"""AMD-compatible Sage Attention for MiniMax H3.

For normal sequences this node is a transparent pass-through: it delegates to
the original H3 attention path (``optimized_attention`` → global sage when
``--use-sage-attention`` is on), so there is zero speed overhead.

Its only value is VRAM safety on very long sequences (seq > 30000, e.g.
Ref2VA dual-clock sampling), where head-chunking cuts peak VRAM enough to
avoid the OOM/GPU-hang that the unpatched path hits on 16 GB gfx12 GPUs.
"""

import logging
import torch

import comfy.model_management as mm
from comfy.ldm.modules.attention import optimized_attention
from comfy.quant_ops import ck as _ck

try:
    from sageattention import sageattn
    _SAGE_AVAILABLE = True
except ImportError:
    _SAGE_AVAILABLE = False

# Auto head-chunking for very long sequences (e.g. Ref2VA dual-clock
# sampling, seq > 30k), where keeping full fp16 q/k/v copies plus the
# qkv buffer alive would otherwise blow past 16 GB VRAM.
AUTO_CHUNK_THRESHOLD = 30000
AUTO_CHUNK_HEADS = 4


def _sageattn_fp16_nhd_amd(q, k, v, dtype):
    """fp16/bf16 sage attention (same kernel as global --use-sage-attention).

    q, k, v: fp16 tensors in NHD layout [batch, seq, heads, head_dim].
    Returns output in NHD layout with the given dtype.
    """
    o = sageattn(q, k, v, tensor_layout="NHD", is_causal=False, smooth_k=False)
    del q, k, v
    return o.to(dtype)


def minimax_sageattn_forward_amd(self, x, rope_freqs=None, transformer_options={}):
    """AMD-compatible replacement for H3 Attention.forward.

    Normal sequences (seq <= 30000) delegate to the original H3 attention
    path (``optimized_attention``), matching global sage attention with zero
    overhead. Very long sequences (seq > 30000) use head-chunked fp16 sage
    attention to cut peak VRAM.
    """
    # List pop trick (compatible with MiniMaxLowVRAMAttention block patch)
    if isinstance(x, list):
        x = x.pop()

    s = x.shape[0]
    device = x.device
    dtype = x.dtype

    # QKV projection - free input immediately
    qkv = self.qkv_proj(x)
    del x

    q, k, v = qkv.split(self.heads * self.head_dim, dim=-1)
    q = q.view(1, s, self.heads, self.head_dim)  # NHD
    k = k.view(1, s, self.heads, self.head_dim)
    v = v.view(1, s, self.heads, self.head_dim)

    # RoPE + RMSNorm (identical to original H3 forward)
    if rope_freqs is not None:
        qw = mm.cast_to(self.q_norm.weight, device=device)
        kw = mm.cast_to(self.k_norm.weight, device=device)
        rot = rope_freqs.shape[-3] * 2
        if mm.in_training:
            q, k = _ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot
            )
        else:
            _ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot
            )
    else:
        q = self.q_norm(q)
        k = self.k_norm(k)

    # Head-chunks support (compatible with MiniMaxLowVRAMAttention)
    n = (
        min(transformer_options.get("minimax_head_chunks", 1), self.heads)
        if isinstance(transformer_options, dict)
        else 1
    )
    # Auto-enable head-chunking for very long sequences (e.g. Ref2VA
    # dual-clock sampling, seq=41414) so the fp16 q/k/v copies plus the
    # qkv buffer don't blow past 16 GB VRAM.
    if n <= 1 and s > AUTO_CHUNK_THRESHOLD:
        n = min(AUTO_CHUNK_HEADS, self.heads)

    if n <= 1:
        # Normal sequence: delegate straight to the original H3 attention
        # path (optimized_attention, which global --use-sage-attention
        # hooks). Zero overhead vs the unpatched model for seq <= 30000.
        q = q.transpose(1, 2)  # NHD [1,s,h,d] -> HND [1,h,s,d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = optimized_attention(
            q, k, v, self.heads, mask=None, skip_reshape=True,
            transformer_options=transformer_options,
        )
        return self.out_proj(out.squeeze(0))

    # Head-chunks: process per head group to reduce peak VRAM
    out = torch.empty((s, self.heads * self.head_dim), dtype=dtype, device=device)
    out_nhd = out.view(1, s, self.heads, self.head_dim)
    hs = 0
    for i in range(n):
        he = hs + self.heads // n + (1 if i < self.heads % n else 0)
        q_fp16 = q[:, :, hs:he, :].to(torch.float16).contiguous()
        k_fp16 = k[:, :, hs:he, :].to(torch.float16).contiguous()
        v_fp16 = v[:, :, hs:he, :].to(torch.float16).contiguous()
        o = _sageattn_fp16_nhd_amd(q_fp16, k_fp16, v_fp16, dtype)
        out_nhd[:, :, hs:he, :] = o
        del o
        hs = he
    del q, k, v, qkv
    return self.out_proj(out)


class MiniMaxH3SageAttentionPatchAMD:
    """AMD-compatible Sage Attention Patch for MiniMax H3.

    fp16 sageattn for normal sequences. Lowers peak VRAM on long sequences
    (seq > 30000) via auto head-chunking, avoiding the OOM/GPU-hang that
    the unpatched path hits on 16 GB gfx12 GPUs.

    Compatible with:
    - --use-sage-attention startup flag (this patch takes over for H3 attention)
    - MiniMaxLowVRAMAttention (composes via minimax_head_chunks transformer_option)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "KJNodes/minimax"
    DESCRIPTION = (
        "AMD-compatible sage attention for MiniMax H3. Transparent for "
        "normal sequences (delegates to optimized_attention). Lowers peak "
        "VRAM on long sequences via auto head-chunking (seq > 30000). "
        "Works on AMD gfx12 (RX 9070 XT)."
    )

    def patch(self, model):
        if not _SAGE_AVAILABLE:
            raise RuntimeError(
                "sageattention is not installed. "
                "Install with: pip install sageattention"
            )

        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")

        # Verify it's an H3 model
        try:
            from comfy.ldm.minimax.model import MiniMaxH3Model
        except ImportError:
            raise RuntimeError(
                "This ComfyUI version does not support MiniMax H3."
            )

        if not isinstance(diffusion_model, MiniMaxH3Model):
            raise RuntimeError(
                "MiniMax H3 Sage Attention Patch (AMD) can only be applied "
                "to a MiniMax H3 model."
            )

        logging.info(
            "Applying MiniMax H3 Sage Attention Patch (AMD) to all "
            f"{len(diffusion_model.blocks)} transformer blocks"
        )

        for idx, block in enumerate(diffusion_model.blocks):
            model_clone.add_object_patch(
                f"diffusion_model.blocks.{idx}.attn.forward",
                minimax_sageattn_forward_amd.__get__(
                    block.attn, block.attn.__class__
                ),
            )

        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3SageAttentionPatchAMD": MiniMaxH3SageAttentionPatchAMD,
    # Alias: KJNodes' N卡 node name — AMD auto-fallback so existing
    # workflows (Work-Fisher integrated, dual-clock sampling, etc.)
    # that reference this node type work without modification.
    "MiniMaxH3MemoryEfficientSageAttentionPatch": MiniMaxH3SageAttentionPatchAMD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SageAttentionPatchAMD": "MiniMax H3 Mem Eff Sage Attn (AMD)",
    "MiniMaxH3MemoryEfficientSageAttentionPatch": "MiniMax H3 Mem Eff Sage Attn (AMD alias)",
}
