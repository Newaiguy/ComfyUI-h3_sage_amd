"""AMD-compatible Memory-Efficient Sage Attention for MiniMax H3.

Mirrors kjnodes' MiniMaxH3MemoryEfficientSageAttentionPatch but uses
gfx12 native HIP kernels instead of CUDA-only _qattn_smXX kernels,
enabling ~33% peak VRAM savings on AMD gfx12 GPUs.

Primary path: sageattn_qk_int8_pv_gfx12_native (compiled 61MB HIP C++ kernel,
equivalent to N卡 _qattn_sm80/sm89 path). Does smooth_k + int8 quantization +
attention in one fused pass.
Fallback: per_block_int8_triton + attn_false (pure Triton, sm75-equivalent).
"""

import logging
import torch

import comfy.model_management as mm
from comfy.quant_ops import ck as _ck

try:
    from sageattention.core import sageattn_qk_int8_pv_gfx12_native
    from sageattention.core import GFX12_NATIVE_ENABLED
    _NATIVE_GFX12_AVAILABLE = GFX12_NATIVE_ENABLED
except (ImportError, AttributeError):
    _NATIVE_GFX12_AVAILABLE = False

try:
    from sageattention.core import per_block_int8_triton, attn_false
    _TRITON_FALLBACK_AVAILABLE = True
except ImportError:
    _TRITON_FALLBACK_AVAILABLE = False

_SAGE_AVAILABLE = _NATIVE_GFX12_AVAILABLE or _TRITON_FALLBACK_AVAILABLE


def _sageattn_int8_fp16_nhd_amd(q, k, v, dtype):
    """int8 sage attention for AMD gfx12.

    Prefers native HIP kernel (sageattn_qk_int8_pv_gfx12_native) which is
    a compiled C++ kernel equivalent to N卡 _qattn_sm80/sm89 path. The native
    kernel does smooth_k + per_warp int8 quantization + attention in one
    fused pass — faster than the Triton fallback.

    Falls back to per_block_int8_triton + attn_false (pure Triton,
    sm75-equivalent) if native module unavailable.

    q, k, v: fp16 tensors in NHD layout [batch, seq, heads, head_dim].
    Returns output in NHD layout with the given dtype.
    """
    if _NATIVE_GFX12_AVAILABLE:
        # Native HIP kernel: fused smooth_k + int8 quant + attention.
        # No need to manually smooth_k or quantize — the kernel does it all.
        o = sageattn_qk_int8_pv_gfx12_native(
            q, k, v,
            tensor_layout="NHD",
            is_causal=False,
            value_dtype="fp16",
        )
        del q, k, v
        return o.to(dtype)

    # Triton fallback (sm75-equivalent path)
    k.sub_(k.mean(dim=1, keepdim=True))
    sm_scale = q.shape[-1] ** -0.5
    q_int8, q_scale, k_int8, k_scale = per_block_int8_triton(
        q, k, sm_scale=sm_scale, tensor_layout="NHD",
    )
    del q, k
    o, _ = attn_false(
        q_int8, k_int8, v, q_scale, k_scale,
        tensor_layout="NHD", output_dtype=dtype, attn_mask=None, return_lse=False,
    )
    del q_int8, q_scale, k_int8, k_scale, v
    return o


def minimax_sageattn_forward_amd(self, x, rope_freqs=None, transformer_options={}):
    """AMD-compatible replacement for H3 Attention.forward.

    Same structure as kjnodes' minimax_sageattn_forward but calls
    _sageattn_int8_fp16_nhd_amd instead of the CUDA-only _sageattn_int8_fp8_nhd.

    Key memory optimization: convert q/k/v to fp16 (independent copies),
    then free the qkv buffer BEFORE running attention. This saves ~33%
    peak VRAM compared to keeping qkv buffer alive during attention.
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

    if n <= 1:
        # Full attention with progressive freeing
        # Convert to fp16 (independent copies), then free qkv buffer
        q_fp16 = q.to(torch.float16)
        k_fp16 = k.to(torch.float16)
        v_fp16 = v.to(torch.float16)
        del q, k, v, qkv  # qkv buffer freed here

        o = _sageattn_int8_fp16_nhd_amd(q_fp16, k_fp16, v_fp16, dtype)
        return self.out_proj(o.view(s, self.heads * self.head_dim))

    # Head-chunks: process per head group to reduce peak VRAM
    out = torch.empty((s, self.heads * self.head_dim), dtype=dtype, device=device)
    out_nhd = out.view(1, s, self.heads, self.head_dim)
    hs = 0
    for i in range(n):
        he = hs + self.heads // n + (1 if i < self.heads % n else 0)
        q_fp16 = q[:, :, hs:he, :].to(torch.float16).contiguous()
        k_fp16 = k[:, :, hs:he, :].to(torch.float16).contiguous()
        v_fp16 = v[:, :, hs:he, :].to(torch.float16).contiguous()
        o = _sageattn_int8_fp16_nhd_amd(q_fp16, k_fp16, v_fp16, dtype)
        out_nhd[:, :, hs:he, :] = o
        del o
        hs = he
    del q, k, v, qkv
    return self.out_proj(out)


class MiniMaxH3SageAttentionPatchAMD:
    """AMD-compatible Memory Efficient Sage Attention Patch for MiniMax H3.

    Uses gfx12 native HIP kernel (sageattn_qk_int8_pv_gfx12_native) that
    matches the N卡 _qattn_smXX path in performance. Falls back to Triton
    (attn_false) if native module unavailable.

    Saves ~33% peak VRAM by freeing the qkv buffer before attention runs.

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
        "AMD-compatible memory efficient sage attention for MiniMax H3. "
        "Uses gfx12 native HIP kernel (fast, like N卡 _qattn_smXX) with "
        "Triton fallback. Saves ~33% peak VRAM by freeing the qkv buffer "
        "before attention. Works on AMD gfx12 (RX 9070 XT)."
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
