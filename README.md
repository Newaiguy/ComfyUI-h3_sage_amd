# h3_sage_amd

AMD gfx12 (RX 9070 / 9070 XT) 专用的 MiniMax H3 记忆体高效 Sage Attention 节点。

对标 KJNodes 的 `MiniMaxH3MemoryEfficientSageAttentionPatch`（N卡），用 AMD gfx12 native HIP kernel 替代 CUDA-only 的 `_qattn_smXX`，实现相同的省显存效果和接近的性能。

## 效果

- **省显存 33%**：在 attention 前释放 qkv buffer，峰值 VRAM 降低约 1/3
- **性能追平 N 卡**：使用编译好的 gfx12 native HIP kernel（61MB），比纯 Triton 路径快 15-20%
- **H3 工作流实测**：RX 9070 XT 采样 10 步约 300s，与 RTX 4070 Ti 的 ~300s 一致（使用 Work-Fisher 8月8日工作流测试）

## 适用 GPU

- AMD gfx12 架构：RX 9070、RX 9070 XT
- 需要 SageAttention 2.2.0+ with gfx12 native support（PR368）

## 一键安装包（夸克网盘）

不想自己编译 SageAttention gfx12 native？下载预编译整合包，开箱即用：

- 链接：https://pan.quark.cn/s/da1aeaa701cd
- 提取码：`Bg47`

包含：预编译 SageAttention（含 61MB gfx12 native kernel）+ h3_sage_amd 节点 + AMD 专用补丁 + 启动脚本 + 完整构建指南。

## 原理

```
N卡 (KJNodes):                    本节点 (AMD):
┌─────────────────────┐           ┌──────────────────────┐
│ qkv = qkv_proj(x)   │           │ qkv = qkv_proj(x)    │
│ q,k,v = split(qkv)  │           │ q,k,v = split(qkv)   │
│ q_fp16 = q.to(fp16) │           │ q_fp16 = q.to(fp16)  │
│ del q, k, v, qkv    │ ← 省显存  │ del q, k, v, qkv     │ ← 省显存
│                     │           │                      │
│ _qattn_sm89(...)   │ ← CUDA   │ sageattn_qk_int8_     │ ← HIP
│ (native C++ kernel) │           │   pv_gfx12_native()   │   (native C++)
└─────────────────────┘           └──────────────────────┘
```

省显存逻辑在 wrapper 层（释放 qkv buffer），与 kernel 无关。换 kernel 不影响省显存效果。

## 安装

### 方法一：手动安装

将 `h3_sage_amd` 文件夹复制到 ComfyUI 的 `custom_nodes/` 目录：

```
ComfyUI/custom_nodes/h3_sage_amd/
├── __init__.py
└── nodes.py
```

### 方法二：通过 ComfyUI-Manager 安装

在 ComfyUI-Manager 中搜索 "MiniMax H3 Mem Eff Sage Attn (AMD)" 安装。

### 前置条件

1. **SageAttention 2.2.0+ with gfx12 native support**

   需要编译好的 `_qattn_gfx12_native.cp313-win_amd64.pyd`（61MB）。
   来源：[SageAttention PR #368](https://github.com/thu-ml/SageAttention/pull/368)

   如果没有自行编译的条件，可使用预编译包（见 release）。

2. **启动参数**

   ComfyUI 启动时需要加 `--use-sage-attention` 参数。

3. **推荐环境变量**

   ```bash
   SAGEATTN_QK_DTYPE=INT8
   SAGEATTN_M=128
   SAGEATTN_N=16
   TORCH_BLAS_PREFER_HIPBLASLT=1
   ROCBLAS_USE_HIPBLASLT=1
   ```

## 使用方法

1. 在 ComfyUI 工作流中添加 **MiniMax H3 Mem Eff Sage Attn (AMD)** 节点
2. 将 MODEL 输入连接到此节点
3. 将此节点的 MODEL 输出连接到后续节点（如 sampler）
4. 此节点会替换 H3 所有 transformer block 的 attention forward

节点位置：`KJNodes/minimax` 分类下

## 与 N卡 KJNodes 节点的关系

| 对比项 | KJNodes (N卡) | 本节点 (AMD) |
|--------|--------------|-------------|
| 省显存 wrapper | ✅ 释放 qkv buffer | ✅ 释放 qkv buffer |
| attention kernel | _qattn_sm80/89/90 (CUDA C++) | sageattn_qk_int8_pv_gfx12_native (HIP C++) |
| v 的精度 | fp8 (sm89+) | fp16 |
| int8 量化 | per_thread_int8_triton | per_warp (native kernel 内部) |
| smooth_k | ✅ | ✅ (native kernel 内部) |
| Triton fallback | sm75 路径 | 有（native 不可用时自动降级）|

## 文件说明

- `nodes.py` - 节点实现（核心代码）
- `__init__.py` - ComfyUI 节点注册

## 致谢

- [SageAttention](https://github.com/thu-ml/SageAttention) - 清华大学 SageAttention 团队
- [KJNodes](https://github.com/kijai/ComfyUI-KJNodes) - Kijai 的 MiniMax H3 优化节点
- [SageAttention PR #368](https://github.com/thu-ml/SageAttention/pull/368) - gfx12 native kernel 实现

## License

MIT
