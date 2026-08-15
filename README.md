# h3_sage_amd

AMD gfx12 (RX 9070 / 9070 XT) 专用的 MiniMax H3 Sage Attention 节点。

兼容 KJNodes 的 `MiniMaxH3MemoryEfficientSageAttentionPatch`（N卡），并注册了同名别名，
现有引用该节点的工作流无需修改即可在 AMD 上使用。

## 定位（2026-08 实测后重新定义）

这个节点**不是提速器，是 16GB 显存卡跑长序列的保险丝**：

- **正常序列（seq ≤ 30000，如 8s/0.4 场景）**：透明直通 ComfyUI 原始 attention 路径
  （`optimized_attention`，全局 `--use-sage-attention` 时即全局 sage 内核），
  与不打补丁的模型**零开销**。
- **超长序列（seq > 30000，如 Ref2VA 双时钟采样 seq=41414/54545）**：自动启用
  head-chunking（42 头分 4 组逐组 fp16 sageattn），把 attention 峰值显存从
  **~11.3GB 降到 ~7.1GB（约 -37%）**，避免全局 sage 路径在 16GB gfx12 卡上的
  OOM / GPU 挂死。全局路径在该场景下直接死机，本节点可以跑完。

### 为什么不用 int8 内核（重要）

早期版本使用 `sageattn_qk_int8_pv_gfx12_native`（PR368 的 gfx12 native HIP kernel）。
实测在长序列上 int8 内核（fp16 V 模式）比标准 fp16 `sageattn` **更慢**：

| seq | fp16 sageattn | int8 (fp16 V) |
|-----|---------------|---------------|
| 2048 | 1.51 ms | 1.28 ms（略快） |
| 4096 | 3.87 ms | 4.10 ms（持平） |
| 8192 | 12.77 ms | 16.75 ms（**慢 1.31x**，随 seq 拉大） |

fp8 V 模式微基准略快于 fp16（~7%），但真实 H3 工作流（50 block × 分头）中反而慢
约 1.6-1.8x。gfx12 目前只有一个未调优的 native int8 内核（对比 N 卡按架构手调的
`_qattn_sm80/86/89/90`），所以 AMD 上省显存只能靠 head-chunking，性能靠标准
fp16 sageattn。详见上游实测反馈（thu-ml/SageAttention#368）。

## 适用 GPU

- AMD gfx12 架构：RX 9070、RX 9070 XT
- 需要 SageAttention 2.2.0+ with gfx12 support（PR368）

## 一键安装包（夸克网盘）

不想自己编译 SageAttention gfx12？下载预编译整合包，开箱即用：

- 链接：https://pan.quark.cn/s/8d916cd79b5d
- 提取码：`yhqi`

包含：预编译 SageAttention（含 gfx12 native kernel，支持 Python 3.12 + 3.13）+
h3_sage_amd 节点 + AMD 专用补丁 + 启动脚本 + 完整构建指南。

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

1. **SageAttention 2.2.0+ with gfx12 support**

   需要编译好的 `sageattention`（含 `_qattn_gfx12_native` .pyd）。
   来源：[SageAttention PR #368](https://github.com/thu-ml/SageAttention/pull/368)

   如果没有自行编译的条件，可使用预编译包（见上文夸克网盘）。

2. **启动参数**

   ComfyUI 启动时建议加 `--use-sage-attention` 参数（正常序列直通路径会用到全局 sage）。

## 使用方法

1. 在 ComfyUI 工作流中添加 **MiniMax H3 Mem Eff Sage Attn (AMD)** 节点
2. 将 MODEL 输出连接到 sampler
3. 此节点会替换 H3 所有 50 个 transformer block 的 attention forward

节点位置：`KJNodes/minimax` 分类下。

自动行为（无需配置）：
- seq ≤ 30000：直通原始路径，无任何额外开销
- seq > 30000：自动 head-chunking（n=4）
- 也兼容 `MiniMaxLowVRAMAttention` 的 `minimax_head_chunks` transformer_option 和
  block 级 list 传参，可组合使用

### 调试探针（可选）

创建空文件 `ComfyUI根目录/h3_debug_on`（或修改 `nodes.py` 中 `_DEBUG_FLAG` 路径）
后，每次 attention 调用会向 `h3_attn_debug.log` 追加一行 `分支/seq/n/耗时`，用于
确认实际走的代码路径。删除该文件即关闭。

## 与 N卡 KJNodes 节点的关系

| 对比项 | KJNodes (N卡) | 本节点 (AMD) |
|--------|--------------|-------------|
| 省显存机制 | int8/fp8 量化 + 释放 qkv | head-chunking + fp16 |
| attention kernel | _qattn_sm80/89/90（按架构手调） | fp16 sageattn（gfx12 PR368） |
| 正常序列开销 | 低 | 零（直通原始路径） |
| 超长序列 | 量化省显存且更快 | head-chunks 省显存，速度同 fp16 |
| Triton fallback | sm75 路径 | sageattn 自带 |

## 文件说明

- `nodes.py` - 节点实现（核心代码）
- `__init__.py` - ComfyUI 节点注册

## 致谢

- [SageAttention](https://github.com/thu-ml/SageAttention) - 清华大学 SageAttention 团队
- [KJNodes](https://github.com/kijai/ComfyUI-KJNodes) - Kijai 的 MiniMax H3 优化节点
- [SageAttention PR #368](https://github.com/thu-ml/SageAttention/pull/368) - gfx12 native backend

## License

MIT
