# 🚀 Ascend Ops - 昇腾 AI 自定义算子扩展

[![License](https://img.shields.io/badge/license-CANN%20Open%20Software%20License%20v2.0-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Ascend%20AI-orange)](https://www.hiascend.com/)
[![Python](https://img.shields.io/badge/python-3.8+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red)](https://pytorch.org/)

> **Ascend Ops** 是基于华为昇腾 AI 处理器的 PyTorch 自定义算子扩展库，专注于实现高性能的Flash Attention 算子，为大语言模型（LLM）推理提供加速支持。

---

## 📋 目录

- [✨ 特性](#-特性)
- [🏗️ 项目结构](#️-项目结构)
- [🔧 环境要求](#-环境要求)
- [📦 安装部署](#-安装部署)
- [🎯 快速开始](#-快速开始)
- [🧪 测试验证](#-测试验证)
- [📖 API 文档](#-api-文档)
- [📄 许可证](#-许可证)

---

## ✨ 特性

- 🔥 **高性能 Flash Attention** - 基于昇腾 AI 处理器优化的推理注意力算子
- ⚡ **PagedAttention 支持** - 支持 KV Cache 分块存储和块表管理
- 🎨 **灵活量化** - 支持 INT8 量化的 Key/Value Cache
- 📊 **AICPU Tiling下沉优化** - 将FA算子分核信息计算下沉到device侧
- 🔗 **PyBind 调用** - 支持 Python/C++ 双向调用接口
- 🌐 **ACL Graph 集成** - 支持 ACL Graph 计算图构建
- 🚀 **SuperKernel 融合** - 支持自动融合 SuperKernel 优化
- 🔌 **PyTorch 集成** - 无缝集成到 PyTorch 生态，支持 torch.ops 调用

---

## 🏗️ 项目结构

```
ascend_ops/
├── CMakeLists.txt                            # CMake 构建配置
├── setup.py                                  # Python 安装脚本
├── README.md                                 # 项目文档（本文件）
│
└── src/
    ├── __init__.py                           # Python 包初始化
    ├── ops_def_registration.cpp              # PyTorch 算子注册
    │
    ├── csrc/                                 # Ascend C 算子实现
    │   │
    │   ├── incre_flash_attention/            # Flash Attention 主算子
    │   │   ├── CMakeLists.txt                # 算子构建配置
    │   │   ├── npu_fused_infer_attention_score.cpp  # 算子CPP入口
    │   │   │
    │   │   ├── op_host/                      # Host 端实现（Tiling）
    │   │   │   ├── incre_flash_attention_tiling.cpp
    │   │   │   ├── incre_flash_attention_tiling_check.cpp
    │   │   │   ├── incre_flash_attention_tiling_context.h
    │   │   │   ├── incre_flash_attention_tiling_base.h
    │   │   │   ├── incre_flash_attention_tiling_impl.h
    │   │   │   └── incre_flash_attention_tiling_struct.h
    │   │   │
    │   │   └── op_kernel/                    # Kernel 端实现
    │   │       ├── incre_flash_attention_arch32.h
    │   │       ├── incre_flash_attention_tilingdata.h
    │   │       ├── incre_flash_attention_preload_dd.h
    │   │       └── ifa_public_define.h
    │   │
    │   └── incre_flash_attention_meta/       # AICPU Tiling下沉算子——计算FA数据分核信息
    │       ├── CMakeLists.txt
    │       ├── npu_fused_infer_attention_score_metadata.cpp  算子CPP入口
    │       │
    │       └── op_kernel/
    │           ├── incre_flash_attention_metadata.cpp
    │           └── ifa_meta_public_define.h
    │       
    │
    └── test/                                  # 测试脚本
        ├── test.py                            # AICPU Tiling下沉 + <<<>>> 调用 测试
        ├── test_aclgraph.py                   # AICPU Tiling下沉 + <<<>>> 调用 + ACLGraph 测试
        └── test_aclgraph_sk.py                # AICPU Tiling下沉 + <<<>>> 调用 + ACLGraph + SK 测试
```

---

## 🔧 环境要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **昇腾 AI 处理器** | Ascend 910B | Ascend 910B |

### 软件依赖

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| **Python** | 3.8 - 3.10 | 推荐 3.9 |
| **CANN** | 9.0.0+ | 推荐 9.0.0 |
| **PyTorch** | 2.0+ | 需匹配 CANN 版本 |
| **torch-npu** | 2.0+ | 昇腾 PyTorch 扩展 |
| **CMake** | 3.16+ | 构建工具 |

---

## 📦 安装部署

### 1️⃣ 安装 CANN 环境

```bash
# 确保 CANN 已安装并配置环境变量
source cann包环境
#默认安装环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 验证 CANN 环境
python -c "import torch; import torch_npu; print(torch.__version__, torch_npu.__version__)"
```

### 2️⃣ 克隆项目

```bash
git clone https://gitcode.com/cann/ops-transformer.git
cd ops-transformer/experimental/ascend_ops
```

### 3️⃣ 构建并安装

```bash
# 安装到当前 Python 环境
python -m build --wheel -n  
pip3 install dist/[xxx].whl --force-reinstall --no-deps
```

### 4️⃣ 验证安装

```bash
python -c "import ascend_ops; print('✅ 安装成功！')"
```

---

## 🎯 快速开始

### FA算子使用

```python
import torch
import torch_npu
import ascend_ops

# 初始化 NPU 设备
torch_npu.npu.set_device("npu:0")

# 准备输入张量
batch_size = 18
q_head_num = 64
kv_head_num = 1
q_seq = 1
block_size = 128
head_dim = 128
kv_seq_length = 8192

# Query
q = torch.randn(batch_size, q_head_num, q_seq, head_dim).bfloat16().npu()

# KV Cache (INT8 量化)
block_num = batch_size * (kv_seq_length // block_size + 1)
key_cache = torch.randint(0, 100, (block_num, kv_head_num, head_dim // 32, block_size, 32)).to(dtype=torch.int8).npu()
value_cache = torch.randint(0, 100, (block_num, kv_head_num, head_dim // 32, block_size, 32)).to(dtype=torch.int8).npu()

# Block Table
max_block_num = kv_seq_length // block_size + 1
block_table = torch.arange(batch_size * max_block_num, dtype=torch.int32).view(batch_size, max_block_num).npu()

# 序列长度
actual_seq_kvlen = torch.tensor([kv_seq_length] * batch_size, dtype=torch.int64).npu()

# 反量化缩放因子
dequant_scale_key = torch.randn(kv_head_num, 1, head_dim).to(dtype=torch.bfloat16).npu()
dequant_scale_value = torch.randn(kv_head_num, 1, head_dim).to(dtype=torch.bfloat16).npu()

# 调用 Flash Attention 算子
result, softmax_lse = torch.ops.custom.npu_fused_infer_attention_score(
    query=q,
    key=key_cache,
    value=value_cache,
    actual_seq_kvlen=actual_seq_kvlen,
    block_table=block_table,
    dequant_scale_key=dequant_scale_key,
    dequant_scale_value=dequant_scale_value,
    num_query_heads=q_head_num,
    num_key_value_heads=kv_head_num,
    softmax_scale=1.0 / (head_dim ** 0.5),
    block_size=block_size,
    input_layout="BNSD",
    sparse_mode=0,
    inner_precise=1,
    key_quant_mode=0,
    value_quant_mode=0
)

print(f"✅ 计算完成！输出 shape: {result.shape}")
```

### AICPU Tiling下沉分核算子使用

```python
import torch
import ascend_ops

# 准备元数据计算参数
batch_size = 18
query_seq_size = 1
query_head_num = 64
head_dim = 128
key_seq_size = 8192
key_head_num = 1
block_size = 128
max_block_num_per_batch = 64

# 计算元数据
metadata = torch.ops.custom.npu_fused_infer_attention_score_metadata(
    batch_size=batch_size,
    query_seq_size=query_seq_size,
    query_head_num=query_head_num,
    head_dim=head_dim,
    key_head_num=key_head_num,
    block_size=block_size,
    max_block_num_per_batch=max_block_num_per_batch,
    actual_seq_lengths_kv=torch.tensor([key_seq_size] * batch_size, dtype=torch.int64).npu(),
    layout_query="BSND",
)

print(f"✅ 元数据计算完成！Shape: {metadata.shape}")
```

---

## 🧪 测试验证

### 运行单元测试

```bash
cd  ops-transformer/experimental/ascend_ops/src/test

# 运行AICPU Tiling下沉+FA算子(<<<>>>调用)测试
python test.py

# 运行AICPU Tiling下沉+FA算子(<<<>>>调用)+aclpraph测试
python test_aclgraph.py

# 运行AICPU Tiling下沉+FA算子(<<<>>>调用)+Aclpraph+SuperKernel测试
python test_aclgraph_sk.py
```

---

## 📖 API 文档

### npu_fused_infer_attention_score

**功能**: Flash Attention 推理算子

**参数**:

| 参数名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | Tensor | ✅ | Query 张量 |
| `key` | Tensor | ✅ | Key Cache 张量，INT8 量化 |
| `value` | Tensor | ✅ | Value Cache 张量，INT8 量化 |
| `block_table` | Tensor | ✅ | KV Cache 块表 |
| `actual_seq_kvlen` | Tensor | ✅ | 实际 KV 序列长度 |
| `num_query_heads` | int | ✅ | Query 头数 |
| `num_key_value_heads` | int | ✅ | Key/Value 头数 |
| `softmax_scale` | float | ✅ | Softmax 缩放因子 |
| `block_size` | int | ✅ | KV Cache 块大小 |
| `input_layout` | str | ✅ | 输入布局格式（BNSD/BSH/BSND） |
| `sparse_mode` | int | ❌ | 稀疏模式，默认 0 |
| `inner_precise` | int | ❌ | 内部精度，默认 1 |
| `dequant_scale_key` | Tensor | ✅ | Key 反量化缩放因子 |
| `dequant_scale_value` | Tensor | ✅ | Value 反量化缩放因子 |
| `key_quant_mode` | int | ❌ | Key 量化模式，默认 0 |
| `value_quant_mode` | int | ❌ | Value 量化模式，默认 0 |
| `atten_mask` | Tensor | ❌ | 注意力掩码 |
| `metadata` | Tensor | ❌ | 分核信息，由npu_fused_infer_attention_score_metadata输出 |

**返回值**:

- `output`: Tensor - Attention 输出
- `softmax_lse`: Tensor - Softmax Log-Sum-Exp（可选）

### npu_fused_infer_attention_score_metadata

**功能**: 计算 Flash Attention 分核信息

**参数**:

| 参数名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `batch_size` | int | ✅ | 批次大小 |
| `query_seq_size` | int | ✅ | Query 序列长度 |
| `query_head_num` | int | ✅ | Query 头数 |
| `head_dim` | int | ✅ | 头维度 |
| `key_head_num` | int | ✅ | Key 头数 |
| `block_size` | int | ✅ | 块大小 |
| `max_block_num_per_batch` | int | ✅ | 每批次最大块数 |
| `actaul_seq_lengths_kv` | Tensor | ✅ | Key/Value实际序列长度 |
| `layout_query` | str | ❌ | Query 布局，默认 "BSND" |

**返回值**:

- `metadata`: Tensor - 分核信息

---

## 📄 许可证

本项目基于 **CANN Open Software License Agreement Version 2.0** 开源。

---

## 📞 联系方式

- 🏢 **华为技术有限公司**
- 📧 **技术支持**: [support@huawei.com](mailto:support@huawei.com)
- 🌐 **CANN 官方文档**: [https://www.hiascend.com/document](https://www.hiascend.com/document)
- 💬 **开发者社区论坛**: [https://www.hiascend.com/forum](https://www.hiascend.com/forum)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给一个 Star！**

</div>
