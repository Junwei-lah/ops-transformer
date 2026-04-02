# KvQuantSparseFlashAttentionPioneer

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A5 推理系列产品</term>   | √  |

## 功能说明

- API功能：`kv_quant_sparse_flash_attention_pioneer`在`sparse_flash_attention`的基础上支持了[Per-Token-Head-Tile-128量化]输入。随着大模型上下文长度的增加，Sparse Attention的重要性与日俱增，这一技术通过“只计算关键部分”大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。

- 计算公式：

    $$
    Attention=\text{softmax}(\frac{Q @ \text{Dequant}({\tilde{K}^{INT8}},{Scale_K})^T}{\sqrt{d_k}})@\text{Dequant}(\tilde{V}^{INT8},{Scale_V}),
    $$

    其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`LightningIndexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度，$\text{Dequant}(\cdot,\cdot)$为反量化函数。
本次公布的`kv_quant_sparse_flash_attention_pioneer`是面向Sparse Attention的全新算子，针对离散访存进行了指令缩减及搬运聚合的细致优化。

## 函数原型

```
torch_npu.npu_kv_quant_sparse_flash_attention_pioneer(query, key, value, sparse_indices, scale_value, key_quant_mode, value_quant_mode, *, key_dequant_scale=None, value_dequant_scale=None, block_table=None, actual_seq_lengths_query=None, actual_seq_lengths_kv=None, key_sink=None, value_sink=None, sparse_block_size=1, layout_query="BSND", layout_kv="BSND", sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1, attention_mode=0, quant_scale_repo_mode=1, tile_size=128, rope_head_dim=64) -> Tensor
```

## 参数说明

**说明：**<br> 

> - query、key、value参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
> - Q\_S和S1表示query shape中的S，KV\_S和S2表示key shape中的S，Q\_N表示num\_query\_heads，KV\_N表示num\_key\_value\_heads，Q\_T表示query shape中的T，KV\_T表示key shape中的T。

- **query**（`Tensor`）：必选参数，表示attention结构的Q输入，不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`和`float16`，query由相同dtype的q_nope和q_rope按D维度拼接得到。`layout_query`为BSND时shape为[B,S1,Q\_N,D]，当`layout_query`为TND时shape为[Q\_T,Q\_N,D]，其中Q\_N支持1/2/3/4/8/16/24/32/48/64。

- **key**（`Tensor`）：必选参数，表示attention结构的K输入，不支持非连续，数据格式支持$ND$，数据类型支持`float8_e4m3`和`hifloat8`，`float8_e4m3`或`hifloat8`的k_nope、query相同dtype的k_rope和`float32`的量化参数按D维度拼接得到，layout\_kv为PA\_BSND时shape为[block\_num, block\_size, KV\_N, D]，其中block\_num为PageAttention时block总数，block\_size为一个block的token数，block\_size取值为16的整数倍，最大支持到1024。`layout_kv`为BSND时shape为[B, S2, KV\_N, D]，`layout_kv`为TND时shape为[KV\_T, KV\_N, D]，其中KV\_N只支持1。

- **value**（`Tensor`）：必选参数，表示attention结构的V输入，不支持非连续，数据格式支持$ND$，数据类型支持`float8_e4m3`和`hifloat8`。value的N仅支持1。

- **sparse\_indices**（`Tensor`）：必选参数，代表离散取kvCache的索引，不支持非连续，数据格式支持$ND$，数据类型支持`int32`，当`layout_query`为BSND时，shape需要传入[B, Q\_S, KV\_N, sparse\_size]，当`layout_query`为TND时，shape需要传入[Q\_T, KV\_N, sparse\_size]，其中sparse\_size为一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，且需要满足sparse\_size大于0。

- **scale\_value**（`float`）：必选参数，公式中$d_k$开根号的倒数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值，数据类型支持`float`。

- **key\_quant\_mode**（`int`）：必选参数，代表key的量化模式，数据类型支持`int64`，仅支持传入2，代表per_tile量化模式。

- **value\_quant\_mode**（`int`）：必选参数，代表value的量化模式，数据类型支持`int64`，仅支持传入2，代表per_tile量化模式。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **key\_dequant\_scale**（`Tensor`）：可选参数，预留参数，仅支持默认值。

- **value\_dequant\_scale**（`Tensor`）：可选参数，预留参数，仅支持默认值。

- **block\_table**（`Tensor`）：可选参数，表示PageAttention中kvCache存储使用的block映射表。数据格式支持$ND$，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的s2对应的block数量，即s2\_max / block\_size向上取整。

- **actual\_seq\_lengths\_query**（`Tensor`）：可选参数，表示不同Batch中`query`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和`query`的shape的S长度相同。该参数中每个Batch的有效token数不超过`query`中的维度S大小且不小于0。支持长度为B的一维tensor。<br>当`layout_query`为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

- **actual\_seq\_lengths\_kv**（`Tensor`）：可选参数，表示不同Batch中`key`和`value`的有效token数，数据类型支持`int32`。如果不指定None，表示和key的shape的S长度相同。该参数中每个Batch的有效token数不超过`key/value`中的维度S大小且不小于0。支持长度为B的一维tensor。<br>当`layout_kv`为TND或PA_BSND时，该入参必须传入，`layout_kv`为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

- **key\_sink**（`Tensor`）：可选参数，表示添加在压缩`key`的序列维度上的额外参数。layout为[sink_num, KV\_N, D]，其中D包含nope和rope两部分，数据类型与`query`一致，支持`bfloat16`和`float16`。

- **value\_sink**（`Tensor`）：可选参数，表示添加在压缩`value`的序列维度上的额外参数。layout为[sink_num, KV\_N, D]，其中D只包含nope，数据类型与`query`一致，支持`bfloat16`和`float16`。

- **sparse\_block\_size**（`int`）：可选参数，代表sparse阶段的block大小，在计算importance score时使用，数据类型支持`int64`，仅支持1。

- **layout\_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，默认值"BSND"，支持传入BSND和TND。

- **layout\_kv**（`str`）：可选参数，用于标识输入`key`的数据排布格式，默认值"BSND"，支持传入BSND、TND和PA\_BSND，PA\_BSND在使能PageAttention时使用。

- **sparse\_mode**（`int`）：可选参数，表示sparse的模式。数据类型支持`int64`。
    - sparse\_mode为0时，代表全部计算。
    - sparse\_mode为3时，代表rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。

- **pre\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和前几个Token计算关联。数据类型支持`int64`，仅支持默认值2^63-1。

- **next\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和后几个Token计算关联。数据类型支持`int64`，仅支持默认值2^63-1。

- **attention\_mode**（`int`）：可选参数，表示attention的模式。数据类型支持`int64`，仅支持传入2，表示MLA-absorb模式，即QK的D包含rope和nope两部分，且KV是同一份，默认值为0。

- **quant\_scale\_repo\_mode**（`int`）：可选参数，表示量化参数的存放模式。数据类型支持`int64`，仅支持传入1，表示combine模式，即量化参数和数据混合存放，默认值1。

- **tile\_size**（`int`）：可选参数，表示per_tile时每个参数对应的数据块大小，仅在per_tile时有效。数据类型支持`int64`，仅支持默认值128。

- **rope\_head\_dim**（`int`）：可选参数，表示MLA架构下的rope\_head\_dim大小，仅在attention\_mode为2时有效。数据类型支持`int64`，仅支持默认值64。

## 返回值说明

`Tensor`

代表公式中的输出Attention。数据格式支持$ND$，数据类型支持`bfloat16`和`float16`。输出shape与入参`query`的shape保持一致。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- 参数query中的D值为576，即nope\+rope=512\+64。
- 参数key、value中的D值为656，即nope\+rope\*2\+dequant\_scale\*4=512\+64\*2\+4\*4。
- 支持sparse\_block\_size整除block\_size。
- 非PageAttention场景layout\_query和layout\_kv需要保持一致。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import random

    device = torch.device("npu")

    b, s1, s2, n1, n2 = 2, 1, 512, 64, 1
    dn, dr, tile_size = 512, 64, 128
    sparse_block_count, block_size = 2048, 256
    s2_act = 4096

    query_type = torch.bfloat16
    key_dtype = torch_npu.hifloat8
    value_dtype = torch_npu.hifloat8

    query_base = torch.tensor(np.random.uniform(-10, 100, (b, s1, n1, dn)), dtype=query_type, device=device)
    key_base = torch.tensor(np.random.uniform(-5, 100, (b * (s2 // block_size), block_size, n2, dn)), dtype=torch.uint8, device=device)
    value_base = torch.tensor(np.random.uniform(-5, 100, (b * (s2 // block_size), block_size, n2, dn)), dtype=torch.uint8, device=device)
    query_rope = torch.tensor(np.random.uniform(-10, 100, (b, s1, n1, dr)), dtype=query_type, device=device)
    key_rope = torch.tensor(np.random.uniform(-10, 10, (b * (s2 // block_size), block_size, n2, dr)), dtype=query_type, device=device)
    antiquant_scale = torch.tensor(np.random.uniform(-100, 100, (b * (s2 // block_size), block_size, n2, dn // tile_size)), dtype=torch.float32, device=device)

    idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
    sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)], dtype=torch.int32, device=device).reshape(b, s1, n2, sparse_block_count)

    act_seq_q = torch.full((b,), s1, dtype=torch.int32, device=device)
    act_seq_kv = torch.full((b,), s2_act, dtype=torch.int32, device=device)
    block_table = torch.arange(b * s2 // block_size, dtype=torch.int32, device=device).reshape(b, -1)

    key = torch.cat((key_base, key_rope.view(torch.uint8), antiquant_scale.view(torch.uint8)), dim=3)
    finally_d = dn + 2 * dr + 4 * (dn // tile_size)
    key = torch.as_strided(key, size=(b * (s2 // block_size), block_size, n2, finally_d), stride=(n2 * finally_d * block_size, n2 * finally_d, finally_d, 1))

    query = torch.cat((query_base, query_rope), dim=3)
    value = value_base
    key_dequant_scale = antiquant_scale
    value_dequant_scale = antiquant_scale

    output = torch_npu._npu_kv_quant_sparse_flash_attention_pioneer(query=query, key=key, value=value,
                            sparse_indices=sparse_indices,
                            scale_value=0.041666666666666664, sparse_block_size=1,
                            key_quant_mode=2, value_quant_mode=2,
                            key_dequant_scale=key_dequant_scale, value_dequant_scale=value_dequant_scale,
                            block_table=block_table, actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
                            key_sink=None, value_sink=None, layout_query="BSND", layout_kv="PA_BSND",
                            sparse_mode=3, attention_mode=2, quant_scale_repo_mode=1,
                            tile_size=tile_size, rope_head_dim=64,
                            key_dtype=key_dtype, value_dtype=value_dtype,
                            pre_tokens=9223372036854775807, next_tokens=9223372036854775807
    )

    # 执行上述代码的输出out类似如下
    tensor([[[[ 0.0000,  -72.0000,  0.0000,  ...,  0.0000, 0.0000, 189.0000],
            [ -390.0000,  780.0000, -390.0000,  ...,  168.0000,  84.0000, -504.0000],
            [ 386.0000,  290.0000,  -386.0000,  ...,  -10.6250,  0.0000, 10.6250],
            ..
            [ -768.0000,  384.0000, -868.0000,  ...,  322.0000,  -215.0000, 430.0000],
            [ 440.0000,  146.0000, 97.5000,  ...,  -253.0000, -760.0000, 84.5000],
            [ -256.0000,  256.0000, 596.0000,  ...,  92.0000,  -736.0000, 0.0000]]]],
            device='npu:0', dtype=torch.bfloat16)
    ```

- AclGraph模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import random
    import torchair as tng
    import torch.nn as nn
    from torchair.configs.compiler_config import CompilerConfig

    b, s1, s2, n1, n2 = 2, 1, 512, 64, 1
    pre_tokens = 9223372036854775807
    next_tokens = 9223372036854775807
    query_type = torch.bfloat16
    scale_value = 0.041666666666666664
    sparse_block_size = 1
    dn = 512
    dr = 64
    tile_size = 128
    sparse_block_count = 2048
    block_size = 256
    layout_query = 'BSND'
    layout_kv = 'PA_BSND'
    s2_act = 4096

    key_dtype = torch_npu.hifloat8
    value_dtype = torch_npu.hifloat8

    query_base = torch.tensor(np.random.uniform(-10, 100, (b, s1, n1, dn))).to(query_type).npu()
    key_base = torch.tensor(np.random.uniform(-5, 100, (b * (s2 // block_size), block_size, n2, dn))).to(torch.uint8).npu()
    value_base = torch.tensor(np.random.uniform(-5, 100, (b * (s2 // block_size), block_size, n2, dn))).to(torch.uint8).npu()

    query_rope = torch.tensor(np.random.uniform(-10, 100, (b, s1, n1, dr))).to(query_type).npu()
    key_rope = torch.tensor(np.random.uniform(-10, 10, (b * (s2 // block_size), block_size, n2, dr))).to(query_type).npu()

    idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
    sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count).to(torch.int32).npu()

    act_seq_q = torch.tensor([s1] * b).to(torch.int32).npu()
    act_seq_kv = torch.tensor([s2_act] * b).to(torch.int32).npu()

    antiquant_scale = torch.tensor(np.random.uniform(-100, 100, (b * (s2 // block_size), block_size, n2, dn // tile_size))).to(torch.float32).npu()

    key = torch.cat((key_base, key_rope.view(torch.uint8), antiquant_scale.view(torch.uint8)), axis=3)

    finally_d = dn + 2 * dr + 4 * (dn // tile_size)
    key = torch.as_strided(
        key,
        size=[b * (s2 // block_size), block_size, n2, finally_d],
        stride=[n2 * finally_d * block_size, n2 * finally_d, finally_d, 1]
    )

    query = torch.cat((query_base, query_rope), axis=3)
    value = value_base

    block_table = torch.tensor([range(b * s2 // block_size)], dtype=torch.int32).reshape(b, -1).npu()

    key_dequant_scale = antiquant_scale
    value_dequant_scale = antiquant_scale

    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, query, key, value, sparse_indices, scale_value, sparse_block_size,
                    key_quant_mode, value_quant_mode, block_table=None,
                    actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
                    key_sink=None, value_sink=None,
                    key_dequant_scale=None, value_dequant_scale=None,
                    layout_query='BSND', layout_kv='BSND',
                    sparse_mode=3, attention_mode=0,
                    quant_scale_repo_mode=0, tile_size=128,
                    rope_head_dim=0, key_dtype=None, value_dtype=None,
                    pre_tokens=9223372036854775807, next_tokens=9223372036854775807):

            return torch_npu._npu_kv_quant_sparse_flash_attention_pioneer(
                query, key, value, sparse_indices, scale_value,
                key_quant_mode, value_quant_mode,
                sparse_block_size=sparse_block_size,
                key_dequant_scale=key_dequant_scale,
                value_dequant_scale=value_dequant_scale,
                block_table=block_table,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                key_sink=key_sink, value_sink=value_sink,
                layout_query=layout_query, layout_kv=layout_kv,
                sparse_mode=sparse_mode, attention_mode=attention_mode,
                quant_scale_repo_mode=quant_scale_repo_mode,
                tile_size=tile_size, rope_head_dim=rope_head_dim,
                key_dtype=key_dtype, value_dtype=value_dtype,
                pre_tokens=pre_tokens, next_tokens=next_tokens
            )


    torch._dynamo.reset()
    npu_mode = Network().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)

    output = npu_mode(query, key, value, sparse_indices, scale_value, sparse_block_size,
        key_quant_mode=2, value_quant_mode=2, key_dequant_scale=key_dequant_scale, value_dequant_scale=value_dequant_scale,
        actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv, key_sink=None, value_sink=None,
        layout_query=layout_query, layout_kv=layout_kv, block_table=block_table, sparse_mode=3,
        attention_mode=2, quant_scale_repo_mode=1, tile_size=tile_size, rope_head_dim=64,
        key_dtype=key_dtype, value_dtype=value_dtype, pre_tokens=pre_tokens, next_tokens=next_tokens
    )


    # 执行上述代码的输出类似如下
    tensor([[[[  0.0000,  -72.0000,  0.0000,  ...,  0.0000, 0.0000, 189.0000],
            [ -390.0000,  780.0000, -390.0000,  ...,  168.0000,  84.0000, -504.0000],
            [ 386.0000,  290.0000,  -386.0000,  ...,  -10.6250,  0.0000, 10.6250],
            ...,
            [ -768.0000,  384.0000, -868.0000,  ...,  322.0000,  -215.0000, 430.0000],
            [ 440.0000,  146.0000, 97.5000,  ...,  -253.0000, -760.0000, 84.5000],
            [ -256.0000,  256.0000, 596.0000,  ...,  92.0000,  -736.0000, 0.0000]]]],
            device='npu:0', dtype=torch.bfloat16)
    ```
