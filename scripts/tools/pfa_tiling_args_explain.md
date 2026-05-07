下面是一版可以直接对照填写的入参说明。脚本入参是一个 JSON，建议分成四段：`shape`、`attrs`、`platform`、`flags`。

**整体结构**
```json
{
  "shape": {},
  "attrs": {},
  "platform": {},
  "flags": {}
}
```

**shape：必填为主，描述 Q/K/V 的核心形状**
```json
"shape": {
  "batch_size": 2,
  "head_num_size": 16,
  "seq_size": 64,
  "seq_inner_size": 2048,
  "qk_head_size": 128,
  "v_head_size": 128,
  "q_head_size": 128
}
```

字段说明：

| 字段 | 含义 | 必填 | 默认/说明 |
|---|---|---:|---|
| `batch_size` | B，batch 数 | 是 | 也支持别名 `b` |
| `head_num_size` | N，Query head 数 | 是 | 也支持别名 `n` |
| `seq_size` | S1，Query 序列长度 | 是 | 也支持别名 `s1` |
| `seq_inner_size` | S2，KV 序列长度 | 是 | 也支持别名 `s2` |
| `qk_head_size` | D，Q/K head dim | 是 | 也支持别名 `d` |
| `v_head_size` | DV，V head dim | 是 | 不填时默认取 `d` |
| `q_head_size` | Q head dim | 否 | 不填时默认取 `qk_head_size`，DN 判断会用到 |

**attrs：算子属性、layout、稀疏模式、实际长度**
```json
"attrs": {
  "input_dtype": "fp16",
  "output_dtype": "fp16",
  "inner_precise": "high_precision",
  "layout": "BSH",
  "sparse_mode": 0,
  "pre_tokens": 2147483647,
  "next_tokens": 2147483647,
  "actual_seq_lengths": [64, 64],
  "actual_seq_lengths_kv": [2048, 2048],
  "actual_shared_prefix_len": 0,
  "g_size": 1,
  "head_num_ratio": 1,
  "pa_layout_type": 0,
  "block_size": 128,
  "aligned_head_size": 128
}
```

字段说明：

| 字段 | 含义 | 常用值/说明 |
|---|---|---|
| `input_dtype` | 输入 dtype | `fp16`、`bf16`、`int8` |
| `output_dtype` | 输出 dtype | 通常 `fp16` / `bf16` |
| `inner_precise` | 内部精度模式 | `high_precision` 时 fp16 BMM 输出按 fp32 处理 |
| `layout` | 输入 layout | `BSH`、`BSND`、`BNSD`、`TND` |
| `sparse_mode` | sparse mode | 见下方枚举 |
| `pre_tokens` | mask 左窗口 | 默认可用 `2147483647` |
| `next_tokens` | mask 右窗口 | 默认可用 `2147483647` |
| `actual_seq_lengths` | 每个 batch 的真实 S1 | 长度建议等于 `batch_size` |
| `actual_seq_lengths_kv` | 每个 batch 的真实 S2 | 长度建议等于 `batch_size` |
| `actual_shared_prefix_len` | KV prefix 长度 | 无 prefix 填 `0` |
| `g_size` | GQA/MLA 分组大小 | PFA merge / IFA / MLA 路径会影响归一化 shape |
| `head_num_ratio` | Q/KV head ratio | BMM orgShape stride 会用到 |
| `pa_layout_type` | PA layout 类型 | `0` 或 `1` |
| `block_size` | PA block size | 默认 `128` |
| `aligned_head_size` | 对齐后的 D | 不填时脚本按 16 对齐 `qk_head_size` |

`sparse_mode` 枚举：

```text
0 = NO_MASK
1 = ALL_MASK
2 = LEFT_UP
3 = RIGHT_DOWN
4 = BAND
```

**platform：平台核数和 buffer 信息**
```json
"platform": {
  "core_num": 48,
  "aic_num": 24,
  "l1_size": 1048576,
  "l0c_size": 262144
}
```

字段说明：

| 字段 | 含义 | 说明 |
|---|---|---|
| `core_num` | AIV 核数 | 源码里 `coreNum = aivNum`，分核主要用它 |
| `aic_num` | AIC 核数 | 当前脚本保留输出/输入，不主导 `SPLIT_NBS_CUBE` |
| `l1_size` | L1 buffer size | BMM check plan 会输出 |
| `l0c_size` | L0C buffer size | BMM check plan 会输出 |

**flags：各种路径开关**
```json
"flags": {
  "fa_run_flag": true,
  "enable_pfa_mla": false,
  "enable_pfa_rope": false,
  "enable_pfa_merge": false,
  "enable_ifa_mla": false,
  "enable_ifa": false,
  "enable_pa": false,
  "enable_kv_antiquant": false,
  "enable_mask": false,
  "enable_pse_shift": false,
  "enable_alibi_pse": false,
  "enable_perblock_quant": false,
  "enable_matmul_norm": false,
  "enable_kv_prefix": false,
  "is_d_no_tail": true,
  "split_s2": 1,
  "normalize_gs1_merge": true
}
```

重点字段说明：

| 字段 | 影响 |
|---|---|
| `fa_run_flag` | 大 DV、IFA 路径下会影响 `Souter/Sinner` |
| `enable_pfa_mla` | 会避开部分 small-V 路径，也影响 DN |
| `enable_pfa_rope` | 会阻止 small-V sparse path 和 DN |
| `enable_pfa_merge` | 触发 PFA merge 下 `Souter=32,Sinner=256` 的逻辑 |
| `enable_ifa` / `enable_ifa_mla` | 触发 IFA/MLA 专门切分逻辑 |
| `enable_pa` | 影响 BMM1 fixSplit 和 DN |
| `enable_mask` | 分核时决定是否按 mask 计算有效 block |
| `enable_pse_shift` / `enable_alibi_pse` | 影响 IFA 路径和 DN |
| `enable_perblock_quant` | small-V path / DN 判断会用到 |
| `enable_matmul_norm` | BMM check 的 fallback/fixSplit 策略不同 |
| `enable_kv_prefix` | DN 判断会用到 |
| `split_s2` | v2 主路径通常保持 `1`；脚本保留该字段输出 |
| `normalize_gs1_merge` | 为 `true` 时，PFA merge / IFA / MLA 会模拟源码里的 G/S1 归一化 |

`normalize_gs1_merge=true` 时，如果开启了 `enable_pfa_merge`、`enable_ifa` 或 `enable_ifa_mla`，脚本会模拟源码逻辑：

```text
head_num_size = head_num_size / g_size
seq_size = seq_size * g_size
```

所以输出里的 `normalizedInput` 可能和你原始输入不同，这是预期行为。

**最小可用示例**
```json
{
  "shape": {
    "batch_size": 2,
    "head_num_size": 16,
    "seq_size": 64,
    "seq_inner_size": 2048,
    "qk_head_size": 128,
    "v_head_size": 128
  },
  "attrs": {
    "input_dtype": "fp16",
    "layout": "BSH",
    "sparse_mode": 0,
    "pre_tokens": 2147483647,
    "next_tokens": 2147483647,
    "actual_seq_lengths": [64, 64],
    "actual_seq_lengths_kv": [2048, 2048],
    "g_size": 1,
    "head_num_ratio": 1
  },
  "platform": {
    "core_num": 48,
    "aic_num": 24,
    "l1_size": 1048576,
    "l0c_size": 262144
  },
  "flags": {
    "fa_run_flag": true,
    "enable_mask": false,
    "enable_pfa_merge": false,
    "enable_ifa": false,
    "enable_ifa_mla": false,
    "enable_pa": false,
    "split_s2": 1
  }
}
```

运行：

```bash
python3 scripts/tools/pfa_tiling_sim.py -i /tmp/pfa_case.json
```

输出里主要看这几段：

```text
normalizedInput     # 归一化后的输入
tiling              # Souter / CubeSouter / Sinner / SoftmaxSouter / splitS2 / DN / BMM check
splitCore           # 分核结果
notes               # 命中的切分路径说明
limitations         # 脚本模拟限制
```

建议必填字段

{
  "shape": {
    "batch_size": 1,
    "head_num_size": 1,
    "seq_size": 1024,
    "seq_inner_size": 1024,
    "qk_head_size": 128,
    "v_head_size": 128
  },
  "attrs": {
    "input_dtype": "fp16",
    "layout": "BNSD"
  },
  "platform": {
    "core_num": 64,
    "aic_num": 32
  }
}
必填含义：

字段	说明
shape.batch_size	B
shape.head_num_size	Q head 数
shape.seq_size	S1 / query seq
shape.seq_inner_size	S2 / KV seq
shape.qk_head_size	Q/K head dim
shape.v_head_size	V head dim
attrs.input_dtype	fp16 / bf16 / int8 等
attrs.layout	BSH / BSND / BNSD / TND
platform.core_num	AIV 核数，分核主要依赖它
platform.aic_num	AIC 核数，部分策略/对齐分析会用
shape 也支持别名：

batch_size      -> b
head_num_size   -> n
seq_size        -> s1
seq_inner_size  -> s2
qk_head_size    -> d
v_head_size     -> dv
条件必填字段

这些字段不是每个 case 都必须填，但相关功能打开时建议填：

字段	什么时候需要
attrs.actual_seq_lengths	有实际 S1 长度，或 batch 内长度不一致时
attrs.actual_seq_lengths_kv	有实际 S2 长度，或 batch 内长度不一致时
attrs.pre_tokens / attrs.next_tokens	分析 sparse/mask 裁剪逻辑时
attrs.sparse_mode	开启 mask/sparse 相关分析时
attrs.actual_shared_prefix_len	KV prefix 场景
attrs.g_size	GQA / PFA merge / IFA / MLA 场景
attrs.head_num_ratio	Q/KV head ratio 不为 1 时
attrs.pa_layout_type / attrs.block_size	PA 场景
flags.enable_mask	有 attention mask 时
flags.enable_pa	Page Attention 场景
flags.enable_pfa_merge	PFA merge 场景
flags.enable_ifa / flags.enable_ifa_mla	IFA / MLA 场景
flags.enable_pfa_rope	rope 场景
flags.enable_perblock_quant	perblock quant 场景
flags.enable_matmul_norm	matmul norm 模板分析
非必填字段及默认值

字段	默认值
shape.q_head_size	默认等于 qk_head_size
attrs.output_dtype	fp16
attrs.inner_precise	high_performance
attrs.sparse_mode	0
attrs.pre_tokens	2147483647
attrs.next_tokens	2147483647
attrs.actual_seq_lengths	默认 [seq_size] * batch_size
attrs.actual_seq_lengths_kv	默认 [seq_inner_size] * batch_size
attrs.actual_shared_prefix_len	0
attrs.g_size	1
attrs.head_num_ratio	1
attrs.pa_layout_type	0
attrs.block_size	128
attrs.aligned_head_size	默认按 qk_head_size 16 对齐
platform.core_num	48
platform.aic_num	24
platform.l1_size	0
platform.l0c_size	0
flags.fa_run_flag	true
flags.split_s2	1
flags.normalize_gs1_merge	true
其他 enable_xxx	false
最小推荐模板

{
  "shape": {
    "batch_size": 1,
    "head_num_size": 1,
    "seq_size": 1024,
    "seq_inner_size": 1024,
    "qk_head_size": 128,
    "v_head_size": 128
  },
  "attrs": {
    "input_dtype": "fp16",
    "layout": "BNSD"
  },
  "platform": {
    "core_num": 64,
    "aic_num": 32
  }
}
这个最小模板可以跑出基本的 Souter/Sinner/SoftmaxSouter/splitS2 和分核结果。需要精确复现源码 case 时，再补齐 actual_seq_lengths、mask/sparse、PA/IFA/merge 等相关字段。