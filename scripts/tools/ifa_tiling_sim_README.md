# IFA Tiling V2 Simulator

`ifa_tiling_v2_sim.py` 是一个用于分析 `incre_flash_attention_tiling_v2.cpp` faRun v2 主路径切块和分核行为的 Python 辅助脚本。

脚本根据输入的 shape、actual seq length、稀疏/mask/PSE/FlashDecode 等开关，输出 IFA v2 的 host 侧 `Souter`、`Sinner`、`SoftmaxSouter`、`splitS2`、N-B-S 分核结果、softmax 空间信息，并可生成每个 candidate cube core 任务量的 SVG 条形图。

## 快速开始

生成输入模板：

```bash
python3 scripts/tools/ifa_tiling_v2_sim.py --example > /tmp/ifa_case.json
```

编辑 `/tmp/ifa_case.json` 后运行：

```bash
python3 scripts/tools/ifa_tiling_v2_sim.py -i /tmp/ifa_case.json
```

生成紧凑 JSON：

```bash
python3 scripts/tools/ifa_tiling_v2_sim.py -i /tmp/ifa_case.json --compact
```

生成负载均衡 SVG 图：

```bash
python3 scripts/tools/ifa_tiling_v2_sim.py -i /tmp/ifa_case.json --plot /tmp/ifa_load_balance.svg
```

也可以继续使用单参数快速输入：

```bash
python3 scripts/tools/ifa_tiling_v2_sim.py \
  --batch-size 2 --q-heads 32 --kv-heads 4 --head-dim 64 \
  --q-seq 1 --kv-seq 4096 --aic-num 24
```

## 输入结构

推荐使用和 `pfa_tiling_sim.py` 一致的四段 JSON：

```json
{
  "shape": {},
  "attrs": {},
  "platform": {},
  "flags": {}
}
```

- `shape`：`batch_size`、`q_heads`、`kv_heads`、`q_seq`、`kv_seq`、`head_dim`
- `attrs`：`input_dtype`、`block_type_size`、`sparse_mode`、`pre_tokens`、`next_tokens`、`actual_q_lens`、`actual_kv_lens`、`actual_shared_prefix_len`
- `platform`：`aic_num`、`aiv_num`、`core_num`
- `flags`：`enable_pse_shift`、`is_gqa`、`fa_run_gs`、`enable_mask`、`is_pfa`、`force_flash_decode`

`force_flash_decode` 可设为 `true`、`false` 或 `null`。设为 `null` 时，脚本按源码风格的 FlashDecode 判定逻辑自动决定。

## 输出重点

常用字段：

```text
normalizedInput
tiling
softmax
batchLoopInfo
splitCore
flashDecodeSplitS2
```

`tiling` 中包含：

- `Souter`
- `Sinner`
- `SinnerForTilingKey`
- `sInnerLoopTimes`
- `sInnerSizeTail`
- `sInnerSizeAlign`
- `SoftmaxSouter`
- `SoftmaxSinnerAlign`
- `splitS2`

`splitCore` 中包含：

- `candidate_cube_cores`
- `used_core_num`
- `actualCoreNums`
- `coreTaskBlocks`
- `candidateCoreTaskBlocks`
- `loadBalance`
- `core_ranges`
- `per_row_assignment`

`candidateCoreTaskBlocks` 会把未使用的 candidate core 以 `0` 填充，SVG 中显示为灰色 idle core。

## 限制

- 不调用真实 CANN tiling API
- softmax tmp size 输出源码 API 公式，不计算 AscendC 内部真实返回值
- 分核权重是 host 侧估算 task blocks，用于理解源码切分策略，不等价于 kernel 实际耗时
