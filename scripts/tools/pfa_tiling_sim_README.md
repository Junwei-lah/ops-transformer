# PFA Tiling Simulator

`pfa_tiling_sim.py` 是一个用于分析 `prompt_flash_attention_tiling_v2.cpp` 主路径切块和分核行为的 Python 辅助脚本。

脚本根据输入的 shape、layout、dtype、稀疏模式、PA/IFA/MLA/PFA Merge/DN 等开关，输出 PFA v2 的 host 侧切分信息、分核结果、BMM check 参数计划，并可生成每个 cube core 任务量的 SVG 条形图。

## 能力范围

- 计算 `Souter`、`CubeSouter`、`Sinner`、`SoftmaxSouter`、`splitS2`
- 模拟 DN 对 `Sinner` 的二次调整
- 模拟 v2 主路径 `SPLIT_NBS_CUBE` 的分核结果
- 输出 BMM1 / BMM2 check 的 shape、orgShape、fixSplit 计划
- 输出每个 cube core 的任务量和负载均衡指标
- 生成每个 candidate cube core 的任务量条形图，idle core 以 0 任务灰色柱显示

注意：脚本不会调用 CANN 的 `MatmulApiTiling::GetTiling`，因此 BMM check 输出的是参数计划，不是真实平台 tiling 结果。

## 快速开始

生成一份输入模板：

```bash
python3 scripts/tools/pfa_tiling_sim.py --example > /tmp/pfa_case.json
```

编辑 `/tmp/pfa_case.json` 后运行：

```bash
python3 scripts/tools/pfa_tiling_sim.py -i /tmp/pfa_case.json
```

生成紧凑 JSON：

```bash
python3 scripts/tools/pfa_tiling_sim.py -i /tmp/pfa_case.json --compact
```

生成负载均衡 SVG 图：

```bash
python3 scripts/tools/pfa_tiling_sim.py -i /tmp/pfa_case.json --plot /tmp/pfa_load_balance.svg
```

保存 JSON 输出：

```bash
python3 scripts/tools/pfa_tiling_sim.py -i /tmp/pfa_case.json > /tmp/pfa_result.json
```

也可以从 stdin 输入：

```bash
cat /tmp/pfa_case.json | python3 scripts/tools/pfa_tiling_sim.py
```

## 输入结构

输入是 JSON，推荐分成四段：

```json
{
  "shape": {},
  "attrs": {},
  "platform": {},
  "flags": {}
}
```

- `shape`：B/N/S1/S2/D/DV 等核心 shape
- `attrs`：dtype、layout、sparse mode、actual seq length、GQA/PA 等属性
- `platform`：平台核数和 L1/L0C buffer 信息
- `flags`：PFA/IFA/MLA/PA/mask/pse/perblock quant 等开关

详细字段说明见：

```text
scripts/tools/pfa_tiling_args_explain.md
```

## 输出重点

输出 JSON 中常用字段如下：

```text
normalizedInput
```

归一化后的输入。开启 `enable_pfa_merge`、`enable_ifa` 或 `enable_ifa_mla` 且 `normalize_gs1_merge=true` 时，脚本会模拟源码中的 G/S1 归一化，输出 shape 可能不同于原始输入。

```text
tiling
```

切块结果，包括：

- `Souter`
- `CubeSouter`
- `Sinner`
- `SoftmaxSouter`
- `splitS2`
- `enableDN`
- `dnAdjusted`
- `bmm_checks`

```text
splitCore
```

分核结果，包括：

- `split_core_mode`
- `candidate_cube_cores`
- `cubeUsedCores`
- `actualCoreNums`
- `coreTaskBlocks`
- `candidateCoreTaskBlocks`
- `loadBalance`
- `coreRanges`

其中：

```text
coreTaskBlocks
```

只包含实际分到任务的 cube core。

```text
candidateCoreTaskBlocks
```

包含全部 candidate cube core。未分到任务的 cube core 用 `0` 填充，可视化时显示为灰色 idle core。

## 负载均衡图

使用 `--plot` 生成 SVG：

```bash
python3 scripts/tools/pfa_tiling_sim.py -i /tmp/pfa_case.json --plot /tmp/pfa_load_balance.svg
```

图中：

- 横轴是 cube core id
- 一个 cube core 对应两个 AIV core，即 `[2*i, 2*i+1]`
- 柱高表示该 cube core 分到的有效 task blocks
- 灰色柱表示 idle core，任务量为 0
- 红色虚线是 mean line
- 蓝色虚线是 target line

`loadBalance` 中会给出：

```text
usedCubeCores
candidateCubeCores
idleCandidateCubeCores
totalTaskBlocks
meanTaskBlocks
targetTaskBlocks
maxTaskBlocks
minTaskBlocks
coefficientOfVariation
maxOverMean
minOverMean
rating
interpretation
```

`meanTaskBlocks` 当前按全部 candidate cube core 计算，因此 idle core 会计入负载均衡分析。

## 常见命令

查看帮助：

```bash
python3 scripts/tools/pfa_tiling_sim.py --help
```

生成 example 并直接运行：

```bash
python3 scripts/tools/pfa_tiling_sim.py --example | python3 scripts/tools/pfa_tiling_sim.py
```

生成 example、运行、并画图：

```bash
python3 scripts/tools/pfa_tiling_sim.py --example \
  | python3 scripts/tools/pfa_tiling_sim.py --plot /tmp/pfa_load_balance.svg
```

## 限制

- 不调用真实 CANN `MatmulApiTiling::GetTiling`
- BMM1 / BMM2 结果只表示源码里尝试设置的 tiling 参数计划
- 当前主要覆盖 PFA v2 主路径 `SPLIT_NBS_CUBE`
- 分核权重是 host 侧估算值，用于理解源码分核策略，不等价于实际 kernel 耗时
