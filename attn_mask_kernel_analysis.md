# PFA 场景下 attn mask kernel 侧逻辑分析（BNSD=8,8,2432,64，BF16，下三角）

## 调用链路确认
- host 侧：`prompt_flash_attention_tiling_v2.cpp`
- kernel 入口：`prompt_flash_attention_entry_regbase.h` BF16 分支，走 `FlashAttentionNoQuantKernelInfer`
- kernel 主流程：`Process() -> ProcessMainLoop() -> IterateBmm1 / ProcessVec1 / IterateBmm2`
- attn mask 生效点：`ProcessVec1Nd()` 中 `hasAtten == true` 分支，随后进入 `ProcessVec1Vf`，再进入 `ProcessVec1Update...`（update 场景）

## host 侧与 mask 相关关键点
1. **BF16 mask dtype 限制**：BF16 输入仅允许 `bool/int8/uint8` mask。
2. **hasAttenMask 模板位**：`enableMask=true` 时置 `hasAttenMask=true`。
3. **稀疏类型映射**：
   - `sparseMode=2(left_up)` 直接映射为 `PFA_CAUSAL`。
   - `sparseMode=3(right_down)` 在 `qS==S2` 且非 PA 时也会映射到 `PFA_CAUSAL`，否则 `PFA_BAND`。

## kernel 侧 mask 核心逻辑
1. **流水并行**：`ProcessMainLoop()` 中按 taskId 做 4 级流水，`taskId>0` 时进入 `ProcessVec1`。
2. **mask copy-in**：`ProcessVec1Nd()` 在 `hasAtten==true` 时，将 GM mask 拷入 UB（`AttenMaskCopyIn` 或 MLA/GQA 专用 copy-in）。
3. **band 压缩模式**：若 `compressMode == BAND_MODE`，会额外取 preMask 并 `MergeBandModeMask`。
4. **mask 在 VF 中的作用**：
   - 在 `ProcessVec1UpdateImpl64VF` 里，先做 scale/pse，再根据 `preg_compare` 执行 `Select(vreg_sel, vreg_min, vreg_input_x, preg_compare)`；
   - 被 mask 的位置会写入 `minValue`（负无穷近似），从而在后续 softmax 中贡献约 0。

## 关于 `ProcessVec1UpdateImpl64VF` 是否命中
- `ProcessVec1Update` 只有在 `oriNRange == GT_0_AND_LTE_64` 时才会调 `ProcessVec1UpdateImpl64`。
- `ProcessVec1Nd` 是按 `runInfo.s2RealSize` 分支：
  - `<=64` 才会进入该 64 分支；
  - `==128` 走 `Impl128`；`(64,128)` 走 `GeneralImpl128`；`(128,256]` 走 256 分支。
- 对 `S2=2432`：
  - 若 `s2BaseSize=128`，2432 可整除 128，通常各 tile 都是 128；
  - 若 `s2BaseSize=256`，尾块是 128。
  - 两种常见情形都**不易进入 <=64 分支**，因此 `ProcessVec1UpdateImpl64VF` 往往不是主路径。

## 实际定位建议
- 如要确认线上确实命中 `Impl64VF`，建议打印/抓取 `runInfo.s2RealSize` 与 `s2BaseSize`。
- 若目标是“下三角 mask 语义正确性”，应优先检查：
  1) host 下发的 `sparseMode` 与 `sparseType` 是否为 causal；
  2) `attenMaskInfo.compressMode` 与 mask shape 是否匹配；
  3) `AttenMaskCopyIn` 后 UB 内容与 `Select(minValue, x, preg_compare)` 对应关系。


## 更正：oriNRange 不是 D（hidden dim）
- 你给的输入是 `BNSD=[8,8,2432,64]`，其中 `D=64` 是 head_dim。
- 但在这条实现里，`oriNRange`/`originN` 对应的是 softmax 最后一维的 **N 方向列数**（即本轮 `s2RealSize`），不是 `D`。
- 代码证据：`ProcessVec1Nd` 调 `ProcessVec1Vf` 时把 `runInfo.s2RealSize` 作为 `originN` 传入。随后 `ProcessVec1Update` 依据 `oriNRange` 选择 `Impl64/Impl128/...` 分支。
- 因此 `D=64` 并不直接等价于 `oriNRange==GT_0_AND_LTE_64`。

## 继续细化：ProcessVec1UpdateImpl64VF 的 mask 行为（按执行顺序）
1. **入口参数语义**
   - `pltOriginalN = originN`：真实有效列数（本 tile 的 N）。
   - `pltSrcN = s2BaseSize`：向量加载/存储按 tile 宽度进行。
   - `maskUb` 来自 `AttenMaskCopyIn`，每行按 32B 对齐，padding 值为 1（valid）。
2. **每行 i 的主流程**
   - 先 `LoadAlign(srcUb + i * s2BaseSize)` 取 bmm1 结果；
   - 做 `scale/dScaleQK` 与 `pse` 累加；
   - `hasAtten==1` 时加载 `preg_compare`，执行 `Select(vreg_sel, vreg_min, vreg_input_x, preg_compare)`；
   - 将 `vreg_sel` 回写到 `srcUb`，再对 `vreg_sel` 做 `ReduceMax`。
3. **mask 数学效果**
   - 被 mask 的元素替换为 `minValue`（负极小值），
   - 后续 `exp(x-max)` 时这部分趋近 0，达到下三角屏蔽效果。
4. **update 场景的累计逻辑**
   - 与 `inMaxUb` 比较得到 `vreg_max_new`；
   - 再基于新 max 计算 `exp` 与 `sum`，用于与历史块做稳定 softmax 累计。

## 对你这个 case（S2=2432, D=64）的直接结论
- `D=64` 只影响 Q/K matmul 的 K 维，不决定 `oriNRange`。
- `oriNRange` 是否走 64 分支，取决于 **每个 s2 tile 的 `runInfo.s2RealSize`**。
- 在常见 tiling（`s2BaseSize=128/256`）下，`S2=2432` 往往对应 `s2RealSize` 为 128 或 256（尾块可能 128），通常不会进入 `Impl64VF`。
- 若你观测到确实进入 `Impl64VF`，一般意味着该轮 tile 的 `s2RealSize<=64`（比如特殊切分/尾块/稀疏裁剪后）。

## 你要的“128 分支”分析（`oriNRange == EQ_128`）

### 1) 分支选择条件
- `ProcessVec1Nd` 中，当 `runInfo.s2RealSize == 128` 时，调用 `ProcessVec1Vf<..., EQ_128, ...>`。
- 在 `ProcessVec1Update` 里，`EQ_128` 会落到 `ProcessVec1UpdateImpl128`（不是 Impl64）。

### 2) `ProcessVec1UpdateImpl128VF` 的数据组织
- 128 列被拆成两个 64 向量块：
  - 前 64：`srcUb + i * s2BaseSize`
  - 后 64：`srcUb + floatRepSize + i * s2BaseSize`
- mask 也对应双路：`maskUb` + `maskUbUnroll`。

### 3) mask 在 128 分支里的关键执行点
- 每行 i：先完成 scale/pse，再加载两路 mask predicate（`preg_compare` 和 `preg_compare_unroll`）。
- 两路分别执行：
  - `Select(vreg_sel, vreg_min, vreg_input_x, preg_compare)`
  - `Select(vreg_sel_unroll, vreg_min, vreg_input_x_unroll, preg_compare_unroll)`
- 再回写两路 `srcUb`，并以 `Max + ReduceMax` 计算该行新 max。
- 语义与 64 分支一致：被 mask 的元素替换为 `minValue`，后续 softmax 贡献趋近 0。

### 4) 为什么你这个 case 更像走 128 分支
- 对 `S2=2432`，常见切分下每轮 `s2RealSize` 常出现 128（尤其 `s2BaseSize=128` 时全是 128；`s2BaseSize=256` 时尾块常为 128）。
- 所以相较 Impl64，`EQ_128 -> ProcessVec1UpdateImpl128VF` 更可能是主路径。

## 针对该 kernel 的结论：attenmask 实现原理总结

### A. 总体机制（不是“乘 0/1”，而是“替换为极小值”）
- 在 `ProcessVec1` 阶段（softmax 前），kernel 先把 bmm1 分数做 scale/pse。
- 然后用 mask predicate 执行 `Select(minValue, score, predicate)`：
  - 有效位置保留原始 score；
  - 无效位置改写为 `minValue`（负极大值近似）。
- 后续 softmax 计算 `exp(score - row_max)` 时，无效位置近似 0，实现“屏蔽”。

### B. 数据流分层
1. **Host 决策层**
   - 通过 `enableMask/hasAttenMask`、`sparseMode`、mask dtype/shape 等决定模板与压缩模式。
2. **Kernel copy-in 层**
   - `AttenMaskCopyIn` 将 GM mask 拷到 UB；band 压缩场景会取 preMask 并 merge。
3. **Kernel vector 计算层**
   - `ProcessVec1Update*` 在每行上应用 mask select，再做 max/sum 更新。
4. **稳定 softmax 累计层**
   - 用 `inMax/inExpSum` 与当前块结果做增量融合，保证多 s2 分块数值稳定。

### C. 对下三角 mask 的语义映射
- 下三角 mask 本质是一个逐元素有效位图（或压缩表达）。
- kernel 不关心“下三角”字面概念，而只消费 copy-in 后的 predicate：
  - predicate=true -> 保留 score；
  - predicate=false -> 写 `minValue`。
- 因此“下三角是否正确”最终等价于：copy-in 到 UB 的 predicate 是否与期望的下三角有效域一致。

### D. 你这个 case（BNSD=8,8,2432,64, BF16）的关键观察点
- 主路径大概率是 `s2RealSize==128` 对应的 `EQ_128` 分支，而不是 `Impl64`。
- 验证建议（按优先级）：
  1) 打点 `runInfo.s2RealSize` / `oriNRange`，确认分支命中；
  2) 抽样检查 `AttenMaskCopyIn` 后 UB mask 与理论下三角是否一致；
  3) 抽样检查 `Select` 后被屏蔽位是否为 `minValue`，以及 softmax 后接近 0。

## 重点补充：VF 逻辑（核心计算视角）

### 1) VF 在整条流水中的职责
- 输入：bmm1 的 score tile（`srcUb`），以及 mask/pse/drop 等辅助张量。
- 输出：
  - `dstTensor`：写给后续 mm2 的 `P = softmax(mask(score + bias))`（或其量化形态）；
  - `max/sum`：每行 softmax 统计量，用于跨 s2 分块增量融合。
- 因而 VF 的本质是“**逐行 masked-softmax + 稳定累计**”。

### 2) VF 的统一计算框架（update 分支）
对每一行 i（`m = halfS1RealSize`）可概括为：
1. 读取 score：`x <- srcUb[i, :]`
2. 缩放与偏置：`x <- scale(x) + pse`
3. 掩码：`x <- select(minValue, x, mask_pred)`
4. 行最大值：`row_max_cur <- max(x)`
5. 与历史块融合最大值：`row_max_new <- max(row_max_cur, inMax[i])`
6. 指数：`e <- exp(x - row_max_new)`
7. 行和：`row_sum_cur <- sum(e)`
8. 写回 `e`（或 cast 后格式）到 `dstTensor`，并把 `row_max_cur/row_sum_cur` 写临时缓冲，供后续累计。

### 3) 64 分支 VF（`ProcessVec1UpdateImpl64VF`）要点
- 单路向量处理一整行（<=64 有效列，按 `s2BaseSize` 对齐加载）。
- mask 核心就是一次 `Select(vreg_min, vreg_input_x, preg_compare)`。
- `preg_ori_src_n` 控制有效列掩码，避免 tail 无效列污染 max/sum。

### 4) 128 分支 VF（`ProcessVec1UpdateImpl128VF`）要点
- 双路 64 向量并行：前 64 与后 64 分开算，再合并。
- mask 也双路：`preg_compare` + `preg_compare_unroll`，分别做 `Select`。
- 行统计先 `Max(lane0, lane1)` 再 `ReduceMax/ReduceSum`，保证与 64 分支语义一致。

### 5) 为什么 VF 能保证 mask 正确性
- 无效位在 softmax 前被强制为 `minValue`，数学上等价于对这些位置施加 `-inf`。
- 只要 copy-in 的 predicate 正确（下三角有效域正确），VF 输出的 `P` 就天然满足下三角约束。

### 6) 对该 case 的结论（BNSD=8,8,2432,64）
- VF 的关键观测不是 `D=64`，而是每轮 `s2RealSize`。
- 常见情况下 `s2RealSize` 更常是 128，因此 VF 热路径更偏向 `ProcessVec1UpdateImpl128VF`。
- 若线上想定责 mask 异常，优先验证：predicate(copy-in) -> Select 后值 -> softmax 后概率，三者是否一致。

## 按你要求仅聚焦两点

### （1）kernel 侧 mask 矩阵搬入（GM -> UB）
1. **偏移计算**
   - 每个 tile 先通过 `ComputeAttenMaskOffset(...)` 计算当前块在 GM mask 的起始偏移（会考虑 prefix/fd/rope/推理模板差异）。
2. **基础搬运函数**
   - `AttenMaskCopyIn(...)` 内部调用 `BoolCopyInRegbase(...)` 把二维 mask 子块搬到 UB。
   - 搬运维度是 `(s1Size=halfS1RealSize, s2Size=s2RealSize)`，但 UB 按 `s2BaseSize` 对齐存放。
3. **对齐与尾块处理**
   - 若 `s2Size`/`totalS2Size` 是 block 对齐，走 `DataCopy`；
   - 否则走 `DataCopyPad`，自动处理尾块与 stride。
4. **压缩模式额外逻辑**
   - `BAND_MODE/RIGHT_DOWN_CAUSAL_BAND/BAND_LEFT_UP_CAUSAL` 在条件满足时会再搬一份 `attenMaskOffsetPre` 到 `attenMaskUbPre`，再 `MergeBandModeMask(...)`。
   - `PREFIX_MODE` 会根据 `computeMode` 选择主 mask / pre mask / 二者 merge。

> 结论：kernel 搬入阶段做的事是“按当前 tile 精确切 mask + 必要的 pre/next 合并 + UB 对齐铺排”，为 VF 直接消费 predicate 做准备。

### （2）`ProcessVec1UpdateImpl128VF` 的 attenmask 逻辑
1. **输入组织**
   - 128 列按两路 64 处理：`vreg_input_x`（前 64）和 `vreg_input_x_unroll`（后 64）。
   - mask 同样双路：`maskUb` 与 `maskUbUnroll`。
2. **mask 应用点（softmax 前）**
   - 先做 scale/pse；
   - 再加载双路 predicate：`preg_compare` 与 `preg_compare_unroll`；
   - 分别执行：
     - `Select(vreg_sel, vreg_min, vreg_input_x, preg_compare)`
     - `Select(vreg_sel_unroll, vreg_min, vreg_input_x_unroll, preg_compare_unroll)`
3. **行统计与稳定更新**
   - 两路 mask 后结果先 `Max` 合并，再 `ReduceMax` 得 `row_max_cur`；
   - 与 `inMax` 融合出 `row_max_new`；
   - 用 `exp(x - row_max_new)` 做行和，完成 update 形态 softmax 累计。
4. **语义本质**
   - 无效位在 softmax 前被替换为 `minValue`（负极大值近似）；
   - 因此 softmax 后这些位概率约等于 0。

> 结论：`Impl128VF` 的 mask 实现本质是“双 lane 的 Select(-inf, x, pred) + 稳定 softmax 累计”。
