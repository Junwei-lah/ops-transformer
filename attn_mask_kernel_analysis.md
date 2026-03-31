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
