# FlashAttention Softmax 近似优化说明

## 1. 背景

本次改动围绕 IFA/FA 的 softmax 路径做近似优化。

原始 online softmax 的核心递推为：

- `m_j = max(m_{j-1}, rowmax(tile_j))`
- `l_j = l_{j-1} * exp(m_{j-1} - m_j) + sum(exp(tile_j - m_j))`

其中：

- `tile_j` 表示第 `j` 个 `s2` 分块
- `m_j` 表示处理到第 `j` 个分块后的逐行最大值
- `l_j` 表示处理到第 `j` 个分块后的逐行 softmax 分母累积

在数值观察中，很多 query 行的最大值主要出现在 `j=1` 和 `j=i` 这两类分块上，因此提出了如下近似思路：

- 在进入 softmax 前，预先给出一个逐行的 `m_pre`
- `j=1` 时，不再仅用本块的 rowmax 初始化，而是执行 `max(m_pre, rowmax(tile_1))`
- 中间分块 `j=2 ... i-1` 近似认为最大值不再更新，即令 `m_j = m_{j-1}`
- 最后一块 `j=i` 仍保留原始的完整更新

这套方法的目标不是完全保持原始算法的逐块最大值更新方式，而是用较小的数值改动换取后续进一步优化 softmax 的空间。

## 2. 本次实现的总体思路

这次代码实现先完成了“控制流接入”和“数值域恢复”两件事：

1. 给 antiquant infer 路径增加一个近似 softmax 开关
2. 允许 `j=1` 从外部预计算的 `m_pre` 启动
3. 对中间分块保留“沿用旧 `m`”的语义
4. 在 Vec2 路径中同步恢复对应的 `PV` 累积语义

需要特别说明的是：

- 当前实现已经把近似 softmax 的流程接入主 kernel
- 当前实现还没有把 VF 内核里“中间块 rowmax 的内部归约计算”完全裁掉
- 也就是说，这一版优先保证语义接通和状态一致性，后续还可以继续下沉到 VF helper 做真正的快路径裁剪

## 3. 算法改进如何映射到代码

### 3.1 开关与输入

实现位置：

- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_kernel.h`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h`

本次使用 `inputParamsRegbase.rsv1` 的 bit0 作为开关：

- `0`：走原始 online softmax
- `1`：走近似 softmax 路径

当开关打开时：

- `workspace` 的起始地址被解释为一段预计算好的逐行 `m_pre` buffer
- 该 buffer 按 `softmaxLse` 相同的行布局读取

对应代码：

- `flash_attention_score_antiquant_kernel.h` 中将 `rsv1 bit0` 写入 `constInfo.rsvd1`
- `flash_attention_score_antiquant_block_vec.h` 中将 `workspace` 绑定为 `softmaxApproxMaxGm`

### 3.2 j=1 分块

原始算法中，`j=1` 一般以当前块的 rowmax 作为 softmax 的第一次初始化。

本次实现中，`j=1` 改为：

1. 从 `softmaxApproxMaxGm` 读取外部的逐行 `m_pre`
2. 将 `sumUb` 清零
3. 进入 update 路径
4. 让 update kernel 内部完成 `max(m_pre, rowmax(tile_1))`

这样就实现了用户提出的：

- “`j=1` 的块需要再加上一次 m 值比较”
- “比较对象是提前算出来的 m”

对应 helper：

- `InitApproxSoftmaxFirstTile(...)`

### 3.3 中间分块 j=2 ... i-1

用户提出的近似策略是：

- 非 `j=1`、非 `j=i` 的块令 `m_j = m_{j-1}`

为了在现有实现基础上尽量少改主干逻辑，这次采用了“先走现有 update，再恢复回旧 m 域”的方式。

具体步骤如下：

1. 在进入当前中间块前，先把旧的 `maxUb` 备份到 `approxPrevMaxBuf`
2. 调用现有 `ProcessVec1Vf<..., true>` 和 `UpdateExpSumAndExpMax(...)`
3. 现有 update 路径会临时得到新域下的：
   - `m_new`
   - `l_new`
   - `exp(m_old - m_new)`
4. 然后执行恢复：
   - 将 `sum` 除以 `exp(m_old - m_new)`，恢复回旧 `m` 域
   - 将 `maxUb` 恢复为 `m_old`

这样，逻辑上就等价于：

- 本块处理完成后仍然沿用 `m_{j-1}`

对应 helper：

- `RestoreApproxSoftmaxState(...)`

### 3.4 最后一块 j=i

最后一个分块不走近似恢复逻辑，仍然保留原始 online softmax 的完整更新。

这样做的原因是：

- 最后一块仍然是更可能出现真实最大值的位置
- 保留最后一块的完整更新，数值上更稳妥

## 4. Vec2 是如何配套修改的

softmax 不只影响 `sum/max`，还会影响 `PV` 累积。

如果中间块在 Vec1 中恢复到了旧 `m` 域，而 Vec2 还停留在新 `m` 域，那么后续块的累积就会不一致。

因此本次在 `ProcessVec2(...)` 中也加入了配套处理：

- 中间块调用 `FlashUpdateNew(...)` 后
- 再额外执行一次 `LastDivNew(..., expUb)`

这里利用的是：

- `expUb` 中保存了 `exp(m_old - m_new)`

通过这一步，可以把 Vec2 的结果重新拉回旧 `m` 域，使它与 Vec1 中恢复后的 `sum/max` 语义一致。

## 5. 新增/修改的主要代码点

### 5.1 核心文件

- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_kernel.h`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h`

### 5.2 关键新增项

在 `flash_attention_score_antiquant_block_vec.h` 中新增了以下内容：

- `APPROX_SOFTMAX_PREMAX_FLAG`
- `softmaxApproxMaxGm`
- `approxPrevMaxBuf[3]`
- `UseApproxSoftmax(...)`
- `IsApproxFirstTile(...)`
- `IsApproxMiddleTile(...)`
- `InitApproxSoftmaxFirstTile(...)`
- `RestoreApproxSoftmaxState(...)`

### 5.3 关键流程改动

`ProcessVec1(...)` 中新增：

- `j=1` 的 `m_pre` 初始化逻辑
- 中间块旧 `m` 的备份
- 中间块更新后的恢复逻辑

`ProcessVec2(...)` 中新增：

- 中间块执行 `FlashUpdateNew(...)` 后，补做一次恢复到旧 `m` 域的处理

## 6. 其它一并提交的改动

与本次提交一并进入仓库的文件还包括：

- `attention/common/op_kernel/arch35/flash_attention_kvsame_bn2gs1s2.h`
- `attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp`
- `attention/incre_flash_attention/op_kernel/arch32/incre_flash_attention_preload.h`
- `attention/incre_flash_attention/op_kernel/arch35/kernel/incre_flash_attention_antiquant_Bbn2s2_Us2_regbase.h`

这些文件的改动主要包括：

- 命名调整
- 注释补充
- 流水和分块处理逻辑的可读性增强
- 与增量 attention 路径相关的小幅配套修改

本说明文档的重点仍然是 softmax 近似优化本身。

## 7. 当前实现的边界与限制

本次实现已经完成了以下事情：

- 将外部 `m_pre` 接入 softmax 主流程
- 将 `j=1 / 中间块 / 最后一块` 三类分块区分处理
- 在 Vec1 与 Vec2 两条路径上保持近似语义的一致性

但当前仍有以下限制：

1. 还没有彻底删除中间块在 VF 内核内部的 rowmax 归约计算  
   当前做法是“先复用原始 update 逻辑，再恢复回旧 `m` 域”

2. `workspace` 中的 `m_pre` 需要由外部提前准备好  
   当前实现只负责消费该 buffer，不负责生成它

3. 这是近似优化，不是严格等价替换  
   设计目标是贴合“最大值主要出现在首块和尾块”的经验观察

## 8. 后续可继续优化的方向

如果后续继续推进这条方案，可以考虑：

1. 在 VF helper 中新增真正的“固定旧 `m`”快路径  
   让中间块直接跳过 rowmax/update 中不必要的分支

2. 将 `m_pre` 的生成流程也纳入同一条 kernel/host 配置链路  
   降低使用门槛

3. 对不同稀疏模式、不同布局、不同量化配置分别做数值验证  
   评估该近似对精度和性能的实际收益

## 9. 本次提交信息

- 提交号：`f85f07b6d6f40d3c2957b2881c06abc1513c4574`
- 提交信息：`Update flash attention kernels and tiling`

本说明文档用于记录其中与 softmax 近似优化相关的设计和实现细节。