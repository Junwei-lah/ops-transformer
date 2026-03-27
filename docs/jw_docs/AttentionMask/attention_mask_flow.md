先给结论：在 prompt_flash_attention_tiling_v2 这套实现里，attentionMask 不是单纯“整块都加载然后逐元素减”，而是分成两层处理：

先按 preTokens/nextTokens/sparse_mode 算出当前 Q 块真正需要看的 K 范围，整块不相交的 K tile 直接跳过，不算。
只对靠近边界的 K tile读取attentionMask`，把被 mask 的分数位置改成极小值，再进 softmax，相当于这些位置权重变 0。
关键入口在 prompt_flash_attention_tiling_v2.cpp (line 3759)、prompt_flash_attention_tiling_v2.cpp (line 2152)、prompt_flash_attention_s1s2_bns1_x910.h (line 2403)。

有几种模式

对这份 v2 代码本身来说，只支持 sparse_mode = 0~4，也就是 5 种。这个限制写死在 prompt_flash_attention_tiling_v2.cpp (line 2152)。
文档虽然写了 08，但 58 不在这个 v2 host 实现里放开。

5 种分别是：

0 = defaultMask
1 = allMask
2 = leftUpCausal
3 = rightDownCausal
4 = band
模式语义和形状检查在 prompt_flash_attention_tiling_v2.cpp (line 1297)、prompt_flash_attention_tiling_v2.cpp (line 1362)、prompt_flash_attention_tiling_v2.cpp (line 2124)。

attentionMask 怎么被使用

enableMask 条件是：attentionMask 张量存在，且 shape 非空。见 prompt_flash_attention_tiling_v2.cpp (line 3759)。

host 侧做三件事：

归一化 sparseMode / preTokens / nextTokens
检查 mask dtype/shape
把这些参数写进 tilingData，给 kernel 用
kernel 侧真正做 mask 的地方是：

算当前 Q 块需要的 K 区间：
sInnerFirstToken = clip(sOuterOffset - preTokens, 0, kvLen)
sInnerLastToken = clip(sOuterOffset + nextTokens + qBlockSize, 0, kvLen)
见 prompt_flash_attention_s1s2_bns1_x910.h (line 2403)。

如果 sInnerLastToken <= sInnerFirstToken，这个 Q 块整块跳过。

对边界 tile，再通过 SelectWithBytesMask 把被 mask 的分数替换成极小值。实现见 prompt_flash_attention_s1s2_bns1_x910_base.h (line 1320) 和 prompt_flash_attention_s1s2_bns1_x910_base.h (line 1268)。

按文档语义，mask=True 表示遮挡；代码里对应把这些位置改成 fp16/fp32 的最小值，所以 softmax 后基本就是 0。

每种模式怎么“减除 mask 块”

0 = defaultMask
这模式最特殊。

如果没传 attentionMask，host 直接把 preTokens/nextTokens 归成 INT_MAX，等价于“不做 mask，全看”。见 prompt_flash_attention_tiling_v2.cpp (line 1297)。
如果传了完整 mask，它本质上走“全尺寸 mask”逻辑。
同时它还会根据 preTokens/nextTokens 被内部再归类成 CAUSAL / ALL / BAND 三种 sparseType，见 prompt_flash_attention_tiling_v2.cpp (line 2124)。
粗粒度减块：靠 sInnerFirstToken/sInnerLastToken 只保留可能有效的 K tile。
细粒度减块：边界 tile 用完整 attentionMask 把无效元素打成极小值。
1 = allMask

必须传完整 mask。
忽略 preTokens/nextTokens。
不做压缩，直接按完整 mask 逐块搬运，边界和内部都可能读 mask。
本质上没有“对角线优化”，是最直观的全矩阵 mask。
2 = leftUpCausal

必须传压缩后的 2048x2048 mask。
host 强制改成 preTokens = INT_MAX, nextTokens = 0，见 prompt_flash_attention_tiling_v2.cpp (line 1297)。
所以当前 Q 块只保留从开头到当前对角线右边界的 K 范围。
整块减除：对角线右上方完全不相交的 K tile 直接不进循环。
块内减除：只有碰到对角线的边界 tile 才 useMask=true，见 prompt_flash_attention_s1s2_bns1_x910.h (line 2192)。
mask 偏移用 ComputeAttenMaskOffset() 按左上角对角线算，见 prompt_flash_attention_s1s2_bns1_x910_base.h (line 1320)。
3 = rightDownCausal

必须传压缩后的 2048x2048 mask。
preTokens = INT_MAX，nextTokens 不直接固定，而是按 batch 动态算：
nextTokensPerBatch = actualSeqKV + prefix - actualSeqQ
见 prompt_flash_attention_tiling_v2.cpp (line 2412) 和 prompt_flash_attention_s1s2_bns1_x910_base.h (line 2248)。
含义是把因 Q/KV 长度不等带来的“右下对齐对角线偏移”补进去。
整块减除：对角线外整块 K tile 不进。
块内减除：只对碰右下对角线的边界 tile 读压缩 mask。
4 = band

必须传压缩后的 2048x2048 mask。
这是唯一会做“两次 mask 选择”的模式。
每个 batch 会先把用户传的 pre/next 转成 batch 内真实窗口：
preTokensPerBatch = preTokens - actualSeqKV - prefix + actualSeqQ
nextTokensPerBatch = nextTokens + actualSeqKV + prefix - actualSeqQ
见 prompt_flash_attention_tiling_v2.cpp (line 2419) 和 prompt_flash_attention_s1s2_bns1_x910_base.h (line 2253)。
整块减除：窗口外的 K tile 直接跳过。
块内减除：band 有左右两个边界，所以会算两个偏移：
attenMaskOffset：处理一侧边界
attenMaskOffsetPre：处理另一侧边界
见 prompt_flash_attention_s1s2_bns1_x910_base.h (line 1320) 和 prompt_flash_attention_s1s2_bns1_x910_base.h (line 1344)。
然后用 sparseBandSelect0/1 判断当前 tile 是否碰到这两侧边界，见 prompt_flash_attention_s1s2_bns1_x910.h (line 2199)。
真正执行在 prompt_flash_attention_s1s2_bns1_x910.h (line 1052)：
第一次 ElewiseCompute(..., type=0) 减掉一侧
第二次重新搬 attenMaskOffsetPre，再 ElewiseCompute(..., type=1) 减掉另一侧
所以 band 是“双边界、两次减块”。