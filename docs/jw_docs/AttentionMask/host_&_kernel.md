可以。以这条 arch35 infer/common kernel 链为例，host tiling 侧和 kernel 侧对 attention mask 的分工，其实很清楚:

一句话概括

host tiling 侧主要做“判定和规划”：
检查 mask 合不合法，决定 sparse 语义，估算每个 batch 实际会算多少块，并把这些信息编码进 tiling / inputParamsRegbase。

kernel 侧主要做“执行和落地”：
按当前 batch / 当前 s1,s2 block，把 sparse 语义转成真正的 pre/next、循环范围、mask 偏移和 merge 行为，然后把 mask 从 GM 拷到 UB 并参与 softmax 前处理。

先看两边之间传了什么

host 最后会把这些字段写进 tiling：
prompt_flash_attention_tiling_v2.cpp (line 4993)
prompt_flash_attention_tiling_v2.cpp (line 5006)
prompt_flash_attention_tiling_v2.cpp (line 5016)
fused_infer_attention_score_impl.cpp (line 1814)

kernel 初始化时把这些字段搬进 sharedParams/attenMaskInfo：
flash_attention_noquant_kernel_base.h (line 523)
flash_attention_noquant_block_vec_infer.h (line 168)

桥接字段最重要的是这些：

preTokens
nextTokens
sparseType
attenMaskCompressMode
attenMaskShapeType
attenMaskS1Size
attenMaskS2Size
bandIndex
prefixNAddr / prefix 长度
Host Tiling 侧通过 attention mask 做了什么

校验“有没有 mask、mask 类型对不对、shape 对不对”
prompt_flash_attention_tiling_v2.cpp (line 1547) 会按 sparse mode 校验 mask 维度和大小。
比如：
默认/全 mask 模式下，mask 可以是 [B,QS,KVS] / [1,QS,KVS] / [B,1,QS,KVS]
LEFT_UP/RIGHT_DOWN/BAND 压缩 mask 模式下，要求是固定优化尺寸的压缩 mask
某些 sparse mode 下 mask 不能为空
这些交叉检查在：
prompt_flash_attention_tiling_v2.cpp (line 2774)
把用户侧 sparse 语义归一化
host 先把 attr 里的 sparseMode/preTokens/nextTokens 规整成内部统一语义：
prompt_flash_attention_tiling_v2.cpp (line 1598)
这里会做几件事：

截断 preTokens/nextTokens 到内部上限
LEFT_UP 直接改成 pre=INF, next=0
RIGHT_DOWN 只先把 pre=INF，真正的 next 留给后面按 batch 长度再算
ALL_MASK 变成 pre=INF, next=INF
BAND 保留原始窗口语义
先在 host 上按 batch 推导一遍 preTokensPerBatch/nextTokensPerBatch
这是和 kernel 最像的一段，但目的不是执行，而是做 workload 估算、needInit 判断、分块规划：
prompt_flash_attention_tiling_v2.cpp (line 3054)
这里 host 会按每个 batch 的实际 actualSeqLengths / actualSeqLengthsKV / prefixLen 计算：

RIGHT_DOWN: next = actualKV(+prefix) - actualQ
BAND: pre = input.pre - actualKV - prefix + actualQ
BAND: next = input.next + actualKV + prefix - actualQ
然后判断这一批是否会出现“上方/下方无效行”，从而设置 needInit：
prompt_flash_attention_tiling_v2.cpp (line 3072)

用 mask/sparse 信息估算分块量和是否有 row-invalid
host 会进一步修正“逻辑上的左上窗口参数”，用于估算 block 数和核内切分：
prompt_flash_attention_tiling_v2.cpp (line 3754)
这一步不是读 mask 数据，而是为了：

算总 block 数
决定 sOuter/sInner 怎么切
判断输出是否要预初始化
让 tiling 更贴近真实有效区域
选 coarse-grained sparse 类别，用于 tiling key / 模板选择
host 通常会同时写两类“模式信息”：
一类是细粒度 attenMaskCompressMode：
prompt_flash_attention_tiling_v2.cpp (line 4807)
fused_infer_attention_score_impl.cpp (line 1661)

它把 sparse mode 映射成：

0: no-compress
1: left-up causal
2: right-down causal
3: band
另一类是更粗的 sparseType：
prompt_flash_attention_tiling_v2.cpp (line 2648)
fused_infer_attention_score_impl.cpp (line 1673)

它更多用于：

tiling key 选择
模板选择
走 ALL / CAUSAL / BAND 哪一大类实现
把 mask 元信息写进 kernel 能直接消费的寄存结构
比如：
attenMaskShapeType
attenMaskS1Size
attenMaskS2Size
preTokens
nextTokens
bandIndex
见：
fused_infer_attention_score_impl.cpp (line 1637)
fused_infer_attention_score_impl.cpp (line 1814)

注意一点：
在这条 infer host 链里，bandIndex 基本被写成默认 0，没有训练侧那种复杂 hybrid band batch 选择：
prompt_flash_attention_tiling_v2.cpp (line 5002)
fused_infer_attention_score_impl.cpp (line 1857)

Kernel 侧通过 attention mask 做了什么

把 host 下发的 mask 元信息装入运行时结构
kernel base 初始化时把 host 传下来的字段装进 attenMaskInfo：
flash_attention_noquant_kernel_base.h (line 523)

每个 batch 再精确计算一遍 preTokensPerBatch/nextTokensPerBatch
这是 kernel 真正执行前的 runtime 版本：
infer_flash_attention_kvcache.h (line 249)
infer_flash_attention_sparse.h (line 25)

它和 host 的区别是：

host 为了估算和切分而算
kernel 为了当前 block 真正执行而算
这里会按 compressMode 把窗口重新翻译成当前 batch 的左上语义：

RIGHT_DOWN 变成 pre=INF, next=actualS2-actualS1(+prefix)
BAND 变成与当前 actualS1/actualS2 对齐的 pre/next
MLA / GQA / prefix 会再做额外修正
用这些 pre/next 直接裁掉不需要算的 s1/s2 块
这一步是 kernel 真正减少计算量的关键：
infer_flash_attention_kvcache.h (line 153)
infer_flash_attention_kvcache.h (line 669)
效果是：

修正 actualS1Size
计算 s2LineStartIdx/s2LoopEndIdx
某些 gS1 块直接 continue
所以 attention mask 在 kernel 里首先不是“加到分数上”，而是先决定“哪些块根本不用算”。
对每个 block 算真实的 mask offset
这一步在：
attenmask.h (line 340)
它根据：

compressMode
当前 s1Offset
当前 s2Offset
actualS1Size/actualS2Size
vecCoreOffset
prefix 信息
算出：

当前主 mask 的 maskOffset
若需要，还算第二块 attenMaskOffsetPre
判断当前 block 要不要读第二块 mask、要不要 merge
这一步在：
attenmask.h (line 112)
会把当前 block 归类成：

NO_NEED_COMPUTE_MODE
CAUSAL_OR_NEXT_ONLY_MODE
PRE_ONLY_MODE
PRE_AND_NEXT_MODE
PREFIX_N_COMPUTE_MODE
PREFIX_COMPUTE_MODE
含义很直接：

只读主块
只读 pre 块
两块都读并合并
整块其实全 0，不必复杂处理
真正把 mask 从 GM 搬到 UB，并做 band/prefix 合并
普通 copy-in 在：
attenmask.h (line 532)
band / prefix 的二次读取与 merge 在：
attenmask.h (line 540)
attenmask.h (line 567)

vec block 把这件事接到真正的 ProcessVec1 前：
flash_attention_noquant_block_vec_base.h (line 760)

所以 kernel 侧对 mask 的最终动作是：

读 mask
必要时读第二块 pre/prefix mask
merge 成当前 block 真正使用的逻辑 mask
然后再参与 softmax 前的计算
最值得记住的边界

host tiling 侧不处理 mask 数据本身，只处理 mask 的“元信息和后果”：

合法性
语义归一化
计算量估算
模板/tiling 决策
参数打包
kernel 侧才处理 mask 数据本身：

当前 block 该看哪一块 mask
偏移怎么算
是否要读第二块
怎么 merge
怎么据此裁剪和执行
为什么 host 和 kernel 都算了一遍 pre/next

这是最容易混淆的点。

host 那一遍是“预测版”：
用于切核、算 block 数、判断 needInit、选择 tiling。

kernel 那一遍是“执行版”：
用于当前 batch / 当前 block 的真实循环边界和 mask 偏移。

两边公式看起来相似，但目的不同，所以不是重复劳动。

举例：
会分两种情况，结论不是一刀切。

结论

如果是“普通显式 mask”路径，也就是 NO_COMPRESS_MODE / ALL_MASK 这类，masked 部分通常会先算 q * K^T，然后在 vec/softmax 前再用 mask 把对应分数改成极小值。
如果是“压缩 sparse mask”路径，比如 LEFT_UP_CAUSAL / RIGHT_DOWN_CAUSAL / BAND / PREFIX，那么窗口外整块通常不会算 q * K^T；但窗口边界块内部那些被 mask 的元素，通常还是会先算，再在 vec 阶段被 mask 掉。
为什么这么说

kernel 会先决定 s2 哪些块要参与 bmm1
infer_flash_attention_kvcache.h (line 668) 里 ComputeS2LoopInfo(...) 会根据 preTokensPerBatch / nextTokensPerBatch 算：
s2LineStartIdx
s2LineEndIdx
s2LoopEndIdx
这里如果是 sparse/causal/band 语义，很多完全落在窗口外的 s2 block 会直接被裁掉，不进入后面的 IterateBmm1(...)。

bmm1(q * K^T) 真正用的就是这个裁剪后的 s2 block 范围
runInfo.s2StartIdx = runParam.s2LineStartIdx 在：
infer_flash_attention_kvcache.h (line 870)
而 IterateBmm1Nd(...) 里实际取 K 的起点就是：
flash_attention_noquant_block_cube.h (line 1580)

也就是：
startPos.s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * s2BaseSize

这说明：

被 s2LoopInfo 整块裁掉的区域，连 K 都不会搬，当然也不会算 q * K^T
进入循环的块，会正常做 bmm1
进入循环后的块，mask 是在 vec 阶段生效的，不是 cube 阶段
你能看到 vec 里会把已经算出来的分数和 mask 一起做 Select(...)：
vf_mul_sel_softmaxflashv2_cast_nz_dn.h (line 131)
vf_mul_sel_softmaxflashv2_cast_nz_dn.h (line 495)
这里是先 LoadAlign(src...) 读 qK^T 结果，再 LoadAlign(... mask ...)，然后：

Select(src, src, vreg_min, preg_compare)
也就是把 masked 元素替换成极小值，再进 softmax。

所以可以把它理解成两层粒度

块级裁剪
由 ComputeS2LoopInfo(...) 完成。
对 sparse/causal/band/prefix，完全无效的 s2 block 不会做 q * K^T。

块内元素级 mask
由 vec 阶段完成。
对已经进入计算的 block，哪怕其中一部分元素最终会被 mask，这部分元素通常也已经做了 q * K^T，只是随后被 Select 改成极小值。

对应到你最关心的几类场景

纯显式 dense mask，没做 sparse 压缩语义：
基本是“全算 q * K^T，再 mask”。

left-up causal / right-down causal / band / prefix：
是“先按窗口裁掉整块，再对边界块内部做 mask”。

换句话说：
attention mask 在这里不是纯后处理，也不是纯前裁剪，而是“块级前裁剪 + 块内后屏蔽”。