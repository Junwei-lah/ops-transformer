在分核前，totalBlockNumsOneHead 是在 prompt_flash_attention_tiling_v2.cpp (line 3326) 里累计出来的。它的含义是：

单个 head 下，所有 batch 总共需要计算多少个 (sOuterBlock, sInnerBlock) 块。

也就是后面做负载均衡时的“总工作量”。

计算链是这几步：

1. 对每个 batch 先把 sparse 参数统一到 left-up 视角

调用 prompt_flash_attention_tiling_v2.cpp (line 2991) 的 GetPreNextTokensLeftUp(...)：

rightDown 会转成
preTokensLeftUp = INT_MAX
nextTokensLeftUp = actualSeqLengthKV - actualSeqLength
band 会转成
preTokensLeftUp = pre - kv + q
nextTokensLeftUp = next + kv - q
其他模式直接用 baseParams 里的 preTokens/nextTokens
这样后面都能用统一的 left-up 几何去估算块数。

2. 修正“整行无效”带来的长度误差

调用 prompt_flash_attention_tiling_v2.cpp (line 3184) 的 FixParamWithRowInvalid(...)：

如果 nextTokensLeftUp < 0，说明上方有整行无效，要把 actualSeqLength 缩短
如果 actualSeqLength > actualSeqLengthKV + preTokensLeftUp，说明下方有整行无效，也要缩短
修正后得到真正用于估算工作量的 actualSeqLengthsTmp。

3. 计算单个 batch、单个 head 的块数

核心函数是 prompt_flash_attention_tiling_v2.cpp (line 3199) 的 GetCalcBlockNumsOneHead(...)。

分两种情况：

情况 A：没有 attentionMask

直接按矩形全块数算：

outerBlockNums = ceil(actualSeqLength / sOuterSize)
innerBlockNums = ceil(actualSeqLengthKV / sInnerSize)
toCalcBlockNums = outerBlockNums * innerBlockNums
也就是：

if (!isAttenMaskUsed) {
    outerBlockNums = ceil(q / sOuterSize);
    innerBlockNums = ceil(kv / sInnerSize);
    return outerBlockNums * innerBlockNums;
}
情况 B：有 attentionMask

先算满矩形块数，再减去“能整块裁掉”的块：

innerBlockNums   = ceil(actualSeqLengthKV / sInnerSize)
outerBlockNums   = ceil(actualSeqLength   / sOuterSize)

blockSeqLengthKV = innerBlockNums * sInnerSize
blockSeqLength   = outerBlockNums * sOuterSize

toCalcBlockNums = innerBlockNums * outerBlockNums
                - GetCutBlockNums(..., nextTokensLeftUp)
                - GetCutBlockNums(..., blockSeqLengthKV - blockSeqLength + preTokensLeftUp)
对应源码就在 prompt_flash_attention_tiling_v2.cpp (line 3208)。

含义是：

先把整个 Q x KV 的块矩阵都算上
再减掉右上/带外这类整块不需要算的块
GetCutBlockNums(...) 本质是在按对角线边界估算“整块裁掉的三角区域块数”，实现见 prompt_flash_attention_tiling_v2.cpp (line 3159)



而 GetCalcBlockNumsOneHead() 里之所以要减两次，是因为一个带状/因果可见区域，放在整个 Q x KV 矩形里看，无效区域天然分成两个三角角落：

右上角一块三角形
左下角一块三角形
对应代码在：
prompt_flash_attention_tiling_v2.cpp

1. 先看 GetCalcBlockNumsOneHead() 在做什么

它先算满矩形块数：

toCalcBlockNums = outerBlockNums * innerBlockNums
如果用了 mask/sparse，再减两次：

toCalcBlockNums
= 满矩形块数
- 右上角整块可裁掉的三角块数
- 左下角整块可裁掉的三角块数
源码就是这两次减法：

toCalcBlockNums -= GetCutBlockNums(..., nextTokensLeftUp);
toCalcBlockNums -= GetCutBlockNums(..., blockSeqLengthKV - blockSeqLength + preTokensLeftUp);
2. 为什么是“两次三角”

把有效带状区域画出来。设：

高度 H = blockSeqLength
宽度 W = blockSeqLengthKV
有效区在两条斜线之间：

col <= row + nextTokensLeftUp
col >= row - preTokensLeftUp
所以整个矩形会变成这样：

            sInner(KV blocks) ->
         +------------------------+
sOuter   | V V V V X X X X X X   |
(Q rows) | V V V V V X X X X X   |
    |    | L V V V V V X X X X   |
    v    | L L V V V V V X X X   |
         | L L L V V V V V X X   |
         | L L L L V V V V V X   |
         +------------------------+

V = 有效块
X = 右上角无效块
L = 左下角无效块
所以：

X 是一个右上三角
L 是一个左下三角
满矩形减掉这两块，剩下的就是实际要算的块。

3. 第一减：减右上角三角

第一项：

GetCutBlockNums(..., nextTokensLeftUp)
对应这条边界：

col = row + nextTokensLeftUp
右上方都是整块无效区：

         +------------------+
         | V V V X X X X X |
         | V V V V X X X X |
         | V V V V V X X X |
         | V V V V V V X X |
         +------------------+
这块三角为什么能用等差数列算？

因为每往下一行 sOuter block，可整块裁掉的 sInner block 数通常会少一个固定步长，于是形成：

n + (n-1) + (n-2) + ...
所以 GetCutBlockNums() 里会出现三角数/等差数列求和。

4. 第二减：减左下角三角

第二项：

GetCutBlockNums(..., W - H + preTokensLeftUp)
这里很多人第一眼会疑惑：
为什么不是直接传 preTokensLeftUp，而是传 W - H + preTokensLeftUp？

原因是：

GetCutBlockNums() 只会算“右上角三角”。
左下角三角不是直接算，而是先做一次 180° 旋转等价变换，把左下角问题变成“右上角问题”再复用同一个 helper。

原来的左下边界是：

col = row - preTokensLeftUp
左下无效区：

         +------------------+
         | X X X V V V V V |
         | X X V V V V V V |
         | X V V V V V V V |
         | V V V V V V V V |
         +------------------+
把整个矩形旋转 180° 后，它会变成另一个右上三角。
旋转后的等效边界偏移量，正好就是：

W - H + preTokensLeftUp
所以第二次调用本质是：

“把左下三角翻转成右上三角，再交给同一个 GetCutBlockNums() 去算”
这就是“为什么减两次三角块数”。

5. GetCutBlockNums() 自己在几何上做了什么

它不是直接按元素算，而是按“块网格”算。

先把 token 对齐到块边界：

blockToken = ceil_or_floor_to_sInner(token)
意思是：

只裁掉那些“整块完全落在无效区”的块
贴边的块不能整块裁掉，要留给后面的 mask 细裁剪
然后它选一个更小的几何粒度 smallSize，再算一个三角形块数。

可以把它理解成先在更细的格子上得到一个理想三角：

         X X X X
         X X X
         X X
         X
然后再做 3 个修正：

innerCutBlockNums：理想三角总量
innerCutBlockLeftNums：减掉跑到左边界外的部分
innerCutBlockDownNums：减掉跑到下边界外的部分
也就是：

最终三角块数
= 理想三角
- 左边溢出的部分
- 下边溢出的部分
所以 helper 本质上是在算：

“有限矩形窗口内，被某条斜线完整切掉的整块数量”

6. 最后一句总结

GetCalcBlockNumsOneHead() 的几何本质就是：

总块数 = 满矩形块数 - 右上三角无效块 - 左下三角无效块
其中：

第一减直接算右上三角
第二减把左下三角变换成右上三角后复用同一个 GetCutBlockNums()
如果你愿意，我下一条我可以继续把这两个 token：

nextTokensLeftUp
W - H + preTokensLeftUp
分别在图上标出来，直接画成“边界线方程 + 三角裁剪区域”的对应图。