下面这张图把 sOuter(Q块)、sInner(K块)、pre/next 和 5 种模式放到同一张“块级视角”里。

约定：

纵轴 sOuter：Q 的块索引
横轴 sInner：K/V 的块索引
.：整块直接计算，不需要读 mask
M：边界块，需要读 attentionMask 做块内裁剪
X：整块跳过，不计算
对任一 Q 块，先粗裁剪出
sInner in [sOuter - pre, sOuter + next + qBlockSize)
再只对边界块做 mask 细裁剪
总览

sOuter (Q blocks)
  ^
  |
  +----------------------------------> sInner (K blocks)
1. sparse_mode = 0, defaultMask

这模式本身是“通用模式”：

不传 attentionMask 时，通常退化成“全算”
传完整 attentionMask 时，块级上像“全矩阵 mask”
若 pre/next 形成因果或带状窗口，块级裁剪又会像 causal/band
可以把它理解成：

情况A：没传 mask，pre/next 也不收紧
      sInner ->
sOuter  . . . . .
   |    . . . . .
   v    . . . . .
        . . . . .
        . . . . .

情况B：传完整 mask
      sInner ->
sOuter  M M M M M
   |    M M M M M
   v    M M M M M
        M M M M M
        M M M M M
2. sparse_mode = 1, allMask

完整 mask，忽略 pre/next。
块级含义最直接：所有参与块都可能要按显式 mask 裁剪。

      sInner ->
sOuter  M M M M M
   |    M M M M M
   v    M M M M M
        M M M M M
        M M M M M

3. sparse_mode = 2, leftUpCausal（左上角对齐形成对角线）

左上角对齐的下三角。
host 侧等价成 pre = +inf, next = 0，所以每个 Q 块只能看“自己左边和自己”。

      sInner ->
sOuter  M X X X X
   |    . M X X X
   v    . . M X X
        . . . M X
        . . . . M
理解：

对角线左下的大块 .：整块都合法，直接算
对角线块 M：块内一部分合法、一部分非法，要读压缩 mask
右上 X：整块非法，直接跳过
这就是 leftUp 的“单边界切块”。

4. sparse_mode = 3, rightDownCausal（右下角对齐形成对角线）
右下角对齐的下三角。
当 Q/KV 长度不同，对角线会整体平移。可以把它看成“左上因果线向右平移”。

例子：KV 比 Q 长，所以允许 Q 看到更右边一些块：

      sInner ->
sOuter  . M X X X
   |    . . M X X
   v    . . . M X
        . . . . M
        . . . . .
如果把 Q 更短、KV 更长的感觉画得更明显，就是：

      sInner ->
sOuter  . . M X X X
   |    . . . M X X
   v    . . . . M X
        . . . . . M
理解：

仍然只有一条边界线
只是这条线不是“左上主对角线”，而是“右下对齐后的对角线”
所以它也是“单边界切块”，只是边界位置按
nextTokensPerBatch = kvLen - qLen
动态偏移
5. sparse_mode = 4, band

这是最值得单独看的一种。
它不是一条边界，而是“两条边界围成的一条带”。

      sInner ->
sOuter  M M X X X X
   |    . M M X X X
   v    . . M M X X
        X . . M M X
        X X . . M M
        X X X . M M
更抽象地看：

      sInner ->
sOuter  X M . . M X
   |    X X M . . M
   v    M X X M . .
        . M X X M .
        . . M X X M
理解：

左边界一条线，右边界一条线
两条线中间的块 .：整块合法，直接算
两侧边界块 M：要读 mask 做块内裁剪
带外块 X：整块跳过
这就是为什么 band 模式在 kernel 里会有两套偏移：

attenMaskOffset
attenMaskOffsetPre
并且会做两次 ElewiseCompute：

第一次裁掉一侧边界
第二次再裁掉另一侧边界
也就是“双边界切块”。

把 pre/next 放回图里

对某个 sOuter = i 的 Q 块，允许的 sInner 大致是：

left  boundary  = i - pre
right boundary  = i + next
于是：

leftUpCausal
pre = +inf, next = 0
只保留左边到当前块

rightDownCausal
pre = +inf, next = kvLen - qLen
只保留左边到“右下对齐后”的当前块

band
保留 [i - pre, i + next] 之间的带状区域

一句话总结

allMask/default full-mask：没有明显几何边界，整图都可能读 mask
leftUp：一条左上到右下的边界线，单边界切块
rightDown：一条平移后的边界线，单边界切块
band：两条边界线夹一条带，双边界切块