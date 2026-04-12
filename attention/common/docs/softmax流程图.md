输入:
  srcUb / inMaxUb / inExpSumUb / maskUb / pseUb / dropMaskUb

        |
        v
+------------------------------+
| 1. 读入 score 两半           |
|    vreg_input_x             |
|    vreg_input_x_unroll      |
+------------------------------+
        |
        v
+------------------------------+
| 2. 预处理 score              |
|    Muls / Mul / Add / Select |
|    scale + pse + mask        |
+------------------------------+
        |
        v
+------------------------------+
| 3. 求当前块最大值 cur_m      |
|    Max + Reduce(MAX)         |
|    -> vreg_input_max         |
|    -> tmpMaxUb               |
+------------------------------+
        |
        v
+------------------------------+
| 4. 跟历史 max 合并           |
|    old_m = vreg_in_max       |
|    new_m = Max(cur_m, old_m) |
|    -> vreg_max_new           |
|    -> tmpMaxUb2              |
+------------------------------+
        |
        v
+------------------------------+
| 5. 用 new_m 重新算 exp       |
|    ExpSub                    |
|    -> vreg_exp_even          |
|    -> vreg_exp_odd           |
+------------------------------+
        |
        v
+------------------------------+
| 6. 求当前块和 cur_l          |
|    Add + Reduce(SUM)         |
|    -> vreg_exp_sum           |
|    -> tmpExpSumUb            |
+------------------------------+
        |
        v
+------------------------------+
| 7. 可选 dropout              |
|    Select + Muls             |
+------------------------------+
        |
        v
+------------------------------+
| 8. 写出当前块 exp(x-new_m)   |
|    StoreAlign / Cast / Or    |
|    -> dstTensor              |
+------------------------------+
        |
        v
+----------------------------------------------+
| 9. 外层合并历史 old_l / old_m                |
|    alpha = exp(old_m - new_m)                |
|    new_l = alpha * old_l + cur_l             |
|    UpdateExpSumAndExpMax(...)                |
+----------------------------------------------+


第 1 步：读入当前块 score
代码位置：vf_basic_block_unaligned128_update.h (line 111)

LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
LoadAlign(vreg_input_x_unroll, srcUb + floatRepSize + i * s2BaseSize);
代码图：

srcUb (一行 128 列)
   |
   +--> 前 64 列  --> vreg_input_x
   |
   +--> 后 64 列  --> vreg_input_x_unroll
变量：

vreg_input_x = x[0:64]
vreg_input_x_unroll = x[64:128]
第 2 步：scale / pse / mask 预处理
代码位置：

scale: vf_basic_block_unaligned128_update.h (line 115)
pse: vf_basic_block_unaligned128_update.h (line 133)
mask: vf_basic_block_unaligned128_update.h (line 170)
核心代码：

Muls(vreg_input_x, vreg_input_x, dScale, preg_all);
Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScale, preg_ori_tail_n);

Add(vreg_input_x, vreg_input_x, vreg_pse, preg_all);
Add(vreg_input_x_unroll, vreg_input_x_unroll, vreg_pse_unroll, preg_ori_tail_n);

Select(vreg_sel, vreg_min, vreg_input_x, preg_compare);
Select(vreg_sel_unroll, vreg_min, vreg_input_x_unroll, preg_compare_unroll);
代码图：

vreg_input_x / vreg_input_x_unroll
        |
        +--> Muls / Mul       (缩放)
        |
        +--> Add              (+ pse / alibi)
        |
        +--> Select           (mask 位置替换成 minValue)
        v
vreg_sel / vreg_sel_unroll
数学上相当于：

x = mask(scale(score) + pse)
变量：

vreg_sel / vreg_sel_unroll = masked 后的 x
第 3 步：求当前块最大值 cur_m
代码位置：vf_basic_block_unaligned128_update.h (line 186)

核心代码：

Max(vreg_max_tmp, vreg_sel, vreg_sel_unroll_new, preg_all);
Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
    vreg_input_max, vreg_max_tmp, preg_all);
StoreUnAlign(... tmpMaxUb, vreg_input_max, ...);
代码图：

vreg_sel            vreg_sel_unroll_new
    |                       |
    +---------- Max --------+
                |
                v
          vreg_max_tmp
                |
         Reduce(MAX)
                |
                v
         vreg_input_max
                |
                v
             tmpMaxUb
数学上：

cur_m = max(x)
变量：

vreg_input_max = cur_m
第 4 步：和历史 old_m 合并成 new_m
代码位置：vf_basic_block_unaligned128_update.h (line 209)

核心代码：

LoadAlign(vreg_in_max, inMaxUb);
LoadAlign(vreg_input_max, tmpMaxUb2);
Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all);
StoreAlign(... tmpMaxUb2, vreg_max_new, preg_all);
代码图：

tmpMaxUb2 ---> vreg_input_max ----+
                                  |
                                  +---- Max ----> vreg_max_new ---> tmpMaxUb2
                                  |
inMaxUb  ---> vreg_in_max --------+
数学上：

old_m = inMaxUb
new_m = max(cur_m, old_m)
变量：

vreg_in_max = old_m
vreg_max_new = new_m
第 5 步：用 new_m 重新算指数
代码位置：vf_basic_block_unaligned128_update.h (line 229)

核心代码：

LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max, tmpMaxUb2 + i);

ExpSub(vreg_exp_even, vreg_input_x, vreg_max, preg_all);
ExpSub(vreg_exp_odd, vreg_input_x_unroll, vreg_max, preg_all);
代码图：

vreg_input_x           vreg_input_x_unroll
      |                        |
      +---- ExpSub(new_m) -----+
                |
        +-------+-------+
        |               |
        v               v
  vreg_exp_even    vreg_exp_odd
数学上：

e = exp(x - new_m)
变量：

vreg_exp_even = exp(x[0:64] - new_m)
vreg_exp_odd = exp(x[64:128] - new_m)
第 6 步：求当前块局部和 cur_l
代码位置：vf_basic_block_unaligned128_update.h (line 244)

核心代码：

Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
    vreg_exp_sum, vreg_exp_sum, preg_all);
StoreUnAlign(... tmpExpSumUb, vreg_exp_sum, ...);
代码图：

vreg_exp_even       vreg_exp_odd
      |                  |
      +------ Add -------+
               |
               v
         vreg_exp_sum
               |
         Reduce(SUM)
               |
               v
          vreg_exp_sum
               |
               v
           tmpExpSumUb
数学上：

cur_l = sum(exp(x - new_m))
变量：

vreg_exp_sum = cur_l
第 7 步：可选 dropout
代码位置：vf_basic_block_unaligned128_update.h (line 260)

核心代码：

Select(vreg_sel_drop, vreg_exp_even, vreg_zero, preg5);
Muls(vreg_exp_even, vreg_sel_drop, divValue, preg_all);

Select(vreg_sel_drop2, vreg_exp_odd, vreg_zero, preg6);
Muls(vreg_exp_odd, vreg_sel_drop2, divValue, preg_all);
代码图：

vreg_exp_even -- Select(drop mask) -- Muls(1/keepProb) --> vreg_exp_even
vreg_exp_odd  -- Select(drop mask) -- Muls(1/keepProb) --> vreg_exp_odd
数学上：

e = dropout(e) / keepProb
第 8 步：把 exp(x - new_m) 写出
代码位置：vf_basic_block_unaligned128_update.h (line 276)

这里根据 T2 的类型分很多支，但语义一致：

float：直接 StoreAlign
half/bf16：Cast + Or + StoreAlign
fp8/hifloat8/int8：Cast + Or/Gather/Pack + StoreAlign
代码图：

vreg_exp_even / vreg_exp_odd
        |
        +--> Cast / Or / Gather / Pack (按输出类型)
        |
        v
      dstTensor
这里存下的是：

exp(x - new_m)
供后续 Vec2 做 P * V 的在线更新。

第 9 步：外层把历史 old_l / old_m 真正合并
这一步不在本文件里，而在 vf_mul_sel_softmaxflashv2_cast_nz.h (line 141)。

核心代码：

LoadAlign(vreg_max, tmpMaxUb);       // new_m
LoadAlign(vreg_in_max, inMaxUb);     // old_m
ExpSub(vreg_exp_max, vreg_in_max, vreg_max, preg_all);   // alpha = exp(old_m - new_m)

LoadAlign(vreg_in_exp_sum, inExpSumUb);   // old_l
LoadAlign(vreg_exp_sum_brc, tmpExpSumUb); // cur_l
Mul(vreg_exp_sum_update, vreg_exp_max, vreg_in_exp_sum, preg_all);
Add(vreg_exp_sum_update, vreg_exp_sum_update, vreg_exp_sum_brc, preg_all);
StoreAlign(... expSumUb, vreg_exp_sum_update, ...);      // new_l
代码图：

old_m ----+
          +--> ExpSub --> alpha = exp(old_m - new_m)
new_m ----+

old_l ----*
          |
alpha ----*--> alpha * old_l ----+
                                 +--> Add --> new_l
cur_l ---------------------------+
数学上就是：

alpha = exp(old_m - new_m)
new_l = alpha * old_l + cur_l
整条链最终可以浓缩成这一张图

score(srcUb)
   |
   v
[Muls/Mul/Add/Select]
   |
   v
x
   |
   +--> [Max + Reduce(MAX)] --------------------> cur_m
   |                                              |
   |                                              +--> [Max with old_m] ---> new_m
   |                                                                         |
   +--> [ExpSub(x, new_m)] ---> e = exp(x-new_m)                             |
                |                                                            |
                +--> [Add + Reduce(SUM)] ---> cur_l -------------------------+
                                                                             |
old_m ----------------------> alpha = exp(old_m - new_m)                     |
old_l ----------------------> alpha * old_l ---------------------------------+
                                                                             |
                                                                             v
                                                                           new_l