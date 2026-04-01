# QuantGroupedMatMulAlltoAllv

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：完成路由专家GroupedMatMul、Unpermute、AlltoAllv融合并实现与共享专家MatMul并行融合，**先计算后通信**。

- 计算公式：
    - 路由专家：
        $$
        gmmY = (gmmX @ gmmWeight) * gmmXScale * gmmWeightScale \\
        unpermuteOut = Unpermute(gmmY) \\
        y = AlltoAllv(unpermuteOut)
        $$

    - 共享专家：

        $$
        mmY = (mmX @  mmWeight) * mmXScaleOptional * mmWeightScaleOptional
        $$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1200px">
      <colgroup>
          <col style="width: 188px">
          <col style="width: 85px">
          <col style="width: 220px">
          <col style="width: 280px">
          <col style="width: 200px">
          <col style="width: 90px">
          <col style="width: 100px">
          <col style="width: 80px">
      </colgroup>
      <thead>
          <tr>
              <th>参数名</th>
              <th>输入/输出</th>
              <th>描述</th>
              <th>使用说明</th>
              <th>数据类型</th>
              <th>数据格式</th>
              <th>维度(shape)</th>
              <th>非连续Tensor</th>
          </tr>
      </thead>
      <tbody>
          <!-- gmmX -->
          <tr>
              <td>gmmX</td>
              <td>输入</td>
              <td>公式中的输入 gmmX。</td>
              <td>支持的 shape 包括 (A, H)。</td>
              <td><li>TT量化：HIFLOAT8</li><li>Mx量化：FLOAT8_E4M3FN、FLOAT8_E5M2</li></td>
              <td>ND</td>
              <td>2</td>
              <td>x</td>
          </tr>
          <!-- gmmWeight -->
          <tr>
              <td>gmmWeight</td>
              <td>输入</td>
              <td>公式中的输入 gmmWeight。</td>
              <td>支持的 shape 包括 (e, H, N1)。e 为每卡部署的专家数，H 为 hidden size，N1 为 MoE FFN 中间维度。</td>
              <td><li>TT量化：HIFLOAT8</li><li>Mx量化：FLOAT8_E4M3FN、FLOAT8_E5M2</li></td>
              <td>ND</td>
              <td>3</td>
              <td>x</td>
          </tr>
          <!-- gmmXScale -->
          <tr>
              <td>gmmXScale</td>
              <td>输入</td>
              <td>gmmX 的量化系数。</td>
              <td><li>默认为空。支持的 shape 包括 (1, )。</li><li>Mx量化时shape：(M, ceil(K/64), 2)</li></td>
              <td><li>TT量化：FLOAT32</li><li>Mx量化：FLOAT8_E8M0</li></td>
              <td>ND</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- gmmWeightScale -->
          <tr>
              <td>gmmWeightScale</td>
              <td>输入</td>
              <td>gmmWeight 的量化系数。</td>
              <td><li>默认为空。支持的 shape 包括 (1, )。</li><li>Mx量化时shape：(e, ceil(K/64), N, 2)，weight转置时为(e, N, ceil(K/64), 2)</li></td>
              <td><li>TT量化：FLOAT32</li><li>Mx量化：FLOAT8_E8M0</li></td>
              <td>ND</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- sendCountsTensorOptional -->
          <tr>
              <td>sendCountsTensorOptional</td>
              <td>输入</td>
              <td>AlltoAllv 使用的 send count。</td>
              <td>当前仅支持空。支持的 shape 包括 (e * ep, )。e 为每卡部署的专家个数，ep 为 ep 域大小。</td>
              <td>INT32、INT64</td>
              <td>ND</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- recvCountsTensorOptional -->
          <tr>
              <td>recvCountsTensorOptional</td>
              <td>输入</td>
              <td>AlltoAllv 使用的 recv count。</td>
              <td>默认为空 Tensor。支持的 shape 包括 (e * ep, )。e 为每卡部署的专家个数，ep 为 ep 域大小。</td>
              <td>INT32、INT64</td>
              <td>ND</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- mmXOptional -->
          <tr>
              <td>mmXOptional</td>
              <td>输入</td>
              <td>公式中的输入 mmX。</td>
              <td>支持的 shape 包括 (bs, H)。bs 为每卡部署的专家个数，H 为 hidden size。</td>
              <td><li>TT量化：BFLOAT16、FLOAT16</li><li>Mx量化：FLOAT8_E4M3FN、FLOAT8_E5M2</li></td>
              <td>ND</td>
              <td>2</td>
              <td>x</td>
          </tr>
          <!-- mmWeightOptional -->
          <tr>
              <td>mmWeightOptional</td>
              <td>输入</td>
              <td>公式中的输入 mmWeight。</td>
              <td>不支持空。支持的 shape 包括 (H, N1)。H 为 hidden size，N1 为共享专家 FFN 的中间层维度。</td>
              <td><li>TT量化：HIFLOAT8</li><li>Mx量化：FLOAT8_E4M3FN、FLOAT8_E5M2</li></td>
              <td>ND</td>
              <td>2</td>
              <td>x</td>
          </tr>
          <!-- mmXScaleOptional -->
          <tr>
              <td>mmXScaleOptional</td>
              <td>输入</td>
              <td>mmX 的量化系数。</td>
              <td>支持的 shape 包括 (1, )。</td>
              <td><li>TT量化：FLOAT32</li><li>Mx量化：FLOAT8_E8M0</li></td>
              <td>ND</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- mmWeightScaleOptional -->
          <tr>
              <td>mmWeightScaleOptional</td>
              <td>输入</td>
              <td>mmWeight 的量化系数。</td>
              <td>支持的 shape 包括 (1, )。</td>
              <td><li>TT量化：FLOAT32</li><li>Mx量化：FLOAT8_E8M0</li></td>
              <td>ND</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- commQuantScaleOptional -->
          <tr>
              <td>commQuantScaleOptional</td>
              <td>输入</td>
              <td>低比特通信量化系数。</td>
              <td>预留参数，当前仅支持空。</td>
              <td>FLOAT32</td>
              <td>ND</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- gmmXQuantMode -->
          <tr>
              <td>gmmXQuantMode</td>
              <td>输入</td>
              <td>gmmX 的量化模式。</td>
              <td>必须传入量化模式，当前支持 1 （pertensor量化）和 6（Mx量化）。</td>
              <td>INT64</td>
              <td>-</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- gmmWeightQuantMode -->
          <tr>
              <td>gmmWeightQuantMode</td>
              <td>输入</td>
              <td>gmmWeight 的量化模式。</td>
              <td>必须传入量化模式，当前支持 1 （pertensor量化）和 6（Mx量化）。</td>
              <td>INT64</td>
              <td>-</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- mmXQuantMode -->
          <tr>
              <td>mmXQuantMode</td>
              <td>输入</td>
              <td>mmX 的量化模式。</td>
              <td>mmX 非空，则必须传入量化模式，当前支持 1 （pertensor量化）和 6（Mx量化）。</td>
              <td>INT64</td>
              <td>-</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- mmWeightQuantMode -->
          <tr>
              <td>mmWeightQuantMode</td>
              <td>输入</td>
              <td>mmWeight 的量化模式。</td>
              <td>mmWeight 不为空，则必须传入量化模式，当前支持 1 （pertensor量化）和 6（Mx量化）。</td>
              <td>INT64</td>
              <td>-</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- commQuantMode -->
          <tr>
              <td>commQuantMode</td>
              <td>输入</td>
              <td>低比特通信量化模式。</td>
              <td>当前低比特功能预留，必须传入 0，表示不量化。</td>
              <td>INT64</td>
              <td>-</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- commQuantDtypeOptional -->
          <tr>
              <td>commQuantDtypeOptional</td>
              <td>输入</td>
              <td>低比特通信的数据类型。</td>
              <td>当前低比特功能预留，必须传入 -1。</td>
              <td>INT64</td>
              <td>-</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- groupSize（新增） -->
          <tr>
              <td>groupSize</td>
              <td>输入</td>
              <td>PerGroup 量化分组大小。</td>
              <td>用于 Matmul 计算三个方向上的量化分组大小，预留参数，仅支持配置为 0，取值不生效。groupSize 输入由 3 个方向的 groupSizeM，groupSizeN，groupSizeK 三个值拼接组成，每个值占 16 位，共占用 int64_t 类型 groupSize 的低 48 位（高 16 位无效），计算公式为：groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。</td>
              <td>INT64</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- group -->
          <tr>
              <td>group</td>
              <td>输入</td>
              <td>通信域标识。</td>
              <td>字符串长度需大于 0，小于 128。</td>
              <td>char*</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- epWorldSize -->
          <tr>
              <td>epWorldSize</td>
              <td>输入</td>
              <td>通信域大小。</td>
              <td>支持 2/4/8/16/32/64/128/256。</td>
              <td>INT64</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- sendCounts -->
          <tr>
              <td>sendCounts</td>
              <td>输入</td>
              <td>AlltoAllv 使用的 send count。表示其他Rank向当前rank上各expert发送的token数量。</td>
              <td>支持的维度为 e * ep。按<code>sendCounts[fromRank][expertId]</code>一维展开, 例如e=3时顺序为<code>e0,e1,e2,e0,e1,e2, ...</code></td>
              <td>aclIntArray*（元素类型 INT64）</td>
              <td>ND</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- recvCounts -->
          <tr>
              <td>recvCounts</td>
              <td>输入</td>
              <td>AlltoAllv 使用的 recv count。表示AlltoAllv后本卡需要接收到的token数量。</td>
              <td>支持的维度为 e * ep。</td>
              <td>aclIntArray*（元素类型 INT64）</td>
              <td>ND</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- transGmmWeight -->
          <tr>
              <td>transGmmWeight</td>
              <td>输入</td>
              <td>gmm 的右矩阵是否转置。</td>
              <td>必须传入，无默认值。</td>
              <td>BOOL</td>
              <td>ND</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- transMmWeight -->
          <tr>
              <td>transMmWeight</td>
              <td>输入</td>
              <td>mm 的右矩阵是否转置。</td>
              <td>必须传入，无默认值。</td>
              <td>BOOL</td>
              <td>ND</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- y -->
          <tr>
              <td>y</td>
              <td>输出</td>
              <td>grouped matmul 计算输出。</td>
              <td>不支持空 Tensor。支持的 shape 包括 (A, N1)。A 为本 rank 处理的 token 数。</td>
              <td>FLOAT16、BFLOAT16</td>
              <td>ND</td>
              <td>2</td>
              <td>x</td>
          </tr>
          <!-- mmYOptional -->
          <tr>
              <td>mmYOptional</td>
              <td>输出</td>
              <td>matmul 计算输出。</td>
              <td>不支持空 Tensor。支持的 shape 包括 (bs, N1)。</td>
              <td>FLOAT16、BFLOAT16</td>
              <td>ND</td>
              <td>2</td>
              <td>x</td>
          </tr>
          <!-- workspaceSize -->
          <tr>
              <td>workspaceSize</td>
              <td>输出</td>
              <td>返回需要在 Device 侧申请的 workspace 大小。</td>
              <td>-</td>
              <td>UINT64</td>
              <td>ND</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- executor -->
          <tr>
              <td>executor</td>
              <td>输出</td>
              <td>返回 op 执行器，包含了算子计算流程。</td>
              <td>-</td>
              <td>aclOpExecutor*</td>
              <td>ND</td>
              <td>-</td>
              <td>-</td>
          </tr>
      </tbody>
  </table>

  gmmXQuantMode、gmmWeightQuantMode、mmXQuantMode、mmWeightQuantMode、commQuantMode的枚举值跟[量化模式](../../docs/zh/context/量化介绍.md)关系如下:

  * 0: 非量化--当前不支持
  * 1: pertensor
  * 2: perchannel
  * 3: pertoken
  * 4: pergroup
  * 5: perblock
  * 6: mx量化
  * 7: pertoken动态量化

## 约束说明

- 确定性计算：
  - aclnnQuantGroupedMatMulAlltoAllv默认确定性实现。
- e * epWorldSize最大支持256，e表示单卡上的专家数量，最大支持到32，epWorldSize支持2/4/8/16/32/64/128/256;
- gmmX的shape(A, H1)，A为sendCounts之和，H1取值范围(0, 65536);
- gmmWeight的shape(e, H1, N1)，H1取值范围(0, 65536)，N1取值范围(0, 65536);
- y的shape为(BSK, N1)，第一维其中K的范围[2, 8]，BSK为recvCounts之和，N1取值(65536);
- mmX是共享专家的左矩阵，shape为(BS, H2)，H2的取值范围(0, 65536)；
- mmWeight是共享专家的右矩阵，shape为(H2， N2)，N2的取值范围(0, 65536)；
- sendCounts为发送到其他卡的token数，数组大小为e * epWorldSize;
- recvCounts从其他卡的token数，数组大小为e * epWorldSize;
- 路由专家和共享专家量化Scale、Mode等均为必选；
- 低比特通信Mode为必选参数，DType和Scale为可选，当Mode为非0时需要提供DType和Scale；
- 参数说明里shape使用的变量：
  - BSK：本卡接收的token数，是recvCounts参数累加之和，取值范围(0, 52428800)。
  - H1：表示路由专家hidden size隐藏层大小，取值范围(0, 65536)。
  - H2：表示共享专家hidden size隐藏层大小，取值范围(0, 12288]。
  - e：表示单卡上专家个数，e<=32，e * epWorldSize最大支持256。
  - N1：表示路由专家的head_num，取值范围(0, 65536)。
  - N2：表示共享专家的head_num，取值范围(0, 65536)。
  - BS：batch sequence size。
  - K：表示选取TopK个专家，K的范围[2, 8]。
  - A：本卡发送的token数，是sendCounts参数累加之和。
  - ep通信域内所有卡的 A 参数的累加和等于所有卡上的 BSK 参数的累加和。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_quant_grouped_mat_mul_allto_allv.cpp](./examples/test_aclnn_quant_grouped_mat_mul_allto_allv.cpp) | 通过[aclnnQuantGroupedMatMulAlltoAllv](./docs/aclnnQuantGroupedMatMulAlltoAllv.md)接口方式调用量化场景的quant_grouped_mat_mul_allto_allv算子。 |
