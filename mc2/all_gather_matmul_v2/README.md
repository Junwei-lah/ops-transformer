# AllGatherMatmulV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | x |
| <term>Atlas 推理系列产品</term> | x |
| <term>Atlas 训练系列产品</term> | x |

## 功能说明

- **算子功能**：
  aclnnAllGatherMatmulV2接口是对aclnnAllGatherMatmul接口的功能拓展，x1和x2新增支持低精度数据类型（如FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8），同时支持pertensor、perblock[量化方式](../../docs/zh/context/量化介绍.md)。
  
  功能可分为以下4种情形：  
    - 如果x1和x2数据类型为FLOAT16/BFLOAT16时，入参x1进行AllGather后，对x1、x2进行matmul计算；
    - 如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8，不输出amaxOut，入参x1进行AllGather后，对x1、x2进行matmul计算，然后进行dequant操作；
    - 如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8，且输出amaxOut，入参x1进行AllGather后，对x1、x2进行matmul计算，然后进行dequant操作，最后进行quant操作， 当前版本暂不支持；
    - 如果groupSize取值为有效值，入参x1进行AllGather后，对x1、x2进行perblock量化matmul计算，然后进行dequant操作。

- **计算公式**：
    - 情形1：

    $$
    output=AllGather(x1)@x2 + bias
    $$

    $$
    gatherOut=AllGather(x1)
    $$

    - 情形2：

    $$
    output=(x1Scale*x2Scale)*(AllGather(x1)@x2 + bias)
    $$

    $$
    gatherOut=AllGather(x1)
    $$

    - 情形3：

      $$
      output=(x1Scale*x2Scale)*(quantScale)*(AllGather(x1)@x2 + bias)
      $$

      $$
      gatherOut=AllGather(x1)
      $$

      $$
      amaxOut=amax((x1Scale*x2Scale)*(AllGather(x1)@x2 + bias))
      $$

    - 情形4：

      $$
      output[r(i), r(j)] = \sum_{k=1}^{\frac{K}{groupSizeK}} x1Scale[i, k] * x2Scale[k, j] * (AllGather(x1)[r(i), r(j)] @ x2[r(k), r(j)])
      $$

      $$
      r(z) = (groupSizeK * (z - 1) + 1) : (groupSizeK * z)
      $$

      $$
      output = \begin{bmatrix}
            output[r(1), r(1)] & \cdots & output[r(1), r(\frac{N}{groupSizeN})] \\ 
            \vdots & \ddots & \vdots \\
            output[r(\frac{M}{groupSizeM}), r(1)] & \vdots & output[r(\frac{M}{groupSizeM}), r(\frac{N}{groupSizeN})]
          \end{bmatrix}
      $$
    
      其中$output\left[r(y), r(z)\right]$表示从output矩阵中取出第$(groupSizeM*(y-1)+1)$到$(groupSizeM*y)$行和$(groupSizeN*(z-1)+1)$到$(groupSizeN*z)$列构成的块。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>公式中的输入x1。</td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入x2。</td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x1Scale</td>
      <td>输入</td>
      <td>公式中的输入x1Scale。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2Scale</td>
      <td>输入</td>
      <td>公式中的输入x2Scale。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quantScale</td>
      <td>输入</td>
      <td>公式中的输入quantScale。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blockSize</td>
      <td>输入</td>
      <td>公式中的输入blockSize。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>公式中的输出output</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>amaxOut</td>
      <td>输出</td>
      <td>公式中的输出amaxOut</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gatherOut</td>
      <td>输出</td>
      <td>公式中的输出gatherOut</td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>Device侧的整型，返回需要在Device侧申请的workspace大小。</td>
      <td>UINT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group</td>
      <td>属性</td>
      <td><li>Host侧标识通信域的字符串，通信域名称。</li><li>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</li></td>
      <td>CHAR*、STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gatherIndex</td>
      <td>属性</td>
      <td>Host侧的整型，标识gather目标。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>commTurn</td>
      <td>属性</td>
      <td>Host侧的整型，通信数据切分数，即总数据量/单次通信量。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>streamMode</td>
      <td>属性</td>
      <td>Host侧的整型，流模式的枚举。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupSize</td>
      <td>可选属性</td>
      <td>用于表示反量化中x1Scale/x2Scale输入的一个数在其所在的对应维度方向上可以用于该方向x1/x2输入的多少个数的反量化。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

* 输入x1为2维，其维度为\(m, k\)。x2必须是2维，其维度为\(k, n\)，轴满足mm算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)。bias为1维，shape为\(n,\)。
* 输出output为2维，其维度为\(m*rank\_size, n\)，rank\_size为卡数。
* 输出gatherout为2维，其维度为\(m*rank\_size, k\)，rank\_size为卡数。
* 当x1、x2的数据类型为FLOAT16/BFLOAT16时，output计算输出数据类型和x1、x2保持一致，bias暂不支持输入为非0的场景，且不支持amaxOut的输入。
* 当x1、x2的数据类型为FLOAT8_E4M3FN/FLOAT_E5M2/HIFLOAT8时，output输出数据类型支持FLOAT16、BFLOAT16、FLOAT。支持bias输入为FLOAT。
* 当x1、x2的数据类型为FLOAT16/BFLOAT16/HIFLOAT8时，x1和x2数据类型需要保持一致。
* 当x1、x2数据类型为FLOAT8_E4M3FN/FLOAT_E5M2时，x1和x2数据类型可以为其中一种。
* 当x1、x2数据类型为FLOAT16/BFLOAT16/HIFLOAT8/FLOAT8_E4M3FN/FLOAT_E5M2时，x2矩阵支持转置/不转置场景，x1矩阵只支持不转置场景。
* 当groupSize取值为549764202624，bias必须为空。
* <term>Ascend 950PR/Ascend 950DT</term>：支持2、4、8、16、32、64卡。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_all_gather_matmul_v2](./examples/test_aclnn_all_gather_matmul_v2.cpp) | 通过[aclnnAllGatherMatmulV2](./docs/aclnnAllGatherMatmulV2.md)接口方式调用AllGatherMatmulV2算子。 |
