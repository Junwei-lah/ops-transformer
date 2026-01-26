# MatmulAlltoAll

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- 算子功能：完成Matmul计算与AlltoAll通信融合。
- 计算公式：
  假设x1的shape为(BS, H1), x2的shape为(H1, H2)。

  $$
  computeOut = x1 @ x2 + bias \\
  permutedOut = computeOut.view(BS, rankSize, H2/rankSize).permute(1, 0, 2) \\
  output = AlltoAll(permutedOut).view(rankSize*BS, H2/rankSize)
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
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
      <td>融合算子的左矩阵输入，对应公式中的x1</td>
      <td>FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>融合算子的右矩阵输入，对应公式中的x2</td>
      <td>FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>可选输入</td>
      <td>矩阵乘运算后累加的偏置，对应公式中的bias</td>
      <td>x1/x2为FLOAT16时，该参数类型为FLOAT16；x1/x2为BFLOAT16时，该参数类型为FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alltoAllAxesOptional</td>
      <td>可选输入</td>
      <td>AlltoAll和Pemute数据交换的方向，支持配置空或者[-1,-2]，传入空时默认按[-1,-2]处理，表示将输入由(BS, H2)转为(BS * rankSize, H2 / rankSize)</td>
      <td>aclIntArray*(元素类型INT64)</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group</td>
      <td>输入</td>
      <td>通信域名，字符串长度要求(0, 128)</td>
      <td>STRING</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>transposeX1</td>
      <td>输入</td>
      <td>标识左矩阵是否转置过，配置为True时左矩阵Shape为(H1, BS)，暂不支持配置为True</td>
      <td>bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>输入</td>
      <td>标识右矩阵是否转置过，配置为True时右矩阵Shape为(H2, H1)</td>
      <td>bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>公式中的输出output，数据类型与输入x1保持一致</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明
* 默认支持确定性计算
* 右矩阵和输出矩阵的H2必须整除NPU卡数
* 不支持空tensor
* x1、x2计算输入的数据类型要和output计算输出的数据类型一致，传入的x1、x2或者output不为空指针。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：output计算输出的数据类型为FLOAT16时，biasOptional计算输入的数据类型支持FLOAT16；output计算输出的数据类型为BFLOAT16时，biasOptional计算输入的数据类型支持FLOAT32。
* H1范围仅支持[1, 65535]
* ranSize仅支持2,4,8
* x1、x2、output的数据类型必须一致
* 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用。
* 不支持跨超节点通信，只支持超节点内。

## 调用说明


| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_matmul_allto_all](./examples/test_aclnn_matmul_allto_all.cpp) | 通过[aclnnMatmulAlltoAll](./docs/aclnnMatmulAlltoAll.md)接口方式调用MatmulAlltoAll算子。 |

