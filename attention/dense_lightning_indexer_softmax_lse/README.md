# DenseLightningIndexerSoftmaxLse

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾910_95 AI处理器</term>|      ×     |
|<term>Atlas A3 训练系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品 </term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200I/300/500 推理产品</term>|      ×     |

## 功能说明

- 算子功能：DenseLightningIndexerSoftmaxLse算子是DenseLightningIndexerGradKlLoss算子计算Softmax输入的一个分支算子。


- 计算公式：

$$
\text{res}=\text{AttentionMask}\left(\text{ReduceSum}\left(W\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right)\right)
$$

$$
\text{maxIndex}=\text{max}\left(res\right)
$$

$$
\text{sumIndex}=\text{ReduceSum}\left(\text{exp}\left(res-maxIndex\right)\right)
$$

maxIndex，sumIndex作为输出传递给算子DenseLightningIndexerGradKlLoss作为输入计算Softmax使用。


## 参数说明

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
          <td>query</td>
          <td>输入</td>
          <td>attention结构的输入Q</td>
          <td>FLOAT16、BFLOAT16 </td>
          <td>ND</td>
      </tr>
      <tr>
          <td>key</td>
          <td>输入</td>
          <td>attention结构的输入K</td>
          <td>FLOAT16、BFLOAT16 </td>
          <td>ND</td>
      </tr>
      <tr>
          <td>queryIndex</td>
          <td>输入</td>
          <td>lightingIndexer结构的输入queryIndex。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>keyIndex</td>
          <td>输入</td>
          <td>lightingIndexer结构的输入keyIndex。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>weights</td>
          <td>输入</td>
          <td>权重</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>softmaxMax</td>与query的B保持一致
          <td>输入</td>
          <td>Device侧的aclTensor，注意力正向计算的中间输出</td>
          <td>FLOAT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>softmaxSum</td>
          <td>输入</td>
          <td>Device侧的aclTensor，注意力正向计算的中间输出</td>
          <td>FLOAT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>softmaxMaxIndex</td>与query的B保持一致
          <td>输入</td>
          <td>Device侧的aclTensor，注意力正向计算的中间输出</td>
          <td>FLOAT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>softmaxSumIndex</td>
          <td>输入</td>
          <td>Device侧的aclTensor，注意力正向计算的中间输出</td>
          <td>FLOAT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>queryRope</td>
          <td>输入</td>
          <td>MLA rope部分：Query位置编码的输出。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>keyRope</td>
          <td>输入</td>
          <td>MLA rope部分：Key位置编码的输出</<td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>actualSeqLengthsQuery</td>
          <td>输入</td>
          <td>每个Batch中，Query的有效token数</td>
          <td>INT64</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>actualSeqLengthsKey</td>
          <td>输入</td>
          <td>每个Batch中，Key的有效token数</td>
          <td>INT64</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>scaleValue</td>
          <td>输入</td>
          <td>缩放系数</td>
          <td>double</td>
          <td>-</td>
      </tr>
      <tr>
          <td>layout</td>
          <td>输入</td>
          <td>layout格式</td>
          <td>char*</td>
          <td>-</td>
      </tr>
      <tr>
          <td>sparseMode</td>
          <td>输入</td>
          <td>sparse的模式</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>preTokens</td>
          <td>输入</td>
          <td>用于稀疏计算，表示Attention需要和前几个token计算关联</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>nextTokens</td>
          <td>输入</td>
          <td>用于稀疏计算，表示Attention需要和后几个token计算关联</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dQueryIndex</td>
          <td>输出</td>
          <td>QueryIndex的梯度</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>dKeyIndex</td>
          <td>输出</td>
          <td>KeyIndex的梯度</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>dWeights</td>
          <td>输出</td>
          <td>Weights的梯度</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>loss</td>
          <td>输出</td>
          <td>损失函数值</td>
          <td>FLOAT32</td>
          <td>ND</td>
      </tr>
      </tbody>
  </table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dense_lightning_indexer_grad_kl_los](examples/test_aclnn_dense_lightning_indexer_grad_kl_los.cpp) | 通过[aclnnDenseLightningIndexerGradKLLoss](docs/aclnnDenseLightningIndexerGradKLLoss.md)接口方式调用dense_lightning_indexer_grad_kl_los算子。 |