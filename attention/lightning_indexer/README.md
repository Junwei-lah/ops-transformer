# LightningIndexer

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- API功能：`lightning_indexer`基于一系列操作得到每一个token对应的Top-$k$个位置。

- 计算公式：

     $$
     Indices=\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
     $$

     对于某个token对应的Index Query $Q_{index}\in\R^{g\times d}$，给定上下文Index Key $K_{index}\in\R^{S_{k}\times d},W\in\R^{g\times 1}$，其中$g$为GQA对应的group size，$d$为每一个头的维度，$S_{k}$是上下文的长度。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1601px"><colgroup>
  <col style="width: 264px">
  <col style="width: 132px">
  <col style="width: 232px">
  <col style="width: 330px">
  <col style="width: 164px">
  <col style="width: 119px">
  <col style="width: 215px">
  <col style="width: 145px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>公式中的输入Q。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,S1,N1,D)。</li>
                <li>layout_query为TND时，shape为(T1,N1,D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的输入K。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>block_num为PageAttention时block总数，block_size为一个block的token数。</li>
          </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_key为PA_BSND时，shape为(block_num, block_size, N2, D)。</li>
                <li>layout_kv为BSND时，shape为(B, S2, N2, D)。</li>
                <li>layout_kv为TND时，shape为(T2, N2, D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>输入</td>
      <td>公式中的输入W。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,S1,N1)。</li>
                <li>layout_query为TND时，shape为(T1,N1)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQueryOptional</td>
      <td>输入</td>
      <td>每个Batch中，Query的有效token数。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和`query`的shape的S长度相同。</li>
                <li>该入参中每个Batch的有效token数不超过`query`中的维度S大小且不小于0，支持长度为B的一维tensor。</li>
                <li>当`layout_query`为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKeyOptional</td>
      <td>输入</td>
      <td>每个Batch中，Key的有效token数。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和key的shape的S长度相同。</li>
                <li> 该参数中每个Batch的有效token数不超过`key/value`中的维度S大小且不小于0，支持长度为B的一维tensor。</li>
                <li>当`layout_key`为TND或PA_BSND时，该入参必须传入，`layout_key`为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>输入</td>
      <td>表示PageAttention中KV存储使用的block映射表。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>PageAttention场景下，block\_table必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actual\_seq\_lengths\_key对应的block数量）</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>shape支持(B,S2/block_size)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>layoutQueryOptional</td>
      <td>输入</td>
      <td>用于标识输入Query的数据排布格式。</td>
      <td>
          <ul>
                <li>用户不特意指定时可传入默认值"BSND"。</li>
                <li>当前支持BSND、TND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKeyOptional</td>
      <td>输入</td>
      <td>用于标识输入Key的数据排布格式。</td>
      <td>
          <ul>
                <li>用户不特意指定时可传入默认值"BSND"。</li>
                <li>当前支持PA_BSND、BSND、TND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseCount</td>
      <td>输入</td>
      <td>topK阶段需要保留的block数量。</td>
      <td>支持[1, 2048]，以及3072、4096、5120、6144、7168、8192</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>输入</td>
      <td>表示sparse的模式。</td>
      <td>
          <ul>
                <li>sparse_mode为0时，代表defaultMask模式。</li>
                <li>sparse_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>输入</td>
      <td>用于稀疏计算，表示attention需要和前几个Token计算关联。</td>
      <td>仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens</td>
      <td>输入</td>
      <td>用于稀疏计算，表示attention需要和后几个Token计算关联。</td>
      <td>仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnValues</td>
      <td>输入</td>
      <td>表示是否输出sparseValuesOut。</td>
      <td>
          <ul>
                <li>True表示输出，但图模式下不支持，False表示不输出；默认值为False</li>
                <li>仅在训练且layout_key不为PA_BSND场景支持</li>
          </ul>
      </td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseIndicesOut</td>
      <td>输出</td>
      <td>公式中的Indices输出。</td>
      <td>不支持空tensor。</td>
      <td>INT32</td>
      <td>-</td>
      <td>
          <ul>
                <li>layout_query为"BSND"时输出shape为[B, S1, N2, sparseCount]。</li>
                <li>layout_query为"TND"时输出shape为[T1, N2, sparseCount]。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>sparseValuesOut</td>
      <td>输出</td>
      <td>公式中的Indices输出对应的value值。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>shape与sparseIndicesOut保持一致</td>
      <td>x</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- 该接口支持图模式。
- 参数query中的N支持小于等于64，key的N支持1。
- headdim支持128。
- block_size取值为16的倍数，最大支持1024。
- 参数query、key的数据类型应保持一致。
- 参数weights不为`float32`时，参数query、key、weights的数据类型应保持一致。

## 调用示例

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples//test_aclnn_lightning_indexer.cpp">test_aclnn_lightning_indexer
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnLightningIndexer.md">aclnnLightningIndexer
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
