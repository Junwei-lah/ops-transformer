# aclnnFusedInferAttentionScoreV5

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score)

## 产品支持情况

| 产品                                                              | 是否支持 |
| :---------------------------------------------------------------- | :------: |
| `<term>`Ascend 950PR/Ascend 950DT `</term>`                   |    √    |
| `<term>`Atlas A3 训练系列产品/Atlas A3 推理系列产品 `</term>` |    ×    |
| `<term>`Atlas A2 训练系列产品/Atlas A2 推理系列产品 `</term>` |    ×    |
| `<term>`Atlas 200I/500 A2 推理产品 `</term>`                  |    ×    |
| `<term>`Atlas 推理系列产品 `</term>`                          |    ×    |
| `<term>`Atlas 训练系列产品 `</term>`                          |    ×    |

## 功能说明

- 接口功能：适配decode & prefill场景的FlashAttention算子，既可以支持prefill计算场景（PromptFlashAttention），也可支持decode计算场景（IncreFlashAttention）。

  相比于FusedInferAttentionScoreV4，本接口新增qStartIdxOptional、kvStartIdxOptional、pseType参数。

  **说明：**
  decode场景下特有KV Cache：KV Cache是大模型推理性能优化的一个常用技术。采样时，Transformer模型会以给定的prompt/context作为初始输入进行推理（可以并行处理），随后逐一生成额外的token来继续完善生成的序列（体现了模型的自回归性质）。在采样过程中，Transformer会执行自注意力操作，为此需要给当前序列中的每个项目（无论是prompt/context还是生成的token）提取键值（KV）向量。这些向量存储在一个矩阵中，通常被称为kv缓存（KV Cache）。
- 计算公式：

  self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

  self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。

  $$
  Attention(Q,K,V)=Score(Q,K)V
  $$

  本算子中Score函数采用Softmax函数，self-attention计算公式为：

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵。

  **说明**：

  <blockquote>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
    <br>Q_S表示query shape中的S，KV_S表示key和value shape中的S，Q_N表示num_query_heads，KV_N表示num_key_value_heads。P表示Softmax(<span>(QK<sup class="superscript">T</sup>) / <span class="sqrt">d</span></span>)的计算结果。</blockquote>

## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFusedInferAttentionScoreV5GetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnFusedInferAttentionScoreV5”接口执行计算。

```c++
aclnnStatus aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
    const aclTensor     *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor     *pseShiftOptional,
    const aclTensor     *attenMaskOptional,
    const aclIntArray   *actualSeqLengthsOptional,
    const aclIntArray   *actualSeqLengthsKvOptional,
    const aclTensor     *deqScale1Optional,
    const aclTensor     *quantScale1Optional,
    const aclTensor     *deqScale2Optional,
    const aclTensor     *quantScale2Optional,
    const aclTensor     *quantOffset2Optional,
    const aclTensor     *antiquantScaleOptional,
    const aclTensor     *antiquantOffsetOptional,
    const aclTensor     *blockTableOptional,
    const aclTensor     *queryPaddingSizeOptional,
    const aclTensor     *kvPaddingSizeOptional,
    const aclTensor     *keyAntiquantScaleOptional, 
    const aclTensor     *keyAntiquantOffsetOptional, 
    const aclTensor     *valueAntiquantScaleOptional, 
    const aclTensor     *valueAntiquantOffsetOptional, 
    const aclTensor     *keySharedPrefixOptional, 
    const aclTensor     *valueSharedPrefixOptional, 
    const aclIntArray   *actualSharedPrefixLenOptional, 
    const aclTensor     *queryRopeOptional, 
    const aclTensor     *keyRopeOptional, 
    const aclTensor     *keyRopeAntiquantScaleOptional, 
    const aclTensor     *dequantScaleQueryOptional, 
    const aclTensor     *learnableSinkOptional, 
    const aclIntArray   *qStartIdxOptional, 
    const aclIntArray   *kvStartIdxOptional, 
    int64_t             numHeads, 
    double              scaleValue, 
    int64_t             preTokens, 
    int64_t             nextTokens, 
    char                *inputLayout, 
    int64_t             numKeyValueHeads, 
    int64_t             sparseMode, 
    int64_t             innerPrecise, 
    int64_t             blockSize, 
    int64_t             antiquantMode, 
    bool                softmaxLseFlag, 
    int64_t             keyAntiquantMode, 
    int64_t             valueAntiquantMode, 
    int64_t             queryQuantMode, 
    int64_t             pseType, 
    const aclTensor     *attentionOut, 
    const aclTensor     *softmaxLse, 
    uint64_t            *workspaceSize, 
    aclOpExecutor       **executor)
```

```c++
aclnnStatus aclnnFusedInferAttentionScoreV5(
    void                *workspace, 
    uint64_t            workspaceSize, 
    aclOpExecutor       *executor, 
    const aclrtStream   stream)
```

## aclnnFusedInferAttentionScoreV5GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1625px"><colgroup>
    <col style="width: 247px">
    <col style="width: 132px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    <col style="width: 119px">
    <col style="width: 272px">
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
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>query</td>
        <td>输入</td>
        <td>公式中的输入Q。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8_E4M3FN</td>
        <td>ND</td>
        <td>见参数inputLayout</td>
        <td>×</td>
    </tr>
    <tr>
        <td>key</td>
        <td>输入</td>
        <td>公式中的输入K。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8_E4M3FN、INT4（INT32）、FLOAT4_E2M1</td>
        <td>ND</td>
        <td>见参数inputLayout</td>
        <td>×</td>
    </tr>
    <tr>
        <td>value</td>
        <td>输入</td>
        <td>公式中的输入V。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8_E4M3FN、INT4（INT32）、FLOAT4_E2M1</td>
        <td>ND</td>
        <td>见参数inputLayout</td>
        <td>×</td>
    </tr>
    <tr>
        <td>pseShiftOptional</td>
        <td>输入</td>
        <td>位置编码</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>建议shape输入 (B,Q_N,Q_S,KV_S)、(1,Q_N,Q_S,KV_S)</td>
        <td>×</td>
    </tr>
    <tr>
        <td>attenMaskOptional</td>
        <td>输入</td>
        <td>mask矩阵</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>BOOL、INT8、UINT8</td>
        <td>ND</td>
        <td>
        <ul>
            <li>sparseMode = 0、1时
            <ul>
                <term>Ascend 950PR/Ascend 950DT</term>
                <ul>
                    <li>attenMaskOptional的shape输入支持传入:(B, Q_S, KV_S)，(1, Q_S, KV_S)，(B, 1, Q_S, KV_S)，(1, 1, Q_S, KV_S)
                    </li>
                </ul>
            </ul>
            <ul>
                <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> 和 <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
                <li>仅在Layout为BSH、BSND、BNSD、BNSD_BSND时，且query与key的Dvalue的D，并且不传query_rope和key_rope时：
                  <ul>
                      <li>Q_S = 1时，可支持shape传入: (B, KV_S)</li>
                      <li>Q_S > 1时，可支持shape传入: (Q_S, KV_S)</li>
                  </ul>
              </li>
            </ul>
            </li>
            <li>sparseMode = 2、3、4时，attenMaskOptional的shape输入支持传入(2048, 2048)或(1,2048,2048)或(1,1,2048,2048)</li>
            <li>上述Q_S为query的shape中的S，KV_S为key和value的shape中的S；如果输入attenMask shape中的Q_S、KV_S非32B对齐，可以向上取到对齐的Q_S、KV_S。</li>
        </ul>
        </td>
        <td>×</td>
    </tr>
    <tr>
        <td>actualSeqLengthsOptional</td>
        <td>输入</td>
        <td>不同Batch中query的有效序列长度。</td>
        <td>
        <ul>
            <li>不指定序列长度可传入nullptr，表示和query的shape的S长度相同。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        <ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>（1）或（B）或（>B）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>actualSeqLengthsKvOptional</td>
        <td>输入</td>
        <td>不同Batch中key/value的有效序列长度。</td>
        <td>
        <ul>
            <li>不指定序列长度可传入nullptr，表示和key/value的shape的S长度相同。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>（1）或（B）或（>B）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>deqScale1Optional</td>
        <td>输入</td>
        <td>BMM1后面的反量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>支持per-tensor。</li>
            <li>使用全量化功能时，该参数由实际量化过程计算得来。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>UINT64、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#INT8">INT8/FP8量化相关入参数量与输入、输出数据格式的综合限制</a>。</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantScale1Optional</td>
        <td>输入</td>
        <td>BMM2前面的量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>支持per-tensor。 </li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#INT8">INT8/FP8量化相关入参数量与输入、输出数据格式的综合限制</a>。</td>
        <td>-</td>
    </tr>
    <tr>
        <td>deqScale2Optional</td>
        <td>输入</td>
        <td>BMM2后面的反量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>支持per-tensor。 </li>
            <li>使用全量化功能时，该参数由实际量化过程计算得来。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>UINT64、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#INT8">INT8/FP8量化相关入参数量与输入、输出数据格式的综合限制</a>。</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantScale2Optional</td>
        <td>输入</td>
        <td>输出的量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>支持per-tensor，per-channel。 </li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT32、BFLOAT16</td>
        <td>ND</td>
        <td>见<a href="#INT8">INT8/FP8量化相关入参数量与输入、输出数据格式的综合限制</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantOffset2Optional</td>
        <td>输入</td>
        <td>输出的量化偏移。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>支持per-tensor，per-channel。 </li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT32、BFLOAT16</td>
        <td>ND</td>
        <td>与quantScale2Optional保持一致</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantScaleOptional</td>
        <td>输入</td>
        <td>伪量化因子</td>
        <td>不支持</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantOffsetOptional</td>
        <td>输入</td>
        <td>伪量化偏移</td>
        <td>不支持</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr> 
    <tr>
        <td>blockTableOptional</td>
        <td>输入</td>
        <td>PageAttention中KV存储使用的block映射表。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
        </ul>
        </td>
        <td>INT32</td>
        <td>ND</td>
        <td>第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为不同batch中最大actualSeqLengthsKv对应的block数量）</td>
        <td>-</td>
    </tr>
    <tr> 
        <td>queryPaddingSizeOptional</td>
        <td>输入</td>
        <td>表示Query中每个batch的数据是否右对齐，且右对齐的个数是多少。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>仅支持Q_S大于1，其余场景该参数无效。</li>
            <li>不使用该功能时可传入nullptr。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>（1）</td>
        <td>-</td>
    </tr>
    <tr> 
        <td>kvPaddingSizeOptional</td>
        <td>输入</td>
        <td>表示key/value中每个batch的数据是否右对齐，且右对齐的个数是多少。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>（1）</td>
        <td>-</td>
    </tr>
    <tr> 
        <td>keyAntiquantScaleOptional</td>
        <td>输入</td>
        <td>表示key的反量化因子，用于kv伪量化参数分离和FP8 per-block全量化场景。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>支持per-tensor，per-channel，per-token，per-tensor叠加per-head，per-token叠加per-head，per-token叠加使用page attention模式管理scale/offset、per-token叠加per head并使用page attention模式管理scale/offset和per-token-group。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr> 
        <td>keyAntiquantOffsetOptional</td>
        <td>输入</td>
        <td>kv伪量化参数分离时表示key的反量化偏移。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>使用时，shape必须与keyAntiquantScaleOptional保持一致。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>支持per-tensor，per-channel，per-token，per-tensor叠加per-head，per-token叠加per-head，per-token叠加使用page attention模式管理scale/offset、per-token叠加per head并使用page attention模式管理scale/offset。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr> 
        <td>valueAntiquantScaleOptional</td>
        <td>输入</td>
        <td>表示value的反量化因子，用于kv伪量化参数分离和FP8 per-block全量化场景。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>支持per-tensor，per-channel，per-token，per-tensor叠加per-head，per-token叠加per-head，per-token叠加使用page attention模式管理scale/offset、per-token叠加per head并使用page attention模式管理scale/offset和per-token-group。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr> 
        <td>valueAntiquantOffsetOptional</td>
        <td>输入</td>
        <td>kv伪量化参数分离时表示value的反量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>使用时，shape必须与valueAntiquantScaleOptional保持一致。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>支持per-tensor，per-channel，per-token，per-tensor叠加per-head，per-token叠加per-head，per-token叠加使用page attention模式管理scale/offset、per-token叠加per head并使用page attention模式管理scale/offset。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>  
    <tr> 
        <td>keySharedPrefixOptional</td>
        <td>输入</td>
        <td>attention结构中Key的系统前缀部分的参数。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、INT8、INT4/INT32</td>
        <td>ND</td>
        <td>
        <ul>
            <li>input_layout为BSH时，shape为（1，prefix_S，H=KV_N*KV_D）</li>
            <li>input_layout为BSND时，shape为（1，prefix_S，KV_N，KV_D）</li>
            <li>input_layout为BNSD、BNSD_BSND时，shape为（1，KV_N，prefix_S，KV_D）</li>
        </ul>
        </td>
        <td>×</td>
    </tr>
    <tr> 
        <td>valueSharedPrefixOptional</td>
        <td>输入</td>
        <td>attention结构中Value的系统前缀部分的输入。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、INT8、INT4/INT32</td>
        <td>ND</td>
        <td>
        <ul>
            <li>input_layout为BSH时，shape为（1，prefix_S，H=KV_N*KV_D）</li>
            <li>input_layout为BSND时，shape为（1，prefix_S，KV_N，KV_D）</li>
            <li>input_layout为BNSD、BNSD_BSND时，shape为（1，KV_N，prefix_S，KV_D）</li>
        </ul>
        </td>
        <td>×</td>
    </tr>
    <tr> 
        <td>actualSharedPrefixLenOptional</td>
        <td>输入</td>
        <td>keySharedPrefix/valueSharedPrefix的有效Sequence Length。</td>
        <td>
        <ul>
            <li>不使用该功能时可传入nullptr，表示和keySharedPrefix/valueSharedPrefix的s长度相同。</li>
            <li>该入参中的有效Sequence Length应该不大于keySharedPrefix/valueSharedPrefix中的Sequence Length。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>（1）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>queryRopeOptional</td>
        <td>输入</td>
        <td>MLA结构中的query的rope信息。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>queryRope的shape中d为64，其余维度与query一致</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyRopeOptional</td>
        <td>输入</td>
        <td>MLA结构中的key的rope信息。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>keyRope的shape中d为64，其余维度与key一致</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyRopeAntiquantScaleOptional</td>
        <td>输入</td>
        <td>MLA结构中的key的rope信息的反量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>预留参数，当前版本不生效，传入nullptr即可。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>  
    <tr> 
        <td>dequantScaleQueryOptional</td>
        <td>输入</td>
        <td>对query进行反量化的因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>全量化场景涉及。量化模式支持per-token叠加per-head，per-block模式。</li>
            <li>不使用该功能时可传入nullptr。</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>learnableSinkOptional</td>
        <td>输入</td>
        <td>表示通过可学习的"Sink Token"起到吸收Attention Score的作用。</td>
        <td>
        <ul>
            <li>仅支持非量化场景。</li>
            <li>仅支持V_D=128/64。</li>
            <li>不支持pse/左padding/公共前缀/后量化。</li>
        </ul>
        </td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>(Q_N)</a></td>
        <td>×</td>
    </tr>
    <tr> 
        <td>qStartIdxOptional</td>
        <td>输入</td>
        <td>代表外切场景，当前分块的query的sequence在全局中的起始索引。</td>
        <td>
        <ul>
            <li>内部生成pse场景生效（pseType为2或3），其他场景可传入nullptr。</li>
            <li>当pseType为2、3时，不传入该参数按照0处理。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr> 
        <td>kvStartIdxOptional</td>
        <td>输入</td>
        <td>代表外切场景，当前分块的key和value的sequence在全局中的起始索引。</td>
        <td>
        <ul>
            <li>内部生成pse场景生效（pseType为2或3），其他场景可传入nullptr。</li>
            <li>当pseType为2、3时，不传入该参数按照0处理。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>numHeads</td>
        <td>输入</td>
        <td>query的head个数。</td>
        <td>在BNSD、BSND、BNSD_BSND、BSND_BNSD、BNSD_NBSD、BSND_NBSD、TND、NTD、NTD_TND、TND_NTD场景下，需要与shape中的query的N轴shape值相同，否则执行异常。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scaleValue</td>
        <td>输入</td>
        <td>公式中d开根号的倒数。</td>
        <td>
        <ul>
            <li>数据类型与query的数据类型需满足数据类型推导规则。 </li>
            <li>用户不特意指定时建议传入1.0。 </li>
        </ul>
        </td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>preTokens</td>
        <td>输入</td>
        <td>用于稀疏计算，表示attention需要和前几个Token计算关联。</td>
        <td>
        <ul>
            <li>不特意指定时建议传入2147483647。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>  
    <tr>
        <td>nextTokens</td>
        <td>输入</td>
        <td>表示attention需要和后几个Token计算关联。</td>
        <td>
        <ul>
            <li>不特意指定时建议传入2147483647。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>inputLayout</td>
        <td>输入</td>
        <td>标识输入query、key、value的数据排布格式。</td>
        <td>
        <ul>
            <li>当前支持BSH、BSND、BNSD、BNSD_BSND（输入为BNSD时，输出格式为BSND）、BSND_BNSD（输入为BSND时，输出格式为BNSD）、BSH_BNSD（输入为BSH时，输出格式为BNSD）、BNSD_NBSD（输入为BNSD时，输出格式为NBSD）、BSND_NBSD（输入为BSND时，输出格式为NBSD）、BSH_NBSD（输入为BSH时，输出格式为NBSD）、TND（TND相关场景综合约束见<a href="#约束说明">约束说明</a>）、NTD、NTD_TND（输入为NTD时，输出格式为TND）、TND_NTD（输入为TND时，输出格式为NTD）。不特意指定时建议传入"BSH"。</li>
            <li>注意排布格式带下划线时，下划线左边表示输入query的layout，下划线右边表示输出output的格式。</li>
            <li>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。</li>
            <li>inputLayout=BSH_BNSD、BSND_BNSD、NTD、NTD_TND仅支持Q_D=K_D=V_D都等于64或128，或Q_D=K_D等于192，V_D等于128。</li>
            <li>inputLayout=BNSD_BSND仅支持Q_D=K_D=V_D都16对齐(output dtype为int8时为32对齐)，或Q_D=K_D等于192，V_D等于128<br></li>
        </ul>
        </td>
        <td>CHAR</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>numKeyValueHeads</td>
        <td>输入</td>
        <td>key、value中head个数。</td>
        <td>
        <ul>
            <li>用户不特意指定时建议传入0，表示key/value和query的head个数相等。</li>
            <li>需要满足numHeads整除numKeyValueHeads，GQA非量化场景和Prefill MLA非量化场景下，numHeads与numKeyValueHeads的比值无限制; Decode MLA场景仅支持numHeads与numKeyValueHeads的比值为1、2、4、8、16、32、64、128。</li>
            <li>在BNSD、BSND、BNSD_BSND、BSND_BNSD、BNSD_NBSD、BSND_NBSD、TND、NTD、NTD_TND、TND_NTD场景下，还需要与shape中的key/value的N轴shape值相同，否则执行异常</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sparseMode</td>
        <td>输入</td>
        <td>sparse的模式。</td>
        <td>
        <ul>
            <li>inputLayout为TND、TND_NTD、NTD_TND时，综合约束请见<a href="#约束说明">约束说明</a>。</li>
            <li>sparseMode为0时，代表defaultMask模式，如果attenmask未传入则不做mask操作，忽略preTokens和nextTokens（内部赋值为INT_MAX）；如果传入，则需要传入完整的attenmask矩阵（S1 * S2），表示preTokens和nextTokens之间的部分需要计算；要求preTokens + nextTokens >= 0。 </li>
            <li>sparseMode为1时，代表allMask，必须传入完整的attenmask矩阵（S1 * S2）。</li>
            <li>sparseMode为2时，代表leftUpCausal模式的mask，需要传入优化后的attenmask矩阵（2048*2048）。</li>
            <li>sparseMode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的attenmask矩阵（2048*2048）。</li>
            <li>sparseMode为4时，代表band模式的mask，需要传入优化后的attenmask矩阵（2048*2048）；要求preTokens + nextTokens >= 0。</li>
            <li>sparseMode为5、6、7、8时，分别代表prefix、global、dilated、block_local，均暂不支持。</li>
            <li>用户不特意指定时建议传入0。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>innerPrecise</td>
        <td>输入</td>
        <td>表示高精度或者高性能选择。</td>
        <td>
        <ul>
            <li>innerPrecise为0时，代表开启高精度模式，且不做行无效修正。</li>
            <li>innerPrecise为1时，代表高性能模式，且不做行无效修正。</li>
            <li>innerPrecise为2时，代表开启高精度模式，且做行无效修正。</li>
            <li>innerPrecise为3时，代表高性能模式，且做行无效修正。</li>
            <li>sparse_mode为0或1，并传入用户自定义mask的情况下，建议开启行无效修正。</li>
            <li>BFLOAT16和INT8不区分高精度和高性能，行无效修正对FLOAT16、BFLOAT16和INT8均生效。</li>
            <li>当前0、1为保留配置值，当计算过程中“参与计算的mask部分”存在某整行全为1的情况时，精度可能会有损失。此时可以尝试将该参数配置为2或3来使能行无效功能以提升精度，但是该配置会导致性能下降。</li>
            <li>如果算子可判断出存在无效行场景，会自动使能无效行计算，例如sparse_mode为3，Sq > Skv场景。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>blockSize</td>
        <td>输入</td>
        <td>PageAttention中KV存储每个block中最大的token个数。</td>
        <td>不传时按照0处理。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantMode</td>
        <td>输入</td>
        <td>伪量化的方式</td>
        <td>不支持</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxLseFlag</td>
        <td>输入</td>
        <td>是否输出softmax_lse。</td>
        <td>
        <ul>
            <li>支持S轴外切（增加输出）。</li>
            <li>用户不特意指定时建议传入false。</li>
        </ul>
        </td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>  
    <tr>
        <td>keyAntiquantMode</td>
        <td>输入</td>
        <td>key 的反量化的方式。</td>
        <td>
        <ul>
            <li>不特意指定时建议传入0。</li>
            <li>除了keyAntiquantMode为0并且valueAntiquantMode为1的场景外，需要与valueAntiquantMode一致。</li>
            <li>keyAntiquantMode为0时，代表per-channel模式（per-channel包含per-tensor）。</li>
            <li>keyAntiquantMode为1时，代表per-token模式。</li>
            <li>keyAntiquantMode为2时，代表per-tensor叠加per-head模式。</li>
            <li>keyAntiquantMode为3时，代表per-token叠加per-head模式。</li>
            <li>keyAntiquantMode为4时，代表per-token叠加使用page attention模式管理scale/offset模式。</li>
            <li>keyAntiquantMode为5时，代表per-token叠加per head并使用page attention模式管理scale/offset模式。</li>
            <li>keyAntiquantMode为6时，代表per-token-group模式。</li>
            <li>keyAntiquantMode为7时，代表FP8 per-block全量化模式。</li>
            <li>传入0-7之外的其他值会执行异常。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>valueAntiquantMode</td>
        <td>输入</td>
        <td>value 的反量化的方式。</td>
        <td>
        <ul>
            <li>模式编号与keyAntiquantMode一致。</li>
            <li>用户不特意指定时建议传入0。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>  
    <tr>
        <td>queryQuantMode</td>
        <td>输入</td>
        <td>query反量化的模式。</td>
        <td>
        <ul>
            <li>模式编号与keyAntiquantMode一致。</li>
            <li>用户不特意指定时建议传入0。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>pseType</td>
        <td>输入</td>
        <td>pse的方式。</td>
        <td>
        <ul>
            <li>支持配置值为0、2、3（pseType = 1推理场景不支持）。</li>
            <li>pseType为0时，外部传入pse，先mul再add。</li>
            <li>pseType为2时，内部生成pse，计算公式：-alibi_slope * abs(i - j)。</li>
            <li>pseType为3时，内部生成pse，计算公式：-alibi_slope * sqrt(abs(i - j))。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>attentionOut</td>
        <td>输出</td>
        <td>公式中的输出。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8、FLOAT8_E4M3FN、HIFLOAT8</td>
        <td>ND</td>
        <td>该入参的D维度与value的D保持一致，其余维度需要与入参query的shape保持一致。</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxLse</td>
        <td>输出</td>
        <td>ring attention算法对query乘key的结果，先取max得到softmax_max。query乘key的结果减去softmax_max, 再取exp，接着求sum，得到softmax_sum。最后对softmax_sum取log，再加上softmax_max得到的结果。</td>
        <td>
        <ul>
            <li>用户不特意指定时建议传入nullptr。</li>
            <li>数据为inf的代表无效数据；softmaxLseFlag为False时，如果softmaxLse传入的Tensor非空，则直接返回该Tensor数据，如果softmaxLse传入的是nullptr，则返回shape为{1}全0的Tensor。</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>softmaxLseFlag为True时，一般情况下，shape必须为[B, N, Q_S, 1]，当inputLayout为TND/NTD_TND/TND_NTD时，shape必须为[T, N, 1]。</td>
        <td>-</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回用户需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
    <thead>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
    </thead>
    <tbody>
        <tr>
            <td>ACLNN_ERR_PARAM_NULLPTR</td>
            <td>161001</td>
            <td>传入的query、key、value、attentionOut是空指针。</td>
        </tr>
        <tr>
            <td>ACLNN_ERR_PARAM_INVALID</td>
            <td>161002</td>
            <td>query、key、value、pseShift、attenMaskOptional、attentionOut的数据类型和数据格式不在支持的范围内。</td>
        </tr>
        <tr>
            <td>ACLNN_ERR_RUNTIME_ERROR</td>
            <td>361001</td>
            <td>API内存调用npu runtime的接口异常。</td>
        </tr>
    </tbody>
    </table>

## aclnnFusedInferAttentionScoreV5

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
    <col style="width: 184px">
    <col style="width: 134px">
    <col style="width: 833px">
    </colgroup>
    <thead>
        <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        </tr></thead>
    <tbody>
        <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedInferAttentionScoreV5GetWorkspaceSize获取。</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
        </tr>
    </tbody>
    </table>
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

### 一、约束类型说明

FusedInferAttentionScore算子约束分为4个档位，按约束复杂程度递增分为 单参数约束、存在性约束、一致性约束和特性交叉约束，各档位约束内容和示例如下：

- 单参数约束：对于单个接口参数的约束，包含FusedInferAttentionScore算子接口中的 Tensor、TensorList、Array 和 Attributes
  - 对于Tensor、TensorList、Array，单参数约束中包含如下校验
    - 校验 shape，包括shape维度dim、每一维度 dim value
    - 校验 dtype
    - 校验 format
  - 对于属性 Attribute
    - 校验属性取值
- 存在性约束：约束特定场景下，特性参数组内，必须传入某参数，或不支持传入某参数
- 一致性约束：特性参数组内，各个参数间约束。
  - Example 1：属性sparseMode和输入tensor attenMask均属于Attention Mask参数组，sparseMode取值为2/3/4时，attenMask shape必须为（2048，2048），此类约束即为参数组内的一致性约束
- 特性交叉约束：涉及多个参数组，不同参数组间交叉约束
  - Example 1：输入tensor blockTable 和 属性 blockSize 属于Paged Attention（同PA）参数组，输入tensor attenMask属于 Mask参数组；在PA场景下，attenMask输入最后一维（attenMaskS2）需要大于等于maxBlockNumPerSeq * blockSize，此类约束即为多参数组间的交叉约束；**且为保证风格统一，此约束会放在入参顺序靠后的 Paged Attention参数组中**

### 特性参数组

|      特性参数组      |     参数字段名称     |    字段分组    |  字段类型  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      公共参数组      |         query         |      INPUT      |   Tensor   |
|                      |          key          |      INPUT      | TensorList |
|                      |         value         |      INPUT      | TensorList |
|                      |       numHeads       |      ATTR      |   int64   |
|                      |      scaleValue      | ATTR(OPTIONAL) |   double   |
|                      |      inputLayout      | ATTR(OPTIONAL) |    char    |
|                      |   numKeyValueHeads   | ATTR(OPTIONAL) |   int64   |
|                      |     innerPrecise     | ATTR(OPTIONAL) |   int64   |
|                      |     attentionOut     |     OUTPUT     |   Tensor   |
|       PSE参数组       |       pseShift       | INPUT(OPTIONAL) |   Tensor   |
|                      |       qStartIdx       | INPUT(OPTIONAL) |   Tensor   |
|                      |      kvStartIdx      | INPUT(OPTIONAL) |   Tensor   |
|                      |        pseType        | ATTR(OPTIONAL) |   int64   |
|      Mask参数组      |       attenMask       | INPUT(OPTIONAL) |   Tensor   |
|                      |       preTokens       | ATTR(OPTIONAL) |   int64   |
|                      |      nextTokens      | ATTR(OPTIONAL) |   int64   |
|                      |      sparseMode      | ATTR(OPTIONAL) |   int64   |
|  ActualSeqLens参数组  |   actualSeqLengths   | INPUT(OPTIONAL) |  IntArray  |
|                      |  actualSeqLengthsKv  | INPUT(OPTIONAL) |  IntArray  |
|  伪量化&全量化参数组  |       deqScale1       | INPUT(OPTIONAL) |   Tensor   |
|                      |      quantScale1      | INPUT(OPTIONAL) |   Tensor   |
|                      |       deqScale2       | INPUT(OPTIONAL) |   Tensor   |
|                      |    antiquantScale    | INPUT(OPTIONAL) |   Tensor   |
|                      |    antiquantOffset    | INPUT(OPTIONAL) |   Tensor   |
|                      |   keyAntiquantScale   | INPUT(OPTIONAL) |   Tensor   |
|                      |  keyAntiquantOffset  | INPUT(OPTIONAL) |   Tensor   |
|                      |  valueAntiquantScale  | INPUT(OPTIONAL) |   Tensor   |
|                      | valueAntiquantOffset | INPUT(OPTIONAL) |   Tensor   |
|                      |   dequantScaleQuery   | INPUT(OPTIONAL) |   Tensor   |
|                      |     antiquantMode     | ATTR(OPTIONAL) |   int64   |
|                      |   keyAntiquantMode   | ATTR(OPTIONAL) |   int64   |
|                      |  valueAntiquantMode  | ATTR(OPTIONAL) |   int64   |
|                      |    queryQuantMode    | ATTR(OPTIONAL) |   int64   |
|    PostQuant参数组    |      quantScale2      | INPUT(OPTIONAL) |   Tensor   |
|                      |     quantOffset2     | INPUT(OPTIONAL) |   Tensor   |
|                      |        outType        | ATTR(OPTIONAL) |   int64   |
| Paged Attention参数组 |      blockTable      | INPUT(OPTIONAL) |   Tensor   |
|                      |       blockSize       | ATTR(OPTIONAL) |   int64   |
|   LeftPadding参数组   |   queryPaddingSize   | INPUT(OPTIONAL) |   Tensor   |
|                      |     kvPaddingSize     | INPUT(OPTIONAL) |   Tensor   |
|  SystemPrefix参数组  |    keySharedPrefix    | INPUT(OPTIONAL) |   Tensor   |
|                      |   valueSharedPrefix   | INPUT(OPTIONAL) |   Tensor   |
|                      | actualSharedPrefixLen | INPUT(OPTIONAL) |   Tensor   |
|      Rope参数组      |      query_rope      | INPUT(OPTIONAL) |   Tensor   |
|                      |       key_rope       | INPUT(OPTIONAL) |   Tensor   |
|                      | keyRopeAntiquantScale | INPUT(OPTIONAL) |   Tensor   |
|  LearnableSink参数组  |     learnableSink     | INPUT(OPTIONAL) |   Tensor   |
|   SoftmaxLSE参数组   |    softmaxLseFlag    | ATTR(OPTIONAL) |    bool    |
|                      |      softmaxLse      |     OUTPUT     |   Tensor   |

### 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|     GQA     |           在资料约束中，泛指不传入ROPE的所有场景           |
| Prefill MLA | 传入ROPE（包含合并和分离2种模式），且输入Q/K/V headdim为128 |
| Decode MLA |             传入ROPE，且输入Q/K/V headdim为512             |
|      B      |                Batch, 表示输入样本批量大小                |
|     Q_N     |        输入query tensor的头数，对应query shape中的N        |
|    KV_N    |    输入key/value tensor的头数，对应key/value shape中的N    |
|     Q_S     |      输入query tensor的序列长度，对应query shape中的S      |
|    KV_S    |  输入key/value tensor的序列长度，对应key/value shape中的S  |
|     Q_T     |          输入query tensor所有batch序列长度的累加和          |

### 参数组约束

<!--
#### 整体规则
1. 按照 特性参数组 -> 四层约束 -> 量化类型（非量化/伪量化/全量化） -> FA场景（GQA/Prefill MLA/Decode MLA） -> 芯片代际（Atlas A2/Ascend 950PR）层级维护
2. 每个特性组，缩进风格一致，表格宽高无固定约束，保持简洁清晰即可
3. 约束中参数命名，第一次出现必须和接口中参数名一致，后续可通过 “attenMask（同 mask/后称 mask）”方式缩写
4. 对于输入各轴缩写，缩写需要具有自解释性，如QueryS、KeyNumHead、PseShiftS2等，禁止出现 S1、B、n2 等字眼；最好在基准信息说明中补充各个变量含义
-->

#### 公共参数组（ShapeChecker）

- 单参数约束

  - 公共约束

    <table style="undefined;table-layout: fixed; width: 1100px">
        <colgroup>
            <col style="width: 200px">
            <col style="width: 250px">
            <col style="width: 200px">
            <col style="width: 450px">
        </colgroup>
        <thead>
            <tr>
                <th>场景</th>
                <th>Query/Key/Value</th>
                <th>numHeads</th>
                <th>inputLayout</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>GQA</td>
                <td rowspan="3"><ul>
                                    <li>数据类型仅支持均为FLOAT16/BFLOAT16</li>
                                    <li>数据格式均仅支持ND</li>
                                </ul>
                </td>
                <td rowspan="2">无限制</td>
                <td rowspan="2">支持BNSD、BSND、BSH、TND、NTD、BSH_BNSD、BSND_BNSD、NTD_TND、BNSD_BSND</td>
            </tr>
            <tr>
                <td>Prefill MLA</td>
            </tr>
            <tr>
                <td>Decode MLA</td>
                <td>仅支持1,2,4,8,16,32,64,128</td>
                <td>支持BNSD、BSND、BSH、TND、BNSD_NBSD、BSND_NBSD、BSH_NBSD、TND_NTD</td>
            </tr>
        </tbody>
    </table>
- 存在性约束

  <table style="undefined;table-layout: fixed; width: 700px">
        <colgroup>
            <col style="width: 400px">
            <col style="width: 300px">
        </colgroup>
        <thead>
            <tr>
                <th>场景</th>
                <th>结果预期</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>query,key,value,attentionOut存在nullptr</td>
                <td>正常拦截</td>
            </tr>
            <tr>
                <td>query,attentionOut的tensor的shapeSize为0</td>
                <td>attentionOut为返回空tesor</td>
            </tr>
            <tr>
                <td>query,attentionOut的tensor的shapeSize不为0,且key,value的tensor的shapeSize为0</td>
                <td>attentionOut返回全0</td>
            </tr>
        </tbody>
    </table>
- 一致性约束

  <table style="undefined;table-layout: fixed; width: 1500px">
        <colgroup>
            <col style="width: 200px">
            <col style="width: 300px">
            <col style="width: 300px">
            <col style="width: 350px">
            <col style="width: 350px">
        </colgroup>
        <thead>
            <tr>
                <th>场景</th>
                <th>B</th>
                <th>N</th>
                <th>D</th>
                <th>numHeads/numKeyValueHeads</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>GQA</td>
                <td rowspan="3">
                    <ul>
                        <li>支持B轴小于等于65536</li>
                        <li>非连续场景下 key、value的tensorlist中的batch只能为1,B不能大于256</li>
                    </ul>
                </td>
                <td rowspan="3">
                    query的layout不为BSH时，query/attentionOut的N轴与numHeads保持一致,key/value的N轴与numKeyValueHeads保持一致
                </td>
                <td>
                    <ul>
                        <li>支持D轴小于等于512</li>
                        <li>非量化场景下，当query/key/value三组HeadDim均小于等于128且layout不为NTD/NTD_TND/BSH_BNSD/BSND_BNSD/BNSD_BSND时，支持参数query/key的HeadDim与value的HeadDim不相等
                        </li>
                        <li>layout为NTD/BSH_BNSD/BSND_BNSD时，HeadDim仅支持64或128</li>
                        <li>layout为BNSD_BSND时，HeadDim需要16对齐</li>
                        <li>layout为BNSD_BSND，且AttentionOut数据类型为INT8时,HeadDim需要32对齐</li>
                    </ul>
                </td>
                <td rowspan="2">
                    numHeads需可整除numKeyValueHeads，numHeads与numKeyValueHeads的比值无限制
                </td>
            </tr>
            <tr>
                <td>Prefill MLA</td>
                <td>支持参数query/key的HeadDim为192，value的HeadDim为128场景，其余场景query/key/value/attentionOut的HeadDim需保持一致</td>
            </tr>
            <tr>
                <td>Decode MLA</td>
                <td>query/key/value/attentionOut的HeadDim需保持一致</td>
                <td>仅numHeads为1、2、4、8、16、32、64、128, numKeyValueHeads为1。</td>
            </tr>
        </tbody>
    </table>
- 特性交叉约束

  - 非量化场景
    <table style="undefined;table-layout: fixed; width: 1000px">
        <colgroup>
            <col style="width: 200px">
            <col style="width: 400px">
            <col style="width: 400px">
        </colgroup>
        <thead>
            <tr>
                <th>场景</th>
                <th>不支持场景</th>
                <th>PagedAttention场景</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>公共</td>
                <td>
                    query的layout为TND/NTD时，不支持pseShift、tensorlist
                </td>
                <td rowspan="4">
                    <ul>
                        <li>当inputLayout为BNSD、TND、BSH、BSND,Key/Value排布支持BnBsH（blockNum, blockSize, H）、BnNBsD（blockNum, KV_N,
                            blockSize, D）和NZ（blockNum，KV_N，D/16，blockSize，16）三种格式</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>GQA</td>
                <td>query/key的HeadDim与value的HeadDim不相等时,除attenMask参数组外，其余均不支持</td>
            </tr>
            <tr>
                <td>Prefill MLA</td>
                <td>
                - <term> Ascend 950PR/Ascend 950DT </term> 
                  - 不支持全量化、伪量化
                - <term> Atlas A3 训练系列产品/Atlas A3 推理系列产品 </term> 和 <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
                  - 不支持全量化、伪量化、tensorlist、左padding
                </td>
            </tr>
            <tr>
                <td>Decode MLA</td>
                <td>不支持tensorlist、左padding、伪量化、prefix</td>
            </tr>
        </tbody>
    </table>

#### PSE参数组

- 单参数约束
  - 公共
    - 入参 pseType 应满足以下条件：
      - pseType 必须为0, 2 或者 3
      - pseType 为 1 不支持 FA 推理场景，仅支持 FA 训练场景
    - 入参 pseShift 应满足以下条件：
      - tensor 的数据类型应满足以下条件：
        - pseType 为 2 或者 3 时，tensor 的数据类型必须为 FLOAT32
        - pseType 为 0  时，且 query 的数据类型为 FLOAT16 或者 INT8 时，tensor 的数据类型必须为 FLOAT16
        - pseType 为 0 时，且 query 的数据类型为 BFLOAT16 时，tensor 的数据类型必须为 BFLOAT16
      - tensor shape 应满足以下条件：
        - pseType 为 2 或者 3 时，tensor shape 应为 (Q_N)
        - pseType 为 0 时:
          - tensor shape 的维度必须为 4
          - P_S1(tensor shape 的第 3 维) > 1 时：
            - tensor shape 的第 1 维应等于 1 或者 B
            - tensor shape 的第 2 维应等于 Q_N
            - tensor shape 的第 3 为应大于等于 Q_S
            - 非 prefix 场景时，tensor shape 的第 4 维应大于等于 KV_S
            - prefix 场景时，tensor shape 的第 4 维应大于等于KV_S + actualSharedPrefixLen
          - P_S1(tensor shape 的第 3 维) = 1 时：
            - tensor shape 的第 1 维应等于 1 或者 B
            - tensor shape 的第 2 维应等于 Q_N
            - 非 prefix 场景时，tensor shape 的第 4 维应大于等于 KV_S
            - prefix 场景时，tensor shape 的第 4 维应大于等于 KV_S + actualSharedPrefixLen
- 存在性约束
  - 公共
    - 入参 pseShift 应满足以下条件：
      - pseType 为 2 或者 3 时，必须传入 pseShift
- 一致性约束
  - 公共
    - pseType 为 2 或者 3 时， 入参 qStartIdx 和 kvStartIdx 应满足以下条件：
      - qStartIdx 的取值范围应满足 [-2147483648, 2147483647]
      - kStartIdx 的取值范围应满足 [-2147483648, 2147483647]
      - kvStartIdx - qStartIdx 的取值范围应满足 [-1048576, 1048576]
      - 若 qStartIdxOptional 或 kvStartIdxOptional 非空，则取列表中的第一个数据作为 qStartIdx 或 kvStartIdx
- 特性交叉约束
  - 公共
    - PagedAttention场景下，入参 pseShift 应满足以下条件：
      - tensor shape 的最后一维应大于等于 maxBlockNumPerBatch * blockSize
    - alibi 场景下，Q_S 应等于 KV_S
    - MLA场景下，入参 pseShift 应满足以下条件：
      - 不支持 pse，不能传入 pseShift
    - D 不等长场景下，入参 pseShift 应满足以下条件：
      - 不支持 pse，不能传入 pseShift

#### Attention Mask参数组

- 单参数约束
  - 公共
    - 入参 attenMask 需要满足以下条件：

      - tensor dtype 为 INT8/UINT8/BOOL 类型
      - tensor format 为 ND/NCHW/NHWC/NCDHW 类型
    - 入参 sparseMode 需要满足以下条件：

      - sparseMode 支持输入范围为 0-4，默认值为 0
      - sparseMode 在不使能 mask 时，仅支持输入为0
      - sparseMode 含义如下表所示（注：attenMask矩阵示例部分中的1 = masked out，0 = keep）

      | sparseMode |                                                                               含义                                                                               |      attenMask矩阵示例      |                                             备注                                             |
      | :--------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------: | :-------------------------------------------------------------------------------------------: |
      |     0     | defaultMask模式，如果attenMask未传入则不做mask操作，忽略preTokens和nextTokens；<br />如果传入，则需要传入完整的attenMask矩阵，计算preTokens和nextTokens之间的部分 | 11111<br />01111<br />00111 |          完整的attenMask矩阵，即全量的Q_S*KV_S矩阵，<br />因此可以自定义不同mask场景          |
      |     1     |                                                allMask模式，必须传入完整的attenMask矩阵，忽略preTokens和nextTokens                                                | 00101<br />10111<br />10101 |                                             同上                                             |
      |     2     |                                            leftUpCausal模式，需要传入优化后的attenMask矩阵，忽略preTokens和nextTokens                                            | 01111<br />00111<br />00011 | 优化后的attenMask矩阵，固定为2048*2048的下三角矩阵，<br />以左上顶点为参数起点划分，对角线全0 |
      |     3     |                                           rightDownCausal模式，需要传入优化后的attenMask矩阵，忽略preTokens和nextTokens                                           | 00011<br />00001<br />00000 | 优化后的attenMask矩阵，固定为2048*2048的下三角矩阵，<br />以右下顶点为参数起点划分，对角线全0 |
      |     4     |                                           band模式，需要传入优化后的attenMask矩阵，计算preTokens和nextTokens之间的部分                                           | 00011<br />10001<br />11000 | 优化后的attenMask矩阵，固定为2048*2048的下三角矩阵，<br />以右下顶点为参数起点划分，对角线全0 |
      |     5     |                                                                              prefix                                                                              |              -              |                                            不支持                                            |
      |     6     |                                                                              global                                                                              |              -              |                                            不支持                                            |
      |     7     |                                                                              dilated                                                                              |              -              |                                            不支持                                            |
      |     8     |                                                                            block_local                                                                            |              -              |                                            不支持                                            |
- 存在性约束
  - 无
- 一致性约束
  - 公共
    - 入参 attenMask 的输入维度仅支持 2/3/4
      - `<term>` Ascend 950PR/Ascend 950DT `</term>`
        - 维度为 2 时，不支持 sparseMode 为 0/1 模式
      - `<term>` Atlas A3 训练系列产品/Atlas A3 推理系列产品 `</term>` 和 `<term>`Atlas A2 训练系列产品/Atlas A2 推理系列产品 `</term>`
        - 维度为 2 时，支持 sparseMode 为 0/1 模式
    - 入参 sparseMode 为 0/1 模式时，attenMask矩阵的 shape 应满足 [batchSize/1，>=Q_S，>=KV_S]
    - 入参 sparseMode 为 2/3/4 模式时，attenMask矩阵的 shape 最后两维应等于2048
    - 非伪量化或 Q_S 大于1时，preTokens 与 nextTokens 应满足 nextTokens * (-1) <= preTokens，以确保具有有效数据
  - 伪量化
    - Q_S等于1时，attenMask 输入维度仅支持 3/4，且 attenMask 输入的 shape 应满足，第一维等于 batchSize 或 1，最后一维应大于等于 blockTable 的第二维 * blockSize
    - Q_S大于1时，若 sparseMode 为 4 模式，且 attentionOut 为 int8 类型时，则 preTokens 与 nextTokens 均不能为负数
- 特性交叉约束
  - 非量化
    - Decode MLA场景下，sparseMode 仅支持 0/3/4 模式
    - GQA 场景下，当 query，key 及 value 的 head dim 不相等时，sparseMode 仅支持 0/2/3 模式
  - 全量化
    - Decode MLA场景下，Q_S 等于1时，sparseMode 仅支持 0 模式，且不支持传入 attenMask 矩阵
    - Decode MLA场景下，Q_S 大于1时，如果query/key/value的类型为 FLOAT8_E4M3FN，sparseMode 仅支持 0/3 模式，且 0 模式下不支持传入 attenMask 矩阵；如果query/key/value的类型为 INT8，sparseMode 仅支持 3 模式

#### ActualSeqLen参数组

- 单参数约束
  - 公共
    - 入参 actualSeqLengths(query 的 actualSeqLengths)应满足以下条件：
      - 长度应满足以下条件：
        - 当 query 的 layout 为 TND/NTD 时，长度应等于 batch 数
        - 当 query 的 layout 为非 TND/NTD 时，长度应等于 1 或者 大于等于 query 的 batch 值
      - 入参中的数值应满足以下条件：
        - 当 query 的 layout 为 TND/NTD 时，其值应递增(大于等于前一个值)排列
        - 当 query 的 layout 为 TND/NTD 是，最后一个元素应等于 T
        - 当 query 的 layout 为非 TND/NTD 时，其值应不大于 Q_S
        - 其值应为非负数
    - 入参 actualSeqLengthsKv(key/value 的 actualSeqLengths)应满足以下条件：
      - 长度应满足以下条件：
        - 当 key/value 的 layout 为 TND/NTD 时，长度应等于 batch 数
        - 当 key/value 的 layout 为非 TND/NTD 时，长度应等于 key/value 的 batch 值
      - 入参中的数值应满足以下条件：
        - 当 key/value 的 layout 为 TND/NTD 时，最后一个元素应等于 T
        - 当 key/value 的 layout 为 TND/NTD 时，其值应递增(大于等于前一个值)排列
        - 当 key/value 的 layout 为非 TND/NTD 时，其值应不大于 KV_S
        - 其值应为非负数
- 存在性约束
  - 公共
    - 入参 actualSeqLengths(query 的 actualSeqLengths)应满足以下条件:
      - 当 query 的 layout 为 TND/NTD 时，必须传入 actualSeqLengths
    - 入参 actualSeqLengthsKv(key/value 的 actualSeqLengths)应满足以下条件：
      - 当 key/value 的 layout 为 TND/NTD 时，必须传入 actualSeqLengthsKv
      - PagedAttention场景下，必须传入 actualSeqLengthsKv
- 一致性约束
  - 无
- 特性交叉约束
  - 公共
    - alibi pse (pseType 为 2 或 3)场景下，入参 actualSeqLengths 和 actualSeqLengthsKv 应满足以下条件：
      - actualSeqLengths 和 actualSeqLengthsKv 在每个 batch 的数值需要相等
  - 全量化
    - Decode MLA 场景下，若传入 actualSeqLengths，query 的 layout 必须为 TND/NTD

#### 伪量化/全量化参数组（DequantChecker）

- 单参数约束
  - 伪量化场景
    - 入参 keyAntiquantMode 和 valueAntiquantMode 应满足以下条件：
      - 入参中的数值应满足以下条件：
        - 其值应为 0(per-channel/per-tensor)、1(per-token)、2(per-tensor 叠加 per-head)、3(per-token 叠加 per-head)、
          4(per-token 模式使用 page attention 管理 scale/offset)、
          5(per-token 叠加 per-head 模式并使用 page attenion 管理 scale/offser)、6(per-token-group)
        - 除 key 支持 per-channel 叠加 value 支持 per-token，keyAntiquantMode 和 valueAntiquantMode 应相等
    - 入参 keyAntiquantScale 和 valueAntiquantScale 应满足以下条件：
      - 入参的数据类型应满足以下条件：
        - per-channel(per-tensor)，数据类型应与 query 相同
        - per-token，数据类型仅支持 FLOAT32
        - per-tensor 叠加 per-head，数据类型应与 query 相同
        - per-token 叠加 per-head，数据类型仅支持 FLOAT32
        - per-token 模式使用 page attention 管理 scale/offset，数据类型仅支持 FLOAT32
        - key 支持 per-channel 叠加 value 支持 per-token，数据类型仅支持 FLOAT32
        - per-token-group，数据类型仅支持 FLOAT8_E8M0
      - 入参的 shape 应满足以下条件：
        - per-channel，shape 应为 (1, N, 1, D)、(1, 1, N, D)、(1, N, D)、(1, H)、(N, 1, D)、(N, D)、(H)
        - per-tensor，shape 应为 (1)
        - per-token，shape 应为 (1, B, >=KV_S)、(B, >=KV_S)
        - per-tensor 叠加 per-head，shape 应为 (N)
        - per-token 叠加 per-head，shape 应为 (B, N, >=KV_S)
        - per-token 模式使用 page attention 管理 scale/offset，shape 应为 (blockNum, blockSize)
        - ter-token 叠加 per-head 模式并使用 page attention 管理 scale/offset, shape 应为 (blockNum, N, blockSize)
        - key 支持 per-channel 叠加 value 支持 per-token，keyAntiquantScale 的 shape 应为 (1, N, 1, D)、(1, N, D)、(1, H)、(N, 1, D)、(N, D)、(H)
        - key 支持 per-channel 叠加 value 支持 per-token，valueAntiquantScale 的 shape 应为 (1, B, >=KV_S)、(B, >=KV_S)
        - per-token-group，shape 应为 (1, B, N, >=KV_S, D/32)
    - 入参 key 和 value 应满足以下条件：
      - 入参的数据类型应满足以下条件：
        - per-channel(per-tensor) 模式，其支持的数据类型为 INT8、INT4(INT32)、HIFLOAT8、FLOAT8_E4M3FN
        - per-token 模式，其支持的数据类型为 INT8、INT4(INT32)
        - per-tensor 叠加 per-head 模式，其支持的数据类型为 INT8
        - per-token 叠加 per-head 模式，其支持的数据类型为 INT8、INT4(INT32)
        - per-token 模式使用 page attenion 管理 scale/offset，其支持的数据类型为 INT8
        - key 支持 per-channel 叠加 value 支持 per-token，其支持的数据类型为 INT8、INT4(INT32)
        - per-token-group，其支持的数据类型为 FLOAT4_E2M1
  - 全量化场景
    - Decode MLA 全量化
      - 入参 dequantScaleQuery、keyAntiquantScale、valueAntiquantScale 的dtype为FLOAT32类型
      - 入参 keyAntiquantScale、valueAntiquantScale 的shape为（1）
      - 入参 keyAntiquantMode、valueAntiquantMode 为0（per-tensor模式），queryQuantMode为3（per-token叠加per-head模式）
    - per-block 全量化
      - 入参 queryQuantMode、keyAntiquantMode、valueAntiquantMode 为7
    - per-tensor 全量化
      - 入参 deqScale1、deqScale2 的dtype支持 UINT64、FLOAT32
      - 入参 quantScale1 的dtype支持FLOAT32
      - 入参 deqScale1、quantScale1、deqScale2 的shape为（1）
      - 入参 queryQuantMode、keyAntiquantMode、valueAntiquantMode 为0
- 存在性约束
  - 伪量化场景
    - 入参 keyAntiquantScale 和 valueAntiquantScale 应满足以下条件：
      - 必须传入 keyAntiquantScale
      - 必须传入 valueAntiquantScale
    - 入参 keyAntiquantOffset 和 valueAntiquantOffset 应满足以下条件：
      - 传入 keyAntiquantOffset 时，必须传入 valueAntiquantOffset
      - 传入 valueAntiquantOffset 时，必须传入 keyAntiquantOffset
      - 当 key/value 的数据类型为 FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1时，不支持 offset，不能传入 keyAntiquantOffset
      - 当 key/value 的数据类型为 FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1时，不支持 offset，不能传入 valueAntiquantOffset
  - 全量化场景
    - Decode MLA 全量化
      - dequantScaleQuery、keyAntiquantScale、valueAntiquantScale 需同时存在
      - 不支持传入 keyAntiquantOffset、valueAntiquantOffset
      - 不支持传入 deqScale1、quantScale1、deqScale2
      - 不支持传入 antiquantScale、antiquantOffset
      - 不支持传入 keyRopeAntiquantScale
    - per-block 全量化
      - dequantScaleQuery、keyAntiquantScale、valueAntiquantScale 需同时存在
      - 不支持传入 keyAntiquantOffset、valueAntiquantOffset
      - 不支持传入 deqScale1、quantScale1、deqScale2
      - 不支持传入 antiquantScale、antiquantOffset
      - 不支持传入 keyRopeAntiquantScale
    - per-tensor 全量化
      - deqScale1、quantScale1、deqScale2 需同时存在
      - 不支持传入 dequantScaleQuery、keyAntiquantScale、valueAntiquantScale
      - 不支持传入 keyAntiquantOffset、valueAntiquantOffset
      - 不支持传入 antiquantScale、antiquantOffset
      - 不支持传入 keyRopeAntiquantScale
- 一致性约束
  - 无
- 特性交叉约束
  - 伪量化场景
    - key/value 的数据类型为 INT8 时，query 的 layout 不支持 TND
    - Q_S > 1 时：
      - key/value 的数据类型为 INT8 时，keyAntiquantMode 不支持 2，3，4，5
      - key/value 的数据类型为 INT8，且keyAntiquantMode 为0或1时，query 和 output 的数据类型仅支持 BF16
      - key/value 的数据类型为 INT8，且keyAntiquantMode 为0或1时，Q_S长度不能大于16
      - key/value 的数据类型为 INT8，且keyAntiquantMode 为0或1时，不支持 tensor list
      - key/value 的数据类型为 INT8，且keyAntiquantMode 为0或1时，不支持左 padding
      - key/value 的数据类型为 INT8，且keyAntiquantMode 为0或1时，不支持 page attention
      - key/value 的数据类型不支持 INT4、INT32
    - page attention 场景下，入参 keyAntiquantScale 和 valueAntiquantScale 应满足以下条件：
      - tensor shape 应满足以下条件：
        - per-token 模式，shape 的最后一维应大于等于 maxBlockNumPerBatch * blockSize
        - per-token 叠加 per-head 模式，shape 的最后一维应大于等于 maxBlockNumPerBatch * blockSize
        - per-token-group 模式，shape 的倒数第二维应大于等于 maxBlockNumPerBatch * blockSize
  - 全量化场景
    - Decode MLA 全量化
      - query、key、value 的dtype为 FLOAT8_E4M3FN/INT8
      - attenOut 的dtype为 BFLOAT16
      - queryRope、keyRope 的dtype为BFLOAT16
      - 当query/key/value是 FLOAT8_E4M3FN 时, inputLayout 仅支持 BSH、BSND、BNSD、TND；当 query/key/value 类型为 INT8 时, inputLayout 仅支持BSH、BSND、TND、BSH_NBSD、BSND_NBSD、TND_NTD
      - 当 query/key/value 类型为 FLOAT8_E4M3FN 时, Q_N 仅支持32、64、128；当query/key/value类型为INT8时, Q_N 仅支持1、2、4、8、16、32、64、128
      - KV_N必须为1；G 支持 [1, 128]；Q_S 支持[1,16]
      - 当query的inputLayout为BSH时，dequantScaleQuery的shape应该为（B, Q_S, Q_N）；当query的inputLayout为BSND、BNSD、TND时，dequantScaleQuery 的shape相比query仅少一个维度D，且每一维需要和query的对应维度保持一致
      - 不支持公共前缀场景、不支持pse场景、不支持alibi场景、不支持左padding场景
      - 当 query/key/value 类型为 INT8 时，仅支持 PagedAttention场景，且kv cache排布为NZ格式
    - per-block 全量化
      - query、key、value 的dtype支持FLOAT8_E4M3FN、HIFLOAT8
      - inputLayout 支持 BNSD、BSH、BSND、BNSD_BSND、NTD_TND
      - attentionOut dtype 支持 FLOAT16、BFLOAT16
      - Q_S 支持 [1,16]; Q_N 支持 [1, 256]；KV_N 支持 [1, 256]；G 支持 [1, 64]；D 支持 [1, 128]
      - 当inputLayout 为NTD_TND时, keyAntiquantScale、valueAntiquantScale的shape为(KV_N, floor(KV_T, 256) + B, ceil(D, 256))，其他场景shape为(B, KV_N, ceil(KV_S, 256), 1)
      - 当inputLayout 为NTD_TND时，dequantScaleQuery 的shape为(Q_N, floor(Q_T, 128) + B, ceil(D, 256))，其他场景shape为(B, Q_N, ceil(Q_S, 128), 1)
      - innerPrecise 仅支持0、1
      - 不支持tensorlist场景、不支持左padding场景、不支持pse场景、不支持alibi场景、不支持Rope存在、不支持Mask场景、不支持PagedAttention场景、不支持SoftmaxLSE场景、不支持公共前缀场景
    - per-tensor 全量化
      - query、key、value 的dtype为INT8
      - attentionOut 的dtype支持FLOAT16、BFLOAT16
      - Q_S 不能为1
      - Q_N 支持 [1, 256]；KV_N 支持 [1, 256]；G 支持 [1,64]；D 支持 [1,512]
      - inputLayout 支持 BSH、BNSD、BSND、BNSD_BSND
      - 不支持PagedAttention场景、不支持alibi场景、不支持Rope存在、不支持后量化、不支持D不等长场景、不支持公共前缀场景

#### 后量化参数组（PostQuantChecker）

- 单参数约束
  - 公共
    - 入参 quantScale2 需要满足以下条件：
      - tensor dtype 为 BF16/FP32 类型
    - 入参 quantOffset2 需要满足以下条件：
      - tensor dtype 为 BF16/FP32 类型
    - PostQuant 场景下，输出 attenOut 的数据类型仅支持 INT8/FP8_E4M3FN/HIFLOAT8
- 存在性约束
  - 公共
    - PostQuant 场景下，必须传入 quantScale2
- 一致性约束
  - 公共
    - PostQuant 场景下，当 quantScale2 维度大于1且量化方式为 per-channel 时，若 layout 为 BSH/BSND/BNSD/BNSD_BSND，则 quantScale2 仅支持 shape 为 queryN*valueD，否则仅支持 [numHeads, vHeadDim]
    - PostQuant 场景下，当 quantScale2 维度等于1且量化方式为 per-tensor 时，quantScale2 仅支持 shape 为 (1)
    - PostQuant 场景下，当 quantOffset2 存在时，quantOffset2 应与 quantScale2 保持相同 shape 及数据类型
- 特性交叉约束
  - 公共
    - PostQuant 场景下，当 query 输入类型不为 BF16 时，quantScale2 仅支持 FP32 类型
  - 非量化
    - PostQuant 场景下，当存在 prefix 时，仅支持输出 attenOut 的数据类型为 INT8
  - 伪量化
    - PostQuant 场景下，输出 attenOut 的数据类型仅支持与输入 Key、Value 数据类型相同

#### Paged Attention参数组

- 单参数约束
  - 公共
    - 入参 blockTable 需要满足以下条件：
      - tensor dtype 为int32类型
      - tensor shape 为2维，每一维dim value取值均不能为0，第一维长度需等于Batch size，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actualSeqLengthsKv对应的block数量）
    - 入参 blockSize 需要满足以下条件：
      - blockSize 需要大于0
      - blockSize是用户自定义的参数，该参数的取值会影响PagedAttention的性能，通常情况下，PagedAttention可以提高吞吐量，但会带来性能上的下降，调大blockSize会有一定性能收益
  - 非量化
    - 入参 blockSize 需要满足以下条件：
      - Decode MLA/Prefill MLA场景：blockSize 16对齐，最大支持1024
      - GQA场景：QueryHeadDim/KeyHeadDim/ValueHeadDim均为64或128时，blockSize 16对齐，最大支持1024；其他情况下，若Q_S> 1，blockSize 128对齐，最大支持512，若Q_S= 1，blockSize 16对齐，最大支持512
  - 伪量化
    - 入参 blockSize 需要满足以下条件：需要根据key、value dtype size 32B对齐，最大支持512。即当key、value dtype为INT8/HIFLOAT8/FLOAT8_E4M3FN 时，blockSize 需要32对齐，即当key、value dtype为INT4(INT32)、FLOAT4_E2M1 时，blockSize 需要64对齐
  - 全量化
    - Decode MLA 全量化场景下，仅支持blockSize取值128
- 存在性约束
  - 公共
    - PagedAttention 使能情况下，必须传入 actualSeqLengthsKv
    - PagedAttention 不支持tensorlist场景，不支持左padding场景，不支持公共前缀场景，不支持D不等长场景
- 一致性约束
  - 无
- 特性交叉约束
  - 公共
    - PagedAttention的使能场景下，若同时使能attenMask，传入attenMask的最后一维需要大于等于 maxBlockNumPerSeq * blockSize
    - PagedAttention场景下，kv cache排布为BnNBsD时性能通常优于kv cache排布为BnBsH时的性能，建议优先选择BnNBsD格式。
    - PagedAttention场景下，当输入kv cache排布格式为BnBsH（blocknum, blocksize, H），且 KV_N * D 超过65535时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小 KV_N）或调整kv cache排布格式为BnNBsD（blocknum, KV_N, blocksize, D）解决。
  - 全量化
    - Decode MLA场景下：
      - 当query的数据类型为FP8_E4M3FN，且inputLayout为BSH、BSND时，kv cache排布只支持BnBsH（blocknum, blocksize, H）和NZ (blocknum，KV_N，D/D0，blocksize，D0) 两种格式
      - 当query的数据类型为FP8_E4M3FN，且inputLayout为BNSD、TND时，kv cache排布支持BnBsH（blocknum, blocksize, H）、BnNBsD（blocknum, KV_N, blocksize, D）和NZ（blocknum，KV_N，D/D0，blocksize，D0）三种格式
      - 当query的数据类型为INT8时，kv cache排布仅支持NZ，且kv cache排布为 (blocknum，KV_N，D/D0，blocksize，D0)
      - 当kv cache排布为NZ时，最后一维D0是32, keyRope 最后一维D0是16
    - GQA全量化场景不支持PagedAttention

#### 左padding参数组

- 单参数约束
  - 非量化
    - Query 左padding 场景下，queryPaddingSize 的 shape 应为 (1)
    - Key、value 左padding 场景下，kvPaddingSize 的 shape 应为 (1)
- 存在性约束
  - 公共
    - Query 左padding 场景下，必须传入 queryPaddingSize
    - Key、value 左padding 场景下，必须传入 kvPaddingSize
- 一致性约束
  - 无
- 特性交叉约束
  - 公共
    - 左padding 场景下，不支持 pagedAttention
    - 左padding 场景下，不支持 pseType = 2/3
    - 左padding 场景下，不支持 BSH_BNSD、BSND_BNSD、TND、NTD、NTD_TND、TND_NTD 场景
    - 左padding 场景下，必须传入 actualSeqLengths/actualSeqLengthsKv

#### 公共前缀参数组

- 单参数约束
  - 入参 keySharedPrefix 和 valueSharedPrefix 应满足以下条件：
    - tensor shape 应满足以下条件：
      - shape 应为 (1)
      - layout 为 BNSD 和 BSND 时，N 轴和 D 轴应与 key/value 的 N 轴和 D 轴相等
      - layout 为 BSH 时，H 轴应与 key/value 的 H 轴相等
      - keySharedPrefix 和 valueSharedPrefix 的 S 轴应相等
    - tensor 数据类型应满足以下条件：
      - 数据类型应与 key/value 的数据类型相同
  - 入参 actualSharedPrefixLen 应满足以下条件：
    - shape 应满足以下条件：
      shape 应为 1
    - 入参中的数值应满足以下条件：
      - 其值不能大于 keySharedPrefix 和 valueSharedPrefix 的 shape 的 S 轴
- 存在性约束
  - 公共
    - 入参 keySharedPrefix 和 valueSharedPrefix 应满足以下条件：
      - 传入 keySharedPrefix 时，必须传入 valueSharedPrefix
      - 传入 valueSharedPrefix 时，必须传入 keySharedPrefix
- 一致性约束
  - 无
- 特性交叉约束
  - 公共
    - 不支持 PagedAttention 场景
    - 不支持 tensorlist 场景
    - 不支持左 padding 场景
    - 不支持 alibi 场景
    - 不支持 TND 场景
    - 不支持 Prefill MLA (包括 D 不等长和 ROPE 独立输入)场景
    - 不支持 Decode MLA 场景
  - 全量化
    - 全量化（包括 MLA 全量化和 qkv FP8 per-block全量化）场景，不支持 prefix
    - 后量化场景，仅支持数据类型 INT8
  - 伪量化
    - 伪量化 key/value 合成场景所有量化模式 prefix 均支持
    - 伪量化 key/value 分离场景，prefix 仅支持以下量化模式：
      - Q_S > 1 时，伪量化方式为 per-channel(per-tensor)、per-token 时，key/value 数据类型仅支持 INT8
      - Q_S = 1 时，伪量化方式为 per-tensor、per-tensor 叠加 per-head、per-token 叠加使用 page attention 模式管理
        scale/offset、per-token 叠加 per-head 并使用 page attention 模式管理 scale/offset，key/value 数据类型仅支持INT8
      - Q_S = 1 时，伪量化方式为 per-channel、per-token、per-token 叠加 per-head、key 支持 per-channel
        叠加 value 支持 per-token，key/value 数据类型支持 INT8、INT4(INT32)

#### Rope参数组

- 单参数约束
  - 公共
    - 入参 queryRope 和 keyRope 需要满足以下条件
      - tensor dtype 为 FLOAT16/BFLOAT16
      - tensor shape 中D维为 64
- 存在性约束
  - 公共
    - 入参 queryRope 和 keyRope 必须同时存在
- 一致性约束
  - 无
- 特性交叉约束
  - 公共
    - query shape的D仅支持128、512
    - 非tensorlist场景,  queryRope shape 维度需要和 query 保持一致，除了 queryRope shape 的D为64外,  其余维度需要和 query 一致；keyRope shape 维度需要和 key 保持一致，除了 keyRope shape 的D为64外,  其余维度需要和 key 一致
    - 非量化场景，入参 queryRope 和 keyRope 的 dtype 需要和 query、key 保持一致
    - 不支持公共前缀场景、不支持pse场景、不支持alibi场景
    - 不支持伪量化场景
  - Decode MLA
    - Layout 仅支持 BNSD、BSND、BSH、TND、BNSD_NBSD、BSND_NBSD、BSH_NBSD、TND_NTD
    - Q_N 支持 1/2/4/8/16/32/64/128；KV_N 仅支持1
    - 非量化场景，Q_S无限制；全量化场景，Q_S 支持 [1,16]
    - 不支持左padding场景、不支持tensorlist场景
  - Prefill MLA
    - Layout 仅支持 BNSD、BSND、BSH、TND、NTD、BSH_BNSD、BSND_BNSD、NTD_TND、BNSD_BSND
    - 在tensorlist场景下，传入 keyRope shape 中D为64，B需要和 key 的tensorlist长度保持一致，N、S需要与 key 的tensorlist中每个tensor的N、S相等
    - 不支持全量化场景

#### LearnableSink参数组

- 单参数约束
  - 公共
    - 入参 learnableSink 需要满足以下条件
      - tensor dtype 为 FLOAT16/BFLOAT16
      - tensor shape 为 (Q_N)
- 存在性约束
  - 无
- 一致性约束
  - 无
- 特性交叉约束
  - 公共
    - LearnableSink 使能场景下，tensor dtype需要和query dtype保持一致
    - LearnableSink 使能场景下，Q_N 仅支持 64、128、192
    - LearnableSink 使能场景下，innerPrecise 必须为高精度模式 (0)
    - LearnableSink 不支持左padding场景、不支持公共前缀场景、不支持pse场景、不支持alibi场景
    - LearnableSink 不支持全量化场景、不支持伪量化场景、不支持Decode MLA场景

#### SoftmaxLSE参数组

- 单参数约束
  - 公共
    - 输出 lseOut 仅支持数据类型 FP32
- 存在性约束
  - 公共
    - softmaxLSE 场景下，输出 lseOut 不应为空
- 一致性约束
  - 无
- 特性交叉约束
  - 公共
    - softmaxLSE 且非空tensor场景下，如输出 layout 为 TND 或 NTD，则 lseOut 输入维度应为 3，且 shape 匹配 [Q_T，Q_N，1]
    - softmaxLSE 且非空tensor场景下，如输出 layout 不为 TND 或 NTD，则 lseOut 输入维度应为 4，且 shape 匹配 [B，Q_N，Q_S，1]

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
  #include <iostream>
  #include <vector>
  #include <math.h>
  #include <cstring>
  #include "acl/acl.h"
  #include "aclnn/opdev/fp16_t.h"
  #include "aclnnop/aclnn_fused_infer_attention_score_v5.h"

  using namespace std;

  #define CHECK_RET(cond, return_expr) \
      do {                               \
        if (!(cond)) {                   \
          return_expr;                   \
        }                                \
      } while (0)

  #define LOG_PRINT(message, ...)     \
      do {                              \
        printf(message, ##__VA_ARGS__); \
      } while (0)

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream* stream) {
      // 固定写法，资源初始化
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  int main() {
      // 1. （固定写法）device/stream初始化，acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      int32_t batchSize = 1;
      int32_t numHeads = 2;
      int32_t sequenceLengthQ = 1;
      int32_t headDims = 16;
      int32_t keyNumHeads = 2;
      int32_t sequenceLengthKV = 16;
      std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
      std::vector<int64_t> keyShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
      std::vector<int64_t> valueShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
      std::vector<int64_t> attenShape = {batchSize, 1, 1, sequenceLengthKV}; // B11S
      std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
      void *queryDeviceAddr = nullptr;
      void *keyDeviceAddr = nullptr;
      void *valueDeviceAddr = nullptr;
      void *attenDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *queryTensor = nullptr;
      aclTensor *keyTensor = nullptr;
      aclTensor *valueTensor = nullptr;
      aclTensor *attenTensor = nullptr;
      aclTensor *outTensor = nullptr;
      std::vector<float> queryHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);
      std::vector<float> keyHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
      std::vector<float> valueHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
      std::vector<int8_t> attenHostData(batchSize * sequenceLengthKV, 0);
      std::vector<float> outHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);

      // 创建query aclTensor
      ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建key aclTensor
      ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      int kvTensorNum = 1;
      aclTensor *tensorsOfKey[kvTensorNum];
      tensorsOfKey[0] = keyTensor;
      auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
      // 创建value aclTensor
      ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      aclTensor *tensorsOfValue[kvTensorNum];
      tensorsOfValue[0] = valueTensor;
      auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
      // 创建atten aclTensor
      ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_BOOL, &attenTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
      auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

      int64_t numKeyValueHeads = numHeads;
      double scaleValue = 1 / sqrt(headDims); // 1/sqrt(d)
      int64_t preTokens = 65535;
      int64_t nextTokens = 65535;
      string sLayerOut = "BNSD";
      char layerOut[sLayerOut.length()];
      strcpy(layerOut, sLayerOut.c_str());
      int64_t sparseMode = 0;
      int64_t innerPrecise = 1;
      int blockSize = 0;
      int antiquantMode = 0;
      bool softmaxLseFlag = false;
      int keyAntiquantMode = 0;
      int valueAntiquantMode = 0;
      int queryAntiquantMode = 0;
      // 3. 调用CANN算子库API
      uint64_t workspaceSize = 0;
      int64_t pseType = 0;
      aclOpExecutor* executor;
      // 调用第一段接口
      ret = aclnnFusedInferAttentionScoreV5GetWorkspaceSize(queryTensor, tensorKeyList, tensorValueList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, queryAntiquantMode, pseType, outTensor, nullptr, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用第二段接口
      ret = aclnnFusedInferAttentionScoreV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV5 failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<op::fp16_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
      }

      // 6. 释放资源
      aclDestroyTensor(queryTensor);
      aclDestroyTensor(keyTensor);
      aclDestroyTensor(valueTensor);
      aclDestroyTensor(attenTensor);
      aclDestroyTensor(outTensor);
      aclDestroyIntArray(actualSeqLengths);
      aclrtFree(queryDeviceAddr);
      aclrtFree(keyDeviceAddr);
      aclrtFree(valueDeviceAddr);
      aclrtFree(attenDeviceAddr);
      aclrtFree(outDeviceAddr);
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
```