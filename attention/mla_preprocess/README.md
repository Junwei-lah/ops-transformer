# MlaPreprocess
## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
## 功能说明
-  **算子功能**：推理场景，Multi-Head Latent Attention前处理的计算。主要计算过程如下：
    -  首先对输入$x$ RmsNormQuant后乘以$W^{DQKV}$进行下采样后分为通路1和通路2。
    -  通路1做RmsNormQuant后乘以$W^{UQ}$后再分为通路3和通路4。
    -  通路3后乘以$W^{uk}$后输出$q^N$。
    -  通路4后经过旋转位置编码后输出$q^R$。
    -  通路2拆分为通路5和通路6。
    -  通路5经过RmsNorm后传入Cache中得到$k^N$。
    -  通路6经过旋转位置编码后传入另一个Cache中得到$k^R$。
-  **计算公式**：

    RmsNormQuant公式

    $$
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    $$
    \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
    $$

    $$
    RmsNormQuant(x) = ({RmsNorm}(x) + bias) * deqScale
    $$
  
    Query计算公式，包括W^{DQKV}矩阵乘、W^{UK}矩阵乘、RmsNormQuant和ROPE旋转位置编码处理

    $$
    q^N =  RmsNormQuant(x) \cdot W^{DQKV} \cdot W^{UK}
    $$

    $$
    q^R = ROPE(x^Q)
    $$

    Key计算公式，包括RmsNorm和rope，将计算结果存入cache

    $$
    k^N = Cache({RmsNorm}(RmsNormQuant(x)))
    $$

    $$
    k^R = Cache(ROPE(RmsNormQuant(x)))
    $$

## 参数说明
| 参数名                     | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|--------------------------------------------------------------------|----------------|------------|
| input                     | 输入      | Device侧的aclTensor，用于计算Query和Key的x，shape为[tokenNum,hiddenSize] | FLOAT16, BFLOAT16 | ND         |
| gamma0                   | 输入      | Device侧的aclTensor，首次RmsNorm计算中的γ参数，shape为[hiddenSize] | FLOAT16, BFLOAT16 | ND |
| beta0                 | 输入      | Device侧的aclTensor，首次RmsNorm计算中的β参数，shape为[hiddenSize]| FLOAT16, BFLOAT16 | ND |
| quantScale0                   | 输入      | Device侧的aclTensor，首次RmsNorm公式中量化缩放的参数，shape为[1] | FLOAT16, BFLOAT16       | ND         |
| quantOffset0                | 输入      | Device侧的aclTensor，首次RmsNorm公式中的量化偏移参数，shape为[1] | INT8    | NZ |
| wdqkv             | 输入      | Device侧的aclTensor，与输入首次做矩阵乘的降维矩阵，shape为[2112,hiddenSize] | INT8, BFLOAT16      | ND         |
| deScale0            | 输入      | Device侧的aclTensor，输入首次做矩阵乘的降维矩阵中的系数，shape为[2112]。input输入dtype为FLOAT16支持INT64，输入BFLOAT16时支持FLOAT | INT64, FLOAT    | ND         |
| bias0                    | 输入      | Device侧的aclTensor，输入首次做矩阵乘的降维矩阵中的系数，shape为[2112]。支持传入空tensor，quantMode为1、3时不传入 | INT32     | ND         |
| gamma1                    | 输入      | Device侧的aclTensor，第二次RmsNorm计算中的γ参数，shape为[1536] | FLOAT16, BFLOAT16       | ND         |
| beta1                 | 输入      | Device侧的aclTensor，第二次RmsNorm计算中的β参数，shape为[1536] | FLOAT16, BFLOAT16          | ND         |
| quantScale1                 | 输入  | Device侧的aclTensor，第二次RmsNorm公式中量化缩放的参数，shape为[1536]。仅在quantMode为0时传入| FLOAT16，BFLOAT16 | ND         |
| quantOffset1                 | 输入 | Device侧的aclTensor，第二次RmsNorm公式中的量化偏移参数，shape为[1]。仅在quantMode为0时传入 | INT8 | ND         |
| wuq      | 输入      | Device侧的aclTensor，权重矩阵，shape为[headNum * 192,1536]  | INT8, BFLOAT16          | NZ         |
| deScale1    | 输入      | Device侧的aclTensor，参与wuq矩阵乘的系数，shape为[headNum*192,1536]。input输入dtype为FLOAT16支持INT64，输入BFLOAT16时支持FLOAT | INT64, FLOAT          | ND         |
| bias1  | 输入      | Device侧的aclTensor，参与wuq矩阵乘的系数，shape为[[headNum*192]]。quantMode为1、3时不传入 | INT32          | NZ         |
| gamma2 | 输入      | Device侧的aclTensor，参与RmsNormAndreshapeAndCache计算的γ参数，shape为[512]。 | FLOAT16, BFLOAT16          | ND         |
| cos      | 输入      | Device侧的aclTensor，表示用于计算旋转位置编码的正弦参数矩阵，shape为[tokenNum,64] | INT8          | NZ         |
| sin      | 输入      | Device侧的aclTensor，表示用于计算旋转位置编码的余弦参数矩阵，shape为[tokenNum,64] | INT8          | NZ         |
| wuk     | 输入      | Device侧的aclTensor，表示计算Key的上采样权重，shape为[headNum * 192, 1536]。ND格式时的shape为[headNum,128,512]，NZ格式时的shape为[headNum,32,128,16] | FLOAT16，BFLOAT16          | ND, NZ         |
| kvCache           | 输入      | Device侧的aclTensor，与输出的kvCacheOut为同一tensor，输入格式随cacheMode变化。</br><li>cacheMode为0：shape为[blockNum,blockSize,1,576]<li>cacheMode为1：shape为[blockNum,blockSize,1,512]<li> cacheMode为2：shape为[blockNum,headNum\*512/32,block_size,32]<li> cacheMode为3：shape为[blockNum,headNum*512/16,block_size,16]| <li>与input一致<li>与input一致</li>INT8<li>与input一致         | <li>ND<li>ND<li>NZ<li>NZ         |
| kvCacheRope          | 输入      | Device侧的aclTensor，可选参数，支出传入空指针。与输出的krCacheOut为同一tensor，输入格式随cacheMode变化。</br> <li>cacheMode为0：不传入。<li>cacheMode为1：shape为[blockNum,blockSize,1,64]<li>cacheMode为2或3：shape为[blockNum, headNum*64 / 16 ,block_size, 16]| 与input一致         |  <li><li>ND<li>NZ          |
| slotmapping          | 输入      | Device侧的aclTensor，表示用于存储kv_cache和kr_cache的索引，shape为[tokenNum] | INT32          | ND          |
| ctkvScale                  | 输入      | Device侧的aclTensor，输出量化处理中参与计算的系数，仅在cacheMode为2时传入，shape为[1] | FLOAT16, BFLOAT16 | ND         |
| qNopeScale              | 输出      | Device侧的aclTensor，输出量化处理中参与计算的系数，仅在cacheMode为2时传入，shape为[headNum] | FLOAT16, BFLOAT16       | ND |
| wdqDim      | 输入      | 表示经过matmul后拆分的dim大小。预留参数，目前只支持1536 | int64_t          | -         |
| qRopeDim      | 输入      | 表示q传入rope的dim大小。预留参数，目前只支持64。 | int64_t          | -         |
| kRopeDim     | 输入      | 表示k传入rope的dim大小。预留参数，目前只支持64。 | int64_t          | -         |
| epsilon         | 输入      | 表示加在分母上防止除0 | float          | -          |
| qRotaryCoeff                  | 输入      | 表示q旋转系数。预留参数，目前只支持2 | int64_t | -         |
| kRotaryCoeff              | 输入      | 表示k旋转系数。预留参数，目前只支持2 | int64_t       | - |
| transposeWdq      | 输入      | 表示wdq是否转置。预留参数，目前只支持true | bool          | -         |
| transposeWuq      | 输入      | 表示wuq是否转置。预留参数，目前只支持true | bool          | -         |
| transposeWuk     | 输入      | 表示wuk是否转置。预留参数，目前只支持true | bool          | -         |
| cacheMode           | 输入      | 表示指定cache的类型，取值范围[0, 3]</br><li>0：kcache和q均经过拼接后输出<li>1：输出的kvCacheOut拆分为kvCacheOut和krCacheOut，qOut拆分为qOut和qRopeOut<li>2：krope和ctkv转为NZ格式输出，ctkv和qnope经过per_head静态对称量化为int8类型<li>3：krope和ctkv转为NZ格式输出 | int64_t         | -          |
| quantMode         | 输入      | 表示指定RmsNorm量化的类型，取值范围[0, 3]</br>0：per_tensor静态非对称量化，默认量化类型</br>1：per_token动态对称量化，未实现</br>2：per_token动态非对称量化，未实现</br>3：不量化，浮点输出，未实现 | int64_t         | -          |
| doRmsNorm        | 输入      | 表示是否对input输入进行RmsNormQuant操作，false表示不操作，true表示进行操作。预留参数，目前只支持true | bool          | -          |
| wdkvSplitCount                  | 输入      | 表示指定wdkv拆分的个数，支持[1-3]，分别表示不拆分、拆分为2个、拆分为3个降维矩阵。预留参数，目前只支持1 | int64_t | -         |
| qOut             | 输出      | 表示Query的输出tensor，对应计算流图中右侧经过NOPE和矩阵乘后的输出，shape和dtype随cacheMode变化</br><li>cacheMode为0：shape为[tokenNum, headNum, 576]<li>cacheMode为1或3：shape为[tokenNum, headNum, 512]<li>cacheMode为2：shape为[tokenNum, headNum, 512] | <li>与input一致<li>与input一致<li>INT8       | ND |
| kvCacheOut             | 输出      | 表示Key经过ReshapeAndCache后的输出，shape和dtype随cacheMode变化</br><li>cacheMode为0：shape为[blockNum, blockSize, 1, 576] <li> cacheMode为1：shape为[blockNum, blockSize, 1, 512]<li>cacheMode为2：shape为[blockNum, headNum\*512/32, block_size, 32] <li>cacheMode为3：shape为[blockNum, headNum*512/16, block_size, 16] | <li>与input一致<li>与input一致<li>INT8<li>与input一致       | <li>ND<li>ND<li>NZ<li>NZ |
| qRopeOut             | 输出      | 表示Query经过旋转编程后的输出，shape和dtype随cacheMode变化</br><li>cacheMode为0：不输出<li> cacheMode为1或3：shape为[tokenNum, headNum, 64]<li>cacheMode为2：shape为[tokenNum, headNum, 64] | <li><li>与input一致<li>与input一致       | <li><li>ND<li>ND |
| krCacheOut             | 输出      | 表示Key经过ROPE和ReshapeAndCache后的输出，shape和dtype随cacheMode变化，</br><li>cacheMode为0：不输出<li> cacheMode为1：shape为[blockNum, blockSize, 1, 64]<li>cacheMode为2或3：shape为[blockNum, headNum*64 / 16 ,block_size, 16] | <li><li>与input一致<li>与input一致       | <li><li>ND<li>NZ |
## 约束说明
-   shape格式字段含义及约束
    -  tokenNum：tokenNum 表示输入样本批量大小，取值范围：0~256
    -  hiddenSize：hiddenSize 表示隐藏层的大小，取值固定为：2048-10240，为256的倍数
    -  headNum：表示多头数，取值范围：16、32、64、128
    -  blockNum：PagedAttention场景下的块数，取值范围：192
    -  blockSize：PagedAttention场景下的块大小，取值范围：128
    -  当wdqkv和wuq的数据类型为bfloat16时，输入input也需要为bfloat16，且hiddenSize只支持6144