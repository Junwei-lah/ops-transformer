# aclnnAlltoAllvQuantGroupedMatMul

## 产品支持情况

| 产品                                        | 是否支持 |
| :------------------------------------------ | :------: |
| Ascend 950PR/Ascend 950DT系列               |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    ×     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    ×     |
| Atlas 200I/500 A2 推理产品                  |    ×     |
| Atlas 推理系列产品                          |    ×     |
| Atlas 训练系列产品                          |    ×     |

## 功能说明

- **算子功能**：完成路由专家AlltoAllv、量化GroupedMatMul融合并实现与共享专家量化MatMul并行融合，**先通信后计算**。

- **计算公式**：
假设通信域中的总卡数为epWorldSize，每张卡上通信后路由专家个数为e，每张卡分组矩阵乘只负责本卡专家的计算。对于每张卡的计算公式如下：
  - 本卡共享专家分组矩阵乘计算

    ```
    mm_y=(mm_x × mm_x_scale) @ (mm_weight × mm_weight_scale)
    ```

  - Alltoallv通信和permute

    ```
    permute_out=Alltoallv(gmm_x)
    ```

  - 本卡路由专家按专家维度分组矩阵乘计算

    ```
    gmm_y=(permute_out × gmm_x_scale) @ (gmm_weight × gmm_weight_scale)
    ```

## 函数原型

每个算子分为两段式接口，必须先调用`aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize`接口获取入参并根据计算流程计算所需workspace大小，再调用`aclnnAlltoAllvQuantGroupedMatMul`接口执行计算。

```cpp
aclnnStatus aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor*   gmmX,
    const aclTensor*   gmmWeight,
    const aclTensor*   gmmXScale,
    const aclTensor*   gmmWeightScale,
    const aclTensor*   sendCountsTensorOptional,
    const aclTensor*   recvCountsTensorOptional,
    const aclTensor*   mmXOptional,
    const aclTensor*   mmWeightOptional,
    const aclTensor*   mmXScaleOptional,
    const aclTensor*   mmWeightScaleOptional,
    int64_t            gmmXQuantMode,
    int64_t            gmmWeightQuantMode,
    int64_t            mmXQuantMode,
    int64_t            mmWeightQuantMode,
    const char*        group,
    int64_t            epWorldSize,
    const aclIntArray* sendCounts,
    const aclIntArray* recvCounts,
    bool               transGmmWeight,
    bool               transMmWeight,
    int64_t            groupSize,
    bool               permuteOutFlag,
    const aclTensor*   gmmY,
    const aclTensor*   mmYOptional,
    const aclTensor*   permuteOutOptional,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnAlltoAllvQuantGroupedMatMul(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
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
        <td>gmmX</td>
        <td>输入</td>
        <td>该输入进行AlltoAllv通信后结果作为GroupedMatMul计算的左矩阵。</td>
        <td>当前版本仅支持2维输入，shape为(BSK, H1)，且仅支持不转置场景。</td>
        <td>HIFLOAT8</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>gmmWeight</td>
        <td>输入</td>
        <td>GroupedMatMul计算的右矩阵。</td>
        <td>当前版本仅支持3维输入，shape为(e, H1, N1)，支持转置/不转置场景。</td>
        <td>HIFLOAT8</td>
        <td>ND</td>
        <td>3</td>
        <td>√（仅适用转置场景）</td>
    </tr>
    <tr>
        <td>gmmXScale</td>
        <td>输入</td>
        <td>gmmX的量化系数。</td>
        <td>pertensor量化场景支持一维，shape为(1)。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>x</td>
    </tr>
    <tr>
        <td>gmmWeightScale</td>
        <td>输入</td>
        <td>gmmWeight的量化系数。</td>
        <td>pertensor量化场景支持一维，shape为(1)。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√（仅适用转置场景）</td>
    </tr>
    <tr>
        <td>sendCountsTensorOptional</td>
        <td>输入</td>
        <td>预留参数，当前版本仅支持传nullptr。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>recvCountsTensorOptional</td>
        <td>输入</td>
        <td>预留参数，当前版本仅支持传nullptr。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmXOptional</td>
        <td>输入</td>
        <td>可选输入，共享专家MatMul计算中的左矩阵。</td>
        <td><ul><li>支持2维，shape为(BS, H2)。</li><li>需与mmWeightOptional同时传入或同为nullptr。</li></ul></td>
        <td>与gmmX保持一致</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>mmWeightOptional</td>
        <td>输入</td>
        <td>可选输入，共享专家MatMul计算中的右矩阵。</td>
        <td><ul><li>支持2维，shape为(H2, N2)。</li><li>需与mmXOptional同时传入或同为nullptr。</li></ul></td>
        <td>与gmmWeight保持一致</td>
        <td>ND</td>
        <td>2</td>
        <td>√（仅适用转置场景）</td>
    </tr>
    <tr>
        <td>mmXScaleOptional</td>
        <td>输入</td>
        <td>可选输入，mmX的量化系数。</td>
        <td><ul><li>pertensor量化场景支持1维，shape为(1)。</li><li>需与mmXOptional同时传入或同为nullptr。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>x</td>
    </tr>
    <tr>
        <td>mmWeightScaleOptional</td>
        <td>输入</td>
        <td>可选输入，mmWeight的量化系数。</td>
        <td><ul><li>pertensor量化场景支持1维，shape为(1)。</li><li>需与mmXOptional同时传入或同为nullptr。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√（仅适用转置场景）</td>
    </tr>
    <tr>
        <td>gmmXQuantMode</td>
        <td>输入</td>
        <td>gmmX的量化模式。</td>
        <td>当前版本仅支持1(pertensor场景)。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gmmWeightQuantMode</td>
        <td>输入</td>
        <td>gmmWeight的量化模式。</td>
        <td>当前版本仅支持1(pertensor场景)。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmXQuantMode</td>
        <td>输入</td>
        <td>mmX的量化模式。</td>
        <td>当前版本仅支持1(pertensor场景)。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmWeightQuantMode</td>
        <td>输入</td>
        <td>mmWeight的量化模式。</td>
        <td>当前版本仅支持1(pertensor场景)。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group</td>
        <td>输入</td>
        <td>专家并行的通信域名，字符串长度要求(0, 128)。</td>
        <td>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epWorldSize</td>
        <td>输入</td>
        <td>ep通信域大小。</td>
        <td><term>Ascend 950PR/Ascend 950DT</term>支持2、4、8、16、32、64、128、256。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sendCounts</td>
        <td>输入</td>
        <td>表示发送给其他卡的token数。</td>
        <td>数据类型支持INT64，长度为e * epWorldSize，最大为256。输入类型需为list。</td>
        <td>aclIntArray*（元素类型INT64）</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>recvCounts</td>
        <td>输入</td>
        <td>表示接收其他卡的token数。</td>
        <td>数据类型支持INT64，长度为e * epWorldSize，最大为256。输入类型需为list。</td>
        <td>aclIntArray*（元素类型INT64）</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transGmmWeight</td>
        <td>输入</td>
        <td>GroupedMatMul的右矩阵是否需要转置。</td>
        <td>true表示需要转置，false表示不转置。</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transMmWeight</td>
        <td>输入</td>
        <td>共享专家MatMul的右矩阵是否需要转置。</td>
        <td>true表示需要转置，false表示不转置。</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>groupSize</td>
        <td>输入</td>
        <td>用于表示量化中gmmXScale/gmmWeightScale/mmXScale/mmWeightScale输入的一个数在其所在的对应维度方向上可以用于该方向gmmX/gmmWeight/mmX/mmWeight输入的多少个数的量化。</td>
        <td>groupSize输入由3个方向的groupSizeM、groupSizeN、groupSizeK三个值拼接组成，每个值占16位，计算公式为groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>permuteOutFlag</td>
        <td>输入</td>
        <td>permuteOutOptional是否需要输出。</td>
        <td>true表明需要输出，false表明不需要输出。</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gmmY</td>
        <td>输出</td>
        <td>路由专家计算的输出。</td>
        <td>支持2维，shape为(A, N1)。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>mmYOptional</td>
        <td>输出</td>
        <td>共享专家计算的输出。</td>
        <td>支持2维，shape为(BS, N2)，仅当传入mmXOptional与mmWeightOptional才输出。</td>
        <td>与gmmY保持一致</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>permuteOutOptional</td>
        <td>输出</td>
        <td>permute之后的输出。</td>
        <td>支持2维，shape为(A, H1)，仅当permuteOutFlag为true时输出。</td>
        <td>与gmmX保持一致</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>UINT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子的计算流程。</td>
        <td>-</td>
        <td>aclOpExecutor*</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

- gmmXQuantMode、gmmWeightQuantMode、mmXQuantMode、mmWeightQuantMode的枚举值跟量化模式关系如下:
  - 0: 非量化
  - 1: pertensor
  - 2: perchannel
  - 3: pertoken
  - 4: pergroup
  - 5: perblock
  - 6: mx量化
  - 7: pertoken动态量化

- **返回值**
    
    返回aclnnStatus状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn返回码.md)。

    第一阶段接口完成入参校验，出现以下场景报错：

    <table style="undefined;table-layout: fixed; width: 1180px"> <colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 800px">
    </colgroup>
    <thead>
    <tr>
    <th>返回值</th>
    <th>错误码</th>
    <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>传入参数要求是必选输入、输出或者必选属性，但实际传入了空指针。</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>gmmX、gmmWeight、sendCountsTensorOptional、recvCountsTensorOptional、mmXOptional、mmWeightOptional、group、epWorldSize、sendCounts、recvCounts的数据类型、数据格式或者维度不在支持的范围内。</td>
    </tr>
    </tbody></table>

## aclnnAlltoAllvQuantGroupedMatMul

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1180px"> <colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 800px">
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
    <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize</code>获取。</td>
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
    </tbody></table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - `aclnnAlltoAllvQuantGroupedMatMul`默认确定性实现。

- 参数说明里shape使用的变量：
  - BSK：本卡发送的token数，是sendCounts参数累加之和，取值范围(0, 52428800)。
  - H1：表示路由专家hidden size隐藏层大小，取值范围(0, 65536)。
  - H2：表示共享专家hidden size隐藏层大小，取值范围(0, 12288]。
  - e：表示单卡上专家个数，取值范围(0, 32]，e * epWorldSize最大支持256。
  - N1：表示路由专家的head_num，取值范围(0, 65536)。
  - N2：表示共享专家的head_num，取值范围(0, 65536)。
  - BS：batch sequence size。
  - K：表示选取TopK个专家，K的范围[2, 8]。
  - A：本卡收到的token数，是recvCounts参数累加之和。
  - ep通信域内所有卡的 A 参数的累加和等于所有卡上的 BSK 参数的累加和。
  - groupSize: 
    - 仅当gmmXScale/gmmWeightScale/mmXScale/mmWeightScale输入都是2维及以上数据时，groupSize取值有效，其他场景需传入0。
    - groupSize值支持公式推导：传入的groupSize内部会按如下公式分解得到groupSizeM、groupSizeN、groupSizeK，当其中有1个或多个为0，会根据gmmX/gmmWeight/mmX/mmWeight/gmmXScale/gmmWeightScale/mmXScale/mmWeightScale输入shape重新设置groupSizeM、groupSizeN、groupSizeK用于计算。设置原理：如果groupSizeM=0，表示m方向量化分组值由接口推导，推导公式为groupSizeM = m / scaleM（需保证m能被scaleM整除），其中m与gmmX/mmX shape中的m一致，scaleM与gmmXScale/mmXScale shape中的m一致；如果groupSizeK=0，表示k方向量化分组值由接口推导，推导公式为groupSizeK = k / scaleK（需保证k能被scaleK整除），其中k与gmmX/mmX shape中的k一致，scaleK与gmmXScale/mmXScale shape中的k一致；如果groupSizeN=0，表示n方向量化分组值由接口推导，推导公式为groupSizeN = n / scaleN（需保证n能被scaleN整除），其中n与gmmWeight/mmWeight shape中的n一致，scaleN与gmmWeightScale/mmWeightScale shape中的n一致。
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$
    - 如果满足重新设置条件，当gmmXScale/gmmWeightScale/mmXScale/mmWeightScale输入都是3维，且数据类型都为FLOAT8_E8M0时，[groupSizeM，groupSizeN，groupSizeK]取值组合会推导为[1, 1, 32]，对应groupSize的值为4295032864。

- 量化参数约束：
  - 当前版本仅支持pertensor量化。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_allto_allv_quant_grouped_mat_mul.cpp](./examples/test_aclnn_allto_allv_quant_grouped_mat_mul.cpp) | 通过[aclnnAlltoAllvQuantGroupedMatMul](./docs/aclnnAlltoAllvQuantGroupedMatMul.md)接口方式调用AlltoAllvQuantGroupedMatMul算子。 |
