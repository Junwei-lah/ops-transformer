# AlltoAllvGroupedMatMul

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **算子功能**：完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合，**先通信后计算**。

- **计算公式**：
    - 路由专家：

    $$
    ataOut = AlltoAllv(gmmX) \\
    permuteOut = Permute(ataOut) \\
    gmmY = permuteOut \times gmmWeight
    $$

    - 共享专家：

    $$
    mmY = mmX \times mmWeight
    $$

## 函数原型

每个算子分为两段式接口，必须先调用`aclnnAlltoAllvGroupedMatMulGetWorkspaceSize`接口获取入参并根据计算流程计算所需workspace大小，再调用`aclnnAlltoAllvGroupedMatMul`接口执行计算。

```cpp
aclnnStatus aclnnAlltoAllvGroupedMatMulGetWorkspaceSize(
    const aclTensor*   gmmX,
    const aclTensor*   gmmWeight,
    const aclTensor*   sendCountsTensorOptional,
    const aclTensor*   recvCountsTensorOptional,
    const aclTensor*   mmXOptional,
    const aclTensor*   mmWeightOptional,
    const char*        group,
    int64_t            epWorldSize,
    const aclIntArray* sendCounts,
    const aclIntArray* recvCounts,
    bool               transGmmWeight,
    bool               transMmWeight,
    bool               permuteOutFlag,
    aclTensor*         gmmY,
    aclTensor*         mmYOptional,
    aclTensor*         permuteOutOptional,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnAlltoAllvGroupedMatMul(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnAlltoAllvGroupedMatMulGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1010px"><colgroup>
    <col style="width: 185px">
    <col style="width: 111px">
    <col style="width: 429px">
    <col style="width: 160px">
    <col style="width: 125px">
    </colgroup>
    <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>gmmX</td>
    <td>输入</td>
    <td>该输入进行AlltoAllv通信与Permute操作后结果作为GroupedMatMul计算的左矩阵，支持2维，shape为(BSK, H1)。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>gmmWeight</td>
    <td>输入</td>
    <td>GroupedMatMul计算的右矩阵，数据类型与gmmX保持一致，支持3维，shape为(e, H1, N1)。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sendCountsTensorOptional</td>
    <td>输入</td>
    <td>可选输入，shape为(e * epWorldSize,)，当前版本暂不支持，传nullptr。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>recvCountsTensorOptional</td>
    <td>输入</td>
    <td>可选输入，shape为(e * epWorldSize,)，当前版本暂不支持，传nullptr。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mmXOptional</td>
    <td>输入</td>
    <td>可选输入，共享专家MatMul计算中的左矩阵，需与mmWeightOptional同时传入或同为nullptr，数据类型与gmmX保持一致，支持2维，shape为(BS, H2)。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mmWeightOptional</td>
    <td>输入</td>
    <td>可选输入，共享专家MatMul计算中的右矩阵，需与mmXOptional同时传入或同为nullptr，数据类型与gmmX保持一致，支持2维，shape为(H2, N2)。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>group</td>
    <td>输入</td>
    <td>专家并行的通信域名，字符串长度要求(0, 128)。</td>
    <td>STRING</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>输入</td>
    <td>ep通信域size：<br><term>Atlas A3系列产品</term>支持8、16、32、64、128；<br><term>Ascend 950PR/Ascend 950DT</term>支持2、4、8、16、32、64。</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sendCounts</td>
    <td>输入</td>
    <td>表示发送给其他卡的token数，数据类型支持INT64，list大小为e * epWorldSize，最大为256。输入类型需为list。</td>
    <td>aclIntArray*（元素类型INT64）</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>recvCounts</td>
    <td>输入</td>
    <td>表示接收其他卡的token数，数据类型支持INT64，list大小为e * epWorldSize，最大为256。输入类型需为list。</td>
    <td>aclIntArray*（元素类型INT64）</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>transGmmWeight</td>
    <td>输入</td>
    <td>GroupedMatMul的右矩阵是否需要转置，true表示需要转置，false表示不转置。</td>
    <td>BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>transMmWeight</td>
    <td>输入</td>
    <td>共享专家MatMul的右矩阵是否需要转置，true表示需要转置，false表示不转置。</td>
    <td>BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>permuteOutFlag</td>
    <td>输入</td>
    <td>permuteOutOptional是否需要输出，true表明需要输出，false表明不需要输出。</td>
    <td>BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>gmmY</td>
    <td>输出</td>
    <td>最终的计算结果，数据类型与输入gmmX保持一致，支持2维，shape为(A, N1)。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mmYOptional</td>
    <td>输出</td>
    <td>共享专家MatMul的输出，数据类型与mmXOptional保持一致，支持2维，shape为(BS, N2)，仅当传入mmXOptional与mmWeightOptional才输出。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>permuteOutOptional</td>
    <td>输出</td>
    <td>permute之后的输出，数据类型与gmmX保持一致，仅当permuteOutFlag为true时输出。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td>UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输出</td>
    <td>返回op执行器，包含了算子的计算流程。</td>
    <td>aclOpExecutor*</td>
    <td>ND</td>
    </tr>
    </tbody></table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

    第一阶段接口完成入参校验，出现以下场景报错：

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 282px">
    <col style="width: 120px">
    <col style="width: 747px">
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

## aclnnAlltoAllvGroupedMatMul

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
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
    <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnAlltoAllvGroupedMatMulGetWorkspaceSize</code>获取。</td>
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

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnAlltoAllvGroupedMatMul默认确定性实现。

- 参数说明里shape使用的变量：
  - BSK：本卡发送的token数，是sendCounts参数累加之和，取值范围(0, 52428800)。
  - H1：表示路由专家hidden size隐藏层大小，取值范围(0, 65536)。
  - H2：表示共享专家hidden size隐藏层大小，取值范围(0, 12288]。
  - e：表示单卡上专家个数，e<=32，e * epWorldSize最大支持256。
  - N1：表示路由专家的head_num，取值范围(0, 65536)。
  - N2：表示共享专家的head_num，取值范围(0, 65536)。
  - BS：batch sequence size。
  - K：表示选取TopK个专家，K的范围[2, 8]。
  - A：本卡收到的token数，是recvCounts参数累加之和。
  - ep通信域内所有卡的 A 参数的累加和等于所有卡上的 BSK 参数的累加和。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>: 单卡通信量在2MB以下可能存在性能劣化。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>:[test_aclnn_allto_allv_grouped_mat_mul.cpp](./examples/arch32/test_aclnn_allto_allv_grouped_mat_mul.cpp)；<term>Ascend 950PR/Ascend 950DT</term>: [test_aclnn_allto_allv_grouped_mat_mul.cpp](./examples/arch32/test_aclnn_allto_allv_grouped_mat_mul.cpp) | 通过[aclnnAlltoAllvGroupedMatMul](./docs/aclnnAlltoAllvGroupedMatMul.md)接口方式调用allto_allv_grouped_mat_mul算子。 |
