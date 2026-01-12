# aclnnMoeTokenPermuteWithEp

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |


## 功能说明

- **算子功能**：MoE的permute计算，根据索引indices将tokens和可选probs广播后排序并按照rangeOptional中范围切片。
- **计算公式**：
  - paddedMode`false`时

    $$
    sortedIndicesFirst=argSort(indices)
    $$

    $$
    sortedIndicesOut=argSort(sortedIndices)
    $$

    当rangeOptional[0] <= sortedIndices[i] < rangeOptional[1]时

    $$
    permuteTokensOut[sortedIndices[i]-rangeOptional[0]]=tokens[i//topK]
    $$

    $$
    permuteProbsOut[sortedIndices[i]-rangeOptional[0]]=probsOptional[i]
    $$

  - paddedMode为`true`时

    $$
    permuteTokensOut[i]=tokens[indices[i]]
    $$

    $$
    sortedIndicesOut=indices
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 “aclnnMoeTokenPermuteWithEpGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenPermuteWithEp”接口执行计算。

* `aclnnStatus aclnnMoeTokenPermuteWithEpGetWorkspaceSize(const aclTensor *tokens, const aclTensor *indices, const aclTensor *probsOptional, const aclIntArray *rangeOptional, int64_t numOutTokens, bool paddedMode, const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut, const aclTensor *permuteProbsOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeTokenPermuteWithEp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeTokenPermuteWithEpGetWorkspaceSize

- **参数说明：**

  - tokens（aclTensor \*，计算输入）：表示permute中的输入tokens，公式中的`tokens`，Device侧的aclTensor。shape支持2D维度，shape为（num\_tokens，hidden\_size），num\_tokens为tokens的数目，hidden\_size为每个tokens的长度。数据类型支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，不支持空Tensor。
  - indices（aclTensor \*，计算输入）：表示输入tokens对应的专家索引，公式中的`indices`，Device侧的aclTensor。shape支持1D或2D维度。paddedMode为false时表示每一个输入token对应的topK个处理专家索引，shape为（num\_tokens，topK\_num）或（num\_tokens），paddedMode为true时表示每个专家选中的token索引（暂不支持）。要求元素个数小于16777215，值大于等于0小于16777215，topK\_num小于等于512。数据类型支持INT32、INT64，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，不支持空Tensor。
  - probsOptional（aclTensor \*，计算输入）：表示输入tokens对应的专家概率，公式中的`probsOptional`，Device侧的aclTensor。可选计算输入，与计算输出permuteProbsOut对应，传入空则不输出permuteProbsOut。shape与indices的shape相同。数据类型支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  - rangeOptional（aclIntArray \*，计算输入）：ep切分的有效范围，size为2。为空时，忽略probsOptional和permuteTokensOut，执行逻辑回退到[aclnnMoeTokenPermute](aclnnMoeTokenPermute.md)。
  - numOutTokens（int64\_t，计算输入）：有效输出token数，在rangeOptional为空时生效。设置为0时，表示不会删除任何token。不为0时，会按照numOutTokens进行切片丢弃按照indices排序好的token中超过numOutTokens的部分，为负数时按照切片索引为负数时处理。
  - paddedMode（bool，计算输入）：paddedMode为true时表示indices已被填充为代表每个专家选中的token索引，此时不对indices进行排序。目前仅支持paddedMode为false。
  - permuteTokensOut（aclTensor \*，计算输出）：表示根据indices进行扩展并排序过的tokens，公式中的`permuteTokensOut`，Device侧的aclTensor。shape支持2D维度，shape为（rangeOptional[1] - rangeOptional[0]，hidden\_size）。数据类型同tokens，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  - sortedIndicesOut（aclTensor \*，计算输出）：表示permuteTokensOut和tokens的映射关系，公式中的`sortedIndicesOut`，Device侧的aclTensor。shape支持1D维度，Shape为（num\_tokens * topK\_num）。数据类型支持INT32，[数据格式](../../../docs/zh/context/数据格式.md)要求为ND。
  - permuteProbsOut（aclTensor \*，计算输出）：表示根据indices进行扩展并排序过的probs，公式中的`permuteProbsOut`，Device侧的aclTensor。shape支持1D维度，shape为（rangeOptional[1] - rangeOptional[0]）。数据类型同tokens，[数据格式](../../../docs/zh/context/数据格式.md)要求为ND。
  - workspaceSize（uint64\_t \*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor \*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR)：1. 输入和输出的Tensor是空指针。
  161002(ACLNN_ERR_PARAM_INVALID)：1. 输入和输出的数据类型不在支持的范围内。
  ```

## aclnnMoeTokenPermuteWithEp

- **参数说明：**
  - workspace（void \*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenPermuteWithEpGetWorkspaceSize获取。
  - executor（aclOpExecutor \*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的Stream。
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnMoeTokenPermuteWithEp默认确定性实现。

- indices 要求元素个数小于`16777215`，值大于等于`0`小于`16777215`(单点支持int32或int64的最大或最小值，其余不在范围内的排序结果不正确)。
- topK小于等于`512`。
- 不支持paddedMode为`True`。
- 当rangeOptional为空时，忽略probsOptional和permuteTokensOut，执行逻辑回退到[aclnnMoeTokenPermute](aclnnMoeTokenPermute.md)。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute_with_ep.h"
#include <iostream>
#include <vector>

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
int CreateAclIntArray(const std::vector<T>& hostData, void** deviceAddr, aclIntArray** intArray) {
  auto size = GetShapeSize(hostData) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Call aclCreateIntArray to create an aclIntArray.
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
  return 0;
}
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
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
    // 1. 固定写法，device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> idxShape = {3, 2};
    std::vector<int64_t> probsShape = {3, 2};
    std::vector<int64_t> expandedXOutShape = {4, 4};
    std::vector<int64_t> idxOutShape = {6};
    std::vector<int64_t> expandedProbsOutShape = {4};

    void* xDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* probsDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* sortedIndicesOutDeviceAddr = nullptr;
    void* expandedProbsOutDeviceAddr = nullptr;
    void* rangeDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* indices = nullptr;
    aclTensor* probs = nullptr;
    aclIntArray* range = nullptr;
    int64_t numTokenOut = 6;
    bool padMode = false;

    aclTensor* expandedXOut = nullptr;
    aclTensor* sortedIndicesOut = nullptr;
    aclTensor* expandedProbsOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> indicesHostData = {1, 2, 3, 1, 2, 3};
    std::vector<float> probsHostData = {0.5, 0.3, 0.4, 0.2, 0.5, 0.4};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> sortedIndicesOutHostData = {0, 0, 0, 0, 0, 0};
    std::vector<float> expandedProbsOutHostData = {0, 0, 0, 0};
    std::vector<int64_t> rangeHostData = {1, 5};
    // 创建self aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, idxShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr, aclDataType::ACL_BF16, &probs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_BF16, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesOutHostData, idxOutShape, &sortedIndicesOutDeviceAddr, aclDataType::ACL_INT32, &sortedIndicesOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedProbsOutHostData, expandedProbsOutShape, &expandedProbsOutDeviceAddr, aclDataType::ACL_BF16, &expandedProbsOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建相关attr
    ret = CreateAclIntArray(rangeHostData, &rangeDeviceAddr, &range);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMoeTokenPermuteWithEp第一段接口
    ret = aclnnMoeTokenPermuteWithEpGetWorkspaceSize(x, indices, probs, range, numTokenOut, padMode, expandedXOut, sortedIndicesOut, expandedProbsOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithEpGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMoeTokenPermuteWithEp第二段接口
    ret = aclnnMoeTokenPermuteWithEp(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithEp failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<float> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(), expandedXData.size() * sizeof(expandedXData[0]), expandedXOutDeviceAddr, expandedXSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %f\n", i, expandedXData[i]);
    }
    auto sortedIndicesSize = GetShapeSize(idxOutShape);
    std::vector<int> sortedIndicesData(sortedIndicesSize, 0);
    ret = aclrtMemcpy(sortedIndicesData.data(), sortedIndicesData.size() * sizeof(sortedIndicesData[0]), sortedIndicesOutDeviceAddr, sortedIndicesSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < sortedIndicesSize; i++) {
        LOG_PRINT("sortedIndicesData[%ld] is: %d\n", i, sortedIndicesData[i]);
    }
    auto expandedProbsSize = GetShapeSize(expandedProbsOutShape);
    std::vector<float> expandedProbsData(expandedProbsSize, 0);
    ret = aclrtMemcpy(expandedProbsData.data(), expandedProbsData.size() * sizeof(expandedProbsData[0]), expandedProbsOutDeviceAddr, expandedProbsSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedProbsSize; i++) {
        LOG_PRINT("expandedProbsData[%ld] is: %f\n", i, expandedProbsData[i]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(indices);
    aclDestroyTensor(probs);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(sortedIndicesOut);
    aclDestroyTensor(expandedProbsOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(probsDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(sortedIndicesOutDeviceAddr);
    aclrtFree(expandedProbsOutDeviceAddr);
    aclrtFree(rangeDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

