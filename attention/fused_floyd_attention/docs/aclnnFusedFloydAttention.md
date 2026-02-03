
# aclnnFusedFloydAttention

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|     √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|     √     |
|<term>Atlas 200I/500 A2 推理产品</term>|     ×      |
|<term>Atlas  推理系列产品</term>|     ×     |
|<term>Atlas  训练系列产品</term>|     ×     |

## 功能说明

- 接口功能：训练场景下，使用FloydAttention算法实现多维自注意力的计算。

- 计算公式：

    注意力的正向计算公式如下：

    $$
    weights = Softmax(attenMask + scale*(einsum(query, key1^T) + einsum(query, key2^T)))
    $$
    
    $$
    attention\_out = einsum(weights, value1) + einsum(weights, value2)
    $$
    

## 函数原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnFusedFloydAttentionGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFusedFloydAttention”接口执行计算。

```Cpp
aclnnStatus aclnnFusedFloydAttentionGetWorkspaceSize(
    const aclTensor *query, 
    const aclTensor *key1, 
    const aclTensor *value1, 
    const aclTensor *key2, 
    const aclTensor *value2, 
    const aclTensor *attenMaskOptional, 
    double           scaleValueOptional, 
    const aclTensor *softmaxMaxOut, 
    const aclTensor *softmaxSumOut, 
    const aclTensor *attentionOutOut, 
    uint64_t        *workspaceSize, 
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFusedFloydAttention(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    const aclrtStream stream)
```


## aclnnFusedFloydAttentionGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1565px"><colgroup>
      <col style="width: 146px">
      <col style="width: 135px">
      <col style="width: 326px">
      <col style="width: 246px">
      <col style="width: 275px">
      <col style="width: 101px">
      <col style="width: 190px">
      <col style="width: 146px">
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
          <td>Device侧的aclTensor，公式中的query。</td>
          <td>数据类型与key1/value1/key2/value2的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>key1</td>
          <td>输入</td>
          <td>Device侧的aclTensor，公式中的key1。</td>
          <td>数据类型与query/value1/key2/value2的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>value1</td>
          <td>输入</td>
          <td>Device侧的aclTensor，公式中的value1。</td>
          <td>数据类型与query/key1/key2/value2的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>key2</td>
          <td>输入</td>
          <td>Device侧的aclTensor，公式中的key2。</td>
          <td>数据类型与query/key1/value1/value2的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>value2</td>
          <td>输入</td>
          <td>Device侧的aclTensor，公式中的value2。</td>
          <td>数据类型与query/key1/value1/key2的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>attenMaskOptional</td>
          <td>输入</td>
          <td>Device侧的aclTensor，公式中的attenMask。</td>
          <td>取值为1代表该位不参与计算，为0代表该位参与计算。</td>
          <td>BOOL、UINT8</td>
          <td>ND</td>
          <td>[B,1,N,1,K]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>scaleValue</td>
          <td>输入</td>
          <td>Host侧的double，公式中的scale，代表缩放系数。</td>
          <td>-</td>
          <td>DOUBLE</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>softmaxMax</td>
          <td>输出</td>
          <td>Device侧的aclTensor，注意力正向计算的中间输出。</td>
          <td>输出的shape类型为[B,H,N,M,8]。</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>[B,H,N,M,8]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>softmaxSum</td>
          <td>输出</td>
          <td>Device侧的aclTensor，注意力正向计算的中间输出。</td>
          <td>输出的shape类型为[B,H,N,M,8]。</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>[B,H,N,M,8]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>attentionOutOut</td>
          <td>输出</td>
          <td>Device侧的aclTensor，计算公式的最终输出。</td>
          <td>数据类型与query的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>输出</td>
          <td>返回需要在Device侧申请的workspace大小。</td>
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
      </tbody>
    </table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  第一段接口完成入参校验，若出现以下错误码，则对应原因为：

    <table style="undefined;table-layout: fixed; width: 1146px"><colgroup>
    <col style="width: 283px">
    <col style="width: 120px">
    <col style="width: 743px">
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
        <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
    </tr>
    <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>query、key1、value1、key2、value2、attenMaskOptional、softmaxMaxOut、softmaxSumOut、attentionOutOut的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    </tbody>
    </table>

## aclnnFusedFloydAttention

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnFlashAttentionScoreGetWorkspaceSize获取。</td>
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

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明<a name="1"></a>

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配
- 关于数据shape的约束，其中：
  - B：取值范围为1\~2K。
  - H：取值范围为1\~256。
  - N：取值范围为16\~1M且N%16==0。
  - M：取值范围为128\~1M且M%128==0。
  - K：取值范围为128\~1M且K%128==0。
  - D：取值范围为16\~128。

- query与key1的第0/2/4轴需相同。
- key1与value1 shape需相同。
- key2与value2 shape需相同。
- softmaxMax与softmaxSum shape需相同。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
  
```c++
#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include "acl/acl.h"
#include "aclnn_fused_floyd_attention.h"

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

template <typename T> void SaveOutResult(std::string &fileName, std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<T> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                            size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    std::ofstream file(fileName, std::ios::binary);
    // 保存文件
    file.write(static_cast<char *>((void *)resultData.data()), size * sizeof(T));
    file.close();
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
                                    aclFinalize(); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret);
                                    aclrtDestroyContext(context); aclrtResetDevice(deviceId); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
                                    aclrtDestroyContext(context); aclrtResetDevice(deviceId); aclFinalize(); return ret);
    return 0;
}

int ReadBinFileNNop(std::string &filePath, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    CHECK_RET(fileStatus == ACL_SUCCESS, LOG_PRINT("Failed to get file %s\n", filePath.c_str()); return -1);

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    CHECK_RET(file.is_open(), LOG_PRINT("Open file failed.\n"); return -1);
    
    file.seekg(0, file.end);
    uint64_t binFileBufferLen = file.tellg();
    CHECK_RET(binFileBufferLen > 0, std::cout << "File size is 0.\n"; file.close(); return -1);
    
    file.seekg(0, file.beg);
    file.read(static_cast<char *>(buffer), binFileBufferLen);
    file.close();
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int CreateAclTensor(std::string &filePath, const std::vector<int64_t> &shape, int typeSize, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * typeSize;
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMallocHost申请host侧内存
    void *binBufferHost = nullptr;
    ret = aclrtMallocHost(&binBufferHost, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
    
    // 读取文件
    ret = ReadBinFileNNop(filePath, binBufferHost, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ReadBinFileNNop failed. ERROR: %d\n", ret);
                                    (void)aclrtFreeHost(binBufferHost); return ret);
    
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, binBufferHost, size, ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtFreeHost(binBufferHost);
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

void FreeResource(aclTensor *q, aclTensor *k, aclTensor *v, aclTensor *k1, aclTensor *v1, aclTensor *attentionOut, aclTensor *softmaxMax,
    aclTensor *softmaxSum, void *qDeviceAddr, void *kDeviceAddr, void *vDeviceAddr, void *k1DeviceAddr, void *v1DeviceAddr, void *attnDeviceAddr, void *attentionOutDeviceAddr,
    void *softmaxMaxDeviceAddr, void *softmaxSumDeviceAddr, uint64_t workspaceSize, void *workspaceAddr,
    int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    if (q != nullptr) {
        aclDestroyTensor(q);
    }
    if (k != nullptr) {
        aclDestroyTensor(k);
    }
    if (v != nullptr) {
        aclDestroyTensor(v);
    }
    if (k1 != nullptr) {
        aclDestroyTensor(k1);
    }
    if (v1 != nullptr) {
        aclDestroyTensor(v1);
    }
    if (attentionOut != nullptr) {
        aclDestroyTensor(attentionOut);
    }
    if (softmaxMax != nullptr) {
        aclDestroyTensor(softmaxMax);
    }
    if (softmaxSum != nullptr) {
        aclDestroyTensor(softmaxSum);
    }

    // 释放device资源
    if (qDeviceAddr != nullptr) {
        aclrtFree(qDeviceAddr);
    }
    if (kDeviceAddr != nullptr) {
        aclrtFree(kDeviceAddr);
    }
    if (vDeviceAddr != nullptr) {
        aclrtFree(vDeviceAddr);
    }
    if (k1DeviceAddr != nullptr) {
        aclrtFree(k1DeviceAddr);
    }
    if (v1DeviceAddr != nullptr) {
        aclrtFree(v1DeviceAddr);
    }
    if (attnDeviceAddr != nullptr) {
        aclrtFree(attnDeviceAddr);
    }
    if (attentionOutDeviceAddr != nullptr) {
        aclrtFree(attentionOutDeviceAddr);
    }
    if (softmaxMaxDeviceAddr != nullptr) {
        aclrtFree(softmaxMaxDeviceAddr);
    }
    if (softmaxSumDeviceAddr != nullptr) {
        aclrtFree(softmaxSumDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (context != nullptr) {
        aclrtDestroyContext(context);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main(int argc, char **argv)
{
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // 如果需要修改shape值，需要同步修改../scripts/fa_generate_data.py中 test_fused_floyd_attention 分支下生成
    // query、key、value对应的shape值，并重新gen data，再执行
    std::ifstream file("../../../../../examples/transformer/scripts/layout.bin");
    if (!file.is_open()) {
        std::cerr << "无法打开文件 layout.bin" << std::endl;
        return 1;
    }
    int target_line;
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    iss >> target_line;
    for (int i = 1; i < target_line; ++i) {
        std::getline(file, line);
    }
    std::getline(file, line);
    std::istringstream dataStream(line);
    std::vector<int> data;
    int value;
    while (dataStream >> value) {
        std::cout << "layout value: " << value << std::endl;
        data.push_back(value);
    }
    // BHNMD
    int64_t B = data[0];
    int64_t H = data[1];
    int64_t N = data[2];
    int64_t M = data[3];
    int64_t K = data[4];
    int64_t D = data[5];
    double scaleValue = 1.0;
    // 五维数据修改
    std::vector<int64_t> qShape = {B, H, N, M, D};
    std::vector<int64_t> kShape = {B, H, N, K, D};
    std::vector<int64_t> k1Shape = {B, H, K, M, D};
    std::vector<int64_t> vShape = {B, H, N, K, D};
    std::vector<int64_t> attnShape = {B, H, N, M, K};
    std::vector<int64_t> attentionOutShape = {B, H, N, M, D};
    std::vector<int64_t> softmaxMaxShape = {B, H, N, M, 8};
    std::vector<int64_t> softmaxSumShape = {B, H, N, M, 8};

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *vDeviceAddr = nullptr;
    void *k1DeviceAddr = nullptr;
    void *v1DeviceAddr = nullptr;
    void *attnDeviceAddr = nullptr;
    void *attentionOutDeviceAddr = nullptr;
    void *softmaxMaxDeviceAddr = nullptr;
    void *softmaxSumDeviceAddr = nullptr;
    
    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *v = nullptr;
    aclTensor *k1 = nullptr;
    aclTensor *v1 = nullptr;
    aclTensor *attenMask = nullptr;
    aclTensor *softmaxMax = nullptr;
    aclTensor *softmaxSum = nullptr;
    aclTensor *attentionOut = nullptr;
    
    std::vector<float> attentionOutHostData(B*H*N*M*D, 0.0);
    std::vector<float> softmaxMaxHostData(B*H*N*M*8, 0.0);
    std::vector<float> softmaxSumHostData(B*H*N*M*8, 0.0);
    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;
    
    if (argv == nullptr || argv[0] == nullptr) {
        LOG_PRINT("Environment error, Argv=%p, Argv[0]=%p", argv, argv == nullptr ? nullptr : argv[0]);
        return 0;
    }
    std::string exeFile(argv[0]);
    std::string currentPath = std::string(exeFile.substr(0, exeFile.rfind('/')) + "/");
    // std::string qFilePath = currentPath + "query.bin";
    std::string qFilePath = "../../../../../examples/transformer/scripts/query.bin";
    ret = CreateAclTensor(qFilePath, qShape, 2, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS,
                FreeResource(q, k, v, k1, v1, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr, k1DeviceAddr, v1DeviceAddr, attnDeviceAddr,
                    attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                    deviceId, &context, &stream);
                return ret);
    
    // std::string kFilePath = currentPath + "key.bin";
    std::string kFilePath = "../../../../../examples/transformer/scripts/key.bin";
    
    ret = CreateAclTensor(kFilePath, kShape, 2, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS,
                FreeResource(q, k, v, k1, v1, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr, k1DeviceAddr, v1DeviceAddr, attnDeviceAddr,
                    attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                    deviceId, &context, &stream);
                return ret);
    
    // std::string vFilePath = currentPath + "value.bin";
    std::string vFilePath = "../../../../../examples/transformer/scripts/value.bin";
    ret = CreateAclTensor(vFilePath, vShape, 2, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS,
                FreeResource(q, k, v, k1, v1, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr, k1DeviceAddr, v1DeviceAddr, attnDeviceAddr,
                    attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                    deviceId, &context, &stream);
                return ret);
    
    std::string k1FilePath = "../../../../../examples/transformer/scripts/key1.bin";
    
    ret = CreateAclTensor(k1FilePath, k1Shape, 2, &k1DeviceAddr, aclDataType::ACL_FLOAT16, &k1);
    CHECK_RET(ret == ACL_SUCCESS,
                FreeResource(q, k, v, k1, v1, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr, k1DeviceAddr, v1DeviceAddr, attnDeviceAddr,
                    attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                    deviceId, &context, &stream);
                return ret);
    
    std::string v1FilePath = "../../../../../examples/transformer/scripts/value1.bin";
    
    ret = CreateAclTensor(v1FilePath, k1Shape, 2, &v1DeviceAddr, aclDataType::ACL_FLOAT16, &v1);
    CHECK_RET(ret == ACL_SUCCESS,
                FreeResource(q, k, v, k1, v1, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr, k1DeviceAddr, v1DeviceAddr, attnDeviceAddr,
                    attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                    deviceId, &context, &stream);
                return ret);
    
    // std::string attnFilePath = currentPath + "atten_mask.bin";
    std::string attnFilePath = "../../../../../examples/transformer/scripts/atten_mask.bin";
    ret = CreateAclTensor(attnFilePath, attnShape, 1, &attnDeviceAddr, aclDataType::ACL_UINT8, &attenMask);
    CHECK_RET(ret == ACL_SUCCESS,
                FreeResource(q, k, v, k1, v1, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr, k1DeviceAddr, v1DeviceAddr, attnDeviceAddr,
                    attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                    deviceId, &context, &stream);
                return ret);
    
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16,
                            &attentionOut);
    CHECK_RET(ret == ACL_SUCCESS,
                FreeResource(q, k, v, k1, v1, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr, k1DeviceAddr, v1DeviceAddr, attnDeviceAddr,
                    attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                    deviceId, &context, &stream);
                return ret);
    
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT,
                            &softmaxMax);
    CHECK_RET(ret == ACL_SUCCESS,
                FreeResource(q, k, v, k1, v1, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr, k1DeviceAddr, v1DeviceAddr, attnDeviceAddr,
                    attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                    deviceId, &context, &stream);
                return ret);
    
    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT,
                            &softmaxSum);
    CHECK_RET(ret == ACL_SUCCESS,
                FreeResource(q, k, v, k1, v1, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr, k1DeviceAddr, v1DeviceAddr, attnDeviceAddr,
                    attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                    deviceId, &context, &stream);
                return ret);
    
    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    aclOpExecutor *executor;
    
    // 调用aclnnFusedFloydAttention第一段接口
    ret = aclnnFusedFloydAttentionGetWorkspaceSize(
        q, k, v, k1, v1, attenMask, scaleValue, softmaxMax, softmaxSum, attentionOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedFloydAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
                FreeResource(q, k, v, k1, v1, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr, k1DeviceAddr, v1DeviceAddr, attnDeviceAddr,
                    attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                    deviceId, &context, &stream);
                return ret);
    
    // 根据第一段接口计算出的workspaceSize申请device内存
    // LOG_PRINT("workspaceSize: %d\n", workspaceSize);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                FreeResource(q, k, v, k1, v1, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr, k1DeviceAddr, v1DeviceAddr, attnDeviceAddr,
                    attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                    deviceId, &context, &stream);
            return ret);
    }
    
    // 调用aclnnFusedFloydAttention第二段接口
    ret = aclnnFusedFloydAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedFloydAttention failed. ERROR: %d\n", ret);
                FreeResource(q, k, v, k1, v1, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr, k1DeviceAddr, v1DeviceAddr, attnDeviceAddr,
                    attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                    deviceId, &context, &stream);
                return ret);
    
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
                FreeResource(q, k, v, k1, v1, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr, k1DeviceAddr, v1DeviceAddr, attnDeviceAddr,
                    attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                    deviceId, &context, &stream);
                return ret);
    
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    std::string attentionOutFileName = "attentionOut.bin";
    SaveOutResult<short>(attentionOutFileName, attentionOutShape, &attentionOutDeviceAddr);
    
    std::string softmaxMaxFileName = "softmaxMax.bin";
    SaveOutResult<float>(softmaxMaxFileName, softmaxMaxShape, &softmaxMaxDeviceAddr);
    
    std::string softmaxSumFileName = "softmaxSum.bin";
    SaveOutResult<float>(softmaxSumFileName, softmaxSumShape, &softmaxSumDeviceAddr);
    
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改; 释放device资源
    FreeResource(q, k, v, k1, v1, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr, k1DeviceAddr, v1DeviceAddr, attnDeviceAddr,
        attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
        deviceId, &context, &stream);
    
    return 0;
}
```