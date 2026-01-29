# aclnnKvRmsNormRopeCache

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/kv_rms_norm_rope_cache)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |


## 功能说明

- 接口功能：对输入张量（kv）的尾轴，拆分出左半边用于rms_norm计算，右半边用于RoPE计算，再将计算结果分别scatter到两块cache中。

- 计算公式：
  
  (1) interleaveRope:

  $$
  x=kv[...,Dv:]
  $$

  $$
  x1=x[...,::2]
  $$

  $$
  x2=x[...,1::2]
  $$

  $$
  x\_part1=torch.cat((x1,x2),dim=-1)
  $$

  $$
  x\_part2=torch.cat((-x2,x1),dim=-1)
  $$

  $$
  y=x\_part1*cos+x\_part2*sin
  $$

  (2) rmsNorm:

  $$
  x=kv[...,:Dv]
  $$

  $$
  square\_x=x*x
  $$

  $$
  mean\_square\_x=square\_x.mean(dim=-1,keepdim=True)
  $$

  $$
  rms=torch.sqrt(mean\_square\_x+epsilon)
  $$

  $$
  y=(x/rms)*gamma
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnKvRmsNormRopeCacheGetWorkspaceSize”接口获得入参并根据流程计算所需workspace大小，再调用“aclnnKvRmsNormRopeCache”接口执行计算。

```Cpp
aclnnStatus aclnnKvRmsNormRopeCacheGetWorkspaceSize(
  const aclTensor* kv, 
  const aclTensor* gamma, 
  const aclTensor* cos, 
  const aclTensor* sin, 
  const aclTensor* index, 
  aclTensor*       kCacheRef, 
  aclTensor*       ckvCacheRef, 
  const aclTensor* kRopeScaleOptional, 
  const aclTensor* ckvScaleOptional, 
  const aclTensor* kRopeOffsetOptional, 
  const aclTensor* cKvOffsetOptional, 
  double           epsilon, 
  char*            cacheModeOptional, 
  bool             isOutputKv, 
  aclTensor*       kRopeOut, 
  aclTensor*       cKvOut,
  uint64_t         workspaceSize, 
  aclOpExecutor*   executor)
```

```Cpp
aclnnStatus aclnnKvRmsNormRopeCache(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnKvRmsNormRopeCacheGetWorkspaceSize

- **参数说明**
  * kv(aclTensor\*，计算输入)：必选参数，公式中用于切分出rms_norm计算所需数据Dv和RoPE计算所需数据Dk的输入数据。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。shape仅支持4维[Bkv,N,Skv,D]。[数据格式](../../../docs/zh/context/数据格式.md)支持ND，数据类型支持FLOAT6、BFLOAT16。
  * gamma(aclTensor\*，计算输入)：必选参数，公式中用于rms_norm计算的输入数据。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。shape为1维[Dv,]。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。数据类型与输入kv一致。
  * cos(aclTensor\*，计算输入)：必选参数，公式中用于RoPE计算的输入数据，对输入张量Dk进行余弦变换，Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。shape为4维[Bkv,1,Skv,Dk]或[Bkv,1,1,Dk]，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。数据类型与输入kv一致。
  * sin(aclTensor\*，计算输入)：必选参数，公式中用于RoPE计算的输入数据，对输入张量Dk进行正弦变换。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。shape为4维[Bkv,1,Skv,Dk]或[Bkv,1,1,Dk]，与cos的shape保持一致。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。数据类型与输入kv一致。
  * index(aclTensor\*，计算输入)：必选参数，用于指定写入cache的具体索引位置，当index的value数值为-1时，代表跳过更新。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。当cacheModeOptional为Norm时，shape为2维[Bkv,Skv]；当cacheModeOptional为PA_BNSD、PA_NZ时，shape为1维[Bkv * Skv]；当cacheModeOptional为PA_BLK_BSND、PA_BLK_NZ时，shape为1维[Bkv\*ceil_div(Skv,BlockSize)]。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。数据类型支持INT64。 
  * kCacheRef(aclTensor\*，计算输入/输出)：必选参数，提前申请的cache，输入输出同地址复用。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。当cacheModeOptional为PA场景（cacheModeOptional为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ）时，shape为4维[BlockNum,BlockSize,N,Dk]；当cacheModeOptional为Norm场景时，shape为4维[Bcache,N,Scache,Dk]。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。数据类型与输入kv一致或者INT8。
  * ckvCacheRef(aclTensor\*，计算输入/输出)：必选参数，提前申请的cache，输入输出同地址复用。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。当cacheModeOptional为PA场景（cacheModeOptional为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ）时，shape为4维[BlockNum,BlockSize,N,Dv]；当cacheModeOptional为Norm场景时，shape为4维[Bcache,N,Scache,Dv]。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。数据类型与输入kv一致或者INT8。
  * kRopeScaleOptional(aclTensor\*，计算输入)：可选参数，当kCacheRef数据类型为INT8时需要此输入参数。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。shape为2维[N,Dk]；或者shape为1维[Dk,]；或者shape为1维[1,]。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。数据类型支持FLOAT32。
  * ckvScaleOptional(aclTensor\*，计算输入)：可选参数，当ckvCacheRef数据类型为INT8时需要此输入参数。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。shape为2维[N,Dv]；或者shape为1维[Dv,]；或者shape为1维[1,]。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。数据类型支持FLOAT32。
  * kRopeOffsetOptional(aclTensor\*，计算输入)：可选参数。当kCacheRef数据类型为INT8且对应的kRopeScaleOptional输入存在并量化场景为非对称量化时，需要此参数输入。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。shape为2维[N,Dk]；或者shape为1维[Dk,]；或者shape为1维[1,]。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。数据类型支持FLOAT32。
  * cKvOffsetOptional(aclTensor\*，计算输入)：可选参数，当ckvCacheRef数据类型为INT8且对应的ckvScaleOptional输入存在并量化场景为非对称量化时，需要此参数输入。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。shape为2维[N,Dv]；或者shape为1维[Dv,]；或者shape为1维[1,]。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。数据类型支持FLOAT32。
  * epsilon(double，输入)：必选参数，rms_norm计算防止除0。float类型浮点数。建议设为1e-5。
  * cacheModeOptional(char\*，输入)：必选参数，cache格式的选择标记。char\*类型。类型有Norm、PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ，建议设为Norm。
  * isOutputKv(bool，输入)：必选参数，kRopeOut和cKvOut输出控制标记。bool类型。当isOutputKv为true时，表示需输出kRopeOut和cKvOut。建议设为false。
  * kRopeOut(aclTensor\*，计算输出)：由isOutputKv控制，当isOutputKv为true时，需输出。shape为4维[Bkv,N,Skv,Dk]。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。数据类型与输入kv一致。
  * cKvOut(aclTensor\*，计算输出)：由isOutputKv控制，当isOutputKv为true时，需输出。shape为4维[Bkv,N,Skv,Dv]。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。数据类型与输入kv一致。
  * workspaceSize(uint64_t\*，出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor\*\*，出参)：返回op执行器，包含了算子计算流程。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1260px"><colgroup>
  <col style="width: 325px">
  <col style="width: 126px">
  <col style="width: 809px">
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
      <td>输入和输出Tensor是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入和输出数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="4">561002</td>
      <td>参数kv、gamma、cos、sin、index、kCacheRef、ckvCacheRef、kRopeScaleOptional、ckvScaleOptional、kRopeOffsetOptional、cKvOffsetOptional的shape校验非法。</td>
    </tr>
    <tr>
      <td>参数kv、gamma、cos、sin、index、kCacheRef、ckvCacheRef、kRopeScaleOptional、ckvScaleOptional、kRopeOffsetOptional、cKvOffsetOptional的dtype校验非法。</td>
    </tr>
    <tr>
      <td>PA场景（cacheModeOptional为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ）下，cache的BlockSize维度值校验非法。</td>
    </tr>
    <tr>
      <td>NZ场景（cacheModeOptional为PA_NZ、PA_BLK_NZ）下，Dk、Dv的维度值校验非法。</td>
    </tr>
  </tbody>
  </table>

## aclnnKvRmsNormRopeCache

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
      <td>在Device侧申请的workspace大小，由第一段接口clnnKvRmsNormRopeCacheGetWorkspaceSize获取。</td>
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
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

  * 参数说明里shape格式说明：
      * Bkv为输入kv的batch size，Skv为输入kv的sequence length，大小由用户输入场景决定，无明确限制。 
      * N为输入kv的head number。此算子与DeepSeekV3网络结构强相关，仅支持N=1的场景，不存在N非1的场景。
      * D为输入kv的head dim。rms_norm计算所需数据Dv和RoPE计算所需数据Dk由输入kv的D切分而来。故Dk、Dv大小需满足Dk+Dv=D。同时，Dk需满足rope规则。根据rope规则，Dk为偶数。若cacheModeOptional为NZ场景（cacheModeOptional为PA_NZ、PA_BLK_NZ），Dk、Dv需32B对齐。
      * 若cacheModeOptional为PA场景（cacheModeOptional为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ），BlockSize需32B对齐。
      * 关于上述32B对齐的情形，对齐值由cache的数据类型决定。以BlockSize为例，若cache的数据类型为INT8，则需BlockSize%32=0；若cache的数据类型为float16，则需BlockSize%16=0；若kCacheRef与ckvCacheRef参数的dtype不一致，BlockSize需同时满足BlockSize%32=0和BlockSize%16=0。
      * Bcache为输入cache的batch size，Scache为输入cache的sequence length，大小由用户输入场景决定，无明确限制。 
      * BlockNum为写入cache的内存块数，大小由用户输入场景决定，无明确限制。 
  * index相关约束：
      * 当cacheModeOptional为Norm时，shape为2维[Bkv,Skv]，要求index的value值范围为[-1,Scache)。不同的Bkv下，value数值可以重复。
      * 当cacheModeOptional为PA_BNSD、PA_NZ时，shape为1维[Bkv * Skv]，要求index的value值范围为[-1,BlockNum * BlockSize)。value数值不能重复。
      * 当cacheModeOptional为PA_BLK_BSND、PA_BLK_NZ时，shape为1维[Bkv * ceil_div(Skv,BlockSize)]，要求index的value的数值范围为[-1,BlockNum * BlockSize)。value/BlockSize的值不能重复。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_kv_rms_norm_rope_cache.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<int8_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
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
  // 1. 固定写法，device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口定义构造
  std::vector<int64_t> kvShape = {181,1,1,576};
  std::vector<int64_t> gammaShape = {512,};
  std::vector<int64_t> cosShape = {181,1,1,64};
  std::vector<int64_t> sinShape = {181,1,1,64};
  std::vector<int64_t> indexShape = {181,1};
  std::vector<int64_t> kpeCacheShape = {181,1,1,64};
  std::vector<int64_t> ckvCacheShape = {181,1,1,512};
  std::vector<int64_t> kRopeShape = {181,1,1,64};
  std::vector<int64_t> cKvShape = {181,1,1,512};
  
  std::vector<int16_t> kvHostData(181*1*1*576,0);
  std::vector<int16_t> gammaHostData(512,0);
  std::vector<int16_t> cosHostData(181*1*1*64,0);
  std::vector<int16_t> sinHostData(181*1*1*64,0);
  std::vector<int64_t> indexHostData(181*1,0);
  std::vector<int16_t> kpeCacheHostData(181*1*1*64,0);
  std::vector<int16_t> ckvCacheHostData(181*1*1*512,0);
  std::vector<int16_t> kRopeHostData(181*1*1*64,0);
  std::vector<int16_t> cKvHostData(181*1*1*512,0);
  
  void* kvDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* cosDeviceAddr = nullptr;
  void* sinDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* kpeCacheDeviceAddr = nullptr;
  void* ckvCacheDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* cKvDeviceAddr = nullptr;
 
  aclTensor* kv = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* cos = nullptr;
  aclTensor* sin = nullptr;
  aclTensor* index = nullptr;
  aclTensor* kpeCache = nullptr;
  aclTensor* ckvCache = nullptr;
  aclTensor* kRope = nullptr;
  aclTensor* cKv = nullptr;
 

  double epsilon = 1e-5;
  char cacheMode[] = "Norm";
  bool isOutputKv = false;

  ret = CreateAclTensor(kvHostData, kvShape, &kvDeviceAddr, aclDataType::ACL_FLOAT16, &kv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT16, &cos);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT16, &sin);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kpeCacheHostData, kpeCacheShape, &kpeCacheDeviceAddr, aclDataType::ACL_FLOAT16, &kpeCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(ckvCacheHostData, ckvCacheShape, &ckvCacheDeviceAddr, aclDataType::ACL_FLOAT16, &ckvCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cKvHostData, cKvShape, &cKvDeviceAddr, aclDataType::ACL_FLOAT16, &cKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnKvRmsNormRopeCache第一段接口
  ret = aclnnKvRmsNormRopeCacheGetWorkspaceSize(kv,gamma,cos,sin,index, 
                                                kpeCache,ckvCache,nullptr,nullptr,nullptr,nullptr,epsilon,cacheMode,isOutputKv,kRope,cKv,&workspaceSize,&executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvRmsNormRopeCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnKvRmsNormRopeCache第二段接口
  ret = aclnnKvRmsNormRopeCache(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvRmsNormRopeCache failed. ERROR: %d\n", ret); return ret);

  // 4. 固定写法，同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(kpeCacheShape, &kpeCacheDeviceAddr);
  PrintOutResult(ckvCacheShape, &ckvCacheDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(kv);
  aclDestroyTensor(gamma);
  aclDestroyTensor(cos);
  aclDestroyTensor(sin);
  aclDestroyTensor(index);
  aclDestroyTensor(kpeCache);
  aclDestroyTensor(ckvCache);
  aclDestroyTensor(kRope);
  aclDestroyTensor(cKv);

  // 7. 释放device 资源
  aclrtFree(kvDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(cosDeviceAddr);
  aclrtFree(sinDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(kpeCacheDeviceAddr);
  aclrtFree(ckvCacheDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(cKvDeviceAddr);


  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
