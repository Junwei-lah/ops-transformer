/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_allto_all_quant_matmul.h"
#include "securec.h"
#include "acl/acl.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "hccl_util.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/transdata.h"


namespace {

using namespace op;
using namespace l0op;

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

// 需要使用的常量定义
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t NEG_TWO = -2;
static constexpr int64_t ZERO = 0;
static constexpr size_t MAX_GROUP_LEN = 128U;
static constexpr size_t TWO_DIMS = 2U;

// 检查必要输入是否为空，必须非空
static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional, const aclTensor* x2Scale, const aclTensor* output) {
    if (x1 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x1 should not be null.");
        return false;
    }
    if (x2 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x2 should not be null.");
        return false;
    }
    if (biasOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input bias should not be null.");
        return false;
    }
    if (x2Scale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x2Scale should not be null.");
        return false;
    }
    if (output == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Output should not be null.");
        return false;
    }
    return true;
}

// 检查是否有空tensor
// 非量化场景支持x1的m轴为0，即token提示词为空
static bool CheckNotEmptyTensor(const aclTensor* x1, const aclTensor* x2) {
    auto kVal1 = x1->GetViewShape().GetDim(1);
    auto kVal2 = x2->GetViewShape().GetDim(0);
    auto nVal = x2->GetViewShape().GetDim(1);
    OP_API_CHECK((kVal1 == ZERO), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "X1 is empty tensor with zero dimK, which is unsupported.");
      return false;
    });
    OP_API_CHECK((kVal2 == ZERO), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "X2 is empty tensor with zero dimK, which is unsupported.");
      return false;
    });
    OP_API_CHECK((nVal == ZERO), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "X2 is empty tensor with zero dimN, which is unsupported.");
      return false;
    });
    return true;
}

// 检查AlltoAll和Permute数据交换的方向参数, 可以为空和{-2,-1}, 不允许为其他值
static bool CheckAlltoAllAxes(const aclIntArray* alltoAllAxesOptional)
{
    // alltoAllAxesOptional为空时会兼容性处理，不报错
    if (alltoAllAxesOptional == nullptr) {
        OP_LOGW("The alltoAllAxesOptional is nullptr.");
        return true;
    }
    uint64_t alltoallAxesSize = 0U;  // alltoallAxes的大小
    aclGetIntArraySize(alltoAllAxesOptional, &alltoallAxesSize);
    if (alltoallAxesSize != TWO_DIMS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of alltoAllAxesOptional should be 2U, but it is: %zu.", alltoallAxesSize);
        return false;
    }
    int64_t data1 = (*alltoAllAxesOptional)[0];
    int64_t data2 = (*alltoAllAxesOptional)[1];
    OP_API_CHECK((data1 != NEG_TWO), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "The 0-axis of alltoAllAxesOptional should be -2, but it is: %ld.", data1);
      return false;
    });
    OP_API_CHECK((data2 != NEG_ONE), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "The 1-axis of alltoAllAxesOptional should be -1, but it is: %ld.", data2);
      return false;
    });
    return true;
}

// 检查输入的转置配置，x1不允许转置
static bool CheckTransposeX1(bool transposeX1)
{
    OP_API_CHECK(transposeX1, {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The x1 should not be transposed, but it is transposed.");
    return false;
  });
    return true;
}

// 检查通信域名的字符串长度是否符合要求
static bool CheckGroupLength(const char *group)
{
    if (group == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Group should not be nullptr.");
        return false;
    }
    auto len = strnlen(group, MAX_GROUP_LEN);
    if ((len >= MAX_GROUP_LEN) || (len == ZERO)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Required group name length in range (0, 128), but it is %zu.", len);
        return false;
    }
    return true;
}

// 检查所有要用到的输入format是否为ND，不支持私有格式，如果内部不为ND格式，会打印warning日志，并将format转换为ND格式
static bool CheckFormat(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
                        const aclTensor* x2Scale, const aclTensor* output, const aclTensor* alltoAllOutOptional)
{
    // 输入格式不支持私有格式
    if (IsPrivateFormat(x1->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnAlltoAllQuantMatmul, x1 format %s does not support private format.",
                op::ToString(x1->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(x2->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnAlltoAllQuantMatmul, x2 format %s does not support private format.",
                op::ToString(x2->GetStorageFormat()).GetString());
        return false;
    }
    if (biasOptional != nullptr) {
        if (IsPrivateFormat(biasOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnAlltoAllQuantMatmul, biasOptional format %s does not support private format.",
                op::ToString(biasOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    if (IsPrivateFormat(x2Scale->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnAlltoAllQuantMatmul, x2Scale format %s does not support private format.",
                op::ToString(x2Scale->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(output->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnAlltoAllQuantMatmul, output format %s does not support private format.",
                op::ToString(output->GetStorageFormat()).GetString());
        return false;
    }
    if (alltoAllOutOptional != nullptr) {
        if (IsPrivateFormat(alltoAllOutOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnAlltoAllQuantMatmul, alltoAllOutOptional format %s does not support private format.",
                op::ToString(alltoAllOutOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    return true;
}

// 兼容性处理，非ND格式转换为ND格式
static bool ReFormatNotND(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
                          const aclTensor* x2Scale, const aclTensor* output, const aclTensor* alltoAllOutOptional)
{
    // 内部只处理ND格式，这里做reformat操作
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x1 origin format is %s.", op::ToString(x1->GetStorageFormat()).GetString());
        x1 = l0op::ReFormat(x1, op::Format::FORMAT_ND);
        CHECK_RET(x1 != nullptr, false);
    }
    if (x2->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x2 origin format is %s.", op::ToString(x2->GetStorageFormat()).GetString());
        x2 = l0op::ReFormat(x2, op::Format::FORMAT_ND);
        CHECK_RET(x2 != nullptr, false);
    }
    if (biasOptional != nullptr) {
        if (biasOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGW("bias origin format is %s.", op::ToString(biasOptional->GetStorageFormat()).GetString());
            biasOptional = l0op::ReFormat(biasOptional, op::Format::FORMAT_ND);
            CHECK_RET(biasOptional != nullptr, false);
        }
    }
    if (x2Scale->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x2Scale origin format is %s.", op::ToString(x2Scale->GetStorageFormat()).GetString());
        x2Scale = l0op::ReFormat(x2Scale, op::Format::FORMAT_ND);
        CHECK_RET(x2Scale != nullptr, false);
    }
    if (output->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("output origin format is %s.", op::ToString(output->GetStorageFormat()).GetString());
        output = l0op::ReFormat(output, op::Format::FORMAT_ND);
        CHECK_RET(output != nullptr, false);
    }
    if (alltoAllOutOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("alltoallout origin format is %s.", op::ToString(alltoAllOutOptional->GetStorageFormat()).GetString());
        alltoAllOutOptional = l0op::ReFormat(alltoAllOutOptional, op::Format::FORMAT_ND);
        CHECK_RET(alltoAllOutOptional != nullptr, false);
    }
    return true;
}

// 根据API定义，列出allto_all_quant_matmul非量化输入X1所能支持的所有dtype
static const std::initializer_list<op::DataType> X1_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT4
};

// 根据API定义，列出allto_all_quant_matmul非量化输入X2所能支持的所有dtype
static const std::initializer_list<op::DataType> X2_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_INT4
};

// 根据API定义，列出allto_all_quant_matmul非量化输入X2SCALE所能支持的所有dtype
static const std::initializer_list<op::DataType> X2SCALE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT
};

// 根据API定义，列出allto_all_quant_matmul非量化输出Output所能支持的所有dtype
static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16
};

// 校验所有输入的参数类型是否正确
static bool CheckAllDtypesValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
                                const aclTensor* x2Scale, const aclTensor* output, const aclTensor* alltoAllOutOptional) {
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, X1_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, X2_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2Scale, X2SCALE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, OUTPUT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(alltoAllOutOptional, X1_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, alltoAllOutOptional, return false);
    return true;
}

static aclnnStatus CheckAndHandleParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                        const aclTensor *x2Scale, const aclIntArray* alltoAllAxesOptional, const char *group,
                                        bool transposeX1, const aclTensor *output, const aclTensor *alltoAllOutOptional)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, biasOptional, x2Scale, output), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查空tensor
    CHECK_RET(CheckNotEmptyTensor(x1, x2), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckAllDtypesValid(x1, x2, biasOptional, x2Scale, output, alltoAllOutOptional), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查alltoallAxes是否为空或者[-2,-1]
    CHECK_RET(CheckAlltoAllAxes(alltoAllAxesOptional), ACLNN_ERR_PARAM_INVALID);
    // 5. 检查transposeX1是否合法, 目前不能为true
    CHECK_RET(CheckTransposeX1(transposeX1), ACLNN_ERR_PARAM_INVALID);
    // 6. 检查group长度是否小于等于128
    CHECK_RET(CheckGroupLength(group), ACLNN_ERR_PARAM_INVALID);
    // 7. 检查输入的数据格式是否为ND
    CHECK_RET(CheckFormat(x1, x2, biasOptional, x2Scale, output, alltoAllOutOptional), ACLNN_ERR_PARAM_INVALID);
    // 8. 兼容性处理非ND格式
    CHECK_RET(ReFormatNotND(x1, x2, biasOptional, x2Scale, output, alltoAllOutOptional), ACLNN_ERR_PARAM_INVALID);
    // 如果所有检查都通过，且reformat也通过，输出参数检查成功
    OP_LOGD("aclnnAlltoAllQuantMatmul checkParams success");
    return ACLNN_SUCCESS;
}
} // namespace

// L0层两段式接口Inner，根据算子原型op_graph/allto_all_quant_matmul_proto.h，由模板自动生成。非量化L2层接口和量化L2层接口共用一套L0层接口。
// worldSize为硬件方参数，在aclnn侧不感知。yDtype在aclnn侧不感知。这两个参数需要在Inner接口处声明，在aclnn侧通过默认值传参。
extern "C" aclnnStatus aclnnInnerAlltoAllMatmulGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                                                const aclTensor *x1ScaleOptional, const aclTensor *x2ScaleOptional,
                                                                const aclTensor *commScaleOptional,
                                                                const aclTensor *x1OffsetOptional, const aclTensor *x2OffsetOptional,
                                                                const char *group, int64_t worldSize, const aclIntArray* all2allAxesOptional,
                                                                int64_t yDtype, int64_t x1QuantMode, int64_t x2QuantMode,
                                                                int64_t commQuantMode, int64_t x1QuantDtype, int64_t commQuantDtype,
                                                                bool transposeX1, bool transposeX2, int64_t groupSize, bool all2AllOutFlag,
                                                                const aclTensor *out, const aclTensor *all2AllOutOptional,
                                                                uint64_t *workspaceSize, aclOpExecutor **executor);
extern "C" aclnnStatus aclnnInnerAlltoAllMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// 非量化L2接口调用L0时需要设置较多默认值，通过InnerAlltoAllQuantMatmulGetWorkspaceSize完成默认值传参和调用L0层接口
extern "C" aclnnStatus InnerAlltoAllQuantMatmulGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
    const aclTensor* biasOptional, const aclTensor* x1ScaleOptional, const aclTensor* x2Scale, const aclTensor* commScaleOptional,
    const aclTensor* x1OffsetOptional, const aclTensor* x2OffsetOptional, const char* group, const aclIntArray* alltoAllAxesOptional,
    int64_t x1QuantMode, int64_t x2QuantMode, int64_t commQuantMode, int64_t commQuantDtype, int64_t x1QuantDtype, int64_t groupSize,
    bool transposeX1, bool transposeX2, const aclTensor* output, const aclTensor* all2AllOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 需要使用的默认值常量定义
    const int64_t WORLD_SIZE = -1;
    const int64_t Y_DTYPE = 28;  // 代表ge::UNDEFINED，不指定输出类型
    // 根据算子原型定义默认值
    char* str_group = const_cast<char*>(group);
    int64_t worldSize = WORLD_SIZE;
    int64_t yDtype = Y_DTYPE;
    bool all2AllOutFlag = true;
    aclnnStatus ret = aclnnInnerAlltoAllMatmulGetWorkspaceSize(
        x1, x2, biasOptional, x1ScaleOptional, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
        str_group, worldSize, alltoAllAxesOptional, yDtype, x1QuantMode, x2QuantMode, commQuantMode, x1QuantDtype, commQuantDtype,
        transposeX1, transposeX2, groupSize, all2AllOutFlag, output, all2AllOutOptional, workspaceSize, executor);
    OP_LOGD("AlltoAllQuantMatmul, aclnnnInnerGetWorkspaceSize ret %d.", ret);
    return ret;
}

// 两段式接口
extern "C" aclnnStatus aclnnAlltoAllQuantMatmulGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
    const aclTensor* biasOptional, const aclTensor* x1ScaleOptional, const aclTensor* x2Scale, const aclTensor* commScaleOptional,
    const aclTensor* x1OffsetOptional, const aclTensor* x2OffsetOptional, const char* group, const aclIntArray* alltoAllAxesOptional,
    int64_t x1QuantMode, int64_t x2QuantMode, int64_t commQuantMode, int64_t commQuantDtype, int64_t x1QuantDtype, int64_t groupSize,
    bool transposeX1, bool transposeX2, const aclTensor* output, const aclTensor* alltoAllOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    aclnnStatus retParam = CheckAndHandleParams(x1, x2, biasOptional, x2Scale, alltoAllAxesOptional, group, transposeX1, output, alltoAllOutOptional);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    aclnnStatus ret = InnerAlltoAllQuantMatmulGetWorkspaceSize(
        x1, x2, biasOptional, x1ScaleOptional, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional, group, alltoAllAxesOptional,
        x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype, x1QuantDtype, groupSize,
        transposeX1, transposeX2, output, alltoAllOutOptional, workspaceSize, executor);
    OP_LOGD("AlltoAllQuantMatmul, end ret %d", ret);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "This is an error in launch aicore, aclnnAlltoAllQuantMatmulGetWorkspaceSize interface call failed.");
    }
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllQuantMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
    aclnnStatus ret = aclnnInnerAlltoAllMatmul(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "This is an error in launch aicore, aclnnInnerAlltoAllMatmul interface call failed.");
        return ret;
    }
    return ACLNN_SUCCESS;
}
