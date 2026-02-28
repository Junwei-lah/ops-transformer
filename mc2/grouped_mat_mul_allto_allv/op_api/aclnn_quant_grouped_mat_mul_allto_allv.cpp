/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_grouped_mat_mul_allto_allv.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transdata.h"
#include "hccl_util.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "securec.h"
#include <algorithm>

namespace {
using namespace op;

enum class QuantModeType : int64_t {
    NO_QUANT = 0,
    PERTENSOR_QUANT = 1,
    PERCHANNEL_QUANT = 2,
    PERTOKEN_QUANT = 3,
    PERGROUP_QUANT = 4,
    PERBLOCK_QUANT = 5,
    MX_QUANT = 6,
    DYN_PERTOKEN_QUANT = 7
};


enum class NnopbaseHcclServerType : uint32_t { // HCCL Server
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

static constexpr int64_t DIM_TWO = 2;
static constexpr int64_t DIM_THREE = 3;

extern "C" aclnnStatus aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const aclTensor *gmmXScaleOptional, const aclTensor *gmmWeightScaleOptional, const aclTensor *gmmXOffsetOptional,
    const aclTensor *gmmWeightOffsetOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const aclTensor *mmXOffsetOptional, const aclTensor *mmWeightOffsetOptional,
    const aclTensor *commQuantScaleOptional, const char *group, int64_t epWorldSize, const aclIntArray *sendCounts,
    const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight, int64_t gmmXQuantMode,
    int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode, int64_t commQuantMode,
    int64_t groupSize, int64_t commQuantDtypeOptional, const aclTensor *yOut, const aclTensor *mmYOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerGroupedMatMulAlltoAllv(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// 检查必要输入是否为空，必须非空
static bool CheckNotNull(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *y)
{
    if (gmmX == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input gmmX should not be null.");
        return false;
    }
    if (gmmWeight == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input gmmWeight should not be null.");
        return false;
    }
    if (y == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "y should not be null.");
        return false;
    }
    return true;
}

// 检查 mm 系列 optional 参数一致性：全空或全非空
static bool CheckMmOptionalConsistency(const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                                       const aclTensor *mmYOptional)
{
    if ((!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))) &&
        (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr)))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "mmXOptional, mmWeightOptional and mmYOptional should all be null or all not be null, "
                "left: %u, right: %u, mmXOptional is nullptr: %u, mmWeightOptional is nullptr: %u, mmYOptional is "
                "nullptr: %u",
                (!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))),
                (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr))),
                mmXOptional == nullptr, mmWeightOptional == nullptr, mmYOptional == nullptr);
        return false;
    }
    return true;
}

// check nullptr
static bool CheckNullStatus(const aclTensor *gmmX, const aclTensor *gmmWeight,
                            const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
                            const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                            const aclTensor *mmXScaleOptional, const char *group, const aclTensor *y,
                            const aclTensor *mmYOptional)
{
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sendCountsTensorOptional and recvCountsTensorOptional should be empty.");
        return false;
    }
    if ((group == nullptr) || (strnlen(group, HCCL_GROUP_NAME_MAX) == 0)) { // HCCL_GROUP_NAME_MAX = 128U
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name is Empty.");
        return false;
    }
    return CheckMmOptionalConsistency(mmXOptional, mmWeightOptional, mmYOptional);
}

// 检查暂不支持的输入参数是否为空，必须为空
static bool CheckNotSupportNull(const aclTensor *gmmXOffsetOptional, const aclTensor *gmmWeightOffsetOptional,
                                const aclTensor *mmXOffsetOptional, const aclTensor *mmWeightOffsetOptional)
{
    if (gmmXOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input gmmXOffsetOptional should be null.");
        return false;
    }
    if (gmmWeightOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input gmmWeightOffsetOptional should be null.");
        return false;
    }
    if (mmXOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input mmXOffsetOptional should be null.");
        return false;
    }
    if (mmWeightOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input mmWeightOffsetOptional should be null.");
        return false;
    }
    return true;
}

static aclnnStatus CheckIntArrayNotEmpty(const aclIntArray *arr, const char *name)
{
    if (arr == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s should not be null.", name);
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t size = 0U;
    aclGetIntArraySize(arr, &size);
    if (size == 0U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s size should not be 0.", name);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// check send and recv
static aclnnStatus CheckSendAndRecv(const aclIntArray *sendCounts, const aclIntArray *recvCounts)
{
    auto ret = CheckIntArrayNotEmpty(sendCounts, "sendCounts");
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    return CheckIntArrayNotEmpty(recvCounts, "recvCounts");
}

// 检查是否有空tensor
static bool CheckNotEmptyTensor(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *y,
                                const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                                const aclTensor *mmYOptional)
{
    auto h1Val = gmmX->GetViewShape().GetDim(1);

    auto h1Val2 = gmmWeight->GetViewShape().GetDim(1);
    auto n1Val = gmmWeight->GetViewShape().GetDim(DIM_TWO);
    auto yMval = y->GetViewShape().GetDim(0);
    auto yNval = y->GetViewShape().GetDim(1);

    OP_API_CHECK((h1Val == 0), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmX is empty tensor with zero dimN, which is unsupported.");
        return false;
    });
    OP_API_CHECK((h1Val2 == 0), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight is empty tensor with zero dimN, which is unsupported.");
        return false;
    });
    OP_API_CHECK((n1Val == 0), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight is empty tensor with zero dimK, which is unsupported.");
        return false;
    });
    OP_API_CHECK((yMval == 0), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y is empty tensor with zero dimM, which is unsupported.");
        return false;
    });
    OP_API_CHECK((yNval == 0), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y is empty tensor with zero dimN, which is unsupported.");
        return false;
    });

    if (mmXOptional == nullptr || mmWeightOptional == nullptr || mmYOptional == nullptr) {
        return true;
    }

    auto mmDim0 = mmXOptional->GetViewShape().GetDim(0);
    auto mmDim1 = mmXOptional->GetViewShape().GetDim(1);

    auto mmWdim0 = mmWeightOptional->GetViewShape().GetDim(0);
    auto mmWdim1 = mmWeightOptional->GetViewShape().GetDim(1);

    auto mmYdim0 = mmYOptional->GetViewShape().GetDim(0);
    auto mmYdim1 = mmYOptional->GetViewShape().GetDim(1);

    OP_API_CHECK(
        ((!((mmDim0 != 0) && (mmDim1 != 0) && (mmWdim0 != 0) && (mmWdim1 != 0) && (mmYdim0 != 0) && (mmYdim1 != 0))) &&
         (!((mmDim0 == 0) && (mmDim1 == 0) && (mmWdim0 == 0) && (mmWdim1 == 0) && (mmYdim0 == 0) && (mmYdim1 == 0)))),
        {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "mmXOptional, mmWeightOptional and mmYOptional should all be empty tensor or all not be empty tensor.");
            return false;
        });

    return true;
}

// 检查所有要用到的format是否为ND，不支持私有格式，如果内部不为ND格式，打印warning日志，将format转换为ND格式
static bool CheckFormat(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
                        const aclTensor *gmmWeightScaleOptional, const aclTensor *mmXOptional,
                        const aclTensor *mmWeightOptional, const aclTensor *y, const aclTensor *mmYOptional)
{
    // 定义内联检查函数
    auto checkNotPrivate = [](const aclTensor *tensor, const char *name) -> bool {
        if (tensor == nullptr) {
            return true;
        }
        if (IsPrivateFormat(tensor->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "aclnnQuantGroupMatmulAlltoAll, %s format %s does not support private format.", name,
                    op::ToString(tensor->GetStorageFormat()).GetString());
            return false;
        }
        return true;
    };

    // 必传参数检查
    auto checkRequired = [](const aclTensor *tensor, const char *name) -> bool {
        if (IsPrivateFormat(tensor->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "aclnnQuantGroupMatmulAlltoAll, %s format %s does not support private format.", name,
                    op::ToString(tensor->GetStorageFormat()).GetString());
            return false;
        }
        return true;
    };

    // 执行检查
    if (!checkRequired(gmmX, "gmmX"))
        return false;
    if (!checkRequired(gmmWeight, "gmmWeight"))
        return false;
    if (!checkNotPrivate(gmmXScaleOptional, "gmmXScaleOptional"))
        return false;
    if (!checkNotPrivate(gmmWeightScaleOptional, "gmmWeightScaleOptional"))
        return false;
    if (!checkNotPrivate(mmXOptional, "mmXOptional"))
        return false;
    if (!checkNotPrivate(mmWeightOptional, "mmWeightOptional"))
        return false;
    if (!checkRequired(y, "y"))
        return false;
    if (!checkNotPrivate(mmYOptional, "mmYOptional"))
        return false;

    return true;
}

static bool ReFormatTensorToND(const aclTensor *tensor, const char *name)
{
    if (tensor != nullptr && tensor->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("%s origin format is %s.", name, op::ToString(tensor->GetStorageFormat()).GetString());
        tensor = l0op::ReFormat(tensor, op::Format::FORMAT_ND);
        CHECK_RET(tensor != nullptr, false);
    }
    return true;
}

// 兼容性处理，非ND格式转换为ND格式
static bool ReFormatNotND(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
                          const aclTensor *gmmWeightScaleOptional, const aclTensor *mmXOptional,
                          const aclTensor *mmWeightOptional, const aclTensor *y, const aclTensor *mmYOptional)
{
    CHECK_RET(ReFormatTensorToND(gmmX, "gmmX"), false);
    CHECK_RET(ReFormatTensorToND(gmmWeight, "gmmWeight"), false);
    CHECK_RET(ReFormatTensorToND(gmmXScaleOptional, "gmmXScaleOptional"), false);
    CHECK_RET(ReFormatTensorToND(gmmWeightScaleOptional, "gmmWeightScaleOptional"), false);
    CHECK_RET(ReFormatTensorToND(mmXOptional, "mmXOptional"), false);
    CHECK_RET(ReFormatTensorToND(mmWeightOptional, "mmWeightOptional"), false);
    CHECK_RET(ReFormatTensorToND(y, "y"), false);
    CHECK_RET(ReFormatTensorToND(mmYOptional, "mmYOptional"), false);
    return true;
}

static bool CheckSingleQuantMode(int64_t quantMode, const aclTensor *scaleOptional, const char *scaleName)
{
    if (static_cast<QuantModeType>(quantMode) == QuantModeType::NO_QUANT && scaleOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s should be empty.", scaleName);
        return false;
    }
    if (static_cast<QuantModeType>(quantMode) == QuantModeType::PERTENSOR_QUANT && scaleOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s should not be null.", scaleName);
        return false;
    }
    return true;
}

// 检查量化参数是否合法
static bool CheckQuantMode(int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode,
                           int64_t mmWeightQuantMode, const aclTensor *gmmXScaleOptional,
                           const aclTensor *gmmWeightScaleOptional, const aclTensor *mmXScaleOptional,
                           const aclTensor *mmWeightScaleOptional)
{
    CHECK_RET(CheckSingleQuantMode(gmmXQuantMode, gmmXScaleOptional, "gmmXScaleOptional"), false);
    CHECK_RET(CheckSingleQuantMode(gmmWeightQuantMode, gmmWeightScaleOptional, "gmmWeightScaleOptional"), false);
    CHECK_RET(CheckSingleQuantMode(mmXQuantMode, mmXScaleOptional, "mmXScaleOptional"), false);
    CHECK_RET(CheckSingleQuantMode(mmWeightQuantMode, mmWeightScaleOptional, "mmWeightScaleOptional"), false);
    return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
                               const aclTensor *gmmWeightScaleOptional, const aclTensor *gmmXOffsetOptional,
                               const aclTensor *gmmWeightOffsetOptional, const aclTensor *sendCountsTensorOptional,
                               const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
                               const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                               const aclTensor *mmWeightScaleOptional, const aclTensor *mmXOffsetOptional,
                               const aclTensor *mmWeightOffsetOptional, int64_t gmmXQuantMode,
                               int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
                               int64_t commQuantMode, const char *group, int64_t epWorldSize,
                               const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight,
                               bool transMmWeight, const aclTensor *y, const aclTensor *mmYOptional,
                               uint64_t *workspaceSize, aclOpExecutor **executor)
{
    (void)epWorldSize;
    (void)sendCounts;
    (void)recvCounts;
    CHECK_RET(CheckNotNull(gmmX, gmmWeight, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckNullStatus(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional,
                              mmWeightOptional, mmXScaleOptional, group, y, mmYOptional),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(
        CheckNotSupportNull(gmmXOffsetOptional, gmmWeightOffsetOptional, mmXOffsetOptional, mmWeightOffsetOptional),
        ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNotEmptyTensor(gmmX, gmmWeight, y, mmXOptional, mmWeightOptional, mmYOptional),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckQuantMode(gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, gmmXScaleOptional,
                             gmmWeightScaleOptional, mmXScaleOptional, mmWeightScaleOptional),
              ACLNN_ERR_PARAM_INVALID);
    if (commQuantMode != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "commQuantMode only supports 0, but got %ld.", commQuantMode);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strnlen(group, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required group name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("aclnnQuantMatmulAlltoAll checkParams success");

    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
    const aclTensor *gmmWeightScaleOptional, const aclTensor *gmmXOffsetOptional,
    const aclTensor *gmmWeightOffsetOptional, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional, const aclTensor *mmXOffsetOptional,
    const aclTensor *mmWeightOffsetOptional, const aclTensor *commQuantScaleOptional, int64_t gmmXQuantMode,
    int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode, int64_t commQuantMode,
    int64_t commQuantDtypeOptional,
    int64_t groupSize, const char *group, int64_t epWorldSize, const aclIntArray *sendCounts,
    const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight, const aclTensor *y,
    const aclTensor *mmYOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto retParam = CheckParams(gmmX, gmmWeight, gmmXScaleOptional, gmmWeightScaleOptional, gmmXOffsetOptional,
                                gmmWeightOffsetOptional, sendCountsTensorOptional, recvCountsTensorOptional,
                                mmXOptional, mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional,
                                mmXOffsetOptional, mmWeightOffsetOptional, gmmXQuantMode, gmmWeightQuantMode,
                                mmXQuantMode, mmWeightQuantMode, commQuantMode, group, epWorldSize, sendCounts,
                                recvCounts, transGmmWeight, transMmWeight, y, mmYOptional, workspaceSize, executor);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    auto retSendAndRecv = CheckSendAndRecv(sendCounts, recvCounts);
    CHECK_RET(retSendAndRecv == ACLNN_SUCCESS, retSendAndRecv);

    char *strGroup = const_cast<char *>(group);

    aclnnStatus ret = aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
        gmmXScaleOptional, gmmWeightScaleOptional, gmmXOffsetOptional, gmmWeightOffsetOptional, mmXScaleOptional,
        mmWeightScaleOptional, mmXOffsetOptional, mmWeightOffsetOptional, commQuantScaleOptional, strGroup, epWorldSize,
        sendCounts, recvCounts, transGmmWeight, transMmWeight, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
        mmWeightQuantMode, commQuantMode, groupSize, commQuantDtypeOptional, y, mmYOptional, workspaceSize, executor);
    return ret;
}

extern "C" aclnnStatus aclnnQuantGroupedMatMulAlltoAllv(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND950) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    aclnnStatus ret = aclnnInnerGroupedMatMulAlltoAllv(workspace, workspaceSize, executor, stream);
    return ret;
}
} // namespace