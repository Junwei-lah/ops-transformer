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
#include "common/utils/hccl_util.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "platform/soc_spec.h"
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

extern "C" aclnnStatus aclnnInnerQuantGroupedMatMulAlltoAllvGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const aclTensor *gmmXScaleOptional, const aclTensor *gmmWeightScaleOptional, const aclTensor *mmXScaleOptional,
    const aclTensor *mmWeightScaleOptional, const aclTensor *commQuantScaleOptional, const char *group,
    int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight,
    bool transMmWeight, int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode,
    int64_t mmWeightQuantMode, int64_t commQuantMode, int64_t groupSize, int64_t commQuantDtypeOptional, int64_t yDtype,
    int64_t mmDtype, const aclTensor *yOut, const aclTensor *mmYOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerQuantGroupedMatMulAlltoAllv(void *workspace, uint64_t workspaceSize,
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
    const bool hasMmX = (mmXOptional != nullptr);
    const bool hasMmWeight = (mmWeightOptional != nullptr);
    const bool hasMmY = (mmYOptional != nullptr);
    const bool allPresent = hasMmX && hasMmWeight && hasMmY;
    const bool allAbsent = !hasMmX && !hasMmWeight && !hasMmY;
    if (!allPresent && !allAbsent) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "mmXOptional, mmWeightOptional, and mmYOptional must be either all empty or all non-empty, "
                "but got: mmXOptional is %s, mmWeightOptional is %s, mmYOptional is %s.",
                hasMmX ? "non-empty" : "empty", hasMmWeight ? "non-empty" : "empty", hasMmY ? "non-empty" : "empty");
        return false;
    }
    return true;
}

// check nullptr
static bool CheckNullStatus(const aclTensor *gmmX, const aclTensor *gmmWeight,
                            const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
                            const aclTensor *mmXOptional, const aclTensor *mmWeightOptional, const char *group,
                            const aclTensor *y, const aclTensor *mmYOptional)
{
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                "sendCountsTensorOptional and recvCountsTensorOptional should be all empty, "
                "but got: sendCountsTensorOptional is %s, recvCountsTensorOptional is %s.",
                sendCountsTensorOptional ? "non-empty" : "empty", recvCountsTensorOptional ? "non-empty" : "empty");
        return false;
    }
    if ((group == nullptr) || (strnlen(group, HCCL_GROUP_NAME_MAX) == 0)) { // HCCL_GROUP_NAME_MAX = 128U
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name is Empty.");
        return false;
    }
    return CheckMmOptionalConsistency(mmXOptional, mmWeightOptional, mmYOptional);
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
static bool CheckGmmNotEmpty(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *y)
{
    auto h1Val = gmmX->GetViewShape().GetDim(1);
    auto h1Val2 = gmmWeight->GetViewShape().GetDim(1);
    auto n1Val = gmmWeight->GetViewShape().GetDim(DIM_TWO);
    auto yMval = y->GetViewShape().GetDim(0);
    auto yNval = y->GetViewShape().GetDim(1);
    OP_API_CHECK((h1Val == 0), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmX is empty tensor with zero dimH1, which is unsupported.");
        return false;
    });
    OP_API_CHECK((h1Val2 == 0), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight is empty tensor with zero dimH1, which is unsupported.");
        return false;
    });
    OP_API_CHECK((n1Val == 0), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight is empty tensor with zero dimN1, which is unsupported.");
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
    return true;
}

static bool CheckMmNotEmptyOrAllEmpty(const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                                      const aclTensor *mmYOptional)
{
    if (mmXOptional == nullptr || mmWeightOptional == nullptr || mmYOptional == nullptr) {
        return true;
    }
    auto mmDim0 = mmXOptional->GetViewShape().GetDim(0);
    auto mmDim1 = mmXOptional->GetViewShape().GetDim(1);
    auto mmWdim0 = mmWeightOptional->GetViewShape().GetDim(0);
    auto mmWdim1 = mmWeightOptional->GetViewShape().GetDim(1);
    auto mmYdim0 = mmYOptional->GetViewShape().GetDim(0);
    auto mmYdim1 = mmYOptional->GetViewShape().GetDim(1);
    const bool mmXEmpty = (mmDim0 == 0 || mmDim1 == 0);
    const bool mmWEmpty = (mmWdim0 == 0 || mmWdim1 == 0);
    const bool mmYEmpty = (mmYdim0 == 0 || mmYdim1 == 0);
    const bool allEmpty = mmXEmpty && mmWEmpty && mmYEmpty;
    const bool allNonEmpty = !mmXEmpty && !mmWEmpty && !mmYEmpty;
    OP_API_CHECK((!allEmpty && !allNonEmpty), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "mmXOptional, mmWeightOptional, and mmYOptional must be either all empty or all non-empty,"
                "but got: mmXOptional shape is [%ld, %ld], mmWeightOptional shape is [%ld, %ld], mmYOptional shape "
                "is [%ld, %ld].",
                mmDim0, mmDim1, mmWdim0, mmWdim1, mmYdim0, mmYdim1);
        return false;
    });
    return true;
}

// 检查是否有空tensor
static bool CheckNotEmptyTensor(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *y,
                                const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                                const aclTensor *mmYOptional)
{
    CHECK_RET(CheckGmmNotEmpty(gmmX, gmmWeight, y), false);
    return CheckMmNotEmptyOrAllEmpty(mmXOptional, mmWeightOptional, mmYOptional);
}

static bool CheckRequiredScaleTensor(const aclTensor *xScale, const aclTensor *weightScale, const char *xName,
                                     const char *weightName, const char *modeName)
{
    if (xScale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%sScale must not be null when %s quant mode is used.", xName, modeName);
        return false;
    }
    if (weightScale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%sScale must not be null when %s quant mode is used.", weightName, modeName);
        return false;
    }
    return true;
}

static bool CheckUnsupportQuantMode(QuantModeType mode, const char *xName)
{
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s quantMode (%ld) is not supported yet.", xName, static_cast<int64_t>(mode));
    return false;
}

// 检查量化参数是否合法
static bool CheckQuantMode(int64_t xQuantMode, int64_t weightQuantMode, const aclTensor *xScaleOptional,
                           const aclTensor *weightScaleOptional, const aclTensor *x, const aclTensor *weight,
                           const aclTensor *y, const char *xName, const char *weightName)
{
    QuantModeType xMode = static_cast<QuantModeType>(xQuantMode);
    QuantModeType wMode = static_cast<QuantModeType>(weightQuantMode);
    if (xMode != wMode) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s QuantMode and %s QuantMode should be the same, but got %ld and %ld.",
                xName, weightName, static_cast<int64_t>(xMode), static_cast<int64_t>(wMode));
        return false;
    }
    // 按量化模式分支校验
    switch (xMode) {
        case QuantModeType::NO_QUANT:
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Current quant template does not support %s/%s in NO_QUANT mode.",
                    xName, weightName);
            return false;
        case QuantModeType::PERTENSOR_QUANT:
            return CheckRequiredScaleTensor(xScaleOptional, weightScaleOptional, xName, weightName, "PerTensor");
        case QuantModeType::MX_QUANT:
            return CheckRequiredScaleTensor(xScaleOptional, weightScaleOptional, xName, weightName, "MX");
        case QuantModeType::PERCHANNEL_QUANT:
        case QuantModeType::PERTOKEN_QUANT:
        case QuantModeType::PERGROUP_QUANT:
        case QuantModeType::PERBLOCK_QUANT:
        case QuantModeType::DYN_PERTOKEN_QUANT:
            return CheckUnsupportQuantMode(xMode, xName);
        default:
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unknown %s QuantMode: %ld.", xName, static_cast<int64_t>(xMode));
            return false;
    }
}

static bool CheckQuantParams(int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, const aclTensor *gmmX,
                             const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
                             const aclTensor *gmmWeightScaleOptional, const aclTensor *y, int64_t mmXQuantMode,
                             int64_t mmWeightQuantMode, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                             const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
                             const aclTensor *mmYOptional)
{
    // 1) gmm 一定要有量化模式，且当前只支持 TT(1) / MX(6)
    if (!CheckQuantMode(gmmXQuantMode, gmmWeightQuantMode, gmmXScaleOptional, gmmWeightScaleOptional, gmmX, gmmWeight,
                        y, "gmmX", "gmmWeight")) {
        return false;
    }
    // 2) mm 不存在时，允许没有量化模式
    if (mmXOptional == nullptr && mmWeightOptional == nullptr) {
        return true;
    }
    // 3) mm 存在时，mm 不能是非量化，且必须与 gmm 保持完全一致
    if (mmXQuantMode != gmmXQuantMode || mmWeightQuantMode != gmmWeightQuantMode) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When mm inputs are set, mm quant modes should be exactly the same as gmm quant modes. "
                "Expect (%ld, %ld), but got (%ld, %ld).",
                gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode);
        return false;
    }
    if (!CheckQuantMode(mmXQuantMode, mmWeightQuantMode, mmXScaleOptional, mmWeightScaleOptional, mmXOptional,
                        mmWeightOptional, mmYOptional, "mmX", "mmWeight")) {
        return false;
    }
    return true;
}

// 检查tensor最后两维是否转置（stride不连续）
static bool IsTransposeLastTwoDims(const aclTensor *tensor)
{
    if (tensor->GetViewShape().GetDimNum() < 2 || tensor->GetViewShape().GetDimNum() > 6) {
        return false;
    }
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
            return false;
        }
        return true;
    }
    return false;
}

// MX Scale Shape 校验：维度数 >= 3 且最后一维 == 2
static aclnnStatus CheckMxScaleShape(const aclTensor *scale, const char *name)
{
    if (scale == nullptr) {
        return ACLNN_SUCCESS;
    }
    uint64_t scaleDimNum = scale->GetViewShape().GetDimNum();
    if (scaleDimNum < DIM_THREE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In MX quant mode, %s dim num should be >= 3, but got %lu.", name,
                scaleDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t lastDim = scale->GetViewShape().GetDim(scaleDimNum - 1);
    if (lastDim != DIM_TWO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In MX quant mode, %s last dim should be 2, but got %ld.", name, lastDim);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// 交换 tensor view 的两个维度（shape + strides），不改变物理数据，基于 aclCreateTensor
static const aclTensor *SwapTensorDims(const aclTensor *tensor, uint64_t dimA, uint64_t dimB)
{
    uint64_t storageShapeDimNum = tensor->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = tensor->GetStorageShape().GetDim(i);
    }
    uint64_t viewShapeDimNum = tensor->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = tensor->GetViewShape().GetDim(i);
    }
    std::swap(viewDim[dimA], viewDim[dimB]);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(tensor, &dataType);
    auto origStride = tensor->GetViewStrides();
    std::vector<int64_t> stride(origStride.begin(), origStride.end());
    std::swap(stride[dimA], stride[dimB]);
    auto offset = tensor->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;
    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, tensor->GetTensor()->GetAddr());
}

// 检测 gmmWeight stride 转置，同时 reshape weight 和 scale（swap dim[1]/dim[2]）
static aclnnStatus HandleGmmMxTranspose(const aclTensor *&weight, const aclTensor *&scale, bool &transWeight)
{
    bool notContiguous = IsTransposeLastTwoDims(weight);
    OP_LOGD("gmmWeight notContiguous=%d, transWeight(before)=%d", notContiguous, transWeight);
    if (notContiguous && transWeight) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight not contiguous and transGmmWeight is already set!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (notContiguous && op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        transWeight = !transWeight;
        OP_LOGD("gmmWeight transposed detected: transWeight flipped to %d", transWeight);
        weight = SwapTensorDims(weight, 1, 2);
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (scale != nullptr) {
            scale = SwapTensorDims(scale, 1, 2);
            CHECK_RET(scale != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    return ACLNN_SUCCESS;
}

// 检测 mmWeight stride 转置，同时 reshape weight 和 scale（swap dim[0]/dim[1]）
static aclnnStatus HandleMmMxTranspose(const aclTensor *&weight, const aclTensor *&scale, bool &transWeight)
{
    bool notContiguous = IsTransposeLastTwoDims(weight);
    OP_LOGD("mmWeight notContiguous=%d, transWeight(before)=%d", notContiguous, transWeight);
    if (notContiguous && transWeight) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmWeight not contiguous and transMmWeight is already set!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (notContiguous && op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        transWeight = !transWeight;
        OP_LOGD("mmWeight transposed detected: transWeight flipped to %d", transWeight);
        weight = SwapTensorDims(weight, 0, 1);
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (scale != nullptr) {
            scale = SwapTensorDims(scale, 0, 1);
            CHECK_RET(scale != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    return ACLNN_SUCCESS;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
                               const aclTensor *gmmWeightScaleOptional, const aclTensor *sendCountsTensorOptional,
                               const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
                               const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                               const aclTensor *mmWeightScaleOptional, int64_t gmmXQuantMode,
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
                              mmWeightOptional, group, y, mmYOptional),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckNotEmptyTensor(gmmX, gmmWeight, y, mmXOptional, mmWeightOptional, mmYOptional),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckQuantParams(gmmXQuantMode, gmmWeightQuantMode, gmmX, gmmWeight, gmmXScaleOptional,
                               gmmWeightScaleOptional, y, mmXQuantMode, mmWeightQuantMode, mmXOptional,
                               mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, mmYOptional),
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
    const aclTensor *gmmWeightScaleOptional, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional, const aclTensor *commQuantScaleOptional,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
    int64_t commQuantMode, int64_t commQuantDtypeOptional, int64_t groupSize, const char *group, int64_t epWorldSize,
    const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight,
    const aclTensor *y, const aclTensor *mmYOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto retParam = CheckParams(gmmX, gmmWeight, gmmXScaleOptional, gmmWeightScaleOptional, sendCountsTensorOptional,
                                recvCountsTensorOptional, mmXOptional, mmWeightOptional, mmXScaleOptional,
                                mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
                                mmWeightQuantMode, commQuantMode, group, epWorldSize, sendCounts, recvCounts,
                                transGmmWeight, transMmWeight, y, mmYOptional, workspaceSize, executor);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    auto retSendAndRecv = CheckSendAndRecv(sendCounts, recvCounts);
    CHECK_RET(retSendAndRecv == ACLNN_SUCCESS, retSendAndRecv);
    char *strGroup = const_cast<char *>(group);
    int64_t yDtype = y->GetDataType();
    int64_t mmDtype = mmYOptional == nullptr ? 0 : mmYOptional->GetDataType();
    // MX 量化场景通过 stride 检测 weight/scale 的转置状态
    bool isMxQuant = (gmmXQuantMode == static_cast<int64_t>(QuantModeType::MX_QUANT));
    if (isMxQuant) {
        OP_LOGD("MX quant mode: transGmmWeight(input)=%d, transMmWeight(input)=%d", transGmmWeight, transMmWeight);
        // MX Scale Shape 校验
        auto scaleRet = CheckMxScaleShape(gmmWeightScaleOptional, "gmmWeightScale");
        CHECK_RET(scaleRet == ACLNN_SUCCESS, scaleRet);
        scaleRet = CheckMxScaleShape(mmWeightScaleOptional, "mmWeightScale");
        CHECK_RET(scaleRet == ACLNN_SUCCESS, scaleRet);
        // 检测 weight stride 转置，同时 reshape weight 和 scale
        auto transRet = HandleGmmMxTranspose(gmmWeight, gmmWeightScaleOptional, transGmmWeight);
        CHECK_RET(transRet == ACLNN_SUCCESS, transRet);
        if (mmWeightOptional != nullptr) {
            transRet = HandleMmMxTranspose(mmWeightOptional, mmWeightScaleOptional, transMmWeight);
            CHECK_RET(transRet == ACLNN_SUCCESS, transRet);
        }
        OP_LOGD("Final: transGmmWeight=%d, transMmWeight=%d", transGmmWeight, transMmWeight);
    }
    aclnnStatus ret = aclnnInnerQuantGroupedMatMulAlltoAllvGetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
        gmmXScaleOptional, gmmWeightScaleOptional, mmXScaleOptional, mmWeightScaleOptional, commQuantScaleOptional,
        strGroup, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, gmmXQuantMode, gmmWeightQuantMode,
        mmXQuantMode, mmWeightQuantMode, commQuantMode, groupSize, commQuantDtypeOptional, yDtype, mmDtype, y,
        mmYOptional, workspaceSize, executor);
    return ret;
}

extern "C" aclnnStatus aclnnQuantGroupedMatMulAlltoAllv(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    aclnnStatus ret = aclnnInnerQuantGroupedMatMulAlltoAllv(workspace, workspaceSize, executor, stream);
    return ret;
}
} // namespace
