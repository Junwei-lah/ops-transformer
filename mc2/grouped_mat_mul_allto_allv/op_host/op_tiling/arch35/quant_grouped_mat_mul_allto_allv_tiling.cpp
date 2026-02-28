/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_grouped_mat_mul_allto_allv_tiling.cpp
 * \brief
 */

#include "op_mc2.h"
#include "mc2_log.h"
#include "quant_grouped_mat_mul_allto_allv_tiling.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_adapter.h"
#include "tiling/mc2_tiling_utils.h"
#include <tiling/tiling_api.h>
#include <numeric>

using namespace Mc2Log;
using namespace AscendC;
using namespace optiling;
using namespace optiling::Mc2GroupedMatmul;

// namespace Mc2GroupedMatmul {

const std::vector<uint32_t> QUANT_GMM_X_DTYPE_LIST = {ge::DT_HIFLOAT8,};
const std::vector<uint32_t> QUANT_GMM_WEIGHT_DTYPE_LIST = {ge::DT_HIFLOAT8,};
const std::vector<uint32_t> QUANT_GMM_X_SCALE_DTYPE_LIST = {ge::DT_FLOAT,};
const std::vector<uint32_t> QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST = {ge::DT_FLOAT,};
const std::vector<uint32_t> QUANT_GMM_Y_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<int64_t> SUPPORT_RANK_SIZE{2, 4, 8, 16, 32, 64, 128, 256};
constexpr int64_t RANK_DEFAULT_NUM = -1;

static bool IsContains(const std::vector<uint32_t> &list, uint32_t value)
{
    return std::count(list.begin(), list.end(), value) > 0;
}

static ge::graphStatus CheckShapeDimensions(const gert::StorageShape *shape, uint64_t dims, const char *shapeName,
    const char *opName_)
{
    uint64_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((dimNum != dims),
        OP_LOGE(opName_, "The %s dimNum should be %lu, now is %lu.", shapeName, dims, dimNum), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::GetShapeAttrsInfo()
{
    // base check required para
    auto status = GmmAlltoAllvTilingBase::GetShapeAttrsInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK((opName_ == nullptr),
        OP_LOGE("quantGMMALLTOALLV", "The opName_ is null."), return ge::GRAPH_FAILED);
    
    localParams_.opName = opName_;
    return ge::GRAPH_SUCCESS;
}

// not support param
ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckOpInputSingleParamsTensorNotSupport()
{
    auto sendCountsTensorDesc = context_->GetOptionalInputDesc(SEND_COUNTS_TENSOR_OPTIONAL_INDEX);
    auto recvCountsTensorDesc = context_->GetOptionalInputDesc(RECV_COUNTS_TENSOR_OPTIONAL_INDEX);
    bool sendCountsTensorDescNotNull = sendCountsTensorDesc != nullptr;
    bool recvCountsTensorDescNotNull = recvCountsTensorDesc != nullptr;
    OP_TILING_CHECK(sendCountsTensorDescNotNull || recvCountsTensorDescNotNull,
        OP_LOGE(opName_, "sendCountsTensorNotNull = %d and recvCountsTensorNotNull = %d, should all be nullptr now!",
                sendCountsTensorDescNotNull, recvCountsTensorDescNotNull),
        return ge::GRAPH_FAILED);

    auto gmmXOffsetTensorDesc = context_->GetOptionalInputDesc(GMM_X_OFFSET_OPTIONAL_INDEX);
    auto gmmWeightOffsetTensorDesc = context_->GetOptionalInputDesc(GMM_WEIGHT_OFFSET_OPTIONAL_INDEX);
    bool gmmXOffsetTensorDescNotNull = gmmXOffsetTensorDesc != nullptr;
    bool gmmWeightOffsetTensorDescNotNull = gmmWeightOffsetTensorDesc != nullptr;
    OP_TILING_CHECK(gmmXOffsetTensorDescNotNull || gmmWeightOffsetTensorDescNotNull,
        OP_LOGE(opName_, "gmmXOffsetTensorNotNull=%d and gmmWeightOffsetTensorNotNull=%d, should all be nullptr now!",
                gmmXOffsetTensorDescNotNull, gmmWeightOffsetTensorDescNotNull),
        return ge::GRAPH_FAILED);

    auto mmXOffsetTensorDesc = context_->GetOptionalInputDesc(MM_X_OFFSET_OPTIONAL_INDEX);
    auto mmWeightOffsetTensorDesc = context_->GetOptionalInputDesc(MM_WEIGHT_OFFSET_OPTIONAL_INDEX);
    bool mmXOffsetTensorDescNotNull = mmXOffsetTensorDesc != nullptr;
    bool mmWeightOffsetTensorDescNotNull = mmWeightOffsetTensorDesc != nullptr;
    OP_TILING_CHECK(mmXOffsetTensorDescNotNull || mmWeightOffsetTensorDescNotNull,
        OP_LOGE(opName_, "mmXOffsetTensorNotNull=%d and mmWeightOffsetTensorNotNull=%d, should all be nullptr now!",
                mmXOffsetTensorDescNotNull, mmWeightOffsetTensorDescNotNull),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// quant must support param
ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckOpInputSingleParamsTensorSupport()
{
    auto gmmXScaleTensorDesc = context_->GetOptionalInputDesc(GMM_X_SCALE_OPTIONAL_INDEX);
    auto gmmWeightScaleTensorDesc = context_->GetOptionalInputDesc(GMM_WEIGHT_SCALE_OPTIONAL_INDEX);
    bool gmmXScaleTensorDescNull = gmmXScaleTensorDesc == nullptr;
    bool gmmWeightScaleTensorDescNull = gmmWeightScaleTensorDesc == nullptr;
    OP_TILING_CHECK(gmmXScaleTensorDescNull || gmmWeightScaleTensorDescNull,
        OP_LOGE(opName_, "gmmXScaleTensorNull=%d and gmmWeightScaleTensorNull=%d, should all not be nullptr!",
                gmmXScaleTensorDescNull, gmmWeightScaleTensorDescNull),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckOpInputSingleParamsTensorMM()
{
    auto mmXTensorShape = context_->GetOptionalInputShape(MM_X_OPTIONAL_INDEX);
    auto mmWeightTensorShape = context_->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX);
    auto mmYShape = context_->GetOutputShape(OUTPUT_MM_Y_OPTIONAL_INDEX);

    bool isMmXNull = (mmXTensorShape == nullptr);
    bool isMmWeightNull = (mmWeightTensorShape == nullptr);
    bool isMmYNull = (mmYShape == nullptr);

    bool isAllSame = (isMmXNull == isMmWeightNull) && (isMmWeightNull == isMmYNull);
    OP_TILING_CHECK(!isAllSame, 
                    OP_LOGE(opName_, "mmXTensor, mmWeightTensor, mmYTensor must exist or not exist at same time."),
                    return ge::GRAPH_FAILED);
    if (!isMmXNull) {
        localParams_.hasSharedMm = true;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckOpInputSingleParamsTensor()
{
    auto status = CheckOpInputSingleParamsTensorNotSupport();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckOpInputSingleParamsTensorSupport();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckOpInputSingleParamsTensorMM();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckAndSetLocalParamsGmm()
{
    localParams_.gmmXDtype = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    localParams_.gmmWeightDtype = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(QUANT_GMM_X_DTYPE_LIST, localParams_.gmmXDtype),
        OP_LOGE(opName_, "The Input gmmX Dtype should be in (hifloat8, ), but gmmX is %s.",
        Ops::Base::ToString(localParams_.gmmXDtype).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(QUANT_GMM_WEIGHT_DTYPE_LIST, localParams_.gmmWeightDtype),
        OP_LOGE(opName_, "The Input gmmWeight Dtype should be in (hifloat8, ), but gmmWeight is %s.",
        Ops::Base::ToString(localParams_.gmmWeightDtype).c_str()), return ge::GRAPH_FAILED);
    localParams_.yDtype = context_->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(QUANT_GMM_Y_DTYPE_LIST, localParams_.yDtype),
        OP_LOGE(opName_, "The Output y Dtype should be in (fp16, bf16), but y Dtype is %s.",
        Ops::Base::ToString(localParams_.yDtype).c_str()), return ge::GRAPH_FAILED);
    localParams_.gmmYDtype = localParams_.yDtype;
    
    const gert::StorageShape* gmmXStorageShape = context_->GetInputShape(GMM_X_INDEX);
    const gert::StorageShape* gmmWeightStorageShape = context_->GetInputShape(GMM_WEIGHT_INDEX);
    const gert::StorageShape* yStorageShape = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(gmmXStorageShape == nullptr, OP_LOGE(opName_, "gmmXStorageShape is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmWeightStorageShape == nullptr, OP_LOGE(opName_, "gmmWeightStorageShape is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE(opName_, "yStorageShape is null!"),
        return ge::GRAPH_FAILED);
    auto status = CheckShapeDimensions(gmmXStorageShape, DIM_TWO, "gmmXShape", opName_);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(gmmWeightStorageShape, DIM_THREE, "gmmWeightShape", opName_);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(yStorageShape, DIM_TWO, "yShape", opName_);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    localParams_.A = gmmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.H1 = gmmXStorageShape->GetStorageShape().GetDim(DIM_ONE);

    localParams_.ep = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.gmmWeightDim1 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);
    localParams_.gmmWeightDim2 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_TWO);

    localParams_.BsK = yStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.N1 = yStorageShape->GetStorageShape().GetDim(DIM_ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckAndSetLocalParamsMm()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    localParams_.mmXDtype = context_->GetOptionalInputDesc(MM_X_OPTIONAL_INDEX)->GetDataType();
    localParams_.mmWeightDtype = context_->GetOptionalInputDesc(MM_WEIGHT_OPTIONAL_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(QUANT_GMM_X_DTYPE_LIST, localParams_.mmXDtype),
        OP_LOGE(opName_, "The Input mmX Dtype should be in (hifloat8, ), but mmX is %s.",
        Ops::Base::ToString(localParams_.mmXDtype).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(QUANT_GMM_WEIGHT_DTYPE_LIST, localParams_.mmWeightDtype),
        OP_LOGE(opName_, "The Input mmWeight Dtype should be in (hifloat8, ), but mmWeight is %s.",
        Ops::Base::ToString(localParams_.mmWeightDtype).c_str()), return ge::GRAPH_FAILED);
    localParams_.mmYDtype = context_->GetOutputDesc(OUTPUT_MM_Y_OPTIONAL_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(QUANT_GMM_Y_DTYPE_LIST, localParams_.mmYDtype),
        OP_LOGE(opName_, "The Output mmY Dtype should be in (fp16, bf16), but mmY is %s.",
        Ops::Base::ToString(localParams_.mmYDtype).c_str()), return ge::GRAPH_FAILED);
    
    const gert::StorageShape* mmXStorageShape = context_->GetOptionalInputShape(MM_X_OPTIONAL_INDEX);
    const gert::StorageShape* mmWeightStorageShape = context_->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX);
    const gert::StorageShape* mmYStorageShape = context_->GetOutputShape(OUTPUT_MM_Y_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXStorageShape == nullptr, OP_LOGE(opName_, "mmXStorageShape is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmWeightStorageShape == nullptr, OP_LOGE(opName_, "mmWeightStorageShape is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmYStorageShape == nullptr, OP_LOGE(opName_, "mmYStorageShape is null!"),
        return ge::GRAPH_FAILED);
    auto status = CheckShapeDimensions(mmXStorageShape, DIM_TWO, "mmXShape", opName_);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(mmWeightStorageShape, DIM_TWO, "mmWeightShape", opName_);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(mmYStorageShape, DIM_TWO, "mmYShape", opName_);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    localParams_.Bs = mmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.H2 = mmXStorageShape->GetStorageShape().GetDim(DIM_ONE);

    localParams_.mmWeightDim0 = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.mmWeightDim1 = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);

    uint64_t mmYDim0 = mmYStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(localParams_.Bs != mmYDim0,
        OP_LOGE(opName_, "mmX DIM0 %lu and mmY DIM0 %lu is not valid!", localParams_.Bs, mmYDim0),
        return ge::GRAPH_FAILED);

    localParams_.N2 = mmYStorageShape->GetStorageShape().GetDim(DIM_ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckAndSetLocalParamsAttr()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(opName_, "Failed to get attrs."), return ge::GRAPH_FAILED);

    auto gmmXQuantModeptr = attrs->GetAttrPointer<int64_t>(ATTR_GMM_X_QUANT_MODE_INDEX);
    OP_TILING_CHECK(gmmXQuantModeptr == nullptr, OP_LOGE(opName_, "gmmXQuantModeptr is null."), return ge::GRAPH_FAILED);
    localParams_.gmmXQuantMode = *gmmXQuantModeptr;
    auto gmmWeightQuantModeptr = attrs->GetAttrPointer<int64_t>(ATTR_GMM_WEIGHT_QUANT_MODE_INDEX);
    OP_TILING_CHECK(gmmWeightQuantModeptr == nullptr, OP_LOGE(opName_, "gmmWeightQuantModeptr is null."), return ge::GRAPH_FAILED);
    localParams_.gmmWeightQuantMode = *gmmWeightQuantModeptr;
    auto gmmTransWeightptr = attrs->GetAttrPointer<bool>(ATTR_TRANS_GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(gmmTransWeightptr == nullptr, OP_LOGE(opName_, "gmmTransWeightptr is null."), return ge::GRAPH_FAILED);
    localParams_.isGmmWeightTrans = *gmmTransWeightptr;

    auto commQuantModeptr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_QUANT_MODE_INDEX);
    OP_TILING_CHECK(commQuantModeptr == nullptr, OP_LOGE(opName_, "commQuantModeptr is null."), return ge::GRAPH_FAILED);
    localParams_.commQuantMode = *commQuantModeptr;
    auto commQuantDtypeptr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_QUANT_DTYPE_INDEX);
    OP_TILING_CHECK(commQuantDtypeptr == nullptr, OP_LOGE(opName_, "commQuantDtypeptr is null."), return ge::GRAPH_FAILED);
    localParams_.commQuantDtype = *commQuantDtypeptr;
    OP_TILING_CHECK(localParams_.commQuantMode != QUANT_NONE,
        OP_LOGE(opName_, "not support commQuant now, but commQuantMode is %ld !", localParams_.commQuantMode),
        return ge::GRAPH_FAILED);

    auto mmXQuantModeptr = attrs->GetAttrPointer<int64_t>(ATTR_MM_X_QUANT_MODE_INDEX);
    OP_TILING_CHECK(mmXQuantModeptr == nullptr, OP_LOGE(opName_, "mmXQuantModeptr is null."), return ge::GRAPH_FAILED);
    localParams_.mmXQuantMode = *mmXQuantModeptr;
    auto mmWeightQuantModeptr = attrs->GetAttrPointer<int64_t>(ATTR_MM_WEIGHT_QUANT_MODE_INDEX);
    OP_TILING_CHECK(mmWeightQuantModeptr == nullptr, OP_LOGE(opName_, "mmWeightQuantModeptr is null."), return ge::GRAPH_FAILED);
    localParams_.mmWeightQuantMode = *mmWeightQuantModeptr;
    auto mmTransWeightptr = attrs->GetAttrPointer<bool>(ATTR_TRANS_MM_WEIGHT_INDEX);
    OP_TILING_CHECK(mmTransWeightptr == nullptr, OP_LOGE(opName_, "mmTransWeightptr is null."), return ge::GRAPH_FAILED);
    localParams_.isMmWeightTrans = *mmTransWeightptr;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckFormat()
{
    OP_LOGD(opName_, "start CheckFormat.");
    OP_TILING_CHECK(context_->GetInputDesc(GMM_X_INDEX)->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE(opName_, "gmmX storage format should be ND."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE(opName_, "gmmWeight storage format should be ND."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(GMM_X_SCALE_OPTIONAL_INDEX)->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE(opName_, "gmmXScale storage format should be ND."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(GMM_WEIGHT_SCALE_OPTIONAL_INDEX)->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE(opName_, "gmmWeightScale storage format should be ND."), return ge::GRAPH_FAILED);
    auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(yDesc == nullptr,
        OP_LOGE(opName_, "y tensor desc can not be null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE(opName_, "y storage format should be ND."), return ge::GRAPH_FAILED);
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    // 即使传入nullptr，GetOptionalInputDesc接口也有可能拿到非nullptr的地址？？？
    auto mmXDesc = context_->GetOptionalInputDesc(MM_X_OPTIONAL_INDEX);
    if (mmXDesc != nullptr) {
        OP_TILING_CHECK(mmXDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
            OP_LOGE(opName_, "mmX storage format should be ND."), return ge::GRAPH_FAILED);
    }
    auto mmWeightDesc = context_->GetOptionalInputDesc(MM_WEIGHT_OPTIONAL_INDEX);
    if (mmWeightDesc != nullptr) {
        OP_TILING_CHECK(mmWeightDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
            OP_LOGE(opName_, "mmWeight storage format should be ND."), return ge::GRAPH_FAILED);
    }
    auto mmXScaleDesc = context_->GetOptionalInputDesc(MM_X_SCALE_OPTIONAL_INDEX);
    if (mmXScaleDesc != nullptr) {
        OP_TILING_CHECK(mmXScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
            OP_LOGE(opName_, "mmXScale storage format should be ND."), return ge::GRAPH_FAILED);
    }
    auto mmWeightScaleDesc = context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_OPTIONAL_INDEX);
    if (mmWeightScaleDesc != nullptr) {
        OP_TILING_CHECK(mmWeightScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
            OP_LOGE(opName_, "mmWeightScale storage format should be ND."), return ge::GRAPH_FAILED);
    }
    auto mmYDesc = context_->GetOutputDesc(OUTPUT_MM_Y_OPTIONAL_INDEX);
    if (mmYDesc != nullptr) {
        OP_TILING_CHECK(mmYDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
            OP_LOGE(opName_, "mmY storage format should be ND."), return ge::GRAPH_FAILED);
    }
    OP_LOGD(opName_, "end CheckFormat.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckAndSetLocalParams()
{
    auto status = CheckAndSetLocalParamsGmm();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckAndSetLocalParamsMm();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckAndSetLocalParamsAttr();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckFormat();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckParamsRelationGmm()
{
    localParams_.gmmXScaleDtype = context_->GetOptionalInputDesc(GMM_X_SCALE_OPTIONAL_INDEX)->GetDataType();
    localParams_.gmmWeightScaleDtype = context_->GetOptionalInputDesc(GMM_WEIGHT_SCALE_OPTIONAL_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(QUANT_GMM_X_SCALE_DTYPE_LIST, localParams_.gmmXScaleDtype),
        OP_LOGE(opName_, "The Input gmmX Scale Dtype should be in (float32, ), but Scale is %s.",
        Ops::Base::ToString(localParams_.gmmXScaleDtype).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST, localParams_.gmmWeightScaleDtype),
        OP_LOGE(opName_, "The Input gmmWeight Scale Dtype should be in (float32, ), but Scale is %s.",
        Ops::Base::ToString(localParams_.gmmWeightScaleDtype).c_str()), return ge::GRAPH_FAILED);

    const gert::StorageShape* gmmXScaleStorageShape = context_->GetOptionalInputShape(GMM_X_SCALE_OPTIONAL_INDEX);
    const gert::StorageShape* gmmWeightScaleStorageShape = context_->GetOptionalInputShape(GMM_WEIGHT_SCALE_OPTIONAL_INDEX);
    OP_TILING_CHECK(gmmXScaleStorageShape == nullptr, OP_LOGE(opName_, "gmmXScaleStorageShape is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmWeightScaleStorageShape == nullptr, OP_LOGE(opName_, "gmmWeightScaleStorageShape is null!"),
        return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(localParams_.gmmXQuantMode != QUANT_PERTENSOR,
        OP_LOGE(opName_, "gmmXQuantMode just support tensor mode now, mode is %ld !", localParams_.gmmXQuantMode),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localParams_.gmmWeightQuantMode != QUANT_PERTENSOR,
        OP_LOGE(opName_, "gmmWeightQuantMode just support tensor mode now, mode is %ld !", localParams_.gmmWeightQuantMode),
        return ge::GRAPH_FAILED);
    ge::graphStatus status = CheckShapeDimensions(gmmXScaleStorageShape, DIM_ONE, "gmmXScaleShape", opName_);
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(gmmWeightScaleStorageShape, DIM_ONE, "gmmWeightScaleShape", opName_);
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    localParams_.gmmQuantSuit = QUANT_PAIR_TT;
    if (localParams_.isGmmWeightTrans) {
        OP_TILING_CHECK(localParams_.H1 != localParams_.gmmWeightDim2,
            OP_LOGE(opName_, "gmmX shape K %lu not match gmmWeight shape K %lu !", localParams_.H1, localParams_.gmmWeightDim2),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N1 != localParams_.gmmWeightDim1,
            OP_LOGE(opName_, "y shape N %lu not match gmmWeight shape N %lu !", localParams_.N1, localParams_.gmmWeightDim1),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(localParams_.H1 != localParams_.gmmWeightDim1,
            OP_LOGE(opName_, "gmmX shape %lu not match gmmWeight shape %lu !", localParams_.H1, localParams_.gmmWeightDim1),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N1 != localParams_.gmmWeightDim2,
            OP_LOGE(opName_, "y shape N %lu not match gmmWeight shape N %lu !", localParams_.N1, localParams_.gmmWeightDim2),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckParamsRelationMm()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    localParams_.mmXScaleDtype = context_->GetOptionalInputDesc(MM_X_SCALE_OPTIONAL_INDEX)->GetDataType();
    localParams_.mmWeightScaleDtype = context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_OPTIONAL_INDEX)->GetDataType();
    
    OP_TILING_CHECK(!IsContains(QUANT_GMM_X_SCALE_DTYPE_LIST, localParams_.mmXScaleDtype),
        OP_LOGE(opName_, "The Input mmX Scale Dtype should be in (float32, ), but Scale is %s.",
        Ops::Base::ToString(localParams_.mmXScaleDtype).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST, localParams_.mmWeightScaleDtype),
        OP_LOGE(opName_, "The Input mmWeight Scale Dtype should be in (float32, ), but Scale is %s.",
        Ops::Base::ToString(localParams_.mmWeightScaleDtype).c_str()), return ge::GRAPH_FAILED);

    const gert::StorageShape* mmXScaleStorageShape = context_->GetOptionalInputShape(MM_X_SCALE_OPTIONAL_INDEX);
    const gert::StorageShape* mmWeightScaleStorageShape = context_->GetOptionalInputShape(MM_WEIGHT_SCALE_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXScaleStorageShape == nullptr, OP_LOGE(opName_, "mmXScaleStorageShape is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmWeightScaleStorageShape == nullptr, OP_LOGE(opName_, "mmWeightScaleStorageShape is null!"),
        return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(localParams_.mmXQuantMode != QUANT_PERTENSOR,
        OP_LOGE(opName_, "mmXQuantMode just support tensor mode now, mode is %ld !", localParams_.mmXQuantMode),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(localParams_.mmWeightQuantMode != QUANT_PERTENSOR,
        OP_LOGE(opName_, "mmWeightQuantMode just support tensor mode now, mode is %ld !", localParams_.mmWeightQuantMode),
        return ge::GRAPH_FAILED);
    ge::graphStatus status = CheckShapeDimensions(mmXScaleStorageShape, DIM_ONE, "mmXScaleShape", opName_);
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(mmWeightScaleStorageShape, DIM_ONE, "mmWeightScaleShape", opName_);
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    localParams_.mmQuantSuit = QUANT_PAIR_TT;
    if (localParams_.isMmWeightTrans) {
        OP_TILING_CHECK(localParams_.H2 != localParams_.mmWeightDim1,
            OP_LOGE(opName_, "mmX shape %lu not match mmWeight shape %lu !", localParams_.H2, localParams_.mmWeightDim1),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N2 != localParams_.mmWeightDim0,
            OP_LOGE(opName_, "mmY shape N %lu not match mmWeight shape N %lu !", localParams_.N2, localParams_.mmWeightDim0),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(localParams_.H2 != localParams_.mmWeightDim0,
            OP_LOGE(opName_, "mmX shape %lu not match mmWeight shape %lu !", localParams_.H2, localParams_.mmWeightDim0),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N2 != localParams_.mmWeightDim1,
            OP_LOGE(opName_, "mmY shape N %lu not match mmWeight shape N %lu !", localParams_.N2, localParams_.mmWeightDim1),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckParamsAttrEpAndSetLocalParams()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(group == nullptr, OP_LOGE(opName_, "group is null."), return ge::GRAPH_FAILED);
    int64_t rankDim = 0;
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(opName_, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    if (*epWorldSizePtr == RANK_DEFAULT_NUM) {
        OP_TILING_CHECK(!mc2tiling::GetRankSize(opName_, group, rankDim), OP_LOGE(opName_, "GetRankSize failed."),
                        return ge::GRAPH_FAILED);
    } else {
        rankDim = *epWorldSizePtr;
    }
    std::string supportRankSizeRange;
    for (const auto& v : SUPPORT_RANK_SIZE) {
        supportRankSizeRange += (std::to_string(v) + " ");
    }
    OP_TILING_CHECK(SUPPORT_RANK_SIZE.find(rankDim) == SUPPORT_RANK_SIZE.end(),
        OP_LOGE(opName_, "World_size should be %s, but the actual value is %ld.", supportRankSizeRange, rankDim),
        return ge::GRAPH_FAILED);
    localParams_.epWorldSize = rankDim;

    bool isEpNumMatch = (localParams_.ep > 0) && (localParams_.ep < 33);
    if (isEpNumMatch) {
        uint64_t expertNum = localParams_.ep * rankDim;
        OP_TILING_CHECK(expertNum > MAX_EXPERT_NUM,
            OP_LOGE(opName_, "expert Num lager than MAX_EXPERT_NUM, expertNum is %lu !", expertNum),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(opName_, "expert Per Rank is not match range, expertNum is %lu !", localParams_.ep);
        return ge::GRAPH_FAILED;
    }

    auto groupSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_SIZE_OPTIONAL_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr, OP_LOGE(opName_, "groupSizePtr is null !"), return ge::GRAPH_FAILED);
    localParams_.groupSize = *groupSizePtr;
    OP_TILING_CHECK(localParams_.groupSize != 0,
        OP_LOGE(opName_, "not support group quant now, but groupSize is %ld !", localParams_.groupSize),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckAndSetSendRecvCountsAttr()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    uint64_t expertNum = localParams_.ep * localParams_.epWorldSize;
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr == nullptr, OP_LOGE(opName_, "sendCountsPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvCountsPtr == nullptr, OP_LOGE(opName_, "recvCountsPtr is null."), return ge::GRAPH_FAILED);
    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    uint64_t recvCountsSize = recvCountsPtr->GetSize();
    OP_TILING_CHECK(sendCountsSize != recvCountsSize,
        OP_LOGE(opName_, "sendCountSize %lu should be equal to recvCountSize %lu !", sendCountsSize, recvCountsSize),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sendCountsSize != expertNum,
        OP_LOGE(opName_, "sendCountSize %lu should be equal to expertTotalNums %lu !", sendCountsSize, expertNum),
        return ge::GRAPH_FAILED);

    const int64_t* sendCounts = static_cast<const int64_t*>(sendCountsPtr->GetData());
    const int64_t* recvCounts = static_cast<const int64_t*>(recvCountsPtr->GetData());
    uint64_t sendCountsSum = std::accumulate(sendCounts, sendCounts + sendCountsSize, 0ULL);
    OP_TILING_CHECK(sendCountsSum != localParams_.A,
        OP_LOGE(opName_, "sendCountsSum %lu should be equal to A %lu !", sendCountsSum, localParams_.A),
        return ge::GRAPH_FAILED);
    uint64_t recvCountsSum = std::accumulate(recvCounts, recvCounts + recvCountsSize, 0ULL);
    OP_TILING_CHECK(recvCountsSum != localParams_.BsK,
        OP_LOGE(opName_, "recvCountsSum %lu should be equal to BsK %lu !", recvCountsSum, localParams_.BsK),
        return ge::GRAPH_FAILED);

    auto gmmQTilingCommonInfoPtr = &localTilingData_.taskTilingInfo;
    uint64_t maxCountsSize = std::min<uint64_t>(expertNum, MAX_EXPERT_NUM);
    for (uint64_t i = 0; i < maxCountsSize; i++) {
        OP_TILING_CHECK(sendCounts[i] < 0,
            OP_LOGE(opName_, "sendCounts value %ld should not be < 0 !", sendCounts[i]), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(recvCounts[i] < 0,
            OP_LOGE(opName_, "recvCounts value %ld should not be < 0 !", recvCounts[i]), return ge::GRAPH_FAILED);
        gmmQTilingCommonInfoPtr->sendCnt[i] = static_cast<int32_t>(sendCounts[i]);
        gmmQTilingCommonInfoPtr->recvCnt[i] = static_cast<int32_t>(recvCounts[i]);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckLocalParams()
{
    OP_TILING_CHECK((localParams_.H1 == 0) || (localParams_.H1 >= MAX_H1_VALUE),
        OP_LOGE(opName_, "H1 should be less than %lu, but got %lu.", MAX_H1_VALUE, localParams_.H1),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((localParams_.BsK >= MAX_BSK_VALUE),
        OP_LOGE(opName_, "BsK should be less than %lu, but got %lu.", MAX_BSK_VALUE, localParams_.BsK),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((localParams_.N1 == 0) || (localParams_.N1 >= MAX_N1_VALUE),
        OP_LOGE(opName_, "N1 should be less than %lu, but got %lu.", MAX_N1_VALUE, localParams_.N1),
        return ge::GRAPH_FAILED);

    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK((localParams_.Bs == 0) || (localParams_.BsK % localParams_.Bs != 0),
        OP_LOGE(opName_, "BSK should be divisible by BS, got BSK[%lu] and BS[%lu].", localParams_.BsK,localParams_.Bs),
        return ge::GRAPH_FAILED);
    uint64_t k = localParams_.BsK / localParams_.Bs;
    OP_TILING_CHECK((k < MIN_K_VALUE) || (k > MAX_K_VALUE),
        OP_LOGE(opName_, "K should be in [%lu, %lu], but got %lu.", MIN_K_VALUE, MAX_K_VALUE, k),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((localParams_.H2 == 0) || (localParams_.H2 > MAX_SHARED_H_SHAPE_SIZE),
        OP_LOGE(opName_, "H2 should be less than %lu, but got %lu.", MAX_SHARED_H_SHAPE_SIZE, localParams_.H2),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((localParams_.N2 == 0) || (localParams_.N2 > MAX_N2_VALUE),
        OP_LOGE(opName_, "N2 should be less than %lu, but got %lu.", MAX_N2_VALUE, localParams_.N2),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckParamsRelationAndSetLocalParams()
{
    auto status = CheckParamsRelationGmm();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckParamsRelationMm();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckParamsAttrEpAndSetLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckAndSetSendRecvCountsAttr();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::GetPlatformInfo()
{
    fe::PlatFormInfos *platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "Fail to get platform info."), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    localParams_.aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    localParams_.aicCoreNum = ascendcPlatform.GetCoreNumAic();
    return ge::GRAPH_SUCCESS;
};

bool QuantGroupedMatmulAllToAllvTiling::IsCapable()
{
    QuantModePair mode = GetQuantMode(context_, opName_);
    OP_TILING_CHECK(mode == QUANT_PAIR_ERROR, OP_LOGE(opName_, "Fail to get attr quant mode."), return false);
    if (mode == QUANT_PAIR_TT) {
        OP_LOGI(opName_, "QuantGroupedMatmulAllToAllvTiling TT mode capable.");
        return true;
    }
    OP_LOGI(opName_, "Skip QuantGroupedMatmulAllToAllvTiling TT.");
    return false;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckAndSetInputOutputInfo()
{
    auto status = CheckOpInputSingleParamsTensor();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckAndSetLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckParamsRelationAndSetLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::SetTilingCommonInfo()
{
    auto gmmQTilingCommonInfoPtr = &localTilingData_.taskTilingInfo;
    gmmQTilingCommonInfoPtr->BSK = localParams_.BsK;
    gmmQTilingCommonInfoPtr->BS = localParams_.Bs;
    gmmQTilingCommonInfoPtr->H1 = localParams_.H1;
    gmmQTilingCommonInfoPtr->H2 = localParams_.H2;
    gmmQTilingCommonInfoPtr->A = localParams_.A;
    gmmQTilingCommonInfoPtr->N1 = localParams_.N1;
    gmmQTilingCommonInfoPtr->N2 = localParams_.N2;
    gmmQTilingCommonInfoPtr->epWorldSize = localParams_.epWorldSize;
    gmmQTilingCommonInfoPtr->e = localParams_.ep;

    gmmQTilingCommonInfoPtr->mainLoopExpertNum = 1;
    gmmQTilingCommonInfoPtr->tailLoopExpertNum = 1;
    gmmQTilingCommonInfoPtr->totalLoopCount = localParams_.ep;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::SetGmmA2avWorkspaceInfo()
{
    constexpr uint64_t alignAddrLen = 512;
    auto gmmYDtypeSize = mc2tiling::GetDataTypeSize(opName_, localParams_.gmmYDtype);
    inferredInfo_.gmmResultLen = mc2tiling::AlignUp(
        localParams_.A * localParams_.N1 * gmmYDtypeSize, alignAddrLen);
    localTilingData_.workspaceInfo.wsGmmOutputSize = inferredInfo_.gmmResultLen;
    localTilingData_.workspaceInfo.wsGmmComputeWorkspaceSize = 1 * 1024 * 1024;
    localTilingData_.workspaceInfo.wsSharedGmmComputeWorkspaceSize = 1 * 1024 * 1024;
    workSpaceSize_ = libApiWorkSpaceSize_ + inferredInfo_.gmmResultLen +
        localTilingData_.workspaceInfo.wsGmmComputeWorkspaceSize +
        localTilingData_.workspaceInfo.wsSharedGmmComputeWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::DoQuantGMMTiling()
{
    // 设置公共信息
    QuantGroupedMatmulAllToAllvAdapter gmmTile(context_);
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.SetCommonInputParams(localParams_));
    // GMM 第一个矩阵块
    uint64_t gmmX_epSize = 0;
    for (uint64_t i = 0; i < localParams_.epWorldSize; i++) {
        gmmX_epSize += localTilingData_.taskTilingInfo.sendCnt[i*localParams_.ep];
    }
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.SetGroupExpertInputParameters(localParams_, gmmX_epSize));
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.Process());
    localTilingData_.gmmBaseTiling = gmmTile.GetGmmQuantTilingAdapterData();

    // SharedMM
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.SetSharedExpertInputParameters(localParams_));
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.Process());
    localTilingData_.sharedGmmTiling = gmmTile.GetGmmQuantTilingAdapterData();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::SetHcclTiling()
{
    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";

    auto attrs = context_->GetAttrs();
    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);

    const uint32_t alltoAllvReduceType = 0u;
    auto outputDataType = context_->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(
        mc2tiling::HCCL_DATA_TYPE.find(outputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
        OP_LOGE(opName_, "%s is Unsupported outputdata type!", Ops::Base::ToString(outputDataType).c_str()),
        return ge::GRAPH_FAILED);

    auto alltoAllvDstDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);
    auto alltoAllvSrcDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);

    Mc2CcTilingConfig hcclCcTilingConfig(groupEpPtr, alltoAllvCmd, alltoAllvConfig, 
                                         alltoAllvReduceType, alltoAllvDstDataType, alltoAllvSrcDataType);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(localTilingData_.hcclA2avTiling.hcclInitTiling) != 0,
        OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling hcclInitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(localTilingData_.hcclA2avTiling.a2avCcTiling) != 0,
        OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling alltoAllvCcTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::DoOpTiling()
{
    // 输入参数的校验:Attrs,Dtype,Shape等
    GE_ASSERT_GRAPH_SUCCESS(CheckAndSetInputOutputInfo());

    GE_ASSERT_GRAPH_SUCCESS(SetTilingCommonInfo());
    // 调用量化Matmul的tiling方法进行切分
    GE_ASSERT_GRAPH_SUCCESS(DoQuantGMMTiling());
    // hccl的tiling参数赋值处理
    GE_ASSERT_GRAPH_SUCCESS(SetHcclTiling());
    GE_ASSERT_GRAPH_SUCCESS(SetGmmA2avWorkspaceInfo());
    return ge::GRAPH_SUCCESS;
}

void PrintGmmA2avWorkspaceInfo(const GmmA2avWorkspaceInfo &workspaceInfo, const char *opName_)
{
    std::stringstream ss;
    ss << "workspaceInfo: ";
    ss << "wsGmmOutputSize=" << workspaceInfo.wsGmmOutputSize <<", wsGmmComputeWorkspaceSize=" <<
        workspaceInfo.wsGmmComputeWorkspaceSize << ", wsSharedGmmComputeWorkspaceSize=" <<
        workspaceInfo.wsSharedGmmComputeWorkspaceSize;
    OP_LOGD(opName_, "%s", ss.str().c_str());
}

void PrintTaskTilingInfo(const MC2KernelTemplate::TaskTilingInfo &taskTilingInfo,
    const QuantGmmAlltoAllvParamsInfo &localParams, const char *opName_)
{
    std::stringstream ss;
    ss << "TaskTilingInfo: ";
    ss << "BSK=" << taskTilingInfo.BSK << ", BS=" << taskTilingInfo.BS << ", H1=" << taskTilingInfo.H1 << ", H2=" <<
        taskTilingInfo.H2 << ", A=" << taskTilingInfo.A << ", N1=" << taskTilingInfo.N1 << ", N2=" << taskTilingInfo.N2;
    ss << ", epWorldSize=" << taskTilingInfo.epWorldSize << ", e=" << taskTilingInfo.e;
    ss << ", mainLoopExpertNum=" << taskTilingInfo.mainLoopExpertNum << ", tailLoopExpertNum=" <<
        taskTilingInfo.tailLoopExpertNum << ", totalLoopCount=" << taskTilingInfo.totalLoopCount;
    ss << "\nSendCounts: ";
    for (int64_t i = 0; i < localParams.ep * localParams.epWorldSize; i++) {
        if (taskTilingInfo.sendCnt[i] != 0) {
            if (i != 0) {
                ss << " ,";
            }
            ss << taskTilingInfo.sendCnt[i];
        }
    }
    ss << "\nRecvCounts: ";
    for (int64_t i = 0; i < localParams.ep * localParams.epWorldSize; i++) {
        if (taskTilingInfo.recvCnt[i] != 0) {
            if (i != 0) {
                ss << " ,";
            }
            ss << taskTilingInfo.recvCnt[i];
        }
    }
    OP_LOGI(opName_, "%s", ss.str().c_str());
}

void PrintGMMQuantTilingData(const MC2KernelTemplate::GMMQuantTilingData &data, const char *opName_)
{
    const auto &mm = data.mmTilingData;
    const auto &quantParams = data.gmmQuantParams;
    const auto &gmmArray = data.gmmArray;

    std::stringstream ss;
    ss << "MM Tiling: M=" << mm.M << ", N=" << mm.N << ", K=" << mm.Ka << ", usedCoreNum=" << mm.usedCoreNum <<
        ", baseM=" << mm.baseM << ", baseN=" << mm.baseN << ", baseK=" << mm.baseK << ", singleCoreM=" <<
        mm.singleCoreM << ", singleCoreN=" << mm.singleCoreN << ", singleCoreK=" << mm.singleCoreK << ", dbL0C=" <<
        mm.dbL0C << ", depthA1=" << mm.depthA1 << ", depthB1=" << mm.depthB1 << ", stepKa=" << mm.stepKa <<
        ", stepKb=" << mm.stepKb << ", stepM=" << mm.stepM << ", stepN=" << mm.stepN << ", iterateOrder=" <<
        mm.iterateOrder;

    ss << "\nQuant Params: groupNum=" << quantParams.groupNum << ", activeType=" << quantParams.activeType <<
        ", aQuantMode=" << quantParams.aQuantMode << ", bQuantMode=" << quantParams.bQuantMode <<
        ", singleX=" << static_cast<int32_t>(quantParams.singleX) <<
        ", singleW=" << static_cast<int32_t>(quantParams.singleW) <<
        ", singleY=" << static_cast<int32_t>(quantParams.singleY) <<
        ", groupType=" << static_cast<int32_t>(quantParams.groupType) <<
        ", groupListType=" << static_cast<uint32_t>(quantParams.groupListType) <<
        ", hasBias=" << static_cast<int32_t>(quantParams.hasBias) << ", reserved=" << quantParams.reserved;

    ss << "\nArray: mList[0]=" << gmmArray.mList[0] << ", kList[0]=" << gmmArray.kList[0] << ", nList[0]=" <<
        gmmArray.nList[0];

    OP_LOGI(opName_, "QuantGmmA2AvTiling TilingParams:\n%s", ss.str().c_str());
}

void QuantGroupedMatmulAllToAllvTiling::PrintQuantGmmA2avTilingData(QuantGmmA2avTilingData &outTilingData)
{
    PrintGmmA2avWorkspaceInfo(outTilingData.workspaceInfo, opName_);
    PrintTaskTilingInfo(outTilingData.taskTilingInfo, localParams_, opName_);
    OP_LOGD(opName_, "------------- PrintGMMQuantTilingData -------------------");
    PrintGMMQuantTilingData(outTilingData.gmmBaseTiling, opName_);
    OP_LOGD(opName_, "------------- PrintGMMSharedQuantTilingData -------------");
    PrintGMMQuantTilingData(outTilingData.sharedGmmTiling, opName_);
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::PostTiling()
{
    PrintQuantGmmA2avTilingData(localTilingData_);
    context_->SetBlockDim(localParams_.aicCoreNum);
    QuantGmmA2avTilingData *outTilingData = context_->GetTilingData<QuantGmmA2avTilingData>();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "failed to get tiling data from context"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tilingBufCap < sizeof(localTilingData_)),
        OP_LOGE(opName_, "TilingBuffer too small, capacity = %zu, need = %zu.", tilingBufCap, sizeof(localTilingData_)),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sizeof(localTilingData_) % sizeof(uint64_t) != 0,
        OP_LOGE(opName_, "Tiling data size[%zu] is not aligned to 8", sizeof(localTilingData_)),
        return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(
        outTilingData, tilingBufCap, reinterpret_cast<void *>(&localTilingData_), sizeof(localTilingData_));
    if (ret != EOK) {
        OP_LOGE(opName_, "postTiling: memcpy_s tiling data failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(localTilingData_));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "get workspace failed"), return ge::GRAPH_FAILED);
    workspaces[0] = workSpaceSize_;
    OP_LOGD(opName_, "Workspaces[0] size=%ld", workspaces[0]);

    return ge::GRAPH_SUCCESS;
}

uint64_t QuantGroupedMatmulAllToAllvTiling::GetTilingKey() const
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(localParams_.hasSharedMm, localParams_.isGmmWeightTrans,
        localParams_.isMmWeightTrans, localParams_.gmmQuantSuit, localParams_.mmQuantSuit);
    OP_LOGD(opName_, "GET_TPL_TILING_KEY: [%d,%d,%d,%d,%d], TilingKey is [%lu].", localParams_.hasSharedMm,
        localParams_.isGmmWeightTrans, localParams_.isMmWeightTrans, localParams_.gmmQuantSuit,
        localParams_.mmQuantSuit, tilingKey);
    return tilingKey;
}

// 注册tiling类
REGISTER_OPS_TILING_TEMPLATE(GroupedMatMulAlltoAllv, QuantGroupedMatmulAllToAllvTiling, 1);

// }
