/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_quant_grouped_mat_mul_tiling.cpp
 * \brief
 */

#include "allto_allv_tt_quant_grouped_mat_mul_tiling.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
bool AlltoAllvTTQuantGmmTiling::IsCapable()
{
    // hifloat8 quant
    if (gmmXDataType_ != ge::DT_HIFLOAT8) {
        return false;
    }
    if (gmmWeightDataType_ != ge::DT_HIFLOAT8) {
        return false;
    }
    OP_LOGD(context_->GetNodeName(), "AlltoAllvTTQuantGmmTiling is capable.");
    return true;
}

uint64_t AlltoAllvTTQuantGmmTiling::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(ADD_TPL_HIF8, hasSharedExpertFlag_, transGmmWeight_, transMmWeight_);
    return tilingKey;
}

ge::graphStatus AlltoAllvTTQuantGmmTiling::DoGmmTiling(uint64_t gmmxMSzie)
{
    OP_LOGD(context_->GetNodeName(), "start DoGmmTiling.");
    // gmm group matmul tiling
    if (gmmxMSzie != 0) {
        auto &gmmQuantTilingData = tilingData->gmmQuantTilingData;
        SetGMMQuantParams(gmmQuantTilingData);
        SetTilingArray(gmmQuantTilingData, gmmxMSzie, n1_, h1_);
        SetTilingParams(gmmQuantTilingData, gmmxMSzie, n1_, h1_, transGmmWeight_);
        PrintGMMQuantTilingData(gmmQuantTilingData);
    }
    // mm group matmul tiling
    if (bs_ != 0) {
        auto &mmQuantTilingData = tilingData->mmQuantTilingData;
        SetGMMQuantParams(mmQuantTilingData);
        SetTilingArray(mmQuantTilingData, bs_, n2_, h2_);
        SetTilingParams(mmQuantTilingData, bs_, n2_, h2_, transMmWeight_);
        PrintGMMQuantTilingData(mmQuantTilingData);
    }
    OP_LOGD(context_->GetNodeName(), "end DoGmmTiling.");
    return ge::GRAPH_SUCCESS;
}

void AlltoAllvTTQuantGmmTiling::SetGMMQuantParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData) const
{
    gmmQuantTilingData.gmmQuantParams.groupNum = SINGLE_GROUP_NUM;
    gmmQuantTilingData.gmmQuantParams.activeType = GMM_ACT_TYPE_NONE;
    gmmQuantTilingData.gmmQuantParams.aQuantMode = PERTENSOR_QUANT_MODE;
    gmmQuantTilingData.gmmQuantParams.bQuantMode = PERTENSOR_QUANT_MODE;
    gmmQuantTilingData.gmmQuantParams.singleX = 0;
    gmmQuantTilingData.gmmQuantParams.singleW = 0;
    gmmQuantTilingData.gmmQuantParams.singleY = 0;
    gmmQuantTilingData.gmmQuantParams.groupType = 0;
    gmmQuantTilingData.gmmQuantParams.groupListType = 1;
    gmmQuantTilingData.gmmQuantParams.hasBias = 0;
    gmmQuantTilingData.gmmQuantParams.reserved = 0;
}

void AlltoAllvTTQuantGmmTiling::SetTilingArray(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData, uint64_t M, uint64_t N, uint64_t K) const
{
    gmmQuantTilingData.gmmArray.mList[0] = static_cast<int32_t>(M);
    gmmQuantTilingData.gmmArray.kList[0] = static_cast<int32_t>(K);
    gmmQuantTilingData.gmmArray.nList[0] = static_cast<int32_t>(N);
}

void AlltoAllvTTQuantGmmTiling::SetTilingParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData, uint64_t M, uint64_t N, uint64_t K, bool transB) const
{
    auto &mm = gmmQuantTilingData.mmTilingData;

    mm.M = M;
    mm.N = N;
    mm.Ka = K;
    mm.Kb = K;
    mm.usedCoreNum = aicCoreNum_;
    mm.isBias = 0;
    mm.dbL0A = DOUBLE_BUFFER;
    mm.dbL0B = DOUBLE_BUFFER;

    mm.baseM = std::min(static_cast<int32_t>(M), static_cast<int32_t>(BASIC_BLOCK_SIZE_256));
    mm.baseM = Ops::Base::CeilAlign(mm.baseM, static_cast<int32_t>(CUBE_BLOCK));
    mm.baseN = std::min(static_cast<int32_t>(N), static_cast<int32_t>(BASIC_BLOCK_SIZE_256));
    mm.baseN = Ops::Base::CeilAlign(mm.baseN, transB ? static_cast<int32_t>(CUBE_BLOCK) : static_cast<int32_t>(L1_ALIGN_SIZE));
    
    mm.baseK = std::min(static_cast<int32_t>(K), static_cast<int32_t>(BASIC_BLOCK_SIZE_128));
    mm.baseK = Ops::Base::CeilAlign(mm.baseK, static_cast<int32_t>(CUBE_REDUCE_BLOCK));

    mm.singleCoreM = std::min(static_cast<int32_t>(M), mm.baseM);
    mm.singleCoreN = std::min(static_cast<int32_t>(N), mm.baseN);
    mm.singleCoreK = K;

    uint64_t l0cRequired = static_cast<uint64_t>(mm.baseM) * mm.baseN * DATA_SIZE_L0C * DB_SIZE;
    mm.dbL0C = (l0cRequired <= l0cSize_) ? DB_SIZE : 1;

    mm.iterateOrder = 0U;

    uint64_t baseASize = static_cast<uint64_t>(mm.baseM) * mm.baseK;
    uint64_t baseBSize = static_cast<uint64_t>(mm.baseN) * mm.baseK;
    uint64_t baseL1Size = baseASize + baseBSize;

    OP_TILING_CHECK(baseL1Size == 0, OP_LOGW(context_->GetNodeName(), "baseL1Size cannot be zero."), return );

    uint64_t leftL1Size = l1Size_;

    uint64_t depthInit = leftL1Size / baseL1Size;
    depthInit = std::max(depthInit, static_cast<uint64_t>(1));

    uint64_t depthScale = depthInit;
    while (depthScale * mm.baseK % BASIC_BLOCK_SIZE_512 != 0 && depthScale > 1) {
        depthScale--;
    }
    depthScale = std::max(depthScale, static_cast<uint64_t>(1));

    mm.depthA1 = depthScale;
    mm.depthB1 = depthScale;

    mm.stepKa = (mm.depthA1 > 1) ? (mm.depthA1 / DB_SIZE) : 1;
    mm.stepKb = (mm.depthB1 > 1) ? (mm.depthB1 / DB_SIZE) : 1;

    OP_TILING_CHECK(mm.baseK == 0, OP_LOGW(context_->GetNodeName(), "baseK cannot be zero."), return );

    if (mm.stepKa * mm.baseK > mm.Ka) {
        mm.stepKa = Ops::Base::CeilDiv(mm.Ka, mm.baseK);
    }
    if (mm.stepKb * mm.baseK > mm.Kb) {
        mm.stepKb = Ops::Base::CeilDiv(mm.Kb, mm.baseK);
    }

    mm.depthA1 = mm.stepKa * DB_SIZE;
    mm.depthB1 = mm.stepKb * DB_SIZE;

    mm.stepM = 1;
    mm.stepN = 1;
}

ge::graphStatus AlltoAllvTTQuantGmmTiling::CheckQuantMode() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckQuantMode.");
    // check gmmXQuantMode null
    OP_TILING_CHECK(gmmXQuantModePtr_ == nullptr,
        OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, gmmXQuantMode attr can not be null."), return ge::GRAPH_FAILED);
    // check gmmXQuantMode
    int64_t gmmXQuantMode = *gmmXQuantModePtr_;
    OP_TILING_CHECK(gmmXQuantMode != static_cast<int64_t>(PERTENSOR_QUANT_MODE),
        OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, gmmXQuantMode should be 1, but actual is %lu.", \
            gmmXQuantMode), return ge::GRAPH_FAILED);
    // check gmmWeightQuantMode null
    OP_TILING_CHECK(gmmWeightQuantModePtr_ == nullptr,
        OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, gmmWeightQuantMode attr can not be null."), return ge::GRAPH_FAILED);
    // check gmmWeightQuantMode
    int64_t gmmWeightQuantMode = *gmmWeightQuantModePtr_;
    OP_TILING_CHECK(gmmWeightQuantMode != static_cast<int64_t>(PERTENSOR_QUANT_MODE),
        OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, gmmWeightQuantMode should be 1, but actual is %lu.", \
            gmmWeightQuantMode), return ge::GRAPH_FAILED);
    if (hasSharedExpertFlag_) {
        // mmXQuantMode(same as gmmXQuantMode)
        int64_t mmXQuantMode = *mmXQuantModePtr_;
        OP_TILING_CHECK(mmXQuantMode != gmmXQuantMode,
            OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, mmXQuantMode should be same as gmmXQuantMode(1), but actual is %lu.", \
                mmXQuantMode), return ge::GRAPH_FAILED);
        // mmWeightQuantMode
        int64_t mmWeightQuantMode = *mmWeightQuantModePtr_;
        OP_TILING_CHECK(mmWeightQuantMode != gmmWeightQuantMode,
            OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, mmWeightQuantMode should be same as gmmWeightQuantMode(1), but actual is %lu.", \
                mmWeightQuantMode), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    // check group size
    auto groupSizePtr = context_->GetAttrs()->GetAttrPointer<int64_t>(ATTR_GROUP_SIZE_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr, OP_LOGE(context_->GetNodeName(), "The groupSize can not be null."),
        return ge::GRAPH_FAILED);
    uint64_t groupSize = *groupSizePtr;
    OP_TILING_CHECK(groupSize != 0,
        OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, the groupSize should be 0, "
        "but actual is %lu.", groupSize), return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckQuantMode.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvTTQuantGmmTiling::CheckScaleFormatAndDtype() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckScaleFormatAndDtype.");
    // check gmmXScale null
    auto gmmXScaleDesc = context_->GetOptionalInputDesc(GMM_X_SCALE_INDEX);
    OP_TILING_CHECK(gmmXScaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, gmmXScale should not be null."), return ge::GRAPH_FAILED);
    // check gmmXScale format
    OP_TILING_CHECK(gmmXScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND, OP_LOGE(context_->GetNodeName(), "gmmXScale storage format should be ND, but actual is %s.", \
        Ops::Base::ToString(gmmXScaleDesc->GetStorageFormat()).c_str()), return ge::GRAPH_FAILED);
    // check gmmXScale dataType
    OP_TILING_CHECK(gmmXScaleDesc->GetDataType() != ge::DT_FLOAT,OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, gmmXScale should be float32, but actual is %s.", \
        ge::TypeUtils::DataTypeToSerialString(gmmXScaleDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    // check gmmWeightScale null
    auto gmmWeightScaleDesc = context_->GetOptionalInputDesc(GMM_WEIGHT_SCALE_INDEX);
    OP_TILING_CHECK(gmmWeightScaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, gmmWeightScale should not be null."), return ge::GRAPH_FAILED);
    // check gmmWeightScale format
    OP_TILING_CHECK(gmmWeightScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND, OP_LOGE(context_->GetNodeName(), "gmmWeightScale storage format should be ND, but actual is %s.", \
        Ops::Base::ToString(gmmWeightScaleDesc->GetStorageFormat()).c_str()), return ge::GRAPH_FAILED);    
    // check gmmWeightScale dataType
    OP_TILING_CHECK(gmmWeightScaleDesc->GetDataType() != ge::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, gmmWeightScale should be float32, but actual is %s.", \
        ge::TypeUtils::DataTypeToSerialString(gmmWeightScaleDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    if (hasSharedExpertFlag_) {
        // check mmXScale null
        auto mmXScaleDesc = context_->GetOptionalInputDesc(MM_X_SCALE_INDEX);
        OP_TILING_CHECK(mmXScaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, mmXScale should not be null."), return ge::GRAPH_FAILED);
        // check mmXScale format
        OP_TILING_CHECK(mmXScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND, OP_LOGE(context_->GetNodeName(), "mmXScale storage format should be ND, but actual is %s.", \
            Ops::Base::ToString(mmXScaleDesc->GetStorageFormat()).c_str()), return ge::GRAPH_FAILED);
        // check mmXScale dataType(same as gmmXScale)
        OP_TILING_CHECK(mmXScaleDesc->GetDataType() != gmmXScaleDesc->GetDataType(), OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, "
            "mmXScale should be same as gmmXScale(float32), but actual is %s.",ge::TypeUtils::DataTypeToSerialString(mmXScaleDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
        // check mmWeightScale null
        auto mmWeightScaleDesc = context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_INDEX);
        OP_TILING_CHECK(mmWeightScaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, mmWeightScale should not be null."), return ge::GRAPH_FAILED);
        // check mmWeightScale format
        OP_TILING_CHECK(mmWeightScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND, OP_LOGE(context_->GetNodeName(), "mmWeightScale storage format should be ND, but actual is %s.", \
            Ops::Base::ToString(mmWeightScaleDesc->GetStorageFormat()).c_str()), return ge::GRAPH_FAILED);
        // check mmWeightScale dataType(same as gmmWeightScale)
        OP_TILING_CHECK(mmWeightScaleDesc->GetDataType() != gmmWeightScaleDesc->GetDataType(),
            OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, mmWeightScale should be same as gmmWeightScale(float32), but actual is %s.", \
            ge::TypeUtils::DataTypeToSerialString(context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_INDEX)->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    }
    OP_LOGD(context_->GetNodeName(), "end CheckScaleFormatAndDtype.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvTTQuantGmmTiling::CheckInputDtype() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckInputDtype.");
    // check gmmX datatype
    ge::DataType gmmXDataType = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    OP_TILING_CHECK(gmmXDataType != ge::DT_HIFLOAT8, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, gmmX should be hifloat8, but actual is %s.", \
                ge::TypeUtils::DataTypeToSerialString(gmmXDataType).c_str()), return ge::GRAPH_FAILED);
    // check gmmWeight datatype
    ge::DataType gmmWeightDataType = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK(gmmWeightDataType != ge::DT_HIFLOAT8, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, gmmWeight should be hifloat8, but actual is %s.", \
                ge::TypeUtils::DataTypeToSerialString(gmmWeightDataType).c_str()), return ge::GRAPH_FAILED);
    // check gmmY dataType
    ge::DataType gmmYDataType = context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(gmmYDataType != ge::DT_FLOAT16 && gmmYDataType != ge::DT_BF16, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, gmmY should be float16 or bfloat16, "
        "but actual is %s.", ge::TypeUtils::DataTypeToSerialString(gmmYDataType).c_str()), return ge::GRAPH_FAILED);
    if (permuteOutFlag_) {
        // check permuteOut dtype
        ge::DataType permuteOutDataType = context_->GetOutputDesc(OUTPUT_PERMUTE_OUT_INDEX)->GetDataType();
        OP_TILING_CHECK(permuteOutDataType != gmmXDataType, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, permuteOut should be same as gmmX dataType(hifloat8), "
            "but actual is %s.", ge::TypeUtils::DataTypeToSerialString(permuteOutDataType).c_str()), return ge::GRAPH_FAILED);
    }
    if (hasSharedExpertFlag_) {
        // check mmX dataType(same as gmmX)
        ge::DataType mmXDataType = context_->GetOptionalInputDesc(MM_X_INDEX)->GetDataType();
        OP_TILING_CHECK(mmXDataType != gmmXDataType, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, mmX should be same as gmmX(hifloat8), but actual is %s.", \
                    ge::TypeUtils::DataTypeToSerialString(mmXDataType).c_str()), return ge::GRAPH_FAILED);
        // check mmWeight dataType(same as gmmWeight)
        ge::DataType mmWeightDataType = context_->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType();
        OP_TILING_CHECK(mmWeightDataType != gmmWeightDataType, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, mmWeight should be same as gmmWeight(hifloat8), "
            "but actual is %s.", ge::TypeUtils::DataTypeToSerialString(mmWeightDataType).c_str()), return ge::GRAPH_FAILED);
        // check mmY dataType(same as gmmY)
        ge::DataType mmYDataType = context_->GetOutputDesc(OUTPUT_MM_Y_INDEX)->GetDataType();
        OP_TILING_CHECK(mmYDataType != gmmYDataType, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, mmY should be same as gmmY(%s), but actual is %s.", \
                    ge::TypeUtils::DataTypeToSerialString(mmYDataType).c_str(), ge::TypeUtils::DataTypeToSerialString(gmmYDataType).c_str()), return ge::GRAPH_FAILED);     
    }
    OP_LOGD(context_->GetNodeName(), "end CheckInputDtype.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvTTQuantGmmTiling::CheckScaleShape() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckScaleShape.");
    // check gmmXScale dimNum
    size_t gmmXScaleDimNum = context_->GetOptionalInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(gmmXScaleDimNum != DIM_ONE, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, gmmXScale input dimNum should be 1, "
                    "but actual dimNum is %lu.", gmmXScaleDimNum), return ge::GRAPH_FAILED);
    // check gmmXScale shape
    int64_t gmmXScaleShape = context_->GetOptionalInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(gmmXScaleShape != DIM_ONE, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, gmmXScale input shape should be [1], "
                    "but actual shape is [%lu].", gmmXScaleShape), return ge::GRAPH_FAILED);
    // check gmmWeightScale dimNum
    size_t gmmWeightScaleDimNum = context_->GetOptionalInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(gmmWeightScaleDimNum != DIM_ONE, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, gmmWeightScale input dimNum should be 1, "
                    "but actual dimNum is %lu.", gmmWeightScaleDimNum), return ge::GRAPH_FAILED);
    // check gmmWeightScale shape
    int64_t gmmWeightScaleShape = context_->GetOptionalInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(gmmWeightScaleShape != DIM_ONE, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, gmmWeightScale input shape should be [1], "
                    "but actual shape is [%lu].", gmmWeightScaleShape), return ge::GRAPH_FAILED);
    if (hasSharedExpertFlag_) {
        // check mmXScale dimNum(same as gmmXScale)
        size_t mmXScaleDimNum = context_->GetOptionalInputShape(MM_X_SCALE_INDEX)->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(mmXScaleDimNum != gmmXScaleDimNum, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, mmXScaleDimNum input dimNum should be same as gmmXScale(1), "
                        "but actual dimNum is %lu.", mmXScaleDimNum), return ge::GRAPH_FAILED);
        // check mmXScale shape
        int64_t mmXScaleShape = context_->GetOptionalInputShape(MM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
        OP_TILING_CHECK(mmXScaleShape != DIM_ONE, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, mmXScaleDimNum input shape should be [1], "
                        "but actual shape is [%lu].", mmXScaleShape), return ge::GRAPH_FAILED);
        // check mmWeightScale dimNum(same as gmmWeightScale)
        size_t mmWeightScaleDimNum = context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(mmWeightScaleDimNum != gmmWeightScaleDimNum, OP_LOGE(context_->GetNodeName(), "When pertensor quant mode, mmWeightScaleDimNum input dimNum should be same as gmmWeightScale(1), "
                        "but actual dimNum is %lu.", mmWeightScaleDimNum), return ge::GRAPH_FAILED);
        // check mmWeightScale shape
        int64_t mmWeightScaleShape = context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
        OP_TILING_CHECK(mmWeightScaleShape != DIM_ONE, OP_LOGE(context_->GetNodeName(), "mmWeightScale input shape should be [1], but actual shape is [%lu]", mmWeightScaleShape), 
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGD(context_->GetNodeName(), "end CheckScaleShape.");
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AlltoAllvQuantGroupedMatMul, AlltoAllvTTQuantGmmTiling, 1);
} // namespace optiling