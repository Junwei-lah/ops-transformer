/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "grouped_mat_mul_allto_allv_tiling_base.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace optiling;

namespace optiling {
constexpr uint32_t MAX_GROUP_BUFFER_SIZE = 128;

// base check required
ge::graphStatus GmmAlltoAllvTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    auto gmmXTensorDesc = context_->GetInputDesc(GMM_X_INDEX);
    auto gmmWeightTensorDesc = context_->GetInputDesc(GMM_WEIGHT_INDEX);
    auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((gmmXTensorDesc == nullptr), OP_LOGE(opName_, "the input gmmX tensor is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((gmmWeightTensorDesc == nullptr), OP_LOGE(opName_, "the input gmmWeight tensor is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((yDesc == nullptr), OP_LOGE(opName_, "the output y tensor is null."),
                    return ge::GRAPH_FAILED);
    
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(opName_, "Failed to get attrs."), return ge::GRAPH_FAILED);
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    // 判断为空或者空字符串
    OP_TILING_CHECK(group == nullptr, OP_LOGE(opName_, "The input attr group is null pointer."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(group[0] == '\0', OP_LOGE(opName_, "The input attr group is empty string."),
                    return ge::GRAPH_FAILED);
    // 判断group是否超过127
    const char* nullTerminator = static_cast<const char*>(memchr(group, '\0', MAX_GROUP_BUFFER_SIZE));
    OP_TILING_CHECK(nullTerminator == nullptr, OP_LOGE(opName_, "The input attr group length is large than 128!"),
                    return ge::GRAPH_FAILED);
    
    auto epWorldSizePtr = attrs->GetAttrPointer<int>(ATTR_EP_WORLD_SIZE_INDEX);
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(opName_, "The input attr epWorldSizePtr is null!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sendCountsPtr == nullptr, OP_LOGE(opName_, "The input attr sendCountsPtr is null!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvCountsPtr == nullptr, OP_LOGE(opName_, "The input attr recvCountsPtr is null!"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(
        platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "fail to get platform info"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingBase::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}
uint64_t GmmAlltoAllvTilingBase::GetTilingKey() const
{
    return 0;
}

QuantModePair GmmAlltoAllvTilingBase::GetQuantMode(const gert::TilingContext *context, const char *opName)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(opName, "Failed to get attrs.");
        return QUANT_PAIR_ERROR;
    }
    // 获取量化模式属性（默认为0，表示非量化）
    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    if (const int64_t *ptr = attrs->GetAttrPointer<int64_t>(ATTR_GMM_X_QUANT_MODE_INDEX)) {
        gmmXQuantMode = *ptr;
    }
    if (const int64_t *ptr = attrs->GetAttrPointer<int64_t>(ATTR_GMM_WEIGHT_QUANT_MODE_INDEX)) {
        gmmWeightQuantMode = *ptr;
    }

    if (gmmXQuantMode == QUANT_NONE && gmmWeightQuantMode == QUANT_NONE) {
        return QUANT_PAIR_NONE;
    }
    if (gmmXQuantMode == QUANT_PERTENSOR && gmmWeightQuantMode == QUANT_PERTENSOR) {
        return QUANT_PAIR_TT;
    } else {
        OP_LOGD(opName,
                "Quantization mode error, TT quantization gmmXQuantMode should be one, gmmWeightQuantMode should be one."
                "currently gmmXQuantMode=%d, gmmWeightQuantMode=%d.",
                gmmXQuantMode, gmmWeightQuantMode);
    }
    return QUANT_PAIR_ERROR;
}
} // namespace
