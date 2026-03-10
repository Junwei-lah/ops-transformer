/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file left_padding_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "left_padding_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// 公共校验函数

// Utility functions for common checkers.
ge::graphStatus LeftPaddingChecker::CheckShapeSupport(const gert::Tensor *tensor,
                                                      const std::vector<int64_t> &expectShapeList) const
{
    if (tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectShapeList.begin(), expectShapeList.end(), tensor->GetShapeSize()) == expectShapeList.end()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// CheckSingle
ge::graphStatus LeftPaddingChecker::CheckSingleDesc(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.qPaddingSizeFlag) {
        OP_CHECK_IF(fiaInfo.opParamInfo.queryPaddingSize.desc == nullptr,
                    OP_LOGE(fiaInfo.opName, "Desc of tensor query_padding_size is nullptr!"), return ge::GRAPH_FAILED);
    }

    if (fiaInfo.kvPaddingSizeFlag) {
        OP_CHECK_IF(fiaInfo.opParamInfo.kvPaddingSize.desc == nullptr,
                    OP_LOGE(fiaInfo.opName, "Desc of tensor kv_padding_size is nullptr!"), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// CheckFeature
ge::graphStatus LeftPaddingChecker::CheckFeatureActualLen(const FiaTilingInfo &fiaInfo)
{
    // When left-padding is enabled for Query and Key/Value, the actual sequence length information must be provided.
    if (fiaInfo.qPaddingSizeFlag && !fiaInfo.isMaxWorkspace) {
        OP_CHECK_IF(fiaInfo.opParamInfo.actualSeqLengthsQ.tensor == nullptr ||
                        fiaInfo.opParamInfo.actualSeqLengthsQ.tensor->GetShapeSize() == 0U,
                    OP_LOGE(fiaInfo.opName,
                            "When query_padding_size exists, the query's actual sequence lengths are required!"),
                    return ge::GRAPH_FAILED);
    }

    if (fiaInfo.kvPaddingSizeFlag && !fiaInfo.isMaxWorkspace) {
        OP_CHECK_IF(fiaInfo.opParamInfo.actualSeqLengths.tensor == nullptr ||
                        fiaInfo.opParamInfo.actualSeqLengths.tensor->GetShapeSize() == 0U,
                    OP_LOGE(fiaInfo.opName,
                            "When kv_padding_size exists, the key/value's actual sequence lengths are required!"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckFeatureLayOut(const FiaTilingInfo &fiaInfo)
{
    // When left-padding is enabled for Query and Key/Value, TND/NTD scenarios are not supported.
    if (fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag) {
        OP_CHECK_IF(fiaInfo.qLayout == FiaLayout::TND,
                    OP_LOGE(fiaInfo.opName, "QueryLeftPadding illegal condition:input layout TND!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(fiaInfo.qLayout == FiaLayout::NTD,
                    OP_LOGE(fiaInfo.opName, "QueryLeftPadding illegal condition:input layout NTD!"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckFeatureAlibiPse(const FiaTilingInfo &fiaInfo)
{
    // When left-padding is enabled for Query and Key/Value, PseType = 2/3 is not supported.
    if (fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag) {
        OP_CHECK_IF(fiaInfo.enableAlibiPse,
                    OP_LOGE(fiaInfo.opName, "When pseType = 2/3, left padding is not supported!"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckFeaturePageAttention(const FiaTilingInfo &fiaInfo)
{
    // When left-padding is enabled for Query and Key/Value, PageAttention scenarios are not supported.
    if (fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag) {
        OP_CHECK_IF(fiaInfo.pageAttentionFlag,
                    OP_LOGE(fiaInfo.opName, "When page attention is used, left padding is not supported!"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// enableNonQuant 相关校验函数
// CheckMuiltPara
ge::graphStatus LeftPaddingChecker::CheckMultiParaShapeAndDim(const FiaTilingInfo &fiaInfo)
{
    // When left-padding is enabled for Query and Key/Value,
    // the Shape size and Dim number corresponding to the Padding size must both be 1.
    if (fiaInfo.qPaddingSizeFlag) {
        const std::vector<int64_t> querypaddingsizeShapeNumList = {SHAPE_NUM_ONE};
        OP_CHECK_IF(ge::GRAPH_SUCCESS !=
                        CheckShapeSupport(fiaInfo.opParamInfo.queryPaddingSize.tensor, querypaddingsizeShapeNumList),
                    OP_LOGE(fiaInfo.opName, "The shape size of query paddingsize is not 1!"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(fiaInfo.opParamInfo.queryPaddingSize.tensor->GetStorageShape().GetDimNum() != DIM_NUM_1,
                    OP_LOGE(fiaInfo.opName, "The dim number of query paddingsize is not 1!"), return ge::GRAPH_FAILED);
    }

    if (fiaInfo.kvPaddingSizeFlag) {
        const std::vector<int64_t> kvpaddingsizeShapeNumList = {SHAPE_NUM_ONE};
        OP_CHECK_IF(
            ge::GRAPH_SUCCESS != CheckShapeSupport(fiaInfo.opParamInfo.kvPaddingSize.tensor, kvpaddingsizeShapeNumList),
            OP_LOGE(fiaInfo.opName, "The shape size of kv paddingsize is not 1!"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(fiaInfo.opParamInfo.kvPaddingSize.tensor->GetStorageShape().GetDimNum() != DIM_NUM_1,
                    OP_LOGE(fiaInfo.opName, "The dim number of kv paddingsize is not 1!"), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// enableFullQuant 相关校验函数

// enableAntiQuant 相关校验函数

ge::graphStatus LeftPaddingChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin LeftPaddingChecker::CheckSinglePara!");
    if (ge::GRAPH_SUCCESS != CheckSingleDesc(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (enableNonQuant_) {
        ;
    } else if (enableFullQuant_) {
        ;
    } else if (enableAntiQuant_) {
        ;
    }
    OP_LOGI(fiaInfo.opName, "End LeftPaddingChecker::CheckSinglePara!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin LeftPaddingChecker::CheckParaExistence!");
    if (enableNonQuant_) {
        ;
    } else if (enableFullQuant_) {
        ;
    } else if (enableAntiQuant_) {
        ;
    }
    OP_LOGI(fiaInfo.opName, "End LeftPaddingChecker::CheckParaExistence!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckFeature(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin LeftPaddingChecker::CheckFeature!");
    if (ge::GRAPH_SUCCESS != CheckFeatureActualLen(fiaInfo) || ge::GRAPH_SUCCESS != CheckFeatureLayOut(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureAlibiPse(fiaInfo) || ge::GRAPH_SUCCESS != CheckFeaturePageAttention(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (enableNonQuant_) {
        ;
    } else if (enableFullQuant_) {
        ;
    } else if (enableAntiQuant_) {
        ;
    }
    OP_LOGI(fiaInfo.opName, "End LeftPaddingChecker::CheckFeature!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckMultiPara(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin LeftPaddingChecker::CheckMultiPara!");
    if (enableNonQuant_) {
        if (ge::GRAPH_SUCCESS != CheckMultiParaShapeAndDim(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    } else if (enableFullQuant_) {
        ;
    } else if (enableAntiQuant_) {
        ;
    }
    OP_LOGI(fiaInfo.opName, "End LeftPaddingChecker::CheckMultiPara!");
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
