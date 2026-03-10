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
 * \file mask_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "mask_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// CheckSinglePara
ge::graphStatus MaskChecker::CheckDtypeAndFormat(const FiaTilingInfo &fiaInfo)
{
    // AttentionMask data type must be int8/uint8/bool, and data format must be ND/NCHW/NHWC/NCDHW.
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(fiaInfo.opParamInfo.attenMask.desc, ATTEN_MASK_NAME)) {
        OP_LOGE(fiaInfo.opName, "AttentionMask data type must be int8/uint8/bool!");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckFormatSupport(fiaInfo.opParamInfo.attenMask.desc, ATTEN_MASK_NAME)) {
        OP_LOGE(fiaInfo.opName, "Data format must be ND/NCHW/NHWC/NCDHW!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSparseMode(const FiaTilingInfo &fiaInfo)
{
    // SparseMode only supports 0/1/2/3/4.
    const std::vector<int32_t> sparseModeList = {SPARSE_MODE_NO_MASK, SPARSE_MODE_ALL_MASK, SPARSE_MODE_LEFT_UP,
                                                 SPARSE_MODE_RIGHT_DOWN, SPARSE_MODE_BAND};
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckValueSupport(fiaInfo.sparseMode, sparseModeList),
                OP_LOGE(fiaInfo.opName, "SparseMode only supports 0/1/2/3/4, but got %u", fiaInfo.sparseMode),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// CheckFeature
ge::graphStatus MaskChecker::CheckNoQuantIFAMLA(const FiaTilingInfo &fiaInfo)
{
    // For IFA MLA, input sparse mode only supports 0/3/4.
    enableIFAMLA = (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512);
    if (enableIFAMLA) {
        OP_CHECK_IF(
            fiaInfo.sparseMode != SPARSE_MODE_NO_MASK && fiaInfo.sparseMode != SPARSE_MODE_RIGHT_DOWN &&
                fiaInfo.sparseMode != SPARSE_MODE_BAND,
            OP_LOGE(fiaInfo.opName, "Only support sparse(%d) 0/3/4 when ifa mla is enable!", fiaInfo.sparseMode),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckFullQuantIFAMLA(const FiaTilingInfo &fiaInfo)
{
    // For IFA MLA, input sparse mode %d, sparse 0 without mask is supported only when sequence length is 1,
    // and sparse 3 with mask or sparse 0 without mask is supported only when sequence length > 1.
    enableIFAMLA = (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512);
    if (enableIFAMLA) {
        if (fiaInfo.s1Size == 1U) {
            OP_CHECK_IF(!((fiaInfo.sparseMode == SPARSE_MODE_NO_MASK) && (!fiaInfo.attenMaskFlag)),
                        OP_LOGE(fiaInfo.opName,
                                "Only support sparse 0 without mask when ifa mla and query's sequence length is 1, "
                                "input sparse mode is %d and there has%smask",
                                fiaInfo.sparseMode, fiaInfo.attenMaskFlag ? " " : " no "),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(!(((fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) && (fiaInfo.attenMaskFlag)) ||
                          ((fiaInfo.sparseMode == SPARSE_MODE_NO_MASK) && (!fiaInfo.attenMaskFlag))),
                        OP_LOGE(fiaInfo.opName,
                                "Only support sparse 3 with mask, or sparse 0 without mask when ifa mla and "
                                "query's sequence length is > 1, "
                                "input sparse mode is %d and there has%smask",
                                fiaInfo.sparseMode, fiaInfo.attenMaskFlag ? " " : " no "),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckQKVDDifferent(const FiaTilingInfo &fiaInfo)
{
    // Only support sparse mode 0/2/3 when query and key headdim is not equal to value headdim.
    OP_CHECK_IF(
        fiaInfo.isQKVDDifferent && fiaInfo.sparseMode != SPARSE_MODE_NO_MASK &&
            fiaInfo.sparseMode != SPARSE_MODE_LEFT_UP && fiaInfo.sparseMode != SPARSE_MODE_RIGHT_DOWN,
        OP_LOGE(fiaInfo.opName, "Not support sparse mode %d when query and key headdim is not equal to value headdim.",
                fiaInfo.sparseMode),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// CheckMultiPara
ge::graphStatus MaskChecker::CheckNoMaskPretokenAndNexttoken(const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF((fiaInfo.sparseMode == SPARSE_MODE_NO_MASK) && (fiaInfo.nextToken < 0) &&
                    ((fiaInfo.nextToken * (-1)) >= static_cast<int32_t>(fiaInfo.s1Size)),
                OP_LOGE(fiaInfo.opName,
                        "NextTokens absolute value should be smaller than length of query, nextTokens = %ld, "
                        "length of query = %u.",
                        fiaInfo.nextToken, fiaInfo.s1Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.sparseMode == SPARSE_MODE_NO_MASK) && (fiaInfo.preToken < 0) &&
                    ((fiaInfo.preToken * (-1)) >=
                     (static_cast<int32_t>(fiaInfo.s2Size) + static_cast<int32_t>(fiaInfo.systemPrefixLen))),
                OP_LOGE(fiaInfo.opName,
                        "PreToken absolute value should be smaller than length of key and value "
                        "(length of key and value + length of prefix when enable prefix), "
                        "preTokens = %ld, seqLengthKV = %u, actualSharedPrefixLen = %ld",
                        fiaInfo.preToken, fiaInfo.s2Size, fiaInfo.systemPrefixLen),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckBandPretokenAndNexttoken(const FiaTilingInfo &fiaInfo)
{
    ge::DataType outputType = fiaInfo.opParamInfo.attenOut.desc->GetDataType();
    OP_CHECK_IF((fiaInfo.sparseMode == SPARSE_MODE_BAND) && (fiaInfo.preToken < 0) &&
                    ((fiaInfo.preToken * (-1)) >= static_cast<int32_t>(fiaInfo.s1Size)),
                OP_LOGE(fiaInfo.opName,
                        "PreTokens absolute value should be smaller than length of query in band mode, preTokens = "
                        "%ld, length of query = %u.",
                        fiaInfo.preToken, fiaInfo.s1Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.sparseMode == SPARSE_MODE_BAND) && (fiaInfo.nextToken < 0) &&
                    ((fiaInfo.nextToken * (-1)) >=
                     (static_cast<int32_t>(fiaInfo.s2Size) + static_cast<int32_t>(fiaInfo.systemPrefixLen))),
                OP_LOGE(fiaInfo.opName,
                        "NextToken absolute value should be smaller than length of key and value in band mode"
                        "(length of key and value + length of prefix when enable prefix), "
                        "NextTokens = %ld, seqLengthKV = %u, actualSharedPrefixLen = %ld",
                        fiaInfo.nextToken, fiaInfo.s2Size, fiaInfo.systemPrefixLen),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (fiaInfo.sparseMode == SPARSE_MODE_BAND && outputType == ge::DT_INT8 &&
         ((fiaInfo.preToken < 0) || fiaInfo.nextToken < 0)),
        OP_LOGE(fiaInfo.opName,
                "When output type is int8, sparse mode = 4, preTokens (%ld) or nextTokens (%ld) cannot be negative.",
                fiaInfo.preToken, fiaInfo.nextToken),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckPretokenAndNexttoken(const FiaTilingInfo &fiaInfo)
{
    // In PFA mode, the values of pretoken and nexttoken must ensure the mask range remains valid.
    isIFAFlag = (enableAntiQuant_) && (fiaInfo.s1Size == 1);
    if (isIFAFlag) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((fiaInfo.nextToken * (-1)) > fiaInfo.preToken,
                OP_LOGE(fiaInfo.opName,
                        "Nexttoken line should be higher than pretoken line, preTokens = %ld, nextTokens = %ld.",
                        fiaInfo.preToken, fiaInfo.nextToken),
                return ge::GRAPH_FAILED);
    // Check the specific conditions that pretoken and nexttoken must satisfy under the sparse mode.
    if (ge::GRAPH_SUCCESS != CheckNoMaskPretokenAndNexttoken(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckBandPretokenAndNexttoken(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckIFADimAndShape(const FiaTilingInfo &fiaInfo)
{
    uint32_t minAttenMaskSize = 0;
    const gert::Tensor *maskShape = fiaInfo.opParamInfo.attenMask.tensor;
    size_t attenMaskDim = maskShape->GetStorageShape().GetDimNum();
    uint32_t attenMaskBatch = maskShape->GetStorageShape().GetDim(DIM_NUM_0);
    uint32_t attenMaskSize = maskShape->GetStorageShape().GetDim(maskShape->GetStorageShape().GetDimNum() - 1);
    if (fiaInfo.pageAttentionFlag) {
        uint32_t maxBlockNumPerSeq = fiaInfo.opParamInfo.blockTable.tensor->GetStorageShape().GetDim(DIM_NUM_1);
        uint32_t sMax = maxBlockNumPerSeq * fiaInfo.blockSize;
        minAttenMaskSize = sMax;
    } else {
        minAttenMaskSize = fiaInfo.maxActualseq;
    }
    if (attenMaskDim == MASK_DIM_SS) {
        OP_LOGE(fiaInfo.opName,
                "The current dimension of the mask is 2. "
                "Please use 3D mask \[B,QS,KVS\]\/\[1,QS,KVS\] or 4D mask \[B,1,QS,KVS\]\/\[1,1,QS,KVS\].");
        return ge::GRAPH_FAILED;
    } else {
        bool checkMask = (attenMaskSize >= minAttenMaskSize) &&
                    ((attenMaskBatch == fiaInfo.bSize) || (attenMaskBatch == DIM_NUM_1));
        OP_CHECK_IF(!checkMask,
                    OP_LOGE(fiaInfo.opName,
                            "AttenMask batch(%u) must be %u or 1, attenMask KV_S(%u) must be larger than or equal to"
                            "the second dimension of blockTable * blockSize(%u), please check",
                            attenMaskBatch, fiaInfo.bSize, attenMaskSize, minAttenMaskSize),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::GetMaskInfo(const FiaTilingInfo &fiaInfo, MaskInfo &maskInfo)
{
    const gert::Tensor *maskShape = fiaInfo.opParamInfo.attenMask.tensor;
    size_t attenMaskDim = maskShape->GetStorageShape().GetDimNum();
    if (attenMaskDim == MASK_DIM_SS) {
        if (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK) {
            OP_LOGE(fiaInfo.opName,
                    "Attenmask does not support inputs with dim 2 when sparse mode = %u. "
                    "Please use 3D mask \[B,QS,KVS\]\/\[1,QS,KVS\] or 4D mask \[B,1,QS,KVS\]\/\[1,1,QS,KVS\].",
                    fiaInfo.sparseMode);
            return ge::GRAPH_FAILED;
        }
        maskInfo.attenMaskQSize = maskShape->GetStorageShape().GetDim(DIM_NUM_0);
        maskInfo.attenMaskSize = maskShape->GetStorageShape().GetDim(DIM_NUM_1);
        maskInfo.strMaskShape = std::to_string(maskInfo.attenMaskQSize) + ", " + std::to_string(maskInfo.attenMaskSize);
    } else if (attenMaskDim == MASK_DIM_BSS) {
        maskInfo.attenMaskBatch = maskShape->GetStorageShape().GetDim(DIM_NUM_0);
        maskInfo.attenMaskQSize = maskShape->GetStorageShape().GetDim(DIM_NUM_1);
        maskInfo.attenMaskSize =
            maskShape->GetStorageShape().GetDim(DIM_NUM_2);  // 2: When the dim is 3, the second dimension is S2.
        maskInfo.strMaskShape = std::to_string(maskInfo.attenMaskBatch) + ", " +
                                std::to_string(maskInfo.attenMaskQSize) + ", " + std::to_string(maskInfo.attenMaskSize);
    } else if (attenMaskDim == MASK_DIM_B1SS) {
        maskInfo.attenMaskBatch = maskShape->GetStorageShape().GetDim(DIM_NUM_0);
        maskInfo.attenMaskN = maskShape->GetStorageShape().GetDim(DIM_NUM_1);
        maskInfo.attenMaskQSize =
            maskShape->GetStorageShape().GetDim(DIM_NUM_2);  // 2: When the dim is 4, the second dimension is S1.
        maskInfo.attenMaskSize = maskShape->GetStorageShape().GetDim(DIM_NUM_3);  // 3:The third dimension is S2.
        maskInfo.strMaskShape = std::to_string(maskInfo.attenMaskBatch) + ", " + std::to_string(maskInfo.attenMaskN) +
                                ", " + std::to_string(maskInfo.attenMaskQSize) + ", " +
                                std::to_string(maskInfo.attenMaskSize);
    } else {
        OP_LOGE(fiaInfo.opName, "AttenMask dim(%zu) must be 2 or 3 or 4!", attenMaskDim);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckDimAndShape(const FiaTilingInfo &fiaInfo)
{
    // In PFA mode, the attenmask dimensions must be 2/3/4.
    // The allowed shape specifications for attenmask vary depending on the sparse mode.
    if (!fiaInfo.attenMaskFlag) {
        return ge::GRAPH_SUCCESS;
    }
    isIFAFlag = (fiaInfo.antiQuantFlag) && (fiaInfo.s1Size == 1);
    if (isIFAFlag) {
        return CheckIFADimAndShape(fiaInfo);
    }

    MaskInfo maskInfo;
    if (GetMaskInfo(fiaInfo, maskInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }

    bool checkMask = false;
    if (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK) {
        checkMask = (maskInfo.attenMaskQSize >= fiaInfo.s1Size) && (maskInfo.attenMaskSize >= fiaInfo.s2Size) &&
                    (maskInfo.attenMaskBatch == NUM1 || maskInfo.attenMaskBatch == fiaInfo.bSize) &&
                    (static_cast<uint32_t>(maskInfo.attenMaskN) == NUM1);
    } else if ((fiaInfo.sparseMode == SPARSE_MODE_LEFT_UP) || (fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) ||
               (fiaInfo.sparseMode == SPARSE_MODE_BAND)) {
        checkMask = (maskInfo.attenMaskBatch == NUM1) && (static_cast<uint32_t>(maskInfo.attenMaskN) == NUM1) &&
                    (maskInfo.attenMaskQSize == SPARSE_OPTIMIZE_ATTENTION_SIZE) &&
                    (maskInfo.attenMaskSize == SPARSE_OPTIMIZE_ATTENTION_SIZE);
    }

    if (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK) {
        OP_CHECK_IF(
            !checkMask,
            OP_LOGE(fiaInfo.opName,
                    "attenMask batch(%u) must be 1 or %u, attenMask Q_S(%u) must be larger than or equal to sQ(%u),"
                    "attenMask KV_S(%u) must be larger than or equal to sK(%u), please check",
                    maskInfo.attenMaskBatch, fiaInfo.bSize, maskInfo.attenMaskQSize, fiaInfo.s1Size,
                    maskInfo.attenMaskSize, fiaInfo.s2Size),
            return ge::GRAPH_FAILED);
    }
    if (((fiaInfo.sparseMode == SPARSE_MODE_LEFT_UP) || (fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) ||
         (fiaInfo.sparseMode == SPARSE_MODE_BAND)) &&
        !checkMask) {
        OP_LOGE(fiaInfo.opName,
                "attenMask shape must be (2048, 2048) or (1, 2048, 2048) or (1, 1, 2048, 2048) when sparse mode = %u.",
                fiaInfo.sparseMode);
        OP_LOGE(fiaInfo.opName, "attenMask shape is (%s).", maskInfo.strMaskShape.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
// enableNonQuant 相关校验函数

// enableFullQuant 相关校验函数

// enableAntiQuant 相关校验函数

ge::graphStatus MaskChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin MaskChecker::CheckSinglePara!");
    if (ge::GRAPH_SUCCESS != CheckDtypeAndFormat(fiaInfo) || ge::GRAPH_SUCCESS != CheckSparseMode(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (enableNonQuant_) {
        ;
    } else if (enableFullQuant_) {
        ;
    } else if (enableAntiQuant_) {
        ;
    }
    OP_LOGI(fiaInfo.opName, "End MaskChecker::CheckSinglePara!");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin MaskChecker::CheckParaExistence!");
    if (enableNonQuant_) {
        ;
    } else if (enableFullQuant_) {
        ;
    } else if (enableAntiQuant_) {
        ;
    }
    OP_LOGI(fiaInfo.opName, "End MaskChecker::CheckParaExistence!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckFeature(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin MaskChecker::CheckFeature!");
    if (enableNonQuant_) {
        if (ge::GRAPH_SUCCESS != CheckNoQuantIFAMLA(fiaInfo) || ge::GRAPH_SUCCESS != CheckQKVDDifferent(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    } else if (enableFullQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFullQuantIFAMLA(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    } else if (enableAntiQuant_) {
        ;
    }
    OP_LOGI(fiaInfo.opName, "End MaskChecker::CheckFeature!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckMultiPara(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin MaskChecker::CheckMultiPara!");
    if (ge::GRAPH_SUCCESS != CheckPretokenAndNexttoken(fiaInfo) || ge::GRAPH_SUCCESS != CheckDimAndShape(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (enableNonQuant_) {
        ;
    } else if (enableFullQuant_) {
        ;
    } else if (enableAntiQuant_) {
        ;
    }
    OP_LOGI(fiaInfo.opName, "End MaskChecker::CheckMultiPara!");
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
