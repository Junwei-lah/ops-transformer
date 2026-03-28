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
            OP_CHECK_IF(fiaInfo.inputQType == ge::DT_FLOAT8_E4M3FN &&
                        !(((fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) && (fiaInfo.attenMaskFlag)) ||
                          ((fiaInfo.sparseMode == SPARSE_MODE_NO_MASK) && (!fiaInfo.attenMaskFlag))),
                        OP_LOGE(fiaInfo.opName,
                                "Only support sparse 3 with mask, or sparse 0 without mask when ifa mla and "
                                "query's sequence length is > 1, and input datatype is FLOAT8_E4M3, "
                                "input sparse mode is %d and there has%smask",
                                fiaInfo.sparseMode, fiaInfo.attenMaskFlag ? " " : " no "),
                        return ge::GRAPH_FAILED);
            
            OP_CHECK_IF(fiaInfo.inputQType == ge::DT_INT8 &&
                        !((fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) && (fiaInfo.attenMaskFlag)),
                        OP_LOGE(fiaInfo.opName,
                                "Only support sparse 3 with mask and "
                                "query's sequence length is > 1,and input datatype is INT8, "
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

ge::graphStatus MaskChecker::CheckPretokenAndNexttoken(const FiaTilingInfo &fiaInfo)
{
    // In PFA mode, the values of pretoken and nexttoken must ensure the mask range remains valid.
    isIFAFlag = (enableAntiQuant_) && (fiaInfo.s1Size == 1);
    ge::DataType outputType = fiaInfo.opParamInfo.attenOut.desc->GetDataType();
    if (isIFAFlag) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((fiaInfo.nextToken * (-1)) > fiaInfo.preToken,
                OP_LOGE(fiaInfo.opName,
                        "Nexttoken line should be higher than pretoken line, preTokens = %ld, nextTokens = %ld.",
                        fiaInfo.preToken, fiaInfo.nextToken),
                return ge::GRAPH_FAILED);
    // Check the specific conditions that pretoken and nexttoken must satisfy under the band mode.
    OP_CHECK_IF(
        (fiaInfo.antiQuantFlag && fiaInfo.sparseMode == SPARSE_MODE_BAND && outputType == ge::DT_INT8 &&
         ((fiaInfo.preToken < 0) || fiaInfo.nextToken < 0)),
        OP_LOGE(fiaInfo.opName,
                "When output type is int8, sparse mode = 4, preTokens (%ld) or nextTokens (%ld) cannot be negative.",
                fiaInfo.preToken, fiaInfo.nextToken),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckIFADimAndShape(const FiaTilingInfo &fiaInfo)
{
    uint32_t minAttenMaskSize = 0;
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    const gert::Tensor *maskShape = fiaInfo.opParamInfo.attenMask.tensor;
    size_t attenMaskDim = maskShape->GetStorageShape().GetDimNum();
    uint32_t attenMaskBatch = maskShape->GetStorageShape().GetDim(DIM_NUM_0);
    uint32_t attenMaskSize = maskShape->GetStorageShape().GetDim(maskShape->GetStorageShape().GetDimNum() - 1);
    if (fiaInfo.pageAttentionFlag) {
        minAttenMaskSize = fiaInfo.s2Size;
    } else {
        minAttenMaskSize = fiaInfo.maxActualseq;
    }
    if (attenMaskDim == MASK_DIM_SS) {
        if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B && (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK)) {
            if (fiaInfo.ropeMode == RopeMode::NO_ROPE ) {
                const std::vector<std::string> layoutSupportList = {
                    "BSH", "BSND", "BNSD", "BNSD_BSND",
                };
                OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layoutStr) == layoutSupportList.end(),
                    OP_LOGE(fiaInfo.opName,
                        "In gqa noquant situation, rope not exits and qkHeadDim = vHeadDim, when sparseMode = 0 or 1, "
                        "two dim mask only support for layout BSH,BSND,BNSD,BNSD_BSND, but got %s",
                        layoutStr.c_str()),
                    return ge::GRAPH_FAILED);
            } else {
                OP_LOGE(fiaInfo.opName,
                        "In gqa noquant situation, rope exits or qkHeadDim != vHeadDim, when sparseMode = 0 or 1, two dim mask is not supported.");
                return ge::GRAPH_FAILED;
            }
        } else {
            OP_LOGE(fiaInfo.opName,
                "The current dimension of the mask is 2. "
                "Please use 3D mask \[B,QS,KVS\]\/\[1,QS,KVS\] or 4D mask \[B,1,QS,KVS\]\/\[1,1,QS,KVS\].");
            return ge::GRAPH_FAILED;
        }
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
       if ((fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK) && fiaInfo.socVersion != platform_ascendc::SocVersion::ASCEND910B) {
            OP_LOGE(fiaInfo.opName,
                    "Attenmask does not support inputs with dim 2 when sparse mode = %u. "
                    "Please use 3D mask \[B,QS,KVS\]\/\[1,QS,KVS\] or 4D mask \[B,1,QS,KVS\]\/\[1,1,QS,KVS\].",
                    fiaInfo.sparseMode);
            return ge::GRAPH_FAILED;
        }
        else {
            if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B && (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK)) {
                if (fiaInfo.ropeMode == RopeMode::NO_ROPE) {
                    const std::vector<std::string> layoutSupportList = {
                        "BSH", "BSND", "BNSD", "BNSD_BSND",
                    };
                    std::string layoutStr(fiaInfo.opParamInfo.layOut);
                    std::string layout = layoutStr;
                    OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end(),
                        OP_LOGE(fiaInfo.opName,
                            "In gqa no quant situation, rope not exits and qkHeadDim = vHeadDim, when sparseMode = 0 or 1, "
                            "two dim mask only support for layout BSH,BSND,BNSD,BNSD_BSND, but got %s",
                            layout.c_str()),
                        return ge::GRAPH_FAILED);
                } else {
                    OP_LOGE(fiaInfo.opName,
                            "In gqa no quant situation, rope exits or qkHeadDim != vHeadDim, when sparseMode = 0 or 1, two dim mask is not supported.");
                    return ge::GRAPH_FAILED;
                }
            }
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
    if ((!fiaInfo.attenMaskFlag) && (fiaInfo.sparseMode != SPARSE_MODE_NO_MASK)) {
        OP_LOGE(fiaInfo.opName, "when sparse_mode is %d, it not 0, atten_mask should not be null.",
                fiaInfo.sparseMode);
        return ge::GRAPH_FAILED;
    }
    if ((fiaInfo.isMaxWorkspace && fiaInfo.socVersion != platform_ascendc::SocVersion::ASCEND910B) || !fiaInfo.attenMaskFlag) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B && (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK)) {
        if (fiaInfo.isMaxWorkspace) {
            return ge::GRAPH_SUCCESS;
        }
    }
    isIFAFlag = (fiaInfo.antiQuantFlag) && (fiaInfo.s1Size == 1);
    if (isIFAFlag) {
        return CheckIFADimAndShape(fiaInfo);
    }

    MaskInfo maskInfo;
    if (GetMaskInfo(fiaInfo, maskInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
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

ge::graphStatus MaskChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeAndFormat(fiaInfo) || ge::GRAPH_SUCCESS != CheckSparseMode(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckFeature(const FiaTilingInfo &fiaInfo)
{
    if (enableNonQuant_) {
        if (ge::GRAPH_SUCCESS != CheckNoQuantIFAMLA(fiaInfo) || ge::GRAPH_SUCCESS != CheckQKVDDifferent(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    } else if (enableFullQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFullQuantIFAMLA(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckMultiPara(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckPretokenAndNexttoken(fiaInfo) || ge::GRAPH_SUCCESS != CheckDimAndShape(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
