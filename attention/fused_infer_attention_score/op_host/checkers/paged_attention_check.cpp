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
 * \file paged_attention_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "paged_attention_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// 公共校验函数
ge::graphStatus PagedAttentionChecker::CheckBlockTableExistence(const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF(fiaInfo.opParamInfo.blockTable.tensor == nullptr,
        OP_LOGE(fiaInfo.opName,
                "When page attention enable, block table should not be null."),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckFeatureExistence(const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF((fiaInfo.opParamInfo.queryPaddingSize.tensor != nullptr) ||
        (fiaInfo.opParamInfo.kvPaddingSize.tensor != nullptr),
        OP_LOGE(fiaInfo.opName,
            "When page attention enable, left padding is not supported."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.opParamInfo.keySharedPrefix.tensor != nullptr) ||
        (fiaInfo.opParamInfo.valueSharedPrefix.tensor != nullptr),
        OP_LOGE(fiaInfo.opName,
            "When page attention enable, system prefix is not supported."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.isQKVDDifferent,
        OP_LOGE(fiaInfo.opName,
            "When page attention enable, the head dim of query and key should be equal."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckSeqLengthKVExistence(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(fiaInfo.opParamInfo.actualSeqLengths.tensor == nullptr ||
        fiaInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>() == nullptr ||
            fiaInfo.opParamInfo.actualSeqLengths.tensor->GetShapeSize() == 0,
        OP_LOGE(fiaInfo.opName,
            "When page attention enable, actualSeqLengths of KV should not be null."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

int64_t PagedAttentionChecker::GetMaxBlockNumPerBatch(const FiaTilingInfo &fiaInfo)
{
    const int32_t blockSize = fiaInfo.blockSize;
    const gert::Tensor* actSeqLenKV = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    uint32_t actualSeqLengthsKVSize = static_cast<uint32_t>(actSeqLenKV->GetShapeSize());
    int64_t actualSeqKVPerBatch = 0;
    int64_t blockNumPerBatch = 0;
    int64_t maxBlockNumPerBatch = 0;
    uint32_t loop = std::min(actualSeqLengthsKVSize, fiaInfo.bSize);

    for (uint32_t i = 0; i < loop; i++) {
        actualSeqKVPerBatch = actSeqLenKV->GetData<int64_t>()[i];
        blockNumPerBatch = (actualSeqKVPerBatch + blockSize - 1) / blockSize;
        if (blockNumPerBatch > maxBlockNumPerBatch) {
            maxBlockNumPerBatch = blockNumPerBatch;
        }
    }
    return maxBlockNumPerBatch;
}

// check mask shape
ge::graphStatus PagedAttentionChecker::CheckMaskShape(const FiaTilingInfo &fiaInfo)
{
    if ((fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK) &&
        fiaInfo.opParamInfo.attenMask.tensor != nullptr) {
        if (fiaInfo.opParamInfo.actualSeqLengths.tensor == nullptr) {
            return ge::GRAPH_SUCCESS;
        }
        const gert::Shape attenMaskShape = fiaInfo.opParamInfo.attenMask.tensor->GetStorageShape();
        int64_t maxBlockNumPerBatch = GetMaxBlockNumPerBatch(fiaInfo);
        uint32_t attenMaskDimNum = attenMaskShape.GetDimNum();
        OP_CHECK_IF(attenMaskDimNum > 0 &&
            (attenMaskShape.GetDim(attenMaskDimNum-1) < maxBlockNumPerBatch * fiaInfo.blockSize),
            OP_LOGE(fiaInfo.opName,
                "When page attention enable and attenMask enable (sparseMode = %u), the last dimension of "
                "attenMask(%ld) shoule be greater than maxBlockNumPerBatch(%ld) * blockSize(%d).",
                fiaInfo.sparseMode, attenMaskShape.GetDim(attenMaskDimNum-1), maxBlockNumPerBatch, fiaInfo.blockSize),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// check blocktable dtype
ge::graphStatus PagedAttentionChecker::CheckBlockTableDtype(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.opParamInfo.blockTable.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::CompileTimeTensorDesc *blockTableDesc = fiaInfo.opParamInfo.blockTable.desc;
    OP_CHECK_IF(blockTableDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(fiaInfo.opName, "When page attention enable, blockTable datatype only support INT32."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check pa cache shape
ge::graphStatus PagedAttentionChecker::CheckPACacheShape(const FiaTilingInfo &fiaInfo, const gert::Shape tempShape,
    const std::string& inputName)
{
    uint32_t shapeDim = tempShape.GetDimNum();
    int64_t tempBlockSize = 0;
    int64_t tempH = 0;
    int64_t tempN = 0;
    int64_t tempD = 0;
    int64_t tempD0 = 0;
    int64_t tempD1 = 0;

    uint32_t compareD = 0;
    if (inputName == "key") {
        compareD = fiaInfo.qkHeadDim;
    } else if (inputName == "keyRope") {
        compareD = fiaInfo.ropeHeadDim;
    } else {
        compareD = fiaInfo.vHeadDim;
    }

    if (shapeDim == DIM_NUM_3) { // [blockNums, blockSize, H]
        tempBlockSize = tempShape.GetDim(DIM_NUM_1);
        tempH = tempShape.GetDim(DIM_NUM_2);
        OP_CHECK_IF(tempBlockSize != fiaInfo.blockSize,
            OP_LOGE(fiaInfo.opName, "When page attention enable, blocksize of %s(%u) should be %u",
                inputName.c_str(), tempBlockSize, fiaInfo.blockSize),
            return ge::GRAPH_FAILED);

        if (fiaInfo.inputKvType == ge::DT_INT4) {
            OP_CHECK_IF(tempH != fiaInfo.n2Size * compareD,
                OP_LOGE(fiaInfo.opName, "When page attention enable, if input kv dataType is INT32, "
                    "the axis H of %s(%u) should be %u, "
                    "if input kv dataType is INT4, the axis H of %s(%u) should be %u",
                    inputName.c_str(), tempH / NUM8, fiaInfo.n2Size * compareD / NUM8,
                    inputName.c_str(), tempH, fiaInfo.n2Size * compareD),
                return ge::GRAPH_FAILED);

            OP_CHECK_IF(tempH > H_LIMIT,
                OP_LOGE(fiaInfo.opName, "When page attention enable and layout is BSH, "
                    "if input kv dataType is INT32, the axis H of %s(%ld) should not > %u."
                    "if input kv dataType is INT4, the axis H of %s(%ld) should not > %u.",
                    inputName.c_str(), tempH / NUM8, H_LIMIT / NUM8, inputName.c_str(), tempH, H_LIMIT),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(tempH != fiaInfo.n2Size * compareD,
                OP_LOGE(fiaInfo.opName, "When page attention enable, the axis H(%u) of kvCache should be %u",
                    tempH, fiaInfo.n2Size * compareD),
                return ge::GRAPH_FAILED);

            OP_CHECK_IF(tempH > H_LIMIT,
                OP_LOGE(fiaInfo.opName, "When page attention enable and layout is BSH, "
                    "the axis H of %s(%ld) should not > %u.", inputName.c_str(), tempH, H_LIMIT),
                return ge::GRAPH_FAILED);
        }
    } else if (shapeDim == DIM_NUM_4) { // [blockNums, N, blockSize, D]
        tempN = tempShape.GetDim(DIM_NUM_1);
        tempBlockSize = tempShape.GetDim(DIM_NUM_2);
        tempD = tempShape.GetDim(DIM_NUM_3);

        OP_CHECK_IF(tempN != fiaInfo.n2Size,
            OP_LOGE(fiaInfo.opName, "When page attention enable, the axis N(%u) of kvCache should be %u",
                tempN, fiaInfo.n2Size),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(tempBlockSize != fiaInfo.blockSize,
            OP_LOGE(fiaInfo.opName, "When page attention enable, blocksize(%u) of kvCache should be %u",
                tempBlockSize, fiaInfo.blockSize),
            return ge::GRAPH_FAILED);
        
        if (fiaInfo.inputKvType == ge::DT_INT4) {
            OP_CHECK_IF(tempD != compareD,
                OP_LOGE(fiaInfo.opName, "When page attention enable, if input kv dataType is INT32, "
                    "the axis D of %s(%u) should be %u, "
                    "if input kv dataType is INT4, the axis D of %s(%u) should be %u",
                    inputName.c_str(), tempD / NUM8, compareD / NUM8,
                    inputName.c_str(), tempD, compareD),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(tempD != compareD,
                OP_LOGE(fiaInfo.opName, "When page attention enable, the axis D of %s(%u) should be %u",
                    inputName.c_str(), tempD, compareD),
                return ge::GRAPH_FAILED);
        }
    } else { // [blockNums, N, D1, blocksize, D0]
        tempN = tempShape.GetDim(DIM_NUM_1);
        tempD1 = tempShape.GetDim(DIM_NUM_2);
        tempBlockSize = tempShape.GetDim(DIM_NUM_3);
        tempD0 = tempShape.GetDim(DIM_NUM_4);
        const ge::DataType inputKvType = fiaInfo.inputKvType;
        uint32_t d0Size = NUM_16; // D0 = 16

        OP_CHECK_IF(tempN != fiaInfo.n2Size,
            OP_LOGE(fiaInfo.opName, "When page attention enable, the axis N of %s(%u) should be %u",
                inputName.c_str(), tempN, fiaInfo.n2Size),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(tempBlockSize != fiaInfo.blockSize,
            OP_LOGE(fiaInfo.opName, "When page attention enable, blocksize of %s(%u) should be %u",
                inputName.c_str(), tempBlockSize, fiaInfo.blockSize),
            return ge::GRAPH_FAILED);

        if (enableAntiQuant_) {
            d0Size = NUM_16;
            if (fiaInfo.inputKvType == ge::DT_INT4) {
                OP_CHECK_IF(tempD0 != d0Size,
                    OP_LOGE(fiaInfo.opName, "When PA_NZ enable, if input kv dataType is INT32, "
                        "the last dim (D0) of kvCache(%u) should be %u; "
                        "if input kv dataType is INT4, the last dim (D0) of %s(%u) should be %u",
                        inputName.c_str(), tempD0/NUM8, d0Size/NUM8, tempD0, d0Size),
                    return ge::GRAPH_FAILED);
            } else {
                OP_CHECK_IF(tempD0 != d0Size,
                    OP_LOGE(fiaInfo.opName, "When PA_NZ enable, the last dim (D0) of %s(%u) should be %u",
                        inputName.c_str(), tempD0, d0Size),
                    return ge::GRAPH_FAILED);
            }
        } else {
            std::unordered_map<ge::DataType, float> typeSizeMap = {{ge::DT_FLOAT16, FLOAT16SIZE},
                {ge::DT_BF16, BFLOAT16SIZE}, {ge::DT_INT8, INT8SIZE}, {ge::DT_HIFLOAT8, FLOAT8SIZE},
                {ge::DT_FLOAT8_E4M3FN, FLOAT8SIZE}};
            uint32_t dataTypeSizeValue = FLOAT16SIZE;
            auto inputTypeCheck = typeSizeMap.find(fiaInfo.inputKvType);
            if (inputTypeCheck != typeSizeMap.end()) {
                dataTypeSizeValue = inputTypeCheck->second;
            }

            if (enableFullQuant_ && inputName == "keyRope") {
                dataTypeSizeValue = BFLOAT16SIZE;
            }

            d0Size = BYTE_BLOCK / dataTypeSizeValue;
            OP_CHECK_IF(tempD0 != d0Size,
                OP_LOGE(fiaInfo.opName, "When PA_NZ enable, the last dim (D0) of %s(%u) should be %u",
                    inputName.c_str(), tempD0, d0Size),
                return ge::GRAPH_FAILED);
        }

        OP_CHECK_IF(tempD1 != compareD / d0Size,
            OP_LOGE(fiaInfo.opName, "When PA_NZ enable, the third dim (D1) of %s(%u) should be %u",
                inputName.c_str(), tempD1, compareD / d0Size),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// check input kv latyout
ge::graphStatus PagedAttentionChecker::CheckKVDtypeSupport(const FiaTilingInfo &fiaInfo)
{
    const ge::DataType inputKvType = fiaInfo.inputKvType;
    const std::vector<ge::DataType> dtypeSupportList = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT4,
        ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1};
    if (std::find(dtypeSupportList.begin(), dtypeSupportList.end(), inputKvType) == dtypeSupportList.end()) {
            OP_LOGE(fiaInfo.opName, "When page attention enable, the datatype of kv only supports "
            "FLOAT16/BFLOAT16/INT8/INT4(INT32)/HIFLOAT8/FLOAT8_E4M3FN/FLOAT4_E2M1.");
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckBlockTableShape(const FiaTilingInfo &fiaInfo)
{
    const gert::Shape blockTableShape = fiaInfo.opParamInfo.blockTable.tensor->GetStorageShape();

    // check dim num
    OP_CHECK_IF(blockTableShape.GetDimNum() != 2,
        OP_LOGE(fiaInfo.opName, "When page attention enable, the dim num(%zu) of blockTable should be 2.",
                blockTableShape.GetDimNum()),
            return ge::GRAPH_FAILED);

    // check blockTable each dim cannot be 0
    OP_CHECK_IF(blockTableShape.GetShapeSize() == 0,
        OP_LOGE(fiaInfo.opName, "When page attention enable, blockTable each dim can not be 0, now is [%ld, %ld].",
            blockTableShape.GetDim(0), blockTableShape.GetDim(1)),
        return ge::GRAPH_FAILED);
    
    if (fiaInfo.opParamInfo.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    // 每个batch的 blockNum 小于 blocktable dim2
    int64_t maxBlockNumPerBatch = GetMaxBlockNumPerBatch(fiaInfo);

    OP_CHECK_IF(((blockTableShape.GetDim(0) != fiaInfo.bSize) || (blockTableShape.GetDim(1) < maxBlockNumPerBatch)),
        OP_LOGE(fiaInfo.opName,
            "When page attention enable, block table shape should be [%u, >=%ld], now is [%ld, %ld].",
            fiaInfo.bSize, maxBlockNumPerBatch, blockTableShape.GetDim(0), blockTableShape.GetDim(1)),
        return ge::GRAPH_FAILED);

    // check key cache shape
    if (ge::GRAPH_SUCCESS != CheckPACacheShape(fiaInfo, fiaInfo.opParamInfo.key.shape->GetStorageShape(), "key") ||
        ge::GRAPH_SUCCESS != CheckPACacheShape(fiaInfo, fiaInfo.opParamInfo.value.shape->GetStorageShape(), "value")) {
        return ge::GRAPH_FAILED;
    }

    // check rope cache shape
    if (fiaInfo.ropeMode == RopeMode::ROPE_SPLIT) {
        if (ge::GRAPH_SUCCESS != CheckPACacheShape(fiaInfo, fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape(),
            "keyRope")) {
            return ge::GRAPH_FAILED;
        }
    }

    // warning: S2 <= 20M
    if (maxBlockNumPerBatch * fiaInfo.blockSize > S_LIMIT) {
        OP_LOGW(fiaInfo.opName,
            "When page attention enable, sequence length(%ld) of kv should <= 20M.",
            maxBlockNumPerBatch * fiaInfo.blockSize);
    }
    return ge::GRAPH_SUCCESS;
}

// check blocksize
ge::graphStatus PagedAttentionChecker::CheckBlockSize(const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF(fiaInfo.opParamInfo.blockSize == nullptr,
        OP_LOGE(fiaInfo.opName,
            "When page attention enable, blockSize shuld not be null."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.blockSize == 0,
        OP_LOGE(fiaInfo.opName,
            "When page attention enable, blockSize should not be 0."),
        return ge::GRAPH_FAILED);

    if (enableNonQuant_) {
        OP_CHECK_IF(fiaInfo.blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT || fiaInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_16,
            OP_LOGE(fiaInfo.opName,
                "In the No_Quant scenario, when page attention enable,  blockSize(%d) should be in range of [%u, %u].",
                fiaInfo.blockSize, BLOCK_SIZE_ALIGN_SIZE_16, BLOCK_SIZE_MAX_FOR_NO_QUANT),
        return ge::GRAPH_FAILED);

        OP_CHECK_IF(fiaInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0,
            OP_LOGE(fiaInfo.opName,
                "In the NO_QUANT scenario, when page attention enable, blockSize(%d) should be a multiple of %u.",
                fiaInfo.blockSize, BLOCK_SIZE_ALIGN_SIZE_16),
            return ge::GRAPH_FAILED);
    } else {
        if (fiaInfo.s1Size > 1) {
            OP_CHECK_IF(fiaInfo.blockSize > BLOCK_SIZE_MAX || fiaInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_128,
                OP_LOGE(fiaInfo.opName,
                    "In the ANTI_QUANT or FULL_QUANT scenario (Q_S > 1), when page attention enable, "
                    "blockSize(%d) should be in range of [%u, %u].",
                    fiaInfo.blockSize, BLOCK_SIZE_ALIGN_SIZE_128, BLOCK_SIZE_MAX),
                return ge::GRAPH_FAILED);

            OP_CHECK_IF(fiaInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_128 != 0,
                OP_LOGE(fiaInfo.opName,
                    "In the ANTI_QUANT or FULL_QUANT scenario (Q_S > 1), when page attention enable, "
                    "blockSize(%d) should be a multiple of %d.",
                    fiaInfo.blockSize, BLOCK_SIZE_ALIGN_SIZE_128),
                return ge::GRAPH_FAILED);
        } else {
            std::unordered_map<ge::DataType, float> typeSizeMap = {{ge::DT_FLOAT16, FLOAT16SIZE},
                {ge::DT_BF16, BFLOAT16SIZE}, {ge::DT_INT8, INT8SIZE}, {ge::DT_HIFLOAT8, FLOAT8SIZE},
                {ge::DT_FLOAT8_E4M3FN, FLOAT8SIZE}, {ge::DT_INT4, INT4SIZE}, {ge::DT_FLOAT4_E2M1, FLOAT4SIZE}};
            uint32_t dataTypeSizeValue = FLOAT16SIZE;
            auto inputTypeCheck = typeSizeMap.find(fiaInfo.inputKvType);
            if (inputTypeCheck != typeSizeMap.end()) {
                dataTypeSizeValue = inputTypeCheck->second;
            }
            uint32_t blockSizeAlign = static_cast<uint32_t>(BYTE_BLOCK / dataTypeSizeValue);

            OP_CHECK_IF(fiaInfo.blockSize > BLOCK_SIZE_MAX || fiaInfo.blockSize < blockSizeAlign,
                OP_LOGE(fiaInfo.opName,
                    "In the ANTI_QUANT or FULL_QUANT scenario (Q_S = 1), when page attention enable, "
                    "blockSize(%d) should be in range of [%u, %u].",
                    fiaInfo.blockSize, blockSizeAlign, BLOCK_SIZE_MAX),
                return ge::GRAPH_FAILED);

            OP_CHECK_IF(fiaInfo.blockSize % blockSizeAlign != 0,
                OP_LOGE(fiaInfo.opName,
                    "In the ANTI_QUANT or FULL_QUANT scenario (Q_S = 1), when page attention enable, "
                    "blockSize(%d) should be a multiple of %u.",
                    fiaInfo.blockSize, blockSizeAlign),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckPADimNum(const FiaTilingInfo &fiaInfo)
{
    const string inputLayout = fiaInfo.opParamInfo.layOut;
    const uint32_t dimNum = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
    if (inputLayout == "BSH" || inputLayout == "BSND") {
        OP_CHECK_IF(dimNum == 4,
            OP_LOGE(fiaInfo.opName,
                "When page attention enable and input layout is %s, PA BnNBsD is not supported.", inputLayout.c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin PagedAttentionChecker::CheckSinglePara!");

    if (!fiaInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }

    if (ge::GRAPH_SUCCESS != CheckBlockTableDtype(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckKVDtypeSupport(fiaInfo)) {
            return ge::GRAPH_FAILED;
    }

    if (enableNonQuant_) {
        ;
    } else if (enableFullQuant_) {
        ;
    } else if (enableAntiQuant_) {
        ;
    }
    OP_LOGI(fiaInfo.opName, "End PagedAttentionChecker::CheckSinglePara!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin PagedAttentionChecker::CheckParaExistence!");

    if (!fiaInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }

    if (ge::GRAPH_SUCCESS != CheckBlockTableExistence(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureExistence(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckSeqLengthKVExistence(fiaInfo)) {
            return ge::GRAPH_FAILED;
    }

    if (enableNonQuant_) {
        ;
    } else if (enableFullQuant_) {
        ;
    } else if (enableAntiQuant_) {
        ;
    }
    OP_LOGI(fiaInfo.opName, "End PagedAttentionChecker::CheckParaExistence!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckFeature(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin PagedAttentionChecker::CheckFeature!");

    if (!fiaInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckMaskShape(fiaInfo)) {
            return ge::GRAPH_FAILED;
    }

    if (enableNonQuant_) {
        ;
    } else if (enableFullQuant_) {
        ;
    } else if (enableAntiQuant_) {
        ;
    }
    OP_LOGI(fiaInfo.opName, "End PagedAttentionChecker::CheckFeature!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckMultiPara(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin PagedAttentionChecker::CheckMultiPara!");

    if (!fiaInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }

    if (ge::GRAPH_SUCCESS != CheckBlockSize(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckPADimNum(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckBlockTableShape(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }

    if (enableNonQuant_) {
        ;
    } else if (enableFullQuant_) {
        ;
    } else if (enableAntiQuant_) {
        ;
    }
    OP_LOGI(fiaInfo.opName, "End PagedAttentionChecker::CheckMultiPara!");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling