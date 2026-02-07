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
 * \file fused_infer_attention_score_tiling_check_feature.cpp
 * \brief
 */

#include <numeric>
#include "tiling/tiling_api.h"
#include "fused_infer_attention_score_tiling_check.h"

using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus FiaTilingCheck::CheckFeatureNoQuantDtype() const
{
    if (quantMode_ != FiaQuantMode::NO_QUANT) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(inputQType_ != ge::DT_BF16 && inputQType_ != ge::DT_FLOAT16,
        OP_LOGE(opName_, "In %s situation, query dtype only support %s and %s, but got %s",
            QuantModeToSerialString(quantMode_).c_str(),
            FusedDataTypeToSerialString(ge::DT_BF16).c_str(), FusedDataTypeToSerialString(ge::DT_FLOAT16).c_str(),
            FusedDataTypeToSerialString(inputQType_).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(inputQType_ != inputKvType_,
        OP_LOGE(opName_, "In %s situation, key and value dtype(%s) must equal to query dtype(%s)",
            QuantModeToSerialString(quantMode_).c_str(),
            FusedDataTypeToSerialString(inputQType_).c_str(),
            FusedDataTypeToSerialString(inputKvType_).c_str()),
        return ge::GRAPH_FAILED);

    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        OP_CHECK_IF((opParamInfo_.queryRope.desc->GetDataType() != opParamInfo_.query.desc->GetDataType()),
            OP_LOGE(opName_, "%s(%s) and %s(%s) must have same dType",
                QUERY_ROPE_NAME.c_str(), FusedDataTypeToSerialString(opParamInfo_.queryRope.desc->GetDataType()).c_str(),
                QUERY_NAME.c_str(), FusedDataTypeToSerialString(opParamInfo_.query.desc->GetDataType()).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF((opParamInfo_.keyRope.desc->GetDataType() != opParamInfo_.key.desc->GetDataType()),
            OP_LOGE(opName_, "%s(%s) and %s(%s) must have same dType",
                KEY_ROPE_NAME.c_str(), FusedDataTypeToSerialString(opParamInfo_.keyRope.desc->GetDataType()).c_str(),
                KEY_NAME.c_str(), FusedDataTypeToSerialString(opParamInfo_.key.desc->GetDataType()).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureBlockSize() const
{
    constexpr int32_t BLOCK_SIZE_ALIGN_SIZE = 16;
    constexpr int32_t BLOCK_SIZE_MAX_SIZE = 1024;
    if (blockSize_ % BLOCK_SIZE_ALIGN_SIZE != 0) {
        OP_LOGE(opName_, "In %s situation, %s should aligned to 16, but got %d.",
            QuantModeToSerialString(quantMode_).c_str(), BLOCK_SIZE_NAME.c_str(), blockSize_);
            return ge::GRAPH_FAILED;
    }

    if (blockSize_ > BLOCK_SIZE_MAX_SIZE) {
        OP_LOGE(opName_, "In %s situation, %s should less equal than 1024, but got %d.",
            QuantModeToSerialString(quantMode_).c_str(), BLOCK_SIZE_NAME.c_str(), blockSize_);
            return ge::GRAPH_FAILED;
    }

    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION && blockSize_ == 0) {
        OP_LOGE(opName_, "In %s and storage mode is page attention, %s should not be 0",
            QuantModeToSerialString(quantMode_).c_str(), BLOCK_SIZE_NAME.c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureLse() const
{
    if (!fiaInfo_.softmaxLseFlag) {
        return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureTensorList() const
{
    if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        const std::vector<std::string> layoutSupportList = {
            "BSND", "BNSD", "BSH", "BNSD_BSND",
        };
        std::string layout = opParamInfo_.layOut;
        if (std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end()) {
            OP_LOGE(opName_,
                "when the tensor number of key/value is greater than 1 (this scenario is called tensor list), "
                "input_layout only supports BSH, BSND, BNSD, and BNSD_BSND, but got %s", layout.c_str());
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF(ropeMode_ != RopeMode::NO_ROPE,
            OP_LOGE(opName_,
                "when the tensor number of key/value is greater than 1 (this scenario is called tensor list), "
                "query_rope and key_rope should be not exist and the head_dim(D) dimension of query and key should be "
                "equal to the head_dim(D) dimension of value."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoquant()
{
    OP_CHECK_IF(socVersion_ == platform_ascendc::SocVersion::ASCEND310P,
        OP_LOGE(opName_, "In %s %s situation, Ascend310P is not supported",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str()),
        return ge::GRAPH_FAILED);
    if (ge::GRAPH_SUCCESS != CheckFeatureTensorList() ||
        ge::GRAPH_SUCCESS != CheckFeatureBlockSize() ||
        ge::GRAPH_SUCCESS != CheckFeatureInOutDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckFeatureMask() ||
        ge::GRAPH_SUCCESS != CheckFeatureNoQuantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureLse() ||
        ge::GRAPH_SUCCESS != CheckFeatureLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureAxisInfo() ||
        ge::GRAPH_SUCCESS != CheckFeatureLearnableSink() ||
        ge::GRAPH_SUCCESS != CheckFeatureHeadDim()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaAntiquant() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaFullquant() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMla()
{
    if (quantMode_ == FiaQuantMode::NO_QUANT) {
        return CheckFeatureMlaNoquant();
    } else if (quantMode_ == FiaQuantMode::ANTI_QUANT) {
        return CheckFeatureMlaAntiquant();
    } else if (quantMode_ == FiaQuantMode::FULL_QUANT) {
        return CheckFeatureMlaFullquant();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMask() const
{
    if ((!attenMaskFlag_) && (fiaInfo_.sparseMode != SPARSE_MODE_NO_MASK)) {
        OP_LOGE(opName_, "when %s is %d, it not 0, %s should not be null.",
            SPARSE_MODE_NAME.c_str(), fiaInfo_.sparseMode, ATTEN_MASK_NAME.c_str());
        return ge::GRAPH_FAILED;
    }

    if (attenMaskFlag_) {
        size_t maskDimNum = opParamInfo_.attenMask.tensor->GetStorageShape().GetDimNum();
        if ((fiaInfo_.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo_.sparseMode == SPARSE_MODE_ALL_MASK) && 
            maskDimNum == DIM_NUM_TWO) {
            if (ropeMode_ == RopeMode::NO_ROPE) {
                const std::vector<std::string> layoutSupportList = {
                    "BSH", "BSND", "BNSD", "BNSD_BSND",
                };
                std::string layout = opParamInfo_.layOut;
                OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end(),
                    OP_LOGE(opName_,
                        "In %s situation, rope not exits and qkHeadDim = vHeadDim, when sparseMode = 0 or 1, "
                        "two dim mask only support for layout BSH,BSND,BNSD,BNSD_BSND, but got %s",
                        QuantModeToSerialString(quantMode_).c_str(), layout.c_str()),
                    return ge::GRAPH_FAILED);
            } else {
                OP_LOGE(opName_,
                        "In %s situation, rope exits or qkHeadDim != vHeadDim, when sparseMode = 0 or 1, two dim mask is not supported.",
                        QuantModeToSerialString(quantMode_).c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }

    if (ropeMode_ == RopeMode::ROPE_SPLIT && vHeadDim_ == 512U) {
        int32_t sparseMode = fiaInfo_.sparseMode;
        if (sparseMode != SPARSE_MODE_NO_MASK && sparseMode != SPARSE_MODE_RIGHT_DOWN && sparseMode != SPARSE_MODE_BAND) {
            OP_LOGE(opName_,
                    "In %s situation, when query_rope and key_rope exsists and the head dim of value is %u, %s only "
                    "support 0/3/4, but got %d.",
                    QuantModeToSerialString(quantMode_).c_str(), vHeadDim_, SPARSE_MODE_NAME.c_str(), sparseMode);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureLeftPadding() const
{
    if (fiaInfo_.qPaddingSizeFlag || fiaInfo_.kvPaddingSizeFlag) {
        const std::vector<std::string> layoutSupportList = {
            "BSND", "BNSD", "BSH", "BNSD_BSND",
        };
        std::string layout = opParamInfo_.layOut;
        if (std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end()) {
            OP_LOGE(opName_,
                    "when query_padding_size or kv_padding_size exists, input_layout only supports BSH, BSND, BNSD, "
                    "and BNSD_BSND, but got %s",
                    layout.c_str());
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF(ropeMode_ != RopeMode::NO_ROPE,
            OP_LOGE(opName_,
                    "when query_padding_size or kv_padding_size exists, query_rope and key_rope should be not exist "
                    "and the "
                    "head_dim(D) dimension of query and key should be equal to the head_dim(D) dimension of value."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvStorageMode_ == KvStorageMode::TENSOR_LIST,
            OP_LOGE(opName_,
                "when query_padding_size or kv_padding_size exists, key/value tensorlist is not suppoprted; in this "
                "case, the tensor number of key/value should be 1"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvStorageMode_ == KvStorageMode::PAGE_ATTENTION,
            OP_LOGE(opName_,
                "when query_padding_size or kv_padding_size exists, page attention is not suppoprted; in this case, "
                "block_table should exist and block_size is not 0"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(fiaInfo_.sysPrefixFlag,
            OP_LOGE(opName_,
                    "when query_padding_size exists, key_shared_prefix and key_shared_prefix should be not exist."),
            return ge::GRAPH_FAILED);
    }

    if (fiaInfo_.qPaddingSizeFlag) {
        OP_CHECK_IF(!(fiaInfo_.actualLenQDims != 0U),
            OP_LOGE(opName_, "when query_padding_size exists, the query's actual sequence lengths are required."),
            return ge::GRAPH_FAILED);
    }

    if (fiaInfo_.kvPaddingSizeFlag) {
        OP_CHECK_IF(!(fiaInfo_.actualLenDims != 0U),
            OP_LOGE(opName_, "when kv_padding_size exists, the key/value's actual sequence lengths are required."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeaturePSE() const
{
    if (fiaInfo_.pseShiftFlag) {
        const std::vector<std::string> layoutSupportList = {
            "BSND", "BNSD", "BSH", "BNSD_BSND",
        };
        std::string layout = opParamInfo_.layOut;
        if (std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end()) {
            OP_LOGE(opName_,
                    "when pse_shift exists, input_layout only supports BSH, BSND, BNSD, and BNSD_BSND, but got %s",
                    layout.c_str());
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF(ropeMode_ != RopeMode::NO_ROPE,
            OP_LOGE(opName_, "when pse_shift exists, query_rope and key_rope should be not exist and the head_dim(D) "
                             "dimension of query and key should be equal to the head_dim(D) dimension of value."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureLearnableSink() const
{
    if (fiaInfo_.learnableSinkFlag == false) {
        return ge::GRAPH_SUCCESS;
    }

    const std::vector<size_t> sinkDimNumList = {DIM_NUM_ONE};
    if (ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.learnableSink.tensor, sinkDimNumList, LEARNABLE_SINK_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t sinkDim = opParamInfo_.learnableSink.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(sinkDim != fiaInfo_.n1Size,
        OP_LOGE(opName_, "learnable_sink enable, sink shape(%u) must be same equal queryN(%u)!", sinkDim, fiaInfo_.n1Size),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((opParamInfo_.learnableSink.desc->GetDataType() != ge::DT_BF16),
            OP_LOGE(opName_, "When learnable_sink enable, sink dtype must be bf16!"),
            return ge::GRAPH_FAILED);
            
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureLayout() const
{
    const std::vector<std::string> layoutSupportList = {
        "BSH", "BSND", "BNSD", "TND", "NTD", "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD", "NTD_TND", "BSH_BNSD", "BSND_BNSD", "BNSD_BSND"
    };
    std::string layout = opParamInfo_.layOut;
    OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end(),
        OP_LOGE(opName_, "In %s %s situation, layout only supports BSH, BSND, BNSD, TND, NTD, BSH_NBSD, BSND_NBSD, BNSD_NBSD, TND_NTD, NTD_TND, BSH_BNSD, BSND_BNSD, BNSD_BSND, but got %s",
            QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), layout.c_str()),
        return ge::GRAPH_FAILED);

    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        OP_CHECK_IF(kvLayout_ != FiaLayout::BSH && kvLayout_ != FiaLayout::BSND && kvLayout_ != FiaLayout::BNSD &&
            kvLayout_ != FiaLayout::TND && kvLayout_ != FiaLayout::NTD,
            OP_LOGE(opName_, "In %s %s situation, key/value's layout only support BSH, BSND, BNSD, TND and NTD in batch continuous scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), LayoutToSerialString(kvLayout_).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvLayout_ != qLayout_,
            OP_LOGE(opName_, "In %s %s situation, key/value's layout and query's layout should be same in batch continuous scene.",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str()),
            return ge::GRAPH_FAILED);
    } else if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        OP_CHECK_IF(kvLayout_ != FiaLayout::BSH && kvLayout_ != FiaLayout::BSND && kvLayout_ != FiaLayout::BNSD,
            OP_LOGE(opName_, "In %s %s situation, key/value's layout only support BSH, BSND and BNSD in tensor list scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), LayoutToSerialString(kvLayout_).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvLayout_ != qLayout_,
            OP_LOGE(opName_, "In %s %s situation, key/value's layout and query's layout should be same in tensor list scene.",
            QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str()),
            return ge::GRAPH_FAILED);
    } else if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
        OP_CHECK_IF(kvLayout_ == FiaLayout::BnBsH && (qLayout_ != FiaLayout::BSH && qLayout_ != FiaLayout::BSND &&
                        qLayout_ != FiaLayout::BNSD && qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD),
            OP_LOGE(opName_, "In %s %s situation, the key/value's layout is BnBsH, %s layout must be BSH, BSND, BNSD TND and TND in page attention scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvLayout_ == FiaLayout::BnNBsD && (qLayout_ != FiaLayout::BSH && qLayout_ != FiaLayout::BSND &&
                        qLayout_ != FiaLayout::BNSD && qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD),
            OP_LOGE(opName_, "In %s %s situation, the key/value's layout is BnNBsD, %s layout must be BSH, BSND, BNSD TND and TND in page attention scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvLayout_ == FiaLayout::NZ && (qLayout_ != FiaLayout::BSH && qLayout_ != FiaLayout::BSND &&
                        qLayout_ != FiaLayout::BNSD && qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD),
            OP_LOGE(opName_, "In %s %s situation, the key/value's layout is BnNBsD, %s layout must be BSH, BSND, BNSD TND and TND in page attention scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureAxisInfo() const
{
    constexpr uint32_t MAX_ACTUAL_SEQ_LEN_BYTE = 64U * 1024U;
    constexpr uint32_t MAX_B_SIZE = 256U;

    OP_CHECK_IF(actualSeqLengthsQSize_ > MAX_ACTUAL_SEQ_LEN_BYTE,
    OP_LOGE(opName_, "In %s situation, actual sequence length q should be smaller or equal to 64K, but got %u",
        QuantModeToSerialString(quantMode_).c_str(), actualSeqLengthsQSize_),
    return ge::GRAPH_FAILED);

    OP_CHECK_IF(actualSeqLengthsKvSize_ > MAX_ACTUAL_SEQ_LEN_BYTE,
    OP_LOGE(opName_, "In %s situation, actual sequence length kv should be smaller or equal to 64K, but got %u",
        QuantModeToSerialString(quantMode_).c_str(), actualSeqLengthsKvSize_),
    return ge::GRAPH_FAILED);

    if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        OP_CHECK_IF(bSize_ > MAX_B_SIZE,
            OP_LOGE(opName_, "In %s situation, batch size(%u) cannot be greater than %u in tensor list scene.",
                QuantModeToSerialString(quantMode_).c_str(), bSize_, MAX_B_SIZE),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaPrefix() const
{
    if (!fiaInfo_.sysPrefixFlag) {
        return ge::GRAPH_SUCCESS;
    }
    const std::vector<std::string> layoutSupportList = {
        "BSND", "BNSD", "BSH", "BNSD_BSND",
    };
    std::string layout = opParamInfo_.layOut;
    if (std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end()) {
        OP_LOGE(opName_,
                "when system prefix exists, input_layout only supports BSH, BSND, BNSD, and BNSD_BSND, but got %s",
                layout.c_str());
        return ge::GRAPH_FAILED;
    }
    int32_t sparseMode = *opParamInfo_.sparseMode;
    auto *maskTensor = opParamInfo_.attenMask.tensor;
    if (attenMaskFlag_ && (sparseMode == SPARSE_MODE_NO_MASK || sparseMode == SPARSE_MODE_ALL_MASK)) {
        uint32_t maskS2 = maskTensor->GetStorageShape().GetDim(maskTensor->GetStorageShape().GetDimNum() - 1);
        uint32_t totalLen = fiaInfo_.systemPrefixLen + fiaInfo_.maxActualseq;
        if (totalLen > maskS2) {
            OP_LOGE(opName_, "s2Size + systemPrefix (%u) is greater than mask s2 size (%u)", totalLen, maskS2);
            return ge::GRAPH_FAILED;
        }
    }
    if (fiaInfo_.ropeMode !=  RopeMode::NO_ROPE) {
        OP_LOGE(opName_, "system prefix do not support rope");
        return ge::GRAPH_FAILED;
    }

    if (fiaInfo_.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        OP_LOGE(opName_, "system prefix do not support PAGE_ATTENTION");
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo_.kvStorageMode == KvStorageMode::TENSOR_LIST && fiaInfo_.s1Size > 1) {
        OP_LOGE(opName_, "system prefix do not support qs > 1 and enable TENSORLIST");
        return ge::GRAPH_FAILED;
    }

    if (fiaInfo_.pseShiftFlag) {
        if (fiaInfo_.s2Size + fiaInfo_.systemPrefixLen > fiaInfo_.pseShiftS2) {
            OP_LOGE(opName_, "when enable pse and system prefix, pse s2 Size greater than kv s2size + systemPrefixLen");
            return ge::GRAPH_FAILED;   
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureHeadDim() const
{
    constexpr uint32_t MAX_HEAD_DIM = 512;
    constexpr uint32_t MAX_ROPE_DIM = 64;

    OP_CHECK_IF(vHeadDim_ > MAX_HEAD_DIM,
    OP_LOGE(opName_, "In %s situation, headDim of value should be smaller or equal to 512, but got %u",
        QuantModeToSerialString(quantMode_).c_str(), vHeadDim_),
    return ge::GRAPH_FAILED);

    OP_CHECK_IF(ropeHeadDim_ > MAX_ROPE_DIM,
    OP_LOGE(opName_, "In %s situation, headDim of Rope should be smaller or equal to 64, but got %u",
        QuantModeToSerialString(quantMode_).c_str(), ropeHeadDim_),
    return ge::GRAPH_FAILED);

    constexpr int32_t D_ALIGN_SIZE = 16;
    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION && kvLayout_ == FiaLayout::NZ) {
        OP_CHECK_IF((vHeadDim_ % D_ALIGN_SIZE != 0) || (qkHeadDim_ % D_ALIGN_SIZE != 0),
        OP_LOGE(opName_, "In %s situation, when the dim of key&value is 5, headDim of query|key|value should be align to 16, but got keyHeadDim:%u, queryHeadDim and keyHeadDim:%u",
            QuantModeToSerialString(quantMode_).c_str(), vHeadDim_, qkHeadDim_),
        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoquant()
{
    OP_CHECK_IF(socVersion_ == platform_ascendc::SocVersion::ASCEND310P,
        OP_LOGE(opName_, "In %s %s situation, Ascend310P is not supported",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str()),
        return ge::GRAPH_FAILED);
    if (ge::GRAPH_SUCCESS != CheckFeatureTensorList() ||
        ge::GRAPH_SUCCESS != CheckFeatureBlockSize() ||
        ge::GRAPH_SUCCESS != CheckFeatureInOutDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckFeatureMask() ||
        ge::GRAPH_SUCCESS != CheckFeatureNoQuantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureLse() ||
        ge::GRAPH_SUCCESS != CheckFeatureLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureAxisInfo() ||
        ge::GRAPH_SUCCESS != CheckFeatureLearnableSink() ||
        ge::GRAPH_SUCCESS != CheckFeatureGqaPrefix() ||
        ge::GRAPH_SUCCESS != CheckFeatureLeftPadding() ||
        ge::GRAPH_SUCCESS != CheckFeaturePSE() ||
        ge::GRAPH_SUCCESS != CheckFeatureHeadDim()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaAntiquant() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaFullquant() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqa()
{
    if (quantMode_ == FiaQuantMode::NO_QUANT) {
        return CheckFeatureGqaNoquant();
    } else if (quantMode_ == FiaQuantMode::ANTI_QUANT) {
        return CheckFeatureGqaAntiquant();
    } else if (quantMode_ == FiaQuantMode::FULL_QUANT) {
        return CheckFeatureGqaFullquant();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureActualSeqLensExistence() const
{
    if ((qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD)) {
        OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor == nullptr,
            OP_LOGE(opName_, "when %s's layout is %s, %s should not be null.", QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                ACTUAL_SEQ_Q_LEN_NAME.c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
            OP_LOGE(opName_, "when %s's layout is %s, %s should not be null.", KEY_NAME.c_str(), LayoutToSerialString(kvLayout_).c_str(),
                ACTUAL_SEQ_KV_LEN_NAME.c_str()),
            return ge::GRAPH_FAILED);

        if (!fiaInfo_.isMaxWorkspace) {
            OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>() == nullptr,
                OP_LOGE(opName_, "when %s's layout is %s, %s data should not be null.", QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                    ACTUAL_SEQ_Q_LEN_NAME.c_str()),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() == nullptr,
                OP_LOGE(opName_, "when %s's layout is %s, %s data should not be null.", KEY_NAME.c_str(), LayoutToSerialString(kvLayout_).c_str(),
                    ACTUAL_SEQ_KV_LEN_NAME.c_str()),
                return ge::GRAPH_FAILED);
        }
    } else {
        if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
            OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
                OP_LOGE(opName_, "In page attention scene, %s should not be null.", ACTUAL_SEQ_KV_LEN_NAME.c_str()),
                return ge::GRAPH_FAILED);
            if (!fiaInfo_.isMaxWorkspace) {
                OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() == nullptr,
                    OP_LOGE(opName_, "In page attention scene, %s data should not be null.", ACTUAL_SEQ_KV_LEN_NAME.c_str()),
                    return ge::GRAPH_FAILED);
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    const FiaLayout &layout, const std::string &actualSeqLenName, const std::string &attrName)
{
    if (tensor == nullptr) {
        OP_LOGE(opName_, "when layout of %s is %s, %s must be provided.",
            attrName.c_str(), LayoutToSerialString(layout).c_str(), actualSeqLenName.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OP_LOGE(opName_, "%s shape size is %ld, it should be greater than 0.",
            actualSeqLenName.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureActualSeqLensQData()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        qSize.push_back(s1Size_);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t *actualSeq = opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>();
    if (actualSeq == nullptr) {
        qSize.push_back(s1Size_);
        return ge::GRAPH_SUCCESS;
    }

    if (GetActualSeqLenSize(actualSeqLengthsQSize_, opParamInfo_.actualSeqLengthsQ.tensor,
        qLayout_, ACTUAL_SEQ_Q_LEN_NAME, QUERY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t loop = std::min(actualSeqLengthsQSize_, bSize_);
    for (uint32_t i = 0; i < loop; i++) {
        int64_t tmpS1 = 0;
        if (qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD) {
            OP_CHECK_IF(actualSeq[i] < 0,
                OP_LOGE(opName_, "when %s's layout is %s, %s[%u] should not be a negative number, but got %ld.",
                    QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                    ACTUAL_SEQ_Q_LEN_NAME.c_str(), i, actualSeq[i]),
                    return ge::GRAPH_FAILED);

            OP_CHECK_IF(i > 0U && (actualSeq[i] < actualSeq[i - 1U]),
                OP_LOGE(opName_, "when %s's layout is %s, %s[%u](%ld) should not be less than %s[%u](%ld).",
                    QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                    ACTUAL_SEQ_Q_LEN_NAME.c_str(), i, actualSeq[i],
                    ACTUAL_SEQ_Q_LEN_NAME.c_str(), (i - 1U), actualSeq[i - 1U]),
                    return ge::GRAPH_FAILED);

            tmpS1 = (i == 0U) ? actualSeq[0] : (actualSeq[i] - actualSeq[i - 1U]);
        } else {
            tmpS1 = actualSeq[i];
        }
        if (tmpS1 > static_cast<int64_t>(s1Size_) || tmpS1 < 0) {
            OP_LOGE(opName_,
                "%s[%u] computed is %ld, it should be in range [0, Q_S(%u)].",
                ACTUAL_SEQ_Q_LEN_NAME.c_str(), i, tmpS1, s1Size_);
            return ge::GRAPH_FAILED;
        }
        qSize.push_back(tmpS1);
    }

    OP_CHECK_IF((qLayout_ == FiaLayout::TND) && (qTSize_ != actualSeq[actualSeqLengthsQSize_ - 1]),
        OP_LOGE(opName_, "when %s's layout is %s, T(%u) should be equal to the last element of %s(%ld).",
            QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(), qTSize_, ACTUAL_SEQ_Q_LEN_NAME.c_str(),
            actualSeq[actualSeqLengthsQSize_ - 1]),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureActualSeqLensKvData()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        kvSize.push_back(s2Size_);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t *actualSeq = opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>();
    if (actualSeq == nullptr) {
        kvSize.push_back(s2Size_);
        return ge::GRAPH_SUCCESS;
    }

    if(GetActualSeqLenSize(actualSeqLengthsKvSize_, opParamInfo_.actualSeqLengths.tensor,
        kvLayout_, ACTUAL_SEQ_KV_LEN_NAME, KEY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t loop = std::min(actualSeqLengthsKvSize_, bSize_);
    for (uint32_t i = 0; i < loop; i++) {
        int64_t tmpS2 = 0;
        if (kvLayout_ == FiaLayout::TND || kvLayout_ == FiaLayout::NTD) {
            OP_CHECK_IF(actualSeq[i] < 0,
                OP_LOGE(opName_, "when kv's layout is %s, %s[%u] should not be a negative number, but got %ld.",
                    LayoutToSerialString(kvLayout_).c_str(),
                    ACTUAL_SEQ_KV_LEN_NAME.c_str(), i, actualSeq[i]),
                    return ge::GRAPH_FAILED);

            OP_CHECK_IF(i > 0U && (actualSeq[i] < actualSeq[i - 1U]),
                OP_LOGE(opName_, "when kv's layout is %s, %s[%u](%ld) should not be less than %s[%u](%ld).",
                    LayoutToSerialString(kvLayout_).c_str(),
                    ACTUAL_SEQ_KV_LEN_NAME.c_str(), i, actualSeq[i],
                    ACTUAL_SEQ_KV_LEN_NAME.c_str(), (i - 1U), actualSeq[i - 1U]),
                    return ge::GRAPH_FAILED);

            tmpS2 = (i == 0U) ? actualSeq[0] : (actualSeq[i] - actualSeq[i - 1U]);
        } else {
            tmpS2 = actualSeq[i];
        }

        OP_CHECK_IF(tmpS2 < 0 || tmpS2 > s2Size_,
            OP_LOGE(opName_, "%s(%u) is %ld, it should be in range [0, KV_S(%ld)].",
                ACTUAL_SEQ_KV_LEN_NAME.c_str(), i, tmpS2, s2Size_),
            return ge::GRAPH_FAILED);
        kvSize.push_back(tmpS2);
    }

    OP_CHECK_IF((kvLayout_ == FiaLayout::TND) && (kTSize_ != actualSeq[actualSeqLengthsKvSize_ - 1]),
        OP_LOGE(opName_, "when kv's layout is %s, T(%u) should be equal to the last element of %s(%ld).",
            LayoutToSerialString(kvLayout_).c_str(), kTSize_, ACTUAL_SEQ_KV_LEN_NAME.c_str(),
            actualSeq[actualSeqLengthsKvSize_ - 1]),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureInOutDtype() const
{
    const std::vector<std::pair<ge::DataType, ge::DataType>> inOutDtypePairSupported = {
        {ge::DT_INT8, ge::DT_INT8},
        {ge::DT_INT8, ge::DT_FLOAT16},
        {ge::DT_FLOAT16, ge::DT_INT8},
        {ge::DT_FLOAT16, ge::DT_FLOAT16},
        {ge::DT_BF16, ge::DT_BF16},
        {ge::DT_BF16, ge::DT_INT8},
        {ge::DT_INT8, ge::DT_INT8},
    };

    std::pair<ge::DataType, ge::DataType> inOutDtypePair = {inputQType_, outputType_};
    if (!VecContains(inOutDtypePairSupported, inOutDtypePair)) {
        OP_LOGE(opName_, "input dtype %d with output dtype %d is not currently supported.", static_cast<int32_t>(inputQType_),
                  static_cast<int32_t>(outputType_));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureActualSeqLens()
{
    if (ge::GRAPH_SUCCESS != CheckFeatureActualSeqLensExistence() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLensQData() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLensKvData()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeature()
{
    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        return CheckFeatureMla();
    } else {
        return CheckFeatureGqa();
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
