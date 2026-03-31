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
 * \file dequant_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "dequant_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// check Q, KV, OUT 类型 全量化
ge::graphStatus DequantChecker::CheckDataTypeFullquant(const FiaTilingInfo &fiaInfo)
{
    // {Q, KV, attenOut}
    const std::vector<std::tuple<ge::DataType, ge::DataType, ge::DataType>> fullQuantDtypeSupported = {
        {ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16},
        {ge::DT_INT8, ge::DT_INT8, ge::DT_BF16},
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16},
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_BF16},
        {ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT16},
        {ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_BF16},
        {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}};

    std::tuple<ge::DataType, ge::DataType, ge::DataType> inOutDtypeTuple = {fiaInfo.inputQType, fiaInfo.inputKvType,
                                                                            fiaInfo.outputType};
    OP_CHECK_IF(
        ge::GRAPH_SUCCESS != CheckValueSupport(inOutDtypeTuple, fullQuantDtypeSupported),
        OP_LOGE(fiaInfo.opName,
                "In fullquant scenario, query datatype(%s), key/value datatype(%s), "
                "attentionOut datatype(%s) is not currently supported.",
                DataTypeToSerialString(fiaInfo.inputQType).c_str(), DataTypeToSerialString(fiaInfo.inputKvType).c_str(),
                DataTypeToSerialString(fiaInfo.outputType).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.inputKvType != fiaInfo.opParamInfo.value.desc->GetDataType(),
                OP_LOGE(fiaInfo.opName, "In fullquant scenario, the datatype of value(%s) should be equal to key(%s).",
                        DataTypeToSerialString(fiaInfo.opParamInfo.value.desc->GetDataType()).c_str(),
                        DataTypeToSerialString(fiaInfo.inputKvType).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// MLA dequantscale dtype:fp32
ge::graphStatus DequantChecker::CheckDequantScaleDtypeMLAFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableIFAMLAFullQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.opParamInfo.dequantScaleQuery.desc == nullptr ||
        fiaInfo.opParamInfo.keyAntiquantScale.desc == nullptr ||
        fiaInfo.opParamInfo.valueAntiquantScale.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(fiaInfo.opParamInfo.dequantScaleQuery.desc->GetDataType() != ge::DT_FLOAT ||
                    fiaInfo.opParamInfo.keyAntiquantScale.desc->GetDataType() != ge::DT_FLOAT ||
                    fiaInfo.opParamInfo.valueAntiquantScale.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(fiaInfo.opName,
                        "In MLA fullquant scenario, datatype of dequantScaleQuery(%s), keyAntiquantScale(%s) "
                        "and valueAntiquantScale(%s) must be FLOAT32.",
                        DataTypeToSerialString(fiaInfo.opParamInfo.dequantScaleQuery.desc->GetDataType()).c_str(),
                        DataTypeToSerialString(fiaInfo.opParamInfo.keyAntiquantScale.desc->GetDataType()).c_str(),
                        DataTypeToSerialString(fiaInfo.opParamInfo.valueAntiquantScale.desc->GetDataType()).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// GQA perblock dequantscale dtype:fp32
ge::graphStatus DequantChecker::CheckDequantScaleDtypeGQAPerblock(const FiaTilingInfo &fiaInfo)
{
    if (!enablePerblockQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.opParamInfo.dequantScaleQuery.desc == nullptr ||
        fiaInfo.opParamInfo.keyAntiquantScale.desc == nullptr ||
        fiaInfo.opParamInfo.valueAntiquantScale.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(fiaInfo.opParamInfo.dequantScaleQuery.desc->GetDataType() != ge::DT_FLOAT ||
                    fiaInfo.opParamInfo.keyAntiquantScale.desc->GetDataType() != ge::DT_FLOAT ||
                    fiaInfo.opParamInfo.valueAntiquantScale.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(fiaInfo.opName,
                        "In per-block quant scenario, datatype of dequantScaleQuery(%s), keyAntiquantScale(%s) "
                        "and valueAntiquantScale(%s) must be float32.",
                        DataTypeToSerialString(fiaInfo.opParamInfo.dequantScaleQuery.desc->GetDataType()).c_str(),
                        DataTypeToSerialString(fiaInfo.opParamInfo.keyAntiquantScale.desc->GetDataType()).c_str(),
                        DataTypeToSerialString(fiaInfo.opParamInfo.valueAntiquantScale.desc->GetDataType()).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDequantScaleDtypeGQAPertensor(const FiaTilingInfo &fiaInfo)
{
    if (!enablePertensorQuant_) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *quantScale1Desc = fiaInfo.opParamInfo.quantScale1.desc;
    const gert::CompileTimeTensorDesc *deqScale1Desc = fiaInfo.opParamInfo.deqScale1.desc;
    const gert::CompileTimeTensorDesc *deqScale2Desc = fiaInfo.opParamInfo.deqScale1.desc;
    if (quantScale1Desc == nullptr || deqScale1Desc == nullptr || deqScale2Desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(quantScale1Desc, QUANT_SCALE1_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(deqScale1Desc, DEQUANT_SCALE1_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(deqScale2Desc, DEQUANT_SCALE2_NAME)) {
        OP_LOGE(fiaInfo.opName,
                "In per-tensor quant scenario, the datatype of "
                "%s(%s), %s(%s), or %s(%s) is not supported.",
                QUANT_SCALE1_NAME.c_str(), DataTypeToSerialString(quantScale1Desc->GetDataType()).c_str(),
                DEQUANT_SCALE1_NAME.c_str(), DataTypeToSerialString(deqScale1Desc->GetDataType()).c_str(),
                DEQUANT_SCALE2_NAME.c_str(), DataTypeToSerialString(deqScale2Desc->GetDataType()).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDequantScaleDtypeFullquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckDequantScaleDtypeMLAFullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantScaleDtypeGQAPerblock(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantScaleDtypeGQAPertensor(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// MLA全量化 Q: per-token-head (3), KV: per-tensor (0)
ge::graphStatus DequantChecker::CheckDequantModeMLAFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableIFAMLAFullQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        *fiaInfo.opParamInfo.queryQuantMode != PER_TOKEN_HEAD_MODE,
        OP_LOGE(fiaInfo.opName, "In MLA fullquant scenario, queryQuantMode(%d) only support per-token-head(%u).",
                *fiaInfo.opParamInfo.queryQuantMode, PER_TOKEN_HEAD_MODE),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(*fiaInfo.opParamInfo.keyAntiquantMode != PER_CHANNEL_MODE,
                OP_LOGE(fiaInfo.opName, "In MLA fullquant scenario, keyAntiquantMode(%d) only support per-tensor(%u).",
                        *fiaInfo.opParamInfo.keyAntiquantMode, PER_CHANNEL_MODE),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        *fiaInfo.opParamInfo.valueAntiquantMode != PER_CHANNEL_MODE,
        OP_LOGE(fiaInfo.opName, "In MLA fullquant scenario, valueAntiquantMode(%d) only support per-tensor(%u).",
                *fiaInfo.opParamInfo.valueAntiquantMode, PER_CHANNEL_MODE),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// per-block qkv antiquantMode(7)
ge::graphStatus DequantChecker::CheckDequantModeGQAPerblock(const FiaTilingInfo &fiaInfo)
{
    if (!enablePerblockQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((*fiaInfo.opParamInfo.keyAntiquantMode != PER_BLOCK_MODE ||
                 *fiaInfo.opParamInfo.valueAntiquantMode != PER_BLOCK_MODE ||
                 *fiaInfo.opParamInfo.queryQuantMode != PER_BLOCK_MODE),
                OP_LOGE(fiaInfo.opName,
                        "In per-block quant scenario, queryQuantMode(%u)/"
                        "keyAntiquantMode(%u)/valueAntiquantMode(%u) only support per-block(%u).",
                        *fiaInfo.opParamInfo.keyAntiquantMode, *fiaInfo.opParamInfo.valueAntiquantMode,
                        *fiaInfo.opParamInfo.queryQuantMode, PER_BLOCK_MODE),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// per-tensor qkv antiquantMode(0)
ge::graphStatus DequantChecker::CheckDequantModeGQAPertensor(const FiaTilingInfo &fiaInfo)
{
    if (!enablePertensorQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((*fiaInfo.opParamInfo.keyAntiquantMode != PER_CHANNEL_MODE ||
                 *fiaInfo.opParamInfo.valueAntiquantMode != PER_CHANNEL_MODE ||
                 *fiaInfo.opParamInfo.queryQuantMode != PER_CHANNEL_MODE),
                OP_LOGE(fiaInfo.opName,
                        "In per-tensor quant scenario, queryQuantMode(%u)/"
                        "keyAntiquantMode(%u)/valueAntiquantMode(%u) only support per-tensor(%u).",
                        *fiaInfo.opParamInfo.keyAntiquantMode, *fiaInfo.opParamInfo.valueAntiquantMode,
                        *fiaInfo.opParamInfo.queryQuantMode, PER_CHANNEL_MODE),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// check dequant scale Mode
ge::graphStatus DequantChecker::CheckDequantModeFullquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckDequantModeGQAPertensor(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantModeGQAPerblock(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantModeMLAFullquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckTensorExistFullquant(const FiaTilingInfo &fiaInfo, const gert::Tensor *tensor,
                                                          const std::string &quantModeName,
                                                          const std::string &inputName)
{
    OP_CHECK_IF(
        tensor == nullptr,
        OP_LOGE(fiaInfo.opName, "In %s scenario, %s should not be null.", quantModeName.c_str(), inputName.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckTensorNotExistFullquant(const FiaTilingInfo &fiaInfo, const gert::Tensor *tensor,
                                                             const std::string &quantModeName,
                                                             const std::string &inputName)
{
    OP_CHECK_IF(tensor != nullptr,
                OP_LOGE(fiaInfo.opName, "In %s scenario, %s should be null.", quantModeName.c_str(), inputName.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus DequantChecker::CheckExistenceNoquant(const FiaTilingInfo &fiaInfo)
{
    auto deqScale1 = fiaInfo.opParamInfo.deqScale1.tensor;
    auto quantScale1 = fiaInfo.opParamInfo.quantScale1.tensor;
    auto deqScale2 = fiaInfo.opParamInfo.deqScale2.tensor;
    auto dequantScaleQuery = fiaInfo.opParamInfo.dequantScaleQuery.tensor;
    auto antiquantScale = fiaInfo.opParamInfo.antiquantScale.tensor;
    auto antiquantOffset = fiaInfo.opParamInfo.antiquantOffset.tensor;
    auto keyAntiquantScale = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto keyAntiquantOffset = fiaInfo.opParamInfo.keyAntiquantOffset.tensor;
    auto valueAntiquantScale = fiaInfo.opParamInfo.valueAntiquantScale.tensor;
    auto valueAntiquantOffset = fiaInfo.opParamInfo.valueAntiquantOffset.tensor;
    bool checkExistenece = deqScale1 == nullptr && quantScale1 == nullptr && deqScale2 == nullptr &&
                           dequantScaleQuery == nullptr && antiquantScale == nullptr && antiquantOffset == nullptr &&
                           keyAntiquantScale == nullptr && keyAntiquantOffset == nullptr &&
                           valueAntiquantScale == nullptr && valueAntiquantOffset == nullptr;
    if (checkExistenece) {
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(fiaInfo.opName,
                "When query,key and value is all float16 or bfloat16, cannot exist quant related param.");
        return ge::GRAPH_FAILED;
    }
}
ge::graphStatus DequantChecker::CheckExistencePertensorFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enablePertensorQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    string quantModeName = "per-tensor quant";
    const gert::Tensor *quantScale1Tensor = fiaInfo.opParamInfo.quantScale1.tensor;
    const gert::Tensor *deqScale1Tensor = fiaInfo.opParamInfo.deqScale1.tensor;
    const gert::Tensor *deqScale2Tensor = fiaInfo.opParamInfo.deqScale2.tensor;
    // Q/K/V antiquantScale
    if (ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, quantScale1Tensor, quantModeName, "quantScale1") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, deqScale1Tensor, quantModeName, "deqScale1") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, deqScale2Tensor, quantModeName, "deqScale2")) {
        return ge::GRAPH_FAILED;
    }

    // shapesize != 0
    OP_CHECK_IF(
        (quantScale1Tensor->GetStorageShape().GetShapeSize() == 0 ||
         deqScale1Tensor->GetStorageShape().GetShapeSize() == 0 ||
         deqScale2Tensor->GetStorageShape().GetShapeSize() == 0),
        OP_LOGE(fiaInfo.opName, "In %s scenario, deqScale1, quantScale1 or deqScale2 should not be empty tensor.",
                quantModeName.c_str()),
        return ge::GRAPH_FAILED);

    // 其他量化方式参数不能存在
    if (ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantOffset.tensor,
                                                          quantModeName, "antiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantOffset.tensor,
                                                          quantModeName, "keyAntiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantOffset.tensor,
                                                          quantModeName, "valueAntiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantScale.tensor,
                                                          quantModeName, "antiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantScale.tensor,
                                                          quantModeName, "keyAntiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantScale.tensor,
                                                          quantModeName, "valueAntiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.dequantScaleQuery.tensor,
                                                          quantModeName, "dequantScaleQuery") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyRopeAntiquantScale.tensor,
                                                          quantModeName, "keyRopeAntiquantScale")) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckExistenceMLAFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableIFAMLAFullQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    string quantModeName = "MLA fullquant";
    // Q/K/V antiquantScale
    if (ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.dequantScaleQuery.tensor,
                                                       quantModeName, "dequantScaleQuery") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantScale.tensor,
                                                       quantModeName, "keyAntiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantScale.tensor,
                                                       quantModeName, "valueAntiquantScale")) {
        return ge::GRAPH_FAILED;
    }

    // 不支持offset
    if (ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantOffset.tensor,
                                                          quantModeName, "antiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantOffset.tensor,
                                                          quantModeName, "keyAntiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantOffset.tensor,
                                                          quantModeName, "valueAntiquantOffset")) {
        return ge::GRAPH_FAILED;
    }

    // 其他量化方式参数不能存在
    if (ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.quantScale1.tensor,
                                                          quantModeName, "quantScale1") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.deqScale1.tensor, quantModeName,
                                                          "deqScale1") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.deqScale2.tensor, quantModeName,
                                                          "deqScale2") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantScale.tensor,
                                                          quantModeName, "antiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyRopeAntiquantScale.tensor,
                                                          quantModeName, "keyRopeAntiquantScale")) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckExistencePerblockFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enablePerblockQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    string quantModeName = "per-block quant";
    // Q/K/V antiquantScale
    if (ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.dequantScaleQuery.tensor,
                                                       quantModeName, "dequantScaleQuery") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantScale.tensor,
                                                       quantModeName, "keyAntiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantScale.tensor,
                                                       quantModeName, "valueAntiquantScale")) {
        return ge::GRAPH_FAILED;
    }

    // 不支持offset
    if (ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantOffset.tensor,
                                                          quantModeName, "antiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantOffset.tensor,
                                                          quantModeName, "keyAntiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantOffset.tensor,
                                                          quantModeName, "valueAntiquantOffset")) {
        return ge::GRAPH_FAILED;
    }

    // 其他量化方式参数不能存在
    if (ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.quantScale1.tensor,
                                                          quantModeName, "quantScale1") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.deqScale1.tensor, quantModeName,
                                                          "deqScale1") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.deqScale2.tensor, quantModeName,
                                                          "deqScale2") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantScale.tensor,
                                                          quantModeName, "antiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyRopeAntiquantScale.tensor,
                                                          quantModeName, "keyRopeAntiquantScale")) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// GQA per-tensor
ge::graphStatus DequantChecker::CheckFeaturePertensorFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enablePertensorQuant_) {
        return ge::GRAPH_SUCCESS;
    }

    // 不支持后量化
    OP_CHECK_IF(fiaInfo.isOutQuantEnable,
                OP_LOGE(fiaInfo.opName, "In per-tensor quant scenario, postquant is not supported."),
                return ge::GRAPH_FAILED);

    // 不支持 rope
    OP_CHECK_IF(fiaInfo.mlaMode != MlaMode::NO_MLA,
                OP_LOGE(fiaInfo.opName, "In per-tensor quant scenario, rope is not supported."),
                return ge::GRAPH_FAILED);

    // 不支持alibipse
    OP_CHECK_IF(fiaInfo.enableAlibiPse,
                OP_LOGE(fiaInfo.opName, "In per-tensor quant scenario, pseType = 2/3 is not supported."),
                return ge::GRAPH_FAILED);

    // 不支持PA_NZ
    const uint32_t keyDim = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(keyDim == DIM_NUM_5, OP_LOGE(fiaInfo.opName, "In per-tensor quant scenario, PA_NZ is not supported."),
                return ge::GRAPH_FAILED);

    // 不支持QS=1
    OP_CHECK_IF(fiaInfo.s1Size == 1,
                OP_LOGE(fiaInfo.opName, "The q_s should not be equal to 1 in per-tensor quant scenario."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// GQA per-block
ge::graphStatus DequantChecker::CheckFeaturePerblockFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enablePerblockQuant_) {
        return ge::GRAPH_SUCCESS;
    }

    // 不支持alibipse
    OP_CHECK_IF(fiaInfo.enableAlibiPse,
                OP_LOGE(fiaInfo.opName, "In per-block quant scenario, pseType should not be 2/3."),
                return ge::GRAPH_FAILED);
    // 不支持pse
    OP_CHECK_IF(fiaInfo.pseShiftFlag, OP_LOGE(fiaInfo.opName, "In per-block quant scenario, pse is not supported."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag,
                OP_LOGE(fiaInfo.opName, "In per-block quant scenario, left padding is not supported."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.sysPrefixFlag,
                OP_LOGE(fiaInfo.opName, "In per-block quant scenario, system prefix is not supported."),
                return ge::GRAPH_FAILED);
                
    OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
                OP_LOGE(fiaInfo.opName, "In per-block quant scenario, key/value tensorlist is not supported."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION,
                OP_LOGE(fiaInfo.opName, "In per-block quant scenario, page attention is not supported."),
                return ge::GRAPH_FAILED);

    // 不支持 softmaxlse
    OP_CHECK_IF(fiaInfo.softmaxLseFlag,
                OP_LOGE(fiaInfo.opName, "In per-block quant scenario, softmax lse is not supported."),
                return ge::GRAPH_FAILED);

    // 不支持mask
    OP_CHECK_IF(fiaInfo.attenMaskFlag, OP_LOGE(fiaInfo.opName, "In per-block quant scenario, mask is not supported."),
                return ge::GRAPH_FAILED);

    // innerPrecise 仅支持 0/1
    OP_CHECK_IF((fiaInfo.innerPrecise != HIGH_PRECISION) && (fiaInfo.innerPrecise != HIGH_PERFORMANCE),
                OP_LOGE(fiaInfo.opName, "In per-block quant scenario, innerPrecise(%d) only support %u or %u.",
                        fiaInfo.innerPrecise, HIGH_PRECISION, HIGH_PERFORMANCE),
                return ge::GRAPH_FAILED);

    // 不支持 rope
    OP_CHECK_IF(fiaInfo.mlaMode != MlaMode::NO_MLA,
                OP_LOGE(fiaInfo.opName, "In per-block quant scenario, rope is not supported."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 不支持左padding、tensorlist、prefix、pse、alibipse
ge::graphStatus DequantChecker::CheckFeatureMLAFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableIFAMLAFullQuant_) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(fiaInfo.enableAlibiPse,
                OP_LOGE(fiaInfo.opName, "In MLA fullquant scenario, pseType should not be 2/3."),
                return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(fiaInfo.pseShiftFlag, OP_LOGE(fiaInfo.opName, "In MLA fullquant scenario, pse is not supported."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag,
                OP_LOGE(fiaInfo.opName, "In MLA fullquant scenario, left padding is not supported."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.sysPrefixFlag,
                OP_LOGE(fiaInfo.opName, "In MLA fullquant scenario, system prefix is not supported."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
                OP_LOGE(fiaInfo.opName, "In MLA fullquant scenario, key/value tensorlist is not supported."),
                return ge::GRAPH_FAILED);
    // MLA int8全量化仅支持
    const uint32_t keyDim = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(fiaInfo.inputQType == ge::DT_INT8 &&
        !(fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION && keyDim == DIM_NUM_5),
                OP_LOGE(fiaInfo.opName, "In MLA fullquant scenario, key/value layout must be PA_NZ when input datatype is INT8."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check scale shape
ge::graphStatus DequantChecker::CheckDequantScaleShapeMLAFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableIFAMLAFullQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    // query
    const gert::Shape queryInputShape = fiaInfo.opParamInfo.query.shape->GetStorageShape();
    const uint32_t queryDimNum = queryInputShape.GetDimNum();
    const gert::Shape dequantScaleQueryShape = fiaInfo.opParamInfo.dequantScaleQuery.tensor->GetStorageShape();
    const uint32_t dequantScaleQueryDimNum = dequantScaleQueryShape.GetDimNum();

    if (fiaInfo.qLayout == FiaLayout::BSH) {
        OP_CHECK_IF((queryDimNum != dequantScaleQueryDimNum),
                    OP_LOGE(fiaInfo.opName,
                            "In MLA fullquant scenario (layout of query is %s), "
                            "the dim num of dequantScaleQuery(%u) should be equal to query(%u).",
                            LayoutToSerialString(fiaInfo.qLayout).c_str(), dequantScaleQueryDimNum, queryDimNum),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF((queryInputShape.GetDim(DIM_NUM_0) != dequantScaleQueryShape.GetDim(DIM_NUM_0)) ||
                        (queryInputShape.GetDim(DIM_NUM_1) != dequantScaleQueryShape.GetDim(DIM_NUM_1)) ||
                        (*fiaInfo.opParamInfo.numHeads != dequantScaleQueryShape.GetDim(DIM_NUM_2)),
                    OP_LOGE(fiaInfo.opName,
                            "In MLA fullquant scenario (layout of query is %s), "
                            "the shape of dequantScaleQuery([%ld, %ld, %ld]) should be [%ld, %ld, %ld].",
                            LayoutToSerialString(fiaInfo.qLayout).c_str(), dequantScaleQueryShape.GetDim(DIM_NUM_0),
                            dequantScaleQueryShape.GetDim(DIM_NUM_1), dequantScaleQueryShape.GetDim(DIM_NUM_2),
                            queryInputShape.GetDim(DIM_NUM_0), queryInputShape.GetDim(DIM_NUM_1),
                            *fiaInfo.opParamInfo.numHeads),
                    return ge::GRAPH_FAILED);
    } else if (fiaInfo.qLayout == FiaLayout::BSND || fiaInfo.qLayout == FiaLayout::BNSD ||
               fiaInfo.qLayout == FiaLayout::TND) {
        OP_CHECK_IF((dequantScaleQueryDimNum != queryDimNum - NUM1),
                    OP_LOGE(fiaInfo.opName,
                            "In MLA fullquant scenario (layout of query is %s), "
                            "the dim num of dequantScaleQuery(%u) should be equal to [queryDimNum - 1](%u).",
                            LayoutToSerialString(fiaInfo.qLayout).c_str(), dequantScaleQueryDimNum, queryDimNum - 1),
                    return ge::GRAPH_FAILED);

        for (uint32_t i = 0; i < dequantScaleQueryDimNum; i++) {
            OP_CHECK_IF((queryInputShape.GetDim(i) != dequantScaleQueryShape.GetDim(i)),
                        OP_LOGE(fiaInfo.opName,
                                "In MLA fullquant scenario (layout of query is %s), "
                                "the %urd dim of dequantScaleQuery(%ld) should be equal to query(%ld).",
                                LayoutToSerialString(fiaInfo.qLayout).c_str(), i, dequantScaleQueryShape.GetDim(i),
                                queryInputShape.GetDim(i)),
                        return ge::GRAPH_FAILED);
        }
    }

    // kv: [1]
    OP_CHECK_IF((fiaInfo.opParamInfo.keyAntiquantScale.tensor->GetStorageShape().GetDimNum() != NUM1 ||
                 fiaInfo.opParamInfo.keyAntiquantScale.tensor->GetShapeSize() != NUM1),
                OP_LOGE(fiaInfo.opName, "In MLA fullquant scenario, the shape of keyAntiquantScale must be [1]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.opParamInfo.valueAntiquantScale.tensor->GetStorageShape().GetDimNum() != NUM1 ||
                 fiaInfo.opParamInfo.valueAntiquantScale.tensor->GetShapeSize() != NUM1),
                OP_LOGE(fiaInfo.opName, "In MLA fullquant scenario, the shape of valueAntiquantScale must be [1]."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDequantScaleShapePertensor(const FiaTilingInfo &fiaInfo)
{
    if (!enablePertensorQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    // shape:[1]
    OP_CHECK_IF((fiaInfo.opParamInfo.quantScale1.tensor->GetStorageShape().GetDimNum() != NUM1 ||
                 fiaInfo.opParamInfo.quantScale1.tensor->GetShapeSize() != NUM1),
                OP_LOGE(fiaInfo.opName, "In per-tensor quant scenario, the shape of quantScale1 must be [1]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.opParamInfo.deqScale1.tensor->GetStorageShape().GetDimNum() != NUM1 ||
                 fiaInfo.opParamInfo.deqScale1.tensor->GetShapeSize() != NUM1),
                OP_LOGE(fiaInfo.opName, "In per-tensor quant scenario, the shape of deqScale1 must be [1]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.opParamInfo.deqScale2.tensor->GetStorageShape().GetDimNum() != NUM1 ||
                 fiaInfo.opParamInfo.deqScale2.tensor->GetShapeSize() != NUM1),
                OP_LOGE(fiaInfo.opName, "In per-tensor quant scenario, the shape of deqScale2 must be [1]."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// GQA antiquantscale
ge::graphStatus DequantChecker::CheckDequantScaleShapePerblock(const FiaTilingInfo &fiaInfo)
{
    if (!enablePerblockQuant_) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::Shape dequantScaleQueryShape = fiaInfo.opParamInfo.dequantScaleQuery.tensor->GetStorageShape();
    const gert::Shape keyAntiquantScaleShape = fiaInfo.opParamInfo.keyAntiquantScale.tensor->GetStorageShape();
    const gert::Shape valueAntiquantScaleShape = fiaInfo.opParamInfo.valueAntiquantScale.tensor->GetStorageShape();

    constexpr uint32_t fp8QBlockSize = 128U;   // 128 is SOuterSize
    constexpr uint32_t fp8KVBlockSize = 256U;  // 256 is SInnerSize

    // NTD_TND格式 scale dim = 3
    if (fiaInfo.qLayout == FiaLayout::NTD) {
        OP_CHECK_IF((dequantScaleQueryShape.GetDimNum() != DIM_NUM_3) ||
                        (keyAntiquantScaleShape.GetDimNum() != DIM_NUM_3) ||
                        (valueAntiquantScaleShape.GetDimNum() != DIM_NUM_3),
                    OP_LOGE(fiaInfo.opName,
                            "In per-block quant scenario, when layout is %s, the dim num of "
                            "dequantScaleQuery(%u), keyAntiquantScale(%u) and valueAntiquantScale(%u) must be %u.",
                            fiaInfo.opParamInfo.layOut, dequantScaleQueryShape.GetDimNum(),
                            keyAntiquantScaleShape.GetDimNum(), valueAntiquantScaleShape.GetDimNum(), DIM_NUM_3),
                    return ge::GRAPH_FAILED);

        if (fiaInfo.isMaxWorkspace) {
            return ge::GRAPH_SUCCESS;
        }

        // dequantScaleQuery [N, T/128+B, D/256]
        OP_CHECK_IF(
            (dequantScaleQueryShape.GetDim(DIM_NUM_0) != fiaInfo.n1Size) ||
                (dequantScaleQueryShape.GetDim(DIM_NUM_1) != fiaInfo.qTSize / fp8QBlockSize + fiaInfo.bSize) ||
                (dequantScaleQueryShape.GetDim(DIM_NUM_2) != CeilDivision(fiaInfo.qkHeadDim, fp8KVBlockSize)),
            OP_LOGE(fiaInfo.opName,
                    "In per-block quant scenario, when layout is %s, "
                    "the shape of dequantScaleQuery([%ld, %ld, %ld]) should be [%u, %u, %u].",
                    LayoutToSerialString(fiaInfo.qLayout).c_str(), dequantScaleQueryShape.GetDim(DIM_NUM_0),
                    dequantScaleQueryShape.GetDim(DIM_NUM_1), dequantScaleQueryShape.GetDim(DIM_NUM_2), fiaInfo.n1Size,
                    fiaInfo.qTSize / fp8QBlockSize + fiaInfo.bSize, CeilDivision(fiaInfo.qkHeadDim, fp8KVBlockSize)),
            return ge::GRAPH_FAILED);

        // keyAntiquantScale/valueAntiquantScale [N, T/256+B, D/256]
        OP_CHECK_IF(
            (keyAntiquantScaleShape.GetDim(DIM_NUM_0) != fiaInfo.n2Size) ||
                (keyAntiquantScaleShape.GetDim(DIM_NUM_1) != fiaInfo.kTSize / fp8KVBlockSize + fiaInfo.bSize) ||
                (keyAntiquantScaleShape.GetDim(DIM_NUM_2) != CeilDivision(fiaInfo.qkHeadDim, fp8KVBlockSize)),
            OP_LOGE(fiaInfo.opName,
                    "In per-block quant scenario, when layout is %s, "
                    "the shape of keyAntiquantScale([%ld, %ld, %ld]) should be [%u, %u, %u].",
                    LayoutToSerialString(fiaInfo.qLayout).c_str(), keyAntiquantScaleShape.GetDim(DIM_NUM_0),
                    keyAntiquantScaleShape.GetDim(DIM_NUM_1), keyAntiquantScaleShape.GetDim(DIM_NUM_2), fiaInfo.n2Size,
                    fiaInfo.kTSize / fp8KVBlockSize + fiaInfo.bSize, CeilDivision(fiaInfo.qkHeadDim, fp8KVBlockSize)),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            (valueAntiquantScaleShape.GetDim(DIM_NUM_0) != fiaInfo.n2Size) ||
                (valueAntiquantScaleShape.GetDim(DIM_NUM_1) != fiaInfo.kTSize / fp8KVBlockSize + fiaInfo.bSize) ||
                (valueAntiquantScaleShape.GetDim(DIM_NUM_2) != CeilDivision(fiaInfo.vHeadDim, fp8KVBlockSize)),
            OP_LOGE(fiaInfo.opName,
                    "In per-block quant scenario, when layout is %s, "
                    "the shape of keyAntiquantScale([%ld, %ld, %ld]) should be [%u, %u, %u].",
                    LayoutToSerialString(fiaInfo.qLayout).c_str(), valueAntiquantScaleShape.GetDim(DIM_NUM_0),
                    valueAntiquantScaleShape.GetDim(DIM_NUM_1), valueAntiquantScaleShape.GetDim(DIM_NUM_2),
                    fiaInfo.n2Size, fiaInfo.kTSize / fp8KVBlockSize + fiaInfo.bSize,
                    CeilDivision(fiaInfo.vHeadDim, fp8KVBlockSize)),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((dequantScaleQueryShape.GetDimNum() != DIM_NUM_4) ||
                        (keyAntiquantScaleShape.GetDimNum() != DIM_NUM_4) ||
                        (valueAntiquantScaleShape.GetDimNum() != DIM_NUM_4),
                    OP_LOGE(fiaInfo.opName,
                            "In per-block quant scenario, when layout is %s, the dim num of "
                            "dequantScaleQuery(%u), keyAntiquantScale(%u) and valueAntiquantScale(%u) must be %u.",
                            LayoutToSerialString(fiaInfo.qLayout).c_str(), dequantScaleQueryShape.GetDimNum(),
                            keyAntiquantScaleShape.GetDimNum(), valueAntiquantScaleShape.GetDimNum(), DIM_NUM_4),
                    return ge::GRAPH_FAILED);

        // dequantScaleQuery [B, N, S/128, 1]
        OP_CHECK_IF((dequantScaleQueryShape.GetDim(DIM_NUM_0) != fiaInfo.bSize) ||
                        (dequantScaleQueryShape.GetDim(DIM_NUM_1) != fiaInfo.n1Size) ||
                        (dequantScaleQueryShape.GetDim(DIM_NUM_2) != CeilDivision(fiaInfo.s1Size, fp8QBlockSize)) ||
                        (dequantScaleQueryShape.GetDim(DIM_NUM_3) != NUM1),
                    OP_LOGE(fiaInfo.opName,
                            "In per-block quant scenario, when layout is %s, "
                            "the shape of dequantScaleQuery([%ld, %ld, %ld, %ld]) should be [%u, %u, %u, %u].",
                            LayoutToSerialString(fiaInfo.qLayout).c_str(), dequantScaleQueryShape.GetDim(DIM_NUM_0),
                            dequantScaleQueryShape.GetDim(DIM_NUM_1), dequantScaleQueryShape.GetDim(DIM_NUM_2),
                            dequantScaleQueryShape.GetDim(DIM_NUM_3), fiaInfo.bSize, fiaInfo.n1Size,
                            CeilDivision(fiaInfo.s1Size, fp8QBlockSize), NUM1),
                    return ge::GRAPH_FAILED);

        // keyAntiquantScale/valueAntiquantScale [B, N, S/256, 1]
        OP_CHECK_IF(
            (keyAntiquantScaleShape.GetDim(DIM_NUM_0) != fiaInfo.bSize) ||
                (keyAntiquantScaleShape.GetDim(DIM_NUM_1) != fiaInfo.n2Size) ||
                (keyAntiquantScaleShape.GetDim(DIM_NUM_2) != CeilDivision(fiaInfo.s2Size, int64_t(fp8KVBlockSize))) ||
                (keyAntiquantScaleShape.GetDim(DIM_NUM_3) != NUM1),
            OP_LOGE(fiaInfo.opName,
                    "In per-block quant scenario, when layout is %s, "
                    "the shape of keyAntiquantScale([%ld, %ld, %ld, %ld]) should be [%u, %u, %ld, %u].",
                    LayoutToSerialString(fiaInfo.qLayout).c_str(), keyAntiquantScaleShape.GetDim(DIM_NUM_0),
                    keyAntiquantScaleShape.GetDim(DIM_NUM_1), keyAntiquantScaleShape.GetDim(DIM_NUM_2),
                    keyAntiquantScaleShape.GetDim(DIM_NUM_3), fiaInfo.bSize, fiaInfo.n2Size,
                    CeilDivision(fiaInfo.s2Size, int64_t(fp8KVBlockSize)), NUM1),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            (valueAntiquantScaleShape.GetDim(DIM_NUM_0) != fiaInfo.bSize) ||
                (valueAntiquantScaleShape.GetDim(DIM_NUM_1) != fiaInfo.n2Size) ||
                (valueAntiquantScaleShape.GetDim(DIM_NUM_2) != CeilDivision(fiaInfo.s2Size, int64_t(fp8KVBlockSize))) ||
                (valueAntiquantScaleShape.GetDim(DIM_NUM_3) != NUM1),
            OP_LOGE(fiaInfo.opName,
                    "In per-block quant scenario, when layout is %s, "
                    "the shape of valueAntiquantScale([%ld, %ld, %ld, %ld]) should be [%u, %u, %ld, %u].",
                    LayoutToSerialString(fiaInfo.qLayout).c_str(), valueAntiquantScaleShape.GetDim(DIM_NUM_0),
                    valueAntiquantScaleShape.GetDim(DIM_NUM_1), valueAntiquantScaleShape.GetDim(DIM_NUM_2),
                    valueAntiquantScaleShape.GetDim(DIM_NUM_3), fiaInfo.bSize, fiaInfo.n2Size,
                    CeilDivision(fiaInfo.s2Size, int64_t(fp8KVBlockSize)), NUM1),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDequantScaleShapeFullquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckDequantScaleShapeMLAFullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantScaleShapePertensor(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantScaleShapePerblock(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// check 不同量化方式 支持的数据类型
ge::graphStatus DequantChecker::CheckInputDTypeFullquant(const FiaTilingInfo &fiaInfo)
{
    if (enableIFAMLAFullQuant_) {  // MLA 全量化 QKV : fp8_e4m3/int8
        OP_CHECK_IF(!(fiaInfo.inputQType == ge::DT_FLOAT8_E4M3FN || fiaInfo.inputQType == ge::DT_INT8),
                    OP_LOGE(fiaInfo.opName,
                            "In MLA fullquant scenario, query datatype(%s) and key/value datatype(%s), "
                            "should be FLOAT8_E4M3FN or INT8",
                            DataTypeToSerialString(fiaInfo.inputQType).c_str(),
                            DataTypeToSerialString(fiaInfo.inputKvType).c_str()),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(fiaInfo.outputType != ge::DT_BF16,
                    OP_LOGE(fiaInfo.opName,
                            "In MLA fullquant scenario, attentionOut datatype(%s) should be BF16.",
                            DataTypeToSerialString(fiaInfo.outputType).c_str()),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(!(fiaInfo.inputQRopeType == ge::DT_BF16 && fiaInfo.inputKRopeType == ge::DT_BF16),
                    OP_LOGE(fiaInfo.opName,
                            "In MLA fullquant scenario, queryRope datatype(%s), "
                            "keyRope datatype(%s) should be BF16.",
                            DataTypeToSerialString(fiaInfo.inputQRopeType).c_str(),
                            DataTypeToSerialString(fiaInfo.inputKRopeType).c_str()),
                    return ge::GRAPH_FAILED);
    } else if (enablePertensorQuant_) {  // GQA Pertensor QKV:int8
        OP_CHECK_IF((fiaInfo.inputQType != ge::DT_INT8),
                    OP_LOGE(fiaInfo.opName, "In per-tensor quant scenario, input datatype(%s) should be INT8.",
                            DataTypeToSerialString(fiaInfo.inputQType).c_str()),
                    return ge::GRAPH_FAILED);
    } else if (enablePerblockQuant_) {  // GQA perblock fp8_e4m3/hifp8
        OP_CHECK_IF((fiaInfo.inputQType != ge::DT_FLOAT8_E4M3FN && fiaInfo.inputQType != ge::DT_HIFLOAT8),
                    OP_LOGE(fiaInfo.opName,
                            "In per-block quant scenario, input datatype(%s) should be FLOAT8_E4M3FN or HIFLOAT8.",
                            DataTypeToSerialString(fiaInfo.inputQType).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// check per-block layout
ge::graphStatus DequantChecker::CheckInputLayoutPerblock(const FiaTilingInfo &fiaInfo)
{
    if (!enablePerblockQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    const std::vector<std::string> unsupportedLayoutList = {
        "BNSD_NBSD", "BSH_NBSD", "BSH_BNSD", "BSND_BNSD", "BSND_NBSD", "NTD", "TND", "TND_NTD"};
    const std::string inputLayout = fiaInfo.opParamInfo.layOut;

    OP_CHECK_IF((std::find(unsupportedLayoutList.begin(), unsupportedLayoutList.end(), inputLayout) !=
                 unsupportedLayoutList.end()),
                OP_LOGE(fiaInfo.opName, "In per-block quant scenario, input layout(%s) is not supported.",
                        fiaInfo.opParamInfo.layOut),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// check per-tensor layout
ge::graphStatus DequantChecker::CheckInputLayoutPertensor(const FiaTilingInfo &fiaInfo)
{
    if (!enablePertensorQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string inputLayout = fiaInfo.opParamInfo.layOut;
    const std::vector<std::string> unsupportedLayoutList = {
        "BNSD_NBSD", "BSND_NBSD", "BSH_NBSD", "BSH_BNSD", "BSND_BNSD", "TND", "NTD", "NTD_TND", "TND_NTD"};

    OP_CHECK_IF(
        (std::find(unsupportedLayoutList.begin(), unsupportedLayoutList.end(), inputLayout) != unsupportedLayoutList.end()),
        OP_LOGE(fiaInfo.opName, "In per-tensor scenario, input layout(%s) is not supported.",
                fiaInfo.opParamInfo.layOut),
        return ge::GRAPH_FAILED);
    

    return ge::GRAPH_SUCCESS;
}

// check mla fullqunat layout
ge::graphStatus DequantChecker::CheckInputLayoutMLAFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableIFAMLAFullQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string inputLayout = fiaInfo.opParamInfo.layOut;
    const std::vector<std::string> supportedLayoutListFP8 = {
        "BSH", "BSND", "BNSD", "TND"};
    const std::vector<std::string> supportedLayoutListINT8 = {
        "BSH", "BSND", "TND", "BSH_NBSD", "BSND_NBSD", "TND_NTD"};
    
    OP_CHECK_IF(fiaInfo.inputQType == ge::DT_FLOAT8_E4M3FN &&
        (std::find(supportedLayoutListFP8.begin(), supportedLayoutListFP8.end(), inputLayout) ==
                 supportedLayoutListFP8.end()),
                OP_LOGE(fiaInfo.opName, "In MLA full quant scenario, input layout(%s) must be BSH/BSND/BNSD/TND "
                        "When input datatype is float8_e4m3.",fiaInfo.opParamInfo.layOut),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.inputQType == ge::DT_INT8 &&
        (std::find(supportedLayoutListINT8.begin(), supportedLayoutListINT8.end(), inputLayout) ==
                 supportedLayoutListINT8.end()),
                OP_LOGE(fiaInfo.opName, "In MLA full quant scenario, input layout(%s) must be BSH/BSND/TND/BSH_NBSD/BSND_NBSD/TND_NTD.",
                        "When input datatype is int8.",fiaInfo.opParamInfo.layOut),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// check layout
ge::graphStatus DequantChecker::CheckInputLayoutFullquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckInputLayoutPerblock(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckInputLayoutPertensor(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckInputLayoutMLAFullquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// check n1 size
ge::graphStatus DequantChecker::CheckN1SizeFullquant(const FiaTilingInfo &fiaInfo)
{
    if (enableIFAMLAFullQuant_) {
        static const std::set<uint32_t> supportNumHeadForMLAFullQuantFP8 = {32U, 64U, 128U};
        static const std::set<uint32_t> supportNumHeadForMLAFullQuantINT8 = {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U};
        OP_CHECK_IF(fiaInfo.inputQType == ge::DT_FLOAT8_E4M3FN &&
            (supportNumHeadForMLAFullQuantFP8.find(fiaInfo.n1Size) == supportNumHeadForMLAFullQuantFP8.end()),
                    OP_LOGE(fiaInfo.opName,
                            "In MLA fullquant scenario, "
                            "the heads num(%u) of query should be in range of {32, 64, 128} When input datatype is float8_e4m3",
                            fiaInfo.n1Size),
                    return ge::GRAPH_FAILED);
         OP_CHECK_IF(fiaInfo.inputQType == ge::DT_INT8 &&
            (supportNumHeadForMLAFullQuantINT8.find(fiaInfo.n1Size) == supportNumHeadForMLAFullQuantINT8.end()),
                    OP_LOGE(fiaInfo.opName,
                            "In MLA fullquant scenario, "
                            "the heads num(%u) of query should be in range of {1, 2, 4, 8, 16, 32, 64, 128} When input datatype is int8",
                            fiaInfo.n1Size),
                    return ge::GRAPH_FAILED);
    } else {  // QN <= 256
        OP_CHECK_IF((fiaInfo.n1Size > N1_LIMIT || fiaInfo.n1Size < NUM1),
                    OP_LOGE(fiaInfo.opName,
                            "In GQA fullquant scenario, the heads num(%u) of query should be in range of [%u, %u].",
                            fiaInfo.n1Size, NUM1, N1_LIMIT),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF((fiaInfo.n1Size % fiaInfo.n2Size != 0),
        OP_LOGE(fiaInfo.opName, "In fullquant scenario, the heads num(%u) of query should be a multiple of KV(%u).",
                fiaInfo.n1Size, fiaInfo.n2Size),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check n2 size
ge::graphStatus DequantChecker::CheckN2SizeFullquant(const FiaTilingInfo &fiaInfo)
{
    if (enableIFAMLAFullQuant_) {
        OP_CHECK_IF((fiaInfo.n2Size != NUM1),
                    OP_LOGE(fiaInfo.opName, "In MLA fullquant scenario, the head num(%u) of KV should be %u.",
                            fiaInfo.n2Size, NUM1),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((fiaInfo.n2Size > N2_LIMIT || fiaInfo.n1Size < NUM1),
            OP_LOGE(fiaInfo.opName, "In GQA fullquant scenario, the head num(%u) of KV should be in range of [%u, %u].",
                    fiaInfo.n2Size, NUM1, N2_LIMIT),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// check QS size
ge::graphStatus DequantChecker::CheckQSSizeFullquant(const FiaTilingInfo &fiaInfo)
{
    if (enableIFAMLAFullQuant_) {
        OP_CHECK_IF((fiaInfo.s1Size > NUM_16 || fiaInfo.s1Size < NUM1),
            OP_LOGE(fiaInfo.opName,
                    "In MLA fullquant scenario, the sequence length(%u) of query should be in range of [%u, %u].",
                    fiaInfo.s1Size, NUM1, NUM_16),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// check g size
ge::graphStatus DequantChecker::CheckGSizeFullquant(const FiaTilingInfo &fiaInfo)
{
    if (enableIFAMLAFullQuant_) { // G <= 128
        OP_CHECK_IF((fiaInfo.gSize < NUM1 || fiaInfo.gSize > NUM_128),
                    OP_LOGE(fiaInfo.opName, "In MLA fullquant scenario, the axis G(%u) should be in range of [%u, %u].",
                            fiaInfo.gSize, NUM1, NUM_128),
                    return ge::GRAPH_FAILED);
    } else {  // G <= 64
        OP_CHECK_IF((fiaInfo.gSize < NUM1 || fiaInfo.gSize > G_LIMIT),
                    OP_LOGE(fiaInfo.opName, "In GQA fullquant scenario, the axis G(%u) should be in range of [%u, %u].",
                            fiaInfo.gSize, NUM1, G_LIMIT),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// check d size
ge::graphStatus DequantChecker::CheckDSizeFullquant(const FiaTilingInfo &fiaInfo)
{
    if (enableIFAMLAFullQuant_) {
        OP_CHECK_IF((fiaInfo.qkHeadDim != NUM_512),
                    OP_LOGE(fiaInfo.opName, "In MLA fullquant scenario, the axis D(%u) of query should be %u.",
                            fiaInfo.qkHeadDim, NUM_512),
                    return ge::GRAPH_FAILED);
    } else if (enablePerblockQuant_) {
        OP_CHECK_IF(
            (fiaInfo.qkHeadDim > NUM_128 || fiaInfo.qkHeadDim < 1),
            OP_LOGE(fiaInfo.opName, "In per-block quant scenario, the axis D(%u) of query and key ony support [1, %u].",
                    fiaInfo.qkHeadDim, NUM_128),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF((fiaInfo.vHeadDim > NUM_128 || fiaInfo.vHeadDim < 1),
                    OP_LOGE(fiaInfo.opName, "In per-block quant scenario, the axis D(%u) of value ony support [1, %u].",
                            fiaInfo.vHeadDim, NUM_128),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((fiaInfo.qkHeadDim > D_LIMIT || fiaInfo.qkHeadDim < NUM1),
                    OP_LOGE(fiaInfo.opName,
                            "In per-tensor quant scenario, the axis D(%u) of query and key ony support [1, %u].",
                            fiaInfo.qkHeadDim, D_LIMIT),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            (fiaInfo.qkHeadDim != fiaInfo.vHeadDim),
            OP_LOGE(fiaInfo.opName, "In per-tensor quant scenario, the axis D of value(%u) should be equal to key(%u).",
                    fiaInfo.vHeadDim, fiaInfo.qkHeadDim),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// check axis
ge::graphStatus DequantChecker::CheckInputAxisFullquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckN1SizeFullquant(fiaInfo) || ge::GRAPH_SUCCESS != CheckN2SizeFullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckQSSizeFullquant(fiaInfo) || ge::GRAPH_SUCCESS != CheckGSizeFullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDSizeFullquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// enableAntiQuant 相关校验函数
// singlePara
ge::graphStatus DequantChecker::CheckSingleParaForAntiquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckAntiquantModeForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckInputKVTypeForAntiquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckAntiquantModeForAntiquant(const FiaTilingInfo &fiaInfo)
{
    // antiquantMode合法性校验
    // keyAntiquantMode
    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
        keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
    }
    if (fiaInfo.opParamInfo.valueAntiquantMode != nullptr) {
        valueAntiquantMode = *fiaInfo.opParamInfo.valueAntiquantMode;
    }
    OP_LOGI(fiaInfo.opName, "keyAntiquantMode is %ld.", keyAntiquantMode);
    OP_LOGI(fiaInfo.opName, "valueAntiquantMode is %ld.", valueAntiquantMode);
    OP_CHECK_IF(((keyAntiquantMode != PER_CHANNEL_MODE) && (keyAntiquantMode != PER_TOKEN_MODE) &&
                 (keyAntiquantMode != PER_TENSOR_HEAD_MODE) && (keyAntiquantMode != PER_TOKEN_HEAD_MODE) &&
                 (keyAntiquantMode != PER_TOKEN_PA_MODE) && (keyAntiquantMode != PER_TOKEN_HEAD_PA_MODE) &&
                 (keyAntiquantMode != PER_TOKEN_GROUP_MODE)),
                OP_LOGE(fiaInfo.opName,
                        "keyAntiquantMode(%ld) is invalid. "
                        "It should be per-channel(0), per-token(1), per-tensor-head(2), per-token-head(3), "
                        "per-token-PA(4), per-token-head-PA(5), per-token-group(6).",
                        keyAntiquantMode),
                return ge::GRAPH_FAILED);
    // valueAntiquantMode
    OP_CHECK_IF(((valueAntiquantMode != PER_CHANNEL_MODE) && (valueAntiquantMode != PER_TOKEN_MODE) &&
                 (valueAntiquantMode != PER_TENSOR_HEAD_MODE) && (valueAntiquantMode != PER_TOKEN_HEAD_MODE) &&
                 (valueAntiquantMode != PER_TOKEN_PA_MODE) && (valueAntiquantMode != PER_TOKEN_HEAD_PA_MODE) &&
                 (valueAntiquantMode != PER_TOKEN_GROUP_MODE)),
                OP_LOGE(fiaInfo.opName,
                        "valueAntiquantMode(%ld) is invalid. "
                        "It should be per-channel(0), per-token(1), per-tensor-head(2), per-token-head(3), "
                        "per-token-PA(4), per-token-head-PA(5), per-token-group(6).",
                        valueAntiquantMode),
                return ge::GRAPH_FAILED);
    bool kPerChnVPerTokFlag = (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_TOKEN_MODE);
    // KV分离场景
    OP_CHECK_IF((!kPerChnVPerTokFlag && (keyAntiquantMode != valueAntiquantMode)),
                OP_LOGE(fiaInfo.opName,
                        "keyAntiquantMode(%ld) and valueAntiquantMode(%ld) are not equal. "
                        "keyAntiquantMode and valueAntiquantMode must be equal when keyAntiquantMode is not 0 "
                        "and valueAntiquantMode is not 1.",
                        keyAntiquantMode, valueAntiquantMode),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckInputKVTypeForAntiquant(const FiaTilingInfo &fiaInfo)
{
    // 根据keyAntiquantMode和valueAntiquantMode，校验输入key、value的datatype的合法性
    auto inputKvType = fiaInfo.inputKvType;
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;
    if (keyAntiquantScaleTensor == nullptr || valueAntiquantScaleTensor == nullptr) {
        // 若不存在keyAntiquantScaleTensor和valueAntiquantScaleTensor，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }

    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
        keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
    }
    if (fiaInfo.opParamInfo.valueAntiquantMode != nullptr) {
        valueAntiquantMode = *fiaInfo.opParamInfo.valueAntiquantMode;
    }
    if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_CHANNEL_MODE) {
        // per-channel/per-tensor模式，支持key/value的数据类型为INT8、INT4(INT32)、HIFLOAT8、FLOAT8_E4M3FN
        OP_CHECK_IF((inputKvType != ge::DT_INT8 && inputKvType != ge::DT_INT4 && inputKvType != ge::DT_HIFLOAT8 &&
                     inputKvType != ge::DT_FLOAT8_E4M3FN),
                    OP_LOGE(fiaInfo.opName,
                            "datatype of key and value(%s) is not supported. "
                            "datatype of key and value must be INT8, INT4(INT32), HIFLOAT8 or FLOAT8_E4M3FN when "
                            "keyAntiquantMode is per-channel(per-tensor) mode and "
                            "valueAntiquantMode is per-channel(per-tensor) mode.",
                            DataTypeToSerialString(inputKvType).c_str()),
                    return ge::GRAPH_FAILED);
        // per-tensor模式，仅当key/value的数类型为INT8时支持
        gert::Shape expectedShape1 = gert::Shape({1});
        auto keyAntiquantScaleShape = keyAntiquantScaleTensor->GetStorageShape();
        if (inputKvType == ge::DT_INT4) {
            OP_CHECK_IF(keyAntiquantScaleShape == expectedShape1,
                        OP_LOGE(fiaInfo.opName,
                                "Datatype of key and value(INT4/INT32) is not supported. "
                                "Datatype of key and value must be INT8 when "
                                "keyAntiquantMode is per-tensor mode and valueAntiquantMode is per-tensor mode."),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF((keyAntiquantScaleShape == expectedShape1) && inputKvType != ge::DT_INT8,
                        OP_LOGE(fiaInfo.opName,
                                "Datatype of key and value(%s) is not supported. "
                                "Datatype of key and value must be INT8 when "
                                "keyAntiquantMode is per-tensor mode and valueAntiquantMode is per-tensor mode.",
                                DataTypeToSerialString(inputKvType).c_str()),
                        return ge::GRAPH_FAILED);
        }
    }
    if (keyAntiquantMode == PER_TOKEN_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        // per-token模式，支持key/value的数据类型为INT8、INT4(INT32)
        OP_CHECK_IF((inputKvType != ge::DT_INT8 && inputKvType != ge::DT_INT4 && inputKvType != ge::DT_FLOAT8_E4M3FN),
                    OP_LOGE(fiaInfo.opName,
                            "datatype of key and value(%s) is not supported. "
                            "datatype of key and value must be INT8, INT4(INT32) or FLOAT8_E4M3FN when "
                            "keyAntiquantMode is per-token mode and valueAntiquantMode is per-token mode.",
                            DataTypeToSerialString(inputKvType).c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_TENSOR_HEAD_MODE && valueAntiquantMode == PER_TENSOR_HEAD_MODE) {
        // per-tensor-head模式，支持key/value的数据类型为INT8
        OP_CHECK_IF((inputKvType != ge::DT_INT8),
                    OP_LOGE(fiaInfo.opName,
                            "datatype of key and value(%s) is not supported. "
                            "datatype of key and value must be INT8 when "
                            "keyAntiquantMode is per-tensor-head mode and "
                            "valueAntiquantMode is per-tensor-head mode.",
                            DataTypeToSerialString(inputKvType).c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_TOKEN_HEAD_MODE && valueAntiquantMode == PER_TOKEN_HEAD_MODE) {
        // per-token-head模式，支持key/value的数据类型为INT8、INT4(INT32)
        OP_CHECK_IF((inputKvType != ge::DT_INT8 && inputKvType != ge::DT_INT4),
                    OP_LOGE(fiaInfo.opName,
                            "datatype of key and value(%s) is not supported. "
                            "datatype of key and value must be INT8 or INT4(INT32) when "
                            "keyAntiquantMode is per-token-head mode and "
                            "valueAntiquantMode is per-token-head mode.",
                            DataTypeToSerialString(inputKvType).c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_TOKEN_PA_MODE && valueAntiquantMode == PER_TOKEN_PA_MODE) {
        // per-token模式使用page attention管理scale/offset，支持key/value数据类型为INT8
        OP_CHECK_IF((inputKvType != ge::DT_INT8 && inputKvType != ge::DT_FLOAT8_E4M3FN),
                    OP_LOGE(fiaInfo.opName,
                            "datatype of key and value(%s) is not supported. "
                            "datatype of key and value must be INT8 or FLOAT8_E4M3FN when "
                            "keyAntiquantMode is per-token-PA mode and "
                            "valueAntiquantMode is per-token-PA mode.",
                            DataTypeToSerialString(inputKvType).c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_TOKEN_HEAD_PA_MODE && valueAntiquantMode == PER_TOKEN_HEAD_PA_MODE) {
        // per-token-head模式使用page attention管理scale/offset，支持key/value的数据类型为INT8
        OP_CHECK_IF((inputKvType != ge::DT_INT8),
                    OP_LOGE(fiaInfo.opName,
                            "datatype of key and value(%s) is not supported. "
                            "datatype of key and value must be INT8 when "
                            "keyAntiquantMode is per-token-head-PA mode and "
                            "valueAntiquantMode is per-token-head-PA mode.",
                            DataTypeToSerialString(inputKvType).c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        // key支持per-channel叠加value支持per-token，支持key/value的数据类型为INT8、INT4(INT32)
        OP_CHECK_IF((inputKvType != ge::DT_INT8 && inputKvType != ge::DT_INT4),
                    OP_LOGE(fiaInfo.opName,
                            "datatype of key and value(%s) is not supported. "
                            "datatype of key and value must be INT8 or INT4(INT32) when "
                            "keyAntiquantMode is per-channel mode and valueAntiquantMode is per-token mode.",
                            DataTypeToSerialString(inputKvType).c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_TOKEN_GROUP_MODE && valueAntiquantMode == PER_TOKEN_GROUP_MODE) {
        // per-token-group模式，支持key/value的数据类型为FLOAT4_E2M1
        OP_CHECK_IF((inputKvType != ge::DT_FLOAT4_E2M1),
                    OP_LOGE(fiaInfo.opName,
                            "datatype of key and value(%s) is not supported. "
                            "datatype of key and value must be FLOAT4_E2M1 when "
                            "keyAntiquantMode is per-token-group mode and "
                            "valueAntiquantMode is per-token-group mode.",
                            DataTypeToSerialString(inputKvType).c_str()),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// existence
ge::graphStatus DequantChecker::CheckExistenceForAntiquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckScaleExistenceForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDescExistenceForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckOffsetExistenceForAntiquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckScaleExistenceForAntiquant(const FiaTilingInfo &fiaInfo)
{
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;

    int64_t antiquantMode = 0;
    if (fiaInfo.opParamInfo.antiquantMode != nullptr) {
        antiquantMode = *fiaInfo.opParamInfo.antiquantMode;
    }

    // 只支持KV分离场景，key和value的antiquantScaleTensor都应非空
    OP_CHECK_IF(keyAntiquantScaleTensor == nullptr || valueAntiquantScaleTensor == nullptr,
                OP_LOGE(fiaInfo.opName,
                        "In antiquant scenario, keyAntiquantScale and valueAntiquantScale must exist!"),
                return ge::GRAPH_FAILED);
    // 不支持KV不分离场景
    OP_CHECK_IF((antiquantMode != 0),
                OP_LOGE(fiaInfo.opName,
                    "Antiquant scenario only supports key/value split mode. antiquantMode "
                    "is not supported."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((fiaInfo.opParamInfo.antiquantScale.tensor != nullptr),
                OP_LOGE(fiaInfo.opName,
                    "Antiquant scenario only supports key/value split mode. antiquantScale "
                    "is not supported."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((fiaInfo.opParamInfo.antiquantOffset.tensor != nullptr),
                OP_LOGE(fiaInfo.opName,
                    "Antiquant scenario only supports key/value split mode. antiquantOffset "
                    "is not supported."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDescExistenceForAntiquant(const FiaTilingInfo &fiaInfo)
{
    // 所有输入若Tensor存在，则Desc必须存在
    // keyantiquantScale
    OP_CHECK_IF((fiaInfo.opParamInfo.keyAntiquantScale.tensor != nullptr &&
                 fiaInfo.opParamInfo.keyAntiquantScale.desc == nullptr),
                OP_LOGE(fiaInfo.opName, "keyAntiqauntScaleTensor exists, but keyAntiquantScaleDesc does not exist."),
                return ge::GRAPH_FAILED);
    // valueAntiquantScale
    OP_CHECK_IF(
        (fiaInfo.opParamInfo.valueAntiquantScale.tensor != nullptr &&
         fiaInfo.opParamInfo.valueAntiquantScale.desc == nullptr),
        OP_LOGE(fiaInfo.opName, "valueAntiqauntScaleTensor exists, but valueAntiquantScaleDesc does not exist."),
        return ge::GRAPH_FAILED);
    // keyAntiquantOffset
    OP_CHECK_IF((fiaInfo.opParamInfo.keyAntiquantOffset.tensor != nullptr &&
                 fiaInfo.opParamInfo.keyAntiquantOffset.desc == nullptr),
                OP_LOGE(fiaInfo.opName, "keyAntiqauntOffsetTensor exists, but keyAntiquantOffsetDesc does not exist."),
                return ge::GRAPH_FAILED);
    // valueAntiquantOffset
    OP_CHECK_IF(
        (fiaInfo.opParamInfo.valueAntiquantOffset.tensor != nullptr &&
         fiaInfo.opParamInfo.valueAntiquantOffset.desc == nullptr),
        OP_LOGE(fiaInfo.opName, "valueAntiqauntOffsetTensor exists, but valueAntiquantOffsetDesc does not exist."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckOffsetExistenceForAntiquant(const FiaTilingInfo &fiaInfo)
{
    // 校验Offset的存在性
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;
    auto &keyAntiquantOffsetTensor = fiaInfo.opParamInfo.keyAntiquantOffset.tensor;
    auto &valueAntiquantOffsetTensor = fiaInfo.opParamInfo.valueAntiquantOffset.tensor;
    auto inputKvType = fiaInfo.opParamInfo.key.desc->GetDataType();

    // keyAntiquantOffset和valueAntiquantOffset要么同时存在，要么同时不存在
    OP_CHECK_IF((keyAntiquantOffsetTensor != nullptr && valueAntiquantOffsetTensor == nullptr),
                OP_LOGE(fiaInfo.opName, "keyAntiquantOffset does not exist, but valueAntiquantOffset exists."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((valueAntiquantOffsetTensor != nullptr && keyAntiquantOffsetTensor == nullptr),
                OP_LOGE(fiaInfo.opName, "valueAntiquantOffset does not exist, but keyAntiquantOffset exists."),
                return ge::GRAPH_FAILED);
    // key、value的datatype为FLOAT8_E4M3FN、HIFLOAT8和FLOAT4_E2M1时，不支持Offset
    if (inputKvType == ge::DT_FLOAT8_E4M3FN || inputKvType == ge::DT_HIFLOAT8 || inputKvType == ge::DT_FLOAT4_E2M1) {
        OP_CHECK_IF((keyAntiquantOffsetTensor != nullptr),
                    OP_LOGE(fiaInfo.opName, "When the datatype is FLOAT8_E4M3FN, HIFLOAT8 or FLOAT4_E2M1, "
                                            "keyAntiquantOffsetTensor is not supported."),
                    return ge::GRAPH_FAILED);
    }
    if (inputKvType == ge::DT_FLOAT8_E4M3FN || inputKvType == ge::DT_HIFLOAT8 || inputKvType == ge::DT_FLOAT4_E2M1) {
        OP_CHECK_IF((valueAntiquantOffsetTensor != nullptr),
                    OP_LOGE(fiaInfo.opName, "When the datatype is FLOAT8_E4M3FN, HIFLOAT8 or FLOAT4_E2M1, "
                                            "valueAntiquantOffsetTensor is not supported."),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// feature
ge::graphStatus DequantChecker::CheckFeatureForAntiquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureExtendForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeaturePAForAntiquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckFeatureExtendForAntiquant(const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF(
        (fiaInfo.inputKvType == ge::DT_INT8 &&fiaInfo.qLayout == FiaLayout::TND),
        OP_LOGE(fiaInfo.opName, "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario,"
                                "the layout of input does not support TND."),
        return ge::GRAPH_FAILED);
    if (fiaInfo.s1Size > 1) {
        int64_t keyAntiquantMode = 0;
        if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
            keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
        }
        OP_CHECK_IF((keyAntiquantMode == PER_TENSOR_HEAD_MODE || keyAntiquantMode == PER_TOKEN_HEAD_MODE ||
                     keyAntiquantMode == PER_TOKEN_PA_MODE || keyAntiquantMode == PER_TOKEN_HEAD_PA_MODE) &&
                     fiaInfo.inputKvType == ge::DT_INT8,
                    OP_LOGE(fiaInfo.opName,
                            "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario,"
                            "if S of query > 1, keyAntiquantMode/valueAntiquantMode 2, 3, 4, 5 are not supported!"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(((keyAntiquantMode == PER_CHANNEL_MODE || keyAntiquantMode == PER_TOKEN_MODE) &&
                    fiaInfo.inputKvType == ge::DT_INT8 &&
                     (fiaInfo.inputQType != ge::DT_BF16 || fiaInfo.outputType != ge::DT_BF16)),
                    OP_LOGE(fiaInfo.opName,
                            "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario,"
                            "if keyAntiquantMode/valueAntiquantMode is 0 or 1, the data type of query and output only "
                            "support BF16!"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(((keyAntiquantMode == PER_CHANNEL_MODE || keyAntiquantMode == PER_TOKEN_MODE) &&
                    fiaInfo.inputKvType == ge::DT_INT8 && fiaInfo.s1Size > NUM_16),
                    OP_LOGE(fiaInfo.opName,
                            "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario,"
                            "if keyAntiquantMode/valueAntiquantMode is 0 or 1, S of query should not be greater than 16!"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(((keyAntiquantMode == PER_CHANNEL_MODE || keyAntiquantMode == PER_TOKEN_MODE) &&
                    fiaInfo.inputKvType == ge::DT_INT8 && !fiaInfo.batchContinuousFlag),
                    OP_LOGE(fiaInfo.opName,
                            "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario,"
                            "if keyAntiquantMode/valueAntiquantMode is 0 or 1, tensorlist is not supported!"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(((keyAntiquantMode == PER_CHANNEL_MODE || keyAntiquantMode == PER_TOKEN_MODE) &&
                    fiaInfo.inputKvType == ge::DT_INT8 && (fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag)),
                    OP_LOGE(fiaInfo.opName,
                            "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario,"
                            "if keyAntiquantMode/valueAntiquantMode is 0 or 1, leftpadding is not supported!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(((keyAntiquantMode == PER_CHANNEL_MODE || keyAntiquantMode == PER_TOKEN_MODE) &&
                    fiaInfo.inputKvType == ge::DT_INT8 && fiaInfo.pageAttentionFlag),
                    OP_LOGE(fiaInfo.opName,
                            "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario,"
                            "if keyAntiquantMode/valueAntiquantMode is 0 or 1, page attention is not supported!"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF((fiaInfo.inputKvType == ge::DT_INT4 || fiaInfo.inputKvType == ge::DT_INT32),
                    OP_LOGE(fiaInfo.opName, "In keyAntiquant/valueAntiquant split mode scenario, int4 and int32 data "
                                            "types are not supported for the key and value."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckFeaturePAForAntiquant(const FiaTilingInfo &fiaInfo)
{
    // 校验交叉特性page attention约束
    if (!fiaInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;
    auto &keyAntiquantOffsetTensor = fiaInfo.opParamInfo.keyAntiquantOffset.tensor;
    auto &valueAntiquantOffsetTensor = fiaInfo.opParamInfo.valueAntiquantOffset.tensor;
    int32_t blockSize = fiaInfo.blockSize;
    uint32_t maxBlockNumPerBatch = fiaInfo.maxBlockNumPerBatch;

    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
        keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
    }
    if (fiaInfo.opParamInfo.valueAntiquantMode != nullptr) {
        valueAntiquantMode = *fiaInfo.opParamInfo.valueAntiquantMode;
    }
    // per-token模式，per-token叠加per-head模式，antiquantScale输入的最后一维需要大于等于maxBlockNumPerBatch * blockSize
    // per-token模式
    if (keyAntiquantMode == PER_TOKEN_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        // shape 可能为(1, B, S)或(B, S)
        uint32_t dimNum = keyAntiquantScaleTensor->GetStorageShape().GetDimNum();
        OP_CHECK_IF(
            (keyAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - 1) < maxBlockNumPerBatch * blockSize),
            OP_LOGE(fiaInfo.opName,
                    "The last dimension(%u) of keyAntiquantScale is less than "
                    "maxBlockNumPerBatch(%u) * blockSize(%u). "
                    "The last dimension of keyAntiquantScale should be larger than or equal to "
                    "maxBlockNumPerBatch * blockSize when "
                    "keyAntiquantMode, valueAntiquantMode are per-token mode and "
                    "keyAntiquant/valuAntiquant is splited.",
                    keyAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - 1), maxBlockNumPerBatch, blockSize),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (valueAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - 1) < maxBlockNumPerBatch * blockSize),
            OP_LOGE(fiaInfo.opName,
                    "The last dimension(%u) of valueAntiquantScale is less than "
                    "maxBlockNumPerBatch(%u) * blockSize(%u). "
                    "The last dimension of valueAntiquantScale should be larger than or equal to "
                    "maxBlockNumPerBatch * blockSize when "
                    "keyAntiquantMode, valueAntiquantMode are per-token mode and "
                    "keyAntiquant/valuAntiquant is splited.",
                    valueAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - 1), maxBlockNumPerBatch, blockSize),
            return ge::GRAPH_FAILED);
    }
    // per-token叠加per-head模式
    // shape可能为(B,N,S)
    if (keyAntiquantMode == PER_TOKEN_HEAD_MODE && valueAntiquantMode == PER_TOKEN_HEAD_MODE) {
        uint32_t dimNum = keyAntiquantScaleTensor->GetStorageShape().GetDimNum();
        OP_CHECK_IF(
            (keyAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - 1) < maxBlockNumPerBatch * blockSize),
            OP_LOGE(fiaInfo.opName,
                    "The last dimension(%u) of keyAntiquantScale is less than "
                    "maxBlockNumPerBatch(%u) * blockSize(%u). "
                    "The last dimension of keyAntiquantScale should be larger than or equal to "
                    "maxBlockNumPerBatch * blockSize when "
                    "keyAntiquantMode, valueAntiquantMode are per-token-head mode and "
                    "keyAntiquant/valuAntiquant is splited.",
                    keyAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - 1), maxBlockNumPerBatch, blockSize),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (valueAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - 1) < maxBlockNumPerBatch * blockSize),
            OP_LOGE(fiaInfo.opName,
                    "The last dimension(%u) of valueAntiquantScale is less than "
                    "maxBlockNumPerBatch(%u) * blockSize(%u). "
                    "The last dimension of valueAntiquantScale should be larger than or equal to "
                    "maxBlockNumPerBatch * blockSize when "
                    "keyAntiquantMode, valueAntiquantMode are per-token-head mode and "
                    "keyAntiquant/valuAntiquant is splited.",
                    valueAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - 1), maxBlockNumPerBatch, blockSize),
            return ge::GRAPH_FAILED);
    }
    // per-token-group模式，antiquantScale输入倒数第二维需要大于等于maxBlockNumPerSeq * blockSize
    // shape为(1,B,N,S,D/32)
    if (keyAntiquantMode == PER_TOKEN_GROUP_MODE && valueAntiquantMode == PER_TOKEN_GROUP_MODE) {
        uint32_t dimNum = keyAntiquantScaleTensor->GetStorageShape().GetDimNum();
        OP_CHECK_IF(
            (keyAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - 2) < maxBlockNumPerBatch * blockSize),
            OP_LOGE(fiaInfo.opName,
                    "The second-to-last dimension(%u) of keyAntiquantScale is less than "
                    "maxBlockNumPerSeq(%u) * blockSize(%u). "
                    "The second-to-last dimension of keyAntiquantScale should be larger than or equal to "
                    "maxBlockNumPerSeq * blockSize when "
                    "keyAntiquantMode, valueAntiquantMode are per-token-group mode and "
                    "keyAntiquant/valuAntiquant is splited.",
                    keyAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - 2), maxBlockNumPerBatch, blockSize),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// multipara
ge::graphStatus DequantChecker::CheckMultiParaForAntiquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckScaleTypeForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckScaleShapeForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckOffsetTypeForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckOffsetShapeForAntiquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckScaleTypeForAntiquant(const FiaTilingInfo &fiaInfo)
{
    // 校验Scale矩阵的datatype的合法性
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;

    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
        keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
    }
    if (fiaInfo.opParamInfo.valueAntiquantMode != nullptr) {
        valueAntiquantMode = *fiaInfo.opParamInfo.valueAntiquantMode;
    }
    auto &keyAntiquantScaleDesc = fiaInfo.opParamInfo.keyAntiquantScale.desc;
    auto &valueAntiquantScaleDesc = fiaInfo.opParamInfo.valueAntiquantScale.desc;
    auto &queryDesc = fiaInfo.opParamInfo.query.desc;

    if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_CHANNEL_MODE) {
        // per-channel/per-tensor模式，keyAntiquantScale和valueAntiquantScale的数据类型与query相同
        OP_CHECK_IF((keyAntiquantScaleDesc->GetDataType() != queryDesc->GetDataType()),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of keyAntiquantScale(%s) and query(%s) are different. "
                            "The datatype of keyAntiquantScale and query must be the same when "
                            "keyAntiquantMode is per-channel mode or per-tensor mode and "
                            "keyAntiquant/valueAntiquant is in split mode.",
                            DataTypeToSerialString(keyAntiquantScaleDesc->GetDataType()).c_str(),
                            DataTypeToSerialString(queryDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((valueAntiquantScaleDesc->GetDataType() != queryDesc->GetDataType()),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of valueAntiquantScale(%s) and query(%s) are different. "
                            "The datatype of valueAntiquantScale and query must be the same when "
                            "valueAntiquantMode is per-channel mode or per-tensor mode and "
                            "keyAntiquant/valueAntiquant is in split mode.",
                            DataTypeToSerialString(valueAntiquantScaleDesc->GetDataType()).c_str(),
                            DataTypeToSerialString(queryDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    }

    if (keyAntiquantMode == PER_TOKEN_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        // per-token模式，keyAntiquantScale和valueAntiquantScale的数据类型固定为FLOAT32
        OP_CHECK_IF((keyAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of keyAntiquantScale(%s) is not FLOAT32. "
                            "The datatype of keyAntiquantScale must be FLOAT32 when "
                            "keyAntiquantMode is per-token mode and "
                            "keyAntiquant/valueAntiquant is in split mode.",
                            DataTypeToSerialString(keyAntiquantScaleDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((valueAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of valueAntiquantScale(%s) is not FLOAT32. "
                            "The datatype of valueAntiquantScale must be FLOAT32 when "
                            "valueAntiquantMode is per-token mode and "
                            "valueAntiquant/valueAntiquant is in split mode.",
                            DataTypeToSerialString(valueAntiquantScaleDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_TENSOR_HEAD_MODE && valueAntiquantMode == PER_TENSOR_HEAD_MODE) {
        // per-tensor-head模式，keyAntiquantScale和valueAntiquantScale的数据类型与query相同
        OP_CHECK_IF((keyAntiquantScaleDesc->GetDataType() != queryDesc->GetDataType()),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of keyAntiquantScale(%s) and query(%s) are different. "
                            "The datatype of keyAntiquantScale and query must be the same when "
                            "keyAntiquantMode is per-tensor-head mode and "
                            "keyAntiquant/valueAntiquant is in split mode.",
                            DataTypeToSerialString(keyAntiquantScaleDesc->GetDataType()).c_str(),
                            DataTypeToSerialString(queryDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((valueAntiquantScaleDesc->GetDataType() != queryDesc->GetDataType()),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of valueAntiquantScale(%s) and query(%s) are different. "
                            "The datatype of valueAntiquantScale and query must be the same when "
                            "valueAntiquantMode is per-tensor-head mode and "
                            "keyAntiquant/valueAntiquant is in split mode.",
                            DataTypeToSerialString(valueAntiquantScaleDesc->GetDataType()).c_str(),
                            DataTypeToSerialString(queryDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_TOKEN_HEAD_MODE && valueAntiquantMode == PER_TOKEN_HEAD_MODE) {
        // per-token-head模式，keyAntiquantScale和valueAntiquantScale的数据类型固定为FLOAT32
        OP_CHECK_IF((keyAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of keyAntiquantScale(%s) is not FLOAT32. "
                            "The datatype of keyAntiquantScale must be FLOAT32 when "
                            "keyAntiquantMode is per-token-head mode and "
                            "keyAntiquant/valueAntiquant is in split mode.",
                            DataTypeToSerialString(keyAntiquantScaleDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((valueAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of valueAntiquantScale(%s) is not FLOAT32. "
                            "The datatype of valueAntiquantScale must be FLOAT32 when "
                            "valueAntiquantMode is per-token-head mode and "
                            "valueAntiquant/valueAntiquant is in split mode.",
                            DataTypeToSerialString(valueAntiquantScaleDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_TOKEN_PA_MODE && valueAntiquantMode == PER_TOKEN_PA_MODE) {
        // per-token-PA模式，数据类型固定为FLOAT32
        OP_CHECK_IF((keyAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of keyAntiquantScale(%s) is not FLOAT32. "
                            "The datatype of keyAntiquantScale must be FLOAT32 when "
                            "keyAntiquantMode is per-token-PA mode and "
                            "keyAntiquant/valueAntiquant is in split mode.",
                            DataTypeToSerialString(keyAntiquantScaleDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((valueAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of valueAntiquantScale(%s) is not FLOAT32. "
                            "The datatype of valueAntiquantScale must be FLOAT32 when "
                            "valueAntiquantMode is per-token-PA mode and "
                            "valueAntiquant/valueAntiquant is in split mode.",
                            DataTypeToSerialString(valueAntiquantScaleDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        // key支持per-channel叠加value支持per-token
        // keyAntiquantScale的数据类型与query相同
        OP_CHECK_IF((keyAntiquantScaleDesc->GetDataType() != queryDesc->GetDataType()),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of keyAntiquantScale(%s) and query(%s) are different. "
                            "The datatype of keyAntiquantScale and query must be the same when "
                            "keyAntiquantMode is per-channel mode and valueAntiquantMode is per-token mode.",
                            DataTypeToSerialString(keyAntiquantScaleDesc->GetDataType()).c_str(),
                            DataTypeToSerialString(queryDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
        // valueAntiquantScale的数据类型固定为FLOAT32
        OP_CHECK_IF((valueAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of valueAntiquantScale(%s) is not FLOAT32. "
                            "The datatype of valueAntiquantScale must be FLOAT32 when "
                            "keyAntiquantMode is per-channel mode and valueAntiquantMode is per-token mode.",
                            DataTypeToSerialString(valueAntiquantScaleDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_TOKEN_GROUP_MODE && valueAntiquantMode == PER_TOKEN_GROUP_MODE) {
        // per-token-group模式，keyAntiquantScale和valueAntiquantScale的数据类型固定为FLOAT8_E8M0
        OP_CHECK_IF((keyAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of keyAntiquantScale(%s) is not FLOAT8_E8M0. "
                            "The datatype of keyAntiquantScale must be FLOAT8_E8M0 when "
                            "keyAntiquantMode is per-token-group mode and "
                            "keyAntiquant/valueAntiquant is in split mode.",
                            DataTypeToSerialString(keyAntiquantScaleDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((valueAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of valueAntiquantScale(%s) is not FLOAT8_E8M0. "
                            "The datatype of valueAntiquantScale must be FLOAT8_E8M0 when "
                            "valueAntiquantMode is per-token-group mode and "
                            "valueAntiquant/valueAntiquant is in split mode.",
                            DataTypeToSerialString(valueAntiquantScaleDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckScaleShapeForAntiquant(const FiaTilingInfo &fiaInfo)
{
    // 校验scale矩阵的shape的合法性
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;

    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
        keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
    }
    if (fiaInfo.opParamInfo.valueAntiquantMode != nullptr) {
        valueAntiquantMode = *fiaInfo.opParamInfo.valueAntiquantMode;
    }

    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    gert::Shape valueAntiquantScaleTensorShape = valueAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    uint32_t valueAntiquantScaleTensorDimNum = valueAntiquantScaleTensorShape.GetDimNum();
    uint32_t numKeyValueHeads = fiaInfo.n2Size;
    uint32_t headDim = fiaInfo.qkHeadDim;
    uint32_t batchSize = fiaInfo.bSize;
    uint64_t seqLength = fiaInfo.s2Size;
    if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_CHANNEL_MODE) {
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerChannelPerTensorMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (keyAntiquantMode == PER_TOKEN_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerTokenMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (keyAntiquantMode == PER_TENSOR_HEAD_MODE && valueAntiquantMode == PER_TENSOR_HEAD_MODE) {
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerTensorHeadMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (keyAntiquantMode == PER_TOKEN_HEAD_MODE && valueAntiquantMode == PER_TOKEN_HEAD_MODE) {
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerTokenHeadMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (keyAntiquantMode == PER_TOKEN_PA_MODE && valueAntiquantMode == PER_TOKEN_PA_MODE) {
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerTokenPAMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (keyAntiquantMode == PER_TOKEN_HEAD_PA_MODE && valueAntiquantMode == PER_TOKEN_HEAD_PA_MODE) {
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerTokenHeadPAMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        // key支持per-channel叠加value支持per-token
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerChannelPerTensorMode(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckVScaleShapeForPerTokenMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (keyAntiquantMode == PER_TOKEN_GROUP_MODE && valueAntiquantMode == PER_TOKEN_GROUP_MODE) {
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerTokenGroupMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (!((keyAntiquantMode == PER_CHANNEL_MODE) && (valueAntiquantMode == PER_TOKEN_MODE))) {
        // 除key支持per-channel叠加value支持per-token模式以外
        // keyAntiquantScale的shape必须与valueAntiquantScale一致
        OP_CHECK_IF((keyAntiquantScaleTensorShape != valueAntiquantScaleTensorShape),
                    OP_LOGE(fiaInfo.opName,
                            "The shape of keyAntiquantScale and valueAntiquantScale are different. "
                            "The shape of keyAntiquantScale and valueAntiquantScale must be the same when "
                            "keyAntiquantMode is not per-channal mode and valueAntiquantMode is not per-token mode."),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckOffsetTypeForAntiquant(const FiaTilingInfo &fiaInfo)
{
    // 校验offset矩阵的datatype的合法性
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;

    // keyAntiquantOffset的datatype与keyAntiquantScale一致
    auto &keyAntiquantOffsetDesc = fiaInfo.opParamInfo.keyAntiquantOffset.desc;
    auto &keyAntiquantScaleDesc = fiaInfo.opParamInfo.keyAntiquantScale.desc;
    if (keyAntiquantOffsetDesc != nullptr && keyAntiquantScaleDesc != nullptr) {
        OP_CHECK_IF((keyAntiquantOffsetDesc->GetDataType() != keyAntiquantScaleDesc->GetDataType()),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of keyAntiquantOffset(%s) and keyAntiquantScale(%s) are different. "
                            "The datatype of keyAntiquantOffset and keyAntiquantScale must be the same when "
                            "keyAntiquant/valueAntiquant is in split mode.",
                            DataTypeToSerialString(keyAntiquantOffsetDesc->GetDataType()).c_str(),
                            DataTypeToSerialString(keyAntiquantScaleDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    }
    // valueAntiquantOffset的datatype与valueAntiquantScale一致
    auto &valueAntiquantOffsetDesc = fiaInfo.opParamInfo.valueAntiquantOffset.desc;
    auto &valueAntiquantScaleDesc = fiaInfo.opParamInfo.valueAntiquantScale.desc;
    if (valueAntiquantOffsetDesc != nullptr && valueAntiquantScaleDesc != nullptr) {
        OP_CHECK_IF((valueAntiquantOffsetDesc->GetDataType() != valueAntiquantScaleDesc->GetDataType()),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of valueAntiquantOffset(%s) and valueAntiquantScale(%s) are different. "
                            "The datatype of valueAntiquantOffset and valueAntiquantScale must be the same when "
                            "keyAntiquant/valueAntiquant is in split mode.",
                            DataTypeToSerialString(valueAntiquantOffsetDesc->GetDataType()).c_str(),
                            DataTypeToSerialString(valueAntiquantScaleDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckOffsetShapeForAntiquant(const FiaTilingInfo &fiaInfo)
{
    // 校验offset矩阵的shape的合法性
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;
    auto &keyAntiquantOffsetTensor = fiaInfo.opParamInfo.keyAntiquantOffset.tensor;
    auto &valueAntiquantOffsetTensor = fiaInfo.opParamInfo.valueAntiquantOffset.tensor;

    if (keyAntiquantScaleTensor == nullptr || valueAntiquantScaleTensor == nullptr ||
        keyAntiquantOffsetTensor == nullptr || valueAntiquantOffsetTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    gert::Shape keyAntiquantOffsetTensorShape = keyAntiquantOffsetTensor->GetStorageShape();
    gert::Shape valueAntiquantOffsetTensorShape = valueAntiquantOffsetTensor->GetStorageShape();
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    gert::Shape valueAntiquantScaleTensorShape = valueAntiquantScaleTensor->GetStorageShape();

    OP_CHECK_IF((keyAntiquantOffsetTensorShape != keyAntiquantScaleTensorShape),
                OP_LOGE(fiaInfo.opName,
                        "The shape of keyAntiquantOffset and keyAntiquantScale are different. "
                        "The shape of keyAntiquantOffset and keyAntiquantScale must be the same when "
                        "keyAntiquant/valueAntiquant is in split mode."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (valueAntiquantOffsetTensorShape != valueAntiquantScaleTensorShape),
        OP_LOGE(fiaInfo.opName,
                "The shape of valueAntiquantOffset and valueAntiquantScale are different. "
                "The shape of valueAntiquantOffset and valueAntiquantScale must be the same when "
                "keyAntiquant/valueAntiquant is in split mode."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckScaleShapeForPerChannelPerTensorMode(const FiaTilingInfo &fiaInfo)
{
    auto &antiquantScaleTensor = fiaInfo.opParamInfo.antiquantScale.tensor;
    gert::Shape antiquantScaleTensorShape = antiquantScaleTensor->GetStorageShape();
    uint32_t antiquantScaleTensorDimNum = antiquantScaleTensorShape.GetDimNum();
    uint32_t numKeyValueHeads = fiaInfo.n2Size;
    uint32_t headDim = fiaInfo.qkHeadDim;
    // per-channel/per-tensor模式
    if (antiquantScaleTensorDimNum == DIM_NUM_1) {
        // per-tensor模式，antiquantSclae的dimNum为1
        // antiquantScale的shape支持[2]
        gert::Shape expectedShape2 = gert::Shape({2});
        OP_CHECK_IF((antiquantScaleTensorShape != expectedShape2),
                    OP_LOGE(fiaInfo.opName,
                            "The shape of antiquantScale([%u]) is not [2]. "
                            "The shape of antiquantScale must be [2] when "
                            "antiquantMode is per-tensor mode and "
                            "keyAntiquant/valueAntiquant is not in split mode.",
                            antiquantScaleTensorShape.GetDim(DIM_NUM_0)),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    // per-channel模式，antiquantScale的dimNum为2,3,4
    if (antiquantScaleTensorDimNum == DIM_NUM_2) {
        // antiquantScale的shape支持[2,H]
        gert::Shape expectedShape2H = gert::Shape({2, numKeyValueHeads * headDim});
        OP_CHECK_IF((antiquantScaleTensorShape != expectedShape2H),
                    OP_LOGE(fiaInfo.opName,
                            "The shape of antiquantScale([%u, %u]) is not [2, H(%u)]. "
                            "The shape of antiquant must be [2, H] when "
                            "antiquantMode is per-channel mode, "
                            "the dimNum of antiquantScale is 2 and "
                            "keyAntiquant/valueAntiquant is not in split mode.",
                            antiquantScaleTensorShape.GetDim(DIM_NUM_0), antiquantScaleTensorShape.GetDim(DIM_NUM_1),
                            numKeyValueHeads * headDim),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    if (antiquantScaleTensorDimNum == DIM_NUM_3) {
        // antiquantScale的shape支持[2, N, D]
        gert::Shape expectedShape2ND = gert::Shape({2, numKeyValueHeads, headDim});
        OP_CHECK_IF((antiquantScaleTensorShape != expectedShape2ND),
                    OP_LOGE(fiaInfo.opName,
                            "The shape of antiquantScale([%u, %u, %u]) is not [2, N(%u), D(%u)]. "
                            "The shape of antiquantScale must be [2, N, D] when "
                            "antiquantMode is per-channel mode, "
                            "the dimNum of antiquantScale is 3 and "
                            "keyAntiquant/valueAntiquant is not in split mode.",
                            antiquantScaleTensorShape.GetDim(DIM_NUM_0), antiquantScaleTensorShape.GetDim(DIM_NUM_1),
                            antiquantScaleTensorShape.GetDim(DIM_NUM_2), numKeyValueHeads, headDim),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    if (antiquantScaleTensorDimNum == DIM_NUM_4) {
        // antiquantScale的shape支持[2, N, 1, D]
        gert::Shape expectedShape2N1D = gert::Shape({2, numKeyValueHeads, 1, headDim});
        OP_CHECK_IF((antiquantScaleTensorShape != expectedShape2N1D),
                    OP_LOGE(fiaInfo.opName,
                            "The shape of antiquantScale[%u, %u, %u, %u] is not [2, N(%u), 1, D(%u)]. "
                            "The shape of antiquantScale must be [2, N, 1, D] when "
                            "antiquantMode is per-channel mode, "
                            "the dimNum of antiquantScale is 4 and "
                            "keyAntiquant/valueAntiquant is not in split mode.",
                            antiquantScaleTensorShape.GetDim(DIM_NUM_0), antiquantScaleTensorShape.GetDim(DIM_NUM_1),
                            antiquantScaleTensorShape.GetDim(DIM_NUM_2), antiquantScaleTensorShape.GetDim(DIM_NUM_3),
                            numKeyValueHeads, headDim),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(fiaInfo.opName, "The dimNum(%u) of antiquantScale is invalid.", antiquantScaleTensorDimNum);
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckScaleShapeForPerTokenMode(const FiaTilingInfo &fiaInfo)
{
    auto &antiquantScaleTensor = fiaInfo.opParamInfo.antiquantScale.tensor;
    gert::Shape antiquantScaleTensorShape = antiquantScaleTensor->GetStorageShape();
    uint32_t antiquantScaleTensorDimNum = antiquantScaleTensorShape.GetDimNum();
    uint32_t batchSize = fiaInfo.bSize;
    uint64_t seqLength = fiaInfo.s2Size;
    // per-token模式
    // antiquantScale的shape支持[2, B, S]
    if (antiquantScaleTensorDimNum == DIM_NUM_3) {
        gert::Shape expectedShape2BS = gert::Shape({2, batchSize, seqLength});
        OP_CHECK_IF((antiquantScaleTensorShape != expectedShape2BS),
                    OP_LOGE(fiaInfo.opName,
                            "The shape of antiquantScale[%u, %u, %u] is not [2, B(%u), S(%llu)]. "
                            "The shape of antiquantScale must be [2, B, S] when "
                            "antiquantMode is per-token mode and "
                            "keyAntiquant/valueAntiquant is not in split mode.",
                            antiquantScaleTensorShape.GetDim(DIM_NUM_0), antiquantScaleTensorShape.GetDim(DIM_NUM_1),
                            antiquantScaleTensorShape.GetDim(DIM_NUM_2), batchSize, seqLength),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(fiaInfo.opName, "The dimNum(%u) of antiquantScale is invalid.", antiquantScaleTensorDimNum);
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckKScaleShapeForPerChannelPerTensorMode(const FiaTilingInfo &fiaInfo)
{
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    uint32_t numKeyValueHeads = fiaInfo.n2Size;
    uint32_t headDim = fiaInfo.qkHeadDim;
    // per-channel/per-tensor模式
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_1) {
        // 维度为1，shape可能为[H]，per-channel模式
        // 维度为1，shape可能为[1]，per-tensor模式
        gert::Shape expectedShapeH = gert::Shape({numKeyValueHeads * headDim});
        gert::Shape expectedShape1 = gert::Shape({1});
        OP_CHECK_IF(
            ((keyAntiquantScaleTensorShape != expectedShapeH) && (keyAntiquantScaleTensorShape != expectedShape1)),
            OP_LOGE(fiaInfo.opName,
                    "The shape of keyAntiquantScale([%u]) is not [H(%u)] or [1]. "
                    "The shape of keyAntiquantScale must be [H]"
                    "when keyAntiquantMode is per-channel mode "
                    "or [1] when keyAntiquantMode is per-tensor mode "
                    "when dimNum of keyAntiquantScale is 1.",
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0), numKeyValueHeads * headDim),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_2) {
        // 维度为2，shape可能为[1, H]或者[N, D]
        gert::Shape expectedShape1H = gert::Shape({1, numKeyValueHeads * headDim});
        gert::Shape expectedShapeND = gert::Shape({numKeyValueHeads, headDim});
        OP_CHECK_IF(
            ((keyAntiquantScaleTensorShape != expectedShape1H) && (keyAntiquantScaleTensorShape != expectedShapeND)),
            OP_LOGE(fiaInfo.opName,
                    "The shape of keyAntiquantScale([%u, %u]) is not [1, H(%u)] or [N(%u), D(%u)]. "
                    "The shape of keyAntiquantScale must be [1, H] or [N, D] when "
                    "keyAntiquantMode is per-channel mode, dimNum of keyAntiquantScale is 2.",
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0), keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1),
                    numKeyValueHeads * headDim, numKeyValueHeads, headDim),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_3) {
        // 维度为3，shape可能为[1, N, D]或者[N, 1, D]
        gert::Shape expectedShape1ND = gert::Shape({1, numKeyValueHeads, headDim});
        gert::Shape expectedShapeN1D = gert::Shape({numKeyValueHeads, 1, headDim});
        OP_CHECK_IF(
            ((keyAntiquantScaleTensorShape != expectedShape1ND) && (keyAntiquantScaleTensorShape != expectedShapeN1D)),
            OP_LOGE(fiaInfo.opName,
                    "The shape of keyAntiquantScale[%u, %u, %u] is not [1, N(%u), D(%u)] "
                    "or [N(%u), 1, D(%u)]. "
                    "The shape of keyAntiquantScale must be [1, N, D] or [N, 1, D] when "
                    "keyAntiquantMode is per-channel mode, dimNum of keyAntiquantScale is 3.",
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0), keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1),
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_2), numKeyValueHeads, headDim, numKeyValueHeads,
                    headDim),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_4) {
        // 维度为4，shape仅支持[1, N, 1, D] 或者 [1, 1, N, D]
        gert::Shape expectedShape1N1D = gert::Shape({1, numKeyValueHeads, 1, headDim});
        gert::Shape expectedShape11ND = gert::Shape({1, 1, numKeyValueHeads, headDim});
        OP_CHECK_IF(
            ((keyAntiquantScaleTensorShape != expectedShape1N1D && keyAntiquantScaleTensorShape != expectedShape11ND)),
            OP_LOGE(fiaInfo.opName,
                    "The shape of keyAntiquantScale([%u, %u, %u, %u]) is not [1, N(%u), 1, D(%u)] or "
                    "[1, 1, N(%u), D(%u)]. The shape of keyAntiquantScale must be [1, N, 1, D] when "
                    "keyAntiquantMode is per-channel mode, dimNum of keyAntiquantScale is 4.",
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0), keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1),
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_2), keyAntiquantScaleTensorShape.GetDim(DIM_NUM_3),
                    numKeyValueHeads, headDim, numKeyValueHeads, headDim),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(fiaInfo.opName, "The dimNum(%u) of keyAntiquantScale is invalid.", keyAntiquantScaleTensorDimNum);
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckKScaleShapeForPerTokenMode(const FiaTilingInfo &fiaInfo)
{
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    uint32_t batchSize = fiaInfo.bSize;
    uint64_t seqLength = fiaInfo.s2Size;
    // per-token模式
    // 支持shape为[1, B, S],[B, S]
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_2) {
        // 维度为2，shape可能为[B, >=KV_S]
        OP_CHECK_IF(keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0) != batchSize ||
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1) < seqLength,
                    OP_LOGE(fiaInfo.opName,
                            "The shape of keyAntiquantScale([%u, %u]) is not supported."
                            "The shape of keyAntiquantScale must be "
                            "[1, B(%llu), >=S(%llu)] or [B(%llu), >=S(%llu)] when keyAntiquantMode is per-token mode.",
                            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0),
                            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1), batchSize, seqLength, batchSize, seqLength),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_3) {
        // 维度为3，shape可能为[1, B, >=KV_S]
        OP_CHECK_IF(keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0) != 1 ||
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1) != batchSize ||
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_2) < seqLength,
                    OP_LOGE(fiaInfo.opName,
                            "The shape of keyAntiquantScale([%u, %u]) is not supported."
                            "The shape of keyAntiquantScale must be "
                            "[1, B(%llu), >=S(%llu)] or [B(%llu), >=S(%llu)] when keyAntiquantMode is per-token mode.",
                            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0),
                            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1), batchSize, seqLength, batchSize, seqLength),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(fiaInfo.opName, "The dimNum(%u) of keyAntiquantScale is invalid.", keyAntiquantScaleTensorDimNum);
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckKScaleShapeForPerTensorHeadMode(const FiaTilingInfo &fiaInfo)
{
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    uint32_t numKeyValueHeads = fiaInfo.n2Size;
    // per-tensor叠加per-head模式
    // shape仅支持[N]
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_1) {
        gert::Shape expectedShapeN = gert::Shape({numKeyValueHeads});
        OP_CHECK_IF((keyAntiquantScaleTensorShape != expectedShapeN),
                    OP_LOGE(fiaInfo.opName,
                            "The shape of keyAntiquantScale([%u]) is not [N(%u)]. "
                            "The shape of keyAntiquantScale must be [N] when "
                            "keyAntiquantMode is per-tensor-head mode.",
                            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0), numKeyValueHeads),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(fiaInfo.opName, "The dimNum(%u) of keyAntiquantScale is invalid.", keyAntiquantScaleTensorDimNum);
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckKScaleShapeForPerTokenHeadMode(const FiaTilingInfo &fiaInfo)
{
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    uint32_t batchSize = fiaInfo.bSize;
    uint32_t numKeyValueHeads = fiaInfo.n2Size;
    uint64_t seqLength = fiaInfo.s2Size;
    // per-token叠加per-head模式
    // shape仅支持[B, N, >=S]
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_3) {
        OP_CHECK_IF(keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0) != batchSize ||
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1) != numKeyValueHeads ||
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_2) < seqLength,
                    OP_LOGE(fiaInfo.opName,
                            "The shape of keyAntiquantScale([%u, %u, %u]) is not supported."
                            "The shape of keyAntiquantScale must be "
                            "[1, B(%llu), >=S(%llu)] or [B(%llu), >=S(%llu)] when keyAntiquantMode is per-token mode.",
                            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0),
                            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1), 
                            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_2),
                            batchSize, seqLength, batchSize, seqLength),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(fiaInfo.opName, "The dimNum(%u) of keyAntiquantScale is invalid.", keyAntiquantScaleTensorDimNum);
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckKScaleShapeForPerTokenPAMode(const FiaTilingInfo &fiaInfo)
{
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    // per-token模式使用page attention管理scale/offset
    // shape仅支持[blockNum, blockSize]
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_2) {
        uint32_t blockNum = fiaInfo.totalBlockNum;
        uint32_t blockSize = fiaInfo.blockSize;
        gert::Shape expectedShape = gert::Shape({blockNum, blockSize});
        OP_CHECK_IF((keyAntiquantScaleTensorShape != expectedShape),
                    OP_LOGE(fiaInfo.opName,
                            "The shape of keyAntiquantScale[%u, %u] is not [blockNum(%u), blockSize(%u)]. "
                            "The shape of keyAntiquantScale must be [blockNum, blockSize] when "
                            "keyAntiquantMode is per-token-pa mode.",
                            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0),
                            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1), blockNum, blockSize),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(fiaInfo.opName, "The dimNum(%u) of keyAntiquantScale is invalid.", keyAntiquantScaleTensorDimNum);
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckKScaleShapeForPerTokenHeadPAMode(const FiaTilingInfo &fiaInfo)
{
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    uint32_t numKeyValueHeads = fiaInfo.n2Size;
    // per-token叠加per-head模式并使用page attention管理scale/offset
    // shape仅支持[blockNum, N, blockSize]
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_3) {
        uint32_t blockNum = fiaInfo.totalBlockNum;
        uint32_t blockSize = fiaInfo.blockSize;
        gert::Shape expectedShape = gert::Shape({blockNum, numKeyValueHeads, blockSize});
        OP_CHECK_IF(
            ((keyAntiquantScaleTensorShape != expectedShape)),
            OP_LOGE(fiaInfo.opName,
                    "The shape of keyAntiquantScale([%u, %u, %u]) is not "
                    "[blockNum(%u), N(%u), blockSize(%u)]. "
                    "The shape of keyAntiquantScale must be [blockNum, N, blockSize] when "
                    "keyAntiquantMode is per-token-head-pa mode.",
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0), keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1),
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_2), blockNum, numKeyValueHeads, blockSize),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(fiaInfo.opName, "The dimNum(%u) of keyAntiquantScale is invalid.", keyAntiquantScaleTensorDimNum);
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckKScaleShapeForPerTokenGroupMode(const FiaTilingInfo &fiaInfo)
{
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    uint32_t batchSize = fiaInfo.bSize;
    uint32_t numKeyValueHeads = fiaInfo.n2Size;
    uint64_t seqLength = fiaInfo.s2Size;
    uint32_t headDim = fiaInfo.qkHeadDim;
    // per-token-group模式
    // shape支持[1, B, N, >=KV_S, D/32]
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_5) {
        OP_CHECK_IF(keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0) != 1 ||
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1) != batchSize ||
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_2) != numKeyValueHeads ||
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_3) < fiaInfo.s2Size ||
                    keyAntiquantScaleTensorShape.GetDim(DIM_NUM_4) != headDim / 32,
                    OP_LOGE(fiaInfo.opName,
                            "The shape of keyAntiquantScale([%u, %u, %u, %u, %u]) is not supported."
                            "The shape of keyAntiquantScale must be "
                            "[1, B(%llu), N(%llu), >=S(%llu)] or [B(%llu), >=S(%llu)] when keyAntiquantMode is per-token mode.",
                            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0),
                            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1), 
                            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_2),
                            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_3),
                            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_4),
                            batchSize, numKeyValueHeads, seqLength, batchSize, seqLength),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(fiaInfo.opName, "The dimNum(%u) of keyAntiquantScale is invalid.", keyAntiquantScaleTensorDimNum);
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckVScaleShapeForPerTokenMode(const FiaTilingInfo &fiaInfo)
{
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;
    gert::Shape valueAntiquantScaleTensorShape = valueAntiquantScaleTensor->GetStorageShape();
    uint32_t valueAntiquantScaleTensorDimNum = valueAntiquantScaleTensorShape.GetDimNum();
    uint32_t batchSize = fiaInfo.bSize;
    uint64_t seqLength = fiaInfo.s2Size;
    // 校验value支持per-token
    // shape支持[1, B, S]或[B, S]
    if (valueAntiquantScaleTensorDimNum == DIM_NUM_2) {
        // [B, >=KV_S]
        OP_CHECK_IF(valueAntiquantScaleTensorShape.GetDim(DIM_NUM_0) != batchSize ||
                    valueAntiquantScaleTensorShape.GetDim(DIM_NUM_1) < seqLength,
                    OP_LOGE(fiaInfo.opName,
                            "The shape of valueAntiquantScale([%u, %u]) is not supported."
                            "The shape of valueAntiquantScale must be "
                            "[1, B(%llu), >=S(%llu)] or [B(%llu), >=S(%llu)] when valueAntiquantMode is per-token mode.",
                            valueAntiquantScaleTensorShape.GetDim(DIM_NUM_0),
                            valueAntiquantScaleTensorShape.GetDim(DIM_NUM_1),
                            batchSize, seqLength, batchSize, seqLength),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    if (valueAntiquantScaleTensorDimNum == DIM_NUM_3) {
        // [1, B, >=KV_S]
        OP_CHECK_IF(valueAntiquantScaleTensorShape.GetDim(DIM_NUM_0) != 1 ||
                    valueAntiquantScaleTensorShape.GetDim(DIM_NUM_1) != batchSize ||
                    valueAntiquantScaleTensorShape.GetDim(DIM_NUM_2) < seqLength,
                    OP_LOGE(fiaInfo.opName,
                            "The shape of valueAntiquantScale([%u, %u, %u]) is not supported."
                            "The shape of valueAntiquantScale must be "
                            "[1, B(%llu), >=S(%llu)] or [B(%llu), >=S(%llu)] when valueAntiquantMode is per-token mode.",
                            valueAntiquantScaleTensorShape.GetDim(DIM_NUM_0),
                            valueAntiquantScaleTensorShape.GetDim(DIM_NUM_1),
                            valueAntiquantScaleTensorShape.GetDim(DIM_NUM_2),
                            batchSize, seqLength, batchSize, seqLength),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(fiaInfo.opName, "The dimNum(%u) of valueAntiquantScale is invalid.", valueAntiquantScaleTensorDimNum);
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (enableFullQuant_) {
        // 量化方式
        if (fiaInfo.ropeMode == RopeMode::ROPE_SPLIT) {
            enableIFAMLAFullQuant_ = true;
        } else if (fiaInfo.inputQType == ge::DT_INT8) {
            enablePertensorQuant_ = true;
        } else {
            enablePerblockQuant_ = true;
        }

        if (ge::GRAPH_SUCCESS != CheckDataTypeFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckDequantScaleDtypeFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckDequantModeFullquant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    } else if (enableAntiQuant_) {
        if (CheckSingleParaForAntiquant(fiaInfo) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (enableNonQuant_) {
        if (fiaInfo.socVersion != platform_ascendc::SocVersion::ASCEND910B) {
            return CheckExistenceNoquant(fiaInfo);
        }
    } else if (enableFullQuant_) {
        if (ge::GRAPH_SUCCESS != CheckExistencePertensorFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckExistenceMLAFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckExistencePerblockFullquant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    } else if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckExistenceForAntiquant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckFeature(const FiaTilingInfo &fiaInfo)
{
    if (enableFullQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeaturePertensorFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckFeaturePerblockFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckFeatureMLAFullquant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }

        if (ge::GRAPH_SUCCESS != CheckInputDTypeFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckInputLayoutFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckInputAxisFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckDequantScaleShapeFullquant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    } else if (enableAntiQuant_) {
        if (CheckFeatureForAntiquant(fiaInfo) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckMultiPara(const FiaTilingInfo &fiaInfo)
{
    if (enableAntiQuant_) {
        if (CheckMultiParaForAntiquant(fiaInfo) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling