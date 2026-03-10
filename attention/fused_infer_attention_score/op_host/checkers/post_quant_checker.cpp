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
 * \file post_quant_checker.cpp
 * \brief
 */
 
#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "post_quant_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// 公共校验函数

// CheckSingle
ge::graphStatus PostQuantChecker::CheckSingleDtype(const FiaTilingInfo &fiaInfo)
{
    // QuantScale2 and quantOffset2 only support bf16/fp32 data type.
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(fiaInfo.opParamInfo.quantScale2.desc, QUANT_SCALE2_NAME)) {
        OP_LOGE(fiaInfo.opName, "QuantScale2 only support bf16/fp32 data type!");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(fiaInfo.opParamInfo.quantOffset2.desc, QUANT_OFFSET2_NAME)) {
        OP_LOGE(fiaInfo.opName, "QuantOffset2 only support bf16/fp32 data type!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// CheckParaExistence
ge::graphStatus PostQuantChecker::CheckExistenceQuantScale2(const FiaTilingInfo &fiaInfo)
{
    // Post-quantization scenarios must include quantScale2.
    if (fiaInfo.isOutQuantEnable) {
        int64_t quantScale2ShapeSize = fiaInfo.opParamInfo.quantScale2.tensor->GetShapeSize();
        OP_CHECK_IF(quantScale2ShapeSize <= 0,
                    OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "Quant_scale2 is empty tensor in post quant scenario!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// CheckFeature
ge::graphStatus PostQuantChecker::CheckFeatureOutput(const FiaTilingInfo &fiaInfo)
{
    // Post-quantization scenarios only support int8/fp8_e4m3fn_t/hifloat8_t output data types.
    if (fiaInfo.isOutQuantEnable) {
        ge::DataType outputType = fiaInfo.opParamInfo.attenOut.desc->GetDataType();
        OP_CHECK_IF(outputType != ge::DT_INT8 && outputType != ge::DT_FLOAT8_E4M3FN && outputType != ge::DT_HIFLOAT8,
                    OP_LOGE(fiaInfo.opName,
                            "The quantScale2 exists, output data type only supports int8, fp8_e4m3fn_t, hifloat8_t!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeatureOutputEqual(const FiaTilingInfo &fiaInfo)
{
    // In the Anti-quantization scenario,
    // only KV input and post-quantization outputs with the same data type are supported.
    if (fiaInfo.isOutQuantEnable) {
        ge::DataType outputType = fiaInfo.opParamInfo.attenOut.desc->GetDataType();
        OP_CHECK_IF(outputType != fiaInfo.inputKvType,
                    OP_LOGE(fiaInfo.opName,
                            "The quantScale2 exists, output data type only supports that matches the KV input type!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeaturePrefix(const FiaTilingInfo &fiaInfo)
{
    // When prefix exists, post-quantization scenarios only support int8 output data types.
    if (fiaInfo.isOutQuantEnable) {
        ge::DataType outputType = fiaInfo.opParamInfo.attenOut.desc->GetDataType();
        OP_CHECK_IF(fiaInfo.sysPrefixFlag && outputType != ge::DT_INT8,
                    OP_LOGE(fiaInfo.opName, "When prefix exists, the output data type only supports int8!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// CheckCheckMultiPara
ge::graphStatus PostQuantChecker::CheckMultiParaQuantOffset2(const FiaTilingInfo &fiaInfo)
{
    // Scale2 and offset2 should have same dtype and shape

    if (fiaInfo.isOutQuantEnable && fiaInfo.opParamInfo.quantOffset2.tensor != nullptr &&
        fiaInfo.opParamInfo.quantOffset2.desc != nullptr) {
        const ge::DataType quantScale2Type = fiaInfo.opParamInfo.quantScale2.tensor->GetDataType();
        int64_t quantScale2ShapeSize = fiaInfo.opParamInfo.quantScale2.tensor->GetShapeSize();
        size_t quantScale2Dim = fiaInfo.opParamInfo.quantScale2.tensor->GetStorageShape().GetDimNum();

        const ge::DataType quantOffset2Datatype = fiaInfo.opParamInfo.quantOffset2.tensor->GetDataType();
        int64_t quantOffset2ShapeSize = fiaInfo.opParamInfo.quantOffset2.tensor->GetShapeSize();
        size_t quantOffset2Dim = fiaInfo.opParamInfo.quantOffset2.tensor->GetStorageShape().GetDimNum();

        OP_CHECK_IF(quantOffset2Datatype != quantScale2Type,
                    OP_LOGE(fiaInfo.opName, "QuantScale2 and quantOffset2 should have same dtype!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(quantOffset2Dim != quantScale2Dim,
                    OP_LOGE(fiaInfo.opName, "QuantScale2 and quantOffset2 should have same dim!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(quantOffset2ShapeSize != quantScale2ShapeSize,
                    OP_LOGE(fiaInfo.opName, "QuantScale2 and quantOffset2 should have same shape size!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckMultiParaDtype(const FiaTilingInfo &fiaInfo)
{
    // Post-quant scale dtype must be FP32. BF16 is allowed only if the query is BF16.

    if (fiaInfo.isOutQuantEnable) {
        const ge::DataType quantScale2Type = fiaInfo.opParamInfo.quantScale2.tensor->GetDataType();
        OP_CHECK_IF(fiaInfo.inputQType == ge::DT_BF16 && quantScale2Type != ge::DT_FLOAT &&
                    quantScale2Type != ge::DT_BF16,
                    OP_LOGE(fiaInfo.opName,
                            "When query is bf16, the post quant scale dtype only supports float32 and bf16!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(fiaInfo.inputQType != ge::DT_BF16 && quantScale2Type != ge::DT_FLOAT,
                    OP_LOGE(fiaInfo.opName,
                            "When query is not bf16, the post quant scale dtype only supports float32!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckMultiParaShape(const FiaTilingInfo &fiaInfo)
{
    // For post quant per-tensor, quant scale/offset only support [1].
    // For post quant per-channel, quant scale/offset dim multiply result only support qN * vD.

    if (fiaInfo.isOutQuantEnable) {
        int64_t quantScale2ShapeSize = fiaInfo.opParamInfo.quantScale2.tensor->GetShapeSize();
        size_t quantScale2Dim = fiaInfo.opParamInfo.quantScale2.tensor->GetStorageShape().GetDimNum();
        uint32_t numHeads = *(fiaInfo.opParamInfo.numHeads);
        uint64_t quantScale2ShapeSizePerChannel =
            static_cast<uint64_t>(numHeads) * static_cast<uint64_t>(fiaInfo.qkHeadDim);
        // per-tensor or per-channel verification
        if (quantScale2Dim == 1) {
            if (static_cast<uint64_t>(quantScale2ShapeSize) != quantScale2ShapeSizePerChannel) {
                OP_CHECK_IF(
                    (static_cast<uint64_t>(quantScale2ShapeSize) != 1U),
                    OPS_REPORT_VECTOR_INNER_ERR(
                        fiaInfo.opName, "For post quant per-tensor, quant scale/offset only support [1], now is [%d]",
                        quantScale2ShapeSize),
                    return ge::GRAPH_FAILED);
            }
        } else {
            OP_CHECK_IF((static_cast<uint64_t>(quantScale2ShapeSize) != quantScale2ShapeSizePerChannel),
                        OPS_REPORT_VECTOR_INNER_ERR(
                            fiaInfo.opName,
                            "For post quant per-channel, quant scale/offset dim multiply result only "
                            "support qN * vD(%u * %u = %lu), now is (%ld).",
                            numHeads, fiaInfo.qkHeadDim, quantScale2ShapeSizePerChannel, quantScale2ShapeSize),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}
// enableNonQuant 相关校验函数

// enableFullQuant 相关校验函数

// enableAntiQuant 相关校验函数

ge::graphStatus PostQuantChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin PostQuantChecker::CheckSinglePara!");
    if (ge::GRAPH_SUCCESS != CheckSingleDtype(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (enableNonQuant_) {
        ;
    } else if (enableFullQuant_) {
        ;
    } else if (enableAntiQuant_) {
        ;
    }
    OP_LOGI(fiaInfo.opName, "End PostQuantChecker::CheckSinglePara!");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin PostQuantChecker::CheckParaExistence!");
    if (ge::GRAPH_SUCCESS != CheckExistenceQuantScale2(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (enableNonQuant_) {
        ;
    } else if (enableFullQuant_) {
        ;
    } else if (enableAntiQuant_) {
        ;
    }
    OP_LOGI(fiaInfo.opName, "End PostQuantChecker::CheckParaExistence!");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeature(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin PostQuantChecker::CheckFeature!");
    if (enableNonQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureOutput(fiaInfo) || ge::GRAPH_SUCCESS != CheckFeaturePrefix(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    } else if (enableFullQuant_) {
        ;
    } else if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureOutputEqual(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_LOGI(fiaInfo.opName, "End PostQuantChecker::CheckFeature!");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckMultiPara(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName, "Begin PostQuantChecker::CheckMultiPara!");
    if (ge::GRAPH_SUCCESS != CheckMultiParaQuantOffset2(fiaInfo) || ge::GRAPH_SUCCESS != CheckMultiParaDtype(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckMultiParaShape(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (enableNonQuant_) {
        ;
    } else if (enableFullQuant_) {
        ;
    } else if (enableAntiQuant_) {
        ;
    }
    OP_LOGI(fiaInfo.opName, "End PostQuantChecker::CheckMultiPara!");

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
