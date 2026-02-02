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
 * \file matmul_allto_all_infershape.cpp
 * \brief 图模式（动态图/静态图）走infershape
 */

#include <register/op_impl_registry.h>
#include "util/math_util.h"
#include "mc2_log.h"
#include "mc2_common_infershape.h"

namespace ops {

using Ops::Base::CeilDiv;

constexpr size_t INDEX_IN_X1 = 0;
constexpr size_t INDEX_IN_X2 = 1;
constexpr size_t INDEX_IN_X1_SCALE = 3;
constexpr size_t INDEX_IN_X2_SCALE = 4;
constexpr size_t INDEX_ATTR_GROUP = 0;
constexpr size_t INDEX_ATTR_WORLD_SIZE = 1;
constexpr size_t INDEX_ATTR_Y_DTYPE = 3;
constexpr size_t INDEX_ATTR_X1_QUANT_MODE = 4;
constexpr size_t INDEX_ATTR_X2_QUANT_MODE = 5;
constexpr size_t INDEX_ATTR_TRANS_X1 = 8;
constexpr size_t INDEX_ATTR_TRANS_X2 = 9;
constexpr size_t INDEX_OUT = 0;
constexpr uint64_t NUM_ONE = 1;
constexpr uint64_t DIM_TWO = 2;
constexpr uint64_t NUM_MINUS_ONE = -1;
constexpr uint64_t X1_QUANT_MODE_NUM = 3;
constexpr uint64_t X2_QUANT_MODE_NUM = 2;
static const char* INNER_DEBUG = "MC2: MatmulAlltoAll InferShape Debug";
const std::set<int> SUPPORT_RANK_NUM{2, 4, 8, 16};

struct MatmulAlltoAllShapeInfo {
    uint64_t output_dim;
    uint64_t rankNum;
    uint64_t m;
    uint64_t n;
    uint64_t k;
};

/**
 * @brief 校验MatmulAlltoAll输入shape，并记录输入m，n，k大小
 *
 * @param context
 * @param shape
 */
static ge::graphStatus CheckShapeForMatmulAlltoAll(const gert::InferShapeContext* context, MatmulAlltoAllShapeInfo& shape)
{
    const auto x1_shape = context->GetInputShape(INDEX_IN_X1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    const auto x2_shape = context->GetInputShape(INDEX_IN_X2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2_shape);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* groupStr = attrs->GetAttrPointer<char>(INDEX_ATTR_GROUP);
    OP_LOGE_IF(groupStr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get matmul allto all group failed.");
    const bool* isTransX1 = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_X1);
    OPS_CHECK(
        isTransX1 == nullptr || *isTransX1, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
        "x1 does not support transpose in matmul allto all."), return ge::GRAPH_FAILED);
    const bool* isTransX2 = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_X2);
    const bool trans_x2 = ((isTransX2 != nullptr) && (*isTransX2));
    shape.m = x1_shape->GetDim(0U);
    shape.k = x1_shape->GetDim(1U);
    shape.n = trans_x2 ? x2_shape->GetDim(0U) : x2_shape->GetDim(1U);
    shape.output_dim = x1_shape->GetDimNum();
    OP_LOGD(INNER_DEBUG, "Matmul m %ld n %ld k %ld.", shape.m, shape.n, shape.k);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取，校验并记录卡数
 *
 * @param context
 * @param shape
 */
static ge::graphStatus CheckRankDim(gert::InferShapeContext* context, MatmulAlltoAllShapeInfo& shape)
{
    const auto attrs = context->GetAttrs();
    const int* rankDim = attrs->GetAttrPointer<int>(INDEX_ATTR_WORLD_SIZE);
    OPS_CHECK(rankDim == nullptr,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Invalid rank number %zu in matmul allto all.", *rankDim),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SUPPORT_RANK_NUM.find(*rankDim) == SUPPORT_RANK_NUM.end(),
                    OP_LOGE(INNER_DEBUG, "Rank number should be 2 or 4 or 8 or 16, but the actual value is %ld.", *rankDim),
                    return ge::GRAPH_FAILED);
    shape.rankNum = *rankDim;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导输出shape
 *
 * @param context
 */
static ge::graphStatus InferShapeMatmulAlltoAll(gert::InferShapeContext* context)
{
    OPS_CHECK(context == nullptr, OP_LOGE(INNER_DEBUG, "Context is null."), return ge::GRAPH_FAILED);
    MatmulAlltoAllShapeInfo shape;
    OPS_CHECK(
        CheckShapeForMatmulAlltoAll(context, shape) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to check shape for matmul allto all"),
        return ge::GRAPH_FAILED);
    OPS_CHECK(
        CheckRankDim(context, shape) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to check rank dim for matmul allto all."),
        return ge::GRAPH_FAILED);
    auto shape_out = context->GetOutputShape(INDEX_OUT);
    OPS_CHECK_NULL_WITH_CONTEXT(context, shape_out);
    shape_out->SetDimNum(shape.output_dim);
    if (shape.m == NUM_MINUS_ONE) {
        shape_out->SetDim(0U, shape.m);
        shape_out->SetDim(1U, shape.n);
    } else {
        uint64_t out_first_dim = shape.m * shape.rankNum;
        uint64_t out_second_dim = CeilDiv(shape.n, shape.rankNum);
        shape_out->SetDim(0U, out_first_dim);
        shape_out->SetDim(1U, out_second_dim);
        OP_LOGI(INNER_DEBUG, "Matmul allto all output shape after infer shape, dim: %zu m: %ld n: %ld.", shape.output_dim, out_first_dim, out_second_dim);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导输出数据类型
 *
 * @param context
 */
static ge::graphStatus InferDataTypeMatmulAlltoAll(gert::InferDataTypeContext* context)
{
    OPS_CHECK(context == nullptr, OP_LOGE(INNER_DEBUG, "Context is null."), return ge::GRAPH_FAILED);
    OP_LOGD(INNER_DEBUG, "Start to infer datatype of matmul allto all.");
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int* x1_quant_mode = attrs->GetAttrPointer<int>(INDEX_ATTR_X1_QUANT_MODE);
    const int* x2_quant_mode = attrs->GetAttrPointer<int>(INDEX_ATTR_X2_QUANT_MODE);
    const int64_t* y_dtype_ptr = attrs->GetInt(INDEX_ATTR_Y_DTYPE);
    auto y_type = ge::DataType::DT_UNDEFINED;
    ge::DataType x1_type = context->GetInputDataType(INDEX_IN_X1);
    if (*x1_quant_mode == 0 && *x2_quant_mode == 0) {
        if ((y_dtype_ptr != nullptr && *y_dtype_ptr != static_cast<uint64_t>(ge::DataType::DT_UNDEFINED))) {
            y_type = static_cast<ge::DataType>(*y_dtype_ptr);
        } else {
            return ge::GRAPH_FAILED;
        }
    } else if (*x1_quant_mode == X1_QUANT_MODE_NUM && *x2_quant_mode == X2_QUANT_MODE_NUM) {
        if ((y_dtype_ptr != nullptr && *y_dtype_ptr != static_cast<uint64_t>(ge::DataType::DT_UNDEFINED))) {
            y_type = static_cast<ge::DataType>(*y_dtype_ptr);
        } else {
            return ge::GRAPH_FAILED;
        }
    }
    context->SetOutputDataType(0, y_type);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatmulAlltoAll)
    .InferShape(InferShapeMatmulAlltoAll)
    .InferDataType(InferDataTypeMatmulAlltoAll);
} // namespace ops