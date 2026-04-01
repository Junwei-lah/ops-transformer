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
 * \file allto_all_matmul_infershape.cpp
 * \brief 图模式（动态图/静态图）走infershape
 */

#include <register/op_impl_registry.h>
#include "util/math_util.h"
#include "mc2_log.h"
#include "common/utils/op_mc2.h"
#include "op_host/mc2_common_infershape.h"

namespace ops {

using Ops::Base::CeilDiv;

namespace {

// input tensor index
enum All_TO_ALL_MATMUL_INPUT_IDX : size_t {
    INDEX_IN_X1 = 0,
    INDEX_IN_X2,
    INDEX_IN_BIAS,
    INDEX_IN_X1_SCALE,
    INDEX_IN_X2_SCALE,
};
// attr index
enum All_TO_ALL_MATMUL_ATTR_IDX : size_t {
    INDEX_ATTR_WORLD_SIZE = 1,
    INDEX_ATTR_ALLTO_ALL_AXES,
    INDEX_ATTR_Y_DTYPE,
    INDEX_ATTR_X1_QUANT_MODE,
    INDEX_ATTR_X2_QUANT_MODE,
    INDEX_ATTR_TRANS_X1 = 9,
    INDEX_ATTR_TRANS_X2,
    INDEX_ATTR_ALLTOALL_OUT_FLAG,
};
// output tensor index
enum All_TO_ALL_MATMUL_OUTPUT_IDX : size_t {
    INDEX_OUT = 0,
    INDEX_ALLTO_ALL_OUT,
};

// 维度信息
constexpr uint64_t DIM_ONE = 1;
constexpr uint64_t DIM_TWO = 2;
constexpr uint64_t DIM_THREE = 3;
// kc量化模式
constexpr uint64_t X1_DYN_PERTOKEN_QUANT_NUM = 7;
constexpr uint64_t X2_PERCHANNEL_QUANT_NUM = 2;
// mx量化模式
constexpr uint64_t X1_MXFP8_QUANT_NUM = 6;
constexpr uint64_t X2_MXFP8_QUANT_NUM = 6;
// 合法性校验
constexpr int64_t NUM_MINUS_ONE = -1;
constexpr int64_t NUM_MINUS_TWO = -2;
constexpr int64_t OUTPUT_INFER_SHAPE = 2;
constexpr int64_t SCALE_LAST_DIM = 2;
constexpr int64_t AXIS_K_UPPER_LIMIT = 65535;
const std::vector<int64_t> SUPPORT_RANK_NUM{2, 4, 8, 16};

static const char* INNER_DEBUG = "MC2: AlltoAllMatmul InferShape Debug";

struct AlltoAllMatmulShapeInfo {
    int64_t outputDim;
    int64_t rankNum;
    int64_t m;
    int64_t n;
    int64_t k1;
    int64_t k2;
};

} // namespace

/**
 * @brief x1和x2合法性校验
 *
 * @param context
 */
static ge::graphStatus CheckXShapeForAlltoAllMatmul(const gert::InferShapeContext* context)
{
    const auto x1Shape = context->GetInputShape(INDEX_IN_X1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OPS_CHECK(x1Shape->GetDimNum() != DIM_TWO, CUBE_INNER_ERR_REPORT(INNER_DEBUG,
              "In AlltoAllMatmul, x1 shape should be %ld, but the actual value is %ld.",
              DIM_TWO, x1Shape->GetDimNum()), return ge::GRAPH_FAILED);
    const auto x2Shape = context->GetInputShape(INDEX_IN_X2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    OPS_CHECK(x2Shape->GetDimNum() != DIM_TWO, CUBE_INNER_ERR_REPORT(INNER_DEBUG,
              "In AlltoAllMatmul, x2 shape should be %ld, but the actual value is %ld.",
              DIM_TWO, x2Shape->GetDimNum()), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief allToAllAxes合法性校验
 *
 * @param context
 */
static ge::graphStatus CheckAllToAllAxesShapeForAlltoAllMatmul(const gert::InferShapeContext* context)
{
    const auto attrs = context->GetAttrs();
    const auto alltoAllAxesPtr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_ATTR_ALLTO_ALL_AXES);
    if (alltoAllAxesPtr != nullptr) {
        OPS_CHECK((alltoAllAxesPtr->GetSize() != DIM_TWO), CUBE_INNER_ERR_REPORT(INNER_DEBUG,
                  "In AlltoAllMatmul, the size of alltoAllAxes should be %ld, but the actual value is %ld.",
                  DIM_TWO, alltoAllAxesPtr->GetSize()), return ge::GRAPH_FAILED);
        const auto alltoAllAxes = static_cast<const int64_t*>(alltoAllAxesPtr->GetData());
        OPS_CHECK((alltoAllAxes[0] != NUM_MINUS_TWO || alltoAllAxes[1] != NUM_MINUS_ONE),
                  CUBE_INNER_ERR_REPORT(INNER_DEBUG,
                  "In AlltoAllMatmul, the alltoAllAxes should be [-2, -1], but the actual value is [%ld, %ld].",
                  alltoAllAxes[0], alltoAllAxes[1]), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 静态shape图bias合法性校验
 *
 * @param context
 * @param shape
 */
static ge::graphStatus CheckBiasShapeForAlltoAllMatmul(const gert::InferShapeContext* context,
                                                       AlltoAllMatmulShapeInfo& shape)
{
    const auto biasShape = context->GetInputShape(INDEX_IN_BIAS);
    if (biasShape != nullptr) {
        // bias一定为1维，且维度值和n轴一致
        OPS_CHECK(biasShape->GetDimNum() != DIM_ONE, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In AlltoAllMatmul, bias shape must be %ld, but actual value is: %ld",
                  DIM_ONE, biasShape->GetDimNum()), return ge::GRAPH_FAILED);
        OPS_CHECK(biasShape->GetDim(0) != shape.n, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In AlltoAllMatmul, bias dim0 must be the same with matmul axis n, "
                  "but actual bias dim0 is: %ld, axis n is: %ld",
                  biasShape->GetDim(0), shape.n), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 静态shape图x1Scale和x2Scale合法性校验
 *
 * @param context
 * @param shape
 */
static ge::graphStatus CheckXScaleShapeForAlltoAllMatmul(const gert::InferShapeContext* context,
                                                         AlltoAllMatmulShapeInfo& shape)
{
    const auto attrs = context->GetAttrs();
    const int64_t* x1QuantMode = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_X1_QUANT_MODE);
    const int64_t* x2QuantMode = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_X2_QUANT_MODE);
    const auto x1ShapeScale = context->GetInputShape(INDEX_IN_X1_SCALE);
    if (x1ShapeScale != nullptr) {
        int64_t x1ShapeScaleDimNum = x1ShapeScale->GetDimNum();
        if (*x1QuantMode == X1_MXFP8_QUANT_NUM && *x2QuantMode == X2_MXFP8_QUANT_NUM) {
            // 只有mxfp8量化模式下，x1Scale才是3维
            OPS_CHECK(x1ShapeScaleDimNum != DIM_THREE, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                      "In AlltoAllMatmul, x1Scale dim num must be %ld, but actual value is: %ld",
                      DIM_THREE, x1ShapeScaleDimNum), return ge::GRAPH_FAILED);
            // x1Scale最后一维一定是2
            OPS_CHECK(x1ShapeScale->GetDim(x1ShapeScaleDimNum - 1) != SCALE_LAST_DIM,
                      CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                      "In AlltoAllMatmul, x1Scale last dim must be %ld, but actual value is: %ld",
                      SCALE_LAST_DIM, x1ShapeScale->GetDim(x1ShapeScaleDimNum - 1)), return ge::GRAPH_FAILED);
        } else {
            OPS_CHECK(x1ShapeScaleDimNum != DIM_ONE, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                      "In AlltoAllMatmul, x1Scale shape must be %ld, but actual value is: %ld",
                      DIM_ONE, x1ShapeScaleDimNum), return ge::GRAPH_FAILED);
        }
        // x1Scale第0维与m轴一致
        OPS_CHECK(x1ShapeScale->GetDim(0) != shape.m, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In AlltoAllMatmul, x1Scale dim0 must be the same with matmul axis m, "
                  "but actual x1Scale dim0 is: %ld, axis m is: %ld",
                  x1ShapeScale->GetDim(0), shape.m), return ge::GRAPH_FAILED);
    }
    const auto x2ShapeScale = context->GetInputShape(INDEX_IN_X2_SCALE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2ShapeScale);
    int64_t x2ShapeScaleDimNum = x2ShapeScale->GetDimNum();
    if (*x1QuantMode == X1_MXFP8_QUANT_NUM && *x2QuantMode == X2_MXFP8_QUANT_NUM) {
        // 只有mxfp8量化模式下，x2Scale才是3维
        OPS_CHECK(x2ShapeScaleDimNum != DIM_THREE, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In AlltoAllMatmul, x2Scale dim num must be %ld, but actual value is: %ld",
                  DIM_THREE, x2ShapeScaleDimNum), return ge::GRAPH_FAILED);
        // x2Scale最后一维一定是2
        OPS_CHECK(x2ShapeScale->GetDim(x2ShapeScaleDimNum - 1) != SCALE_LAST_DIM,
                  CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In AlltoAllMatmul, x2Scale last dim must be %ld, but actual value is: %ld",
                  SCALE_LAST_DIM, x2ShapeScale->GetDim(x2ShapeScaleDimNum - 1)), return ge::GRAPH_FAILED);
    } else {
        OPS_CHECK(x2ShapeScaleDimNum != DIM_ONE, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In AlltoAllMatmul, x2Scale shape must be %ld, but actual value is: %ld",
                  DIM_ONE, x2ShapeScaleDimNum), return ge::GRAPH_FAILED);
    }
    // x2Scale第0维与n轴一致
    OPS_CHECK(x2ShapeScale->GetDim(0) != shape.n, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
              "In AlltoAllMatmul, x2Scale dim0 must be the same with matmul axis n, "
              "but actual x2Scale dim0 is: %ld, axis n is: %ld",
              x2ShapeScale->GetDim(0), shape.n), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 静态shape图k轴合法性校验
 *
 * @param context
 * @param shape
 */
static ge::graphStatus CheckAxisKShapeForAlltoAllMatmul(const gert::InferShapeContext* context,
                                                        AlltoAllMatmulShapeInfo& shape)
{
    OPS_CHECK(shape.k1 > AXIS_K_UPPER_LIMIT || shape.k2 > AXIS_K_UPPER_LIMIT,
              CUBE_INNER_ERR_REPORT(context->GetNodeName(),
              "axis k cannot exceed upper limit %ld, but actual k1 is: %ld, k2 is: %ld",
              AXIS_K_UPPER_LIMIT, shape.k1, shape.k2), return ge::GRAPH_FAILED);
    if (shape.k1 != shape.k2 / shape.rankNum) {
        OP_LOGE(context->GetNodeName(),
                "In allto_all_matmul, x1.k must be the same to x2.k / rankSize, "
                "but actual get x1.k: %ld, x2.k: %ld, rankSize: %ld", shape.k1, shape.k2, shape.rankNum);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取输入的m，n，k轴大小
 *
 * @param context
 * @param shape
 */
static ge::graphStatus GetMatmulAxisInfoForAlltoAllMatmul(const gert::InferShapeContext* context,
                                                          AlltoAllMatmulShapeInfo& shape)
{
    const auto attrs = context->GetAttrs();
    const bool* isTransX1 = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_X1);
    OPS_CHECK(isTransX1 == nullptr || *isTransX1, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
              "x1 does not support transpose in allto all matmul."), return ge::GRAPH_FAILED);
    const bool* isTransX2 = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_X2);
    const bool transX2 = ((isTransX2 != nullptr) && (*isTransX2));

    const auto x1Shape = context->GetInputShape(INDEX_IN_X1);
    const auto x2Shape = context->GetInputShape(INDEX_IN_X2);
    shape.m = x1Shape->GetDim(0U);
    shape.k1 = x1Shape->GetDim(1U);
    shape.n = transX2 ? x2Shape->GetDim(0U) : x2Shape->GetDim(1U);
    shape.k2 = transX2 ? x2Shape->GetDim(1U) : x2Shape->GetDim(0U);
    shape.outputDim = x1Shape->GetDimNum();

    OP_LOGD(INNER_DEBUG, "Matmul m is: %ld, n is: %ld, k1 is: %ld, k2 is: %ld. transX2 is: %d",
            shape.m, shape.n, shape.k1, shape.k2, transX2);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取卡数rankSize
 *
 * @param context
 * @param shape
 */
static ge::graphStatus CheckRankDimForAlltoAllMatmul(gert::InferShapeContext* context,
                                                     AlltoAllMatmulShapeInfo& shape)
{
    const auto attrs = context->GetAttrs();
    const int64_t* rankDim = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_WORLD_SIZE);
    OPS_CHECK(rankDim == nullptr,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Rank number is null in allto all matmul."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(std::find(SUPPORT_RANK_NUM.begin(), SUPPORT_RANK_NUM.end(), *rankDim) >= SUPPORT_RANK_NUM.end(),
                    OP_LOGE(INNER_DEBUG,
                            "Rank number should be in %s, but the actual value is %ld.",
                            VectorToString(SUPPORT_RANK_NUM).c_str(), *rankDim),
                    return ge::GRAPH_FAILED);
    shape.rankNum = *rankDim;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导输出shape
 *
 * @param context
 */
static ge::graphStatus InferShapeAlltoAllMatmul(gert::InferShapeContext* context)
{
    OPS_CHECK(context == nullptr, OP_LOGE(INNER_DEBUG, "Context is null."), return ge::GRAPH_FAILED);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    // 动静态shape图都要进行校验
    CheckXShapeForAlltoAllMatmul(context);
    CheckAllToAllAxesShapeForAlltoAllMatmul(context);
    // 初始化shape结构体
    AlltoAllMatmulShapeInfo shape;
    OPS_CHECK(CheckRankDimForAlltoAllMatmul(context, shape) != ge::GRAPH_SUCCESS,
              CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to check rank dim for allto all matmul."),
              return ge::GRAPH_FAILED);
    OPS_CHECK(GetMatmulAxisInfoForAlltoAllMatmul(context, shape) != ge::GRAPH_SUCCESS,
              CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to check shape for allto all matmul"),
              return ge::GRAPH_FAILED);
    if (shape.m != NUM_MINUS_ONE) {
        // 静态shape图校验
        CheckAxisKShapeForAlltoAllMatmul(context, shape);
        CheckBiasShapeForAlltoAllMatmul(context, shape);
        CheckXScaleShapeForAlltoAllMatmul(context, shape);
    }
    // 推导output shape
    const bool* allToAllOutFlag = attrs->GetAttrPointer<bool>(INDEX_ATTR_ALLTOALL_OUT_FLAG);
    OPS_CHECK_NULL_WITH_CONTEXT(context, allToAllOutFlag);
    auto allToAllOut = context->GetOutputShape(INDEX_ALLTO_ALL_OUT);
    OPS_CHECK_NULL_WITH_CONTEXT(context, allToAllOut);
    allToAllOut->SetDimNum(OUTPUT_INFER_SHAPE);
    if (allToAllOutFlag) {
        if (shape.m != NUM_MINUS_ONE) {
            int64_t allToAllOutFirstDim = CeilDiv(shape.m, shape.rankNum);
            int64_t allToAllOutSecondDim = shape.k1 * shape.rankNum;
            allToAllOut->SetDim(0U, allToAllOutFirstDim);
            allToAllOut->SetDim(1U, allToAllOutSecondDim);
            OP_LOGI(
            INNER_DEBUG, "Allto all matmul alltoallout shape after infer shape, outputDim: %ld, m: %ld n: %ld.",
            OUTPUT_INFER_SHAPE, allToAllOutFirstDim, allToAllOutSecondDim);
        }
    }
    auto shapeOut = context->GetOutputShape(INDEX_OUT);
    OPS_CHECK_NULL_WITH_CONTEXT(context, shapeOut);
    shapeOut->SetDimNum(OUTPUT_INFER_SHAPE);
    if (shape.m == NUM_MINUS_ONE) {
        shapeOut->SetDim(0U, shape.m);
        shapeOut->SetDim(1U, shape.n);
    } else {
        int64_t outFirstDim = CeilDiv(shape.m, shape.rankNum);
        int64_t outSecondDim = shape.n;
        shapeOut->SetDim(0U, outFirstDim);
        shapeOut->SetDim(1U, outSecondDim);
        OP_LOGI(INNER_DEBUG, "allto all matmul output shape after infer shape, m: %ld n: %ld.",
                outFirstDim, outSecondDim);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导输出datatype
 *
 * @param context
 */
static ge::graphStatus InferDataTypeAlltoAllMatmul(gert::InferDataTypeContext* context)
{
    OPS_CHECK(context == nullptr, OP_LOGE(INNER_DEBUG, "Context is null."), return ge::GRAPH_FAILED);
    OP_LOGD(INNER_DEBUG, "Start to infer datatype of allto all matmul.");

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* x1QuantMode = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_X1_QUANT_MODE);
    const int64_t* x2QuantMode = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_X2_QUANT_MODE);
    OPS_CHECK(!(*x1QuantMode == 0 && *x2QuantMode == 0)
               && !(*x1QuantMode == X1_DYN_PERTOKEN_QUANT_NUM && *x2QuantMode == X2_PERCHANNEL_QUANT_NUM)
               && !(*x1QuantMode == X1_MXFP8_QUANT_NUM && *x2QuantMode == X2_MXFP8_QUANT_NUM),
               OP_LOGE(INNER_DEBUG,
                       "The x1 or x2 quant mode is invalid, x1QuantMode is: %ld, x2QuantMode is: %ld",
                       x1QuantMode, x1QuantMode),
               return ge::GRAPH_FAILED);

    // 初始默认值
    auto yType = ge::DataType::DT_UNDEFINED;
    ge::DataType x1Type = context->GetInputDataType(INDEX_IN_X1);
    const int64_t* yDtypePtr = attrs->GetInt(INDEX_ATTR_Y_DTYPE);
    if ((yDtypePtr != nullptr && *yDtypePtr != static_cast<uint64_t>(ge::DataType::DT_UNDEFINED))) {
        OP_LOGI(INNER_DEBUG, "The yDtype value is: %ld", *yDtypePtr);
        yType = static_cast<ge::DataType>(*yDtypePtr);
    } else {
        OP_LOGE(INNER_DEBUG, "The yDtypePtr is null or get invalid yDtype value: DT_UNDEFINED.");
        return ge::GRAPH_FAILED;
    }
    // 设置推导的datatype
    context->SetOutputDataType(0, yType);
    context->SetOutputDataType(1, x1Type);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AlltoAllMatmul)
    .InferShape(InferShapeAlltoAllMatmul)
    .InferDataType(InferDataTypeAlltoAllMatmul);
} // namespace ops