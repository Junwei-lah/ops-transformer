// /**
//  * Copyright (c) 2026 Huawei Technologies Co., Ltd.
//  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
//  * CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
//  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// /*!
//  * \file test_quant_gmm_alltoallv_tiling.cpp
//  * \brief tiling ut
//  */

#include <iostream>
#include <gtest/gtest.h>

#include "mc2_tiling_case_executor.h"
#include "../../../../op_host/op_tiling/arch35/quant_grouped_mat_mul_allto_allv_tiling.h"

using namespace std;

struct GroupedMatMulAlltoAllvTilingTestParam {
    string caseName;

    // input
    std::vector<int64_t> gmmXShape;
    ge::DataType gmmXDataType;
    ge::Format gmmXFormat;

    std::vector<int64_t> gmmWeightShape;
    ge::DataType gmmWeightDataType;
    ge::Format gmmWeightFormat;

    std::vector<int64_t> gmmXScaleShape;
    ge::DataType gmmXScaleDataType;
    ge::Format gmmXScaleFormat;

    std::vector<int64_t> gmmWeightScaleShape;
    ge::DataType gmmWeightScaleDataType;
    ge::Format gmmWeightScaleFormat;

    std::vector<int64_t> mmXShape;
    ge::DataType mmXDataType;
    ge::Format mmXFormat;

    std::vector<int64_t> mmWeightShape;
    ge::DataType mmWeightDataType;
    ge::Format mmWeightFormat;

    std::vector<int64_t> mmXScaleShape;
    ge::DataType mmXScaleDataType;
    ge::Format mmXScaleFormat;

    std::vector<int64_t> mmWeightScaleShape;
    ge::DataType mmWeightScaleDataType;
    ge::Format mmWeightScaleFormat;

    std::vector<int64_t> sendCounts;
    std::vector<int64_t> recvCounts;

    // output: expected output tensor
    std::vector<int64_t> gmmYShape;
    ge::DataType gmmYDataType;
    ge::Format gmmYFormat;

    std::vector<int64_t> mmYShape;
    ge::DataType mmYDataType;
    ge::Format mmYFormat;

    // Attributes
    bool gmm_x_quant_mode;
    bool gmm_weight_quant_mode;
    bool mm_x_quant_mode;
    bool mm_weight_quant_mode;

    bool trans_gmm_weight_flag;
    bool trans_mm_weight_flag;

    int64_t world_size;
    int64_t ep_world_size;
    int64_t graph_type;

    // Expected result
    ge::graphStatus expectedStatus;

    // Expected tiling key
    uint64_t expectTilingKey;
};

static const vector<GroupedMatMulAlltoAllvTilingTestParam> groupedMatMulAlltoAllvTilingTestParam = {
    // hif8 全量化
    // 正常测试用例
    {
        "gmmalltoallv_hif8_quant_normal",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmX
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmWeight
        {1}, ge::DT_FLOAT, ge::FORMAT_ND, // gmmXScale
        {1}, ge::DT_FLOAT, ge::FORMAT_ND, // gmmWeightScale
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // mmX
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // mmWeight
        {1}, ge::DT_FLOAT, ge::FORMAT_ND, // mmXScale
        {1}, ge::DT_FLOAT, ge::FORMAT_ND, // mmWeightScale
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024}, // sendCounts
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024}, // recvCounts
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // gmmYShape
        {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // mmYShape
        1, 1, 1, 1, // gmmXQuantMode gmmWeightQuantMode mmXQuantMode mmWeightQuantMode
        false, false, // gmmTrans mmTrans
        2, 2, 0, // worldSize epWorldSize graphType
        ge::GRAPH_SUCCESS, 137 // expectedStatus expectTilingKey
    },

    // 异常测试用例
    // 数据类型非法异常
    {
        "gmmalltoallv_hif8_quant_exception_gmmx_datatype_invalid",
        {8192, 7168}, ge::DT_INT8, ge::FORMAT_ND, // gmmX数据类型非法
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_gmmweight_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_INT8, ge::FORMAT_ND, // gmmWeight数据类型非法
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_gmmy_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_INT8, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // gmmY数据类型非法
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_gmmxscale_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_INT8, ge::FORMAT_ND, // gmmXScale数据类型非法
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_gmmweightscale_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_INT8, ge::FORMAT_ND, // gmmWeightScale数据类型非法
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_mmx_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_INT8, ge::FORMAT_ND, // mmX数据类型非法
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_mmweight_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_INT8, ge::FORMAT_ND, // mmWeight数据类型非法
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_mmxscale_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_INT8, ge::FORMAT_ND, // mmXScale数据类型非法
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_mmweightscale_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_INT8, ge::FORMAT_ND, // mmWeightScale数据类型非法
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },

    // 空tensor异常
    {
        "gmmalltoallv_hif8_quant_exception_gmmx_empty_dim0",
        {0, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmX为空tensor，第一维度为0
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_gmmx_empty_dim1",
        {8192, 0}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmX为空tensor，第二维度为0
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_gmmweight_empty_dim0",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {0, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmWeight为空tensor，第一维度为0
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_gmmweight_empty_dim1",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 0, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmWeight为空tensor，第二维度为0
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_gmmweight_empty_dim2",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 0}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmWeight为空tensor，第三维度为0
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_gmmy_empty_dim0",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {0,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // gmmY为空tensor，第一维为0
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_gmmy_empty_dim1",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,0}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // gmmY为空tensor，第二维为0
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },

    // K轴不匹配异常
    {
        "gmmalltoallv_hif8_quant_exception_gmmx_gmmweight_k_mismatch_no_transpose",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 8192, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmX和gmmWeight的k轴不匹配（无转置），gmmX的K=7168，gmmWeight的K=8192
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_gmmx_gmmweight_k_mismatch_transpose",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmX和gmmWeight的k轴不匹配（转置），gmmX的K=7168，gmmWeight转置后K=4096
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0 // trans_gmm_weight_flag=true
    },
    {
        "gmmalltoallv_hif8_quant_exception_mm_not_null_mmx_mmweight_k_mismatch",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {8192, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // mm不为空时mmX和mmWeight的K轴不匹配，mmX的K=7168，mmWeight的K=8192
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },

    // mm不为空时mmX的第一维与mmWeight的第二维不匹配
    {
        "gmmalltoallv_hif8_quant_exception_mm_not_null_mmx_dim0_mmweight_dim1_mismatch",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 8192}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // mm不为空时mmX的第一维(4096)与mmWeight的第二维(8192)不匹配
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    // mm不为空时mmY与mmX的第一维不匹配
    {
        "gmmalltoallv_hif8_quant_exception_mm_not_null_mmy_mmx_dim0_mismatch",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // mmY(8192)与mmX的第一维(4096)不匹配
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },

    // group相关异常
    {
        "gmmalltoallv_hif8_quant_exception_group_length_exceed_128",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024,1024,1024,1024,1024,1024,1024,1024,1024}, // 9个元素，超过8
        {1024,1024,1024,1024,1024,1024,1024,1024,1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0 // group长度超过128（这里sendCounts长度超过8）
    },

    // trans flag与shape不匹配
    {
        "gmmalltoallv_hif8_quant_exception_transgmmweight_value_shape_mismatch",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // shape为[4, 4096, 7168]
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0 // transGmmWeight=false但shape不匹配（应该是[4,7168,4096]）
    },
    {
        "gmmalltoallv_hif8_quant_exception_transmmweight_value_shape_mismatch",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // shape为[4096,7168]
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0 // transMmWeight=false但shape不匹配（应该是[7168,4096]）
    },

    // epWorldSize异常
    {
        "gmmalltoallv_hif8_quant_exception_epworldsize_not_equal_256",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 4, 0, ge::GRAPH_FAILED, 0 // e*epWorldSize不等于256 (2*4=8)
    },

    // TopK异常
    {
        "gmmalltoallv_hif8_quant_exception_topk_greater_than_8",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {9, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // TopK的值大于8 (第一维=9)
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },

    // sendCounts和recvCounts总和异常
    {
        "gmmalltoallv_hif8_quant_exception_sendcounts_sum_not_equal_gmmx_dim0",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024,1024,1024,1024,1024,1024,1024,1025}, // 总和=8193，不等于gmmX的第一维大小8192
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_recvcounts_sum_not_equal_gmmx_dim0",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024,1024,1024,1024,1024,1024,1024,1025}, // 总和=8193，不等于gmmX的第一维大小8192
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    }
};

class GroupedMatMulAlltoAllvTilingTest : public ::testing::TestWithParam<GroupedMatMulAlltoAllvTilingTestParam> {
protected:
    static void SetUpTestCase() {
        cout << "GroupedMatMulAlltoAllvTilingTest SetUp" << endl;
    }

    static void TearDownTestCase() {
        cout << "GroupedMatMulAlltoAllvTilingTest TearDown" << endl;
    }
    void SetUp() override {
        const auto& param = GetParam();
        cout << "Running test case: " << param.caseName << endl;
    }
};

INSTANTIATE_TEST_SUITE_P(
    GroupedMatMulAlltoAllvTilingTestSuite,
    GroupedMatMulAlltoAllvTilingTest,
    testing::ValuesIn(groupedMatMulAlltoAllvTilingTestParam),
    [](const testing::TestParamInfo<GroupedMatMulAlltoAllvTilingTestParam>& info) {
        return info.param.caseName;
    }
);

TEST_P(GroupedMatMulAlltoAllvTilingTest, test_grouped_quant_mat_mul_allto_allv_tiling) {
    const auto& param = GetParam();

    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    uint64_t coreNum = 36;
    uint64_t ubSize = 256 * 1024;
    size_t tilingDataSize = sizeof(QuantGmmA2avTilingData);

    gert::StorageShape mmXStorageShape;
    if (param.mmXShape.size() > 0 && param.mmXShape[0] > 0) {
        mmXStorageShape = {{param.mmXShape[0], param.mmXShape[1]}, {param.mmXShape[0], param.mmXShape[1]}};
    } else {
        mmXStorageShape = {};
    }
    
    gert::StorageShape mmWeightStorageShape;
    if (param.mmWeightShape.size() > 0 && param.mmWeightShape[0] > 0) {
        mmWeightStorageShape = {{param.mmWeightShape[0], param.mmWeightShape[1]}, {param.mmWeightShape[0], param.mmWeightShape[1]}};
    } else {
        mmWeightStorageShape = {};
    }
    
    gert::StorageShape mmYStorageShape;
    if (param.mmYShape.size() > 0 && param.mmYShape[0] > 0) {
        mmYStorageShape = {{param.mmYShape[0], param.mmYShape[1]}, {param.mmYShape[0], param.mmYShape[1]}};
    } else {
        mmYStorageShape = {};
    }

    // gmmXScaleStorageShape
    gert::StorageShape gmmXScaleStorageShape;
    if (param.gmmXScaleShape.size() > 0 && param.gmmXScaleShape[0] > 0) {
        gmmXScaleStorageShape = {{param.gmmXScaleShape[0]}, {param.gmmXScaleShape[0]}};
    } else {
        gmmXScaleStorageShape = {};
    }

    // gmmWeightScaleStorageShape
    gert::StorageShape gmmWeightScaleStorageShape;
    if (param.gmmWeightScaleShape.size() > 0 && param.gmmWeightScaleShape[0] > 0) {
        gmmWeightScaleStorageShape = {{param.gmmWeightScaleShape[0]}, {param.gmmWeightScaleShape[0]}};
    } else {
        gmmWeightScaleStorageShape = {};
    }

    // mmXScaleStorageShape
    gert::StorageShape mmXScaleStorageShape;
    if (param.mmXScaleShape.size() > 0 && param.mmXScaleShape[0] > 0) {
        mmXScaleStorageShape = {{param.mmXScaleShape[0]}, {param.mmXScaleShape[0]}};
    } else {
        mmXScaleStorageShape = {};
    }

    // mmWeightScaleStorageShape
    gert::StorageShape mmWeightScaleStorageShape;
    if (param.mmWeightScaleShape.size() > 0 && param.mmWeightScaleShape[0] > 0) {
        mmWeightScaleStorageShape = {{param.mmWeightScaleShape[0]}, {param.mmWeightScaleShape[0]}};
    } else {
        mmWeightScaleStorageShape = {};
    }

    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{param.gmmXShape[0], param.gmmXShape[1]},{param.gmmXShape[0], param.gmmXShape[1]}}, param.gmmXDataType, param.gmmXFormat},
            {{{param.gmmWeightShape[0], param.gmmWeightShape[1],param.gmmWeightShape[2]},{param.gmmWeightShape[0], param.gmmWeightShape[1],param.gmmWeightShape[2]}},
                 param.gmmWeightDataType, param.gmmWeightFormat},
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // send_counts_tensor
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // recv_counts_tensor
            {mmXStorageShape, param.mmXDataType, param.mmXFormat},
            {mmWeightStorageShape,param.mmWeightDataType, param.mmWeightFormat},
            {gmmXScaleStorageShape, param.gmmXScaleDataType, param.gmmXScaleFormat},
            {gmmWeightScaleStorageShape, param.gmmWeightScaleDataType, param.gmmWeightScaleFormat},
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // gmmXOffset
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // gmmWeightOffset
            {mmXScaleStorageShape, param.mmXScaleDataType, param.mmXScaleFormat},
            {mmWeightScaleStorageShape, param.mmWeightScaleDataType, param.mmWeightScaleFormat},
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // mmXOffset
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // mmWeightOffset
            {{}, ge::DT_INT64, ge::FORMAT_ND}  // commQunatScale
        },
        {
            {{{param.gmmYShape[0], param.gmmYShape[1]},{param.gmmYShape[0], param.gmmYShape[1]}}, param.gmmYDataType, param.gmmYFormat},
            {mmYStorageShape, param.mmYDataType, param.mmYFormat}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.ep_world_size)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(param.sendCounts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(param.recvCounts)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.trans_gmm_weight_flag)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.trans_mm_weight_flag)},
            {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmm_x_quant_mode)},
            {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmm_weight_quant_mode)},
            {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mm_x_quant_mode)},
            {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mm_weight_quant_mode)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"gmm_y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"mm_y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        },
        &compileInfo,
        "Ascend950",
        coreNum,
        ubSize,
        tilingDataSize
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectedStatus, param.expectTilingKey);
}
