/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_quant_matmul_allto_all.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class TestAclnnQuantMatmulAlltoAll : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        cout << "TestAclnnQuantMatmulAlltoAll SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "test_aclnn_matmul_allto_all TearDown" << endl;
    }
};

struct QuantMatmulAlltoAllAclnnTestParam {
    // 用例名
    string caseName;
    // 通信域卡数，ut测试默认为2
    int worldSize;
    // 数据形状
    int64_t x1Quantmode; // x1量化模式
    int64_t x2Quantmode; // x2量化模式
    vector<int64_t> x1Shape; // x1数据shape，正常为（BS，H1）
    vector<int64_t> x2Shape; // x2数据shape，正常为（H1，H2）
    vector<int64_t> biasShape; // bias数据shape，正常为（H2）
    vector<int64_t> x1ScaleShape; // x1scales数据shape，正常为（BS）
    vector<int64_t> x2ScaleShape; // x2scales数据shape，正常为（H2）
    vector<int64_t> outputShape; // output数据shape，正常为（BS * world_size，H2 / world_size）
    // 数据类型
    aclDataType x1Dtype; // x1数据dtype，仅支持float8_e5m2和float8_e4m3fn
    aclDataType x2Dtype; // x2数据dtype，仅支持float8_e5m2和float8_e4m3fn
    aclDataType biasDtype; // bias数据dtype，仅支持float32
    aclDataType x1ScaleDtype; // x1scales数据dtype，仅支持float32
    aclDataType x2ScaleDtype; // x2scales数据dtype，仅支持float32
    aclDataType outputDtype; // 输出数据dtype，支持bfloat16、float16和float32
    // 数据格式
    aclFormat x1Format; // x1数据format，仅支持ND
    aclFormat x2Format; // x2数据format，仅支持ND
    aclFormat biasFormat; // bias数据format，仅支持ND
    aclFormat x1ScaleFormat; // x1Scale数据format，仅支持ND
    aclFormat x2ScaleFormat; // x2Scale数据format，仅支持ND
    aclFormat outputFormat; // output数据format，仅支持ND
    // 其它属性
    vector<int64_t> alltoAllAxesOptional; // alltoall数据交换的方向，只能为空或者[-1,-2]
    char* group; // 通信域标识，字符串，长度要求（0，128）
    bool transposeX1; // x1是否转置，现不支持为true
    bool transposeX2; // x2是否转置，为true时x2shape为（H2，H1）
    aclnnStatus aclnnStatusUt; //期望状态
};

static QuantMatmulAlltoAllAclnnTestParam g_quantCasesParams[] = {
    // 正常用例 48条，caseid按照[算子名-x1-x2-bias-x1scale-x2scale-output-format-transpose-id]构成
    // x1Dtype = ACL_FLOAT8_E4M3FN, x2Dtype = ACL_FLOAT8_E4M3FN (共12个), 按output（bf16、fp16、fp32）分组
    {"AclnnQuantMatmulAlltoAll-e4m3-e4m3-f32-f32-f32-bf16-nd-notrans-01", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e4m3-f32-f32-f32-bf16-nd-trans-02", 2, 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e4m3-undef-f32-f32-bf16-nd-notrans-03", 2, 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e4m3-undef-f32-f32-bf16-nd-trans-04", 2, 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e4m3-f32-f32-f32-fp16-nd-notrans-05", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e4m3-f32-f32-f32-fp16-nd-trans-06", 2, 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e4m3-undef-f32-f32-fp16-nd-notrans-07", 2, 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e4m3-undef-f32-f32-fp16-nd-trans-08", 2, 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e4m3-f32-f32-f32-fp32-nd-notrans-09", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e4m3-f32-f32-f32-fp32-nd-trans-10", 2, 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e4m3-undef-f32-f32-fp32-nd-notrans-11", 2, 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e4m3-undef-f32-f32-fp32-nd-trans-12", 2, 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    // x1Dtype = ACL_FLOAT8_E4M3FN, x2Dtype = ACL_FLOAT8_E5M2 (共12个)
    {"AclnnQuantMatmulAlltoAll-e4m3-e5m2-f32-f32-f32-bf16-nd-notrans-13", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e5m2-f32-f32-f32-bf16-nd-trans-14", 2, 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e5m2-undef-f32-f32-bf16-nd-notrans-15", 2, 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e5m2-undef-f32-f32-bf16-nd-trans-16", 2, 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e5m2-f32-f32-f32-fp16-nd-notrans-17", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e5m2-f32-f32-f32-fp16-nd-trans-18", 2, 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e5m2-undef-f32-f32-fp16-nd-notrans-19", 2, 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e5m2-undef-f32-f32-fp16-nd-trans-20", 2, 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e5m2-f32-f32-f32-fp32-nd-notrans-21", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e5m2-f32-f32-f32-fp32-nd-trans-22", 2, 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e5m2-undef-f32-f32-fp32-nd-notrans-23", 2, 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e4m3-e5m2-undef-f32-f32-fp32-nd-trans-24", 2, 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    // x1Dtype = ACL_FLOAT8_E5M2, x2Dtype = ACL_FLOAT8_E4M3FN (共12个)
    {"AclnnQuantMatmulAlltoAll-e5m2-e4m3-f32-f32-f32-bf16-nd-notrans-25", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e4m3-f32-f32-f32-bf16-nd-trans-26", 2, 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e4m3-undef-f32-f32-bf16-nd-notrans-27", 2, 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e4m3-undef-f32-f32-bf16-nd-trans-28", 2, 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e4m3-f32-f32-f32-fp16-nd-notrans-29", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e4m3-f32-f32-f32-fp16-nd-trans-30", 2, 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e4m3-undef-f32-f32-fp16-nd-notrans-31", 2, 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e4m3-undef-f32-f32-fp16-nd-trans-32", 2, 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e4m3-f32-f32-f32-fp32-nd-notrans-33", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e4m3-f32-f32-f32-fp32-nd-trans-34", 2, 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e4m3-undef-f32-f32-fp32-nd-notrans-35", 2, 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e4m3-undef-f32-f32-fp32-nd-trans-36", 2, 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    // x1Dtype = ACL_FLOAT8_E5M2, x2Dtype = ACL_FLOAT8_E5M2 (共12个)
    {"AclnnQuantMatmulAlltoAll-e5m2-e5m2-f32-f32-f32-bf16-nd-notrans-37", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e5m2-f32-f32-f32-bf16-nd-trans-38", 2, 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e5m2-undef-f32-f32-bf16-nd-notrans-39", 2, 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e5m2-undef-f32-f32-bf16-nd-trans-40", 2, 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e5m2-f32-f32-f32-fp16-nd-notrans-41", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e5m2-f32-f32-f32-fp16-nd-trans-42", 2, 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e5m2-undef-f32-f32-fp16-nd-notrans-43", 2, 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e5m2-undef-f32-f32-fp16-nd-trans-44", 2, 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e5m2-f32-f32-f32-fp32-nd-notrans-45", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e5m2-f32-f32-f32-fp32-nd-trans-46", 2, 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e5m2-undef-f32-f32-fp32-nd-notrans-47", 2, 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnQuantMatmulAlltoAll-e5m2-e5m2-undef-f32-f32-fp32-nd-trans-48", 2, 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_SUCCESS},

    // 异常用例 32条，caseid按照[error-算子名-异常原因-id]构成
    // 1. x1 dtype不合法(ACL_INT8)
    {"error-AclnnQuantMatmulAlltoAll-x1dtype_invalid-01", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_INT8, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 2. x2 dtype不合法 (ACL_UINT8)
    {"error-AclnnQuantMatmulAlltoAll-x2dtype_invalid-02", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_UINT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 3. bias dtype不合法
    {"error-AclnnQuantMatmulAlltoAll-biasdtype_invalid-03", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 4. x1scale dtype不合法
    {"error-AclnnQuantMatmulAlltoAll-x1scaledtype_invalid-04", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 5. x2scale dtype不合法
    {"error-AclnnQuantMatmulAlltoAll-x2scaledtype_invalid-05", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT8_E4M3FN, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 6. output dtype不合法
    {"error-AclnnQuantMatmulAlltoAll-outdtype_mismatch_06", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 7. 空tensor
    // 7.1 x1有维度为0，first dim
    {"error-AclnnQuantMatmulAlltoAll-x1empty-07", 2, 3, 2, {0, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 7.2 x1有维度为0，second dim
    {"error-AclnnQuantMatmulAlltoAll-x1empty-08", 2, 3, 2, {256, 0}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 7.3 x2有维度为0，first dim
    {"error-AclnnQuantMatmulAlltoAll-x2empty-09", 2, 3, 2, {256, 128}, {0, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 7.4 x2有维度为0，second dim
    {"error-AclnnQuantMatmulAlltoAll-x2empty-10", 2, 3, 2, {256, 128}, {128, 0}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 8. format为私有格式 (6条)
    {"error-AclnnQuantMatmulAlltoAll-private_fmt1-11", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    {"error-AclnnQuantMatmulAlltoAll-private_fmt2-12", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    {"error-AclnnQuantMatmulAlltoAll-private_fmt3-13", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    {"error-AclnnQuantMatmulAlltoAll-private_fmt4-14", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    {"error-AclnnQuantMatmulAlltoAll-private_fmt5-15", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    {"error-AclnnQuantMatmulAlltoAll-private_fmt6-16", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 9. AlltoAllAxes不合法
    {"error-AclnnQuantMatmulAlltoAll-invalid_axes-17", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {338, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10. group不合法
    // 10.1 group为空
    {"error-AclnnQuantMatmulAlltoAll-group_empty-18", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.2 group长度超过128
    {"error-AclnnQuantMatmulAlltoAll-group_empty-19", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567",
        false, false, ACLNN_ERR_PARAM_INVALID},
    // 11. transposeX1=true
    {"error-AclnnQuantMatmulAlltoAll-transx1-20", 2, 3, 2, {128, 256}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", true, false, ACLNN_ERR_PARAM_INVALID},
    // 12. shape不合法
    // 12.1 x1维度不合法
    {"error-AclnnQuantMatmulAlltoAll-invalid_x1dim-21", 2, 3, 2, {16, 256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.2 x2维度不合法
    {"error-AclnnQuantMatmulAlltoAll-invalid_x2dim-22", 2, 3, 2, {256, 128}, {128, 256, 32}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.3 output维度不合法
    {"error-AclnnQuantMatmulAlltoAll-invalid_outputdim-23", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {512, 128, 32},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.4 bias维度不合法
    {"error-AclnnQuantMatmulAlltoAll-invalid_biasdim-24", 2, 3, 2, {256, 128}, {128, 256}, {256, 32}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.5 x1scale维度不合法
    {"error-AclnnQuantMatmulAlltoAll-invalid_x1scaledim-25", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256, 32}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.6 x2scale维度不合法
    {"error-AclnnQuantMatmulAlltoAll-invalid_x2scaledim-26", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256, 32}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.7 x1和x2的k轴不匹配(x2不转置)
    {"error-AclnnQuantMatmulAlltoAll-mismatch_kdim-27", 2, 3, 2, {256, 128}, {32, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.8 x1和x2的k轴不匹配(x2转置)
    {"error-AclnnQuantMatmulAlltoAll-mismatch_kdim-28", 2, 3, 2, {256, 128}, {256, 32}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, true, ACLNN_ERR_PARAM_INVALID},
    // 10.9 k轴超出范围
    {"error-AclnnQuantMatmulAlltoAll-outrange_kdim-29", 2, 3, 2, {256, 65536}, {65536, 256}, {256}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.10 bias和x2不匹配
    {"error-AclnnQuantMatmulAlltoAll-invalid_bias_shape-30", 2, 3, 2, {256, 128}, {128, 256}, {128}, {256}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.11 x1scale与x1不匹配
    {"error-AclnnQuantMatmulAlltoAll-invalid_x1scale_shape-31", 2, 3, 2, {256, 128}, {128, 256}, {256}, {64}, {256}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.12 x2scale与x2不匹配
    {"error-AclnnQuantMatmulAlltoAll-invalid_x2scale_shape-32", 2, 3, 2, {256, 128}, {128, 256}, {256}, {256}, {64}, {512, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_quant_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID}
};

static void TestQuantOneParamCase(const QuantMatmulAlltoAllAclnnTestParam& param)
{
    std::cout << "run case " << param.caseName << std::endl;
    // 从结构体list中获取实际用例属性
    int64_t x1quantmode = param.x1Quantmode;
    int64_t x2quantmode = param.x2Quantmode;
    vector<int64_t> x1Shape = param.x1Shape;
    vector<int64_t> x2Shape = param.x2Shape;
    vector<int64_t> biasShape = param.biasShape;
    vector<int64_t> x1scalesShape = param.x1ScaleShape;
    vector<int64_t> x2scalesShape = param.x2ScaleShape;
    vector<int64_t> outputShape = param.outputShape;
    aclDataType x1Dtype = param.x1Dtype;
    aclDataType x2Dtype = param.x2Dtype;
    aclDataType biasDtype = param.biasDtype;
    aclDataType x1scalesDtype = param.x1ScaleDtype;
    aclDataType x2scalesDtype = param.x2ScaleDtype;
    aclDataType outputDtype = param.outputDtype;
    aclFormat x1Format = param.x1Format;
    aclFormat x2Format = param.x2Format;
    aclFormat biasFormat = param.biasFormat;
    aclFormat x1ScaleFormat = param.x1ScaleFormat;
    aclFormat x2ScaleFormat = param.x2ScaleFormat;
    aclFormat outputFormat = param.outputFormat;
    vector<int64_t> axesAcl = param.alltoAllAxesOptional;
    aclIntArray *alltoAllAxesOptional = aclCreateIntArray(axesAcl.data(), axesAcl.size());
    const char* group = param.group;
    bool transposeX1 = param.transposeX1;
    bool transposeX2 = param.transposeX2;
    aclnnStatus retStatus = param.aclnnStatusUt;
    TensorDesc x1 = TensorDesc(x1Shape, x1Dtype, x1Format);
    TensorDesc x2 = TensorDesc(x2Shape, x2Dtype, x2Format);
    TensorDesc x1scales = TensorDesc(x1scalesShape, x1scalesDtype, x1ScaleFormat);
    TensorDesc x2scales = TensorDesc(x2scalesShape, x2scalesDtype, x2ScaleFormat);
    TensorDesc output = TensorDesc(outputShape, outputDtype, outputFormat);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (biasShape.empty()) {
        auto ut = OP_API_UT(aclnnQuantMatmulAlltoAll,
                            INPUT(x1, x2, nullptr, x1scales, x2scales, nullptr, nullptr, nullptr, alltoAllAxesOptional, group,
                                  x1quantmode, x2quantmode, 0, -1, 0, transposeX1, transposeX2),
                            OUTPUT(output));
        aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
        EXPECT_EQ(aclRet, retStatus);
    } else {
        TensorDesc bias = TensorDesc(biasShape, biasDtype, biasFormat);
        auto ut = OP_API_UT(aclnnQuantMatmulAlltoAll,
                            INPUT(x1, x2, bias, x1scales, x2scales, nullptr, nullptr, nullptr, alltoAllAxesOptional, group,
                                  x1quantmode, x2quantmode, 0, -1, 0, transposeX1, transposeX2),
                            OUTPUT(output));
        aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
        EXPECT_EQ(aclRet, retStatus);
    }
    std::cout << "end case " <<  param.caseName << std::endl;
}

TEST_F(TestAclnnQuantMatmulAlltoAll, QuantCasesParamsTest)
{
    if (std::size(g_quantCasesParams) != 0) {
        uint64_t numCases = sizeof(g_quantCasesParams) / sizeof(g_quantCasesParams[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestQuantOneParamCase(g_quantCasesParams[idx]);
        }
    }
}

// 此表只为构造QuantMatmulAlltoAll暂不支持的参数验证ut
struct QuantMatmulAlltoAllAclnnTest2Param {
    // 用例名
    string caseName;
    vector<int64_t> commScaleOptionalShape; // 低比特通信的量化系数shape，暂不支持。
    vector<int64_t> x1OffsetOptionalShape; // 左矩阵的偏置shape，暂不支持。
    vector<int64_t> x2OffsetOptionalShape; // 右矩阵的偏置shape，暂不支持。
    aclDataType commScaleOptionalDtype; // 低比特通信的量化系数dtype，暂不支持。
    aclDataType x1OffsetOptionalDtype; // 左矩阵的偏置dtype，暂不支持。
    aclDataType x2OffsetOptionalDtype; // 右矩阵的偏置dtype，暂不支持。
    aclFormat commScaleOptionalFormat; // 低比特通信的量化系数format，暂不支持。
    aclFormat x1OffsetOptionalFormat; // 左矩阵的偏置format，暂不支持。
    aclFormat x2OffsetOptionalFormat; // 右矩阵的偏置format，暂不支持。
    int64_t commQuantMode; // 低比特通信的量化模式，预留参数，当前仅支持配置为0，表示不量化。
    int64_t commQuantDtype; // 低比特通信的量化类型，预留参数，当前仅支持配置为-1，表示ACL_DT_UNDEFINED。
    int64_t groupSize; // 用于Matmul计算三个方向上的量化分组大小，预留参数，T-C量化模式下仅支持配置为0，取值不生效。
    aclnnStatus aclnnStatusUt; //期望状态
};

static QuantMatmulAlltoAllAclnnTest2Param g_reservedCases[] = {
    // 正常用例 1条
    // 低比特通信tensor和偏置tensor都为空，commQuantMode取0，commQuantDtype取-1，groupSize取0
    {"AclnnQuantMatmulAlltoAll-all_reserved_params_valid", {}, {}, {}, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, 0, -1, 0, ACLNN_SUCCESS},
    // 异常用例 6条
    {"error-AclnnQuantMatmulAlltoAll-commScale_invalid-01", {256}, {}, {}, ACL_FLOAT, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, 0, -1, 0, ACLNN_ERR_PARAM_INVALID},
    {"error-AclnnQuantMatmulAlltoAll-x1Offset_invalid-02", {}, {256}, {}, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_DT_UNDEFINED,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, 0, -1, 0, ACLNN_ERR_PARAM_INVALID},
    {"error-AclnnQuantMatmulAlltoAll-x2Offset_invalid-03", {}, {}, {256}, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, 0, -1, 0, ACLNN_ERR_PARAM_INVALID},
    {"error-AclnnQuantMatmulAlltoAll-commQuantMode_invalid-04", {}, {}, {}, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, 1, -1, 0, ACLNN_ERR_PARAM_INVALID},
    {"error-AclnnQuantMatmulAlltoAll-commQuantDtype_invalid-05", {}, {}, {}, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, 0, 0, 0, ACLNN_ERR_PARAM_INVALID},
    {"error-AclnnQuantMatmulAlltoAll-groupSize_invalid-06", {}, {}, {}, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, 0, -1, 32, ACLNN_ERR_PARAM_INVALID}
};

static void TestQuantReservedCase(const QuantMatmulAlltoAllAclnnTest2Param& param)
{
    std::cout << "run case " << param.caseName << std::endl;
    // 从结构体list中获取实际用例属性
    vector<int64_t> commScaleShape = param.commScaleOptionalShape;
    vector<int64_t> x1OffsetShape = param.x1OffsetOptionalShape;
    vector<int64_t> x2OffsetShape = param.x2OffsetOptionalShape;
    aclDataType commScaleDtype = param.commScaleOptionalDtype;
    aclDataType x1OffsetDtype = param.x1OffsetOptionalDtype;
    aclDataType x2OffsetDtype = param.x2OffsetOptionalDtype;
    aclFormat commScaleFormat = param.commScaleOptionalFormat;
    aclFormat x1OffsetFormat = param.x1OffsetOptionalFormat;
    aclFormat x2OffsetFormat = param.x2OffsetOptionalFormat;
    aclnnStatus retStatus = param.aclnnStatusUt;
    TensorDesc commScale = TensorDesc(commScaleShape, commScaleDtype, commScaleFormat);
    TensorDesc x1Offset = TensorDesc(x1OffsetShape, x1OffsetDtype, x1OffsetFormat);
    TensorDesc x2Offset = TensorDesc(x2OffsetShape, x2OffsetDtype, x2OffsetFormat);
    int64_t commQuantMode = param.commQuantMode;
    int64_t commQuantDtype = param.commQuantDtype;
    int64_t groupSize = param.groupSize;
    // 生成其它参数
    TensorDesc x1 = TensorDesc({256, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({128, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x1scales = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scales = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({512, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    vector<int64_t> axesAcl = {-1, -2};
    aclIntArray *alltoAllAxesOptional = aclCreateIntArray(axesAcl.data(), axesAcl.size());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (!commScaleShape.empty()) {
        auto ut = OP_API_UT(aclnnQuantMatmulAlltoAll,
                    INPUT(x1, x2, bias, x1scales, x2scales, commScale, nullptr, nullptr, alltoAllAxesOptional, "ut_test_quant_matmul_allto_all",
                          3, 2, commQuantMode, commQuantDtype, groupSize, false, false),
                    OUTPUT(output));
        aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
        EXPECT_EQ(aclRet, retStatus);
    } else if (!x1OffsetShape.empty()) {
        auto ut = OP_API_UT(aclnnQuantMatmulAlltoAll,
                    INPUT(x1, x2, bias, x1scales, x2scales, nullptr, x1Offset, nullptr, alltoAllAxesOptional, "ut_test_quant_matmul_allto_all",
                          3, 2, commQuantMode, commQuantDtype, groupSize, false, false),
                    OUTPUT(output));
        aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
        EXPECT_EQ(aclRet, retStatus);
    } else if (!x2OffsetShape.empty()) {
        auto ut = OP_API_UT(aclnnQuantMatmulAlltoAll,
            INPUT(x1, x2, bias, x1scales, x2scales, nullptr, nullptr, x2Offset, alltoAllAxesOptional, "ut_test_quant_matmul_allto_all",
                  3, 2, commQuantMode, commQuantDtype, groupSize, false, false),
            OUTPUT(output));
        aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
        EXPECT_EQ(aclRet, retStatus);
    } else {
        auto ut = OP_API_UT(aclnnQuantMatmulAlltoAll,
            INPUT(x1, x2, bias, x1scales, x2scales, nullptr, nullptr, nullptr, alltoAllAxesOptional, "ut_test_quant_matmul_allto_all",
                  3, 2, commQuantMode, commQuantDtype, groupSize, false, false),
            OUTPUT(output));
        aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
        EXPECT_EQ(aclRet, retStatus);
    }
    std::cout << "end case " <<  param.caseName << std::endl;
}

TEST_F(TestAclnnQuantMatmulAlltoAll, ReservedCasesTest)
{
    if (std::size(g_reservedCases) != 0) {
        uint64_t numCases = sizeof(g_reservedCases) / sizeof(g_reservedCases[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestQuantReservedCase(g_reservedCases[idx]);
        }
    }
}