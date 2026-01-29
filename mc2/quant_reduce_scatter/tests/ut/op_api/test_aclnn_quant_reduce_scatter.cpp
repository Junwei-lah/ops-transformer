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
#include "../../../op_api/aclnn_quant_reduce_scatter.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class test_aclnn_quant_reduce_scatter : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        cout << "test_aclnn_quant_reduce_scatter SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "test_aclnn_quant_reduce_scatter TearDown" << endl;
    }
};

struct QuantReduceScatterAclnnTestParam {
    string case_name;
    vector<int64_t> x_shape; // x数据shape
    vector<int64_t> scales_shape; // scales数据shape
    vector<int64_t> output_shape; // output数据shape
    char* group; // 通信域标识
    aclDataType x_dtype; // x数据dtype
    aclDataType scales_dtype; // scales数据dtype
    aclDataType output_dtype; // 输出数据dtype
    aclnnStatus aclnn_status;
};

static QuantReduceScatterAclnnTestParam cases_params[] = {
    // 正常用例
    {"test_aclnn_quant_reduce_scatter_mx_BS1024_H5120_FLOAT8E4M3FN_FLOAT8E8M0_FLOAT16_true",{1024, 5120}, {1024, 80, 2}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT16,ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_mx_BS1024_H5120_FLOAT8E5M2_FLOAT8E8M0_BF16_true", {1024, 5120}, {1024, 80, 2}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_BF16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_mx_BS1024_H7168_FLOAT8E4M3FN_FLOAT8E8M0_FLOAT16_true", {1024, 7168}, {1024, 112, 2}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_mx_BS1024_H7168_FLOAT8E5M2_FLOAT8E8M0_BF16_true", {1024, 7168}, {1024, 112, 2}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_BF16, ACLNN_SUCCESS},

    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H5120_INT8_FLOAT_FLOAT16_true", {1024, 5120}, {1024, 40}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_INT8, ACL_FLOAT, ACL_FLOAT16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H5120_HIFLOAT8_FLOAT_BF16_true", {1024, 5120}, {1024, 40}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_HIFLOAT8, ACL_FLOAT, ACL_BF16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H5120_FLOAT8E4M3FN_FLOAT_FLOAT_true", {1024, 5120}, {1024, 40}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H5120_FLOAT8E5M2_FLOAT_FLOAT_true", {1024, 5120}, {1024, 40}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACLNN_SUCCESS},

    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H7168_INT8_FLOAT_FLOAT16_true", {1024, 7168}, {1024, 56}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_INT8, ACL_FLOAT, ACL_FLOAT16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H7168_HIFLOAT8_FLOAT_BF16_true", {1024, 7168}, {1024, 56}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_HIFLOAT8, ACL_FLOAT, ACL_BF16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H7168_FLOAT8E4M3FN_FLOAT_FLOAT_true", {1024, 7168}, {1024, 56}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H7168_FLOAT8E5M2_FLOAT_FLOAT_true", {1024, 7168}, {1024, 56}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_BS2048_H5120_INT8_FLOAT_FLOAT16_true", {2048, 5120}, {2048, 40}, {1024, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_INT8, ACL_FLOAT, ACL_FLOAT16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_BS2048_H5120_HIFLOAT8_FLOAT_BF16_true", {2048, 5120}, {2048, 40}, {1024, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_HIFLOAT8, ACL_FLOAT, ACL_BF16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_mx_BS2048_H7168_FLOAT8E5M2_FLOAT8E8M0_BF16_true", {2048, 7168}, {2048, 112, 2}, {1024, 7168}, 
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_BF16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_BS4096_H7168_INT8_FLOAT_FLOAT16_true", {4096, 7168}, {4096, 56}, {2048, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_INT8, ACL_FLOAT, ACL_FLOAT16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_BS8192_H5120_FLOAT8E4M3FN_FLOAT_FLOAT_true", {8192, 5120}, {8192, 40}, {4096, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACLNN_SUCCESS},

    // x类型异常
    {"test_aclnn_quant_reduce_scatter_mx_BS1024_H5120_INT8_FLOAT8E8M0_FLOAT16_xDtype_false", {1024, 5120}, {1024, 80, 2}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_INT8, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID}, // mx量化时，x不应该为INT8,HIFLOAT8
    {"test_aclnn_quant_reduce_scatter_mx_BS1024_H7168_INT8_FLOAT8E8M0_FLOAT16_xDtype_false", {1024, 7168}, {1024, 112, 2}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_INT8, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_mx_BS1024_H5120_HIFLOAT8_FLOAT8E8M0_FLOAT16_xDtype_false", {1024, 5120}, {1024, 80, 2}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_HIFLOAT8, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_mx_BS1024_H7168_HIFLOAT8_FLOAT8E8M0_FLOAT16_xDtype_false", {1024, 7168}, {1024, 112, 2}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_HIFLOAT8, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H5120_FLOAT16_FLOAT_FLOAT_xDtype_false", {1024, 5120}, {1024, 40}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID}, // T-G量化时，x不应该为FLOAT16
    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H7168_FLOAT16_FLOAT_FLOAT_xDtype_false", {1024, 7168}, {1024, 56}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_mx_BS1024_H5120_FLOAT_FLOAT_FLOAT16_xDtype_false", {1024, 5120}, {1024, 80, 2}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},

    {"test_aclnn_quant_reduce_scatter_mx_BS1024_H5120_FLOAT_FLOAT8E8M0_FLOAT16_xDtype_false", {1024, 5120}, {1024, 80, 2}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID}, // x不应该为FLOAT
    {"test_aclnn_quant_reduce_scatter_mx_BS1024_H7168_FLOAT_FLOAT8E8M0_FLOAT16_xDtype_false", {1024, 7168}, {1024, 112, 2}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    // scales类型异常
    {"test_aclnn_quant_reduce_scatter_BS1024_H5120_INT8_FLOAT16_FLOAT16_scalesDtype_false", {1024, 5120}, {1024, 40}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_INT8, ACL_FLOAT16, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID}, // scales不支持FLOAT16
    {"test_aclnn_quant_reduce_scatter_BS1024_H5120_HIFLOAT8_FLOAT16_BF16_scalesDtype_false", {1024, 5120}, {1024, 40}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_HIFLOAT8, ACL_FLOAT16, ACL_BF16, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_BS1024_H5120_FLOAT8E4M3FN_FLOAT16_FLOAT_scalesDtype_false", {1024, 5120}, {1024, 40}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E4M3FN, ACL_FLOAT16, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_BS1024_H5120_FLOAT8E5M2_FLOAT16_FLOAT_scalesDtype_false", {1024, 5120}, {1024, 40}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E5M2, ACL_FLOAT16, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_BS1024_H5120_FLOAT8E5M2_FLOAT8E4M3FN_FLOAT_scalesDtype_false", {1024, 5120}, {1024, 40}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID}, // scales不支持FLOAT8_E4M3FN

    {"test_aclnn_quant_reduce_scatter_BS1024_H7168_INT8_FLOAT16_FLOAT16_scalesDtype_false", {1024, 7168}, {1024, 56}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_INT8, ACL_FLOAT16, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_BS1024_H7168_HIFLOAT8_FLOAT16_FLOAT_scalesDtype_false", {1024, 7168}, {1024, 56}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_HIFLOAT8, ACL_FLOAT16, ACL_BF16, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_BS1024_H7168_FLOAT8E4M3FN_FLOAT16_FLOAT_scalesDtype_false", {1024, 7168}, {1024, 56}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E4M3FN, ACL_FLOAT16, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_BS1024_H7168_FLOAT8E5M2_FLOAT16_FLOAT_scalesDtype_false", {1024, 7168}, {1024, 56}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E5M2, ACL_FLOAT16, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID},
    // output类型异常
    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H5120_INT8_FLOAT_FLOAT8E5M2_outputDtype_false", {1024, 5120}, {1024, 40}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_INT8, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID}, // output不支持ACL_FLOAT8_E5M2
    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H5120_HIFLOAT8_FLOAT_FLOAT8E5M2_outputDtype_false", {1024, 5120}, {1024, 40}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H5120_FLOAT8E4M3FN_FLOAT_FLOAT8E5M2_outputDtype_false", {1024, 5120}, {1024, 40}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H5120_FLOAT8E5M2_FLOAT_FLOAT8E5M2_outputDtype_false", {1024, 5120}, {1024, 40}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID},

    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H7168_INT8_FLOAT_FLOAT8E5M2_outputDtype_false", {1024, 7168}, {1024, 56}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_INT8, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID}, // output不支持ACL_FLOAT8_E5M2
    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H7168_HIFLOAT8_FLOAT_FLOAT8E5M2_outputDtype_false", {1024, 7168}, {1024, 56}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H7168_FLOAT8E4M3FN_FLOAT_FLOAT8E5M2_outputDtype_false", {1024, 7168}, {1024, 56}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_tg_BS1024_H7168_FLOAT8E5M2_FLOAT_FLOAT8E5M2_outputDtype_false", {1024, 7168}, {1024, 56}, {512, 7168},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter__BS1024_H5120_FLOAT8E4M3FN_FLOAT8E8M0_FLOAT8E5M2_outputDtype_false", {1024, 5120}, {1024, 80, 2}, {512, 5120},
        {"quant_reduce_scatter_test_group"}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID}
};

static QuantReduceScatterAclnnTestParam group_cases_params[] = {
    // group长度校验用例, this_is_a_very_long_groupname_长度为30字符
    {"test_aclnn_quant_reduce_scatter_mx_BS1024_H5120_FLOAT8E4M3FN_FLOAT8E8M0_FLOAT16_false_group_length_127",
        {1024, 5120}, {1024, 80, 2}, {512, 5120},
        {"this_is_a_very_long_groupname_"
         "this_is_a_very_long_groupname_"
         "this_is_a_very_long_groupname_"
         "this_is_a_very_long_groupname_"
         "this_is"},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT16,
        ACLNN_SUCCESS}, // group长度为127
    {"test_aclnn_quant_reduce_scatter_mx_BS1024_H5120_FLOAT8E4M3FN_FLOAT8E8M0_FLOAT16_false_group_length_128",
        {1024, 5120}, {1024, 80, 2}, {512, 5120},
        {"this_is_a_very_long_groupname_"
         "this_is_a_very_long_groupname_"
         "this_is_a_very_long_groupname_"
         "this_is_a_very_long_groupname_"
         "this_is_"},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT16,
        ACLNN_ERR_PARAM_INVALID}, // group长度为128-越界
    {"test_aclnn_quant_reduce_scatter_mx_BS1024_H5120_FLOAT8E4M3FN_FLOAT8E8M0_FLOAT16_false_group_length_129",
        {1024, 5120}, {1024, 80, 2}, {512, 5120},
        {"this_is_a_very_long_groupname_"
         "this_is_a_very_long_groupname_"
         "this_is_a_very_long_groupname_"
         "this_is_a_very_long_groupname_"
         "this_is_a"},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT16,
        ACLNN_ERR_PARAM_INVALID}, // group长度为129-越界
};

static void TestOneParamCase(const QuantReduceScatterAclnnTestParam& param)
{
    std::cout << "run case " << param.case_name << std::endl;
    if (param.group == nullptr) {
        std::cerr << "[ERROR]: group is null" << std::endl;
        return;
    }
    vector<int64_t> xShape = param.x_shape;
    vector<int64_t> scalesShape = param.scales_shape;
    vector<int64_t> outputShape = param.output_shape;
    char* group = param.group;
    aclDataType xDtype = param.x_dtype;
    aclDataType scalesDtype = param.scales_dtype;
    aclDataType outputDtype = param.output_dtype;
    aclnnStatus retStatus = param.aclnn_status;
    TensorDesc x = TensorDesc(xShape, xDtype, ACL_FORMAT_ND);
    TensorDesc scales = TensorDesc(scalesShape, scalesDtype, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc(outputShape, outputDtype, ACL_FORMAT_ND);
    const char* reduceOp = "sum";
    auto ut = OP_API_UT(aclnnQuantReduceScatter,
                        INPUT(x, scales, group, reduceOp),
                        OUTPUT(output));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, retStatus);
}

TEST_F(test_aclnn_quant_reduce_scatter, cases_params)
{
    if (std::size(cases_params) != 0) {
        uint64_t numCases = sizeof(cases_params) / sizeof(cases_params[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestOneParamCase(cases_params[idx]);
        }
    }
}

TEST_F(test_aclnn_quant_reduce_scatter, group_cases_params)
{
    if (std::size(group_cases_params) != 0) {
        uint64_t numCases = sizeof(group_cases_params) / sizeof(group_cases_params[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestOneParamCase(group_cases_params[idx]);
        }
    }
}
