/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <sstream>
#include <vector>
#include <gtest/gtest.h>
#include "opdev/platform.h"
#include "matmul_all_reduce_api_ut_param.h"
#include "../../../op_api/aclnn_matmul_all_reduce_v2.h"

namespace MatmulAllReduceUT {

class AclnnMatmulAllReduceV2Test : public testing::TestWithParam<MatmulAllReduceApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce AclnnMatmulAllReduceV2Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce AclnnMatmulAllReduceV2Test TearDown" << std::endl;
    }
};

// 测试 aclTensor 入参为 nullptr 的场景 =================================================================================
TEST_F(AclnnMatmulAllReduceV2Test, aclTensorNull)
{
    TensorDesc x1 = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char* group = "group";
    const char* reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    auto ut_null_x1 = OP_API_UT(
        aclnnMatmulAllReduceV2,
        INPUT(nullptr, x2, bias, x3, group, reduceOp, commTurn, streamMode),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x1.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x2 = OP_API_UT(
        aclnnMatmulAllReduceV2,
        INPUT(x1, nullptr, bias, x3, group, reduceOp, commTurn, streamMode),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x2.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_bias = OP_API_UT(
        aclnnMatmulAllReduceV2,
        INPUT(x1, x2, nullptr, x3, group, reduceOp, commTurn, streamMode),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_SUCCESS, ut_null_bias.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x3 = OP_API_UT(
        aclnnMatmulAllReduceV2,
        INPUT(x1, x2, bias, nullptr, group, reduceOp, commTurn, streamMode),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_SUCCESS, ut_null_x3.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_output = OP_API_UT(
        aclnnMatmulAllReduceV2,
        INPUT(x1, x2, bias, x3, group, reduceOp, commTurn, streamMode),
        OUTPUT(nullptr)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_output.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

// 测试 aclTensor 入参不为 nullptr 的场景 ================================================================================
std::vector<MatmulAllReduceApiUtParam> casesV2 {
    // 正确用例
    {"common_2d", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"common_3d_x1", {{2, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{2, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{2, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"common_bf16", {{32, 64}, ACL_BF16, ACL_FORMAT_ND}, {{64, 128}, ACL_BF16, ACL_FORMAT_ND}, {{128}, ACL_BF16, ACL_FORMAT_ND}, {{32, 128}, ACL_BF16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{32, 128}, ACL_BF16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"empty_K", {{16, 0}, ACL_FLOAT16, ACL_FORMAT_ND}, {{0, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"empty_M", {{0, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{0, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{0, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"empty_N", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 0}, ACL_FLOAT16, ACL_FORMAT_ND}, {{0}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 0}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 0}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"large_shape_no_cube", {{128, 8192}, ACL_FLOAT16, ACL_FORMAT_ND}, {{8192, 11296}, ACL_FLOAT16, ACL_FORMAT_ND}, {{11296}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128, 11296}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{128, 11296}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    // 失败用例：参数无效
    {"invalid_x1_shape_4d", {{2, 8, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{2, 8, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{2, 8, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_x1_shape_empty", {{}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_x2_shape_3d", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16, 1}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_x3_shape_3d", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{2, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_group_nullptr", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, nullptr, "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_NULLPTR},
    {"invalid_reduce_op", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "max", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_stream_mode_zero", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 0, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_stream_mode_negative", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, -1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_dtype_x1_int8", {{16, 32}, ACL_INT8, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_dtype_x2_int8", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_INT8, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    // 失败用例：shape 冲突
    {"invalid_shape_x1_and_x2_k", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_shape_x1_and_output", {{32, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_shape_x2_and_output_n", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_shape_bias_and_output_n", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_shape_x3_and_output", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    // 失败用例：dtype 冲突
    {"invalid_dtype_x1_and_x2", {{16, 32}, ACL_FLOAT, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_dtype_x1_and_bias", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_BF16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_dtype_x1_and_x3", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_BF16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_dtype_x1_and_output", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 16}, ACL_BF16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID}
};

TEST_P(AclnnMatmulAllReduceV2Test, param)
{
    auto param = GetParam();
    auto ut = OP_API_UT(
        aclnnMatmulAllReduceV2,
        INPUT(param.x1, param.x2, param.bias, param.x3, param.group, param.reduceOp, param.commTurn, param.streamMode),
        OUTPUT(param.output)
    );
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(param.expectAclnnStatus, ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce,
    AclnnMatmulAllReduceV2Test,
    testing::ValuesIn(casesV2),
    PrintMatmulAllReduceApiUtParam
);

} // namespace matmul_all_reduce_ut
