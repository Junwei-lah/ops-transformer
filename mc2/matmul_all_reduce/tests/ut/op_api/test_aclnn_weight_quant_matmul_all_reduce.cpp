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
#include "../../../op_api/aclnn_weight_quant_matmul_all_reduce.h"

namespace matmul_all_reduce_ut {

class AclnnWeightQuantMatmulAllReduceTest : public testing::TestWithParam<MatmulAllReduceApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce AclnnWeightQuantMatmulAllReduceTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce AclnnWeightQuantMatmulAllReduceTest TearDown" << std::endl;
    }
};

// 测试 aclTensor 入参为 nullptr 的场景 =================================================================================
TEST_F(AclnnWeightQuantMatmulAllReduceTest, aclTensorNull)
{
    TensorDesc x1 = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({64, 128}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantOffset = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3 = TensorDesc({32, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char* group = "group";
    const char* reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    int64_t groupSize = 0;
    TensorDesc output = TensorDesc({32, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    auto ut_null_x1 = OP_API_UT(
        aclnnWeightQuantMatmulAllReduce,
        INPUT(nullptr, x2, bias, antiquantScale, antiquantOffset, x3, group, reduceOp, commTurn, streamMode, groupSize),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x1.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x2 = OP_API_UT(
        aclnnWeightQuantMatmulAllReduce,
        INPUT(x1, nullptr, bias, antiquantScale, antiquantOffset, x3, group, reduceOp, commTurn, streamMode, groupSize),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x2.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_scale = OP_API_UT(
        aclnnWeightQuantMatmulAllReduce,
        INPUT(x1, x2, bias, nullptr, antiquantOffset, x3, group, reduceOp, commTurn, streamMode, groupSize),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_scale.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_output = OP_API_UT(
        aclnnWeightQuantMatmulAllReduce,
        INPUT(x1, x2, bias, antiquantScale, antiquantOffset, x3, group, reduceOp, commTurn, streamMode, groupSize),
        OUTPUT(nullptr)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_output.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

// 测试 aclTensor 入参不为 nullptr 的场景 ================================================================================
std::vector<MatmulAllReduceApiUtParam> casesWeightQuant {
    // 正确用例
    {"common_1", {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"common_2", {{1, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 256}, ACL_INT8, ACL_FORMAT_ND}, {{256}, ACL_FLOAT16, ACL_FORMAT_ND}, {{256}, ACL_FLOAT16, ACL_FORMAT_ND}, {{256}, ACL_FLOAT16, ACL_FORMAT_ND}, {{1, 256}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{1, 256}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"x1_shape_3d", {{2, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 32}, ACL_INT8, ACL_FORMAT_ND}, {{32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{2, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{2, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"empty_K", {{32, 0}, ACL_FLOAT16, ACL_FORMAT_ND}, {{0, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"pergroup_quant", {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{2, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{2, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 32, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"pertensor_quant", {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{1}, ACL_FLOAT16, ACL_FORMAT_ND}, {{1}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{16, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"perchannel_quant", {{8, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 48}, ACL_INT8, ACL_FORMAT_ND}, {{48}, ACL_FLOAT16, ACL_FORMAT_ND}, {{1, 48}, ACL_FLOAT16, ACL_FORMAT_ND}, {{1, 48}, ACL_FLOAT16, ACL_FORMAT_ND}, {{8, 48}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{8, 48}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    // 失败用例：参数无效
    {"empty_M", {{0, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{0, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{0, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_x1_shape_4d", {{2, 8, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{2, 8, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{2, 8, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_x2_shape_3d", {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 32, 1}, ACL_INT8, ACL_FORMAT_ND}, {{32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{32, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_group_nullptr", {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, nullptr, "sum", 0, 1, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_NULLPTR},
    {"invalid_reduce_op", {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "max", 0, 1, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_stream_mode_zero", {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_stream_mode_negative", {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, -1, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    // 失败用例：shape 冲突
    {"invalid_shape_x1_and_x2", {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{48, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_shape_x1_and_output", {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{16, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{16, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_shape_bias_and_output", {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_shape_scale_and_output", {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_pergroup_size", {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{4, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{4, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 16, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    // 失败用例：dtype 冲突
    {"invalid_dtype_x1_and_x2", {{32, 64}, ACL_FLOAT, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_dtype_x1_and_scale", {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_BF16, ACL_FORMAT_ND}, {{128}, ACL_BF16, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_dtype_x1_and_output", {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32, 128}, ACL_BF16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{32, 128}, ACL_BF16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_dtype_scale_and_offset", {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_BF16, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID}
};

TEST_P(AclnnWeightQuantMatmulAllReduceTest, param)
{
    auto param = GetParam();
    auto ut = OP_API_UT(
        aclnnWeightQuantMatmulAllReduce,
        INPUT(param.x1, param.x2, param.bias, param.antiquantScale, param.antiquantOffset, param.x3,
              param.group, param.reduceOp, param.commTurn, param.streamMode, param.groupSize),
        OUTPUT(param.output)
    );
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(param.expectAclnnStatus, ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce,
    AclnnWeightQuantMatmulAllReduceTest,
    testing::ValuesIn(casesWeightQuant),
    PrintMatmulAllReduceApiUtParam
);

} // namespace matmul_all_reduce_ut
