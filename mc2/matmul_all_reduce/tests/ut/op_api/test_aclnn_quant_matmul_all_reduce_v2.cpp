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
#include "../../../op_api/aclnn_quant_matmul_all_reduce_v2.h"

namespace matmul_all_reduce_ut {

class AclnnQuantMatmulAllReduceV2Test : public testing::TestWithParam<MatmulAllReduceApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce AclnnQuantMatmulAllReduceV2Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce AclnnQuantMatmulAllReduceV2Test TearDown" << std::endl;
    }
};

// 测试 aclTensor 入参为 nullptr 的场景 =================================================================================
TEST_F(AclnnQuantMatmulAllReduceV2Test, aclTensorNull)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc bias = {{128}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc x3 = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc dequantScale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc pertokenScale = {{32}, ACL_FLOAT, ACL_FORMAT_ND};
    const char* group = "group";
    const char* reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    auto ut_null_x1 = OP_API_UT(
        aclnnQuantMatmulAllReduceV2,
        INPUT(nullptr, x2, bias, x3, dequantScale, pertokenScale, group, reduceOp, commTurn, streamMode),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x1.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x2 = OP_API_UT(
        aclnnQuantMatmulAllReduceV2,
        INPUT(x1, nullptr, bias, x3, dequantScale, pertokenScale, group, reduceOp, commTurn, streamMode),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x2.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_dScale = OP_API_UT(
        aclnnQuantMatmulAllReduceV2,
        INPUT(x1, x2, bias, x3, nullptr, pertokenScale, group, reduceOp, commTurn, streamMode),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_dScale.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_output = OP_API_UT(
        aclnnQuantMatmulAllReduceV2,
        INPUT(x1, x2, bias, x3, dequantScale, pertokenScale, group, reduceOp, commTurn, streamMode),
        OUTPUT(nullptr)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_output.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

// 测试 aclTensor 入参不为 nullptr 的场景 ================================================================================
std::vector<MatmulAllReduceApiUtParam> casesQuantV2 {
    // 正确用例
    {"pertoken_quant_fp16", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"pertoken_quant_bf16", {{16, 32}, ACL_INT8, ACL_FORMAT_ND}, {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64}, ACL_INT32, ACL_FORMAT_ND}, {{16, 64}, ACL_BF16, ACL_FORMAT_ND}, {{64}, ACL_BF16, ACL_FORMAT_ND}, {{16}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 64}, ACL_BF16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"x1_3d_pertoken", {{2, 16, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 32}, ACL_INT8, ACL_FORMAT_ND}, {{32}, ACL_INT32, ACL_FORMAT_ND}, {{2, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{2, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    // 参数无效
    {"invalid_group_nullptr", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, nullptr, "sum", 0, 1, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_NULLPTR},
    {"invalid_reduce_op", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "max", 0, 1, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_stream_mode_zero", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_stream_mode_negative", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, -1, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_x1_shape_4d", {{2, 8, 16, 32}, ACL_INT8, ACL_FORMAT_ND}, {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64}, ACL_INT32, ACL_FORMAT_ND}, {{2, 8, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, {{64}, ACL_FLOAT, ACL_FORMAT_ND}, {{16}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{2, 8, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_x2_shape_3d", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 32, 1}, ACL_INT8, ACL_FORMAT_ND}, {{32}, ACL_INT32, ACL_FORMAT_ND}, {{32, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{32, 32}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    // shape 冲突
    {"invalid_shape_x1_and_x2", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{48, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_shape_x1_and_output", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{16, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_shape_bias_and_output", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{64}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_shape_pertokenScale_2d", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32, 1}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_shape_pertokenScale_size", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{16}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    // dtype 冲突
    {"invalid_dtype_x1", {{32, 64}, ACL_INT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_dtype_x2", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT16, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_dtype_bias", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT16, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_dtype_dequant_pertoken_mismatch", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_INT64, ACL_FORMAT_ND}, {{0}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_dtype_x3", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, "group", "sum", 0, 1, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
};

TEST_P(AclnnQuantMatmulAllReduceV2Test, param)
{
    auto param = GetParam();
    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceV2,
        INPUT(param.x1, param.x2, param.bias, param.x3, param.dequantScale, param.pertokenScale,
              param.group, param.reduceOp, param.commTurn, param.streamMode),
        OUTPUT(param.output)
    );
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(param.expectAclnnStatus, ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce,
    AclnnQuantMatmulAllReduceV2Test,
    testing::ValuesIn(casesQuantV2),
    PrintMatmulAllReduceApiUtParam
);

} // namespace matmul_all_reduce_ut
