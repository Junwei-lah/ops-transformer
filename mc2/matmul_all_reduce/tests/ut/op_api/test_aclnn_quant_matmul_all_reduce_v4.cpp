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
#include "../../../op_api/aclnn_quant_matmul_all_reduce_v4.h"

namespace MatmulAllReduceUT {

class AclnnQuantMatmulAllReduceV4Test : public testing::TestWithParam<MatmulAllReduceApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce AclnnQuantMatmulAllReduceV4Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce AclnnQuantMatmulAllReduceV4Test TearDown" << std::endl;
    }
};

// 测试 aclTensor 入参为 nullptr 的场景 =================================================================================
TEST_F(AclnnQuantMatmulAllReduceV4Test, aclTensorNull)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc bias = {{128}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc x3 = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc x1Scale = {{32}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc x2Scale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc commQuantScale1 = {{128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc commQuantScale2 = {{128}, ACL_FLOAT16, ACL_FORMAT_ND};
    const char* group = "group";
    const char* reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    int64_t groupSize = 0;
    int64_t commQuantMode = 0;
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    auto ut_null_x1 = OP_API_UT(
        aclnnQuantMatmulAllReduceV4,
        INPUT(nullptr, x2, bias, x3, x1Scale, x2Scale, commQuantScale1, commQuantScale2,
              group, reduceOp, commTurn, streamMode, groupSize, commQuantMode),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x1.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x2 = OP_API_UT(
        aclnnQuantMatmulAllReduceV4,
        INPUT(x1, nullptr, bias, x3, x1Scale, x2Scale, commQuantScale1, commQuantScale2,
              group, reduceOp, commTurn, streamMode, groupSize, commQuantMode),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x2.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x2Scale = OP_API_UT(
        aclnnQuantMatmulAllReduceV4,
        INPUT(x1, x2, bias, x3, x1Scale, nullptr, commQuantScale1, commQuantScale2,
              group, reduceOp, commTurn, streamMode, groupSize, commQuantMode),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x2Scale.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_output = OP_API_UT(
        aclnnQuantMatmulAllReduceV4,
        INPUT(x1, x2, bias, x3, x1Scale, x2Scale, commQuantScale1, commQuantScale2,
              group, reduceOp, commTurn, streamMode, groupSize, commQuantMode),
        OUTPUT(nullptr)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_output.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

// 测试 aclTensor 入参不为 nullptr 的场景 ================================================================================
std::vector<MatmulAllReduceApiUtParam> casesQuantV4 {
    // 正确用例
    {"int8_pertoken", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"int8_no_x1scale", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_INT64, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"int8_bf16_output", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_BF16, ACL_FORMAT_ND}, {{}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_BF16, ACL_FORMAT_ND}, {{128}, ACL_BF16, ACL_FORMAT_ND}, {{128}, ACL_BF16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, 0, {{32, 128}, ACL_BF16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"fp8_e4m3fn", {{32, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND}, {{64, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"fp8_e5m2", {{32, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND}, {{64, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"hifloat8", {{32, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND}, {{64, 128}, ACL_HIFLOAT8, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"float4_e2m1", {{32, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND}, {{64, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{2, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND}, {{128, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"fp8_dynamic_quant", {{32, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND}, {{64, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, 1, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    {"no_comm_quant", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_SUCCESS},
    // 失败用例：参数无效
    {"invalid_float4_e1m2", {{32, 64}, ACL_FLOAT4_E1M2, ACL_FORMAT_ND}, {{64, 128}, ACL_FLOAT4_E1M2, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{2, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND}, {{128, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, {{}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_group_nullptr", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, nullptr, "sum", 0, 1, 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_NULLPTR},
    {"invalid_reduce_op", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "max", 0, 1, 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_stream_mode_zero", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 0, 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    // 失败用例：shape 冲突
    {"invalid_shape_x1_and_x2", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{48, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    // 失败用例：dtype 冲突
    {"invalid_dtype_x1", {{32, 64}, ACL_INT16, ACL_FORMAT_ND}, {{64, 128}, ACL_INT8, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID},
    {"invalid_dtype_x2", {{32, 64}, ACL_INT8, ACL_FORMAT_ND}, {{64, 128}, ACL_INT16, ACL_FORMAT_ND}, {{128}, ACL_INT32, ACL_FORMAT_ND}, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{32}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, {{128}, ACL_FLOAT16, ACL_FORMAT_ND}, "group", "sum", 0, 1, 0, 0, {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND}, ACLNN_ERR_PARAM_INVALID}
};

TEST_P(AclnnQuantMatmulAllReduceV4Test, param)
{
    auto param = GetParam();
    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceV4,
        INPUT(param.x1, param.x2, param.bias, param.x3, param.x1Scale, param.x2Scale,
              param.commQuantScale1, param.commQuantScale2, param.group, param.reduceOp, 
              param.commTurn, param.streamMode, param.groupSize, param.commQuantMode),
        OUTPUT(param.output)
    );
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(param.expectAclnnStatus, ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce,
    AclnnQuantMatmulAllReduceV4Test,
    testing::ValuesIn(casesQuantV4),
    PrintMatmulAllReduceApiUtParam
);

} // namespace matmul_all_reduce_ut
