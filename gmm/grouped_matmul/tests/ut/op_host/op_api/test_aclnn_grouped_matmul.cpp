/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_v5.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_v4.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_weight_nz.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_grouped_matmul_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_grouped_matmul_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_grouped_matmul_test TearDown" << endl;
    }
};

TEST_F(l2_grouped_matmul_test, Ascend910B2_grouped_matmul_fp16)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = nullptr;
    auto offsetOptionsl = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;
    //  auto sortedIndices = TensorListDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    //  auto routingMapOptional = TensorListDesc({512, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out = TensorListDesc(1,TensorDesc({M,N}, ACL_FLOAT16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptionsl, antiquantScaleOptional,
                              antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                              activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                              splitItem, groupType, groupListType, actType, tuningConfigOptional),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    /*此处校验161002是因为当前框架通过桩函数调用ut，导致无法正常infershape，正常现象*/
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend910D_grouped_matmul_a8w8ofp16_weightNz_static)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, K, N}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, 1}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 10));
    auto offsetOptionsl = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1,TensorDesc({M,N}, ACL_FLOAT16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(
        aclnnGroupedMatmulWeightNz,
        INPUT(x, weight, biasOptional, scaleOptional, offsetOptionsl, antiquantScaleOptional, antiquantOffsetOptional,
              perTokenScaleOptional, groupListOptional, activationInputOptional, activationQuantScaleOptional,
              activationQuantOffsetOptional, splitItem, groupType, groupListType, actType, tuningConfigOptional, 0),
        OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend910D_grouped_matmul_a8w8ofp16_weightNz_pertoken)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptionsl = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1,TensorDesc({M,N}, ACL_FLOAT16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(
        aclnnGroupedMatmulWeightNz,
        INPUT(x, weight, biasOptional, scaleOptional, offsetOptionsl, antiquantScaleOptional, antiquantOffsetOptional,
              perTokenScaleOptional, groupListOptional, activationInputOptional, activationQuantScaleOptional,
              activationQuantOffsetOptional, splitItem, groupType, groupListType, actType, tuningConfigOptional, 0),
        OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}
