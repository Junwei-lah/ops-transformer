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
#include "../../../op_api/aclnn_quant_grouped_mat_mul_allto_allv.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "test_aclnn_quant_grouped_mat_mul_allto_all SetUp" << endl;
    }


    static void TearDownTestCase()
    {
        cout << "test_aclnn_quant_grouped_mat_mul_allto_all TearDown" << endl;
    }
};

struct QuantGroupedMatmulAlltoAllvAclnnTestParam {
    // 用例名
    string case_name;
    // gmmX
    vector<int64_t> gmmX;
    aclDataType gmmX_dtype;
    aclFormat gmmX_format;

    // gmmWeight
    vector<int64_t> gmmWeight;
    aclDataType gmmWeight_dtype;
    aclFormat gmmWeight_format;

    // mm
    vector<int64_t> mmX;
    aclDataType mmX_dtype;
    aclFormat mmX_format;

    // mmweight
    vector<int64_t> mmWeight;
    aclDataType mmWeight_dtype;
    aclFormat mmWeight_format;

    // y
    vector<int64_t> y;
    aclDataType y_dtype;
    aclFormat y_format;

    // mmY
    vector<int64_t> mmYOptional;
    aclDataType mmYOptional_dtype;
    aclFormat mmYOptional_format;

    char *group;
    bool send;
    bool recv;
    bool transGmmWeight;
    bool transMmWeight;
    aclnnStatus aclnn_status;
};

static QuantGroupedMatmulAlltoAllvAclnnTestParam quant_cases_params[] = {
// float16 正常用例
    {"test_quant_allto_allv_grouped_mat_mul_test_float16_00",
            {4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND,
            {4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16,  ACL_FORMAT_ND,
            {4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            "test_grouped_mat_mul_allto_allv_ep_group", 
            false, false, false, false, ACLNN_SUCCESS},

// 异常 sendCounts null
    {"test_quant_allto_allv_grouped_mat_mul_test_float16_01",
            {4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND,
            {4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16,  ACL_FORMAT_ND,
            {4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            "test_grouped_mat_mul_allto_allv_ep_group", 
            true, false, false, false, ACLNN_ERR_PARAM_INVALID},

// recvCounts null
    {"test_quant_allto_allv_grouped_mat_mul_test_float16_02",
            {4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND,
            {4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16,  ACL_FORMAT_ND,
            {4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            "test_grouped_mat_mul_allto_allv_ep_group", 
            false, true, false, false, ACLNN_ERR_PARAM_INVALID},

// gmmx null
    {"test_quant_allto_allv_grouped_mat_mul_test_float16_03",
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            {4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16,  ACL_FORMAT_ND,
            {4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            "test_grouped_mat_mul_allto_allv_ep_group", 
            false, false, false, false, ACLNN_ERR_PARAM_INVALID},

// gmmWeight null
    {"test_quant_allto_allv_grouped_mat_mul_test_float16_04",
            {4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16,  ACL_FORMAT_ND,
            {4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            "test_grouped_mat_mul_allto_allv_ep_group", 
            false, false, false, false, ACLNN_ERR_PARAM_INVALID},

// y null
    {"test_quant_allto_allv_grouped_mat_mul_test_float16_05",
            {4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND,
            {4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16,  ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            "test_grouped_mat_mul_allto_allv_ep_group", 
            false, false, false, false, ACLNN_ERR_PARAM_INVALID},
// group ep null
    {"test_quant_allto_allv_grouped_mat_mul_test_float16_06",
            {4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND,
            {4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16,  ACL_FORMAT_ND,
            {4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            "", 
            false, false, false, false, ACLNN_ERR_PARAM_NULLPTR},

// group ep invalid
    {"test_quant_allto_allv_grouped_mat_mul_test_float16_07",
            {4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND,
            {4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16,  ACL_FORMAT_ND,
            {4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            "test_grouped_mat_mul_allto_allv_ep_group_"
            "test_grouped_mat_mul_allto_allv_ep_group_"
            "test_grouped_mat_mul_allto_allv_ep_group_"
            "test_grouped_mat_mul_allto_allv_ep_group", 
            false, false, false, false, ACLNN_ERR_PARAM_INVALID},

// mmx not_null mmweight null mmy null
    {"test_quant_allto_allv_grouped_mat_mul_test_float16_08",
            {4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND,
            {4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {1024, 7168}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16,  ACL_FORMAT_ND,
            {4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            "test_grouped_mat_mul_allto_allv_ep_group", 
            false, false, false, false, ACLNN_ERR_PARAM_INVALID},

// mmx null mmweight not_null mmy null
    {"test_quant_allto_allv_grouped_mat_mul_test_float16_09",
            {4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND,
            {4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            {7168, 1024}, ACL_FLOAT16,  ACL_FORMAT_ND,
            {4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            "test_grouped_mat_mul_allto_allv_ep_group", 
            false, false, false, false, ACLNN_ERR_PARAM_INVALID},

// mmx null mmweight null mmy not_null
    {"test_quant_allto_allv_grouped_mat_mul_test_float16_10",
            {4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND,
            {4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16, ACL_FORMAT_ND,
            {}, ACL_FLOAT16,  ACL_FORMAT_ND,
            {4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
            {1024}, ACL_FLOAT16, ACL_FORMAT_ND,
            "test_grouped_mat_mul_allto_allv_ep_group", 
            false, false, false, false, ACLNN_ERR_PARAM_INVALID},
};

static void TestQuantParamCase(const QuantGroupedMatmulAlltoAllvAclnnTestParam &param)
{
    std::cout << "run case " << param.case_name << std::endl;
    TensorDesc gmmX_ = TensorDesc(param.gmmX, param.gmmX_dtype, param.gmmX_format);
    TensorDesc gmmWeight_ = TensorDesc(param.gmmWeight, param.gmmWeight_dtype, param.gmmWeight_format);
    TensorDesc mmX_ = TensorDesc(param.mmX, param.mmX_dtype, param.mmX_format);
    TensorDesc mmWeight_ = TensorDesc(param.mmWeight, param.mmWeight_dtype, param.mmWeight_format);
    TensorDesc y_ = TensorDesc(param.y, param.y_dtype, param.y_format);
    TensorDesc mmY_ = TensorDesc(param.mmYOptional, param.mmYOptional_dtype, param.mmYOptional_format);

    const char *group = param.group;
    bool send_ = param.send;
    bool recv_ = param.recv;
    bool transGmmWeight = param.transGmmWeight;
    bool transMmWeight = param.transMmWeight;

    aclnnStatus retStatus = param.aclnn_status;

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t H = 7168;
    constexpr int64_t e = 4;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    if (send_)
    sendCounts = nullptr;
    if (recv_)
    recvCounts = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_, 
                        nullptr, nullptr,       // scale
                        nullptr, nullptr,       // counts tensor
                        mmX_, mmWeight_,            
                        nullptr, nullptr,       // mm scale
                        nullptr,                // commQuantScale
                        0, 0, 0, 0, 0,          // quantmode
                        -1, 0, group, epWorldSize, sendCounts,
                        recvCounts, transGmmWeight, transMmWeight),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, retStatus);
    std::cout << "end case " << param.case_name << std::endl;
}

TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, DISABLED_quant_cases_params)
{
    if (std::size(quant_cases_params) != 0) {
    uint64_t numCases = sizeof(quant_cases_params) / sizeof(quant_cases_params[0]);
    for (size_t idx = 0; idx < numCases; idx += 1) {
        TestQuantParamCase(quant_cases_params[idx]);
        }
    }
}

// ============================================================================
// Group 1: QuantMode/Scale 一致性 (CheckQuantMode)
// ============================================================================

// gmmXQuantMode=0 但 gmmXScale 非空 → PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_gmmX_quantmode0_with_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        gmmXScale_, nullptr,     // gmmXScale 非空, gmmWeightScale 空
                        nullptr, nullptr,        // counts tensor
                        mmX_, mmWeight_,
                        nullptr, nullptr,        // mm scale
                        nullptr,                 // commQuantScale
                        0, 0, 0, 0, 0,           // gmmXQM=0, 其余=0
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gmmXQuantMode=1 但 gmmXScale=nullptr → PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_gmmX_quantmode1_without_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,        // gmmXScale=nullptr
                        nullptr, nullptr,        // counts tensor
                        mmX_, mmWeight_,
                        nullptr, nullptr,        // mm scale
                        nullptr,                 // commQuantScale
                        1, 0, 0, 0, 0,           // gmmXQM=1, 其余=0
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gmmWeightQuantMode=0 但 gmmWeightScale 非空 → PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_gmmWeight_quantmode0_with_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, gmmWeightScale_, // gmmWeightScale 非空
                        nullptr, nullptr,         // counts tensor
                        mmX_, mmWeight_,
                        nullptr, nullptr,         // mm scale
                        nullptr,                  // commQuantScale
                        0, 0, 0, 0, 0,            // gmmWQM=0
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gmmWeightQuantMode=1 但 gmmWeightScale=nullptr → PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_gmmWeight_quantmode1_without_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,        // gmmWeightScale=nullptr
                        nullptr, nullptr,        // counts tensor
                        mmX_, mmWeight_,
                        nullptr, nullptr,        // mm scale
                        nullptr,                 // commQuantScale
                        0, 1, 0, 0, 0,           // gmmWQM=1
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mmXQuantMode=0 但 mmXScale 非空（mm组全非空） → PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_mmX_quantmode0_with_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({1024, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmXScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,        // gmm scale
                        nullptr, nullptr,        // counts tensor
                        mmX_, mmWeight_,
                        mmXScale_, nullptr,       // mmXScale 非空
                        nullptr,                 // commQuantScale
                        0, 0, 0, 0, 0,           // mmXQM=0
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mmXQuantMode=1 但 mmXScale=nullptr（mm组全非空） → PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_mmX_quantmode1_without_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({1024, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,        // gmm scale
                        nullptr, nullptr,        // counts tensor
                        mmX_, mmWeight_,
                        nullptr, nullptr,        // mmXScale=nullptr
                        nullptr,                 // commQuantScale
                        0, 0, 1, 0, 0,           // mmXQM=1
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mmWeightQuantMode=0 但 mmWeightScale 非空 → PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_mmWeight_quantmode0_with_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({1024, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,        // gmm scale
                        nullptr, nullptr,        // counts tensor
                        mmX_, mmWeight_,
                        nullptr, mmWeightScale_, // mmWeightScale 非空
                        nullptr,                 // commQuantScale
                        0, 0, 0, 0, 0,           // mmWQM=0
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mmWeightQuantMode=1 但 mmWeightScale=nullptr → PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_mmWeight_quantmode1_without_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({1024, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,        // gmm scale
                        nullptr, nullptr,        // counts tensor
                        mmX_, mmWeight_,
                        nullptr, nullptr,        // mmWeightScale=nullptr
                        nullptr,                 // commQuantScale
                        0, 0, 0, 1, 0,           // mmWQM=1
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// Group 3: CountsTensor 不支持 (CheckNullStatus)
// ============================================================================

// sendCountsTensorOptional 非空 → PARAM_NULLPTR
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_sendCountsTensor_not_null)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sendCountsTensor_({32}, ACL_INT64, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,        // scale
                        sendCountsTensor_, nullptr, // sendCountsTensor 非空
                        mmX_, mmWeight_,
                        nullptr, nullptr,        // mm scale
                        nullptr,                 // commQuantScale
                        0, 0, 0, 0, 0,
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// recvCountsTensorOptional 非空 → PARAM_NULLPTR
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_recvCountsTensor_not_null)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc recvCountsTensor_({32}, ACL_INT64, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,        // scale
                        nullptr, recvCountsTensor_, // recvCountsTensor 非空
                        mmX_, mmWeight_,
                        nullptr, nullptr,        // mm scale
                        nullptr,                 // commQuantScale
                        0, 0, 0, 0, 0,
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Group 4: SendCounts/RecvCounts 边界 (CheckSendAndRecv)
// ============================================================================

// sendCounts size=0 → PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_sendCounts_empty_array)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    aclIntArray *sendCounts = aclCreateIntArray(nullptr, 0);
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,        // scale
                        nullptr, nullptr,        // counts tensor
                        mmX_, mmWeight_,
                        nullptr, nullptr,        // mm scale
                        nullptr,                 // commQuantScale
                        0, 0, 0, 0, 0,
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// recvCounts size=0 → PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_recvCounts_empty_array)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(nullptr, 0);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,        // scale
                        nullptr, nullptr,        // counts tensor
                        mmX_, mmWeight_,
                        nullptr, nullptr,        // mm scale
                        nullptr,                 // commQuantScale
                        0, 0, 0, 0, 0,
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// sendCounts/recvCounts 全 0（MoE 负载不均衡场景）→ SUCCESS
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_sendRecvCounts_all_zero)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    std::vector<int64_t> sendCountsList(epWorldSize * e, 0);
    std::vector<int64_t> recvCountsList(epWorldSize * e, 0);
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,        // scale
                        nullptr, nullptr,        // counts tensor
                        mmX_, mmWeight_,
                        nullptr, nullptr,        // mm scale
                        nullptr,                 // commQuantScale
                        0, 0, 0, 0, 0,
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// ============================================================================
// Group 5: Group nullptr
// ============================================================================

// group=nullptr → PARAM_NULLPTR
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_group_nullptr)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,        // scale
                        nullptr, nullptr,        // counts tensor
                        mmX_, mmWeight_,
                        nullptr, nullptr,        // mm scale
                        nullptr,                 // commQuantScale
                        0, 0, 0, 0, 0,
                        -1, 0,
                        (const char*)nullptr,    // group=nullptr
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Group 6: 功能正确性
// ============================================================================

// 部分 rank token=0（混合 sendCounts/recvCounts）→ SUCCESS
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_sendRecvCounts_partial_zero)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    // 部分为 0，部分非 0
    std::vector<int64_t> sendCountsList = {256, 0, 128, 0, 256, 0, 128, 0,
                                           256, 0, 128, 0, 256, 0, 128, 0,
                                           256, 0, 128, 0, 256, 0, 128, 0,
                                           256, 0, 128, 0, 256, 0, 128, 0};
    std::vector<int64_t> recvCountsList = {0, 256, 0, 128, 0, 256, 0, 128,
                                           0, 256, 0, 128, 0, 256, 0, 128,
                                           0, 256, 0, 128, 0, 256, 0, 128,
                                           0, 256, 0, 128, 0, 256, 0, 128};
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,        // scale
                        nullptr, nullptr,        // counts tensor
                        mmX_, mmWeight_,
                        nullptr, nullptr,        // mm scale
                        nullptr,                 // commQuantScale
                        0, 0, 0, 0, 0,
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// commQuantMode=1 → PARAM_INVALID（API 层新增校验，仅支持 0）
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_commQuantMode_nonzero)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,        // scale
                        nullptr, nullptr,        // counts tensor
                        mmX_, mmWeight_,
                        nullptr, nullptr,        // mm scale
                        nullptr,                 // commQuantScale
                        0, 0, 0, 0, 1,           // commQuantMode=1
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// TT 量化正常路径：gmmXQM=1, gmmWQM=1, 提供 scale → 实测确认
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_tt_quant_normal)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        gmmXScale_, gmmWeightScale_, // TT: 双 scale 非空
                        nullptr, nullptr,            // counts tensor
                        mmX_, mmWeight_,
                        nullptr, nullptr,            // mm scale
                        nullptr,                     // commQuantScale
                        1, 1, 0, 0, 0,               // gmmXQM=1, gmmWQM=1
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    // TT 量化正常路径，CheckParams 应通过
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Group 7: CheckNotEmptyTensor 边界用例（源码修复后补充）
// ============================================================================

// gmmX dim1=0 → PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_gmmX_empty_dim1)
{
    TensorDesc gmmX_({4096, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,
                        nullptr, nullptr,
                        mmX_, mmWeight_,
                        nullptr, nullptr,
                        nullptr,
                        0, 0, 0, 0, 0,
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gmmWeight dim1=0 → PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_gmmWeight_empty_dim1)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 0, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,
                        nullptr, nullptr,
                        mmX_, mmWeight_,
                        nullptr, nullptr,
                        nullptr,
                        0, 0, 0, 0, 0,
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gmmWeight dim2=0 → PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_gmmWeight_empty_dim2)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,
                        nullptr, nullptr,
                        mmX_, mmWeight_,
                        nullptr, nullptr,
                        nullptr,
                        0, 0, 0, 0, 0,
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// y dim0=0 → PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_y_empty_dim0)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({0, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,
                        nullptr, nullptr,
                        mmX_, mmWeight_,
                        nullptr, nullptr,
                        nullptr,
                        0, 0, 0, 0, 0,
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// y dim1=0 → PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_y_empty_dim1)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,
                        nullptr, nullptr,
                        mmX_, mmWeight_,
                        nullptr, nullptr,
                        nullptr,
                        0, 0, 0, 0, 0,
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mm optional 维度不一致（部分为零部分非零）→ PARAM_INVALID
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_mm_empty_inconsistent)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({1024, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,
                        nullptr, nullptr,
                        mmX_, mmWeight_,
                        nullptr, nullptr,
                        nullptr,
                        0, 0, 0, 0, 0,
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// commQuantMode=1 → PARAM_INVALID（新增校验专用用例）
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_commQuantMode_reject)
{
    TensorDesc gmmX_({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,
                        nullptr, nullptr,
                        mmX_, mmWeight_,
                        nullptr, nullptr,
                        nullptr,
                        0, 0, 0, 0, 1,           // commQuantMode=1
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gmmX dim0=0（MoE token=0 场景）→ SUCCESS（修改后放行）
TEST_F(DISABLED_test_aclnn_quant_grouped_mat_mul_allto_all, test_gmmX_dim0_zero)
{
    TensorDesc gmmX_({0, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllv,
                        INPUT(gmmX_, gmmWeight_,
                        nullptr, nullptr,
                        nullptr, nullptr,
                        mmX_, mmWeight_,
                        nullptr, nullptr,
                        nullptr,
                        0, 0, 0, 0, 0,
                        -1, 0,
                        "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts,
                        recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    // gmmX dim0=0 is now allowed (MoE token=0 scenario), should pass CheckParams
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}