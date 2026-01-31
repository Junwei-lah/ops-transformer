/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cfloat>

#include <array>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_ffn_to_attention.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "platform/platform_info.h"

using namespace op;
using namespace std;

namespace FFNToAttentionUT {

class aclnn_ffn_to_attention_test : public testing::Test {
 protected:
  static void SetUpTestCase()
  {
      op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
      cout << "aclnn_ffn_to_attention_test SetUp" << endl;
  }

  static void TearDownTestCase()
  {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    cout << "aclnn_ffn_to_attention_test TearDown" << endl;
  }
};

TEST_F(aclnn_ffn_to_attention_test, test_ffn_to_attention_attnRank_table) {
  TensorDesc x = TensorDesc({1584, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc sessionIds = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc microBatchIds = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc tokenIds = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc expertOffsets = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc actualTokenNum = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  TensorDesc attnRankTable = TensorDesc({11}, ACL_INT32, ACL_FORMAT_ND);
  int64_t worldSize = 16;
  const char* group = "test_ffn_to_attention_group";
  std::vector<int64_t> tokenInfoTable = {1, 16, 9};
  std::vector<int64_t> tokenData = {1, 16, 9, 7168};
  aclIntArray* tokenInfoTableShape = aclCreateIntArray(tokenInfoTable.data(), tokenInfoTable.size());
  aclIntArray* tokenDataShape = aclCreateIntArray(tokenData.data(), tokenData.size());

  auto ut = OP_API_UT(aclnnFFNToAttention, INPUT(x, sessionIds, microBatchIds, tokenIds, expertOffsets,
  actualTokenNum, attnRankTable, group, worldSize, tokenInfoTableShape, tokenDataShape),
  OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(aclnn_ffn_to_attention_test, test_ffn_to_attention_no_attnRank_table) {
  TensorDesc x = TensorDesc({1584, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc sessionIds = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc microBatchIds = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc tokenIds = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc expertOffsets = TensorDesc({1584}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc actualTokenNum = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  aclTensor* attnRankTable = nullptr;
  int64_t worldSize = 16;
  const char* group = "test_ffn_to_attention_group";
  std::vector<int64_t> tokenInfoTable = {1, 16, 9};
  std::vector<int64_t> tokenData = {1, 16, 9, 7168};
  aclIntArray* tokenInfoTableShape = aclCreateIntArray(tokenInfoTable.data(), tokenInfoTable.size());
  aclIntArray* tokenDataShape = aclCreateIntArray(tokenData.data(), tokenData.size());

  auto ut = OP_API_UT(aclnnFFNToAttention, INPUT(x, sessionIds, microBatchIds, tokenIds, expertOffsets,
  actualTokenNum, attnRankTable, group, worldSize, tokenInfoTableShape, tokenDataShape),
  OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

} // FFNToAttentionUT