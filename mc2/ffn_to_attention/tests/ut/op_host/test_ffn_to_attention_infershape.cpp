/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace FFNToAttentionUT {

class FFNToAttentionInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "FFNToAttentionInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "FFNToAttentionInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(FFNToAttentionInferShapeTest, basic) {
    gert::StorageShape x_shape = {{1584, 7168}, {}};
    gert::StorageShape session_ids_shape = {{1584}, {}};
    gert::StorageShape micro_batch_ids_shape = {{1584}, {}};
    gert::StorageShape token_ids_shape = {{1584}, {}};
    gert::StorageShape expert_offsets_shape = {{1584}, {}};
    gert::StorageShape actual_token_num_shape = {{1}, {}};
    gert::StorageShape attn_rank_table_shape = {{11}, {}};

    gert::InfershapeContextPara infershapeContextPara("FFNToAttention",
        {
            {x_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {session_ids_shape, ge::DT_INT32, ge::FORMAT_ND},
            {micro_batch_ids_shape, ge::DT_INT32, ge::FORMAT_ND},
            {token_ids_shape, ge::DT_INT32, ge::FORMAT_ND},
            {expert_offsets_shape, ge::DT_INT32, ge::FORMAT_ND},
            {actual_token_num_shape, ge::DT_INT64, ge::FORMAT_ND},
            {attn_rank_table_shape, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9, 7168})}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

} // FFNToAttentionUT