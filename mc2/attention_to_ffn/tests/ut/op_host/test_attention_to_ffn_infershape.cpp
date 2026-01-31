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

namespace AttentionToFFNUT {

class AttentionToFFNInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "AttentionToFFNInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "AttentionToFFNInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(AttentionToFFNInferShapeTest, basic) {
    gert::StorageShape x_shape = {{1, 16, 7168}, {}};
    gert::StorageShape session_id_shape = {{1}, {}};
    gert::StorageShape micro_batch_id_shape = {{1}, {}};
    gert::StorageShape layer_id_shape = {{1}, {}};
    gert::StorageShape expert_ids_shape = {{1, 16, 8}, {}};
    gert::StorageShape expert_rank_table_shape = {{1, 9, 4}, {}};
    gert::StorageShape scales_shape = {{9, 7168}, {}};
    gert::StorageShape active_mask_shape = {{1, 16}, {}};

    gert::InfershapeContextPara infershapeContextPara("AttentionToFFN",
        {
            {x_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {session_id_shape, ge::DT_INT32, ge::FORMAT_ND},
            {micro_batch_id_shape, ge::DT_INT32, ge::FORMAT_ND},
            {layer_id_shape, ge::DT_INT32, ge::FORMAT_ND},
            {expert_ids_shape, ge::DT_INT32, ge::FORMAT_ND},
            {expert_rank_table_shape, ge::DT_INT32, ge::FORMAT_ND},
            {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {active_mask_shape, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

} // AttentionToFFNUT