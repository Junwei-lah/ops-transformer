/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_kernel/attention_to_ffn_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace AttentionToFFNUT {

class AttentionToFFNTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "AttentionToFFNTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "AttentionToFFNTiling TearDown" << std::endl;
    }
};

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingNoQuantSyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
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
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 0UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingNoQuantAsyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 2UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingQuantSyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 1UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingQuantAsyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 3UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingNoQuantSyncActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
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
        },
        &compileInfo, "Ascend910_93"
    );
    uint64_t expectTilingKey = 4UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingNoQuantAsyncActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 6UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingQuantSyncActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 5UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingQuantAsyncActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 7UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingMicroBatchNumInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{2, 16, 7168}, {2, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
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
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingBsInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 513, 7168}, {1, 513, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
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
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingKInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 16}, {1, 16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
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
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingHInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 8196}, {1, 16, 8196}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
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
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingLayerNumInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 9, 4}, {2, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
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
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingMoeExpertNumInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1026, 4}, {1, 1026, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1025)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingFfnTokenInfoTableShapeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingFfnTokenDataShapeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1025)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingAttnTokenInfoTableShapeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingXDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168, 1}, {1, 16, 7168, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
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
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingSessionIdDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
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
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingMicroBatchIdDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
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
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingLayerIdDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
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
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingExpertIdsDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
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
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingExpertRankTableDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{9, 4}, {9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
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
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingScalesDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168, 1}, {1, 9, 7168, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingActiveMaskDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingXDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingSessionIdDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingMicroBatchIdDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingLayerIdDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingExpertIdsDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingExpertRankTableDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingScalesDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingActiveMaskDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_INT8, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingXFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingSessionIdFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingMicroBatchIdFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingLayerIdFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingExpertIdsFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingExpertRankTableDormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingScalesFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNTiling, AttentionToFfnTilingActiveMaskFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_FRACTAL_NZ}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

} // AttentionToFFNUT