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
#include "mc2_tiling_case_executor.h"

namespace MoeDistributeDispatchUT {

class MoeDistributeDispatchTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MoeDistributeDispatchTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MoeDistributeDispatchTiling TearDown" << std::endl;
    }
};

//normal: share rank
TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTiling0)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// normal: share rank
TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTiling1)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{16, 7160}, {16, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16*8}, {16*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTiling2)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{16, 7160}, {16, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16*8}, {16*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTiling3)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{16, 7160}, {16, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16*8}, {16*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(31)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTiling4)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{16, 7168}, {16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{33, 7168}, {33, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16*8}, {16*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(257)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(31)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTiling5)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{16, 7168}, {16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16*8}, {16*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTiling6)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{7*8}, {7*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 1024UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTiling7)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTiling8)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTiling9)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{16, 7168}, {16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16*8}, {16*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTiling10)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// 背景：4机 每机器8卡  每卡8个专家  由于deepseek 总共256 专家 分层
// 可选传入默认值   除了quant_mode = 0
// 只改变环境变量HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE
TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTilingA2Quant0Layered)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8*8*32, 7168}, {8*8*32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8*8*32}, {8*8*32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8*8}, {8*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8*32}, {8*32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}
TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTilingA2Quant0)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8*8*32, 7168}, {8*8*32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8*8*32}, {8*8*32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8*8}, {8*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8*32}, {8*32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}
// A2 test
// 背景：4机 每机器8卡  每卡8个专家  由于deepseek 总共256 专家
// 可选传入默认值   除了quant_mode = 0   
TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTilingA2GlobalBs)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8*8*32, 7168}, {8*8*32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8*8*32}, {8*8*32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8*8}, {8*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8*32}, {8*32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}
// 背景：4机 每机器8卡  每卡8个专家  由于deepseek 总共256 专家
// 可选传入默认值   除了 quant_mode = 2 global_bs = 512  正常
TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTilingA2ShapeAndEpRankId)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{8, 7160}, {8, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8*8*32, 7168}, {8*8*32, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{8*8*32}, {8*8*32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8*8}, {8*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8*32}, {8*32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(33)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}


//背景：4机 每机器8卡  每卡8个专家  由于deepseek 总共256 专家
//可选传入默认值   除了ep_rank_id = 33  quant_mode = 2 同时x的shape{16,7160} 异常
TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTilingA2MoeExpertNum)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8*8*32, 7168}, {8*8*32, 7168}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{8*8*32}, {8*8*32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8*8}, {8*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8*32}, {8*32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(257)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

//背景：4机 每机器8卡  每卡8个专家  由于deepseek 总共256 专家
// 可选传入默认值   除了 quant_mode = 2 同时moe_expert_num = 257  异常
TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTilingepWorldSize384)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(384)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}


TEST_F(MoeDistributeDispatchTiling, MoeDistributeDispatchTestTilingepWorldSize72)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32*8}, {32*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(72)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(216)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(18)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

} //moe_distribute_dispatch_ut