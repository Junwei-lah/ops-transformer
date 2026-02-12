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
#include "../../../op_host/rotary_position_embedding_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class RotaryPositionEmbeddingTiling : public testing::Test {
public:
    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";

protected:
    static void SetUpTestCase()
    {
        std::cout << "RotaryPositionEmbeddingTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RotaryPositionEmbeddingTiling TearDown" << std::endl;
    }
};

TEST_F(RotaryPositionEmbeddingTiling, RotaryPositionEmbedding_fp16_001)
{
    optiling::RotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbedding",
                                              {
                                                  // input info
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1032;
    string expectTilingData = "3 8192 1 2 64 32 32 64 1 64 84 5376 5376 64 1 0 1 128 64 8192 4096 64 64 0 1 0 1 128 64 "
                              "64 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RotaryPositionEmbeddingTiling, RotaryPositionEmbedding_bf16_001)
{
    optiling::RotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbedding",
                                              {
                                                  // input info
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1033;
    string expectTilingData = "3 8192 1 2 64 32 32 64 1 64 84 5376 5376 64 1 0 1 128 64 8192 4096 64 64 0 1 0 1 128 64 "
                              "64 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RotaryPositionEmbeddingTiling, RotaryPositionEmbedding_fp32_001)
{
    optiling::RotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbedding",
                                              {
                                                  // input info
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1031;
    string expectTilingData = "3 8192 1 2 64 32 32 64 1 64 126 8064 8064 64 1 0 1 128 64 8192 4096 64 64 0 1 0 1 128 "
                              "64 64 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RotaryPositionEmbeddingTiling, RotaryPositionEmbedding_fp32_001_tnd)
{
    optiling::RotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbedding",
                                              {
                                                  // input info
                                                  {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 1, 64}, {64, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 1, 64}, {64, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1031;
    string expectTilingData = "3 8192 1 2 64 32 32 64 1 64 126 8064 8064 64 1 0 1 128 64 8192 4096 64 64 0 1 0 1 128 "
                              "64 64 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
