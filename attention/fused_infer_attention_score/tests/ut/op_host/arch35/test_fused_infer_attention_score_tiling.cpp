/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/fused_infer_attention_score_tiling_compile_info.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class FusedInferAttentionScoreTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedInferAttentionScoreTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedInferAttentionScoreTiling TearDown" << std::endl;
    }
};

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScore_950_tiling_0)     // learnableSink
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{4, 16, 3, 1}, {4, 16, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476483184f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(156)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sinks_type", Ops::Transformer::AnyValue::CreateFrom<float>(16)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "ASCEND950", 64, 262144, 16384);
    int64_t expectTilingKey = 4294967295;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScore_950_tiling_1)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{4, 13, 10347}, {4, 13, 10347}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{4, 13, 16, 64}, {4, 13, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key_rope_antiquant_scale-input23
            {{{4, 10347, 1, 64}, {4, 10347, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                       // 输出Tensor
         {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5128)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266602241;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScore_950_tiling_2)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{4, 16, 3, 1}, {4, 16, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476483184f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(156)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266600704;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScore_950_tiling_3)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{32, 1, 1024}, {32, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{32, 4096, 128}, {32, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{32, 4096, 128}, {32, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                              // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                            // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_padding_size-input14
            {{{1, 32, 4096}, {1, 32, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // key_antiquant_offset-input16
            {{{1, 32, 4096}, {1, 32, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{32, 1, 1024}, {32, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{32, 8, 1, 1}, {32, 8, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},  // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382977;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScore_950_tiling_4)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{8, 32, 3, 512}, {8, 32, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{8, 1, 256, 512}, {8, 1, 256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{8, 1, 256, 512}, {8, 1, 256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{8, 12}, {8, 12}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{10, 1, 128, 512}, {10, 1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{10, 1, 128, 512}, {10, 1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{8, 32, 3, 64}, {8, 32, 3, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{8, 1, 256, 64}, {8, 1, 256, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{10, 1, 128, 64}, {10, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{8, 32, 3, 512}, {8, 32, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            //  {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382977;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScore_950_tiling_5)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 1, 8, 128}, {4, 1, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // query-input0
            {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{4, 8, 1, 2048}, {4, 8, 1, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // pse_shift-input3
            {{{4, 1, 1, 2048}, {4, 1, 1, 2048}}, ge::DT_INT8, ge::FORMAT_ND},        // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                     // query_padding_size-input13
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                   // kv_padding_size-input14
            {{{1, 1, 128}, {1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // key_antiquant_offset-input16
            {{{1, 1, 128}, {1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                     // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                     // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                     // kv_start_idx-input27
        },
        {                                                                   // 输出Tensor
         {{{4, 1, 8, 128}, {4, 1, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{4, 8, 1, 1}, {4, 8, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382977;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScore_950_tiling_6)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 1, 8, 128}, {4, 1, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT64, ge::FORMAT_ND}, // key-input1
            {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT64, ge::FORMAT_ND}, // value-input2
            {{{4, 8, 1, 2048}, {4, 8, 1, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // pse_shift-input3
            {{{4, 1, 1, 2048}, {4, 1, 1, 2048}}, ge::DT_INT8, ge::FORMAT_ND},      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_padding_size-input14
            {{{1, 1, 128}, {1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{1, 1, 128}, {1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                   // 输出Tensor
         {{{4, 1, 8, 128}, {4, 1, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{4, 8, 1, 1}, {4, 8, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382977;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScore_950_tiling_7)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{3, 1, 16, 128}, {3, 1, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // query-input0
            {{{3, 55648, 8, 128}, {3, 55648, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{3, 55648, 8, 128}, {3, 55648, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_INT8, ge::FORMAT_ND},          // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{3, 1, 16, 64}, {3, 1, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // key_rope_antiquant_scale-input23
            {{{3, 55648, 8, 64}, {3, 55648, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{3, 16, 1, 128}, {3, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687836487032f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(55648)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND_BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266601217;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScore_950_tiling_8)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{5, 1, 512}, {5, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0
            {{{5, 768, 512}, {5, 768, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{5, 768, 512}, {5, 768, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{5, 1, 64}, {5, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{5, 768, 64}, {5, 768, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{5, 1, 512}, {5, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                        // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132385025;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScore_950_tiling_9)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{12, 64, 512}, {12, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0
            {{{32, 1, 1020, 512}, {32, 1, 1020, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{32, 1, 1020, 512}, {32, 1, 1020, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{286, 1, 128, 512}, {286, 1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{286, 1, 128, 512}, {286, 1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{32, 1, 1020, 64}, {32, 1, 1020, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{286, 1, 128, 64}, {286, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{12, 64, 512}, {12, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                        // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417382415922f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1020)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScore_950_tiling_10)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{12, 64, 512}, {12, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{3, 6, 9, 12}, {3, 6, 9, 12}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{1024, 2048, 3072, 4096}, {1024, 2048, 3072, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{4096, 1, 64}, {4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{64, 12, 512}, {64, 12, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                        // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666666664f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND_NTD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132385026;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScore_950_tiling_11)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{12, 64, 512}, {12, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{3, 6, 9, 12}, {3, 6, 9, 12}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{1024, 2048, 3072, 4096}, {1024, 2048, 3072, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{4096, 1, 64}, {4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{12, 64, 512}, {12, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                        // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666666664f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132385026;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScore_950_tiling_12)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{5, 6, 128}, {5, 6, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0
            {{{5, 768, 128}, {5, 768, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{5, 768, 128}, {5, 768, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{2, 4, 6}, {2, 4, 6}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{256, 512, 768}, {256, 512, 768}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{5, 6, 128}, {5, 6, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                        // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("NTD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScore_950_tiling_13)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{5, 6, 128}, {5, 6, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0
            {{{5, 768, 128}, {5, 768, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{5, 768, 128}, {5, 768, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{2, 4, 6}, {2, 4, 6}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{256, 512, 768}, {256, 512, 768}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{5, 6, 64}, {5, 6, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{5, 768, 64}, {5, 768, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{5, 6, 128}, {5, 6, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                        // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("NTD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 400819204;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check1)
{
    // GQA非量化场景n1>256
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 257, 127}, {4, 13, 257, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 127}, {4, 10347, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 1, 127}, {4, 10347, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 13, 257, 127}, {4, 13, 257, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(257)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check2)
{
    // decode mla场景不支持BNSD_BSND
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{4, 13, 16, 64}, {4, 13, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{4, 10347, 1, 64}, {4, 10347, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 16, 13, 512}, {4, 16, 13, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD_BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check3)
{
    // prefill mla场景不支持BSND_NBSD
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 16, 128}, {4, 13, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 128}, {4, 10347, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 1, 128}, {4, 10347, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{4, 13, 16, 64}, {4, 13, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{4, 10347, 1, 64}, {4, 10347, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 13, 16, 128}, {4, 13, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND_NBSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check4)
{
    // GQA非量化场景不支持innerPrecise = 6
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 16, 127}, {4, 13, 16, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 127}, {4, 10347, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 1, 127}, {4, 10347, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 13, 16, 127}, {4, 13, 16, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check5)
{
    // GQA非量化 PA(BBH)场景keyBlockNum != valueBlockNum
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_kvlist[] = {88, 88, 88, 88};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{30, 128, 256}, {30, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{32, 128, 256}, {32, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actual_seq_kvlist},                                 // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{4, 16}, {4, 16}}, ge::DT_INT32, ge::FORMAT_ND},                        // block_table-input12
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                          // 输出Tensor
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check6)
{
    // GQA非量化 PA(BND1BD0)场景keyBlockNum != valueBlockNum
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_kvlist[] = {88, 88, 88, 88};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{30, 2, 8, 128, 16}, {30, 2, 8, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{32, 2, 8, 128, 16}, {32, 2, 8, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actual_seq_kvlist},                                 // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{4, 16}, {4, 16}}, ge::DT_INT32, ge::FORMAT_ND},                        // block_table-input12
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                          // 输出Tensor
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check7)
{
    // GQA非量化 PA(BND1BD0)场景D0 != 16
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_kvlist[] = {88, 88, 88, 88};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{30, 2, 8, 128, 15}, {30, 2, 8, 128, 15}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{30, 2, 8, 128, 15}, {30, 2, 8, 128, 15}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actual_seq_kvlist},                                 // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{4, 16}, {4, 16}}, ge::DT_INT32, ge::FORMAT_ND},                        // block_table-input12
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                          // 输出Tensor
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check8)
{
    // GQA非量化 PA(BND1BD0)场景D1 * 16 != D
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_kvlist[] = {88, 88, 88, 88};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{30, 2, 8, 128, 16}, {30, 2, 8, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{30, 2, 7, 128, 16}, {30, 2, 7, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actual_seq_kvlist},                                 // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{4, 16}, {4, 16}}, ge::DT_INT32, ge::FORMAT_ND},                        // block_table-input12
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                          // 输出Tensor
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check9)
{
    // GQA非量化 PA(BNBD)场景keyBlockNum != valueBlockNum
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_kvlist[] = {88, 88, 88, 88};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{30, 2, 128, 128}, {30, 2, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{32, 2, 128, 128}, {32, 2, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actual_seq_kvlist},                                 // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{4, 16}, {4, 16}}, ge::DT_INT32, ge::FORMAT_ND},                        // block_table-input12
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                          // 输出Tensor
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check10)
{
    // GQA非量化 PA(BNBD)场景不支持BSND
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_kvlist[] = {88, 88, 88, 88};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{30, 2, 128, 128}, {30, 2, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{30, 2, 128, 128}, {30, 2, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actual_seq_kvlist},                                 // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{4, 16}, {4, 16}}, ge::DT_INT32, ge::FORMAT_ND},                        // block_table-input12
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                          // 输出Tensor
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check11)
{
    // GQA非量化场景不开后量化不支持 q 和 out类型不一致
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 16, 127}, {4, 13, 16, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 1024, 1, 127}, {4, 1024, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 1024, 1, 127}, {4, 1024, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 13, 16, 127}, {4, 13, 16, 127}}, ge::DT_INT8, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check12)
{
    // GQA非量化 d不等长场景开后量化 q和out类型只支持bf16或fp16
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 16, 127}, {4, 13, 16, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 1024, 1, 88}, {4, 1024, 1, 88}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 1024, 1, 88}, {4, 1024, 1, 88}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{16, 88}, {16, 88}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 13, 16, 127}, {4, 13, 16, 127}}, ge::DT_INT8, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check13)
{
    // GQA非量化场景 B的取值范围(0, 65536]
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{65537, 13, 16, 127}, {65537, 13, 16, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{65537, 1024, 1, 127}, {65537, 1024, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{65537, 1024, 1, 127}, {65537, 1024, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{65537, 13, 16, 127}, {65537, 13, 16, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check14)
{
    // GQA非量化场景 QD的取值范围(0, 512]
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 16, 513}, {4, 13, 16, 513}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 1024, 1, 513}, {4, 1024, 1, 513}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 1024, 1, 513}, {4, 1024, 1, 513}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 13, 16, 513}, {4, 13, 16, 513}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check15)
{
    // GQA非量化 Q不等长场景 QKVD不能超过128
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 16, 129}, {4, 13, 16, 129}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 1024, 1, 127}, {4, 1024, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 1024, 1, 127}, {4, 1024, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 13, 16, 129}, {4, 13, 16, 129}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check17)
{
    // decode mla场景g轴仅支持{1, 2, 4, 8, 16, 32, 64, 128}
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 17, 512}, {4, 13, 17, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{4, 13, 17, 64}, {4, 13, 17, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{4, 10347, 1, 64}, {4, 10347, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 13, 17, 512}, {4, 13, 17, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(17)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check18)
{
    // GQA非量化场景 D不等于64或128时，G的取值范围[1, 64]
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 127, 127}, {4, 13, 127, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 127}, {4, 10347, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 1, 127}, {4, 10347, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 13, 127, 127}, {4, 13, 127, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_tiling_19)
{
    // GQA非量化场景 q维度需要和out维度一致
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{5, 1, 512}, {5, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0
            {{{5, 768, 512}, {5, 768, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{5, 768, 512}, {5, 768, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{5, 1, 64}, {5, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{5, 768, 64}, {5, 768, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{5, 1, 1, 512}, {5, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                        // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132385025;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check20)
{
    // GQA非量化场景 inputlayout为BSND_BNSD，q与out的N不一致
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 127, 128}, {4, 13, 127, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 128}, {4, 10347, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 1, 128}, {4, 10347, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 125, 13, 128}, {4, 125, 13, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND_BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check21)
{
    // decode mla场景 inputlayout为BNSD_NBSD，q与out的S不一致
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{4, 13, 17, 64}, {4, 13, 17, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{4, 10347, 1, 64}, {4, 10347, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{13, 4, 15, 512}, {13, 4, 15, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD_NBSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check22)
{
    // decode mla场景 inputlayout为BSND_NBSD，q与out的S不一致
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{4, 13, 17, 64}, {4, 13, 17, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{4, 10347, 1, 64}, {4, 10347, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{15, 4, 13, 512}, {15, 4, 13, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND_NBSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check23)
{
    // decode mla场景 inputlayout为BSH_NBSD，q与out的h不一致
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 8192}, {4, 13, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 512}, {4, 10347, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 512}, {4, 10347, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{4, 13, 1024}, {4, 13, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{4, 10347, 64}, {4, 10347, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{15, 4, 13, 512}, {15, 4, 13, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH_NBSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_tiling_24)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{12, 64, 512}, {12, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{3, 6, 9, 12}, {3, 6, 9, 12}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{1024, 2048, 3072, 4096}, {1024, 2048, 3072, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{4096, 1, 64}, {4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{63, 12, 512}, {63, 12, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                        // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666666664f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND_NTD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132385026;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check25)
{
    // GQA非量化场景 K V的dtype类型需要一致
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 127, 128}, {4, 13, 127, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 128}, {4, 10347, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 1, 128}, {4, 10347, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 13, 127, 128}, {4, 13, 127, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check26)
{
    // GQA非量化场景 KV的S不相等
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 127, 128}, {4, 13, 127, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 128}, {4, 10347, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10348, 1, 128}, {4, 10348, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 13, 127, 128}, {4, 13, 127, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check27)
{
    // GQA非量化 D不等长场景 KV的N不相等
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 64, 128}, {4, 13, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 100}, {4, 10347, 1, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 2, 100}, {4, 10347, 2, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 13, 64, 128}, {4, 13, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check28)
{
    // GQA非量化场景 N与QN需要保持一致
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 63, 128}, {4, 13, 63, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 128}, {4, 10347, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 1, 128}, {4, 10347, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 13, 64, 128}, {4, 13, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check29)
{
    // GQA非量化场景 N与QN需要保持一致
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{63, 52, 128}, {63, 52, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{1, 41388, 128}, {1, 41388, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1, 41388, 128}, {1, 41388, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{3, 6, 9, 12}, {3, 6, 9, 12}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{1024, 2048, 3072, 4096}, {1024, 2048, 3072, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},             // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{64, 52, 128}, {64, 52, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("NTD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check30)
{
    // GQA非量化场景 BSH成功
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 8192}, {4, 13, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 128}, {4, 10347, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 128}, {4, 10347, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 13, 8192}, {4, 13, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 107133670105728;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_qkvout_check31)
{
    // GQA非量化场景 D不等长 BSND成功
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 64, 127}, {4, 13, 64, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 126}, {4, 10347, 1, 126}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 1, 126}, {4, 10347, 1, 126}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
            {{{4, 13, 64, 127}, {4, 13, 64, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382980;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// rope check
// D512 rope success
TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_tiling_mla_D512)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 32, 16, 512}, {4, 32, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 1, 2048, 512}, {4, 1, 2048, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 1, 2048, 512}, {4, 1, 2048, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{4,32,16,64}, {4,32,16,64}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{4,1,2048,64}, {4,1,2048,64}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24   
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                       // 输出Tensor
         {{{4, 32, 16, 512}, {4, 32, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5128)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266602241;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// D128 rope success
TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_tiling_mla_D128)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 32, 16, 128}, {4, 32, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 1, 2048, 128}, {4, 1, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 1, 2048, 128}, {4, 1, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{4,32,16,64}, {4,32,16,64}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{4,1,2048,64}, {4,1,2048,64}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24   
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                       // 输出Tensor
         {{{4, 32, 16, 128}, {4, 32, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5128)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266602241;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// check rope dtype需要和 query/key 保持一致
TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_tiling_mla_D512_ropeQKDtype)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 32, 16, 512}, {4, 32, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 1, 2048, 512}, {4, 1, 2048, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 1, 2048, 512}, {4, 1, 2048, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{4,32,16,64}, {4,32,16,64}}, ge::DT_BF16, ge::FORMAT_ND},                             // query_rope-input21
            {{{4,1,2048,64}, {4,1,2048,64}}, ge::DT_BF16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24   
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                       // 输出Tensor
         {{{4, 32, 16, 512}, {4, 32, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5128)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266602241;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

//check rope仅支持bf16/fp16
TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_tiling_mla_ropeDtype)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 32, 16, 512}, {4, 32, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 1, 2048, 512}, {4, 1, 2048, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 1, 2048, 512}, {4, 1, 2048, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{4,32,16,64}, {4,32,16,64}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // query_rope-input21
            {{{4,1,2048,64}, {4,1,2048,64}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24   
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                       // 输出Tensor
         {{{4, 32, 16, 512}, {4, 32, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5128)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266602241;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// check D=512 attenOut dtype需要和query一致
TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_tiling_mla_attenOutDtype)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 32, 16, 512}, {4, 32, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 1, 2048, 512}, {4, 1, 2048, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 1, 2048, 512}, {4, 1, 2048, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{4,32,16,64}, {4,32,16,64}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{4,1,2048,64}, {4,1,2048,64}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24   
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                       // 输出Tensor
         {{{4, 32, 16, 512}, {4, 32, 16, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5128)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266602241;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// check D仅支持128/512
TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_tiling_mla_D)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 32, 16, 256}, {4, 32, 16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 1, 2048, 256}, {4, 1, 2048, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 1, 2048, 256}, {4, 1, 2048, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{4,32,16,64}, {4,32,16,64}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{4,1,2048,64}, {4,1,2048,64}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24   
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                       // 输出Tensor
         {{{4, 32, 16, 256}, {4, 32, 16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5128)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266602241;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// check rope_D仅支持64
TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_tiling_mla_ropeD)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 32, 16, 512}, {4, 32, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 1, 2048, 512}, {4, 1, 2048, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 1, 2048, 512}, {4, 1, 2048, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{4,32,16,32}, {4,32,16,32}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{4,1,2048,32}, {4,1,2048,32}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24   
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                       // 输出Tensor
         {{{4, 32, 16, 512}, {4, 32, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5128)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266602241;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// per-block 全量化 check 不支持pse
TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_tiling_perblock_pse)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // query-input0
            {{{4, 2, 2048, 128}, {4, 2, 2048, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // key-input1
            {{{4, 2, 2048, 128}, {4, 2, 2048, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // value-input2
            {{{1,8,128,2048}, {1,8,128,2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3   1NS1S2
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{4,2,8,1}, {4,2,8,1}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{4,2,8,1}, {4,2,8,1}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key_rope_antiquant_scale-input23
            {{{4,8,1,1}, {4,8,1,1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                       // 输出Tensor
         {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5128)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266602241;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// per-tensor 全量化 check 输入QKV为int8
TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_tiling_pertensor_Vdtype)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_INT8, ge::FORMAT_ND},     // query-input0
            {{{4, 2, 2048, 128}, {4, 2, 2048, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{4, 2, 2048, 128}, {4, 2, 2048, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24   
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                       // 输出Tensor
         {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5128)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266602241;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// per-tensor 全量化 check D支持1-512
TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_tiling_pertensor_Dsize)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 8, 128, 513}, {4, 8, 128, 513}}, ge::DT_INT8, ge::FORMAT_ND},     // query-input0
            {{{4, 2, 2048, 513}, {4, 2, 2048, 513}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{4, 2, 2048, 513}, {4, 2, 2048, 513}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24   
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                       // 输出Tensor
         {{{4, 8, 128, 513}, {4, 8, 128, 513}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5128)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266602241;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// mla 全量化 check Q scale shape dim
TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_tiling_mlafullquant_qScaleShapeDim)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 32, 16, 512}, {4, 32, 16, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // query-input0
            {{{4, 1, 2048, 512}, {4, 1, 2048, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // key-input1
            {{{4, 1, 2048, 512}, {4, 1, 2048, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{4,32,16,64}, {4,32,16,64}}, ge::DT_BF16, ge::FORMAT_ND},                             // query_rope-input21
            {{{4,1,2048,64}, {4,1,2048,64}}, ge::DT_BF16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key_rope_antiquant_scale-input23
            {{{4,32,8, 2}, {4,32,8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                       // 输出Tensor
         {{{4, 32, 16, 512}, {4, 32, 16, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5128)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266602241;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PA_0)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,1,8}, {1,5,1,8}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,0,8}, {1,5,0,8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,0,8}, {1,5,0,8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{1, 16}, {1, 16}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,1,8}, {1,5,1,8}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PA_NonQuant)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_kvlist[] = {1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,1,8}, {1,5,1,8}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,8,8}, {1,5,8,8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,8,8}, {1,5,8,8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actual_seq_kvlist},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{1, 16}, {1, 16}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,1,8}, {1,5,1,8}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PA_QSequal1)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_kvlist[] = {1};
    int64_t antiquant_list[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,1,8}, {1,5,1,8}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,128,8}, {1,5,128,8}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,128,8}, {1,5,128,8}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actual_seq_kvlist},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, antiquant_list},                             // antiquant_scale-input10
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, antiquant_list},                             // antiquant_offset-input11
            {{{1, 16}, {1, 16}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,1,8}, {1,5,1,8}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PA_QSbig1)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_kvlist[] = {1};
    int64_t antiquant_list[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,128,8}, {1,5,128,8}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,128,8}, {1,5,128,8}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,128,8}, {1,5,128,8}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actual_seq_kvlist},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, antiquant_list},                             // antiquant_scale-input10
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, antiquant_list},                             // antiquant_offset-input11
            {{{1, 16}, {1, 16}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,128,8}, {1,5,128,8}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PA_DimNum)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_kvlist[] = {1};
    int64_t antiquant_list[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,128,128}, {1,5,128,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,128,128}, {1,5,128,128}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,128,128}, {1,5,128,128}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actual_seq_kvlist},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, antiquant_list},                             // antiquant_scale-input10
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, antiquant_list},                             // antiquant_offset-input11
            {{{1, 16}, {1, 16}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,128,128}, {1,5,128,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PA_Mask)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_kvlist[] = {1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 16}, {2048, 16}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actual_seq_kvlist},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{1, 16}, {1, 16}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PA_QKVDDifferent)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,64}, {1,5,2048,64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666666664f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PA_actualSeqLengthsKv)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666666664f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PA_Feature_leftpadding)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476483184f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(156)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PA_Feature_Prefix)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1, 16, 3, 128}, {1, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},                   // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{1, 4, 1, 128}, {1, 4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1, 16, 3, 128}, {1, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476483184f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(156)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PA_blockTable_dataType)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_kvlist[] = {1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actual_seq_kvlist},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476483184f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(156)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PA_blockTable_ShapeSize)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{16, 0}, {16, 0}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476483184f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(156)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}


TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PA_blockTable_dim)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_kvlist[] = {1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{4, 4, 16, 128}, {4, 4, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 4, 16, 128}, {4, 4, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actual_seq_kvlist},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476483184f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(156)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PA_blockTableShape_dim4)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_kvlist[] = {64};
    int64_t antiquant_list[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_scale-input10
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_offset-input11
            {{{1, 16}, {1, 16}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PA_KV)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_BOOL, ge::FORMAT_ND}, // key-input1
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_BOOL, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476483184f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(156)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_Sparse01Dim2)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t antiquant_list[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_scale-input10
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_NextHigherPre)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t antiquant_list[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_scale-input10
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-32)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_Sparse0_Shape)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t antiquant_list[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{1, 2048, 1024}, {1, 2048, 1024}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_scale-input10
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_Sparse2_Shape)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t antiquant_list[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 1024}, {2048, 1024}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_scale-input10
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}


TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_Sparse0_Next)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t antiquant_list[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_scale-input10
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4096)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-4096)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_Sparse0_Pre)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t antiquant_list[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_scale-input10
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-4096)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4096)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_Sparse4_Ne_1)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t antiquant_list[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_scale-input10
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-16)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_Sparse4_Ne_2)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t antiquant_list[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_scale-input10
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-16)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}


TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_Sparse4_Pre)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t antiquant_list[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_scale-input10
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-4096)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4096)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_Sparse4_Next)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t antiquant_list[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_scale-input10
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, antiquant_list},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4096)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-4096)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_Dtype)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_Sparsemode)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_IFAMLA)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,1,512}, {1,5,1,512}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,512}, {1,5,2048,512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,512}, {1,5,2048,512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{1,5,1,64}, {1,5,1,64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{1,5,2048,64}, {1,5,2048,64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,1,512}, {1,5,1,512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666666664f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_Dim)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048}, {2048}}, ge::DT_BOOL, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_Shape_IsnotPFADim_2)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 1}, {2048, 1}}, ge::DT_BOOL, ge::FORMAT_ND},                  // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_Shape_IsnotPFADim_3)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{1, 2048, 1}, {1, 2048, 1}}, ge::DT_BOOL, ge::FORMAT_ND},                  // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}


TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_Mask_QKVDDifferent)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,64}, {1,5,2048,64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666666664f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}


TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PostQuant_Scale2)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PostQuant_ShapeSize)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PostQuant_Prefix)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{1, 5, 1, 128}, {1, 5, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{1, 5, 1, 128}, {1, 5, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PostQuant_Out)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PostQuant_AntiQuant)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PostQuant_Offset_datatype)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PostQuant_Offset_dim)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PostQuant_Offset_shape)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PostQuant_QDtype_bf16)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PostQuant_QDtype_Tensor)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_PostQuant_QDtype_Channel)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{1, 5, 1, 64}, {1, 5, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_LeftPadding_PA)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{2048}, {2048}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{1,16}, {1,16}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_LeftPadding_Pse)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_LeftPadding_TND)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{12, 64, 512}, {12, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{3, 6, 9, 12}, {3, 6, 9, 12}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{1024, 2048, 3072, 4096}, {1024, 2048, 3072, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},             // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},           // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{4096, 1, 64}, {4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{12, 64, 512}, {12, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{12, 64, 2}, {12, 64, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},                        // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666666664f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_LeftPadding_actseqlenQ)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476483184f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(156)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_LeftPadding_actseqlen)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476483184f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(156)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_LeftPadding_PaddingSizeShape)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476483184f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(156)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_SoftmaxLSE_MuiltPara_TND_Shape_Dim)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{12, 64, 256}, {12, 64, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0
            {{{4096, 1, 256}, {4096, 1, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4096, 1, 256}, {4096, 1, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{3, 6, 9, 12}, {3, 6, 9, 12}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{1024, 2048, 3072, 4096}, {1024, 2048, 3072, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},             // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{12, 64, 256}, {12, 64, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{12, 64}, {12, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},                        // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666666664f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_SoftmaxLSE_MuiltPara_TND_Shape)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{12, 64, 256}, {12, 64, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0
            {{{4096, 1, 256}, {4096, 1, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4096, 1, 256}, {4096, 1, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{3, 6, 9, 12}, {3, 6, 9, 12}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{1024, 2048, 3072, 4096}, {1024, 2048, 3072, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},             // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{12, 64, 256}, {12, 64, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{12, 64, 2}, {12, 64, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},                        // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666666664f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_SoftmaxLSE_MuiltPara_NoTND_Shape)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{4, 2048, 2048}, {4, 2048, 2048}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{4, 16, 3, 2}, {4, 16, 3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476483184f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(156)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_SoftmaxLSE_MuiltPara_NoTND_Shape_Dim)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{4, 2048, 2048}, {4, 2048, 2048}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{4, 16, 3}, {4, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476483184f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(156)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_SoftmaxLSE_Existence_ShapeAndDesc)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{4, 2048, 2048}, {4, 2048, 2048}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476483184f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(156)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedInferAttentionScoreTiling, FusedInferAttentionScoreTiling_SoftmaxLSE_SingleDtype)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 4, 962, 128}, {4, 4, 962, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{4, 2048, 2048}, {4, 2048, 2048}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{4, 16, 3, 128}, {4, 16, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{4, 16, 3, 1}, {4, 16, 3, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},      // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476483184f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(156)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}