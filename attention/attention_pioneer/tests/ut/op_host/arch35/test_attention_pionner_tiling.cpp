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
#include "../../../../op_host/attention_pioneer_tiling_compile_info.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class AttentionPioneerTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AttentionPioneerTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AttentionPioneerTiling TearDown" << std::endl;
    }
};

// PFA MLA sink success: TND layout, d=192, v_d=128, sinkLength=128, FP16
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_pfa_mla_fp16)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0 (TND)
            {{{2048, 1, 192}, {2048, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key-input1
            {{{2048, 1, 128}, {2048, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths-input5
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input11
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input12
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input15
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // key_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // value_antiquant_scale-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input20
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_shared_prefix-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_shared_prefix-input22
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-input23
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // query_rope-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope-input25
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input26
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_query-input27
            {{{128, 1, 192}, {128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_sink-input28
            {{{128, 1, 64}, {128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // key_rope_sink-input29
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // value_sink-input30
        },
        {                                                                               // outputs
         {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
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

            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_query-input27
            {{{128, 1, 512}, {128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_sink-input28
            {{{128, 1, 64}, {128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // key_rope_sink-input29
            {{{128, 1, 512}, {128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // value_sink-input30
        },
        {                                                                               // outputs
         {{{16, 32, 512}, {16, 32, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
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
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// Fail: invalid sinkLength (not 0 or 128)
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_invalid_sinkLength)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0 (TND)
            {{{2048, 1, 192}, {2048, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key-input1
            {{{2048, 1, 128}, {2048, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths-input5
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input11
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input12
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input15
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // key_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // value_antiquant_scale-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input20
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_shared_prefix-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_shared_prefix-input22
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-input23
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // query_rope-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope-input25
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input26
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_query-input27
            {{{64, 1, 192}, {64, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // key_sink-input28 (sinkLength=64, invalid)
            {{{64, 1, 64}, {64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},               // key_rope_sink-input29
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // value_sink-input30
        },
        {                                                                               // outputs
         {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
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
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// Fail: wrong dtype (INT8) with sink
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_wrong_dtype)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{16, 32, 192}, {16, 32, 192}}, ge::DT_INT8, ge::FORMAT_ND},              // query-input0 (INT8, wrong dtype)
            {{{2048, 1, 192}, {2048, 1, 192}}, ge::DT_INT8, ge::FORMAT_ND},            // key-input1
            {{{2048, 1, 128}, {2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},            // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths-input5
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input11
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input12
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input15
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // key_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // value_antiquant_scale-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input20
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_shared_prefix-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_shared_prefix-input22
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-input23
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // query_rope-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope-input25
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input26
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_query-input27
            {{{128, 1, 192}, {128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_sink-input28
            {{{128, 1, 64}, {128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // key_rope_sink-input29
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // value_sink-input30
        },
        {                                                                               // outputs
         {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
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
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// Fail: wrong layout (BNSD) with PFA MLA sink
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_wrong_layout_bnsd)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{4, 32, 16, 192}, {4, 32, 16, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // query-input0 (BNSD)
            {{{4, 1, 2048, 192}, {4, 1, 2048, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key-input1
            {{{4, 1, 2048, 128}, {4, 1, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_seq_lengths-input5
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input11
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input12
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input15
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // key_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // value_antiquant_scale-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input20
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_shared_prefix-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_shared_prefix-input22
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-input23
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // query_rope-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope-input25
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input26
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_query-input27
            {{{128, 1, 192}, {128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_sink-input28
            {{{128, 1, 64}, {128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // key_rope_sink-input29
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // value_sink-input30
        },
        {                                                                               // outputs
         {{{4, 32, 16, 192}, {4, 32, 16, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
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
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// Fail: wrong head dim (d=128 without rope for PFA MLA, not valid for sink)
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_wrong_head_dim)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{16, 32, 128}, {16, 32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0 (d=128, not MLA)
            {{{2048, 2, 128}, {2048, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key-input1
            {{{2048, 2, 128}, {2048, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths-input5
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input11
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input12
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input15
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // key_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // value_antiquant_scale-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input20
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_shared_prefix-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_shared_prefix-input22
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-input23
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // query_rope-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope-input25
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input26
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_query-input27
            {{{128, 2, 128}, {128, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_sink-input28
            {{{128, 2, 64}, {128, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // key_rope_sink-input29
            {{{128, 2, 128}, {128, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // value_sink-input30
        },
        {                                                                               // outputs
         {{{16, 32, 128}, {16, 32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
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
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// Fail: sink + PSE (pse_shift has non-empty shape)
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_pse_conflict)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0 (TND)
            {{{2048, 1, 192}, {2048, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key-input1
            {{{2048, 1, 128}, {2048, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value-input2
            {{{1, 32, 16, 2048}, {1, 32, 16, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // pse_shift-i            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132384002;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// PFA MLA sink success: TND layout, d=192, v_d=128, sinkLength=128, BF16
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_pfa_mla_bf16)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{16, 32, 192}, {16, 32, 192}}, ge::DT_BF16, ge::FORMAT_ND},              // query-input0 (TND)
            {{{2048, 1, 192}, {2048, 1, 192}}, ge::DT_BF16, ge::FORMAT_ND},            // key-input1
            {{{2048, 1, 128}, {2048, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},            // value-input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths-input5
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale2-input9
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // quant_scale2-input10
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // quant_offset2-input11
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // antiquant_scale-input12
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input15
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // key_antiquant_scale-input17
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // key_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // value_antiquant_scale-input19
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // value_antiquant_offset-input20
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // key_shared_prefix-input21
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // value_shared_prefix-input22
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-input23
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // query_rope-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // key_rope-input25
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // key_rope_antiquant_scale-input26
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_query-input27
            {{{128, 1, 192}, {128, 1, 192}}, ge::DT_BF16, ge::FORMAT_ND},              // key_sink-input28
            {{{128, 1, 64}, {128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                // key_rope_sink-input29
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},              // value_sink-input30
        },
        {                                                                               // outputs
         {{{16, 32, 192}, {16, 32, 192}}, ge::DT_BF16, ge::FORMAT_ND},                 // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
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

            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input15
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // key_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // value_antiquant_scale-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input20
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_shared_prefix-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_shared_prefix-input22
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-input23
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // query_rope-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope-input25
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input26
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_query-input27
            {{{128, 1, 192}, {128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_sink-input28
            {{{128, 1, 64}, {128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // key_rope_sink-input29
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // value_sink-input30
        },
        {                                                                               // outputs
         {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
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
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// Fail: sink + AlibiPse (pse_type=2)
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_alibipse_conflict)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0 (TND)
            {{{2048, 1, 192}, {2048, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key-input1
            {{{2048, 1, 128}, {2048, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value-input2
            {{{32}, {32}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // pse_shift-input3 (Ali BPE shape)
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths-input5
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input11
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input12
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input15
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // key_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // value_antiquant_scale-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input20
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_shared_prefix-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_shared_prefix-input22
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-input23
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // query_rope-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope-input25
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input26
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_query-input27
            {{{128, 1, 192}, {128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_sink-input28
            {{{128, 1, 64}, {128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // key_rope_sink-input29
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // value_sink-input30
        },
        {                                                                               // outputs
         {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
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
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// Fail: sink + leftpadding (query_padding_size has non-empty shape)
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_leftpadding_conflict)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0 (TND)
            {{{2048, 1, 192}, {2048, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key-input1
            {{{2048, 1, 128}, {2048, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths-input5
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input11
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input12
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // query_padding_size-input15 (non-empty, leftpadding)
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_padding_size-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // key_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // value_antiquant_scale-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input20
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_shared_prefix-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_shared_prefix-input22
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-input23
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // query_rope-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope-input25
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input26
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_query-input27
            {{{128, 1, 192}, {128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_sink-input28
            {{{128, 1, 64}, {128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // key_rope_sink-input29
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // value_sink-input30
        },
        {                                                                               // outputs
         {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132384002;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// IFA MLA sink success: TND_NTD layout, d=512, sinkLength=128, FP16, with query_rope and key_rope
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_ifa_mla_fp16)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{16, 32, 512}, {16, 32, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0 (TND)
            {{{2048, 1, 512}, {2048, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key-input1
            {{{2048, 1, 512}, {2048, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths-input5
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input11
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input12
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input15
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // key_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // value_antiquant_scale-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input20
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_shared_prefix-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_shared_prefix-input22
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-input23
            {{{16, 32, 64}, {16, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // query_rope-input24
            {{{2048, 1, 64}, {2048, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope-input25
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input26nput3 (non-empty, PSE)
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths-input5
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input11
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input12,
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

// Fail: sink + prefix (key_shared_prefix has non-empty shape)
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_prefix_conflict)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0 (TND)
            {{{2048, 1, 192}, {2048, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key-input1
            {{{2048, 1, 128}, {2048, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths-input5
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input11
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input12
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input15
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // key_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // value_antiquant_scale-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input20
            {{{256, 1, 192}, {256, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_shared_prefix-input21 (non-empty, prefix)
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // value_shared_prefix-input22 (non-empty, prefix)
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_shared_prefix_len-input23
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // query_rope-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope-input25
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input26
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_query-input27
            {{{128, 1, 192}, {128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_sink-input28
            {{{128, 1, 64}, {128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // key_rope_sink-input29
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // value_sink-input30
        },
        {                                                                               // outputs
         {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
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
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// Fail: sink + postquant (output is INT8)
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_postquant_conflict)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0 (TND)
            {{{2048, 1, 192}, {2048, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key-input1
            {{{2048, 1, 128}, {2048, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths-input5
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale2-input9
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2-input10 (non-empty for quant)
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_offset2-input11
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input12
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input15
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // key_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // value_antiquant_scale-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input20
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_shared_prefix-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_shared_prefix-input22
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-input23
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // query_rope-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope-input25
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input26
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_query-input27
            {{{128, 1, 192}, {128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_sink-input28
            {{{128, 1, 64}, {128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // key_rope_sink-input29
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // value_sink-input30
        },
        {                                                                               // outputs
         {{{16, 32, 192}, {16, 32, 192}}, ge::DT_INT8, ge::FORMAT_ND},                 // attentionOut (INT8 = postquant)
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
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
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// Fail: sink + perblock quant (FLOAT8 input with antiquant_mode=7)
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_perblock_quant_conflict)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},    // query-input0 (FLOAT8)
            {{{2048, 1, 192}, {2048, 1, 192}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},  // key-input1
            {{{2048, 1, 128}, {2048, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},  // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths-input5
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input11
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input12
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input15
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input16
            {{{4, 1, 12, 1}, {4, 1, 12, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},            // key_antiquant_scale-input17 (perblock)
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input18
            {{{4, 1, 12, 1}, {4, 1, 12, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},            // value_antiquant_scale-input19 (perblock)
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input20
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_shared_prefix-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_shared_prefix-input22
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-input23
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // query_rope-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope-input25
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input26
            {{{4, 32, 1, 1}, {4, 32, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},            // dequant_scale_query-input27 (perblock)
            {{{128, 1, 192}, {128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_sink-input28
            {{{128, 1, 64}, {128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // key_rope_sink-input29
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // value_sink-input30
        },
        {                                                                               // outputs
         {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
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
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// Fail: sink + pertensor quant (INT8 input)
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_pertensor_quant_conflict)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{16, 32, 192}, {16, 32, 192}}, ge::DT_INT8, ge::FORMAT_ND},              // query-input0 (INT8 = pertensor quant)
            {{{2048, 1, 192}, {2048, 1, 192}}, ge::DT_INT8, ge::FORMAT_ND},            // key-input1
            {{{2048, 1, 128}, {2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},            // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths-input5
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-input6
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},                                // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},                                // dequant_scale2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input11
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input12
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input15
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // key_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // value_antiquant_scale-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input20
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_shared_prefix-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_shared_prefix-input22
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-input23
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // query_rope-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope-input25
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input26
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_query-input27
            {{{128, 1, 192}, {128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_sink-input28
            {{{128, 1, 64}, {128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // key_rope_sink-input29
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // value_sink-input30
        },
        {                                                                               // outputs
         {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
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
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// Fail: IFA MLA sink with wrong layout (TND instead of TND_NTD)
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_ifa_mla_wrong_layout)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{16, 32, 512}, {16, 32, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0 (TND)
            {{{2048, 1, 512}, {2048, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key-input1
            {{{2048, 1, 512}, {2048, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths-input5
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input11
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input12
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input15
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // key_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // value_antiquant_scale-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input20
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_shared_prefix-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_shared_prefix-input22
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-input23
            {{{16, 32, 64}, {16, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // query_rope-input24
            {{{2048, 1, 64}, {2048, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope-input25
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input26
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_query-input27
            {{{128, 1, 512}, {128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_sink-input28
            {{{128, 1, 64}, {128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // key_rope_sink-input29
            {{{128, 1, 512}, {128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // value_sink-input30
        },
        {                                                                               // outputs
         {{{16, 32, 512}, {16, 32, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
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
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// sinkLength=0 (no sink, should pass without sink validation)
TEST_F(AttentionPioneerTiling, AttentionPioneerTiling_sink_zero_sinkLength)
{
    optiling::AttentionPioneerCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "AttentionPioneer",
        {
            {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0 (TND)
            {{{2048, 1, 192}, {2048, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key-input1
            {{{2048, 1, 128}, {2048, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths-input5
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-input6
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale1-input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1-input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input11
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input12
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input13
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input14
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input15
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // key_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // value_antiquant_scale-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input20
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_shared_prefix-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_shared_prefix-input22
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-input23
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // query_rope-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope-input25
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input26
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_query-input27
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_sink-input28 (empty = sinkLength=0)
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope_sink-input29
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_sink-input30
        },
        {                                                                               // outputs
         {{{16, 32, 192}, {16, 32, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                     // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
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
    int64_t expectTilingKey = 132384002;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}
