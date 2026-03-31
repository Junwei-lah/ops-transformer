/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 *BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class AttentionPioneerProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AttentionPioneerProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AttentionPioneerProto TearDown" << std::endl;
    }
};

// Case 0: BSND layout, basic infershape, softmax_lse disabled
TEST_F(AttentionPioneerProto, AttentionPioneer_infershape_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AttentionPioneer",
        // 输入Tensor (31个输入)
        {
            {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // 0: query
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // 1: key
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // 2: value
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 3: pse_shift
            {{{4, 13, 10347}, {4, 13, 10347}}, ge::DT_INT8, ge::FORMAT_ND},            // 4: atten_mask
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // 5: actual_seq_lengths
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // 6: actual_seq_lengths_kv
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 7: dequant_scale1
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 8: quant_scale1
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 9: dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 10: quant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 11: quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 12: antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 13: antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // 14: block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // 15: query_padding_size
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // 16: kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 17: key_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 18: key_antiquant_offset
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 19: value_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 20: value_antiquant_offset
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 21: key_shared_prefix
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 22: value_shared_prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // 23: actual_shared_prefix_len
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 24: query_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 25: key_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 26: key_rope_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // 27: dequant_scale_query
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 28: key_sink
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 29: key_rope_sink
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // 30: value_sink
        },
        {                                                                       // 输出Tensor
         {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {                                                                       // 属性
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
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 13, 16, 512}, // attentionOut
        {0},              // softmaxOut (lse disabled)
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 1: BSH layout, softmax_lse enabled
TEST_F(AttentionPioneerProto, AttentionPioneer_infershape_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AttentionPioneer",
        {
            {{{32, 1, 1024}, {32, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // 0: query
            {{{32, 4096, 128}, {32, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // 1: key
            {{{32, 4096, 128}, {32, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // 2: value
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 3: pse_shift
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                              // 4: atten_mask
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                            // 5: actual_seq_lengths
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // 6: actual_seq_lengths_kv
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 7: dequant_scale1
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 8: quant_scale1
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 9: dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 10: quant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 11: quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 12: antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 13: antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // 14: block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // 15: query_padding_size
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // 16: kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 17: key_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 18: key_antiquant_offset
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 19: value_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 20: value_antiquant_offset
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 21: key_shared_prefix
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 22: value_shared_prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // 23: actual_shared_prefix_len
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 24: query_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 25: key_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 26: key_rope_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // 27: dequant_scale_query
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 28: key_sink
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 29: key_rope_sink
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // 30: value_sink
        },
        {
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
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {32, 1, 1024}, // attentionOut
        {32, 8, 1, 1}, // softmaxOut (lse enabled)
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 2: TND layout
TEST_F(AttentionPioneerProto, AttentionPioneer_infershape_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AttentionPioneer",
        {
            {{{100, 16, 192}, {100, 16, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 0: query (TND: T=100, N=16, D=192)
            {{{2048, 1, 192}, {2048, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 1: key
            {{{2048, 1, 128}, {2048, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 2: value
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 3: pse_shift
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                             // 4: atten_mask
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                           // 5: actual_seq_lengths
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // 6: actual_seq_lengths_kv
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 7: dequant_scale1
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 8: quant_scale1
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 9: dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 10: quant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 11: quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 12: antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 13: antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // 14: block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                            // 15: query_padding_size
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                            // 16: kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 17: key_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 18: key_antiquant_offset
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 19: value_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 20: value_antiquant_offset
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 21: key_shared_prefix
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 22: value_shared_prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                            // 23: actual_shared_prefix_len
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 24: query_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 25: key_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 26: key_rope_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // 27: dequant_scale_query
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 28: key_sink
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 29: key_rope_sink
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 30: value_sink
        },
        {
         {{{100, 16, 128}, {100, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut (TND: T=100, N=16, D=128 from value)
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.07216878f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {100, 16, 128}, // attentionOut (value_d = 128)
        {0},            // softmaxOut (lse disabled)
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 3: TND layout, softmax_lse enabled
TEST_F(AttentionPioneerProto, AttentionPioneer_infershape_3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AttentionPioneer",
        {
            {{{100, 16, 192}, {100, 16, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 0: query
            {{{2048, 1, 192}, {2048, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 1: key
            {{{2048, 1, 128}, {2048, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 2: value
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 3: pse_shift
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                             // 4: atten_mask
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                           // 5: actual_seq_lengths
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // 6: actual_seq_lengths_kv
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 7: dequant_scale1
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 8: quant_scale1
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 9: dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 10: quant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 11: quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 12: antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 13: antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // 14: block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                            // 15: query_padding_size
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                            // 16: kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 17: key_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 18: key_antiquant_offset
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 19: value_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 20: value_antiquant_offset
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 21: key_shared_prefix
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 22: value_shared_prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                            // 23: actual_shared_prefix_len
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 24: query_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 25: key_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 26: key_rope_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // 27: dequant_scale_query
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 28: key_sink
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 29: key_rope_sink
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 30: value_sink
        },
        {
         {{{100, 16, 128}, {100, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{100, 16, 1}, {100, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},     // softmax_lse shape for TND
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.07216878f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
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
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {100, 16, 128}, // attentionOut
        {100, 16, 1},   // softmaxOut (TND lse: {T, N, 1})
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 4: BNSD layout
TEST_F(AttentionPioneerProto, AttentionPioneer_infershape_4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AttentionPioneer",
        {
            {{{4, 8, 1, 128}, {4, 8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // 0: query (BNSD)
            {{{4, 1, 2048, 128}, {4, 1, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // 1: key
            {{{4, 1, 2048, 128}, {4, 1, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // 2: value
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 3: pse_shift
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                   // 4: atten_mask
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                 // 5: actual_seq_lengths
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // 6: actual_seq_lengths_kv
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 7: dequant_scale1
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 8: quant_scale1
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 9: dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 10: quant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 11: quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 12: antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 13: antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                  // 14: block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                  // 15: query_padding_size
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                  // 16: kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 17: key_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 18: key_antiquant_offset
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 19: value_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 20: value_antiquant_offset
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 21: key_shared_prefix
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 22: value_shared_prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                  // 23: actual_shared_prefix_len
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 24: query_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 25: key_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 26: key_rope_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // 27: dequant_scale_query
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 28: key_sink
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 29: key_rope_sink
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 30: value_sink
        },
        {
         {{{4, 8, 1, 128}, {4, 8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 8, 1, 128}, // attentionOut (BNSD: B=4, N=8, S=1, D=128)
        {0},             // softmaxOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 5: TND_NTD layout (output layout is NTD)
TEST_F(AttentionPioneerProto, AttentionPioneer_infershape_5)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AttentionPioneer",
        {
            {{{100, 16, 512}, {100, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 0: query (TND)
            {{{2048, 1, 512}, {2048, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 1: key
            {{{2048, 1, 512}, {2048, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 2: value
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 3: pse_shift
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                             // 4: atten_mask
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                           // 5: actual_seq_lengths
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // 6: actual_seq_lengths_kv
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 7: dequant_scale1
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 8: quant_scale1
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 9: dequant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 10: quant_scale2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 11: quant_offset2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 12: antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 13: antiquant_offset
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // 14: block_table
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                            // 15: query_padding_size
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                            // 16: kv_padding_size
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 17: key_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 18: key_antiquant_offset
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 19: value_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 20: value_antiquant_offset
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 21: key_shared_prefix
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 22: value_shared_prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                            // 23: actual_shared_prefix_len
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 24: query_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 25: key_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 26: key_rope_antiquant_scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // 27: dequant_scale_query
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 28: key_sink
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 29: key_rope_sink
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // 30: value_sink
        },
        {
         {{{16, 100, 512}, {16, 100, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // attentionOut (NTD layout)
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND_NTD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {16, 100, 512}, // attentionOut (NTD: N=16, T=100, D=512)
        {0},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 6: BNSD_BSND layout (query BNSD, output BSND)
TEST_F(AttentionPioneerProto, AttentionPioneer_infershape_6)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AttentionPioneer",
        {
            {{{4, 8, 1, 128}, {4, 8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // 0: query (BNSD)
            {{{4, 1, 2048, 128}, {4, 1, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // 1: key
            {{{4, 1, 2048, 128}, {4, 1, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // 2: value
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // 3-30: empty
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
         {{{4, 1, 8, 128}, {4, 1, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // attentionOut (BSND)
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD_BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 1, 8, 128}, // attentionOut (BSND: B=4, S=1, N=8, D=128)
        {0},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 8: BSND layout, query dim mismatch (3 dims instead of 4) - should fail
TEST_F(AttentionPioneerProto, AttentionPioneer_infershape_dim_mismatch)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AttentionPioneer",
        {
            {{{4, 13, 8192}, {4, 13, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // 0: query (3 dims, but BSND expects 4)
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
         {{{4, 13, 8192}, {4, 13, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
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
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 9: NTD_TND layout (query NTD, output TND)
TEST_F(AttentionPioneerProto, AttentionPioneer_infershape_9)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AttentionPioneer",
        {
            {{{16, 100, 512}, {16, 100, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // 0: query (NTD: N=16, T=100, D=512)
            {{{1, 2048, 512}, {1, 2048, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // 1: key
            {{{1, 2048, 512}, {1, 2048, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // 2: value
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
         {{{100, 16, 512}, {100, 16, 512}}, ge::DT_BF16, ge::FORMAT_ND},  // attentionOut (TND)
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("NTD_TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {100, 16, 512}, // attentionOut (TND: T=100, N=16, D=512)
        {0},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 10: numHeads = 0, should fail
TEST_F(AttentionPioneerProto, AttentionPioneer_infershape_num_heads_zero)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AttentionPioneer",
        {
            {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
         {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
