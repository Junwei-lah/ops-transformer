/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_grouped_mat_mul_allto_allv_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops
{
class QuantGroupedMatMulAlltoAllv : public OpDef
{
public:
    explicit QuantGroupedMatMulAlltoAllv(const char* name) : OpDef(name)
    {
        // 14 scenarios total:
        // 0-1: FP16 input/output
        // 2-3: BF16 input/output
        // 4-5: HIFLOAT8 input, FP16/BF16 output
        // 6-13: MX quantization (E4M3/E5M2 combinations with FP16/BF16 output)
        this->Input("gmm_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,  // MX: E4M3-E4M3
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,  // MX: E4M3-E5M2
                       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,      // MX: E5M2-E4M3
                       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2})     // MX: E5M2-E5M2
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gmm_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,  // MX: E4M3-E4M3
                       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,      // MX: E4M3-E5M2
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,  // MX: E5M2-E4M3
                       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2})     // MX: E5M2-E5M2
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("send_counts_tensor")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
                       ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
                       ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
                       ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("recv_counts_tensor")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
                       ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
                       ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
                       ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mm_x")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,  // MX: E4M3-E4M3
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,  // MX: E4M3-E5M2
                       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,      // MX: E5M2-E4M3
                       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2})     // MX: E5M2-E5M2
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mm_weight")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,  // MX: E4M3-E4M3
                       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,      // MX: E4M3-E5M2
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,  // MX: E5M2-E4M3
                       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2})     // MX: E5M2-E5M2
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gmmXScale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,  // MX: E4M3-E4M3
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,  // MX: E4M3-E5M2
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,  // MX: E5M2-E4M3
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0}) // MX: E5M2-E5M2
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gmmWeightScale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,  // MX: E4M3-E4M3
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,  // MX: E4M3-E5M2
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,  // MX: E5M2-E4M3
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0}) // MX: E5M2-E5M2
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mmXScaleOptional")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,  // MX: E4M3-E4M3
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,  // MX: E4M3-E5M2
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,  // MX: E5M2-E4M3
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0}) // MX: E5M2-E5M2
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mmWeightScaleOptional")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,  // MX: E4M3-E4M3
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,  // MX: E4M3-E5M2
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,  // MX: E5M2-E4M3
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0}) // MX: E5M2-E5M2
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("commQuantScaleOptional")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT})  // MX quant does not support low-bit comm
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16,  // MX: E4M3-E4M3 -> FP16/BF16
                       ge::DT_FLOAT16, ge::DT_BF16,  // MX: E4M3-E5M2 -> FP16/BF16
                       ge::DT_FLOAT16, ge::DT_BF16,  // MX: E5M2-E4M3 -> FP16/BF16
                       ge::DT_FLOAT16, ge::DT_BF16}) // MX: E5M2-E5M2 -> FP16/BF16
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("mm_y")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16,  // MX: E4M3-E4M3 -> FP16/BF16
                       ge::DT_FLOAT16, ge::DT_BF16,  // MX: E4M3-E5M2 -> FP16/BF16
                       ge::DT_FLOAT16, ge::DT_BF16,  // MX: E5M2-E4M3 -> FP16/BF16
                       ge::DT_FLOAT16, ge::DT_BF16}) // MX: E5M2-E5M2 -> FP16/BF16
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("ep_world_size").AttrType(REQUIRED).Int();
        this->Attr("send_counts").AttrType(REQUIRED).ListInt();
        this->Attr("recv_counts").AttrType(REQUIRED).ListInt();
        this->Attr("trans_gmm_weight").AttrType(OPTIONAL).Bool(false);
        this->Attr("trans_mm_weight").AttrType(OPTIONAL).Bool(false);
        this->Attr("gmm_x_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("gmm_weight_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("mm_x_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("mm_weight_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("comm_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("group_size").AttrType(OPTIONAL).Int(0);
        this->Attr("comm_quant_dtype").AttrType(OPTIONAL).Int(0);
        this->Attr("y_dtype").AttrType(OPTIONAL).Int(static_cast<int64_t>(ge::DT_UNDEFINED));
        this->Attr("mm_dtype").AttrType(OPTIONAL).Int(static_cast<int64_t>(ge::DT_UNDEFINED));

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false")  // 动态shape,复用二进制,后续图支持后修改
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().AddConfig("ascend950", aicore_config);
        this->MC2().HcclGroup({"group"});
    }
};

OP_ADD(QuantGroupedMatMulAlltoAllv);
}  // namespace ops