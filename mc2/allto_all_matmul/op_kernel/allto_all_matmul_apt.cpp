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
 * \file allto_all_matmul.cpp
 * \brief
 */
#include <cstring>
#include <lib/matmul_intf.h>
#include "basic_api/kernel_basic_intf.h"
#include "common.h"
#include "./arch35/template_head.h"
#include "./arch35/allto_all_matmul_tiling_key.h"
#include "./arch35/allto_all_matmul_tiling_data.h"
#include "./arch35/allto_all_matmul_arch35.h"

using namespace AscendC;
using namespace MC2KernelTemplate;
using namespace AlltoAllMatmulImpl;

#ifndef ALLTO_ALL_MATMUL_APT_FP_IMPL
#define ALLTO_ALL_MATMUL_APT_FP_IMPL(tilingData, pipe)   \
    do {    \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(HcclServerType::HCCL_SERVER_TYPE_CCU, 0, 1, AlltoAllMatmulTilingData, CommunicationType); \
        CommunicationType commImplName(&tilingData);    \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_X1, TransposeType);   \
        TransposeType transposeImplName(&pipe); \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP(Mc2MatMulV3TilingData, ComputationType); \
        ComputationType matmulImplName(&pipe); \
        using SchedulerContextType = PipelineContext<FpQuantExtraData, Mc2MatMulV3TilingData>;  \
        using SchedulerType = MC2KernelPipelineCommTransComputeTemplate<CommunicationType, TransposeType, ComputationType, SchedulerContextType>;   \
        SchedulerType SchedulerImpl(&commImplName, &transposeImplName, &matmulImplName);    \
        AlltoAllMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingData> op(&SchedulerImpl); \
        op.Init(x1, x2, bias, y, all2all_out, workspaceGM, &tilingData, &pipe); \
        op.Process();   \
    } while (0)
#endif

template <uint32_t QUANTMODE, bool X2TRANSPOSE, uint32_t DTYPEBIAS>
__global__ __aicore__ void allto_all_matmul(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR comm_scale, GM_ADDR x1_offset,
                                            GM_ADDR x2_offset, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AlltoAllMatmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AlltoAllMatmulTilingData, tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    if constexpr (DTYPEBIAS == DTYPE_BIAS_SAME_WITH_X) {
        using DtypeBias = DTYPE_X1;
        ALLTO_ALL_MATMUL_APT_FP_IMPL(tilingData, pipe);
    } else if constexpr (DTYPEBIAS == DTYPE_BIAS_FP32) {
        using DtypeBias = float;
        ALLTO_ALL_MATMUL_APT_FP_IMPL(tilingData, pipe);
    }
}