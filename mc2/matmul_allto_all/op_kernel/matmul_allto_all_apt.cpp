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
 * \file matmul_allto_all.cpp
 * \brief
 */
#include <cstring>
#include <lib/matmul_intf.h>
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "common.h"
#include "./arch35/template_head.h"
#include "./arch35/matmul_allto_all_tiling_key.h"
#include "./arch35/matmul_allto_all_arch35.h"
#include "./arch35/kc_quant_matmul_allto_all_arch35.h"
using namespace AscendC;
using namespace MC2KernelTemplate;
using namespace MatmulAlltoAllImpl;

#ifndef MATMUL_ALLTO_ALL_APT_FP_IMPL
#define MATMUL_ALLTO_ALL_APT_FP_IMPL(tilingData, pipe)  \
    do {    \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP(Mc2MatMulV3TilingData, ComputationType); \
        ComputationType matmulImplName(&pipe); \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_Y, TransposeType);    \
        TransposeType transposeImplName(&pipe);    \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(HcclServerType::HCCL_SERVER_TYPE_CCU, 1, 0, MatmulAlltoAllTilingData, CommunicationType); \
        CommunicationType commImplName(&tilingData);  \
        using SchedulerContextType = PipelineContext<FpQuantExtraData, Mc2MatMulV3TilingData>;  \
        using SchedulerType = MC2KernelPipelineTemplate<ComputationType, TransposeType, CommunicationType, SchedulerContextType>;   \
        SchedulerType SchedulerImpl(&matmulImplName, &transposeImplName, &commImplName);    \
        MatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingData> op(&SchedulerImpl); \
        op.Init(x1, x2, bias, y, workspaceGM, &tilingData, &pipe);  \
        op.Process();   \
    } while (0)
#endif

template <uint32_t QUANTMODE, bool X2TRANSPOSE, uint32_t DTYPEBIAS>
__global__ __aicore__ void matmul_allto_all(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale,
                                            GM_ADDR comm_scale, GM_ADDR x1_offset, GM_ADDR x2_offset, GM_ADDR y,
                                            GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    //kernel的使用类型，这里是cube和vic混用，cube是主核，cube:vec=1:2
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
    //注册默认的tilingdata，需要保证有且只有一个默认tilingdata被注册
    REGISTER_TILING_DEFAULT(MatmulAlltoAllTilingData);
    GET_TILING_DATA_WITH_STRUCT(MatmulAlltoAllTilingData, tilingData, tilingGM);

    if constexpr (DTYPEBIAS == DTYPE_BIAS_SAME_WITH_X) {
        using DtypeBias = DTYPE_X1;
        MATMUL_ALLTO_ALL_APT_FP_IMPL(tilingData, pipe);
    } else if constexpr (DTYPEBIAS == DTYPE_BIAS_FP32) {
        using DtypeBias = float;
        MATMUL_ALLTO_ALL_APT_FP_IMPL(tilingData, pipe);
    }
#else
    //注册默认的tilingdata，需要保证有且只有一个默认tilingdata被注册
    REGISTER_TILING_DEFAULT(QuantMatmulAlltoAllTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantMatmulAlltoAllTilingData, tilingData, tilingGM);

    DEFINE_AND_IMPL_MC2_MATMUL_FOR_MATMUL_COMPUTATION_QUANT(DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams, ComputationType, DTYPE_X1, DTYPE_X2);
    ComputationType matmulImplName(&pipe);
    DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_Y, TransposeType);
    TransposeType transposeImplName(&pipe);
    DEFINE_MC2_HCCL_FOR_COMMUNICATION(HcclServerType::HCCL_SERVER_TYPE_CCU, 1, 0, QuantMatmulAlltoAllTilingData, CommunicationType);
    CommunicationType commImplName(&tilingData);
    using SchedulerContextType = PipelineContext<QuantExtraData, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams>;
    using SchedulerType = MC2KernelPipelineTemplate<ComputationType, TransposeType, CommunicationType, SchedulerContextType>;
    SchedulerType SchedulerImpl(&matmulImplName, &transposeImplName, &commImplName);
    KcQuantMatmulAlltoAllArch35<SchedulerType, SchedulerContextType, QuantMatmulAlltoAllTilingData> op(&SchedulerImpl);
    op.Init(x1, x2, bias, y, x1_scale, x2_scale, x2_offset, workspaceGM, &tilingData, &pipe);
    op.Process();
#endif
}