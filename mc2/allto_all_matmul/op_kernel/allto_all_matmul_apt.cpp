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
#include <kernel_operator.h>
#include <lib/matmul_intf.h>
#include "common.h"
#include "./arch35/allto_all_matmul_arch35.h"
#include "./arch35/allto_all_matmul_pipeline.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "./arch32/allto_all_matmul_tiling_data_910_93.h"
#include "./arch32/allto_all_matmul_tiling_key_910_93.h"
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#include "./arch35/allto_all_matmul_tiling_key.h"
#include "./arch35/allto_all_matmul_tiling_data.h"
#include "./arch35/allto_all_kc_quant_matmul_arch35.h"
#include "./arch35/allto_all_kc_quant_matmul_pipeline.h"
#include "./arch35/allto_all_mx_quant_matmul_arch35.h"
#include "./arch35/allto_all_mx_quant_matmul_pipeline.h"
#endif

using AscendC::HcclServerType;
using AscendC::TPipe;
using MC2KernelTemplate::MC2AlltoAllContext;
using MC2KernelTemplate::MC2AlltoAllPrimitives;

#ifndef ALLTO_ALL_MATMUL_A3_FP_IMPL
#define ALLTO_ALL_MATMUL_A3_FP_IMPL(tilingData, pipe)                                                                  \
    do {                                                                                                               \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, HcclServerType::HCCL_SERVER_TYPE_AICPU, MC2AlltoAllContext,           \
                                          AlltoAllMatmulTilingDataA3, MC2AlltoAllPrimitives, 0, 1, CommunicationType); \
        CommunicationType commImplName(&tilingData);                                                                   \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_X1, TransposeType);                                            \
        TransposeType transposeImplName(&pipe);                                                                        \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_A3_FP(ComputationContextType);                                \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_A3_FP(ComputationType);                                               \
        ComputationType matmulImplName(&pipe);                                                                         \
        using SchedulerContextType = Mc2Kernel::AlltoAllMmPipelineContext<ComputationContextType>;                     \
        using SchedulerType = Mc2Kernel::AlltoAllMatmulPipeLine<CommunicationType, TransposeType, ComputationType,     \
                                                                SchedulerContextType>;                                 \
        SchedulerType SchedulerImpl(&commImplName, &transposeImplName, &matmulImplName);                               \
        Mc2Kernel::AlltoAllMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataA3> op(           \
            &SchedulerImpl);                                                                                           \
        op.Init(x1, x2, bias, y, all2all_out, workspaceGM, &tilingData, &pipe);                                        \
        op.Process();                                                                                                  \
    } while (0)
#endif

#ifndef ALLTO_ALL_MATMUL_APT_FP_IMPL
#define ALLTO_ALL_MATMUL_APT_FP_IMPL(tilingData, pipe)                                                                 \
    do {                                                                                                               \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, HcclServerType::HCCL_SERVER_TYPE_CCU, MC2AlltoAllContext,             \
                                          AlltoAllMatmulTilingData, MC2AlltoAllPrimitives, 0, 1, CommunicationType);   \
        CommunicationType commImplName(&tilingData);                                                                   \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_X1, TransposeType);                                            \
        TransposeType transposeImplName(&pipe);                                                                        \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_FP(ComputationContextType);                                   \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP(ComputationType);                                                  \
        ComputationType matmulImplName(&pipe);                                                                         \
        using SchedulerContextType = Mc2Kernel::AlltoAllMmPipelineContext<ComputationContextType>;                     \
        using SchedulerType = Mc2Kernel::AlltoAllMatmulPipeLine<CommunicationType, TransposeType, ComputationType,     \
                                                                SchedulerContextType>;                                 \
        SchedulerType SchedulerImpl(&commImplName, &transposeImplName, &matmulImplName);                               \
        Mc2Kernel::AlltoAllMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingData> op(             \
            &SchedulerImpl);                                                                                           \
        op.Init(x1, x2, bias, y, all2all_out, workspaceGM, &tilingData, &pipe);                                        \
        op.Process();                                                                                                  \
    } while (0)
#endif

#ifndef ALLTO_ALL_KC_QUANT_MATMUL_IMPL
#define ALLTO_ALL_KC_QUANT_MATMUL_IMPL(tilingData, pipe, MMDataTypeX1)                                                 \
    do {                                                                                                               \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, HcclServerType::HCCL_SERVER_TYPE_CCU, MC2AlltoAllContext,             \
                                          AlltoAllQuantMatmulTilingData, MC2AlltoAllPrimitives, 0, 1,                  \
                                          CommunicationType);                                                          \
        CommunicationType commImplName(&tilingData);                                                                   \
        DEFINE_MC2_FP8_DYNAMIC_QUANT_PERTOKEN(DTYPE_X1, MMDataTypeX1, TransAndDynamicQuantType);                       \
        TransAndDynamicQuantType dynamicQuantImplName(&pipe);                                                          \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_QUANT(ComputationContextType);                                \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_QUANT(ComputationType, MMDataTypeX1, DTYPE_X2);                       \
        ComputationType matmulImplName(&pipe);                                                                         \
        using SchedulerContextType = Mc2Kernel::AlltoAllKCQmmPipelineContext<ComputationContextType>;                  \
        using SchedulerType = Mc2Kernel::AlltoAllKCQuantMatmulPipeLine<CommunicationType, TransAndDynamicQuantType,    \
                                                                       ComputationType, SchedulerContextType>;         \
        SchedulerType SchedulerImpl(&commImplName, &dynamicQuantImplName, &matmulImplName);                            \
        Mc2Kernel::AlltoAllKcQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllQuantMatmulTilingData> op( \
            &SchedulerImpl);                                                                                           \
        op.Init(x1, x2, bias, y, all2all_out, x1_scale, x2_scale, x2_offset, workspaceGM, &tilingData, &pipe);         \
        op.Process();                                                                                                  \
    } while (0)
#endif

#ifndef ALLTO_ALL_MX_QUANT_MATMUL_IMPL
#define ALLTO_ALL_MX_QUANT_MATMUL_IMPL(tilingData, pipe, commDataTypeX1, isMxFp4)                                      \
    do {                                                                                                               \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, HcclServerType::HCCL_SERVER_TYPE_CCU, MC2AlltoAllContext,             \
                                          AlltoAllQuantMatmulTilingData, MC2AlltoAllPrimitives, 0, 1,                  \
                                          CommunicationType);                                                          \
        CommunicationType commImplName(&tilingData);                                                                   \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(commDataTypeX1, TransposeType);                                      \
        TransposeType transposeImplName(&pipe);                                                                        \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(AscendC::fp8_e8m0_t, ScaleTransposeType);                            \
        ScaleTransposeType scaleTransposeImplName(&pipe);                                                              \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_MX_QUANT(ComputationContextType);                             \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_MX_QUANT(ComputationType);                                            \
        ComputationType matmulImplName(&pipe);                                                                         \
        using SchedulerContextType = Mc2Kernel::AlltoAllMXQmmPipelineContext<ComputationContextType>;                  \
        using SchedulerType =                                                                                          \
            Mc2Kernel::AlltoAllMXQuantMatmulPipeLine<CommunicationType, TransposeType, ScaleTransposeType,             \
                                                     ComputationType, SchedulerContextType>;                           \
        SchedulerType SchedulerImpl(&commImplName, &transposeImplName, &scaleTransposeImplName, &matmulImplName);      \
        Mc2Kernel::AlltoAllMxQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllQuantMatmulTilingData,     \
                                               isMxFp4>                                                                \
            op(&SchedulerImpl);                                                                                        \
        op.Init(x1, x2, bias, y, all2all_out, x1_scale, x2_scale, workspaceGM, &tilingData, &pipe);                    \
        op.Process();                                                                                                  \
    } while (0)
#endif

template <uint32_t QUANTMODE, bool X2TRANSPOSE, uint32_t DTYPEBIAS>
__global__ __aicore__ void allto_all_matmul(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale,
                                            GM_ADDR comm_scale, GM_ADDR x1_offset, GM_ADDR x2_offset, GM_ADDR y,
                                            GM_ADDR all2all_out, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    REGISTER_TILING_DEFAULT(AlltoAllMatmulTilingDataA3);
    GET_TILING_DATA_WITH_STRUCT(AlltoAllMatmulTilingDataA3, tilingData, tilingGM);

    if constexpr (DTYPEBIAS == DTYPE_BIAS_SAME_WITH_X) {
        using DtypeBias = DTYPE_X1;
        ALLTO_ALL_MATMUL_A3_FP_IMPL(tilingData, pipe);
    } else if constexpr (DTYPEBIAS == DTYPE_BIAS_FP32) {
        using DtypeBias = float;
        ALLTO_ALL_MATMUL_A3_FP_IMPL(tilingData, pipe);
    }
#endif

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
    REGISTER_TILING_DEFAULT(AlltoAllMatmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AlltoAllMatmulTilingData, tilingData, tilingGM);

    if constexpr (DTYPEBIAS == DTYPE_BIAS_SAME_WITH_X) {
        using DtypeBias = DTYPE_X1;
        ALLTO_ALL_MATMUL_APT_FP_IMPL(tilingData, pipe);
    } else if constexpr (DTYPEBIAS == DTYPE_BIAS_FP32) {
        using DtypeBias = float;
        ALLTO_ALL_MATMUL_APT_FP_IMPL(tilingData, pipe);
    }

#else
    REGISTER_TILING_DEFAULT(AlltoAllQuantMatmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AlltoAllQuantMatmulTilingData, tilingData, tilingGM);
#if (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) &&\
     ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2)))
    ALLTO_ALL_MX_QUANT_MATMUL_IMPL(tilingData, pipe, DTYPE_X1, false);
#elif ((ORIG_DTYPE_X1 == DT_FLOAT4_E2M1) && (ORIG_DTYPE_X2 == DT_FLOAT4_E2M1))
    ALLTO_ALL_MX_QUANT_MATMUL_IMPL(tilingData, pipe, uint8_t, true);
#else
    if constexpr (QUANTMODE == KC_QUANT_FP8E5M2_MODE) {
        ALLTO_ALL_KC_QUANT_MATMUL_IMPL(tilingData, pipe, float8_e5m2_t);
    } else if constexpr (QUANTMODE == KC_QUANT_FP8E4M3_MODE) {
        ALLTO_ALL_KC_QUANT_MATMUL_IMPL(tilingData, pipe, float8_e4m3_t);
    }
#endif
#endif
#endif
}