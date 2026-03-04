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
 * \file grouped_mat_mul_allto_allv_tiling.h
 * \brief
 */

#ifndef __GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H__
#define __GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H__

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#if __CCE_AICORE__ == 310
    #if __has_include("../../allto_allv_grouped_mat_mul/mc2_templates/common/a2av_common_tiling.h")
    #include "../../allto_allv_grouped_mat_mul/mc2_templates/common/a2av_common_tiling.h"
    #else
    #include "../../allto_allv_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h"
    #endif
#else
    #if __has_include("../../allto_allv_grouped_mat_mul/mc2_templates/common/a2av_common_tiling.h")
    #include "../../allto_allv_grouped_mat_mul/mc2_templates/common/a2av_common_tiling.h"
    #else
    #include "../../allto_allv_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h"
    #endif
#endif
constexpr uint32_t MAX_EXPERT_SIZE = 256U; // 最大通信域专家的数量

struct GmmAlltoAllvAicpuTiling {
    int32_t sendCnt[MAX_EXPERT_SIZE];
    int32_t recvCnt[MAX_EXPERT_SIZE];
};

struct GmmAlltoAllvCommonTilingInfo {
    uint64_t A;
    uint64_t H;
    uint64_t sharedMatmulH;
    uint64_t E_ep;
    uint64_t N1;
    uint64_t Bs;
    uint64_t N2;
    uint64_t BsK;
    uint64_t epWorldSize;
    uint64_t aivCoreNum;
    uint64_t aicCoreNum;
    bool isGmmWeightTrans;
    bool isMmWeightTrans;
    bool isOptionalMatmul;
    bool isOptionalSendRecvCountTensors;
    bool isNeedMM = false;
};

class GroupedMatMulAlltoAllvTilingData
{
public:
    Mc2InitTiling hcclInitTiling;
    Mc2CcTiling alltoAllvCcTiling;
    GmmAlltoAllvCommonTilingInfo commonTilingInfo;
    TCubeTiling matmulTiling;
    TCubeTiling sharedExpMatmulTiling;
    GmmAlltoAllvAicpuTiling aicpuTilingInfo;
};

#endif // __GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H__