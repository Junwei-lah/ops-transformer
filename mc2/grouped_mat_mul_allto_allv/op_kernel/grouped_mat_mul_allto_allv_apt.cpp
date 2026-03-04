/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
* \file grouped_mat_mul_allto_allv_apt.cpp
* \brief
*/
#include "basic_api/kernel_basic_intf.h"
#include "arch35/quant_grouped_mat_mul_allto_allv_tiling.h"
#include "grouped_mat_mul_allto_allv_tiling.h"
#include "grouped_mat_mul_allto_allv_tiling_key.h"
#include "grouped_mat_mul_allto_allv.h"
#if __CCE_AICORE__ == 310
    #if __has_include("../../allto_allv_grouped_mat_mul/mc2_templates/mc2_templates.h")
    #include "../../allto_allv_grouped_mat_mul/mc2_templates/mc2_templates.h"
    #else
    #include "../../allto_allv_grouped_mat_mul/op_kernel/mc2_templates/mc2_templates.h"
    #endif
#else
    #if __has_include("../../allto_allv_grouped_mat_mul/mc2_templates/mc2_templates.h")
    #include "../../allto_allv_grouped_mat_mul/mc2_templates/mc2_templates.h"
    #else
    #include "../../allto_allv_grouped_mat_mul/op_kernel/mc2_templates/mc2_templates.h"
    #endif
#endif
#if defined(CONST_TILING)
#define GET_NESTED_TILING_DATA_MEMBER_ADDR(outerType, innerType, outerMember, innerMember, var, tiling) \
    const outerType *outerPtr##var = (const outerType *)(tiling);                                       \
    const innerType *innerPtr##var = &(outerPtr##var->outerPtr##var);                                   \
    const int32_t *(var) = (const int32_t)((const uint8_t *)&(innerPtr##var->innerMember))
#else
#define GET_NESTED_TILING_DATA_MEMBER_ADDR(outerType, innerType, outerMember, innerMember, var, tiling) \
    size_t outerOffset##var = (size_t)(&((outerType *)0)->outerMember);                                 \
    size_t innerOffset##var = (size_t)(&((innerType *)0)->innerMember);                                 \
    __gm__ int32_t *(var) = (__gm__ int32_t *)((__gm__ uint8_t *)(tiling) + outerOffset##var + innerOffset##var)
#endif

using namespace AscendC;
using namespace MC2KernelTemplate;
using namespace Mc2GroupedMatmulTilingData;

#if defined(CONST_TILING)
#define TILING_TYPE const int32_t
#else
#define TILING_TYPE __gm__ int32_t
#endif

template <typename X_T, const bool IS_OPT_MM, const bool IS_GMM_WEIGHT_TRANS, const bool IS_OPT_WEIGHT_TRANS>
struct GMMATAVType { // Grouped_Mat_Mul_All_To_Allv_Type
    using xType = X_T;
    static constexpr bool isOptionalMm = IS_OPT_MM;
    static constexpr bool isGmmWeightTrans = IS_GMM_WEIGHT_TRANS;
    static constexpr bool isOptWeightTrans = IS_OPT_WEIGHT_TRANS;
};

#define INVOKE_GMMATAV_OP_IMPL_A5(templateClass, ...)                                                   \
    do {                                                                                                \
        TPipe pipe;                                                                                     \
        templateClass<GMMATAVType<__VA_ARGS__>> op;                                                     \
        op.Init(                                                                                        \
            gmmxGM, gmmweightGM, sendCountsTensorOptionalGM, recvCountsTensorOptionalGM, mmxOptionalGM, \
            mmweightOptionalGM, yGM, mmyOptionalGM, workspaceGM, contextGM, &tilingData, &pipe);        \
        op.Process();                                                                                   \
    } while (0)

#define INVOKE_GMMATAV_OP_IMPL(templateClass, ...)                                                       \
    do {                                                                                                 \
        TPipe pipe;                                                                                      \
        templateClass<GMMATAVType<__VA_ARGS__>> op;                                                      \
        op.Init(                                                                                         \
            gmmxGM, gmmweightGM, sendCountsTensorOptionalGM, recvCountsTensorOptionalGM, mmxOptionalGM,  \
            mmweightOptionalGM, yGM, mmyOptionalGM, workspaceGM, contextGM, &tilingData, hcclInitTiling, \
            alltoAllvCcTiling, &pipe);                                                                   \
        op.Process();                                                                                    \
    } while (0)

template <
    bool TILINGKEY_COMPUTE_MATMUL, bool TILINGKEY_GROUPED_MATMUL_TRANS,
    bool TILINGKEY_MATMUL_TRANS, uint8_t TILINGKEY_GMM_QUANT_MODE, uint8_t TILINGKEY_SHARED_MM_QUANT_MODE>
__global__ __aicore__ void grouped_mat_mul_allto_allv(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorOptionalGM, GM_ADDR recvCountsTensorOptionalGM,
    GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM, GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM,
    GM_ADDR gmmXOffsetOptinalGM, GM_ADDR gmmWeightOffsetOptinalGM, GM_ADDR mmxScaleGM, GM_ADDR mmWeightScaleGM,
    GM_ADDR mmXOffsetOptinalGM, GM_ADDR mmWeightOffsetOptinalGM, GM_ADDR commQuantScaleGM,
    GM_ADDR yGM, GM_ADDR mmyOptionalGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspaceGM == nullptr) {
        return;
    }
    GM_ADDR userWorkspace = GetUserWorkspace(workspaceGM);
    if (userWorkspace == nullptr) {
        return;
    }
#if (ORIG_DTYPE_GMM_X == DT_BF16 || ORIG_DTYPE_GMM_X == DT_FLOAT16)
    REGISTER_TILING_DEFAULT(GroupedMatMulAlltoAllvTilingData);
    auto tiling = (__gm__ GroupedMatMulAlltoAllvTilingData*)tilingGM;
    __gm__ void* hcclInitTiling = (__gm__ void*)(&(tiling->hcclInitTiling));
    __gm__ void* alltoAllvCcTiling = (__gm__ void*)(&(tiling->alltoAllvCcTiling));
    GET_TILING_DATA(tilingData, tilingGM);
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();
#if (ORIG_DTYPE_GMM_X == DT_BF16)
    INVOKE_GMMATAV_OP_IMPL(GroupedMatmulAlltoAllv, DTYPE_GMM_X, TILINGKEY_COMPUTE_MATMUL,
                            TILINGKEY_GROUPED_MATMUL_TRANS, TILINGKEY_MATMUL_TRANS);
#elif (ORIG_DTYPE_GMM_X == DT_FLOAT16)
    INVOKE_GMMATAV_OP_IMPL(GroupedMatmulAlltoAllv, DTYPE_GMM_X, TILINGKEY_COMPUTE_MATMUL,
                            TILINGKEY_GROUPED_MATMUL_TRANS, TILINGKEY_MATMUL_TRANS);
#endif

#else
    TPipe pipe;
    REGISTER_TILING_DEFAULT(QuantGmmA2avTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    const QuantGmmA2avTilingData* tilingData_ = &tilingData;
    const void* hcclInitTiling = &(tilingData_->hcclA2avTiling.hcclInitTiling);
    uint64_t hcclCcTilingOffset = offsetof(QuantGmmA2avTilingData, hcclA2avTiling) +
                    offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling);
    constexpr CubeFormat W_FORMAT = CubeFormat::ND;
    constexpr bool USE_SEND_COUNTS = true;
    constexpr bool IS_SHARED_EXPERT = true;
    constexpr bool IS_NOT_SHARED_EXPERT = false;

    using HcclOpType = HcclA2avOp<DTYPE_Y, false>;
    using GmmASWKernelType = Mc2GroupedMatmul::Mc2GmmASWKernel<DTYPE_GMM_X, DTYPE_GMM_WEIGHT, float, float, DTYPE_Y,
        W_FORMAT, TILINGKEY_GROUPED_MATMUL_TRANS, TILINGKEY_MATMUL_TRANS>;
    using ComputeOpType =
        QuantGroupedMatmul<QuantGmmA2avTilingData, GMMQuantTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT, float, DTYPE_Y,
        CubeFormat::ND,
        false,  // 左矩阵不支持转置
        TILINGKEY_GROUPED_MATMUL_TRANS,
        false,  // isShared
        false>; // isA2avGmm
    using SharedGmmExpertOpType =
        QuantGroupedMatmul<QuantGmmA2avTilingData, GMMQuantTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT, float, DTYPE_Y,
        CubeFormat::ND,
        false,  // 左矩阵不支持转置
        TILINGKEY_MATMUL_TRANS,
        true,   // isShared
        false>; // isA2avGmm
    using GmmA2avSchedulerType = GmmA2avScheduler<HcclOpType, ComputeOpType,
        SharedGmmExpertOpType, TILINGKEY_COMPUTE_MATMUL>;
    // hccl
    HcclOpType hcclOp;
    hcclOp.Init(hcclInitTiling, hcclCcTilingOffset, &tilingData.taskTilingInfo, workspaceGM, yGM);
    // gmm
    GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData,
        gmmBaseTiling, gmmArray, gmmArrayAddr_, tilingGM);
    ComputeOpType computeOp;
    // auto tilingData_ = static_cast<const QuantGmmA2avTilingData*>(&tilingData);
    auto gmmGroupListGM = workspaceGM + tilingData_->workspaceInfo.wsGmmOutputSize;
    computeOp.Init(gmmxGM, gmmweightGM, gmmxScaleGM, gmmWeightScaleGM, workspaceGM, gmmGroupListGM,
        tilingData_, &tilingData_->gmmBaseTiling, gmmArrayAddr_, &pipe, false);
    // sharedmm
    GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData,
        sharedGmmTiling, gmmArray, mmArrayAddr_, tilingGM);
    SharedGmmExpertOpType shareComputeOp;
    auto mmGroupListGM = workspaceGM + tilingData_->workspaceInfo.wsGmmOutputSize +
        tilingData_->workspaceInfo.wsGmmComputeWorkspaceSize;
    shareComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, mmxScaleGM, mmWeightScaleGM, mmyOptionalGM,
        mmGroupListGM, tilingData_, &tilingData_->sharedGmmTiling, mmArrayAddr_, &pipe, false);
    GmmA2avSchedulerType gmmA2avScheduler(hcclOp, computeOp, shareComputeOp, &tilingData.taskTilingInfo);
    gmmA2avScheduler.Process();
#endif
}