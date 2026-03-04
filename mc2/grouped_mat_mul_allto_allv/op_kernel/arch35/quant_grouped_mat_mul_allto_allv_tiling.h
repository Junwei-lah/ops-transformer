/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file quant_grouped_mat_mul_allto_allv_tiling.h
 * \brief Quant Grouped MatMul AlltoAllV TilingData 定义
 */
#ifndef QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H__
#define QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H__

#if __CCE_AICORE__ == 310
    #if __has_include("../../allto_allv_grouped_mat_mul/mc2_templates/common/a2av_common_tiling.h")
    #include "../../allto_allv_grouped_mat_mul/mc2_templates/common/a2av_common_tiling.h"
    #else
    #include "../../../allto_allv_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h"
    #endif
#else
    #if __has_include("../../allto_allv_grouped_mat_mul/mc2_templates/common/a2av_common_tiling.h")
    #include "../../allto_allv_grouped_mat_mul/mc2_templates/common/a2av_common_tiling.h"
    #else
    #include "../../../allto_allv_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h"
    #endif
#endif
#pragma once

// 使用公共命名空间中的类型
using MC2KernelTemplate::GMMQuantTilingData;
using MC2KernelTemplate::TaskTilingInfo;

/**
 * GMM A2AV Workspace 信息
 *
 * 算子级 workspace 分为三部分:
 *   [0, wsGmmOutputSize)                              → 路由专家 GMM 输出缓冲, 传给 GmmComputeOp.Init 的 y 参数
 *   [wsGmmOutputSize, + wsGmmComputeWorkspaceSize)    → 路由专家 GmmComputeOp 内部空间, 传给 GmmComputeOp.Init 的 tempAddr 参数
 *   [+, + wsSharedGmmComputeWorkspaceSize)            → 共享专家 SharedGmmComputeOp 内部空间, 传给 SharedGmmComputeOp.Init 的 tempAddr 参数
 *
 * GmmComputeOp / SharedGmmComputeOp workspace 内部布局 (由各自内部管理, tiling 侧仅需计算并分配总大小):
 *   [0, 64)                   ptrTable:  4 × 16B GetTensorAddr 双重间接指针表 (x, weight, scaleB, y)
 *                              注: scaleA (perTokenScale) 不经过 GetTensorAddr, 直接作为 float* 使用
 *                              注: 若后续 hasBias=1, bias 也经过 GetTensorAddr, 需增加第 5 个 slot (80B)
 *   [64, 64 + expertNum * 8)  groupList: expertNum × int64_t 累积和
 *   总大小 = 64 + expertNum * sizeof(int64_t)
 *   其中 expertNum = 单次 GmmASWKernel 调用中的专家数 (当前 groupNum=1 时 expertNum=1,
 *   后续多专家同时计算时 expertNum 随融合算子切分的专家数量变化)
 */
struct GmmA2avWorkspaceInfo {
    uint64_t wsGmmOutputSize;                    // 路由专家 GMM 主输出缓冲大小 (x @ weight -> gmm_output -> hccl -> y)
    uint64_t wsGmmComputeWorkspaceSize;          // 路由专家 GmmComputeOp 内部空间大小
    uint64_t wsSharedGmmComputeWorkspaceSize;    // 共享专家 SharedGmmComputeOp 内部空间大小
};

struct QuantGmmA2avTilingData {
    // ============ HCCL AlltoAllV Tiling ============
    MC2KernelTemplate::HcclA2avTilingInfo hcclA2avTiling;

    // ============ Workspace 配置信息 ============
    GmmA2avWorkspaceInfo workspaceInfo;

    // ============ 核心配置信息 (Task/Loop/Comm) ============
    TaskTilingInfo taskTilingInfo;

    // ============ 共享专家 GMM Tiling（放在前面）============
    GMMQuantTilingData sharedGmmTiling; // 共享专家 GMM Tiling 数据

    // ============ 普通专家 GMM Tiling ============
    GMMQuantTilingData gmmBaseTiling; // 共享专家 GMM Tiling 数据，后续还会在kernel中根据任务刷新

    // ============ isPermuteOut ============
    bool isPermuteOut = false;
};
#endif
