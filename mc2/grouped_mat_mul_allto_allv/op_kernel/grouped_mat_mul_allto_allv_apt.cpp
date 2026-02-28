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
 * \file grouped_mat_mul_allto_allv_apt.cpp
 * \brief Quant Grouped MatMul AlltoAllV APT 实现
 */
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#include "grouped_mat_mul_allto_allv_tiling.h"
#include "arch35/grouped_mat_mul_allto_allv_tiling_key.h"

using namespace AscendC;

/**
 * APT 入口函数
 *
 * @tparam TILINGKEY_GMM_WEIGHT_TRANS       GMM 权重转置 (0/1)
 * @tparam TILINGKEY_SHARED_MM_WEIGHT_TRANS 共享专家 MM 权重转置 (0/1)
 * @tparam TILINGKEY_GMM_QUANT_MODE         GMM 量化模式 (0-6)
 * @tparam TILINGKEY_SHARED_MM_QUANT_MODE   共享专家 MM 量化模式 (0-6)
 */
template <bool TILINGKEY_GMM_WEIGHT_TRANS, bool TILINGKEY_SHARED_MM_WEIGHT_TRANS,
          uint32_t TILINGKEY_COMM_QUANT_MODE, uint32_t TILINGKEY_GMM_QUANT_MODE,
          uint32_t TILINGKEY_SHARED_MM_QUANT_MODE>
__global__ __aicore__ void quant_grouped_mat_mul_allto_allv(
    GM_ADDR gmmXGM, GM_ADDR gmmWeightGM,
    GM_ADDR gmmXScaleGM, GM_ADDR gmmWeightScaleGM,
    GM_ADDR gmmXOffsetGM, GM_ADDR gmmWeightOffsetGM,
    GM_ADDR mmXGM, GM_ADDR mmWeightGM,
    GM_ADDR mmXScaleGM, GM_ADDR mmWeightScaleGM,
    GM_ADDR mmXOffsetGM, GM_ADDR mmWeightOffsetGM,
    GM_ADDR sendCountsGM, GM_ADDR recvCountsGM,
    GM_ADDR yGM, GM_ADDR mmYGM,
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
}