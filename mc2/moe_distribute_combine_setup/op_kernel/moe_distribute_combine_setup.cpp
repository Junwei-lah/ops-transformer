/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_setup.cpp
 * \brief
 */

#include "basic_api/kernel_basic_intf.h"
#include "moe_distribute_combine_setup_tiling_data.h"
#include "moe_distribute_combine_setup_tiling_key.h"

#ifdef __DAV_C310__
#include "arch35/moe_distribute_combine_setup_arch35.h"
#endif // __DAV_C310__

using namespace AscendC;
using namespace MoeDistributeCombineSetupImpl;

template <bool HasTp>
__global__ __aicore__ void moe_distribute_combine_setup(GM_ADDR expandX, GM_ADDR expertIds,
                                                        GM_ADDR assistInfoForCombine, GM_ADDR quantExpandX,
                                                        GM_ADDR commCmdInfoOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MoeDistributeCombineSetupTilingData);
    auto tiling = (__gm__ MoeDistributeCombineSetupTilingData *)tilingGM;
    __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling->mc2InitTiling));
    __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling->mc2CcTiling));
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if constexpr (!HasTp) { // tp=1
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineSetupTilingData, tilingData, tilingGM);
        MoeDistributeCombineSetup<DTYPE_EXPAND_X, int32_t> op;
        op.Init(expandX, expertIds, assistInfoForCombine, quantExpandX, commCmdInfoOut, workspaceGM, &pipe, &tilingData,
                mc2InitTiling, mc2CcTiling);
        op.Process();
    }
#endif
}