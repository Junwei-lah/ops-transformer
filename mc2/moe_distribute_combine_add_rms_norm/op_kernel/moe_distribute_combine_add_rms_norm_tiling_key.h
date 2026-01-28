/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 /* !
 * \file moe_distribute_combine_add_rms_norm_tiling_key.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_TILING_KEY_H
#define MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

#define TILINGKEY_NO_QUANT 0
#define TILINGKEY_INT8_QUANT 1

namespace Mc2Tiling {
ASCENDC_TPL_ARGS_DECL(MoeDistributeCombineAddRmsNorm, ASCENDC_TPL_BOOL_DECL(TILINGKEY_TP_WORLD_SIZE, 0, 1),
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_QUANT_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            TILINGKEY_NO_QUANT, TILINGKEY_INT8_QUANT),
);
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_TP_WORLD_SIZE, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(TILINGKEY_QUANT_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_NO_QUANT,
                                                          TILINGKEY_INT8_QUANT),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(MoeDistributeCombineV2TilingData)), );
}
#endif