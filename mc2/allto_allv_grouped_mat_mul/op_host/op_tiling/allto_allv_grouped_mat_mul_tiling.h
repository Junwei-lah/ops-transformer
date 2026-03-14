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
 * \file allto_allv_grouped_mat_mul_tiling.h
 * \brief
 */
#ifndef MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_STRUCT_H
#define MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_STRUCT_H

#include "allto_allv_grouped_mat_mul_tiling_base.h"

namespace optiling {
class AlltoAllvGmmTilingStruct : public AlltoAllvGmmTilingBase
{
public:
    explicit AlltoAllvGmmTilingStruct(gert::TilingContext* context) : AlltoAllvGmmTilingBase(context){};

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
};
} // namespace optiling
#endif