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
 * \file compute_stage.h
 * \brief
 */

#ifndef MC2_COMPUTE_STAGE_H
#define MC2_COMPUTE_STAGE_H

#include "../../arch35/3rd_head.h"
#include "./matmul/fp_matmul.h"
#include "./matmul/quant_matmul.h"
#include "./math/mc2_vec_transpose.h"

namespace MC2KernelTemplate {

// 使用matmulv3算子作为计算节点的计算实现,后续是否转置的参数通过算子的模板参数获取
#ifndef DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP
#define DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP(TilingType, ComputationType) \
    using ComputationType = FPMatmul<\
        Mc2MatmulV3Advanced::Mc2MatmulAswKernel<\
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, false>,\
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, X2TRANSPOSE>,\
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>,\
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DtypeBias>,\
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD>,\
        FpQuantExtraData, TilingType>
#endif

#ifndef DEFINE_AND_IMPL_MC2_MATMUL_FOR_MATMUL_COMPUTATION_QUANT
#define DEFINE_AND_IMPL_MC2_MATMUL_FOR_MATMUL_COMPUTATION_QUANT(TilingType, ComputationType) \
    using ComputationType = QuantMatmul<\
        Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel<DTYPE_X1, DTYPE_X2, float, float, float,\
            DTYPE_Y, CubeFormat::ND, CubeFormat::ND, CubeFormat::ND, false, X2TRANSPOSE, float, Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock>,\
        QuantExtraData, TilingType>

#endif

#ifndef DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_WEIGHT_QUANT
#define DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_WEIGHT_QUANT() \
    do {} while (0)
#endif

// 使用math算子作为计算节点的计算实现
#ifndef DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION
#define DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(TransposeDataType, TransposeType) \
    using TransposeType = MC2VecTranspose<TransposeDataType>
#endif

};

#endif // MC2_COMPUTE_STAGE_H