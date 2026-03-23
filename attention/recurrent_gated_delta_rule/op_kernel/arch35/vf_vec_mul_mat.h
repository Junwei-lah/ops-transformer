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
 * \file vf_vec_mul_mat.h
 * \brief
 */

#ifndef VF_VEC_MUL_MAT_H
#define VF_VEC_MUL_MAT_H

#include "kernel_tensor.h"

namespace AscendC {

template <typename T>
__aicore__ inline void VecMulMatVF(const LocalTensor<T>& matrixOutUb, const LocalTensor<T>& rowVecInUb,
                                   const LocalTensor<T>& matrixInUb, const uint16_t row, const uint32_t col)
{
    __ubuf__ float * matrixUb = (__ubuf__ float*)matrixInUb.GetPhyAddr();
    __ubuf__ float * rowVecUb = (__ubuf__ float*)rowVecInUb.GetPhyAddr();
    __ubuf__ float * outputUb = (__ubuf__ float*)matrixOutUb.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64; // 一个寄存器能够存放64个FP32
    uint16_t dLoops = col / floatRepSize;
    uint32_t dTail = col % floatRepSize;
    uint16_t dTailLoop = dTail > 0 ? 1 : 0;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregMatrix;
        MicroAPI::RegTensor<T> vregRowVec;
        MicroAPI::RegTensor<T> vregOutput;
        MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg tailMask;
        tailMask = MicroAPI::UpdateMask<float>(dTail);
        constexpr static MicroAPI::CastTrait castTraitPack2 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        constexpr static MicroAPI::CastTrait castTraitF32ToHalf = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ODD};

        uint32_t colOffset = 0;
        uint32_t rowOffset = 0;
        for (uint16_t j = 0; j < dLoops; j++) {
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregRowVec, rowVecUb + colOffset);
            rowOffset = 0;
            for (uint16_t i = 0; i < row; i++) {
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregMatrix, matrixUb + colOffset + rowOffset);
                MicroAPI::Mul(vregOutput, vregMatrix, vregRowVec, fullMask);
                MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(outputUb + colOffset + rowOffset, vregOutput, fullMask);
                rowOffset += col;
            }
            colOffset += floatRepSize; 
        }

        if (dTailLoop > 0) {
            rowOffset = 0;
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregRowVec, rowVecUb + dLoops * floatRepSize);
            for (uint16_t i = 0; i < row; i++) {
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregMatrix, matrixUb + dLoops * floatRepSize + rowOffset);
                MicroAPI::Mul(vregOutput, vregMatrix, vregRowVec, tailMask);
                MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(outputUb + dLoops * floatRepSize + rowOffset, vregOutput, tailMask);
                rowOffset += col;
            }
        }
    }
}

} // namespace

#endif // VF_VEC_MUL_MAT_H