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
 * \file moe_distribute_v2_quant.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_V2_QUANT_H
#define MOE_DISTRIBUTE_COMBINE_V2_QUANT_H

#if __has_include("../moe_distribute_dispatch_v2/check_winsize.h")
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_constant.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_base.h"
#else
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_constant.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_base.h"
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if __has_include("../moe_distribute_dispatch_v2/quantize_functions.h")
#include "../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif
#endif

namespace Mc2Kernel {
using namespace AscendC;
using namespace MoeDistributeV2Base;

template <typename ExpandXType, typename XType, typename ExpandIdxType, bool IsNeedReduceScatter,
    uint8_t QuantMode, bool HasAddRmsNorm>
class MoeDistributeCombineQuant{
public:
    float scaleValFloat_;
    LocalTensor<half> fp16CastTensor_;
    LocalTensor<float> absFloatTensor_;
    LocalTensor<float> reduceMaxFloatTensor_;
    LocalTensor<float> scaleDivFloatTensor_;
    LocalTensor<float> scaleDupLocalTensor_;
    LocalTensor<float> winTpSendCountFloatTensor_;
    LocalTensor<float> floatLocalTemp_;
    uint32_t axisH_{0};
    uint32_t mask_{0};
    uint32_t repeatNum_{0};
    uint32_t hAlign32Size_{0};
    uint32_t quantScaleNum_{0};
    LocalTensor<int8_t> castLocalTensor_;
    LocalTensor<XType> scaleDivTensor_;

    __aicore__ inline MoeDistributeCombineQuant() = default;

    __aicore__ inline void SetQuantInitParams(LocalTensor<float> winTpSendCountFloatTensor,
        LocalTensor<half> fp16CastTensor, LocalTensor<float> absFloatTensor,
        LocalTensor<float> reduceMaxFloatTensor, LocalTensor<float> scaleDupLocalTensor)
    {
        winTpSendCountFloatTensor_ = winTpSendCountFloatTensor;
        floatLocalTemp_ = winTpSendCountFloatTensor;
        fp16CastTensor_ = fp16CastTensor;
        absFloatTensor_ = absFloatTensor;
        reduceMaxFloatTensor_ = reduceMaxFloatTensor;
        scaleDupLocalTensor_ = scaleDupLocalTensor;
    }

    __aicore__ inline void SetDeQuantInitParams(LocalTensor<half> fp16CastTensor, LocalTensor<float> absFloatTensor,
        LocalTensor<float> scaleDupLocalTensor, LocalTensor<float> scaleDivFloatTensor)
    {
        fp16CastTensor_ = fp16CastTensor;
        absFloatTensor_ = absFloatTensor;
        scaleDupLocalTensor_ = scaleDupLocalTensor;
        scaleDivFloatTensor_ = scaleDivFloatTensor;
    }

     __aicore__ inline void QuantInit(uint32_t &scaleNum_, uint32_t &hExpandXAlign32Size_, uint32_t &hExpandXAlignSize_,
        uint32_t &scaleNumAlignSize_, uint32_t &hFloatAlign256Size_, uint32_t &tokenScaleCnt_, uint32_t axisH)
    {
        axisH_ = axisH;
        if constexpr (QuantMode == INT8_COMM_QUANT) {
            hAlign32Size_ = Ceil(axisH_, UB_ALIGN) * UB_ALIGN;
            scaleValFloat_ = static_cast<float>(1.0f / SCALE_PARAM);
            uint32_t scaleGranu = static_cast<uint32_t>(UB_ALIGN / sizeof(float)); // 计算每个block得到的reducemax结果数量
            quantScaleNum_ = (hExpandXAlign32Size_ / sizeof(ExpandXType)) / scaleGranu; // 得到有效scale的个数
            scaleNum_ = quantScaleNum_;
            hExpandXAlignSize_ = hExpandXAlign32Size_;
            scaleNumAlignSize_ = Ceil(scaleNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
            repeatNum_ = static_cast<uint32_t>(hFloatAlign256Size_ / ALIGNED_LEN); // 每次256b参与计算
            mask_ = static_cast<uint32_t>(ALIGNED_LEN / sizeof(float));
            tokenScaleCnt_ = hAlign32Size_ / sizeof(ExpandXType) + quantScaleNum_; // int8_align + scale有效个数
        }
        #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        else if constexpr(QuantMode == MXFP8_E5M2_COMM_QUANT || QuantMode == MXFP8_E4M3_COMM_QUANT) {
            hExpandXAlignSize_ = Align128(axisH) * sizeof(ExpandXType);
            quantScaleNum_ = Align2(Ceil32(axisH));
            scaleNum_ = quantScaleNum_;
            scaleNumAlignSize_ = Align128(scaleNum_) * sizeof(ExpandXType) * BUFFER_NUM; // 双搬
            tokenScaleCnt_ = Align256(axisH) / sizeof(ExpandXType) + scaleNum_;
        }
        #endif
    }

    __aicore__ inline void Int8QuantProcess(LocalTensor<ExpandXType> &outLocal, LocalTensor<ExpandXType> &inLocal)
    {
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        castLocalTensor_ = outLocal.template ReinterpretCast<int8_t>(); // 长度为int8H_Align + scaleNum
        scaleDivTensor_ = castLocalTensor_[hAlign32Size_].template ReinterpretCast<ExpandXType>(); // 偏移前面的int8

        Cast(winTpSendCountFloatTensor_, inLocal, RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        Abs(absFloatTensor_, winTpSendCountFloatTensor_, axisH_); // absFloatTensor_ align到256并写0，支持ReduceMax与Brcb
        PipeBarrier<PIPE_V>();
        BlockReduceMax(reduceMaxFloatTensor_, absFloatTensor_, repeatNum_, mask_, 1, 1, BLOCK_NUM); // 32->1 256->8
        PipeBarrier<PIPE_V>();
        Muls(reduceMaxFloatTensor_, reduceMaxFloatTensor_, scaleValFloat_, quantScaleNum_); // 有效个数
        PipeBarrier<PIPE_V>();
        Cast(scaleDivTensor_, reduceMaxFloatTensor_, RoundMode::CAST_RINT, quantScaleNum_); // 有效个数
        PipeBarrier<PIPE_V>();
        Brcb(scaleDupLocalTensor_, reduceMaxFloatTensor_, repeatNum_, {1, BLOCK_NUM}); // 一次256
        PipeBarrier<PIPE_V>();
        Div(winTpSendCountFloatTensor_, winTpSendCountFloatTensor_, scaleDupLocalTensor_, axisH_); // 有效个数
        PipeBarrier<PIPE_V>();
        Cast(fp16CastTensor_, winTpSendCountFloatTensor_, RoundMode::CAST_RINT, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(castLocalTensor_, fp16CastTensor_, RoundMode::CAST_RINT, axisH_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
    }

    __aicore__ inline void Int8DequantProcess(LocalTensor<XType>& inLocal, LocalTensor<XType> &outLocal)
    {
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        castLocalTensor_ = inLocal.template ReinterpretCast<int8_t>();
        scaleDivTensor_ = inLocal[hAlign32Size_ / INT8_DIVIVE];
        SyncFunc<AscendC::HardEvent::S_V>();
        Cast(scaleDivFloatTensor_, scaleDivTensor_, RoundMode::CAST_NONE, quantScaleNum_);
        Cast(fp16CastTensor_, castLocalTensor_, RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(absFloatTensor_, fp16CastTensor_, RoundMode::CAST_NONE, axisH_);
        Brcb(scaleDupLocalTensor_, scaleDivFloatTensor_, repeatNum_, {1, BLOCK_NUM});
        PipeBarrier<PIPE_V>();
        Mul(absFloatTensor_, absFloatTensor_, scaleDupLocalTensor_, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(outLocal, absFloatTensor_, RoundMode::CAST_RINT, axisH_);
        PipeBarrier<PIPE_V>();
    }

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    __aicore__ inline void QuantMxFp8(LocalTensor<ExpandXType>& outLocal, LocalTensor<ExpandXType>& inLocal)
    {
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        uint32_t mxScaleNum = Align2(Ceil32(axisH_));
        __ubuf__ ExpandXType* srcAddr = (__ubuf__ ExpandXType*)inLocal.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)floatLocalTemp_.GetPhyAddr();
        __ubuf__ uint16_t* halfScaleLocalAddr = (__ubuf__ uint16_t*)floatLocalTemp_[Align32(mxScaleNum)].GetPhyAddr();
        __ubuf__ int8_t* outLocalAddr = (__ubuf__ int8_t*)outLocal.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleLocalAddr =
            (__ubuf__ uint16_t*)outLocal[Align256<uint32_t>(axisH_) / INT8_DIVIVE].GetPhyAddr();
        quant::ComputeMaxExp(srcAddr, maxExpAddr, axisH_); // 计算最大Exp
        if constexpr (QuantMode == MXFP8_E5M2_COMM_QUANT) {
            // 计算scales并填充
            quant::ComputeScale<fp8_e5m2_t>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
            quant::ComputeFp8Data<ExpandXType, fp8_e5m2_t,
                AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
                srcAddr, halfScaleLocalAddr, outLocalAddr, axisH_); // 计算量化后的expandx并填充
        } else if constexpr (QuantMode == MXFP8_E4M3_COMM_QUANT) {
            // 计算scales并填充
            quant::ComputeScale<fp8_e4m3fn_t>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
            quant::ComputeFp8Data<ExpandXType, fp8_e4m3fn_t,
                AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
                srcAddr, halfScaleLocalAddr, outLocalAddr, axisH_); // 计算量化后的expandx并填充
        }
    }
    
    template <typename T>
    __aicore__ inline void DeQuantMxFp8(LocalTensor<XType>& inLocal, LocalTensor<float>& sumTensor)
    {
        LocalTensor<T> castFp8LocalTensor_ = inLocal.template ReinterpretCast<T>();
        // bf16/fp16量化为mxfp8后，字节差2倍
        LocalTensor<fp8_e8m0_t> scaleDivFp8Tensor_ =
            inLocal[Align256<uint32_t>(axisH_) / INT8_DIVIVE].template ReinterpretCast<fp8_e8m0_t>();
        __ubuf__ bfloat16_t *dyScaleBf16Ptr = (__ubuf__ bfloat16_t *)scaleDivFloatTensor_.GetPhyAddr();
        __ubuf__ float *dyScaleFp32Ptr = (__ubuf__ float *)scaleDupLocalTensor_.GetPhyAddr(); // 大小是h*4字节
        __ubuf__ fp8_e8m0_t *srcPtr0 = (__ubuf__ fp8_e8m0_t *)scaleDivFp8Tensor_.GetPhyAddr();
        __ubuf__ T *tokenPtr0 = (__ubuf__ T *)castFp8LocalTensor_.GetPhyAddr();
        __ubuf__ float *sumDstPtr = (__ubuf__ float *)sumTensor.GetPhyAddr();
        uint32_t bf16RepeatSize = quant::GetVRegSizeDispatch() / sizeof(bfloat16_t);
        uint32_t fp32RepeatSize = quant::GetVRegSizeDispatch() / sizeof(float);
        uint16_t repeatTimes = Ceil(quantScaleNum_, bf16RepeatSize);
        uint16_t fp32RepeatTimes = Ceil(axisH_, fp32RepeatSize);
        uint16_t repeatTimes2 = Ceil(quantScaleNum_ * INT8_DIVIVE, fp32RepeatSize);
        uint32_t quantCount2 = quantScaleNum_ * INT8_DIVIVE;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<fp8_e8m0_t> vSrcReg;
            AscendC::MicroAPI::RegTensor<T> tokenSrcReg;
            AscendC::MicroAPI::RegTensor<float> tokenFp32SrcReg;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vDstReg;
            AscendC::MicroAPI::RegTensor<bfloat16_t> dyScaleBf16Reg;
            AscendC::MicroAPI::RegTensor<float> dyScaleFp32Reg;
            AscendC::MicroAPI::RegTensor<float> sumDstReg;
            AscendC::MicroAPI::RegTensor<float> sumLocalDstReg;
            static constexpr AscendC::MicroAPI::CastTrait FP82BF16CastTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
            static constexpr AscendC::MicroAPI::CastTrait FP162FP32CastTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
            AscendC::MicroAPI::MaskReg maskReg;
            AscendC::MicroAPI::MaskReg maskReg1;
            AscendC::MicroAPI::MaskReg maskReg2;

            for (uint16_t i = 0; i < repeatTimes; i++) {
                maskReg = AscendC::MicroAPI::UpdateMask<bfloat16_t>(quantScaleNum_);
                // 一次搬128个u8 unpack成128个u16
                MicroAPI::DataCopy<fp8_e8m0_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(vSrcReg,
                    srcPtr0 + i * bf16RepeatSize);
                MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, FP82BF16CastTraitZero>(vDstReg, vSrcReg, maskReg);
                MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                    dyScaleBf16Ptr + i * bf16RepeatSize * INT8_DIVIVE, vDstReg, vDstReg, maskReg); // bf16，双搬出元素
            }
            MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < repeatTimes2; i++) {
                maskReg1 = AscendC::MicroAPI::UpdateMask<float>(quantCount2);
                MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(dyScaleBf16Reg,
                    dyScaleBf16Ptr + i * fp32RepeatSize); // 128/2=64 搬入64个u16 unpack成64个u32
                MicroAPI::Cast<float, bfloat16_t, FP162FP32CastTraitZero>(dyScaleFp32Reg, dyScaleBf16Reg, maskReg1);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(
                    dyScaleFp32Ptr + i * fp32RepeatSize * INT8_DIVIVE, dyScaleFp32Reg, dyScaleFp32Reg, maskReg1);
            }

            MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < fp32RepeatTimes; i++) {
                maskReg2 = AscendC::MicroAPI::UpdateMask<float>(axisH_);
                // 广播4B->32B，8倍
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(dyScaleFp32Reg, dyScaleFp32Ptr + i * 8);
                // 接收到的token float8_e5m2_t
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(tokenSrcReg,
                    tokenPtr0 + i * fp32RepeatSize);
                MicroAPI::Cast<float, T, FP82BF16CastTraitZero>(tokenFp32SrcReg, tokenSrcReg, maskReg2);
                MicroAPI::Mul(sumLocalDstReg, dyScaleFp32Reg, tokenFp32SrcReg, maskReg2); // token与量化参数相乘
                MicroAPI::DataCopy(sumDstPtr + i * fp32RepeatSize, sumLocalDstReg, maskReg2); // 最后搬出 float类型
            }
        }
    }
#endif
    __aicore__ inline void QuantProcess(LocalTensor<ExpandXType>& outLocal, LocalTensor<ExpandXType>& inLocal)
    {
        if constexpr (QuantMode == INT8_COMM_QUANT) {
            Int8QuantProcess(outLocal, inLocal);
        }
        #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        else if constexpr (QuantMode == MXFP8_E5M2_COMM_QUANT || QuantMode == MXFP8_E4M3_COMM_QUANT) {
            QuantMxFp8(outLocal, inLocal);
        }
        #endif
    }
    __aicore__ inline void DeQuantProcess(LocalTensor<XType>& inLocal, LocalTensor<XType>& outLocal,
        LocalTensor<float>& sumTensor)
    {
        if constexpr (QuantMode == INT8_COMM_QUANT) {
            Int8DequantProcess(inLocal, outLocal);
        }
        #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        else if constexpr (QuantMode == MXFP8_E5M2_COMM_QUANT) {
            DeQuantMxFp8<fp8_e5m2_t>(inLocal, sumTensor);
        } else if constexpr (QuantMode == MXFP8_E4M3_COMM_QUANT) {
            DeQuantMxFp8<fp8_e4m3fn_t>(inLocal, sumTensor);
        }
        #endif
    }
};
}
#endif // MOE_DISTRIBUTE_V2_QUANT_H