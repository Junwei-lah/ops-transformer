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
 * \file chunk_gated_delta_rule_stage3.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE3_H
#define CHUNK_GATED_DELTA_RULE_STAGE3_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

using aT3 = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using bT3 = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using cT3 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using StageThreeMT = matmul::MatmulImpl<aT3, bT3, cT3>;

struct StageThreeParams {
    GlobalTensor<float> qkt;       // (Nv, Sp, Dk)
    GlobalTensor<float> gCumExp;           // (Nv, Sp)
    GlobalTensor<float> attnInter;
    GlobalTensor<float> vInner;
    GlobalTensor<float> maskTensor;
    GM_ADDR ws;
    GlobalTensor<bfloat16_t> attnOut;
    StageThreeMT *mm3;
    TPipe *pipe;
    ChunkGroup *cg;
    float scale;
    int64_t Nv;
    int64_t Nk;
    int64_t Dv;
    int64_t Dk;
    bool gOptional;
};

class Stage3 {
public:
    __aicore__ inline void Init(StageThreeParams *initParams, int32_t coreNum)
    {
        coreId_ = GetBlockIdx();
        sTP_ = initParams;
        pipe_ = sTP_->pipe;
        chunkSize_ = sTP_->cg->chunkSize;
        seqLength_ = sTP_->cg->length;
        Sp_ = (seqLength_ + chunkSize_ - 1) / chunkSize_  * chunkSize_;
        chunkNum_ = (seqLength_ + chunkSize_ - 1) / chunkSize_ ;
        coreNum_ = coreNum;
        Nv_ = sTP_->Nv;
        Nk_ = sTP_->Nk;
        Dv_ = sTP_->Dv;
        Dk_ = sTP_->Dk;
        paddedDv_ = Ceil(Dv_, BLOCK_SIZE / sizeof(bfloat16_t)) * (BLOCK_SIZE / sizeof(bfloat16_t));
        gOptional_ = sTP_->gOptional;
        uint64_t workSpaceOffset = 0;
        tmpGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams->ws + workSpaceOffset +
                                                                coreNum_ * chunkSize_ * chunkSize_ * sizeof(float)));
        if ASCEND_IS_AIC {
            return;
        }
        coreId_ /= AIC_AIV_1_1;
        if (GetSubBlockIdx() == 1) {
            return;
        }
        uint64_t inQueueSize = static_cast<uint64_t>(chunkSize_) *
                               AscendC::Std::max((int64_t)chunkSize_, paddedDv_) * sizeof(float);
        pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, inQueueSize);
        pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, chunkSize_ > paddedDv_ ?
                          chunkSize_ * chunkSize_ * sizeof(float) : chunkSize_ * paddedDv_ * sizeof(float));
        pipe_->InitBuffer(tmpBuff_, (STAGE3_BUFFER_COUNT * chunkSize_ * chunkSize_ * sizeof(float)));
        uint64_t buffOffset = 0;
        uint64_t tmpOffset = chunkSize_ * chunkSize_;
        tmpBuffer1_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpOffset), buffOffset);
        buffOffset += tmpOffset * sizeof(float);
        tmpBuffer2_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpOffset), buffOffset);
        buffOffset += tmpOffset* sizeof(float);
        maskBuffer_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpOffset), buffOffset);

        // 搬入mask
        DataCopyExtParams inParams{static_cast<uint16_t>(chunkSize_),
                                   static_cast<uint32_t>(chunkSize_ * sizeof(float)),
                                   0, 0, 0};
        DataCopyPadExtParams<float> copyPadParams{false, 0, 0, 0};
        DataCopyPad(maskBuffer_, sTP_->maskTensor, inParams, copyPadParams);
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);
    }

    __aicore__ inline void Process()
    {
        int64_t totalChunks = Nv_ * chunkNum_;  // Nv Nc 融合
        int64_t chunksPerCore = (totalChunks + coreNum_ -1) / coreNum_;
        int64_t lastChunkSize = seqLength_ % chunkSize_ == 0 ? chunkSize_ : seqLength_ % chunkSize_;
        int64_t startChunk = coreId_ * chunksPerCore;
        int64_t endChunk = startChunk + chunksPerCore > totalChunks ? totalChunks : startChunk + chunksPerCore;
        for (int64_t idx = startChunk; idx < endChunk; idx++) {
            int64_t nvId = idx / chunkNum_;
            int64_t chunkId = idx % chunkNum_;
            int64_t chunkPos = chunkId * chunkSize_;    // 当前chunk起始位置
            curChunkSize_ = (chunkId == chunkNum_ - 1) ? lastChunkSize : chunkSize_; // 尾块
            if ASCEND_IS_AIV {
                if (GetSubBlockIdx() == 0) {
                    CalMaskedQKT(tmpGM_[coreId_ * chunkSize_ * chunkSize_], nvId, chunkPos);
                }
                CrossCoreSetFlag<0x2, PIPE_MTE3>(0x4);
                CrossCoreWaitFlag(0x3);
                if (GetSubBlockIdx() == 0) {
                    CalAttnOut(sTP_->attnInter[nvId * Sp_ * Dv_ + chunkPos * Dv_],
                               sTP_->attnOut[chunkPos * Nv_ * Dv_ + nvId * Dv_]);
                }
            }

            if ASCEND_IS_AIC {
                CrossCoreWaitFlag(0x4);
                AICProcess(tmpGM_[coreId_ * chunkSize_ * chunkSize_],
                           sTP_->vInner[nvId * Sp_ * Dv_ + chunkPos * Dv_],
                           sTP_->attnInter[nvId * Sp_ * Dv_ + chunkPos * Dv_]);
                CrossCoreSetFlag<0x2, PIPE_FIX>(0x3);
            }
        }
    }
    
    __aicore__ inline void CalMaskedQKT(GlobalTensor<float> outGM, int nvId, int chunkPos)
    {
        // chunkSize 大小进行自动补齐
        if (gOptional_) {
            AlignedCopyIn(sTP_->gCumExp[nvId * Sp_ + chunkPos], 1, curChunkSize_);  // 自动补齐
            auto g_cum_exp = inQueue_.DeQue<float>();
            const uint32_t srcShape1[] = {static_cast<uint32_t>(chunkSize_), static_cast<uint32_t>(1)};
            const uint32_t srcShape2[] = {static_cast<uint32_t>(1), static_cast<uint32_t>(chunkSize_)};
            const uint32_t dstShape[] = {static_cast<uint32_t>(chunkSize_),
                                         static_cast<uint32_t>(chunkSize_)};
            Broadcast<float, BROADCAST_AXIS, 1>(tmpBuffer1_, g_cum_exp, dstShape, srcShape1);
            Broadcast<float, BROADCAST_AXIS, 0>(tmpBuffer2_, g_cum_exp, dstShape, srcShape2);
            PipeBarrier<PIPE_V>();
            Div(tmpBuffer1_, tmpBuffer1_, tmpBuffer2_, curChunkSize_ * chunkSize_);
            inQueue_.FreeTensor(g_cum_exp);
        } else {
            Duplicate(tmpBuffer1_, static_cast<float>(1.0f), curChunkSize_ * chunkSize_);
            PipeBarrier<PIPE_V>();
        }
 
        // qkt
        AlignedCopyIn(sTP_->qkt[nvId * Sp_ * chunkSize_ + chunkPos * chunkSize_], curChunkSize_, curChunkSize_);
        auto qkt = inQueue_.DeQue<float>();
        auto scale_qkt = outQueue_.AllocTensor<float>();
        Muls(scale_qkt, qkt, sTP_->scale, curChunkSize_ * chunkSize_);
        Mul(scale_qkt, scale_qkt, tmpBuffer1_, curChunkSize_ * chunkSize_);
        inQueue_.FreeTensor(qkt);
        // mask
        Mul(scale_qkt, scale_qkt, maskBuffer_, curChunkSize_ * chunkSize_);
        outQueue_.EnQue(scale_qkt);
        AlignedCopyOut(outGM, curChunkSize_, curChunkSize_);
    }

    __aicore__ inline void CalAttnOut(GlobalTensor<float> inTensor, GlobalTensor<bfloat16_t> outTensor)
    {
        // 匹配到BF16对齐, 余数为8以上时自动对齐无需额外偏移
        int64_t dstStride = 0;
        if (Dv_ % BLOCK_BF16_NUM != 0) {
            dstStride = Dv_ % BLOCK_BF16_NUM > BLOCK_FLOAT_NUM ? 0 : BLOCK_FLOAT_NUM;
        }
        LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();
        DataCopyExtParams inParams{static_cast<uint16_t>(curChunkSize_),
                                   static_cast<uint32_t>(Dv_ * sizeof(float)),
                                   static_cast<uint32_t>(0),
                                   static_cast<uint32_t>((dstStride) * sizeof(float) / BLOCK_SIZE),
                                   0};
        DataCopyPadExtParams<float> copyPadParams{false, 0, 0, 0};
        DataCopyPad(inLocal, inTensor, inParams, copyPadParams);
        inQueue_.EnQue(inLocal);
        auto fp32AttnIn = inQueue_.DeQue<float>();
        auto attnOut = outQueue_.AllocTensor<bfloat16_t>();
        Cast(attnOut, fp32AttnIn, RoundMode::CAST_RINT, curChunkSize_ * paddedDv_);
        inQueue_.FreeTensor(fp32AttnIn);  // Cast 完成后立即释放
        outQueue_.EnQue(attnOut);
        auto outLocal = outQueue_.DeQue<bfloat16_t>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(curChunkSize_);
        copyParams.blockLen = static_cast<uint32_t>(Dv_ * sizeof(bfloat16_t));
        copyParams.srcStride = static_cast<uint32_t>((0) * sizeof(bfloat16_t));
        copyParams.dstStride = static_cast<uint32_t>((Nv_ * Dv_ - Dv_) * sizeof(bfloat16_t));
        DataCopyPad(outTensor, outLocal, copyParams);
        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void AICProcess(GlobalTensor<float>tmpGM,
                                      GlobalTensor<float>vInner,
                                      GlobalTensor<float>attnInter)
    {
        // masked_qkt @ v_inner
        sTP_->mm3->SetOrgShape(curChunkSize_, Dv_, curChunkSize_);    // MNK
        sTP_->mm3->SetSingleShape(curChunkSize_, Dv_, curChunkSize_); // SingleCoreMNK
        sTP_->mm3->SetTensorA(tmpGM);
        sTP_->mm3->SetTensorB(vInner);
        sTP_->mm3->IterateAll(attnInter, 1);
        sTP_->mm3->End();
    }

    template <typename inType>
    __aicore__ inline void AlignedCopyIn(GlobalTensor<inType> tmpGM, int32_t row, int32_t col)
    {
        LocalTensor<inType> inLocal = inQueue_.AllocTensor<inType>();
        // 非对齐拷入会自动对齐, 然后离散拷入UB
        int paddingCol = Ceil(col, BLOCK_SIZE / sizeof(float)) * (BLOCK_SIZE / sizeof(float));
        DataCopyExtParams inParams{static_cast<uint16_t>(row),
                                   static_cast<uint32_t>(col * sizeof(inType)),
                                   static_cast<uint32_t>(0),
                                   static_cast<uint32_t>((chunkSize_ - paddingCol) * sizeof(float) / BLOCK_SIZE), 0};
        DataCopyPadExtParams<inType> copyPadParams{false, 0, 0, 0};
        DataCopyPad(inLocal, tmpGM, inParams, copyPadParams);
        inQueue_.EnQue(inLocal);
    }

    template <typename outType>
    __aicore__ inline void AlignedCopyOut(GlobalTensor<outType> tmpGM, int32_t row, int32_t col)
    {
        auto outLocal = outQueue_.DeQue<outType>();
        int paddingCol = Ceil(col, BLOCK_SIZE / sizeof(float)) * (BLOCK_SIZE / sizeof(float));
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(row);
        copyParams.blockLen = static_cast<uint32_t>(col * sizeof(outType));
        copyParams.srcStride = static_cast<uint32_t>((chunkSize_ - paddingCol) * sizeof(float) / BLOCK_SIZE);
        copyParams.dstStride = static_cast<uint32_t>((0) * sizeof(outType));
        DataCopyPad(tmpGM, outLocal, copyParams);
        outQueue_.FreeTensor(outLocal);
    }

private:
    StageThreeParams *sTP_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    GlobalTensor<float> tmpGM_;
    LocalTensor<float> tmpBuffer1_;
    LocalTensor<float> tmpBuffer2_;
    LocalTensor<float> maskBuffer_;
    int64_t seqLength_;
    int64_t Sp_;
    int64_t Nv_;
    int64_t Nk_;
    int64_t Dv_;
    int64_t Dk_;
    int64_t paddedDv_;
    int32_t chunkNum_;
    int32_t coreNum_;
    int32_t curChunkSize_;
    int32_t chunkSize_;
    int32_t coreId_;
    bool gOptional_;
};

} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE3_H