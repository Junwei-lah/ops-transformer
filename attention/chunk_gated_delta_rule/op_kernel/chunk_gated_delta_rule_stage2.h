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
 * \file chunk_gated_delta_rule_stage2.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE2_H
#define CHUNK_GATED_DELTA_RULE_STAGE2_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

using aT2 = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using bT2 = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using cT2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using StageTwoMT = matmul::MatmulImpl<aT2, bT2, cT2>;

struct StageTwoParams {
    GlobalTensor<float> qPrime;    // (Nv, Sp, Dk)
    GlobalTensor<float> vInner;    // (Nv, Sp, Dv)
    GlobalTensor<float> gCumExp;   // (Nv, Sp)
    GlobalTensor<float> kCumdecay; // (Nv, Sp, Dk)
    GlobalTensor<float> curState;  // (Nv, Dv, Dk)
    GlobalTensor<float> kg;
    GlobalTensor<float> attnInter;
    GM_ADDR ws;
    StageTwoMT *mm1;
    TPipe *pipe;
    ChunkGroup *cg;
    int64_t Nv;
    int64_t Nk;
    int64_t Dv;
    int64_t Dk;
    bool gOptional;
};

class Stage2 {
public:
    __aicore__ inline void Init(StageTwoParams *initParams, int32_t coreNum)
    {
        sTP_ = initParams;
        pipe_ = sTP_->pipe;
        chunkSize_ = sTP_->cg->chunkSize;
        seqLength_ = sTP_->cg->length;
        Sp_ = (seqLength_ + chunkSize_ - 1) / chunkSize_  * chunkSize_;
        chunkNum_ = Sp_ / chunkSize_;
        coreNum_ = coreNum;
        Nv_ = sTP_->Nv;
        Nk_ = sTP_->Nk;
        Dv_ = sTP_->Dv;
        Dk_ = sTP_->Dk;
        curDk_ = Ceil(Dk_, BLOCK_SIZE / sizeof(float)) * (BLOCK_SIZE / sizeof(float));
        curChunkSize_ = chunkSize_;
        gOptional_ = sTP_->gOptional;
        InitLocalBuffers();
    }

    __aicore__ inline void InitLocalBuffers()
    {
        if ASCEND_IS_AIC {
            return;
        }
        pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE,
                          chunkSize_ > Dv_ ? chunkSize_ * curDk_ * sizeof(float) : Dv_ * curDk_ * sizeof(float));
        uint64_t outQueueSize = AscendC::Std::max((uint64_t)chunkSize_ * chunkSize_ * sizeof(float),
                                                  (uint64_t)Dv_ * curDk_ * sizeof(bfloat16_t));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, outQueueSize);
        pipe_->InitBuffer(tmpBuff_,
                          chunkSize_ > Dv_ ? chunkSize_ * curDk_ * sizeof(float) : Dv_ * curDk_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t coreId = GetBlockIdx();
        if ASCEND_IS_AIV {
            coreId /= AIC_AIV_1_1;
        }
        int64_t nvPerCore = (Nv_ + coreNum_ - 1) / coreNum_;
        int64_t nvStart = coreId * nvPerCore;
        int64_t nvEnd = nvStart + nvPerCore;
        nvEnd = nvEnd > Nv_ ? Nv_ : nvEnd;
        int64_t lastChunkSize = seqLength_ % chunkSize_ == 0 ? chunkSize_ : seqLength_ % chunkSize_;
        for (int64_t nvId = nvStart; nvId < nvEnd; nvId++) {
            curChunkSize_ = chunkSize_;
            auto curState = sTP_->curState[nvId * Dv_ * Dk_];
            for (int64_t cId = 0; cId < chunkNum_; cId++) {
                int64_t length = cId * chunkSize_;
                if (cId == chunkNum_ - 1) {
                    curChunkSize_ = lastChunkSize;
                }
                if ASCEND_IS_AIV {
                    if (GetSubBlockIdx() == 0) {
                        CopyIn<float>(curState, Dv_, Dk_);
                    }
                    CrossCoreWaitFlag(0x2);
                    if (GetSubBlockIdx() == 0) {
                        CalGCumExp(curState, sTP_->gCumExp[nvId * Sp_ + length]);
                    }
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x3);  // 当前state非空，无法直接原子累加，需要覆盖写完通知AIC
                    CrossCoreWaitFlag(0x4);
                }
                if ASCEND_IS_AIC {
                    uint64_t mm_offset0 = nvId * Sp_ * Dk_ + length * Dk_;
                    uint64_t mm_offset1 = nvId * Sp_ * Dv_ + length * Dv_;
                    CalVPrime(sTP_->kCumdecay[mm_offset0], curState, sTP_->vInner[mm_offset1]);
                    CalAttnInter(sTP_->qPrime[mm_offset0], curState, sTP_->attnInter[mm_offset1]);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);   // 读完之前AIV不能写
                    CrossCoreWaitFlag(0x3);
                    CalStateNew(sTP_->vInner[mm_offset1], sTP_->kg[mm_offset0], curState);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(0x4);
                    int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
                    SetFlag<HardEvent::FIX_MTE2>(eventID);
                    WaitFlag<HardEvent::FIX_MTE2>(eventID);
                }
            }
        }
    }

    __aicore__ inline void CalGCumExp(GlobalTensor<float> stateNew, GlobalTensor<float> gCumExp)
    {
        // 刷新cache
        DataCacheCleanAndInvalid<float,
                                 CacheLine::SINGLE_CACHE_LINE,
                                 DcciDst::CACHELINE_OUT>(gCumExp[curChunkSize_ - 1]);
        float last_g_cum_exp = gOptional_? gCumExp.GetValue(curChunkSize_ - 1) : 1.0f;
        auto state_in = inQueue_.DeQue<float>();
        auto state_out = outQueue_.AllocTensor<float>();
        Muls(state_out, state_in, last_g_cum_exp, Dv_ * curDk_);
        outQueue_.EnQue(state_out);
        CopyOut<float>(stateNew, Dv_, Dk_);
        inQueue_.FreeTensor(state_in);
    }

    __aicore__ inline void CalAttnInter(GlobalTensor<float> qPrime,
                                        GlobalTensor<float> state,
                                        GlobalTensor<float> attnInter)
    {
        // q_prime @ state.transpose(0, 1)
        sTP_->mm1->SetOrgShape(curChunkSize_, Dv_, Dk_);    // MNK
        sTP_->mm1->SetSingleShape(curChunkSize_, Dv_, Dk_); // SingleCoreMNK
        sTP_->mm1->SetTensorA(qPrime, false);
        sTP_->mm1->SetTensorB(state, true);
        sTP_->mm1->IterateAll(attnInter);
        sTP_->mm1->End();
    }

    __aicore__ inline void CalVPrime(GlobalTensor<float> kCumdecay,
                                     GlobalTensor<float> state,
                                     GlobalTensor<float> vPrime)
    {
        // v_inner += k_cumdecay @ state.transpose(0, 1)
        sTP_->mm1->SetOrgShape(curChunkSize_, Dv_, Dk_);    // MNK
        sTP_->mm1->SetSingleShape(curChunkSize_, Dv_, Dk_); // SingleCoreMNK
        sTP_->mm1->SetTensorA(kCumdecay, false);
        sTP_->mm1->SetTensorB(state, true);
        sTP_->mm1->IterateAll(vPrime, 1);
        sTP_->mm1->End();
    }

    __aicore__ inline void CalStateNew(GlobalTensor<float> vInner,
                                       GlobalTensor<float> kg,
                                       GlobalTensor<float> state)
    {
        // state_out = v_new.transpose(0, 1) @ kg
        sTP_->mm1->SetOrgShape(Dv_, Dk_, curChunkSize_);    // MNK
        sTP_->mm1->SetSingleShape(Dv_, Dk_, curChunkSize_); // SingleCoreMNK
        sTP_->mm1->SetTensorA(vInner, true);
        sTP_->mm1->SetTensorB(kg, false);
        sTP_->mm1->IterateAll(state, 1);
        sTP_->mm1->End();
    }

    template <typename inType>
    __aicore__ inline void CopyIn(GlobalTensor<inType> tmpGM, int32_t row, int32_t col)
    {
        LocalTensor<inType> inLocal = inQueue_.AllocTensor<inType>();
        DataCopyExtParams inParams{static_cast<uint16_t>(row),
                                   static_cast<uint32_t>(col * sizeof(inType)),                // 非对齐情况需要补0
                                   static_cast<uint32_t>(0),
                                   0, 0};
        int padding = Ceil(col, BLOCK_SIZE / sizeof(inType)) * (BLOCK_SIZE / sizeof(inType)) - col;
        DataCopyPadExtParams<inType> copyPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(inLocal, tmpGM, inParams, copyPadParams);
        inQueue_.EnQue(inLocal);
    }

    template <typename outType>
    __aicore__ inline void CopyOut(GlobalTensor<outType> tmpGM, int32_t row, int32_t col, bool setAtomic = false)
    {
        auto outLocal = outQueue_.DeQue<outType>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(row);
        copyParams.blockLen = static_cast<uint32_t>(col * sizeof(outType));
        copyParams.srcStride = static_cast<uint32_t>(0);
        copyParams.dstStride = static_cast<uint32_t>((0) * sizeof(outType));
        if (setAtomic) {
            SetAtomicAdd<float>();
        }
        DataCopyPad(tmpGM, outLocal, copyParams);
        if (setAtomic) {
            SetAtomicNone();
        }
        outQueue_.FreeTensor(outLocal);
    }

private:
    StageTwoParams *sTP_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    int64_t Nk_;
    int64_t Nv_;
    int64_t Dk_;
    int64_t Dv_;
    int64_t seqLength_;
    int32_t chunkSize_;
    int32_t curChunkSize_;
    int32_t curDk_;
    int64_t Sp_;
    int32_t chunkNum_;
    int32_t coreNum_;
    bool gOptional_;
};
} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE2_H