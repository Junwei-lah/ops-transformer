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
 * \file chunk_gated_delta_rule_stage1.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE1_H
#define CHUNK_GATED_DELTA_RULE_STAGE1_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

using aT1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using bT1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using cT1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using StageOneMT = matmul::MatmulImpl<aT1, bT1, cT1>;

constexpr uint64_t UB_REST_BYTES = 100 * 1024;  // 100KB
constexpr uint64_t INVERSE_SHAPE = 32;          // 对角块边长
constexpr uint64_t INVERSE_COUNT = 5;           // 求逆所需空间
constexpr uint32_t ALIGN_SIZE = 16;

// Matmul 形状参数结构体
struct MatmulShapeParams {
    uint64_t m;    // 原始 M 维度
    uint64_t n;    // 原始 N 维度
    uint64_t k;    // 原始 K 维度
    uint64_t sm;   // 单次计算 M 维度
    uint64_t sn;   // 单次计算 N 维度
    uint64_t sk;   // 单次计算 K 维度
};

struct GDRStageOneInitParams {
    // input
    GlobalTensor<bfloat16_t> query;     // (T, Nk, Dk)
    GlobalTensor<bfloat16_t> key;       // (T, Nk, Dk)
    GlobalTensor<bfloat16_t> value;     // (T, Nv, Dv)
    GlobalTensor<bfloat16_t> beta;      // (T, Nv)
    GlobalTensor<float> g;              // (T, Nv)
    // ouput
    GlobalTensor<float> gCumExp;        // (Nv, cg_len)
    GlobalTensor<float> kCumdecay;      // (Nv, cg_len, Dk)
    GlobalTensor<float> vInner;         // (Nv, cg_len, Dv)
    GlobalTensor<float> qPrime;         // (Nv, cg_len, Dk)
    GlobalTensor<float> kG;             // (Nv, cg_len, Dk)
    GlobalTensor<float> qK;             // (Nv, cg_len, C)
    // other
    GM_ADDR ws;
    GlobalTensor<float> stageOneMask;   // (Nv, cg_len, C)
    ChunkGroup cg;
    bool gOptional;
};

class Stage1 {
public:
    __aicore__ inline Stage1(StageOneMT &mmFp32) : mmFp32(mmFp32) {}
    __aicore__ inline void SetGlobalTensors(const GDRStageOneInitParams &initParams)
    {
        queryBaseGm_ = initParams.query;
        keyBaseGm_ = initParams.key;
        valueBaseGm_ = initParams.value;
        betaBaseGm_ = initParams.beta;

        outGCumExpBaseGm_ = initParams.gCumExp;
        outKCumdecayBaseGm_ = initParams.kCumdecay;
        outVInnerBaseGm_ = initParams.vInner;
        outQPrimeBaseGm_ = initParams.qPrime;
        outKgBaseGm_ = initParams.kG;
        outQkBaseGm_ = initParams.qK;
        stageOneMask_ = initParams.stageOneMask;

        if (gOptional_) {
            gBaseGm_ = initParams.g;
        }

        uint64_t workSpaceOffset = 0;
        gBKWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                  coreIdx_ * chunkSize_ * dk_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * dk_ * sizeof(float);
        kkWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                 coreIdx_ * chunkSize_ * chunkSize_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * chunkSize_ * sizeof(float);
        vBetaWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                    coreIdx_ * chunkSize_ * dv_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * dv_ * sizeof(float);
        attnWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                   coreIdx_ * chunkSize_ * chunkSize_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * chunkSize_ * sizeof(float);
        queryContinousGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                          coreIdx_ * chunkSize_ * dk_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * dk_ * sizeof(float);
        keyContinousGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                         coreIdx_ * chunkSize_ * dk_ * sizeof(float)));
    }

    __aicore__ inline void InitLocalBuffers()
    {
        uint32_t maxLen = AscendC::Std::max(AscendC::Std::max(dvAligned_ / 2, dkAligned_ / 2), chunkSize_);
        pipe_->InitBuffer(fp32InQueue_, BUFFER_NUM_ONE, chunkSize_ * maxLen * sizeof(float));
        pipe_->InitBuffer(fp32OutQueue_, BUFFER_NUM_ONE, chunkSize_ * maxLen * sizeof(float));
        if (gOptional_) {
            pipe_->InitBuffer(gOutQueue_, BUFFER_NUM_ONE, chunkSize_ * sizeof(float));
        }

        pipe_->InitBuffer(tmpBuff_, UB_REST_BYTES);
        uint32_t buffOffset = 0;
        betaUbBfloat16_ = tmpBuff_.GetWithOffset<bfloat16_t>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(bfloat16_t);
        
        gCumUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_), buffOffset);
        buffOffset += chunkSize_ * sizeof(float);

        gBUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(float);

        gEndBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(float);

        betaUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(float);

        gBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * maxLen), buffOffset);
        gammaUbFloat_ = gBroadUbFloat_;
        kUbFloat_ = gBroadUbFloat_;
        valueUbFloat_ = gBroadUbFloat_;
        qUbFloat_ = gBroadUbFloat_;
        buffOffset += chunkSize_ * maxLen  * sizeof(float);
        
        gTransBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * maxLen), buffOffset);
        attnUbFloat_ = gTransBroadUbFloat_;
        gCumExpBroadUbFloat_ = gTransBroadUbFloat_;
        buffOffset += chunkSize_ * maxLen * sizeof(float);

        qUbFloatCon_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_ * dkAligned_), buffOffset);
        buffOffset += halfChunkSize_ * dkAligned_ * sizeof(float);

        kUbFloatCon_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_ * dkAligned_), buffOffset);
        buffOffset += halfChunkSize_ * dkAligned_ * sizeof(float);

        inverseUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_ * halfChunkSize_ *
                                                        INVERSE_COUNT), buffOffset);
        buffOffset += halfChunkSize_ * halfChunkSize_ * INVERSE_COUNT * sizeof(float);

        colBuffer_ = tmpBuff_.GetWithOffset<uint32_t>(static_cast<uint32_t>(INVERSE_SHAPE), buffOffset);
        buffOffset += INVERSE_SHAPE * sizeof(uint32_t);

        gatherOffsetFp32_ = tmpBuff_.GetWithOffset<uint32_t>(static_cast<uint32_t>(chunkSize_), buffOffset);
        buffOffset += chunkSize_ * sizeof(uint32_t);

        gatherOffsetBf16_ = tmpBuff_.GetWithOffset<uint32_t>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
    }

    __aicore__ inline void InitGatherBuffer()
    {
        for (uint32_t i = 0; i < chunkSize_; ++i) {
            gatherOffsetFp32_.SetValue(i, i * BLOCK_SIZE);
        }
        for (uint32_t i = 0; i < halfChunkSize_; ++i) {
            gatherOffsetBf16_.SetValue(i, i * BLOCK_SIZE);
        }
        for (uint32_t i = 0; i < INVERSE_SHAPE; ++i) {
            colBuffer_.SetValue<uint32_t>(i, (i * chunkSize_) * sizeof(float));
        }
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }

    __aicore__ inline void Init(const GDRStageOneInitParams &initParams, TPipe *pipe,
                                const ChunkGatedDeltaRuleTilingData *tilingData)
    {
        pipe_ = pipe;
        tiling_ = tilingData;
        nk_ = tiling_->nk;
        nv_ = tiling_->nv;
        chunkSize_ = tiling_->chunkSize;
        dk_ = tiling_->dk;
        dv_ = tiling_->dv;
        dkAligned_ = (dk_ + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
        dvAligned_ = (dv_ + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
        scale_ = tiling_->scale;
        coreNum_ = tiling_->aiCoreNum;
        cg_ = initParams.cg;
        gOptional_ = initParams.gOptional;
        vRowStride_ = nv_ * dv_;
        numChunk_ = (cg_.length + chunkSize_ - 1) / chunkSize_;
        subBlockIdx_ = GetSubBlockIdx();
        halfChunkSize_ = chunkSize_ / TASK_RATIO;
        validLen_ = chunkSize_;
        subValidRows_ = halfChunkSize_;
        subOffset_ = subBlockIdx_ * halfChunkSize_;
        coreIdx_ = GetBlockIdx();
        if ASCEND_IS_AIV{
            coreIdx_ /= TASK_RATIO;
            InitLocalBuffers();
            InitGatherBuffer();
        }
        SetGlobalTensors(initParams);
    }

    __aicore__ inline void Process()
    {
        uint32_t totalChunk = nv_ * numChunk_;
        uint32_t tailChunkNum = totalChunk / coreNum_;   // tail核处理的块数
        uint32_t formerChunkNum = tailChunkNum + 1;      // former核处理的块数
        uint32_t formerCoreNum = totalChunk % coreNum_;  // former核数量
        uint32_t start;
        uint32_t end;
        if (coreIdx_ < formerCoreNum) {
            start = coreIdx_ * formerChunkNum;
            end = start + formerChunkNum;
        } else {
            start = formerCoreNum * formerChunkNum + (coreIdx_ - formerCoreNum) * tailChunkNum;
            end = start + tailChunkNum;
        }

        for (int32_t taskId = start; taskId < end; ++taskId) {
            validLen_ = chunkSize_;
            uint64_t nId   = taskId % nv_;
            uint64_t cgId = taskId / nv_;
            // 尾chunk处理
            if (cgId == numChunk_ - 1 && cg_.length % chunkSize_ != 0) {
                validLen_ = cg_.length % chunkSize_;
            }
            if (validLen_ < halfChunkSize_) {
                subValidRows_ = (subBlockIdx_ == 0) ? validLen_ : 0;
            } else {
                subValidRows_ = (subBlockIdx_ == 0) ? halfChunkSize_ : validLen_ - halfChunkSize_;
            }
            // chunk在全局T上的起始行 = chunkGroup起始行 + chunk内偏移
            uint64_t chunkStartRow = cg_.startPos + cgId * chunkSize_;
            SetChunkTensors(nId, cgId, chunkStartRow);
            ProcessOneChunk();
        }
    }

private:
    // ----------------------------------------------------------
    // SetChunkTensors
    //   nId       : head 编号 (Nv 维度)
    //   localChunkId : CG 内的 chunk 编号 (0 ~ CG_CHUNKS-1)
    //   chunkStartRow   : 当前 chunk 在全局 T 上的起始行
    // ----------------------------------------------------------
   __aicore__ inline void SetChunkTensors(uint64_t nId, uint64_t localChunkId, uint64_t chunkStartRow)
    {
        uint64_t kid = nId * nk_ / nv_;
        uint64_t subRow = chunkStartRow + subOffset_;
        uint64_t qk_base = subRow * nk_ * dk_ + kid * dk_;
        queryGm_ = queryBaseGm_[qk_base];
        keyGm_   = keyBaseGm_[qk_base];

        uint64_t vOffset = chunkStartRow * vRowStride_ + nId * dv_;
        valueGm_ = valueBaseGm_[vOffset];

        uint64_t bgOffset = chunkStartRow * nv_ + nId;
        betaGm_ = betaBaseGm_[bgOffset];
        if (gOptional_) {
            gGm_ = gBaseGm_[bgOffset];
        }

        uint64_t cgLenPad = (cg_.length + chunkSize_ - 1) / chunkSize_ * chunkSize_;
        uint64_t chunkRowBase = nId * cgLenPad + localChunkId * chunkSize_;

        outGCumExpGm_ = outGCumExpBaseGm_[chunkRowBase];
        outKCumdecayGm_ = outKCumdecayBaseGm_[chunkRowBase * dk_];
        outQPrimeGm_ = outQPrimeBaseGm_[chunkRowBase * dk_];
        outKgGm_ = outKgBaseGm_[chunkRowBase * dk_];
        outVInnerGm_ = outVInnerBaseGm_[chunkRowBase * dv_];
        outQkGm_ = outQkBaseGm_[chunkRowBase * chunkSize_];
    }

    __aicore__ inline void ProcessOneChunk()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(0x9);  // 同步0
            // key @ key.transpose(-1,-2)
            AICProcess(keyContinousGm_, keyContinousGm_, kkWsGm_,
                       {chunkSize_, chunkSize_, dk_, chunkSize_, chunkSize_, dk_}, true);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);  // 同步1
            // query @ key.transpose(-1,-2)   stage1 out
            AICProcess(queryContinousGm_, keyContinousGm_, outQkGm_,
                       {validLen_, validLen_, dk_, validLen_, validLen_, dk_}, true);
            AscendC::CrossCoreWaitFlag(0x7);  // 同步2
            // 求逆左下角矩阵
            AttnInverseMMCompute(INVERSE_SHAPE);
            AscendC::CrossCoreWaitFlag(0x6);  // 同步3
            // attn @ k_cumdecay
            AICProcess(attnWsGm_, gBKWsGm_, outKCumdecayGm_,
                       {chunkSize_, dk_, chunkSize_, chunkSize_, dk_, chunkSize_});
            AscendC::CrossCoreWaitFlag(0x5);  // 同步4
            // attn @ v_beta    stage1 out
            AICProcess(attnWsGm_, vBetaWsGm_, outVInnerGm_,
                       {chunkSize_, dv_, chunkSize_, chunkSize_, dv_, chunkSize_});
        }
        if ASCEND_IS_AIV {
            // 获取连续QK
            QKPreProcess();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9);  // 同步0
            if (gOptional_) {
                // g_cum_exp = g.cumsum(dim=-1).exp()
                GCumExpCompute();
                // attn_1 = (g_cum_exp[:None] / g_cum_exp[None,:]) * mask
                GammaCompute();
            }
            BetaCopyInWithStride();
            AscendC::CrossCoreWaitFlag(0x8);  // 同步1
            // attn_1 = kkt * attn_1
            KKBetaCompute();
            // attn_1对角块求逆，对角块shape为INVERSE_SHAPE=32
            InverseCompute();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x7);  // 同步2
            // kg = key * (g_cum_exp[-1, None] / g_cum_exp)[..., None] && k_cumdecay = -1.0 * k * beta * g_cum_exp
            GBKCompute();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6);  // 同步3
            // v_beta = value * beta.unsqueeze(-1)  # (C, Dv)
            VBetaCompute();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);  // 同步4
            // q_prime = query * scale_ * g_cum_exp[:, None]       # (C, Dk)
            QPrimeCompute();
        }
    }

    __aicore__ inline void QKPreProcessCompute(const GlobalTensor<bfloat16_t>& srcGm, const GlobalTensor<float>& dstGm,
                                                LocalTensor<float>& dstBuffer, bool kgFlag = false)
    {
        // copyIn
        DataCopyInBf16WithStride(subValidRows_, dk_, srcGm, nk_ * dk_);
        // compute
        LocalTensor<bfloat16_t> bf16Tensor = fp32InQueue_.DeQue<bfloat16_t>();
        Cast(dstBuffer, bf16Tensor, AscendC::RoundMode::CAST_NONE, subValidRows_ * dkAligned_);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(bf16Tensor);

        if (subValidRows_ < halfChunkSize_) {
            Duplicate(dstBuffer[subValidRows_ * dkAligned_], static_cast<float>(0.0f),
                      (halfChunkSize_ - subValidRows_) * dkAligned_);
            PipeBarrier<PIPE_V>();
        }

        // copyOut
        auto tmpTensor = fp32OutQueue_.AllocTensor<float>();
        DataCopy(tmpTensor, dstBuffer, halfChunkSize_ * dkAligned_);
        fp32OutQueue_.EnQue(tmpTensor);
        tmpTensor = fp32OutQueue_.DeQue<float>();

        uint32_t srcStride = (dkAligned_ - dk_) * sizeof(float) / BLOCK_SIZE;
        DataCopyExtParams outParams{static_cast<uint16_t>(halfChunkSize_),
                                    static_cast<uint32_t>(dk_ * sizeof(float)), srcStride, 0, 0};
        DataCopyPad(dstGm, tmpTensor, outParams);
        if (!gOptional_ && kgFlag) {
            DataCopyPad(outKgGm_[subOffset_ * dk_], tmpTensor, outParams);
        }
        fp32OutQueue_.FreeTensor(tmpTensor);
    }

    __aicore__ inline void QKPreProcess()
    {
        uint64_t outOffset = subOffset_ * dk_;
        QKPreProcessCompute(queryGm_, queryContinousGm_[outOffset], qUbFloatCon_);
        QKPreProcessCompute(keyGm_, keyContinousGm_[outOffset], kUbFloatCon_, true);
    }

    __aicore__ inline void GCumExpCompute()
    {
        // Copy g
        GCopyInWithStride();
        // CumSum计算
        uint32_t outer = 1;
        uint32_t inner = chunkSize_;
        CumSumInfo cumSumInfo{outer, inner};
        CumSum<float>(gCumUbFloat_, gCumUbFloat_, gCumUbFloat_, cumSumInfo);
        PipeBarrier<PIPE_V>();
        // Exp计算
        gCumExpUbFloat_ = gOutQueue_.AllocTensor<float>();
        Exp<float, 0, true>(gCumExpUbFloat_, gCumUbFloat_, chunkSize_);
        gOutQueue_.EnQue<float>(gCumExpUbFloat_);
        DataCopyOutG(validLen_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GammaCompute()
    {
        // BroadCast
        uint32_t divShape[2] = {chunkSize_, chunkSize_};
        uint32_t gShape[2] = {chunkSize_, 1};
        uint32_t gTransShape[2] = {1, chunkSize_};
        Broadcast<float, BROADCAST_AXIS, 1>(gBroadUbFloat_, gCumExpUbFloat_, divShape, gShape);
        Broadcast<float, BROADCAST_AXIS, 0>(gTransBroadUbFloat_, gCumExpUbFloat_, divShape, gTransShape);
        PipeBarrier<PIPE_V>();
        // div
        Div(gammaUbFloat_, gBroadUbFloat_, gTransBroadUbFloat_, chunkSize_ * chunkSize_);
        PipeBarrier<PIPE_V>();
        // mask
        DataCopyInFp32(chunkSize_ * chunkSize_, stageOneMask_[GetBlockIdx() * chunkSize_ * chunkSize_]);
        kkLocal_ = fp32InQueue_.DeQue<float>();
        Mul(gammaUbFloat_, gammaUbFloat_, kkLocal_, chunkSize_ * chunkSize_);
        fp32InQueue_.FreeTensor(kkLocal_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void KKBetaCompute()
    {
        // copy value
        uint32_t kkLength = chunkSize_ * halfChunkSize_;
        uint64_t kkBeginOffset = subOffset_ * chunkSize_;
        DataCopyInFp32(kkLength, kkWsGm_[kkBeginOffset]);
        kkLocal_ = fp32InQueue_.DeQue<float>();

        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t kkShape[2] = {halfChunkSize_, chunkSize_};
        Broadcast<float, BROADCAST_AXIS, 1>(attnUbFloat_, betaUbFloat_, kkShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(attnUbFloat_, kkLocal_, attnUbFloat_, chunkSize_ * halfChunkSize_);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(kkLocal_);
    }

    __aicore__ inline void InverseCompute()
    {
        uint64_t curVecLen = chunkSize_ * halfChunkSize_;
        if (gOptional_) {
            Mul(attnUbFloat_, attnUbFloat_, gammaUbFloat_[subOffset_ * chunkSize_], curVecLen);
        } else {
            DataCopyInFp32(curVecLen, stageOneMask_[subOffset_ * chunkSize_]);
            kkLocal_ = fp32InQueue_.DeQue<float>();
            Mul(attnUbFloat_, attnUbFloat_, kkLocal_, curVecLen);
            fp32InQueue_.FreeTensor(kkLocal_);
        }
        PipeBarrier<PIPE_V>();

        inverseLocal_ = fp32OutQueue_.AllocTensor<float>();
        Muls(inverseLocal_, attnUbFloat_, static_cast<float>(-1.0), curVecLen);
        PipeBarrier<PIPE_V>();

        InverseAIV(subOffset_, INVERSE_SHAPE);
        fp32OutQueue_.EnQue(inverseLocal_);
        DataCopyOutFp32(halfChunkSize_, chunkSize_, chunkSize_, attnWsGm_[subOffset_ * chunkSize_]);
    }

    __aicore__ inline void InverseAIV(uint64_t offset, uint32_t inverseVecLen)
    {
        PipeBarrier<PIPE_V>();
        uint64_t inverseBufferOffset = 0;
        auto row = inverseUbFloat_[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen;
        auto col = inverseUbFloat_[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen + inverseVecLen;
        auto yLocal = inverseUbFloat_[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen;
        auto ei = inverseUbFloat_[inverseBufferOffset];

        Duplicate(ei, static_cast<float>(0.0), inverseVecLen);
        Duplicate(yLocal, static_cast<float>(0.0), inverseVecLen * inverseVecLen); // yLocal清零
        inverseLocal_.SetValue(offset, static_cast<float>(1.0));
        
        uint32_t srcShape[2] = {1, inverseVecLen};
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
        for (int i = 1; i < inverseVecLen; ++i) {
            uint32_t curI = i - 1;
            uint32_t validRows = inverseVecLen - i;
            Gather(col, attnUbFloat_[offset + i * chunkSize_ + curI], colBuffer_, (uint32_t)0, validRows);
            PipeBarrier<PIPE_V>();
            uint32_t dstShape[2] = {validRows, inverseVecLen};
            uint32_t colSrcShape[2] = {validRows, 1};
            Broadcast<float, BROADCAST_AXIS, 1>(col[inverseVecLen], col, dstShape, colSrcShape);
            Broadcast<float, BROADCAST_AXIS, 0>(row, inverseLocal_[offset + curI * chunkSize_], dstShape, srcShape);
            PipeBarrier<PIPE_V>();
            MulAddDst(yLocal[i * inverseVecLen], col[inverseVecLen], row, inverseVecLen * validRows);
            PipeBarrier<PIPE_V>();
            ei.SetValue(i - 1, static_cast<float>(0.0));
            ei.SetValue(i, static_cast<float>(1.0));
            eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventID);
            WaitFlag<HardEvent::S_V>(eventID);
            // xi = (I - SUM) / Lii = I - SUM
            Sub(inverseLocal_[offset + i * chunkSize_], ei, yLocal[i * inverseVecLen], inverseVecLen);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void GBKCompute()
    {
        if (gOptional_) {
            // tmp = -1.0 * beta * g_cum_exp
            Mul(gBUbFloat_, betaUbFloat_, gCumExpUbFloat_[subOffset_], halfChunkSize_);
            PipeBarrier<PIPE_V>();
            Muls(gBUbFloat_, gBUbFloat_, static_cast<float>(-1), halfChunkSize_);
        } else {
            Muls(gBUbFloat_, betaUbFloat_, static_cast<float>(-1), halfChunkSize_);
        }
        PipeBarrier<PIPE_V>();
        // k_cumdecay = k * tmp =  -1.0 * k * beta * g_cum_exp
        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t kShape[2] = {halfChunkSize_, dkAligned_};
        gBKLocal_ = fp32OutQueue_.AllocTensor<float>();
        Broadcast<float, BROADCAST_AXIS, 1>(gBKLocal_, gBUbFloat_, kShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(gBKLocal_, gBKLocal_, kUbFloatCon_, halfChunkSize_ * dkAligned_);
        fp32OutQueue_.EnQue<float>(gBKLocal_);
        uint64_t gBKBeginOffset = subOffset_ * dk_;
        DataCopyOutFp32(halfChunkSize_, dk_, dkAligned_, gBKWsGm_[gBKBeginOffset]);
        PipeBarrier<PIPE_V>();
        if (gOptional_) {
            // kg = k * (g_cum_exp[-1, None] / g_cum_exp)[..., None]
            uint32_t gEndShape[2] = {1, 1};
            uint32_t gBroadShape[2] = {halfChunkSize_, 1};
            Broadcast<float, BROADCAST_AXIS, 0>(gEndBroadUbFloat_, gCumExpUbFloat_[chunkSize_ - 1],
                                                gBroadShape, gEndShape);
            PipeBarrier<PIPE_V>();
            Div(gEndBroadUbFloat_, gEndBroadUbFloat_, gCumExpUbFloat_[subOffset_], halfChunkSize_);
            PipeBarrier<PIPE_V>();
            kgLocal_ = fp32OutQueue_.AllocTensor<float>();
            Broadcast<float, BROADCAST_AXIS, 1>(kgLocal_, gEndBroadUbFloat_, kShape, gBroadShape);
            PipeBarrier<PIPE_V>();
            Mul(kgLocal_, kgLocal_, kUbFloatCon_, halfChunkSize_ * dkAligned_);
            PipeBarrier<PIPE_V>();
            fp32OutQueue_.EnQue<float>(kgLocal_);
            uint64_t kgBeginOffset = subOffset_ * dk_;
            DataCopyOutFp32(halfChunkSize_, dk_, dkAligned_, outKgGm_[kgBeginOffset]);  // stage1 out
        }
    }

    __aicore__ inline void VBetaCompute()
    {
        uint64_t vBeginOffset = subOffset_ * vRowStride_;
        DataCopyInBf16WithStride(subValidRows_, dv_, valueGm_[vBeginOffset], vRowStride_);
        valueLocal_ = fp32InQueue_.DeQue<bfloat16_t>();
        vBetaLocal_ = fp32OutQueue_.AllocTensor<float>();
        Cast(valueUbFloat_, valueLocal_, AscendC::RoundMode::CAST_NONE, subValidRows_ * dvAligned_);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(valueLocal_);
        if (subValidRows_ < halfChunkSize_) {
            Duplicate(valueUbFloat_[subValidRows_ * dvAligned_], static_cast<float>(0.0f),
                      (halfChunkSize_ - subValidRows_) * dvAligned_);
            PipeBarrier<PIPE_V>();
        }
        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t vShape[2] = {halfChunkSize_, dvAligned_};
        Broadcast<float, BROADCAST_AXIS, 1>(vBetaLocal_, betaUbFloat_, vShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(vBetaLocal_, valueUbFloat_, vBetaLocal_, halfChunkSize_ * dvAligned_);
        PipeBarrier<PIPE_V>();
        fp32OutQueue_.EnQue<float>(vBetaLocal_);
        DataCopyOutFp32(halfChunkSize_, dv_, dvAligned_, vBetaWsGm_[subOffset_ * dv_]);
    }

    __aicore__ inline void QPrimeCompute()
    {
        qPrimeLocal_ = fp32OutQueue_.AllocTensor<float>();
        // query * scale
        if (gOptional_) {
            Muls(qUbFloat_, qUbFloatCon_, scale_, halfChunkSize_ * dkAligned_);
            PipeBarrier<PIPE_V>();
            uint32_t gCumExpShape[2] = {halfChunkSize_, 1};
            uint32_t qShape[2] = {halfChunkSize_, dkAligned_};
            Broadcast<float, BROADCAST_AXIS, 1>(gCumExpBroadUbFloat_, gCumExpUbFloat_[subOffset_],
                                                qShape, gCumExpShape);
            PipeBarrier<PIPE_V>();
            // query * scale * g_cum_exp[:, None]       # (C, Dk)
            Mul(qPrimeLocal_, qUbFloat_, gCumExpBroadUbFloat_, halfChunkSize_ * dkAligned_);
            gOutQueue_.FreeTensor(gCumExpUbFloat_);
        } else {
            Muls(qPrimeLocal_, qUbFloatCon_, scale_, halfChunkSize_ * dkAligned_);
        }
        PipeBarrier<PIPE_V>();
        fp32OutQueue_.EnQue<float>(qPrimeLocal_);
        uint64_t qgBeginOffset = subOffset_ * dk_;
        DataCopyOutFp32(halfChunkSize_, dk_, dkAligned_, outQPrimeGm_[qgBeginOffset]);  // stage1 out
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DataCopyInFp32(uint64_t len, GlobalTensor<float> y)
    {
        DataCopyPadExtParams<float> padParams;
        DataCopyExtParams kkParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        fp32InLocal_ = fp32InQueue_.AllocTensor<float>();
        DataCopyPad(fp32InLocal_, y, kkParams, padParams);
        fp32InQueue_.EnQue<float>(fp32InLocal_);
    }

    __aicore__ inline void BetaCopyInWithStride()
    {
        uint64_t betaBeginOffset = subOffset_ * nv_;
        DataCopyInBf16WithStride(subValidRows_, 1, betaGm_[betaBeginOffset], nv_);
        betaLocal_ = fp32InQueue_.DeQue<bfloat16_t>();
        if (subValidRows_ < halfChunkSize_) {
            Duplicate(betaUbBfloat16_, bfloat16_t(0.0f), halfChunkSize_);
            PipeBarrier<PIPE_V>();
        }
        Gather(betaUbBfloat16_, betaLocal_, gatherOffsetBf16_, static_cast<uint32_t>(0), subValidRows_);
        PipeBarrier<PIPE_V>();

        Cast(betaUbFloat_, betaUbBfloat16_, AscendC::RoundMode::CAST_NONE, halfChunkSize_);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(betaLocal_);
    }

    __aicore__ inline void GCopyInWithStride()
    {
        DataCopyInFp32WithStride(validLen_, 1, gGm_, nv_);
        gLocal_ = fp32InQueue_.DeQue<float>();
        if (validLen_ < chunkSize_) {
            Duplicate(gCumUbFloat_, 0.0f, chunkSize_);
            PipeBarrier<PIPE_V>();
        }
        Gather(gCumUbFloat_, gLocal_, gatherOffsetFp32_, static_cast<uint32_t>(0), validLen_);
        PipeBarrier<PIPE_V>();

        fp32InQueue_.FreeTensor(gLocal_);
    }

    __aicore__ inline void DataCopyInFp32WithStride(uint64_t rows,  // 要搬的行数
                                                    uint64_t cols,  // 每行的元素数
                                                    const GlobalTensor<float> src,
                                                    uint64_t srcRowStride) // GM上相邻行的间距(元素数)
    {
        DataCopyPadExtParams<float> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                 static_cast<float>(0)};
        uint32_t srcGap = (srcRowStride - cols) * sizeof(float);
        DataCopyExtParams params{static_cast<uint16_t>(rows),
                                 static_cast<uint32_t>(cols * sizeof(float)),
                                 static_cast<uint32_t>(srcGap), 0, 0};
        fp32InLocal_ = fp32InQueue_.AllocTensor<float>();
        DataCopyPad(fp32InLocal_, src, params, padParams);
        fp32InQueue_.EnQue<float>(fp32InLocal_);
    }

    __aicore__ inline void DataCopyInBf16WithStride(uint64_t rows,  // 要搬的行数
                                                    uint64_t cols,  // 每行的元素数
                                                    GlobalTensor<bfloat16_t> src,
                                                    uint64_t srcRowStride) // GM上相邻行的间距(元素数)
    {
        DataCopyPadExtParams<bfloat16_t> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                      static_cast<float>(0)};
        uint32_t srcGap = (srcRowStride - cols) * sizeof(bfloat16_t);
        DataCopyExtParams params{static_cast<uint16_t>(rows),
                                 static_cast<uint32_t>(cols * sizeof(bfloat16_t)),
                                 static_cast<uint32_t>(srcGap), 0, 0};
        bf16InLocal_ = fp32InQueue_.AllocTensor<bfloat16_t>();
        DataCopyPad(bf16InLocal_, src, params, padParams);
        fp32InQueue_.EnQue<bfloat16_t>(bf16InLocal_);
    }

    __aicore__ inline void DataCopyOutFp32(uint32_t rows, uint32_t cols,
                                                uint32_t colsAligned, GlobalTensor<float> y)
    {
        fp32OutLocal_ = fp32OutQueue_.DeQue<float>();
        uint32_t srcStride = (colsAligned - cols) * sizeof(float) / BLOCK_SIZE;
        DataCopyExtParams yGMParams{static_cast<uint16_t>(rows),
                                    static_cast<uint32_t>(cols * sizeof(float)),
                                    static_cast<uint32_t>(srcStride), 0, 0};
        DataCopyPad(y, fp32OutLocal_, yGMParams);
        fp32OutQueue_.FreeTensor(fp32OutLocal_);
    }

    __aicore__ inline void DataCopyOutG(uint64_t length)
    {
        gCumExpUbFloat_ = gOutQueue_.DeQue<float>();
        if (subBlockIdx_ == 0) {
            DataCopyExtParams params{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(length * sizeof(float)),
                                     0, 0, 0};
            DataCopyPad(outGCumExpGm_, gCumExpUbFloat_, params);  // stage1 out
        }
    }

    __aicore__ inline void AttnInverseMMCompute(uint64_t curLen)
    {
        uint64_t leftDown = chunkSize_ * curLen;
        uint64_t rightDown = leftDown + curLen;
        // 右矩阵左下角 @ 右矩阵左上角 -> 右矩阵左下角
        AICProcess(attnWsGm_[leftDown], attnWsGm_, attnWsGm_[leftDown],
                   {chunkSize_, chunkSize_, chunkSize_, curLen, curLen, curLen});
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventID);
        WaitFlag<HardEvent::FIX_MTE2>(eventID);
        // 右矩阵右下角 @ 右矩阵左下角 -> 右矩阵左下角
        AICProcess(attnWsGm_[rightDown], attnWsGm_[leftDown], attnWsGm_[leftDown],
                   {chunkSize_, chunkSize_, chunkSize_, curLen, curLen, curLen});
        eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventID);
        WaitFlag<HardEvent::FIX_MTE2>(eventID);
    }

    __aicore__ inline void AICProcess(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z,
                                      const MatmulShapeParams &shape, bool transB = false)
    {
        mmFp32.SetOrgShape(shape.m, shape.n, shape.k);
        mmFp32.SetSingleShape(shape.sm, shape.sn, shape.sk);
        mmFp32.SetTensorA(x);
        mmFp32.SetTensorB(y, transB);
        mmFp32.IterateAll(z);
        mmFp32.End();
    }
    
    TPipe *pipe_;
    StageOneMT &mmFp32;
    const ChunkGatedDeltaRuleTilingData *tiling_;
    ChunkGroup cg_;
    uint32_t nk_;
    uint32_t nv_;
    uint32_t dk_;
    uint32_t dv_;
    uint32_t dkAligned_;
    uint32_t dvAligned_;
    uint32_t numChunk_;
    uint64_t vRowStride_;
    uint32_t halfChunkSize_;
    uint32_t subBlockIdx_;
    uint32_t subOffset_;
    uint32_t coreIdx_;
    uint32_t chunkSize_;
    uint32_t validLen_;
    uint32_t subValidRows_;
    uint32_t coreNum_;
    float scale_;
    bool gOptional_;

    // base GM pointers
    GlobalTensor<bfloat16_t> queryBaseGm_;
    GlobalTensor<bfloat16_t> keyBaseGm_;
    GlobalTensor<bfloat16_t> valueBaseGm_;
    GlobalTensor<bfloat16_t> betaBaseGm_;
    GlobalTensor<float> gBaseGm_;
    GlobalTensor<float> outGCumExpBaseGm_, outVInnerBaseGm_, outKgBaseGm_, outQkBaseGm_;
    GlobalTensor<float> outKCumdecayBaseGm_, outQPrimeBaseGm_;

    // chunk GM pointers
    GlobalTensor<bfloat16_t> queryGm_;
    GlobalTensor<bfloat16_t> keyGm_;
    GlobalTensor<bfloat16_t> valueGm_;
    GlobalTensor<bfloat16_t> betaGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<float> outGCumExpGm_;
    GlobalTensor<float> outKCumdecayGm_;
    GlobalTensor<float> outVInnerGm_;
    GlobalTensor<float> outQPrimeGm_;
    GlobalTensor<float> outKgGm_;
    GlobalTensor<float> outQkGm_;

    GlobalTensor<float> vBetaWsGm_;
    GlobalTensor<float> kkWsGm_;
    GlobalTensor<float> attnWsGm_;
    GlobalTensor<float> gBKWsGm_;
    GlobalTensor<float> queryContinousGm_;
    GlobalTensor<float> keyContinousGm_;
    GlobalTensor<float> querytmpGm_;
    GlobalTensor<float> stageOneMask_;

    // UB queues
    TQue<QuePosition::VECIN, 1> fp32InQueue_;
    TQue<QuePosition::VECOUT, 1> fp32OutQueue_;
    TQue<QuePosition::VECOUT, 1> gOutQueue_;

    TBuf<TPosition::VECCALC> tmpBuff_;

    // UB tensors
    LocalTensor<bfloat16_t> betaUbBfloat16_;
    LocalTensor<float> betaUbFloat_;
    LocalTensor<float> valueUbFloat_;
    LocalTensor<float> attnUbFloat_;
    LocalTensor<float> inverseUbFloat_;
    LocalTensor<float> inverseLocal_;
    LocalTensor<float> gCumUbFloat_;
    LocalTensor<float> gCumExpUbFloat_;
    LocalTensor<float> gCumExpBroadUbFloat_;
    LocalTensor<float> gBUbFloat_;
    LocalTensor<float> kUbFloat_;
    LocalTensor<float> qUbFloat_;
    LocalTensor<float> gBroadUbFloat_;
    LocalTensor<float> gTransBroadUbFloat_;
    LocalTensor<float> gEndBroadUbFloat_;
    LocalTensor<float> gammaUbFloat_;
    LocalTensor<uint32_t> colBuffer_;
    LocalTensor<float> qUbFloatCon_;
    LocalTensor<float> kUbFloatCon_;
    LocalTensor<uint32_t> gatherOffsetFp32_;
    LocalTensor<uint32_t> gatherOffsetBf16_;

    LocalTensor<bfloat16_t> betaLocal_;
    LocalTensor<bfloat16_t> valueLocal_;
    LocalTensor<bfloat16_t> kLocal_;
    LocalTensor<float> qPrimeLocal_;
    LocalTensor<float> vBetaLocal_;
    LocalTensor<float> kkLocal_;
    LocalTensor<float> gLocal_;
    LocalTensor<float> gBKLocal_;
    LocalTensor<float> kgLocal_;
    
    LocalTensor<bfloat16_t> bf16InLocal_;
    LocalTensor<float> fp32InLocal_;
    LocalTensor<float> fp32OutLocal_;
};
} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE1_H