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
 * \file sparse_lightning_indexer_grad_kl_loss_kernel_base.h
 * \brief
 */
 
#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_KERNEL_BASE_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_KERNEL_BASE_H

#include "sparse_lightning_indexer_grad_kl_loss_tiling_regbase.h"
#include "sparse_lightning_indexer_grad_kl_loss_regbase_common.h"
#include "sparse_lightning_indexer_grad_kl_loss_cube_block.h"
#include "sparse_lightning_indexer_grad_kl_loss_vector_block.h"

namespace SligKlLoss {
template <typename CubeBlockType, typename VecBlockType>
class SparseLightningIndexerGradKLLossKernelBase {
public:
    ARGS_TRAITS;
    __aicore__ inline SparseLightningIndexerGradKLLossKernelBase(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex, GM_ADDR weight,
                                GM_ADDR sparseIndices, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR queryRope,
                                GM_ADDR keyRope, GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKey,
                                GM_ADDR dQueryIndex, GM_ADDR dKeyIndex, GM_ADDR dWeight, GM_ADDR loss,
                                GM_ADDR workspace,
                                const optiling::SparseLightningIndexerGradKLLossRegBaseTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessSY(SLIGradKLLossRunInfo &runInfo, int64_t taskId, int64_t bIdx, int64_t s1Idx);
    __aicore__ inline void ProcessP(SLIGradKLLossRunInfo &runInfo, int64_t taskId, int64_t bIdx, int64_t s1Idx);
    __aicore__ inline void InitMMResBuf();
    __aicore__ inline void FreeBuf();
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace);
    __aicore__ inline void InitBuffer(TPipe *pipe);
    __aicore__ inline int32_t GetS2SparseLen(int32_t s1Idx, int32_t actualSeqLensQ, int32_t actualSeqLensK, SLISparseMode sparseMode);
    __aicore__ inline void SetRunInfo(SLIGradKLLossRunInfo &runInfo, int64_t taskId, int64_t bIdx, int64_t s1Idx, int64_t s1IdxEnd, bool isLastB);
    __aicore__ inline void SetRunInfoP(SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void SetSYRunInfo(SLIGradKLLossKRunInfo &kRunInfo, SLIGradKLLossRunInfo &runInfo, int64_t s2TaskId);
    __aicore__ inline void SetPRunInfo(SLIGradKLLossKRunInfo &kRunInfo, SLIGradKLLossRunInfo &runInfo, int64_t s2TaskId);
    __aicore__ inline void SetSYSingleRunInfo(SLIGradKLLossKRunInfo &kRunInfo, SLIGradKLLossRunInfo &runInfo, int64_t s2TaskId, int64_t taskId);
    __aicore__ inline void SetPSingleRunInfo(SLIGradKLLossKRunInfo &kRunInfo, SLIGradKLLossRunInfo &runInfo, int64_t s2TaskId, int64_t taskId);
    __aicore__ inline void SetConstInfo();
    __aicore__ inline int64_t FindBIndex(int64_t bIndex, int64_t curIndex, int64_t &accumulateLen);
    __aicore__ inline void CalcMultiCoreOffset(int64_t &bStartIdx, int64_t &s1StartIdx, int64_t &bEndIdx,
                                               int64_t &s1EndIdx);
    __aicore__ inline int32_t GetActualSeqLens(int32_t bIdx, int32_t defaultLens,
                                               GlobalTensor<int64_t> &actualSeqLensGm, SLILayout layout,
                                               int64_t &accumLen);
    __aicore__ inline int64_t GetEndS1Etx(int32_t bIdx, int32_t defaultLens, GlobalTensor<int64_t> &actualSeqLensGm,
                                          SLILayout layout);

    TPipe *pipe = nullptr;
    const optiling::SparseLightningIndexerGradKLLossRegBaseTilingData *__restrict tilingData = nullptr;

    // input gm
    GlobalTensor<INPUT_T> queryGm, queryIndexGm, weightGm;
    GlobalTensor<INPUT_T> keyGm, keyIndexGm;
    GlobalTensor<INPUT_T> queryRopeGm;
    GlobalTensor<INPUT_T> keyRopeGm;
    GlobalTensor<T> softmaxMaxGm, softmaxSumGm;
    GlobalTensor<int32_t> topKIndexGm;
    GlobalTensor<int64_t> actualSeqLengthsQueryGm;
    GlobalTensor<int64_t> actualSeqLengthsKeyGm;

    // output gm
    GlobalTensor<OUT_T> dQueryIndexGm;
    GlobalTensor<OUT_T> dKeyIndexGm;
    GlobalTensor<OUT_T> dWeightGm;
    GlobalTensor<T> lossGm;

    // workspace
    GlobalTensor<T> reduceSumRes;
    GlobalTensor<T> reluRes;
    GlobalTensor<INPUT_T> gatherSYRes;
    GlobalTensor<T> scatterAddResGm;

    // CV间共享buffer
    TBuf<> mm12ResBuf[3]; //UB上3buffer
    TBuf<> mm3ResBuf;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;

    BuffersPolicy3buff<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm3Buffer;
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> sYL1Buf;
    BuffersPolicySingleBuffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> reluGradResL1Buf;

    SLIGradKLLossConstInfo constInfo;
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::Init(
    GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex, GM_ADDR weight, GM_ADDR sparseIndices,
    GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR actualSeqLengthsQuery,
    GM_ADDR actualSeqLengthsKey, GM_ADDR dQueryIndex, GM_ADDR dKeyIndex, GM_ADDR dWeight, GM_ADDR loss,
    GM_ADDR workspace, const optiling::SparseLightningIndexerGradKLLossRegBaseTilingData *__restrict tiling,
    TPipe *tPipe)
{
    pipe = tPipe;
    tilingData = tiling;
    SetConstInfo();

    queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    queryIndexGm.SetGlobalBuffer((__gm__ INPUT_T *)queryIndex);
    keyIndexGm.SetGlobalBuffer((__gm__ INPUT_T *)keyIndex);
    weightGm.SetGlobalBuffer((__gm__ INPUT_T *)weight);
    topKIndexGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    softmaxMaxGm.SetGlobalBuffer((__gm__ T *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ T *)softmaxSum);
    if constexpr (HAS_ROPE) {
        queryRopeGm.SetGlobalBuffer((__gm__ INPUT_T *)queryRope);
        keyRopeGm.SetGlobalBuffer((__gm__ INPUT_T *)keyRope);
    }
    if (actualSeqLengthsQuery != nullptr) {
        actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsQuery, constInfo.bSize);
    } else {
        actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsQuery, 0);
    }
    if (actualSeqLengthsKey != nullptr) {
        actualSeqLengthsKeyGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsKey, constInfo.bSize);
    } else {
        actualSeqLengthsKeyGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsKey, 0);
    }

    dQueryIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dQueryIndex);
    dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dKeyIndex);
    dWeightGm.SetGlobalBuffer((__gm__ OUT_T *)dWeight);
    lossGm.SetGlobalBuffer((__gm__ T *)loss);
    lossGm.SetValue(0, 0.0F);
    AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(lossGm);

    InitMMResBuf();
    InitWorkspace(workspace);
    InitBuffer(pipe);

    // init vec block
    if ASCEND_IS_AIV {
        // InitVecOP
        vecBlock.InitParams(constInfo, tilingData);
        vecBlock.InitGlobalBuffer(keyGm, keyRopeGm, keyIndexGm, topKIndexGm,
                                  actualSeqLengthsQueryGm, actualSeqLengthsKeyGm,
                                  softmaxMaxGm, softmaxSumGm, gatherSYRes, reduceSumRes,
                                  scatterAddResGm, weightGm, lossGm, reluRes, dWeightGm,
                                  dKeyIndexGm);
    } else if ASCEND_IS_AIC {
        // initCubeOP
        cubeBlock.SetCubeBlockParams(tPipe, &l1BufferManager);
        cubeBlock.InitCubeBuffers();
        cubeBlock.InitGlobalBuffer(query, queryIndex, queryRope, dQueryIndex, reluRes, gatherSYRes);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::ProcessSY(SLIGradKLLossRunInfo &runInfo,
                                                                                   int64_t taskId, int64_t bIdx, int64_t s1Idx)
{
    bool notLastLoop = true;
    SLIGradKLLossKRunInfo kRunInfos[MODE_NUM_2];
    if ASCEND_IS_AIV {
        vecBlock.CopyInWeight(runInfo);
    }
    for (int32_t s2TaskId = 0; s2TaskId < runInfo.s2LoopTimes + 1; ++s2TaskId) {
        if (s2TaskId == runInfo.s2LoopTimes) {
            notLastLoop = false;
        }
        if (s2TaskId >= 0 && notLastLoop) {
            SetSYRunInfo(kRunInfos[s2TaskId % MODE_NUM_2], runInfo, s2TaskId);
            for (int32_t loopIdx = 0; loopIdx < kRunInfos[s2TaskId % MODE_NUM_2].s2SingleLoopTimes; ++loopIdx) {
                SetSYSingleRunInfo(kRunInfos[s2TaskId % MODE_NUM_2], runInfo, s2TaskId, loopIdx);
                if ASCEND_IS_AIV {
                    vecBlock.ProcessVector0(this->sYL1Buf.Get(), runInfo, kRunInfos[s2TaskId % MODE_NUM_2]);
                }
            }
            if ASCEND_IS_AIC {
                cubeBlock.ComputeMmSy(this->bmm1Buffers.GetCube(), sYL1Buf, runInfo, constInfo, kRunInfos[s2TaskId % MODE_NUM_2]);
            }
        }

        if (s2TaskId >= 1) {
            Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Res = this->bmm1Buffers.GetVec();
            for (int32_t loopIdx = 0; loopIdx < kRunInfos[(s2TaskId + 1) % MODE_NUM_2].s2SingleLoopTimes; ++loopIdx) {
                SetSYSingleRunInfo(kRunInfos[(s2TaskId + 1) % MODE_NUM_2], runInfo, s2TaskId - 1, loopIdx);
                if ASCEND_IS_AIV {
                    vecBlock.ProcessVector1(bmm1Res, runInfo, kRunInfos[(s2TaskId + 1) % MODE_NUM_2]);
                }
            }
        }
    }
    if ASCEND_IS_AIV {
        vecBlock.ProcessVector2(this->bmm1Buffers.GetPre(), runInfo);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::ProcessP(SLIGradKLLossRunInfo &runInfo,
                                                                                   int64_t taskId, int64_t bIdx, int64_t s1Idx)
{
    bool notLastLoop = true;
    bool notLastThreeLoop = true;
    bool notSecondLast = true;
    bool notThirdLast = true;
    SLIGradKLLossKRunInfo kRunInfos[MODE_NUM_4];
    bool isFixOut = false;
    if ASCEND_IS_AIV {
        vecBlock.CopyInMaxSum(runInfo);
    }
    for (int32_t s2TaskId = 0; s2TaskId < runInfo.s2LoopTimes + 3; ++s2TaskId) {
        if (s2TaskId == runInfo.s2LoopTimes + 2) {
            notLastLoop = false; 
        } else if (s2TaskId == runInfo.s2LoopTimes + 1) {
            notSecondLast = false;
        } else if (s2TaskId == runInfo.s2LoopTimes) {
            notThirdLast = false;
        }
        notLastThreeLoop = notLastLoop && notSecondLast && notThirdLast;
        if (s2TaskId >= 0 && notLastThreeLoop) {
            SetPRunInfo(kRunInfos[s2TaskId % MODE_NUM_4], runInfo, s2TaskId);
            for (int32_t loopIdx = 0; loopIdx < kRunInfos[s2TaskId % MODE_NUM_4].s2SingleLoopTimes; ++loopIdx) {
                SetPSingleRunInfo(kRunInfos[s2TaskId % MODE_NUM_4], runInfo, s2TaskId, loopIdx);
                if ASCEND_IS_AIV {
                    vecBlock.ProcessVector3(this->sYL1Buf.Get(), runInfo, kRunInfos[s2TaskId % MODE_NUM_4]);
                }
            }
            if ASCEND_IS_AIC {
                cubeBlock.ComputeMmP(this->bmm1Buffers.GetCube(), sYL1Buf, runInfo, constInfo, kRunInfos[s2TaskId % MODE_NUM_4]);
            }
        }
        if (s2TaskId >= 3) {
            if ASCEND_IS_AIC {
                isFixOut = !notLastLoop;
                cubeBlock.ComputeMm3(this->bmm3Buffer.Get(), this->reluGradResL1Buf.Get(), runInfo, constInfo, kRunInfos[(s2TaskId + 1) % MODE_NUM_4]);
                cubeBlock.ComputeMm4(reluGradResL1Buf.Get(), runInfo, constInfo, kRunInfos[(s2TaskId + 1) % MODE_NUM_4], isFixOut);
            }
            for (int32_t loopIdx = 0; loopIdx < kRunInfos[(s2TaskId + 1) % MODE_NUM_4].s2SingleLoopTimes; ++loopIdx) {
                SetPSingleRunInfo(kRunInfos[(s2TaskId + 1) % MODE_NUM_4], runInfo, s2TaskId - 3, loopIdx);
                if ASCEND_IS_AIV {
                    vecBlock.ProcessVector7(this->bmm3Buffer.Get(), runInfo, kRunInfos[(s2TaskId + 1) % MODE_NUM_4]);
                }
            }
        }
        if (s2TaskId >= 2 && notLastLoop) {
            Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Res = this->bmm1Buffers.GetVec();
            for (int32_t loopIdx = 0; loopIdx < kRunInfos[(s2TaskId + 2) % MODE_NUM_4].s2SingleLoopTimes; ++loopIdx) {
                SetPSingleRunInfo(kRunInfos[(s2TaskId + 2) % MODE_NUM_4], runInfo, s2TaskId - 2, loopIdx);
                if ASCEND_IS_AIV {
                    vecBlock.ProcessVector4(bmm1Res, runInfo, kRunInfos[(s2TaskId + 2) % MODE_NUM_4]);
                    vecBlock.ProcessVector5(runInfo, kRunInfos[(s2TaskId + 2) % MODE_NUM_4]);
                    vecBlock.ProcessVector6(this->reluGradResL1Buf.Get(), runInfo, kRunInfos[(s2TaskId + 2) % MODE_NUM_4]);
                }
            }
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::Process()
{
    int64_t bStartIdx, s1StartIdx, bEndIdx, s1EndIdx;
    CalcMultiCoreOffset(bStartIdx, s1StartIdx, bEndIdx, s1EndIdx);
    int64_t taskId = 0;
    SLIGradKLLossRunInfo runInfo;
    for (int64_t bIdx = bStartIdx; bIdx <= bEndIdx; bIdx++) {
        bool lastB = (bIdx == bEndIdx);
        int64_t s1StartIdxThisBatch = 0;
        int64_t s1EndIdxThisBatch = 0;

        if constexpr (LAYOUT_Q == SLILayout::TND) {
            s1StartIdxThisBatch = (bIdx == bStartIdx) ? s1StartIdx : 0;
            s1EndIdxThisBatch =
                (!lastB) ? GetEndS1Etx(bIdx, constInfo.s1Size, actualSeqLengthsQueryGm, LAYOUT_Q) : s1EndIdx;
        } else if constexpr (LAYOUT_Q == SLILayout::BSND) {
            s1StartIdxThisBatch = (bIdx == bStartIdx) ? s1StartIdx : 0;
            s1EndIdxThisBatch = (!lastB) ? constInfo.s1Size : s1EndIdx;
        }

        for (int64_t s1Idx = s1StartIdxThisBatch; s1Idx < s1EndIdxThisBatch; s1Idx++) {
            SetRunInfo(runInfo, taskId, bIdx, s1Idx, s1EndIdxThisBatch, lastB);
            ProcessSY(runInfo, taskId, bIdx, s1Idx);
            SetRunInfoP(runInfo);
            ProcessP(runInfo, taskId, bIdx, s1Idx);
            taskId++;
        }
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        vecBlock.ReInitBuffers(pipe);
        vecBlock.ProcessVector8();
    }
    FreeBuf();    
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int32_t SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::GetS2SparseLen(int32_t s1Idx,
    int32_t actualSeqLensQ, int32_t actualSeqLensK, SLISparseMode sparseMode)
{
    if (sparseMode == SLISparseMode::RightDown) {
        return Max(actualSeqLensK - actualSeqLensQ + s1Idx + 1, 0);
    } else {
        return 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::SetRunInfo(
    SLIGradKLLossRunInfo &runInfo, int64_t taskId, int64_t bIdx, int64_t s1Idx, int64_t s1IdxEnd, bool isLastB)
{
    if (s1Idx >= s1IdxEnd) {        // extra循环阶段，不生产任务
        runInfo.isValid = false;
        return;
    }

    runInfo.taskId = taskId;
    runInfo.bIdx = bIdx;
    runInfo.s1Idx = s1Idx;
    if constexpr (LAYOUT_Q == SLILayout::TND) {
        int32_t actualSeqLensQ =
            GetActualSeqLens(runInfo.bIdx, constInfo.s1Size, actualSeqLengthsQueryGm, LAYOUT_Q, runInfo.accumS1Idx);
        int32_t actualSeqLensK =
            GetActualSeqLens(runInfo.bIdx, constInfo.s2Size, actualSeqLengthsKeyGm, LAYOUT_KT, runInfo.accumS2Idx);
        runInfo.actS1Size = actualSeqLensQ;
        runInfo.actS2Size = actualSeqLensK;
        runInfo.accumS1Idx += s1Idx;
    } else if constexpr (LAYOUT_Q == SLILayout::BSND) {
        runInfo.actS1Size = constInfo.s1Size;
        runInfo.actS2Size = constInfo.s2Size;
        runInfo.accumS1Idx = bIdx * constInfo.s1Size + s1Idx;
        runInfo.accumS2Idx = bIdx * constInfo.s2Size;
    }
    runInfo.kBaseSize = 2048;
    runInfo.s2SparseLen = GetS2SparseLen(runInfo.s1Idx, runInfo.actS1Size, runInfo.actS2Size, constInfo.sparseMode);
    runInfo.s2RealSize = Min(constInfo.kSize, runInfo.s2SparseLen);
    runInfo.kRealSize = runInfo.s2RealSize;
    runInfo.kRealSizeAlign8 = (runInfo.kRealSize + 7) >> 3 << 3;
    runInfo.s2LoopTimes = CeilDiv(runInfo.s2RealSize, constInfo.syKBaseSize);
    runInfo.s2TailSize = (runInfo.s2RealSize % constInfo.syKBaseSize == 0) ?
        constInfo.syKBaseSize : (runInfo.s2RealSize % constInfo.syKBaseSize);
    runInfo.s2BaseSize = VEC_SY_BASESIZE;
    runInfo.s2CurSize = 0;

    runInfo.kLoopTimes = CeilDiv(runInfo.kRealSize, runInfo.kBaseSize);
    runInfo.kTailSize = (runInfo.kRealSize % runInfo.kBaseSize == 0) ?
        runInfo.kBaseSize : (runInfo.kRealSize % runInfo.kBaseSize);

    if constexpr (LAYOUT_Q == SLILayout::TND) {
        runInfo.queryTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQuery * (constInfo.dSizeQuery);
        runInfo.queryRopeTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQuery * (constInfo.dSizeQueryRope);
        runInfo.queryIndexTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQueryIndex * constInfo.dSizeQueryIndex;
    } else if constexpr (LAYOUT_Q == SLILayout::BSND) {
        runInfo.queryTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQuery * (constInfo.dSizeQuery);
        runInfo.queryRopeTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQuery * (constInfo.dSizeQueryRope);
        runInfo.queryIndexTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQueryIndex * constInfo.dSizeQueryIndex;
    }

    if (constInfo.subBlockIdx == 0) {
        runInfo.nIndexSize = AlignTo(constInfo.gSizeQueryIndex, ALIGN_NUM_2) / 2;
        runInfo.nVecSize = AlignTo(constInfo.gSizeQuery, ALIGN_NUM_2) / 2;
    } else {
        runInfo.nIndexSize = constInfo.gSizeQueryIndex - AlignTo(constInfo.gSizeQueryIndex, ALIGN_NUM_2) / 2;
        runInfo.nVecSize = constInfo.gSizeQuery - AlignTo(constInfo.gSizeQuery, ALIGN_NUM_2) / 2;
    }

    if constexpr (LAYOUT_Q == SLILayout::TND) {
        runInfo.topkGmBaseOffset = runInfo.accumS1Idx * constInfo.kSize;
        runInfo.weightOffset = runInfo.accumS1Idx * constInfo.gSizeQueryIndex + 
            AlignTo(constInfo.gSizeQueryIndex, ALIGN_NUM_2) * constInfo.subBlockIdx / 2;
        runInfo.softmaxInputOffset = runInfo.accumS1Idx * constInfo.gSizeQuery +
            AlignTo(constInfo.gSizeQuery, ALIGN_NUM_2) * constInfo.subBlockIdx / 2;
    } else if constexpr (LAYOUT_Q == SLILayout::BSND) {
        runInfo.topkGmBaseOffset = runInfo.bIdx * constInfo.s1Size * constInfo.kSize + runInfo.s1Idx * constInfo.kSize;
        runInfo.weightOffset = runInfo.bIdx * constInfo.s1Size * constInfo.gSizeQueryIndex +
            runInfo.s1Idx * constInfo.gSizeQueryIndex + AlignTo(constInfo.gSizeQueryIndex, ALIGN_NUM_2) *
            constInfo.subBlockIdx / 2;
        runInfo.softmaxInputOffset = runInfo.bIdx * constInfo.s1Size * constInfo.gSizeQuery +
            runInfo.s1Idx * constInfo.gSizeQuery + AlignTo(constInfo.gSizeQuery, ALIGN_NUM_2) *
            constInfo.subBlockIdx / 2;
    }
    runInfo.isValid = true;
    runInfo.isLastK = isLastB && s1Idx == s1IdxEnd - 1;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::SetRunInfoP(
    SLIGradKLLossRunInfo &runInfo)
{
    runInfo.s2LoopTimes = CeilDiv(runInfo.s2RealSize, constInfo.pKBaseSize);
    runInfo.s2TailSize = (runInfo.s2RealSize % constInfo.pKBaseSize == 0) ?
        constInfo.pKBaseSize : (runInfo.s2RealSize % constInfo.pKBaseSize);
    runInfo.s2CurSize = 0;
    runInfo.s2BaseSize = VEC_P_BASESIZE;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::SetSYRunInfo(SLIGradKLLossKRunInfo &kRunInfo,
                                                                                     SLIGradKLLossRunInfo &runInfo,
                                                                                     int64_t s2TaskId)
{
    runInfo.s2TailSize = (runInfo.s2RealSize % VEC_SY_BASESIZE == 0) ?
        VEC_SY_BASESIZE : (runInfo.s2RealSize % VEC_SY_BASESIZE);
    kRunInfo.kTaskId = s2TaskId;
    kRunInfo.kTaskIdMod2 = kRunInfo.kTaskId & 1;
    kRunInfo.kProcessSize = (kRunInfo.kTaskId  == runInfo.s2LoopTimes - 1) ?
        (runInfo.s2RealSize - (runInfo.s2LoopTimes - 1) * constInfo.syKBaseSize) : constInfo.syKBaseSize;
    kRunInfo.s2SingleLoopTimes = CeilDiv(kRunInfo.kProcessSize, VEC_SY_BASESIZE);
    kRunInfo.dValue = constInfo.dSizeQueryIndex;
    kRunInfo.dRopeValue = 0;
    kRunInfo.syGmEn = true;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::SetPRunInfo(SLIGradKLLossKRunInfo &kRunInfo,
                                                                                     SLIGradKLLossRunInfo &runInfo,
                                                                                     int64_t s2TaskId)
{
    runInfo.s2TailSize = (runInfo.s2RealSize % VEC_P_BASESIZE == 0) ?
        VEC_P_BASESIZE : (runInfo.s2RealSize % VEC_P_BASESIZE);
    kRunInfo.kTaskId = s2TaskId;
    kRunInfo.kTaskIdMod2 = kRunInfo.kTaskId & 1;
    kRunInfo.kProcessSize = (kRunInfo.kTaskId  == runInfo.s2LoopTimes - 1) ?
        (runInfo.s2RealSize - (runInfo.s2LoopTimes - 1) * constInfo.pKBaseSize) : constInfo.pKBaseSize;
    kRunInfo.s2SingleLoopTimes = CeilDiv(kRunInfo.kProcessSize, VEC_P_BASESIZE);
    kRunInfo.dValue = constInfo.dSizeQuery;
    kRunInfo.dRopeValue = constInfo.dSizeRope;
    kRunInfo.syGmEn = false;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::SetSYSingleRunInfo(SLIGradKLLossKRunInfo &kRunInfo,
                                                                                     SLIGradKLLossRunInfo &runInfo,
                                                                                     int64_t s2TaskId,
                                                                                     int64_t taskId)
{
    kRunInfo.s2Idx = s2TaskId * constInfo.syKBaseSize / VEC_SY_BASESIZE + taskId;
    kRunInfo.s2SingleIdx = taskId;
    kRunInfo.isS2end = taskId >= kRunInfo.s2SingleLoopTimes - 1 && s2TaskId >= runInfo.s2LoopTimes - 1;
    kRunInfo.s2RealBaseSize = kRunInfo.isS2end ? runInfo.s2TailSize : VEC_SY_BASESIZE;
    runInfo.s2CurSize = s2TaskId * constInfo.syKBaseSize + taskId * VEC_SY_BASESIZE;
    kRunInfo.s2SingleCurSize = kRunInfo.s2SingleIdx * VEC_SY_BASESIZE;
    kRunInfo.isAlign64 = (!kRunInfo.isS2end) || (kRunInfo.isS2end && runInfo.s2TailSize % 64 == 0);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::SetPSingleRunInfo(SLIGradKLLossKRunInfo &kRunInfo,
                                                                                     SLIGradKLLossRunInfo &runInfo,
                                                                                     int64_t s2TaskId,
                                                                                     int64_t taskId)
{
    kRunInfo.s2Idx = s2TaskId * constInfo.pKBaseSize * 2 / VEC_SY_BASESIZE + taskId;
    kRunInfo.s2SingleIdx = taskId;
    kRunInfo.isS2end = taskId >= kRunInfo.s2SingleLoopTimes - 1 && s2TaskId >= runInfo.s2LoopTimes - 1;
    kRunInfo.s2RealBaseSize = kRunInfo.isS2end ? runInfo.s2TailSize : VEC_P_BASESIZE;
    runInfo.s2CurSize = s2TaskId * constInfo.pKBaseSize + taskId * VEC_P_BASESIZE;
    kRunInfo.s2SingleCurSize = kRunInfo.s2SingleIdx * VEC_P_BASESIZE;
    kRunInfo.isAlign64 = (!kRunInfo.isS2end) || (kRunInfo.isS2end && runInfo.s2TailSize % 64 == 0);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::SetConstInfo()
{
    if ASCEND_IS_AIV {
        constInfo.aivIdx = GetBlockIdx();
        constInfo.aicIdx = constInfo.aivIdx / 2;
        constInfo.subBlockIdx = constInfo.aivIdx % MODE_NUM_2;
    } else {
        constInfo.aicIdx = GetBlockIdx();
    }

    auto &baseInfo = tilingData->baseParams;
    constInfo.bSize = baseInfo.bSize;
    constInfo.n2Size = baseInfo.n2Size;
    constInfo.gSizeQuery = baseInfo.gSizeQuery;
    constInfo.gSizeQueryIndex = baseInfo.gSizeQueryIndex;
    constInfo.s1Size = baseInfo.s1Size;
    constInfo.s2Size = baseInfo.s2Size;
    constInfo.kSize = baseInfo.kSize;

    constInfo.dSizeQuery = baseInfo.dSizeQuery;
    constInfo.dSizeQueryIndex = baseInfo.dSizeQueryIndex;
    constInfo.dSizeRope = 64;
    constInfo.sparseMode = static_cast<SLISparseMode>(baseInfo.sparseMode);
    constInfo.scaleValue = baseInfo.scaleValue;
    constInfo.sparseBlockSize = 1;
    constInfo.gatherKBaseSize = 128; // gather每次处理的行数(v0+v1)
    constInfo.syKBaseSize = 16384 / (constInfo.n2Size * constInfo.gSizeQueryIndex);
    constInfo.pKBaseSize = 16384 / (constInfo.n2Size * constInfo.gSizeQuery);
    constInfo.syKBaseSize = Min(1024, constInfo.syKBaseSize);
    constInfo.pKBaseSize = Min(1024, constInfo.pKBaseSize);
    constInfo.pScaler = 1.0f / static_cast<float>(static_cast<int64_t>(constInfo.gSizeQuery));
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::InitMMResBuf()
{
    l1BufferManager.Init(pipe, L1_MAX_SIZE);
    sYL1Buf.Init(l1BufferManager, 128 * 576 * sizeof(INPUT_T));
    reluGradResL1Buf.Init(l1BufferManager, 64 * 256 * sizeof(INPUT_T));
    ubBufferManager.Init(pipe, UB_MAX_SIZE);
    bmm1Buffers.Init(ubBufferManager, 32 * 256 * sizeof(T));
    bmm3Buffer.Init(ubBufferManager, 64 * 128 * sizeof(T));
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[0]);
        CrossCoreSetFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[1]);
        CrossCoreSetFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[2]);
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_C3_TO_V7_FLAG);
    } else if ASCEND_IS_AIC {
        CrossCoreSetFlag<2, PIPE_MTE1>(SYNC_GATHER_TO_MM12_FLAG[0]);
        CrossCoreSetFlag<2, PIPE_MTE1>(SYNC_GATHER_TO_MM12_FLAG[1]);      
        CrossCoreSetFlag<2, PIPE_MTE1>(SYNC_V6_TO_C3_FLAG);  
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::FreeBuf()
{
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_GATHER_TO_MM12_FLAG[0]);
        CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_GATHER_TO_MM12_FLAG[1]);      
        CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_V6_TO_C3_FLAG);  
    } else if ASCEND_IS_AIC {
        CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_MM2_TO_V1_FLAG[0]);
        CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_MM2_TO_V1_FLAG[1]);
        CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_MM2_TO_V1_FLAG[2]);
        CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_C3_TO_V7_FLAG);        
    }
}
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::InitWorkspace(__gm__ uint8_t *workspace)
{
    int64_t reduceSumOffset = 2 * constInfo.kSize * sizeof(float); // * 2
    int64_t reluOffset = constInfo.gSizeQueryIndex * constInfo.kSize * sizeof(float);
    int64_t gatherSYOffset = constInfo.kSize * constInfo.dSizeQueryIndex * sizeof(INPUT_T);

    int64_t coreTotalOffset = constInfo.aicIdx * (reduceSumOffset + reluOffset + gatherSYOffset);
    int64_t totalOffset = GetBlockNum() * (reduceSumOffset + reluOffset + gatherSYOffset);

    uint64_t offset = 0;
    reduceSumRes.SetGlobalBuffer((__gm__ T *)(workspace + offset + coreTotalOffset));
    offset += reduceSumOffset;
    reluRes.SetGlobalBuffer((__gm__ T *)(workspace + offset + coreTotalOffset));
    offset += reluOffset;
    gatherSYRes.SetGlobalBuffer((__gm__ INPUT_T *)(workspace + offset + coreTotalOffset));
    offset += gatherSYOffset;
    scatterAddResGm.SetGlobalBuffer((__gm__ T *)(workspace + totalOffset));
    if ASCEND_IS_AIV {
        int64_t totalCost = 0;
        if constexpr (LAYOUT_Q == SLILayout::TND) {
            totalCost = actualSeqLengthsKeyGm.GetValue(constInfo.bSize - 1);
        } else {
            totalCost = constInfo.bSize * constInfo.s2Size;
        }

        int64_t totalCoreNum = GetBlockNum() * GetTaskRation();
        int64_t avgCost = CeilDiv(totalCost, totalCoreNum);

        int32_t t2Start = Min(constInfo.aivIdx * avgCost, totalCost);
        int32_t t2End = Min(t2Start + avgCost, totalCost);
        GlobalTensor<T> sccatterAddTensor = scatterAddResGm[t2Start * constInfo.dSizeQueryIndex];
        AscendC::Fill(sccatterAddTensor, constInfo.dSizeQueryIndex * (t2End - t2Start), static_cast<T>(0));
    }
    SyncAll();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::InitBuffer(TPipe *pipe)
{
    if ASCEND_IS_AIV {
        vecBlock.InitBuffers(pipe);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::CalcMultiCoreOffset(
    int64_t &bStartIdx, int64_t &s1StartIdx, int64_t &bEndIdx, int64_t &s1EndIdx)
{
    int64_t actualSum = 0;
    int64_t bS1Index = tilingData->multiCoreParams.bS1Index[constInfo.aicIdx];
    int64_t bS1EndIndex = constInfo.aicIdx + 1 < optiling::MAX_CORE_NUM_REGBASE ?
                              tilingData->multiCoreParams.bS1Index[constInfo.aicIdx + 1] :
                              tilingData->multiCoreParams.totalSize;
    if constexpr (LAYOUT_Q == SLILayout::TND) {
        bStartIdx = FindBIndex(0, bS1Index, actualSum);
        s1StartIdx = bS1Index - actualSum;
        bEndIdx = FindBIndex(bStartIdx, bS1EndIndex - 1, actualSum);
        s1EndIdx = bS1EndIndex - actualSum;
    } else {
        bStartIdx = bS1Index / constInfo.s1Size;
        bEndIdx = (bS1EndIndex - 1) / constInfo.s1Size;
        s1StartIdx = bS1Index - bStartIdx * constInfo.s1Size;
        s1EndIdx = bS1EndIndex - bEndIdx * constInfo.s1Size;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::FindBIndex(int64_t bIndex, int64_t curIndex,
                                                                                    int64_t &accumulateLen)
{
    for (int index = bIndex; index < constInfo.bSize; index++) {
        int64_t actualLen = actualSeqLengthsQueryGm.GetValue(index);
        if (curIndex < actualLen) {
            return index;
        }
        accumulateLen = actualLen;
    }
    return tilingData->multiCoreParams.totalSize >= curIndex ? constInfo.bSize : -1;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int32_t SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::GetActualSeqLens(
    int32_t bIdx, int32_t defaultLens, GlobalTensor<int64_t> &actualSeqLensGm, SLILayout layout, int64_t &accumLen)
{
    if (actualSeqLensGm.GetSize() <= 0) {
        return defaultLens;
    }

    if (layout == SLILayout::TND) {
        if (bIdx == 0) {
            accumLen = 0;
            return actualSeqLensGm.GetValue(0);
        } else {
            accumLen = actualSeqLensGm.GetValue(bIdx - 1);
            return (actualSeqLensGm.GetValue(bIdx) - accumLen);
        }
    } else {
        return 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t SparseLightningIndexerGradKLLossKernelBase<CubeBlockType, VecBlockType>::GetEndS1Etx(
    int32_t bIdx, int32_t defaultLens, GlobalTensor<int64_t> &actualSeqLensGm, SLILayout layout)
{
    if (actualSeqLensGm.GetSize() <= 0) {
        return defaultLens;
    }

    if (layout == SLILayout::TND) {
        if (bIdx == 0) {
            return actualSeqLensGm.GetValue(0);
        } else {
            return (actualSeqLensGm.GetValue(bIdx) - actualSeqLensGm.GetValue(bIdx - 1));
        }
    } else {
        assert(false, "do not support current layout!\n");
        return 0;
    }
}
} // namespace SligKlLoss

#endif