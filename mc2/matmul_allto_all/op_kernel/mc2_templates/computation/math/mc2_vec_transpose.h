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
 * \file mc2_vec_transpose.h
 * \brief
 */

#ifndef MC2_VEC_TRANSPOSE_H
#define MC2_VEC_TRANSPOSE_H

namespace MC2KernelTemplate{
using namespace AscendC;
template <typename tranposeDataType>
class MC2VecTranspose {
public:
    __aicore__ inline MC2VecTranspose(TPipe *tPipe) {
        tPipe_ = tPipe;
    }

protected:
    static constexpr uint32_t MAIN_TILEM_SPLIT_SIZE = 64;
    static constexpr uint32_t MAIN_INNER_AXIS_SPLIT_SIZE = 256;
    static constexpr uint32_t MULTIPLE_AIV_TO_AIC = 2;
    static constexpr uint32_t TWO_FACTOR = 2;
    static constexpr uint32_t ONE_RATIO_TWO = 2;

//成员变量定义
    GlobalTensor<tranposeDataType> tranposeGm_;
    GlobalTensor<tranposeDataType> ubOutGm_;
    TPipe *tPipe_;
    TQue<QuePosition::VECIN, 1> vecInQueue_;
    uint64_t srcStride_ = 0;
    uint64_t dstStride_ = 0;
    uint64_t tileM_ = 0;
    uint64_t rankSize_ = 0;
    uint64_t innerAxis_ = 0;
    uint64_t innerOffsetIn_ = 0;
    uint64_t innerOffsetOut_ = 0;

    //定义切块变量
    uint32_t tileMSplitCnt_ = 0;
    uint32_t innerAxisSplitCnt_ = 0;
    uint32_t totalCnt_ = 0;
    uint32_t tailBlockCnt_ = 0;
    uint32_t tailSplitTileM_ = 0;
    uint32_t tailSplitInnerAxis_ = 0;

    uint32_t calBlockRound_ = 0;
    uint32_t usedCoreNum_ = 0;

//内部函数定义
    __aicore__ inline void getSplitCnt() {
        tileMSplitCnt_ = Ceil(tileM_, MAIN_TILEM_SPLIT_SIZE);   //可优化为移位操作
        innerAxisSplitCnt_ = Ceil(innerAxis_, MAIN_INNER_AXIS_SPLIT_SIZE);
        totalCnt_ = tileMSplitCnt_ * innerAxisSplitCnt_;

        tailSplitTileM_ = tileM_ - (tileMSplitCnt_ - 1) * MAIN_TILEM_SPLIT_SIZE;
        tailSplitInnerAxis_ = innerAxis_ - (innerAxisSplitCnt_ - 1) * MAIN_INNER_AXIS_SPLIT_SIZE;

        if (totalCnt_ < usedCoreNum_) {
            usedCoreNum_ = totalCnt_;
        }

        calBlockRound_ = totalCnt_ / usedCoreNum_;
        tailBlockCnt_ = totalCnt_ - calBlockRound_ * usedCoreNum_;
    }

    __aicore__ inline void VecSync() {
        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_AIV_ONLY_ALL);
        CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL);
    }

    __aicore__ inline void CopyInToUB(const uint64_t & srcGmOffset, const uint64_t & curTileMCntIdx,
                            const uint64_t & curInnerAxisCntIdx) {
        LocalTensor<tranposeDataType> vecInBuf = vecInQueue_.template AllocTensor<tranposeDataType>();

        DataCopyExtParams loadGm2UbParams;
        uint32_t blockCount = (curTileMCntIdx == (tileMSplitCnt_ - 1)) ? (tailSplitTileM_) :
                                            (MAIN_TILEM_SPLIT_SIZE);  //根据tileM方向是否是尾块确定blockCount的大小
        loadGm2UbParams.blockCount = static_cast<uint16_t>(blockCount);
        uint64_t blockLen = (curInnerAxisCntIdx == (innerAxisSplitCnt_ - 1)) ? (tailSplitInnerAxis_) :
                                        (MAIN_INNER_AXIS_SPLIT_SIZE);  //根据innerAxis方向是否是尾块确定blockLen的大小
        loadGm2UbParams.blockLen = static_cast<uint32_t>(blockLen * sizeof(tranposeDataType));
        loadGm2UbParams.srcStride = static_cast<int64_t>(innerOffsetIn_ * sizeof(tranposeDataType) -
                                                    loadGm2UbParams.blockLen);
        loadGm2UbParams.dstStride = static_cast<int64_t>(0);

        DataCopyPadExtParams<tranposeDataType> padExtParams{true, 0, 0, *reinterpret_cast<tranposeDataType *>(uint8_t(0))};

        DataCopyPad<tranposeDataType, PaddingMode::Normal>(vecInBuf, tranposeGm_[srcGmOffset], loadGm2UbParams,
                                                        padExtParams);
        vecInQueue_.EnQue(vecInBuf);
    }

    __aicore__ inline void CopyOutToGm(const uint64_t & dstGmOffset, const uint64_t & curTileMCntIdx,
                            const uint64_t & curInnerAxisCntIdx) {
        LocalTensor<tranposeDataType> vecOutBuf = vecInQueue_.template DeQue<tranposeDataType>();

        DataCopyExtParams loadUb2GmParams;
        uint32_t blockCount = (curTileMCntIdx == (tileMSplitCnt_ - 1)) ? (tailSplitTileM_) :
                                            (MAIN_TILEM_SPLIT_SIZE);  //根据tileM方向是否是尾块确定blockCount的大小
        loadUb2GmParams.blockCount = static_cast<uint16_t>(blockCount);
        uint64_t blockLen = (curInnerAxisCntIdx == (innerAxisSplitCnt_ - 1)) ? (tailSplitInnerAxis_) :
                                        (MAIN_INNER_AXIS_SPLIT_SIZE);  //根据innerAxis方向是否是尾块确定blockLen的大小
        loadUb2GmParams.blockLen = static_cast<uint32_t>(blockLen * sizeof(tranposeDataType));
        loadUb2GmParams.srcStride = 0;
        loadUb2GmParams.dstStride = static_cast<int64_t>(innerOffsetOut_ * sizeof(tranposeDataType) - loadUb2GmParams.blockLen);

        DataCopyPad<tranposeDataType, PaddingMode::Normal>(ubOutGm_[dstGmOffset], vecOutBuf, loadUb2GmParams);
        vecInQueue_.FreeTensor(vecOutBuf);
    }

    __aicore__ inline void VecTransposeProcess(const uint64_t & srcGmOffset, const uint64_t & dstGmOffset, const uint64_t & curTileMCntIdx,
                            const uint64_t & curInnerAxisCntIdx) {
        CopyInToUB(srcGmOffset, curTileMCntIdx, curInnerAxisCntIdx);
        PipeBarrier<PIPE_ALL>();
        CopyOutToGm(dstGmOffset, curTileMCntIdx, curInnerAxisCntIdx);
        PipeBarrier<PIPE_ALL>();
    }
public:
    //innerAxis为内轴的大小，上层调用Init时传入的innerAxis大小已经除以卡数了
    __aicore__ inline void Init(GM_ADDR tranposeSrcAddr, GM_ADDR tranposeDstAddr, uint64_t rankSize, uint64_t innerAxis,
        uint64_t tileM,  uint64_t srcStride, uint64_t dstStride, uint64_t innerOffsetIn, uint64_t innerOffsetOut) {
        tPipe_->Reset();
        tileM_ = tileM;
        rankSize_ = rankSize;

        usedCoreNum_ = GetBlockNum() * ONE_RATIO_TWO;   //CV(1:2) 先初始化为满核
        srcStride_ = srcStride;
        dstStride_ = dstStride;
        innerAxis_ = innerAxis;
        innerOffsetIn_ = innerOffsetIn;
        innerOffsetOut_ = innerOffsetOut;

        getSplitCnt();

        tranposeGm_.SetGlobalBuffer((__gm__ tranposeDataType *)tranposeSrcAddr);
        ubOutGm_.SetGlobalBuffer((__gm__ tranposeDataType *)tranposeDstAddr);

        const uint32_t twoUbSize = AscendC::TOTAL_UB_SIZE / TWO_FACTOR;
        tPipe_->InitBuffer(vecInQueue_, 1, twoUbSize);
    }
    __aicore__ inline void Process() {
        if (GetBlockIdx() >= usedCoreNum_) {
            return;
        }

        if (GetBlockIdx() < tailBlockCnt_) {
            ++calBlockRound_;
        }

        for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) { //
            //uint64_t srcGmOffset = rankIdx * rankStride_;   //根据rankID计算偏移
            uint64_t srcGmOffset = rankIdx * srcStride_;   //获取卡源数据的起始地址
            uint64_t dstGmOffset = rankIdx * dstStride_;   //获取卡目的数据的起始地址

            for (uint64_t blockIdx = 0; blockIdx < calBlockRound_; blockIdx++) {
                uint64_t curblockIdxInTotal = blockIdx * usedCoreNum_ + GetBlockIdx();
                uint64_t curInnerAxisCntIdx = curblockIdxInTotal % innerAxisSplitCnt_;
                uint64_t curTileMCntIdx = curblockIdxInTotal / innerAxisSplitCnt_;
                //计算切块的额外偏移
                uint64_t srcGmOffsetFinal = srcGmOffset + (curTileMCntIdx * MAIN_TILEM_SPLIT_SIZE * innerOffsetIn_ + 
                    curInnerAxisCntIdx * MAIN_INNER_AXIS_SPLIT_SIZE);
                uint64_t dstGmOffsetFinal = dstGmOffset + (curTileMCntIdx  * MAIN_TILEM_SPLIT_SIZE * innerOffsetOut_) +
                    (curInnerAxisCntIdx * MAIN_INNER_AXIS_SPLIT_SIZE);
                VecTransposeProcess(srcGmOffsetFinal, dstGmOffsetFinal, curTileMCntIdx, curInnerAxisCntIdx);
            }
        }
    }

    __aicore__ inline void Destroy()
    {
        vecInQueue_.FreeAllEvent();
    }
};
}

#endif