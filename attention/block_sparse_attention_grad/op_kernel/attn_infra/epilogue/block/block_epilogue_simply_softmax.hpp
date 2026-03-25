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
 * \file block_epliogue_simply_softmax.h
 * \brief Block Epliogue Simply Softmax Kernel Implementation
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SIMPLY_SOFTMAX_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SIMPLY_SOFTMAX_HPP

#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "kernel_operator.h"


using namespace AscendC;

template<typename InDtype>
struct SimplySoftMaxInfo {
    LocalTensor<float> sTensor;
    LocalTensor<float> lseTensor;
    LocalTensor<float> lseBrocTensor;
    LocalTensor<float> pFp32Tensor;
    LocalTensor<InDtype> pFp16Tensor;

    GlobalTensor<float> sGm;
    GlobalTensor<float> lseGm;
    GlobalTensor<InDtype> pGm;
};

template<typename InDtype>
struct CalDsInfo {
    LocalTensor<float> dpFp32Tensor;
    LocalTensor<float> softmaxGradTensor;
    LocalTensor<float> pFp32Tensor;
    LocalTensor<InDtype> dsFp16Tensor;

    GlobalTensor<float> dpGm;
    GlobalTensor<float> softmaxGradGm;
    GlobalTensor<InDtype> dsGm;
};

namespace NpuArch::Epilogue::Block {
template <
    typename InputDType,
    typename OutputDtype,
    uint32_t INPUT_LAYOUT>
class SimpltSoftmax
{
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    struct Params {
        // Data members
        GM_ADDR s; // 连续
        GM_ADDR softmaxLse; //需要跳着搬运
        GM_ADDR dp; // 连续
        GM_ADDR blockSparseMask;
        GM_ADDR actualQSeqlen;
        GM_ADDR actualKvSeqlen;
        GM_ADDR softGradworkspace; // 需要跳着搬运
        GM_ADDR pWorkspace; // 连续
        GM_ADDR dsWorkspace; // 连续
        GM_ADDR tilingData;
        uint64_t actualRow = 0;
        uint64_t actualCol = 0;
        uint64_t processNums = 0;
        uint64_t curCoreBatch = 0;
        uint64_t curCoreN1Idx = 0;
        uint64_t curCoreS1Idx = 0;
        uint64_t curT1Idx = 0;

        // Methods
        __aicore__ inline
        Params() {}

        __aicore__ inline
        Params(
            GM_ADDR s_, GM_ADDR softmaxLse_,  GM_ADDR dp_, GM_ADDR blockSparseMask_,
            GM_ADDR actualQSeqlen_, GM_ADDR actualKvSeqlen_, GM_ADDR softGradworkspace_, GM_ADDR pWorkspace_, GM_ADDR dsWorkspace_, GM_ADDR tilingData_,
            uint64_t acutualRow_, uint64_t actualCol_, uint64_t processNums_, uint64_t curBatch_, uint64_t curN1_, uint64_t curS1_, uint64_t curT1_
        ) : s(s_), softmaxLse(softmaxLse_), dp(dp_), blockSparseMask(blockSparseMask_),
            actualQSeqlen(actualQSeqlen_), actualKvSeqlen(actualKvSeqlen_),
            softGradworkspace(softGradworkspace_), pWorkspace(pWorkspace_), dsWorkspace(dsWorkspace_), tilingData(tilingData_),
            actualRow(acutualRow_), actualCol(actualCol_), processNums(processNums_),
            curCoreBatch(curBatch_), curCoreN1Idx(curN1_), curCoreS1Idx(curS1_), curT1Idx(curT1_)
        {
            
        }    
    };

    NpuArch::Arch::Resource<ArchTag> resource;

    constexpr static uint64_t STAGES=1;
    constexpr static uint64_t INPUT_NUM = 2;
    constexpr static uint64_t DOUBLE_BUFFER = 2;
    constexpr static uint64_t BNSD = 1;
    constexpr static uint64_t TND = 0;
    constexpr static uint64_t proceeM = 128;
    constexpr static uint64_t proceeK = 128;
    constexpr static uint64_t BRCB_BASE_NUM = 8;
    constexpr static uint64_t REAPTE_BYTE = 256;

    constexpr static uint64_t BLOCK_BYTE_SIZE = 32;
    constexpr static uint64_t BLOCK_FP32_NUM = 8;
    constexpr static uint64_t BLOCK_16_NUM = 16;
    constexpr static uint64_t SFMG_HIGH_PERF_N_FACTOR = 8;
    constexpr static uint64_t SFMG_HIGH_PERF_D_FACTOR = 64;
    constexpr static uint64_t baseM =  128;

    uint64_t cBlockIdx = 0;
    uint64_t cubeCoreIdx = 0;
    uint64_t vecCoreIdx = 0;
    uint64_t row = 0; // 当前core需要处理q方向的s数
    uint64_t col = 0; // 当前core需要处理kv方向的s数
    uint64_t align32Col = 0;
    uint64_t align16Col = 0;
    uint64_t alignCol = 0;
    uint64_t alignRow = 0;
    uint64_t curCoreBatch = 0;
    uint64_t curCoreN1Idx = 0;      // q_n
    uint64_t curT1Idx = 0;          // q_t
    uint64_t curCoreS1Idx = 0;     // q_s 
    uint64_t maxQSeqlen = 0;
    uint64_t maxKvSeqlen = 0;
    uint64_t n1 = 0; // q_n
    uint64_t transpseStride = 0;

    uint64_t usedVecCoreNums = 0;
    uint64_t p16BaseBufLen = 0;
    uint64_t p32BaseBufLen = 0;
    float scaleValue = 0.0f;

    GlobalTensor<float> sGm;  // (N s1 s2)
    GlobalTensor<float> softmaxLseGm; // (N s1 1)
    GlobalTensor<float> dpGm; // (N s1 s2)
    GlobalTensor<InputDType> pWorkspaceGm; // (N s1 s2)
    GlobalTensor<float> softGradworkspaceGm; // (N s1 8)
    GlobalTensor<InputDType> dsWorkspaceGm; // (N s1 s2)

    GM_ADDR actualQSeqlen;
    GM_ADDR actualKvSeqlen;

    LocalTensor<float> sTensor[STAGES];
    LocalTensor<float> lseTensor[STAGES];
    LocalTensor<float> lseBrocTensor[STAGES];
    LocalTensor<float> pFp32Tensor[STAGES];
    LocalTensor<InputDType> p16Tensor[STAGES];
    LocalTensor<float> dpFp32Tensor[STAGES];
    LocalTensor<float> softmaxGradTensor[STAGES];
    LocalTensor<float> dsTensor[STAGES];
    LocalTensor<InputDType> ds16Tensor[STAGES];

    __aicore__ inline
    SimpltSoftmax(Params const &params)
    {
        vecCoreIdx = GetBlockIdx();
        cBlockIdx = vecCoreIdx;
        __gm__ BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tilingData);
        usedVecCoreNums = tilingData->usedVecCoreNum;

        if (cBlockIdx >= usedVecCoreNums) {
            return;
        }

        maxQSeqlen = tilingData -> maxQSeqlen;
        maxKvSeqlen = tilingData -> maxKvSeqlen;
        n1 = tilingData -> numHeads; // q_n
        col = params.actualCol;
        align32Col = (col + BLOCK_FP32_NUM - 1) / BLOCK_FP32_NUM * BLOCK_FP32_NUM; // fp32 对齐后的列数
        align16Col = (col + BLOCK_16_NUM - 1) / BLOCK_16_NUM * BLOCK_16_NUM;
        alignCol = (align32Col % BLOCK_16_NUM != 0) ? align16Col : align32Col;
        curCoreBatch = params.curCoreBatch;
        curCoreN1Idx = params.curCoreN1Idx;
        curT1Idx = params.curT1Idx;
        curCoreS1Idx = params.curCoreS1Idx;
        actualQSeqlen = params.actualQSeqlen;
        actualKvSeqlen = params.actualKvSeqlen;
        scaleValue = tilingData->scaleValue;

        uint64_t s2 = ((__gm__ uint64_t *)actualKvSeqlen)[curCoreBatch];
        uint64_t ubSize = tilingData->ubSize;
        uint64_t ubSizeEeachStage = ubSize  / STAGES / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE; // 32字节对齐

        row = params.actualRow;
        if (row <= 0) {
            return;
        }
        if constexpr(INPUT_LAYOUT == TND) {
            transpseStride = (n1 * 1 - 1) * sizeof(float);
        } else if constexpr(INPUT_LAYOUT == BNSD){
            transpseStride = 0;
        }

        // baseM =  128;
        // 分核 一个core 最大 128 * 128 一个vec 64 * 128
        uint64_t sBufferLen = baseM * 128 * sizeof(float); // max
        uint64_t lBufferLen = baseM * sizeof(float); // max
        uint64_t lBrobBufferLen = BRCB_BASE_NUM * baseM * sizeof(float);  // max
        uint64_t p32BufferLen = baseM * 128 * sizeof(float);
        uint64_t dpBufLen = sBufferLen;
        uint64_t p16BufLen = baseM * 128 * sizeof(InputDType);
        uint64_t ds16BufLen = p16BufLen;
        uint64_t dBufLen = BRCB_BASE_NUM * baseM * sizeof(float);
        p16BaseBufLen = p16BufLen;
        p32BaseBufLen = p32BufferLen;

        for (uint64_t i = 0; i < STAGES; i++) {
            // 第一轮 softmax 计算空间划分
            sTensor[i] = resource.ubBuf.template GetBufferByByte<float>((sBufferLen / 2) * i);
            pFp32Tensor[i] = sTensor[i]; // 复用s
            lseTensor[i] = resource.ubBuf.template GetBufferByByte<float>((lBufferLen / 2) * i); // 复用s
            p16Tensor[i] = resource.ubBuf.template GetBufferByByte<InputDType>(sBufferLen + (p16BufLen / 2) * i);
            lseBrocTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                sBufferLen + p16BufLen + (lBrobBufferLen / 2) * i);

            // 第二轮 ds 计算空间划分
            dpFp32Tensor[i] = 
                resource.ubBuf.template GetBufferByByte<float>(p32BufferLen + p16BufLen + (dpBufLen / 2) * i);
            softmaxGradTensor[i] = 
                resource.ubBuf.template GetBufferByByte<float>(p32BufferLen + p16BufLen + dpBufLen + (dBufLen / 2) * i);
            dsTensor[i] = pFp32Tensor[i]; // 复用s
            ds16Tensor[i] = p16Tensor[i]; // 复用p16
        }

        // 初始化 GM
        sGm.SetGlobalBuffer((__gm__ float *)params.s);
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)params.softmaxLse);
        dpGm.SetGlobalBuffer((__gm__ float *)params.dp );
        pWorkspaceGm.SetGlobalBuffer((__gm__ InputDType *)params.pWorkspace);
        softGradworkspaceGm.SetGlobalBuffer((__gm__ float *)params.softGradworkspace);
        dsWorkspaceGm.SetGlobalBuffer((__gm__ InputDType *)params.dsWorkspace);
    }
        
    __aicore__ inline
    ~SimpltSoftmax()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline
    void operator()();

    template <>
    __aicore__ inline
    void operator()<AscendC::AIC>()
    {

    }

    template <>
    __aicore__ inline
    void operator()<AscendC::AIV>()
    {
        if (cBlockIdx >= usedVecCoreNums || row <= 0) {
            return;
        }
        
        // col <= 128
        // 计算单loop的计算量及loop次数
        uint64_t eleBaseBuffNum = p32BaseBufLen / STAGES / sizeof(float); // 基本buffer块的元素数量
        uint64_t bufferRows = baseM / STAGES; // 一次lopp可以执行的最多row行数
        uint64_t rowLoopTimes = row / bufferRows;
        uint64_t tailRowNum = row - rowLoopTimes * bufferRows;

        uint64_t ping = 0;
        // 不包含尾行处理
        for (uint64_t i = 0; i < rowLoopTimes; i++) {
            auto eventId = ping ? EVENT_ID1 : EVENT_ID0;
            uint64_t curS1 = curCoreS1Idx + i * bufferRows;
            int32_t gmRowOffset = i * bufferRows * col;
            compute(gmRowOffset, bufferRows, col, curS1, ping);
      
            if (STAGES == DOUBLE_BUFFER) {
                ping = 1 - ping;
            }
        }

        if (tailRowNum > 0) {
            auto eventId = ping ? EVENT_ID1 : EVENT_ID0;

            uint64_t curS1 = curCoreS1Idx + rowLoopTimes * bufferRows;
            int32_t gmOffset =  rowLoopTimes * bufferRows * col;
            uint64_t tempRow = tailRowNum;
            compute(gmOffset, tempRow, col, curS1, ping);
            if (STAGES == DOUBLE_BUFFER) {
                ping = 1 - ping;
            }
        }
    }

    __aicore__ inline
    void compute(int32_t gmOffset, uint64_t row, uint64_t col, uint64_t curS1, uint64_t ping)
    {
        struct SimplySoftMaxInfo<InputDType> runSftInfo = {sTensor[ping], lseTensor[ping], lseBrocTensor[ping], pFp32Tensor[ping], p16Tensor[ping], sGm[gmOffset], softmaxLseGm, pWorkspaceGm[gmOffset]};
        struct CalDsInfo<InputDType> runDsInfo = {dpFp32Tensor[ping], softmaxGradTensor[ping], pFp32Tensor[ping], ds16Tensor[ping], dpGm[gmOffset], softGradworkspaceGm, dsWorkspaceGm[gmOffset]};
        SimplySoftmax(runSftInfo, row, col, curS1);
        CalDs(runDsInfo, row, col, curS1 );
    }


   /*
    * lse copy and brocast
    * lse input shape (b n s 1) or (t n 1)
    * out shape (b n s 8) or (n s 8)
    * dtype float
    */
    __aicore__ inline
    void LseBrocast(GlobalTensor<float> &LseGm, LocalTensor<float> &lse, LocalTensor<float> &lseFp32Brc, uint64_t count, uint64_t curS1)
    {
        uint64_t startOffset = 0;
        auto event_id = EVENT_ID0;
        if constexpr (INPUT_LAYOUT == TND) {
            uint64_t bOffset = n1 * ((__gm__ int64_t *)actualQSeqlen)[curCoreBatch];
            startOffset = bOffset + curS1 * n1 + curCoreN1Idx;
            // 对于TND 格式来说， 会进行leis (s n) -> (n s) 的transpose转换
            DataCopyPad(lseFp32Brc, LseGm[startOffset],
                    {static_cast<uint16_t>(count), static_cast<uint32_t>(1 * sizeof(float)),
                    static_cast<uint32_t>(transpseStride), 0, 0},
                    {false, 0, 0, 0});
        } else {
            startOffset = curCoreBatch * (n1 * maxQSeqlen) + curCoreN1Idx * maxQSeqlen + curS1;
            DataCopyPad(lse, LseGm[startOffset],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(count * sizeof(float)),
                    static_cast<uint32_t>(0), 0, 0},
                    {false, 0, 0, 0});

            set_flag(PIPE_MTE2, PIPE_V, event_id);
            wait_flag(PIPE_MTE2, PIPE_V, event_id);

            uint8_t repeatimes = CeilDiv(count, BRCB_BASE_NUM);
            Brcb(lseFp32Brc, lse, repeatimes, {1, 8});

            set_flag(PIPE_V, PIPE_MTE2, event_id);
            wait_flag(PIPE_V, PIPE_MTE2, event_id);
        }
    }

    /*
        * brief: Compute the elementwise multiplication of a tensor of shape (m, n) and a tensor of shape
        * ubIn0:[m, n], ubIn1[m, 8]
    */
    __aicore__ inline
    void SubBrcb(LocalTensor<float> const &ubOut, LocalTensor<float> const &ubIn0, LocalTensor<float> const &ubIn1, uint64_t row, uint64_t col)
    {
        uint32_t maxRepeatNum = 255;
        uint32_t eleNumPerBlk = static_cast<uint32_t>(BLOCK_BYTE_SIZE) / static_cast<uint32_t>(sizeof(float));

        uint32_t blkNumPerColumn = col / eleNumPerBlk;
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = blkNumPerColumn;
        repeatParams.src0BlkStride = blkNumPerColumn;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = 1;
        repeatParams.src0RepStride = 1;
        repeatParams.src1RepStride = 0;

        // 执行一次sub，迭代次数 col / oneblock ， 一次迭代计算 row * oneblock 元素， 
        uint32_t rowNumPerCompute = BLK_NUM_PER_VECTOR_FRACTAL; // 256 / 32 = 8 即per_repeat / per_block = 8
        uint32_t colNumPerCompute = eleNumPerBlk * maxRepeatNum; // 255 * 8 最多可以计算列的元素
        for (uint32_t rowOffset = 0; rowOffset < row; rowOffset += rowNumPerCompute) {
            uint32_t residueM = row - rowOffset;
            uint32_t currentRowNum = (residueM > rowNumPerCompute) ? rowNumPerCompute : residueM;
            uint64_t mask = static_cast<uint64_t>(currentRowNum) * static_cast<uint64_t>(eleNumPerBlk);
            for (uint32_t colOffset = 0; colOffset < col; colOffset += colNumPerCompute) {
                uint32_t residueN = col - colOffset;
                uint32_t currentColNum = (residueN > colNumPerCompute) ? colNumPerCompute : residueN;
                uint8_t repeatTimes = static_cast<uint8_t>(currentColNum / eleNumPerBlk);
                uint32_t tailColNUm = currentColNum - static_cast<uint32_t>(repeatTimes) * eleNumPerBlk;
                uint32_t trailTimes = tailColNUm == 0 ? 0 : 1;
                AscendC::Sub(
                    ubOut[rowOffset * col + colOffset],
                    ubIn0[rowOffset * col + colOffset],
                    ubIn1[rowOffset * eleNumPerBlk],
                    mask, repeatTimes, repeatParams
                );
            }
        }
    }

    /*
        * p = exp(s - lse_brc)
        * lse  shape (n s1 8) fp32 需非连续搬运
        * s shape (n s1 s2) fp32 连续
    */
    __aicore__ inline
    void SimplySoftmax(struct SimplySoftMaxInfo<InputDType> runInfo, uint64_t row, uint64_t col, uint64_t curS1)
    {
        // row <= 128 col <= 128
        LocalTensor<float> &sLocal = runInfo.sTensor;
        LocalTensor<float> &lse = runInfo.lseTensor;
        LocalTensor<float> &lseFp32Brc = runInfo.lseBrocTensor;
        LocalTensor<float> &p32Local = runInfo.pFp32Tensor;
        LocalTensor<InputDType> &p16Local = runInfo.pFp16Tensor;

        GlobalTensor<float> s = runInfo.sGm;
        GlobalTensor<float> lseGm = runInfo.lseGm;
        GlobalTensor<InputDType> pGm = runInfo.pGm;

        uint64_t countAlign = row * alignCol;
        uint64_t count =  row * col;
        
        auto event_id = EVENT_ID0;
        set_flag(PIPE_MTE3, PIPE_MTE2, event_id);
        wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);

        LseBrocast(lseGm, lse, lseFp32Brc, row, curS1);

        if (align32Col * sizeof(InputDType) % BLOCK_BYTE_SIZE == 0) {
            DataCopyPad(sLocal, s, {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 0, 0}, 
                    {true, 0, static_cast<uint8_t>(align32Col - col), 0});
        } else {
            DataCopyPad(sLocal, s, {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 1, 0}, 
                    {true, 0, static_cast<uint8_t>(align32Col - col), 0});
        }

        set_flag(PIPE_MTE2, PIPE_V, event_id);
        wait_flag(PIPE_MTE2, PIPE_V, event_id);
        
        Muls(sLocal, sLocal, (float)scaleValue, countAlign);
        AscendC::PipeBarrier<PIPE_V>();

        SubBrcb(p32Local, sLocal, lseFp32Brc, row, alignCol);
        AscendC::PipeBarrier<PIPE_V>();

        Exp(p32Local, p32Local, countAlign);
        AscendC::PipeBarrier<PIPE_V>();

        Cast(p16Local, p32Local, AscendC::RoundMode::CAST_ROUND, countAlign);

        set_flag(PIPE_V, PIPE_MTE3, event_id);
        wait_flag(PIPE_V, PIPE_MTE3, event_id);

        DataCopyPad(pGm, p16Local, {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(InputDType)), 0, 0, 0});

        set_flag(PIPE_MTE3, PIPE_MTE2, event_id);
        wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);
    }

    /*
        * ds = p * (dp - D)
        * dp  shape (n s1 s2) fp32 连续
        * D shape (n s1 8) fp32 需非连续搬运
        * p shape (n s1 s2) fp32 连续
    */
    __aicore__ inline
    void CalDs(struct CalDsInfo<InputDType> runInfo,  uint64_t row, uint64_t col, uint64_t curS1)
    {
        LocalTensor<float> dpLocal = runInfo.dpFp32Tensor;
        LocalTensor<float> dLocal = runInfo.softmaxGradTensor;
        LocalTensor<float> p32Local = runInfo.pFp32Tensor;
        LocalTensor<InputDType> ds16Tensor = runInfo.dsFp16Tensor;

        GlobalTensor<float> dp = runInfo.dpGm;
        GlobalTensor<float> d = runInfo.softmaxGradGm;
        GlobalTensor<InputDType> ds = runInfo.dsGm;

        uint64_t count = row * col;
        uint64_t countAlign = row * alignCol;

        auto event_id = EVENT_ID0;
        set_flag(PIPE_MTE3, PIPE_MTE2, event_id);
        wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);

        if (align32Col * sizeof(InputDType) % BLOCK_BYTE_SIZE == 0) {
            DataCopyPad(dpLocal, dp, {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 0, 0}, 
                    {true, 0, static_cast<uint8_t>(align32Col - col), 0});
         } else {
            DataCopyPad(dpLocal, dp, {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 1, 0}, 
                    {true, 0, static_cast<uint8_t>(align32Col - col), 0});
        }


        CopyDIn(d, dLocal, row, curS1);

        set_flag(PIPE_MTE2, PIPE_V, event_id);
        wait_flag(PIPE_MTE2, PIPE_V, event_id);

        SubBrcb(dpLocal, dpLocal, dLocal, row, alignCol);
        AscendC::PipeBarrier<PIPE_V>();

        Mul(dpLocal, p32Local, dpLocal, countAlign);
        AscendC::PipeBarrier<PIPE_V>();

        Cast(ds16Tensor, dpLocal, AscendC::RoundMode::CAST_ROUND, countAlign);

        set_flag(PIPE_V, PIPE_MTE3, event_id);
        wait_flag(PIPE_V, PIPE_MTE3, event_id);

        DataCopyPad(ds, ds16Tensor, {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(InputDType)), 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();

        set_flag(PIPE_MTE3, PIPE_MTE2, event_id);
        wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);
    }

    /*
        * D shape ((n s1 8) or (b n s1 8) fp32 
    */
    __aicore__ inline
    void CopyDIn(GlobalTensor<float> &d, LocalTensor<float> &dLocal, uint64_t count, uint64_t curS1)
    {
        uint64_t startOffset = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            uint64_t bOffset = n1 * ((__gm__ int64_t *)actualQSeqlen)[curCoreBatch] * BRCB_BASE_NUM;
            startOffset = bOffset + curS1 * n1 * BRCB_BASE_NUM + curCoreN1Idx * BRCB_BASE_NUM;               
        } else {
            startOffset = curCoreBatch * (n1 * maxQSeqlen * BRCB_BASE_NUM) + curCoreN1Idx * maxQSeqlen * BRCB_BASE_NUM + curS1 * BRCB_BASE_NUM;
        }
        DataCopyPad(dLocal, d[startOffset],
                    {static_cast<uint16_t>(count), static_cast<uint32_t>(BRCB_BASE_NUM * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
    }
};

}

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SIMPLY_SOFTMAX_HPP
