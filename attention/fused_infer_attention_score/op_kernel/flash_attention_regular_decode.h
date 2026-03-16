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
 * \file flash_attention_regular_decode.h
 * \brief
 */
#ifndef FLASH_ATTENTION_REGULAR_DECODE_H
#define FLASH_ATTENTION_REGULAR_DECODE_H

#include "kernel_common.hpp"

using namespace NpuArch;
using namespace KernelCommon;

namespace SplitFuse {
    template <
        class BlockMmadQK,
        class BlockMmadPV,
        class EpilogueOnlineSoftmax,
        class EpilogueRescaleO,
        class EpilogueInitOut,
        bool PAGED_CACHE_FLAG,
        FaiKernel::MaskType MASK_TYPE = FaiKernel::MaskType::NO_MASK,
        FaiKernel::inputLayout INPUT_LAYOUT = FaiKernel::inputLayout::BSND>
    class FAInferKernelDecoding {
    public:
        using ArchTag = typename BlockMmadQK::ArchTag;
        using L1TileShape = typename BlockMmadQK::L1TileShape;
        using ElementQ = typename BlockMmadQK::ElementA;
        using LayoutQ = typename BlockMmadQK::LayoutA;
        using ElementK = typename BlockMmadQK::ElementB;
        using LayoutK = typename BlockMmadQK::LayoutB;
        using ElementS = typename BlockMmadQK::ElementC;
        using LayoutS = typename BlockMmadQK::LayoutC;

        using ElementP = typename BlockMmadPV::ElementA;
        using LayoutP = typename BlockMmadPV::LayoutA;
        using ElementV = typename BlockMmadPV::ElementB;
        using LayoutV = typename BlockMmadPV::LayoutB;

        using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
        using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;

        using ElementO = typename EpilogueRescaleO::ElementOutput;
        using LayoutO = typename EpilogueRescaleO::LayoutOutput;

        using ElementOTmp = typename EpilogueRescaleO::ElementInput;
        using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;

        using ElementLse = typename EpilogueRescaleO::ElementLse;
        using LayoutLse = typename EpilogueRescaleO::LayoutLse;

        using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
        using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

        static constexpr Epilogue::LseMode LSE_MODE = EpilogueRescaleO::LSE_MODE;
        static constexpr Epilogue::SinkMode SINK_MODE = EpilogueOnlineSoftmax::SINK_MODE;

        // Methods
        __aicore__ inline
        FAInferKernelDecoding() {}

        __aicore__ inline
        void operator()(FAIKernelParams const &params)
        {
            __gm__ FAInferTilingData *fATilingData = reinterpret_cast<__gm__ FAInferTilingData *>(params.tiling);
            uint64_t mm1OutSize = fATilingData->mm1OutSize;
            uint64_t smOnlineOutSize = fATilingData->smOnlineOutSize;
            uint64_t mm2OutSize = fATilingData->mm2OutSize;
            uint32_t batch = fATilingData->batch;
            uint32_t qHeads = fATilingData->numHeads;
            uint32_t kvHeads = fATilingData->kvHeads;
            uint32_t embed = fATilingData->embeddingSize;
            uint32_t embedV = fATilingData->embeddingSizeV;
            uint32_t pagedBlockSize = fATilingData->blockSize;
            uint32_t maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;
            uint32_t firstBatchTaskNum = fATilingData->firstBatchTaskNum;
            uint32_t totalTaskNum = fATilingData->totalTaskNum;
            uint32_t blockSize = fATilingData->blockSize;
            uint32_t maskType = fATilingData->maskType;
            float scaleValue = fATilingData->scaleValue;
            uint32_t sparseMode = fATilingData->sparseMode;
            int64_t preToken = fATilingData->preToken;
            int64_t nextToken = fATilingData->nextToken;
            AscendC::GlobalTensor<ElementQ> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
            AscendC::ListTensorDesc keyListTensorDescInit((__gm__ void*)params.k);
            AscendC::ListTensorDesc valueListTensorDescInit((__gm__ void*)params.v);
            __gm__ uint8_t* currentKey = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
            __gm__ uint8_t* currentValue = (__gm__ uint8_t*)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
            AscendC::GlobalTensor<ElementK> gK;
            gK.SetGlobalBuffer((__gm__ ElementK *)currentKey);
            AscendC::GlobalTensor<ElementK> gV;
            gV.SetGlobalBuffer((__gm__ ElementK *)currentValue);
            AscendC::GlobalTensor<ElementMask> gMask;
            gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
            AscendC::GlobalTensor<int32_t> gBlockTable;
            gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
            AscendC::GlobalTensor<int64_t> gActualQseqlen;
            gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            AscendC::GlobalTensor<int64_t> gActualKvseqlen;
            gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
            AscendC::GlobalTensor<ElementO> gO;
            gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
            AscendC::GlobalTensor<ElementLse> gLse;
            gLse.SetGlobalBuffer((__gm__ ElementLse *)params.lse);
            AscendC::GlobalTensor<ElementS> gS;
            gS.SetGlobalBuffer((__gm__ ElementS *)(params.workSpace));
            AscendC::GlobalTensor<ElementP> gP;
            gP.SetGlobalBuffer((__gm__ ElementP *)(params.workSpace + mm1OutSize));
            AscendC::GlobalTensor<ElementOTmp> gOTmp;
            gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + mm1OutSize + smOnlineOutSize));
            AscendC::GlobalTensor<ElementOTmp> gOUpdate;
            gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace +
                mm1OutSize + smOnlineOutSize + mm2OutSize));
            AscendC::GlobalTensor<bfloat16_t> gSink;
            gSink.SetGlobalBuffer((__gm__ bfloat16_t *)(params.sink));

            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
#ifdef __DAV_C220_CUBE__
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
            
            uint32_t kDynNum = NpuArch::Detail::Alignment::RoundUp(embed, NUM_128);
            kDynNum = kDynNum < NUM_256 ? NUM_256 : kDynNum;
            uint32_t maxQKPL1Size = L1_MAX_SIZE - embedV * MAX_KV_STACK_LEN * sizeof(ElementV);
            uint32_t maxQL1Size = Q_TILE_CEIL * kDynNum * sizeof(ElementQ);
            uint32_t maxNDynNum =
                ((maxQKPL1Size - maxQL1Size) / kDynNum / sizeof(ElementV) / DOUBLE_BUFFER) / NUM_32 * NUM_32;

            uint32_t nDynNum = maxNDynNum < L1_MAX_N_NUM ? maxNDynNum : L1_MAX_N_NUM;
            nDynNum = L1_MAX_N_NUM % nDynNum != 0 ?
                NpuArch::Detail::Alignment::RoundDown((nDynNum - 1), NUM_32) : nDynNum;

            uint32_t L1_QK_SIZE = BlockMmadQK::L1TileShape::M * kDynNum * sizeof(ElementQ);
            blockMmadQK.init(resource, nDynNum, kDynNum);
            uint32_t kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M;
            blockMmadPV.init(resource, nDynNum, kPVDynNum, L1_QK_SIZE);
#endif
#ifdef __DAV_C220_VEC__
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);

            epilogueOnlineSoftmax.init(resource, scaleValue);
            epilogueRescaleO.init(resource);
            epilogueInitOut.init(resource);

            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
            uint64_t strideQ = static_cast<uint64_t>(qHeads * embed);
            uint64_t strideO = static_cast<uint64_t>(qHeads * embedV);
            uint64_t strideK = static_cast<uint64_t>(kvHeads * embed);
            uint64_t strideV = static_cast<uint64_t>(kvHeads * embedV);
            uint32_t embedRound = NpuArch::Detail::Alignment::RoundUp(embed, FaiKernel::BLOCK_SIZE);
            uint32_t embedRoundV = NpuArch::Detail::Alignment::RoundUp(embedV, FaiKernel::BLOCK_SIZE);
            uint32_t groupSize = qHeads / kvHeads;

            uint64_t qBOffset = 0;
            uint64_t kBOffset = 0;
            uint64_t vBOffset = 0;
            uint64_t oBOffset = 0;
            uint64_t lseBOffset = 0;
            uint64_t blockBOffset = 0;

            uint32_t preTotalTaskNum = 0;
            uint32_t curBatch = 0;
            uint32_t totalQTokens = static_cast<uint32_t>(gActualQseqlen.GetValue(batch - 1));
            uint32_t qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
            uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
            if constexpr(INPUT_LAYOUT == FaiKernel::inputLayout::TND) {
                uint32_t prevQSeqlenSum = (curBatch == 0) ?
                    0 : static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch - 1));
                qSeqlen = qSeqlen - prevQSeqlenSum;
                if constexpr (!PAGED_CACHE_FLAG) {
                    uint32_t prevKvSeqlenSum = (curBatch == 0) ?
                        0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch - 1));
                    kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                }
            }
            uint32_t curGBlockTile = GetQNBlockTile(qSeqlen, groupSize);
            uint32_t curGBlockNum = NpuArch::Detail::Alignment::CeilDiv(groupSize, curGBlockTile); // 8
            uint32_t curQSBlockTile = GetQSBlockTileDecode(qSeqlen);
            uint32_t curQSBlockNum = NpuArch::Detail::Alignment::CeilDiv(qSeqlen, curQSBlockTile);
            uint32_t curQSGBlockTile = curGBlockTile * curQSBlockTile;
            uint32_t curKvNBlockTile = curGBlockTile < groupSize ? 1 : GetKvNBlockTile(curQSGBlockTile, kvHeads); // 2
            uint32_t curKvNBlockNum = NpuArch::Detail::Alignment::CeilDiv(kvHeads, curKvNBlockTile); // 1
            uint32_t curTotalTaskNum = firstBatchTaskNum;

            for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
                while (taskIdx >= curTotalTaskNum) {
                    ++curBatch;
                    preTotalTaskNum = curTotalTaskNum;
                    qBOffset += qSeqlen * strideQ;
                    if constexpr (!PAGED_CACHE_FLAG) {
                        kBOffset += static_cast<uint64_t>(kvSeqlen * strideK);
                        vBOffset += static_cast<uint64_t>(kvSeqlen * strideV);
                    } else {
                        blockBOffset += static_cast<uint64_t>(maxNumBlocksPerBatch);
                    }
                    oBOffset += static_cast<uint64_t>(qSeqlen * strideO);
                    lseBOffset += static_cast<uint64_t>(qSeqlen * qHeads);

                    qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
                    kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                    if constexpr(INPUT_LAYOUT == FaiKernel::inputLayout::TND) {
                        uint32_t prevQSeqlenSum = (curBatch == 0) ?
                            0 : static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch - 1));
                        qSeqlen = qSeqlen - prevQSeqlenSum;
                        if constexpr (!PAGED_CACHE_FLAG) {
                            uint32_t prevKvSeqlenSum = (curBatch == 0) ?
                                0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch - 1));
                            kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                        }
                    }
                    curGBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                    curGBlockNum = NpuArch::Detail::Alignment::CeilDiv(groupSize, curGBlockTile);
                    curQSBlockTile = GetQSBlockTileDecode(qSeqlen);
                    curQSBlockNum = NpuArch::Detail::Alignment::CeilDiv(qSeqlen, curQSBlockTile);
                    curQSGBlockTile = curGBlockTile * curQSBlockTile;
                    curKvNBlockTile = curGBlockTile < groupSize ? 1 : GetKvNBlockTile(curQSGBlockTile, kvHeads);
                    curKvNBlockNum = NpuArch::Detail::Alignment::CeilDiv(kvHeads, curKvNBlockTile);
                    curTotalTaskNum += curQSBlockNum * curGBlockNum * curKvNBlockNum;
                }
                uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
                uint32_t qSBlockIdx = taskIdxCurBatch / (curGBlockNum * curKvNBlockNum);
                uint32_t gKvNBlockIdx = taskIdxCurBatch - qSBlockIdx * (curGBlockNum * curKvNBlockNum);
                uint32_t gBlockIdx = gKvNBlockIdx / curKvNBlockNum;
                uint32_t kvNBlockIdx = gKvNBlockIdx - gBlockIdx * curKvNBlockNum;

                uint32_t kvNStartIdx = kvNBlockIdx * curKvNBlockTile;
                uint32_t qNStartIdx = kvNStartIdx * groupSize + gBlockIdx * curGBlockTile;

                uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1U)) ?
                    (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
                uint32_t gBlockSize = (gBlockIdx == (curGBlockNum - 1U)) ?
                    (groupSize - gBlockIdx * curGBlockTile) : curGBlockTile;
                uint32_t kvNBlockSize = (kvNBlockIdx == (curKvNBlockNum - 1U)) ?
                    (kvHeads - kvNBlockIdx * curKvNBlockTile) : curKvNBlockTile;
                uint32_t rowNum = qSBlockSize * gBlockSize;
                uint32_t rowNumRound = NpuArch::Detail::Alignment::RoundUp(rowNum, FaiKernel::BLOCK_SIZE);

                uint64_t qSOffset = static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideQ;
                uint64_t qNStartOffset = static_cast<uint64_t>(qNStartIdx * embed);
                uint64_t kNStartOffset = static_cast<uint64_t>(kvNStartIdx * embed);
                uint64_t vNStartOffset = static_cast<uint64_t>(kvNStartIdx * embedV);
                uint64_t oSOffset = static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideO;
                uint64_t oNStartOffset = static_cast<uint64_t>(qNStartIdx * embedV);
                uint64_t lseTokenOffset = static_cast<uint64_t>(qSBlockIdx * curQSBlockTile * qHeads);

                int64_t noSkipKvS = static_cast<int64_t>(kvSeqlen);
                if (maskType != 0U) {
                    int64_t diffS = kvSeqlen - qSeqlen;
                    diffS = (diffS < 0) ? 0 : diffS;
                    noSkipKvS = (qSBlockIdx + 1U) * curQSBlockTile + diffS;
                    noSkipKvS = AscendC::Std::min(static_cast<int64_t>(kvSeqlen), noSkipKvS);
                }
                uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, pagedBlockSize);

                uint32_t blockStackNum = MAX_KV_STACK_LEN / pagedBlockSize;
                uint32_t stackSeqTile;
                uint32_t stackSeqTilePad = blockStackNum * pagedBlockSize;
                uint32_t preKVNum = PRE_LAUNCH * blockStackNum;
                int32_t stackSeqCount = 0;
#ifdef __DAV_C220_CUBE__
                LayoutQ layoutQTemp(rowNum, embed);
                LayoutK layoutKTemp(strideK, blockStackNum * pagedBlockSize);
                LayoutV layoutVTemp(blockStackNum * pagedBlockSize, strideV);
#endif
                for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal + preKVNum; kvSIdx += blockStackNum) {

                    if (kvSIdx < kvSLoopNumTotal) {

                        if (kvSIdx + blockStackNum > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - kvSIdx * pagedBlockSize;
                        } else {
                            stackSeqTile = pagedBlockSize * blockStackNum;
                        }
                        uint32_t curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1U);
#ifdef __DAV_C220_CUBE__

                        uint64_t gmOffsetQGmtoL1 = qBOffset + qSOffset + qNStartOffset;
                        if (kvSIdx == 0) {
                            uint32_t taskRowNum = rowNum * kvNBlockSize;
                            LayoutQ layoutQL1(taskRowNum, embed);
                            uint32_t taskColNum = gBlockSize * kvNBlockSize;
                            blockMmadQK.loadQGM(gQ[gmOffsetQGmtoL1], layoutQL1, taskRowNum, taskColNum, qHeads, kvNBlockSize);
                        }
#endif

                        for (uint32_t kvNIncreIdx = 0; kvNIncreIdx < kvNBlockSize; kvNIncreIdx++) {

                            uint64_t gmOffsetQ = qBOffset + qSOffset + qNStartOffset +
                                static_cast<uint64_t>(kvNIncreIdx * groupSize * embed);
                            uint64_t gmOffsetK = kBOffset + kNStartOffset +
                                static_cast<uint64_t>(kvNIncreIdx * embed);
                            uint32_t sWorkspaceIncreOffset = kvNIncreIdx * rowNum * MAX_KV_STACK_LEN;
                            uint64_t gmOffsetS =
                                static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                                curStackTileMod * WORKSPACE_BLOCK_SIZE_DB + sWorkspaceIncreOffset);
                            GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                            LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
#ifdef __DAV_C220_CUBE__
                            blockMmadQK(
                                gQ[gmOffsetQ],
                                gK[gmOffsetK],
                                gS[gmOffsetS],
                                gBlockTable[blockBOffset],
                                layoutQTemp,
                                layoutKTemp,
                                layOutS,
                                actualBlockShapeQK,
                                kvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                strideK,
                                kvNIncreIdx);
                            if (kvNIncreIdx == kvNBlockSize - 1) {
                                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                            }
#endif
                        }
#ifdef __DAV_C220_VEC__
                        LayoutP layOutP(rowNum * kvNBlockSize, stackSeqTile, stackSeqTilePad);

                        uint64_t gmOffsetSBase = 
                            static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                        uint64_t gmOffsetPBase = gmOffsetSBase;
                        LayoutS layOutS(rowNum * kvNBlockSize, stackSeqTile, stackSeqTilePad);

                        GemmCoord actualBlockShapeQK{rowNum * kvNBlockSize, stackSeqTile, embed};

                        epilogueOnlineSoftmax(
                            gP[gmOffsetPBase],
                            gS[gmOffsetSBase],
                            layOutP,
                            layOutS,
                            actualBlockShapeQK,
                            (stackSeqCount == 0),
                            0,
                            qSBlockSize,
                            gBlockSize,
                            curStackTileMod,
                            kvNBlockSize,
                            gmOffsetSBase,
                            gmOffsetPBase,
                            qkReady,
                            softmaxReady);
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                    }
                    if (kvSIdx >= preKVNum) {
                        uint32_t nowkvSIdx = kvSIdx - preKVNum;
                        if (nowkvSIdx + blockStackNum > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - nowkvSIdx * pagedBlockSize;
                        } else {
                            stackSeqTile = pagedBlockSize * blockStackNum;
                        }
                        uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1U);

                        for (uint32_t kvNIncreIdx = 0; kvNIncreIdx < kvNBlockSize; kvNIncreIdx++) {
                            uint64_t gmOffsetV = vBOffset + vNStartOffset +
                                static_cast<uint64_t>(kvNIncreIdx * embedV);
                            uint64_t gmOffsetO = oBOffset + oSOffset + oNStartOffset +
                                static_cast<uint64_t>(kvNIncreIdx * groupSize * embed);
                            uint64_t gmOffsetLse = lseBOffset + lseTokenOffset + qNStartIdx +
                                static_cast<uint64_t>(kvNIncreIdx * groupSize);
  
                            uint32_t oWorkspaceIncreOffset = kvNIncreIdx * rowNum * embedRoundV;
                            uint64_t gmOffsetOTmp =
                                static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                                curStackTileMod * WORKSPACE_BLOCK_SIZE_DB + oWorkspaceIncreOffset);
                            GemmCoord actualBlockShapePV{rowNum, embedV, stackSeqTile};
                            LayoutOTmp layoutOTmp(rowNum, embedV, embedRoundV);
#ifdef __DAV_C220_CUBE__

                            uint32_t pWorkspaceIncreOffset = kvNIncreIdx * rowNum * stackSeqTilePad;
                            uint64_t gmOffsetP = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                curStackTileMod * WORKSPACE_BLOCK_SIZE_DB + pWorkspaceIncreOffset;
                            LayoutP layoutPTemp(rowNum, stackSeqTile, stackSeqTilePad);
                            blockMmadPV(
                                gP[gmOffsetP],
                                gV[gmOffsetV],
                                gOTmp[gmOffsetOTmp],
                                gBlockTable[blockBOffset],
                                layoutPTemp,
                                layoutVTemp,
                                layoutOTmp,
                                actualBlockShapePV,
                                nowkvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                noSkipKvS,
                                strideV,
                                blockStackNum,
                                softmaxReady,
                                (kvNIncreIdx == 0));
                            if (kvNIncreIdx == kvNBlockSize - 1) {
                                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);

                            }
#endif
                        }
#ifdef __DAV_C220_VEC__

                        LayoutO layoutO(qSeqlen, embed * qHeads);
                        LayoutOTmp layoutUpdate(rowNum * kvNBlockSize, embed, embedRound);
                        LayoutLse layoutLse(totalQTokens, qHeads);
                        LayoutOTmp layoutOTmp(rowNum * kvNBlockSize, embedV, embedRoundV);
                        GemmCoord actualBlockShapePV{rowNum * kvNBlockSize, embedV, stackSeqTile};

                        Arch::CrossCoreWaitFlag(pvReady);

                        uint64_t gmOffsetO = oBOffset + oSOffset + oNStartOffset;
                        uint64_t gmOffsetUpdate = static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB);
                        uint64_t gmOffsetOTmp = static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                                curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                        uint64_t gmOffsetLse = lseBOffset + lseTokenOffset + qNStartIdx;

                        epilogueRescaleO(
                            // kvNIncreIdx,
                            gO[gmOffsetO],
                            gOTmp[gmOffsetOTmp],
                            gOUpdate[gmOffsetUpdate],
                            gLse[gmOffsetLse],
                            layoutO,
                            layoutOTmp,
                            layoutUpdate,
                            layoutLse,
                            actualBlockShapePV,
                            qSBlockSize,
                            gBlockSize,
                            kvNBlockSize,
                            (stackSeqCount - PRE_LAUNCH == 0),
                            nowkvSIdx + blockStackNum >= kvSLoopNumTotal,
                            curStackTileMod,
                            1U);
#endif
                    }
                    stackSeqCount++;
                }
            }
#ifdef __DAV_C220_CUBE__
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
#ifdef __DAV_C220_VEC__
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
            AscendC::PipeBarrier<PIPE_ALL>();
        }

    private:
        Arch::Resource<ArchTag> resource;
        Arch::CrossCoreFlag qkReady{QK_READY_ID};
        Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
        Arch::CrossCoreFlag pvReady{PV_READY_ID};

        BlockMmadQK blockMmadQK;
        BlockMmadPV blockMmadPV;
        EpilogueOnlineSoftmax epilogueOnlineSoftmax;
        EpilogueRescaleO epilogueRescaleO;
        EpilogueInitOut epilogueInitOut;
    };
}
#endif
