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
 * \file allto_all_matmul.h
 * \brief
 */

#ifndef ALL_TO_ALL_MATMUL_A4W4_H
#define ALL_TO_ALL_MATMUL_A4W4_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "allto_all_matmul_tiling.h"
#include "allto_all_matmul_util.h"
#include "block_epilogue_dequant.hpp"
#include "tile_broadcast_add.hpp"

#include "../../3rd/template_linear_algebra/include/template_linear_algebra/catlass.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/arch/arch.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/arch/resource.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/layout/layout.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_swizzle.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/gemm/dispatch_policy.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/gemm/gemm_type.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/gemm_coord.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/copy_gm_to_ub.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/epilogue/dispatch_policy.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_broadcast_mul.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_swizzle.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_copy.hpp"

#include "matmul.hpp"

using namespace AscendC;
using namespace Catlass;

namespace AlltoAllMatmulImpl {
// A2AMM : AlltoAllMatmulA4W4
#define TemplateA2AMMClass typename AType, typename BType, typename BiasType, typename PerTokenScaleType, typename ScaleType, typename CType, typename AllToAllResultType, bool hasBias, bool transB
#define TemplateA2AMMFunc AType, BType, BiasType, PerTokenScaleType, ScaleType, CType, AllToAllResultType, hasBias, transB

using namespace AscendC;
template <TemplateA2AMMClass>
class AlltoAllMatmulA4W4 : public CommBase{
public:
    __aicore__ inline AlltoAllMatmulA4W4() {};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                GM_ADDR perTokenScaleGM, GM_ADDR scaleGM,
                                GM_ADDR cGM, GM_ADDR allToAllResult,
                                GM_ADDR workspaceGM, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void AIVInit();
    __aicore__ inline void AICInit();
    __aicore__ inline void CatlassMatmul();
    __aicore__ inline void AlltoAll();
    __aicore__ inline void Dequant();

private:
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR biasGM_;
    GM_ADDR scaleGM_;
    GM_ADDR perTokenScaleGM_;
    GM_ADDR allToAllResultGM_;
    GM_ADDR workspaceGM_;

    __gm__ AType* gmPeerMem_;
    __gm__ int32_t* dequantCGM_;

    Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
};


template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmulA4W4<TemplateA2AMMFunc>::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                                               GM_ADDR perTokenScaleGM, GM_ADDR scaleGM,
                                                               GM_ADDR cGM, GM_ADDR allToAllResultGM,
                                                               GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AlltoAllMatmulTilingData);
    auto tiling = (__gm__ AlltoAllMatmulTilingData*)tilingGM;
    GET_TILING_DATA(tilingData, tilingGM);

    auto contextGM = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclCombineOpParam *)contextGM;
    rank = winContext_ -> rankId;
    rankSize = tilingData.allToAllMatmulInfo.rankSize;

    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    scaleGM_ = scaleGM;
    perTokenScaleGM_ = perTokenScaleGM;
    allToAllResultGM_ = allToAllResultGM;
    workspaceGM_ = GetUserWorkspace(workspaceGM);

    CommBase::SetArgs<AType>(rank, rankSize, tilingData);
    perTokenScaleGM_ += rank * (m / rankSize) * sizeof(PerTokenScaleType);
    this->ub_offset = Catlass::BytesToBits(UB_OFFSET) / Catlass::SizeOfBits<int8_t>::value;

    dequantCGM_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_);
    kBytes = Catlass::BitsToBytes(k * Catlass::SizeOfBits<AType>::value);
    tokenBytes = Catlass::BitsToBytes(tokenSize * Catlass::SizeOfBits<AType>::value);

    AlltoAllMatmulA4W4<TemplateA2AMMFunc>::AICInit();
    AlltoAllMatmulA4W4<TemplateA2AMMFunc>::AIVInit();
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmulA4W4<TemplateA2AMMFunc>::AICInit()
{
    if ASCEND_IS_AIC {
        SetLoadDataPaddingValue(0);
        SetAtomicNone();
        SetFixpipeNz2ndFlag(1, 0, 0);
        gmPeerMem_ = reinterpret_cast<__gm__ AType*>(buff[rank]);
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmulA4W4<TemplateA2AMMFunc>::AIVInit()
{
    if ASCEND_IS_AIV {
        SetAtomicNone();
        SetMaskNorm();
        SetVectorMask<int32_t>((uint64_t)-1, (uint64_t)-1);
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmulA4W4<TemplateA2AMMFunc>::CatlassMatmul()
{
    if ASCEND_IS_AIC {
        using ArchTag = Arch::AtlasA2;

        constexpr bool ENABLE_UNIT_FLAG = false;
        constexpr bool ENABLE_SHUFFLE_K = false;
        constexpr bool aicCalBias = false;  // 计算量化后的矩阵乘，bias不由CatlassMatmul负责

        using ElementA = AscendC::int4b_t;
        using ElementB = AscendC::int4b_t;
        using ElementC = int32_t;
        using ElementBias = BiasType;
        using LayoutA = layout::RowMajor;
        // B转置
        using LayoutB = std::conditional_t<transB, layout::ColumnMajor, layout::RowMajor>;
        using LayoutC = layout::RowMajor;
        using LayoutBias = layout::VectorLayout;

        uint32_t realM = m / rankSize;
        uint32_t realK = k * rankSize;
        LayoutA layoutA{static_cast<uint32_t>(realM), static_cast<uint32_t>(realK)};
        LayoutB layoutB{static_cast<uint32_t>(realK), static_cast<uint32_t>(n)};
        LayoutC layoutC{static_cast<uint32_t>(realM), static_cast<uint32_t>(n)};
        LayoutBias layoutBias{static_cast<uint32_t>(n)};

        using DispatchPolicy = Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>;

        using AType_ = Gemm::GemmType<ElementA, LayoutA>;
        using BType_ = Gemm::GemmType<ElementB, LayoutB>;
        using CType_ = Gemm::GemmType<ElementC, LayoutC>;
        using BiasType_ = std::conditional_t<aicCalBias, Gemm::GemmType<ElementBias, LayoutBias>, void>;

        struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_> {
            using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_>;
            using ElementA = typename Base::ElementA;
            using ElementB = typename Base::ElementB;
            using ElementAccumulator = typename Base::ElementAccumulator;

            using CopyGmToL1A = typename Base::CopyGmToL1A;
            using CopyGmToL1B = typename Base::CopyGmToL1B;

            using CopyL1ToL0A = typename Base::CopyL1ToL0A;
            using CopyL1ToL0B = typename Base::CopyL1ToL0B;

            using CopyL0CToGm = typename Base::CopyL0CToGm;
            using BiasTypeSelector = typename Base::BiasTypeSelector;
            using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
            using CopyL1ToBT = typename Base::CopyL1ToBT;
        };

        using TileCopy = TileCopyOpt;
        GemmCoord processSize{static_cast<uint32_t>(realM), static_cast<uint32_t>(n), static_cast<uint32_t>(realK)};
        using BlockScheduler30 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        GM_ADDR matmulResultGM = reinterpret_cast<GM_ADDR>(dequantCGM_);  // 量化矩阵乘法时，需要修改c矩阵存放地址
        if (m0 == 128) {
            using L1TileShape = GemmShape<128, 256, 1024>;
            using L0TileShape = GemmShape<128, 256, 256>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_, CType_, BiasType_, TileCopy>;
            using MatmulKernel = Gemm::Kernel::AlltoAllMatmulKernel<void, void, BlockMmadOpt, void, BlockScheduler30, aicCalBias>;
            MatmulKernel matmul_op;
            typename MatmulKernel::Params params{processSize,
                                    reinterpret_cast<GM_ADDR>(gmPeerMem_), layoutA,
                                    reinterpret_cast<GM_ADDR>(bGM_), layoutB,
                                    reinterpret_cast<GM_ADDR>(biasGM_),
                                    reinterpret_cast<GM_ADDR>(matmulResultGM), layoutC,
                                    pValue, 3, 0, static_cast<int32_t>(rankSize), MAX_BLOCK_COUNT};
            matmul_op(params);
        } else {
            using L1TileShape = GemmShape<256, 128, 1024>;
            using L0TileShape = GemmShape<256, 128, 256>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_, CType_, BiasType_, TileCopy>;
            using MatmulKernel = Gemm::Kernel::AlltoAllMatmulKernel<void, void, BlockMmadOpt, void, BlockScheduler30, aicCalBias>;
            MatmulKernel matmul_op;
            typename MatmulKernel::Params params{processSize,
                                    reinterpret_cast<GM_ADDR>(gmPeerMem_), layoutA,
                                    reinterpret_cast<GM_ADDR>(bGM_), layoutB,
                                    reinterpret_cast<GM_ADDR>(biasGM_),
                                    reinterpret_cast<GM_ADDR>(matmulResultGM), layoutC,
                                    pValue, 3, 0, static_cast<int32_t>(rankSize), MAX_BLOCK_COUNT};
            matmul_op(params);
        }
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmulA4W4<TemplateA2AMMFunc>::Dequant()
{
    using ArchTag = Arch::AtlasA2;

    using ElementC = int32_t;
    using ElementBias = BiasType;
    using LayoutD = layout::RowMajor;

    uint32_t realM = m / rankSize;
    uint32_t realK = k * rankSize;
    LayoutD layoutD{static_cast<uint32_t>(realM), static_cast<uint32_t>(n)};

    using CType_ = Gemm::GemmType<ElementC, layout::RowMajor>;

    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using ScaleGType = Gemm::GemmType<ScaleType, layout::VectorLayout>;
    using PerTokenScaleGType = Gemm::GemmType<PerTokenScaleType, layout::VectorLayout>;
    using BiasGType = Gemm::GemmType<BiasType, layout::VectorLayout>;
    using DType = Gemm::GemmType<CType, layout::RowMajor>;
    layout::VectorLayout layoutScale{static_cast<uint32_t>(n)};
    layout::VectorLayout layoutPerTokenScale{static_cast<uint32_t>(realM)};
    layout::VectorLayout layoutBias{static_cast<uint32_t>(n)};
    using LayoutScale = layout::VectorLayout;
    using LayoutPerTokenScale = layout::VectorLayout;
    using ElementD = CType;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using RowBroadcastAddType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    struct TileCopyDequant : public Catlass::Epilogue::Tile::TileCopy<ArchTag, CType_, ScaleGType, PerTokenScaleGType, DType> {
        using Base = Catlass::Epilogue::Tile::TileCopy<ArchTag, CType_, ScaleGType, PerTokenScaleGType, DType>;
        using ElementC = typename Base::ElementC;
        using ElementScale = typename Base::ElementX;
        using ElementPerTokenScale = typename Base::ElementY;
        using ElementBias = typename BiasGType::Element;
        using ElementD = typename Base::ElementD;

        using CopyGmToUbC = typename Base::CopyGmToUbC;
        using CopyGmToUbScale = typename Base::CopyGmToUbX;
        using CopyGmToUbPerTokenScale = typename Base::CopyGmToUbY;
        using CopyGmToUbBias = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, BiasGType>;
        using CopyUbToGmD = typename Base::CopyUbToGmD;
    };

    using EpilogueTileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;
    GemmCoord problemShape{static_cast<uint32_t>(realM), static_cast<uint32_t>(n), static_cast<uint32_t>(realK)};

    AscendC::GlobalTensor<ElementD> gmD;
    gmD.SetGlobalBuffer((__gm__ ElementD *)cGM_);
    AscendC::GlobalTensor<ElementC> gmC;
    gmC.SetGlobalBuffer((__gm__ ElementC *)dequantCGM_);

    uint32_t rowsPerCore = DivCeil(problemShape.m(), blockNum);
    uint32_t rowsThisCore = rowsPerCore;
    uint32_t stRowPerCore = aicIdx * rowsPerCore;
    if (stRowPerCore < problemShape.m()) {
        if (rowsThisCore + stRowPerCore > problemShape.m()) {
            rowsThisCore = problemShape.m() - stRowPerCore;
        }
    } else {
        rowsThisCore = 0;
    }
    MatrixCoord coreOffset(stRowPerCore, 0u);
    auto layoutC = layout::RowMajor{problemShape.m(), n};
    int64_t gmOffsetC = layoutC.GetOffset(coreOffset);
    GemmCoord actualBlockShape{rowsThisCore, n, 1};

    if (m0 == 128) {
        using EpilogueTileShape = MatrixShape<32, 256>;
        using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
        using TileRowBroadcastAdd = Epilogue::Tile::TileRowBroadcastAdd<ArchTag, RowBroadcastAddType, EpilogueTileShape>;
        using TileBroadcastOneBlk = Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType,
            EpilogueTileShape::ROW>;
        using TileOneBlkColumnBroadcastMul = Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag,
            OneBlkColumnBroadcastMulType, EpilogueTileShape>;
        using QuantBlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType_, ScaleGType, PerTokenScaleGType, BiasGType, DType,
            TileRowBroadcastMul, TileRowBroadcastAdd, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, TileCopyDequant, EpilogueTileScheduler>;
        QuantBlockEpilogue blockEpilogue(resource);

        using EpilogueParams = typename QuantBlockEpilogue::Params;
        EpilogueParams epilogueParams {
            scaleGM_, layoutScale,
            perTokenScaleGM_, layoutPerTokenScale.GetTileLayout(problemShape.template GetCoordByAxis<0>()),
            biasGM_, layoutBias
        };
        blockEpilogue.UpdateParams(epilogueParams);

        blockEpilogue(coreOffset, actualBlockShape, gmC[gmOffsetC], layoutC, gmD[gmOffsetC], layoutC);
    } else {
        using EpilogueTileShape = MatrixShape<64, 128>;
        using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;

        using TileRowBroadcastAdd = Epilogue::Tile::TileRowBroadcastAdd<ArchTag, RowBroadcastAddType, EpilogueTileShape>;

        using TileBroadcastOneBlk = Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType,
            EpilogueTileShape::ROW>;
        using TileOneBlkColumnBroadcastMul = Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag,
            OneBlkColumnBroadcastMulType, EpilogueTileShape>;
        using QuantBlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType_, ScaleGType, PerTokenScaleGType, BiasGType, DType,
            TileRowBroadcastMul, TileRowBroadcastAdd, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, TileCopyDequant, EpilogueTileScheduler>;
        QuantBlockEpilogue blockEpilogue(resource);

        using EpilogueParams = typename QuantBlockEpilogue::Params;
        EpilogueParams epilogueParams {
            scaleGM_, layoutScale,
            perTokenScaleGM_, layoutPerTokenScale.GetTileLayout(problemShape.template GetCoordByAxis<0>()),
            biasGM_, layoutBias
        };
        blockEpilogue.UpdateParams(epilogueParams);

        blockEpilogue(coreOffset, actualBlockShape, gmC[gmOffsetC], layoutC, gmD[gmOffsetC], layoutC);
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmulA4W4<TemplateA2AMMFunc>::AlltoAll()
{
    if ASCEND_IS_AIV {
        ResetIpcFlags(BUFFER_NUM);
        PipeBarrier<PIPE_ALL>();

        int64_t src_offset = 0;
        for (int32_t commIdx = 0; commIdx <= commCount; ++commIdx) {
            uint64_t flagIdx = commIdx % MAX_BLOCK_COUNT;

            if (commIdx == commCount - 1) {
                allToAllSizePerRankPerLoop = allToAllSizePerRank - src_offset;
            }

            if (commIdx >= MAX_BLOCK_COUNT && commIdx < commCount) {
                WaitEvent(flagIdx);
            }

            SetAndWaitAivSync(flagIdx);
            if (commIdx < commCount) {
                CrossRankSyncV1(FLAG_ZERO_IDX, commIdx + 1);
            }
            SetAndWaitAivSync(flagIdx);

            if (aivIdx == 0 && commIdx < commCount && aicIdx < allToAllSendCoreNum) {
                int32_t dstRank = aicIdx / coreNumPerRank;
                int32_t dstLoc = aicIdx % coreNumPerRank;
                int32_t coreOffset = dstLoc * allToAllSizePerCore;
                int32_t dataLen = coreOffset + allToAllSizePerCore > allToAllSizePerRankPerLoop ?
                        allToAllSizePerRankPerLoop - coreOffset : allToAllSizePerCore;
                int64_t dataSrc = dstRank * allToAllSizePerRank + src_offset + coreOffset;
                int64_t dataDst = flagIdx * pingPongBlockSize + coreOffset * rankSize + rank * k;
                uint32_t copyBytes = Catlass::BitsToBytes(dataLen * Catlass::SizeOfBits<AType>::value);

                if (dataLen > 0) {
                    CopyTokensFromGMToGM(reinterpret_cast<__gm__ int8_t*>(aGM_) + dataSrc / 2, (__gm__ int8_t *)buff[dstRank] + dataDst / 2, copyBytes, kBytes, tokenBytes);
                }
                src_offset += allToAllSizePerRankPerLoop;
            }
            else if (aivIdx == 1 && commIdx > 0 && aicIdx >= allToAllSendCoreNum && aicIdx < usedCoreNum) {
                int32_t blockDst = ((commIdx - 1) % MAX_BLOCK_COUNT) * pingPongBlockSize;
                int32_t mThisLoop = commIdx == commCount ? m / rankSize - (commIdx - 1) * mPerLoop : mPerLoop;
                int32_t mThisLoopPerCore = DivCeil(mThisLoop, allToAllRecvCoreNum);
                int32_t mSt = (aicIdx - allToAllSendCoreNum) * mThisLoopPerCore;
                int32_t mThisCoreThisLoop = mSt + mThisLoopPerCore > mThisLoop ? mThisLoop - mSt : mThisLoopPerCore;
                int64_t srcSt = blockDst + mSt * tokenSize;
                int64_t dstSt = ((commIdx - 1) * mPerLoop + mSt) * tokenSize;
                if (mThisCoreThisLoop > 0) {
                    CopyTokensFromGMToGM((__gm__ int8_t *)buff[rank] + srcSt / 2, reinterpret_cast<__gm__ int8_t*>(allToAllResultGM_) + dstSt / 2, mThisCoreThisLoop * tokenBytes, tokenBytes, tokenBytes);
                }
            }

            SetAndWaitAivSync(flagIdx);
            if (commIdx < commCount) {
                CrossRankSyncV1(FLAG_ONE_IDX, commIdx + 1);
            }
            SetAndWaitAivSync(flagIdx);

            if (commIdx < commCount) {
                SetAicSync(flagIdx);
            }
        }

        WaitEvent(FLAG_ZERO_IDX);
        if (commCount % 2 == 0) {  // 若AIC计算次数为偶数，则多等一次
            WaitEvent(FLAG_ONE_IDX);
        }

        if constexpr (AscendC::IsSameType<BType, AscendC::int4b_t>::value) {
            SetAndWaitAivSync(FLAG_ONE_IDX);
            Dequant();
        }

        PipeBarrier<PIPE_ALL>();
        ResetIpcFlags(1);
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmulA4W4<TemplateA2AMMFunc>::Process()
{
    AlltoAll();
    CatlassMatmul();
    SyncAll<false>();
}

} // AlltoAllMatmulImpl
#endif // ALL_TO_ALL_MATMUL_A4W4_H