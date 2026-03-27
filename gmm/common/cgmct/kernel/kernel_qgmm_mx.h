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
 * \file kernel_qgmm_mx.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_QGMM_MX_H
#define MATMUL_KERNEL_KERNEL_QGMM_MX_H
#include "kernel_basic_intf.h"
#include "../utils/common_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/fill_utils.h"
#include "../utils/grouped_matmul_constant.h"
#include "../utils/layout_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/tuple_utils.h"

namespace Cgmct {
namespace Gemm {
namespace Kernel {
#define QGMM_MX_KERNEL_CLASS_TEM_PARAMS                                                                                \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QGMM_MX_KERNEL_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::GroupedMatmul;

namespace {
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_X1SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_X2SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_BIAS_OFFSET = 4UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_M_TILEIDX = 0UL;
constexpr uint64_t IDX_N_TILEIDX = 1UL;
constexpr uint64_t IDX_M_TAIL_SPLIT_TILEIDX = 2UL;
constexpr uint64_t IDX_N_TAIL_SPLIT_TILEIDX = 3UL;
} // namespace

QGMM_MX_KERNEL_CLASS_TEM_PARAMS
class KernelQGmmMx {
public:
    __aicore__ inline KernelQGmmMx()
    {
    }
    __aicore__ inline ~KernelQGmmMx()
    {
    }

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;
    // schedulerOp
    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmad::L1TileShape,
                                                                    typename BlockMmad::L0TileShape, BlockScheduler,
                                                                    transA, transB>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutB = typename BlockMmad::LayoutB;
    static constexpr CubeFormat formatB = TagToFormat<LayoutB>::format;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // x, w, x1Scale, x2Scale, bias, y
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using CoordClass = Coordinate<transA, transB, CubeFormat::ND, formatB, CubeFormat::ND>;

    struct GMMTiling {
        uint32_t groupNum;
        uint32_t m;
        uint32_t n;
        uint32_t k;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t kAL1;
        uint32_t kBL1;
        uint32_t scaleKAL1;
        uint32_t scaleKBL1;
        uint8_t isBias;
        uint8_t dbL0C;
        int8_t groupType;
        uint8_t groupListType;
        __aicore__ GMMTiling()
        {
        }
        __aicore__ GMMTiling(uint32_t groupNum_, uint32_t m_, uint32_t n_, uint32_t k_, uint32_t baseM_,
                             uint32_t baseN_, uint32_t baseK_, uint32_t kAL1_, uint32_t kBL1_, uint32_t scaleKAL1_,
                             uint32_t scaleKBL1_, uint8_t isBias_, uint8_t dbL0C_, int8_t groupType_,
                             uint8_t groupListType_)
            : groupNum(groupNum_), m(m_), n(n_), k(k_), baseM(baseM_), baseN(baseN_), baseK(baseK_), kAL1(kAL1_),
              kBL1(kBL1_), scaleKAL1(scaleKAL1_), scaleKBL1(scaleKBL1_), isBias(isBias_), dbL0C(dbL0C_),
              groupType(groupType_), groupListType(groupListType_)
        {
        }
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        GMMTiling gmmParams;
        Params() = default;
    };

public:
    __aicore__ inline void Init(const Params &params);
    __aicore__ inline void Run(const Params &params);
    __aicore__ inline void operator()(const Params &params)
    {
        Run(params);
    }

private:
    __aicore__ inline void SetMNK(uint32_t groupIdx);
    __aicore__ inline void BaseMBalance(BlockSchedulerOp &bs, int64_t m, int64_t baseM);
    __aicore__ inline void ProcessSingleGroup(const Params &params, BlockSchedulerOp &bs, uint32_t groupIdx);
    __aicore__ inline void UpdateOffset(uint32_t groupIdx);
    __aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx);
    __aicore__ inline void UpdateMMGlobalAddr();
    __aicore__ inline void Iterate(int64_t singleCoreM, int64_t singleCoreN);
    __aicore__ inline bool IsLastGroupAndNeedSplit(const BlockSchedulerOp &bs, uint32_t groupIdx);

private:
    BlockMmad mmadOp_;
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};

    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    AscendC::GlobalTensor<AscendC::fp8_e8m0_t> x1ScaleGlobal_;
    AscendC::GlobalTensor<AscendC::fp8_e8m0_t> x2ScaleGlobal_;
    AscendC::GlobalTensor<int64_t> groupListGlobal_;
    AscendC::GlobalTensor<CType> yGlobal_;

    GM_ADDR groupListPtr_;
    GM_ADDR xTensorPtr_;
    GM_ADDR wTensorPtr_;
    GM_ADDR x1ScaleTensorPtr_;
    GM_ADDR x2ScaleTensorPtr_;
    GM_ADDR biasTensorPtr_;
    GM_ADDR yTensorPtr_;

    int32_t preOffset_ = 0;
    uint32_t groupNum_;
    uint32_t curBaseM_;
    int8_t groupType_;
    uint8_t groupListType_;
    bool isBias_{false};
};

QGMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void KernelQGmmMx<QGMM_MX_KERNEL_FUN_TEM_PARAMS>::Run(const Params &params)
{
    Init(params);
    BlockSchedulerOp bs(params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
    if constexpr (formatB == CubeFormat::NZ) {
        if constexpr (transB) {
            bs.SetTailAlign(1, AscendC::BLOCK_CUBE);
        } else {
            bs.SetTailAlign(1, MATMUL_MNK_ALIGN_INT8);
        }
    }
    for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
        UpdateOffset(groupIdx);
        // Update the group-specific M/N/K values.
        SetMNK(groupIdx);
        if (Get<MNK_M>(problemShape_) <= 0 || Get<MNK_K>(problemShape_) <= 0) {
            continue;
        }
        if ASCEND_IS_AIC {
            mmadOp_.UpdateParamsForNextProblem(problemShape_);
        }

        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> bsProblemShape{
            Get<MNK_M>(problemShape_), Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_), 0L};
        BaseMBalance(bs, Get<MNK_M>(problemShape_), params.gmmParams.baseM);
        bs.UpdateNextProblem(bsProblemShape);
        // Further split the tail tiles of the last group to use more cores when possible.
        if (IsLastGroupAndNeedSplit(bs, groupIdx)) {
            bs.UpdateTailTile();
        }
        UpdateMMGlobalAddr();
        ProcessSingleGroup(params, bs, groupIdx);
    }
}

QGMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void KernelQGmmMx<QGMM_MX_KERNEL_FUN_TEM_PARAMS>::Init(const Params &params)
{
    xTensorPtr_ = params.mmadParams.aGmAddr;
    wTensorPtr_ = params.mmadParams.bGmAddr;
    x1ScaleTensorPtr_ = params.mmadParams.x1ScaleGmAddr;
    x2ScaleTensorPtr_ = params.mmadParams.x2ScaleGmAddr;
    biasTensorPtr_ = params.mmadParams.biasGmAddr;
    groupListPtr_ = params.mmadParams.groupListGmAddr;
    yTensorPtr_ = params.mmadParams.cGmAddr;

    groupNum_ = params.gmmParams.groupNum;
    curBaseM_ = params.gmmParams.baseM;
    groupType_ = params.gmmParams.groupType;
    groupListType_ = params.gmmParams.groupListType;
    isBias_ = params.gmmParams.isBias == 1;
    Get<MNK_M>(problemShape_) = params.gmmParams.m;
    Get<MNK_N>(problemShape_) = params.gmmParams.n;
    Get<MNK_K>(problemShape_) = params.gmmParams.k;

    if (groupListPtr_ != nullptr) {
        groupListGlobal_.SetGlobalBuffer((__gm__ int64_t *)groupListPtr_);
    }
    TupleShape l0Shape{static_cast<int64_t>(params.gmmParams.baseM), static_cast<int64_t>(params.gmmParams.baseN),
                       static_cast<int64_t>(params.gmmParams.baseK)};
    L1Params l1Params{static_cast<uint64_t>(params.gmmParams.kAL1), static_cast<uint64_t>(params.gmmParams.kBL1),
                      static_cast<uint64_t>(params.gmmParams.scaleKAL1), 2UL}; // Enable double buffering by default.
    mmadOp_.Init(problemShape_, l0Shape, l1Params, isBias_, params.gmmParams.dbL0C == DOUBLE_BUFFER_COUNT);
}

QGMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void KernelQGmmMx<QGMM_MX_KERNEL_FUN_TEM_PARAMS>::BaseMBalance(BlockSchedulerOp &bs, int64_t m,
                                                                                 int64_t baseM)
{
    if constexpr (!transA) {
        int64_t mCnt = CeilDiv(m, baseM);
        curBaseM_ = CeilAlign(CeilDiv(m, mCnt), AscendC::BLOCK_CUBE);
        bs.UpdateBaseM(curBaseM_);
    }
}

QGMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline bool KernelQGmmMx<QGMM_MX_KERNEL_FUN_TEM_PARAMS>::IsLastGroupAndNeedSplit(const BlockSchedulerOp &bs,
                                                                                            uint32_t groupIdx)
{
    // Consider tail split only when at least half of the cores are still available.
    return groupIdx == groupNum_ - 1 && (bs.GetEndBlockIdx() + 1) <= AscendC::GetBlockNum() / 2;
}

QGMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void KernelQGmmMx<QGMM_MX_KERNEL_FUN_TEM_PARAMS>::UpdateOffset(uint32_t groupIdx)
{
    // baseOffset_ remains 0 for the first group.
    if (groupIdx == 0) {
        return;
    }
    int64_t m = Get<MNK_M>(problemShape_);
    int64_t n = Get<MNK_N>(problemShape_);
    int64_t k = Get<MNK_K>(problemShape_);
    // aBaseOffset += m * k
    Get<IDX_A_OFFSET>(baseOffset_) += m * k;
    if constexpr (formatB == CubeFormat::ND) {
        // bBaseOffset += n * k
        Get<IDX_B_OFFSET>(baseOffset_) += n * k;
    } else {
        if constexpr (transB) {
            int64_t nAlign = (n + AscendC::BLOCK_CUBE - 1) & (~(AscendC::BLOCK_CUBE - 1));
            int64_t kAlign = (k + MATMUL_MNK_ALIGN_INT8 - 1) & (~(MATMUL_MNK_ALIGN_INT8 - 1));
            Get<IDX_B_OFFSET>(baseOffset_) += nAlign * kAlign;
        } else {
            int64_t nAlign = (n + MATMUL_MNK_ALIGN_INT8 - 1) & (~(MATMUL_MNK_ALIGN_INT8 - 1));
            int64_t kAlign = (k + AscendC::BLOCK_CUBE - 1) & (~(AscendC::BLOCK_CUBE - 1));
            Get<IDX_B_OFFSET>(baseOffset_) += nAlign * kAlign;
        }
    }

    if constexpr (transA) { // split k, x1Scale:(k/64+g, m, 2) x2Scale:(k/64+g, n, 2)
        int64_t scaleK = (Get<IDX_B_OFFSET>(baseOffset_) / n / MXFP_DIVISOR_SIZE + groupIdx) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_X1SCALE_OFFSET>(baseOffset_) = m * scaleK;
        Get<IDX_X2SCALE_OFFSET>(baseOffset_) = n * scaleK;
    } else { // split m, x1Scale:(m, ceil(k/64), 2) x2Scale:(g, n, ceil(k/64), 2) or (g, ceil(k/64), n, 2)
        int64_t scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_X1SCALE_OFFSET>(baseOffset_) += m * scaleK;
        Get<IDX_X2SCALE_OFFSET>(baseOffset_) += n * scaleK;
    }
    // yBaseOffset += m * n
    Get<IDX_C_OFFSET>(baseOffset_) += m * n;
    Get<IDX_BIAS_OFFSET>(baseOffset_) += n;
}

QGMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void KernelQGmmMx<QGMM_MX_KERNEL_FUN_TEM_PARAMS>::ProcessSingleGroup(const Params &params,
                                                                                       BlockSchedulerOp &bs,
                                                                                       uint32_t groupIdx)
{
    CoordClass coord(Get<MNK_M>(problemShape_), Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_),
                     static_cast<int64_t>(curBaseM_), params.gmmParams.baseN, params.gmmParams.baseK);
    BlockCoord tileIdx;
    while (bs.GetTileIdx(tileIdx)) {
        BlockShape singleShape = bs.GetBlockShape(tileIdx);
        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            return;
        }
        blockOffset_ = coord.template GetQuantOffset<QuantMode::MX_PERGROUP_MODE>(
            Get<IDX_M_TILEIDX>(tileIdx), Get<IDX_N_TILEIDX>(tileIdx), Get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape),
            Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape));
        Iterate(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape));
    }
}

QGMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void KernelQGmmMx<QGMM_MX_KERNEL_FUN_TEM_PARAMS>::Iterate(int64_t singleCoreM, int64_t singleCoreN)
{
    AscendC::Std::tuple<int64_t, int64_t, int64_t> blockShape{singleCoreM, singleCoreN,
                                                              static_cast<int64_t>(Get<MNK_K>(problemShape_))};
    if (isBias_) {
        mmadOp_(aGlobal_[Get<IDX_A_OFFSET>(blockOffset_)], bGlobal_[Get<IDX_B_OFFSET>(blockOffset_)],
                x1ScaleGlobal_[Get<IDX_X1SCALE_OFFSET>(blockOffset_)],
                x2ScaleGlobal_[Get<IDX_X2SCALE_OFFSET>(blockOffset_)], biasGlobal_[Get<IDX_BIAS_OFFSET>(blockOffset_)],
                yGlobal_[Get<IDX_C_OFFSET>(blockOffset_)], blockShape);
    } else {
        mmadOp_(aGlobal_[Get<IDX_A_OFFSET>(blockOffset_)], bGlobal_[Get<IDX_B_OFFSET>(blockOffset_)],
                x1ScaleGlobal_[Get<IDX_X1SCALE_OFFSET>(blockOffset_)],
                x2ScaleGlobal_[Get<IDX_X2SCALE_OFFSET>(blockOffset_)], yGlobal_[Get<IDX_C_OFFSET>(blockOffset_)],
                blockShape);
    }
}

QGMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void KernelQGmmMx<QGMM_MX_KERNEL_FUN_TEM_PARAMS>::SetMNK(uint32_t groupIdx)
{
    int32_t splitValue = GetSplitValueFromGroupList(groupIdx);
    if (groupType_ == GMM_SPLIT_M) {
        Get<MNK_M>(problemShape_) = splitValue;
    } else {
        Get<MNK_K>(problemShape_) = splitValue;
    }
}

QGMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void KernelQGmmMx<QGMM_MX_KERNEL_FUN_TEM_PARAMS>::UpdateMMGlobalAddr()
{
    // Update global tensor addresses for the current grouped matmul instance.
    aGlobal_.SetGlobalBuffer(GetTensorAddr<AType>(0, xTensorPtr_) + Get<IDX_A_OFFSET>(baseOffset_));
    bGlobal_.SetGlobalBuffer(GetTensorAddr<BType>(0, wTensorPtr_) + Get<IDX_B_OFFSET>(baseOffset_));
    if (isBias_) {
        biasGlobal_.SetGlobalBuffer(GetTensorAddr<BiasType>(0, biasTensorPtr_) + Get<IDX_BIAS_OFFSET>(baseOffset_));
    }
    x1ScaleGlobal_.SetGlobalBuffer((__gm__ AscendC::fp8_e8m0_t *)(x1ScaleTensorPtr_) +
                                   Get<IDX_X1SCALE_OFFSET>(baseOffset_)); // optional input
    x2ScaleGlobal_.SetGlobalBuffer(GetTensorAddr<AscendC::fp8_e8m0_t>(0, x2ScaleTensorPtr_) +
                                   Get<IDX_X2SCALE_OFFSET>(baseOffset_));
    yGlobal_.SetGlobalBuffer(GetTensorAddr<CType>(0, yTensorPtr_) + Get<IDX_C_OFFSET>(baseOffset_));
}

QGMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline int32_t KernelQGmmMx<QGMM_MX_KERNEL_FUN_TEM_PARAMS>::GetSplitValueFromGroupList(uint32_t groupIdx)
{
    int32_t splitValue = 0;
    if (groupListType_ == 0) {
        int32_t offset = static_cast<int32_t>(groupListGlobal_.GetValue(groupIdx));
        splitValue = offset - preOffset_;
        preOffset_ = offset;
    } else {
        splitValue = static_cast<int32_t>(groupListGlobal_.GetValue(groupIdx));
    }
    return splitValue;
}

} // namespace Kernel
} // namespace Gemm
} // namespace Cgmct

#endif
