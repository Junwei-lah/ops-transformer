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
 * \file allto_all_matmul_arch35.h
 * \brief
 */

#ifndef ALLTO_ALL_MATMUL_ARCH35_H
#define ALLTO_ALL_MATMUL_ARCH35_H

#include "allto_all_matmul_tiling_data.h"

namespace AlltoAllMatmulImpl
{
using namespace AscendC;
template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
class AlltoAllMatmulArch35
{
public:
    __aicore__ inline AlltoAllMatmulArch35(SchedulerType* pipeLine) : pipeLine_(pipeLine){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out,
                                GM_ADDR workspaceGM, AlltoAllMatmulTilingDataType* tilingData,
                                TPipe* tPipe);
    __aicore__ inline void Process();

private:
    SchedulerType* pipeLine_;
    SchedulerContextType pipeLineContext_;
    AlltoAllMatmulTilingDataType* tilingData_;
    TPipe* tPipe_;
    GM_ADDR x1_;
    GM_ADDR x2_;
    GM_ADDR y_;
    GM_ADDR bias_;
    GM_ADDR workspaceGM_;
    GM_ADDR commOutGM_;
    GM_ADDR transOutGM_;

private:
    __aicore__ inline void ProcessTile(uint32_t taskCnt);
    __aicore__ inline void ProcessTail(uint32_t taskCnt);
    // 执行流水线
    __aicore__ inline void ProcessPipeLine(uint32_t taskCnt);
};


template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void AlltoAllMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y,
    GM_ADDR all2all_out, GM_ADDR workspaceGM, AlltoAllMatmulTilingDataType* tilingData,TPipe* tPipe)
{   
    // 获取tilingdata数据
    tilingData_ = tilingData;
    auto&& mc2Tiling_ = tilingData_->alltoAllMatmulTilingInfo;
    // 管道初始化
    tPipe_ = tPipe;
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    workspaceGM_ = workspaceGM;
    commOutGM_ = workspaceGM;
    transOutGM_ = all2all_out;
    if (all2all_out == nullptr) {
        transOutGM_ = (GM_ADDR)((uint64_t)commOutGM_ + mc2Tiling_.commLen);
    }
    // 初始化流水线
    pipeLine_->Init();
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void AlltoAllMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::Process()
{
    auto&& mc2Tiling_ = tilingData_->alltoAllMatmulTilingInfo;
    // 启动主块流水
    if (mc2Tiling_.tileCnt>0) {
        ProcessTile(mc2Tiling_.tileCnt);
    }

    // 启动尾块流水
    if (mc2Tiling_.tailCnt>0) {
        ProcessTail(mc2Tiling_.tailCnt);
    }

    // 结束流水线
    pipeLine_->End();
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void AlltoAllMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::ProcessTile(uint32_t taskCnt)
{
    auto&& mc2Tiling_ = tilingData_->alltoAllMatmulTilingInfo;
    // 复用的变量
    uint64_t tileMMultiRankK = (uint64_t)mc2Tiling_.tileM * (uint64_t)mc2Tiling_.rankK;

    // matmul计算相关地址和偏移
    pipeLineContext_.aGM = transOutGM_;
    pipeLineContext_.bGM = x2_;
    pipeLineContext_.cGM = y_;
    pipeLineContext_.biasGM = bias_;
    pipeLineContext_.extraData.a_offset = tileMMultiRankK * (uint64_t)mc2Tiling_.rankDim * (uint64_t)sizeof(DTYPE_X1);
    pipeLineContext_.extraData.b_offset = (uint64_t)0UL;
    pipeLineContext_.extraData.c_offset = (uint64_t)mc2Tiling_.tileM * (uint64_t)mc2Tiling_.rankN * (uint64_t)sizeof(DTYPE_Y);
    pipeLineContext_.tilingData = &(tilingData_->mc2MmV3TileTilingData);

    // 转置相关地址和偏移
    pipeLineContext_.transposeSrcAddr = commOutGM_;
    pipeLineContext_.transposeDstAddr = transOutGM_;
    pipeLineContext_.transposeSrcOffset = tileMMultiRankK * (uint64_t)sizeof(DTYPE_X1);
    pipeLineContext_.transposeDstOffset = tileMMultiRankK * (uint64_t)mc2Tiling_.rankDim * (uint64_t)sizeof(DTYPE_X1);
    pipeLineContext_.nextSrcBlockOffset = (uint64_t)mc2Tiling_.rankM * (uint64_t)mc2Tiling_.rankK / (uint64_t)mc2Tiling_.rankDim;
    pipeLineContext_.nextDstBlockOffset =  (uint64_t)mc2Tiling_.rankK;
    pipeLineContext_.rankCnt = (uint64_t)mc2Tiling_.rankDim;
    pipeLineContext_.innerAxis = (uint64_t)mc2Tiling_.rankK;
    pipeLineContext_.transM = (uint64_t)mc2Tiling_.tileM;

    // 通信相关地址和偏移
    pipeLineContext_.taskCnt = mc2Tiling_.tileCnt;
    pipeLineContext_.sendBuffer = x1_;
    pipeLineContext_.recvBuffer = commOutGM_;
    pipeLineContext_.sendOffset = tileMMultiRankK * (uint64_t)sizeof(DTYPE_X1);
    pipeLineContext_.recvOffset = pipeLineContext_.sendOffset;
    pipeLineContext_.sendCount = tileMMultiRankK;
    pipeLineContext_.strideCount = pipeLineContext_.nextSrcBlockOffset;
    pipeLineContext_.hcclDataType = mc2Tiling_.hcclDataType;

    ProcessPipeLine(taskCnt);
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void AlltoAllMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::ProcessTail(uint32_t taskCnt)
{
    auto&& mc2Tiling_ = tilingData_->alltoAllMatmulTilingInfo;
    // 复用的变量
    uint64_t tailMMultiRankK = (uint64_t)mc2Tiling_.tailM * (uint64_t)mc2Tiling_.rankK;
    uint64_t tileCntMultitileMMultiRankK = (uint64_t)mc2Tiling_.tileCnt * (uint64_t)mc2Tiling_.tileM * (uint64_t)mc2Tiling_.rankK;

    pipeLineContext_.aGM = transOutGM_ + tileCntMultitileMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.bGM = x2_;
    pipeLineContext_.cGM = y_ + (uint64_t)mc2Tiling_.tileCnt * mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.extraData.a_offset = tailMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.extraData.b_offset = (uint64_t)0UL;
    pipeLineContext_.extraData.c_offset = (uint64_t)mc2Tiling_.tailM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.tilingData = &(tilingData_->mc2MmV3TailTilingData);

    pipeLineContext_.transposeSrcAddr = commOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.transposeDstAddr = transOutGM_ + tileCntMultitileMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.transposeSrcOffset = tailMMultiRankK * (uint64_t)sizeof(DTYPE_X1);
    pipeLineContext_.transposeDstOffset = tailMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.transM = (uint64_t)mc2Tiling_.tailM;

    pipeLineContext_.taskCnt = mc2Tiling_.tailCnt;
    pipeLineContext_.sendBuffer = x1_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.recvBuffer = commOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.sendOffset = tailMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.recvOffset = pipeLineContext_.sendOffset;
    pipeLineContext_.sendCount = tailMMultiRankK;

    ProcessPipeLine(taskCnt);
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void AlltoAllMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::ProcessPipeLine(uint32_t taskCnt)
{
    pipeLine_->ChangeSpecification(&pipeLineContext_);
    pipeLine_->Process(taskCnt);
}
};
#endif