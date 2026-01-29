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
 * \file matmul_allto_all_arch35.h
 * \brief
 */

#ifndef MATMUL_ALLTO_ALL_ARCH35_H
#define MATMUL_ALLTO_ALL_ARCH35_H

#include "matmul_allto_all_tiling_data.h"

namespace MatmulAlltoAllImpl
{
using namespace AscendC;
/**
 * SchedulerType: 流水线类的数据类型
 * SchedulerContextType: 流水线类使用的上下文的数据类型
 * MatmulAlltoAllTilingDataType: tilingdata的数据类型
 */
template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType>
class MatmulAlltoAllArch35
{
public:
    __aicore__ inline MatmulAlltoAllArch35(SchedulerType* pipeLine) : pipeLine_(pipeLine){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y,
                                GM_ADDR workspaceGM, MatmulAlltoAllTilingDataType* tilingData,
                                TPipe* tPipe);
    __aicore__ inline void Process();

private:
    SchedulerType* pipeLine_;
    SchedulerContextType pipeLineContext_;
    MatmulAlltoAllTilingDataType* tilingData_;
    // 数据控制
    TPipe* tPipe_;
    // 输入输出矩阵地址
    GM_ADDR x1_;
    GM_ADDR x2_;
    GM_ADDR y_;
    GM_ADDR bias_;
    //临时空间的地址
    GM_ADDR workspaceGM_;
    GM_ADDR tempComputeOutGM_;
    GM_ADDR transOutGM_;

private:
    __aicore__ inline void ProcessTile(uint32_t taskCnt);
    __aicore__ inline void ProcessTail(uint32_t taskCnt);
    // 执行流水线
    __aicore__ inline void ProcessPipeLine(uint32_t taskCnt);
};

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType>
__aicore__ inline void MatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspaceGM, MatmulAlltoAllTilingDataType* tilingData,TPipe* tPipe)
{   
    // 获取tilingdata数据
    tilingData_ = tilingData;
    auto&& mc2Tiling_ = tilingData_->matmulAlltoAllTilingInfo;
    // 数据初始化
    tPipe_ = tPipe;
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    workspaceGM_ = workspaceGM;
    tempComputeOutGM_ = workspaceGM;
    transOutGM_ = (GM_ADDR)(tempComputeOutGM_ + mc2Tiling_.mmResultLen);
    // 初始化流水线
    pipeLine_->Init();
}

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType>
__aicore__ inline void MatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType>::Process()
{
    auto&& mc2Tiling_ = tilingData_->matmulAlltoAllTilingInfo;
    // 启动主块流水
    if(mc2Tiling_.tileCnt>0) {
        ProcessTile(mc2Tiling_.tileCnt);
    }

    // 启动尾块流水
    if(mc2Tiling_.tailCnt>0) {
        ProcessTail(mc2Tiling_.tailCnt);
    }

    // 结束流水线
    pipeLine_->End();
}

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType>
__aicore__ inline void MatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType>::ProcessTile(uint32_t taskCnt)
{
    auto&& mc2Tiling_ = tilingData_->matmulAlltoAllTilingInfo;
    //matmul矩阵乘计算的输入输出地址，到下一轮计算的数据地址的偏移
    pipeLineContext_.aGM = x1_;
    pipeLineContext_.bGM = x2_;
    pipeLineContext_.cGM = tempComputeOutGM_;
    pipeLineContext_.biasGM = bias_;
    pipeLineContext_.extraData.a_offset = (uint64_t)mc2Tiling_.tileM * mc2Tiling_.rankK * sizeof(DTYPE_X1);
    pipeLineContext_.extraData.b_offset = (uint64_t)0UL;
    pipeLineContext_.extraData.c_offset = (uint64_t)mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.tilingData = &(tilingData_->mc2MmV3TileTilingData);

    //转置操作的输入输出地址，单轮转置内部数据块的偏移，到下一轮转置数据地址的偏移
    pipeLineContext_.transposeSrcAddr = tempComputeOutGM_;
    pipeLineContext_.transposeDstAddr = transOutGM_;
    pipeLineContext_.transposeSrcOffset = (uint64_t)mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.nextSrcBlockOffset = (uint64_t)mc2Tiling_.rankN / mc2Tiling_.rankDim;
    pipeLineContext_.nextDstBlockOffset = (uint64_t)mc2Tiling_.rankM * mc2Tiling_.rankN / mc2Tiling_.rankDim;
    pipeLineContext_.transposeDstOffset = pipeLineContext_.transposeSrcOffset / mc2Tiling_.rankDim;
    pipeLineContext_.rankCnt = mc2Tiling_.rankDim;
    pipeLineContext_.innerAxis = (uint64_t)mc2Tiling_.rankN / mc2Tiling_.rankDim;
    pipeLineContext_.transM = (uint64_t)mc2Tiling_.tileM;

    //通信操作的输入输出地址，到下一轮的地址偏移
    pipeLineContext_.taskCnt = mc2Tiling_.tileCnt;
    pipeLineContext_.sendBuffer = pipeLineContext_.transposeDstAddr;
    pipeLineContext_.recvBuffer = y_;
    pipeLineContext_.sendOffset = (uint64_t)mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y) / mc2Tiling_.rankDim;
    pipeLineContext_.recvOffset = pipeLineContext_.sendOffset;
    pipeLineContext_.sendCount = (uint64_t)mc2Tiling_.tileM * pipeLineContext_.innerAxis;
    pipeLineContext_.strideCount = pipeLineContext_.nextDstBlockOffset;
    pipeLineContext_.hcclDataType = mc2Tiling_.hcclDataType;

    ProcessPipeLine(taskCnt);
}

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType>
__aicore__ inline void MatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType>::ProcessTail(uint32_t taskCnt)
{
    auto&& mc2Tiling_ = tilingData_->matmulAlltoAllTilingInfo;
    pipeLineContext_.aGM = x1_ + mc2Tiling_.tileCnt * mc2Tiling_.tileM * mc2Tiling_.rankK * sizeof(DTYPE_X1);
    pipeLineContext_.bGM = x2_;
    pipeLineContext_.cGM = tempComputeOutGM_ + mc2Tiling_.tileCnt * mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.extraData.a_offset = (uint64_t)mc2Tiling_.tailM * mc2Tiling_.rankK * sizeof(DTYPE_X1);
    pipeLineContext_.extraData.b_offset = (uint64_t)0UL;
    pipeLineContext_.extraData.c_offset = (uint64_t)mc2Tiling_.tailM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.tilingData = &(tilingData_->mc2MmV3TailTilingData);

    uint64_t commonOffset = (uint64_t)mc2Tiling_.rankM * (uint64_t)mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.transposeSrcAddr = tempComputeOutGM_ + mc2Tiling_.tileCnt * mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.transposeDstAddr = transOutGM_ + mc2Tiling_.tileCnt * mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y) / (uint64_t)mc2Tiling_.rankDim;
    pipeLineContext_.transposeSrcOffset = (uint64_t)mc2Tiling_.tailM * (uint64_t)mc2Tiling_.rankN * (uint64_t)sizeof(DTYPE_Y);
    pipeLineContext_.transposeDstOffset = pipeLineContext_.transposeSrcOffset / (uint64_t)mc2Tiling_.rankDim;
    pipeLineContext_.transM = (uint64_t)mc2Tiling_.tailM;

    pipeLineContext_.taskCnt = mc2Tiling_.tailCnt;
    pipeLineContext_.sendBuffer = pipeLineContext_.transposeDstAddr;
    pipeLineContext_.recvBuffer = y_ + mc2Tiling_.tileCnt * mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y) / (uint64_t)mc2Tiling_.rankDim;
    pipeLineContext_.sendOffset = (uint64_t)mc2Tiling_.tailM * (uint64_t)mc2Tiling_.rankN * sizeof(DTYPE_Y) / (uint64_t)mc2Tiling_.rankDim;
    pipeLineContext_.recvOffset = pipeLineContext_.sendOffset;
    pipeLineContext_.sendCount = (uint64_t)mc2Tiling_.tailM * (uint64_t)pipeLineContext_.innerAxis;

    ProcessPipeLine(taskCnt);
}

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType>
__aicore__ inline void MatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType>::ProcessPipeLine(uint32_t taskCnt)
{
    pipeLine_->ChangeSpecification(&pipeLineContext_);
    pipeLine_->Process(taskCnt);
}

}; // namespace MatmulAlltoAllImpl

#endif