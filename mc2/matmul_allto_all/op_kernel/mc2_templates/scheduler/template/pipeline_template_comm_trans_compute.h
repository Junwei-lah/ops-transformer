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
 * \file pipeline_template_comm_trans_compute.h
 * \brief
 */

#ifndef MC2_PIPELINE_TEMPLATE_COMM_TRANS_COMPUTE_H
#define MC2_PIPELINE_TEMPLATE_COMM_TRANS_COMPUTE_H

#include "pipeline_context.h"

// 流水线模板
namespace MC2KernelTemplate {
//通信转置计算模板
template <typename CommunicationType, typename TransposeType, typename ComputationType, typename ContextType>
class MC2KernelPipelineCommTransComputeTemplate {
public:
    __aicore__ inline MC2KernelPipelineCommTransComputeTemplate(CommunicationType* commStage, TransposeType* transStage, ComputationType* computeStage) : commStage_(commStage), transStage_(transStage), computeStage_(computeStage){};

    __aicore__ inline void Init();

    __aicore__ inline void ChangeSpecification(void* updateContext);

    __aicore__ inline void Process(uint32_t taskCnt);

    __aicore__ inline void End();

private:
    CommunicationType* commStage_; // 通信节点
    TransposeType* transStage_; // 转置计算的计算节点
    ComputationType* computeStage_; // 矩阵乘的计算节点
    ContextType* context_; //相关上下文
};

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename ContextType>
__aicore__ inline void MC2KernelPipelineCommTransComputeTemplate<CommunicationType, TransposeType, ComputationType, ContextType>::Init()
{
    commStage_->Init();
}

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename ContextType>
__aicore__ inline void MC2KernelPipelineCommTransComputeTemplate<CommunicationType, TransposeType, ComputationType, ContextType>::ChangeSpecification(void* updateContext)
{
    context_ = (ContextType*) updateContext;
    commStage_->Update(context_->taskCnt, context_->sendBuffer, context_->recvBuffer, 
        context_->sendOffset, context_->recvOffset, context_->sendCount, context_->strideCount, context_->hcclDataType);
    computeStage_->Update(context_->aGM, context_->bGM, context_->cGM, context_->biasGM, &(context_->extraData), context_->tilingData);
}

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename ContextType>
__aicore__ inline void MC2KernelPipelineCommTransComputeTemplate<CommunicationType, TransposeType, ComputationType, ContextType>::Process(uint32_t taskCnt)
{
    uint32_t index;
    for (index = 0 ; index < taskCnt; index++) {
        if ASCEND_IS_AIV {
            commStage_->Process();
            AscendC::SyncAll<true>();
            transStage_->Init(context_->transposeSrcAddr, context_->transposeDstAddr, context_->rankCnt, context_->innerAxis, context_->transM, context_->nextSrcBlockOffset, context_->nextDstBlockOffset, context_->innerAxis, context_->innerAxis * context_->rankCnt);
            transStage_->Process();
            transStage_->Destroy();
            context_->transposeSrcAddr = (GM_ADDR)((uint64_t)context_->transposeSrcAddr + context_->transposeSrcOffset);
            context_->transposeDstAddr = (GM_ADDR)((uint64_t)context_->transposeDstAddr + context_->transposeDstOffset);
            CrossCoreSetFlag<0, PIPE_MTE3>(8);
            CrossCoreWaitFlag(8);
            CrossCoreSetFlag<2, PIPE_MTE3>(9);
        }
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(9);
            computeStage_->Process(index == 0);
        }
        AscendC::SyncAll<false>();
    }
}

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename ContextType>
__aicore__ inline void MC2KernelPipelineCommTransComputeTemplate<CommunicationType, TransposeType, ComputationType, ContextType>::End()
{
    commStage_->End();
    computeStage_->End();
}
};

#endif