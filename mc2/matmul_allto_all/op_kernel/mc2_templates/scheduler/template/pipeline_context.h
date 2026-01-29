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
 * \file pipeline_context.h
 * \brief
 */

#ifndef MC2_PIPELINE_CONTEXT_H
#define MC2_PIPELINE_CONTEXT_H

namespace MC2KernelTemplate {
//todo 后续可以按节点拆成对应的上下文复用
template <typename ExtraDataType, typename TilingDataType>
struct PipelineContext {
// computation
    GM_ADDR aGM;
    GM_ADDR bGM;
    GM_ADDR cGM;
    GM_ADDR biasGM;
    ExtraDataType extraData;
    TilingDataType* tilingData;
// transpose
    GM_ADDR transposeSrcAddr;
    GM_ADDR transposeDstAddr;
    uint64_t transposeSrcOffset;
    uint64_t nextSrcBlockOffset;
    uint64_t nextDstBlockOffset;
    uint64_t transposeDstOffset;
    uint32_t rankCnt;
    uint64_t innerAxis;
    uint64_t transM;
// communication
    uint32_t taskCnt;
    GM_ADDR sendBuffer;
    GM_ADDR recvBuffer;
    uint64_t sendOffset;
    uint64_t recvOffset;
    uint64_t sendCount;
    uint64_t strideCount;
    uint64_t hcclDataType;
};
};

#endif