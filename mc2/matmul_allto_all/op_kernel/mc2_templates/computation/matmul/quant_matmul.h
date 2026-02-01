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
 * \file quant_matmul.h
 * \brief
 */

#ifndef MC2_QUANT_MATMUL_H
#define MC2_QUANT_MATMUL_H

namespace MC2KernelTemplate {
struct QuantExtraData {
    uint64_t a_offset;
    uint64_t b_offset;
    uint64_t c_offset;
    uint64_t x1_scale_offset;
    GM_ADDR x1_scale;
    GM_ADDR x2_scale;
    GM_ADDR x2_offset;
};

template <typename MMKernel, typename ExtraDataType, typename TilingDataType>
class QuantMatmul {
protected:
    __aicore__ inline void Init();
public:
    __aicore__ inline QuantMatmul(AscendC::TPipe* tPipe): tPipe_(tPipe) {};
    __aicore__ inline void Process(bool hasNext);
    __aicore__ inline void Update(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, ExtraDataType* extraData, TilingDataType* tilingData);
    __aicore__ inline void End();
private:
    MMKernel mmOp_;
    BaseGmAddrs baseAddrs_;
    ExtraDataType extraData_;
    AscendC::TPipe* tPipe_;
    TilingDataType* tilingData_;
};

template <typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline void QuantMatmul<MMKernel, ExtraDataType, TilingDataType>::Init() 
{
    tPipe_->Reset();
    mmOp_.Init(baseAddrs_.aGM, baseAddrs_.bGM, extraData_.x2_scale, extraData_.x2_offset, baseAddrs_.biasGM, extraData_.x1_scale, baseAddrs_.cGM, nullptr, tilingData_, tPipe_);
}

template <typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline void QuantMatmul<MMKernel, ExtraDataType, TilingDataType>::Process(bool isFirst){
    
    if (!isFirst) {
        baseAddrs_.aGM = (GM_ADDR)((uint64_t)baseAddrs_.aGM + extraData_.a_offset);
        baseAddrs_.bGM = (GM_ADDR)((uint64_t)baseAddrs_.bGM + extraData_.b_offset);
        baseAddrs_.cGM = (GM_ADDR)((uint64_t)baseAddrs_.cGM + extraData_.c_offset);
        extraData_.x1_scale = (GM_ADDR)((uint64_t)extraData_.x1_scale + extraData_.x1_scale_offset);
    }
    Init();
    mmOp_.Process();
}

template <typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline void QuantMatmul<MMKernel, ExtraDataType, TilingDataType>::Update(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, ExtraDataType* extraData, TilingDataType* tilingData) 
{    
    baseAddrs_.aGM = aGM;
    baseAddrs_.bGM = bGM;
    baseAddrs_.cGM = cGM;
    baseAddrs_.biasGM = biasGM;
    if (extraData != nullptr) {
        extraData_.a_offset = extraData->a_offset;
        extraData_.b_offset = extraData->b_offset;
        extraData_.c_offset = extraData->c_offset;
        extraData_.x1_scale_offset = extraData->x1_scale_offset;
        extraData_.x1_scale = extraData->x1_scale;
        extraData_.x2_scale = extraData->x2_scale;
        extraData_.x2_offset = extraData->x2_offset;
    }
    if (tilingData != nullptr) {
        tilingData_ = tilingData;
    }
}

template <typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline void QuantMatmul<MMKernel, ExtraDataType, TilingDataType>::End()
{
}
};
#endif

