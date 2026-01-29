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
 * \file fp_matmul.h 
 * \brief
 */

#ifndef MC2_FP_MATMUL_H
#define MC2_FP_MATMUL_H
#include "matmul_factory.h"

namespace MC2KernelTemplate
{
//额外的数据，输入和输出每一轮计算前后的地址偏移
struct FpQuantExtraData {
    uint64_t a_offset;
    uint64_t b_offset;
    uint64_t c_offset;
};

/**
 * MMKernel：使用的matmul的数据类型
 * ExtraDataType：额外数据结构体的数据类型
 * TilingDataType：matmul算子使用的tiling的数据类型
 */
template <typename MMKernel, typename ExtraDataType, typename TilingDataType>
class FPMatmul {
protected:
    __aicore__ inline void Init();
public:
    __aicore__ inline FPMatmul(AscendC::TPipe* tPipe): tPipe_(tPipe) {};
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

//初始化一个matmul的算子
template <typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline void FPMatmul<MMKernel, ExtraDataType, TilingDataType>::Init() 
{
    if ASCEND_IS_AIV {
        return;
    }
    tPipe_->Reset();
    mmOp_.Init(baseAddrs_.aGM, baseAddrs_.bGM, baseAddrs_.cGM, baseAddrs_.biasGM, nullptr, nullptr, tilingData_, tPipe_);
}

//执行一轮matmul计算节点的计算，如果还有下一轮，为下一轮计算准备参数
template <typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline void FPMatmul<MMKernel, ExtraDataType, TilingDataType>::Process(bool isFirst) 
{
    if ASCEND_IS_AIV {
        return;
    }

    if (!isFirst) {
        baseAddrs_.aGM = (GM_ADDR)((uint64_t)baseAddrs_.aGM + extraData_.a_offset);
        baseAddrs_.bGM = (GM_ADDR)((uint64_t)baseAddrs_.bGM + extraData_.b_offset);
        baseAddrs_.cGM = (GM_ADDR)((uint64_t)baseAddrs_.cGM + extraData_.c_offset);
        Init();
    }
    mmOp_.Process();
}

//更换流水线上matmul计算节点的规格，包括输入输出地址和tiling信息，然后重新初始化matmul算子
template <typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline void FPMatmul<MMKernel, ExtraDataType, TilingDataType>::Update(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, ExtraDataType* extraData, TilingDataType* tilingData) 
{
    if ASCEND_IS_AIV {
        return;
    }   
    baseAddrs_.aGM = aGM;
    baseAddrs_.bGM = bGM;
    baseAddrs_.cGM = cGM;
    baseAddrs_.biasGM = biasGM;
    if (extraData != nullptr) {
        extraData_.a_offset = extraData->a_offset;
        extraData_.b_offset = extraData->b_offset;
        extraData_.c_offset = extraData->c_offset;
    }
    if (tilingData != nullptr) {
        tilingData_ = tilingData;
        Init();
    }
}

//流水线释放时，释放matmul计算节点
template <typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline void FPMatmul<MMKernel, ExtraDataType, TilingDataType>::End()
{
    if ASCEND_IS_AIV {
        return;
    }
    mmOp_.End();
}
};
#endif