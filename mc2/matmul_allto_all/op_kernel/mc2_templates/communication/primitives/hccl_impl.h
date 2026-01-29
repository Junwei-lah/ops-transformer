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
 * \file hccl_impl.h 
 * \brief
 */

#ifndef MC2_HCCL_IMPL_H
#define MC2_HCCL_IMPL_H

#include "lib/hccl/hccl.h"

namespace MC2KernelTemplate {
using namespace AscendC;
/**
 * ServerType:通信控制方式，ccu/mte/aicpu等
 * SendCnt:每轮发送的次数
 * RecvCnt:每轮等待发送完毕的次数
 * TilingDataType:tilingdata具体数据类型
 * CommunicationPrimitive:通信原语，alltoall等,暂不启用
 */
template <HcclServerType ServerType, typename TilingDataType, uint32_t SendCnt, uint32_t RecvCnt>
class HcclCommunication
{
public:
    __aicore__ inline HcclCommunication(TilingDataType* tiling) : tiling_(tiling)
    {}

    __aicore__ inline void Init()
    {
        notifyFlag_ = false;   
        if ASCEND_IS_AIV {
            if (AscendC::GetBlockIdx() == 0) {
                notifyFlag_ = true; 
            }
        }
        
        hccl_.InitV2(GetHcclContext<0>(), &(tiling_->mc2InitTiling));
        hccl_.SetCcTilingV2(offsetof(TilingDataType, mc2CcTiling));
        if constexpr (SendCnt == 1U && RecvCnt == 0U) {
            communicationType_ = Communicationtype::COMMUNICATION_SEND_ONE;
        } else if constexpr (SendCnt == 0U && RecvCnt == 1U) {
            communicationType_ = Communicationtype::COMMUNICATION_WAIT_ONE;
        }
        sendIndex_ = 0;
        recvIndex_ = 0;
        sendBuffer_ = 0;
        recvBuffer_ = 0;
        sendOffset_ = 0;
        recvOffset_ = 0;
        sendCount_ = 0;
        strideCount_ = 0;
    }

    /**
     * 初始化通信相关上下文，包括地址信息和每一轮的偏移信息等
     * loopType:首尾轮通信区分
     * sendBuffer:发送地址
     * recvBuffer:接收地址
     * sendOffset:每一轮与下一轮发送地址偏移
     * recvOffset:每一轮与下一轮接收地址偏移
     * sendCount:发送数据量(个数)
     * strideCount:发往不同卡数据之间的偏移(个数)
     * 
     */
    __aicore__ inline void Update(uint32_t taskCnt, GM_ADDR sendBuffer, GM_ADDR recvBuffer, 
        uint64_t sendOffset, uint64_t recvOffset, uint64_t sendCount, uint64_t strideCount, uint8_t hcclDataType)
    {
        //只有通信核参与通信
        if (!notifyFlag_) {
            return;
        }
        sendBuffer_ = (uint64_t)sendBuffer;
        recvBuffer_ = (uint64_t)recvBuffer;
        sendOffset_ = sendOffset;
        recvOffset_ = recvOffset;
        sendCount_ = sendCount;
        strideCount_ = strideCount;
        hcclDataType_ = (AscendC::HcclDataType)(static_cast<uint8_t>(hcclDataType));
        // 如果是先通后算就全量启动通信
        if (communicationType_ == Communicationtype::COMMUNICATION_WAIT_ONE) {
            //alltoall接口职责不单一，这里只使用repeat=1的模式
            uint8_t repeat = 1;
            for (uint32_t i = 0; i < taskCnt; i++) {
                hTasks_[sendIndex_ + i] = hccl_.template AlltoAll<true>((GM_ADDR)sendBuffer, (GM_ADDR)recvBuffer, sendCount, hcclDataType_, strideCount, repeat);
                sendBuffer += sendOffset;
                recvBuffer += recvOffset;
            }
            //更新全局变量
            sendIndex_ += taskCnt;
        }
    }

    // 执行一轮流水，包括通信地址更新，先通后算等待一轮通信完成，先算后通开始一轮通信
    __aicore__ inline void Process()
    {
        //只有通信核参与通信
        if (!notifyFlag_) {
            return;
        }
        if (communicationType_ == Communicationtype::COMMUNICATION_WAIT_ONE) {
            hccl_.Wait(hTasks_[recvIndex_]);
            recvIndex_++;
        } else if (communicationType_ == Communicationtype::COMMUNICATION_SEND_ONE) {
            uint8_t repeat = 1;
            hTasks_[sendIndex_] = hccl_.template AlltoAll<true>((GM_ADDR)sendBuffer_, (GM_ADDR)recvBuffer_, sendCount_, hcclDataType_, strideCount_, repeat);
            sendBuffer_ += sendOffset_;
            recvBuffer_ += recvOffset_;
            sendIndex_++;
        }
    }

    // 结束hccl通信
    __aicore__ inline void End()
    {
        // 如果是先算后通就全量等待通信
        if (notifyFlag_ && communicationType_ == Communicationtype::COMMUNICATION_SEND_ONE) {
            for (;recvIndex_ < sendIndex_;++recvIndex_) {
                hccl_.Wait(hTasks_[recvIndex_]);
            }
        }

        // 防止block_idx0执行过快清空RcvCnt,增加全核同步
        SyncAll<false>();
        if (notifyFlag_) {
            hccl_.Finalize();
        }
    }       

private:
    enum Communicationtype{
        COMMUNICATION_WAIT_ONE,
        COMMUNICATION_SEND_ONE
    };
    TilingDataType* tiling_;
    Hccl<ServerType> hccl_;
    // todo 属性参数可能可以转化为方法入参或者临时变量
    uint32_t sendIndex_ = 0;
    uint32_t recvIndex_ = 0;
    uint64_t sendBuffer_ = 0;
    uint64_t recvBuffer_ = 0;
    uint64_t sendOffset_ = 0;
    uint64_t recvOffset_ = 0;
    uint64_t sendCount_ = 0;
    uint64_t strideCount_ = 0;
    bool notifyFlag_ = false;
    AscendC::HcclDataType hcclDataType_;
    Communicationtype communicationType_ = COMMUNICATION_WAIT_ONE;
    AscendC::HcclHandle hTasks_[16]; //hccl只支持最多16个任务并行
};

};

#endif