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
 * \file moe_distribute_combine_setup_arch35.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_SETUP_ARCH35_H
#define MOE_DISTRIBUTE_COMBINE_SETUP_ARCH35_H

#if __has_include("../common/inc/kernel/mc2_kernel_utils.h")
#include "../common/inc/kernel/mc2_kernel_utils.h"
#else
#include "../../common/inc/kernel/mc2_kernel_utils.h"
#endif

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../moe_distribute_base.h"
#include "../moe_distribute_combine_setup_tiling_data.h"

namespace MoeDistributeCombineSetupImpl {

struct WriteWithNotifySQEInfoParams {
    uint64_t dataSrcAddr;
    uint64_t dataDstAddr;
    uint32_t length;
    uint64_t notifyAddr;
    uint64_t notifyData;
    uint8_t cqe;
};

#define TemplateMC2TypeClass typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeFunc ExpandXType, ExpandIdxType

using namespace AscendC;
template <TemplateMC2TypeClass>
class MoeDistributeCombineSetup {
    constexpr static uint8_t BUFFER_NUM = 2;              // 多buf
    constexpr static uint64_t STATE_OFFSET = 512U;        // 状态空间偏移地址
    constexpr static uint32_t STATE_SIZE = 1024U * 1024U; // 1M
    constexpr static uint32_t UB_ALIGN = 32U;             // UB按32字节对齐
    constexpr static uint64_t WIN_STATE_OFFSET = 350U * 1024U;
    constexpr static uint64_t STATE_WIN_OFFSET = 950U * 1024U;
    constexpr static uint64_t STATE_SIZE_PER_CORE = 512U;   // 数据和状态的0/1区标识占用空间
    constexpr static uint64_t COMBINE_STATE_OFFSET = 0U;    // 本卡状态空间偏移地址，前面的地址给dispatch用
    constexpr static uint32_t STATE_COUNT_THRESHOLD = 512U; // moeExpertNumPerRank*epWorldSize状态数阈值

public:
    __aicore__ inline MoeDistributeCombineSetup(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR quantExpandX,
                                GM_ADDR commCmdInfoOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeCombineSetupTilingData *tilingData, __gm__ void *mc2InitTiling,
                                __gm__ void *mc2CcTiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void CurRankComm(const LocalTensor<int32_t> &assistInfoForCombineLocal,
                                       uint32_t curRankExpertNum);
    __aicore__ inline void InitCqeStatus();
    __aicore__ inline void Communication();
    __aicore__ inline void BuffInit();
    __aicore__ inline void AssistInfoLocalCopy();
    __aicore__ inline void UrmaInit(const LocalTensor<uint8_t> &sqInfoU8, const LocalTensor<uint8_t> &cqInfoU8,
                                    const LocalTensor<uint8_t> &cqeTensorU8, const LocalTensor<uint8_t> &jfcDoorBellU8,
                                    const LocalTensor<uint8_t> &templateSqeU8, uint32_t epIdx, uint32_t &sqPi,
                                    uint32_t &sqCi, uint32_t &cqPi, uint32_t &cqCi, uint32_t &sqPiLinear,
                                    uint32_t &cqCiLinear);
    __aicore__ inline void SendPerExpert(const LocalTensor<uint8_t> &tokenSqeU8,
                                         const LocalTensor<uint8_t> &templateSqeU8,
                                         const LocalTensor<int32_t> &assistInfoForCombineLocal, uint32_t epIdx,
                                         uint32_t curRankExpertNum, WriteWithNotifySQEInfoParams &notifySqeInfo,
                                         uint32_t &tokenSqeNum);

    __aicore__ GM_ADDR GetWinAddrByRankId(const uint32_t rankId, const uint8_t expertLocalId = 0U)
    {
        return (GM_ADDR)(hcclContext_->windowsIn[rankId]) + winDataSizeOffset_ +
               expertPerSizeOnWin_ * static_cast<uint64_t>(expertLocalId);
    }

    __aicore__ GM_ADDR GetWinStateAddrByRankId(uint32_t rankId)
    {
        return (GM_ADDR)(hcclContext_->windowsOut[rankId]) + COMBINE_STATE_OFFSET +
               WIN_STATE_OFFSET * static_cast<uint64_t>(dataState_);
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<int32_t> assistInfoForCombineGlobal_;
    GM_ADDR epWindowGM_;
    GM_ADDR epStatusSpaceGM_;
    GM_ADDR expandXGM_;

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    const MoeDistributeCombineSetupInfo *moeDistributeCombineSetupInfo_{nullptr};
    uint32_t axisMaxBS_{0};
    uint32_t coreIdx_{0};    // aiv id
    uint32_t moeSendNum_{0}; // moeExpertPerRankNum * epWorldSize
    uint64_t epDataOffsetOnWin_{0};
    uint64_t epStateOffsetOnWin_{0};
    uint64_t axisHExpandXTypeSize_{0};
    uint32_t startRankId_{0}; // 当前核处理的起始卡号
    uint32_t endRankId_{0};   // 当前核处理的结束卡号
    uint32_t sendRankNum_{0};
    uint32_t dataState_{0};
    uint64_t stateOffset_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    bool isShardExpert_{false};

    TQue<QuePosition::VECIN, 1> assistInfoQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1>
        expertTokenTmpQueue_; // URMA本卡回环通信拷贝数据Local Memory暂存

    // URMA通信新增变量、Buf
    TBuf<> urmaSqInfoBuf_;  // 80B
    TBuf<> urmaCqInfoBuf_;  // 48B
    TBuf<> jfsDoorBellBuf_; // 32B
    TBuf<> jfcDoorBellBuf_; // 32B
    TBuf<> cqeBuf_;         // 32B * CQ_DEPTH_256
    TBuf<> templateSqeBuf_; // 96B
    TBuf<> tokenSqeBuf_;    // 96B

    __gm__ HcclCombinOpParam *hcclContext_{nullptr};
};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Init(
    GM_ADDR expandX, GM_ADDR /*expertIds*/, GM_ADDR assistInfoForCombine, GM_ADDR /*quantExpandX*/,
    GM_ADDR /*commCmdInfoOut*/, GM_ADDR /*workspaceGM*/, TPipe *pipe,
    const MoeDistributeCombineSetupTilingData *tilingData, __gm__ void *mc2InitTiling, __gm__ void *mc2CcTiling)
{
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();
    moeDistributeCombineSetupInfo_ = &(tilingData->moeDistributeCombineSetupInfo);
    hcclContext_ = (__gm__ HcclCombinOpParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();

    // 获取win状态区地址，并保证数据一致
    // 在1M中选择512K偏移后的1.5k空间记录本卡历史状态
    GlobalTensor<int32_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)hcclContext_->windowsOut[moeDistributeCombineSetupInfo_->epRankId];
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + STATE_WIN_OFFSET +
                                                            STATE_SIZE_PER_CORE * static_cast<uint64_t>(coreIdx_)));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);

    dataState_ = selfDataStatusTensor(0); // win状态区的0/1标识
    expandXGM_ = expandX;
    assistInfoForCombineGlobal_.SetGlobalBuffer((__gm__ int32_t *)assistInfoForCombine);

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    axisMaxBS_ = moeDistributeCombineSetupInfo_->globalBs / moeDistributeCombineSetupInfo_->epWorldSize;
    moeSendNum_ = moeDistributeCombineSetupInfo_->epWorldSize * moeDistributeCombineSetupInfo_->moeExpertPerRankNum;
    isShardExpert_ = (moeDistributeCombineSetupInfo_->epRankId <
                      moeDistributeCombineSetupInfo_->sharedExpertRankNum); // 当前rank是否为共享专家

    axisHExpandXTypeSize_ = static_cast<uint64_t>(moeDistributeCombineSetupInfo_->h) *
                            static_cast<uint64_t>(sizeof(ExpandXType));              // 一个token占用内存
    expertPerSizeOnWin_ = static_cast<uint64_t>(axisMaxBS_) * axisHExpandXTypeSize_; // 单个专家可能占用的最大空间

    winDataSizeOffset_ =
        static_cast<uint64_t>(dataState_) * (static_cast<uint64_t>(moeDistributeCombineSetupInfo_->totalWinSize) >> 1);
    stateOffset_ = (moeSendNum_ > STATE_COUNT_THRESHOLD) ? (STATE_OFFSET >> 1) : STATE_OFFSET;
    epStateOffsetOnWin_ = static_cast<uint64_t>(moeDistributeCombineSetupInfo_->epRankId) * stateOffset_;
    epDataOffsetOnWin_ = static_cast<uint64_t>(moeDistributeCombineSetupInfo_->epRankId) *
                         static_cast<uint64_t>(moeDistributeCombineSetupInfo_->moeExpertPerRankNum) *
                         expertPerSizeOnWin_; // 前面rank数据区占用内存地址偏移

    epWindowGM_ = GetWinAddrByRankId(moeDistributeCombineSetupInfo_->epRankId);
    epStatusSpaceGM_ = GetWinStateAddrByRankId(moeDistributeCombineSetupInfo_->epRankId);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    OOMCheckAddrRange<ExpandXType>((__gm__ ExpandXType *)(epWindowGM_), moeDistributeCombineSetupInfo_->totalWinSize);
    OOMCheckAddrRange<float>((__gm__ float *)(epStatusSpaceGM_), STATE_SIZE);
#endif

    SplitCoreCal(); // 分核计算
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::SplitCoreCal()
{
    // 对worldSize按卡分核，得到每个核上处理的卡的数量，保证一个jetty只由一个核维护
    uint32_t coreIdxNew = (coreIdx_ + moeDistributeCombineSetupInfo_->epRankId) %
                          moeDistributeCombineSetupInfo_->aivNum; // 按照卡去进行偏移
    sendRankNum_ = moeDistributeCombineSetupInfo_->epWorldSize / moeDistributeCombineSetupInfo_->aivNum;
    uint32_t remainderRankNum = moeDistributeCombineSetupInfo_->epWorldSize % moeDistributeCombineSetupInfo_->aivNum;
    startRankId_ = sendRankNum_ * coreIdxNew;
    if (coreIdxNew < remainderRankNum) {
        ++sendRankNum_;
        startRankId_ += coreIdxNew;
    } else {
        startRankId_ += remainderRankNum;
    }
    endRankId_ = startRankId_ + sendRankNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::InitCqeStatus()
{
    LocalTensor<uint8_t> cqInfoU8 = urmaCqInfoBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> cqeTensorU8 = cqeBuf_.Get<uint8_t>();
    bool isFirstInitCqe = false;

    for (uint32_t epIdx = startRankId_; epIdx < endRankId_; ++epIdx) {
        if (unlikely(epIdx == moeDistributeCombineSetupInfo_->epRankId)) {
            continue;
        }

        GetIsFirstInComm((GM_ADDR)hcclContext_, moeDistributeCombineSetupInfo_->epRankId, epIdx, isFirstInitCqe);

        // 把CQ中所有CQE的status初始化为无效值0xff
        if (unlikely(!isFirstInitCqe)) {
            GetURMACqInfoTensor(cqInfoU8, (GM_ADDR)hcclContext_, epIdx);
            AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>(); // 等cqInfoU8的GM->Local，后续InvalidateCqeStatus标量读
            InvalidateCqeStatus(cqInfoU8, cqeTensorU8);
            AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); // 等cqGlobalTensor的Local->GM，后续GM->Local
            UpdateIsFirstInComm((GM_ADDR)hcclContext_, moeDistributeCombineSetupInfo_->epRankId, epIdx, true);
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::BuffInit()
{
    tpipe_->Reset();
    tpipe_->InitBuffer(assistInfoQueue_, BUFFER_NUM,
                       moeSendNum_ * sizeof(int32_t)); // epWorldSize * moeExpertPerRankNum * 4

    // 初始化urma相关buf
    tpipe_->InitBuffer(urmaSqInfoBuf_, sizeof(HcclAiRMAWQ));
    tpipe_->InitBuffer(urmaCqInfoBuf_, sizeof(HcclAiRMACQ));
    tpipe_->InitBuffer(jfsDoorBellBuf_, static_cast<uint32_t>(sizeof(uint32_t)));
    tpipe_->InitBuffer(jfcDoorBellBuf_, static_cast<uint32_t>(sizeof(uint32_t)));
    tpipe_->InitBuffer(cqeBuf_, UB_ALIGN * CQ_DEPTH_256);
    tpipe_->InitBuffer(templateSqeBuf_, WRITE_WITH_NOTIFY_SQE_SIZE);
    tpipe_->InitBuffer(tokenSqeBuf_, WRITE_WITH_NOTIFY_SQE_SIZE * moeDistributeCombineSetupInfo_->moeExpertPerRankNum);

    tpipe_->InitBuffer(expertTokenTmpQueue_, 1, static_cast<uint32_t>(expertPerSizeOnWin_));
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::AssistInfoLocalCopy()
{
    LocalTensor<int32_t> assistInfoForCombineLocal = assistInfoQueue_.AllocTensor<int32_t>();

    DataCopyExtParams epSendCntParams;
    if (isShardExpert_) {
        // 对于共享专家来说assistInfoForCombine输入维度为epWordSize个
        epSendCntParams = {1U, moeDistributeCombineSetupInfo_->epWorldSize * static_cast<uint32_t>(sizeof(uint32_t)),
                           0U, 0U, 0U};
    } else {
        epSendCntParams = {1U, moeSendNum_ * static_cast<uint32_t>(sizeof(uint32_t)), 0U, 0U, 0U};
    }
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(assistInfoForCombineLocal, assistInfoForCombineGlobal_, epSendCntParams, copyPadParams);
    assistInfoQueue_.EnQue(assistInfoForCombineLocal);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::UrmaInit(
    const LocalTensor<uint8_t> &sqInfoU8, const LocalTensor<uint8_t> &cqInfoU8, const LocalTensor<uint8_t> &cqeTensorU8,
    const LocalTensor<uint8_t> &jfcDoorBellU8, const LocalTensor<uint8_t> &templateSqeU8, uint32_t epIdx,
    uint32_t &sqPi, uint32_t &sqCi, uint32_t &cqPi, uint32_t &cqCi, uint32_t &sqPiLinear, uint32_t &cqCiLinear)
{
    // URMA 加载SQ CQ
    GetURMASqInfoTensor(sqInfoU8, (GM_ADDR)hcclContext_, epIdx);
    GetURMACqInfoTensor(cqInfoU8, (GM_ADDR)hcclContext_, epIdx);
    AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>(); // 等sqInfoU8、cqInfoU8从GM拷贝Local，后续标量读

    // 获取PI CI
    GetPICI((GM_ADDR)hcclContext_, moeDistributeCombineSetupInfo_->epRankId, epIdx, sqPi, sqCi, cqPi, cqCi, sqPiLinear,
            cqCiLinear);

    PollCommCQUpdateSQCI(sqInfoU8, cqInfoU8, cqeTensorU8, jfcDoorBellU8, sqCi, cqCi, cqCiLinear);

    // 根据当前处理卡号更新WQE模板
    UpdateCommWriteWithNotifySQE(templateSqeU8, sqInfoU8);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::SendPerExpert(
    const LocalTensor<uint8_t> &tokenSqeU8, const LocalTensor<uint8_t> &templateSqeU8,
    const LocalTensor<int32_t> &assistInfoForCombineLocal, uint32_t epIdx, uint32_t curRankExpertNum,
    WriteWithNotifySQEInfoParams &notifySqeInfo, uint32_t &tokenSqeNum)
{
    GM_ADDR dstStateAddr = GetWinStateAddrByRankId(epIdx) + epStateOffsetOnWin_;

    for (uint32_t expertIdx = 0U; expertIdx < curRankExpertNum; ++expertIdx) {
        uint32_t preCount = 0U;
        uint32_t assistInfoIdx = expertIdx * moeDistributeCombineSetupInfo_->epWorldSize + epIdx;
        if (likely(assistInfoIdx > 0U)) {
            // 计算其他卡或专家已经发了多少token
            preCount = assistInfoForCombineLocal.GetValue(assistInfoIdx - 1U);
        }
        // 当前要发送的token数量
        uint32_t curTokenNum = assistInfoForCombineLocal.GetValue(assistInfoIdx) - preCount;
        if (unlikely(curTokenNum == 0U)) {
            continue;
        }

        GM_ADDR srcAddr = expandXGM_ + static_cast<uint64_t>(preCount) * axisHExpandXTypeSize_;
        GM_ADDR dstAddr = GetWinAddrByRankId(epIdx, expertIdx) + epDataOffsetOnWin_;

        // 发送上一个WQE模板
        if (likely(tokenSqeNum > 0U)) {
            AscendC::SyncFunc<AscendC::HardEvent::S_V>(); // 等templateSqeU8标量设置完成
            DataCopy(tokenSqeU8[WRITE_WITH_NOTIFY_SQE_SIZE * (tokenSqeNum - 1)], templateSqeU8,
                     WRITE_WITH_NOTIFY_SQE_SIZE);
            AscendC::SyncFunc<AscendC::HardEvent::V_S>();
            SetCommWriteWithNotifySQE(tokenSqeU8[WRITE_WITH_NOTIFY_SQE_SIZE * (tokenSqeNum - 1)],
                                      notifySqeInfo.dataSrcAddr, notifySqeInfo.dataDstAddr, notifySqeInfo.length,
                                      notifySqeInfo.notifyAddr + sizeof(uint64_t), notifySqeInfo.notifyData,
                                      notifySqeInfo.cqe);
        }

        // 组装下一个WQE模板
        notifySqeInfo = {(uint64_t)srcAddr,
                         (uint64_t)dstAddr,
                         static_cast<uint32_t>(axisHExpandXTypeSize_) * curTokenNum,
                         (uint64_t)dstStateAddr,
                         0U,
                         0U};

        ++tokenSqeNum;
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Communication()
{
    uint32_t curRankExpertNum = (isShardExpert_) ? 1U : moeDistributeCombineSetupInfo_->moeExpertPerRankNum;
    AssistInfoLocalCopy();
    LocalTensor<int32_t> assistInfoForCombineLocal = assistInfoQueue_.DeQue<int32_t>();
    AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>(); // 等assistInfoQueue_的GM->Local，后续标量读

    if (unlikely(moeDistributeCombineSetupInfo_->epRankId >= startRankId_ &&
                 moeDistributeCombineSetupInfo_->epRankId < endRankId_)) {
        // 本卡通信
        CurRankComm(assistInfoForCombineLocal, curRankExpertNum);
    }

    uint32_t sqPi = 0U, sqCi = 0U, cqPi = 0U, cqCi = 0U, sqPiLinear = 0U, cqCiLinear = 0U;
    LocalTensor<uint8_t> sqInfoU8 = urmaSqInfoBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> cqInfoU8 = urmaCqInfoBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> cqeTensorU8 = cqeBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> jfcDoorBellU8 = jfcDoorBellBuf_.Get<uint8_t>(8); // 2*sizeof(uint32_t)=8*sizeof(uint8_t)
    LocalTensor<uint8_t> templateSqeU8 = templateSqeBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> tokenSqeU8 = tokenSqeBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> jfsDoorBellU8 = jfsDoorBellBuf_.Get<uint8_t>(4); // 1*sizeof(uint32_t)=4*sizeof(uint8_t)

    // 同一个卡只发一次状态，因此先对卡做循环
    for (uint32_t epIdx = startRankId_; epIdx < endRankId_; ++epIdx) {
        if (unlikely(epIdx == moeDistributeCombineSetupInfo_->epRankId)) {
            continue;
        }

        UrmaInit(sqInfoU8, cqInfoU8, cqeTensorU8, jfcDoorBellU8, templateSqeU8, epIdx, sqPi, sqCi, cqPi, cqCi,
                 sqPiLinear, cqCiLinear);

        uint32_t tokenSqeNum = 0;

        // 组装第一个SQE，如果后续存在有效SQE则忽略该SQE
        WriteWithNotifySQEInfoParams notifySqeInfo{(uint64_t)expandXGM_,
                                                   (uint64_t)(GetWinAddrByRankId(epIdx) + epDataOffsetOnWin_),
                                                   static_cast<uint32_t>(axisHExpandXTypeSize_),
                                                   (uint64_t)(GetWinStateAddrByRankId(epIdx) + epStateOffsetOnWin_),
                                                   0U, 0U};

        SendPerExpert(tokenSqeU8, templateSqeU8, assistInfoForCombineLocal, epIdx, curRankExpertNum, notifySqeInfo,
                      tokenSqeNum);

        if (unlikely(tokenSqeNum == 0)) {
            tokenSqeNum = 1;
        }

        // 给最后一个要发送的WQ设置flag，保证强保序
        AscendC::SyncFunc<AscendC::HardEvent::S_V>(); // 等templateSqeU8标量设置完成
        DataCopy(tokenSqeU8[WRITE_WITH_NOTIFY_SQE_SIZE * (tokenSqeNum - 1)], templateSqeU8, WRITE_WITH_NOTIFY_SQE_SIZE);
        AscendC::SyncFunc<AscendC::HardEvent::V_S>();
        SetCommWriteWithNotifySQE(tokenSqeU8[WRITE_WITH_NOTIFY_SQE_SIZE * (tokenSqeNum - 1)], notifySqeInfo.dataSrcAddr,
                                  notifySqeInfo.dataDstAddr, notifySqeInfo.length, notifySqeInfo.notifyAddr, 0x3f800000,
                                  1);
        AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>(); // 等tokenSqeU8标量写，后续Local->GM

        // 发数据
        PutCommNotifySQE(sqInfoU8, cqInfoU8, tokenSqeU8, cqeTensorU8, jfcDoorBellU8, tokenSqeNum, sqPi, sqPiLinear,
                         sqCi, cqCi, cqCiLinear);
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_S>(); // 等sqe下发完成后敲doorbell
        SendJFSDoorBell(jfsDoorBellU8, sqInfoU8, sqPiLinear);

        // 更新PI CI
        UpdatePICI((GM_ADDR)hcclContext_, moeDistributeCombineSetupInfo_->epRankId, epIdx, sqPi, sqCi, cqPi, cqCi,
                   sqPiLinear, cqCiLinear);
    }

    assistInfoQueue_.FreeTensor<int32_t>(assistInfoForCombineLocal);
}

template <TemplateMC2TypeClass>
__aicore__ inline void
MoeDistributeCombineSetup<TemplateMC2TypeFunc>::CurRankComm(const LocalTensor<int32_t> &assistInfoForCombineLocal,
                                                            uint32_t curRankExpertNum)
{
    uint32_t curTokenNum = 0;
    for (uint32_t expertIdx = 0U; expertIdx < curRankExpertNum; ++expertIdx) {
        uint32_t preCount = 0U;
        uint32_t assistInfoIdx =
            expertIdx * moeDistributeCombineSetupInfo_->epWorldSize + moeDistributeCombineSetupInfo_->epRankId;
        if (likely(assistInfoIdx > 0U)) {
            // 计算其他卡或专家已经发了多少token
            preCount = assistInfoForCombineLocal.GetValue(assistInfoIdx - 1U);
        }
        // 当前要发送的token数量
        curTokenNum = assistInfoForCombineLocal.GetValue(assistInfoIdx) - preCount;
        if (unlikely(curTokenNum == 0U)) {
            continue;
        }

        GM_ADDR srcAddr = expandXGM_ + static_cast<uint64_t>(preCount) * axisHExpandXTypeSize_;
        GM_ADDR dstAddr = GetWinAddrByRankId(moeDistributeCombineSetupInfo_->epRankId, expertIdx) + epDataOffsetOnWin_;

        DataCopyExtParams copyParams{1U, curTokenNum * static_cast<uint32_t>(axisHExpandXTypeSize_), 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};

        // expandX在GM上，winIn也属于GM，
        // 因此，数据需要GM -> local -> winIn
        GlobalTensor<uint8_t> selfDataSrcTensor;
        selfDataSrcTensor.SetGlobalBuffer((__gm__ uint8_t *)srcAddr);
        LocalTensor<uint8_t> expertTokenTmpU8 = expertTokenTmpQueue_.AllocTensor<uint8_t>();

        DataCopyPad(expertTokenTmpU8, selfDataSrcTensor, copyParams, padParams);
        expertTokenTmpQueue_.EnQue(expertTokenTmpU8);
        expertTokenTmpU8 = expertTokenTmpQueue_.DeQue<uint8_t>();
        AscendC::SyncFunc<AscendC::HardEvent::MTE2_MTE3>();

        GlobalTensor<uint8_t> selfDataDstTensor;
        selfDataDstTensor.SetGlobalBuffer((__gm__ uint8_t *)dstAddr);
        DataCopyPad(selfDataDstTensor, expertTokenTmpU8, copyParams);

        expertTokenTmpQueue_.FreeTensor<uint8_t>(expertTokenTmpU8);
    }

    AscendC::SyncFunc<AscendC::HardEvent::MTE3_S>(); // 数据拷贝完后才能写状态
    // 向本卡状态区写状态
    GlobalTensor<int32_t> selfStatusTensor;
    selfStatusTensor.SetGlobalBuffer(
        (__gm__ int32_t *)(GetWinStateAddrByRankId(moeDistributeCombineSetupInfo_->epRankId) + epStateOffsetOnWin_));
    selfStatusTensor.SetValue(0, 0x3f800000);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfStatusTensor);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        if (startRankId_ >= moeDistributeCombineSetupInfo_->epWorldSize) {
            // 空闲核，直接返回
            return;
        }

        // URMA 通信相关Buffer初始化
        BuffInit();
        // 首次进入通信域后初始化一次CQE
        InitCqeStatus();
        // URMA 生成WQE模板
        LocalTensor<uint8_t> templateSqeU8 = templateSqeBuf_.Get<uint8_t>();
        GenerateCommWriteWithNotifySQE(templateSqeU8);

        Communication();
    }
}
} // namespace MoeDistributeCombineSetupImpl

#endif // MOE_DISTRIBUTE_COMBINE_SETUP_ARCH35_H
