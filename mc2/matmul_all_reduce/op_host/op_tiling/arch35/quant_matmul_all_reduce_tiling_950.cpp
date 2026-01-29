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
 * \file quant_matmul_all_reduce_tiling_950.cc
 * \brief
 */
#ifndef _QUANT_MATMUL_ALL_REDUCE_TILING_950_CC_
#define _QUANT_MATMUL_ALL_REDUCE_TILING_950_CC_
#include "quant_matmul_all_reduce_tiling_950.h"
#include "op_mc2.h"
#include "mc2_log.h"
#include "util/math_util.h"
#include "mc2/matmul_all_reduce/op_kernel/matmul_all_reduce_apt_tiling_key.h"

using namespace Mc2Log;
using namespace Mc2Tiling;
namespace optiling {
constexpr uint64_t HCOMM_CNT = 2;
constexpr uint64_t INT8_WORKSPACE_CNT = 3;
constexpr uint64_t PERTILE_FP8_WORKSPACE_CNT = 3;
constexpr uint64_t PERTILE_FP32_WORKSPACE_CNT = 2;
constexpr uint64_t GROUP_M_OFFSET = 32;
constexpr uint64_t GROUP_N_OFFSET = 16;
constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
constexpr uint64_t GROUP_MAX_BIT_SIZE = 0xFFFFFFFFFFFF;
constexpr uint64_t PERTILE_TILELEN = 128;
constexpr uint64_t DOUBLE_BUFFER = 2;
constexpr uint64_t QUANT_MODE_FP8 = 2;
constexpr uint32_t ALIGN_DATA_SIZE = 32;

static const std::initializer_list<std::tuple<int, int, int>> MXFP_GROUPSIZE_SUPPORT_LIST = {
    std::make_tuple(0, 0, 32), std::make_tuple(1, 1, 32)};
static const std::initializer_list<std::tuple<int, int, int>> PERBLOCK_GROUPSIZE_SUPPORT_LIST = {
    std::make_tuple(128, 128, 128)};

namespace {
const gert::Shape defaultShape = gert::Shape();
gert::StorageShape defaultStorageShape = gert::StorageShape();
} // namespace
bool QuantMatmulAllReduceTilingA5::IsCapable()
{
    if (isA8W8_ || (scenario_ == AllReduceScenario::FP8HIF8) || (scenario_ == AllReduceScenario::MXFP4) ||
        (scenario_ == AllReduceScenario::MXFP8)) {
        OP_LOGI(opName_, "Start with quant tiling.");
        return true;
    }
    OP_LOGI(opName_, "Skip quant tiling as dtype not support.");
    return false;
}

void QuantMatmulAllReduceTilingA5::SetMc2Hcomm()
{
    OP_TILING_CHECK(
        mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType) == mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
        VECTOR_INNER_ERR_REPORT_TILING(
            opName_, "cannot find HcclDataType according to ge datatype = %d.", static_cast<int32_t>(args_.geCType)),
        return );
    quantMatmulAllReduceTilingData_.version = mc2tiling::COMM_VERSION3; // 新版本
    if (MutableRCSTilingData().isInputCommQuantScale == 1) {
        quantMatmulAllReduceTilingData_.hcommCfg.opType = (
            static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_REDUCE_SCATTER));
        quantMatmulAllReduceTilingData_.hcommCfg.srcDataType = (
            static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, ge::DataType::DT_INT8)));
        quantMatmulAllReduceTilingData_.hcommCfg.dstDataType = (
            static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, ge::DataType::DT_FLOAT)));
        quantMatmulAllReduceTilingData_.hcommInt8Cfg.opType = (
            static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER));
        quantMatmulAllReduceTilingData_.hcommInt8Cfg.srcDataType = (
            static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, ge::DataType::DT_INT8)));
        quantMatmulAllReduceTilingData_.hcommInt8Cfg.dstDataType = (
            static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, ge::DataType::DT_INT8)));
        quantMatmulAllReduceTilingData_.hcommCnt = HCOMM_CNT;
    } else if (MutableRCSTilingData().isInputCommQuantScale == QUANT_MODE_FP8) {
        quantMatmulAllReduceTilingData_.hcommCfg.opType = (
            static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLTOALL));
        quantMatmulAllReduceTilingData_.hcommCfg.srcDataType = (
            static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType)));
        quantMatmulAllReduceTilingData_.hcommCfg.dstDataType = (
            static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType)));
        quantMatmulAllReduceTilingData_.hcommInt8Cfg.opType = (
            static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER));
        quantMatmulAllReduceTilingData_.hcommInt8Cfg.srcDataType = (
            static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType)));
        quantMatmulAllReduceTilingData_.hcommInt8Cfg.dstDataType = (
            static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType)));
        quantMatmulAllReduceTilingData_.hcommCnt = HCOMM_CNT;
    } else {
        quantMatmulAllReduceTilingData_.hcommCfg.opType = (
            static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLREDUCE));
        quantMatmulAllReduceTilingData_.hcommCfg.srcDataType = (
            static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType)));
        quantMatmulAllReduceTilingData_.hcommCfg.dstDataType = (
            static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType)));
        quantMatmulAllReduceTilingData_.hcommCnt = 1;
    }
}

ge::graphStatus QuantMatmulAllReduceTilingA5::DoOpTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(CheckA8W8());
    GE_ASSERT_GRAPH_SUCCESS(CheckInput());
    DoRCSTiling();
    DoSplitMTiling();
    GE_ASSERT_GRAPH_SUCCESS(DoQuantTiling());
    if (MutableRCSTilingData().isInputCommQuantScale == 1) {
        isCommInt8Enable_ = true;
    }
    SetMc2Hcomm();
    DoAllReduceTiling(true);
    if (MutableRCSTilingData().isInputCommQuantScale == QUANT_MODE_FP8) {
        isCommFp8Enable_ = true;
        GE_ASSERT_GRAPH_SUCCESS(GetDynamicQuantTempBuffSize());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::GetDynamicQuantTempBuffSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    OP_TILING_CHECK(ascendcPlatform.GetCoreNumAiv() == 0,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "GetCoreNumAiv Failed! Invalid AivNum."),
                    return ge::GRAPH_FAILED);
    auto aivNum = ascendcPlatform.GetCoreNumAiv() > 0 ? ascendcPlatform.GetCoreNumAiv() : 1;
    int64_t procRowsRaw =
        MutableTCubeTileTilingData().M < aivNum ? 1 : MutableTCubeTileTilingData().M / aivNum;
    uint64_t ubSize = static_cast<uint64_t>(aicoreParams_.ubSize);
    uint64_t fp8Size = 1;
    uint64_t outSize = args_.geCType == ge::DataType::DT_FLOAT ? sizeof(float) : 2;
    uint64_t ubDenomQuant =
        3 * PERTILE_TILELEN * sizeof(float) + PERTILE_TILELEN * fp8Size + sizeof(float) + sizeof(uint8_t);
    ubDenomQuant *= DOUBLE_BUFFER;
    int64_t procRows = ((ubSize / ubDenomQuant) < procRowsRaw) ? (ubSize / ubDenomQuant) : procRowsRaw;
    int64_t procRowTileCnt = (ubSize / ubDenomQuant) / procRows;
    OP_LOGD(opName_, "Quant: ubSize=%ld, ubDenomQuant=%ld, procRowsRaw=%ld, procRows=%ld, procRowTileCnt=%ld.",
        ubSize, ubDenomQuant, procRowsRaw, procRows, procRowTileCnt);
    std::vector<int64_t> srcShapeVec = {procRowTileCnt, 1};
    std::vector<int64_t> dstShapeVec = {procRowTileCnt, PERTILE_TILELEN};
    ge::Shape srcShape(srcShapeVec);
    ge::Shape dstShape(dstShapeVec);
    uint32_t maxValBroadCast{0};
    uint32_t minValBroadCast{0};
    AscendC::GetBroadCastMaxMinTmpSize(ascendcPlatform, srcShape, dstShape, sizeof(float), false, maxValBroadCast,
                                       minValBroadCast); // Quant 广播Scale计算量化结果时需要的额外空间
    uint32_t minValBroadCastDequant{0};
    uint64_t ubDenomDequant =
        PERTILE_TILELEN * fp8Size + sizeof(float) + PERTILE_TILELEN * outSize + 2 * PERTILE_TILELEN * sizeof(float);
    ubDenomDequant *= DOUBLE_BUFFER;
    procRows = ((ubSize / ubDenomDequant) < procRowsRaw) ? (ubSize / ubDenomDequant) : procRowsRaw;
    procRowTileCnt =  (ubSize / ubDenomDequant) / procRows;
    OP_LOGD(opName_, "Dequant: ubSize=%ld, ubDenomDequant=%ld, procRowsRaw=%ld, procRows=%ld, procRowTileCnt=%ld.",
        ubSize, ubDenomDequant, procRowsRaw, procRows, procRowTileCnt);
    std::vector<int64_t> srcShapeDequantVec = {procRowTileCnt, 1};
    std::vector<int64_t> dstShapeDequantVec = {procRowTileCnt, PERTILE_TILELEN};
    ge::Shape srcShapeDequant(srcShapeDequantVec);
    ge::Shape dstShapeDequant(dstShapeDequantVec);
    AscendC::GetBroadCastMaxMinTmpSize(ascendcPlatform, srcShapeDequant, dstShapeDequant, sizeof(float), false,
                                       maxValBroadCast, minValBroadCastDequant); // Dequant 广播Scale计算反量化
    uint32_t tempBuffSize = std::max(minValBroadCast, minValBroadCastDequant);
    tempBuffSize = Ops::Base::CeilDiv(tempBuffSize, ALIGN_DATA_SIZE) * ALIGN_DATA_SIZE;
    MutableRCSTilingData().dynamicQuantTempBuffSize = tempBuffSize;
    return ge::GRAPH_SUCCESS;
}

uint64_t QuantMatmulAllReduceTilingA5::GetTilingKey() const
{
    uint8_t commDtype = COMMDTPYE_DEFAULT;
    if (isCommInt8Enable_ == true) {	
        commDtype = COMMDTPYE_INT8; // 适配int8 通信;
    } else if (isCommFp8Enable_ == true) {
        commDtype = COMMDTPYE_FP8; // 适配fp8 通信;
    }
    bool scenarioIsMXFP8 = (scenario_ == AllReduceScenario::MXFP8); // 区分MXFP8 和 FP8HIF8场景
    const uint64_t tilingKey = GET_TPL_TILING_KEY(  \
        MMTYPE_QUANT_MM,                            \
        quantTPlparam_.transB,                      \
        false,                                      \
        SET_NOT_USE_FP_MM_TILING,                   \
        quantTPlparam_.kernelType,                  \
        commDtype,                                  \
        scenarioIsMXFP8,                            \
        SET_NOT_USE_WEIGHT_QUANT_MM_TILING);
    OP_LOGD(opName_, "transB, kernelType,"                              \
            "commDtype, scenarioIsMXFP8 is:[%d,%u,%u,%d].",             \
            quantTPlparam_.transB, quantTPlparam_.kernelType,           \
            commDtype, scenarioIsMXFP8);
    OP_LOGD(opName_, "Mc2MatmulAllReduce: quant_TilingKey=%lu.", tilingKey);
    return tilingKey;
}

void QuantMatmulAllReduceTilingA5::PrintExtendMatmulTiling(bool isTail)
{
    auto& tiling = quantMatmulAllReduceTilingData_.tilematmulTiling;
    if (isTail) {
        tiling = quantMatmulAllReduceTilingData_.tailmatmulTiling;
    }

    OP_LOGD(opName_, "QuantBmmV3Params.batchA=%u.", tiling.params.batchA);
    OP_LOGD(opName_, "QuantBmmV3Params.batchB=%u.", tiling.params.batchB);
    OP_LOGD(opName_, "QuantBmmV3Params.batchC=%u.", tiling.params.batchC);
    OP_LOGD(opName_, "QuantBmmV3Params.batchA1=%u.", tiling.params.batchA1);
    OP_LOGD(opName_, "QuantBmmV3Params.batchA2=%u.", tiling.params.batchA2);
    OP_LOGD(opName_, "QuantBmmV3Params.batchA3=%u.", tiling.params.batchA3);
    OP_LOGD(opName_, "QuantBmmV3Params.batchA4=%u.", tiling.params.batchA4);
    OP_LOGD(opName_, "QuantBmmV3Params.batchB1=%u.", tiling.params.batchB1);
    OP_LOGD(opName_, "QuantBmmV3Params.batchB2=%u.", tiling.params.batchB2);
    OP_LOGD(opName_, "QuantBmmV3Params.batchB3=%u.", tiling.params.batchB3);
    OP_LOGD(opName_, "QuantBmmV3Params.batchB4=%u.", tiling.params.batchB4);
    OP_LOGD(opName_, "QuantBmmV3Params.batchC1=%u.", tiling.params.batchC1);
    OP_LOGD(opName_, "QuantBmmV3Params.batchC2=%u.", tiling.params.batchC2);
    OP_LOGD(opName_, "QuantBmmV3Params.batchC3=%u.", tiling.params.batchC3);
    OP_LOGD(opName_, "QuantBmmV3Params.batchC4=%u.", tiling.params.batchC4);
    OP_LOGD(opName_, "QuantBmmV3Params.singleCoreBatch=%u.", tiling.params.singleCoreBatch);
    OP_LOGD(opName_, "QuantBmmV3Params.isPerTensor=%u.", tiling.params.isPerTensor);
    OP_LOGD(opName_, "QuantBmmV3Params.isPertoken=%u.", tiling.params.isPertoken);
    OP_LOGD(opName_, "QuantBmmV3Params.isDoubleScale=%u.", tiling.params.isDoubleScale);
    OP_LOGD(opName_, "QuantBmmV3Params.biasThreeDim=%u.", tiling.params.biasThreeDim);
    OP_LOGD(opName_, "QuantBmmV3Params.ubCalcM=%u.", tiling.params.ubCalcM);
    OP_LOGD(opName_, "QuantBmmV3Params.ubCalcN=%u.", tiling.params.ubCalcN);
    OP_LOGD(opName_, "QuantBmmV3Params.needUbBuffer=%u.", tiling.params.needUbBuffer);
    OP_LOGD(opName_, "QuantBmmV3Params.realSingleCoreM=%u.", tiling.params.realSingleCoreM);
    OP_LOGD(opName_, "QuantBmmV3Params.realSingleCoreN=%u.", tiling.params.realSingleCoreN);
    OP_LOGD(opName_, "QuantBmmV3Params.biasDtype=%u.", tiling.params.biasDtype);
    OP_LOGD(opName_, "QuantBmmV3Params.ubSize=%u.", tiling.params.ubSize);
    OP_LOGD(opName_, "QuantBmmV3Params.isMClash=%u.", tiling.params.isMClash);
    OP_LOGD(opName_, "QuantBmmV3Params.isNClash=%u.", tiling.params.isNClash);
    OP_LOGD(opName_, "QuantBmmV3Params.groupSizeM=%u.", tiling.params.groupSizeM);
    OP_LOGD(opName_, "QuantBmmV3Params.groupSizeK=%u.", tiling.params.groupSizeK);
    OP_LOGD(opName_, "QuantBmmV3Params.groupSizeN=%u.", tiling.params.groupSizeN);

    OP_LOGD(opName_, "TileL2cacheTiling.mTileCntL2=%u.", tiling.tileL2cacheTiling.mTileCntL2);
    OP_LOGD(opName_, "TileL2cacheTiling.nTileCntL2=%u.", tiling.tileL2cacheTiling.nTileCntL2);
    OP_LOGD(opName_, "TileL2cacheTiling.mTileBlock=%u.", tiling.tileL2cacheTiling.mTileBlock);
    OP_LOGD(opName_, "TileL2cacheTiling.nTileBlock=%u.", tiling.tileL2cacheTiling.nTileBlock);
    OP_LOGD(opName_, "TileL2cacheTiling.calOrder=%u.", tiling.tileL2cacheTiling.calOrder);
    OP_LOGD(opName_, "TileL2cacheTiling.isBasicTiling=%u.", tiling.tileL2cacheTiling.isBasicTiling);

    OP_LOGD(opName_, "AdaptiveSlidingWin.mTailTile=%u.", tiling.adaptiveSlidingWin.mTailTile);
    OP_LOGD(opName_, "AdaptiveSlidingWin.nTailTile=%u.", tiling.adaptiveSlidingWin.nTailTile);
}

ge::graphStatus QuantMatmulAllReduceTilingA5::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
    uint64_t commInt8WorkSpace = 0UL;
    uint64_t commFp32WorkSpace = 0UL;
    uint64_t gmcFloat = static_cast<uint64_t>(MutableRCSTilingData().rankM) *
                        static_cast<uint64_t>(MutableRCSTilingData().rankN) *
                        static_cast<uint64_t>(args_.outputDtypeSize);
    bool isFp8 = MutableRCSTilingData().isInputCommQuantScale == QUANT_MODE_FP8;
    if (MutableRCSTilingData().isInputCommQuantScale == 1 || isFp8) {
        uint64_t padTileM = MutableTCubeTileTilingData().M;
        uint64_t padTailM = MutableTCubeTailTilingData().M;
        if (padTileM % args_.rankDim != 0) {
            padTileM += args_.rankDim - (padTileM % args_.rankDim); // args_.rankDim :1/2/4/8 不会为0
        }
        uint64_t tempPadTileM = padTileM * MutableTCubeTileTilingData().N * sizeof(int8_t);
        if (padTailM % args_.rankDim != 0) {
            padTailM += args_.rankDim - (padTailM % args_.rankDim); // args_.rankDim :1/2/4/8 不会为0
        }
        uint64_t tempPadTailM = padTailM * MutableTCubeTailTilingData().N * sizeof(int8_t);
        commFp32WorkSpace = (tempPadTileM * MutableRCSTilingData().tileCnt +
                             tempPadTailM * MutableRCSTilingData().tailCnt) * sizeof(float);
        if (isFp8) {
            uint64_t tileN = MutableTCubeTileTilingData().N;
            uint64_t tailN = MutableTCubeTailTilingData().N;
            tileN += Ops::Base::CeilDiv(tileN, PERTILE_TILELEN);
            tailN += Ops::Base::CeilDiv(tailN, PERTILE_TILELEN);
            tempPadTileM = padTileM * tileN;
            tempPadTailM = padTailM * tailN;
        }
        commInt8WorkSpace =
            tempPadTileM * MutableRCSTilingData().tileCnt + tempPadTailM * MutableRCSTilingData().tailCnt;
        commInt8WorkSpace *= isFp8 ? sizeof(float) : sizeof(int8_t);
        OP_LOGI(opName_, "Set commInt8WorkSpace size=%lu, commFp32WorkSpace size=%lu to context.", commInt8WorkSpace,
                commFp32WorkSpace);
    }
    uint64_t commWorkSpace = myWorkSpaceSize_ - libApiWorkSpaceSize_;
    MutableRCSTilingData().commWorkSpaceSize = (commWorkSpace); // myWorkSpaceSize_去除系统空间后剩余大小
    MutableRCSTilingData().commInt8WorkSpace = (commInt8WorkSpace); // int8 通信用于存放reduceScatter输入 workspace 的开销
    if (isFp8) {
        // 存放Matmul输出+quant输出+alltoall输出+(dequant+reduce+quant)混合输出+allgather输出+dequant输出
        myWorkSpaceSize_ = myWorkSpaceSize_ + PERTILE_FP8_WORKSPACE_CNT * commInt8WorkSpace +
                           commInt8WorkSpace / args_.rankDim + PERTILE_FP32_WORKSPACE_CNT * commFp32WorkSpace;
    } else {
        myWorkSpaceSize_ = myWorkSpaceSize_ + gmcFloat + INT8_WORKSPACE_CNT * commInt8WorkSpace + commFp32WorkSpace;
    }
    OP_LOGI(opName_, "Set max workspace size=%lu to context.", myWorkSpaceSize_);
    workspaces[0] = myWorkSpaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::PostTiling()
{
    OP_LOGD(
        opName_, "Final tiling data size=%zu and context capacity size=%zu.",
        sizeof(QuantMatmulAllReduceTilingDataA5), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(sizeof(QuantMatmulAllReduceTilingDataA5));

    OP_TILING_CHECK(
        (sizeof(QuantMatmulAllReduceTilingDataA5) % sizeof(uint64_t)) != 0,
        VECTOR_INNER_ERR_REPORT_TILING(
            opName_, "Tiling data size=%zu not aligned to 8.", sizeof(QuantMatmulAllReduceTilingDataA5)),
        return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void *>(&quantMatmulAllReduceTilingData_), sizeof(QuantMatmulAllReduceTilingDataA5));
    if (ret != EOK){
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }

    PrintTilingData();
    context_->SetBlockDim(args_.aicCoreNum);
    // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要设置避免出现网络挂死
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

Mc2Tiling::Mc2Msg& QuantMatmulAllReduceTilingA5::MutableMc2MsgData()
{
    return quantMatmulAllReduceTilingData_.msg;
}
Mc2Tiling::RCSTiling& QuantMatmulAllReduceTilingA5::MutableRCSTilingData()
{
    return quantMatmulAllReduceTilingData_.param;
}
::TCubeTiling& QuantMatmulAllReduceTilingA5::MutableTCubeTileTilingData()
{
    return quantMatmulAllReduceTilingData_.tilematmulTiling.matmulTiling;
}
::TCubeTiling& QuantMatmulAllReduceTilingA5::MutableTCubeTailTilingData()
{
    return quantMatmulAllReduceTilingData_.tailmatmulTiling.matmulTiling;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::DoQuantTiling()
{
    args_.mValue = tileMValue_;
    QuantTilingTransferHelperA5 mmTile(*this, quantMatmulAllReduceTilingData_.tilematmulTiling);
    if (args_.enableSplitK) {
        OP_LOGD(opName_, "Enable SplitK Tiling.");
        GE_ASSERT_GRAPH_SUCCESS(mmTile.DoTiling());
        quantTPlparam_ = mmTile.GetQuantMMAllReduceTPLParam(mmTile.GetKernelType());
        OP_LOGD(opName_, "quantmmAllReduce get kernelType: %d", mmTile.GetKernelType());
        return ge::GRAPH_SUCCESS;
    } else {
        GE_ASSERT_GRAPH_SUCCESS(mmTile.DoTiling());;
        if (MutableRCSTilingData().tailCnt == 0) {
            quantTPlparam_ = mmTile.GetQuantMMAllReduceTPLParam(mmTile.GetKernelType());
            OP_LOGD(opName_, "quantmmAllReduce get kernelType: %d", mmTile.GetKernelType());
            return ge::GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        QuantTilingTransferHelperA5 mmTail(*this, quantMatmulAllReduceTilingData_.tailmatmulTiling);
        GE_ASSERT_GRAPH_SUCCESS(mmTail.DoTiling());

        quantTPlparam_ = mmTail.GetQuantMMAllReduceTPLParam(mmTail.GetKernelType());
        OP_LOGD(opName_, "quantmmAllReduce get kernelType: %d", mmTail.GetKernelType());
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckAxisSize()
{
    const uint64_t m = MatmulAllReduceTilingBase::GetMValue();
    OP_TILING_CHECK(
        m > static_cast<uint64_t>(INT32_MAX),
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(), "The size of m-axis=%lu exceeds the upper limit=%d.", m, INT32_MAX),
        return ge::GRAPH_FAILED);
    const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
    OP_TILING_CHECK(
        k > static_cast<uint64_t>(UINT16_MAX),
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(), "The size of k-axis=%lu exceeds the upper limit=%d.", k, UINT16_MAX),
        return ge::GRAPH_FAILED);
    // A2有ND2NZ指令的长度约束65535，为了兼容A2，x2最后一维限制65535
    uint64_t x2FirstDim = mmrCtxInfo_.x2_shape->GetStorageShape().GetDim(0);
    uint64_t x2LastDim = mmrCtxInfo_.x2_shape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        (x2FirstDim > static_cast<uint64_t>(INT32_MAX)) || (x2LastDim > static_cast<uint64_t>(UINT16_MAX)),
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(), "The size of x2 first-axis=%lu exceeds the upper limit=%d or last-axis=%lu"
            " exceeds the upper limit=%d.", x2FirstDim, INT32_MAX, x2LastDim, UINT16_MAX),
        return ge::GRAPH_FAILED);

    return CheckQuantEmptyTensor();
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckA8W8ScenarioScaleType()
{
    if (scenario_ == AllReduceScenario::A8W8) {
        auto dequantScaleType = mmrCtxInfo_.dequant_scale->GetDataType();
        auto yType = mmrCtxInfo_.y->GetDataType();
        auto pertokenScaleShape = mmrCtxInfo_.pertoken_scale_shape;
        OP_LOGD(opName_, "DequantScaleType=%d, yType=%d.", dequantScaleType, yType);
        // 1. y = bf16 时，dequantScale = bf16
        // 2. y = fp16 且 protoken 不存在时， dequantScale = int64、uint64
        // 3. y = fp16 且 protoken 存在时，dequantScale = fp32
        if (yType == ge::DT_BF16) {
            OP_TILING_CHECK(
                dequantScaleType != ge::DT_BF16,
                VECTOR_INNER_ERR_REPORT_TILING(
                    context_->GetNodeName(),
                    "In the dequant scenario, when output type is bf16, "
                    "type of dequantScale should be bf16."),
                return ge::GRAPH_FAILED);
        } else if (pertokenScaleShape == nullptr) {
            OP_TILING_CHECK(
                !((dequantScaleType == ge::DT_UINT64) || (dequantScaleType == ge::DT_INT64)),
                VECTOR_INNER_ERR_REPORT_TILING(
                    context_->GetNodeName(),
                    "In the dequant scenario, when output type is fp16, "
                    "type of dequantScale should be uint64 or int64 without pertoken."),
                return ge::GRAPH_FAILED);
        } else {
            OP_TILING_CHECK(
                dequantScaleType != ge::DT_FLOAT,
                VECTOR_INNER_ERR_REPORT_TILING(
                    context_->GetNodeName(),
                    "In the dequant scenario, when output type is fp16, "
                    "type of dequantScale should be bf16 with pertoken."),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckDequantScaleType()
{
    // dequantScale数据类型范围
    auto dequantScaleType = mmrCtxInfo_.dequant_scale->GetDataType();
    OP_TILING_CHECK(
        !((dequantScaleType == ge::DT_UINT64) || (dequantScaleType == ge::DT_BF16) ||
          (dequantScaleType == ge::DT_INT64) || (dequantScaleType == ge::DT_FLOAT) ||
          (dequantScaleType == ge::DT_FLOAT8_E8M0)),
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(),
            "In the dequant scenario, type of dequantScale should be uint64, int64, bf16 or float32, "
            "get type=%s.",
            ge::TypeUtils::DataTypeToSerialString(dequantScaleType).c_str()),
        return ge::GRAPH_FAILED);
    if (scenario_ == AllReduceScenario::FP8HIF8) {
        if (dequantScaleType == ge::DT_FLOAT) {
            // fp8/hif8场景下，pertoken+pertensor/perchannel场景必须提供两个scale，类型为float
            OP_TILING_CHECK(
                mmrCtxInfo_.pertoken_scale == nullptr,
                VECTOR_INNER_ERR_REPORT_TILING(
                    context_->GetNodeName(),
                    "In the dequant scenario, when dequantScale type is float, "
                    "per_token_scale should not be null."),
                return ge::GRAPH_FAILED);
            auto perTokenScaleType = mmrCtxInfo_.pertoken_scale->GetDataType();
            OP_TILING_CHECK(
                perTokenScaleType != ge::DT_FLOAT,
                VECTOR_INNER_ERR_REPORT_TILING(
                    context_->GetNodeName(),
                    "In the dequant scenario, when dequantScale type is float, "
                    "type of per_token_scale should be float, get type=%s.",
                    ge::TypeUtils::DataTypeToSerialString(perTokenScaleType).c_str()),
                return ge::GRAPH_FAILED);
        } else {
            // fp8/hif8场景下，单路pertensor场景必须提供1个scale，类型为uint64
            OP_TILING_CHECK(
                dequantScaleType != ge::DT_UINT64,
                VECTOR_INNER_ERR_REPORT_TILING(
                    context_->GetNodeName(),
                    "In the dequant scenario, "
                    "type of dequantScale should be uint64_t, get type=%s.",
                    ge::TypeUtils::DataTypeToSerialString(dequantScaleType).c_str()),
                return ge::GRAPH_FAILED);
        }
    }

    OP_TILING_CHECK(
        CheckMXFPScenarioScaleType() != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "Check scale type failed."), return ge::GRAPH_FAILED);
    return CheckA8W8ScenarioScaleType();
} // namespace optiling

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckMXFPScenarioScaleType()
{
    auto dequantScaleType = mmrCtxInfo_.dequant_scale->GetDataType();
    if ((scenario_ == AllReduceScenario::MXFP4) || (scenario_ == AllReduceScenario::MXFP8)) {
        OP_TILING_CHECK(
            mmrCtxInfo_.pertoken_scale == nullptr,
            VECTOR_INNER_ERR_REPORT_TILING(
                context_->GetNodeName(), "In the dequant MXfp4/MXfp8 scenario, per_token_scale should not be null."),
            return ge::GRAPH_FAILED);
        auto perTokenScaleType = mmrCtxInfo_.pertoken_scale->GetDataType();
        OP_TILING_CHECK(
            (dequantScaleType != ge::DT_FLOAT8_E8M0) || (perTokenScaleType != ge::DT_FLOAT8_E8M0),
            VECTOR_INNER_ERR_REPORT_TILING(
                context_->GetNodeName(),
                "In the dequant MXfp4/MXfp8 scenario, "
                "type of dequantScale should be float8_e8m0, "
                "get type of dequantScale=%s, type of pertokenScale=%s.",
                ge::TypeUtils::DataTypeToSerialString(dequantScaleType).c_str(),
                ge::TypeUtils::DataTypeToSerialString(perTokenScaleType).c_str()),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckBias()
{
    // bias数据类型为int32, fp8/hif8,MX场景下为float
    if (mmrCtxInfo_.bias_shape != nullptr) {
        auto biasType = mmrCtxInfo_.bias->GetDataType();
        if (isA8W8_) {
            OP_TILING_CHECK(
                biasType != ge::DT_INT32,
                VECTOR_INNER_ERR_REPORT_TILING(
                    context_->GetNodeName(),
                    "In the dequant scenario, type of bias should be int32, "
                    "but got type of bias=%d.",
                    biasType),
                return ge::GRAPH_FAILED);
        } else if (
            (scenario_ == AllReduceScenario::FP8HIF8) || (scenario_ == AllReduceScenario::MXFP4) ||
            (scenario_ == AllReduceScenario::MXFP8)) {
            OP_TILING_CHECK(
                biasType != ge::DT_FLOAT,
                VECTOR_INNER_ERR_REPORT_TILING(
                    context_->GetNodeName(),
                    "In the dequant scenario, type of bias should be float, "
                    "but got type of bias=%d.",
                    biasType),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckCommQuantScale()
{
    // comm_quant_scale不为空时校验数据类型
    if ((mmrCtxInfo_.comm_quant_scale_1_shape != nullptr) && (mmrCtxInfo_.comm_quant_scale_2_shape != nullptr)) {
        auto commQuantScaleType1 = mmrCtxInfo_.comm_quant_scale_1->GetDataType();
        auto commQuantScaleType2 = mmrCtxInfo_.comm_quant_scale_2->GetDataType();
        auto cType = mmrCtxInfo_.y->GetDataType();
        OP_TILING_CHECK(
            ((commQuantScaleType1 != cType) || (commQuantScaleType2 != cType)),
            VECTOR_INNER_ERR_REPORT_TILING(
                context_->GetNodeName(),
                "The type of comm_quant_scale_1 Type=%d or comm_quant_scale_2 Type=%d should be same to cType=%d.",
                static_cast<int32_t>(commQuantScaleType1), static_cast<int32_t>(commQuantScaleType2),
                static_cast<int32_t>(cType)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

bool CheckGroupSizeVaild(
    std::tuple<int, int, int> groupSizeMNK, std::initializer_list<std::tuple<int, int, int>> supportGroupSizeList)
{
    return std::find(supportGroupSizeList.begin(), supportGroupSizeList.end(), groupSizeMNK) !=
           supportGroupSizeList.end();
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckQuantGroupSize()
{
    OP_TILING_CHECK(
        mmrCtxInfo_.groupSizePtr == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "the groupSize is nullptr"),
        return false);
    auto groupSizePtr = mmrCtxInfo_.groupSizePtr;
    uint64_t groupSizeK = static_cast<uint64_t>(*groupSizePtr) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    std::tuple<uint64_t, uint64_t, uint64_t> groupSizeMNK(groupSizeM, groupSizeN, groupSizeK);
    if (isPerBlock_) {
        OP_TILING_CHECK(
            !(CheckGroupSizeVaild(groupSizeMNK, PERBLOCK_GROUPSIZE_SUPPORT_LIST)),
            CUBE_INNER_ERR_REPORT(
                opName_,
                "groupSizeM, groupSizeN and groupSizeK should be 128 in perblock scene,"
                " but actual is [groupSizeM = %lu, groupSizeN = %lu, groupSizeK = %lu]",
                groupSizeM, groupSizeN, groupSizeK),
            return ge::GRAPH_FAILED);
    } else if ((scenario_ == AllReduceScenario::MXFP4) || (scenario_ == AllReduceScenario::MXFP8)) {
        OP_TILING_CHECK(
            !(CheckGroupSizeVaild(groupSizeMNK, MXFP_GROUPSIZE_SUPPORT_LIST)),
            CUBE_INNER_ERR_REPORT(
                opName_,
                "groupSizeM, groupSizeN and groupSizeK should be surported in mxfp scene,"
                " but actual is [groupSizeM = %lu, groupSizeN = %lu, groupSizeK = %lu]",
                groupSizeM, groupSizeN, groupSizeK),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckX1X2()
{
    // x2 shape 为 2 维
    size_t x2DimNum = mmrCtxInfo_.x2_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        x2DimNum != DIM_NUM_TWO && x2DimNum != DIM_NUM_FOUR,
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(),
            "In the dequant scenario, Expect x2 dim to be 2 or 4, "
            "but got x2 dim=%lu.",
            x2DimNum),
        return ge::GRAPH_FAILED);
    // x1，x2数据类型相同
    auto x1Type = mmrCtxInfo_.x1->GetDataType();
    auto x2Type = mmrCtxInfo_.x2->GetDataType();
    OP_TILING_CHECK(
        (isA8W8_ && (x1Type != x2Type)),
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(),
            "In the dequant scenario, type of x1 and x2 should be same, "
            "but got type of x1=%d, type of x2=%d.",
            x1Type, x2Type),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((scenario_ == AllReduceScenario::FP8HIF8) && (x1Type != x2Type) &&
         ((x1Type == ge::DT_HIFLOAT8) || (x2Type == ge::DT_HIFLOAT8))),
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(),
            "In the dequant scenario, when type is HIFLOAT8, "
            "type of x1 and x2 should be same, "
            "but got type of x1=%d, type of x2=%d.",
            x1Type, x2Type),
        return ge::GRAPH_FAILED);
    if ((scenario_ == AllReduceScenario::MXFP4) || (scenario_ == AllReduceScenario::MXFP8)) {
        OP_TILING_CHECK(args_.isBTrans == false,
                        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(),
                                                        "In the dequant MXfp4/MXfp8 scenario, x2 must be transposed."),
                        return ge::GRAPH_FAILED);
    }
    if (scenario_ == AllReduceScenario::MXFP4) {
        uint64_t x1K = GetKValue();
        OP_TILING_CHECK(
            Ops::Base::CeilDiv(x1K, MX_FP4_GROUP_SIZE) % 2 != 0,
            VECTOR_INNER_ERR_REPORT_TILING(
                context_->GetNodeName(), "In the dequant MXfp4 scenario, k=%lu ceil dev 32 must be even.", x1K),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckInput()
{
    GE_ASSERT_GRAPH_SUCCESS(MatmulAllReduceTilingBase::CheckInput());
    OP_TILING_CHECK(
        CheckX1X2() != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "Check input_X failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        CheckBias() != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "Check bias failed."), return ge::GRAPH_FAILED);
    // dequantScale数据类型范围
    GE_ASSERT_GRAPH_SUCCESS(CheckDequantScaleType());
    // comm_quant_scale不为空时校验数据类型
    OP_TILING_CHECK(
        CheckCommQuantScale() != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "Check commQuantScale failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckQuantGroupSize() != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "Check groupSize failed."), return ge::GRAPH_FAILED);

    return CheckAxisSize();
}

QuantMatmulAllReduceTilingA5::QuantMatmulAllReduceTilingA5(gert::TilingContext* context)
    : MatmulAllReduceTilingBase(context), quantMatmulAllReduceTilingData_(quantMatmulAllReduceTilingDataSelf_)
{}

// 使用外部传入的tilingdata和ctxinfo
QuantMatmulAllReduceTilingA5::QuantMatmulAllReduceTilingA5(
    gert::TilingContext* context, MMRCtxInfo* mmrCtxInfo, QuantMatmulAllReduceTilingDataA5* out)
    : MatmulAllReduceTilingBase(context, mmrCtxInfo), quantMatmulAllReduceTilingData_(*out)
{}

const gert::Shape QuantTilingTransferHelperA5::GetX1Shape(const size_t index)
{
    (void)index;
    // 判断原始的m是否128对齐,对于perblock三维场景m轴非128对齐按照3维传值，和matmul保持一致
    bool isNotBatchOne = (tilingProcesser_.args_.batchValue != 1ULL);
    bool is128Aligned = ((tilingProcesser_.args_.orgMValue / tilingProcesser_.args_.batchValue) & 127ULL) == 0;
    // perblock场景下，若x1为（b,m,k)时，量化参数shape为[b,ceilDiv(m,128),ceilDiv(k,128)],仅当ceilDiv(m,128)为整数时，才能保证 b*ceilDiv(m,128) = ceilDiv(b*m,128)
    // mxfp场景下，若x1为（b,m,k)时，量化参数shape为[m,ceilDiv(k,64),2],若batch合轴，量化参数shape为[b*m,ceilDiv(k,64),2]，原shape无法满足计算
    if ((tilingProcesser_.isPerBlock_ && isNotBatchOne && !is128Aligned) ||
        ((tilingProcesser_.scenario_ == AllReduceScenario::MXFP8) && isNotBatchOne) || 
        ((tilingProcesser_.scenario_ == AllReduceScenario::MXFP4) && isNotBatchOne)) {
        return gert::Shape(
            {static_cast<int64_t>(tilingProcesser_.args_.batchValue),
            static_cast<int64_t>(tilingProcesser_.args_.mValue) / static_cast<int64_t>(tilingProcesser_.args_.batchValue),
            static_cast<int64_t>(tilingProcesser_.args_.kValue)});
    } else {
        return gert::Shape(
            {static_cast<int64_t>(tilingProcesser_.args_.mValue), static_cast<int64_t>(tilingProcesser_.args_.kValue)});
    }
}
const gert::Shape QuantTilingTransferHelperA5::GetX2Shape(const size_t index)
{
    (void)index;
    if (tilingProcesser_.args_.isBTrans) {
        return gert::Shape(
            {static_cast<int64_t>(tilingProcesser_.args_.nValue), static_cast<int64_t>(tilingProcesser_.args_.kValue)});
    }
    return gert::Shape(
        {static_cast<int64_t>(tilingProcesser_.args_.kValue), static_cast<int64_t>(tilingProcesser_.args_.nValue)});
}

const gert::Shape& QuantTilingTransferHelperA5::GetScaleShape(const size_t index)
{
    (void)index;
    OP_TILING_CHECK(
        tilingProcesser_.mmrCtxInfo_.dequant_scale_shape == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(
            tilingProcesser_.opName_, "%s is quant, but has no quant shape.", inputParams_.opName),
        return defaultShape);
    return tilingProcesser_.mmrCtxInfo_.dequant_scale_shape->GetStorageShape();
}

const gert::StorageShape* QuantTilingTransferHelperA5::GetOffsetShape(const size_t index)
{
    (void)index;
    return nullptr;
}

const gert::StorageShape* QuantTilingTransferHelperA5::GetPertokenShape(const size_t index)
{
    (void)index;
    if (tilingProcesser_.mmrCtxInfo_.pertoken_scale_shape == nullptr) {
        return nullptr;
    }

    if (tilingProcesser_.isPerBlock_) {
        return tilingProcesser_.mmrCtxInfo_.pertoken_scale_shape;
    }

    defaultStorageShape = gert::StorageShape(
        {static_cast<int64_t>(tilingProcesser_.args_.mValue)}, {static_cast<int64_t>(tilingProcesser_.args_.mValue)});
    return &defaultStorageShape;
}

const gert::StorageShape* QuantTilingTransferHelperA5::GetBiasShape(const size_t index)
{
    (void)index;
    return tilingProcesser_.mmrCtxInfo_.bias_shape;
}

ge::graphStatus QuantTilingTransferHelperA5::GetShapeAttrsInfo()
{
    OP_LOGI(tilingProcesser_.opName_, "Start assemble input params for matmul tiling.");
    auto&& tilingArgs = tilingProcesser_.args_;
    inputParams_.opName = tilingProcesser_.opName_;
    inputParams_.transA = tilingArgs.isATrans;
    inputParams_.transB = tilingArgs.isBTrans;
    inputParams_.hasBias = tilingArgs.isBias;
    inputParams_.libApiWorkSpaceSize = tilingProcesser_.libApiWorkSpaceSize_;
    inputParams_.aDtype = tilingArgs.geAType;
    inputParams_.bDtype = tilingArgs.geBType;
    inputParams_.cDtype = tilingArgs.geCType;
    inputParams_.outDtype = static_cast<int64_t>(tilingArgs.geCType);
    inputParams_.biasDtype = tilingArgs.isBias ? tilingArgs.geBiasType : ge::DT_INT32;
    inputParams_.scaleDtype = tilingProcesser_.mmrCtxInfo_.dequant_scale->GetDataType();
    if (tilingProcesser_.mmrCtxInfo_.pertoken_scale != nullptr) {
        inputParams_.perTokenScaleDtype = tilingProcesser_.mmrCtxInfo_.pertoken_scale->GetDataType();
    }

    if ((tilingProcesser_.scenario_ == AllReduceScenario::MXFP4) ||
        (tilingProcesser_.scenario_ == AllReduceScenario::MXFP8)) {
        inputParams_.groupSizeK = MX_GROUP_SIZE;
    }

    if (tilingProcesser_.isPerBlock_) {
        inputParams_.groupSizeM = SUPPORTED_BLOCK_SIZE;
        inputParams_.groupSizeN = SUPPORTED_BLOCK_SIZE;
        inputParams_.groupSizeK = SUPPORTED_BLOCK_SIZE;
    }
    // optiling::PlatformInfo::GetInstance().intrinsic_fix_pipe_l0c2out = tilingProcesser_.supportL0c2Out_;
    GE_ASSERT_TRUE(AnalyzeInputs());
    inputParams_.isPerTensor = (tilingProcesser_.quantType_ == Mc2QuantType::PER_TENSOR);
    PrintTilingInputParam(inputParams_);
    return ge::GRAPH_SUCCESS;
}

QuantMMAllReduceTPLParam QuantTilingTransferHelperA5::GetQuantMMAllReduceTPLParam(const uint64_t kernelType)
{
    QuantMMAllReduceTPLParam param;
    param.transB = inputParams_.transB;
    param.kernelType = kernelType;
    return param;
}

ge::graphStatus QuantTilingTransferHelperA5::PostTiling()
{
    tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
    OP_LOGI(tilingProcesser_.opName_, "Set mm workspace size=%lu to mc2.", tilingProcesser_.myWorkSpaceSize_);
    return ge::GRAPH_SUCCESS;
}

void QuantTilingTransferHelperA5::PrintTilingInputParam(Mc2QuantBatchMatmulInfo quantBatchMatmulInfo)
{
    OP_LOGD(
        tilingProcesser_.opName_, "The transA_=%d, transB_=%d, hasBias_=%d.", quantBatchMatmulInfo.transA,
        quantBatchMatmulInfo.transB, quantBatchMatmulInfo.hasBias);
    OP_LOGD(
        tilingProcesser_.opName_, "The mSize_=%ld, kSize_=%ld, nSize_=%ld, libApiWorkSpaceSize=%u.",
        quantBatchMatmulInfo.mSize, quantBatchMatmulInfo.kSize, quantBatchMatmulInfo.nSize,
        quantBatchMatmulInfo.libApiWorkSpaceSize);
    OP_LOGD(
        tilingProcesser_.opName_, "The aDtype_=%d, bDtype_=%d, cDtype_=%d, biasDtype_=%d, outDtype=%ld.",
        static_cast<int32_t>(quantBatchMatmulInfo.aDtype), static_cast<int32_t>(quantBatchMatmulInfo.bDtype),
        static_cast<int32_t>(quantBatchMatmulInfo.cDtype), static_cast<int32_t>(quantBatchMatmulInfo.biasDtype),
        quantBatchMatmulInfo.outDtype);
    OP_LOGD(
        tilingProcesser_.opName_,
        "The batchA=%lu, batchA1-A4=[%lu:%lu:%lu:%lu], "
        "batchB=%lu, batchB1-B4=[%lu:%lu:%lu:%lu], batchC=%lu, batchBias=%lu.",
        quantBatchMatmulInfo.batchA, quantBatchMatmulInfo.batchA1, quantBatchMatmulInfo.batchA2,
        quantBatchMatmulInfo.batchA3, quantBatchMatmulInfo.batchA4, quantBatchMatmulInfo.batchB,
        quantBatchMatmulInfo.batchB1, quantBatchMatmulInfo.batchB2, quantBatchMatmulInfo.batchB3,
        quantBatchMatmulInfo.batchB4, quantBatchMatmulInfo.batchC, quantBatchMatmulInfo.batchBias);
    OP_LOGD(tilingProcesser_.opName_, "Check isperTensor=%d.", static_cast<int32_t>(quantBatchMatmulInfo.isPerTensor));
}

//注册tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAllReduce,QuantMatmulAllReduceTilingA5,static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND950),0);
} // namespace optiling

#endif //_QUANT_MATMUL_ALL_REDUCE_TILING_950_CC_