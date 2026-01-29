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
 * \file matmul_all_reduce_tiling_950.cc
 * \brief
 */
#include "matmul_all_reduce_tiling_950.h"
#include "tiling/new_mc2_tiling_utils.h"
#include "op_mc2.h"
#include "mc2/matmul_all_reduce/op_kernel/matmul_all_reduce_apt_tiling_key.h"

using namespace Mc2Tiling;
namespace optiling {
bool MatmulAllReduceTilingA5::IsCapable()
{
    OP_LOGI(opName_, "Start with MatmulAllReduceTilingA5 tiling.");
    return true;
}

void MatmulAllReduceTilingA5::SetMc2Hcomm()
{
    OP_TILING_CHECK(
        mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType) == mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
        VECTOR_INNER_ERR_REPORT_TILING(
            opName_, "cannot find HcclDataType according to ge datatype = %d.", static_cast<int32_t>(args_.geCType)),
        return );
    matmulAllReduce910TilingData_.hcommCfg.opType = (
        static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLREDUCE));
    matmulAllReduce910TilingData_.hcommCfg.srcDataType = (
        static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType)));
    matmulAllReduce910TilingData_.hcommCfg.dstDataType = (
        static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType)));

    matmulAllReduce910TilingData_.version = mc2tiling::COMM_VERSION3; // 新版本
    matmulAllReduce910TilingData_.hcommCnt = 1;                       // allreduce 通信域数量为1
}

void MatmulAllReduceTilingA5::DoEmptyTensorTiling()
{
    MutableTCubeTileTilingData().M = args_.orgMValue;
    MutableTCubeTileTilingData().N = args_.orgNValue;
    MutableTCubeTileTilingData().isBias = args_.isBias;
    MutableTCubeTileTilingData().usedCoreNum = 1;
}

ge::graphStatus MatmulAllReduceTilingA5::DoOpTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(CheckA16W16());
    GE_ASSERT_GRAPH_SUCCESS(CheckInput());
    SetMc2Hcomm();
    DoRCSTiling();
    DoSplitMTiling();
    if (!isKZero_) {
        GE_ASSERT_GRAPH_SUCCESS(Do910Tiling());
    } else {
        DoEmptyTensorTiling();
    }
    DoAllReduceTiling(true);
    return ge::GRAPH_SUCCESS;
}

uint64_t MatmulAllReduceTilingA5::GetTilingKey() const
{
    if (unlikely(isKZero_)) {
        const uint64_t tilingKey = GET_TPL_TILING_KEY(  \
            MMTYPE_FP_NULL_TENSOR,                      \
            false,                                      \
            false,                                      \
            SET_NOT_USE_FP_MM_TILING,                   \
            SET_NOT_USE_QUANT_MM_TILING,                \
            SET_NOT_USE_WEIGHT_QUANT_MM_TILING);
        return tilingKey;
    }
    bool matmulWithAdd = true;
    if (!matmulAllReduce910TilingData_.param.isAdd) {
        matmulWithAdd = false;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(  \
        MMTYPE_FP_MM,                               \
        false,                                      \
        false,                                      \
        matmulWithAdd,                              \
        SET_NOT_USE_QUANT_MM_TILING,                \
        SET_NOT_USE_WEIGHT_QUANT_MM_TILING);
    OP_LOGD(opName_, "Mc2MatmulAllReduce: matmulWithAdd is:[%d].", matmulWithAdd);
    OP_LOGD(opName_, "Mc2MatmulAllReduce: TilingKey=%lu.", tilingKey);
    return tilingKey;
}

void MatmulAllReduceTilingA5::PrintExtendMatmulTiling(bool isTail)
{
    auto& tiling = matmulAllReduce910TilingData_.mC2Mmv3TailTilingData;
    if (isTail) {
        tiling = matmulAllReduce910TilingData_.mC2Mmv3TailTilingData;
    }

    OP_LOGD(opName_, "Matmul tiling mTailCnt=%u", tiling.mTailCnt);
    OP_LOGD(opName_, "Matmul tiling nTailCnt=%u", tiling.nTailCnt);
    OP_LOGD(opName_, "Matmul tiling kTailCnt=%u", tiling.kTailCnt);
    OP_LOGD(opName_, "Matmul tiling mBaseTailSplitCnt=%u", tiling.mBaseTailSplitCnt);
    OP_LOGD(opName_, "Matmul tiling nBaseTailSplitCnt=%u", tiling.nBaseTailSplitCnt);
    OP_LOGD(opName_, "Matmul tiling mTailMain=%u", tiling.mTailMain);
    OP_LOGD(opName_, "Matmul tiling nTailMain=%u", tiling.nTailMain);
    OP_LOGD(opName_, "Matmul tiling isHf32=%u", tiling.isHf32);
    OP_LOGD(opName_, "Matmul tiling aswWindowLen=%u", tiling.aswWindowLen);
}

ge::graphStatus MatmulAllReduceTilingA5::GetWorkspaceSize()
{
    GE_ASSERT_GRAPH_SUCCESS(MatmulAllReduceTilingBase::GetWorkspaceSize());
    OP_LOGI(
        opName_, "Select max workspace size to context, myWorkSpaceSize_=%lu, workspaceSize_=%lu.", myWorkSpaceSize_,
        workspaceSize_);
    myWorkSpaceSize_ = std::max(myWorkSpaceSize_, workspaceSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = myWorkSpaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingA5::PostTiling()
{
    OP_LOGD(
        opName_, "Final tiling data size=%zu and context capacity size=%zu.",
        sizeof(MatmulAllReduce910TilingDataA5), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(sizeof(MatmulAllReduce910TilingDataA5));

    OP_TILING_CHECK(
        (sizeof(MatmulAllReduce910TilingDataA5) % sizeof(uint64_t)) != 0,
        VECTOR_INNER_ERR_REPORT_TILING(
            opName_, "Tiling data size=%zu not aligned to 8.", sizeof(MatmulAllReduce910TilingDataA5)),
        return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void *>(&matmulAllReduce910TilingData_), sizeof(MatmulAllReduce910TilingDataA5));
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

ge::graphStatus MatmulAllReduceTilingA5::Do910Tiling()
{
    OP_LOGD(opName_, "Start to excute DoMatmulV3Tiling!");
    // 获取芯片平台信息
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "get platform info failed"),
                    return ge::GRAPH_FAILED);
    // 获取compileInfo
    OP_TILING_CHECK(mc2_matmul_v3_advanced::InitCompileInfo(platformInfo, &compileInfo_) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "init compile info failed"), return ge::GRAPH_FAILED);

    // 根据芯片型号获取策略模板
    std::vector<int32_t> priorities;
    OP_TILING_CHECK(mc2tiling::NewGetMatmulV3PriorityPolicy(socVersion_, priorities, opName_) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "get mmv3 priority policy failed"),
                    return ge::GRAPH_FAILED);
    Mc2MMRegisterCfg registerCfg {"Mc2MatMulV3", socVersion_, priorities};
    mc2tiling::NewUpdateMatmulV3Args(mmV3Args_, args_, opName_);

    // 获取tileTiling
    mmV3Args_.mValue = tileMValue_;
    OP_LOGD(opName_, "Do Mc2MatmulV3 tile tiling!");
    Mc2MatmulHelper::Mc2MatmulTilingCfg tileTilingCfg(reinterpret_cast<const void*>(&compileInfo_),
                                                      reinterpret_cast<const void*>(&mmV3Args_));
    GE_ASSERT_GRAPH_SUCCESS(DoMatmulV3Tiling(tileTilingCfg, registerCfg, MutableMC2MmV3TileTilingData()));
    if (tailMValue_ != 0UL) {
        mmV3Args_.mValue = tailMValue_;
        OP_LOGD(opName_, "Do Mc2MatmulV3 tail tiling!");
        Mc2MatmulHelper::Mc2MatmulTilingCfg tailTilingCfg(reinterpret_cast<const void*>(&compileInfo_),
                                                          reinterpret_cast<const void*>(&mmV3Args_));
        GE_ASSERT_GRAPH_SUCCESS(DoMatmulV3Tiling(tileTilingCfg, registerCfg, MutableMC2MmV3TailTilingData()));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingA5::DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg& tilingCfg,
    Mc2MMRegisterCfg& registerCfg, Mc2MatMulV3TilingData& tilingData)
{
    tilingCfg.SetRankDim(args_.rankDim);
    tilingCfg.SetMatMulV3TilingData(tilingData);
    if (Mc2MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Failed to do MatmulV3Tiling.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

Mc2Tiling::Mc2Msg& MatmulAllReduceTilingA5::MutableMc2MsgData()
{
    return matmulAllReduce910TilingData_.msg;
}

Mc2Tiling::RCSTiling& MatmulAllReduceTilingA5::MutableRCSTilingData()
{
    return matmulAllReduce910TilingData_.param;
}

ge::graphStatus MatmulAllReduceTilingA5::CheckAxisSize()
{
    const uint64_t m = MatmulAllReduceTilingBase::GetMValue();
    OP_TILING_CHECK(
        m > static_cast<uint64_t>(INT32_MAX),
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "The size of m-axis=%lu exceeds the upper limit.", m),
        return ge::GRAPH_FAILED);
    const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
    OP_TILING_CHECK(
        k > static_cast<uint64_t>(INT32_MAX),
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "The size of k-axis=%lu exceeds the upper limit.", k),
        return ge::GRAPH_FAILED);
    const uint64_t n = MatmulAllReduceTilingBase::GetNValue();
    OP_TILING_CHECK(
        n > static_cast<uint64_t>(INT32_MAX),
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "The size of n-axis=%lu exceeds the upper limit.", n),
        return ge::GRAPH_FAILED);

    return CheckEmptyTensor();
}

ge::graphStatus MatmulAllReduceTilingA5::CheckX1X2()
{
    // x2 shape 为 2 维
    size_t x2DimNum = mmrCtxInfo_.x2_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        x2DimNum != DIM_NUM_TWO,
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(),
            "In the not quant scenario, Expect x2 dim to be 2, "
            "but got x2 dim=%lu.",
            x2DimNum),
        return ge::GRAPH_FAILED);
    auto x1Type = mmrCtxInfo_.x1->GetDataType();
    //  x1 为fp16 或者bf16
    OP_TILING_CHECK(
        !((x1Type == ge::DT_FLOAT16) || (x1Type == ge::DT_BF16)),
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(),
            "In the not quant scenario, type of x1 should be fp16 or bf16, "
            "but got type of x1 type=%d.",
            x1Type),
        return ge::GRAPH_FAILED);
    // x1，x2数据类型相同
    auto x2Type = mmrCtxInfo_.x2->GetDataType();
    OP_TILING_CHECK(
        x1Type != x2Type,
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(),
            "In the not quant scenario, type of x1 and x2 should be same, "
            "but got type of x1=%d, type of x2=%d.",
            x1Type, x2Type),
        return ge::GRAPH_FAILED);
    // x1,bias数据类型相同
    if (mmrCtxInfo_.bias_shape != nullptr) {
        auto biasType = mmrCtxInfo_.bias->GetDataType();
        OP_TILING_CHECK(
            x1Type != biasType,
            VECTOR_INNER_ERR_REPORT_TILING(
                context_->GetNodeName(),
                "In the not quant scenario, type of x1 and bias should be same, "
                "but got type of x1=%d, type of bias=%d.",
                x1Type, biasType),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingA5::CheckInput()
{
    GE_ASSERT_GRAPH_SUCCESS(MatmulAllReduceTilingBase::CheckInput());
    OP_TILING_CHECK(
        CheckX1X2() != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "Check input_X failed."), return ge::GRAPH_FAILED);
    auto outputDimNum = mmrCtxInfo_.y_shape->GetStorageShape().GetDimNum();
    if (mmrCtxInfo_.x3_shape != nullptr) {
        auto x3DimNum = mmrCtxInfo_.x3_shape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            outputDimNum != x3DimNum,
            VECTOR_INNER_ERR_REPORT_TILING(
                context_->GetNodeName(),
                "In the not quant scenario, shape of x3 and output should be same, "
                "but got dim of x3=%lu, dim of output=%lu.",
                outputDimNum, x3DimNum),
            return ge::GRAPH_FAILED);
        for (size_t i = 0U; i < outputDimNum; i++) {
            auto outputDimValue = mmrCtxInfo_.y_shape->GetStorageShape().GetDim(i);
            auto x3DimValue = mmrCtxInfo_.x3_shape->GetStorageShape().GetDim(i);
            OP_TILING_CHECK(
                outputDimValue != x3DimValue,
                VECTOR_INNER_ERR_REPORT_TILING(
                    context_->GetNodeName(),
                    "In the not quant scenario, shape of x3 and output should be same, "
                    "but when dim=%lu, value of x3=%ld, value of output=%ld.",
                    i, outputDimValue, x3DimValue),
                return ge::GRAPH_FAILED);
        }
    }

    return CheckAxisSize();
}

MatmulAllReduceTilingA5::MatmulAllReduceTilingA5(gert::TilingContext* context)
    : MatmulAllReduceTilingBase(context), matmulAllReduce910TilingData_(matmulAllReduce910TilingDataSelf_)
{}

MatmulAllReduceTilingA5::MatmulAllReduceTilingA5(
    gert::TilingContext* context, MMRCtxInfo* mmrCtxInfo, MatmulAllReduce910TilingDataA5* out)
    : MatmulAllReduceTilingBase(context, mmrCtxInfo), matmulAllReduce910TilingData_(*out)
{}

//注册tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAllReduce,MatmulAllReduceTilingA5,static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND950),2);
} // namespace optiling
