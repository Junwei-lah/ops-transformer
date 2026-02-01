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
 * \file fp_matmul_allto_all_tiling_base.cpp
 * \brief
 */
#include "fp_matmul_allto_all_tiling_base.h"
#include "op_mc2.h"
#include "mc2_log.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace MC2Tiling {

/**
 * @brief 当前非量化过程的准入条件
 *
 * @return true
 */
bool FpMatmulAllToAllTilingBase::IsCapable()
{
    QuantMode mode = MatmulAlltoAllTilingUtil::GetQuantMode(context_, opName_);
    if (mode == QuantMode::NON_QUANT) {
        OP_LOGI(opName_, "Start with FpMatmulAllToAll tiling.");
        return true;
    }
    OP_LOGI(opName_, "Skip FpMatmulAllToAll tiling when not NON_QUANT.");
    return false;
}

/**
 * @brief 校验输入信息是否合规:attr,Dtype,shape等，使用通用校验util中的check方法
 *
 * @return ge::graphStatus
 */
ge::graphStatus FpMatmulAllToAllTilingBase::CheckOpInputInfo()
{
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckAttrsInfo(context_, opName_, MATMUL_ALLTOALL_INDEX_SCHEMA) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check Attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckTensorFormat(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check format failed."), return ge::GRAPH_FAILED);              
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckNonQuantTensorDataType(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check Dtype failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckShapeInfo(context_, opName_, MATMUL_ALLTOALL_INDEX_SCHEMA) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(Check2DMatrixMulShapes(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check shape failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 根据输入设置tiling参数
 *
 * @return ge::graphStatus
 */
ge::graphStatus FpMatmulAllToAllTilingBase::InitTilingContextParameters()
{
    GE_ASSERT_GRAPH_SUCCESS(
        MatmulAlltoAllTilingUtil::SetAttrsInfo(context_, opName_, contextInfo, MATMUL_ALLTOALL_INDEX_SCHEMA));
    GE_ASSERT_GRAPH_SUCCESS(MatmulAlltoAllTilingUtil::SetDataTypeInfo(context_, opName_, contextInfo));
    GE_ASSERT_GRAPH_SUCCESS(MatmulAlltoAllTilingUtil::SetShapeInfo(context_, contextInfo));
    contextInfo.quantMode = QuantMode::NON_QUANT; // 在isCapable判断过，直接赋值即可
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hccl参数；进行通算切分, 获取mm tiling等
 *
 * @return ge::graphStatus
 */
ge::graphStatus FpMatmulAllToAllTilingBase::DoOpTiling()
{
    // 输入参数的校验:Attrs,Dtype,Shape等
    GE_ASSERT_GRAPH_SUCCESS(CheckOpInputInfo());
    // 参数校验通过后赋值给全局上下文变量
    GE_ASSERT_GRAPH_SUCCESS(InitTilingContextParameters());
    // 进行通算切分
    GE_ASSERT_GRAPH_SUCCESS(TileCommAndCompute());
    // 调用非量化Matmul的tiling方法进行切分
    GE_ASSERT_GRAPH_SUCCESS(DoMMTiling());
    // hccl的tiling参数赋值处理
    GE_ASSERT_GRAPH_SUCCESS(SetHcclTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 进行通算切分之后单个块的MM TILING
 *
 * @return ge::graphStatus
 */
ge::graphStatus FpMatmulAllToAllTilingBase::DoMMTiling()
{
    // 非空校验已在GetPlatformInfo校验过
    fe::PlatFormInfos *platformInfo = context_->GetPlatformInfo();
    if (mc2_matmul_v3_advanced::InitCompileInfo(platformInfo, &compileInfo_) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Fail to Init CompileInfo!");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatForm = platform_ascendc::PlatformAscendC(platformInfo);

    std::vector<int32_t> priorities;
    GE_ASSERT_GRAPH_SUCCESS(mc2tiling::NewGetMatmulV3PriorityPolicy(socVersion_, priorities, opName_));

    Mc2MMRegisterCfg registerCfg{"Mc2MatMulV3", socVersion_, priorities};

    mc2tiling::NewUpdateMatmulV3Args(mmV3Args_, contextInfo.args_, opName_);

    // tile tiling, 对于matmulAlltoAll,mvalue就是切块大小
    mmV3Args_.mValue = inferredInfo.tileM;
    Mc2MatmulHelper::Mc2MatmulTilingCfg tileTilingCfg(reinterpret_cast<const void *>(&compileInfo_),
                                                      reinterpret_cast<const void *>(&mmV3Args_));
    GE_ASSERT_GRAPH_SUCCESS(DoMatmulV3Tiling(tileTilingCfg, registerCfg, localTilingData_.mc2MmV3TileTilingData));

    if (inferredInfo.tailM > 0) {
        //  tail  tiling
        mmV3Args_.mValue = inferredInfo.tailM;
        Mc2MatmulHelper::Mc2MatmulTilingCfg tailTilingCfg(reinterpret_cast<const void *>(&compileInfo_),
                                                          reinterpret_cast<const void *>(&mmV3Args_));
        GE_ASSERT_GRAPH_SUCCESS(DoMatmulV3Tiling(tailTilingCfg, registerCfg, localTilingData_.mc2MmV3TailTilingData));
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 调用MM的tiling
 *
 * @param tilingCfg MM的tiling的编译与参数信息
 * @param registerCfg 实际MM的注册信息
 * @param tilingData 对应首块或尾块的tilingData
 * @return ge::graphStatus
 */
ge::graphStatus FpMatmulAllToAllTilingBase::DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg &tilingCfg,
                                                             Mc2MMRegisterCfg &registerCfg,
                                                             Mc2MatMulV3TilingData &tilingData)
{
    tilingCfg.SetRankDim(contextInfo.args_.rankDim);
    tilingCfg.SetMatMulV3TilingData(tilingData);
    if (Mc2MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "DoMatmulV3Tiling failed.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hccl的config,进行hccl对应的通信任务设置
 *
 * @return ge::graphStatus
 */
ge::graphStatus FpMatmulAllToAllTilingBase::SetHcclTiling()
{
    OP_TILING_CHECK(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.args_.geCType) ==
                        mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "Cannot find HcclDataType according to ge datatype = %d.",
                                                   static_cast<int32_t>(contextInfo.args_.geCType)),
                    return ge::GRAPH_FAILED;);

    Mc2CcTilingConfigBuilder allToAllBuilder =
        Mc2CcTilingConfigBuilder::create(contextInfo.group, mc2tiling::AicpuComType::HCCL_CMD_ALLTOALL,
                                         Mc2CcTilingConfigBuilder::AlgConfigType::ALL_TO_ALL);
    //reducetype接口附带的数据类型优先于调用通信接口传入的数据类型，因此这里需要设置
    AscendC::Mc2CcTilingConfig allToAllTilingConfig = allToAllBuilder.withCommEngine(mc2tiling::A5_CCU_ENGINE).
        withReduceType(opName_, AscendC::HcclReduceOp::HCCL_REDUCE_SUM, contextInfo.args_.geCType, contextInfo.args_.geCType).build();
    if (!allToAllBuilder.isSuccess()) {
        OP_LOGE(opName_, "Build hccl tiling config failed: %s", allToAllBuilder.errorMsg().c_str());
        return ge::GRAPH_FAILED;
    }
    allToAllTilingConfig.GetTiling(localTilingData_.mc2InitTiling);
    allToAllTilingConfig.GetTiling(localTilingData_.mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 打印matmul tiling的信息,注：当前蓝区冒烟找不到mc2_log.h的对应方法，暂时自己实现
 *
 * @param opName
 * @param tiling
 */
void FpMatmulAllToAllTilingBase::PrintMMV3TilingData(const std::string &opName, Mc2MatMulV3TilingData &tiling)
{
    PrintTCubeTilingData(opName, tiling.tCubeTiling);
    OP_LOGD(opName, " tiling.mTailCnt %d", tiling.mTailCnt);
    OP_LOGD(opName, " tiling.nTailCnt %d", tiling.nTailCnt);
    OP_LOGD(opName, " tiling.kTailCnt %d", tiling.kTailCnt);
    OP_LOGD(opName, " tiling.isHf32 %d", tiling.isHf32);
    OP_LOGD(opName, " tiling.mBaseTailSpiltCnt %d", tiling.mBaseTailSplitCnt);
    OP_LOGD(opName, " tiling.nBaseTailSpiltCnt %d", tiling.nBaseTailSplitCnt);
    OP_LOGD(opName, " tiling.mTailMain %d", tiling.mTailMain);
    OP_LOGD(opName, " tiling.nTailMain %d", tiling.nTailMain);
    OP_LOGD(opName, " tiling.aswWindowLen %d", tiling.aswWindowLen);
}

/**
 * @brief 打印tilingInfo信息
 *
 * @param opName
 * @param tilingInfo
 */
void FpMatmulAllToAllTilingBase::PrintMatmulAlltoAllTilingInfo(const std::string &opName,
                                                               MatmulAlltoAllTilingInfo &tilingInfo)
{
    OP_LOGD(opName, "tilingInfo.rankDim: %u", tilingInfo.rankDim);
    OP_LOGD(opName, "tilingInfo.tileM: %u", tilingInfo.tileM);
    OP_LOGD(opName, "tilingInfo.tileCnt: %u", tilingInfo.tileCnt);
    OP_LOGD(opName, "tilingInfo.tailM: %u", tilingInfo.tailM);
    OP_LOGD(opName, "tilingInfo.tailCnt: %u", tilingInfo.tailCnt);
    OP_LOGD(opName, "tilingInfo.biasLen: %u", tilingInfo.biasLen);
    OP_LOGD(opName, "tilingInfo.rankM: %u", tilingInfo.rankM);
    OP_LOGD(opName, "tilingInfo.rankN: %u", tilingInfo.rankN);
    OP_LOGD(opName, "tilingInfo.rankK: %u", tilingInfo.rankK);
    OP_LOGD(opName, "tilingInfo.mmResultLen: %u", tilingInfo.mmResultLen);
    OP_LOGD(opName, "tilingInfo.permuteLen: %u", tilingInfo.permuteLen);
    OP_LOGD(opName, "tilingInfo.biasLen: %u", tilingInfo.biasLen);
    OP_LOGD(opName, "tilingInfo.aicCoreNum: %u", tilingInfo.aicCoreNum);
    OP_LOGD(opName, "tilingInfo.hcclDataType: %u", tilingInfo.hcclDataType);
}

/**
 * @brief 打印传递给kernel的tilingData
 *
 * @param outTilingData tilingData参数
 */
void FpMatmulAllToAllTilingBase::PrintMatmulAlltoAllTilingData(MatmulAlltoAllTilingData &outTilingData)
{
    PrintMatmulAlltoAllTilingInfo(opName_, outTilingData.matmulAlltoAllTilingInfo);
    PrintMMV3TilingData(opName_, outTilingData.mc2MmV3TileTilingData);
    if (outTilingData.matmulAlltoAllTilingInfo.tailCnt == 0) {
        return;
    }
    OP_LOGD(opName_, "Matmulalltoall has tail");
    PrintMMV3TilingData(opName_, outTilingData.mc2MmV3TailTilingData);
}

/**
 * @brief 获取对应的tilingKey
 * 使用QUANT_MODE来区分tilingKey,此处的QUANT_MODE指的是x1,x2的QUANT模式组合，以x1为pertoken量化(K)，x2为perchannel量化(C)
 * 为例子，K-C量化就代表一种组合
 *
 * @return uint64_t tilingKey结果
 */
uint64_t FpMatmulAllToAllTilingBase::GetTilingKey() const
{
    // 按照量化组合模式，是否转置，bias数据类型进行展开
    bool x2TransposeFlag = contextInfo.args_.isBTrans ? true : false;
    // 0代表数据类型和x一致(FP16 OR BF16)，1代表FP32
    uint32_t biasDType = DTYPE_BIAS_SAME_WITH_X;
    if (contextInfo.args_.geBiasType != contextInfo.args_.geAType) {
        biasDType = DTYPE_BIAS_FP32;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(NON_QUANT_MODE, x2TransposeFlag, biasDType);
    OP_LOGD(opName_, "QUANTMODE,X2TRANSPOSE,DTYPEBIAS: [%d,%d,%d], TilingKey is [%lu].", NON_QUANT_MODE,
            x2TransposeFlag, biasDType, tilingKey);
    return tilingKey;
}

/**
 * @brief 保存tiling数据到context
 *
 * @return ge::graphStatus
 */
ge::graphStatus FpMatmulAllToAllTilingBase::PostTiling()
{
    SetTilingInfo(localTilingData_.matmulAlltoAllTilingInfo);
    MatmulAlltoAllTilingData *outTilingData = context_->GetTilingData<MatmulAlltoAllTilingData>();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "Failed to get tiling data from context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tilingBufCap < sizeof(localTilingData_)),
                    OP_LOGE(opName_, "TilingBuffer capacity too small, capacity = %zu, need = %zu.", tilingBufCap,
                            sizeof(localTilingData_)),
                    return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(outTilingData, tilingBufCap, &localTilingData_, sizeof(localTilingData_));
    if (ret != EOK) {
        OP_LOGE(opName_, "MatmulAlltoAll postTiling: memcpy_s tiling data failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.", sizeof(MatmulAlltoAllTilingData),
            context_->GetRawTilingData()->GetCapacity());

    context_->GetRawTilingData()->SetDataSize(sizeof(MatmulAlltoAllTilingData));
    context_->SetBlockDim(contextInfo.args_.aicCoreNum);
    PrintMatmulAlltoAllTilingData(*outTilingData);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tilingInfo结构体
 *
 * @param tilingInfo 目标结构体
 */
void FpMatmulAllToAllTilingBase::SetTilingInfo(MatmulAlltoAllTilingInfo &tilingInfo) const
{
    // 基本字段拷贝
    tilingInfo.tileM = inferredInfo.tileM;
    tilingInfo.tileCnt = inferredInfo.tileCnt;
    tilingInfo.tailM = inferredInfo.tailM;
    tilingInfo.tailCnt = inferredInfo.tailCnt;
    tilingInfo.rankM = contextInfo.args_.mValue;
    tilingInfo.rankN = contextInfo.args_.nValue;
    tilingInfo.rankK = contextInfo.args_.kValue;
    tilingInfo.mmResultLen = inferredInfo.mmResultLen;
    tilingInfo.permuteLen = inferredInfo.permuteLen;
    tilingInfo.biasLen = inferredInfo.biasLen;
    tilingInfo.aicCoreNum = contextInfo.args_.aicCoreNum;
    tilingInfo.rankDim = contextInfo.args_.rankDim;
    tilingInfo.hcclDataType =
        (static_cast<uint64_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.args_.geAType))); // hccl数据类型
}

/**
 * @brief 构造函数，创建一个FpMatmulAllToAllTilingBase对象
 *
 * @param context
 */
FpMatmulAllToAllTilingBase::FpMatmulAllToAllTilingBase(gert::TilingContext *context) : MatmulAllToAllTilingBase(context)
{
}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAlltoAll, FpMatmulAllToAllTilingBase,
                                         static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND950), 0);
} // namespace MC2Tiling
