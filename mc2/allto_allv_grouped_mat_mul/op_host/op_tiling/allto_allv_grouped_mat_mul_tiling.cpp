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
 * \file allto_allv_grouped_mat_mul_tiling.cc
 * \brief
 */

#include "allto_allv_grouped_mat_mul_tiling.h"
#include <string>
#include <numeric>
#include <climits>
#include "tiling/matmul_formulaic_tiling.h"
#include "tiling/hccl_formulaic_tiling.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "tiling/mc2_calc_num_blocks.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/mc2_tiling_utils.h"
#include "../../op_kernel/allto_allv_grouped_mat_mul_tiling.h"
#include "allto_allv_grouped_mat_mul_tiling_base.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "context_util.h"
#include "../../op_kernel/allto_allv_grouped_mat_mul_tiling_key.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;
namespace optiling {
constexpr uint32_t GMM_X_INDEX = 0U;
constexpr uint32_t OUTPUT_Y_INDEX = 0U;
constexpr uint32_t GMM_WEIGHT_INDEX = 1U;

constexpr uint32_t SEND_COUNTS_TENSOR_INDEX = 2U;
constexpr uint32_t RECV_COUNTS_TENSOR_INDEX = 3U;
constexpr uint32_t MM_X_INDEX = 4U;
constexpr uint32_t MM_WEIGHT_INDEX = 5U;
constexpr uint32_t OUTPUT_GMM_Y_INDEX = 0U;
constexpr uint32_t OUTPUT_MM_Y_INDEX = 1U;
constexpr uint32_t OUTPUT_PERMUTE_OUT_INDEX = 2U;

constexpr uint32_t DIM_TWO = 2;
constexpr uint32_t DIM_ONE = 1;
constexpr uint32_t DIM_THREE = 3;

constexpr uint32_t NUM_ZERO = 0;
constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;
constexpr uint32_t NUM_EIGHT = 8;
constexpr uint32_t NUM_SIXTEEN = 16;
constexpr uint32_t NUM_THIRTYTWO = 32;
constexpr uint32_t NUM_SIXTYFOUR = 64;
constexpr uint32_t MAX_EXPERT_NUM = 256;
constexpr uint32_t MAX_BSK = 52428800;
constexpr uint32_t MAX_SHAPE_SIZE = 65536;
constexpr uint32_t MAX_SHARED_H_SHAPE_SIZE = 12288;

constexpr uint32_t ATTR_GROUP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_SEND_COUNTS_INDEX = 2;
constexpr uint32_t ATTR_RECV_COUNTS_INDEX = 3;
constexpr uint32_t ATTR_TRANS_GMM_WEIGHT_INDEX = 4;
constexpr uint32_t ATTR_TRANS_MM_WEIGHT_INDEX = 5;
constexpr uint32_t ATTR_PERMUTE_OUT_FLAG_INDEX = 6;

constexpr int64_t BEST_L1_PARTA = 256 * 1024;
constexpr int64_t BEST_L1_PARTB = 128 * 1024;
constexpr int64_t BEST_BASE_N = 256;
constexpr uint32_t UB_DIVIDE_NUM = 2;
constexpr uint32_t UB_CALSIZE_PER_BLOCK = 16 * 1024;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint64_t DOUBLE_BUFFER_STEPKA_STEPKB = 2;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t MAX_TURN_NUM = 24;
constexpr int32_t MAX_BASE_K = 128;
constexpr uint64_t COMM_TILE = 8; // 每卡数据分配几次计算

const char* A_INNER_DEBUG = "AlltoAllvGroupedMatMul Tiling";

static inline uint32_t SixteenAlign(uint32_t a, bool up = false)
{
    if (up) {
        a += 15; // 15: 16 bytes up-align
    }
    return a & ~15; // ~15: 16 bytes down-align
}

static inline uint32_t Ceil(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

static uint64_t GMMGetSizePlatForm(
    const platform_ascendc::CoreMemType memType, platform_ascendc::PlatformAscendC ascendcPlatform)
{
    uint64_t size = 0;
    ascendcPlatform.GetCoreMemSize(memType, size);
    return size;
}

struct PlatFormMemSize {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;

    explicit PlatFormMemSize(platform_ascendc::PlatformAscendC ascendcPlatform)
        : ubSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::UB, ascendcPlatform)),
          l1Size(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L1, ascendcPlatform)),
          l0CSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_C, ascendcPlatform)),
          l0ASize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_A, ascendcPlatform)),
          l0BSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_B, ascendcPlatform))
    {}
};

// 定义参数结构体
struct MMTilingParams {
    int32_t curMaxM;
    int32_t curMaxK;
    int32_t curMaxN;
    int32_t* curBaseM;
    int32_t* curBaseK;
    int32_t* curBaseN;
};

struct SetMMTilingParams {
    matmul_tiling::DataType matmulDtype;
    int32_t curMaxM;
    int32_t curMaxK;
    int32_t curMaxN;
    int32_t curBaseM;
    int32_t curBaseN;
    int32_t type;
};

static void PrintTilingDataGMM(::TCubeTiling msg)
{
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.usedCoreNum %d.", msg.usedCoreNum);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.M %d.", msg.M);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.N %d.", msg.N);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.Ka %d.", msg.Ka);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.Kb %d.", msg.Kb);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.singleCoreM %d.", msg.singleCoreM);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.singleCoreN %d.", msg.singleCoreN);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.singleCoreK %d.", msg.singleCoreK);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.baseM %d.", msg.baseM);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.baseN %d.", msg.baseN);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.baseK %d.", msg.baseK);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.stepKa %d.", msg.stepKa);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.stepKb %d.", msg.stepKb);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.stepM %d.", msg.stepM);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.stepN %d.", msg.stepN);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.isBias %d.", msg.isBias);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.transLength %d.", msg.transLength);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.iterateOrder %d.", msg.iterateOrder);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.dbL0A %d.", msg.dbL0A);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.dbL0B %d.", msg.dbL0B);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.dbL0C %d.", msg.dbL0C);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.shareMode %d.", msg.shareMode);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.shareL1Size %d.", msg.shareL1Size);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.shareL0CSize %d.", msg.shareL0CSize);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.shareUbSize %d.", msg.shareUbSize);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.batchM %d.", msg.batchM);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.batchN %d.", msg.batchN);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.singleBatchM %d.", msg.singleBatchM);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.singleBatchN %d.", msg.singleBatchN);
}

static void PrintTilingDataMM(::TCubeTiling msg)
{
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.usedCoreNum %d.", msg.usedCoreNum);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.M %d.", msg.M);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.N %d.", msg.N);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.Ka %d.", msg.Ka);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.Kb %d.", msg.Kb);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.singleCoreM %d.", msg.singleCoreM);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.singleCoreN %d.", msg.singleCoreN);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.singleCoreK %d.", msg.singleCoreK);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.baseM %d.", msg.baseM);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.baseN %d.", msg.baseN);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.baseK %d.", msg.baseK);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.stepKa %d.", msg.stepKa);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.stepKb %d.", msg.stepKb);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.stepM %d.", msg.stepM);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.stepN %d.", msg.stepN);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.isBias %d.", msg.isBias);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.transLength %d.", msg.transLength);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.iterateOrder %d.", msg.iterateOrder);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.dbL0A %d.", msg.dbL0A);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.dbL0B %d.", msg.dbL0B);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.dbL0C %d.", msg.dbL0C);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.shareMode %d.", msg.shareMode);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.shareL1Size %d.", msg.shareL1Size);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.shareL0CSize %d.", msg.shareL0CSize);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.shareUbSize %d.", msg.shareUbSize);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.batchM %d.", msg.batchM);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.batchN %d.", msg.batchN);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.singleBatchM %d.", msg.singleBatchM);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.singleBatchN %d.", msg.singleBatchN);
}

static void PrintCommonTilingInfo(AlltoAllvGmmCommonTilingInfo &commonTilingInfo)
{
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.BSK %lu.", commonTilingInfo.BSK);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.BS %lu.", commonTilingInfo.BS);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.H1 %lu.", commonTilingInfo.H1);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.H2 %lu.", commonTilingInfo.H2);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.A %lu.", commonTilingInfo.A);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.N1 %lu.", commonTilingInfo.N1);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.N2 %lu.", commonTilingInfo.N2);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.epWorldSize %lu.", commonTilingInfo.epWorldSize);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.E_ep %lu.", commonTilingInfo.E_ep);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.commOut %lu.", commonTilingInfo.commOut);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.aivCoreNum %lu.", commonTilingInfo.aivCoreNum);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.aicCoreNum %lu.", commonTilingInfo.aicCoreNum);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.isGmmWeightTrans %d.", commonTilingInfo.isGmmWeightTrans);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.isMmWeightTrans %d.", commonTilingInfo.isMmWeightTrans);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.isSendCntsTensor %d.", commonTilingInfo.isSendCntsTensor);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.isRecvCntsTensor %d.", commonTilingInfo.isRecvCntsTensor);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.isPermuteOut %d.", commonTilingInfo.isPermuteOut);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.isNeedMM %d.", commonTilingInfo.isNeedMM);
}

class AlltoAllvGmmTiling
{
public:
    AlltoAllvGmmTilingData* tilingData;

    ge::graphStatus Init(gert::TilingContext* context);
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext* context);

protected:
    ge::graphStatus GetContextAttr(const gert::TilingContext* context);
    ge::graphStatus GetShapeAndFormat(const gert::TilingContext* context);
    ge::graphStatus CheckMKN(const gert::TilingContext* context);
    ge::graphStatus CheckShapeSize(const gert::TilingContext* context) const;
    ge::graphStatus CheckAttrsShapeSize(const gert::TilingContext* context) const;
    ge::graphStatus CheckAttrsShapeRelation(const gert::TilingContext* context) const;
    ge::graphStatus CheckSendRecvDataVolumn(const gert::TilingContext* context) const;
    ge::graphStatus CheckShapeRelation(const gert::TilingContext* context) const;
    ge::graphStatus CheckShapeDims(const gert::TilingContext* context);
    ge::graphStatus CheckDType(const gert::TilingContext* context) const;
    ge::graphStatus CheckMmShapeDims(const gert::TilingContext* context) const;
    ge::graphStatus SetHcclTiling(const gert::TilingContext* context) const;

    ge::graphStatus CalMMTiling(const gert::TilingContext* context, MMTilingParams& params) const;
    ge::graphStatus SetMMTiling(const gert::TilingContext* context, SetMMTilingParams& params) const;
    ge::graphStatus DoAiCoreTiling(const gert::TilingContext* context);
    uint64_t GetTilingKey(const gert::TilingContext* context) const;
    ge::graphStatus setNumBlocks(gert::TilingContext* context); 

private:
    int32_t maxM_;
    int32_t maxN_;
    int32_t maxK_;
    int32_t baseM_;
    int32_t baseN_;
    int32_t baseK_;
    uint32_t mmDataTypeSize;

    int32_t maxMForMM_;
    int32_t maxNForMM_;
    int32_t maxKForMM_;
    int32_t baseMForMM_;
    int32_t baseNForMM_;
    int32_t baseKForMM_;

    const char* epGroup_;
    uint32_t rankSize_;
    uint32_t libApiWorkSpaceSize_;
    uint64_t epWorldSize_;

    ge::DataType mmDType_ = ge::DT_UNDEFINED;
};

// 获取入参参数
ge::graphStatus AlltoAllvGmmTiling::GetContextAttr(const gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned nullptr!"), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int>(ATTR_EP_WORLD_SIZE_INDEX);
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    auto transGmmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_GMM_WEIGHT_INDEX);
    auto transMmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_MM_WEIGHT_INDEX);
    auto permuteOutFlagPtr = attrs->GetAttrPointer<bool>(ATTR_PERMUTE_OUT_FLAG_INDEX);

    OP_TILING_CHECK(groupEpPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "groupEpPtr is null!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        epWorldSizePtr == nullptr, OP_LOGE(A_INNER_DEBUG, "epWorldSizePtr is null!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        sendCountsPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "sendCountsPtr is null!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        recvCountsPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "recvCountsPtr is null!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        transGmmWeightPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "transGmmWeightPtr is null!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        transMmWeightPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "transMmWeightPtr is null!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        permuteOutFlagPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "permuteOutFlagPtr is null!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(A_INNER_DEBUG, "tilingData is null!"), return ge::GRAPH_FAILED);

    tilingData->commonTilingInfo.epWorldSize = *epWorldSizePtr;
    tilingData->commonTilingInfo.isGmmWeightTrans = *transGmmWeightPtr;
    tilingData->commonTilingInfo.isMmWeightTrans = *transMmWeightPtr;
    tilingData->commonTilingInfo.isPermuteOut = *permuteOutFlagPtr;

    const gert::StorageShape* mmXStorageShape = context->GetOptionalInputShape(MM_X_INDEX);
    const gert::StorageShape* mmWeightStorageShape = context->GetOptionalInputShape(MM_WEIGHT_INDEX);
    const gert::StorageShape* outputMmYStorageShape = context->GetOutputShape(OUTPUT_MM_Y_INDEX);
    if (!((mmXStorageShape == nullptr) && (mmWeightStorageShape == nullptr) &&
          (outputMmYStorageShape == nullptr || outputMmYStorageShape->GetStorageShape().GetDimNum() == NUM_ZERO)) &&
        !((mmXStorageShape != nullptr) && (mmWeightStorageShape != nullptr) &&
          (outputMmYStorageShape != nullptr && outputMmYStorageShape->GetStorageShape().GetDimNum() != NUM_ZERO))) {
        OP_LOGE(A_INNER_DEBUG, "mmX, mmWeight and mmY should all be nullptr or all be not nullptr!");
        return ge::GRAPH_FAILED;
    }
    tilingData->commonTilingInfo.isNeedMM = (mmXStorageShape != nullptr);

    epGroup_ = groupEpPtr;
    epWorldSize_ = *epWorldSizePtr;

    OP_LOGI(A_INNER_DEBUG, "epGroup is %s, epWorldSize is %lu.", epGroup_, epWorldSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::GetShapeAndFormat(const gert::TilingContext* context)
{
    OP_TILING_CHECK(
        (context->GetInputShape(GMM_X_INDEX) == nullptr) || (context->GetInputShape(GMM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(A_INNER_DEBUG, "GetInputShape gmmX or gmmWeight returned null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        context->GetOutputShape(OUTPUT_GMM_Y_INDEX) == nullptr,
        OP_LOGE(A_INNER_DEBUG, "GetOutputShape gmmY returned null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        context->GetInputDesc(GMM_X_INDEX) == nullptr, OP_LOGE(A_INNER_DEBUG, "GetInputDesc gmmX returned null."),
        return ge::GRAPH_FAILED);

    tilingData->commonTilingInfo.BSK = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(0);
    tilingData->commonTilingInfo.H1 = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(1);
    tilingData->commonTilingInfo.E_ep = context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(0);
    tilingData->commonTilingInfo.N1 = tilingData->commonTilingInfo.isGmmWeightTrans ?
                                          context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(1) :
                                          context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(NUM_TWO);

    tilingData->commonTilingInfo.A = context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(0);
    mmDType_ = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    mmDataTypeSize = GetSizeByDataType(mmDType_);

    maxM_ = tilingData->commonTilingInfo.A;
    maxK_ = tilingData->commonTilingInfo.H1;
    maxN_ = tilingData->commonTilingInfo.N1;
    if (tilingData->commonTilingInfo.isNeedMM) {
        tilingData->commonTilingInfo.BS = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(0);
        tilingData->commonTilingInfo.H2 = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(1);
        tilingData->commonTilingInfo.N2 =
            tilingData->commonTilingInfo.isMmWeightTrans ?
                context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(0) :
                context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(1);
        maxMForMM_ = tilingData->commonTilingInfo.BS;
        maxKForMM_ = tilingData->commonTilingInfo.H2;
        maxNForMM_ = tilingData->commonTilingInfo.N2;
    } else {
        tilingData->commonTilingInfo.BS = 0U;
        tilingData->commonTilingInfo.H2 = 0U;
        tilingData->commonTilingInfo.N2 = 0U;
        maxMForMM_ = 0U;
        maxKForMM_ = 0U;
        maxNForMM_ = 0U;
    }
    // 暂时非空拦截 aclnn侧也校验了
    OP_TILING_CHECK(
        (context->GetOptionalInputShape(SEND_COUNTS_TENSOR_INDEX) != nullptr) ||
            (context->GetOptionalInputShape(RECV_COUNTS_TENSOR_INDEX) != nullptr),
        OP_LOGE(A_INNER_DEBUG, "sendCountsTensor and recvCountsTensor should all be null!"), return ge::GRAPH_FAILED);

    tilingData->commonTilingInfo.isSendCntsTensor =
        (context->GetOptionalInputShape(SEND_COUNTS_TENSOR_INDEX) == nullptr) ? false : true;
    tilingData->commonTilingInfo.isRecvCntsTensor =
        (context->GetOptionalInputShape(RECV_COUNTS_TENSOR_INDEX) == nullptr) ? false : true;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckMKN(const gert::TilingContext* context)
{
    (void)context; // Unused
    OP_TILING_CHECK(
        mmDataTypeSize == 0,
        OP_LOGE(
            A_INNER_DEBUG, "GMM get matmul dtype[%s] size is 0.",
            TypeUtils::DataTypeToAscendString(mmDType_).GetString()),
        return ge::GRAPH_FAILED);
    uint32_t numInOneBlk = ONE_BLK_SIZE / mmDataTypeSize;
    OP_TILING_CHECK(numInOneBlk == 0, OP_LOGE(A_INNER_DEBUG, "GMM numInOneBlk cannot be 0."), return ge::GRAPH_FAILED);
    int64_t maxMKN = INT_MAX / numInOneBlk * numInOneBlk;
    OP_TILING_CHECK(
        maxM_ > maxMKN || maxN_ > maxMKN || maxK_ > maxMKN,
        OP_LOGE(A_INNER_DEBUG, "32B-aligned m, n or k axis for gmm is out of range int32!"), return ge::GRAPH_FAILED);
    if (tilingData->commonTilingInfo.isNeedMM) {
        OP_TILING_CHECK(
            maxMForMM_ > maxMKN || maxNForMM_ > maxMKN || maxKForMM_ > maxMKN,
            OP_LOGE(A_INNER_DEBUG, "32B-aligned m, n or k axis for mm is out of range int32!"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckSendRecvDataVolumn(const gert::TilingContext* context) const
{
    // 单卡之间通信数据 [2M,100M]
    uint64_t eExpert = tilingData->commonTilingInfo.E_ep;
    uint64_t epWorldSize = tilingData->commonTilingInfo.epWorldSize;
    uint64_t recvSendMin = static_cast<uint64_t>(2U * 1024U * 1024U);       // 通信量下限 2MB=2*1024*1024
    
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK((sendCountsPtr == nullptr) || (recvCountsPtr == nullptr),
        OP_LOGE(A_INNER_DEBUG, "sendCountsPtr or recvCountsPtr is null."), return ge::GRAPH_FAILED);

    const uint64_t* sendCounts = static_cast<const uint64_t*>(sendCountsPtr->GetData());
    const uint64_t* recvCounts = static_cast<const uint64_t*>(recvCountsPtr->GetData());
    uint64_t recvSum = 0U;
    uint64_t sendSum = 0U;
    uint64_t H1 = tilingData->commonTilingInfo.H1;
    uint64_t bsk = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(0);
    uint64_t a = context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(0);
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_93) {
        for (uint64_t i = 1U; i <= epWorldSize; i++) {
            recvSum = 0U;
            sendSum = 0U;
            for (uint64_t j = (i - 1U) * eExpert; j <= i * eExpert - 1U; j++) {
                OP_TILING_CHECK((sendCounts[j] < NUM_ZERO) || (sendCounts[j] > bsk),
                    OP_LOGE(A_INNER_DEBUG, "sendCounts[%lu] should be in [0, bsK[%lu]], but get %lu",j, bsk, sendCounts[j]),
                    return ge::GRAPH_FAILED);
                OP_TILING_CHECK((recvCounts[j] < NUM_ZERO) || (recvCounts[j] > a),
                    OP_LOGE(A_INNER_DEBUG, "recvCounts[%lu] should be in [0, a[%lu]], but get %lu",j, a, recvCounts[j]),
                    return ge::GRAPH_FAILED);
                recvSum += recvCounts[j] * H1 * 2U;
                sendSum += sendCounts[j] * H1 * 2U; // /sizeof(gmmX) = 2U
            }
            OP_TILING_CHECK(recvSum < recvSendMin,
                OP_LOGE(A_INNER_DEBUG,
                    "rank %lu:sum(recvCounts[%lu, %lu]) * H1 * sizeof dtype(gmmx) should be greater than or equal to 2MB,"
                    "but got %lu Byte!",
                    i - 1U, (i - 1U) * eExpert, i * eExpert - 1U, recvSum),
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK(sendSum < recvSendMin,
                OP_LOGE(A_INNER_DEBUG,
                    "rank %lu:sum(sendCounts[%lu, %lu]) * H1 * sizeof dtype(gmmx) should be greater than or equal to 2MB,"
                    "but got %lu Byte!",
                    i - 1U, (i - 1U) * eExpert, i * eExpert - 1U, sendSum),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

// 检查入参 shape size
ge::graphStatus AlltoAllvGmmTiling::CheckShapeSize(const gert::TilingContext* context) const
{
    OP_TILING_CHECK(
        (context->GetInputShape(GMM_X_INDEX) == nullptr) || (context->GetInputShape(GMM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(A_INNER_DEBUG, "GetInputShape gmmX or gmmWeight returned null."), return ge::GRAPH_FAILED);

    uint64_t BSK = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(0);
    if (BSK <= NUM_ZERO || BSK >= MAX_BSK) {
        OP_LOGE(A_INNER_DEBUG, "BSK should be in (0, 52428800), but got %lu!", BSK);
        return ge::GRAPH_FAILED;
    }
    uint64_t H1 = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        (H1 <= NUM_ZERO) || (H1 >= MAX_SHAPE_SIZE),
        OP_LOGE(A_INNER_DEBUG, "H1 should be in (0, 65536), but got %lu!", H1), return ge::GRAPH_FAILED);

    uint64_t N1 = tilingData->commonTilingInfo.isGmmWeightTrans ?
                      context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(1) :
                      context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(NUM_TWO);
    OP_TILING_CHECK(
        N1 <= NUM_ZERO || N1 >= MAX_SHAPE_SIZE, OP_LOGE(A_INNER_DEBUG, "N1 should be in (0, 65536), but got %lu!", N1),
        return ge::GRAPH_FAILED);

    if (tilingData->commonTilingInfo.isNeedMM) {
        uint64_t BS = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(0);
        if (BS <= NUM_ZERO) {
            OP_LOGE(A_INNER_DEBUG, "BS should be larger than 0, but got %lu!", BS);
            return ge::GRAPH_FAILED;
        }
        uint64_t H2 = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(1);
        if (H2 <= NUM_ZERO || H2 > MAX_SHARED_H_SHAPE_SIZE) {
            OP_LOGE(A_INNER_DEBUG, "H2 should be in (0, 12288], but got %lu!", H2);
            return ge::GRAPH_FAILED;
        }
        uint64_t N2 = tilingData->commonTilingInfo.isMmWeightTrans ?
                          context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(0) :
                          context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(1);
        if (N2 <= NUM_ZERO || N2 >= MAX_SHAPE_SIZE) {
            OP_LOGE(A_INNER_DEBUG, "N2 should be in (0, 65536), but got %lu!", N2);
            return ge::GRAPH_FAILED;
        }
        uint64_t topK = BSK / BS;
        if (topK < NUM_TWO || topK > NUM_EIGHT) {
            OP_LOGE(A_INNER_DEBUG, "topK should be in [2, 8], but got %lu!", topK);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckAttrsShapeSize(const gert::TilingContext* context) const
{
    uint64_t E_ep = tilingData->commonTilingInfo.E_ep;
    if (E_ep <= NUM_ZERO || E_ep > NUM_THIRTYTWO) {
        OP_LOGE(A_INNER_DEBUG, "E_ep should be in (0, 32], but got %lu!", E_ep);
        return ge::GRAPH_FAILED;
    }
    uint64_t epWorldSize = tilingData->commonTilingInfo.epWorldSize;
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    std::vector<int64_t> epWorldSizeOptional;
    std::string epWorldSizeNum;
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
        epWorldSizeOptional = {2, 4, 8, 16, 32, 64}; //A5限制epWorldSize为{2，4，8，16，32，64}
    } else {
        epWorldSizeOptional = {8, 16, 32, 64, 128}; //A3限制epWorldSize为{8，16，32，64, 128}
    }
    for (size_t i = 0; i < epWorldSizeOptional.size(); i++) {
        epWorldSizeNum += (std::to_string(epWorldSizeOptional[i]) + " ");
    }
    OP_TILING_CHECK(
        std::find(epWorldSizeOptional.begin(), epWorldSizeOptional.end(), epWorldSize) == epWorldSizeOptional.end(),
        OP_LOGE(A_INNER_DEBUG, "epWorldSize[%lu] should be %s!", epWorldSize, epWorldSizeNum.c_str()), return ge::GRAPH_FAILED);
    // 对sendCounts和recvCounts校验
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK((sendCountsPtr == nullptr) || (recvCountsPtr == nullptr),
        OP_LOGE(A_INNER_DEBUG, "sendCountsPtr or recvCountsPtr is null."), return ge::GRAPH_FAILED);
    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    uint64_t recvCountsSize = recvCountsPtr->GetSize();
    OP_TILING_CHECK(sendCountsSize != recvCountsSize,
        OP_LOGE(
            A_INNER_DEBUG, "The size of sendCounts(e*ep) %lu should be equal to recvCounts(e*ep) %lu !", sendCountsSize,
            recvCountsSize),
        return ge::GRAPH_FAILED);
    if (E_ep * epWorldSize != sendCountsSize) {
        OP_LOGE(A_INNER_DEBUG,
            "The first dim of gmmWeight(e, H1, N1) %lu  multi epWorldSize %lu shoubl be equal to the size of "
            "sendCounts(e*ep) %lu!", E_ep, epWorldSize, sendCountsSize);
        return ge::GRAPH_FAILED;
    }
    if ((E_ep * epWorldSize <= NUM_ZERO) || (E_ep * epWorldSize > MAX_EXPERT_NUM)) {
        OP_LOGE(
            A_INNER_DEBUG, "The size of send_counts(e*ep) and recv_counts(e*ep) should be in (0, 256], but got %lu!",
            E_ep * epWorldSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 检查sendcounts recvcounts shape 之间的关系
ge::graphStatus AlltoAllvGmmTiling::CheckAttrsShapeRelation(const gert::TilingContext* context) const
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(
        (sendCountsPtr == nullptr) || (recvCountsPtr == nullptr),
        OP_LOGE(A_INNER_DEBUG, "sendCountsPtr or recvCountsPtr is null."), return ge::GRAPH_FAILED);

    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    uint64_t recvCountsSize = recvCountsPtr->GetSize();

    errno_t ret = memcpy_s(
        &(tilingData->aicpuTiling.sendCnt), MAX_EXPERT_NUM * sizeof(int64_t), sendCountsPtr->GetData(),
        sendCountsPtr->GetSize() * sizeof(int64_t));
    if (ret != EOK) {
        OP_LOGE(A_INNER_DEBUG, "memcpy_s failed, ret = %d.", ret);
        return ge::GRAPH_FAILED;
    }
    ret = memcpy_s(
        &(tilingData->aicpuTiling.recvCnt), MAX_EXPERT_NUM * sizeof(int64_t), recvCountsPtr->GetData(),
        recvCountsPtr->GetSize() * sizeof(int64_t));
    if (ret != EOK) {
        OP_LOGE(A_INNER_DEBUG, "memcpy_s failed, ret = %d.", ret);
        return ge::GRAPH_FAILED;
    }

    const uint64_t* sendCounts = static_cast<const uint64_t*>(sendCountsPtr->GetData());
    uint64_t sendCountsSum = std::accumulate(sendCounts, sendCounts + sendCountsSize, 0ULL);
    OP_TILING_CHECK(
        sendCountsSum != tilingData->commonTilingInfo.BSK,
        OP_LOGE(
            A_INNER_DEBUG, "The sum of sendCounts %lu should be equal to BSK %lu!", sendCountsSum,
            tilingData->commonTilingInfo.BSK),
        return ge::GRAPH_FAILED);

    const uint64_t* recvCounts = static_cast<const uint64_t*>(recvCountsPtr->GetData());
    uint64_t recvCountsSum = std::accumulate(recvCounts, recvCounts + recvCountsSize, 0ULL);
    OP_TILING_CHECK(
        recvCountsSum != tilingData->commonTilingInfo.A,
        OP_LOGE(
            A_INNER_DEBUG, "The sum of recvCounts %lu should be equal to A %lu!", recvCountsSum,
            tilingData->commonTilingInfo.A),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 检查入参 shape 之间的关系
ge::graphStatus AlltoAllvGmmTiling::CheckShapeRelation(const gert::TilingContext* context) const
{
    OP_TILING_CHECK((context->GetInputShape(GMM_WEIGHT_INDEX) == nullptr) || (context->GetInputShape(GMM_X_INDEX) == nullptr),
                     OP_LOGE(A_INNER_DEBUG, "GetInputShape gmmX or gmmWeight returned nullptr."), return ge::GRAPH_FAILED);

    uint64_t gmmWeightH1 = tilingData->commonTilingInfo.isGmmWeightTrans ?
                               context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(NUM_TWO) :
                               context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(1);
    uint64_t gmmXH1 = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(gmmXH1 != gmmWeightH1, OP_LOGE(A_INNER_DEBUG,
                    "The H1 %lu of gmmX(BSK, H1) should be equal to the H1 %lu of gmmWeight(e, H1, N1) !", gmmXH1, gmmWeightH1),
                    return ge::GRAPH_FAILED);

    if (tilingData->commonTilingInfo.isNeedMM) {
        uint64_t mmXH2 = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(1);
        uint64_t mmWeightH2 = tilingData->commonTilingInfo.isMmWeightTrans ?
                                  context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(1) :
                                  context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(mmXH2 != mmWeightH2, OP_LOGE(A_INNER_DEBUG,
                        "The H2 %lu of mmX(BS, H2) should be equal to the H2 %lu of mmWeight(H2, N2)!", mmXH2, mmWeightH2),
                        return ge::GRAPH_FAILED);

        uint64_t mmXBS = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(0);
        uint64_t mmYBS = context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(mmXBS != mmYBS, OP_LOGE(A_INNER_DEBUG,
                        "The BS %lu of mmX(BS, H2) should be equal to the BS %lu of mmY(BS, N2)!", mmXBS, mmYBS),
                        return ge::GRAPH_FAILED);
    }

    if (tilingData->commonTilingInfo.isPermuteOut) {
        OP_TILING_CHECK((context->GetOutputShape(OUTPUT_GMM_Y_INDEX) == nullptr) ||
                        (context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) == nullptr),
                        OP_LOGE(A_INNER_DEBUG, "GetPermuteOutputShape GmmY or permuteOut returned null."),
                        return ge::GRAPH_FAILED);
        uint64_t gmmYA = context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(0);
        uint64_t permuteA = context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDim(0);
        uint64_t permuteH1 = context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(gmmXH1 != permuteH1, OP_LOGE(A_INNER_DEBUG,
                        "The H1 %lu of gmmX(BSK, H1) should be equal to the H1 %lu of permuteOut(A, H1)!", gmmXH1, permuteH1),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(gmmYA != permuteA, OP_LOGE(A_INNER_DEBUG,
                        "The A %lu of gmmY(A, H1) should be equal to the A %lu of permuteOut(A, H1)!", gmmYA, permuteA),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckMmShapeDims(const gert::TilingContext* context) const
{
    if (tilingData->commonTilingInfo.isNeedMM) {
        if (context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
            OP_LOGE(
                A_INNER_DEBUG, "The dim of mmX(BS, H2) should be 2, but got %lu!",
                context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDimNum());
            return ge::GRAPH_FAILED;
        }
        if (context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
            OP_LOGE(
                A_INNER_DEBUG, "The dim of mmWeight(H2, N2) should be 2, but got %lu!",
                context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDimNum());
            return ge::GRAPH_FAILED;
        }
        if (context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
            OP_LOGE(
                A_INNER_DEBUG, "The dim of mmY(BS, N2) should be 2, but got %lu!",
                context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_TILING_CHECK(
            (context->GetOutputShape(OUTPUT_MM_Y_INDEX) != nullptr &&
             context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDimNum() != NUM_ZERO),
            OP_LOGE(A_INNER_DEBUG, "The mmY should be null when mmX and mmWeight are null!"), return ge::GRAPH_FAILED);
        if (tilingData->commonTilingInfo.isMmWeightTrans) {
            OP_LOGE(A_INNER_DEBUG, "The trans_mm_weight should be false when mmX mmWeight mmY is null!");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// 检查入参 shape 维度
ge::graphStatus AlltoAllvGmmTiling::CheckShapeDims(const gert::TilingContext* context)
{
    OP_TILING_CHECK(
        (context->GetInputShape(GMM_X_INDEX) == nullptr) || (context->GetInputShape(GMM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(A_INNER_DEBUG, "GetInputShape gmmX or gmmWeight returned null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (context->GetOutputShape(OUTPUT_GMM_Y_INDEX) == nullptr),
        OP_LOGE(A_INNER_DEBUG, "GetOutputShape gmmY returned null."), return ge::GRAPH_FAILED);

    if (context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
        OP_LOGE(A_INNER_DEBUG, "The dim of gmmX(BSK, H1) should be 2, but got %lu!",
            context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDimNum() != NUM_THREE) {
        OP_LOGE(A_INNER_DEBUG, "The dim of gmmWeight(e, H1, N1) should be 3, but got %lu!",
            context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
        OP_LOGE(A_INNER_DEBUG, "The dim of gmmY(A, N1) should be 2, but got %lu!",
            context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (tilingData->commonTilingInfo.isNeedMM) {
        OP_TILING_CHECK(context->GetOptionalInputShape(MM_WEIGHT_INDEX) == nullptr,
            OP_LOGE(A_INNER_DEBUG, "GetOptionalInputShape of mm_weight is null."),
            return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(
        CheckMmShapeDims(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check mm shape dim failed!"),
        return ge::GRAPH_FAILED);

    if (tilingData->commonTilingInfo.isPermuteOut) {
        if (context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) == nullptr ||
            context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum() == NUM_ZERO) {
            OP_LOGE(A_INNER_DEBUG, "The permuteOut should not be null when permuteOutFlag is true!");
            return ge::GRAPH_FAILED;
        }
        if (context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
            OP_LOGE(A_INNER_DEBUG, "The dim of permuteOut(A, H1) should be 2, but got %lu!",
                context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) != nullptr &&
            context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum() != NUM_ZERO) {
            OP_LOGE(A_INNER_DEBUG, "The permuteOut should be null when permuteOutFlag is false!");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::SetHcclTiling(const gert::TilingContext* context) const
{
    (void)context; // Unused
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(A_INNER_DEBUG, "Tiling Data is null!"), return ge::GRAPH_FAILED);

    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";

    const uint32_t alltoAllvReduceType = 0u;
    auto outputDataType = context->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    auto inputDataType = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    OP_TILING_CHECK(
        mc2tiling::HCCL_DATA_TYPE.find(outputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
        OP_LOGE(A_INNER_DEBUG, "%s is Unsupported outputdata type!", Ops::Base::ToString(outputDataType).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        mc2tiling::HCCL_DATA_TYPE.find(inputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
        OP_LOGE(A_INNER_DEBUG, "%s is Unsupported inputdata type!", Ops::Base::ToString(inputDataType).c_str()),
        return ge::GRAPH_FAILED);   

    auto alltoAllvDstDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);
    auto alltoAllvSrcDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(inputDataType)->second);

    Mc2CcTilingConfig hcclCcTilingConfig(epGroup_, alltoAllvCmd, alltoAllvConfig,
                                         alltoAllvReduceType, alltoAllvDstDataType, alltoAllvSrcDataType);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclInitTiling) != 0,
        OP_LOGE(A_INNER_DEBUG, "mc2CcTilingConfig mc2tiling GetTiling hcclInitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->alltoAllvCcTiling) != 0,
        OP_LOGE(A_INNER_DEBUG, "mc2CcTilingConfig mc2tiling GetTiling alltoAllvCcTiling failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckDType(const gert::TilingContext* context) const
{
    OP_TILING_CHECK(
        (context->GetInputDesc(GMM_X_INDEX) == nullptr) || (context->GetInputDesc(GMM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(A_INNER_DEBUG, "GetInputDesc gmmX or gmmWeight returned null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        context->GetOutputDesc(OUTPUT_Y_INDEX) == nullptr, OP_LOGE(A_INNER_DEBUG, "GetOutputDesc y returned null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (context->GetInputDesc(GMM_X_INDEX)->GetDataType() != ge::DT_FLOAT16) &&
            (context->GetInputDesc(GMM_X_INDEX)->GetDataType() != ge::DT_BF16),
        OP_LOGE(A_INNER_DEBUG, "Unsupported dataType, gmmx only support float16 and bfloat16!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (context->GetInputDesc(GMM_X_INDEX)->GetDataType() != context->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType()) ||
            (context->GetInputDesc(GMM_X_INDEX)->GetDataType() !=
             context->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType()),
        OP_LOGE(A_INNER_DEBUG, "The dataType of gmmWeight and gmmY should be the same with gmmX."), return ge::GRAPH_FAILED);
    if (tilingData->commonTilingInfo.isNeedMM) {
        auto mmXDex = context->GetOptionalInputDesc(MM_X_INDEX);
        OP_TILING_CHECK(mmXDex == nullptr, OP_LOGE(A_INNER_DEBUG, "Flag isNeedMM is True, but MM_X is null."), return ge::GRAPH_FAILED);
        auto mmWeightDesc = context->GetOptionalInputDesc(MM_WEIGHT_INDEX);
        OP_TILING_CHECK(
            mmWeightDesc == nullptr, OP_LOGE(A_INNER_DEBUG, "Flag isNeedMM is True, MM_WEIGHT is null."), return ge::GRAPH_FAILED);
        auto mmYDesc = context->GetOutputDesc(OUTPUT_MM_Y_INDEX);
        OP_TILING_CHECK(mmYDesc == nullptr, OP_LOGE(A_INNER_DEBUG, "GetOutputDesc mmY returned null."), return ge::GRAPH_FAILED);

        OP_TILING_CHECK(
            (context->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() != ge::DT_FLOAT16) &&
                (context->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() != ge::DT_BF16),
            OP_LOGE(A_INNER_DEBUG, "Unsupported dataType, mmx only support float16 and bfloat16!"), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            (context->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() !=
             context->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType()) ||
                (context->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() !=
                 context->GetOutputDesc(OUTPUT_MM_Y_INDEX)->GetDataType()),
            OP_LOGE(A_INNER_DEBUG, "The dataType of mmWeight and mmY should be the same with mmX."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::Init(gert::TilingContext* context)
{
    tilingData = context->GetTilingData<AlltoAllvGmmTilingData>();
    OP_TILING_CHECK(
        GetContextAttr(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Get context attr failed!"),
        return ge::GRAPH_FAILED);

    if (tilingData->commonTilingInfo.isNeedMM) {
        OP_TILING_CHECK(context->GetOptionalInputShape(MM_X_INDEX) == nullptr,
            OP_LOGE(A_INNER_DEBUG, "GetOptionalInputShape of mm_x returns null."),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(context->GetOutputShape(OUTPUT_MM_Y_INDEX) == nullptr,
            OP_LOGE(A_INNER_DEBUG, "GetOutputShape of mm_y returns null."),
            return ge::GRAPH_FAILED);
    }   
    OP_TILING_CHECK(
        CheckShapeDims(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check shape dim failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckDType(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check dtype failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckShapeRelation(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check shape relation failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        GetShapeAndFormat(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Get shape and format failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckShapeSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check shape size failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckAttrsShapeSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check Attrs shape size failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckAttrsShapeRelation(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG, "Check Attrs Shape Relation failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckSendRecvDataVolumn(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG, "Check Send Recv Data Volumn failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckMKN(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMM CheckMKN failed."),
        return ge::GRAPH_FAILED);
    OP_LOGI(
        A_INNER_DEBUG,
        "AlltoAllvGmmTiling: maxM_ is %d, maxK_ is %d, maxN_ is %d, maxMForMM_ is %d, maxKForMM_ is %d, maxNForMM_ "
        "is %d.",
        maxM_, maxK_, maxN_, maxMForMM_, maxKForMM_, maxNForMM_);
    return ge::GRAPH_SUCCESS;
}

uint64_t AlltoAllvGmmTiling::GetTilingKey(const gert::TilingContext* context) const
{
    uint32_t templateMmDType = ADD_TPL_FP16; 
    bool tilingkeyMm = false;
    bool tilingekyGmmTrans = false;
    bool tilingekyMmTrans = false;
    if (context->GetInputDesc(GMM_X_INDEX)->GetDataType() == ge::DT_FLOAT16) {
        templateMmDType = ADD_TPL_FP16;
    } else if (context->GetInputDesc(GMM_X_INDEX)->GetDataType() == ge::DT_BF16) {
        templateMmDType = ADD_TPL_BP16;
    }
    if (tilingData->commonTilingInfo.isNeedMM) { 
        tilingkeyMm = true;
    } else {
        tilingkeyMm = false;
    }
    if (tilingData->commonTilingInfo.isGmmWeightTrans) {
        tilingekyGmmTrans = true;
    } else {
        tilingekyGmmTrans = false;
    }
    if (tilingData->commonTilingInfo.isMmWeightTrans) {
        tilingekyMmTrans = true;
    } else {
        tilingekyMmTrans = false;
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(templateMmDType, tilingkeyMm, 
                                    tilingekyGmmTrans, tilingekyMmTrans);

    PrintCommonTilingInfo(tilingData->commonTilingInfo);
    OP_LOGD(A_INNER_DEBUG, "end RunFusionKernelTiling, tilingKey is %lu", tilingKey);
    return tilingKey;
}

ge::graphStatus AlltoAllvGmmTiling::setNumBlocks(gert::TilingContext* context){
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    // 设置 CV 核数
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    uint64_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);
    static const platform_ascendc::SocVersion SOC_VERSION = ascendcPlatform.GetSocVersion();
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t numBlocks = mc2tiling::GetNumBlocks(aicNum, aivNum, A_INNER_DEBUG);
    OP_TILING_CHECK(
        (PLATFORM_SIZE.ubSize == 0U) || (PLATFORM_SIZE.l1Size == 0U) || (PLATFORM_SIZE.l0CSize == 0U) ||
        (PLATFORM_SIZE.l0ASize == 0U) || (PLATFORM_SIZE.l0BSize == 0U),
        OP_LOGE(
            A_INNER_DEBUG,
            "platform info is invalid, ubSize=%lu, l1Size=%lu, l0CSize=%lu, l0ASize=%lu, l0BSize=%lu",
            PLATFORM_SIZE.ubSize, PLATFORM_SIZE.l1Size, PLATFORM_SIZE.l0CSize,
            PLATFORM_SIZE.l0ASize, PLATFORM_SIZE.l0BSize),
        return ge::GRAPH_FAILED);
    tilingData->commonTilingInfo.aicCoreNum = numBlocks;
    tilingData->commonTilingInfo.aivCoreNum = numBlocks * NUM_TWO;    // aic:aiv按照1：2配比
    context->SetBlockDim(static_cast<uint32_t>(numBlocks));           // 通算融合场景 AIC_NUM:AIV_NUM = 1:2 默认启动

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::RunFusionKernelTiling(gert::TilingContext* context)
{
    OP_LOGD(A_INNER_DEBUG, "begin RunFusionKernelTiling.");

    OP_TILING_CHECK(
        SetHcclTiling(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "set hccl tiling failed!"),
        return ge::GRAPH_FAILED);

    // aicore tiling
    OP_TILING_CHECK(
        DoAiCoreTiling(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMMAlltoAllv DoAiCoreTiling failed."),
        return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(
        setNumBlocks(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMMAlltoAllv setNumBlocks failed."),
        return ge::GRAPH_FAILED);

    // set workspaces
    size_t* workspaces = context->GetWorkspaceSizes(1); // 1: fixed value
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(A_INNER_DEBUG, "get workspace failed"), return ge::GRAPH_FAILED);

    uint64_t commOut = tilingData->commonTilingInfo.A * tilingData->commonTilingInfo.H1 * mmDataTypeSize;
    uint64_t permuteOut = tilingData->commonTilingInfo.isPermuteOut ?
                              0 :
                              (tilingData->commonTilingInfo.A * tilingData->commonTilingInfo.H1 * mmDataTypeSize);
    tilingData->commonTilingInfo.commOut = commOut;
    workspaces[0] = libApiWorkSpaceSize_ + commOut + permuteOut;
    uint64_t tilingKey = GetTilingKey(context);
    context->SetTilingKey(tilingKey);

    OP_LOGD(A_INNER_DEBUG, "end RunFusionKernelTiling, tilingKey is %lu", tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::DoAiCoreTiling(const gert::TilingContext* context)
{
    OP_LOGD(A_INNER_DEBUG, "begin DoAiCoreTiling.");
    auto dTypeForMM = matmul_tiling::DataType::DT_FLOAT16;
    if (mmDType_ == ge::DT_BF16) {
        dTypeForMM = matmul_tiling::DataType::DT_BF16;
    }
    OP_LOGD(
        A_INNER_DEBUG, "mmDType_ is %d, dTypeForMM is %d.", static_cast<int>(mmDType_), static_cast<int>(dTypeForMM));
    // TCubeTiling mmTilingData
    MMTilingParams mmParams = {maxM_, maxK_, maxN_, &baseM_, &baseK_, &baseN_};
    OP_TILING_CHECK(
        CalMMTiling(context, mmParams) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMM CalMMTiling failed."),
        return ge::GRAPH_FAILED);
    SetMMTilingParams setMnParams = {dTypeForMM, maxM_, maxK_, maxN_, baseM_, baseN_, 0};
    OP_TILING_CHECK(
        SetMMTiling(context, setMnParams) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMM SetMMTiling failed."),
        return ge::GRAPH_FAILED);

    if (tilingData->commonTilingInfo.isNeedMM) {
        mmParams = {maxMForMM_, maxKForMM_, maxNForMM_, &baseMForMM_, &baseKForMM_, &baseNForMM_};
        OP_TILING_CHECK(
            CalMMTiling(context, mmParams) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "MM CalMMTiling failed."),
            return ge::GRAPH_FAILED);
        setMnParams = {dTypeForMM, maxMForMM_, maxKForMM_, maxNForMM_, baseMForMM_, baseNForMM_, 1};
        OP_TILING_CHECK(
            SetMMTiling(context, setMnParams) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "MM SetMMTiling failed."),
            return ge::GRAPH_FAILED);
        PrintTilingDataMM(tilingData->mmTilingData);
    }
    PrintTilingDataGMM(tilingData->gmmTilingData);
    OP_LOGD(A_INNER_DEBUG, "end DoAiCoreTiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CalMMTiling(const gert::TilingContext* context, MMTilingParams& params) const
{
    OP_LOGD(A_INNER_DEBUG, "begin CalMMTlingData.");

    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);

    uint32_t tempBaseN = BEST_BASE_N;
    while (tempBaseN > static_cast<uint32_t>(params.curMaxN)) {
        tempBaseN = tempBaseN >> 1;
    }
    if (tempBaseN < static_cast<uint32_t>(params.curMaxN)) {
        tempBaseN = tempBaseN << 1;
    }
    *params.curBaseN = std::min<int32_t>(BEST_BASE_N, tempBaseN);

    // 基于使能double buffer的L0B内存计算baseK
    *params.curBaseK =
        (PLATFORM_SIZE.l0BSize / DOUBLE_BUFFER_L0A_L0B) / (*params.curBaseN * mmDataTypeSize); // 相关*怎么处理 未知
    *params.curBaseK = static_cast<int32_t>(SixteenAlign(static_cast<uint32_t>(*params.curBaseK)));
    if (*params.curBaseK > MAX_BASE_K) {
        *params.curBaseK = MAX_BASE_K;
        int32_t maxBaseN =
            SixteenAlign(PLATFORM_SIZE.l0BSize / DOUBLE_BUFFER_L0A_L0B / (*params.curBaseK * mmDataTypeSize));
        *params.curBaseN = std::min<int32_t>(*params.curBaseN, maxBaseN);
        *params.curBaseN = std::max<int32_t>(
            16, SixteenAlign(static_cast<uint32_t>(*params.curBaseN), true)); // 16: minimum value for baseN
    }
    if (*params.curBaseK > params.curMaxK) {
        *params.curBaseK =
            std::min<int32_t>(*params.curBaseK, SixteenAlign(static_cast<uint32_t>(params.curMaxK), true));
    }
    OP_TILING_CHECK(
        *params.curBaseK == 0, OP_LOGE(A_INNER_DEBUG, "curBaseK should not be 0."), return ge::GRAPH_FAILED);
    // 基于使能double buffer的L0A内存和L0B内存计算baseM(cube)
    uint32_t maxBaseM = PLATFORM_SIZE.l0CSize / (*params.curBaseN * sizeof(float));
    *params.curBaseM = std::min<uint32_t>(
        (PLATFORM_SIZE.l0ASize / DOUBLE_BUFFER_L0A_L0B) / (*params.curBaseK * mmDataTypeSize), maxBaseM);
    *params.curBaseM = static_cast<int32_t>(SixteenAlign(static_cast<uint32_t>(*params.curBaseM)));
    if (*params.curBaseM > params.curMaxM) {
        *params.curBaseM = static_cast<int32_t>(SixteenAlign(static_cast<uint32_t>(params.curMaxM), true));
    }
    OP_TILING_CHECK(
        *params.curBaseM == 0, OP_LOGE(A_INNER_DEBUG, "curBaseM should not be 0."), return ge::GRAPH_FAILED);
    OP_LOGD(A_INNER_DEBUG, "end CalMMTlingData");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::SetMMTiling(const gert::TilingContext* context, SetMMTilingParams& params) const
{
    OP_LOGD(A_INNER_DEBUG, "Begin SetMMTiling.");

    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);

    matmul_tiling::MatmulApiTiling mm(ascendcPlatform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, params.matmulDtype, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, params.matmulDtype, false);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN, params.matmulDtype);
    mm.SetOrgShape(params.curMaxM, params.curMaxN, params.curMaxK);
    mm.SetShape(params.curMaxM, params.curBaseN, params.curMaxK);
    mm.SetFixSplit(std::min(params.curBaseM, params.curMaxM), params.curBaseN);
    mm.SetBufferSpace(PLATFORM_SIZE.l1Size, PLATFORM_SIZE.l0CSize, PLATFORM_SIZE.ubSize);
    if (params.type == 0) {
        OP_TILING_CHECK(
            mm.GetTiling(tilingData->gmmTilingData) == -1, OP_LOGE(A_INNER_DEBUG, "gmm matmul getTiling failed."),
            return ge::GRAPH_FAILED);
    } else if (params.type == 1) {
        OP_TILING_CHECK(
            mm.GetTiling(tilingData->mmTilingData) == -1, OP_LOGE(A_INNER_DEBUG, "mm matmul getTiling failed."),
            return ge::GRAPH_FAILED);
    }

    OP_LOGD(A_INNER_DEBUG, "End SetMMTiling.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AlltoAllvGmmTilingFuncA3(gert::TilingContext* context)
{
    AlltoAllvGmmTiling tiling;
    OP_TILING_CHECK(
        tiling.Init(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMM tiling init failed."),
        return ge::GRAPH_FAILED);
    return tiling.RunFusionKernelTiling(context);
}

bool AlltoAllvGmmTilingStruct::IsCapable()
{
    return true;
}

ge::graphStatus AlltoAllvGmmTilingStruct::DoOpTiling()
{
    return AlltoAllvGmmTilingFuncA3(context_);
}

uint64_t AlltoAllvGmmTilingStruct::GetTilingKey() const
{
    const uint64_t tilingKey = context_->GetTilingKey();
    OP_LOGD(A_INNER_DEBUG, "AlltoAllvGmmTiling get tiling key %lu", tilingKey);
    return tilingKey;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(
        platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILING(A_INNER_DEBUG, "fail to get platform info"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();

    return ge::GRAPH_SUCCESS;
}

// Every thing is done by DoOptiling.
ge::graphStatus AlltoAllvGmmTilingBase::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

// 后续开源至gitcode需要使用__DAV_C310__的宏隔离
REGISTER_OPS_TILING_TEMPLATE(AlltoAllvGroupedMatMul, AlltoAllvGmmTilingStruct, 0);

static ge::graphStatus AlltoAllvGmmTilingFunc(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

struct AlltoAllvGmmCompileInfo {
};
static ge::graphStatus TilingParseForAlltoAllvGmm(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<AlltoAllvGmmCompileInfo>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AlltoAllvGroupedMatMul)
    .Tiling(AlltoAllvGmmTilingFunc)
    .TilingParse<AlltoAllvGmmCompileInfo>(TilingParseForAlltoAllvGmm); // 向框架注册入口函数
} // namespace optiling