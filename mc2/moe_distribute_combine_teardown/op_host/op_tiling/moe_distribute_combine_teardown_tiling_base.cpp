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
 * \file moe_distribute_combine_teardown_tiling_base.cpp
 * \brief
 */

#include "op_host/op_tiling/moe_tiling_base.h"
#include "moe_distribute_combine_teardown_tiling_base.h"

namespace {
constexpr uint32_t EXPAND_X_INDEX = 0U;
constexpr uint32_t QUANT_EXPAND_X_INDEX = 1U;
constexpr uint32_t EXPERT_IDS_INDEX = 2U;
constexpr uint32_t EXPAND_IDX_INDEX = 3U;
constexpr uint32_t EXPERT_SCALES_INDEX = 4U;
constexpr uint32_t COMM_CMD_INFO_INDEX = 5U;
constexpr uint32_t X_ACTIVE_MASK_INDEX = 6U;
constexpr uint32_t SHARED_EXPERT_X_INDEX = 7U;
constexpr uint32_t X_OUT_INDEX = 0U;

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 4;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 5;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 6;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 7;
constexpr uint32_t ATTR_COMM_QUANT_MODE_INDEX = 8;
constexpr uint32_t ATTR_COMM_TYPE_INDEX = 9;
constexpr uint32_t ATTR_COMM_ALG_INDEX = 10;

constexpr uint32_t THREE_DIMS = 3U;
constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIM = 1U;

constexpr int64_t MIN_H = 1024;
constexpr int64_t MAX_H = 8192;
constexpr int64_t MAX_BS = 512;
constexpr int64_t MAX_K = 16;

constexpr uint32_t NO_SCALES = 0U;
constexpr uint32_t STATIC_SCALES = 1U;
constexpr uint32_t DYNAMIC_SCALES = 2U;
constexpr uint32_t OP_TYPE_BATCH_WRITE = 18U;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr int64_t MIN_GROUP_EP_SIZE = 2;
constexpr int64_t MAX_GROUP_EP_SIZE = 384;
constexpr int64_t NON_QUANT = 0;
constexpr int64_t DYNAMIC_QUANT = 2;
constexpr int64_t MAX_MOE_EXPERT_NUM = 512;
constexpr int64_t SDMA_COMM = 0;
constexpr int64_t URMA_COMM = 2;
constexpr int64_t QUANT_HS_OFFSET = 4;
constexpr int64_t MAX_EP_WORLD_SIZE = 4;
constexpr int64_t BS_UPPER_BOUND = 4;
constexpr int64_t ALIGN_8 = 8UL;
constexpr int64_t ALIGN_32 = 32UL;
constexpr int64_t ALIGN_512 = 512UL;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;
constexpr uint32_t AICPUNUM = 4U;
constexpr uint32_t USED_AIV_NUMS = 40U;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr uint32_t SDMA_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr uint32_t COMM_CMD_INFO_SIZE = 16U;
constexpr uint64_t MIN_AVAILABLE_BUFF_SIZE = 2;
constexpr int64_t HCCL_BUFFER_SIZE = 44;
} // namespace
namespace MC2Tiling {

void MoeDistributeCombineTeardownTilingBase::PrintTilingDataInfo()
{
    const MoeDistributeCombineTeardownInfo &info = tilingData_->moeDistributeCombineTeardownInfo;
    OP_LOGD(nodeName_, "epWorldSize is %u.", info.epWorldSize);
    OP_LOGD(nodeName_, "epRankId is %u.", info.epRankId);
    OP_LOGD(nodeName_, "expertShardType is %u.", info.expertShardType);
    OP_LOGD(nodeName_, "sharedExpertNum is %u.", info.sharedExpertNum);
    OP_LOGD(nodeName_, "sharedExpertRankNum is %u.", info.sharedExpertRankNum);
    OP_LOGD(nodeName_, "moeExpertNum is %u.", info.moeExpertNum);
    OP_LOGD(nodeName_, "moeExpertPerRankNum is %u.", info.moeExpertPerRankNum);
    OP_LOGD(nodeName_, "globalBs is %u.", info.globalBs);
    OP_LOGD(nodeName_, "bs is %u.", info.bs);
    OP_LOGD(nodeName_, "k is %u.", info.k);
    OP_LOGD(nodeName_, "h is %u.", info.h);
    OP_LOGD(nodeName_, "aivNum is %u.", info.aivNum);
    OP_LOGD(nodeName_, "isActiveMask is %u.", static_cast<uint32_t>(info.isActiveMask));
    OP_LOGD(nodeName_, "hasSharedExpertX is %u.", static_cast<uint32_t>(info.hasSharedExpertX));
    OP_LOGD(nodeName_, "totalUbSize is %lu.", info.totalUbSize);
    OP_LOGD(nodeName_, "totalWinSize is %lu.", info.totalWinSize);
}

void MoeDistributeCombineTeardownTilingBase::SetAttrToTilingData()
{
    auto attrs = context_->GetAttrs();

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto expertShardTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_NUM_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);

    groupEp_ = string(groupEpPtr);

    // 填tilingdata
    tilingData_->moeDistributeCombineTeardownInfo.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
    tilingData_->moeDistributeCombineTeardownInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData_->moeDistributeCombineTeardownInfo.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    tilingData_->moeDistributeCombineTeardownInfo.expertShardType = static_cast<uint32_t>(*expertShardTypePtr);
    tilingData_->moeDistributeCombineTeardownInfo.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData_->moeDistributeCombineTeardownInfo.sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    tilingData_->moeDistributeCombineTeardownInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);

    uint32_t localMoeExpertNum = 1;
    if (*epRankIdPtr >= *sharedExpertRankNumPtr) {
        localMoeExpertNum = *moeExpertNumPtr / (*epWorldSizePtr - *sharedExpertRankNumPtr);
    }
    tilingData_->moeDistributeCombineTeardownInfo.moeExpertPerRankNum = localMoeExpertNum;

    auto *xActiveMaskStorageShape = context_->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    tilingData_->moeDistributeCombineTeardownInfo.isActiveMask = (xActiveMaskStorageShape != nullptr);

    auto *sharedExpertXShape = context_->GetOptionalInputShape(SHARED_EXPERT_X_INDEX);
    tilingData_->moeDistributeCombineTeardownInfo.hasSharedExpertX = (sharedExpertXShape != nullptr);
}

void MoeDistributeCombineTeardownTilingBase::SetDimsToTilingData()
{
    auto expandXStorageShape = context_->GetInputShape(EXPAND_X_INDEX); // A, H
    auto expertIdsShape = context_->GetInputShape(EXPERT_IDS_INDEX);    // Bs, K
    int64_t H = expandXStorageShape->GetStorageShape().GetDim(1);
    int64_t Bs = expertIdsShape->GetStorageShape().GetDim(0);
    int64_t K = expertIdsShape->GetStorageShape().GetDim(1);
    tilingData_->moeDistributeCombineTeardownInfo.bs = static_cast<uint32_t>(Bs);
    tilingData_->moeDistributeCombineTeardownInfo.h = static_cast<uint32_t>(H);
    tilingData_->moeDistributeCombineTeardownInfo.k = static_cast<uint32_t>(K);
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckAttrsWithoutRelation()
{
    return ge::GRAPH_FAILED;
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckAttrsComplex()
{
    return ge::GRAPH_FAILED;
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName_, "attrs is null."), return ge::GRAPH_FAILED);

    if (CheckAttrsNullptr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckAttrsWithoutRelation() != ge::GRAPH_SUCCESS || CheckAttrsComplex() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckAttrsNullptr()
{
    auto attrs = context_->GetAttrs();

    // 判空指针
    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    OP_TILING_CHECK(groupEpPtr == nullptr, OP_LOGE(nodeName_, "groupEp is nullptr."), return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName_, "epWorldSizePtr is nullptr"),
                    return ge::GRAPH_FAILED);

    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName_, "epRankIdPtr is nullptr"), return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName_, "moeExpertNumPtr is nullptr"),
                    return ge::GRAPH_FAILED);

    auto expertShardTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    OP_TILING_CHECK(expertShardTypePtr == nullptr, OP_LOGE(nodeName_, "expertShardTypePtr is nullptr"),
                    return ge::GRAPH_FAILED);

    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_NUM_INDEX);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE(nodeName_, "sharedExpertNumPtr is nullptr"),
                    return ge::GRAPH_FAILED);

    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName_, "sharedExpertRankNumPtr is nullptr"),
                    return ge::GRAPH_FAILED);

    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName_, "globalBsPtr is nullptr"), return ge::GRAPH_FAILED);

    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_QUANT_MODE_INDEX);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE(nodeName_, "commQuantModePtr is nullptr"),
                    return ge::GRAPH_FAILED);

    auto commTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_TYPE_INDEX);
    OP_TILING_CHECK(commTypePtr == nullptr, OP_LOGE(nodeName_, "commTypePtr is nullptr"), return ge::GRAPH_FAILED);

    // 判非空指针
    auto commAlgPtr = attrs->GetAttrPointer<char>(ATTR_COMM_ALG_INDEX);
    if (commAlgPtr != nullptr) {
        const std::string commAlg = std::string(commAlgPtr);
        OP_TILING_CHECK((commAlg != ""), OP_LOGE(nodeName_, "commAlg shoud be null or empty string."),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckOneTensorDim(std::string name, TensorType tensortype,
                                                                          uint32_t index, uint32_t dims)
{
    const gert::StorageShape *StorageShape;
    if (tensortype == INPUT) {
        StorageShape = context_->GetInputShape(index);
    } else if (tensortype == OUTPUT) {
        StorageShape = context_->GetOutputShape(index);
    } else if (tensortype == OPTIONINPUT) {
        StorageShape = context_->GetOptionalInputShape(index);
    } else {
        OP_LOGE(nodeName_, "TensorType Only Support input or output. type:%u, name: %s, index:%u", tensortype,
                name.c_str(), index);
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK(StorageShape == nullptr, OP_LOGE(nodeName_, "%s is null.", name.c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(StorageShape->GetStorageShape().GetDimNum() != dims,
                    OP_LOGE(nodeName_, "%s got %lu dim, expected %u dim", name.c_str(),
                            StorageShape->GetStorageShape().GetDimNum(), dims),
                    return ge::GRAPH_FAILED);

    for (uint32_t d = 0; d < dims; d++) {
        OP_LOGD(nodeName_, "%s dim%u = %ld", name.c_str(), d, StorageShape->GetStorageShape().GetDim(d));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckInputTensorDim()
{
    OP_TILING_CHECK(CheckOneTensorDim("expandX", INPUT, EXPAND_X_INDEX, TWO_DIMS) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName_, "expandX checkdim failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOneTensorDim("quantExpandX", INPUT, QUANT_EXPAND_X_INDEX, TWO_DIMS) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName_, "quantExpandX checkdim failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOneTensorDim("expertIds", INPUT, EXPERT_IDS_INDEX, TWO_DIMS) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName_, "expertIds checkdim failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOneTensorDim("expandIdx", INPUT, EXPAND_IDX_INDEX, ONE_DIM) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName_, "expandIdx checkdim failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOneTensorDim("expertScales", INPUT, EXPERT_SCALES_INDEX, TWO_DIMS) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName_, "expertScales checkdim failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOneTensorDim("commCmdInfo", INPUT, COMM_CMD_INFO_INDEX, ONE_DIM) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName_, "commCmdInfo checkdim failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckOptionalInputTensorDim()
{
    if (tilingData_->moeDistributeCombineTeardownInfo.isActiveMask) {
        OP_TILING_CHECK(CheckOneTensorDim("xActiveMask", OPTIONINPUT, X_ACTIVE_MASK_INDEX, ONE_DIM) !=
                            ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName_, "xActiveMask checkdim failed."), return ge::GRAPH_FAILED);
    }

    if (tilingData_->moeDistributeCombineTeardownInfo.hasSharedExpertX) {
        // 传入有效数字时，sharedExpertNum需为0
        OP_TILING_CHECK(tilingData_->moeDistributeCombineTeardownInfo.sharedExpertNum != 0,
                        OP_LOGE(nodeName_, "sharedExpertNum[%u] must be 0, when sharedExpertX is not nullptr",
                                tilingData_->moeDistributeCombineTeardownInfo.sharedExpertNum),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            CheckOneTensorDim("sharedExpertX", OPTIONINPUT, SHARED_EXPERT_X_INDEX, TWO_DIMS) != ge::GRAPH_SUCCESS &&
                CheckOneTensorDim("sharedExpertX", OPTIONINPUT, SHARED_EXPERT_X_INDEX, THREE_DIMS) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName_, "sharedExpertX checkdim failed, must be 2 or 3 dim."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckOutputTensorDim()
{
    OP_TILING_CHECK(CheckOneTensorDim("xOut", OUTPUT, X_OUT_INDEX, TWO_DIMS) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName_, "xOut checkdim failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckTensorDim()
{
    OP_TILING_CHECK(CheckInputTensorDim() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName_, "Input param shape is invalid."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOptionalInputTensorDim() != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName_, "Optional input param shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckOutputTensorDim() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName_, "Output param shape is invalid."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckTensorShapeRelation()
{
    auto expandXStorageShape = context_->GetInputShape(EXPAND_X_INDEX);            // A, H
    auto quantExpandXStorageShape = context_->GetInputShape(QUANT_EXPAND_X_INDEX); // A, tokenMsgSize
    auto expertIdsStorageShape = context_->GetInputShape(EXPERT_IDS_INDEX);        // Bs, K
    auto expandIdxStorageShape = context_->GetInputShape(EXPAND_IDX_INDEX);        // Bs * K
    auto expertScalesStorageShape = context_->GetInputShape(EXPERT_SCALES_INDEX);  // Bs, K
    auto xOutStorageShape = context_->GetOutputShape(X_OUT_INDEX);                 // Bs, H

    const int64_t H = expandXStorageShape->GetStorageShape().GetDim(1);
    const int64_t Bs = expertIdsStorageShape->GetStorageShape().GetDim(0);
    const int64_t K = expertIdsStorageShape->GetStorageShape().GetDim(1);

    // 校验tokenMsgSize
    const int64_t tempTokenMsgSize = ops::CeilAlign(
        static_cast<int64_t>(ops::CeilAlign(H, ALIGN_32) + ops::CeilAlign(H, ALIGN_8) / ALIGN_8 * sizeof(float)),
        ALIGN_512);
    OP_TILING_CHECK(quantExpandXStorageShape->GetStorageShape().GetDim(1) != tempTokenMsgSize,
                    OP_LOGE(nodeName_, "quantexpandx's dim1[%ld] should be equal to tokenMsgSize[%ld]",
                            quantExpandXStorageShape->GetStorageShape().GetDim(1), tempTokenMsgSize),
                    return ge::GRAPH_FAILED);

    // 校验xOut的H
    OP_TILING_CHECK(xOutStorageShape->GetStorageShape().GetDim(1) != H,
                    OP_LOGE(nodeName_, "expandx's dim1[%ld] should be equal to xout's dim1[%ld]", H,
                            xOutStorageShape->GetStorageShape().GetDim(1)),
                    return ge::GRAPH_FAILED);

    // 校验expertScales、xOut的Bs
    OP_TILING_CHECK(expertScalesStorageShape->GetStorageShape().GetDim(0) != Bs,
                    OP_LOGE(nodeName_, "expertIds's dim0[%ld] should be equal to expertScales' dim0[%ld]", Bs,
                            expertScalesStorageShape->GetStorageShape().GetDim(0)),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xOutStorageShape->GetStorageShape().GetDim(0) != Bs,
                    OP_LOGE(nodeName_, "expertIds's dim0[%ld] should be equal to xOut's dim0[%ld]", Bs,
                            xOutStorageShape->GetStorageShape().GetDim(0)),
                    return ge::GRAPH_FAILED);

    // 校验expertScales的K
    OP_TILING_CHECK(expertScalesStorageShape->GetStorageShape().GetDim(1) != K,
                    OP_LOGE(nodeName_, "expertIds's dim1[%ld] should be equal to expertscales' dim1[%ld]", K,
                            expertScalesStorageShape->GetStorageShape().GetDim(1)),
                    return ge::GRAPH_FAILED);

    // 校验expandIdx的Bs*K
    OP_TILING_CHECK(expandIdxStorageShape->GetStorageShape().GetDim(0) != Bs * K,
                    OP_LOGE(nodeName_, "expandIdx's dim0[%ld] should be equal to Bs[%ld] * K[%ld]",
                            expandIdxStorageShape->GetStorageShape().GetDim(0), Bs, K),
                    return ge::GRAPH_FAILED);

    return CheckTensorShapeRelationSecondPart();
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckTensorShapeRelationSecondPart()
{
    const int64_t A = context_->GetInputShape(EXPAND_X_INDEX)->GetStorageShape().GetDim(0);
    const int64_t Bs = context_->GetInputShape(EXPERT_IDS_INDEX)->GetStorageShape().GetDim(0);
    const int64_t K = context_->GetInputShape(EXPERT_IDS_INDEX)->GetStorageShape().GetDim(1);

    const auto sharedExpertRankNum = tilingData_->moeDistributeCombineTeardownInfo.sharedExpertRankNum;
    const auto epWorldSize = tilingData_->moeDistributeCombineTeardownInfo.epWorldSize;
    const auto sharedExpertNum = tilingData_->moeDistributeCombineTeardownInfo.sharedExpertNum;
    const auto localMoeExpertNum = tilingData_->moeDistributeCombineTeardownInfo.moeExpertPerRankNum;
    const auto globalBs = tilingData_->moeDistributeCombineTeardownInfo.globalBs;

    // 补充Attr的globalBs校验
    OP_TILING_CHECK((globalBs != 0) && ((globalBs < Bs * epWorldSize) || (globalBs > MAX_BS * epWorldSize)),
                    OP_LOGE(nodeName_,
                            "globalBs[%u] should be >= Bs * epWorldSize[%lu] and <= MaxBs * epWorldSize, or = 0",
                            globalBs, Bs * epWorldSize),
                    return ge::GRAPH_FAILED);

    // 校验A的取值约束
    if (tilingData_->moeDistributeCombineTeardownInfo.epRankId < sharedExpertRankNum) { // 共享专家
        bool AisValid = (globalBs == 0) ? (A == Bs * epWorldSize * sharedExpertNum / sharedExpertRankNum) :
                                          (A == globalBs * sharedExpertNum / sharedExpertRankNum);
        OP_TILING_CHECK(
            !AisValid,
            OP_LOGE(nodeName_,
                    "For shared expert, globalBs[%u], A[%ld] should be equal to Bs[%ld] * epWorldSize[%u] * "
                    "sharedExpertNum[%u] / sharedExpertRankNum[%u] if globalBs =0, else equal to globalBs[%u] * "
                    "sharedExpertNum[%u] / sharedExpertRankNum[%u]",
                    globalBs, A, Bs, epWorldSize, sharedExpertNum, sharedExpertRankNum, globalBs, sharedExpertNum,
                    sharedExpertRankNum),
            return ge::GRAPH_FAILED);
    } else { // MoE专家
        bool AisValid = (globalBs == 0) ?
                            (A >= (Bs * epWorldSize * std::min(static_cast<int64_t>(localMoeExpertNum), K))) :
                            (A >= (globalBs * std::min(static_cast<int64_t>(localMoeExpertNum), K)));
        OP_TILING_CHECK(
            !AisValid,
            OP_LOGE(
                nodeName_,
                "For moe expert, globalBs[%u], if globalBs =0, A[%ld] should >= Bs[%ld] * epWorldSize[%u] * "
                "min(localMoeExpertNum[%u], K[%ld]) / "
                "sharedExpertRankNum[%u]; else A[%ld] should >= globalBs[%u] * min(localMoeExpertNum[%u], K[%ld]) / "
                "sharedExpertRankNum[%u]",
                globalBs, A, Bs, epWorldSize, localMoeExpertNum, K, sharedExpertRankNum, A, globalBs, localMoeExpertNum,
                K, sharedExpertRankNum),
            return ge::GRAPH_FAILED);
    }

    // 校验quantExpandX的A
    OP_TILING_CHECK(context_->GetInputShape(QUANT_EXPAND_X_INDEX)->GetStorageShape().GetDim(0) != A,
                    OP_LOGE(nodeName_, "expandx's dim0[%ld] should be equal to quantexpandx's dim0[%ld]", A,
                            context_->GetInputShape(QUANT_EXPAND_X_INDEX)->GetStorageShape().GetDim(0)),
                    return ge::GRAPH_FAILED);

    return CheckTensorShapeRelationThirdPart();
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckTensorShapeRelationThirdPart()
{
    auto expandXStorageShape = context_->GetInputShape(EXPAND_X_INDEX);     // A, H
    auto expertIdsStorageShape = context_->GetInputShape(EXPERT_IDS_INDEX); // Bs, K
    auto commCmdInfoShape = context_->GetInputShape(COMM_CMD_INFO_INDEX);   // 一维

    const int64_t Bs = expertIdsStorageShape->GetStorageShape().GetDim(0);
    const int64_t H = expandXStorageShape->GetStorageShape().GetDim(1);
    const int64_t A = expandXStorageShape->GetStorageShape().GetDim(0);
    const int64_t commCmdInfoSize = commCmdInfoShape->GetStorageShape().GetDim(0);
    const auto epWorldSize = tilingData_->moeDistributeCombineTeardownInfo.epWorldSize;

    // 校验commCmdInfoSize的取值约束
    OP_TILING_CHECK((commCmdInfoSize != (A + epWorldSize) * COMM_CMD_INFO_SIZE),
                    OP_LOGE(nodeName_, "commCmdInfoSize[%ld] should be (A[%ld] + epWorldSize[%u]) * %u",
                            commCmdInfoSize, A, epWorldSize, COMM_CMD_INFO_SIZE),
                    return ge::GRAPH_FAILED);

    if (tilingData_->moeDistributeCombineTeardownInfo.isActiveMask) {
        auto xActiveMaskStorageShape = context_->GetOptionalInputShape(X_ACTIVE_MASK_INDEX); // Bs
        OP_TILING_CHECK(xActiveMaskStorageShape->GetStorageShape().GetDim(0) != Bs,
                        OP_LOGE(nodeName_, "xactivemask's dim0[%lu] should be equal to expertIds's dim0[%lu]",
                                xActiveMaskStorageShape->GetStorageShape().GetDim(0), Bs),
                        return ge::GRAPH_FAILED);
    }

    if (tilingData_->moeDistributeCombineTeardownInfo.hasSharedExpertX) {
        auto sharedExpertXStorageShape =
            context_->GetOptionalInputShape(SHARED_EXPERT_X_INDEX);                 // Bs, H 或 a, b, H (a * b = Bs)
        if (sharedExpertXStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) { // Bs, H
            OP_TILING_CHECK(sharedExpertXStorageShape->GetStorageShape().GetDim(0) != Bs,
                            OP_LOGE(nodeName_, "sharedExpertX's dim0[%ld] should be equal to Bs[%ld]",
                                    sharedExpertXStorageShape->GetStorageShape().GetDim(0), Bs),
                            return ge::GRAPH_FAILED);
            OP_TILING_CHECK(sharedExpertXStorageShape->GetStorageShape().GetDim(1) != H,
                            OP_LOGE(nodeName_, "sharedExpertX's dim1[%ld] should be equal to expandx's dim1[%ld]",
                                    sharedExpertXStorageShape->GetStorageShape().GetDim(1), H),
                            return ge::GRAPH_FAILED);
        } else if (sharedExpertXStorageShape->GetStorageShape().GetDimNum() == THREE_DIMS) { // a, b, H (a * b = Bs)
            OP_TILING_CHECK(sharedExpertXStorageShape->GetStorageShape().GetDim(0) *
                                    sharedExpertXStorageShape->GetStorageShape().GetDim(1) !=
                                Bs,
                            OP_LOGE(nodeName_, "sharedExpertX's dim0[%ld] * dim1[%ld] should be equal to Bs[%ld]",
                                    sharedExpertXStorageShape->GetStorageShape().GetDim(0),
                                    sharedExpertXStorageShape->GetStorageShape().GetDim(1), Bs),
                            return ge::GRAPH_FAILED);
            OP_TILING_CHECK(sharedExpertXStorageShape->GetStorageShape().GetDim(2) != H,
                            OP_LOGE(nodeName_, "sharedExpertX's dim2[%ld] should be equal to expandx's dim1[%ld]",
                                    sharedExpertXStorageShape->GetStorageShape().GetDim(2), H),
                            return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckBsHKSize(int64_t bs, int64_t h, int64_t k)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckTensorShapeSize()
{
    auto expandXStorageShape = context_->GetInputShape(EXPAND_X_INDEX);     // A, H
    auto expertIdsStorageShape = context_->GetInputShape(EXPERT_IDS_INDEX); // Bs, K

    auto Bs = expertIdsStorageShape->GetStorageShape().GetDim(0);
    auto K = expertIdsStorageShape->GetStorageShape().GetDim(1);
    auto H = expandXStorageShape->GetStorageShape().GetDim(1);
    if (CheckBsHKSize(Bs, H, K) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto expandIdxStorageShape = context_->GetInputShape(EXPAND_IDX_INDEX); // Bs * K
    OP_TILING_CHECK((expandIdxStorageShape->GetStorageShape().GetDim(0) != Bs * K),
                    OP_LOGE(nodeName_, "ExpandIdx should be BS * K [%ld], but got [%ld]", Bs * K,
                            expandIdxStorageShape->GetStorageShape().GetDim(0)),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckTensorDataType()
{
    auto expandXDesc = context_->GetInputDesc(EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName_, "expandXDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expandXDesc->GetDataType() != ge::DT_BF16) && (expandXDesc->GetDataType() != ge::DT_FLOAT16),
                    OP_LOGE(nodeName_, "expandX dataType is invalid, dataType should be bf16 or float16, but is %s",
                            Ops::Base::ToString(expandXDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    auto quantExpandXDesc = context_->GetInputDesc(QUANT_EXPAND_X_INDEX);
    OP_TILING_CHECK(quantExpandXDesc == nullptr, OP_LOGE(nodeName_, "quantExpandXDesc is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((quantExpandXDesc->GetDataType() != ge::DT_INT8),
                    OP_LOGE(nodeName_, "quantExpandX dataType is invalid, dataType should be int8, but is %s",
                            Ops::Base::ToString(quantExpandXDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    auto expertIdsDesc = context_->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE(nodeName_, "expertIdsDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expertIdsDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName_, "expertIds dataType is invalid, dataType should be int32, but is %s",
                            Ops::Base::ToString(expertIdsDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    auto expandIdxDesc = context_->GetInputDesc(EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxDesc == nullptr, OP_LOGE(nodeName_, "expandIdxDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expandIdxDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName_, "expandIdx dataType is invalid, dataType should be int32, but is %s",
                            Ops::Base::ToString(expandIdxDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    auto expertScalesDesc = context_->GetInputDesc(EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesDesc == nullptr, OP_LOGE(nodeName_, "expertScalesDesc is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expertScalesDesc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE(nodeName_, "expertScales dataType is invalid, dataType should be float32, but is %s",
                            Ops::Base::ToString(expertScalesDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    auto commCmdInfoDesc = context_->GetInputDesc(COMM_CMD_INFO_INDEX);
    OP_TILING_CHECK(commCmdInfoDesc == nullptr, OP_LOGE(nodeName_, "commCmdInfoDesc is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((commCmdInfoDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName_, "commCmdInfo dataType is invalid, dataType should be int32, but is %s",
                            Ops::Base::ToString(commCmdInfoDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    return CheckTensorDataTypeSecondPart();
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckTensorDataTypeSecondPart()
{
    if (tilingData_->moeDistributeCombineTeardownInfo.isActiveMask) {
        auto xActiveMaskDesc = context_->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName_, "xActiveMaskDesc is null."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK((xActiveMaskDesc->GetDataType() != ge::DT_BOOL),
                        OP_LOGE(nodeName_, "xActiveMask dataType is invalid, dataType should be bool, but is %s",
                                Ops::Base::ToString(xActiveMaskDesc->GetDataType()).c_str()),
                        return ge::GRAPH_FAILED);
    }

    auto expandXDesc = context_->GetInputDesc(EXPAND_X_INDEX);
    if (tilingData_->moeDistributeCombineTeardownInfo.hasSharedExpertX) {
        auto shardExpertXDesc = context_->GetOptionalInputDesc(SHARED_EXPERT_X_INDEX);
        OP_TILING_CHECK(shardExpertXDesc == nullptr, OP_LOGE(nodeName_, "shardExpertXDesc is null."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            (shardExpertXDesc->GetDataType() != expandXDesc->GetDataType()),
            OP_LOGE(nodeName_,
                    "shardExpertX dataType is invalid, dataType should be equal to expandX dataType %s, but is %s",
                    Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                    Ops::Base::ToString(shardExpertXDesc->GetDataType()).c_str()),
            return ge::GRAPH_FAILED);
    }

    auto xOutDesc = context_->GetOutputDesc(X_OUT_INDEX);
    OP_TILING_CHECK(xOutDesc == nullptr, OP_LOGE(nodeName_, "xOutDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((xOutDesc->GetDataType() != expandXDesc->GetDataType()),
                    OP_LOGE(nodeName_,
                            "xOut dataType is invalid, dataType should be equal to expandX dataType %s, but is %s",
                            Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                            Ops::Base::ToString(xOutDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void MoeDistributeCombineTeardownTilingBase::SetTilingKey()
{
    bool tp = false;

    // 设置tilingKey模板参数
    const uint64_t tilingKey = GET_TPL_TILING_KEY(tp);
    context_->SetTilingKey(tilingKey);
    OP_LOGD(nodeName_, "tilingKey is [%lu].", tilingKey);
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::CheckHcclBuffsize()
{
    const uint64_t hcclBuffSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize() / 2;
    OP_TILING_CHECK(hcclBuffSize < MIN_AVAILABLE_BUFF_SIZE,
                    OP_LOGE(nodeName_, "HCCL_BUFFSIZE[%ld] is less than [%ld]", hcclBuffSize, MIN_AVAILABLE_BUFF_SIZE),
                    return ge::GRAPH_FAILED);

    auto sharedExpertNum = tilingData_->moeDistributeCombineTeardownInfo.sharedExpertNum;
    auto epWorldSize = tilingData_->moeDistributeCombineTeardownInfo.epWorldSize;
    auto Bs = tilingData_->moeDistributeCombineTeardownInfo.bs;
    auto H = tilingData_->moeDistributeCombineTeardownInfo.h;
    auto K = tilingData_->moeDistributeCombineTeardownInfo.k;
    auto localExpertNum = tilingData_->moeDistributeCombineTeardownInfo.moeExpertPerRankNum;

    const int64_t tempBuffSize =
        MIN_AVAILABLE_BUFF_SIZE *
        (localExpertNum * Bs * epWorldSize *
             ops::CeilAlign((ops::CeilAlign(static_cast<int64_t>(H) * 2, ALIGN_32) + HCCL_BUFFER_SIZE), ALIGN_512) +
         (static_cast<int64_t>(K) + static_cast<int64_t>(sharedExpertNum)) * static_cast<int64_t>(Bs) *
             ops::CeilAlign(static_cast<int64_t>(H) * 2, ALIGN_512));

    OP_TILING_CHECK(hcclBuffSize < tempBuffSize,
                    OP_LOGE(nodeName_, "HCCL_BUFFSIZE[%ld] is less than [%ld]", hcclBuffSize, tempBuffSize),
                    return ge::GRAPH_FAILED);

    tilingData_->moeDistributeCombineTeardownInfo.totalWinSize = hcclBuffSize;

    return ge::GRAPH_SUCCESS;
}

void MoeDistributeCombineTeardownTilingBase::SetPlatformInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t blockDim = 1U;
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context_->SetBlockDim(blockDim);
    context_->SetAicpuBlockDim(AICPUNUM);
    tilingData_->moeDistributeCombineTeardownInfo.totalUbSize = ubSize;
    tilingData_->moeDistributeCombineTeardownInfo.aivNum = aivNum;
    context_->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    OP_LOGD(nodeName_, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::SetWorkspace()
{
    size_t *workspace = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName_, "get workspace failed"),
                    return ge::GRAPH_FAILED);
    workspace[0] = static_cast<size_t>(SYSTEM_NEED_WORKSPACE) + SDMA_NEED_WORKSPACE; // 可能的URMA适配点
    OP_LOGD(nodeName_, "workspce[0] size is %ld", workspace[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::SetHcommCfg()
{
    return ge::GRAPH_FAILED;
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::MoeDistributeCombineTeardownTilingFuncImpl()
{
    OP_LOGD(nodeName_, "MoeDistributeCombineTeardownTilingFunc start");
    tilingData_ = context_->GetTilingData<MoeDistributeCombineTeardownTilingData>();
    OP_TILING_CHECK(tilingData_ == nullptr, OP_LOGE(nodeName_, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    // 实现Tiling拦截，并在类变量"tilingData_"中设置相关信息
    if (CheckAttrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 获取入参属性
    SetAttrToTilingData();

    if (!(CheckTensorDim() == ge::GRAPH_SUCCESS && CheckTensorShapeSize() == ge::GRAPH_SUCCESS &&
          CheckTensorShapeRelation() == ge::GRAPH_SUCCESS && CheckTensorDataType() == ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    SetDimsToTilingData();

    OP_TILING_CHECK(CheckHcclBuffsize() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "Tiling set hccl failed"),
                    return ge::GRAPH_FAILED);
    SetPlatformInfo();

    OP_TILING_CHECK(SetWorkspace() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "Tiling set workspace failed"),
                    return ge::GRAPH_FAILED);

    SetTilingKey();
    OP_TILING_CHECK(SetHcommCfg() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName_, "SetHcommCfg failed."),
                    return ge::GRAPH_FAILED);
    PrintTilingDataInfo();
    OP_LOGD(nodeName_, "MoeDistributeCombineTeardownTilingFunc success");
    return ge::GRAPH_SUCCESS;
}

bool MoeDistributeCombineTeardownTilingBase::IsCapable()
{
    return false;
}

ge::graphStatus MoeDistributeCombineTeardownTilingBase::DoOpTiling()
{
    return MoeDistributeCombineTeardownTilingFuncImpl();
}

uint64_t MoeDistributeCombineTeardownTilingBase::GetTilingKey() const
{
    // TilingKey calculation is done in DoOptiling
    const uint64_t tilingKey = context_->GetTilingKey();
    OP_LOGD(nodeName_, "%s get tiling key %lu", this->socTilingName_, tilingKey);
    return tilingKey;
}

} // namespace MC2Tiling