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
 * \file moe_distribute_dispatch_setup_tiling_base.cpp
 * \brief
 */

#include "op_host/op_tiling/moe_tiling_base.h"
#include "moe_distribute_dispatch_setup_tiling_base.h"

namespace {
constexpr uint32_t X_INDEX = 0U;
constexpr uint32_t EXPERT_IDS_INDEX = 1U;
constexpr uint32_t SCALES_INDEX = 2U;
constexpr uint32_t X_ACTIVE_MASK_INDEX = 3U;
constexpr uint32_t OUTPUT_Y_INDEX = 0U;
constexpr uint32_t OUTPUT_EXPAND_IDX_INDEX = 1U;
constexpr uint32_t OUTPUT_COMM_CMD_INFO_INDEX = 2U;

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0U;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1U;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2U;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3U;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 4U;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 5U;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 6U;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 7U;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 8U;
constexpr uint32_t ATTR_COMM_TYPE_INDEX = 9U;
constexpr uint32_t ATTR_COMM_ALG_INDEX = 10U;

constexpr uint32_t ONE_DIMS = 1U;
constexpr uint32_t TWO_DIMS = 2U;

constexpr uint32_t INIT_TILINGKEY = 1000U;
constexpr uint32_t NO_SCALES = 0U;
constexpr uint32_t STATIC_SCALES = 1U;
constexpr uint32_t DYNAMIC_SCALES = 2U;
constexpr uint32_t OP_TYPE_BATCH_WRITE = 18U;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr size_t MAX_COMM_ALG_LENGTH = 1UL;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4UL;
constexpr int64_t MIN_GROUP_EP_SIZE = 2UL;
constexpr int64_t MAX_GROUP_EP_SIZE = 384UL;
constexpr int64_t UNQUANT = 0;
constexpr int64_t STATIC_QUANT = 1;
constexpr int64_t PERTOKEN_DYNAMIC_QUANT = 2;
constexpr int64_t PERGROUP_DYNAMIC_QUANT = 3;
constexpr int64_t MX_QUANT = 4;
constexpr int64_t MAX_MOE_EXPERT_NUM = 512UL;
constexpr int64_t SDMA_COMM = 0UL;
constexpr int64_t URMA_COMM = 2UL;
constexpr int64_t QUANT_HS_OFFSET = 4UL;
constexpr int64_t MAX_EP_WORLD_SIZE = 4UL;
constexpr int64_t BS_UPPER_BOUND = 4UL;
constexpr int64_t COMM_CMD_INFO_MULTIPLY = 16UL;
constexpr int64_t MIN_AVAILABLE_BUFF_SIZE = 2UL;
constexpr int64_t HCCL_BUFFER_SIZE = 44UL;

constexpr int64_t MIN_H = 1024UL;
constexpr int64_t MAX_H = 8192UL;
constexpr int64_t MAX_BS = 512UL;
constexpr int64_t MAX_K = 16UL;

constexpr uint32_t LOCAL_STREAM_MAX_NUM = 40U;
constexpr uint32_t TILINGKEY_SCALES = 10U;
constexpr int64_t MOE_EXPERT_MAX_NUM = 512U;
constexpr int64_t MB_SIZE = 1024UL * 1024UL;
constexpr int64_t WIN_ADDR_ALIGN = 512UL;
constexpr int64_t EVEN_ALIGN = 2;
constexpr int64_t UB_ALIGN = 32UL;
constexpr int64_t ALIGN_32 = 32UL;
constexpr int64_t ALIGN_128 = 128UL;
constexpr int64_t ALIGN_256 = 256UL;
constexpr int64_t ALIGN_512 = 512UL;
constexpr int64_t SCALE_EXPAND_IDX_BUFFER = 44UL;
constexpr int64_t DOUBLE_DATA_BUFFER = 2UL;
constexpr int64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr int64_t AICPUNUM = 4UL;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr uint32_t SDMA_NEED_WORKSPACE = 16U * 1024 * 1024;
} // namespace

namespace optiling {

uint64_t MoeDistributeDispatchSetupTilingBase::GetTilingKey() const
{
    // TilingKey calculation is done in DoOptiling
    const uint64_t tilingKey = context_->GetTilingKey();
    OP_LOGD(nodeName_, "%s get tiling key %lu", this->socTilingName_, tilingKey);
    return tilingKey;
}

const void MoeDistributeDispatchSetupTilingBase::PrintTilingDataInfo()
{
    const MoeDistributeDispatchSetupInfo& info = tilingData_->moeDistributeDispatchSetupInfo;
    OP_LOGD(nodeName_, "epWorldSize is %u.", info.epWorldSize);
    OP_LOGD(nodeName_, "epRankId is %u.", info.epRankId);
    OP_LOGD(nodeName_, "expertShardType is %u.", info.expertShardType);
    OP_LOGD(nodeName_, "sharedExpertNum is %u.", info.sharedExpertNum);
    OP_LOGD(nodeName_, "sharedExpertRankNum is %u.", info.sharedExpertRankNum);
    OP_LOGD(nodeName_, "moeExpertNum is %u.", info.moeExpertNum);
    OP_LOGD(nodeName_, "quantMode is %u.", info.quantMode);
    OP_LOGD(nodeName_, "globalBs is %u.", info.globalBs);
    OP_LOGD(nodeName_, "bs is %u.", info.bs);
    OP_LOGD(nodeName_, "k is %u.", info.k);
    OP_LOGD(nodeName_, "h is %u.", info.h);
    OP_LOGD(nodeName_, "aivNum is %u.", info.aivNum);
    OP_LOGD(nodeName_, "isQuant is %u.", static_cast<uint32_t>(info.isQuant));
    OP_LOGD(nodeName_, "isActiveMask is %u.", static_cast<uint32_t>(info.isActiveMask));
    OP_LOGD(nodeName_, "totalUbSize is %lu.", info.totalUbSize);
    OP_LOGD(nodeName_, "totalWinSize is %lu.", info.totalWinSize);
    OP_LOGD(nodeName_, "sdmaUsedStreamPerCore is %u.", info.sdmaUsedStreamPerCore);
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckRequiredAttrValue()
{
    auto attrs = context_->GetAttrs();
    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);

    OP_TILING_CHECK(
        ((strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
         (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH)),
        OP_LOGE(nodeName_, "groupEp is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((*epWorldSizePtr < MIN_GROUP_EP_SIZE) || (*epWorldSizePtr > MAX_GROUP_EP_SIZE)),
        OP_LOGE(
            nodeName_, "epWorldSize should be [%ld, %ld], get %ld", MIN_GROUP_EP_SIZE, MAX_GROUP_EP_SIZE,
            *epWorldSizePtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((*epRankIdPtr < 0) || (*epRankIdPtr >= *epWorldSizePtr)),
        OP_LOGE(nodeName_, "epRankId should be within the range of [0, %ld], get %ld", *epRankIdPtr, *epWorldSizePtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((*moeExpertNumPtr <= 0) || (*moeExpertNumPtr > MAX_MOE_EXPERT_NUM)),
        OP_LOGE(
            nodeName_, "moeExpertNum should be within the range of [0, %ld], get %ld", MAX_MOE_EXPERT_NUM,
            *moeExpertNumPtr),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchSetupTilingBase::GetRequiredAttrAndSetTilingData()
{
    OP_LOGD("GetRequiredAttrAndSetTilingData");
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName_, "attrs is null."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);

    // 判空
    OP_TILING_CHECK(groupEpPtr == nullptr, OP_LOGE(nodeName_, "groupEp is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName_, "epWorldSize is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName_, "epRankId is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName_, "moeExpertNum is null."), return ge::GRAPH_FAILED);

    if (CheckRequiredAttrValue() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 设置 tilingdata
    groupEp_ = string(groupEpPtr);
    tilingData_->moeDistributeDispatchSetupInfo.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
    tilingData_->moeDistributeDispatchSetupInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData_->moeDistributeDispatchSetupInfo.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckSharedExpertAttrValue()
{
    const uint32_t& sharedExpertNum = tilingData_->moeDistributeDispatchSetupInfo.sharedExpertNum;
    const uint32_t& sharedExpertRankNum = tilingData_->moeDistributeDispatchSetupInfo.sharedExpertRankNum;
    // 共享专家卡数>=共享专家数且可以整除
    if (sharedExpertRankNum == 0) {
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK(
        (sharedExpertNum == 0),
        OP_LOGE(
            nodeName_,
            "attribute must comply with sharedExpertNum != 0 when sharedExpertRankNum != 0, but got "
            "sharedExpertNum=%u, sharedExpertRankNum=%u.",
            sharedExpertNum, sharedExpertRankNum),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (sharedExpertNum > sharedExpertRankNum),
        OP_LOGE(
            nodeName_,
            "attribute must comply with sharedExpertNum <= sharedExpertRankNum, but got sharedExpertNum=%u, "
            "sharedExpertRankNum=%u.",
            sharedExpertNum, sharedExpertRankNum),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (sharedExpertRankNum % sharedExpertNum != 0),
        OP_LOGE(
            nodeName_,
            "attribute must comply with sharedExpertRankNum %% sharedExpertNum == 0, but got sharedExpertNum=%u, "
            "sharedExpertRankNum=%u.",
            sharedExpertNum, sharedExpertRankNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckOptionalAttrValue()
{
    auto attrs = context_->GetAttrs();
    auto expertShardTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_NUM_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    auto commTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_TYPE_INDEX);
    auto commAlgPtr = attrs->GetAttrPointer<char>(ATTR_COMM_ALG_INDEX);

    OP_TILING_CHECK(
        (*expertShardTypePtr != 0), OP_LOGE(nodeName_, "expertShardType only support 0, get %ld", *expertShardTypePtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((*sharedExpertNumPtr < 0) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM)),
        OP_LOGE(
            nodeName_, "sharedExpertNum should be within the range of [0, %ld], get %ld", MAX_SHARED_EXPERT_NUM,
            *sharedExpertNumPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((*sharedExpertRankNumPtr < 0) ||
         (*sharedExpertRankNumPtr > (tilingData_->moeDistributeDispatchSetupInfo.epWorldSize / 2))),
        OP_LOGE(
            nodeName_, "sharedExpertRankNum should be within the range of [0, %u], get %ld.",
            tilingData_->moeDistributeDispatchSetupInfo.epWorldSize / 2, *sharedExpertRankNumPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((*quantModePtr < UNQUANT) && (*quantModePtr > MX_QUANT)),
        OP_LOGE(nodeName_, "quantMode only support 0 to 4, get %ld.", *quantModePtr), return ge::GRAPH_FAILED);
    // globalBs 会在后面获取 BS 后再次校验
    OP_TILING_CHECK(
        (*globalBsPtr < 0), OP_LOGE(nodeName_, "globalBs should be 0 or maxBs * epWorldSize, get %ld", *globalBsPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*commTypePtr != URMA_COMM), OP_LOGE(nodeName_, "commType only support 2, get %ld.", *commTypePtr),
        return ge::GRAPH_FAILED);
    if (commAlgPtr != nullptr) {
        const std::string commAlg = std::string(commAlgPtr);
        OP_TILING_CHECK(
            (commAlg != ""), OP_LOGE(nodeName_, "commAlg should be null or empty string."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchSetupTilingBase::GetOptionalAttrAndSetTilingData()
{
    OP_LOGD("GetOptionalAttrAndSetTilingData");
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName_, "attrs is null."), return ge::GRAPH_FAILED);

    auto expertShardTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_NUM_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    auto commTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_TYPE_INDEX);

    // 判空
    OP_TILING_CHECK(
        expertShardTypePtr == nullptr, OP_LOGE(nodeName_, "expertShardType is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        sharedExpertNumPtr == nullptr, OP_LOGE(nodeName_, "sharedExpertNum is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName_, "sharedExpertRankNum is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE(nodeName_, "quantMode is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName_, "globalBs is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commTypePtr == nullptr, OP_LOGE(nodeName_, "commType is null."), return ge::GRAPH_FAILED);

    if (CheckOptionalAttrValue() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 校验共享专家限制
    if (CheckSharedExpertAttrValue() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 设置 tilingdata
    tilingData_->moeDistributeDispatchSetupInfo.expertShardType = static_cast<uint32_t>(*expertShardTypePtr);
    tilingData_->moeDistributeDispatchSetupInfo.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData_->moeDistributeDispatchSetupInfo.sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    tilingData_->moeDistributeDispatchSetupInfo.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData_->moeDistributeDispatchSetupInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);

    if (tilingData_->moeDistributeDispatchSetupInfo.quantMode != UNQUANT) {
        tilingData_->moeDistributeDispatchSetupInfo.isQuant = true;
    } else {
        tilingData_->moeDistributeDispatchSetupInfo.isQuant = false;
    }
    if (context_->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX) != nullptr) {
        tilingData_->moeDistributeDispatchSetupInfo.isActiveMask = true;
    } else {
        tilingData_->moeDistributeDispatchSetupInfo.isActiveMask = false;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchSetupTilingBase::GetComplexAttrAndSetTilingData()
{
    const uint32_t& epRankId = tilingData_->moeDistributeDispatchSetupInfo.epRankId;
    const uint32_t& epWorldSize = tilingData_->moeDistributeDispatchSetupInfo.epWorldSize;
    const uint32_t& sharedExpertNum = tilingData_->moeDistributeDispatchSetupInfo.sharedExpertNum;
    const uint32_t& sharedExpertRankNum = tilingData_->moeDistributeDispatchSetupInfo.sharedExpertRankNum;
    const uint32_t& moeExpertNum = tilingData_->moeDistributeDispatchSetupInfo.moeExpertNum;

    // localMoeExpertNum
    if (epRankId >= sharedExpertRankNum) {
        // MoE 专家卡
        tilingData_->moeDistributeDispatchSetupInfo.moeExpertPerRankNum =
            moeExpertNum / (epWorldSize - sharedExpertNum);
    } else {
        // 共享专家卡
        tilingData_->moeDistributeDispatchSetupInfo.moeExpertPerRankNum = 1U;
    }

    OP_TILING_CHECK(
        (moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0),
        OP_LOGE(
            nodeName_,
            "attribute must comply with moeExpertNum %% (epWorldSize - sharedExpertRankNum) == 0, but got moeExpertNum "
            "%u, epWorldSize %u, sharedExpertRankNum %u.",
            moeExpertNum, epWorldSize, sharedExpertRankNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckOneTensorDim(
    std::string name, TensorType tensortype, uint32_t index, uint32_t dims)
{
    const gert::StorageShape* shape;
    if (tensortype == TensorType::INPUT) {
        shape = context_->GetInputShape(index);
    } else if (tensortype == TensorType::OUTPUT) {
        shape = context_->GetOutputShape(index);
    } else if (tensortype == TensorType::OPTIONINPUT) {
        shape = context_->GetOptionalInputShape(index);
    } else {
        OP_LOGE(
            nodeName_, "TensorType Only Support input or output. type:%u, name:%s, index:%u", tensortype, name.c_str(),
            index);
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK(shape == nullptr, OP_LOGE(nodeName_, "%s is null.", name.c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        shape->GetStorageShape().GetDimNum() != dims,
        OP_LOGE(nodeName_, "%s got %lu dim, expected %u dim", name.c_str(), shape->GetStorageShape().GetDimNum(), dims),
        return ge::GRAPH_FAILED);

    for (uint32_t d = 0; d < dims; d++) {
        OP_LOGD(nodeName_, "%s %u dim = %ld", name.c_str(), d, shape->GetStorageShape().GetDim(d));
    }
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckInputTensorDim()
{
    OP_TILING_CHECK(
        CheckOneTensorDim("x", TensorType::INPUT, X_INDEX, TWO_DIMS) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName_, "x checkdim failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckOneTensorDim("expertIds", TensorType::INPUT, EXPERT_IDS_INDEX, TWO_DIMS) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName_, "expertIds checkdim failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckOptionalInputTensorDim()
{
    const gert::StorageShape* scalesStorageShape = context_->GetOptionalInputShape(SCALES_INDEX);
    const gert::StorageShape* xActiveMaskStorageShape = context_->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    uint64_t scalesCount = 0;
    if (scalesStorageShape != nullptr) {
        size_t scalesDimNum = scalesStorageShape->GetStorageShape().GetDimNum();
        const int64_t scalesDim0 = scalesStorageShape->GetStorageShape().GetDim(0);
        if (scalesDimNum == ONE_DIMS) {
            //A3不会进此分支
            auto YDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
            scalesCount = static_cast<uint64_t>(scalesDim0);
        } else {
            const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
            scalesCount = static_cast<uint64_t>(scalesDim0 * scalesDim1);
        }
    }
    if (xActiveMaskStorageShape != nullptr) {
        OP_TILING_CHECK(
            xActiveMaskStorageShape->GetStorageShape().GetDimNum() != ONE_DIMS,
            OP_LOGE(
                nodeName_, "xActiveMask's dim is %lu but should be 1!",
                xActiveMaskStorageShape->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
    }
    tilingData_->moeDistributeDispatchSetupInfo.scalesCount = scalesCount;
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckOutputTensorDim()
{
    OP_TILING_CHECK(
        CheckOneTensorDim("yOut", TensorType::OUTPUT, OUTPUT_Y_INDEX, TWO_DIMS) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName_, "yOut checkdim failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckOneTensorDim("expandIdxOut", TensorType::OUTPUT, OUTPUT_EXPAND_IDX_INDEX, ONE_DIMS) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName_, "expandIdxOut checkdim failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckOneTensorDim("commCmdInfoOut", TensorType::OUTPUT, OUTPUT_COMM_CMD_INFO_INDEX, ONE_DIMS) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName_, "commCmdInfoOut checkdim failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckTensorDim()
{
    OP_TILING_CHECK(
        CheckInputTensorDim() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName_, "Input param shape is invalid."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckOptionalInputTensorDim() != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName_, "Optional input param shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckOutputTensorDim() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName_, "Output param shape is invalid."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckTensorShapeRelation()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    auto expertIdsShape = context_->GetInputShape(EXPERT_IDS_INDEX);
    auto xActiveMaskShape = context_->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    auto scalesShape = context_->GetOptionalInputShape(SCALES_INDEX);

    // BS校验
    OP_TILING_CHECK(
        xStorageShape->GetStorageShape().GetDim(0) != expertIdsShape->GetStorageShape().GetDim(0),
        OP_LOGE(
            nodeName_, "x's dim0[%lu] should be equal to expertIds's dim0[%lu]",
            xStorageShape->GetStorageShape().GetDim(0), expertIdsShape->GetStorageShape().GetDim(0)),
        return ge::GRAPH_FAILED);
    if (xActiveMaskShape != nullptr) {
        OP_TILING_CHECK(
            xStorageShape->GetStorageShape().GetDim(0) != xActiveMaskShape->GetStorageShape().GetDim(0),
            OP_LOGE(
                nodeName_, "x's dim0[%lu] should be equal to xActiveMask's dim0[%lu]",
                xStorageShape->GetStorageShape().GetDim(0), xActiveMaskShape->GetStorageShape().GetDim(0)),
            return ge::GRAPH_FAILED);
    }

    // H校验
    if (scalesShape != nullptr) {
        OP_TILING_CHECK(
            xStorageShape->GetStorageShape().GetDim(1) != scalesShape->GetStorageShape().GetDim(1),
            OP_LOGE(
                nodeName_, "x's dim1[%lu] should be equal to scales's dim1[%lu]",
                xStorageShape->GetStorageShape().GetDim(1), scalesShape->GetStorageShape().GetDim(1)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckInputTensorDataType()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    auto expertIdsDesc = context_->GetInputDesc(EXPERT_IDS_INDEX);

    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName_, "x is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE(nodeName_, "expertIds is null."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        (xDesc->GetDataType() != ge::DT_FLOAT16) && (xDesc->GetDataType() != ge::DT_BF16),
        OP_LOGE(
            nodeName_, "Unsupported dataType, x only support float16 and bfloat16, but is %s!",
            Ops::Base::ToString(xDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (expertIdsDesc->GetDataType() != ge::DT_INT32),
        OP_LOGE(
            nodeName_, "Unsupported dataType, expertIds only support int32, but is %s!",
            Ops::Base::ToString(expertIdsDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckOptionalInputTensorDataType()
{
    auto scalesDesc = context_->GetOptionalInputDesc(SCALES_INDEX);
    auto xActiveMaskDesc = context_->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
    if (scalesDesc != nullptr) {
        OP_TILING_CHECK(
            (scalesDesc->GetDataType() != ge::DT_FLOAT),
            OP_LOGE(
                nodeName_, "Unsupported dataType, scales only support float, but is %s!",
                Ops::Base::ToString(scalesDesc->GetDataType()).c_str()),
            return ge::GRAPH_FAILED);
    }

    if (xActiveMaskDesc != nullptr) {
        OP_TILING_CHECK(
            (xActiveMaskDesc->GetDataType() != ge::DT_BOOL),
            OP_LOGE(
                nodeName_, "Unsupported dataType, xActiveMask only support bool, but is %s!",
                Ops::Base::ToString(xActiveMaskDesc->GetDataType()).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckOutputTensorDataType()
{
    auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    auto expandIdxDesc = context_->GetOutputDesc(OUTPUT_EXPAND_IDX_INDEX);
    auto commCmdInfoDesc = context_->GetOutputDesc(OUTPUT_COMM_CMD_INFO_INDEX);

    OP_TILING_CHECK(yDesc == nullptr, OP_LOGE(nodeName_, "yOut is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expandIdxDesc == nullptr, OP_LOGE(nodeName_, "expandIdxOut is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commCmdInfoDesc == nullptr, OP_LOGE(nodeName_, "commCmdInfoOut is null."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        (expandIdxDesc->GetDataType() != ge::DT_INT32),
        OP_LOGE(
            nodeName_, "Unsupported dataType, expandIdxOut only support int32, but is %s!",
            Ops::Base::ToString(expandIdxDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (commCmdInfoDesc->GetDataType() != ge::DT_INT32),
        OP_LOGE(
            nodeName_, "Unsupported dataType, commCmdInfoOut only support int32, but is %s!",
            Ops::Base::ToString(commCmdInfoDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    int64_t quantMode = tilingData_->moeDistributeDispatchSetupInfo.quantMode;
    OP_TILING_CHECK((quantMode == UNQUANT)
        && (yDesc->GetDataType() != context_->GetInputDesc(X_INDEX)->GetDataType()),
        OP_LOGE(
            nodeName_,
            "Invalid yOut's datatype for quantMode %ld. Only bf16/fp16 is supported, but got %s.",
            quantMode, Ops::Base::ToString(yDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((quantMode == STATIC_QUANT)
        && (yDesc->GetDataType() != ge::DT_HIFLOAT8) && (yDesc->GetDataType() != ge::DT_INT8),
        OP_LOGE(
            nodeName_, "Invalid yOut's datatype for quantMode %ld. Only int8/hif8 is supported, but got %s.",
            quantMode, Ops::Base::ToString(yDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((quantMode == PERTOKEN_DYNAMIC_QUANT)
        && (yDesc->GetDataType() != ge::DT_INT8) && (yDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN)
        && (yDesc->GetDataType() != ge::DT_FLOAT8_E5M2),
        OP_LOGE(
            nodeName_, "Invalid yOut's datatype for quantMode %ld. Only int8/fp8_e4m3fn/fp8_e5m2 is supported, but got %s.",
            quantMode, Ops::Base::ToString(yDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((quantMode == PERGROUP_DYNAMIC_QUANT) || (quantMode == MX_QUANT))
        && (yDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) && (yDesc->GetDataType() != ge::DT_FLOAT8_E5M2),
        OP_LOGE(
            nodeName_, "Invalid yOut's datatype for quantMode %ld. Only fp8_e4m3fn/fp8_e5m2 is supported, but got %s.",
            quantMode, Ops::Base::ToString(yDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckTensorDataType()
{
    if (!((CheckInputTensorDataType() == ge::GRAPH_SUCCESS) &&
          (CheckOptionalInputTensorDataType() == ge::GRAPH_SUCCESS) &&
          (CheckOutputTensorDataType() == ge::GRAPH_SUCCESS))) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckTensorShapeSizeAndSetTilingData()
{
    auto* xStorageShape = context_->GetInputShape(X_INDEX);
    auto* expertIdsShape = context_->GetInputShape(EXPERT_IDS_INDEX);

    int64_t H = xStorageShape->GetStorageShape().GetDim(1);
    int64_t BS = xStorageShape->GetStorageShape().GetDim(0);
    int64_t K = expertIdsShape->GetStorageShape().GetDim(1);

    uint32_t& epWorldSize = tilingData_->moeDistributeDispatchSetupInfo.epWorldSize;
    uint32_t& moeExpertNum = tilingData_->moeDistributeDispatchSetupInfo.moeExpertNum;
    uint32_t& globalBs = tilingData_->moeDistributeDispatchSetupInfo.globalBs;

    OP_TILING_CHECK(
        (H < MIN_H) || (H > MAX_H), OP_LOGE(nodeName_, "H[%ld] should be within range [%ld, %ld].", H, MIN_H, MAX_H),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (BS <= 0) || (BS > MAX_BS), OP_LOGE(nodeName_, "BS[%ld] should be within range (0, %ld].", BS, MAX_BS),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (K < 0) || (K > MAX_K) || (K > static_cast<int64_t>(moeExpertNum)),
        OP_LOGE(nodeName_, "K[%ld] should be within range [0, min(16, moeExpertNum[%u])].", K, moeExpertNum),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        (globalBs != 0) && ((globalBs < BS * epWorldSize) || (globalBs > MAX_BS * epWorldSize)),
        OP_LOGE(
            nodeName_, "globalBs[%u] should be >= BS * epWorldSize[%lu] and <= 512 * epWorldSize, or = 0.", globalBs,
            BS * epWorldSize),
        return ge::GRAPH_FAILED);

    tilingData_->moeDistributeDispatchSetupInfo.h = static_cast<uint32_t>(H);
    tilingData_->moeDistributeDispatchSetupInfo.bs = static_cast<uint32_t>(BS);
    tilingData_->moeDistributeDispatchSetupInfo.k = static_cast<uint32_t>(K);

    if (CheckComplexTensorShapeSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}
const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckComplexTensorShapeSize()
{
    int64_t sharedExpertNum = tilingData_->moeDistributeDispatchSetupInfo.sharedExpertNum;
    int64_t moeExpertNum = tilingData_->moeDistributeDispatchSetupInfo.moeExpertNum;

    int64_t BS = tilingData_->moeDistributeDispatchSetupInfo.bs;
    int64_t K = tilingData_->moeDistributeDispatchSetupInfo.k;

    auto scalesShape = context_->GetOptionalInputShape(SCALES_INDEX);
    if (scalesShape != nullptr) {
        int64_t scalesDim0GoldenSize = moeExpertNum + sharedExpertNum;
        int64_t scalesDim0Size = context_->GetOptionalInputShape(SCALES_INDEX)->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(
            scalesDim0Size != scalesDim0GoldenSize,
            OP_LOGE(
                nodeName_, "scales's dim0[%lu] should equal to moeExpertNum + sharedExpertNum[%lu].", scalesDim0Size,
                scalesDim0GoldenSize),
            return ge::GRAPH_FAILED);
    }

    int64_t yOutDim0GoldenSize = (BS * (K + sharedExpertNum));
    int64_t yOutDim0Size = context_->GetOutputShape(OUTPUT_Y_INDEX)->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(
        yOutDim0Size != yOutDim0GoldenSize,
        OP_LOGE(
            nodeName_, "yOut's dim1[%lu] should equal to (BS * (K + sharedExpertNum))[%lu].", yOutDim0Size,
            yOutDim0GoldenSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckCalcTensorShapeSizeAndSetTilingData()
{
    auto yOutShape = context_->GetOutputShape(OUTPUT_Y_INDEX);
    auto expandIdxOutShape = context_->GetOutputShape(OUTPUT_EXPAND_IDX_INDEX);
    auto commCmdInfoOutShape = context_->GetOutputShape(OUTPUT_COMM_CMD_INFO_INDEX);
    int64_t K = tilingData_->moeDistributeDispatchSetupInfo.k;
    int64_t BS = tilingData_->moeDistributeDispatchSetupInfo.bs;
    int64_t H = tilingData_->moeDistributeDispatchSetupInfo.h;

    int64_t epWorldSize = tilingData_->moeDistributeDispatchSetupInfo.epWorldSize;
    int64_t sharedExpertNum = tilingData_->moeDistributeDispatchSetupInfo.sharedExpertNum;
    int64_t localExpertNum = tilingData_->moeDistributeDispatchSetupInfo.moeExpertPerRankNum;

    int64_t tokenMsgSize = yOutShape->GetStorageShape().GetDim(1);
    int64_t expandIdxOutSize = expandIdxOutShape->GetStorageShape().GetDim(0);
    int64_t commCmdInfoOutSize = commCmdInfoOutShape->GetStorageShape().GetDim(0);
    int64_t quantMode = tilingData_->moeDistributeDispatchSetupInfo.quantMode;
    int64_t tokenMsgSizeGolden = 0;
    if (quantMode == UNQUANT) {
        tokenMsgSizeGolden = ops::CeilAlign(H, ALIGN_256);
    } else if (quantMode == STATIC_QUANT) {
        tokenMsgSizeGolden = ops::CeilAlign(H, ALIGN_512);
    } else if (quantMode == PERTOKEN_DYNAMIC_QUANT) {
        tokenMsgSizeGolden = ops::CeilAlign(ops::CeilAlign(H, ALIGN_32) + QUANT_HS_OFFSET, ALIGN_512);
    } else if (quantMode == PERGROUP_DYNAMIC_QUANT) {
        tokenMsgSizeGolden = ops::CeilAlign(ops::CeilAlign(H, ALIGN_128) + ops::CeilDiv(H, ALIGN_128), ALIGN_512);
    } else if (quantMode == MX_QUANT) {
        tokenMsgSizeGolden = ops::CeilAlign(ops::CeilAlign(H, ALIGN_256) + ops::CeilAlign(ops::CeilDiv(H, ALIGN_32), EVEN_ALIGN), ALIGN_512);
    }

    OP_TILING_CHECK(
        tokenMsgSize != tokenMsgSizeGolden,
        OP_LOGE(nodeName_, "yOut's dim1[%ld] should be equal to tokenMsgSize[%ld]", tokenMsgSize, tokenMsgSizeGolden),
        return ge::GRAPH_FAILED);

    const int64_t expandIdxOutSizeGolden = BS * K;
    OP_TILING_CHECK(
        expandIdxOutSize != expandIdxOutSizeGolden,
        OP_LOGE(
            nodeName_, "expandIdxOut's dim0[%ld] should be equal to expandIdxOutSize[%ld]", expandIdxOutSize,
            expandIdxOutSizeGolden),
        return ge::GRAPH_FAILED);

    const int64_t commCmdInfoOutSizeGolden =
        (BS * (K + sharedExpertNum) + epWorldSize * localExpertNum) * COMM_CMD_INFO_MULTIPLY;
    OP_TILING_CHECK(
        commCmdInfoOutSize != commCmdInfoOutSizeGolden,
        OP_LOGE(
            nodeName_, "commCmdInfo's dim1[%ld] should be equal to commCmdInfoOutSize[%ld]", commCmdInfoOutSize,
            commCmdInfoOutSizeGolden),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void MoeDistributeDispatchSetupTilingBase::SetTilingKey()
{
    /*
     * tilingkey说明
     * 4位的十进制数
     * 第1位（个位）：quantMode:
     *     0: 不量化, 1: 静态量化, 2: 动态量化
     * 第2位（十位）：是否有smoothScale:
     *     0: 无, 1: 有
     * 第3位（百位）：0
     * 第4位（千位）：1
     */
    uint64_t tilingKey = INIT_TILINGKEY;
    tilingKey += static_cast<uint64_t>(tilingData_->moeDistributeDispatchSetupInfo.quantMode);
    // 这里的是判断scales的
    if (context_->GetOptionalInputShape(SCALES_INDEX) != nullptr) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_SCALES);
    }
    context_->SetTilingKey(tilingKey);
    return;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckHcclBuffSize()
{
    const uint64_t hcclBuffSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize() / 2;
    OP_TILING_CHECK(
        hcclBuffSize < MIN_AVAILABLE_BUFF_SIZE,
        OP_LOGE(nodeName_, "HCCL_BUFFSIZE [%ld] < [%ld].", hcclBuffSize, MIN_AVAILABLE_BUFF_SIZE),
        return ge::GRAPH_FAILED);

    uint32_t& epWorldSize = tilingData_->moeDistributeDispatchSetupInfo.epWorldSize;
    uint32_t& sharedExpertNum = tilingData_->moeDistributeDispatchSetupInfo.sharedExpertNum;
    uint32_t& globalBs = tilingData_->moeDistributeDispatchSetupInfo.globalBs;
    uint32_t& BS = tilingData_->moeDistributeDispatchSetupInfo.bs;
    uint32_t& H = tilingData_->moeDistributeDispatchSetupInfo.h;
    uint32_t& K = tilingData_->moeDistributeDispatchSetupInfo.k;
    uint32_t& localExpertNum = tilingData_->moeDistributeDispatchSetupInfo.moeExpertPerRankNum;
    uint32_t maxBs = BS;
    if (globalBs != 0) {
        maxBs = globalBs / epWorldSize;
    }
    uint64_t align = ops::CeilAlign(
        (ops::CeilAlign(2U * H, static_cast<uint32_t>(ALIGN_32)) + static_cast<uint64_t>(HCCL_BUFFER_SIZE)), static_cast<uint64_t>(ALIGN_512));
    const uint64_t hcclBuffSizeGolden =
        (MIN_AVAILABLE_BUFF_SIZE * localExpertNum * maxBs * epWorldSize * align) +
        (K + sharedExpertNum) * maxBs * ops::CeilAlign(2U * H, static_cast<uint32_t>(ALIGN_512));

    OP_TILING_CHECK(
        hcclBuffSize < hcclBuffSizeGolden,
        OP_LOGE(nodeName_, "HCCL_BUFFSIZE [%lu] < [%lu].", hcclBuffSize, hcclBuffSizeGolden), return ge::GRAPH_FAILED);

    tilingData_->moeDistributeDispatchSetupInfo.totalWinSize = hcclBuffSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchSetupTilingBase::SetWorkspace()
{
    size_t* workspace = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(
        workspace == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName_, "get workspace failed"),
        return ge::GRAPH_FAILED);
    workspace[0] = static_cast<size_t>(SYSTEM_NEED_WORKSPACE) + SDMA_NEED_WORKSPACE;
    OP_LOGD(nodeName_, "workspce[0] size is %lu", workspace[0]);
    return ge::GRAPH_SUCCESS;
}

void MoeDistributeDispatchSetupTilingBase::SetPlatformInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t blockDim = 1U;
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context_->SetBlockDim(blockDim);
    context_->SetAicpuBlockDim(AICPUNUM);
    tilingData_->moeDistributeDispatchSetupInfo.totalUbSize = ubSize;
    tilingData_->moeDistributeDispatchSetupInfo.aivNum = aivNum;
    context_->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    OP_LOGD(nodeName_, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    uint32_t sdmaUsedStreamPerCore = LOCAL_STREAM_MAX_NUM / aivNum;
    tilingData_->moeDistributeDispatchSetupInfo.sdmaUsedStreamPerCore = sdmaUsedStreamPerCore;
}

void MoeDistributeDispatchSetupTilingBase::SetHcommCfg() {};

ge::graphStatus MoeDistributeDispatchSetupTilingBase::MoeDistributeDispatchSetupTilingFuncImpl()
{
    OP_LOGD(nodeName_, "MoeDistributeDispatchSetupTilingFunc start");
    tilingData_ = context_->GetTilingData<MoeDistributeDispatchSetupTilingData>();
    OP_TILING_CHECK(tilingData_ == nullptr, OP_LOGE(nodeName_, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    // 获取入参属性
    if (!((GetRequiredAttrAndSetTilingData() == ge::GRAPH_SUCCESS) &&
          (GetOptionalAttrAndSetTilingData() == ge::GRAPH_SUCCESS) &&
          (GetComplexAttrAndSetTilingData() == ge::GRAPH_SUCCESS))) {
        return ge::GRAPH_FAILED;
    }

    if (!((CheckTensorDataType() == ge::GRAPH_SUCCESS) && (CheckTensorDim() == ge::GRAPH_SUCCESS) &&
          (CheckTensorShapeRelation() == ge::GRAPH_SUCCESS) &&
          (CheckTensorShapeSizeAndSetTilingData() == ge::GRAPH_SUCCESS) &&
          (CheckCalcTensorShapeSizeAndSetTilingData() == ge::GRAPH_SUCCESS))) {
        return ge::GRAPH_FAILED;
    }

    if (CheckHcclBuffSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SetPlatformInfo();

    if (SetWorkspace() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SetTilingKey();
    SetHcommCfg();
    PrintTilingDataInfo();
    OP_LOGD(nodeName_, "MoeDistributeDispatchSetupTilingFunc success");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling