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
 * \file moe_distribute_combine_teardown_tiling_arch32.cpp
 * \brief
 */

#include "mc2_log.h"
#include "moe_distribute_combine_teardown_tiling_arch32.h"

namespace {
constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 4;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 5;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 6;
constexpr uint32_t ATTR_COMM_QUANT_MODE_INDEX = 8;
constexpr uint32_t ATTR_COMM_TYPE_INDEX = 9;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr int64_t MIN_GROUP_EP_SIZE = 2;
constexpr int64_t MAX_GROUP_EP_SIZE = 384;
constexpr int64_t NON_QUANT = 0;
constexpr int64_t DYNAMIC_QUANT = 2;
constexpr int64_t MAX_MOE_EXPERT_NUM = 512;
constexpr int64_t SDMA_COMM = 0;
constexpr int64_t URMA_COMM = 2;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr uint32_t SDMA_NEED_WORKSPACE = 16U * 1024 * 1024;
} // namespace

namespace MC2Tiling {

ge::graphStatus MoeDistributeCombineTeardownTilingA3::CheckAttrsWithoutRelation()
{
    auto attrs = context_->GetAttrs();

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto expertShardTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_QUANT_MODE_INDEX);
    auto commTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_TYPE_INDEX);

    OP_TILING_CHECK((strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE(nodeName_, "groupEp length should be [1, %lu), get %lu.", MAX_GROUP_NAME_LENGTH,
                            strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH)),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epWorldSizePtr < MIN_GROUP_EP_SIZE) || (*epWorldSizePtr > MAX_GROUP_EP_SIZE),
                    OP_LOGE(nodeName_, "epWorldSize should be [%lu, %lu], get %lu", MIN_GROUP_EP_SIZE,
                            MAX_GROUP_EP_SIZE, *epWorldSizePtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*expertShardTypePtr != 0,
                    OP_LOGE(nodeName_, "expertShardType only support 0, get %lu", *expertShardTypePtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*sharedExpertNumPtr < 0) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM),
                    OP_LOGE(nodeName_, "sharedExpertNum shoud be within the range of [0, %lu], get %lu",
                            MAX_SHARED_EXPERT_NUM, *sharedExpertNumPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*quantModePtr != NON_QUANT) && (*quantModePtr != DYNAMIC_QUANT),
        OP_LOGE(nodeName_, "quantMode shoud be equal to %lu or %lu, get %lu", NON_QUANT, DYNAMIC_QUANT, *quantModePtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*commTypePtr < SDMA_COMM) || (*commTypePtr > URMA_COMM),
        OP_LOGE(nodeName_, "commType only support [%lu, %lu], get [%lu]", SDMA_COMM, URMA_COMM, *commTypePtr),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingA3::CheckAttrsComplex()
{
    auto attrs = context_->GetAttrs();

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= *epWorldSizePtr),
                    OP_LOGE(nodeName_, "ep_rankId shoud be within the range of epWorldSize[0, %lu), get %lu",
                            *epWorldSizePtr, *epRankIdPtr),
                    return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    OP_TILING_CHECK((*moeExpertNumPtr <= 0) || (*moeExpertNumPtr > MAX_MOE_EXPERT_NUM),
                    OP_LOGE(nodeName_, "moeExpertNum shoud be within the range of (0, %lu], get %lu",
                            MAX_MOE_EXPERT_NUM, *moeExpertNumPtr),
                    return ge::GRAPH_FAILED);

    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    OP_TILING_CHECK((*sharedExpertRankNumPtr < 0) || (*sharedExpertRankNumPtr > *epWorldSizePtr / 2),
                    OP_LOGE(nodeName_, "sharedExpertRankNum shoud be within the range of [0, %lu], get %lu",
                            *epWorldSizePtr / 2, *sharedExpertRankNumPtr),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        (*moeExpertNumPtr % (*epWorldSizePtr - *sharedExpertRankNumPtr) != 0),
        OP_LOGE(nodeName_,
                "moeExpertNum should follow moeExpertNum %% (epWorldSize - sharedExpertRankNum) = 0, got %ld",
                (*moeExpertNumPtr % (*epWorldSizePtr - *sharedExpertRankNumPtr))),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingA3::SetHcommCfg()
{
    const char *nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeCombine groupEp = %s", groupEp_.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp_, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tilingData_->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData_->mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}
} // namespace MC2Tiling
