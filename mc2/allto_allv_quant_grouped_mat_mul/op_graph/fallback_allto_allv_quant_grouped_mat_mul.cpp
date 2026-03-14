/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file fallback_allto_allv_quant_grouped_mat_mul.cpp
 * \brief fallback function of op AlltoAllvQuantGroupedMatMul
 */
#include "fallback/fallback.h"
#include "op_mc2.h"
#include "mc2_log.h"

namespace fallback {
constexpr size_t ATTR_K_GROUP = 0;
constexpr size_t ATTR_K_EP_WORLD_SIZE = 1;
constexpr size_t ATTR_K_SEND_COUNTS = 2;
constexpr size_t ATTR_K_RECV_COUNTS = 3;
constexpr size_t ATTR_K_TRANS_GMM_WEIGHT = 4;
constexpr size_t ATTR_K_TRANS_MM_WEIGHT = 5;
constexpr size_t ATTR_K_PERMUTE_OUT_FLAG = 6;

constexpr size_t INPUT_K_GMM_X = 0;
constexpr size_t INPUT_K_GMM_WEIGHT = 1;
constexpr size_t INPUT_K_SEND_COUNTS_TENSOR = 2;
constexpr size_t INPUT_K_RECV_COUNTS_TENSOR = 3;
constexpr size_t INPUT_K_MM_X = 4;
constexpr size_t INPUT_K_MM_WEIGHT = 5;

constexpr size_t OUTPUT_K_GMM_Y = 0;
constexpr size_t OUTPUT_K_MM_Y = 1;
constexpr size_t OUTPUT_K_PERMUTE_OUT = 2;

// 输入参数和属性的校验
static ge::graphStatus CheckInputsAndAttrs(const gert::Tensor *gmmX, const gert::Tensor *gmmWeight, const char *group,
    const int64_t *epWorldSize, const bool *transGmmWeight)
{
    OPS_ERR_IF(gmmX == nullptr, OP_LOGE("AlltoAllvQuantGroupedMatMulFallback", "gmmX is null"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(gmmWeight == nullptr, OP_LOGE("AlltoAllvQuantGroupedMatMulFallback", "gmmWeight is null"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(group == nullptr, OP_LOGE("AlltoAllvQuantGroupedMatMulFallback", "group is null"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(epWorldSize == nullptr, OP_LOGE("AlltoAllvQuantGroupedMatMulFallback", "epWorldSize is null"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(transGmmWeight == nullptr, OP_LOGE("AlltoAllvQuantGroupedMatMulFallback", "transGmmWeight is null"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 解析 sendCounts 和 recvCounts
static ge::graphStatus ParseSendRecvCounts(const gert::TypedContinuousVector<int64_t> *sendCounts,
    const gert::TypedContinuousVector<int64_t> *recvCounts, std::vector<int64_t> &actSendCountsSeqArray,
    std::vector<int64_t> &actRecvCountsSeqArray)
{
    OPS_ERR_IF(sendCounts == nullptr, OP_LOGE("AlltoAllvQuantGroupedMatMulFallback", "sendCounts is null"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(recvCounts == nullptr, OP_LOGE("AlltoAllvQuantGroupedMatMulFallback", "recvCounts is null"),
        return ge::GRAPH_FAILED);

    const int64_t *actSendSeqData = sendCounts->GetData();
    const size_t sendLen = sendCounts->GetSize();
    for (size_t i = 0UL; i < sendLen; i++) {
        actSendCountsSeqArray.push_back(actSendSeqData[i]);
    }

    const int64_t *actRecvSeqData = recvCounts->GetData();
    const size_t recvLen = recvCounts->GetSize();
    for (size_t i = 0UL; i < recvLen; i++) {
        actRecvCountsSeqArray.push_back(actRecvSeqData[i]);
    }

    return ge::GRAPH_SUCCESS;
}

// 校验输出参数
static ge::graphStatus CheckOutputTensors(const gert::Tensor *gmmY, const gert::Tensor *&mmY,
    const gert::Tensor *&permuteOut)
{
    OPS_ERR_IF(gmmY == nullptr, OP_LOGE("AlltoAllvQuantGroupedMatMulFallback", "gmmY is null"),
        return ge::GRAPH_FAILED);
    if ((permuteOut != nullptr) && (permuteOut->GetStorageShape().GetDimNum() == 0)) {
        permuteOut = nullptr;
    }
    if ((mmY != nullptr) && (mmY->GetStorageShape().GetDimNum() == 0)) {
        mmY = nullptr;
    }

    return ge::GRAPH_SUCCESS;
}

// 主执行函数
static ge::graphStatus AlltoAllvQuantGroupedMatMulExecuteFunc(gert::OpExecuteContext *host_api_ctx)
{
    OPS_LOG_D("AlltoAllvQuantGroupedMatMulFallback", "Start AlltoAllvQuantGroupedMatMulFallback.");
    OPS_ERR_IF(host_api_ctx == nullptr, OP_LOGE("AlltoAllvQuantGroupedMatMulFallback", "host_api_ctx is null"),
        return ge::GRAPH_FAILED);
    const gert::Tensor *gmmX = host_api_ctx->GetInputTensor(INPUT_K_GMM_X);
    const gert::Tensor *gmmWeight = host_api_ctx->GetInputTensor(INPUT_K_GMM_WEIGHT);
    const gert::Tensor *sendCountsTensor =
        host_api_ctx->GetOptionalInputTensor(INPUT_K_SEND_COUNTS_TENSOR);
    const gert::Tensor *recvCountsTensor =
        host_api_ctx->GetOptionalInputTensor(INPUT_K_RECV_COUNTS_TENSOR);
    const gert::Tensor *mmX = host_api_ctx->GetOptionalInputTensor(INPUT_K_MM_X);
    const gert::Tensor *mmWeight =
        host_api_ctx->GetOptionalInputTensor(INPUT_K_MM_WEIGHT);
    const gert::RuntimeAttrs *attrs = host_api_ctx->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OP_LOGE("AlltoAllvQuantGroupedMatMulFallback", "attrs is null"),
        return ge::GRAPH_FAILED);
    const char *group = attrs->GetStr(ATTR_K_GROUP);
    const int64_t *epWorldSize = attrs->GetInt(ATTR_K_EP_WORLD_SIZE);
    const gert::TypedContinuousVector<int64_t> *sendCounts =
        attrs->GetListInt(ATTR_K_SEND_COUNTS);
    const gert::TypedContinuousVector<int64_t> *recvCounts =
        attrs->GetListInt(ATTR_K_RECV_COUNTS);
    const bool *transGmmWeight = attrs->GetBool(ATTR_K_TRANS_GMM_WEIGHT);
    const bool *transMmWeight = attrs->GetBool(ATTR_K_TRANS_MM_WEIGHT);
    const bool *permuteOutFlag = attrs->GetBool(ATTR_K_PERMUTE_OUT_FLAG);
    if (CheckInputsAndAttrs(gmmX, gmmWeight, group, epWorldSize, transGmmWeight) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    std::vector<int64_t> actSendCountsSeqArray;
    std::vector<int64_t> actRecvCountsSeqArray;
    if (ParseSendRecvCounts(sendCounts, recvCounts, actSendCountsSeqArray, actRecvCountsSeqArray) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto gmmY = host_api_ctx->GetOutputTensor(OUTPUT_K_GMM_Y);
    auto mmY = host_api_ctx->GetOutputTensor(OUTPUT_K_MM_Y);
    auto permuteOut = host_api_ctx->GetOutputTensor(OUTPUT_K_PERMUTE_OUT);
    if (CheckOutputTensors(gmmY, mmY, permuteOut) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const auto apiRet = EXEC_OPAPI_CMD(aclnnAlltoAllvQuantGroupedMatMul, gmmX, gmmWeight, sendCountsTensor,
        recvCountsTensor, mmX, mmWeight, group, *epWorldSize, actSendCountsSeqArray, actRecvCountsSeqArray,
        *transGmmWeight, *transMmWeight, *permuteOutFlag, gmmY, mmY, permuteOut);
    OPS_ERR_IF(apiRet != ge::GRAPH_SUCCESS,
        OP_LOGE("AlltoAllvQuantGroupedMatMulFallback", "Aclnn api error code %u", apiRet), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP(AlltoAllvQuantGroupedMatMul).OpExecuteFunc(AlltoAllvQuantGroupedMatMulExecuteFunc);
} // namespace fallback