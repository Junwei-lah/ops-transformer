/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <algorithm>
#include <unordered_map>
#include "kv_quant_sparse_flash_attention_pioneer.h"
#include "aclnn_kv_quant_sparse_flash_attention_pioneer.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static bool kvQuantSparseFlashAttentionPioneerCheckDataType(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *blockTableOptional,
    const aclTensor *out)
{
    auto qDtype = query->GetDataType();
    auto kDtype = key->GetDataType();
    auto vDtype = value->GetDataType();
    auto sparseIndDtype = sparseIndices->GetDataType();
    auto blockTableDtype = blockTableOptional->GetDataType();
    auto outDtype = out->GetDataType();

    static const std::unordered_map<DataType, std::vector<DataType>> dTypeMappingKv = {
        {DataType::DT_FLOAT8_E4M3FN, {DataType::DT_FLOAT8_E4M3FN}},
        {DataType::DT_HIFLOAT8, {DataType::DT_HIFLOAT8}},
    };
    if (dTypeMappingKv.find(kDtype) == dTypeMappingKv.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "key data type is invalid, please check.");
        return false;
    } else {
        auto validVDtypeList = dTypeMappingKv.at(kDtype);
        if (std::find(validVDtypeList.begin(), validVDtypeList.end(), vDtype) == validVDtypeList.end()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "value data type is invalid, please check.");
            return false;
        }
    }
    if (sparseIndDtype != DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sparseIndices data type must be int32, please check.");
        return false;
    }
    if (blockTableDtype != DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockTable data type must be int32, please check.");
        return false;
    }
    if (outDtype != qDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out data type must be equal qDtype, please check.");
        return false;
    }
    return true;
}

aclnnStatus kvQuantSparseFlashAttentionPioneerCheckTensorNull(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const char *layoutQueryOptional,
    const char *layoutKvOptional,
    const aclTensor *out,
    uint64_t *workspaceSize)
{
    // 参数指针判空
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(sparseIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(layoutQueryOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(layoutKvOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus kvQuantSparseFlashAttentionPioneerContiguous(
    const aclTensor *&query,
    const aclTensor *&sparseIndices,
    const aclTensor *&keyDequantScaleOptional,
    const aclTensor *&valueDequantScaleOptional,
    const aclTensor *&blockTableOptional,
    const aclTensor *&actualSeqLengthsQueryOptional,
    const aclTensor *&actualSeqLengthsKvOptional,
    const aclTensor *keySinkOptional,
    const aclTensor *valueSinkOptional,
    aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    sparseIndices = l0op::Contiguous(sparseIndices, executor);
    CHECK_RET(sparseIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (keyDequantScaleOptional) {
        keyDequantScaleOptional = l0op::Contiguous(keyDequantScaleOptional, executor);
        CHECK_RET(keyDequantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (valueDequantScaleOptional) {
        valueDequantScaleOptional = l0op::Contiguous(valueDequantScaleOptional, executor);
        CHECK_RET(valueDequantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (blockTableOptional) {
        blockTableOptional = l0op::Contiguous(blockTableOptional, executor);
        CHECK_RET(blockTableOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (actualSeqLengthsQueryOptional) {
        actualSeqLengthsQueryOptional = l0op::Contiguous(actualSeqLengthsQueryOptional, executor);
        CHECK_RET(actualSeqLengthsQueryOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (actualSeqLengthsKvOptional) {
        actualSeqLengthsKvOptional = l0op::Contiguous(actualSeqLengthsKvOptional, executor);
        CHECK_RET(actualSeqLengthsKvOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (keySinkOptional) {
        keySinkOptional = l0op::Contiguous(keySinkOptional, executor);
        CHECK_RET(keySinkOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (valueSinkOptional) {
        valueSinkOptional = l0op::Contiguous(valueSinkOptional, executor);
        CHECK_RET(valueSinkOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static const aclTensor* GetTensorContiguous(const aclTensor *tensor, aclOpExecutor *executor, const char *tensorName)
{
    if (tensor == nullptr) {
        return nullptr;
    }
    if (!IsContiguous(tensor)) {
        tensor = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetStorageShape(),
                                            tensor->GetViewStrides(), tensor->GetViewOffset());
    } else {
        tensor = l0op::Contiguous(tensor, executor);
    }

    CHECK_RET(tensor != nullptr, nullptr);
    return tensor;
}

aclnnStatus aclnnKvQuantSparseFlashAttentionPioneerGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *keyDequantScaleOptional,
    const aclTensor *valueDequantScaleOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *keySinkOptional,
    const aclTensor *valueSinkOptional,
    double scaleValue,
    int64_t keyQuantMode,
    int64_t valueQuantMode,
    int64_t sparseBlockSize,
    const char *layoutQueryOptional,
    const char *layoutKvOptional,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t attentionMode,
    int64_t quantScaleRepoMode,
    int64_t tileSize,
    int64_t ropeHeadDim,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_RET(kvQuantSparseFlashAttentionPioneerCheckTensorNull(query, key, value, sparseIndices, layoutQueryOptional,
            layoutKvOptional, out, workspaceSize) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    
    CHECK_RET(kvQuantSparseFlashAttentionPioneerCheckDataType(query, key, value, sparseIndices, blockTableOptional,
            out), ACLNN_ERR_PARAM_INVALID);

    L2_DFX_PHASE_1(aclnnKvQuantSparseFlashAttentionPioneer,
            DFX_IN(query, key, value, sparseIndices, keyDequantScaleOptional, valueDequantScaleOptional,
            blockTableOptional, actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional, keySinkOptional,
            valueSinkOptional, scaleValue, keyQuantMode, valueQuantMode, sparseBlockSize, layoutQueryOptional,
            layoutKvOptional, sparseMode, preTokens, nextTokens, attentionMode, quantScaleRepoMode, tileSize, ropeHeadDim),
            DFX_OUT(out));
    
    // 获取executor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    // 非连续转连续
    CHECK_RET(kvQuantSparseFlashAttentionPioneerContiguous(query, sparseIndices, keyDequantScaleOptional,
            valueDequantScaleOptional, blockTableOptional, actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional,
            keySinkOptional, valueSinkOptional, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const aclTensor *newKey = GetTensorContiguous(key, l0Executor, "key");
    const aclTensor *newValue = GetTensorContiguous(value, l0Executor, "value");

    // 调用L0接口获得输出
    auto l0KvQuantSparseFlashAttentionPioneerOuts = l0op::KvQuantSparseFlashAttentionPioneer(query, newKey, newValue, sparseIndices,
            keyDequantScaleOptional, valueDequantScaleOptional, blockTableOptional, actualSeqLengthsQueryOptional,
            actualSeqLengthsKvOptional, keySinkOptional, valueSinkOptional, scaleValue, keyQuantMode, valueQuantMode, sparseBlockSize,
            layoutQueryOptional, layoutKvOptional, sparseMode, preTokens, nextTokens, attentionMode, quantScaleRepoMode, tileSize,
            ropeHeadDim, l0Executor);

    // 检查输出
    if (l0KvQuantSparseFlashAttentionPioneerOuts == nullptr) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(l0KvQuantSparseFlashAttentionPioneerOuts, out, l0Executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKvQuantSparseFlashAttentionPioneer(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnKvQuantSparseFlashAttentionPioneer);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

}  // namespace

#ifdef __cplusplus
}
#endif