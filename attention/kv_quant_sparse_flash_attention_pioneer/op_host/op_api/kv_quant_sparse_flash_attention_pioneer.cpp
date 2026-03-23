/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(KvQuantSparseFlashAttentionPioneer);

const aclTensor *KvQuantSparseFlashAttentionPioneer(
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
        aclOpExecutor *executor)
{
    int64_t keyBlockStride = 0;
    auto blockStrides = key->GetViewStrides();
    keyBlockStride = blockStrides[0];

    // L0接口时延统计以及入参打印
    L0_DFX(KvQuantSparseFlashAttentionPioneer, query, key, value, sparseIndices, keyDequantScaleOptional,
        valueDequantScaleOptional, blockTableOptional, actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional,
        keySinkOptional, valueSinkOptional, scaleValue, keyQuantMode, valueQuantMode, sparseBlockSize,
        layoutQueryOptional, layoutKvOptional, sparseMode, preTokens, nextTokens, attentionMode, quantScaleRepoMode,
        tileSize, ropeHeadDim, keyBlockStride);
    
    // 构造输出
    auto output = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);

    // 调用inferShape
    auto ret = INFER_SHAPE(KvQuantSparseFlashAttentionPioneer,
                            OP_INPUT(query, key, value, sparseIndices, keyDequantScaleOptional, valueDequantScaleOptional,
                                blockTableOptional, actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional,
                                keySinkOptional, valueSinkOptional),
                            OP_OUTPUT(output),
                            OP_ATTR(scaleValue, keyQuantMode, valueQuantMode, sparseBlockSize, layoutQueryOptional,
                                layoutKvOptional, sparseMode, preTokens, nextTokens, attentionMode,
                                quantScaleRepoMode, tileSize, ropeHeadDim, keyBlockStride));
    
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "KvQuantSparseFlashAttentionPioneer InferShape failed.");
        return nullptr;
    }

    // 发起aicore任务
    ret = ADD_TO_LAUNCHER_LIST_AICORE(KvQuantSparseFlashAttentionPioneer,
                            OP_INPUT(query, key, value, sparseIndices, keyDequantScaleOptional, valueDequantScaleOptional,
                                blockTableOptional, actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional,
                                keySinkOptional, valueSinkOptional),
                            OP_OUTPUT(output),
                            OP_ATTR(scaleValue, keyQuantMode, valueQuantMode, sparseBlockSize, layoutQueryOptional,
                                layoutKvOptional, sparseMode, preTokens, nextTokens, attentionMode,
                                quantScaleRepoMode, tileSize, ropeHeadDim, keyBlockStride));   

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }

    return output;
}

} // namespace l0op
