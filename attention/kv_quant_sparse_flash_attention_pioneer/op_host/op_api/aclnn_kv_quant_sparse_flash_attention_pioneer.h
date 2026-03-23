/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_KV_QUANT_SPARSE_FLASH_ATTENTION_PIONEER_H_
#define OP_API_INC_LEVEL2_ACLNN_KV_QUANT_SPARSE_FLASH_ATTENTION_PIONEER_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnKvQuantSparseFlashAttentionPioneer的第一段接口，根据具体的计算流程，计算workspace大小。
 * function: aclnnKvQuantSparseFlashAttentionPioneerGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * value : required
 * sparseIndices : required
 * keyDequantScaleOptional : optional
 * valueDequantScaleOptional : optional
 * blockTableOptional : optional
 * actualSeqLengthsQueryOptional : optional
 * actualSeqLengthsKvOptional : optional
 * keySinkOptional : optional
 * valueSinkOptional : optional
 * scaleValue : required
 * keyQuantMode : required
 * valueQuantMode : required
 * sparseBlockSize : optional
 * layoutQueryOptional : optional
 * layoutKvOptional : optional
 * sparseMode : optional
 * preTokens : optional
 * nextTokens : optional
 * attentionMode : optional
 * quantScaleRepoMode : optional
 * tileSize : optional
 * ropeHeadDim : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */

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
    aclOpExecutor **executor);

/**
 * function: aclnnKvQuantSparseFlashAttentionPioneer
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */

aclnnStatus aclnnKvQuantSparseFlashAttentionPioneer(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif