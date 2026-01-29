/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_interleave_rope.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "common/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "posembedding/rotary_position_embedding/op_host/op_api/aclnn_rotary_position_embedding.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

constexpr int64_t HALF_INTERLEAVE_MODE = 3;

aclnnStatus aclnnInnerRotaryPositionEmbeddingGetWorkspaceSize(
    const aclTensor* x, const aclTensor* cos, const aclTensor* sin, int64_t mode, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnInnerRotaryPositionEmbedding(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
aclnnStatus aclnnInnerInterleaveRopeGetWorkspaceSize(
    const aclTensor* x, const aclTensor* cos, const aclTensor* sin, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);
aclnnStatus aclnnInnerInterleaveRope(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnInterleaveRopeGetWorkspaceSize(
    const aclTensor* x, const aclTensor* cos, const aclTensor* sin, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    bool useRotaryPositionEmbedding = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950;
    if (useRotaryPositionEmbedding) {
        return aclnnInnerRotaryPositionEmbeddingGetWorkspaceSize(
            x, cos, sin, HALF_INTERLEAVE_MODE, out, workspaceSize, executor);
    } else {
        return aclnnInnerInterleaveRopeGetWorkspaceSize(x, cos, sin, out, workspaceSize, executor);
    }
}

aclnnStatus aclnnInterleaveRope(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    bool useRotaryPositionEmbedding = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950;
    if (useRotaryPositionEmbedding) {
        return aclnnInnerRotaryPositionEmbedding(workspace, workspaceSize, executor, stream);
    } else {
        return aclnnInnerInterleaveRope(workspace, workspaceSize, executor, stream);
    }
}

#ifdef __cplusplus
}
#endif
