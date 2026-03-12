/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "block_sparse_attention_tiling.h"
#include <cmath>
#include <cstring>
#include "log/log.h"

#include <cstdint>
#include <string>
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_base.h"

using namespace ge;
using namespace std;

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;
constexpr int TND_DIM_NUM = 3;

constexpr int BNSD_DIM_B = 0;
constexpr int BNSD_DIM_N = 1;
constexpr int BNSD_DIM_S = 2;
constexpr int BNSD_DIM_D = 3;
constexpr int BNSD_DIM_NUM = 4;

constexpr int BSH_DIM_B = 0;
constexpr int BSH_DIM_S = 1;
constexpr int BSH_DIM_H = 2;

constexpr int QUERY_INDEX = 0;
constexpr int KEY_INDEX = 1;
constexpr int VALUE_INDEX = 2;
constexpr int BLOCK_SHAPE_INDEX = 5;

//新增的blockSparseMask参数的下标
constexpr int BLOCK_SPARSE_MASK_INDEX = 3;

constexpr int ATTENTION_MASK_INDEX = 4;
constexpr int ACTUAL_SEQ_LENGTHS_INDEX = 6;
constexpr int ACTUAL_SEQ_LENGTHS_KV_INDEX = 7;
constexpr int BLOCK_TABLE_INDEX = 8;
constexpr int SOFTMAX_LSE_INDEX = 10;
constexpr int MAX_BLOCK_NUM_INDEX = 2;


constexpr int Q_INPUT_LAYOUT_INDEX = 0;
constexpr int KV_INPUT_LAYOUT_INDEX = 1;
constexpr int NUM_KEY_VALUE_HEADS_INDEX = 2;
constexpr int MASK_TYPE_INDEX = 3;
constexpr int SCALE_VALUE_INDEX = 4;
constexpr int INNER_PRECISE_INDEX = 5;
constexpr int BLOCK_SIZE_INDEX = 6;
constexpr int SOFTMAX_LSE_FLAG_INDEX = 9;

constexpr int VALID_EMBEDDING_SIZE_64 = 64;
constexpr int VALID_EMBEDDING_SIZE_128 = 128;

constexpr int LSE_NO_OUT = 0;
constexpr int LSE_OUT = 1;

namespace optiling {

constexpr uint32_t BASIC_BLOCK_SIZE = 128;
constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 131072;
constexpr uint32_t NUM3 = 3;

static inline uint32_t CeilDiv(uint32_t n1, uint32_t n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? ((n1 + n2 - 1) / n2) : n1;
}

static inline uint32_t GetQBlocks(int32_t qseqlen, int32_t x)
{
    uint32_t qBlocksInX = CeilDiv(x, BASIC_BLOCK_SIZE);
    uint32_t completeXBlocks = x != 0 ? qseqlen / x : qseqlen / BASIC_BLOCK_SIZE;
    uint32_t remainingSeqlen = x != 0 ? qseqlen - completeXBlocks * x : qseqlen % BASIC_BLOCK_SIZE;
    uint32_t remainingBlocks = CeilDiv(remainingSeqlen, BASIC_BLOCK_SIZE);
    return qBlocksInX * completeXBlocks + remainingBlocks;
}

static inline uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    uint32_t qNBlockTile = 1;
    return qNBlockTile;
}

ge::graphStatus BSATiling::GetNpuInfo(gert::TilingContext *rfaContext)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(rfaContext->GetPlatformInfo());
    
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ParseKvInputLayout(gert::TilingContext *rfaContext)
{
    if (rfaContext->GetAttrs()->GetAttrPointer<char>(KV_INPUT_LAYOUT_INDEX) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    std::string kvLayout(rfaContext->GetAttrs()->GetAttrPointer<char>(KV_INPUT_LAYOUT_INDEX));
    if (kvLayout == "TND") {
        kvCacheLayout_ = RFAKvCacheLayout::TND;
    } else if (kvLayout == "BNSD") {
        kvCacheLayout_ = RFAKvCacheLayout::BNSD;
    } else {
        OP_LOGE(rfaContext->GetNodeName(), "Unsupported KV layout: %s. Supported formats: TND, BNSD.", 
                rfaContext->GetAttrs()->GetAttrPointer<char>(KV_INPUT_LAYOUT_INDEX));
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ParseQInputLayout(gert::TilingContext *rfaContext)
{
    if (rfaContext->GetAttrs()->GetAttrPointer<char>(Q_INPUT_LAYOUT_INDEX) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    std::string qLayout(rfaContext->GetAttrs()->GetAttrPointer<char>(Q_INPUT_LAYOUT_INDEX));
    if (qLayout == "TND") {
        qInputLayout_ = RFAQInputLayout::TND_Q;
    } else if (qLayout == "BNSD") {
        qInputLayout_ = RFAQInputLayout::BNSD_Q;
    } else {
        OP_LOGE(rfaContext->GetNodeName(), "Unsupported Q layout: %s. Supported formats: TND, BNSD", 
                rfaContext->GetAttrs()->GetAttrPointer<char>(Q_INPUT_LAYOUT_INDEX));
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}
//新增校验blockSparseMask合法
ge::graphStatus BSATiling::ValidateBlockSparseMask(gert::TilingContext *rfaContext)
{   
    const auto *blockSparseMaskShape = rfaContext->GetInputShape(BLOCK_SPARSE_MASK_INDEX);
    //验证每一维的数据是否合法
    int64_t  blockSparseMaskBatchSize = blockSparseMaskShape->GetStorageShape().GetDim(0);  // batch_size
    int64_t  blockSparseMaskNumHead = blockSparseMaskShape->GetStorageShape().GetDim(1);  // num_heads
    int64_t  blockSparseMaskSeqLenQ = blockSparseMaskShape->GetStorageShape().GetDim(2);  // seq_len_q
    int64_t  blockSparseMaskSeqLenKV = blockSparseMaskShape->GetStorageShape().GetDim(3);  // seq_len_kv
    
    if(blockSparseMaskBatchSize != batch_){
        OP_LOGE(rfaContext->GetNodeName(), "blockSparseMask batch mismatch: shape=%ld, tiling=%u", blockSparseMaskBatchSize, batch_);
        return ge::GRAPH_FAILED;        
    }
    if(blockSparseMaskNumHead != numHeads_){
        OP_LOGE(rfaContext->GetNodeName(), "blockSparseMask numHeads mismatch: shape=%ld, tiling=%u", blockSparseMaskNumHead, numHeads_);
        return ge::GRAPH_FAILED; 
    }
    if(blockSparseMaskSeqLenQ != maxQSeqlen_){
        OP_LOGE(rfaContext->GetNodeName(), "blockSparseMask maxQSeqlen mismatch: shape=%ld, tiling=%u", blockSparseMaskSeqLenQ, maxQSeqlen_);
        return ge::GRAPH_FAILED; 
    }
    if(blockSparseMaskSeqLenKV != maxKvSeqlen_){
        OP_LOGE(rfaContext->GetNodeName(), "blockSparseMask maxKvSeqlen_ mismatch: shape=%ld, tiling=%u", blockSparseMaskSeqLenKV, maxKvSeqlen_);
        return ge::GRAPH_FAILED; 
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ValidateTNDFormat(gert::TilingContext *rfaContext)
{
    const auto *kvShape = rfaContext->GetInputShape(KEY_INDEX);
    if (kvShape == nullptr || kvShape->GetStorageShape().GetDimNum() != TND_DIM_NUM) {
        OP_LOGE(rfaContext->GetNodeName(), "TND format KV cache must have 3 dimensions");
        return ge::GRAPH_FAILED;
    }
    
    // TND: [T, N, D] where T=total tokens, N=num_kv_heads, D=head_dim
    totalTokensKv_ = kvShape->GetStorageShape().GetDim(0);
    int64_t kvHeads = kvShape->GetStorageShape().GetDim(1);
    int64_t headDim = kvShape->GetStorageShape().GetDim(2);
    
    if (kvHeads != kvHeads_) {
        OP_LOGE(rfaContext->GetNodeName(), "KV heads mismatch: shape=%ld, attr=%u", kvHeads, kvHeads_);
        return ge::GRAPH_FAILED;
    }
    
    if (headDim != embeddingSize_) {
        OP_LOGE(rfaContext->GetNodeName(), "Head dimension mismatch: shape=%ld, attr=%u", headDim, embeddingSize_);
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ValidateBNSDFormat(gert::TilingContext *rfaContext)
{
    const auto *kvShape = rfaContext->GetInputShape(KEY_INDEX);
    if (kvShape == nullptr || kvShape->GetStorageShape().GetDimNum() != BNSD_DIM_NUM) {
        OP_LOGE(rfaContext->GetNodeName(), "BNSD format KV cache must have 4 dimensions");
        return ge::GRAPH_FAILED;
    }
    
    // BNSD: [B, N, S, D] where B=batch, N=num_kv_heads, S=seq_len, D=head_dim
    int64_t kvBatch = kvShape->GetStorageShape().GetDim(BNSD_DIM_B);
    int64_t kvHeads = kvShape->GetStorageShape().GetDim(BNSD_DIM_N);
    int64_t kvSeqlen = kvShape->GetStorageShape().GetDim(BNSD_DIM_S);
    int64_t headDim = kvShape->GetStorageShape().GetDim(BNSD_DIM_D);
    
    // 保存BNSD格式KV的第三维（S维度）作为maxKvSeqlen
    maxKvSeqlen_ = static_cast<uint32_t>(kvSeqlen);
    
    if (kvBatch != batch_) {
        OP_LOGE(rfaContext->GetNodeName(), "KV batch mismatch: shape=%ld, tiling=%u", kvBatch, batch_);
        return ge::GRAPH_FAILED;
    }
    
    if (kvHeads != kvHeads_) {
        OP_LOGE(rfaContext->GetNodeName(), "KV heads mismatch: shape=%ld, attr=%u", kvHeads, kvHeads_);
        return ge::GRAPH_FAILED;
    }
    
    if (headDim != embeddingSize_) {
        OP_LOGE(rfaContext->GetNodeName(), "Head dimension mismatch: shape=%ld, attr=%u", headDim, embeddingSize_);
        return ge::GRAPH_FAILED;
    }
    
    // 验证Value shape
    const auto *valueShape = rfaContext->GetInputShape(VALUE_INDEX);
    if (valueShape == nullptr || valueShape->GetStorageShape().GetDimNum() != BNSD_DIM_NUM) {
        OP_LOGE(rfaContext->GetNodeName(), "BNSD format Value cache must have 4 dimensions");
        return ge::GRAPH_FAILED;
    }
    
    if (valueShape->GetStorageShape().GetDim(BNSD_DIM_B) != kvBatch ||
        valueShape->GetStorageShape().GetDim(BNSD_DIM_N) != kvHeads ||
        valueShape->GetStorageShape().GetDim(BNSD_DIM_S) != kvSeqlen ||
        valueShape->GetStorageShape().GetDim(BNSD_DIM_D) != headDim) {
        OP_LOGE(rfaContext->GetNodeName(), "Value shape mismatch with Key shape in BNSD format");
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::CheckKvCacheLayout(gert::TilingContext *rfaContext)
{
    ge::graphStatus ret = ParseKvInputLayout(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    
    // 验证Q和KV格式一致性：如果其中一个是BNSD，另一个也必须是BNSD
    bool isQBNSD = (qInputLayout_ == RFAQInputLayout::BNSD_Q);
    bool isKvBNSD = (kvCacheLayout_ == RFAKvCacheLayout::BNSD);
    
    if (isQBNSD != isKvBNSD) {
        OP_LOGE(rfaContext->GetNodeName(), 
                "Q and KV layouts must match: if one is BNSD, the other must also be BNSD. "
                "Q layout: %s, KV layout: %s",
                (isQBNSD ? "BNSD" : "non-BNSD"),
                (isKvBNSD ? "BNSD" : "non-BNSD"));
        return ge::GRAPH_FAILED;
    }
    
    if (kvCacheLayout_ == RFAKvCacheLayout::TND) {
        ret = ValidateTNDFormat(rfaContext);
    } else if (kvCacheLayout_ == RFAKvCacheLayout::BNSD) {
        ret = ValidateBNSDFormat(rfaContext);
    } else {
        OP_LOGE(rfaContext->GetNodeName(), "Unsupported KV cache layout");
        return ge::GRAPH_FAILED;
    }
    
    return ret;
}

ge::graphStatus BSATiling::ProcessQueryShape(gert::TilingContext *rfaContext)
{
    const auto *queryShape = rfaContext->GetInputShape(QUERY_INDEX);
    if (queryShape == nullptr) {
        OP_LOGE(rfaContext->GetNodeName(), "Query shape is null");
        return ge::GRAPH_FAILED;
    }
    
    if (qInputLayout_ == RFAQInputLayout::TND_Q) {
        if (queryShape->GetStorageShape().GetDimNum() != TND_DIM_NUM) {
            OP_LOGE(rfaContext->GetNodeName(), "TND format must have 3 dimensions");
            return ge::GRAPH_FAILED;
        }
        totalTokensT_ = queryShape->GetStorageShape().GetDim(TND_DIM_T);
        numHeads_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_N));
        embeddingSize_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_D));
    } else if (qInputLayout_ == RFAQInputLayout::BNSD_Q) {
        if (queryShape->GetStorageShape().GetDimNum() != BNSD_DIM_NUM) {
            OP_LOGE(rfaContext->GetNodeName(), "BNSD format must have 4 dimensions");
            return ge::GRAPH_FAILED;
        }
        // 保存BNSD格式的batch和S维度
        batch_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BNSD_DIM_B));
        numHeads_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BNSD_DIM_N));
        embeddingSize_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BNSD_DIM_D));
        maxQSeqlen_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BNSD_DIM_S));
    }
    
    if (numHeads_ == 0) {
        OP_LOGE(rfaContext->GetNodeName(), "numHeads deduced from query is zero");
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}



ge::graphStatus BSATiling::GetKvSeqlenFromShape(gert::TilingContext *rfaContext, uint32_t &kvSeqlen)
{
    const auto *kvShape = rfaContext->GetInputShape(KEY_INDEX);
    if (kvShape == nullptr || kvShape->GetStorageShape().GetDimNum() != BNSD_DIM_NUM) {
        OP_LOGE(rfaContext->GetNodeName(), "BNSD format: KV shape is invalid, cannot get S dimension");
        return ge::GRAPH_FAILED;
    }
    int64_t kvSeqlenInt64 = kvShape->GetStorageShape().GetDim(BNSD_DIM_S);
    if (kvSeqlenInt64 <= 0) {
        OP_LOGE(rfaContext->GetNodeName(), "BNSD format: KV seqlen (%ld) is invalid", kvSeqlenInt64);
        return ge::GRAPH_FAILED;
    }
    kvSeqlen = static_cast<uint32_t>(kvSeqlenInt64);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ValidateBNSDQSeqlen(gert::TilingContext *rfaContext)
{
    if (qInputLayout_ != RFAQInputLayout::BNSD_Q || qSeqLenList == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    
    if (maxQSeqlen_ == 0) {
        OP_LOGE(rfaContext->GetNodeName(), "BNSD format: maxQSeqlen_ is 0, cannot validate seq lengths");
        return ge::GRAPH_FAILED;
    }
    
    for (uint32_t i = 0; i < batch_; i++) {
        if (qSeqLenList[i] > static_cast<int64_t>(maxQSeqlen_)) {
            OP_LOGE(rfaContext->GetNodeName(), 
                    "BNSD format validation failed: qseqlen[%u] (%ld) > maxQSeqlen (%u)", 
                    i, qSeqLenList[i], maxQSeqlen_);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ValidateBNSDKvSeqlen(gert::TilingContext *rfaContext)
{
    if (kvCacheLayout_ != RFAKvCacheLayout::BNSD || kvSeqLenList == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    
    if (maxKvSeqlen_ == 0) {
        OP_LOGE(rfaContext->GetNodeName(), "BNSD format: maxKvSeqlen_ is 0, cannot validate kv seq lengths");
        return ge::GRAPH_FAILED;
    }
    
    for (uint32_t i = 0; i < batch_; i++) {
        if (kvSeqLenList[i] > static_cast<int64_t>(maxKvSeqlen_)) {
            OP_LOGE(rfaContext->GetNodeName(), 
                    "BNSD format validation failed: kvseqlen[%u] (%ld) > maxKvSeqlen (%u)", 
                    i, kvSeqLenList[i], maxKvSeqlen_);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ProcessQSeqLengths(gert::TilingContext *rfaContext, 
                                              const gert::Tensor *actualSeqLengths)
{
    // BNSD格式下，actualSeqLengths可以为nullptr，使用BNSD中的S值
    // batch已经从BNSD的B维度获取（在ProcessQueryShape中），不需要从actualSeqLengths获取
    if (qInputLayout_ == RFAQInputLayout::BNSD_Q) {
        // BNSD格式下，如果actualSeqLengths为nullptr或batch不匹配，都视为nullptr处理
        // 因为batch应该从BNSD的B维度获取，而不是从actualSeqLengths获取
        bool shouldUseUniform = false;
        if (actualSeqLengths == nullptr) {
            shouldUseUniform = true;
        } else {
            uint32_t actualBatch = static_cast<uint32_t>(actualSeqLengths->GetShapeSize());
            if (actualBatch != batch_) {
                // batch不匹配，视为nullptr处理
                shouldUseUniform = true;
            }
        }
        
        if (shouldUseUniform) {
            // BNSD格式下，actualSeqLengths为nullptr或batch不匹配，使用BNSD中的S值
            if (batch_ == 0) {
                OP_LOGE(rfaContext->GetNodeName(), "BNSD format: batch_ is 0, cannot process seq lengths");
                return ge::GRAPH_FAILED;
            }
            if (maxQSeqlen_ == 0) {
                OP_LOGE(rfaContext->GetNodeName(), "BNSD format: maxQSeqlen_ is 0, cannot process seq lengths");
                return ge::GRAPH_FAILED;
            }
            useUniformQSeqlen_ = true;
            qSeqLenList = nullptr;
            return ge::GRAPH_SUCCESS;
        }
        // BNSD格式下，提供了actualSeqLengths且batch匹配，使用actualSeqLengths
    } else {
        // TND格式
        if (actualSeqLengths == nullptr) {
            OP_LOGE(rfaContext->GetNodeName(), "TND format: Actual seq lengths cannot be null");
            return ge::GRAPH_FAILED;
        }
        // TND格式下，从actualSeqLengths获取batch
        batch_ = static_cast<uint32_t>(actualSeqLengths->GetShapeSize());
        if (batch_ == 0) {
            OP_LOGE(rfaContext->GetNodeName(), "batch size is 0");
            return ge::GRAPH_FAILED;
        }
    }
    
    qSeqLenList = actualSeqLengths->GetData<int64_t>();
    if (qSeqLenList == nullptr) {
        OP_LOGE(rfaContext->GetNodeName(), "Actual seq lengths GetData is nullptr");
        return ge::GRAPH_FAILED;
    }
    useUniformQSeqlen_ = false;
    
    // BNSD格式下，校验每个batch的qseqlen都小于maxQSeqlen_
    return ValidateBNSDQSeqlen(rfaContext);
}

bool BSATiling::CheckShouldUseUniformKvSeqlen(const gert::Tensor *actualSeqLengthsKv)
{
    if (actualSeqLengthsKv == nullptr) {
        return true;
    }
    uint32_t kvBatch = static_cast<uint32_t>(actualSeqLengthsKv->GetShapeSize());
    return (kvBatch != batch_);
}

ge::graphStatus BSATiling::SetupUniformKvSeqlen(gert::TilingContext *rfaContext)
{
    if (batch_ == 0) {
        OP_LOGE(rfaContext->GetNodeName(), "BNSD format: batch_ is 0, cannot process kv seq lengths");
        return ge::GRAPH_FAILED;
    }
    
    uint32_t kvSeqlen = 0;
    ge::graphStatus ret = GetKvSeqlenFromShape(rfaContext, kvSeqlen);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    
    maxKvSeqlen_ = kvSeqlen;
    useUniformKvSeqlen_ = true;
    kvSeqLenList = nullptr;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ProcessKvSeqLengthsBNSD(gert::TilingContext *rfaContext, 
                                                   const gert::Tensor *actualSeqLengthsKv)
{
    // BNSD格式下，如果actualSeqLengthsKv为nullptr或batch不匹配，都视为nullptr处理
    // 因为batch应该从BNSD的B维度获取，而不是从actualSeqLengthsKv获取
    if (CheckShouldUseUniformKvSeqlen(actualSeqLengthsKv)) {
        return SetupUniformKvSeqlen(rfaContext);
    }
    // BNSD格式下，提供了actualSeqLengthsKv且batch匹配，使用actualSeqLengthsKv
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ProcessKvSeqLengthsTND(gert::TilingContext *rfaContext, 
                                                  const gert::Tensor *actualSeqLengthsKv)
{
    if (actualSeqLengthsKv == nullptr) {
        OP_LOGE(rfaContext->GetNodeName(), "TND format: Actual seq lengths kv cannot be null");
        return ge::GRAPH_FAILED;
    }
    
    uint32_t kvBatch = static_cast<uint32_t>(actualSeqLengthsKv->GetShapeSize());
    if (kvBatch != batch_) {
        OP_LOGE(rfaContext->GetNodeName(), 
                "TND format: actualSeqLengthsKv batch (%u) mismatch with batch (%u)", 
                kvBatch, batch_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ProcessKvSeqLengthsWithArray(gert::TilingContext *rfaContext, 
                                                        const gert::Tensor *actualSeqLengthsKv)
{
    // BNSD格式下，如果maxKvSeqlen_还没有设置，从KV shape中获取
    if (kvCacheLayout_ == RFAKvCacheLayout::BNSD && maxKvSeqlen_ == 0) {
        uint32_t kvSeqlen = 0;
        ge::graphStatus ret = GetKvSeqlenFromShape(rfaContext, kvSeqlen);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        maxKvSeqlen_ = kvSeqlen;
    }
    
    kvSeqLenList = actualSeqLengthsKv->GetData<int64_t>();
    if (kvSeqLenList == nullptr) {
        OP_LOGE(rfaContext->GetNodeName(), "Actual seq lengths kv GetData is nullptr");
        return ge::GRAPH_FAILED;
    }
    useUniformKvSeqlen_ = false;
    
    // BNSD格式下，校验每个batch的kvseqlen都小于maxKvSeqlen_
    return ValidateBNSDKvSeqlen(rfaContext);
}

ge::graphStatus BSATiling::ProcessKvSeqLengths(gert::TilingContext *rfaContext, 
                                               const gert::Tensor *actualSeqLengthsKv)
{
    // BNSD格式下，actualSeqLengthsKv可以为nullptr，使用BNSD中的S值
    // batch已经从BNSD的B维度获取（在ProcessQueryShape中），不需要从actualSeqLengthsKv获取
    if (kvCacheLayout_ == RFAKvCacheLayout::BNSD) {
        ge::graphStatus ret = ProcessKvSeqLengthsBNSD(rfaContext, actualSeqLengthsKv);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        // 如果使用了uniform值，已经返回了，否则继续处理数组
        if (useUniformKvSeqlen_) {
            return ge::GRAPH_SUCCESS;
        }
    } else {
        ge::graphStatus ret = ProcessKvSeqLengthsTND(rfaContext, actualSeqLengthsKv);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }
    
    // 处理使用数组的情况
    return ProcessKvSeqLengthsWithArray(rfaContext, actualSeqLengthsKv);
}

ge::graphStatus BSATiling::ProcessActualSeqLengths(gert::TilingContext *rfaContext)
{
    // 先解析KV layout（用于判断是否需要处理BNSD格式的nullptr情况）
    ge::graphStatus ret = ParseKvInputLayout(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    
    // 处理Q序列长度
    auto actualSeqLengths = rfaContext->GetOptionalInputTensor(ACTUAL_SEQ_LENGTHS_INDEX);
    ret = ProcessQSeqLengths(rfaContext, actualSeqLengths);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    
    // 处理KV序列长度
    auto actualSeqLengthsKv = rfaContext->GetOptionalInputTensor(ACTUAL_SEQ_LENGTHS_KV_INDEX);
    ret = ProcessKvSeqLengths(rfaContext, actualSeqLengthsKv);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ProcessBlockShape(gert::TilingContext *rfaContext)
{
    auto blockShape = rfaContext->GetInputTensor(BLOCK_SHAPE_INDEX);
    if (blockShape == nullptr) {
        OP_LOGE(rfaContext->GetNodeName(), "Block shape tensor is null");
        return ge::GRAPH_FAILED;
    }
    
    blockShapeList = blockShape->GetData<int64_t>();
    if (blockShapeList == nullptr) {
        OP_LOGE(rfaContext->GetNodeName(), "Block shape GetData is nullptr");
        return ge::GRAPH_FAILED;
    }
    
    blockShapeX_ = blockShapeList[0];
    blockShapeY_ = blockShapeList[1];
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ProcessSoftmaxLse(gert::TilingContext *rfaContext)
{
    auto softmaxLsePtr = rfaContext->GetAttrs()->GetAttrPointer<uint32_t>(SOFTMAX_LSE_FLAG_INDEX);
    if (softmaxLsePtr == nullptr) {
        OP_LOGE(rfaContext->GetNodeName(), "softmaxLsePtr is null");
        return ge::GRAPH_FAILED;
    }
    switch (*softmaxLsePtr) {
        case LSE_NO_OUT:
            softmaxLseFlag_ = false;
            break;
        case LSE_OUT:
            softmaxLseFlag_ = true;
            break;
        default:
            OP_LOGE(rfaContext->GetNodeName(), "invalid softmaxLseFlag:%u", *softmaxLsePtr);
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ValidateTNDSeqlenSum(gert::TilingContext *rfaContext)
{
    // 只在TND格式时进行校验
    if (qInputLayout_ != RFAQInputLayout::TND_Q || kvCacheLayout_ != RFAKvCacheLayout::TND) {
        return ge::GRAPH_SUCCESS;
    }
    
    // TND格式下，不应该使用统一值，必须提供actualSeqLengths数组
    if (useUniformQSeqlen_ || useUniformKvSeqlen_) {
        OP_LOGE(rfaContext->GetNodeName(), 
                "TND format: useUniformQSeqlen or useUniformKvSeqlen should be false, "
                "actualSeqLengths must be provided");
        return ge::GRAPH_FAILED;
    }
    
    if (qSeqLenList == nullptr) {
        OP_LOGE(rfaContext->GetNodeName(), "qSeqLenList is nullptr, cannot validate TND seqlen sum");
        return ge::GRAPH_FAILED;
    }
    
    if (kvSeqLenList == nullptr) {
        OP_LOGE(rfaContext->GetNodeName(), "kvSeqLenList is nullptr, cannot validate TND seqlen sum");
        return ge::GRAPH_FAILED;
    }
    
    if (batch_ == 0) {
        OP_LOGE(rfaContext->GetNodeName(), "batch_ is 0, cannot validate TND seqlen sum");
        return ge::GRAPH_FAILED;
    }
    
    // 计算所有batch的qseqlen之和
    int64_t sumQSeqlen = 0;
    int64_t sumKvSeqlen = 0;

    for (uint32_t i = 0; i < batch_; i++) {
        sumQSeqlen += qSeqLenList[i];
        sumKvSeqlen += kvSeqLenList[i];
    }
    
    // 校验qseqlen之和是否等于Q的T
    if (sumQSeqlen != totalTokensT_) {
        OP_LOGE(rfaContext->GetNodeName(), 
                "TND format validation failed: sum of qseqlen across all batches (%ld) != Q T (%ld)", 
                sumQSeqlen, totalTokensT_);
        return ge::GRAPH_FAILED;
    }
    
    // 校验kvseqlen之和是否等于KV的T
    if (sumKvSeqlen != totalTokensKv_) {
        OP_LOGE(rfaContext->GetNodeName(), 
                "TND format validation failed: sum of kvseqlen across all batches (%ld) != KV T (%ld)", 
                sumKvSeqlen, totalTokensKv_);
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ValidateConfiguration(gert::TilingContext *rfaContext)
{
    if (embeddingSize_ == 0 || numHeads_ == 0) {
        OP_LOGE(rfaContext->GetNodeName(), "Invalid head or embedding configuration");
        return ge::GRAPH_FAILED;
    }

    if (embeddingSize_ != VALID_EMBEDDING_SIZE_64 && embeddingSize_ != VALID_EMBEDDING_SIZE_128) {
        OP_LOGE(rfaContext->GetNodeName(), "Invalid embedding size, embeddingSize must be 64 or 128");
        return ge::GRAPH_FAILED;
    }

    if (blockShapeX_ <= 0 || blockShapeY_ <= 0) {
        OP_LOGE(rfaContext->GetNodeName(), "Invalid block shape, blockShapeX and blockShapeY must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    
    if (blockShapeY_ % BASIC_BLOCK_SIZE != 0) {
        OP_LOGE(rfaContext->GetNodeName(), "Invalid block shape, blockShapeY must be divisible by %u", BASIC_BLOCK_SIZE);
        return ge::GRAPH_FAILED;
    }
    dataType_ = rfaContext->GetInputDesc(QUERY_INDEX)->GetDataType();
    if (innerPrecise_ == 1 && dataType_ == ge::DT_BF16) {
        OP_LOGE(rfaContext->GetNodeName(), "Invalid innerPrecise, innerPrecise must be 0 when dataType is BF16");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ProcessInput(gert::TilingContext *rfaContext)
{
    ge::graphStatus ret;
    
    // 1. 解析Q layout
    ret = ParseQInputLayout(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    
    // 2. 处理Query shape
    ret = ProcessQueryShape(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }  
    // 4. 处理实际序列长度
    ret = ProcessActualSeqLengths(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    
    // 5. 处理Block shape
    ret = ProcessBlockShape(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 6. 处理softmax lse flag
    ret = ProcessSoftmaxLse(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    
    // 7. 验证配置
    ret = ValidateConfiguration(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::CheckAttr(gert::TilingContext *rfaContext)
{
    if (rfaContext->GetAttrs()->GetAttrPointer<uint32_t>(NUM_KEY_VALUE_HEADS_INDEX) == nullptr) {
        OP_LOGE(rfaContext->GetNodeName(), "numKeyValueHeads is null");
        return ge::GRAPH_FAILED;
    }
    kvHeads_ = *rfaContext->GetAttrs()->GetAttrPointer<uint32_t>(NUM_KEY_VALUE_HEADS_INDEX);
    
    if (rfaContext->GetAttrs()->GetAttrPointer<float>(SCALE_VALUE_INDEX) == nullptr) {
        scaleValue_ = 1.0f / std::sqrt(static_cast<float>(embeddingSize_));
    } else {
        scaleValue_ = *rfaContext->GetAttrs()->GetAttrPointer<float>(SCALE_VALUE_INDEX);
    }
    
    if (rfaContext->GetAttrs()->GetAttrPointer<uint32_t>(MASK_TYPE_INDEX) != nullptr) {
        maskType_ = *rfaContext->GetAttrs()->GetAttrPointer<uint32_t>(MASK_TYPE_INDEX);
    }
    
    // 获取innerPrecise参数
    if (rfaContext->GetAttrs()->GetAttrPointer<uint32_t>(INNER_PRECISE_INDEX) != nullptr) {
        innerPrecise_ = *rfaContext->GetAttrs()->GetAttrPointer<uint32_t>(INNER_PRECISE_INDEX);
    }
    auto softmaxLsePtr = rfaContext->GetAttrs()->GetAttrPointer<uint32_t>(SOFTMAX_LSE_FLAG_INDEX);
    if (softmaxLsePtr == nullptr) {
        softmaxLseFlag_ = false;
    } else {
        softmaxLseFlag_ = *softmaxLsePtr == 1 ? true : false;
    }
    
    return ge::GRAPH_SUCCESS;
}

void BSATiling::CalculateBatchTaskSplit(uint32_t batchIdx, int64_t qSeqlen, uint32_t groupSize,
                                        uint32_t &curTaskNum, uint32_t &curQBlockNum)
{
    uint32_t curQBlockTile = GetQNBlockTile(qSeqlen, groupSize);
    uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQBlockTile);
    uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads_;
    curTaskNum = GetQBlocks(qSeqlen, blockShapeX_) * curQNBlockNum;
    curQBlockNum = CeilDiv(qSeqlen, blockShapeX_) * numHeads_;
}

ge::graphStatus BSATiling::CalculateTaskSplit(gert::TilingContext *rfaContext)
{
    // 计算总的Q块数量和最大KV块数量
    totalQBlocks_ = 0;

    if (kvHeads_ == 0) {
        OP_LOGE(rfaContext->GetNodeName(), "kvHeads_ is 0, cannot calculate groupSize");
        return ge::GRAPH_FAILED;
    }
    if (batch_ == 0) {
        OP_LOGE(rfaContext->GetNodeName(), "batch_ is 0 in CalculateTaskSplit");
        return ge::GRAPH_FAILED;
    }
    
    // 根据useUniformQSeqlen_标志位决定分核时使用actualSeqLengths数组还是maxQSeqlen_
    uint32_t groupSize = numHeads_ / kvHeads_;
    
    // 遍历每个batch进行分核计算
    for (auto i = 0; i < batch_; i++) {
        // 根据useUniformQSeqlen_标志位决定使用actualSeqLengths数组还是maxQSeqlen_
        int64_t qSeqlen;
        if (useUniformQSeqlen_) {
            // BNSD格式下actualSeqLengths为nullptr，使用maxQSeqlen_作为统一的qseqlen值
            qSeqlen = static_cast<int64_t>(maxQSeqlen_);
        } else {
            // 使用actualSeqLengths数组（TND格式或BNSD格式但提供了actualSeqLengths）
            if (qSeqLenList == nullptr) {
                OP_LOGE(rfaContext->GetNodeName(), "qSeqLenList is nullptr, cannot calculate task split");
                return ge::GRAPH_FAILED;
            }
            qSeqlen = qSeqLenList[i];
        }

        uint32_t curTaskNum = 0;
        uint32_t curQBlockNum = 0;
        CalculateBatchTaskSplit(i, qSeqlen, groupSize, curTaskNum, curQBlockNum);
        
        if (i == 0) {
            firstBatchTaskNum_ = curTaskNum;
            firstQBlockNum_ = curQBlockNum;
        }
        totalTaskNum_ += curTaskNum;
        totalQBlocks_ += curQBlockNum;
    }
    blockDim_ = std::min(aicNum_, totalTaskNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::CalculateWorkSpace(gert::TilingContext *rfaContext)
{
    if (blockDim_ == 0) {
        OP_LOGE(rfaContext->GetNodeName(), "blockDim is 0");
        return ge::GRAPH_FAILED;
    }

    const auto *blockSparseMaskShape = rfaContext->GetInputShape(BLOCK_SPARSE_MASK_INDEX);
    maxKvBlockNum_ = blockSparseMaskShape->GetStorageShape().GetDim(3);
    maxQBlockNum_ = blockSparseMaskShape->GetStorageShape().GetDim(2);
    selectIdxSize_ = CeilDiv(blockShapeX_, 128) * CeilDiv(maxKvBlockNum_, 32) * 32 * sizeof(uint32_t) * batch_ * numHeads_ * maxQBlockNum_;
    selectNumIdxSize_ = CeilDiv(blockShapeX_, 128) * sizeof(uint32_t) * 32 * batch_ * numHeads_ * maxQBlockNum_;
    int32_t syncSize_ = sizeof(uint32_t) * 256;
    
    mm1OutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
    smOnlineOutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(uint16_t) * NUM3;
    mm2OutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
    updateSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
    
    workSpaceSize_ = libapiSize_ + mm1OutSize_ + smOnlineOutSize_ + mm2OutSize_ + updateSize_ + selectNumIdxSize_ + selectIdxSize_ + syncSize_;
    rfaContext->GetWorkspaceSizes(1)[0] = workSpaceSize_;
    uint32_t totalTaskNumMask = batch_ * numHeads_ * maxQBlockNum_;
    avgRowPerSubCore_ = CeilDiv(totalTaskNumMask, blockDim_ * 2);
    preActivateSubCoreNum_ = CeilDiv(totalTaskNumMask, avgRowPerSubCore_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::FillTilingData(gert::TilingContext *rfaContext)
{
    if (tilingData_ == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_embeddingSize(embeddingSize_);
    tilingData_->set_blockSize(blockSize_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_batch(batch_);
    tilingData_->set_maxNumBlocksPerBatch(maxNumBlocksPerBatch_);
    tilingData_->set_firstBatchTaskNum(firstBatchTaskNum_);
    tilingData_->set_totalTaskNum(totalTaskNum_);
    tilingData_->set_maskType(maskType_);
    
    tilingData_->set_blockShapeX(blockShapeX_);
    tilingData_->set_blockShapeY(blockShapeY_);
    
    tilingData_->set_firstQBlockNum(firstQBlockNum_);
    tilingData_->set_totalQBlocks(totalQBlocks_);
    tilingData_->set_maxKvBlockNum(maxKvBlockNum_);
    tilingData_->set_maxQBlockNum(maxQBlockNum_);
    tilingData_->set_avgRowPerSubCore(avgRowPerSubCore_);
    tilingData_->set_preActivateSubCoreNum(preActivateSubCoreNum_);
    
    tilingData_->set_kvCacheLayout(static_cast<uint32_t>(kvCacheLayout_));
    tilingData_->set_queryLayout(static_cast<uint32_t>(qInputLayout_));
    tilingData_->set_maxQSeqlen(maxQSeqlen_);
    tilingData_->set_maxKvSeqlen(maxKvSeqlen_);
    
    // BNSD格式下当actualSeqLengths为nullptr时，使用maxQSeqlen和maxKvSeqlen作为统一值
    tilingData_->set_useUniformQSeqlen(useUniformQSeqlen_ ? 1 : 0);
    tilingData_->set_useUniformKvSeqlen(useUniformKvSeqlen_ ? 1 : 0);
    
    // 生成tilingKey（按照开发规范：在tiling层生成）
    uint64_t tilingKey = GenerateTilingKey(rfaContext);
    tilingData_->set_tilingKey(tilingKey);
    rfaContext->SetTilingKey(tilingKey);
    rfaContext->SetBlockDim(blockDim_);
    
    tilingData_->set_mm1OutSize(mm1OutSize_);
    tilingData_->set_smOnlineOutSize(smOnlineOutSize_);
    tilingData_->set_mm2OutSize(mm2OutSize_);
    tilingData_->set_updateSize(updateSize_);
    tilingData_->set_workSpaceSize(workSpaceSize_);
    tilingData_->set_scaleValue(scaleValue_);
    tilingData_->set_selectNumIdxSize(selectNumIdxSize_);
    tilingData_->set_selectIdxSize(selectIdxSize_);
    
    return ge::GRAPH_SUCCESS;
}

uint64_t BSATiling::GenerateTilingKey(gert::TilingContext *rfaContext)
{
    /**
     * 64位整数，使用十进制位域表示：
     * AAAABBBBCCCCDDDDEEEE
     * - 位0-1:   Q Layout（个位）      2=TND, 3=BNSD
     * - 位2-4:   Mask Type（千位）    0=NoMask, 3=CausalMask
     * - 位5-7:   Softmax Precision（十万位） 0=Float, 1=Half
     * - 位8-10:  PagedCache Flag（千万位）  0=NoCache, 1=WithCache
     * - 位11-13: KV Layout（十亿位）   00=TND, 20=BNSD
     * - 位14-15: Data Type（百亿位）  00=FP16, 22=BF16
     * - 位16-18: Operator Category（千万亿位） 900=BlockSparseAttention
     * 
     * 示例：
     * - FP16, TND, TND, NoCache, Half, NoMask = 9000000030100002
     * - FP16, TND, TND, NoCache, Float, NoMask = 9000000030000002
     */
    
    uint64_t tilingKey = 9000000000000000ULL;  // RFA基础值（Operator Category = 900）
    
    // [位14-15] Data Type（百亿位）
    if (dataType_ == ge::DT_FLOAT16) {
        tilingKey += 0;  // 00 for FP16
    } else if (dataType_ == ge::DT_BF16) {
        tilingKey += 22220ULL;  // 22 for BF16 -> 9000000030000002 + 22220 = 9000000030022222
    }
    
    // [位11-13] KV Layout（十亿位）
    if (kvCacheLayout_ == RFAKvCacheLayout::TND) {
        tilingKey += 30000000ULL;  // 00 for TND
    } else if (kvCacheLayout_ == RFAKvCacheLayout::BNSD) {
        tilingKey += 50000000ULL;  // 20 for BNSD
    }
    
    // [位8-10] PagedCache Flag（千万位）
    bool hasPagedCache = (rfaContext->GetOptionalInputTensor(BLOCK_TABLE_INDEX) != nullptr);
    if (hasPagedCache) {
        tilingKey += 1000000ULL;  // 1 for WithCache
    }
    
    // [位5-7] Softmax Precision（十万位）
    if (innerPrecise_ == 1) {
        tilingKey += 100000ULL;  // 1 for Half (FP16) Softmax
    }
    // innerPrecise_ == 0: 0 for Float Softmax（默认值）
    
    // [位2-4] Mask Type（千位）
    if (maskType_ == 3) {  // Causal mask
        tilingKey += 3000ULL;
    }
    // maskType_ == 0: 0 for NoMask（默认值）
    
    // [位0-1] Q Layout（个位）
    if (qInputLayout_ == RFAQInputLayout::TND_Q) {
        tilingKey += 2;  // 2 for TND
    } else if (qInputLayout_ == RFAQInputLayout::BNSD_Q) {
        tilingKey += 3;  // 3 for BNSD
    }

    // Softmax LSE（亿位）
    if (softmaxLseFlag_) {
        tilingKey += 100000000ULL; // 1 for lse out
    }
    
    return tilingKey;
}

ge::graphStatus BSATiling::GetRFATiling(gert::TilingContext *rfaContext,
                                         BlockSparseAttentionTilingData &tilingData)
{
    tilingData_ = &tilingData;
    ge::graphStatus ret = GetNpuInfo(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(rfaContext->GetNodeName(), "GetNpuInfo failed");
        return ret;
    }
    
    ret = CheckAttr(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(rfaContext->GetNodeName(), "CheckAttr failed");
        return ret;
    }
    ret = ProcessInput(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(rfaContext->GetNodeName(), "ProcessInput failed");
        return ret;
    }
    
    ret = CheckKvCacheLayout(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(rfaContext->GetNodeName(), "CheckKvCacheLayout failed");
        return ret;
    }
    
    // 校验TND格式下qseqlen和kvseqlen之和是否分别等于Q和KV的T
    ret = ValidateTNDSeqlenSum(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(rfaContext->GetNodeName(), "ValidateTNDSeqlenSum failed");
        return ret;
    }
    
    ret = CalculateTaskSplit(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(rfaContext->GetNodeName(), "CalculateTaskSplit failed");
        return ret;
    }
    
    ret = CalculateWorkSpace(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(rfaContext->GetNodeName(), "CalculateWorkSpace failed");
        return ret;
    }
    
    ret = FillTilingData(rfaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(rfaContext->GetNodeName(), "FillTilingData failed");
        return ret;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::RFASetTilingData(gert::TilingContext *context,
    BlockSparseAttentionTilingData &tilingData)
{
    OP_CHECK_IF(context->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("BlockSparseAttention",
        "RawTilingData got from GE context is nullptr."), return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingBlockSparseAttention(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("BlockSparseAttention",
        "Context is nullptr."), return ge::GRAPH_FAILED);
    BlockSparseAttentionTilingData tilingData;
    BSATiling rfiTiling;
    if (rfiTiling.GetRFATiling(context, tilingData) == ge::GRAPH_SUCCESS) {
        rfiTiling.RFASetTilingData(context, tilingData);
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(context->GetNodeName(), "GetRFATiling failed");
        return ge::GRAPH_FAILED;
    }
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForBlockSparseAttention(gert::TilingParseContext* context)
{
    (void) context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BlockSparseAttention)
    .Tiling(TilingBlockSparseAttention)
    .TilingInputsDataDependency({5, 6, 7}, {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU})
    .TilingParse<BlockSparseAttentionCompileInfo>(TilingPrepareForBlockSparseAttention); // 向框架注册入口函数;

}  // namespace optiling
