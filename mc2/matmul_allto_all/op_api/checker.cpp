/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "checker.h"
#include "securec.h"
#include "acl/acl.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "hccl_util.h"

namespace matmul_allto_all_check {

using namespace op;

// 需要使用的常量定义
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t NEG_TWO = -2;
static constexpr size_t ONE_DIM = 1;
static constexpr size_t TWO_DIMS = 2U;
static constexpr int64_t ZERO = 0;
static constexpr size_t MAX_GROUP_LEN = 128U;
static constexpr int64_t KVALUE_MIN = 1;
static constexpr int64_t KVALUE_MAX = 65535;

// 检查AlltoAll和Permute数据交换的方向参数, 可以为空和{-1,-2}, 不允许为其他值
bool CheckAlltoAllAxes(const aclIntArray* alltoAllAxesOptional)
{
    // alltoAllAxesOptional为空时会兼容性处理，不报错
    if (alltoAllAxesOptional == nullptr) {
        OP_LOGW("The alltoAllAxesOptional is nullptr.");
        return true;
    }
    uint64_t alltoallAxesSize = 0U;  // alltoallAxes的大小
    aclGetIntArraySize(alltoAllAxesOptional, &alltoallAxesSize);
    if (alltoallAxesSize != TWO_DIMS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of alltoAllAxesOptional should be 2U, but it is: %zu.", alltoallAxesSize);
        return false;
    }
    int64_t data1 = (*alltoAllAxesOptional)[0];
    int64_t data2 = (*alltoAllAxesOptional)[1];
    OP_API_CHECK((data1 != NEG_ONE), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "The 0-axis of alltoAllAxesOptional should be -1, but it is: %ld.", data1);
      return false;
    });
    OP_API_CHECK((data2 != NEG_TWO), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "The 1-axis of alltoAllAxesOptional should be -2, but it is: %ld.", data2);
      return false;
    });
    return true;
}

// 检查输入的转置配置，x1不允许转置
bool CheckTransposeX1(bool transposeX1)
{
    OP_API_CHECK(transposeX1, {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The x1 should not be transposed, but it is transposed.");
    return false;
  });
    return true;
}

// 检查通信域名的字符串长度是否符合要求
bool CheckGroupLength(const char *group)
{
    if (group == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Group should not be nullptr.");
        return false;
    }
    auto len = strnlen(group, MAX_GROUP_LEN);
    if ((len >= MAX_GROUP_LEN) || (len == ZERO)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Required group name length in range (0, 128), but it is %zu.", len);
        return false;
    }
    return true;
}

// 校验输入属性shape
bool CheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
                bool transposeX2, const aclTensor* output)
{
    OP_CHECK_WRONG_DIMENSION(x1, TWO_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(x2, TWO_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(output, TWO_DIMS, return false);

    auto kdimX1 = x1->GetViewShape().GetDim(1);
    auto kdimX2 = transposeX2 ? x2->GetViewShape().GetDim(1) : x2->GetViewShape().GetDim(0);
    if (kdimX1 != kdimX2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "The k-axis of x1 and x2 should be same, but x1's k-axis is: %ld and x2's k-axis is: %ld.", kdimX1, kdimX2);
        return false;
    }
    if (kdimX1 < KVALUE_MIN || kdimX1 > KVALUE_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "The k-axis should be in range[1, 65535], but it is: %ld.", kdimX1);
        return false;
    }

    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    if (biasOptional != nullptr){
        OP_CHECK_WRONG_DIMENSION(biasOptional, ONE_DIM, return false);
        auto biasDim = biasOptional->GetViewShape().GetDim(0);
        if (biasDim != nVal) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "The n-axis of x2 and bias should be same, but x2's n-axis is: %ld and bias's n-axis is: %ld.", nVal, biasDim);
            return false;
        }
    }

    return true;
}

// 处理支持转置的tensor物理排布不连续问题
aclTensor *TransX2Tensor(const aclTensor *x2)
{
    uint64_t storageShapeDimNum = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = x2->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = x2->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = x2->GetViewShape().GetDim(i);
    }
    // transpose the viewshape last two dimensions
    viewDim[0] = x2->GetViewShape().GetDim(1);
    viewDim[1] = x2->GetViewShape().GetDim(0);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2, &dataType);
    std::vector<int64_t> stride(viewShapeDimNum);
    auto transStride = x2->GetViewStrides();
    stride = std::vector<int64_t>(transStride.begin(), transStride.end());
    // transpose the two dimensions
    stride[0] = transStride[1];
    stride[1] = transStride[0];

    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, x2->GetTensor()->GetAddr());
}

// 检查tensor是否连续
bool IsTransposeLastTwoDims(const aclTensor *tensor) {
    // 当输入tensor的shape小于2或者大于6的时候，返回错误
    if (tensor->GetViewShape().GetDimNum() < 2 || tensor->GetViewShape().GetDimNum() > 6) {
        return false;
    }
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
    // BMM 场景下，Batch维度的stride需要等于 N, D 的乘积
    if (tensor->GetViewStrides()[dim2] == 1
      && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        if (tensor->GetViewShape().GetDim(dim1) == 1
          && tensor->GetViewShape().GetDim(dim2) == 1) {
            return false;
          }
        return true;
      }
    return false;
}

// 检查x2是否合法
bool CheckX2Valid(const aclTensor* x2) {
    if (x2 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x2 should not be null.");
        return false;
    }
  	if (x2->IsEmpty()) {
    	OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x2 do not support empty tensor.");
    	return false;
  	}
    OP_CHECK_WRONG_DIMENSION(x2, TWO_DIMS, return false);
    return true;
}
} // namespace matmul_allto_all_check