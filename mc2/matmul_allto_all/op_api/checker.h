/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_H
#define CHECKER_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

namespace matmul_allto_all_check {

// 检查AlltoAll和Permute数据交换的方向参数, 可以为空和{-2,-1}, 不允许为其他值
bool CheckAlltoAllAxes(const aclIntArray* alltoAllAxesOptional);

// 检查输入的转置配置，x1不允许转置
bool CheckTransposeX1(bool transposeX1);

// 检查通信域名的字符串长度是否符合要求
bool CheckGroupLength(const char *group);

// 校验输入属性shape
bool CheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional, bool transposeX2, const aclTensor* output);

// 处理支持转置的tensor物理排布不连续问题
aclTensor *TransX2Tensor(const aclTensor *x2);

// 检查tensor是否连续
bool IsTransposeLastTwoDims(const aclTensor *tensor);

// 检查x2是否合法，空指针、空tensor和维度
bool CheckX2Valid(const aclTensor* x2);
} // namespace matmul_allto_all_check

#endif //CHECKER_H
