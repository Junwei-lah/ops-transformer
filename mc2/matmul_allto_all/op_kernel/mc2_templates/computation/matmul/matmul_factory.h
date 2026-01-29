/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_factory.h
 * \brief
 */

#ifndef MC2_MATMUL_FACTORY_H
#define MC2_MATMUL_FACTORY_H 

namespace MC2KernelTemplate {
//基本输入输出
struct BaseGmAddrs {
    GM_ADDR aGM;
    GM_ADDR bGM;
    GM_ADDR cGM;
    GM_ADDR biasGM;
};
};  // namespace MC2kernelComputation

#endif