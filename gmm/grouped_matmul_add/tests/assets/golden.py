#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__golden__ = {
        "kernel": {
            "grouped_matmul_add": "grouped_matmul_add_golden"
        }
}

import numpy as np
import torch

def grouped_matmul_add_golden(x, weight, group_list, y, transpose_x: bool = True, transpose_weight: bool = False,
                              group_type: int = 2, group_list_type: int = 0, **kwargs):
    
    # group_list_type 0: cumsum, 1: count
    output_dtypes = kwargs['output_dtypes']
    out_dtype = output_dtypes[0]
    outs = []
    group_num = len(group_list)

    if group_list_type == 1:
        group_list = np.cumsum(group_list)
    x = torch.from_numpy(x.astype(np.float32))
    weight = torch.from_numpy(weight.astype(np.float32))
    for i in range(group_num):
        M = x.shape[-1] 
        N = weight.shape[-1] 
        pre = 0 if i == 0 else group_list[i-1]
        cur = group_list[i]
        if cur - pre == 0: # k is 0
            out = np.zeros((M, N), dtype=out_dtype)
            outs.append(out)
            continue
        x_g = x[pre:cur, :]
        x_g = np.swapaxes(x_g, -1, -2)
        weight_g = weight[pre:cur, :]
        out = torch.matmul(x_g, weight_g).numpy().astype(out_dtype)
        outs.append(out)
        
    real_out = outs if not outs else np.concatenate(outs, axis=0)
    real_out = real_out.reshape(y.shape)
    real_out = real_out + y
    return real_out