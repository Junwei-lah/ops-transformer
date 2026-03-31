# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import itertools
import torch
import torch_npu

# ******入参调用
from test_chunk_gated_delta_rule_paramset import ENABLED_PARAMS

# ******CPU侧算子逻辑实现获取golden与npu算子直调结果
import chunk_gated_delta_rule_operator_single
import pytest

locals()["param_combinations"] = []

for _, params in enumerate(ENABLED_PARAMS):
    # 将params的所有字段注册为局部变量
    for key, value in params.items():
        locals()[f"param_{key}"] = value

    # 生成所有参数组合
    param_names = [
        "B", "seqlen", "nk", "nv", "dk", "dv", "chunk_size", "data_type"
    ]

    param_values = [
        locals()["param_B"],
        locals()["param_seqlen"],
        locals()["param_nk"],
        locals()["param_nv"],
        locals()["param_dk"],
        locals()["param_dv"],
        locals()["param_chunk_size"],
        locals()["param_data_type"],
    ]

    # 生成所有的组合，并转换为字典列表
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        locals()["param_combinations"].append(param_dict)

    @pytest.mark.ci
    @pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
    def test_chunk_gated_delta_rule(param_combinations):
        # 初始化参数和tensor
        B = param_combinations['B']
        seqlen = param_combinations['seqlen']
        nk = param_combinations['nk']
        nv = param_combinations['nv']
        dk = param_combinations['dk']
        dv = param_combinations['dv']
        chunk_size = param_combinations['chunk_size']
        data_type = param_combinations['data_type']

        test_data = B, seqlen, nk, nv, dk, dv, chunk_size, data_type

        torch_npu.npu.set_device(0)

        # 豀得cpu结果(真值)和算子结果（测试值)
        chunk_gated_delta_rule_operator_single.run_precision_test(test_data)
