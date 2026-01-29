# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import re


def should_skip_directory(dir_name):
    """
    判断是否应该跳过该目录
    """
    skip_dirs = {
        'build', 'cmake', 'common', 'docs', 'examples',
        'experimental', 'scripts', 'tests', 'third_party', '3rd'
    }
    return dir_name in skip_dirs


def should_skip_op(op_name):
    """
    判断是否应该跳过该算子
    """
    skip_ops = [
        "mla_prolog", "mla_prolog_v2", "mla_prolog_v3",
        "all_gather_matmul", "all_gather_matmul_v2", "allto_all_all_gather_batch_mat_mul", "allto_allv_grouped_mat_mul",
        "batch_mat_mul_reduce_scatter_allto_all", "distribute_barrier", "elastic_receivable_info_collect",
        "elastic_receivable_test", "grouped_mat_mul_all_reduce", "grouped_mat_mul_allto_allv",
        "inplace_matmul_all_reduce_add_rms_norm", "matmul_all_reduce", "matmul_all_reduce_add_rms_norm",
        "matmul_reduce_scatter", "matmul_reduce_scatter_v2", "moe_distribute_buffer_reset", "moe_distribute_combine",
        "moe_distribute_combine_add_rms_norm", "moe_distribute_combine_v2", "moe_distribute_dispatch",
        "moe_distribute_dispatch_v2", "moe_update_expert", "quant_all_reduce", "quant_reduce_scatter",
        "moe_finalize_routing_v2", "moe_finalize_routing_v2_grad", "moe_gating_top_k", "moe_gating_top_k_softmax",
        "moe_gating_top_k_softmax_v2", "moe_init_routing", "moe_init_routing_quant_v2", "moe_init_routing_v2",
        "moe_init_routing_v2_grad", "moe_init_routing_v3", "moe_re_routing", "moe_token_permute_with_routing_map"
    ]
    return op_name in skip_ops


def parse_foreach_config(config_str):
    """
    解析 FOREACH_OPDEF 中的配置字符串
    """
    config_mapping = {
        'A2': 'ascend910b',
        '910_93': 'ascend910_93',
        'A5': 'ascend950',
        '910B': 'ascend910b',
        '910B_93': 'ascend910_93',
        '910B_95': 'ascend950',
        '950': 'ascend950',
        '910': 'ascend910',
        '910_55': 'ascend910_55',
    }

    found_configs = []
    config_str_upper = config_str.upper()

    priority_checks = [
        ('A2', 'ascend910b'),
        ('910_93', 'ascend910_93'),
        ('A5', 'ascend950'),
        ('910_55', 'ascend910_55'),
        ('910B', 'ascend910b'),
        ('910B_93', 'ascend910_93'),
        ('910B_95', 'ascend950'),
        ('950', 'ascend950'),
        ('910', 'ascend910'),
    ]

    for key, value in priority_checks:
        if key in config_str_upper and value not in found_configs:
            found_configs.append(value)

    return found_configs


def extract_static_map_configs(content):
    """
    从静态map中提取配置名称
    """
    configs = []

    map_patterns = [
        r'static\s+const\s+std::map<std::string[^>]*>\s+\w+\s*=\s*\{([^}]+)\}',
        r'\{"([a-zA-Z0-9_]+)"[^}]*\}',
    ]

    for pattern in map_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            config_matches = re.findall(r'"([a-zA-Z0-9_]+)"', match)
            for config in config_matches:
                if config not in configs:
                    configs.append(config)

    return list(set(configs))


def extract_set_ascend_config_calls(content):
    """
    提取 SetAscendConfig 调用中的配置名称
    """
    configs = []

    pattern1 = r'SetAscendConfig\([^,]+,\s*"([^"]+)"\)'
    pattern2 = r'SetAscendConfig\([^,]+,\s*"([^"]+)",\s*"([^"]+)"\)'

    matches1 = re.findall(pattern1, content)
    for match in matches1:
        if match not in configs:
            configs.append(match)

    matches2 = re.findall(pattern2, content)
    for match in matches2:
        version, dst_version = match
        if version not in configs:
            configs.append(version)
        if dst_version not in configs:
            configs.append(dst_version)

    return list(set(configs))


def extract_foreach_opdef_configs(content):
    """
    提取 FOREACH_OPDEF 相关格式的配置
    """
    configs = []

    pattern1 = r'FOREACH_OPDEF\(([^,]+),'
    matches1 = re.findall(pattern1, content)
    for match in matches1:
        config_str = match.strip()
        configs.extend(parse_foreach_config(config_str))

    pattern2 = r'FOREACH_OPDEF_END_([^(]+)\('
    matches2 = re.findall(pattern2, content)
    for match in matches2:
        config_str = match.strip()
        configs.extend(parse_foreach_config(config_str))

    return list(set(configs))


def extract_traditional_aicore_configs(content):
    """
    提取传统的 AICore 配置名称
    """
    configs = []

    traditional_patterns = [
        r'this->AICore\(\)\.AddConfig\("([a-zA-Z0-9_]+)"',
        r'\.AddConfig\("([a-zA-Z0-9_]+)"',
        r'AddConfig\("([a-zA-Z0-9_]+)"',
    ]

    for pattern in traditional_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if match not in configs:
                configs.append(match)

    return configs


def extract_ai_core_configs(file_path):
    """
    从 _def.cpp 文件中提取 AICore 配置名称
    """
    configs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 方法1：匹配传统的 AICore 配置
        traditional_configs = extract_traditional_aicore_configs(content)
        if traditional_configs:
            configs.extend(traditional_configs)

        # 方法2：匹配 FOREACH_OPDEF 相关格式
        foreach_configs = extract_foreach_opdef_configs(content)
        if foreach_configs:
            configs.extend(foreach_configs)

        # 方法3：匹配静态map + SetAscendConfig 格式
        static_map_configs = extract_static_map_configs(content)
        set_ascend_configs = extract_set_ascend_config_calls(content)

        all_other_configs = list(set(static_map_configs + set_ascend_configs))
        if all_other_configs:
            configs.extend(all_other_configs)

        # 去重并返回
        return list(set(configs))

    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return []


def update_ai_core_configs(op_name, ai_core_configs):
    if should_skip_op(op_name) and "ascend950" in ai_core_configs:
        ai_core_configs.remove("ascend950")
    return ai_core_configs


def main(repository_path):
    result = []
    for root, dirs, files in os.walk(repository_path):
        # 过滤掉不需要的目录
        dirs[:] = [d for d in dirs if not should_skip_directory(d)]

        for file in files:
            if file.endswith('_def.cpp'):
                full_path = os.path.join(root, file)
                op_name = file.replace('_def.cpp', '')

                # 提取 AICore 配置
                ai_core_configs = extract_ai_core_configs(full_path)
                ai_core_configs = update_ai_core_configs(op_name, ai_core_configs)

                # 创建字典
                op_dict = {
                    op_name: ai_core_configs
                }
                result.append(op_dict)
    result.sort(key=lambda x: next(iter(x)))
    return result