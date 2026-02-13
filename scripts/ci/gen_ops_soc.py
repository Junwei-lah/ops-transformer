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


def get_sh_files(gen_dir):
    """获取目录中所有 .sh 文件名（不包含路径）"""
    sh_files = []
    for item in os.listdir(gen_dir):
        item_path = os.path.join(gen_dir, item)
        if os.path.isfile(item_path) and item.lower().endswith('.sh'):
            sh_files.append(item)
    return sh_files


def parse_opname_from_filename(filename):
    """
    从文件名解析 op_name。
    要求格式：xxx-<opname>-<digits>.sh
    成功返回 op_name，失败返回 None
    """
    parts = filename.split('-')
    if len(parts) < 3:
        return None

    return parts[1]


def count_opnames(sh_filenames):
    """统计每个 op_name 出现的次数"""
    opname_to_count = {}
    for filename in sh_filenames:
        op_name = parse_opname_from_filename(filename)
        if op_name is not None:
            opname_to_count[op_name] = opname_to_count.get(op_name, 0) + 1
    return opname_to_count


def grouped(gen_path, soc, group_size):
    result: list[list[str]] = [[] for _ in range(group_size)]
    if not os.path.isdir(gen_path):
        return result
    sh_files = get_sh_files(gen_path)
    op_counts = count_opnames(sh_files)

    all_rows = []
    for op_name, count in op_counts.items():
        for i in range(count):
            row_string = f"{op_name},{count}-{i}"
            all_rows.append(row_string)

    for idx, row in enumerate(all_rows):
        result[idx % group_size].append(row)

    return result


def main(repository_path, soc, group_size=1):
    project_path = os.path.abspath(repository_path)
    gen_path = os.path.abspath(os.path.join(project_path, "build", "binary", soc, "gen"))
    op_data = grouped(gen_path, soc, group_size)
    return op_data