#!/bin/bash
# -*- coding: utf-8 -*-
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

# Iterate over all subdirectories in kernels/ and build each
for dir in kernels/*/; do
    echo "Building kernel in: $dir"
    pushd "$dir"
    # Check if build.sh exists and is executable, then run it
    if [ -f "./build.sh" ]; then
        chmod +x ./build.sh  # Ensure it's executable
        ./build.sh
    else
        echo "Warning: build.sh not found in $dir"
    fi
    popd
done