#!/bin/bash
# -*- coding: utf-8 -*-
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

declare -A NumCoresMap
NumCoresMap["Ascend910A"]="32"
NumCoresMap["Ascend910B1"]="25"
NumCoresMap["Ascend910B2"]="24"
NumCoresMap["Ascend910B3"]="20"
NumCoresMap["Ascend910B4"]="20"

export NumCoresMap