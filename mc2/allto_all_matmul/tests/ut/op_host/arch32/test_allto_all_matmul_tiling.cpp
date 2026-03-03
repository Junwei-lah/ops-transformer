/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_kernel/arch32/allto_all_matmul_tiling_data_910_93.h"
#include "mc2_tiling_case_executor.h"

namespace AlltoAllMatmulUT {

class AlltoAllMatmulA3TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllMatmulA3TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllMatmulA3TilingTest TearDown" << std::endl;
    }
};

TEST_F(AlltoAllMatmulA3TilingTest, Float16Test1)
{
    struct AlltoAllMatmulCompileInfo {} compileInfo;
    uint64_t coreNum = 20;

    gert::TilingContextPara tilingContextPara("AlltoAllMatmul",
        {
            {{{88, 128}, {88, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{44, 256}, {44, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{44, 256}, {44, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"worldsize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"alltoAllAxes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"yDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"x1QuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"x2QuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"x1QuantDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"commQuantDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transposex1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transposex2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"groupSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"alltoalloutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend910_93", coreNum);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

} // AlltoAllMatmulUT