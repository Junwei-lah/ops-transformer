/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_quant_arch35_tiling.cpp
 * \brief CSV-driven unit tests for GroupedQmm arch35 tiling.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../../op_host/op_tiling/arch35/grouped_quant_matmul_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

namespace {

std::string GetExeDirPath()
{
    std::string exe_path("./");
    char path[1024];
    ssize_t n = readlink("/proc/self/exe", path, sizeof(path));
    if (n > 0) {
        path[n] = '\0';
        exe_path.assign(path);
        auto pos = exe_path.find_last_of('/');
        if (pos != std::string::npos) {
        exe_path.erase(pos + 1);
        } else {
        exe_path.assign("./");
        }
    }
    
    return exe_path;
}

string ToLower(string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return value;
}

bool ParseBool(const string &value)
{
    auto lower = ToLower(value);
    return lower == "true" || lower == "1" || lower == "yes";
}

void SplitStr2Vec(const string &input, const string &delimiter, vector<string> &output)
{
    auto delimiterLen = delimiter.size();
    string::size_type currPos = 0;
    string::size_type nextPos = input.find(delimiter, currPos);
    while (nextPos != string::npos) {
        output.emplace_back(input.substr(currPos, nextPos - currPos));
        currPos = nextPos + delimiterLen;
        nextPos = input.find(delimiter, currPos);
    }

    if (currPos <= input.size()) {
        output.emplace_back(input.substr(currPos));
    }
}

DataType ParseDtype(const string &dtype)
{
    static const map<string, DataType> dtypeMap = {
        {"FLOAT", ge::DT_FLOAT},
        {"FLOAT16", ge::DT_FLOAT16},
        {"BF16", ge::DT_BF16},
        {"INT64", ge::DT_INT64},
        {"FLOAT8-E8M0", ge::DT_FLOAT8_E8M0},
        {"FLOAT8-E4M3", ge::DT_FLOAT8_E4M3FN},
        {"FLOAT8-E5M2", ge::DT_FLOAT8_E5M2},
    };
    auto it = dtypeMap.find(dtype);
    if (it == dtypeMap.end()) {
        cerr << "Unsupported dtype in csv: " << dtype << endl;
        return ge::DT_UNDEFINED;
    }
    return it == dtypeMap.end() ? ge::DT_UNDEFINED : it->second;
}

gert::Shape BuildShape(const vector<int64_t> &dims)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
    return shape;
}

gert::StorageShape MakeShape(const vector<int64_t> &originDims, const vector<int64_t> &storageDims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape() = BuildShape(originDims);
    shape.MutableStorageShape() = BuildShape(storageDims);
    return shape;
}

gert::StorageShape MakeEmptyShape()
{
    return gert::StorageShape();
}

string TilingData2Str(const void *tilingData, size_t tilingSize)
{
    ostringstream oss;
    const auto *data = reinterpret_cast<const int32_t *>(tilingData);
    const size_t len = tilingSize / sizeof(int32_t);
    for (size_t i = 0; i < len; ++i) {
        if (i != 0) {
            oss << " ";
        }
        oss << data[i];
    }
    return oss.str();
}

vector<gert::TilingContextPara::OpAttr> GetGroupedQmmAttrs(bool transposeX, bool transposeWeight, int64_t groupType)
{
    return {
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeX)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupType)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    };
}

class GroupedQuantArch35TilingTestParam {
public:
    void Prepare(optiling::GMMCompileInfo &compileInfo) const
    {
        compileInfo = {
            static_cast<uint32_t>(coreNum > 0 ? coreNum : 32),     // aicNum
            static_cast<uint32_t>(coreNum > 0 ? coreNum * 2 : 64), // aivNum
            262144,                                                // ubSize
            524288,                                                // l1Size
            134217728,                                             // l2Size
            262144,                                                // l0CSize
            65536,                                                 // l0ASize
            65536,                                                 // l0BSize
            platform_ascendc::SocVersion::ASCEND950,               // socVersion
            NpuArch::DAV_3510,
        };
    }

    void InvokeTilingFunc(optiling::GMMCompileInfo &compileInfo) const
    {
        vector<int64_t> xOriginDims = transposeX ? vector<int64_t>{k, m} : vector<int64_t>{m, k};
        gert::StorageShape xShape = MakeShape(xOriginDims, xOriginDims);
        gert::StorageShape biasShape = hasBias ? MakeShape({groupNum, n}, {groupNum, n}) : MakeEmptyShape();
        gert::StorageShape groupListShape = MakeShape({groupNum}, {groupNum});
        int64_t kScaleBlocks = (k + 63) / 64;
        gert::StorageShape perTokenScaleShape = MakeShape({m, kScaleBlocks, 2}, {m, kScaleBlocks, 2});

        vector<int64_t> weightOriginDims =
            transposeWeight ? vector<int64_t>{groupNum, n, k} : vector<int64_t>{groupNum, k, n};
        vector<int64_t> weightStorageDims;
        if (weightFormat == "NZ") {
            weightStorageDims = transposeWeight ? vector<int64_t>{groupNum, (k + 31) / 32, (n + 15) / 16, 16, 32} :
                                                  vector<int64_t>{groupNum, (n + 31) / 32, (k + 15) / 16, 16, 32};
        } else {
            weightStorageDims = weightOriginDims;
        }
        gert::StorageShape weightShape = MakeShape(weightOriginDims, weightStorageDims);

        vector<int64_t> scaleDims = transposeWeight ? vector<int64_t>{groupNum, n, kScaleBlocks, 2} :
                                                      vector<int64_t>{groupNum, kScaleBlocks, n, 2};
        gert::StorageShape scaleShape = MakeShape(scaleDims, scaleDims);

        gert::TilingContextPara tilingContextPara(
            "GroupedMatmul",
            {
                {xShape, xDtype, ge::FORMAT_ND},                                                          // x
                {weightShape, weightDtype, weightFormat == "NZ" ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND}, // weight
                {biasShape, biasDtype, ge::FORMAT_ND},                                                    // bias
                {scaleShape, scaleDtype, ge::FORMAT_ND},                                                  // scale
                {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},                                          // offset
                {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},         // antiquantScale
                {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},         // antiquantOffset
                {groupListShape, ge::DT_INT64, ge::FORMAT_ND},           // groupList
                {perTokenScaleShape, perTokenScaleDtype, ge::FORMAT_ND}, // perTokenScale
            },
            {{MakeShape({m}, {n}), yDtype, ge::FORMAT_ND}}, GetGroupedQmmAttrs(transposeX, transposeWeight, groupType),
            &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize);

        TilingInfo tilingInfo;
        bool tilingResult = ExecuteTiling(tilingContextPara, tilingInfo);
        ASSERT_EQ(tilingResult, result) << "prefix=" << prefix << ", case=" << caseName;
        if (!result) {
            return;
        }

        ASSERT_EQ(tilingInfo.blockNum, expectBlockDim) << "prefix=" << prefix;
        ASSERT_EQ(tilingInfo.tilingKey, expectTilingKey) << "prefix=" << prefix;
        ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMQuantBasicApiTilingData))
            << "prefix=" << prefix;

        string actualTilingData = TilingData2Str(tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
        ASSERT_EQ(actualTilingData, expectTilingData) << "prefix=" << prefix;
    }

    void Test() const
    {
        optiling::GMMCompileInfo compileInfo;
        Prepare(compileInfo);
        InvokeTilingFunc(compileInfo);
    }

    string socVersion;
    string caseName;
    bool enable = true;
    string prefix;
    int64_t coreNum = -1;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t groupNum = 0;
    bool transposeX = false;
    bool transposeWeight = false;
    int64_t groupType = 0;
    string weightFormat;
    ge::DataType xDtype = ge::DT_UNDEFINED;
    ge::DataType weightDtype = ge::DT_UNDEFINED;
    ge::DataType scaleDtype = ge::DT_UNDEFINED;
    ge::DataType perTokenScaleDtype = ge::DT_UNDEFINED;
    ge::DataType biasDtype = ge::DT_UNDEFINED;
    ge::DataType yDtype = ge::DT_UNDEFINED;
    bool hasBias = false;
    bool result = false;
    uint64_t expectBlockDim = 0;
    uint64_t expectTilingKey = 0;
    string expectTilingData;
};

vector<GroupedQuantArch35TilingTestParam> GetParams(const string &socVersion)
{
    vector<GroupedQuantArch35TilingTestParam> params;
    std::string rootPath(GetExeDirPath() + "../../../../../");
    std::string csvPath(rootPath + "gmm/grouped_matmul/tests/ut/op_host/test_grouped_quant_arch35_tiling.csv");
    ifstream csvData(csvPath, ios::in);
    if (!csvData.is_open()) {
        cout << "cannot open case file " << csvPath << ", maybe not exist" << endl;
        return params;
    }

    string line;
    while (getline(csvData, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        vector<string> items;
        SplitStr2Vec(line, ",", items);
        if (items.empty() || items[0] == "socVersion" || items.size() < 24U) {
            continue;
        }

        GroupedQuantArch35TilingTestParam param;
        size_t idx = 0UL;
        param.socVersion = items[idx++];
        if (param.socVersion != socVersion) {
            continue;
        }

        param.caseName = items[idx++];
        param.enable = ParseBool(items[idx++]);
        if (!param.enable) {
            continue;
        }
        param.prefix = items[idx++];
        param.coreNum = items[idx].empty() ? -1 : stoll(items[idx]);
        idx++;
        param.m = stoll(items[idx++]);
        param.k = stoll(items[idx++]);
        param.n = stoll(items[idx++]);
        param.groupNum = stoll(items[idx++]);
        param.transposeX = ParseBool(items[idx++]);
        param.transposeWeight = ParseBool(items[idx++]);
        param.groupType = stoll(items[idx++]);
        param.weightFormat = items[idx++];
        param.xDtype = ParseDtype(items[idx++]);
        param.weightDtype = ParseDtype(items[idx++]);
        param.scaleDtype = ParseDtype(items[idx++]);
        param.perTokenScaleDtype = ParseDtype(items[idx++]);
        param.biasDtype = ParseDtype(items[idx++]);
        param.yDtype = ParseDtype(items[idx++]);
        param.hasBias = ParseBool(items[idx++]);
        param.result = ParseBool(items[idx++]);
        param.expectBlockDim = static_cast<uint64_t>(stoull(items[idx++]));
        param.expectTilingKey = static_cast<uint64_t>(stoull(items[idx++]));
        param.expectTilingData = items[idx++];
        params.push_back(param);
    }
    return params;
}

string MakeParamName(const testing::TestParamInfo<GroupedQuantArch35TilingTestParam> &info)
{
    string name = info.param.prefix;
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return isalnum(c) ? static_cast<char>(c) : '_'; });
    return name;
}

} // namespace

namespace GroupedQuantArch35TilingUT {

const vector<GroupedQuantArch35TilingTestParam> &GetAscend950Params()
{
    static const vector<GroupedQuantArch35TilingTestParam> params = GetParams("Ascend950");
    return params;
}

class TestGroupedQuantArch35Tiling : public testing::TestWithParam<GroupedQuantArch35TilingTestParam> {
protected:
    static void SetUpTestCase()
    {
    }
    static void TearDownTestCase()
    {
    }
};

TEST_P(TestGroupedQuantArch35Tiling, generalTest)
{
    GetParam().Test();
}

INSTANTIATE_TEST_SUITE_P(GROUPED_QMM_950, TestGroupedQuantArch35Tiling, testing::ValuesIn(GetAscend950Params()),
                         MakeParamName);

} // namespace GroupedQuantArch35TilingUT
