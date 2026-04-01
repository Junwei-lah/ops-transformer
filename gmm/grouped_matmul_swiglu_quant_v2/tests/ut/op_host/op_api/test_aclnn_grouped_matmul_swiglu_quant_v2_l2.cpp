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
 * \file test_aclnn_grouped_matmul_swiglu_quant_v2_l2.cpp
 * \brief CSV-driven unit tests for grouped_matmul_swiglu_quant_v2 opapi.
 */

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_grouped_matmul_swiglu_quant_v2.h"
#include "../../../../op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace std;
using namespace op;

namespace {

constexpr size_t kPathBufferSize = 4096;
constexpr size_t kSwigluCsvColumnCount = 45;

string GetExeDirPath()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    DWORD len = GetModuleFileNameA(nullptr, path, MAX_PATH);
    string exePath(path, len);
    auto pos = exePath.find_last_of("\\/");
    return pos == string::npos ? string(".\\") : exePath.substr(0, pos + 1);
#else
    char path[kPathBufferSize] = {0};
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len <= 0) {
        return "./";
    }
    path[len] = '\0';
    string exePath(path);
    auto pos = exePath.find_last_of('/');
    return pos == string::npos ? string("./") : exePath.substr(0, pos + 1);
#endif
}

string GetCurrentFileDir()
{
    string path = __FILE__;
    auto pos = path.find_last_of("\\/");
    return pos == string::npos ? string(".") : path.substr(0, pos);
}

string JoinPath(const string &dir, const string &file)
{
    if (dir.empty()) {
        return file;
    }
    char last = dir.back();
    if (last == '/' || last == '\\') {
        return dir + file;
    }
    return dir + "/" + file;
}

bool FileExists(const string &path)
{
    ifstream in(path);
    return in.is_open();
}

string ResolveCsvPath(const string &csvName)
{
    const string exePath = JoinPath(GetExeDirPath(), csvName);
    if (FileExists(exePath)) {
        return exePath;
    }
    const string srcPath = JoinPath(GetCurrentFileDir(), csvName);
    if (FileExists(srcPath)) {
        return srcPath;
    }
    return exePath;
}

void SplitStr2Vec(const string &input, const string &delimiter, vector<string> &output)
{
    const auto delimiterLen = delimiter.size();
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

string Trim(string value)
{
    auto isNotSpace = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), isNotSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), isNotSpace).base(), value.end());
    return value;
}

bool ParseBool(const string &value)
{
    string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower == "true" || lower == "1" || lower == "yes";
}

vector<int64_t> ParseI64List(const string &value, const string &sep = "|")
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> tokens;
    SplitStr2Vec(trimmed, sep, tokens);
    vector<int64_t> out;
    out.reserve(tokens.size());
    for (const auto &token : tokens) {
        out.emplace_back(stoll(Trim(token)));
    }
    return out;
}

vector<int64_t> ParseDims(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> tokens;
    SplitStr2Vec(trimmed, ":", tokens);
    vector<int64_t> out;
    out.reserve(tokens.size());
    for (const auto &token : tokens) {
        out.emplace_back(stoll(Trim(token)));
    }
    return out;
}

vector<vector<int64_t>> ParseDimsList(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> items;
    SplitStr2Vec(trimmed, ";", items);
    vector<vector<int64_t>> out;
    out.reserve(items.size());
    for (const auto &item : items) {
        out.emplace_back(ParseDims(item));
    }
    return out;
}

aclDataType ParseDtype(const string &dtype)
{
    static const map<string, aclDataType> kMap = {
        {"FLOAT", ACL_FLOAT},
        {"FLOAT16", ACL_FLOAT16},
        {"BF16", ACL_BF16},
        {"INT8", ACL_INT8},
        {"INT4", ACL_INT4},
        {"INT64", ACL_INT64},
        {"UINT64", ACL_UINT64},
        {"FLOAT8_E5M2", ACL_FLOAT8_E5M2},
        {"FLOAT8_E8M0", ACL_FLOAT8_E8M0},
    };
    auto it = kMap.find(dtype);
    return it == kMap.end() ? ACL_DT_UNDEFINED : it->second;
}

aclFormat ParseFormat(const string &format)
{
    static const map<string, aclFormat> kMap = {
        {"ND", ACL_FORMAT_ND},
        {"FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ},
    };
    auto it = kMap.find(format);
    return it == kMap.end() ? ACL_FORMAT_ND : it->second;
}

void SetupPlatformForCase(const string &socVersion)
{
    static const map<string, op::SocVersion> socMap = {
        {"Ascend910B", op::SocVersion::ASCEND910B},
        {"Ascend950", op::SocVersion::ASCEND950},
    };
    auto it = socMap.find(socVersion);
    op::SetPlatformSocVersion(it == socMap.end() ? op::SocVersion::ASCEND910B : it->second);
}

TensorDesc MakeTensorDesc(const vector<int64_t> &shape, aclDataType dtype, aclFormat format,
                          const vector<int64_t> &storageShape, const vector<int64_t> &stride,
                          bool allowNdCustomStorage = false)
{
    if (format == ACL_FORMAT_ND && !allowNdCustomStorage) {
        return TensorDesc(shape, dtype, format).ValueRange(-10, 10);
    }
    if (!storageShape.empty() || !stride.empty()) {
        return TensorDesc(shape, dtype, format, stride, 0, storageShape).ValueRange(-10, 10);
    }
    return TensorDesc(shape, dtype, format).ValueRange(-10, 10);
}

TensorDesc MakeTensorDescWithRange(const vector<int64_t> &shape, aclDataType dtype, aclFormat format,
                                   const vector<int64_t> &storageShape, const vector<int64_t> &stride,
                                   int64_t low, int64_t high, bool allowNdCustomStorage = false)
{
    TensorDesc desc = MakeTensorDesc(shape, dtype, format, storageShape, stride, allowNdCustomStorage);
    return desc.ValueRange(low, high);
}

TensorListDesc MakeTensorListDesc(const vector<vector<int64_t>> &shapes, aclDataType dtype, aclFormat format,
                                  const vector<vector<int64_t>> &storageShapes, const vector<vector<int64_t>> &strides,
                                  bool allowNdCustomStorage = false)
{
    vector<TensorDesc> descs;
    descs.reserve(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
        static const vector<int64_t> empty;
        const auto &storage = i < storageShapes.size() ? storageShapes[i] : empty;
        const auto &stride = i < strides.size() ? strides[i] : empty;
        descs.emplace_back(MakeTensorDesc(shapes[i], dtype, format, storage, stride, allowNdCustomStorage));
    }
    return TensorListDesc(descs);
}

TensorListDesc MakeTensorListDescWithRange(const vector<vector<int64_t>> &shapes, aclDataType dtype, aclFormat format,
                                           const vector<vector<int64_t>> &storageShapes,
                                           const vector<vector<int64_t>> &strides, int64_t low, int64_t high,
                                           bool allowNdCustomStorage = false)
{
    vector<TensorDesc> descs;
    descs.reserve(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
        static const vector<int64_t> empty;
        const auto &storage = i < storageShapes.size() ? storageShapes[i] : empty;
        const auto &stride = i < strides.size() ? strides[i] : empty;
        descs.emplace_back(MakeTensorDescWithRange(shapes[i], dtype, format, storage, stride, low, high, allowNdCustomStorage));
    }
    return TensorListDesc(descs);
}

void FallbackSingleNzWeightToNdStorage(const string &api, aclFormat &weightFormat,
                                       vector<vector<int64_t>> &weightShapesVec,
                                       vector<vector<int64_t>> &weightStorageShapesVec)
{
    if (api != "WeightNzV2" || weightFormat != ACL_FORMAT_FRACTAL_NZ) {
        return;
    }
    if (weightShapesVec.size() != 1 || !weightStorageShapesVec.empty()) {
        return;
    }
    const auto &w = weightShapesVec[0];
    if (w.size() != 5 || w[0] <= 0 || w[3] != 16 || w[4] != 32) {
        return;
    }
    // Some environments crash while creating single 5D FRACTAL_NZ desc in TensorList.
    // Fallback to equivalent ND view + NZ storage, same as stable hand-written UT path.
    const int64_t e = w[0];
    const int64_t n = w[1] * w[4];
    const int64_t k = w[2] * w[3];
    weightShapesVec[0] = {e, k, n};
    weightStorageShapesVec = {w};
    weightFormat = ACL_FORMAT_ND;
}

struct SwigluOpApiCase {
    string socVersion;
    string caseName;
    string api;
    string checkRet;
    uint64_t expectRet;
    string xShape;
    string xDtype;
    string xFormat;
    string xStorageShape;
    string xStride;
    string weightShapes;
    string weightDtype;
    string weightFormat;
    string weightStorageShapes;
    string weightStrides;
    string weightScaleShapes;
    string weightScaleDtype;
    string weightScaleFormat;
    string weightScaleStorageShapes;
    string weightScaleStrides;
    string weightAssistShapes;
    string weightAssistDtype;
    string weightAssistFormat;
    string weightAssistStorageShapes;
    string weightAssistStrides;
    string xScaleShape;
    string xScaleDtype;
    string xScaleFormat;
    string smoothScaleShape;
    string smoothScaleDtype;
    string smoothScaleFormat;
    string groupListShape;
    string groupListDtype;
    string groupListFormat;
    int64_t dequantMode;
    int64_t dequantDtype;
    int64_t quantMode;
    int64_t quantDtype;
    string tuningConfig;
    string out1Shape;
    string out1Dtype;
    string out1Format;
    string out2Shape;
    string out2Dtype;
    string out2Format;

    void Run() const
    {
        SetupPlatformForCase(socVersion);
        auto x = MakeTensorDescWithRange(ParseDims(xShape), ParseDtype(xDtype), ParseFormat(xFormat), ParseDims(xStorageShape),
                                         ParseDims(xStride), -1, 1);
        auto weightShapesVec = ParseDimsList(weightShapes);
        auto weightStorageShapesVec = ParseDimsList(weightStorageShapes);
        auto weightFormatVal = ParseFormat(weightFormat);
        auto weightScaleShapesVec = ParseDimsList(weightScaleShapes);
        FallbackSingleNzWeightToNdStorage(api, weightFormatVal, weightShapesVec, weightStorageShapesVec);
        const bool forceWeightStorage = (api == "WeightNzV2");
        auto weight = MakeTensorListDescWithRange(weightShapesVec, ParseDtype(weightDtype), weightFormatVal,
                                                  weightStorageShapesVec, ParseDimsList(weightStrides),
                                                  -1, 1, forceWeightStorage);
        auto xScale = MakeTensorDescWithRange(ParseDims(xScaleShape), ParseDtype(xScaleDtype), ParseFormat(xScaleFormat), {}, {},
                                              0, 3);
        auto groupList = MakeTensorDescWithRange(ParseDims(groupListShape), ParseDtype(groupListDtype),
                                                 ParseFormat(groupListFormat), {}, {}, 0, 64);
        auto out1 = MakeTensorDescWithRange(ParseDims(out1Shape), ParseDtype(out1Dtype), ParseFormat(out1Format), {}, {},
                                            -1, 1);
        auto out2 = MakeTensorDescWithRange(ParseDims(out2Shape), ParseDtype(out2Dtype), ParseFormat(out2Format), {}, {},
                                            -1, 1);

        const auto weightAssistShapesVec = ParseDimsList(weightAssistShapes);
        const auto smoothScaleDims = ParseDims(smoothScaleShape);
        const bool hasWeightScale = !weightScaleShapesVec.empty();
        const bool hasWeightAssist = !weightAssistShapesVec.empty();
        const bool hasSmoothScale = !smoothScaleDims.empty();

        vector<int64_t> tuningValues = ParseI64List(tuningConfig, "|");
        aclIntArray *tuningConfigArr = tuningValues.empty() ? nullptr : aclCreateIntArray(tuningValues.data(), tuningValues.size());

        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACL_SUCCESS;

        // Optional inputs must use nullptr; do not build empty TensorListDesc.
        if (!hasWeightScale) {
            auto weightScale = nullptr;
            if (hasWeightAssist && hasSmoothScale) {
                auto weightAssist = MakeTensorListDesc(weightAssistShapesVec, ParseDtype(weightAssistDtype),
                                                       ParseFormat(weightAssistFormat), ParseDimsList(weightAssistStorageShapes),
                                                       ParseDimsList(weightAssistStrides));
                auto smoothScale =
                    MakeTensorDesc(smoothScaleDims, ParseDtype(smoothScaleDtype), ParseFormat(smoothScaleFormat), {}, {});
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else if (hasWeightAssist) {
                auto weightAssist = MakeTensorListDesc(weightAssistShapesVec, ParseDtype(weightAssistDtype),
                                                       ParseFormat(weightAssistFormat), ParseDimsList(weightAssistStorageShapes),
                                                       ParseDimsList(weightAssistStrides));
                auto smoothScale = nullptr;
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else if (hasSmoothScale) {
                auto weightAssist = nullptr;
                auto smoothScale =
                    MakeTensorDesc(smoothScaleDims, ParseDtype(smoothScaleDtype), ParseFormat(smoothScaleFormat), {}, {});
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else {
                auto weightAssist = nullptr;
                auto smoothScale = nullptr;
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            }
        } else {
            auto weightScale = MakeTensorListDescWithRange(weightScaleShapesVec, ParseDtype(weightScaleDtype),
                                                           ParseFormat(weightScaleFormat),
                                                           ParseDimsList(weightScaleStorageShapes),
                                                           ParseDimsList(weightScaleStrides), 0, 3);
            if (hasWeightAssist && hasSmoothScale) {
                auto weightAssist = MakeTensorListDesc(weightAssistShapesVec, ParseDtype(weightAssistDtype),
                                                       ParseFormat(weightAssistFormat), ParseDimsList(weightAssistStorageShapes),
                                                       ParseDimsList(weightAssistStrides));
                auto smoothScale =
                    MakeTensorDesc(smoothScaleDims, ParseDtype(smoothScaleDtype), ParseFormat(smoothScaleFormat), {}, {});
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else if (hasWeightAssist) {
                auto weightAssist = MakeTensorListDesc(weightAssistShapesVec, ParseDtype(weightAssistDtype),
                                                       ParseFormat(weightAssistFormat), ParseDimsList(weightAssistStorageShapes),
                                                       ParseDimsList(weightAssistStrides));
                auto smoothScale = nullptr;
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else if (hasSmoothScale) {
                auto weightAssist = nullptr;
                auto smoothScale =
                    MakeTensorDesc(smoothScaleDims, ParseDtype(smoothScaleDtype), ParseFormat(smoothScaleFormat), {}, {});
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else {
                auto weightAssist = nullptr;
                auto smoothScale = nullptr;
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1, out2));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            }
        }

        if (tuningConfigArr != nullptr) {
            aclDestroyIntArray(tuningConfigArr);
        }
        if (ParseBool(checkRet)) {
            EXPECT_EQ(ret, static_cast<aclnnStatus>(expectRet));
        }
    }
};

vector<SwigluOpApiCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<SwigluOpApiCase> cases;
    string line;
    bool headerSkipped = false;
    while (getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }
        vector<string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() != kSwigluCsvColumnCount) {
            continue;
        }
        SwigluOpApiCase c;
        size_t i = 0;
        c.socVersion = Trim(cols[i++]);
        c.caseName = Trim(cols[i++]);
        c.api = Trim(cols[i++]);
        c.checkRet = Trim(cols[i++]);
        c.expectRet = static_cast<uint64_t>(stoull(Trim(cols[i++])));
        c.xShape = Trim(cols[i++]);
        c.xDtype = Trim(cols[i++]);
        c.xFormat = Trim(cols[i++]);
        c.xStorageShape = Trim(cols[i++]);
        c.xStride = Trim(cols[i++]);
        c.weightShapes = Trim(cols[i++]);
        c.weightDtype = Trim(cols[i++]);
        c.weightFormat = Trim(cols[i++]);
        c.weightStorageShapes = Trim(cols[i++]);
        c.weightStrides = Trim(cols[i++]);
        c.weightScaleShapes = Trim(cols[i++]);
        c.weightScaleDtype = Trim(cols[i++]);
        c.weightScaleFormat = Trim(cols[i++]);
        c.weightScaleStorageShapes = Trim(cols[i++]);
        c.weightScaleStrides = Trim(cols[i++]);
        c.weightAssistShapes = Trim(cols[i++]);
        c.weightAssistDtype = Trim(cols[i++]);
        c.weightAssistFormat = Trim(cols[i++]);
        c.weightAssistStorageShapes = Trim(cols[i++]);
        c.weightAssistStrides = Trim(cols[i++]);
        c.xScaleShape = Trim(cols[i++]);
        c.xScaleDtype = Trim(cols[i++]);
        c.xScaleFormat = Trim(cols[i++]);
        c.smoothScaleShape = Trim(cols[i++]);
        c.smoothScaleDtype = Trim(cols[i++]);
        c.smoothScaleFormat = Trim(cols[i++]);
        c.groupListShape = Trim(cols[i++]);
        c.groupListDtype = Trim(cols[i++]);
        c.groupListFormat = Trim(cols[i++]);
        c.dequantMode = stoll(Trim(cols[i++]));
        c.dequantDtype = stoll(Trim(cols[i++]));
        c.quantMode = stoll(Trim(cols[i++]));
        c.quantDtype = stoll(Trim(cols[i++]));
        c.tuningConfig = Trim(cols[i++]);
        c.out1Shape = Trim(cols[i++]);
        c.out1Dtype = Trim(cols[i++]);
        c.out1Format = Trim(cols[i++]);
        c.out2Shape = Trim(cols[i++]);
        c.out2Dtype = Trim(cols[i++]);
        c.out2Format = Trim(cols[i++]);
        if (c.socVersion == "Ascend950") {
            cases.emplace_back(c);
        }
    }
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<SwigluOpApiCase> &info)
{
    string name = info.param.caseName;
    for (char &c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
            c = '_';
        }
    }
    return name;
}

class grouped_matmul_swiglu_quant_v2_opapi_csv_test : public testing::TestWithParam<SwigluOpApiCase> {};

TEST_P(grouped_matmul_swiglu_quant_v2_opapi_csv_test, run_case)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(
    grouped_matmul_swiglu_quant_v2_opapi_csv,
    grouped_matmul_swiglu_quant_v2_opapi_csv_test,
    testing::ValuesIn(LoadCases(ResolveCsvPath("test_aclnn_grouped_matmul_swiglu_quant_v2_l2.csv"))),
    BuildCaseName);

}  // namespace
