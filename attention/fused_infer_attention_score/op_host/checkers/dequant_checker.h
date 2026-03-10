/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file dequant_checker.h
 * \brief
 */


#ifndef DEQUANT_CHECKER_H
#define DEQUANT_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {

class DequantChecker : public BaseChecker {
public:
    DequantChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~DequantChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiPara(const FiaTilingInfo &fiaInfo) override;

private:
    // 公共校验函数

    // enableNonQuant 相关校验函数
    ge::graphStatus CheckExistenceNoquant(const FiaTilingInfo &fiaInfo);

    // enableFullQuant 相关校验函数
    ge::graphStatus CheckDataTypeFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleDtypeMLAFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleDtypeGQAPerblock(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleDtypeGQAPertensor(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleDtypeFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantModeMLAFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantModeGQAPerblock(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantModeGQAPertensor(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantModeFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleDimNumGQAPerblock(const FiaTilingInfo &fiaInfo);

    ge::graphStatus CheckTensorExistFullquant(const FiaTilingInfo &fiaInfo, const gert::Tensor *tensor,
        const std::string &quantModeName, const std::string &inputName);
    ge::graphStatus CheckTensorNotExistFullquant(const FiaTilingInfo &fiaInfo, const gert::Tensor *tensor,
        const std::string &quantModeName, const std::string &inputName);
   ge::graphStatus CheckExistencePertensorFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckExistenceMLAFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckExistencePerblockFullquant(const FiaTilingInfo &fiaInfo);

    ge::graphStatus CheckFeaturePertensorFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeaturePerblockFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureMLAFullquant(const FiaTilingInfo &fiaInfo);

    ge::graphStatus CheckDequantScaleShapeMLAFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleShapePertensor(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleShapePerblock(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleShapeFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckInputDTypeFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckInputLayoutPerblock(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckInputLayoutPertensor(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckInputLayoutMLAFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckInputLayoutFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckInputAxisFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckN1SizeFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckN2SizeFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckQSSizeFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckGSizeFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDSizeFullquant(const FiaTilingInfo &fiaInfo);

    // enableAntiQuant 相关校验函数
    // SinglePara
    ge::graphStatus CheckSingleParaForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckAntiquantModeForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckInputKVTypeForAntiquant(const FiaTilingInfo &fiaInfo);
    
    // Existence
    ge::graphStatus CheckExistenceForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDescExistenceForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckScaleExistenceForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckOffsetExistenceForAntiquant(const FiaTilingInfo &fiaInfo);

    // Feature
    ge::graphStatus CheckFeatureForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeaturePAForAntiquant(const FiaTilingInfo &fiaInfo);

    // MultiPara
    ge::graphStatus CheckMultiParaForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckScaleTypeForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckScaleShapeForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckOffsetTypeForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckOffsetShapeForAntiquant(const FiaTilingInfo &fiaInfo);

    ge::graphStatus CheckScaleShapeForPerChannelPerTensorMode(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckScaleShapeForPerTokenMode(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKScaleShapeForPerChannelPerTensorMode(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKScaleShapeForPerTokenMode(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKScaleShapeForPerTensorHeadMode(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKScaleShapeForPerTokenHeadMode(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKScaleShapeForPerTokenPAMode(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKScaleShapeForPerTokenHeadPAMode(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKScaleShapeForPerTokenGroupMode(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckVScaleShapeForPerTokenMode(const FiaTilingInfo &fiaInfo);

private:
    bool enablePerblockQuant_ = false;
    bool enablePertensorQuant_ = false;
    bool enableIFAMLAFullQuant_ = false;
};

} // namespace optiling
#endif // DEQUANT_CHECKER_H