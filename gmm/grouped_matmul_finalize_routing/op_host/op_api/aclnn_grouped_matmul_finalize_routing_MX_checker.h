/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_QUANT_GROUPED_MATMUL_FINALIZE_ROUTING_950_CHECKER_H
#define OP_API_INC_QUANT_GROUPED_MATMUL_FINALIZE_ROUTING_950_CHECKER_H
#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_quant_grouped_matmul_finalize_routing_util.h"
#include "../../../grouped_matmul/op_host/op_api/aclnn_grouped_matmul_util.h"
#include "aclnn_quant_grouped_matmul_finalize_routing_util.h"
#include "util/math_util.h"

using namespace GmmFinalizeRouting;

namespace GmmFinalizeRouting {

constexpr size_t ZERO_DIM = 0UL;
constexpr size_t ONE_DIM = 1UL;
constexpr size_t TWO_DIM = 2UL;
constexpr size_t THERE_DIM = 3UL;
constexpr size_t FOUR_DIM = 4UL;
constexpr int64_t GMMFR_SPLIT_SIZE = 64L;
constexpr int64_t GMMFR_SPLIT_FACTOR = 2L;
constexpr int64_t MOD2 = 2L;
constexpr int64_t MAX_NUM_EXPERTS = 1024L;

const std::initializer_list<DataType> MX_IN_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2,
                                                                 op::DataType::DT_FLOAT4_E1M2, op::DataType::DT_FLOAT4_E2M1};
const std::initializer_list<DataType> MXFP4_IN_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT4_E1M2, op::DataType::DT_FLOAT4_E2M1};
const std::initializer_list<DataType> MXFP8_IN_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT4_E1M2, op::DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<op::DataType> MX_SCALE_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> MX_ROW_INDEX_TYPE_SUPPORT_LIST = {op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> MX_BIAS_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> MX_PERTOKEN_SCALE_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> GROUP_LIST_TYPE_SUPPORT_LIST = {op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> SHARED_INPUT_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> LOGIT_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};


class AclnnGroupedMatmulFinalizeRoutingDAV3510Checker {
public:
    explicit AclnnGroupedMatmulFinalizeRoutingDAV3510Checker() {};
    ~AclnnGroupedMatmulFinalizeRoutingDAV3510Checker() {};
    aclnnStatus CheckParams(GroupedMatmulParams &gmmParams)
    {
        gmmParams_ = gmmParams;
        // 1. 检查参数是否为空指针
        CHECK_RET(CheckNotNull() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
        // 2. 检查输入的数据类型是否在支持的数据类型范围之内
        CHECK_RET(CheckDtypeValid(), ACLNN_ERR_PARAM_INVALID);
        // 3. 校验输入、输出参数维度
        CHECK_RET(CheckInputOutDims() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        // 4. 校验输入、输出shape参数
        CHECK_RET(CheckInputOutShape(), ACLNN_ERR_PARAM_INVALID);
        // 5. 校验输入、输出shape参数针对MXFP4
        if (CheckType(gmmParams_.x1->GetDataType(), MXFP4_IN_TYPE_SUPPORT_LIST)) {
            CHECK_RET(CheckInputOutShapeForMXFP4(), ACLNN_ERR_PARAM_INVALID);
        }
        // 6. 检查数据形状是否支持
        CHECK_RET(CheckFormat(), ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckNotNull()
    {
        CHECK_COND(gmmParams_.x1 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In MX quant , x should not be nullptr.");
        CHECK_COND(gmmParams_.x2 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In MX quant , weight should not be nullptr.");
        CHECK_COND(gmmParams_.scale != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "In MX quant, scaleOptional should not be nullptr.");
        CHECK_COND(gmmParams_.pertokenScaleOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "In MX quant, perTokenScaleOptional should not be nullptr.");
        CHECK_COND(gmmParams_.groupList != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "In MX quant, groupListOptional should not be nullptr.");
        CHECK_COND(gmmParams_.logit != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In MX quant, logit should not be nullptr.");
        CHECK_COND(gmmParams_.rowIndex != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "In MX quant, rowIndex should not be nullptr.");
        CHECK_COND(gmmParams_.out != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "In MX quant, out should not be nullptr.");
        CHECK_COND(gmmParams_.offset == nullptr, ACLNN_ERR_PARAM_INVALID, "MX quant case does not support offset.");
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckInputOutDims()
    {
        auto xDimNumber = gmmParams_.x1->GetViewShape().GetDimNum();
        auto wDimNumber = gmmParams_.x2->GetViewShape().GetDimNum();
        auto xScaleDimNumber = gmmParams_.pertokenScaleOptional->GetViewShape().GetDimNum();
        auto wScaleDimNumber = gmmParams_.scale->GetViewShape().GetDimNum();
        auto grouplistDimNumber = gmmParams_.groupList->GetViewShape().GetDimNum();
        auto logitDimNumber = gmmParams_.logit->GetViewShape().GetDimNum();
        auto rowindexDimNumber = gmmParams_.rowIndex->GetViewShape().GetDimNum();
        auto outDimNumber = gmmParams_.out->GetViewShape().GetDimNum();
        CHECK_COND(xDimNumber == TWO_DIM, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of x should be equal 2, current dim is %lu", xDimNumber);
        CHECK_COND(wDimNumber == THERE_DIM, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of w should be equal 3, current dim is %lu", wDimNumber);
        CHECK_COND(xScaleDimNumber == THERE_DIM, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of pertokenscale should be equal 3, current dim is %lu", xScaleDimNumber);
        CHECK_COND(wScaleDimNumber == FOUR_DIM, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of scale should be equal 4, current dim is %lu", wScaleDimNumber);
        CHECK_COND(grouplistDimNumber == ONE_DIM, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of grouplist should be equal 1, current dim is %lu", grouplistDimNumber);
        CHECK_COND(logitDimNumber == ONE_DIM, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of logit should be equal 1, current dim is %lu", logitDimNumber);
        CHECK_COND(rowindexDimNumber == ONE_DIM, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of rowindex should be equal 1, current dim is %lu", rowindexDimNumber);
        CHECK_COND(outDimNumber == TWO_DIM, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of out should be equal 1, current dim is %lu", outDimNumber);
        if (gmmParams_.bias != nullptr) {
            auto baisDimNumber = gmmParams_.bias->GetViewShape().GetDimNum();
            CHECK_COND(baisDimNumber == TWO_DIM, ACLNN_ERR_PARAM_INVALID,
                       "The dim num of bais should be equal 2, current dim is %lu", baisDimNumber);
        }
        if (gmmParams_.shareInput != nullptr) {
            auto shareInputDimNumber = gmmParams_.shareInput->GetViewShape().GetDimNum();
            CHECK_COND(shareInputDimNumber == TWO_DIM, ACLNN_ERR_PARAM_INVALID,
                       "The dim num of shareinput should be equal 2, current dim is %lu", shareInputDimNumber);
        }
        return ACLNN_SUCCESS;
    }

    bool CheckInputOutShape()
    {
        if (CheckInputOutShapeConsistency() == false) {
            return false;
        }
        int64_t m = gmmParams_.x1->GetViewShape().GetDim(ZERO_DIM); // 从x的第0维获取m
        int64_t k = gmmParams_.x1->GetViewShape().GetDim(ONE_DIM);  // 从x的第1维获取k
        // 转置情况下从weight的第1维获取n，非转置情况下从weight的第2维获取n
        int64_t n = gmmParams_.transposeX2 ? (gmmParams_.x2)->GetViewShape().GetDim(ONE_DIM) :
                                             (gmmParams_.x2)->GetViewShape().GetDim(TWO_DIM);
        int64_t e = (gmmParams_.x2)->GetViewShape().GetDim(0);          // 从weight的第0维获取e
        int64_t outputBS = gmmParams_.out->GetViewShape().GetDim(0);
        op::Shape xExpectShape = {m, k};
        op::Shape weightExpectShape = {e, k, n};
        op::Shape xScaleExpectShape = {m, Ops::Base::CeilDiv(k, GMMFR_SPLIT_SIZE), GMMFR_SPLIT_FACTOR};
        op::Shape weightScaleExpectShape = {e, Ops::Base::CeilDiv(k, GMMFR_SPLIT_SIZE), n, GMMFR_SPLIT_FACTOR};
        op::Shape weightTransExpectShape = {e, n, k};
        op::Shape weightScaleTransExpectShape = {e, n, Ops::Base::CeilDiv(k, GMMFR_SPLIT_SIZE), GMMFR_SPLIT_FACTOR};
        op::Shape grouplistExpectShape = {e};
        op::Shape logitExpectShape = {m};
        op::Shape rowindexExpectShape = {m};
        op::Shape outputExpectShape = {outputBS, n};

        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.x1, xExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.pertokenScaleOptional, xScaleExpectShape, return false);
        if (gmmParams_.transposeX2) {
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.scale, weightScaleTransExpectShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.x2, weightTransExpectShape, return false);
        } else {
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.scale, weightScaleExpectShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.x2, weightExpectShape, return false);
        }
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.groupList, grouplistExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.logit, logitExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.rowIndex, rowindexExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.out, outputExpectShape, return false);

        if (gmmParams_.bias != nullptr) {
            // bias的shape期望为[E, N]
            op::Shape biasExpectShape = {e, n};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.bias, biasExpectShape, return false);
        }
        if (gmmParams_.shareInput != nullptr) {
            // shareInput的shape期望为[bsdp, N]
            int64_t bsdp = gmmParams_.shareInput->GetViewShape().GetDim(0); // 从share_input 第一维获取bsdp
            op::Shape shareInputExpectShape = {bsdp, n};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.shareInput, shareInputExpectShape, return false);
        }
        return true;
    }

    bool CheckInputOutShapeConsistency()
    {
        int64_t k = gmmParams_.x1->GetViewShape().GetDim(ONE_DIM); // 从x的第1维获取k
        int64_t kInWeight = gmmParams_.transposeX2 ? (gmmParams_.x2)->GetViewShape().GetDim(TWO_DIM) :
                                                     (gmmParams_.x2)->GetViewShape().GetDim(ONE_DIM);
        int64_t e = (gmmParams_.x2)->GetViewShape().GetDim(0); // 从weight的第0维获取e
        if (kInWeight != k) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The dimension (k) of 'x' (%ld) must be equal to the dimension (k) of 'weight' (%ld)", k,
                    kInWeight);
            return false;
        }
        // groupList的长度应等于weight的专家数
        int64_t groupListLen = gmmParams_.groupList->GetViewShape().GetDim(ZERO_DIM);
        if (groupListLen != e) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Length of 'groupList' should be equal to the number of experts in weight. But got %ld.", e);
            return false;
        }
        if (e > MAX_NUM_EXPERTS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In MXFP4/MXFP8, e must be less than 1024. But got %ld.", e);
            return false;
        }
        return true;
    }

    bool CheckInputOutShapeForMXFP4()
    {   
        int64_t m = gmmParams_.x1->GetViewShape().GetDim(ZERO_DIM); // 从x的第0维获取m
        int64_t k = gmmParams_.x1->GetViewShape().GetDim(ONE_DIM);  // 从x的第1维获取k
        // 转置情况下从weight的第1维获取n，非转置情况下从weight的第2维获取n
        int64_t n = gmmParams_.transposeX2 ? (gmmParams_.x2)->GetViewShape().GetDim(ONE_DIM) :
                                             (gmmParams_.x2)->GetViewShape().GetDim(TWO_DIM);
        DataType xDtype = gmmParams_.x1->GetDataType();
        DataType weightDtype = gmmParams_.x2->GetDataType();
        if ((xDtype == DataType::DT_FLOAT4_E2M1 || xDtype == DataType::DT_FLOAT4_E1M2) &&
            (weightDtype == DataType::DT_FLOAT4_E2M1 || weightDtype == DataType::DT_FLOAT4_E1M2)) {
            if (!(k % MOD2 == 0)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In MXFP4 , k must be divisible by 2. But got %ld.", k);
                return false;
            }
            if (gmmParams_.transposeX2 == false && (n % MOD2 != 0)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In MXFP4 , n must be even when x2 is not transposed. But got %ld.", n);
                return false;
            }
            if (k == MOD2) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In MXFP4 , k cannot be 2");
                return false;
            }
        }
        return true;
    }

    bool CheckDtypeValid()
    {
        DataType xDtype = gmmParams_.x1->GetDataType();
        DataType weightDtype = gmmParams_.x2->GetDataType();
        if (CheckType(xDtype, MX_IN_TYPE_SUPPORT_LIST) && CheckType(weightDtype, MX_IN_TYPE_SUPPORT_LIST)) {
            OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.x1, MX_IN_TYPE_SUPPORT_LIST, return false);
            OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.x2, MX_IN_TYPE_SUPPORT_LIST, return false);
            OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.scale, MX_SCALE_TYPE_SUPPORT_LIST, return false);
            OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.rowIndex, MX_ROW_INDEX_TYPE_SUPPORT_LIST, return false);
            OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.pertokenScaleOptional, MX_PERTOKEN_SCALE_TYPE_SUPPORT_LIST,
                                       return false);
            if (gmmParams_.bias != nullptr) {
                OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.bias, MX_BIAS_TYPE_SUPPORT_LIST, return false);
            }
            OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.groupList, GROUP_LIST_TYPE_SUPPORT_LIST, return false);
            if (gmmParams_.shareInput != nullptr) {
                OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.shareInput, SHARED_INPUT_TYPE_SUPPORT_LIST, return false);
            }
            OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.logit, LOGIT_TYPE_SUPPORT_LIST, return false);
            OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.out, OUT_TYPE_SUPPORT_LIST, return false);

            if ((CheckType(gmmParams_.x1->GetDataType(), MXFP4_IN_TYPE_SUPPORT_LIST) !=
                 CheckType(gmmParams_.x2->GetDataType(), MXFP4_IN_TYPE_SUPPORT_LIST)) ||
                (CheckType(gmmParams_.x1->GetDataType(), MXFP8_IN_TYPE_SUPPORT_LIST) !=
                 CheckType(gmmParams_.x2->GetDataType(), MXFP8_IN_TYPE_SUPPORT_LIST))) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "X1 and x2 dtype should be same, actual x1 dtype is %s and x2 dtype is %s.",
                        op::ToString(gmmParams_.x1->GetDataType()).GetString(),
                        op::ToString(gmmParams_.x2->GetDataType()).GetString());
                return false;
            }
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Quant case with x dtype %s and weight dtype %s is not supported.",
                    op::ToString(xDtype).GetString(), op::ToString(weightDtype).GetString());
            return false;
        }
        return true;
    }

    bool CheckFormat()
    {
        if (op::IsPrivateFormat(gmmParams_.x1->GetStorageFormat()) ||
            op::IsPrivateFormat(gmmParams_.pertokenScaleOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x and pertokenScaleOptional must be ND format, but got: %s, %s.",
                    op::ToString(gmmParams_.x1->GetStorageFormat()).GetString(),
                    op::ToString(gmmParams_.pertokenScaleOptional->GetStorageFormat()).GetString());
            return false;
        }
        if (op::IsPrivateFormat(gmmParams_.x2->GetStorageFormat()) ||
            op::IsPrivateFormat(gmmParams_.scale->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight and scale must be ND format, but got: %s, %s.",
                    op::ToString(gmmParams_.x2->GetStorageFormat()).GetString(),
                    op::ToString(gmmParams_.scale->GetStorageFormat()).GetString());
            return false;
        }
        if (op::IsPrivateFormat(gmmParams_.groupList->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of groupList should be ND, current format is %s.",
                    op::ToString(gmmParams_.groupList->GetStorageFormat()).GetString());
            return false;
        }
        if (op::IsPrivateFormat(gmmParams_.logit->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of logit should be ND, current format is %s.",
                    op::ToString(gmmParams_.logit->GetStorageFormat()).GetString());
            return false;
        }
        if (op::IsPrivateFormat(gmmParams_.rowIndex->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of rowIndex should be ND, current format is %s.",
                    op::ToString(gmmParams_.rowIndex->GetStorageFormat()).GetString());
            return false;
        }
        if (op::IsPrivateFormat(gmmParams_.out->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of out should be ND, current format is %s.",
                    op::ToString(gmmParams_.out->GetStorageFormat()).GetString());
            return false;
        }
        if (gmmParams_.bias != nullptr && op::IsPrivateFormat(gmmParams_.bias->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of bias should be ND, current format is %s.",
                    op::ToString(gmmParams_.bias->GetStorageFormat()).GetString());
            return false;
        }
        if (gmmParams_.shareInput != nullptr && op::IsPrivateFormat(gmmParams_.shareInput->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of shareInput should be ND, current format is %s.",
                    op::ToString(gmmParams_.shareInput->GetStorageFormat()).GetString());
            return false;
        }
        return true;
    }

private:
    GroupedMatmulParams gmmParams_;
};
} // namespace GmmFinalizeRouting
#endif