/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_ALL_REDUCE_API_UT_PARAM_H
#define MATMUL_ALL_REDUCE_API_UT_PARAM_H

#include <cstdint>
#include <string>
#include <sstream>
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

namespace MatmulAllReduceUT {

struct MatmulAllReduceApiUtParam {
    std::string case_name;
    TensorDesc x1;
    TensorDesc x2;
    TensorDesc bias;
    TensorDesc x3;
    TensorDesc dequantScale;
    TensorDesc pertokenScale;
    TensorDesc x1Scale;
    TensorDesc x2Scale;
    TensorDesc commQuantScale1;
    TensorDesc commQuantScale2;
    TensorDesc antiquantScale;
    TensorDesc antiquantOffset;
    const char* group;
    const char* reduceOp;
    int64_t commTurn;
    int64_t streamMode;
    int64_t groupSize;
    int64_t commQuantMode;
    TensorDesc output;
    aclnnStatus expectAclnnStatus;

    // aclnn_matmul_all_reduce
    MatmulAllReduceApiUtParam(std::string case_name, TensorDesc x1, TensorDesc x2, TensorDesc bias, const char* group,
        const char* reduceOp, int64_t commTurn, int64_t streamMode, TensorDesc output, aclnnStatus expectAclnnStatus): 
        case_name(case_name), x1(x1), x2(x2), bias(bias), group(group), reduceOp(reduceOp), commTurn(commTurn),
        streamMode(streamMode), output(output), expectAclnnStatus(expectAclnnStatus) {}

    // aclnn_matmul_all_reduce_v2
    MatmulAllReduceApiUtParam(std::string case_name, TensorDesc x1, TensorDesc x2, TensorDesc bias, TensorDesc x3,
        const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode,
        TensorDesc output, aclnnStatus expectAclnnStatus): 
        case_name(case_name), x1(x1), x2(x2), bias(bias), x3(x3), group(group), reduceOp(reduceOp), commTurn(commTurn),
        streamMode(streamMode), output(output), expectAclnnStatus(expectAclnnStatus) {}

    // aclnn_quant_matmul_all_reduce
    MatmulAllReduceApiUtParam(std::string case_name, TensorDesc x1, TensorDesc x2, TensorDesc bias, TensorDesc x3,
        TensorDesc dequantScale, const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode,
        TensorDesc output, aclnnStatus expectAclnnStatus): 
        case_name(case_name), x1(x1), x2(x2), bias(bias), x3(x3), dequantScale(dequantScale), group(group),
        reduceOp(reduceOp), commTurn(commTurn), streamMode(streamMode), output(output),
        expectAclnnStatus(expectAclnnStatus) {}

    // aclnn_quant_matmul_all_reduce_v2
    MatmulAllReduceApiUtParam(std::string case_name, TensorDesc x1, TensorDesc x2, TensorDesc bias, TensorDesc x3,
        TensorDesc dequantScale, TensorDesc pertokenScale, const char* group, const char* reduceOp, int64_t commTurn, 
        int64_t streamMode, TensorDesc output, aclnnStatus expectAclnnStatus): 
        case_name(case_name), x1(x1), x2(x2), bias(bias), x3(x3), dequantScale(dequantScale), 
        pertokenScale(pertokenScale), group(group), reduceOp(reduceOp), commTurn(commTurn), streamMode(streamMode), 
        output(output), expectAclnnStatus(expectAclnnStatus) {}

    // aclnn_quant_matmul_all_reduce_v3
    MatmulAllReduceApiUtParam(std::string case_name, TensorDesc x1, TensorDesc x2, TensorDesc bias, TensorDesc x3,
        TensorDesc dequantScale, TensorDesc pertokenScale, TensorDesc commQuantScale1, TensorDesc commQuantScale2,
        const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode, TensorDesc output,
        aclnnStatus expectAclnnStatus): 
        case_name(case_name), x1(x1), x2(x2), bias(bias), x3(x3), dequantScale(dequantScale), 
        pertokenScale(pertokenScale), commQuantScale1(commQuantScale1), commQuantScale2(commQuantScale2), 
        group(group), reduceOp(reduceOp), commTurn(commTurn), streamMode(streamMode), output(output),
        expectAclnnStatus(expectAclnnStatus) {}

    // aclnn_quant_matmul_all_reduce_v4
    MatmulAllReduceApiUtParam(std::string case_name, TensorDesc x1, TensorDesc x2, TensorDesc bias, TensorDesc x3, 
        TensorDesc x1Scale, TensorDesc x2Scale, TensorDesc commQuantScale1, TensorDesc commQuantScale2,
        const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode, int64_t groupSize, 
        int64_t commQuantMode, TensorDesc output, aclnnStatus expectAclnnStatus): 
        case_name(case_name), x1(x1), x2(x2), bias(bias), x3(x3), x1Scale(x1Scale), x2Scale(x2Scale),
        commQuantScale1(commQuantScale1), commQuantScale2(commQuantScale2), group(group), reduceOp(reduceOp),
        commTurn(commTurn), streamMode(streamMode), groupSize(groupSize), commQuantMode(commQuantMode), output(output),
        expectAclnnStatus(expectAclnnStatus) {}

    // aclnnWeightQuantMatmulAllReduce
    MatmulAllReduceApiUtParam(std::string case_name, TensorDesc x1, TensorDesc x2, TensorDesc bias,
        TensorDesc antiquantScale, TensorDesc antiquantOffset, TensorDesc x3, const char* group, const char* reduceOp,
        int64_t commTurn, int64_t streamMode, int64_t groupSize, TensorDesc output, aclnnStatus expectAclnnStatus):
        case_name(case_name), x1(x1), x2(x2), bias(bias), antiquantScale(antiquantScale),
        antiquantOffset(antiquantOffset), x3(x3), group(group), reduceOp(reduceOp), commTurn(commTurn), 
        streamMode(streamMode), groupSize(groupSize), output(output),
        expectAclnnStatus(expectAclnnStatus) {}
};

inline std::ostream& operator<<(std::ostream& os, const MatmulAllReduceApiUtParam& param)
{
    return os << param.case_name;
}

inline std::string PrintMatmulAllReduceApiUtParam(const testing::TestParamInfo<MatmulAllReduceApiUtParam>& info)
{
    return info.param.case_name;
}

} // namespace matmul_all_reduce_ut

#endif // MATMUL_ALL_REDUCE_API_UT_PARAM_H
