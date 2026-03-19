/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_all_gather_matmul.h"
#include "securec.h"
#include "acl/acl.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "common/op_host/op_api/matmul_util.h"
#include "hccl_util.h"
#include "mc2_aclnn_util.h"

using namespace op;
using namespace Ops::Transformer;

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  uint32_t id;
  const char *funcName;
  bool hasReg;
} NnopbaseDfxId;

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerAllGatherMatmulGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, 
                                                             const char *group, bool transposeX1, bool transposeX2,
                                                             int64_t gatherIndex, int64_t commTurn, int64_t rankSize,
                                                             bool isGatherOut, bool isAMaxOut, int64_t yDtype, 
                                                             const char* commMode, aclTensor *output, aclTensor *gatherOut, 
                                                             uint64_t *workspaceSize, aclOpExecutor **executor);
extern aclnnStatus aclnnInnerAllGatherMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream);
extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

static const aclTensor *TransX2Tensor(const aclTensor *x2)
{
  uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
  std::vector<int64_t> storageDims(storageDimsNum);
  for (uint64_t i = 0; i < storageDimsNum; i++) {
    storageDims[i] = x2->GetStorageShape().GetDim(i);
  }

  uint64_t viewDimsNum = x2->GetViewShape().GetDimNum();
  std::vector<int64_t> viewDims;
  viewDims.resize(viewDimsNum);
  for (uint64_t i = 0; i < viewDimsNum; i++) {
    viewDims[i] = x2->GetViewShape().GetDim(i);
  }
  // transpose the viewshape last two dimensions
  viewDims[0] = x2->GetViewShape().GetDim(1);
  viewDims[1] = x2->GetViewShape().GetDim(0);

  aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
  aclGetDataType(x2, &dataType);
  std::vector<int64_t> stride(viewDimsNum);
  auto transStride = x2->GetViewStrides();
  stride = std::vector<int64_t>(transStride.begin(), transStride.end());
  // transpose the two dimensions
  stride[0] = transStride[1];
  stride[1] = transStride[0];
  auto offset = x2->GetViewOffset();
  aclFormat format = aclFormat::ACL_FORMAT_ND;

  return aclCreateTensor(viewDims.data(), viewDimsNum, dataType, stride.data(), offset, format, storageDims.data(),
                          storageDimsNum, x2->GetTensor()->GetAddr());
}

aclnnStatus allGatherMatmulGetWorkspaceSizeCCUMode(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, 
                                                   const char* group, int64_t gatherIndex, int64_t commTurn, 
                                                   int64_t streamMode, const char* commMode, aclTensor* output, aclTensor* gatherOut,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor)
{
  uint64_t timeStamp = NnopbaseMsprofSysTime();
  uint32_t rankSize = 0;
  bool transposeX1 = IsTransposeLastTwoDims(x1);
  bool transposeX2 = IsTransposeLastTwoDims(x2);

  bool isGatherOut = true;
  bool isAMaxOut = true;

  uint64_t outDtype = static_cast<uint64_t>(output->GetDataType());
  auto transX2 = x2;
  if (transposeX2) {
    // x2转置时将两轴shape调换
    transX2 = TransX2Tensor(x2);
  }
  aclnnStatus ret = aclnnInnerAllGatherMatmulGetWorkspaceSize(x1, transX2, bias, group,
                                                                transposeX1, transposeX2, gatherIndex, commTurn,
                                                                rankSize, isGatherOut, isAMaxOut,
                                                                outDtype, commMode, output, gatherOut,  
                                                                workspaceSize, executor);
  static NnopbaseDfxId dfxId = {0x60000, __func__, false};
  NnopbaseReportApiInfo(timeStamp, dfxId);
  return ret;
}

aclnnStatus aclnnAllGatherMatmulGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                                   const char* group,int64_t gatherIndex, int64_t commTurn, 
                                                   int64_t streamMode, const char* commMode, aclTensor* output, 
                                                   aclTensor* gatherOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    aclnnStatus ret = ACLNN_SUCCESS;
    ret = allGatherMatmulGetWorkspaceSizeCCUMode(x1, x2, bias, group, gatherIndex, commTurn,
                                                       streamMode, commMode, output, gatherOut,  
                                                       workspaceSize, executor);
                                                      
    return ret;
}

aclnnStatus aclnnAllGatherMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream)
{
  if (NnopbaseSetHcclServerType) {
      NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
  }

  return aclnnInnerAllGatherMatmul(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif