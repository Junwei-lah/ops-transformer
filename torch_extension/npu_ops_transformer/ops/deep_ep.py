# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY
from .moe_distribute_combine_v3 import npu_moe_distribute_combine_v3
from .moe_distribute_dispatch_v3 import npu_moe_distribute_dispatch_v3
from .update_context import update_context


class MoeDistributeBuffer:
    def __init__(self, group, ccl_buffer_size: int = 0, comm_alg: int = 0):
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.world_size = torch.distributed.get_world_size(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank_id, init_comm=False)
        self.ccl_buffer_size = ccl_buffer_size
        context_struct_size = (2 + 1024)
        context_tensor_size = ((context_struct_size * 8) + 3) // 4
        self.context = torch.zeros(context_tensor_size, dtype=torch.int32).npu()
        update_context(self.group_name, self.world_size, self.context)

    def get_low_latency_ccl_buffer_size(self, num_max_dispatch_tokens_per_rank: int, hidden: int,
                                             num_moe_expert: int, num_shared_expert: int = 0,
                                             num_shared_expert_ranks: int = 0) -> int:
        total_buffsize = self.world_size
        return total_buffsize

    def update_ctx(self, new_group) -> bool:
        self.group = new_group
        self.rank_id = torch.distributed.get_rank(new_group)
        self.group_name = new_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank_id, init_comm=False)
        new_world_size = torch.distributed.get_world_size(new_group)
        torch._check(new_world_size == self.world_size,
                    lambda: (f"New world size should be the same as orginal world size, "
                            f"but got {new_world_size=}, orginial={self.world_size}"
                            f"{ops_error(ErrCode.VALUE)}."),)
        return update_context(self.group_name, self.world_size, self.context)
        
    def npu_low_latency_dispatch(self, x, topk_idx, num_experts: int, *,
                             quant_mode=0, comm_alg="", x_smooth_scale=None,
                             x_active_mask=None, topk_weights=None, zero_expert_num=0, copy_expert_num=0,
                             const_expert_num=0, elastic_info=None, expert_shard_type=0, shared_expert_num=1,
                             shared_expert_rank_num=0, expert_token_nums_type=1, num_max_dispatch_tokens_per_rank=0):
        (expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales) \
            = torch.ops.npu_ops_transformer.npu_moe_distribute_dispatch_v3(
                                             context=self.context,
                                             x=x,
                                             expert_ids=topk_idx,
                                             ep_world_size=self.world_size,
                                             ep_rank_id=self.rank_id,
                                             moe_expert_num=num_experts,
                                             ccl_buffer_size=self.ccl_buffer_size,
                                             scales=x_smooth_scale,
                                             x_active_mask=x_active_mask,
                                             expert_scales=topk_weights,
                                             elastic_info=elastic_info,
                                             performance_info=None,
                                             expert_shard_type=expert_shard_type,
                                             shared_expert_num=shared_expert_num,
                                             shared_expert_rank_num=shared_expert_rank_num,
                                             quant_mode=quant_mode,
                                             expert_token_nums_type=expert_token_nums_type,
                                             global_bs=num_max_dispatch_tokens_per_rank * self.world_size,
                                             comm_alg=comm_alg,
                                             zero_expert_num=zero_expert_num,
                                             copy_expert_num=copy_expert_num,
                                             const_expert_num=const_expert_num)
        return expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, expand_scales

    def npu_low_latency_combine(self, x, topk_idx, topk_weights, assist_info_for_combine, ep_send_counts, *,
                            num_experts=0, comm_alg="", comm_quant_mode=0, x_active_mask=None, expand_scales=None,
                            shared_expert_x=None, elastic_info=None, ori_x=None, const_expert_alpha_1=None,
                            const_expert_alpha_2=None, const_expert_v=None, zero_expert_num=0, copy_expert_num=0,
                            const_expert_num=0, expert_shared_type=0, shared_expert_num=1, shared_expert_rank_num=0,
                            num_max_dispatch_tokens_per_rank=0):
        return torch.ops.npu_ops_transformer.npu_moe_distribute_combine_v3(
                                             context=self.context,
                                             expand_x=x,
                                             expert_ids=topk_idx,
                                             assist_info_for_combine=assist_info_for_combine,
                                             ep_send_counts=ep_send_counts,
                                             expert_scales=topk_weights,
                                             ep_world_size=self.world_size,
                                             ep_rank_id=self.rank_id,
                                             moe_expert_num=num_experts,
                                             ccl_buffer_size=self.ccl_buffer_size,
                                             tp_send_counts=None,
                                             x_active_mask=x_active_mask,
                                             expand_scales=expand_scales,
                                             shared_expert_x=shared_expert_x,
                                             elastic_info=elastic_info,
                                             ori_x=ori_x,
                                             const_expert_alpha_1=const_expert_alpha_1,
                                             const_expert_alpha_2=const_expert_alpha_2,  
                                             const_expert_v=const_expert_v,
                                             performance_info=None,
                                             expert_shard_type=expert_shared_type,
                                             shared_expert_num=shared_expert_num,
                                             shared_expert_rank_num=shared_expert_rank_num,
                                             copy_expert_num=copy_expert_num,
                                             zero_expert_num=zero_expert_num,
                                             const_expert_num=const_expert_num,
                                             comm_alg=comm_alg,
                                             comm_quant_mode=comm_quant_mode,
                                             global_bs=num_max_dispatch_tokens_per_rank * self.world_size)
  