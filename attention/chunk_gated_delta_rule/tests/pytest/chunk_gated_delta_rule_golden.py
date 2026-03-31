# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import time
import torch
import torch_npu
import torch.nn.functional as F
import numpy as np
import logging
import datetime
import os
import sys
import argparse
from typing import Optional

np.random.seed(21)
np.set_printoptions(suppress=True)
DEVICE_ID = 0
torch.npu.config.allow_internal_format = True
eb_threshold = 2**(-8)
err_threshold = 2**(-8)
CV_MAX_RE = 5               # 最大相对误差
CV_AVER_RE = 1.5            # 平均相对误差
CV_RMSE = 1.5               # 均方根误差
CV_SMALL_VAL = 2            # 小值域错误占比
CV_ERR_BALANCE = 2          # 误差均衡性
MIN_ERR = 1e-3
logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)


# 定义ANSI颜色常量（新增）
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'    # 绿色
    WARNING = '\033[93m'
    FAIL = '\033[91m'       # 红色
    ENDC = '\033[0m'        # 重置颜色
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def chunk_gated_delta_rule_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    num_heads = q.shape[-2]
    num_value_heads = v.shape[-2]

    if num_value_heads // num_heads > 1:
        q = q.repeat_interleave(num_value_heads // num_heads, dim=2)
        k = k.repeat_interleave(num_value_heads // num_heads, dim=2)

    batch_size = initial_state.shape[0]
    core_attn_out = []
    last_recurrent_state = torch.empty_like(initial_state)

    for b_idx in range(batch_size):
        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        cur_q = q[:, start:end, ...]
        cur_k = k[:, start:end, ...]
        cur_v = v[:, start:end, ...]
        cur_g = g[:, start:end, ...]
        cur_beta = beta[:, start:end, ...]
        cur_state = initial_state[b_idx].unsqueeze(0)

        cur_core_attn_out, cur_last_recurrent_state = chunk_gated_delta_rule_native(
            query=cur_q,
            key=cur_k,
            value=cur_v,
            g=cur_g,
            beta=cur_beta,
            initial_state=cur_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out.append(cur_core_attn_out)
        last_recurrent_state[b_idx] = cur_last_recurrent_state

    tar_dtype = core_attn_out[0].dtype
    tar_device = core_attn_out[0].device
    tar_shape = list(core_attn_out[0].shape)
    tar_shape[1] = cu_seqlens[-1]
    final_cor_attn_out = torch.empty(tar_shape, dtype=tar_dtype, device=tar_device)

    for b_idx in range(batch_size):
        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        final_cor_attn_out[:, start:end, ...] = core_attn_out[b_idx]

    return final_cor_attn_out, last_recurrent_state


def chunk_gated_delta_rule_native(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    tot_heads = num_heads + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )
    # for each chunk
    for i in range(0, tot_heads // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :num_heads]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def get_max_re(golden: torch.Tensor, actual: torch.Tensor):
    # 最大相对误差
    abs_error = torch.abs(actual - golden) / (torch.abs(golden) + MIN_ERR)
    max_re = torch.max(abs_error.flatten())
    return max_re


def get_avg_re(golden: torch.Tensor, actual: torch.Tensor):
    # 平均相对误差
    abs_error = torch.abs(actual - golden) / (torch.abs(golden) + MIN_ERR)
    avg_re = torch.mean(abs_error)
    return avg_re


def get_rmse(golden: torch.Tensor, actual: torch.Tensor):
    # 均方根误差
    sqr_err = torch.pow((actual - golden), 2)
    rmse = torch.sqrt(torch.mean(sqr_err))
    return rmse


def get_smra(golden: torch.Tensor, actual: torch.Tensor):
    # 小值域错误占比
    abs_A = torch.abs(golden)
    mask_A = abs_A < 2**(-10)
    num_a = torch.sum(mask_A).item()

    # 统计对应位置 B 中元素绝对值大于 1e-16 的个数
    abs_B = torch.abs(golden - actual)
    mask_B = abs_B > 1e-16
    num_b = torch.sum(mask_A & mask_B).item()
    
    smra = num_b / num_a if num_a > 0 else 0
    return smra


def get_eb(golden: torch.Tensor, actual: torch.Tensor):
    # 误差均衡性
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    actual_error = actual - golden
    error_balance = torch.mean(actual_error / golden_nmax)
    return error_balance


def compare_cv(golden: torch.Tensor, golden_high_type: torch.Tensor, actual: torch.Tensor, name=None):
    golden = golden.to(torch.float32)
    golden_high_type = golden_high_type.to(torch.float32)
    actual = actual.to(torch.float32)
    # 最大相对误差
    max_re_npu = get_max_re(golden, actual)
    max_re_high_type = get_max_re(golden, golden_high_type)
    print(f"{max_re_npu=}, {max_re_high_type=}")
    # 平均相对误差
    avg_re_npu = get_avg_re(golden, actual)
    avg_re_high_type = get_avg_re(golden, golden_high_type)
    # 均方根误差
    rmse_npu = get_rmse(golden, actual)
    rmse_high_type = get_rmse(golden, golden_high_type)
    # 小值域错误占比
    smra_npu = get_smra(golden, actual)
    smra_high_type = get_smra(golden, golden_high_type)
    max_re_rate = max_re_npu / max(max_re_high_type, err_threshold)
    avg_re_rate = avg_re_npu / max(avg_re_high_type, err_threshold)
    rmse_rate = rmse_npu / max(rmse_high_type, err_threshold)
    smra_rate = smra_npu / max(smra_high_type, err_threshold)
    # 误差均衡性
    EB = get_eb(golden_high_type, actual)

    if name is not None:
        print(f"compare_cv for {name}:")
    print(f"\tmax_re_rate={max_re_rate:.3f} ({CV_MAX_RE}), max_re_high_type={max_re_high_type:.3e}")
    print(f"\tavg_re_rate={avg_re_rate:.3f} ({CV_AVER_RE}), avg_re_high_type={avg_re_high_type:.3e}")
    print(f"\trmse_rate={rmse_rate:.3f} ({CV_RMSE}), rmse_high_type={rmse_high_type:.3e}")
    print(f"\tsmra_rate={smra_rate:.3f} ({CV_SMALL_VAL}), smra_high_type={smra_high_type:.3e}")
    result = (max_re_rate < CV_MAX_RE) and (avg_re_rate < CV_AVER_RE) and (rmse_rate < CV_RMSE)
    result = result and smra_rate < CV_SMALL_VAL
    if not result:
        epsilon = 2.0**-7
        if max_re_npu < epsilon:
            print(f"\t max_re_npu={max_re_npu} less than {epsilon}.")
            result = True
    return result


def cgdr_golden(q, k, v, g, beta, scale, initial_state, actual_seq_lengths, use_float64=False):
    t0 = time.time()
    cu_seqlens = F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0)
    if use_float64:
        v = v.cpu().to(torch.float64)
    else:
        v = v.to(torch.float32)
    if g is None:
        g = torch.zeros((v.shape[0], v.shape[1])).to(v.device).to(v.dtype)
    o_golden, state_golden = chunk_gated_delta_rule_npu(
        q.unsqueeze(0).to(v.device).to(v.dtype),
        k.unsqueeze(0).to(v.device).to(v.dtype),
        v.unsqueeze(0).to(v.device).to(v.dtype),
        g.unsqueeze(0).to(v.device).to(v.dtype),
        beta.unsqueeze(0).to(v.device).to(v.dtype),
        scale=scale,
        initial_state=initial_state.transpose(-1, -2).clone().to(v.device).to(v.dtype),
        cu_seqlens=cu_seqlens.to(v.device)
    )
    o_golden = o_golden[0]
    state_golden = state_golden.transpose(-1, -2)
    print(f"cgdr_golden {use_float64=} time cost: {time.time() - t0} s")
    return o_golden.to(torch.float32).npu(), state_golden.to(torch.float32).npu()


def cgdr_benchmark(q, k, v, g, beta, scale, initial_state, actual_seq_lengths):
    cu_seqlens = F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0)
    if g is None:
        g = torch.zeros((v.shape[0], v.shape[1])).to(v.device).to(torch.float32)
    o_bench, state_bench = chunk_gated_delta_rule_npu(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g.unsqueeze(0) if g is not None else None,
        beta.unsqueeze(0),
        scale=scale,
        initial_state=initial_state.transpose(-1, -2).clone(),
        cu_seqlens=cu_seqlens
    )
    o_bench = o_bench[0].to(torch.float32)
    state_bench = state_bench.transpose(-1, -2).to(torch.float32)
    return o_bench, state_bench


def cgdr_npu(q, k, v, g, beta, scale, initial_state, actual_seq_lengths):
    o_npu, state_npu = torch_npu.npu_chunk_gated_delta_rule(
        q, k, v,
        beta=beta,
        initial_state=initial_state.clone(),
        actual_seq_lengths=actual_seq_lengths,
        scale=scale,
        g=g
    )
    o_npu = o_npu.to(torch.float32)
    state_npu = state_npu.to(torch.float32)
    return o_npu, state_npu


def run_chunk_gated_delta_rule_eager(B, seqlen, nk, nv, dk, dv, chunk_size=64,
                                     data_type=torch.bfloat16):
    torch_npu.npu.set_device(int(DEVICE_ID))
    # ======================== gen input data start =============================
    T = B * seqlen
    q = torch.rand((T, nk, dk), dtype=data_type, device="npu:%s" % DEVICE_ID)
    k = torch.rand((T, nk, dk), dtype=data_type, device="npu:%s" % DEVICE_ID)
    v = torch.rand((T, nv, dv), dtype=data_type, device="npu:%s" % DEVICE_ID)
    g = torch.rand((T, nv), dtype=torch.float32, device="npu:%s" % DEVICE_ID) * -1.0
    beta = torch.rand((T, nv), dtype=data_type, device="npu:%s" % DEVICE_ID)
    q = torch.nn.functional.normalize(q, p=2, dim=-1)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    scale = 1 / (dk ** 0.5)
    initial_state = torch.rand((B, nv, dv, dk), dtype=data_type, device="npu:%s" % DEVICE_ID)
    actual_seq_lengths = torch.tensor([seqlen] * B, dtype=torch.int32, device="npu:%s" % DEVICE_ID)
    # ======================== gen input data finish =============================

    # ======================== execute golden/benchmark/npu ================================
    o_golden, state_golden = cgdr_golden(q, k, v, g, beta, scale, initial_state, actual_seq_lengths, use_float64=False)
    o_bench, state_bench = cgdr_benchmark(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)
    o_npu, state_npu = cgdr_npu(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)
    # ======================== check result ================================
    ret = True
    if not compare_cv(o_golden, o_bench, o_npu, name="o"):
        print("compare o failed.")
        err_o = torch.abs(o_golden - o_npu).flatten()
        idx = torch.argmax(err_o)
        print(f"idx={idx}, err_o={err_o[idx]}, o_golden={o_golden.flatten()[idx]}, "
              f"o_npu={o_npu.flatten()[idx]}, o_bench={o_bench.flatten()[idx]}")
        ret = False
    if not compare_cv(state_golden, state_bench, state_npu, name="state"):
        print("compare state failed.")
        err_s = torch.abs(state_golden - state_npu).flatten()
        idx = torch.argmax(err_s)
        print(f"idx={idx}, err_s={err_s[idx]}, state_golden={state_golden.flatten()[idx]}, "
              f"state_npu={state_npu.flatten()[idx]}, state_bench={state_bench.flatten()[idx]}")
        ret = False
    print("PASSED" if ret else "FAILED")
    return ret


def run_precision_test(inputs):
    run_chunk_gated_delta_rule_eager(
        inputs['B'], inputs['seqlen'], inputs['nk'],
        inputs['nv'], inputs['dk'], inputs['dv'], inputs['chunk_size'],
        data_type=inputs['data_type']
    )
