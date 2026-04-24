#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
#
# A lightweight host-side simulator for PromptFlashAttentionTilingV2.
# It mirrors the Souter/Sinner/SoftmaxSouter selection, the BMM check
# parameter construction, and the N-B-S split-core estimation logic in
# attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp.
#
# This script does not call CANN's MatmulApiTiling, so BMM "check" output is
# the attempted SetShape/SetOrgShape/SetFixSplit plan, not a real GetTiling
# success/failure result.

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


SOUTER_FACTOR_SUB = 32
SOUTER_FACTOR_DEFAULT = 64
SINNER_FACTOR_SUB = 64
SINNER_FACTOR_DEFAULT = 128
SINNER_FACTOR_DOUBLE = 256
CV_RATIO = 2
BLOCK_SIZE_BASE = 128
SPARSE_MODE_INT_MAX = 2_147_483_647

SPARSE_MODE_NO_MASK = 0
SPARSE_MODE_ALL_MASK = 1
SPARSE_MODE_LEFT_UP = 2
SPARSE_MODE_RIGHT_DOWN = 3
SPARSE_MODE_BAND = 4

LAYOUTS_MERGED = {"BSH", "BSND", "TND"}
FP16_DTYPE_NAMES = {"fp16", "float16", "dt_float16"}
BF16_DTYPE_NAMES = {"bf16", "bfloat16", "dt_bf16"}
INT8_DTYPE_NAMES = {"int8", "dt_int8"}


def ceil_div(a: int, b: int) -> int:
    if b == 0:
        return 0
    return (a + b - 1) // b


def align_up(a: int, b: int) -> int:
    if b == 0:
        return 0
    return ceil_div(a, b) * b


def calc_tail_size(a: int, b: int) -> int:
    if b == 0:
        return 0
    mod = a % b
    return mod if mod != 0 else b


def cxx_div(a: int, b: int) -> int:
    """C++ signed integer division truncates toward zero."""
    if b == 0:
        return 0
    sign = -1 if (a < 0) ^ (b < 0) else 1
    return sign * (abs(a) // abs(b))


def norm_dtype(dtype: str) -> str:
    return str(dtype).lower()


def as_bool(data: Dict[str, Any], name: str, default: bool = False) -> bool:
    return bool(data.get(name, default))


def as_int(data: Dict[str, Any], name: str, default: int = 0) -> int:
    return int(data.get(name, default))


@dataclass
class PFAConfig:
    batch_size: int
    head_num_size: int
    seq_size: int
    seq_inner_size: int
    qk_head_size: int
    v_head_size: int
    q_head_size: int | None = None

    input_dtype: str = "fp16"
    output_dtype: str = "fp16"
    inner_precise: str = "high_performance"
    layout: str = "BSH"
    sparse_mode: int = SPARSE_MODE_NO_MASK
    pre_tokens: int = SPARSE_MODE_INT_MAX
    next_tokens: int = SPARSE_MODE_INT_MAX
    actual_seq_lengths: List[int] = field(default_factory=list)
    actual_seq_lengths_kv: List[int] = field(default_factory=list)
    actual_shared_prefix_len: int = 0

    core_num: int = 48
    aic_num: int = 24
    l1_size: int = 0
    l0c_size: int = 0

    g_size: int = 1
    head_num_ratio: int = 1
    pa_layout_type: int = 0
    block_size: int = BLOCK_SIZE_BASE
    aligned_head_size: int | None = None

    fa_run_flag: bool = True
    enable_pfa_mla: bool = False
    enable_pfa_rope: bool = False
    enable_pfa_merge: bool = False
    enable_ifa_mla: bool = False
    enable_ifa: bool = False
    enable_pa: bool = False
    enable_kv_antiquant: bool = False
    enable_mask: bool = False
    enable_pse_shift: bool = False
    enable_alibi_pse: bool = False
    enable_perblock_quant: bool = False
    enable_matmul_norm: bool = False
    enable_kv_prefix: bool = False
    is_d_no_tail: bool = True
    split_s2: int = 1

    normalize_gs1_merge: bool = True

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PFAConfig":
        shape = data.get("shape", data)
        cfg = PFAConfig(
            batch_size=as_int(shape, "batch_size", as_int(shape, "b", 1)),
            head_num_size=as_int(shape, "head_num_size", as_int(shape, "n", 1)),
            seq_size=as_int(shape, "seq_size", as_int(shape, "s1", 1)),
            seq_inner_size=as_int(shape, "seq_inner_size", as_int(shape, "s2", 1)),
            qk_head_size=as_int(shape, "qk_head_size", as_int(shape, "d", 128)),
            v_head_size=as_int(shape, "v_head_size", as_int(shape, "dv", as_int(shape, "d", 128))),
            q_head_size=shape.get("q_head_size"),
        )

        attrs = data.get("attrs", data)
        flags = data.get("flags", data)
        platform = data.get("platform", data)

        cfg.input_dtype = attrs.get("input_dtype", cfg.input_dtype)
        cfg.output_dtype = attrs.get("output_dtype", cfg.output_dtype)
        cfg.inner_precise = attrs.get("inner_precise", cfg.inner_precise)
        cfg.layout = attrs.get("layout", attrs.get("input_layout", cfg.layout))
        cfg.sparse_mode = int(attrs.get("sparse_mode", cfg.sparse_mode))
        cfg.pre_tokens = int(attrs.get("pre_tokens", attrs.get("preToken", cfg.pre_tokens)))
        cfg.next_tokens = int(attrs.get("next_tokens", attrs.get("nextToken", cfg.next_tokens)))
        cfg.actual_shared_prefix_len = int(attrs.get("actual_shared_prefix_len", 0))
        cfg.actual_seq_lengths = list(attrs.get("actual_seq_lengths", []))
        cfg.actual_seq_lengths_kv = list(attrs.get("actual_seq_lengths_kv", []))

        cfg.core_num = int(platform.get("core_num", platform.get("aiv_num", cfg.core_num)))
        cfg.aic_num = int(platform.get("aic_num", cfg.aic_num))
        cfg.l1_size = int(platform.get("l1_size", cfg.l1_size))
        cfg.l0c_size = int(platform.get("l0c_size", cfg.l0c_size))

        cfg.g_size = int(attrs.get("g_size", cfg.g_size))
        cfg.head_num_ratio = int(attrs.get("head_num_ratio", cfg.head_num_ratio))
        cfg.pa_layout_type = int(attrs.get("pa_layout_type", cfg.pa_layout_type))
        cfg.block_size = int(attrs.get("block_size", cfg.block_size))
        cfg.aligned_head_size = attrs.get("aligned_head_size", None)
        if cfg.aligned_head_size is not None:
            cfg.aligned_head_size = int(cfg.aligned_head_size)

        for name in (
            "fa_run_flag",
            "enable_pfa_mla",
            "enable_pfa_rope",
            "enable_pfa_merge",
            "enable_ifa_mla",
            "enable_ifa",
            "enable_pa",
            "enable_kv_antiquant",
            "enable_mask",
            "enable_pse_shift",
            "enable_alibi_pse",
            "enable_perblock_quant",
            "enable_matmul_norm",
            "enable_kv_prefix",
            "is_d_no_tail",
        ):
            setattr(cfg, name, as_bool(flags, name, getattr(cfg, name)))
        cfg.split_s2 = int(flags.get("split_s2", cfg.split_s2))
        cfg.normalize_gs1_merge = as_bool(flags, "normalize_gs1_merge", True)

        cfg.normalize()
        return cfg

    def normalize(self) -> None:
        self.input_dtype = norm_dtype(self.input_dtype)
        self.output_dtype = norm_dtype(self.output_dtype)
        self.layout = str(self.layout).upper()
        self.g_size = max(self.g_size, 1)
        self.head_num_ratio = max(self.head_num_ratio, 1)
        self.core_num = max(self.core_num, 1)
        self.aic_num = max(self.aic_num, 1)
        if self.q_head_size is None:
            self.q_head_size = self.qk_head_size
        if self.aligned_head_size is None:
            self.aligned_head_size = align_up(self.qk_head_size, 16)

        if self.normalize_gs1_merge and (self.enable_ifa_mla or self.enable_ifa or self.enable_pfa_merge):
            if self.head_num_size >= self.g_size:
                self.head_num_size = max(1, self.head_num_size // self.g_size)
            self.seq_size *= self.g_size

        if not self.actual_seq_lengths:
            self.actual_seq_lengths = [self.seq_size for _ in range(self.batch_size)]
        if not self.actual_seq_lengths_kv:
            self.actual_seq_lengths_kv = [self.seq_inner_size for _ in range(self.batch_size)]
        if len(self.actual_seq_lengths) < self.batch_size:
            self.actual_seq_lengths.extend([self.actual_seq_lengths[-1]] * (self.batch_size - len(self.actual_seq_lengths)))
        if len(self.actual_seq_lengths_kv) < self.batch_size:
            self.actual_seq_lengths_kv.extend([self.actual_seq_lengths_kv[-1]] * (self.batch_size - len(self.actual_seq_lengths_kv)))


def get_matmul_type(cfg: PFAConfig) -> Dict[str, str]:
    if cfg.input_dtype in FP16_DTYPE_NAMES and cfg.inner_precise.lower() == "high_precision":
        return {"input": "float16", "output": "float32"}
    if cfg.input_dtype in BF16_DTYPE_NAMES:
        return {"input": "bfloat16", "output": "float32"}
    if cfg.input_dtype in INT8_DTYPE_NAMES:
        return {"input": "int8", "output": "int32"}
    return {"input": "float16", "output": "float16"}


def bmm1_check_plan(cfg: PFAConfig, s_outer: int, s_inner: int) -> Dict[str, Any]:
    effective_s_outer = s_outer * CV_RATIO
    matmul_type = get_matmul_type(cfg)
    stride_q = cfg.qk_head_size * cfg.head_num_size
    stride_k = stride_q // cfg.head_num_ratio if cfg.head_num_ratio else 0

    if cfg.layout in LAYOUTS_MERGED:
        if cfg.enable_kv_antiquant or (cfg.layout == "TND" and cfg.enable_pa and cfg.pa_layout_type == 0):
            org_shape = [cfg.seq_size, cfg.seq_inner_size, stride_q, cfg.qk_head_size]
            org_shape_case = "merged_layout_kv_antiquant_or_tnd_pa"
        elif cfg.enable_ifa_mla or cfg.enable_ifa:
            org_shape = [cfg.seq_size, cfg.seq_inner_size, cfg.qk_head_size, stride_k]
            org_shape_case = "merged_layout_ifa"
        else:
            org_shape = [cfg.seq_size, cfg.seq_inner_size, stride_q, stride_k]
            org_shape_case = "merged_layout_default"
    elif cfg.layout == "BNSD":
        if cfg.enable_pa and cfg.pa_layout_type == 1:
            org_shape = [cfg.seq_size, cfg.seq_inner_size, cfg.qk_head_size, stride_k]
            org_shape_case = "bnsd_pa_layout1"
        else:
            org_shape = [cfg.seq_size, cfg.seq_inner_size, cfg.qk_head_size]
            org_shape_case = "bnsd_default"
    else:
        org_shape = []
        org_shape_case = "unsupported_or_unhandled_layout"

    if cfg.enable_pa and not cfg.enable_ifa_mla:
        initial_fix_split = [effective_s_outer, BLOCK_SIZE_BASE]
    else:
        initial_fix_split = [effective_s_outer, s_inner]

    fallback = None
    if cfg.enable_matmul_norm:
        fallback = {
            "always_reapply": True,
            "fix_split": [min(128, effective_s_outer), min(128, s_inner), 128],
            "reason": "enable_matmul_norm",
        }
    else:
        fallback = {
            "only_if_first_get_tiling_fails": True,
            "fix_split": [min(128, effective_s_outer), BLOCK_SIZE_BASE if cfg.enable_pa else min(128, s_inner), 64],
            "reason": "autoBaseMNK fallback",
        }

    mte2_pipe = {
        "enabled_by_default": cfg.seq_size > 16 or cfg.qk_head_size % 32 != 0,
        "small_seq_recheck": None,
    }
    if cfg.seq_size <= 16 and cfg.qk_head_size % 32 == 0:
        mte2_pipe["small_seq_recheck"] = {
            "fix_split": [
                min(128, effective_s_outer),
                BLOCK_SIZE_BASE if cfg.enable_pa else min(512, s_inner),
                32,
            ]
        }

    return {
        "op": "bmm1_q_k_transpose",
        "split_core_cube_souter": True,
        "set_shape": {"M": effective_s_outer, "N": s_inner, "K": cfg.qk_head_size},
        "types": {
            "A": {"pos": "GM", "format": "ND", "dtype": matmul_type["input"], "transpose": False},
            "B": {"pos": "GM", "format": "ND", "dtype": matmul_type["input"], "transpose": True},
            "C": {"pos": "VECCALC", "format": "ND_ALIGN", "dtype": matmul_type["output"]},
        },
        "set_org_shape": org_shape,
        "set_org_shape_case": org_shape_case,
        "set_buffer_space": {"l1": cfg.l1_size, "l0c": cfg.l0c_size},
        "initial_set_fix_split": initial_fix_split,
        "auto_base_mnk": True,
        "fallback": fallback,
        "post_process": {
            "shareMode": 0,
            "shareL1Size": cfg.l1_size,
            "shareL0CSize": cfg.l0c_size,
            "shareUbSize": 0,
            "enable_double_buffer_when_depthA1_depthB1_are_1": True,
            "mte2_pipe": mte2_pipe,
        },
    }


def bmm2_check_plan(cfg: PFAConfig, s_outer: int, s_inner: int) -> Dict[str, Any]:
    matmul_type = get_matmul_type(cfg)
    stride_v = cfg.v_head_size * cfg.head_num_size // cfg.head_num_ratio if cfg.head_num_ratio else 0

    if cfg.layout in LAYOUTS_MERGED:
        if cfg.enable_kv_antiquant or (cfg.layout == "TND" and cfg.enable_pa and cfg.pa_layout_type == 0):
            org_shape = [cfg.seq_size, cfg.v_head_size, cfg.seq_inner_size]
            org_shape_case = "merged_layout_kv_antiquant_or_tnd_pa"
        else:
            org_shape = [cfg.seq_size, stride_v, cfg.seq_inner_size]
            org_shape_case = "merged_layout_default"
    elif cfg.layout == "BNSD":
        if cfg.enable_pa and cfg.pa_layout_type == 1:
            org_shape = [cfg.seq_size, stride_v, cfg.seq_inner_size]
            org_shape_case = "bnsd_pa_layout1"
        else:
            org_shape = [cfg.seq_size, cfg.v_head_size, cfg.seq_inner_size]
            org_shape_case = "bnsd_default"
    else:
        org_shape = []
        org_shape_case = "unsupported_or_unhandled_layout"

    explicit_fix_split = None
    if cfg.enable_matmul_norm:
        explicit_fix_split = [min(128, s_outer), min(128, cfg.v_head_size), 128]

    inactive_non_auto_branch = {
        "condition": "autoBaseMNK == false, not used by ComputeCVDiffParams",
        "if_isDNoTail_or_splitS2_eq_0": [s_outer, 128],
        "else": [s_outer, cfg.aligned_head_size],
    }

    return {
        "op": "bmm2_softmax_v",
        "split_core_cube_souter": False,
        "set_shape": {"M": s_outer, "N": cfg.v_head_size, "K": s_inner},
        "types": {
            "A": {"pos": "VECCALC", "format": "ND", "dtype": matmul_type["input"], "transpose": False},
            "B": {"pos": "GM", "format": "ND", "dtype": matmul_type["input"], "transpose": False},
            "C": {"pos": "VECCALC", "format": "ND_ALIGN", "dtype": matmul_type["output"]},
        },
        "set_org_shape": org_shape,
        "set_org_shape_case": org_shape_case,
        "set_buffer_space": {"l1": cfg.l1_size, "l0c": cfg.l0c_size},
        "auto_base_mnk": True,
        "explicit_set_fix_split": explicit_fix_split,
        "inactive_non_auto_branch": inactive_non_auto_branch,
        "post_process": {
            "shareMode": 0,
            "shareL1Size": cfg.l1_size,
            "shareL0CSize": cfg.l0c_size,
            "shareUbSize": 0,
        },
    }


def adjust_cv_tiling(cfg: PFAConfig) -> Tuple[Dict[str, Any], List[str]]:
    notes: List[str] = []
    s_outer = SOUTER_FACTOR_DEFAULT
    s_inner = SINNER_FACTOR_DEFAULT
    softmax_s_outer = SOUTER_FACTOR_DEFAULT

    bmm_checks = {
        "candidate_before_heuristics": {
            "Souter": s_outer,
            "Sinner": s_inner,
            "SoftmaxSouter": softmax_s_outer,
        },
        "bmm1": bmm1_check_plan(cfg, s_outer, s_inner),
        "bmm2": bmm2_check_plan(cfg, s_outer, s_inner),
        "note": "Real MatmulApiTiling::GetTiling is not executed in this Python simulator.",
    }

    if cfg.v_head_size <= 128 and not cfg.enable_pfa_mla:
        check_dtype = cfg.input_dtype in FP16_DTYPE_NAMES or cfg.input_dtype in BF16_DTYPE_NAMES
        check_seq = cfg.seq_size <= SOUTER_FACTOR_DEFAULT and cfg.seq_inner_size > SINNER_FACTOR_DEFAULT
        pre_tokens = cfg.pre_tokens
        next_tokens = cfg.next_tokens
        if cfg.sparse_mode == SPARSE_MODE_NO_MASK:
            pre_tokens = 0 if pre_tokens > 0 else pre_tokens
        elif cfg.sparse_mode == SPARSE_MODE_BAND:
            next_tokens = 0 if next_tokens > 0 else next_tokens
        check_sparse = (
            cfg.sparse_mode == SPARSE_MODE_ALL_MASK
            or cfg.sparse_mode == SPARSE_MODE_RIGHT_DOWN
            or (
                cfg.sparse_mode in (SPARSE_MODE_NO_MASK, SPARSE_MODE_BAND)
                and pre_tokens + next_tokens > SINNER_FACTOR_DEFAULT
            )
        )
        if check_dtype and check_seq and check_sparse and not cfg.enable_pfa_rope and not cfg.enable_perblock_quant:
            s_outer = SOUTER_FACTOR_SUB
            s_inner = SINNER_FACTOR_DOUBLE
            softmax_s_outer = SOUTER_FACTOR_SUB
            notes.append("small_vd_sparse_path: Souter=32, Sinner=256, SoftmaxSouter=32")
        elif cfg.layout in LAYOUTS_MERGED and cfg.enable_pfa_merge:
            s_outer = SOUTER_FACTOR_SUB
            s_inner = SINNER_FACTOR_DOUBLE
            notes.append("pfa_merge_small_vd_path: Souter=32, Sinner=256, SoftmaxSouter remains 64")
    elif cfg.v_head_size > 128 and not cfg.enable_ifa_mla and not cfg.enable_ifa:
        if not cfg.fa_run_flag:
            s_outer = SOUTER_FACTOR_SUB
            s_inner = SINNER_FACTOR_SUB
            notes.append("large_vd_non_fa_run_path: Souter=32, Sinner=64")
        elif cfg.layout in LAYOUTS_MERGED and cfg.enable_pfa_merge and cfg.v_head_size <= 256:
            s_outer = SOUTER_FACTOR_SUB
            s_inner = SINNER_FACTOR_DOUBLE
            notes.append("pfa_merge_large_vd_le_256_path: Souter=32, Sinner=256")
        else:
            s_outer = SOUTER_FACTOR_DEFAULT
            s_inner = SINNER_FACTOR_DEFAULT
            notes.append("large_vd_default_path: Souter=64, Sinner=128")
        softmax_s_outer = SOUTER_FACTOR_SUB
    elif cfg.enable_ifa_mla or (cfg.enable_ifa and cfg.v_head_size > 128):
        if not cfg.fa_run_flag or cfg.enable_ifa_mla:
            s_outer = SOUTER_FACTOR_SUB
        else:
            s_outer = SOUTER_FACTOR_DEFAULT
        if not cfg.fa_run_flag and not cfg.enable_ifa_mla and cfg.enable_pse_shift:
            s_inner = SINNER_FACTOR_SUB
        else:
            s_inner = SINNER_FACTOR_DEFAULT
        softmax_s_outer = SOUTER_FACTOR_SUB
        notes.append("ifa_or_ifa_mla_large_vd_path")
    else:
        notes.append("default_path: Souter=64, Sinner=128, SoftmaxSouter=64")

    result = {
        "Souter": s_outer,
        "CubeSouter": s_outer * CV_RATIO,
        "Sinner": s_inner,
        "SoftmaxSouter": softmax_s_outer,
        "splitS2": cfg.split_s2,
        "bmm_checks": bmm_checks,
    }
    return result, notes


def apply_dn_adjustment(cfg: PFAConfig, tiling: Dict[str, Any]) -> Dict[str, Any]:
    q_dim = cfg.q_head_size if cfg.q_head_size is not None else cfg.qk_head_size
    enable_dn = (
        not cfg.enable_mask
        and not cfg.enable_pse_shift
        and not cfg.enable_alibi_pse
        and not cfg.enable_pa
        and not cfg.enable_pfa_mla
        and not cfg.enable_pfa_rope
        and q_dim <= 128
        and cfg.v_head_size <= 128
        and not cfg.enable_kv_prefix
        and (cfg.input_dtype in FP16_DTYPE_NAMES or cfg.input_dtype in BF16_DTYPE_NAMES or cfg.enable_perblock_quant)
        and tiling["Souter"] * 2 > 64
    )
    actual_right = all((s1 % 32 == 0 and s2 > 128) for s1, s2 in zip(cfg.actual_seq_lengths, cfg.actual_seq_lengths_kv))
    changed = False
    reasons = []
    if enable_dn and actual_right and q_dim == cfg.v_head_size and q_dim == 64:
        tiling["Sinner"] = SINNER_FACTOR_DOUBLE
        changed = True
        reasons.append("DN qD==vD==64 and actual seq constraints")
    if enable_dn and q_dim == cfg.v_head_size and q_dim <= 128 and cfg.enable_perblock_quant:
        tiling["Sinner"] = SINNER_FACTOR_DOUBLE
        changed = True
        reasons.append("DN perblock quant qD==vD<=128")
    tiling["enableDN"] = enable_dn
    tiling["dnAdjusted"] = changed
    tiling["dnReasons"] = reasons
    return tiling


def get_pre_next_left_up(cfg: PFAConfig, actual_s1: int, actual_s2_with_prefix: int) -> Tuple[int, int]:
    layout_opt = cfg.layout in LAYOUTS_MERGED
    if cfg.sparse_mode == SPARSE_MODE_RIGHT_DOWN:
        pre = SPARSE_MODE_INT_MAX
        if cfg.enable_ifa_mla:
            nxt = actual_s2_with_prefix * cfg.g_size - actual_s1 if layout_opt else SPARSE_MODE_INT_MAX
        elif cfg.enable_ifa:
            nxt = actual_s2_with_prefix * cfg.g_size - actual_s1
        else:
            nxt = actual_s2_with_prefix - actual_s1
        return pre, nxt

    if cfg.sparse_mode == SPARSE_MODE_BAND:
        if cfg.enable_ifa_mla:
            if layout_opt:
                pre = cfg.pre_tokens * cfg.g_size - actual_s2_with_prefix * cfg.g_size + actual_s1
                nxt = cfg.next_tokens * cfg.g_size + actual_s2_with_prefix * cfg.g_size - actual_s1
            else:
                pre = SPARSE_MODE_INT_MAX
                nxt = SPARSE_MODE_INT_MAX
        elif cfg.enable_ifa:
            pre = cfg.pre_tokens * cfg.g_size - actual_s2_with_prefix * cfg.g_size + actual_s1
            nxt = cfg.next_tokens * cfg.g_size + actual_s2_with_prefix * cfg.g_size - actual_s1
        else:
            pre = cfg.pre_tokens - actual_s2_with_prefix + actual_s1
            nxt = cfg.next_tokens + actual_s2_with_prefix - actual_s1
        return pre, nxt

    if cfg.enable_ifa_mla:
        if layout_opt:
            return cfg.pre_tokens * cfg.g_size, cfg.next_tokens * cfg.g_size
        return SPARSE_MODE_INT_MAX, SPARSE_MODE_INT_MAX
    if cfg.enable_ifa:
        return cfg.pre_tokens * cfg.g_size, cfg.next_tokens * cfg.g_size
    return cfg.pre_tokens, cfg.next_tokens


def fix_param_with_row_invalid(cfg: PFAConfig, actual_s1: int, actual_s2_with_prefix: int, pre: int, nxt: int) -> Tuple[int, int, int]:
    next_error = -nxt if nxt < 0 else 0
    next_error = min(next_error, actual_s1)
    if cfg.enable_ifa_mla:
        pre_error = actual_s1 - actual_s2_with_prefix * cfg.g_size - pre
    else:
        pre_error = actual_s1 - actual_s2_with_prefix - pre
    pre_error = max(0, min(pre_error, actual_s1))

    nxt += next_error
    pre -= next_error
    actual_s1 -= next_error
    actual_s1 -= pre_error
    return actual_s1, pre, nxt


def get_actual_inner_block_nums(start: int, end: int, inner_blocks: int) -> int:
    if end < 0:
        return 0
    if end < inner_blocks:
        return end + 1 if start < 0 else end - start + 1
    return inner_blocks if start < 0 else (inner_blocks - start if start < inner_blocks else 0)


def sum_arithmetic_series(an: int, d: int) -> int:
    if d == 0:
        return 0
    return (an % d + an) * (an // d + 1) // 2 if an > 0 else 0


def get_cut_block_nums(block_seq_kv: int, block_seq: int, s_inner: int, s_outer: int, token: int) -> int:
    block_token = ceil_div(token, s_inner) * s_inner if token > 0 else cxx_div(token, s_inner) * s_inner
    out_div_in = s_outer // s_inner if s_outer > s_inner else 1
    in_div_out = s_inner // s_outer if s_inner > s_outer else 1
    if out_div_in >= 1:
        tolerance = out_div_in
        small_size = s_inner
    else:
        tolerance = in_div_out
        small_size = s_outer

    blocks = 0
    blocks += sum_arithmetic_series(cxx_div(block_seq_kv - block_token, small_size) - tolerance, tolerance)
    blocks -= sum_arithmetic_series(cxx_div(-block_token, small_size) - tolerance, tolerance)
    blocks -= sum_arithmetic_series(cxx_div(block_seq_kv - block_seq - block_token, small_size) - tolerance, tolerance)
    blocks += sum_arithmetic_series(cxx_div(-block_token - block_seq, small_size) - tolerance, tolerance)
    return blocks


def calc_block_nums_one_head(
    cfg: PFAConfig,
    actual_s1: int,
    actual_s2: int,
    s_outer: int,
    s_inner: int,
    pre: int,
    nxt: int,
    is_atten_mask_used: bool,
) -> int:
    prefix = cfg.actual_shared_prefix_len
    if not is_atten_mask_used:
        outer_blocks = ceil_div(actual_s1, s_outer)
        inner_blocks = ceil_div(actual_s2, s_inner) + ceil_div(prefix, s_inner)
        return outer_blocks * inner_blocks

    inner_blocks = ceil_div(actual_s2, s_inner)
    block_seq_kv = inner_blocks * s_inner
    outer_blocks = ceil_div(actual_s1, s_outer)
    block_seq = outer_blocks * s_outer
    calc_blocks = inner_blocks * outer_blocks
    calc_blocks -= get_cut_block_nums(block_seq_kv, block_seq, s_inner, s_outer, nxt - prefix)
    calc_blocks -= get_cut_block_nums(block_seq_kv, block_seq, s_inner, s_outer, block_seq_kv - block_seq + pre + prefix)

    prefix_inner_blocks = ceil_div(prefix, s_inner)
    block_prefix = prefix_inner_blocks * s_inner
    calc_blocks += prefix_inner_blocks * outer_blocks
    calc_blocks -= get_cut_block_nums(block_prefix, block_seq, s_inner, s_outer, nxt)
    calc_blocks -= get_cut_block_nums(block_prefix, block_seq, s_inner, s_outer, block_prefix - block_seq + pre)
    return calc_blocks


def compute_split_core(cfg: PFAConfig, tiling: Dict[str, Any]) -> Dict[str, Any]:
    s_outer = int(tiling["Souter"])
    s_inner = int(tiling["Sinner"])
    cur_core_num = cfg.core_num
    split_by_cube = True
    if split_by_cube:
        s_outer_for_split = s_outer * CV_RATIO
        cur_core_num = max(1, cur_core_num // CV_RATIO)
    else:
        s_outer_for_split = s_outer

    total_blocks_one_head = 0
    multi_smax_inner_loop_times = 0
    is_atten_mask_used = cfg.enable_mask
    for b_idx in range(cfg.batch_size):
        actual_s1_tmp = cfg.actual_seq_lengths[b_idx]
        actual_s2 = cfg.actual_seq_lengths_kv[b_idx]
        pre, nxt = get_pre_next_left_up(cfg, actual_s1_tmp, actual_s2 + cfg.actual_shared_prefix_len)
        actual_s1_tmp, pre, nxt = fix_param_with_row_invalid(
            cfg, actual_s1_tmp, actual_s2 + cfg.actual_shared_prefix_len, pre, nxt
        )
        s_inner_loops = ceil_div(actual_s2, s_inner) + ceil_div(cfg.actual_shared_prefix_len, s_inner)
        multi_smax_inner_loop_times = max(multi_smax_inner_loop_times, s_inner_loops)
        total_blocks_one_head += calc_block_nums_one_head(
            cfg, actual_s1_tmp, actual_s2, s_outer_for_split, s_inner, pre, nxt, is_atten_mask_used
        )

    core_weight_target = (total_blocks_one_head * cfg.head_num_size) / float(cur_core_num)
    core_n_start: List[int] = []
    core_n_end: List[int] = []
    core_sid_start: List[int] = []
    core_sid_end: List[int] = []
    core_spos_start: List[int] = []
    core_spos_end: List[int] = []
    bn_start_idx: List[int] = []
    gs1_start_idx: List[int] = []
    core_weights: List[int] = []

    def ensure_core_start(core: int, head: int, batch: int, spos: int) -> None:
        while len(core_n_start) <= core:
            core_n_start.append(0)
            core_n_end.append(0)
            core_sid_start.append(0)
            core_sid_end.append(0)
            core_spos_start.append(0)
            core_spos_end.append(0)
            bn_start_idx.append(0)
            gs1_start_idx.append(0)
            core_weights.append(0)
        core_n_start[core] = head
        core_sid_start[core] = batch
        core_spos_start[core] = spos
        bn_start_idx[core] = batch * cfg.head_num_size + head
        gs1_start_idx[core] = spos

    cur_weight = 0
    cur_core = 0
    tmp_n_end = 0
    tmp_sid_end = 0
    tmp_spos_end = 0
    ensure_core_start(0, 0, 0, 0)
    prefix_inner_blocks = ceil_div(cfg.actual_shared_prefix_len, s_inner)

    for b_idx in range(cfg.batch_size):
        for head in range(cfg.head_num_size):
            pre, nxt = get_pre_next_left_up(
                cfg, cfg.actual_seq_lengths[b_idx], cfg.actual_seq_lengths_kv[b_idx] + cfg.actual_shared_prefix_len
            )
            actual_s1 = cfg.actual_seq_lengths[b_idx]
            actual_s2 = cfg.actual_seq_lengths_kv[b_idx]
            actual_s1, pre, nxt = fix_param_with_row_invalid(
                cfg, actual_s1, actual_s2 + cfg.actual_shared_prefix_len, pre, nxt
            )
            outer_blocks = ceil_div(actual_s1, s_outer_for_split)
            inner_blocks = ceil_div(actual_s2, s_inner)
            for s_outer_idx in range(outer_blocks):
                diff = int(core_weight_target * float(cur_core + 1)) - cur_weight
                pre_no_prefix = pre + cfg.actual_shared_prefix_len
                next_no_prefix = nxt - cfg.actual_shared_prefix_len

                if pre_no_prefix > 0:
                    inner_start = -ceil_div(pre_no_prefix, s_inner)
                else:
                    inner_start = -cxx_div(pre_no_prefix, s_inner)
                if next_no_prefix > 0:
                    inner_end = ceil_div(next_no_prefix, s_inner)
                else:
                    inner_end = cxx_div(next_no_prefix, s_inner)

                if pre > 0:
                    prefix_start = -ceil_div(pre, s_inner)
                else:
                    prefix_start = -cxx_div(pre, s_inner)
                if nxt > 0:
                    prefix_end = ceil_div(nxt, s_inner)
                else:
                    prefix_end = cxx_div(nxt, s_inner)

                actual_inner_blocks = get_actual_inner_block_nums(inner_start, inner_end, inner_blocks)
                actual_inner_blocks += get_actual_inner_block_nums(prefix_start, prefix_end, prefix_inner_blocks)
                is_first_block = tmp_n_end == 0 and tmp_sid_end == 0 and tmp_spos_end == 0
                if actual_inner_blocks - diff > diff and not is_first_block:
                    core_n_end[cur_core] = tmp_n_end
                    core_sid_end[cur_core] = tmp_sid_end
                    core_spos_end[cur_core] = tmp_spos_end
                    cur_core += 1
                    ensure_core_start(cur_core, head, b_idx, s_outer_idx)

                tmp_n_end = head + 1
                tmp_sid_end = b_idx + 1
                tmp_spos_end = s_outer_idx + 1
                core_weights[cur_core] += actual_inner_blocks
                cur_weight += actual_inner_blocks
                pre -= s_outer_for_split
                nxt += s_outer_for_split

    ensure_core_start(cur_core, core_n_start[cur_core], core_sid_start[cur_core], core_spos_start[cur_core])
    core_n_end[cur_core] = tmp_n_end
    core_sid_end[cur_core] = tmp_sid_end
    core_spos_end[cur_core] = tmp_spos_end
    bn_start_idx.append(cfg.batch_size * cfg.head_num_size)
    gs1_start_idx.append(tmp_spos_end)

    cube_used_cores = cur_core + 1
    actual_core_nums = cube_used_cores * CV_RATIO if split_by_cube else cube_used_cores
    sinner_block_num = ceil_div(cfg.seq_inner_size, s_inner)
    total_size = (total_blocks_one_head // sinner_block_num) * cfg.head_num_size if sinner_block_num else 0
    split_factor_size = ceil_div(total_size, cube_used_cores)
    used_core_weights = core_weights[:cube_used_cores]
    candidate_core_weights = used_core_weights + [0] * max(0, cur_core_num - cube_used_cores)
    load_balance = build_load_balance(candidate_core_weights, cur_core_num, core_weight_target, cube_used_cores)

    core_ranges = []
    for idx in range(cube_used_cores):
        core_ranges.append(
            {
                "cube_core": idx,
                "aiv_core_pair": [idx * 2, idx * 2 + 1] if split_by_cube else [idx],
                "n_start": core_n_start[idx],
                "n_end": core_n_end[idx],
                "batch_start": core_sid_start[idx],
                "batch_end": core_sid_end[idx],
                "souter_start": core_spos_start[idx],
                "souter_end": core_spos_end[idx],
                "bn_start_idx": bn_start_idx[idx],
                "bn_end_idx": bn_start_idx[idx + 1] if idx + 1 < len(bn_start_idx) else None,
                "gs1_start_idx": gs1_start_idx[idx],
                "gs1_end_idx": gs1_start_idx[idx + 1] if idx + 1 < len(gs1_start_idx) else None,
                "task_blocks": used_core_weights[idx],
                "load_ratio_to_mean": (
                    used_core_weights[idx] / load_balance["meanTaskBlocks"]
                    if load_balance["meanTaskBlocks"] > 0
                    else 0
                ),
            }
        )

    return {
        "split_core_mode": "SPLIT_NBS_CUBE",
        "souter_for_split": s_outer_for_split,
        "candidate_cube_cores": cur_core_num,
        "core_weight_target": core_weight_target,
        "total_block_nums_one_head": total_blocks_one_head,
        "multiSmaxsInnerLoopTimes": multi_smax_inner_loop_times,
        "actualCoreNums": actual_core_nums,
        "cubeUsedCores": cube_used_cores,
        "coreTaskBlocks": used_core_weights,
        "candidateCoreTaskBlocks": candidate_core_weights,
        "loadBalance": load_balance,
        "multiCoreParamsRegbase": {
            "coreNum": cube_used_cores,
            "totalSize": total_size,
            "s1OuterSize": ceil_div(cfg.seq_size, s_outer_for_split),
            "splitFactorSize": split_factor_size,
            "splitFactorTailSize": calc_tail_size(total_size, split_factor_size),
            "bnStartIdx": bn_start_idx,
            "sparseStartIdx": gs1_start_idx,
        },
        "promptAttentionSeqParams": {
            "CoreHeadNumTail_coreNStart": core_n_start[:cube_used_cores],
            "actualS1_coreNEnd": core_n_end[:cube_used_cores],
            "actualCoreNums_coreSidStart": core_sid_start[:cube_used_cores],
            "singleCoreHeadNumSize_coreSidEnd": core_sid_end[:cube_used_cores],
            "coreSeqPosStart": core_spos_start[:cube_used_cores],
            "coreSeqPosEnd": core_spos_end[:cube_used_cores],
        },
        "coreRanges": core_ranges,
    }


def build_load_balance(
    core_weights: List[int], candidate_cube_cores: int, target: float, used_cube_cores: int | None = None
) -> Dict[str, Any]:
    used = used_cube_cores if used_cube_cores is not None else len([w for w in core_weights if w > 0])
    total = sum(core_weights)
    measured_cores = len(core_weights)
    mean = total / measured_cores if measured_cores else 0.0
    max_weight = max(core_weights) if core_weights else 0
    min_weight = min(core_weights) if core_weights else 0
    max_core = core_weights.index(max_weight) if core_weights else -1
    min_core = core_weights.index(min_weight) if core_weights else -1
    variance = sum((w - mean) ** 2 for w in core_weights) / measured_cores if measured_cores else 0.0
    stddev = math.sqrt(variance)
    cv = stddev / mean if mean > 0 else 0.0
    max_over_mean = max_weight / mean if mean > 0 else 0.0
    min_over_mean = min_weight / mean if mean > 0 else 0.0
    idle_candidate_cores = max(0, candidate_cube_cores - used)

    if used <= 1:
        rating = "single_core"
    elif max_over_mean <= 1.10 and cv <= 0.08:
        rating = "excellent"
    elif max_over_mean <= 1.25 and cv <= 0.15:
        rating = "good"
    elif max_over_mean <= 1.50 and cv <= 0.25:
        rating = "moderate"
    else:
        rating = "poor"

    interpretation = [
        f"Used {used}/{candidate_cube_cores} cube cores; {idle_candidate_cores} candidate cube cores are idle and shown as zero bars.",
        f"Task blocks total={total}, candidate-core mean={mean:.2f}, max={max_weight} on core {max_core}, min={min_weight} on core {min_core}.",
        f"Max/mean={max_over_mean:.3f}, min/mean={min_over_mean:.3f}, coefficient_of_variation={cv:.3f}; rating={rating}.",
        "Task blocks are the same effective inner-block weights used by the split-core estimator.",
    ]

    return {
        "usedCubeCores": used,
        "candidateCubeCores": candidate_cube_cores,
        "idleCandidateCubeCores": idle_candidate_cores,
        "totalTaskBlocks": total,
        "meanTaskBlocks": mean,
        "targetTaskBlocks": target,
        "maxTaskBlocks": max_weight,
        "maxTaskCore": max_core,
        "minTaskBlocks": min_weight,
        "minTaskCore": min_core,
        "stddevTaskBlocks": stddev,
        "coefficientOfVariation": cv,
        "maxOverMean": max_over_mean,
        "minOverMean": min_over_mean,
        "rating": rating,
        "interpretation": interpretation,
    }


def render_load_balance_svg(result: Dict[str, Any], path: str) -> None:
    split_core = result["splitCore"]
    weights = split_core.get("candidateCoreTaskBlocks") or split_core.get("coreTaskBlocks", [])
    load_balance = split_core.get("loadBalance", {})
    if not weights:
        raise SystemExit("No core task weights are available for plotting.")

    width = max(900, 80 + len(weights) * 34)
    height = 520
    margin_left = 72
    margin_right = 28
    margin_top = 86
    margin_bottom = 92
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_weight = max(max(weights), 1)
    mean = float(load_balance.get("meanTaskBlocks", 0.0))
    target = float(load_balance.get("targetTaskBlocks", 0.0))
    bar_gap = 8
    bar_w = max(10, (plot_w - bar_gap * (len(weights) - 1)) / len(weights))

    def y_of(value: float) -> float:
        return margin_top + plot_h - (value / max_weight) * plot_h

    mean_y = y_of(mean)
    target_y = y_of(target) if target > 0 else None
    title = "PFA split-core task blocks per cube core"
    subtitle = (
        f"rating={load_balance.get('rating', 'n/a')} | "
        f"used={load_balance.get('usedCubeCores', len(weights))}/"
        f"{load_balance.get('candidateCubeCores', len(weights))} cube cores | "
        f"max/mean={float(load_balance.get('maxOverMean', 0.0)):.3f} | "
        f"cv={float(load_balance.get('coefficientOfVariation', 0.0)):.3f}"
    )

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{margin_left}" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="{margin_left}" y="60" font-family="Arial, sans-serif" font-size="13" fill="#475569">{html.escape(subtitle)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#334155" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#334155" stroke-width="1"/>',
    ]

    for tick in range(5):
        value = max_weight * tick / 4
        y = y_of(value)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#64748b">{value:.0f}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{mean_y:.2f}" x2="{width - margin_right}" y2="{mean_y:.2f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 5"/>')
    parts.append(f'<text x="{width - margin_right}" y="{mean_y - 7:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#dc2626">mean {mean:.1f}</text>')
    if target_y is not None:
        parts.append(f'<line x1="{margin_left}" y1="{target_y:.2f}" x2="{width - margin_right}" y2="{target_y:.2f}" stroke="#2563eb" stroke-width="2" stroke-dasharray="3 5"/>')
        parts.append(f'<text x="{width - margin_right}" y="{target_y + 16:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#2563eb">target {target:.1f}</text>')

    for idx, weight in enumerate(weights):
        x = margin_left + idx * (bar_w + bar_gap)
        y = y_of(weight)
        h = margin_top + plot_h - y
        ratio = weight / mean if mean > 0 else 0
        if weight == 0:
            fill = "#cbd5e1"
        else:
            fill = "#0f766e" if ratio <= 1.10 else "#f59e0b" if ratio <= 1.35 else "#dc2626"
        display_y = margin_top + plot_h - 2 if weight == 0 else y
        display_h = 2 if weight == 0 else h
        parts.append(
            f'<rect x="{x:.2f}" y="{display_y:.2f}" width="{bar_w:.2f}" '
            f'height="{display_h:.2f}" rx="2" fill="{fill}"/>'
        )
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{max(y - 6, margin_top - 8):.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#334155">{weight}</text>')
        label_y = margin_top + plot_h + 18
        label = str(idx)
        if len(weights) > 36 and idx % 2 == 1:
            label = ""
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{label_y}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#475569">{label}</text>')

    legend_y = height - 44
    legend = [
        {"kind": "box", "color": "#0f766e", "text": "<=110% mean"},
        {"kind": "box", "color": "#f59e0b", "text": "<=135% mean"},
        {"kind": "box", "color": "#dc2626", "text": ">135% mean"},
        {"kind": "box", "color": "#cbd5e1", "text": "idle core"},
        {"kind": "line", "color": "#dc2626", "dash": "6 5", "text": "mean line"},
        {"kind": "line", "color": "#2563eb", "dash": "3 5", "text": "target line"},
    ]
    x = margin_left
    for item in legend:
        if item["kind"] == "line":
            parts.append(
                f'<line x1="{x}" y1="{legend_y - 4}" x2="{x + 24}" y2="{legend_y - 4}" '
                f'stroke="{item["color"]}" stroke-width="2" stroke-dasharray="{item["dash"]}"/>'
            )
            text_x = x + 32
        else:
            parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" fill="{item["color"]}"/>')
            text_x = x + 18
        parts.append(
            f'<text x="{text_x}" y="{legend_y}" font-family="Arial, sans-serif" '
            f'font-size="12" fill="#475569">{html.escape(item["text"])}</text>'
        )
        x += 128

    parts.append(f'<text x="{margin_left}" y="{height - 18}" font-family="Arial, sans-serif" font-size="12" fill="#64748b">x-axis: cube core id; each cube core maps to two AIV cores in SPLIT_NBS_CUBE.</text>')
    parts.append("</svg>")

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def simulate(cfg: PFAConfig) -> Dict[str, Any]:
    tiling, notes = adjust_cv_tiling(cfg)
    tiling = apply_dn_adjustment(cfg, tiling)
    split_core = compute_split_core(cfg, tiling)
    return {
        "normalizedInput": {
            "batch_size": cfg.batch_size,
            "head_num_size": cfg.head_num_size,
            "seq_size": cfg.seq_size,
            "seq_inner_size": cfg.seq_inner_size,
            "qk_head_size": cfg.qk_head_size,
            "v_head_size": cfg.v_head_size,
            "q_head_size": cfg.q_head_size,
            "actual_seq_lengths": cfg.actual_seq_lengths,
            "actual_seq_lengths_kv": cfg.actual_seq_lengths_kv,
            "g_size": cfg.g_size,
            "layout": cfg.layout,
        },
        "tiling": tiling,
        "splitCore": split_core,
        "notes": notes,
        "limitations": [
            "MatmulApiTiling::GetTiling is not executed; BMM checks are parameter plans only.",
            "The script targets the main v2 SPLIT_NBS_CUBE path and approximates host-side behavior for analysis.",
        ],
    }


def example_input() -> Dict[str, Any]:
    return {
        "shape": {
            "batch_size": 1,
            "head_num_size": 32,
            "seq_size": 1024,
            "seq_inner_size": 1024,
            "qk_head_size": 128,
            "v_head_size": 128,
        },
        "attrs": {
            "input_dtype": "fp16",
            "output_dtype": "fp16",
            "inner_precise": "high_precision",
            "layout": "BSH",
            "sparse_mode": 0,
            "pre_tokens": 2147483647,
            "next_tokens": 2147483647,
            "actual_seq_lengths": [1024],
            "actual_seq_lengths_kv": [1024],
            "actual_shared_prefix_len": 0,
            "g_size": 1,
            "head_num_ratio": 1,
        },
        "platform": {
            "core_num": 64,
            "aic_num": 32,
            "l1_size": 1048576,
            "l0c_size": 262144,
        },
        "flags": {
            "enable_mask": False,
            "enable_pse_shift": False,
            "enable_alibi_pse": False,
            "enable_pa": False,
            "enable_pfa_mla": False,
            "enable_pfa_rope": False,
            "enable_pfa_merge": False,
            "enable_ifa": False,
            "enable_ifa_mla": False,
            "enable_perblock_quant": False,
            "fa_run_flag": True,
            "split_s2": 1,
        },
    }


def load_input(path: str | None) -> Dict[str, Any]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raw = sys.stdin.read()
    if not raw.strip():
        raise SystemExit("No input provided. Use --example to print a sample JSON.")
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate PromptFlashAttentionTilingV2 tiling and split-core decisions.")
    parser.add_argument("-i", "--input", help="Input JSON file. If omitted, reads JSON from stdin.")
    parser.add_argument("--example", action="store_true", help="Print an example input JSON and exit.")
    parser.add_argument("--compact", action="store_true", help="Print compact JSON.")
    parser.add_argument("--plot", help="Write an SVG bar chart of per-cube-core task blocks to this path.")
    args = parser.parse_args()

    if args.example:
        print(json.dumps(example_input(), indent=2, ensure_ascii=False))
        return 0

    data = load_input(args.input)
    cfg = PFAConfig.from_dict(data)
    result = simulate(cfg)
    if args.plot:
        render_load_balance_svg(result, args.plot)
        result["visualization"] = {"loadBalanceSvg": os.path.abspath(args.plot)}
    if args.compact:
        print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
