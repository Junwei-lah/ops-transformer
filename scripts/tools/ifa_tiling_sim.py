#!/usr/bin/env python3
"""
Simulate the core tiling decisions in
attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp.

This is a host-side analysis helper, not a bit-exact replacement for CANN
tiling APIs. In particular, GetSoftMaxFlashV2MinTmpSize is reported as a
symbolic API-sized value because the real implementation lives in AscendC.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


BYTE_BLOCK = 32
NUM16 = 16
NUM32 = 32
NUM64 = 64
NUM128 = 128
NUM256 = 256
NUM512 = 512
NUM1024 = 1024
MAX_CORE_NUM_REGBASE = 66

SPARSE_MODE_RIGHT_DOWN = 3
SPARSE_MODE_BAND = 4


def ceil_div(x: int, y: int) -> int:
    if y == 0:
        raise ValueError("division by zero")
    return (x + y - 1) // y


def align_up(x: int, align: int) -> int:
    if align <= 0:
        return x
    return ceil_div(x, align) * align


def calc_tail_size(x: int, tile: int) -> int:
    if tile == 0:
        return 0
    mod = x % tile
    return mod if mod != 0 else tile


def cxx_div(x: int, y: int) -> int:
    """C++ signed integer division truncates toward zero."""
    if y == 0:
        raise ValueError("division by zero")
    sign = -1 if (x < 0) ^ (y < 0) else 1
    return sign * (abs(x) // abs(y))


def as_bool(data: Dict[str, Any], name: str, default: bool = False) -> bool:
    return bool(data.get(name, default))


def as_int(data: Dict[str, Any], name: str, default: int = 0) -> int:
    return int(data.get(name, default))


def sum_arithmetic_series(an: int, d: int) -> int:
    """Copy of SumOfArithmeticSeries(an, d)."""
    if d == 0:
        return 0
    return (an % d + an) * (an // d + 1) // 2 if an > 0 else 0


def get_cut_block_nums(block_seq_len_kv: int, block_seq_len: int, s_inner: int, s_outer: int, token: int) -> int:
    """Copy of IFATilingV2::GetCutBlockNums."""
    block_token = ceil_div(token, s_inner) * s_inner if token > 0 else cxx_div(token, s_inner) * s_inner
    out_div_in = s_outer // s_inner if s_outer > s_inner else 1
    in_div_out = s_inner // s_outer if s_inner > s_outer else 1
    if out_div_in >= 1:
        tolerance = out_div_in
        small_size = s_inner
    else:
        tolerance = in_div_out
        small_size = s_outer

    block_nums = 0
    block_nums += sum_arithmetic_series(cxx_div(block_seq_len_kv - block_token, small_size) - tolerance, tolerance)
    block_nums -= sum_arithmetic_series(cxx_div(-block_token, small_size) - tolerance, tolerance)
    block_nums -= sum_arithmetic_series(
        cxx_div(block_seq_len_kv - block_seq_len - block_token, small_size) - tolerance, tolerance
    )
    block_nums += sum_arithmetic_series(cxx_div(-block_token - block_seq_len, small_size) - tolerance, tolerance)
    return block_nums


def get_actual_inner_block_nums(s_inner_start: int, s_inner_end: int, inner_block_nums: int) -> int:
    """Copy of IFATilingV2::GetActualInnerBlockNums."""
    if s_inner_end < 0:
        return 0
    if s_inner_end < inner_block_nums:
        return s_inner_end + 1 if s_inner_start < 0 else s_inner_end - s_inner_start + 1
    return inner_block_nums if s_inner_start < 0 else (inner_block_nums - s_inner_start if s_inner_start < inner_block_nums else 0)


@dataclass
class IFATilingInput:
    batch_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    q_seq: int = 1
    kv_seq: int = 0
    aic_num: int = 24
    aiv_num: Optional[int] = None
    core_num: Optional[int] = None
    block_type_size: int = 4
    pse_shift: bool = False
    is_gqa: Optional[bool] = None
    fa_run_gs: bool = False
    is_pfa: bool = False
    atten_mask: bool = False
    sparse_mode: int = 0
    pre_token: int = 0
    next_token: int = 0
    actual_shared_prefix_len: int = 0
    actual_q_lens: Optional[List[int]] = None
    actual_kv_lens: Optional[List[int]] = None
    max_actualseq: Optional[int] = None
    force_flash_decode: Optional[bool] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "IFATilingInput":
        shape = data.get("shape", data)
        attrs = data.get("attrs", data)
        platform = data.get("platform", data)
        flags = data.get("flags", data)

        q_heads = as_int(shape, "q_heads", as_int(shape, "head_num_size", as_int(shape, "n", 1)))
        kv_heads = as_int(shape, "kv_heads", as_int(shape, "num_key_value_heads", as_int(shape, "n_kv", q_heads)))
        q_seq = as_int(shape, "q_seq", as_int(shape, "seq_size", as_int(shape, "s1", 1)))
        kv_seq = as_int(shape, "kv_seq", as_int(shape, "seq_inner_size", as_int(shape, "s2", q_seq)))
        block_type_size = as_int(attrs, "block_type_size", 4)
        dtype = str(attrs.get("input_dtype", attrs.get("dtype", ""))).lower()
        if "fp16" in dtype or "float16" in dtype or "bf16" in dtype or "bfloat16" in dtype:
            block_type_size = int(attrs.get("block_type_size", 2))
        elif "fp32" in dtype or "float32" in dtype:
            block_type_size = int(attrs.get("block_type_size", 4))

        force_flash_decode: Optional[bool]
        if "force_flash_decode" in flags:
            force_flash_decode = None if flags["force_flash_decode"] is None else bool(flags["force_flash_decode"])
        elif "flash_decode" in flags:
            force_flash_decode = None if flags["flash_decode"] is None else bool(flags["flash_decode"])
        else:
            force_flash_decode = None

        return IFATilingInput(
            batch_size=as_int(shape, "batch_size", as_int(shape, "b", 1)),
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=as_int(shape, "head_dim", as_int(shape, "d", 128)),
            q_seq=q_seq,
            kv_seq=kv_seq,
            aic_num=as_int(platform, "aic_num", as_int(platform, "cube_core_num", 24)),
            aiv_num=platform.get("aiv_num"),
            core_num=platform.get("core_num"),
            block_type_size=block_type_size,
            pse_shift=as_bool(flags, "pse_shift", as_bool(flags, "enable_pse_shift", False)),
            is_gqa=flags.get("is_gqa"),
            fa_run_gs=as_bool(flags, "fa_run_gs", False),
            is_pfa=as_bool(flags, "is_pfa", as_bool(flags, "enable_pfa", False)),
            atten_mask=as_bool(flags, "atten_mask", as_bool(flags, "enable_mask", False)),
            sparse_mode=as_int(attrs, "sparse_mode", 0),
            pre_token=as_int(attrs, "pre_token", as_int(attrs, "pre_tokens", 0)),
            next_token=as_int(attrs, "next_token", as_int(attrs, "next_tokens", 0)),
            actual_shared_prefix_len=as_int(attrs, "actual_shared_prefix_len", 0),
            actual_q_lens=attrs.get("actual_q_lens", attrs.get("actual_seq_lengths")),
            actual_kv_lens=attrs.get("actual_kv_lens", attrs.get("actual_seq_lengths_kv")),
            max_actualseq=attrs.get("max_actualseq"),
            force_flash_decode=force_flash_decode,
        )


@dataclass
class BlockSizes:
    s_outer_size: int
    s_inner_size: int
    s_inner_size_for_tiling_key: int
    s_inner_loop_times: int
    s_inner_size_tail: int
    s_inner_size_align: int


@dataclass
class SoftmaxInfo:
    tmp_shape_rows: int
    tmp_shape_cols: int
    tmp_shape_cols_align_unit: int
    tmp_size_formula: str
    arch35_regbase_ub_bytes: int
    arch35_regbase_layout: Dict[str, str]
    fill_tiling_shape_rows: int
    fill_tiling_shape_cols: int


@dataclass
class FlashDecodeInfo:
    enabled: bool
    bng: int
    kv_split_limit: int
    initial_split_s2: int
    split_s2: int
    s_inner_loop_size_per_split: int
    accum_out_size_elems: int
    log_sum_exp_size_elems_one_buffer: int
    fd_workspace_bytes: int


def normalize_lens(values: Optional[Sequence[int]], default_value: int, batch_size: int, name: str) -> List[int]:
    if values is None:
        return [default_value for _ in range(batch_size)]
    if len(values) == 1:
        return [int(values[0]) for _ in range(batch_size)]
    if len(values) != batch_size:
        raise ValueError(f"{name} length should be 1 or batch_size={batch_size}, got {len(values)}")
    return [int(v) for v in values]


class IFATilingV2Simulator:
    def __init__(self, cfg: IFATilingInput):
        self.cfg = cfg
        if cfg.kv_heads <= 0 or cfg.q_heads <= 0:
            raise ValueError("q_heads and kv_heads must be positive")
        if cfg.q_heads % cfg.kv_heads != 0:
            raise ValueError("q_heads must be divisible by kv_heads")

        self.group = cfg.q_heads // cfg.kv_heads
        self.is_gqa = (self.group > 1) if cfg.is_gqa is None else cfg.is_gqa
        self.kv_seq = cfg.kv_seq if cfg.kv_seq > 0 else max(cfg.actual_kv_lens or [cfg.q_seq])
        self.actual_q_lens = normalize_lens(cfg.actual_q_lens, cfg.q_seq, cfg.batch_size, "actual_q_lens")
        self.actual_kv_lens = normalize_lens(cfg.actual_kv_lens, self.kv_seq, cfg.batch_size, "actual_kv_lens")
        self.max_actualseq = cfg.max_actualseq if cfg.max_actualseq is not None else max(self.actual_kv_lens)

        self.aic_num = cfg.aic_num
        self.aiv_num = cfg.aiv_num if cfg.aiv_num is not None else cfg.aic_num * 2
        self.core_num = cfg.core_num if cfg.core_num is not None else self.aiv_num

    def set_fa_run_base_size(self) -> Tuple[int, int]:
        s_outer = NUM16
        if self.cfg.q_seq > 1 and self.is_gqa:
            s_outer = NUM32

        if self.cfg.head_dim <= NUM64:
            s_inner = NUM1024
            if self.cfg.pse_shift or (self.cfg.q_seq > 1 and self.is_gqa):
                s_inner = NUM512
        elif self.cfg.head_dim <= NUM128:
            s_inner = NUM256
        elif self.cfg.head_dim <= NUM256:
            s_inner = NUM256
        else:
            s_inner = NUM128
        return s_outer, s_inner

    def calc_inner_size(self) -> BlockSizes:
        s_outer, s_inner = self.set_fa_run_base_size()
        s_inner_for_key = s_inner
        s_inner_loop_times = ceil_div(self.kv_seq, s_inner)
        s_inner_tail = self.kv_seq - (s_inner_loop_times - 1) * s_inner
        if s_inner > self.kv_seq:
            s_inner = self.kv_seq
        s_inner_align = align_up(s_inner, BYTE_BLOCK)
        return BlockSizes(s_outer, s_inner, s_inner_for_key, s_inner_loop_times, s_inner_tail, s_inner_align)

    def softmax_info(self, blocks: BlockSizes) -> SoftmaxInfo:
        align_unit = BYTE_BLOCK // self.cfg.block_type_size
        cols = align_up(blocks.s_inner_size, align_unit)
        return SoftmaxInfo(
            tmp_shape_rows=self.group,
            tmp_shape_cols=cols,
            tmp_shape_cols_align_unit=align_unit,
            tmp_size_formula=(
                f"GetSoftMaxFlashV2MinTmpSize(Shape({{{self.group}, {cols}}}), "
                f"{self.cfg.block_type_size}, {self.cfg.block_type_size}, true, false)"
            ),
            arch35_regbase_ub_bytes=7 * 512,
            arch35_regbase_layout={
                "0 * 512B": "softmaxMaxUb[0]",
                "1 * 512B": "softmaxMaxUb[1]",
                "2 * 512B": "softmaxSumUb[0]",
                "3 * 512B": "softmaxSumUb[1]",
                "4 * 512B": "softmaxExpUb[0]",
                "5 * 512B": "softmaxExpUb[1]",
                "6 * 512B": "softmaxTmpUb",
            },
            fill_tiling_shape_rows=1,
            fill_tiling_shape_cols=cols,
        )

    def get_pre_next_tokens_left_up(self, actual_q: int, actual_kv_with_prefix: int) -> Tuple[int, int]:
        pre = self.cfg.pre_token
        nxt = self.cfg.next_token
        if self.cfg.sparse_mode == SPARSE_MODE_RIGHT_DOWN:
            nxt = actual_kv_with_prefix - actual_q
        elif self.cfg.sparse_mode == SPARSE_MODE_BAND:
            pre = self.cfg.pre_token - actual_kv_with_prefix + actual_q
            nxt = self.cfg.next_token + actual_kv_with_prefix - actual_q
        return pre, nxt

    @staticmethod
    def fix_param_with_row_invalid(actual_q: int, actual_kv_with_prefix: int, pre: int, nxt: int) -> Tuple[int, int, int]:
        next_err = -nxt if nxt < 0 else 0
        next_err = min(next_err, actual_q)
        pre_err = actual_q - actual_kv_with_prefix - pre if actual_q > actual_kv_with_prefix + pre else 0
        pre_err = min(pre_err, actual_q)
        nxt += next_err
        pre -= next_err
        actual_q -= next_err
        actual_q -= pre_err
        return actual_q, pre, nxt

    def calc_block_nums_one_head(
        self,
        blocks: BlockSizes,
        outer_blocks: int,
        inner_blocks: int,
        prefix_inner_blocks: int,
        pre: int,
        nxt: int,
    ) -> int:
        if not self.cfg.atten_mask:
            return (inner_blocks + prefix_inner_blocks) * outer_blocks

        block_seq_len = outer_blocks * blocks.s_outer_size
        block_seq_len_kv = inner_blocks * blocks.s_inner_size
        total = inner_blocks * outer_blocks
        total -= get_cut_block_nums(
            block_seq_len_kv,
            block_seq_len,
            blocks.s_inner_size,
            blocks.s_outer_size,
            nxt - self.cfg.actual_shared_prefix_len,
        )
        total -= get_cut_block_nums(
            block_seq_len_kv,
            block_seq_len,
            blocks.s_inner_size,
            blocks.s_outer_size,
            block_seq_len_kv - block_seq_len + pre + self.cfg.actual_shared_prefix_len,
        )

        prefix_seq_len = prefix_inner_blocks * blocks.s_inner_size
        total += prefix_inner_blocks * outer_blocks
        total -= get_cut_block_nums(prefix_seq_len, block_seq_len, blocks.s_inner_size, blocks.s_outer_size, nxt)
        total -= get_cut_block_nums(
            prefix_seq_len,
            block_seq_len,
            blocks.s_inner_size,
            blocks.s_outer_size,
            prefix_seq_len - block_seq_len + pre,
        )
        return total

    def compute_split_nb_seq_farun(self, blocks: BlockSizes, s_outer_loops: List[int], s_inner_loops: List[int],
                                   prefix_inner_blocks: int, core_weight_target: float) -> Dict[str, Any]:
        split_heads = self.cfg.kv_heads if self.cfg.fa_run_gs else self.cfg.q_heads
        arr_len = max(self.aic_num, 64)
        bn_start = [0 for _ in range(arr_len)]
        gs1_start = [0 for _ in range(arr_len)]

        cur_weight = 0
        cur_core = 0
        tmp_core_nid_end = 0
        tmp_core_sid_end = 0
        tmp_core_spos_end = 0
        per_row: List[Dict[str, Any]] = []

        for b_idx in range(self.cfg.batch_size):
            actual_q = self.actual_q_lens[b_idx]
            if self.cfg.fa_run_gs:
                actual_q *= self.group
            actual_kv = self.actual_kv_lens[b_idx]

            for head in range(split_heads):
                pre, nxt = self.get_pre_next_tokens_left_up(actual_q, actual_kv + self.cfg.actual_shared_prefix_len)
                fixed_q, pre, nxt = self.fix_param_with_row_invalid(actual_q, actual_kv + self.cfg.actual_shared_prefix_len, pre, nxt)
                outer_blocks = s_outer_loops[b_idx]
                inner_blocks = s_inner_loops[b_idx]

                for s_outer_idx in range(outer_blocks):
                    diff = int(core_weight_target * float(cur_core + 1)) - cur_weight
                    if not self.cfg.atten_mask:
                        actual_inner = inner_blocks + prefix_inner_blocks
                    else:
                        pre_no_prefix = pre + self.cfg.actual_shared_prefix_len
                        next_no_prefix = nxt - self.cfg.actual_shared_prefix_len

                        start = -(
                            ceil_div(pre_no_prefix, blocks.s_inner_size)
                            if pre_no_prefix > 0
                            else cxx_div(pre_no_prefix, blocks.s_inner_size)
                        )
                        end = (
                            ceil_div(next_no_prefix, blocks.s_inner_size)
                            if next_no_prefix > 0
                            else cxx_div(next_no_prefix, blocks.s_inner_size)
                        )
                        start_prefix = -(ceil_div(pre, blocks.s_inner_size) if pre > 0 else cxx_div(pre, blocks.s_inner_size))
                        end_prefix = ceil_div(nxt, blocks.s_inner_size) if nxt > 0 else cxx_div(nxt, blocks.s_inner_size)

                        actual_inner = get_actual_inner_block_nums(start, end, inner_blocks)
                        actual_inner += get_actual_inner_block_nums(start_prefix, end_prefix, prefix_inner_blocks)

                    cut_to_next = False
                    if actual_inner - diff > diff and not (
                        tmp_core_nid_end == 0 and tmp_core_sid_end == 0 and tmp_core_spos_end == 0
                    ):
                        cur_core += 1
                        if cur_core >= arr_len:
                            raise ValueError(f"core split overflow: cur_core={cur_core}, array length={arr_len}")
                        bn_start[cur_core] = b_idx * split_heads + head
                        gs1_start[cur_core] = s_outer_idx
                        cut_to_next = True

                    per_row.append(
                        {
                            "core": cur_core,
                            "batch": b_idx,
                            "head": head,
                            "s_outer_index": s_outer_idx,
                            "actual_inner_block_nums": actual_inner,
                            "pre_tokens": pre,
                            "next_tokens": nxt,
                            "cut_to_next_core_before_row": cut_to_next,
                        }
                    )
                    tmp_core_nid_end = head + 1
                    tmp_core_sid_end = b_idx + 1
                    tmp_core_spos_end = s_outer_idx + 1
                    cur_weight += actual_inner
                    pre -= blocks.s_outer_size
                    nxt += blocks.s_outer_size

        if cur_core + 1 >= arr_len:
            raise ValueError(f"core split overflow at final sentinel: cur_core={cur_core}, array length={arr_len}")
        bn_start[cur_core + 1] = self.cfg.batch_size * split_heads
        gs1_start[cur_core + 1] = tmp_core_spos_end

        used_core = cur_core + 1
        used_core_weights = [0 for _ in range(used_core)]
        for row in per_row:
            used_core_weights[row["core"]] += row["actual_inner_block_nums"]
        candidate_core_weights = used_core_weights + [0] * max(0, self.aic_num - used_core)
        load_balance = build_load_balance(candidate_core_weights, self.aic_num, core_weight_target, used_core)

        core_ranges = []
        for core in range(used_core):
            core_ranges.append(
                {
                    "core": core,
                    "aiv_core_pair": [core * 2, core * 2 + 1],
                    "bn_start": bn_start[core],
                    "bn_end": bn_start[core + 1],
                    "s_outer_start": gs1_start[core],
                    "s_outer_end_marker": gs1_start[core + 1],
                    "task_blocks": used_core_weights[core],
                    "load_ratio_to_mean": (
                        used_core_weights[core] / load_balance["meanTaskBlocks"]
                        if load_balance["meanTaskBlocks"] > 0
                        else 0
                    ),
                }
            )

        return {
            "used_core_num": used_core,
            "cubeUsedCores": used_core,
            "actualCoreNums": used_core * 2,
            "bn_start_idx": bn_start[: used_core + 1],
            "sparse_start_idx_gs1": gs1_start[: used_core + 1],
            "coreTaskBlocks": used_core_weights,
            "candidateCoreTaskBlocks": candidate_core_weights,
            "loadBalance": load_balance,
            "core_ranges": core_ranges,
            "per_row_assignment": per_row,
        }

    def is_flash_decode_farun(self, blocks: BlockSizes) -> bool:
        if self.cfg.force_flash_decode is not None:
            return self.cfg.force_flash_decode
        if self.cfg.actual_shared_prefix_len > 0:
            return False
        if self.kv_seq < blocks.s_inner_size * 2:
            return False
        bng = self.cfg.batch_size * self.cfg.kv_heads * ceil_div(self.group, blocks.s_outer_size)
        if bng < 0.4 * self.aic_num and self.group == 1:
            return True
        if bng < 0.4 * self.aic_num and self.max_actualseq >= 2048:
            return True
        return False

    def split_s2_info(self, blocks: BlockSizes, flash_decode: bool) -> FlashDecodeInfo:
        bng = self.cfg.batch_size * self.cfg.kv_heads * ceil_div(self.group, blocks.s_outer_size)
        kv_split_limit = NUM256 if blocks.s_inner_size <= NUM256 else blocks.s_inner_size
        initial = 1
        split = 1
        if flash_decode:
            if bng <= 0:
                raise ValueError("bng should be positive")
            initial = max(1, self.aic_num // bng)
            split = initial
            while split > 1 and (self.max_actualseq // split) < kv_split_limit:
                split -= 1
        s_inner_loop_size = ceil_div(self.max_actualseq, split)
        head_dim_align = align_up(self.cfg.head_dim, BYTE_BLOCK)
        accum = self.cfg.batch_size * self.cfg.q_heads * split * head_dim_align
        log_sum = self.cfg.batch_size * self.cfg.q_heads * split * (BYTE_BLOCK // self.cfg.block_type_size)
        workspace = (accum + 2 * log_sum) * self.cfg.block_type_size if flash_decode else 0
        return FlashDecodeInfo(flash_decode, bng, kv_split_limit, initial, split, s_inner_loop_size, accum, log_sum, workspace)

    def run(self) -> Dict[str, Any]:
        blocks = self.calc_inner_size()
        softmax = self.softmax_info(blocks)

        prefix_inner_blocks = ceil_div(self.cfg.actual_shared_prefix_len, blocks.s_inner_size)
        s_outer_loops = []
        s_inner_loops = []
        total_block_nums_one_head = 0
        batch_rows = []

        for b_idx in range(self.cfg.batch_size):
            actual_q = self.actual_q_lens[b_idx] * (self.group if self.cfg.fa_run_gs else 1)
            actual_kv = self.actual_kv_lens[b_idx]
            pre, nxt = self.get_pre_next_tokens_left_up(actual_q, actual_kv + self.cfg.actual_shared_prefix_len)
            fixed_q, pre, nxt = self.fix_param_with_row_invalid(actual_q, actual_kv + self.cfg.actual_shared_prefix_len, pre, nxt)
            outer = ceil_div(fixed_q, blocks.s_outer_size)
            inner = ceil_div(actual_kv, blocks.s_inner_size)
            s_outer_loops.append(outer)
            s_inner_loops.append(inner)
            block_nums = self.calc_block_nums_one_head(blocks, outer, inner, prefix_inner_blocks, pre, nxt)
            total_block_nums_one_head += block_nums
            batch_rows.append(
                {
                    "batch": b_idx,
                    "actual_q": actual_q,
                    "actual_kv": actual_kv,
                    "fixed_actual_q_for_row_invalid": fixed_q,
                    "s_outer_loop_times": outer,
                    "s_inner_loop_times": inner,
                    "pre_tokens_left_up": pre,
                    "next_tokens_left_up": nxt,
                    "calc_block_nums_one_head": block_nums,
                }
            )

        split_heads = self.cfg.kv_heads if self.cfg.fa_run_gs else self.cfg.q_heads
        core_weight_target = (float(total_block_nums_one_head * split_heads) / float(self.aic_num)) if self.aic_num else 0.0
        split_core = self.compute_split_nb_seq_farun(blocks, s_outer_loops, s_inner_loops, prefix_inner_blocks, core_weight_target)
        flash_decode = self.is_flash_decode_farun(blocks)
        fd = self.split_s2_info(blocks, flash_decode and not self.cfg.is_pfa)

        s1_outer_size = ceil_div(self.cfg.q_seq, blocks.s_outer_size)
        total_size_for_multicore = 0
        sinner_block_num = ceil_div(max(self.kv_seq, 1), max(blocks.s_inner_size, 1))
        if sinner_block_num:
            total_size_for_multicore = (total_block_nums_one_head // sinner_block_num) * split_heads
        split_factor_size = ceil_div(total_size_for_multicore, split_core["used_core_num"])

        return {
            "normalizedInput": {
                **asdict(self.cfg),
                "nNumOfQInOneGroup": self.group,
                "is_gqa": self.is_gqa,
                "kv_seq_used": self.kv_seq,
                "max_actualseq": self.max_actualseq,
                "aic_num": self.aic_num,
                "aiv_num": self.aiv_num,
                "core_num": self.core_num,
            },
            "tiling": {
                "Souter": blocks.s_outer_size,
                "Sinner": blocks.s_inner_size,
                "SinnerForTilingKey": blocks.s_inner_size_for_tiling_key,
                "sInnerLoopTimes": blocks.s_inner_loop_times,
                "sInnerSizeTail": blocks.s_inner_size_tail,
                "sInnerSizeAlign": blocks.s_inner_size_align,
                "SoftmaxSouter": softmax.tmp_shape_rows,
                "SoftmaxSinnerAlign": softmax.tmp_shape_cols,
                "splitS2": fd.split_s2,
                "s1OuterSize": s1_outer_size,
                "kernelPath": "faRun v2 normal/anti-quant regbase path; FD uses splitS2 workspace when enabled",
            },
            "softmax": asdict(softmax),
            "batchLoopInfo": batch_rows,
            "splitCore": {
                "split_core_mode": "SPLIT_NBS_CUBE",
                "split_heads": split_heads,
                "candidate_cube_cores": self.aic_num,
                "prefix_inner_loop_times": prefix_inner_blocks,
                "total_block_nums_one_head": total_block_nums_one_head,
                "core_weight_target": core_weight_target,
                "multiCoreParamsRegbase": {
                    "coreNum": split_core["used_core_num"],
                    "totalSize": total_size_for_multicore,
                    "s1OuterSize": s1_outer_size,
                    "splitFactorSize": split_factor_size,
                    "splitFactorTailSize": calc_tail_size(total_size_for_multicore, split_factor_size),
                    "bnStartIdx": split_core["bn_start_idx"],
                    "sparseStartIdx": split_core["sparse_start_idx_gs1"],
                },
                **split_core,
            },
            "flashDecodeSplitS2": asdict(fd),
            "notes": [
                "This is a host-side analysis helper for incre_flash_attention_tiling_v2.cpp; it does not call CANN tiling APIs.",
                "Task blocks are the effective inner-block weights used by the N-B-S split-core estimator.",
                "Softmax tmp size is reported as the source API formula because GetSoftMaxFlashV2MinTmpSize is implemented in AscendC.",
            ],
        }


def build_load_balance(
    core_weights: List[int], candidate_cube_cores: int, target: float, used_cube_cores: Optional[int] = None
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
        "Task blocks are the same effective inner-block weights used by the IFA v2 split-core estimator.",
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
    title = "IFA v2 split-core task blocks per cube core"
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
        label = "" if len(weights) > 36 and idx % 2 == 1 else str(idx)
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{margin_top + plot_h + 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#475569">{label}</text>')

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

    parts.append(f'<text x="{margin_left}" y="{height - 18}" font-family="Arial, sans-serif" font-size="12" fill="#64748b">x-axis: cube core id; each cube core maps to two AIV cores in the regbase SPLIT_NBS_CUBE path.</text>')
    parts.append("</svg>")

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def example_input() -> Dict[str, Any]:
    return {
        "shape": {
            "batch_size": 2,
            "q_heads": 32,
            "kv_heads": 4,
            "q_seq": 1,
            "kv_seq": 4096,
            "head_dim": 64,
        },
        "attrs": {
            "input_dtype": "fp16",
            "block_type_size": 2,
            "sparse_mode": 0,
            "pre_tokens": 0,
            "next_tokens": 0,
            "actual_q_lens": [1, 1],
            "actual_kv_lens": [4096, 3072],
            "actual_shared_prefix_len": 0,
        },
        "platform": {
            "aic_num": 24,
            "aiv_num": 48,
            "core_num": 48,
        },
        "flags": {
            "enable_pse_shift": False,
            "is_gqa": True,
            "fa_run_gs": False,
            "enable_mask": False,
            "is_pfa": False,
            "force_flash_decode": None,
        },
    }


def load_input(path: Optional[str]) -> Dict[str, Any]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raw = sys.stdin.read()
    if not raw.strip():
        raise SystemExit("No input provided. Use --example to print a sample JSON.")
    return json.loads(raw)


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate IFA tiling v2 block sizes and core split.")
    parser.add_argument("-i", "--input", help="Input JSON file. If omitted, reads JSON from stdin.")
    parser.add_argument("--config", help="Legacy alias of --input. CLI args override values in the file.")
    parser.add_argument("--example", action="store_true", help="Print an example input JSON and exit.")
    parser.add_argument("--compact", action="store_true", help="Print compact JSON.")
    parser.add_argument("--plot", help="Write an SVG bar chart of per-cube-core task blocks to this path.")
    parser.add_argument("--summary", action="store_true", help="Print the old human-readable summary instead of JSON.")
    parser.add_argument("--json", action="store_true", help="Legacy no-op: JSON is the default output.")

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--q-heads", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--q-seq", type=int)
    parser.add_argument("--kv-seq", type=int)
    parser.add_argument("--aic-num", type=int)
    parser.add_argument("--aiv-num", type=int)
    parser.add_argument("--core-num", type=int)
    parser.add_argument("--block-type-size", type=int)

    parser.add_argument("--pse-shift", action="store_true")
    parser.add_argument("--is-gqa", action="store_true")
    parser.add_argument("--no-is-gqa", action="store_true")
    parser.add_argument("--fa-run-gs", action="store_true")
    parser.add_argument("--is-pfa", action="store_true")
    parser.add_argument("--atten-mask", action="store_true")
    parser.add_argument("--sparse-mode", type=int)
    parser.add_argument("--pre-token", type=int)
    parser.add_argument("--next-token", type=int)
    parser.add_argument("--actual-shared-prefix-len", type=int)
    parser.add_argument("--actual-q-lens", help="Comma separated list, e.g. 1,1,1")
    parser.add_argument("--actual-kv-lens", help="Comma separated list, e.g. 2048,1024")
    parser.add_argument("--max-actualseq", type=int)
    parser.add_argument("--force-flash-decode", action="store_true")
    parser.add_argument("--disable-flash-decode", action="store_true")
    return parser


def merge_args(args: argparse.Namespace) -> Dict[str, Any]:
    data: Dict[str, Any]
    input_path = args.input or args.config
    if input_path:
        data = load_input(input_path)
    else:
        has_cli_shape = any(
            getattr(args, key, None) is not None for key in ("batch_size", "q_heads", "kv_heads", "head_dim")
        )
        data = {} if has_cli_shape else load_input(None)

    if "shape" in data or "attrs" in data or "platform" in data or "flags" in data:
        flat_data: Dict[str, Any] = {}
        flat_data.update(data.get("shape", {}))
        flat_data.update(data.get("attrs", {}))
        flat_data.update(data.get("platform", {}))
        flat_data.update(data.get("flags", {}))
        data = flat_data

    for key in (
        "batch_size",
        "q_heads",
        "kv_heads",
        "head_dim",
        "q_seq",
        "kv_seq",
        "aic_num",
        "aiv_num",
        "core_num",
        "block_type_size",
        "sparse_mode",
        "pre_token",
        "next_token",
        "actual_shared_prefix_len",
        "max_actualseq",
    ):
        val = getattr(args, key, None)
        if val is not None:
            data[key] = val

    for key in ("pse_shift", "fa_run_gs", "is_pfa", "atten_mask"):
        if getattr(args, key, False):
            data[key] = True

    if args.is_gqa:
        data["is_gqa"] = True
    if args.no_is_gqa:
        data["is_gqa"] = False
    if args.force_flash_decode:
        data["force_flash_decode"] = True
    if args.disable_flash_decode:
        data["force_flash_decode"] = False

    actual_q_lens = parse_int_list(args.actual_q_lens)
    actual_kv_lens = parse_int_list(args.actual_kv_lens)
    if actual_q_lens is not None:
        data["actual_q_lens"] = actual_q_lens
    if actual_kv_lens is not None:
        data["actual_kv_lens"] = actual_kv_lens
    return data


def print_summary(result: Dict[str, Any]) -> None:
    blocks = result["tiling"]
    softmax = result["softmax"]
    split_core = result["splitCore"]
    fd = result["flashDecodeSplitS2"]

    print("== IFA tiling v2 simulation ==")
    print(f"G={result['normalizedInput']['nNumOfQInOneGroup']} is_gqa={result['normalizedInput']['is_gqa']}")
    print()
    print("[Block sizes]")
    print(f"Souter/sOuterSize_              : {blocks['Souter']}")
    print(f"Sinner/sInnerSize_              : {blocks['Sinner']}")
    print(f"Sinner for tiling key           : {blocks['SinnerForTilingKey']}")
    print(f"sInnerLoopTimes_                : {blocks['sInnerLoopTimes']}")
    print(f"sInnerSizeTail_                 : {blocks['sInnerSizeTail']}")
    print(f"sInnerSizeAlign_                : {blocks['sInnerSizeAlign']}")
    print()
    print("[Softmax]")
    print(f"tmp shape                       : {{{softmax['tmp_shape_rows']}, {softmax['tmp_shape_cols']}}}")
    print(f"tmp size formula                : {softmax['tmp_size_formula']}")
    print(f"arch35 regbase fixed UB bytes   : {softmax['arch35_regbase_ub_bytes']}")
    print(f"fill tiling shape               : {{{softmax['fill_tiling_shape_rows']}, {softmax['fill_tiling_shape_cols']}}}")
    print()
    print("[Core split]")
    print(f"split heads                     : {split_core['split_heads']}")
    print(f"totalBlockNumsOneHead           : {split_core['total_block_nums_one_head']}")
    print(f"coreWeightTarget                : {split_core['core_weight_target']:.4f}")
    print(f"used core num                   : {split_core['used_core_num']}")
    print(f"bnStartIdx                      : {split_core['bn_start_idx']}")
    print(f"sparseStartIdx(gS1StartIdx)     : {split_core['sparse_start_idx_gs1']}")
    print()
    print("[splitS2 / flash decode]")
    print(f"enabled                         : {fd['enabled']}")
    print(f"bng                             : {fd['bng']}")
    print(f"kvSplitLimit                    : {fd['kv_split_limit']}")
    print(f"initial splitS2                 : {fd['initial_split_s2']}")
    print(f"final splitS2                   : {fd['split_s2']}")
    print(f"sInnerLoopSize per split        : {fd['s_inner_loop_size_per_split']}")
    print(f"FD accumOut elems               : {fd['accum_out_size_elems']}")
    print(f"FD logSumExp elems per buffer   : {fd['log_sum_exp_size_elems_one_buffer']}")
    print(f"FD workspace bytes              : {fd['fd_workspace_bytes']}")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.example:
        print(json.dumps(example_input(), indent=2, ensure_ascii=False))
        return 0

    if args.input or args.config or not any(
        getattr(args, key, None) is not None for key in ("batch_size", "q_heads", "kv_heads", "head_dim")
    ):
        raw_data = load_input(args.input or args.config)
        cfg_data = merge_args(args) if any(vars(args).values()) else raw_data
        cfg = IFATilingInput.from_dict(cfg_data)
    else:
        data = merge_args(args)
        required = ["batch_size", "q_heads", "kv_heads", "head_dim"]
        missing = [key for key in required if key not in data]
        if missing:
            parser.error(f"missing required fields: {', '.join(missing)}")
        cfg = IFATilingInput(**data)

    result = IFATilingV2Simulator(cfg).run()

    if args.plot:
        render_load_balance_svg(result, args.plot)
        result["visualization"] = {"loadBalanceSvg": os.path.abspath(args.plot)}

    if args.summary:
        print_summary(result)
    elif args.compact:
        print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
