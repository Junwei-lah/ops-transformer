#!/usr/bin/env python3
# Compare C++ IFA_TILING_DUMP logs with ifa_tiling_sim.py JSON output.

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


SUMMARY_TAG = "[IFA_TILING_DUMP][summary]"
CORE_RANGE_TAG = "[IFA_TILING_DUMP][coreRange]"
PAIR_RE = re.compile(r"(\w+)=\[(-?\d+),(-?\d+)\]")
KV_RE = re.compile(r"([\w.]+)=(-?\d+(?:\.\d+)?)")


@dataclass
class Diff:
    path: str
    cpp: Any
    py: Any
    status: str
    detail: str = ""


def parse_value(raw: str) -> Any:
    if "." in raw:
        try:
            return float(raw)
        except ValueError:
            return raw
    try:
        return int(raw)
    except ValueError:
        return raw


def parse_kv_line(line: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, first, second in PAIR_RE.findall(line):
        out[key] = [int(first), int(second)]
    line_without_pairs = PAIR_RE.sub("", line)
    for key, value in KV_RE.findall(line_without_pairs):
        out[key] = parse_value(value)
    return out


def parse_cpp_dump(path: str) -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    current_core_ranges: List[Dict[str, Any]] = []
    last_core_ranges: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if SUMMARY_TAG in line:
                summaries.append(parse_kv_line(line))
                last_core_ranges = current_core_ranges
                current_core_ranges = []
            elif CORE_RANGE_TAG in line:
                current_core_ranges.append(parse_kv_line(line))

    if not summaries:
        raise SystemExit(f"No {SUMMARY_TAG} line found in {path}")
    if not last_core_ranges and len(summaries) == 1 and current_core_ranges:
        last_core_ranges = current_core_ranges
    if not last_core_ranges:
        raise SystemExit(f"No {CORE_RANGE_TAG} line found in {path}")

    return {
        "summary": summaries[-1],
        "coreRanges": last_core_ranges,
        "summaryCount": len(summaries),
        "coreRangeCount": len(last_core_ranges),
    }


def load_py_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_path(data: Dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur[part]
        else:
            raise KeyError(dotted)
    return cur


def values_equal(cpp: Any, py: Any, float_tol: float) -> Tuple[bool, str]:
    if isinstance(cpp, float) or isinstance(py, float):
        try:
            cpp_f = float(cpp)
            py_f = float(py)
        except (TypeError, ValueError):
            return False, "not numeric"
        if math.isclose(cpp_f, py_f, rel_tol=float_tol, abs_tol=float_tol):
            return True, ""
        return False, f"abs diff={abs(cpp_f - py_f):.12g}"
    return cpp == py, ""


def compare_one(diffs: List[Diff], path: str, cpp: Any, py: Any, float_tol: float) -> None:
    ok, detail = values_equal(cpp, py, float_tol)
    diffs.append(Diff(path=path, cpp=cpp, py=py, status="OK" if ok else "DIFF", detail=detail))


def compare_summary(cpp_summary: Dict[str, Any], py: Dict[str, Any], float_tol: float) -> List[Diff]:
    mapping = [
        ("candidateCoreNum", "splitCore.candidate_cube_cores"),
        ("usedCoreNum", "splitCore.used_core_num"),
        ("actualCoreNums", "splitCore.actualCoreNums"),
        ("batchSize", "normalizedInput.batch_size"),
        ("qHeads", "normalizedInput.q_heads"),
        ("kvHeads", "normalizedInput.kv_heads"),
        ("qSeq", "normalizedInput.q_seq"),
        ("kvSeq", "normalizedInput.kv_seq_used"),
        ("headDim", "normalizedInput.head_dim"),
        ("splitHeads", "splitCore.split_heads"),
        ("sOuterSize", "tiling.Souter"),
        ("sInnerSize", "tiling.Sinner"),
        ("s1OuterSize", "tiling.s1OuterSize"),
        ("totalBlockNumsOneHead", "splitCore.total_block_nums_one_head"),
        ("prefixInnerLoopTimes", "splitCore.prefix_inner_loop_times"),
        ("coreWeightTarget", "splitCore.core_weight_target"),
        ("multiCore.coreNum", "splitCore.multiCoreParamsRegbase.coreNum"),
        ("multiCore.totalSize", "splitCore.multiCoreParamsRegbase.totalSize"),
        ("multiCore.s1OuterSize", "splitCore.multiCoreParamsRegbase.s1OuterSize"),
        ("multiCore.splitFactorSize", "splitCore.multiCoreParamsRegbase.splitFactorSize"),
        ("multiCore.splitFactorTailSize", "splitCore.multiCoreParamsRegbase.splitFactorTailSize"),
    ]
    diffs: List[Diff] = []
    for cpp_key, py_path in mapping:
        try:
            compare_one(diffs, f"summary.{cpp_key} <-> {py_path}", cpp_summary[cpp_key], get_path(py, py_path), float_tol)
        except KeyError as exc:
            diffs.append(Diff(path=f"summary.{cpp_key} <-> {py_path}", cpp=None, py=None, status="MISSING", detail=str(exc)))
    return diffs


def normalize_cpp_core_range(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "core": item.get("cubeCore", item.get("core")),
        "aiv_core_pair": item.get("aivCorePair"),
        "bn_start": item.get("bnStartIdx"),
        "bn_end": item.get("bnEndIdx"),
        "s_outer_start": item.get("sparseStartIdx"),
        "s_outer_end_marker": item.get("sparseEndIdx"),
        "task_blocks": item.get("taskBlocks"),
    }


def compare_core_ranges(cpp_ranges: List[Dict[str, Any]], py: Dict[str, Any], float_tol: float) -> List[Diff]:
    py_ranges = py.get("splitCore", {}).get("core_ranges", [])
    diffs: List[Diff] = []
    compare_one(diffs, "core_ranges.length", len(cpp_ranges), len(py_ranges), float_tol)

    fields = [
        "core",
        "aiv_core_pair",
        "bn_start",
        "bn_end",
        "s_outer_start",
        "s_outer_end_marker",
        "task_blocks",
    ]
    for idx, cpp_item in enumerate(cpp_ranges[: len(py_ranges)]):
        cpp_norm = normalize_cpp_core_range(cpp_item)
        py_item = py_ranges[idx]
        for field in fields:
            compare_one(diffs, f"core_ranges[{idx}].{field}", cpp_norm.get(field), py_item.get(field), float_tol)
    return diffs


def compare_arrays(cpp_ranges: List[Dict[str, Any]], py: Dict[str, Any], float_tol: float) -> List[Diff]:
    cpp_norm = [normalize_cpp_core_range(item) for item in cpp_ranges]
    cpp_blocks = [item.get("task_blocks") for item in cpp_norm]
    cpp_bn_start = [item.get("bn_start") for item in cpp_norm]
    cpp_sparse_start = [item.get("s_outer_start") for item in cpp_norm]
    if cpp_norm:
        cpp_bn_start.append(cpp_norm[-1].get("bn_end"))
        cpp_sparse_start.append(cpp_norm[-1].get("s_outer_end_marker"))

    split_core = py.get("splitCore", {})
    diffs: List[Diff] = []
    compare_one(diffs, "splitCore.coreTaskBlocks", cpp_blocks, split_core.get("coreTaskBlocks"), float_tol)
    compare_one(diffs, "splitCore.bn_start_idx", cpp_bn_start, split_core.get("bn_start_idx"), float_tol)
    compare_one(diffs, "splitCore.sparse_start_idx_gs1", cpp_sparse_start, split_core.get("sparse_start_idx_gs1"), float_tol)

    candidate = split_core.get("candidateCoreTaskBlocks")
    candidate_cores = split_core.get("candidate_cube_cores")
    if candidate is not None and candidate_cores is not None:
        expected = cpp_blocks + [0] * max(0, int(candidate_cores) - len(cpp_blocks))
        compare_one(diffs, "splitCore.candidateCoreTaskBlocks", expected, candidate, float_tol)
    return diffs


def build_report(cpp: Dict[str, Any], py: Dict[str, Any], float_tol: float) -> Dict[str, Any]:
    diffs: List[Diff] = []
    diffs.extend(compare_summary(cpp["summary"], py, float_tol))
    diffs.extend(compare_core_ranges(cpp["coreRanges"], py, float_tol))
    diffs.extend(compare_arrays(cpp["coreRanges"], py, float_tol))

    status_counts: Dict[str, int] = {}
    for diff in diffs:
        status_counts[diff.status] = status_counts.get(diff.status, 0) + 1

    return {
        "ok": status_counts.get("DIFF", 0) == 0 and status_counts.get("MISSING", 0) == 0,
        "statusCounts": status_counts,
        "cpp": {
            "summaryCount": cpp["summaryCount"],
            "coreRangeCount": cpp["coreRangeCount"],
        },
        "diffs": [diff.__dict__ for diff in diffs],
    }


def print_text_report(report: Dict[str, Any], show_ok: bool) -> None:
    print("IFA tiling dump compare:", "PASS" if report["ok"] else "FAIL")
    print("statusCounts:", json.dumps(report["statusCounts"], ensure_ascii=False, sort_keys=True))
    print("cpp:", json.dumps(report["cpp"], ensure_ascii=False, sort_keys=True))
    print()

    for diff in report["diffs"]:
        if diff["status"] == "OK" and not show_ok:
            continue
        line = f"[{diff['status']}] {diff['path']}: cpp={diff['cpp']} py={diff['py']}"
        if diff["detail"]:
            line += f" ({diff['detail']})"
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare IFA C++ tiling dump logs with ifa_tiling_sim.py JSON.")
    parser.add_argument("--cpp-log", required=True, help="Log file containing [IFA_TILING_DUMP] lines.")
    parser.add_argument("--py-json", required=True, help="JSON produced by scripts/tools/ifa_tiling_sim.py.")
    parser.add_argument("--float-tol", type=float, default=1e-6, help="Tolerance for floating-point fields.")
    parser.add_argument("--show-ok", action="store_true", help="Print matching fields too.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON report.")
    args = parser.parse_args()

    cpp = parse_cpp_dump(args.cpp_log)
    py = load_py_json(args.py_json)
    report = build_report(cpp, py, args.float_tol)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_text_report(report, args.show_ok)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
