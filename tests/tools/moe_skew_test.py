#!/usr/bin/env python3
"""Checks tools/prof/moe_skew.py on synthetic router telemetry."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "prof" / "moe_skew.py"
SPEC = importlib.util.spec_from_file_location("moe_skew", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the moe_skew module")
skew = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(skew)
FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)


def record(layers, experts, used, calls, bucket, hist_fn, slot_weights=None,
           tokens=None, unique_experts=0, tail_lt1pct=(0, 0),
           dup=0, dup_pairs=0, overlap=0, overlap_rows=0):
    """One telemetry record matching QwenMoeSkewHost::AppendRecord JSON schema."""
    buckets = 6
    rows = [[0] * buckets for _ in range(layers)]
    token_rows = [[0] * buckets for _ in range(layers)]
    unique_rows = [[0] * buckets for _ in range(layers)]
    tail_rows = [[[0, 0] for _ in range(buckets)] for _ in range(layers)]
    dup_rows = [[0] * buckets for _ in range(layers)]
    dup_pair_rows = [[0] * buckets for _ in range(layers)]
    default_slots = [1.0 / used] * used if slot_weights is None else slot_weights
    for layer in range(layers):
        rows[layer][bucket] = calls
        token_rows[layer][bucket] = calls if tokens is None else tokens
        unique_rows[layer][bucket] = unique_experts
        tail_rows[layer][bucket] = list(tail_lt1pct)
        dup_rows[layer][bucket] = dup
        dup_pair_rows[layer][bucket] = dup_pairs
    return {
        "ordinal": 1,
        "layers": layers,
        "experts": experts,
        "used": used,
        "rows": rows,
        "tokens": token_rows,
        "unique_experts": unique_rows,
        "tail_lt1pct": tail_rows,
        "hist": [[hist_fn(b) for b in range(buckets)] for _ in range(layers)],
        "slot_weights": [[list(default_slots) if b == bucket else [0.0] * used
                          for b in range(buckets)]
                         for _ in range(layers)],
        "dup": dup_rows,
        "dup_pairs": dup_pair_rows,
        "overlap": [overlap] * layers,
        "overlap_rows": [overlap_rows] * layers,
    }


def test_gini_and_hot_mass():
    uniform = [10] * 8
    check(skew.gini(uniform) == 0.0, "uniform router has zero Gini")
    peaked = [0, 0, 0, 0, 0, 0, 0, 80]
    check(abs(skew.gini(peaked) - 0.875) < 1e-9, "single-expert Gini")
    check(skew.hot_mass(peaked, 0.1) == 1.0, "hot decile carries everything")
    check(skew.hot_mass(uniform, 0.1) >= 0.1, "hot decile at least its share")
    check(skew.hot_set_size(peaked, 0.8) == 1, "one expert covers 80%")
    check(skew.hot_set_size(uniform, 0.8) == 7, "uniform needs 80% of experts")


def test_aggregate_and_summary():
    layers, experts, used = 2, 8, 4
    # rows=1: perfectly skewed router; rows=2: 2 tokens, 4 used each -> 8 slots,
    # 6 unique experts (2 shared -> 25% dedup saving, 50% pairwise dup).
    records = [
        record(layers, experts, used, calls=1, bucket=0,
               hist_fn=lambda bucket: [10, 0, 0, 0, 0, 0, 0, 0]
               if bucket == 0 else [0] * experts,
               slot_weights=[0.6, 0.3, 0.095, 0.005],
               tokens=1, unique_experts=4, tail_lt1pct=(0, 1),
               overlap=2, overlap_rows=1),
        record(layers, experts, used, calls=1, bucket=0,
               hist_fn=lambda bucket: [10, 0, 0, 0, 0, 0, 0, 0]
               if bucket == 0 else [0] * experts,
               slot_weights=[0.6, 0.3, 0.095, 0.005],
               tokens=1, unique_experts=4, tail_lt1pct=(0, 1),
               overlap=2, overlap_rows=1),
        record(layers, experts, used, calls=1, bucket=1,
               hist_fn=lambda bucket: [4, 4, 0, 0, 0, 0, 0, 0]
               if bucket == 1 else [0] * experts,
               slot_weights=[0.5, 0.3, 0.15, 0.05],
               tokens=2, unique_experts=6, tail_lt1pct=(0, 1),
               dup=2, dup_pairs=1),
    ]
    agg = skew.aggregate(records)
    check(agg["rows"][0][0] == 2 and agg["rows"][0][1] == 1,
          "row counters summed per bucket")
    check(agg["tokens"][0][0] == 2 and agg["tokens"][0][1] == 2,
          "token counters summed per bucket")
    check(agg["hist"][0][0][0] == 20, "histograms summed")
    check(agg["dup"][0][1] == 2 and agg["dup_pairs"][0][1] == 1,
          "duplicate counters summed")
    summary = skew.summarize(agg)
    single = summary["per_bucket"][0]
    batched = summary["per_bucket"][1]
    check(single["calls"] == 4, "single-row calls counted over both layers")
    check(single["tokens"] == 4, "single-row tokens counted over both layers")
    check(abs(single["mean_gini"] - 0.875) < 1e-9,
          "skewed router reported as such")
    check(abs(batched["duplicate_share"] - 0.5) < 1e-9,
          "two shared experts of four is a 0.5 duplicate share")
    check(abs(batched["dedup_saving"] - 0.25) < 1e-9,
          "6 unique of 8 slots is a 0.25 dedup saving")
    check(abs(single["tail_slots_7_8_mass"] - 0.1) < 1e-9,
          "last two slots carry 0.10 of total weight")
    check(abs(single["tail7_lt1pct_share"] - 0.0) < 1e-9,
          "slot n-2 never below 1%")
    check(abs(single["tail8_lt1pct_share"] - 1.0) < 1e-9,
          "slot n-1 always below 1%")
    check(abs(batched["tail8_lt1pct_share"] - 0.5) < 1e-9,
          "half of batched tokens have slot n-1 below 1%")
    check(abs(summary["neighbor_overlap_single"] - 0.5) < 1e-9,
          "two shared ids of four is a 0.5 neighbor overlap")


def test_shape_mismatch_rejected(tmp):
    good = record(2, 8, 2, calls=1, bucket=0, hist_fn=lambda bucket: [0] * 8)
    bad = record(3, 8, 2, calls=1, bucket=0, hist_fn=lambda bucket: [0] * 8)
    path_good = tmp / "good.jsonl"
    path_bad = tmp / "bad.jsonl"
    path_good.write_text(json.dumps(good) + "\n", encoding="utf-8")
    path_bad.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    try:
        skew.aggregate(skew.load_records([str(path_good), str(path_bad)]))
        check(False, "shape mismatch must be rejected")
    except SystemExit:
        pass


def test_end_to_end(tmp):
    layers, experts, used = 1, 4, 2
    record_json = json.dumps(
        record(layers, experts, used, calls=1, bucket=0,
               hist_fn=lambda bucket: [3, 1, 0, 0] if bucket == 0
               else [0] * experts,
               slot_weights=[0.995, 0.005], tokens=1, unique_experts=2,
               tail_lt1pct=(0, 1), overlap=1, overlap_rows=1))
    path = tmp / "dump.jsonl"
    path.write_text(record_json + "\n", encoding="utf-8")
    out = tmp / "summary.json"
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = skew.main([str(path), "--json", str(out)])
    check(rc == 0, "report exits zero")
    output = buffer.getvalue()
    check("rows=1" in output and "dedup" in output and "t8<1%" in output,
          "report prints the bucket table with dedup and tail columns")
    saved = json.loads(out.read_text(encoding="utf-8"))
    check(saved["combined"]["per_bucket"][0]["calls"] == 1,
          "saved summary carries the call count")
    check(saved["combined"]["per_bucket"][0]["tail8_lt1pct_share"] == 1.0,
          "saved summary carries the tail <1% share")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_gini_and_hot_mass()
        test_aggregate_and_summary()
        test_shape_mismatch_rejected(Path(tmp))
        test_end_to_end(Path(tmp))
    for failure in FAILURES:
        print("FAIL:", failure)
    if not FAILURES:
        print("moe_skew_test: PASS")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
