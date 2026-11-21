#!/usr/bin/env python3
"""Offline analysis of the GUFO_QWEN_MOE_SKEW=1 router telemetry (W0.4).

The engine appends one JSON record per request to the telemetry file: per
layer, per row-count bucket (1-4 = the MTP verify widths, 5-8, more than 8)
it carries the selected-expert histogram, the routing weight per selection
slot, token and distinct-expert counts, 7th/8th <1% tail counts, the experts
shared between the rows of one call and the overlap of a token with its
predecessor. This script aggregates the records and computes what the Wave 2
decisions need:

  * per-layer Gini and hot mass of the router (W2.2 / W2.6 "hot plate");
  * pairwise duplicate share and batch deduplication saving
    `1 - unique_experts / (tokens * used)` in batched calls (W2.2);
  * the neighbor-token route overlap (W3.5 SpecFit);
  * the routing mass and <1% token share of the 7th/8th expert (W5.3).

Usage:
  tools/prof/moe_skew.py DUMP.jsonl [MORE.jsonl ...] [--json OUT]

Records may come from several files (C1 and C4 passes kept separately); pass
them together to compare, the per-file breakdown is always printed.
"""

from __future__ import annotations

import argparse
import json
import sys

BUCKET_LABELS = ("rows=1", "rows=2", "rows=3", "rows=4", "rows=5-8",
                 "rows>8")


def gini(counts: list[int]) -> float:
    """Gini coefficient of the selection counts; 0 = uniform router."""
    values = sorted(counts)
    total = sum(values)
    if total == 0 or not values:
        return 0.0
    cumulative = 0.0
    weighted = 0.0
    for value in values:
        cumulative += value
        weighted += cumulative
    return 1.0 - 2.0 * (weighted / (len(values) * total)) + 1.0 / len(values)


def hot_mass(counts: list[int], fraction: float = 0.1) -> float:
    """Share of selections carried by the hottest `fraction` of experts."""
    values = sorted(counts, reverse=True)
    total = sum(values)
    if total == 0 or not values:
        return 0.0
    top = max(1, int(round(len(values) * fraction)))
    return sum(values[:top]) / total


def hot_set_size(counts: list[int], coverage: float = 0.8) -> int:
    """Smallest number of hottest experts covering `coverage` of selections."""
    values = sorted(counts, reverse=True)
    total = sum(values)
    if total == 0:
        return 0
    cumulative = 0
    for index, value in enumerate(values, 1):
        cumulative += value
        if cumulative >= coverage * total:
            return index
    return len(values)


def load_records(paths: list[str]) -> list[dict]:
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if not records:
        raise SystemExit("no telemetry records found")
    return records


def aggregate(records: list[dict]) -> dict:
    """Sum histograms and counters of equal-shape records per layer/bucket."""
    layers = records[0]["layers"]
    buckets = len(records[0]["rows"][0])
    experts = records[0]["experts"]
    used = records[0]["used"]
    for record in records:
        if (record["layers"] != layers or record["experts"] != experts
                or record["used"] != used):
            raise SystemExit("records with different shapes cannot be mixed")
    hist = [[[0] * experts for _ in range(buckets)] for _ in range(layers)]
    slot_weights = [[[0.0] * used for _ in range(buckets)]
                    for _ in range(layers)]
    rows = [[0] * buckets for _ in range(layers)]
    tokens = [[0] * buckets for _ in range(layers)]
    unique_experts = [[0] * buckets for _ in range(layers)]
    tail_lt1pct = [[[0, 0] for _ in range(buckets)] for _ in range(layers)]
    dup = [[0] * buckets for _ in range(layers)]
    dup_pairs = [[0] * buckets for _ in range(layers)]
    overlap = [0] * layers
    overlap_rows = [0] * layers
    for record in records:
        rec_tokens = record.get("tokens")
        rec_unique = record.get("unique_experts")
        rec_tail = record.get("tail_lt1pct")
        for layer in range(layers):
            for bucket in range(buckets):
                rows[layer][bucket] += record["rows"][layer][bucket]
                if rec_tokens is not None:
                    tokens[layer][bucket] += rec_tokens[layer][bucket]
                if rec_unique is not None:
                    unique_experts[layer][bucket] += rec_unique[layer][bucket]
                if rec_tail is not None:
                    tail_lt1pct[layer][bucket][0] += rec_tail[layer][bucket][0]
                    tail_lt1pct[layer][bucket][1] += rec_tail[layer][bucket][1]
                dup[layer][bucket] += record["dup"][layer][bucket]
                dup_pairs[layer][bucket] += record["dup_pairs"][layer][bucket]
                for expert in range(experts):
                    hist[layer][bucket][expert] += \
                        record["hist"][layer][bucket][expert]
                for slot in range(used):
                    slot_weights[layer][bucket][slot] += \
                        record["slot_weights"][layer][bucket][slot]
            overlap[layer] += record["overlap"][layer]
            overlap_rows[layer] += record["overlap_rows"][layer]
    return {"layers": layers, "experts": experts, "used": used,
            "hist": hist, "slot_weights": slot_weights, "rows": rows,
            "tokens": tokens, "unique_experts": unique_experts,
            "tail_lt1pct": tail_lt1pct, "dup": dup, "dup_pairs": dup_pairs,
            "overlap": overlap, "overlap_rows": overlap_rows}


def summarize(agg: dict) -> dict:
    """The Wave 2 decision metrics per bucket, aggregated over layers."""
    layers, buckets, used = agg["layers"], len(agg["rows"][0]), agg["used"]
    per_bucket = []
    for bucket in range(buckets):
        total_rows = sum(agg["rows"][layer][bucket]
                         for layer in range(layers))
        total_tokens = sum(agg["tokens"][layer][bucket]
                           for layer in range(layers))
        total_unique = sum(agg["unique_experts"][layer][bucket]
                           for layer in range(layers))
        total_tail7 = sum(agg["tail_lt1pct"][layer][bucket][0]
                          for layer in range(layers))
        total_tail8 = sum(agg["tail_lt1pct"][layer][bucket][1]
                          for layer in range(layers))
        total_pairs = sum(agg["dup_pairs"][layer][bucket]
                          for layer in range(layers))
        total_dup = sum(agg["dup"][layer][bucket] for layer in range(layers))
        total_slots = sum(sum(agg["slot_weights"][layer][bucket])
                          for layer in range(layers))
        tail_slots = (used - 2) if used >= 2 else 0
        tail_mass = 0.0
        if total_slots > 0 and tail_slots > 0:
            tail_mass = sum(
                agg["slot_weights"][layer][bucket][slot]
                for layer in range(layers)
                for slot in range(tail_slots, used)) / total_slots
        active_hists = [
            agg["hist"][layer][bucket] for layer in range(layers)
            if sum(agg["hist"][layer][bucket]) > 0
        ]
        ginis = [gini(counts) for counts in active_hists]
        hots = [hot_mass(counts) for counts in active_hists]
        hot_sets = [hot_set_size(counts) for counts in active_hists]
        dedup_saving = (
            1.0 - (total_unique / (total_tokens * used))
            if total_tokens > 0 and used > 0 and total_unique > 0 else 0.0)
        per_bucket.append({
            "bucket": BUCKET_LABELS[bucket],
            "calls": total_rows,
            "tokens": total_tokens,
            "mean_gini": sum(ginis) / len(ginis) if ginis else 0.0,
            "min_gini": min(ginis) if ginis else 0.0,
            "max_gini": max(ginis) if ginis else 0.0,
            "mean_hot_mass_decile": sum(hots) / len(hots) if hots else 0.0,
            "mean_hot_set_80": (sum(hot_sets) / len(hot_sets))
                               if hot_sets else 0.0,
            "duplicate_share": (total_dup / (total_pairs * used))
                               if total_pairs else 0.0,
            "dedup_saving": dedup_saving,
            "tail_slots_7_8_mass": tail_mass,
            "tail7_lt1pct_share": (total_tail7 / total_tokens)
                                  if total_tokens else 0.0,
            "tail8_lt1pct_share": (total_tail8 / total_tokens)
                                  if total_tokens else 0.0,
        })
    total_single = sum(agg["overlap_rows"])
    overlap_share = (sum(agg["overlap"]) / (total_single * used)
                     if total_single else 0.0)
    return {"per_bucket": per_bucket, "neighbor_overlap_single": overlap_share}


def format_report(per_file: dict[str, dict], combined: dict) -> str:
    lines = []
    for label, summary in list(per_file.items()) + [("<combined>", combined)]:
        lines.append(f"=== {label} ===")
        header = (f"{'bucket':<8} {'calls':>9} {'tokens':>9} {'gini':>7} "
                  f"{'hot10%':>7} {'hot80%':>7} {'dup':>7} {'dedup':>7} "
                  f"{'t7+8w':>7} {'t7<1%':>7} {'t8<1%':>7}")
        lines.append(header)
        for bucket in summary["per_bucket"]:
            lines.append(
                f"{bucket['bucket']:<8} {bucket['calls']:>9} "
                f"{bucket['tokens']:>9} "
                f"{bucket['mean_gini']:>7.4f} "
                f"{bucket['mean_hot_mass_decile']:>7.4f} "
                f"{bucket['mean_hot_set_80']:>7.1f} "
                f"{bucket['duplicate_share']:>7.4f} "
                f"{bucket['dedup_saving']:>7.4f} "
                f"{bucket['tail_slots_7_8_mass']:>7.4f} "
                f"{bucket['tail7_lt1pct_share']:>7.4f} "
                f"{bucket['tail8_lt1pct_share']:>7.4f}")
        lines.append(
            f"neighbor overlap (single-row calls): "
            f"{summary['neighbor_overlap_single']:.4f}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dumps", nargs="+", help="telemetry JSONL files")
    parser.add_argument("--json", help="write the combined summary as JSON")
    args = parser.parse_args(argv)
    per_file = {}
    for path in args.dumps:
        per_file[path] = summarize(aggregate(load_records([path])))
    combined = summarize(aggregate(load_records(args.dumps)))
    print(format_report(per_file, combined))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"files": per_file, "combined": combined}, handle,
                      indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
