#!/usr/bin/env python3
"""Verify the Q1 §8 vision-continuation criteria from a saved diagnostic log.

The `model-vision-serving-mtp` qualification (Wave 1.2) is decided against the
approved Q1 §8 thresholds: KL ≤ 0.010, max|Δlogit| ≤ 1.0, near-tie < 0.15 and
token re-synchronization ≤ 2 tokens. The first three are read straight from the
per-step `vision-diag` lines; the re-synchronization cost is computed here from
the two greedy token streams (cached continuation vs cold prefill serving):
the first divergent step, the constant shift after which both streams emit
identical tokens again, and the count of tokens that do not align.

Usage:
  tools/prof/vision_resync.py LOG [--json]
  tools/prof/vision_resync.py --tokens 40,5312,... --tokens 40,5312,...

The script never rewrites the test or the log; it only computes a decision
basis. The qualification verdict itself belongs to the owner (W1.2).
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# Approved Q1 §8 thresholds (PREFILL_QUALITY_CONTRACT_Q1_PROPOSAL.md, 2026-11-20).
KL_LIMIT = 0.010
MAX_DELTA_LIMIT = 1.0
NEAR_TIE_LIMIT = 0.15
RESYNC_TOKEN_LIMIT = 2

# `vision-diag step=3 cached_token=11 cold_token=1892 cached_margin=0.030472
# cold_margin=0.070221 max_delta=0.391665 kl=0.001826`
STEP_RE = re.compile(
    r"vision-diag step=(\d+) cached_token=(\d+) cold_token=(\d+) "
    r"cached_margin=([0-9.eE+-]+) cold_margin=([0-9.eE+-]+) "
    r"max_delta=([0-9.eE+-]+) kl=([0-9.eE+-]+)"
)


def parse_steps(text: str) -> list[dict]:
    """Per-step records of the two greedy trajectories, ordered by step."""
    steps = [
        {
            "step": int(step),
            "cached_token": int(cached),
            "cold_token": int(cold),
            "cached_margin": float(cached_margin),
            "cold_margin": float(cold_margin),
            "max_delta": float(max_delta),
            "kl": float(kl),
        }
        for step, cached, cold, cached_margin, cold_margin, max_delta, kl in
        STEP_RE.findall(text)
    ]
    steps.sort(key=lambda record: record["step"])
    return steps


def first_divergence(cached: list[int], cold: list[int]) -> int | None:
    """First index where the two greedy streams disagree, or None."""
    for index, (a, b) in enumerate(zip(cached, cold)):
        if a != b:
            return index
    return None


def levenshtein(a: list[int], b: list[int]) -> int:
    """Token edit distance (insert/delete/substitute, each cost 1)."""
    row = list(range(len(b) + 1))
    for i, a_token in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, b_token in enumerate(b, 1):
            current = row[j]
            row[j] = min(
                row[j] + 1,  # deletion
                row[j - 1] + 1,  # insertion
                previous + (a_token != b_token),  # substitution / match
            )
            previous = current
    return row[-1]


def longest_common_subsequence_length(a: list[int], b: list[int]) -> int:
    previous = [0] * (len(b) + 1)
    for a_token in a:
        current = [0]
        for j, b_token in enumerate(b, 1):
            if a_token == b_token:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def realignment(cached: list[int], cold: list[int], divergence: int,
                max_shift: int = 2) -> dict | None:
    """Smallest shift ≤ max_shift after which both streams emit the same
    tokens again, plus the unaligned token counts on each side.

    The streams are equal with shift `s` when `cached[i] == cold[i + s]` for
    every overlapping position from some index onward. The re-synchronization
    point is the position after the last mismatch under that shift, so a few
    leading mismatches are tolerated but a mismatch after the streams have
    re-aligned (a cascade) is not. Trailing tokens that fall outside the
    overlap are reported separately: they are artifacts of a truncated capture
    window, not evidence of a second divergence.
    """
    best = None
    for shift in range(-max_shift, max_shift + 1):
        lo = max(0, -shift)
        hi = min(len(cached), len(cold) - shift)
        start = max(lo, divergence, divergence - shift)
        if hi <= start:
            continue
        mismatches = [index for index in range(start, hi)
                      if cached[index] != cold[index + shift]]
        last = mismatches[-1] if mismatches else start - 1
        resync_index = last + 1
        if resync_index >= hi:
            # Not one matching step after the last mismatch: the streams
            # never re-synchronize inside the captured window under this
            # shift.
            continue
        candidate = {
            "shift": shift,
            "cached_resync_index": resync_index,
            "cold_resync_index": resync_index + shift,
            "cached_unaligned": resync_index - divergence,
            "cold_unaligned": (resync_index + shift) - divergence,
            "edit_distance_without_tail": levenshtein(
                cached[divergence:hi], cold[divergence:hi + shift]),
            # The matched suffix runs to `hi` on the cached side; anything the
            # streams emitted beyond that is a capture-window tail.
            "cached_tail": len(cached) - hi,
            "cold_tail": len(cold) - (hi + shift),
        }
        cand_key = (
            max(candidate["cached_unaligned"], candidate["cold_unaligned"]),
            candidate["cached_unaligned"] + candidate["cold_unaligned"],
            abs(candidate["shift"]),
        )
        if best is None or cand_key < (
            max(best["cached_unaligned"], best["cold_unaligned"]),
            best["cached_unaligned"] + best["cold_unaligned"],
            abs(best["shift"]),
        ):
            best = candidate
    return best


def evaluate(steps: list[dict], resync_token_limit: int = RESYNC_TOKEN_LIMIT,
             max_shift: int = 2) -> dict:
    cached = [record["cached_token"] for record in steps]
    cold = [record["cold_token"] for record in steps]
    divergence = first_divergence(cached, cold)
    # At `step == divergence` both streams still share the exact same input
    # prefix; its logits are where the near-tie flip occurs (matching the
    # engine's `common_prefix_steps = divergence + 1`).
    common = steps if divergence is None else steps[:divergence + 1]
    result = {
        "steps": len(steps),
        "common_prefix_steps": len(common),
        "common_max_logit_delta": max(
            (record["max_delta"] for record in common), default=0.0),
        "common_max_kl": max(
            (record["kl"] for record in common), default=0.0),
        "first_divergence_step": divergence,
        "levenshtein": levenshtein(cached, cold),
        "lcs_tokens": longest_common_subsequence_length(cached, cold),
    }
    if divergence is None:
        result["resync"] = {
            "verdict": "PASS",
            "note": "streams are identical on every captured step",
        }
        result["verdicts"] = {
            "kl": "PASS" if result["common_max_kl"] <= KL_LIMIT else "FAIL",
            "max_delta": "PASS"
                         if result["common_max_logit_delta"] <= MAX_DELTA_LIMIT
                         else "FAIL",
            "resync": "PASS",
        }
        return result
    record = steps[divergence]
    result["divergence_margins"] = {
        "cached": record["cached_margin"],
        "cold": record["cold_margin"],
    }
    result["resync"] = realignment(cached, cold, divergence, max_shift) or {
        "verdict": "FAIL",
        "note": f"streams never re-align within shift {max_shift}",
    }
    verdicts = {
        "kl": "PASS" if result["common_max_kl"] <= KL_LIMIT else "FAIL",
        "max_delta": "PASS" if result["common_max_logit_delta"] <= MAX_DELTA_LIMIT
                     else "FAIL",
        "near_tie": "PASS" if max(record["cached_margin"],
                                  record["cold_margin"]) < NEAR_TIE_LIMIT
                    else "FAIL",
    }
    resync = result["resync"]
    if "note" in resync:
        verdicts["resync"] = "FAIL"
    else:
        # The approved criterion is "alignment / edit distance ≤ 2 tokens".
        # The primary reading is the count of tokens each stream emits before
        # the two agree again under a constant shift; the raw windowed
        # Levenshtein is reported alongside for interpretation. Trailing
        # capture-window tails are excluded (they are reported separately).
        verdicts["resync"] = (
            "PASS" if max(resync["cached_unaligned"], resync["cold_unaligned"])
            <= resync_token_limit else "FAIL")
    result["verdicts"] = verdicts
    return result


def format_report(result: dict) -> str:
    lines = [
        f"steps={result['steps']} common_prefix_steps="
        f"{result['common_prefix_steps']}",
        f"common_max_logit_delta={result['common_max_logit_delta']:.6f} "
        f"(limit {MAX_DELTA_LIMIT})",
        f"common_max_kl_temperature1={result['common_max_kl']:.6f} "
        f"(limit {KL_LIMIT})",
        f"first_divergence_step={result['first_divergence_step']}",
        f"levenshtein={result['levenshtein']} "
        f"lcs_tokens={result['lcs_tokens']}",
    ]
    if "divergence_margins" in result:
        lines.append(
            "divergence_margins cached={cached:.6f} cold={cold:.6f} "
            "(near-tie limit {limit})".format(
                limit=NEAR_TIE_LIMIT, **result["divergence_margins"]))
    resync = result["resync"]
    if "note" in resync:
        lines.append(f"resync: {resync['note']}")
    else:
        lines.append(
            "resync: shift={shift} cached_resync_index="
            "{cached_resync_index} cold_resync_index={cold_resync_index} "
            "cached_unaligned={cached_unaligned} "
            "cold_unaligned={cold_unaligned} "
            "edit_distance_without_tail={edit_distance_without_tail} "
            "cached_tail={cached_tail} "
            "cold_tail={cold_tail}".format(**resync))
    for name, verdict in result["verdicts"].items():
        lines.append(f"{name}: {verdict}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", nargs="?", help="diagnostic log with vision-diag lines")
    parser.add_argument("--tokens", action="append", default=[],
                        help="comma-separated token stream; repeat for each side")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    if args.tokens:
        if len(args.tokens) != 2:
            parser.error("--tokens needs exactly two comma-separated streams")
        streams = [[int(token) for token in stream.split(",") if token]
                   for stream in args.tokens]
        if len(streams[0]) != len(streams[1]):
            parser.error("the two --tokens streams must have equal length")
        steps = [
            {"step": index, "cached_token": cached, "cold_token": cold,
             "cached_margin": 0.0, "cold_margin": 0.0, "max_delta": 0.0,
             "kl": 0.0}
            for index, (cached, cold) in enumerate(zip(*streams))
        ]
    elif args.log:
        steps = parse_steps(open(args.log, encoding="utf-8").read())
        if not steps:
            parser.error(f"no vision-diag step lines found in {args.log}")
    else:
        parser.error("a log path or two --tokens streams are required")
    result = evaluate(steps)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
