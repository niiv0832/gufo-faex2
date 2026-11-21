#!/usr/bin/env python3
"""Checks tools/prof/vision_resync.py on synthetic and real captured data."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "prof" / "vision_resync.py"
SPEC = importlib.util.spec_from_file_location("vision_resync", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the vision_resync module")
vr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vr)
FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)


def step(step, cached, cold, margin=1.0, delta=0.0, kl=0.0):
    return {"step": step, "cached_token": cached, "cold_token": cold,
            "cached_margin": margin, "cold_margin": margin,
            "max_delta": delta, "kl": kl}


def test_identical_streams():
    steps = [step(i, 10 + i, 10 + i) for i in range(6)]
    result = vr.evaluate(steps)
    check(result["verdicts"] == {"kl": "PASS", "max_delta": "PASS",
                                 "resync": "PASS"}, "identical streams pass")
    check(result["levenshtein"] == 0, "identical streams have zero distance")


def test_single_flip_resync():
    # The captured 2026-11-19 shape: one flipped argmax, then the streams
    # agree again with a shift of one; the cold side carries one trailing
    # token cut off by the capture window.
    cached = [40, 5312, 2438, 11, 430, 279, 4307, 2099, 369, 264, 6235, 11,
              13398, 2002, 314, 2438]
    cold = [40, 5312, 2438, 1892, 279, 4307, 2099, 369, 264, 6235, 11, 13398,
            2002, 314, 2438, 440]
    steps = [step(i, a, b, margin=0.03 if i == 3 else 1.0)
             for i, (a, b) in enumerate(zip(cached, cold))]
    result = vr.evaluate(steps)
    check(result["first_divergence_step"] == 3, "divergence at step 3")
    check(result["common_prefix_steps"] == 4,
          "common prefix includes step 3 where input context is still identical")
    check(result["resync"]["shift"] == -1, "streams settle into a shift of 1")
    check(result["resync"]["cached_resync_index"] == 5,
          "cached realigns at step 5")
    check(result["resync"]["cold_resync_index"] == 4,
          "cold realigns at step 4")
    check(result["resync"]["cached_unaligned"] == 2,
          "two cached tokens unaligned (11, 430)")
    check(result["resync"]["cold_unaligned"] == 1,
          "one cold token unaligned (1892)")
    check(result["resync"]["edit_distance_without_tail"] == 2,
          "edit distance excluding window tail is 2")
    check(result["resync"]["cold_tail"] == 1, "trailing token reported as tail")
    check(result["levenshtein"] == 3, "windowed levenshtein is 3")
    check(result["verdicts"]["resync"] == "PASS", "resync within the limit")
    check(result["verdicts"]["near_tie"] == "PASS", "flip at a near-tie")


def test_negative_shift_does_not_look_before_divergence():
    # Even if cached[divergence] equals cold[divergence - 1], resync must not
    # produce negative unaligned counts by matching before the divergence.
    cached = [10, 20, 30, 30, 40, 50, 60]
    cold = [10, 20, 30, 40, 50, 60, 70]
    steps = [step(i, a, b, margin=0.02 if i == 3 else 1.0)
             for i, (a, b) in enumerate(zip(cached, cold))]
    result = vr.evaluate(steps)
    check(result["resync"]["cached_unaligned"] == 1,
          "one extra inserted token on cached side")
    check(result["resync"]["cold_unaligned"] == 0,
          "zero unaligned tokens on cold side (never negative)")
    check(result["resync"]["edit_distance_without_tail"] == 1,
          "single insertion has edit distance 1")


def test_cascade_fails():
    cached = [1, 2, 9, 9, 9, 9]
    cold = [1, 2, 8, 8, 8, 8]
    steps = [step(i, a, b, margin=0.01) for i, (a, b)
             in enumerate(zip(cached, cold))]
    result = vr.evaluate(steps)
    check(result["verdicts"]["resync"] == "FAIL", "cascade fails the resync gate")
    check("note" in result["resync"], "cascade reports no realignment")


def test_thresholds():
    # A violation at the divergence step itself (where both inputs still match)
    # must fail the KL and max_delta gates.
    steps = [step(0, 7, 7, delta=0.1, kl=0.001),
             step(1, 8, 9, margin=0.2, delta=1.5, kl=0.02)]
    result = vr.evaluate(steps)
    check(result["common_prefix_steps"] == 2,
          "divergence step is included in common prefix")
    check(result["verdicts"]["kl"] == "FAIL", "KL above the limit fails")
    check(result["verdicts"]["max_delta"] == "FAIL", "maxΔ above the limit fails")
    check(result["verdicts"]["near_tie"] == "FAIL",
          "flip outside the near-tie band fails")


def test_log_parsing():
    text = (
        "noise before\n"
        "vision-diag step=0 cached_token=40 cold_token=40 cached_margin=0.918016 "
        "cold_margin=0.809122 max_delta=0.513805 kl=0.001420\n"
        "interleaved loader lines\n"
        "vision-diag step=1 cached_token=11 cold_token=1892 "
        "cached_margin=0.030472 cold_margin=0.070221 max_delta=0.391665 "
        "kl=0.001826\n"
        "noise after\n")
    steps = vr.parse_steps(text)
    check(len(steps) == 2, "two step lines parsed")
    check(steps[1]["cold_token"] == 1892, "cold token parsed")
    check(steps[0]["max_delta"] == 0.513805, "max delta parsed")
    result = vr.evaluate(steps)
    check(result["common_prefix_steps"] == 2,
          "common prefix covers steps 0 and 1 (same input prefix)")
    check(abs(result["common_max_kl"] - 0.001826) < 1e-12,
          "KL aggregate includes the divergence step")


def main():
    test_identical_streams()
    test_single_flip_resync()
    test_negative_shift_does_not_look_before_divergence()
    test_cascade_fails()
    test_thresholds()
    test_log_parsing()
    for failure in FAILURES:
        print("FAIL:", failure)
    if not FAILURES:
        print("vision_resync_test: PASS")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
