#!/usr/bin/env python3
"""Checks tools/bench/logits-nll-report.py on small synthetic logits."""

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

try:
    import numpy as np
except ImportError:
    print("numpy is not installed; skipping")
    sys.exit(77)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "bench" / "logits-nll-report.py"
SPEC = importlib.util.spec_from_file_location("logits_nll_report", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the report module")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)
FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)


def close(a, b, tolerance=1e-12):
    return abs(a - b) <= tolerance


def test_metrics():
    # Two tokens: reference p = (0.75, 0.25), candidate q = (0.5, 0.5).
    reference = np.array([[math.log(3.0), 0.0]], dtype=np.float32)
    candidate = np.array([[0.0, 0.0]], dtype=np.float32)
    m = report.document_metrics(reference, candidate, np.array([0]))
    kl = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    check(close(m["kl_mean"], kl, 1e-7), "Bernoulli KL")
    check(close(m["delta_nll"], math.log(0.75 / 0.5), 1e-7), "Bernoulli ΔNLL")
    check(m["top1_agree"] == 1 and m["top1_gold_reference"] == 1,
          "argmax takes the first of equal logits")
    # A common shift of a row changes its bits, not its distribution.
    # (The shift is exact in float32 for these values.)
    exact = np.array([[1.5, 0.25]], dtype=np.float32)
    shifted = report.document_metrics(exact, exact + np.float32(1024.0),
                                      np.array([1]))
    check(close(shifted["kl_mean"], 0.0, 1e-9) and
          close(shifted["delta_nll"], 0.0, 1e-9), "shift invariance")
    check(shifted["bitwise_equal_rows"] == 0, "shifted rows are not bitwise")
    same = report.document_metrics(reference, reference.copy(), np.array([1]))
    check(same["bitwise_equal_rows"] == 1 and same["kl_mean"] == 0.0,
          "identical rows")


def test_bootstrap_and_verdicts():
    worse = report.paired_bootstrap([0.01, 0.02, 0.03, 0.02], 10000, 7)
    check(close(worse[0], 0.02) and worse[1] > 0.0, "all documents worse")
    check(report.verdict(worse[1], worse[2]) == report.WORSE, "worse verdict")
    better = report.paired_bootstrap([-0.01, -0.02, -0.03, -0.02], 10000, 7)
    check(report.verdict(better[1], better[2]) == report.BETTER,
          "better verdict")
    mixed = report.paired_bootstrap([0.05, -0.04, 0.01, -0.02], 10000, 7)
    text = report.verdict(mixed[1], mixed[2])
    check(mixed[1] < 0.0 < mixed[2], "mixed interval contains zero")
    check(text == report.UNDETECTED % ((mixed[2] - mixed[1]) / 2.0),
          "undetected verdict carries the half-width")
    check(report.paired_bootstrap([0.05, -0.04, 0.01, -0.02], 10000, 7) ==
          mixed, "a fixed seed repeats the interval")
    # One document: the interval collapses to its value.
    single = report.paired_bootstrap([0.5], 100, 1)
    check(single == (0.5, 0.5, 0.5), "single document")


def test_files_and_alignment():
    vocab = 5
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "raw").mkdir()
        documents = []
        expected = {}
        for index, domain in enumerate(["RU", "RU", "EN"]):
            name = "doc%d" % index
            ids = rng.integers(0, vocab, size=12)
            (base / (name + ".ids")).write_text(" ".join(map(str, ids)))
            # Reference: positions 2..4 and 7..9; candidate: 7..9 only.
            reference = rng.normal(size=(6, vocab)).astype("<f4")
            candidate = reference[3:6] + rng.normal(
                scale=0.1, size=(3, vocab)).astype("<f4")
            reference.tofile(base / "raw" / (name + ".ref.f32"))
            candidate.tofile(base / "raw" / (name + ".cand.f32"))
            documents.append({"id": name, "domain": domain,
                              "ids": name + ".ids"})
            expected[name] = report.document_metrics(
                reference[3:6], candidate, ids[8:11])
        spec = {"vocab": vocab, "bootstrap_samples": 2000, "seed": 11,
                "documents": documents,
                "variants": {
                    "ref": {"path": "raw/{id}.ref.f32",
                            "windows": [[2, 3], [7, 3]]},
                    "cand": {"path": "raw/{id}.cand.f32",
                             "windows": [[7, 3]]}},
                "comparisons": [{"name": "candidate against reference",
                                 "reference": "ref", "candidate": "cand",
                                 "first": 7, "count": 3}]}
        (base / "spec.json").write_text(json.dumps(spec))
        check(report.main(["tool", str(base / "spec.json"),
                           str(base / "out")]) == 0, "report runs")
        result = json.loads((base / "out" / "report.json").read_text())
        comparison = result["comparisons"][0]
        for row in comparison["documents"]:
            want = expected[row["id"]]
            check(all(row[key] == want[key] for key in want),
                  "second window rows and gold ids of " + row["id"])
        check(comparison["domains"]["RU"]["documents"] == 2 and
              comparison["domains"]["EN"]["documents"] == 1 and
              comparison["all"]["documents"] == 3, "grouping by domain")
        deltas = [d["delta_nll"] for d in comparison["documents"]]
        check(close(comparison["all"]["delta_nll_mean"],
                    sum(deltas) / 3.0), "mean over documents")
        check((base / "out" / "report.md").read_text().count("| doc") == 3,
              "per-document table")
        # An existing output directory is never overwritten.
        try:
            report.main(["tool", str(base / "spec.json"), str(base / "out")])
            check(False, "existing output directory accepted")
        except FileExistsError:
            pass
        # A truncated dump is rejected, not silently misaligned.
        path = base / "raw" / "doc0.cand.f32"
        path.write_bytes(path.read_bytes()[:-4])
        try:
            report.main(["tool", str(base / "spec.json"), str(base / "out2")])
            check(False, "truncated dump accepted")
        except ValueError:
            pass
        # A window that does not hold the compared positions is rejected.
        spec["comparisons"][0]["first"] = 6
        (base / "spec3.json").write_text(json.dumps(spec))
        try:
            report.main(["tool", str(base / "spec3.json"), str(base / "out3")])
            check(False, "positions outside the windows accepted")
        except ValueError:
            pass


def main():
    test_metrics()
    test_bootstrap_and_verdicts()
    test_files_and_alignment()
    for failure in FAILURES:
        print("FAIL:", failure)
    if not FAILURES:
        print("logits_nll_report_test: PASS")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
