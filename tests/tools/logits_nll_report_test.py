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
    check(same["bitwise_equal_rows"] == 1 and same["kl_mean"] == 0.0 and
          same["max_logit_delta"] == 0.0, "identical rows")
    check(close(m["max_logit_delta"], math.log(3.0), 1e-6), "max_logit_delta")


def _write_q1_case(base, docs_spec, q1_cfg, comparisons):
    vocab = 4
    (base / "raw").mkdir(exist_ok=True)
    docs = []
    for doc in docs_spec:
        name = doc["id"]
        ids = np.array(doc["ids"], dtype=int)
        (base / (name + ".ids")).write_text(" ".join(map(str, ids)))
        doc["ref"].astype("<f4").tofile(base / "raw" / (name + ".ref.f32"))
        doc["cand"].astype("<f4").tofile(base / "raw" / (name + ".cand.f32"))
        entry = {"id": name, "domain": doc["domain"], "ids": name + ".ids"}
        if "cluster_id" in doc:
            entry["cluster_id"] = doc["cluster_id"]
        docs.append(entry)
    spec = {
        "vocab": vocab,
        "bootstrap_samples": 1000,
        "seed": 20261119,
        "documents": docs,
        "variants": {
            "g1": {"path": "raw/{id}.ref.f32", "windows": [[0, 2]]},
            "p": {"path": "raw/{id}.cand.f32", "windows": [[0, 2]]},
        },
        "comparisons": comparisons,
        "q1": q1_cfg,
    }
    path = base / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_q1_non_inferiority_and_clusters():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        ref_row = np.array([[3.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        small_cand = np.array([[2.98, 0.0, 0.0, 0.0], [2.98, 0.0, 0.0, 0.0]], dtype=np.float32)
        bad_cand = np.array([[0.0, 3.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]], dtype=np.float32)

        # Case 1: PASS on RU/code/math, non-required EN fails without failing Q1.
        docs = []
        for dom in ("RU", "code", "math"):
            for i in range(4):
                docs.append({"id": f"{dom}_{i}", "domain": dom,
                             "cluster_id": f"{dom}_c{i}", "ids": [0, 0, 0],
                             "ref": ref_row, "cand": small_cand})
        for i in range(4):
            docs.append({"id": f"EN_{i}", "domain": "EN", "cluster_id": f"EN_c{i}",
                         "ids": [0, 0, 0], "ref": ref_row, "cand": bad_cand})
        spec_path = _write_q1_case(
            base / "pass_case" if False else base, docs,
            {"alpha": 0.05, "family_alpha": 0.05, "min_clusters": 4},
            [{"name": "short", "reference": "g1", "candidate": "p", "first": 0, "count": 2,
              "delta_nll": 0.010, "kappa": 0.010, "required_domains": ["RU", "code", "math"]}],
        )
        check(report.main(["tool", str(spec_path), str(base / "out_pass")]) == 0, "q1 pass run")
        res = json.loads((base / "out_pass" / "report.json").read_text())
        q1 = res["q1"]
        check(q1["status"] == "PASS", "q1 overall PASS")
        check(q1["required_cells"] == 3 and q1["family_k"] == 6, "family K for 3 cells")
        en_cell = next(c for c in q1["cells"] if c["domain"] == "EN")
        check(not en_cell["required"] and en_cell["verdict"] == "FAIL",
              "diagnostic EN cell failed independently")
        check("## Контракт Q1" in (base / "out_pass" / "report.md").read_text(),
              "markdown includes Q1 table")

        # Case 2: Duplicate cluster_id prevents fake N inflation (4 docs, 2 clusters < min_clusters=3).
        base2 = base / "cluster_case"
        base2.mkdir()
        docs2 = [{"id": f"RU_{i}", "domain": "RU", "cluster_id": f"c{i % 2}",
                  "ids": [0, 0, 0], "ref": ref_row, "cand": small_cand}
                 for i in range(4)]
        spec2 = _write_q1_case(
            base2, docs2, {"alpha": 0.05, "family_alpha": 0.05, "min_clusters": 3},
            [{"name": "short", "reference": "g1", "candidate": "p", "first": 0, "count": 2,
              "delta_nll": 0.010, "kappa": 0.010}],
        )
        report.main(["tool", str(spec2), str(base2 / "out")])
        res2 = json.loads((base2 / "out" / "report.json").read_text())
        check(res2["q1"]["status"] == "INCONCLUSIVE", "insufficient clusters gives INCONCLUSIVE")
        check(res2["q1"]["cells"][0]["clusters"] == 2 and
              res2["q1"]["cells"][0]["documents"] == 4 and
              res2["q1"]["cells"][0]["reason"] == "insufficient_clusters",
              "cluster count collapsed properly")

        # Case 3: Missing required domain -> INCONCLUSIVE.
        base3 = base / "missing_dom"
        base3.mkdir()
        spec3 = _write_q1_case(
            base3, docs2, {"alpha": 0.05, "family_alpha": 0.05, "min_clusters": 2},
            [{"name": "short", "reference": "g1", "candidate": "p", "first": 0, "count": 2,
              "delta_nll": 0.010, "kappa": 0.010, "required_domains": ["RU", "math"]}],
        )
        report.main(["tool", str(spec3), str(base3 / "out")])
        res3 = json.loads((base3 / "out" / "report.json").read_text())
        check(res3["q1"]["status"] == "INCONCLUSIVE", "missing required domain -> INCONCLUSIVE")

        # Case 4: Confirmed exceedance -> FAIL, and larger K widens joint CI.
        base4 = base / "fail_case"
        base4.mkdir()
        docs4 = [{"id": f"RU_{i}", "domain": "RU", "cluster_id": f"c{i}",
                  "ids": [0, 0, 0], "ref": ref_row, "cand": bad_cand}
                 for i in range(6)]
        spec4 = _write_q1_case(
            base4, docs4, {"alpha": 0.05, "family_alpha": 0.05, "min_clusters": 4},
            [{"name": "short", "reference": "g1", "candidate": "p", "first": 0, "count": 2,
              "delta_nll": 0.010, "kappa": 0.010}],
        )
        report.main(["tool", str(spec4), str(base4 / "out")])
        res4 = json.loads((base4 / "out" / "report.json").read_text())
        check(res4["q1"]["status"] == "FAIL" and
              res4["q1"]["cells"][0]["reason"] == "exceedance_confirmed",
              "confirmed exceedance gives FAIL")

        # Case 5: Interval overlapping margin -> INCONCLUSIVE (not PASS, not FAIL).
        base5 = base / "overlap_case"
        base5.mkdir()
        docs5 = []
        for i in range(4):
            cand_i = small_cand if i < 2 else bad_cand
            docs5.append({"id": f"RU_{i}", "domain": "RU", "cluster_id": f"c{i}",
                          "ids": [0, 0, 0], "ref": ref_row, "cand": cand_i})
        spec5 = _write_q1_case(
            base5, docs5, {"alpha": 0.05, "family_alpha": 0.05, "min_clusters": 4},
            [{"name": "short", "reference": "g1", "candidate": "p", "first": 0, "count": 2,
              "delta_nll": 0.010, "kappa": 0.010}],
        )
        report.main(["tool", str(spec5), str(base5 / "out")])
        res5 = json.loads((base5 / "out" / "report.json").read_text())
        check(res5["q1"]["status"] == "INCONCLUSIVE" and
              res5["q1"]["cells"][0]["reason"] == "interval_overlaps_margin",
              "overlapping interval gives INCONCLUSIVE")
        check(res5["q1"]["cells"][0]["delta_nll"]["sigma"] is not None,
              "observed cluster sigma reported for pilot power planning")

        # Case 6: Invalid min_clusters < 2 or delta_nll <= 0 rejected.
        base6 = base / "bad_cfg"
        base6.mkdir()
        spec6 = _write_q1_case(
            base6, docs2, {"alpha": 0.05, "family_alpha": 0.05, "min_clusters": 1},
            [{"name": "short", "reference": "g1", "candidate": "p", "first": 0, "count": 2,
              "delta_nll": 0.010, "kappa": 0.010}],
        )
        try:
            report.main(["tool", str(spec6), str(base6 / "out")])
            check(False, "min_clusters < 2 accepted")
        except ValueError:
            pass


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
    test_q1_non_inferiority_and_clusters()
    for failure in FAILURES:
        print("FAIL:", failure)
    if not FAILURES:
        print("logits_nll_report_test: PASS")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
