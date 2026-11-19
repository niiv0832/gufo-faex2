#!/usr/bin/env python3
"""Per-document NLL comparison of full-vocabulary logits dumps.

Reads raw float32 logits rows written by qwen35moe_gpu_logits_tool (or any
tool with the same layout) for a frozen corpus and compares a candidate
reading of each document against a reference reading of the same token ids.

The unit of observation is the document: consecutive positions of one
document are not independent. Per document it reports the mean NLL of the
true next token under both readings, their difference, KL(reference ||
candidate) over the whole vocabulary and top-1 agreement. Per domain and
overall it reports the mean document difference with a 95 % paired bootstrap
interval over documents and one of three verdicts. No tolerance is applied:
an interval that contains zero means "no difference detected at this
sensitivity", never "no loss".

usage: logits-nll-report.py SPEC.json OUT_DIR

SPEC.json:
  {"vocab": 248320, "bootstrap_samples": 10000, "seed": 20261113,
   "documents": [{"id": "ru_a", "domain": "RU", "ids": "ids/ru_a.ids"}],
   "variants": {"g1": {"path": "raw/g1/{id}.f32",
                       "windows": [[15, 64], [1023, 64]]}},
   "comparisons": [{"name": "P15 against decode", "reference": "g1",
                    "candidate": "p15", "first": 15, "count": 64}]}

A variant file holds the rows of its windows back to back; window [a, n] is
the rows of positions a .. a + n - 1 and the row of position p predicts
ids[p + 1]. Relative paths resolve against the directory of SPEC.json.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

WORSE = "ухудшение обнаружено"
BETTER = "улучшение обнаружено"
UNDETECTED = "различие не обнаружено, чувствительность ± %.6f"


def log_softmax(rows):
    rows = rows.astype(np.float64)
    shifted = rows - rows.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def load_rows(path, windows, vocab, first, count):
    """Rows of positions first .. first + count - 1 from a variant file."""
    total = sum(n for _, n in windows)
    expected = total * vocab * 4
    actual = Path(path).stat().st_size
    if actual != expected:
        raise ValueError("%s: %d bytes, expected %d" % (path, actual, expected))
    offset = 0
    for start, n in windows:
        if start <= first and first + count <= start + n:
            data = np.memmap(path, dtype="<f4", mode="r",
                             shape=(total, vocab))
            rows = np.array(data[offset + first - start:
                                 offset + first - start + count])
            if not np.isfinite(rows).all():
                raise ValueError("%s: non-finite logits" % path)
            return rows
        offset += n
    raise ValueError("%s: positions %d..%d are in no window"
                     % (path, first, first + count - 1))


def document_metrics(reference, candidate, gold):
    """Metrics of one document from [positions, vocab] logits and gold ids."""
    ref = log_softmax(reference)
    cand = log_softmax(candidate)
    index = np.arange(len(gold))
    nll_ref = -ref[index, gold]
    nll_cand = -cand[index, gold]
    kl = (np.exp(ref) * (ref - cand)).sum(axis=1)
    top_ref = reference.argmax(axis=1)
    top_cand = candidate.argmax(axis=1)
    max_logit_delta = float(
        np.abs(candidate.astype(np.float64) - reference.astype(np.float64)).max())
    return {
        "positions": int(len(gold)),
        "nll_reference": float(nll_ref.mean()),
        "nll_candidate": float(nll_cand.mean()),
        "delta_nll": float((nll_cand - nll_ref).mean()),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "max_logit_delta": max_logit_delta,
        "top1_agree": int((top_ref == top_cand).sum()),
        "top1_gold_reference": int((top_ref == gold).sum()),
        "top1_gold_candidate": int((top_cand == gold).sum()),
        "bitwise_equal_rows": int(
            (reference.view(np.uint32) == candidate.view(np.uint32))
            .all(axis=1).sum()),
    }


def paired_bootstrap(values, samples, seed):
    """Mean and the 95 % percentile interval of its document bootstrap."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def verdict(low, high):
    if low > 0.0:
        return WORSE
    if high < 0.0:
        return BETTER
    return UNDETECTED % ((high - low) / 2.0)


def summarize(documents, samples, seed):
    deltas = [d["delta_nll"] for d in documents]
    mean, low, high = paired_bootstrap(deltas, samples, seed)
    kl_mean, kl_low, kl_high = paired_bootstrap(
        [d["kl_mean"] for d in documents], samples, seed)
    positions = sum(d["positions"] for d in documents)
    return {
        "documents": len(documents),
        "delta_nll_mean": mean, "delta_nll_low": low, "delta_nll_high": high,
        "verdict": verdict(low, high),
        "documents_worse": sum(d > 0.0 for d in deltas),
        "documents_better": sum(d < 0.0 for d in deltas),
        "kl_mean": kl_mean, "kl_low": kl_low, "kl_high": kl_high,
        "top1_agree": sum(d["top1_agree"] for d in documents),
        "positions": positions,
    }


def compare(spec, base, comparison):
    vocab = spec["vocab"]
    first, count = comparison["first"], comparison["count"]
    rows = []
    for document in spec["documents"]:
        ids = [int(x) for x in
               (base / document["ids"]).read_text().split()]
        if len(ids) < first + count + 1:
            raise ValueError(document["id"] + ": too few ids")
        gold = np.array(ids[first + 1:first + count + 1])
        loaded = []
        for side in ("reference", "candidate"):
            variant = spec["variants"][comparison[side]]
            path = base / variant["path"].format(id=document["id"])
            loaded.append(load_rows(path, variant["windows"], vocab, first,
                                    count))
        metrics = document_metrics(loaded[0], loaded[1], gold)
        metrics.update(id=document["id"], domain=document["domain"],
                       cluster_id=document.get("cluster_id", document["id"]))
        rows.append(metrics)
    samples, seed = spec["bootstrap_samples"], spec["seed"]
    domains = {}
    for row in rows:
        domains.setdefault(row["domain"], []).append(row)
    return {
        "name": comparison["name"], "reference": comparison["reference"],
        "candidate": comparison["candidate"], "first": first, "count": count,
        "documents": rows,
        "domains": {name: summarize(group, samples, seed)
                    for name, group in domains.items()},
        "all": summarize(rows, samples, seed),
    }


def paired_bootstrap_q1(values, samples, seed, one_sided_alpha, joint_side_alpha):
    """Mean, one-sided IUT upper bound, simultaneous Bonferroni bounds, and cluster sigma."""
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(arr), size=(samples, len(arr)))
    means = arr[picks].mean(axis=1)
    upper_1sided = float(np.percentile(means, 100.0 * (1.0 - one_sided_alpha)))
    joint_low, joint_high = np.percentile(
        means, [100.0 * joint_side_alpha, 100.0 * (1.0 - joint_side_alpha)])
    sigma = float(np.std(arr, ddof=1)) if len(arr) >= 2 else None
    return {
        "mean": float(arr.mean()),
        "upper_1sided": upper_1sided,
        "joint_low": float(joint_low),
        "joint_high": float(joint_high),
        "sigma": sigma,
    }


def _cluster_means(rows, key):
    grouped = {}
    for row in rows:
        cid = row.get("cluster_id", row["id"])
        grouped.setdefault(cid, []).append(row[key])
    cids = sorted(grouped)
    return cids, [float(np.mean(grouped[cid])) for cid in cids]


def evaluate_q1(spec, results):
    """Evaluate Q1 non-inferiority barriers across (comparison, domain) cells."""
    q1_cfg = spec["q1"]
    alpha = float(q1_cfg.get("alpha", 0.05))
    family_alpha = float(q1_cfg.get("family_alpha", 0.05))
    min_clusters_default = int(q1_cfg.get("min_clusters", 2))
    if not (0.0 < alpha < 0.5) or not (0.0 < family_alpha < 0.5):
        raise ValueError("alpha and family_alpha must be in (0, 0.5)")
    if min_clusters_default < 2:
        raise ValueError("q1.min_clusters must be >= 2")

    samples = int(spec["bootstrap_samples"])
    seed = int(spec["seed"])

    cell_specs = []
    for comp_cfg, comp_res in zip(spec["comparisons"], results):
        comp_required = bool(comp_cfg.get("required", True))
        req_domains = comp_cfg.get("required_domains")
        present_domains = sorted(comp_res["domains"])
        all_domains = sorted(set(present_domains) | set(req_domains or []))
        delta_limit = comp_cfg.get("delta_nll", q1_cfg.get("default_delta_nll"))
        kappa_limit = comp_cfg.get("kappa", q1_cfg.get("default_kappa"))
        if delta_limit is None or kappa_limit is None:
            raise ValueError(comp_cfg["name"] + ": missing delta_nll or kappa")
        delta_limit, kappa_limit = float(delta_limit), float(kappa_limit)
        if not (np.isfinite(delta_limit) and delta_limit > 0.0 and
                np.isfinite(kappa_limit) and kappa_limit > 0.0):
            raise ValueError(comp_cfg["name"] + ": delta_nll and kappa must be finite and > 0")
        min_clusters = int(comp_cfg.get("min_clusters", min_clusters_default))
        if min_clusters < 2:
            raise ValueError(comp_cfg["name"] + ": min_clusters must be >= 2")
        for domain in all_domains:
            is_required = comp_required and (req_domains is None or domain in req_domains)
            docs = [d for d in comp_res["documents"] if d["domain"] == domain]
            cell_specs.append({
                "comparison": comp_res["name"],
                "domain": domain,
                "required": is_required,
                "delta_nll_limit": delta_limit,
                "kappa_limit": kappa_limit,
                "min_clusters": min_clusters,
                "docs": docs,
            })

    required_cells_count = sum(1 for c in cell_specs if c["required"])
    family_k = int(q1_cfg.get("family_k", max(1, 2 * required_cells_count)))
    if family_k < 1:
        raise ValueError("q1.family_k must be >= 1")
    joint_side_alpha = family_alpha / (2.0 * family_k)

    evaluated_cells = []
    for index, cell in enumerate(cell_specs):
        docs = cell["docs"]
        cids, delta_vals = _cluster_means(docs, "delta_nll")
        _, kl_vals = _cluster_means(docs, "kl_mean")
        n_clusters = len(cids)
        base_entry = {
            "comparison": cell["comparison"],
            "domain": cell["domain"],
            "required": cell["required"],
            "documents": len(docs),
            "clusters": n_clusters,
            "min_clusters": cell["min_clusters"],
            "delta_nll_limit": cell["delta_nll_limit"],
            "kappa_limit": cell["kappa_limit"],
        }
        if not docs:
            base_entry.update({"verdict": "INCONCLUSIVE", "reason": "missing_domain_documents"})
            evaluated_cells.append(base_entry)
            continue
        d_stats = paired_bootstrap_q1(
            delta_vals, samples, seed + index * 2, alpha, joint_side_alpha)
        k_stats = paired_bootstrap_q1(
            kl_vals, samples, seed + index * 2 + 1, alpha, joint_side_alpha)
        kl_max = float(max(d["kl_max"] for d in docs))
        max_logit_delta = float(max(d["max_logit_delta"] for d in docs))
        if n_clusters < cell["min_clusters"]:
            q1_verdict, reason = "INCONCLUSIVE", "insufficient_clusters"
        elif (d_stats["joint_low"] > cell["delta_nll_limit"] or
              k_stats["joint_low"] > cell["kappa_limit"]):
            q1_verdict, reason = "FAIL", "exceedance_confirmed"
        elif (d_stats["upper_1sided"] < cell["delta_nll_limit"] and
              k_stats["upper_1sided"] < cell["kappa_limit"]):
            q1_verdict, reason = "PASS", "non_inferior"
        else:
            q1_verdict, reason = "INCONCLUSIVE", "interval_overlaps_margin"
        base_entry.update({
            "delta_nll": d_stats,
            "kl": k_stats,
            "kl_max": kl_max,
            "max_logit_delta": max_logit_delta,
            "verdict": q1_verdict,
            "reason": reason,
        })
        evaluated_cells.append(base_entry)

    required_verdicts = [c["verdict"] for c in evaluated_cells if c["required"]]
    if "FAIL" in required_verdicts:
        overall = "FAIL"
    elif not required_verdicts or "INCONCLUSIVE" in required_verdicts:
        overall = "INCONCLUSIVE"
    else:
        overall = "PASS"

    return {
        "status": overall,
        "alpha_one_sided_iut": alpha,
        "family_alpha_two_sided": family_alpha,
        "family_k": family_k,
        "joint_side_alpha": joint_side_alpha,
        "required_cells": required_cells_count,
        "note": ("Численная оценка Q1 по предъявленному срезу. Полная квалификация Q1 "
                 "также требует точных инвариантов уровня 1 и прикладного набора задач."),
        "cells": evaluated_cells,
    }


def markdown_q1(q1_report):
    out = ["## Контракт Q1 (non-inferiority)\n",
           "Итоговый численный статус: **%s** (`α` IUT = %.3f, Bonferroni `α/(2K)` при `K = %d`, "
           "обязательных ячеек: %d)\n" % (
               q1_report["status"], q1_report["alpha_one_sided_iut"],
               q1_report["family_k"], q1_report["required_cells"]),
           "| Срез | Домен | Обяз. | Класт./док. | ΔNLL средн. | ΔNLL upper | δ_NLL | "
           "KL средн. | KL upper | κ | σ(ΔNLL) | Статус (причина) |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in q1_report["cells"]:
        req = "да" if c["required"] else "нет"
        counts = "%d/%d" % (c["clusters"], c["documents"])
        if "delta_nll" not in c:
            out.append("| %s | %s | %s | %s | — | — | %.4f | — | — | %.4f | — | %s (%s) |" % (
                c["comparison"], c["domain"], req, counts, c["delta_nll_limit"],
                c["kappa_limit"], c["verdict"], c["reason"]))
        else:
            d, k = c["delta_nll"], c["kl"]
            sig = "%.5f" % d["sigma"] if d["sigma"] is not None else "—"
            out.append("| %s | %s | %s | %s | %+.6f | %+.6f | %.4f | %.6f | %.6f | %.4f | %s | %s (%s) |" % (
                c["comparison"], c["domain"], req, counts,
                d["mean"], d["upper_1sided"], c["delta_nll_limit"],
                k["mean"], k["upper_1sided"], c["kappa_limit"], sig,
                c["verdict"], c["reason"]))
    out.append("")
    return "\n".join(out)


def markdown(results, q1_report=None):
    out = []
    for result in results:
        out.append("## %s\n" % result["name"])
        out.append("reference `%s`, candidate `%s`, positions %d..%d; "
                   "ΔNLL = candidate − reference, nats per token\n"
                   % (result["reference"], result["candidate"],
                      result["first"],
                      result["first"] + result["count"] - 1))
        out.append("| Группа | Док. | ΔNLL среднее | 95 % интервал | Хуже/лучше "
                   "| KL среднее | Top-1 | Вывод |")
        out.append("|---|---:|---:|---|---:|---:|---:|---|")
        groups = list(result["domains"].items()) + [("все", result["all"])]
        for name, s in groups:
            out.append("| %s | %d | %+.6f | [%+.6f; %+.6f] | %d/%d | %.6f | "
                       "%d/%d | %s |"
                       % (name, s["documents"], s["delta_nll_mean"],
                          s["delta_nll_low"], s["delta_nll_high"],
                          s["documents_worse"], s["documents_better"],
                          s["kl_mean"], s["top1_agree"], s["positions"],
                          s["verdict"]))
        out.append("")
        out.append("| Документ | Домен | NLL эталона | NLL кандидата | ΔNLL | "
                   "KL среднее | KL макс. | Top-1 |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for d in result["documents"]:
            out.append("| %s | %s | %.6f | %.6f | %+.6f | %.6f | %.6f | %d/%d |"
                       % (d["id"], d["domain"], d["nll_reference"],
                          d["nll_candidate"], d["delta_nll"], d["kl_mean"],
                          d["kl_max"], d["top1_agree"], d["positions"]))
        out.append("")
    if q1_report is not None:
        out.append(markdown_q1(q1_report))
    return "\n".join(out)


def main(argv):
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    spec_path = Path(argv[1]).resolve()
    out_dir = Path(argv[2])
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    out_dir.mkdir(parents=True, exist_ok=False)
    results = [compare(spec, spec_path.parent, comparison)
               for comparison in spec["comparisons"]]
    q1_report = evaluate_q1(spec, results) if "q1" in spec else None
    report = {"spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
              "bootstrap_samples": spec["bootstrap_samples"],
              "seed": spec["seed"], "comparisons": results}
    if q1_report is not None:
        report["q1"] = q1_report
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (out_dir / "report.md").write_text(markdown(results, q1_report) + "\n",
                                       encoding="utf-8")
    for result in results:
        s = result["all"]
        print("%s: %+.6f [%+.6f; %+.6f] %s" % (
            result["name"], s["delta_nll_mean"], s["delta_nll_low"],
            s["delta_nll_high"], s["verdict"]))
    if q1_report is not None:
        print("Q1: %s (required_cells=%d, K=%d)" % (
            q1_report["status"], q1_report["required_cells"], q1_report["family_k"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
