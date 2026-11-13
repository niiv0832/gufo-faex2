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
    return {
        "positions": int(len(gold)),
        "nll_reference": float(nll_ref.mean()),
        "nll_candidate": float(nll_cand.mean()),
        "delta_nll": float((nll_cand - nll_ref).mean()),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
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
        metrics.update(id=document["id"], domain=document["domain"])
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


def markdown(results):
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
    report = {"spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
              "bootstrap_samples": spec["bootstrap_samples"],
              "seed": spec["seed"], "comparisons": results}
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (out_dir / "report.md").write_text(markdown(results) + "\n",
                                       encoding="utf-8")
    for result in results:
        s = result["all"]
        print("%s: %+.6f [%+.6f; %+.6f] %s" % (
            result["name"], s["delta_nll_mean"], s["delta_nll_low"],
            s["delta_nll_high"], s["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
