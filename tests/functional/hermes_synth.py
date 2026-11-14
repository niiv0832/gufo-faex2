#!/usr/bin/env python3
"""Synthetic Hermes Agent requests for the DFlash2-vs-MTP evaluation.

Loads the anonymised capture in fixtures/hermes_synth and composes complete
chat-completion bodies. It starts no server and loads no model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hermes_synth"
SHA256 = {
    "base_request.json": "b0f02262b3a4e728e43549dc89b600c3b3131686692a4de6307242dcc5a337a5",
    "aux_title_request.json": "c66efaeb29be6181f340b0a0e30fe54378d9e560d2d596b09cb15489e1d09d4c",
    "cases.jsonl": "0ef885cf77318a2653b8549b531c0aab8d1193cd5409d1abc78a550bfa893197",
}
CLASSES = ("tool", "mixed", "prose")


def _read(name: str) -> str:
    data = (FIXTURES / name).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256[name]:
        raise ValueError(f"{name}: sha256 {digest} does not match the pinned fixture")
    return data.decode("utf-8")


def load_base() -> dict:
    return json.loads(_read("base_request.json"))


def load_aux_title() -> dict:
    return json.loads(_read("aux_title_request.json"))


def load_cases() -> list[dict]:
    return [json.loads(line) for line in _read("cases.jsonl").splitlines() if line]


def build_request(case: dict, base: dict | None = None) -> dict:
    """Return the body Hermes would send for the next turn of this case."""
    base = base or load_base()
    body = copy.deepcopy(base["params"])
    body["tools"] = copy.deepcopy(base["tools"])
    body["messages"] = [{"role": "system", "content": base["system"]}]
    body["messages"] += copy.deepcopy(case["messages"])
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", metavar="CASE_ID", help="print one composed request")
    args = parser.parse_args()
    base, cases = load_base(), load_cases()
    if args.dump:
        case = next(c for c in cases if c["id"] == args.dump)
        print(json.dumps(build_request(case, base), ensure_ascii=False, indent=1))
        return
    for case in cases:
        body = build_request(case, base)
        size = len(json.dumps(body, ensure_ascii=False))
        print(f'{case["id"]:32} {case["class"]:6} messages={len(body["messages"])} chars={size}')


if __name__ == "__main__":
    main()
