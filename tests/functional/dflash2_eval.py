#!/usr/bin/env python3
"""Step 1/2 HTTP evaluator for DFlash2 vs MTP1 vs off on hermes_synth.

Sends the pinned hermes_synth requests plus aux_title_request.json to an
isolated gufo serve instance over loopback HTTP, records raw responses and
per-request metrics without averaging across requests, validates tool_calls
and JSON-schema outputs with stdlib only, and cross-checks server.log
(`event=load_completed` and `event=completed`).
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hermes_synth  # noqa: E402
from metrics import summarize  # noqa: E402


WARMUP_REQUEST = {
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "28+28="}],
    "reasoning_effort": "none",
    "temperature": 0.0,
    "seed": 1,
    "max_tokens": 16,
    "stream": False,
}


def parse_sse_and_json(raw: bytes, streaming: bool) -> dict:
    """Parse complete chat completion text, reasoning, tool_calls, usage, timings."""
    if streaming:
        lines = [
            line[5:].strip()
            for line in raw.splitlines()
            if line.startswith(b"data:")
        ]
        if not lines or lines[-1] != b"[DONE]" or lines.count(b"[DONE]") != 1:
            raise ValueError("SSE stream missing terminal [DONE] or has trailing data")
        events = [json.loads(line) for line in lines[:-1]]
    else:
        events = [json.loads(raw)]

    if not events:
        raise ValueError("empty response events")

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tools: dict[int, dict] = {}
    finish_reason: str | None = None
    usage: dict = {}
    timings: dict = {}
    response_id: str | None = None

    for event in events:
        if isinstance(event.get("id"), str) and response_id is None:
            response_id = event["id"]
        if isinstance(event.get("usage"), dict) and event["usage"]:
            usage = event["usage"]
        if isinstance(event.get("timings"), dict) and event["timings"]:
            timings.update(event["timings"])
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            msg = choice.get("delta" if streaming else "message", {})
            if isinstance(msg.get("content"), str):
                text_parts.append(msg["content"])
            if isinstance(msg.get("reasoning_content"), str):
                reasoning_parts.append(msg["reasoning_content"])
            for idx, call in enumerate(msg.get("tool_calls") or []):
                slot = call.get("index") if streaming else idx
                if not isinstance(slot, int) or slot < 0:
                    raise ValueError(f"invalid tool call slot: {slot!r}")
                entry = tools.setdefault(
                    slot,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if call.get("id"):
                    entry["id"] += str(call["id"])
                fn = call.get("function") or {}
                if fn.get("name"):
                    entry["function"]["name"] += str(fn["name"])
                if fn.get("arguments"):
                    entry["function"]["arguments"] += str(fn["arguments"])
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

    ordered_tools = [tools[k] for k in sorted(tools)]
    return {
        "response_id": response_id,
        "text": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": ordered_tools,
        "finish_reason": finish_reason,
        "usage": usage,
        "timings": timings,
    }


def validate_tool_calls(
    tool_calls: list[dict],
    declared_tools: dict[str, dict],
) -> tuple[bool, list[dict], str | None]:
    """Validate tool call names, JSON object arguments, and required schema keys."""
    details: list[dict] = []
    if not tool_calls:
        return False, details, "no tool_calls generated"
    for call in tool_calls:
        fn = call.get("function") or {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "")
        declared = name in declared_tools
        parsed_args = None
        valid_json = False
        missing_required: list[str] = []
        error_msg: str | None = None
        try:
            parsed_args = json.loads(raw_args)
            valid_json = isinstance(parsed_args, dict)
            if not valid_json:
                error_msg = "arguments JSON is not an object"
        except json.JSONDecodeError as exc:
            error_msg = f"invalid JSON arguments: {exc}"
        if not declared:
            error_msg = error_msg or f"undeclared tool: {name}"
        elif valid_json:
            required = declared_tools[name].get("parameters", {}).get("required", [])
            missing_required = [k for k in required if k not in parsed_args]
            if missing_required:
                error_msg = f"missing required args: {missing_required}"
        details.append(
            {
                "id": call.get("id", ""),
                "name": name,
                "declared": declared,
                "valid_json": valid_json,
                "missing_required": missing_required,
                "arguments_raw": raw_args,
                "error": error_msg,
            }
        )
    all_ok = all(
        item["declared"] and item["valid_json"] and not item["missing_required"]
        for item in details
    )
    first_err = next((item["error"] for item in details if item["error"]), None)
    return all_ok, details, first_err


def validate_aux_title(text: str, finish_reason: str | None) -> tuple[bool, str | None]:
    """Validate aux_title_request json_schema output: {"title": non-empty str}."""
    if finish_reason != "stop":
        return False, f"unexpected finish_reason={finish_reason!r}"
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON: {exc}"
    if not isinstance(obj, dict):
        return False, "output is not a JSON object"
    if set(obj.keys()) != {"title"}:
        return False, f"unexpected keys: {sorted(obj.keys())}"
    if not isinstance(obj["title"], str) or not obj["title"].strip():
        return False, "title must be a non-empty string"
    return True, None


def parse_server_log(log_text: str) -> dict:
    """Extract load_completed mode/draft_limit and per-request completed lines."""
    loads = []
    for line in log_text.splitlines():
        if "event=load_completed" in line and "kind=text" in line:
            fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", line))
            loads.append(fields)
    completed = {}
    for line in log_text.splitlines():
        if "event=completed " in line:
            fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", line))
            if "request" in fields:
                completed[fields["request"]] = fields
    return {"loads": loads, "completed": completed}


def send_chat_completion(
    host: str,
    port: int,
    body: dict,
    timeout_s: float,
    raw_out_path: Path | None = None,
) -> dict:
    """Send POST /v1/chat/completions over loopback and capture raw + timings."""
    streaming = bool(body.get("stream", False))
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    started = time.monotonic()
    parts: list[tuple[float, bytes]] = []
    try:
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        status = resp.status
        request_id = resp.getheader("X-Request-ID")
        content_type = resp.getheader("Content-Type")
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            parts.append(((time.monotonic() - started) * 1000.0, chunk))
        wall_ms = (time.monotonic() - started) * 1000.0
    finally:
        conn.close()

    raw = b"".join(chunk for _, chunk in parts)
    if raw_out_path is not None:
        raw_out_path.write_bytes(raw)

    result: dict = {
        "http_status": status,
        "request_id": request_id,
        "content_type": content_type,
        "wall_ms": wall_ms,
        "streaming": streaming,
    }
    if status != 200:
        result["error"] = f"HTTP {status}: {raw.decode('utf-8', errors='replace')[:500]}"
        return result

    measured, output_sha256 = summarize(parts, streaming, True, contract=None)
    parsed = parse_sse_and_json(raw, streaming)
    usage = parsed["usage"]
    gufo = usage.get("gufo", {}) if isinstance(usage, dict) else {}
    comp_details = (
        usage.get("completion_tokens_details", {})
        if isinstance(usage, dict)
        else {}
    )
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    reasoning_tokens = comp_details.get("reasoning_tokens", 0)
    if not isinstance(reasoning_tokens, int):
        reasoning_tokens = 0
    non_reasoning_tokens = max(0, completion_tokens - reasoning_tokens)
    draft_tokens = usage.get("draft_tokens", measured.get("draft_tokens", 0))
    draft_accepted = usage.get(
        "draft_tokens_accepted", measured.get("draft_tokens_accepted", 0)
    )
    draft_rounds = gufo.get("draft_rounds", parsed["timings"].get("draft_rounds", 0))
    decode_ms = measured.get("decode_ms", 0.0)
    prefill_ms = measured.get("prefill_ms", 0.0)
    prefill_tokens = measured.get("prefill_tokens", 0)

    result.update(
        {
            "output_sha256": output_sha256,
            "finish_reason": parsed["finish_reason"],
            "text": parsed["text"],
            "reasoning": parsed["reasoning"],
            "tool_calls": parsed["tool_calls"],
            "prompt_tokens": prompt_tokens,
            "cached_tokens": measured.get("cached_tokens", 0),
            "prefill_tokens": prefill_tokens,
            "prefill_ms": prefill_ms,
            "prefill_tps": usage.get(
                "prompt_tokens_per_second",
                (1000.0 * prefill_tokens / prefill_ms) if prefill_ms > 0 else 0.0,
            ),
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "non_reasoning_tokens": non_reasoning_tokens,
            "decode_ms": decode_ms,
            "decode_tps": usage.get(
                "completion_tokens_per_second",
                (1000.0 * completion_tokens / decode_ms) if decode_ms > 0 else 0.0,
            ),
            "ttft_ms": measured.get("ttft_ms"),
            "client_ttft_ms": measured.get("client_ttft_ms"),
            "queue_ms": measured.get("queue_ms"),
            "mean_inter_token_ms": gufo.get("mean_inter_token_ms"),
            "max_inter_token_ms": gufo.get("max_inter_token_ms"),
            "cache_hit": gufo.get("cache_hit"),
            "cache_disk_hit": gufo.get("cache_disk_hit"),
            "cache_miss_reason": gufo.get("cache_miss_reason"),
            "execution_plan": gufo.get("execution_plan"),
            "draft_tokens": draft_tokens,
            "draft_tokens_accepted": draft_accepted,
            "draft_rounds": draft_rounds,
            "acceptance_rate": (
                draft_accepted / draft_tokens if draft_tokens else 0.0
            ),
        }
    )
    return result


def evaluate_record(
    case_class: str,
    res: dict,
    declared_tools: dict[str, dict],
) -> dict:
    """Attach class-specific validity outcome to a completed request result."""
    if res.get("http_status") != 200:
        return {"valid": False, "validation_error": res.get("error", "http error")}
    finish = res.get("finish_reason")
    if case_class in ("tool", "mixed"):
        ok, tool_details, err = validate_tool_calls(res["tool_calls"], declared_tools)
        if finish != "tool_calls":
            ok = False
            err = err or f"finish_reason={finish!r} (expected 'tool_calls')"
        return {"valid": ok, "validation_error": err, "tool_details": tool_details}
    if case_class == "prose":
        ok = finish == "stop" and bool(res.get("text", "").strip()) and not res.get("tool_calls")
        err = None
        if finish != "stop":
            err = f"finish_reason={finish!r}"
        elif res.get("tool_calls"):
            err = "unexpected tool_calls in prose case"
        elif not res.get("text", "").strip():
            err = "empty text in prose case"
        return {"valid": ok, "validation_error": err}
    if case_class == "aux_title":
        ok, err = validate_aux_title(res.get("text", ""), finish)
        return {"valid": ok, "validation_error": err}
    if case_class == "warmup":
        ok = finish == "stop" and bool(res.get("text", "").strip())
        return {"valid": ok, "validation_error": None if ok else f"finish_reason={finish!r}"}
    return {"valid": True, "validation_error": None}


def join_log_fields(records: list[dict], server_log_path: Path) -> dict:
    """Cross-check server.log load_completed and completed lines."""
    parsed = parse_server_log(server_log_path.read_text(encoding="utf-8", errors="replace"))
    completed = parsed["completed"]
    for rec in records:
        rid = rec.get("request_id")
        if rid and rid in completed:
            fields = completed[rid]
            rec["server_log"] = {
                "cache": fields.get("cache"),
                "cached_tokens": int(fields["cached_tokens"]) if "cached_tokens" in fields else None,
                "duration_ms": float(fields["duration_ms"]) if "duration_ms" in fields else None,
                "ttft_ms": float(fields["ttft_ms"]) if "ttft_ms" in fields else None,
                "prefill_tps": float(fields["prefill_tps"]) if "prefill_tps" in fields else None,
                "decode_tps": float(fields["decode_tps"]) if "decode_tps" in fields else None,
                "draft_accepted": int(fields["draft_accepted"]) if "draft_accepted" in fields else None,
                "draft_proposed": int(fields["draft_proposed"]) if "draft_proposed" in fields else None,
                "draft_rounds": int(fields["draft_rounds"]) if "draft_rounds" in fields else None,
                "acceptance_pct": float(fields["acceptance_pct"]) if "acceptance_pct" in fields else None,
                "plan": fields.get("plan"),
            }
    return parsed["loads"]


def run_eval(args: argparse.Namespace) -> int:
    url = urlsplit(args.base_url)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost") or not url.port:
        raise ValueError(f"Expected loopback http URL, got {args.base_url!r}")
    host, port = url.hostname, url.port

    out_dir = Path(args.output)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    base = hermes_synth.load_base()
    cases = hermes_synth.load_cases()
    aux_title = hermes_synth.load_aux_title()
    declared_tools = {
        t["function"]["name"]: t["function"]
        for t in base["tools"]
        if t.get("type") == "function"
    }

    # Probe /health and /props first
    conn = http.client.HTTPConnection(host, port, timeout=5.0)
    try:
        conn.request("GET", "/props")
        resp = conn.getresponse()
        props = json.loads(resp.read())
    finally:
        conn.close()

    schedule: list[tuple[int, str, str, dict]] = []
    if not args.no_warmup:
        schedule.append((0, "warmup-28plus28", "warmup", dict(WARMUP_REQUEST)))
    for p in range(1, args.passes + 1):
        for case in cases:
            body = hermes_synth.build_request(case, base)
            body["temperature"] = args.temperature
            body["seed"] = args.seed
            schedule.append((p, case["id"], case["class"], body))
        if args.include_aux_every_pass or p == 1:
            aux_body = dict(aux_title)
            aux_body["temperature"] = args.temperature
            aux_body["seed"] = args.seed
            schedule.append((p, "aux-title-cold", "aux_title", aux_body))
            if p == 1:
                schedule.append((p, "aux-title-warm", "aux_title", dict(aux_body)))

    records: list[dict] = []
    for seq, (pass_idx, case_id, case_class, body) in enumerate(schedule, start=1):
        tag = f"p{pass_idx}-{seq:02d}-{case_id}"
        (raw_dir / f"{tag}.request.json").write_text(
            json.dumps(body, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        res = send_chat_completion(
            host,
            port,
            body,
            timeout_s=args.request_timeout,
            raw_out_path=raw_dir / f"{tag}.raw",
        )
        val = evaluate_record(case_class, res, declared_tools)
        rec = {
            "seq": seq,
            "pass": pass_idx,
            "id": case_id,
            "class": case_class,
            **res,
            **val,
        }
        records.append(rec)
        print(
            f"[{args.mode}] {tag:34s} status={rec.get('http_status')} "
            f"valid={rec.get('valid')} finish={rec.get('finish_reason')} "
            f"prompt={rec.get('prompt_tokens')} cached={rec.get('cached_tokens')} "
            f"prefill={rec.get('prefill_tokens')} "
            f"comp={rec.get('completion_tokens')} (reas={rec.get('reasoning_tokens')}) "
            f"wall={rec.get('wall_ms', 0):.1f}ms ttft={rec.get('ttft_ms', 0) or 0:.1f}ms "
            f"decode={rec.get('decode_tps', 0):.2f}tok/s "
            f"draft={rec.get('draft_tokens_accepted', 0)}/{rec.get('draft_tokens', 0)}",
            flush=True,
        )

    loads = []
    if args.server_log and Path(args.server_log).is_file():
        loads = join_log_fields(records, Path(args.server_log))

    report = {
        "schema": "gufo.dflash2-eval.v1",
        "mode": args.mode,
        "temperature": args.temperature,
        "seed": args.seed,
        "passes": args.passes,
        "props": props,
        "server_loads": loads,
        "records": records,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:13345")
    parser.add_argument("--mode", choices=("off", "mtp", "dflash2", "auto"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-log", default="")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--include-aux-every-pass", action="store_true")
    args = parser.parse_args()
    return run_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
