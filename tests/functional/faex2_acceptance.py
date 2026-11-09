#!/usr/bin/env python3
"""Focused FAEX2 acceptance; content gates are separate from wire transport."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from faex2 import ALIAS, file_hash, hard_timeout, load_cases, profile, require, validate_drafts, validate_props
from discovery import assert_model_listing
from faex2_images import image_content, text_image_content
from image_inputs import image_cases
from run import write_json


def rubric_pass(scores, critical_failures):
    names = ("correctness", "instruction", "russian_clarity", "explanation")
    return (not critical_failures and set(scores) == set(names)
            and all(type(scores[n]) is int and 1 <= scores[n] <= 2 for n in names)
            and scores["correctness"] == 2 and sum(scores.values()) >= 6)


def grade(result, expected, mode):
    """Never promote partial, foreign, skipped or merely HTTP-200 evidence."""
    if expected.get("cancelled"):
        require(result.get("status") == "CANCELLED" and result.get("http_status") == 200
                and result.get("text", "").strip(), "intentional content cancellation absent")
        return
    status = expected.get("http_status", 200)
    require(result.get("http_status") == status, "unexpected HTTP status")
    if status != 200:
        require(result.get("status") == "HTTP_ERROR", "HTTP error not recorded")
        error = result.get("error", {})
        require(isinstance(error, dict) and error.get("code") and error.get("message"),
                "missing structured error")
        require(expected.get("error_contains", "").lower() in error["message"].lower(),
                "wrong error message")
        return
    require(result.get("status") == "PASS", "transport did not pass")
    if "discovery" in expected:
        value = result.get("response")
        kind = expected["discovery"]
        if kind == "health":
            require(value == {"status": "ok"}, "bad health")
        elif kind == "ready":
            require(value == {"status": "ready", "model": ALIAS}, "bad ready")
        elif kind == "props":
            validate_props(value, expected["config"])
        elif kind == "models":
            settings = profile(expected["config"])
            assert_model_listing(value, ALIAS, settings["context"],
                                 ["text", "image"] if settings["vision"] else ["text"])
        else:
            raise ValueError("unknown discovery check")
        return
    metrics = result.get("metrics", {})
    if expected.get("fresh_prefill"):
        require(type(metrics.get("prompt_tokens")) is int and metrics["prompt_tokens"] > 0
                and metrics.get("cached_tokens") == 0
                and metrics.get("prefill_tokens") == metrics["prompt_tokens"],
                "fresh image prefill absent")
    # C4 may legitimately use the cohort gate; C1 explicitly demands draft work.
    if mode == "off" or expected.get("draft_work"):
        validate_drafts(mode, [metrics])
    else:
        p, a = metrics.get("draft_tokens"), metrics.get("draft_tokens_accepted")
        require(type(p) is int and type(a) is int and 0 <= a <= p,
                "invalid draft counters")
    text = result.get("text", "").strip()
    tools = result.get("tools", [])
    if "tool" in expected:
        require(result.get("finish") == "tool_calls" and len(tools) == 1,
                "missing/extra tool")
        call = tools[0]
        require(call.get("id") and call.get("type") == "function", "invalid tool identity")
        require(call["function"]["name"] == expected["tool"], "wrong tool")
        require(json.loads(call["function"]["arguments"]) == expected["arguments"],
                "wrong tool arguments")
    else:
        require(result.get("finish") == expected.get("finish", "stop") and text and not tools,
                "empty or incomplete visible answer")
        if "text" in expected:
            require(text == expected["text"], "wrong answer")
        if "json" in expected:
            require(json.loads(text) == expected["json"], "wrong JSON answer")
        for value in expected.get("contains", []):
            require(value in text, "missing expected content: " + value)
        for value in expected.get("excludes", []):
            require(value not in text, "foreign/forbidden content: " + value)
    reasoning = result.get("reasoning", "")
    if expected.get("reasoning") is True:
        require(reasoning, "reasoning-on work absent")
    elif not expected.get("allow_reasoning", False):
        require(not reasoning, "unexpected reasoning")
    if "reasoning_budget" in expected:
        count = result.get("usage", {}).get("output_tokens_details", {}).get("reasoning_tokens")
        require(type(count) is int and 0 < count <= expected["reasoning_budget"],
                "missing/out-of-bound reasoning token count")
    for field, bound in (("prompt_tokens", "min_prompt_tokens"),
                         ("cached_tokens", "min_cached_tokens")):
        if bound in expected:
            require(type(metrics.get(field)) is int and metrics[field] >= expected[bound],
                    "insufficient " + field)
    for field, bound in (("prompt_tokens", "max_prompt_tokens"),
                         ("prefill_tokens", "max_prefill_tokens")):
        if bound in expected:
            require(type(metrics.get(field)) is int and metrics[field] <= expected[bound],
                    "excessive " + field)
    if expected.get("disk_hit"):
        detail = result.get("usage", {}).get("gufo", {})
        require(detail.get("cache_disk_hit") is True and metrics.get("cached_tokens", 0) > 0,
                "disk cache restore absent")


class Runner:
    """Each concurrent request has a separate process, deadline and raw directory."""
    def __init__(self, base_url, output, mode, transport_command=None):
        require(mode in ("off", "mtp"), "unsupported mode")
        self.base_url, self.mode = base_url, mode
        self.output = Path(output)
        self.output.mkdir(exist_ok=False)
        (self.output / "inputs").mkdir()
        (self.output / "requests").mkdir()
        self.command = transport_command or [sys.executable,
            str(Path(__file__).with_name("faex2_transport.py"))]
        self.identifiers = set()
        self.report = {"status": "RUNNING", "mode": mode, "requests": [],
                       "started_at": time.time(), "timing_status": "RECORDED_NOT_COMPARED"}
        self.save()

    def save(self):
        write_json(self.output / "report.json", self.report)

    def reserve(self, identifier):
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", identifier), "invalid request id")
        require(identifier not in self.identifiers, "duplicate request id")
        self.identifiers.add(identifier)

    def exchange(self, identifier, request, expected, timeout=120, cancel=None):
        require(type(timeout) in (int, float) and 0 < timeout <= 600, "invalid request timeout")
        input_path = self.output / "inputs" / (identifier + ".json")
        with input_path.open("x") as out:
            json.dump(request, out, ensure_ascii=False, indent=2)
        output = self.output / "requests" / identifier
        command = [*self.command, "request", "--base-url", self.base_url,
                   "--request", str(input_path), "--output", str(output), "--timeout", str(timeout)]
        if cancel is not None:
            command += ["--cancel-after-content", str(cancel)]
        row = {"id": identifier, "status": "RUNNING", "expected": expected,
               "started_at": time.time(),
               "request_file": str(input_path.relative_to(self.output)),
               "result_file": str((output / "result.json").relative_to(self.output)),
               "command": command}
        try:
            with (self.output / (identifier + ".stdout")).open("xb") as stdout, \
                 (self.output / (identifier + ".stderr")).open("xb") as stderr:
                proc = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout + 10)
            row["exit"] = proc.returncode
            result = json.loads((output / "result.json").read_text())
            row["result"] = result
            require(proc.returncode == 0, "transport process failed")
            grade(result, expected, self.mode)
            row["status"] = "NEEDS_REVIEW" if expected.get("manual_quality") else "PASS"
        except Exception as error:
            row.update(status="FAIL", error=type(error).__name__ + ": " + str(error))
        row["finished_at"] = time.time()
        return row

    def record(self, rows):
        self.report["requests"].extend(rows)
        failures = [row for row in rows if row["status"] == "FAIL"]
        if failures:
            self.report["status"] = "FAIL"
        self.save()

        require(not failures, "; ".join(row["id"] + ": " + row["error"] for row in failures))
        return [row["result"] for row in rows]

    def run_case(self, identifier, request, expected, timeout=120, cancel=None):
        self.reserve(identifier)
        return self.record([self.exchange(identifier, request, expected, timeout, cancel)])[0]

    def run_batch(self, cases, timeout=120):
        require(1 <= len(cases) <= 4, "batch must have one to four requests")
        for identifier, _, _ in cases:
            self.reserve(identifier)
        with ThreadPoolExecutor(max_workers=len(cases)) as pool:
            futures = [pool.submit(self.exchange, identifier, request, expected, timeout)
                       for identifier, request, expected in cases]
            rows = [future.result() for future in futures]
        return self.record(rows)

    def finish(self):
        if self.report["status"] != "FAIL":
            self.report["status"] = ("NEEDS_REVIEW" if any(
                row["status"] == "NEEDS_REVIEW" for row in self.report["requests"]) else "PASS")
        self.report["finished_at"] = time.time()
        self.save()


def chat_request(body, streaming=False):
    payload = {"model": ALIAS, "cache_prompt": False,
               "chat_template_kwargs": {"enable_thinking": False},
               **body, "stream": streaming}
    if streaming:
        payload["stream_options"] = {"include_usage": True}
    return {"method": "POST", "path": "/v1/chat/completions", "body": payload}


def tool_followup(body, result, tool_result):
    return {**body, "stream": True, "stream_options": {"include_usage": True},
            "tool_choice": "none", "messages": [*body["messages"],
            {"role": "assistant", "content": result["text"] or None, "tool_calls": result["tools"]},
            {"role": "tool", "tool_call_id": result["tools"][0]["id"],
             "content": json.dumps(tool_result, ensure_ascii=False)}]}


def b2_visual_cases():
    def visual(identifier, images, question, expected, streaming=False):
        body = {"messages": [{"role": "user", "content": [*images,
                {"type": "text", "text": question}]}], "max_completion_tokens": 64}
        if expected.get("http_status", 200) == 200:
            expected = {**expected, "fresh_prefill": True}
        return identifier, chat_request(body, streaming), expected
    color = "Name the dominant color in this image. Reply with one lowercase English color name only."
    cases = [visual("vision-" + name, [{"type": "image_url", "image_url": {"url": url}}],
                    color, {"text": "red"}, name in ("jpeg", "webp_lossy"))
             for name, url in image_cases(image_content)[:4]]
    cases.append(visual("vision-text", [text_image_content()],
                        "Read the text in this image. Reply with the text only, preserving the space.",
                        {"text": "GUFO 742"}))
    question = ("Name the dominant color of each image, in the order shown. "
                "Reply with two lowercase English color names separated by a comma, without spaces.")
    for name, colors in (("forward", ("red", "blue")), ("reverse", ("blue", "red"))):
        cases.append(visual("vision-two-" + name, [image_content(c) for c in colors],
                            question, {"text": ",".join(colors)}))
    cases.append(visual("vision-bad-base64", [{"type": "image_url",
                         "image_url": {"url": "data:image/png;base64,A==="}}], color,
                         {"http_status": 400, "error_contains": "base64"}))
    cases.append(visual("vision-recovery", [image_content("blue")], color, {"text": "blue"}, True))
    return cases


def check_ready(runner, record, stage):
    require(record.get("status") == "READY" and record.get("loaded_mode") == runner.mode,
            "invalid server readiness/mode")
    require(type(record.get("pid")) is int and record["pid"] > 0, "missing owned PID")
    require(time.time() < record["expires_at"], "server window expired")
    config = record["config"]
    require(config.get("stage") == stage, "wrong acceptance stage")
    runner.report["server_record"] = record
    runner.save()
    for path, kind in (("/health", "health"), ("/ready", "ready"),
                       ("/props", "props"), ("/v1/models", "models")):
        runner.run_case("discovery-" + kind, {"method": "GET", "path": path, "body": None},
                        {"discovery": kind, "config": config})


def run_b2(runner):
    root = Path(__file__).parent
    quality_path = root / "faex2_quality.jsonl"
    quality = [json.loads(line) for line in quality_path.read_text().splitlines() if line.strip()]
    runner.report["fixture_sha256"] = {name: file_hash(root / name) for name in
        ("faex2_cases.jsonl", "faex2_quality.jsonl", "faex2_images.py", "image_inputs.py")}
    write_json(runner.output / "quality-rubric.json", quality)
    runner.save()
    for case in load_cases(root / "faex2_cases.jsonl"):
        for transport in case["transports"]:
            request = chat_request(case["request"], transport == "sse")
            result = runner.run_case(case["id"] + "-" + transport, request, case["expect"])
            if "tool_result" in case:
                followup = tool_followup(request["body"], result, case["tool_result"])
                runner.run_case("tool-followup-sse", chat_request(followup, True), case["followup_expect"])
    runner.run_case("mtp-russian-long", chat_request({"messages": [{"role": "user", "content":
        "Объясни школьнику, как проверять ответ в задаче на скорость, время и расстояние. "
        "Напиши 8–10 полных предложений по-русски, с примером 150 км за 3 часа. "
        "Объясни единицы и обратную проверку."}], "max_completion_tokens": 512}, True),
        {"draft_work": True, "manual_quality": True, "contains": ["150", "50"]})
    for identifier, request, expected in b2_visual_cases():
        runner.run_case(identifier, request, expected)
    for case in quality:
        request = chat_request({"messages": case.get("messages", [{"role": "user", "content": case["prompt"]}]),
                                "max_completion_tokens": case["max_completion_tokens"]}, True)
        runner.run_case("quality-" + case["id"], request, {"manual_quality": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["b2"])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--server-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runner = Runner(args.base_url, args.output, "mtp")
    try:
        with hard_timeout(1500):
            check_ready(runner, json.loads(args.server_record.read_text()), "B2")
            run_b2(runner)
    except BaseException as error:
        runner.report.update(status="FAIL", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        runner.finish()


if __name__ == "__main__":
    main()
