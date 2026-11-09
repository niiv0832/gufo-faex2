#!/usr/bin/env python3
"""Bounded FAEX2 B1 harness; server ownership and retained HTTP evidence."""

from __future__ import annotations

import json
import re
import argparse
import contextlib
import hashlib
import http.client
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import platform
from urllib.parse import urlsplit

from discovery import assert_model_listing
from run import write_json

from metrics import integer_number, summarize


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_mode(log, requested, rows):
    modes = [match.group(1) for line in log.splitlines()
             if "event=load_completed" in line and "kind=text" in line
             for match in [re.search(r"\bspeculative=(\w+)", line)] if match]
    require(modes == [requested], f"requested {requested}, loaded {modes}")
    return validate_drafts(requested, rows)


def validate_drafts(requested, rows):
    proposed, accepted = 0, 0
    for row in rows:
        p, a = row.get("draft_tokens"), row.get("draft_tokens_accepted")
        require(type(p) is int and type(a) is int and 0 <= a <= p,
                "missing/invalid draft counters")
        proposed += p
        accepted += a
    if requested == "off":
        require(proposed == accepted == 0, "off executed drafts")
    elif requested == "mtp":
        require(proposed > 0 and accepted > 0, "MTP draft work not demonstrated")
    else:
        raise ValueError("unqualified speculative mode")
    return {"loaded_mode": requested, "draft_tokens": proposed,
            "draft_tokens_accepted": accepted}


def validate_exchange(case, body, raw, status, parts=None):
    """Reuse the shared wire/usage/schema contract, then check fixed B1 goldens."""
    try:
        require(status == 200, f"HTTP {status}")
        streaming = body.get("stream", False)
        if streaming:
            lines = [line[5:].strip() for line in raw.splitlines() if line.startswith(b"data:")]
            require(lines and lines[-1] == b"[DONE]" and lines.count(b"[DONE]") == 1,
                    "stream missing terminal DONE or has events after DONE")
            events = [json.loads(line) for line in lines[:-1]]
        else:
            events = [json.loads(raw)]
        require(events, "empty response")
        measured, fingerprint = summarize(parts or [(0, raw)], streaming, True,
            ("/v1/chat/completions", body, status))
        text, reasoning, tools, finish, identifier = "", "", {}, None, None
        for event in events:
            require(event.get("object") == ("chat.completion.chunk" if streaming else "chat.completion"),
                    "wrong response object")
            require(integer_number(event.get("created")) and event["created"] > 0, "missing/invalid created")
            require(event.get("model") == body["model"], "wrong response model")
            require(event.get("id") and isinstance(event["id"], str), "missing response id")
            require(identifier is None or identifier == event["id"], "mixed response ids")
            identifier = event["id"]
            for choice in event.get("choices", []):
                require(choice.get("index") == 0, "unexpected choice index")
                message = choice.get("delta" if streaming else "message", {})
                require(message.get("role", "assistant") == "assistant", "wrong role")
                require(finish is None, "choice content after finish")
                content = message.get("content") or ""
                require(isinstance(content, str), "non-text content")
                text += content
                reasoning += message.get("reasoning_content") or ""
                for index, call in enumerate(message.get("tool_calls") or []):
                    slot = call.get("index") if streaming else index
                    require(type(slot) is int and 0 <= slot <= len(tools), "invalid tool index")
                    require(call.get("type") == "function" if slot not in tools else
                            call.get("type", "function") == "function", "invalid tool type")
                    target = tools.setdefault(slot, {"id": "", "type": "function",
                        "function": {"name": "", "arguments": ""}})
                    if call.get("id"):
                        require(not target["id"], "duplicate tool id")
                        target["id"] = call["id"]
                    function = call.get("function") or {}
                    target["function"]["name"] += function.get("name") or ""
                    target["function"]["arguments"] += function.get("arguments") or ""
                finish = choice.get("finish_reason")
        require(not reasoning, "B1 reasoning must be off")
        require(measured.get("completion_tokens", 0) > 0, "empty generation")
        validate_drafts("off", [measured])
        expected = case["expect"]
        calls = list(tools.values())
        if "tool" in expected:
            require(finish == "tool_calls" and len(calls) == 1, "missing/extra tool call")
            call = calls[0]
            require(bool(call["id"]), "missing tool id")
            require(call["function"]["name"] == expected["tool"], "wrong tool name")
            require(json.loads(call["function"]["arguments"]) == expected["arguments"],
                    "wrong tool arguments")
        else:
            require(finish == "stop" and not calls and text.strip(), "incomplete/empty answer")
            if "json" in expected:
                require(json.loads(text) == expected["json"], "wrong structured answer")
            else:
                require(text.strip() == expected["text"], "wrong answer or foreign session marker")
        return {"text": text, "tools": calls, "finish": finish,
                "metrics": measured, "output_sha256": fingerprint}
    except Exception as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(str(error)) from error


ALIAS = "qwen3.6-35b-a3b"
SAMPLER = {"temperature": 1.0, "top_k": 20, "top_p": .95, "min_p": .05,
           "presence_penalty": 0.0, "frequency_penalty": 0.0,
           "repeat_penalty": 1.0, "repeat_last_n": 64}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config):
    require(config.get("stage") == "B1", "only B1 is implemented")
    for field, length in [("source_sha", 40), ("binary_sha256", 64), ("model_sha256", 64)]:
        require(isinstance(config.get(field), str) and
                re.fullmatch("[0-9a-f]{" + str(length) + "}", config[field]),
                "missing/invalid " + field)
    for field in ["binary", "model", "library_path"]:
        require(isinstance(config.get(field), str) and Path(config[field]).is_absolute(),
                "absolute path required: " + field)
    require(type(config.get("port")) is int and 1024 <= config["port"] <= 65535,
            "invalid loopback port")
    for field, default, maximum in [("lifetime_s", 900, 900), ("startup_timeout_s", 240, 240)]:
        value = config.get(field, default)
        require(type(value) in (float, int) and math.isfinite(value) and 0 < value <= maximum,
                "invalid " + field)
    for path, digest in config.get("libraries", {}).items():
        require(Path(path).is_absolute() and re.fullmatch(r"[0-9a-f]{64}", digest),
                "invalid library manifest")
    return config


@contextlib.contextmanager
def hard_timeout(seconds):
    """Wall deadline, including a peer which drips headers/body forever (Unix)."""
    def expired(_signum, _frame):
        raise DeadlineExceeded("wall deadline exceeded")
    require(seconds > 0, "deadline already expired")
    old_handler = signal.signal(signal.SIGALRM, expired)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, old_timer[0]) if old_timer[0] else seconds)
    started = time.monotonic()
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0]:
            signal.setitimer(signal.ITIMER_REAL,
                max(.000001, old_timer[0] - (time.monotonic() - started)), old_timer[1])


class DeadlineExceeded(TimeoutError):
    """Distinct from a retryable per-socket timeout during readiness."""


def probe(port, path):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=.5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        require(response.status == 200, "probe HTTP " + str(response.status))
        data = response.read(1024 * 1024 + 1)
        require(len(data) <= 1024 * 1024, "oversized discovery response")
        return json.loads(data)
    finally:
        connection.close()


def validate_props(props, config):
    require(props.get("model_alias") == ALIAS and props.get("model") == ALIAS, "wrong model alias")
    require(props.get("build_info") == "development (" + config["source_sha"] + ")", "wrong served version")
    require(props.get("model_path") == config["model"], "wrong served weights")
    require(props.get("total_slots") == 1, "wrong slot count")
    require(props.get("modalities", {}).get("vision") is False, "unexpected vision")
    defaults = props.get("default_generation_settings", {})
    require(defaults.get("n_ctx") == 8192, "wrong context")
    for field, expected in SAMPLER.items():
        got = defaults.get("params", {}).get(field)
        require(type(got) in (int, float) and math.isfinite(got) and
                math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-7),
                "wrong sampler " + field)


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)
    except ProcessLookupError:
        process.wait(timeout=3)


def linux_listener_inodes(tcp, port):
    address = f"0100007F:{port:04X}"
    return {fields[9] for line in tcp.splitlines()[1:] for fields in [line.split()]
            if len(fields) > 9 and fields[1] == address and fields[3] == "0A"}


def listener_owned(pid, port):
    """Prove the socket belongs to this PID, not just a matching API on its port."""
    if sys.platform.startswith("linux"):
        proc = Path("/proc") / str(pid)
        listening = linux_listener_inodes((proc / "net/tcp").read_text(), port)
        sockets = set()
        for fd in (proc / "fd").iterdir():
            try:
                match = re.fullmatch(r"socket:\[(\d+)\]", os.readlink(fd))
                if match: sockets.add(match.group(1))
            except FileNotFoundError:
                pass  # Another thread may close an unrelated fd while we inspect it.
        return bool(listening & sockets)
    if sys.platform == "darwin":
        result = subprocess.run(["/usr/sbin/lsof", "-nP", "-a", "-p", str(pid),
            "-iTCP@127.0.0.1:" + str(port), "-sTCP:LISTEN", "-Fpn"],
            capture_output=True, text=True, timeout=2)
        lines = result.stdout.splitlines()
        return result.returncode == 0 and "p" + str(pid) in lines and "n127.0.0.1:" + str(port) in lines
    raise ValueError("listener ownership unsupported on this platform")


def serve(config, output):
    """Own exactly one new server process; retain all files even on failure."""
    validate_config(config)
    output = Path(output)
    output.mkdir(exist_ok=False)  # Also rejects existing/dangling symlinks.
    output = output.resolve()
    report = {"status": "STARTING", "config": config, "started_at": time.time()}
    owned = []
    reservation = None
    handlers = {}
    def interrupted(signum, _frame):
        raise InterruptedError("signal " + str(signum))
    try:
        handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        write_json(output / "server-result.json", report)
        with hard_timeout(config.get("lifetime_s", 900)):
            reservation = socket.socket()
            reservation.bind(("127.0.0.1", config["port"]))
            for name in ["binary", "model"]:
                require(file_hash(config[name]) == config[name + "_sha256"], name + " hash mismatch")
            for path, digest in config.get("libraries", {}).items():
                require(file_hash(path) == digest, "library hash mismatch: " + path)
            require(Path(config["library_path"]).is_dir(), "library directory absent")
            for name in ["tmp", "cache"]: (output / name).mkdir()
            environment = {"HOME": str(Path.home()), "PATH": "/usr/bin:/bin", "LC_ALL": "C",
                "TMPDIR": str(output / "tmp"), "XDG_CACHE_HOME": str(output / "cache"),
                "HSA_ENABLE_SDMA": "0", "LD_LIBRARY_PATH": config["library_path"]}
            with (output / "version.stdout").open("w") as stdout, (output / "version.stderr").open("w") as stderr:
                process = subprocess.Popen([config["binary"], "--version"], stdout=stdout,
                    stderr=stderr, env=environment, start_new_session=True)
                owned.append(process)
                require(process.wait(timeout=15) == 0, "version command failed")
            require((output / "version.stdout").read_text() ==
                    "gufo version development (" + config["source_sha"] + ")\n", "wrong binary version")
            require(not (output / "version.stderr").read_bytes(), "unexpected version stderr")
            command = [config["binary"], "serve", "--host", "127.0.0.1", "--port", str(config["port"]),
                "--sessions", "1", "--max-request-bytes", "1048576", "llm", "--model", config["model"],
                "--served-model-name", ALIAS, "--context", "8192", "--speculative", "off", "--think", "off",
                "--cache-ram-bytes", "0", "--cache-disk-bytes", "0", "--stall-timeout-ms", "120000"]
            for name, value in SAMPLER.items(): command += ["--" + name.replace("_", "-"), str(value)]
            report.update(command=command, environment=environment)
            write_json(output / "server-result.json", report)
            with (output / "server.log").open("w") as log:
                reservation.close()
                reservation = None
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    env=environment, start_new_session=True)
                owned.append(process)
                report["pid"] = process.pid
                startup = time.monotonic()
                with hard_timeout(config.get("startup_timeout_s", 240)):
                    until = startup + config.get("startup_timeout_s", 240)
                    while True:
                        require(process.poll() is None, "server exited " + str(process.returncode))
                        try:
                            health = probe(config["port"], "/health")
                            ready = probe(config["port"], "/ready")
                        except DeadlineExceeded:
                            raise
                        except (OSError, ValueError, http.client.HTTPException):
                            health, ready = None, None
                        if health == {"status": "ok"} and ready == {"status": "ready", "model": ALIAS}:
                            break
                        if time.monotonic() >= until: raise TimeoutError("server readiness deadline exceeded")
                        time.sleep(.05)
                    props = probe(config["port"], "/props")
                    listing = probe(config["port"], "/v1/models")
                    validate_props(props, config)
                    assert_model_listing(listing, ALIAS, 8192, ["text"])
                    report.update(validate_mode((output / "server.log").read_text(), "off", []))
                    require(process.poll() is None, "owned server exited before readiness")
                    require(listener_owned(process.pid, config["port"]), "listener belongs to another PID")
                    report.update(status="READY", props=props, models=listing,
                        startup_ms=(time.monotonic() - startup) * 1000,
                        expires_at=report["started_at"] + config.get("lifetime_s", 900))
                    write_json(output / "ready.json", report)
                    write_json(output / "server-result.json", report)
                while not (output / "stop-request").exists():
                    require(process.poll() is None, "owned server exited " + str(process.returncode))
                    time.sleep(.1)
                report["status"] = "STOPPED"
    except BaseException as error:
        report.update(status="FAIL", error=type(error).__name__ + ": " + str(error))
    finally:
        if reservation is not None: reservation.close()
        for process in reversed(owned): stop_owned(process)
        report["finished_at"] = time.time()
        report["child_exit_codes"] = [process.returncode for process in owned]
        write_json(output / "server-result.json", report)
        for sig, previous in handlers.items(): signal.signal(sig, previous)
    return 0 if report["status"] == "STOPPED" else 1


def load_cases(path):
    from jsonschema import Draft202012Validator
    cases = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    identifiers = set()
    require(cases, "no test cases")
    for case in cases:
        identifier = case.get("id", "")
        require(re.fullmatch(r"[a-z][a-z0-9-]{0,63}", identifier) and identifier not in identifiers,
                "invalid/duplicate case id")
        identifiers.add(identifier)
        require(case.get("transports") and set(case["transports"]) <= {"buffered", "sse"}, "invalid transport")
        require(len(set(case["transports"])) == len(case["transports"]), "duplicate transport")
        request = case["request"]
        require(type(request.get("max_completion_tokens")) is int and
                0 < request["max_completion_tokens"] <= 256, "unbounded request")
        schema = request.get("response_format", {}).get("json_schema", {}).get("schema")
        if schema: Draft202012Validator.check_schema(schema)
        for tool in request.get("tools", []):
            Draft202012Validator.check_schema(tool["function"]["parameters"])
    return cases


def check(record, base_url, output, request_timeout=120, total_timeout=900):
    """Check the separately owned server via loopback; never starts a server."""
    for value, maximum in [(request_timeout, 120), (total_timeout, 900)]:
        require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= maximum,
                "invalid client timeout")
    target = urlsplit(base_url)
    require(target.scheme == "http" and target.hostname == "127.0.0.1" and target.port and
            not target.username and not target.password and target.path in ("", "/") and
            not target.query and not target.fragment, "only explicit loopback HTTP is allowed")
    output = Path(output)
    output.mkdir(exist_ok=False)
    report = {"stage": "B1", "status": "RUNNING", "timing_status": "RECORDED_NOT_COMPARED",
              "requests": [], "server_record": record, "started_at": time.time()}
    until = time.monotonic() + total_timeout
    def save(): write_json(output / "report.json", report)

    def exchange(identifier, endpoint, body, validator):
        require(identifier not in [row["id"] for row in report["requests"]], "duplicate request id")
        row = {"id": identifier, "endpoint": endpoint, "status": "RUNNING",
               "request_file": identifier + ".request.json", "response_file": identifier + ".response.raw"}
        report["requests"].append(row)
        write_json(output / row["request_file"], {"method": "GET" if body is None else "POST",
                                               "endpoint": endpoint, "body": body})
        save()
        connection = http.client.HTTPConnection("127.0.0.1", target.port, timeout=request_timeout)
        started = time.monotonic()
        parts, size = [], 0
        try:
            with (output / row["response_file"]).open("xb") as raw_file:
                with hard_timeout(min(request_timeout, until - time.monotonic())):
                    connection.request("GET" if body is None else "POST", endpoint,
                        None if body is None else json.dumps(body, ensure_ascii=False).encode(),
                        {"Content-Type": "application/json"})
                    response = connection.getresponse()
                    row["http_status"] = response.status
                    row["content_type"] = response.getheader("Content-Type")
                    row["request_id"] = response.getheader("X-Request-ID")
                    save()
                    while True:
                        block = response.read1(65536)
                        if not block: break
                        raw_file.write(block); raw_file.flush()
                        parts.append(((time.monotonic() - started) * 1000, block))
                        size += len(block)
                        require(size <= 8 * 1024 * 1024, "response exceeds B1 byte limit")
                    raw = b"".join(block for _, block in parts)
                    expected_type = "text/event-stream" if body and body.get("stream") else "application/json"
                    require((row["content_type"] or "").split(";", 1)[0].strip().lower() == expected_type,
                            "wrong response Content-Type")
                    result = validator(raw, response.status, parts)
            row.update(status="PASS", **result)
            row.setdefault("metrics", {})["wall_ms"] = (time.monotonic() - started) * 1000
            # B1 has reasoning off: the first streamed text is the first visible answer.
            if body and body.get("stream"):
                row["metrics"]["client_ttfa_ms"] = row["metrics"].get("client_ttft_ms")
            return result
        except BaseException as error:
            row.update(status="FAIL", error=type(error).__name__ + ": " + str(error),
                       elapsed_ms=(time.monotonic() - started) * 1000)
            raise
        finally:
            connection.close()
            save()

    try:
        save()
        config = validate_config(record["config"])
        require(record.get("status") == "READY" and record.get("loaded_mode") == "off", "invalid ready record/mode")
        require(type(record.get("pid")) is int and record["pid"] > 0, "missing owned server pid")
        require(time.time() < record["expires_at"], "server window expired")
        from importlib.metadata import version
        require(version("jsonschema") == "4.26.0", "use pinned jsonschema==4.26.0")
        report["client"] = {"python": platform.python_version(), "jsonschema": version("jsonschema")}
        root = Path(__file__).parent
        cases_path = root / "faex2_cases.jsonl"
        report["harness_files"] = {name: file_hash(root / name) for name in
            ["faex2.py", "metrics.py", "discovery.py", "run.py"]}
        report["fixture_sha256"] = file_hash(cases_path)
        cases = load_cases(cases_path)
        def discovery(endpoint, raw, status, _parts):
            require(status == 200, "discovery HTTP " + str(status))
            value = json.loads(raw)
            if endpoint == "/health": require(value == {"status": "ok"}, "bad health")
            elif endpoint == "/ready": require(value == {"status": "ready", "model": ALIAS}, "bad ready")
            elif endpoint == "/props": validate_props(value, config)
            else: assert_model_listing(value, ALIAS, 8192, ["text"])
            return {"value": value}
        for endpoint in ["/health", "/ready", "/props", "/v1/models"]:
            exchange(endpoint.strip("/").replace("/", "-"), endpoint, None,
                     lambda raw, status, parts, path=endpoint: discovery(path, raw, status, parts))
        for case in cases:
            for transport in case["transports"]:
                body = {**case["request"], "model": ALIAS, "stream": transport == "sse",
                        "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
                if body["stream"]: body["stream_options"] = {"include_usage": True}
                result = exchange(case["id"] + "-" + transport, "/v1/chat/completions", body,
                    lambda raw, status, parts: validate_exchange(case, body, raw, status, parts))
                if "tool_result" in case:
                    # Static fixture only: generated tool names/arguments are never executed.
                    followup = {**body, "stream": True, "stream_options": {"include_usage": True},
                        "tool_choice": "none", "messages": [*body["messages"],
                        {"role": "assistant", "content": result["text"] or None, "tool_calls": result["tools"]},
                        {"role": "tool", "tool_call_id": result["tools"][0]["id"],
                         "content": json.dumps(case["tool_result"], ensure_ascii=False)}]}
                    expected = {"expect": case["followup_expect"]}
                    exchange(case["id"] + "-followup-" + transport if transport == "sse" else
                             case["id"] + "-followup-sse", "/v1/chat/completions", followup,
                        lambda raw, status, parts: validate_exchange(expected, followup, raw, status, parts))
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=type(error).__name__ + ": " + str(error))
    finally:
        report["finished_at"] = time.time()
        save()
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    server = sub.add_parser("serve", help="owned B1 server; requires separate model approval")
    server.add_argument("--config", type=Path, required=True)
    server.add_argument("--output", type=Path, required=True)
    check_config = sub.add_parser("validate-config", help="syntax only; no binary/weights execution")
    check_config.add_argument("--config", type=Path, required=True)
    client = sub.add_parser("check", help="B1 API checks against the owned loopback server")
    client.add_argument("--server-record", type=Path, required=True)
    client.add_argument("--base-url", required=True)
    client.add_argument("--output", type=Path, required=True)
    client.add_argument("--request-timeout", type=float, default=120)
    client.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    try:
        if args.action == "check":
            return check(json.loads(args.server_record.read_text()), args.base_url, args.output,
                         args.request_timeout, args.timeout)
        config = validate_config(json.loads(args.config.read_text()))
        if args.action == "validate-config":
            print("CONFIG_VALID_NOT_EXECUTED")
            return 0
        return serve(config, args.output)
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
