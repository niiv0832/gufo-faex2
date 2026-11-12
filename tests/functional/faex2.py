#!/usr/bin/env python3
"""Bounded FAEX2 profiles; server ownership and retained HTTP evidence."""

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
import threading
from urllib.parse import urlsplit

from discovery import assert_model_listing
from run import write_json

from metrics import integer_number, summarize


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_mode(log, requested, rows):
    loaded_mode(log, requested)
    return validate_drafts(requested, rows)


def loaded_mode(log, requested):
    modes = [match.group(1) for line in log.splitlines()
             if "event=load_completed" in line and "kind=text" in line
             for match in [re.search(r"\bspeculative=(\w+)", line)] if match]
    require(modes == [requested], f"requested {requested}, loaded {modes}")
    return {"loaded_mode": requested}


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


def binary_version(config):
    if "expected_version" in config:
        require(config.get("stage") == "C" and config["expected_version"] == "gufo version development (unknown)\n",
                "legacy version is an explicit comparison-only exception")
        return config["expected_version"]
    return "gufo version development (" + config["source_sha"] + ")\n"


def validate_config(config):
    require(isinstance(config, dict), "config must be an object")
    stage = config.get("stage")
    require(stage in ("B1", "B2", "B3", "C"), "unsupported stage")
    if "expected_version" in config:
        binary_version(config)
    for field, length in [("source_sha", 40), ("binary_sha256", 64), ("model_sha256", 64)]:
        require(isinstance(config.get(field), str) and
                re.fullmatch("[0-9a-f]{" + str(length) + "}", config[field]),
                "missing/invalid " + field)
    for field in ["binary", "model", "library_path"]:
        require(isinstance(config.get(field), str) and Path(config[field]).is_absolute(),
                "absolute path required: " + field)
    require(type(config.get("port")) is int and 1024 <= config["port"] <= 65535,
            "invalid loopback port")
    lifetime = {"B1": 900, "B2": 1800, "B3": 7200, "C": 7200}[stage]
    for field, default, maximum in [("lifetime_s", lifetime, lifetime), ("startup_timeout_s", 240, 240)]:
        value = config.get(field, default)
        require(type(value) in (float, int) and math.isfinite(value) and 0 < value <= maximum,
                "invalid " + field)
    require(isinstance(config.get("libraries", {}), dict), "invalid libraries manifest")
    for path, digest in config.get("libraries", {}).items():
        require(isinstance(path, str) and Path(path).is_absolute() and isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest),
                "invalid library manifest")
    if stage != "B1":
        allowed = {"stage", "source_sha", "binary", "binary_sha256", "model", "model_sha256",
                   "library_path", "libraries", "port", "lifetime_s", "startup_timeout_s",
                   "mode", "sessions", "context", "vision", "mmproj", "mmproj_sha256", "think",
                   "reasoning_budget", "draft_tokens", "cache_ram_bytes", "cache_disk_bytes",
                   "cache_disk", "max_request_bytes", "run_root", "expected_version"}
        require(not (set(config) - allowed), "unsupported advanced config fields")
        require(config.get("mode") in ("off", "mtp"), "invalid speculative mode")
        require(type(config.get("sessions")) is int and config["sessions"] in (1, 2, 4), "invalid sessions")
        require(type(config.get("context")) is int and config["context"] in (8192, 131072), "invalid context")
        require(config.get("vision") is True, "advanced profiles require vision")
        require(isinstance(config.get("mmproj"), str) and Path(config["mmproj"]).is_absolute(),
                "absolute path required: mmproj")
        require(isinstance(config.get("mmproj_sha256"), str) and
                re.fullmatch(r"[0-9a-f]{64}", config["mmproj_sha256"]), "missing/invalid mmproj_sha256")
        require(config.get("think") in ("off", "on"), "invalid think mode")
        require(type(config.get("reasoning_budget")) is int and config["reasoning_budget"] == 512,
                "reasoning_budget must be 512")
        require(type(config.get("draft_tokens", 1)) is int and config.get("draft_tokens", 1) == 1,
                "draft_tokens must be 1")
        for field, minimum, maximum in [("cache_ram_bytes", 1, 16*1024**3),
                                        ("cache_disk_bytes", 0, 64*1024**3),
                                        ("max_request_bytes", 1, 64*1024**2)]:
            require(type(config.get(field)) is int and minimum <= config[field] <= maximum,
                    "invalid " + field)
        require(isinstance(config.get("run_root"), str) and Path(config["run_root"]).is_absolute(),
                "absolute path required: run_root")
        root = Path(config["run_root"])
        require(root.is_dir() and not root.is_symlink() and ".." not in root.parts, "invalid run_root")
        if stage == "B2":
            require((config["mode"], config["sessions"], config["context"]) == ("mtp", 1, 8192),
                    "B2 requires mtp, 1 session, 8192 context")
            require(config["cache_disk_bytes"] == 0, "B2 disk cache must be disabled")
        if config["cache_disk_bytes"]:
            require(isinstance(config.get("cache_disk"), str), "cache_disk required")
            owned_directory(config["cache_disk"], root, inspect_contents=True)
        else:
            require("cache_disk" not in config, "disabled disk cache cannot have a path")
    return config


def owned_directory(path, root, inspect_contents=False):
    """Reject escapes and symlinks without touching an existing cache."""
    path, root = Path(path), Path(root)
    require(path.is_absolute() and ".." not in path.parts, "absolute owned path required")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("path is outside run_root") from error
    require(relative.parts, "path must be below run_root")
    current = root
    for part in relative.parts:
        current /= part
        require(not current.is_symlink(), "owned path contains symlink")
        require(not current.exists() or current.is_dir(), "owned path is not a directory")
    require(path.resolve().is_relative_to(root.resolve()), "resolved path escapes run_root")
    if inspect_contents and path.exists():
        for parent, directories, files in os.walk(path, followlinks=False):
            for name in directories + files:
                require(not (Path(parent) / name).is_symlink(), "cache contains symlink")
    return path


def profile(config):
    if config["stage"] == "B1":
        return {"sessions": 1, "context": 8192, "mode": "off", "think": "off",
                "vision": False, "cache_ram_bytes": 0, "cache_disk_bytes": 0,
                "max_request_bytes": 1048576, "lifetime_s": config.get("lifetime_s", 900)}
    return {**config, "lifetime_s": config.get("lifetime_s", 1800 if config["stage"] == "B2" else 7200)}


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
    settings = profile(config)
    require(props.get("model_alias") == ALIAS and props.get("model") == ALIAS, "wrong model alias")
    require(props.get("build_info") == binary_version(config).removeprefix("gufo version ").rstrip("\n"),
            "wrong served version")
    require(props.get("model_path") == config["model"], "wrong served weights")
    require(props.get("total_slots") == settings["sessions"], "wrong slot count")
    require(props.get("modalities", {}).get("vision") is settings["vision"], "wrong vision mode")
    defaults = props.get("default_generation_settings", {})
    require(defaults.get("n_ctx") == settings["context"], "wrong context")
    for field, expected in SAMPLER.items():
        got = defaults.get("params", {}).get(field)
        require(type(got) in (int, float) and math.isfinite(got) and
                math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-7),
                "wrong sampler " + field)
    if config["stage"] != "B1":
        for source in [props, defaults, defaults.get("params", {})]:
            for field in ["reasoning_budget", "reasoning_budget_tokens", "thinking_budget_tokens"]:
                if field in source:
                    require(type(source[field]) is int and source[field] == 512, "wrong reasoning budget")


def validate_loaded_profile(log, config):
    if config["stage"] == "B1":
        return validate_mode(log, "off", [])
    result = loaded_mode(log, config["mode"])
    loads = [line for line in log.splitlines() if "event=load_completed" in line and "kind=text" in line]
    require(len(loads) == 1 and re.search(r"\bdraft_limit=1\b", loads[0]), "wrong loaded draft limit")
    thinking = [match.group(1) for line in log.splitlines() if "event=defaults" in line
                for match in [re.search(r"\bthinking=(\w+)", line)] if match]
    require(thinking == [config["think"]], "wrong loaded thinking mode")
    for match in re.finditer(r"\b(?:reasoning_budget|reasoning_budget_tokens|thinking_budget_tokens)=(-?\d+)", log):
        require(match.group(1) == "512", "wrong loaded reasoning budget")
    vision = [int(match.group(1)) for line in log.splitlines() if "event=vision_prepared" in line
              for match in [re.search(r"\bresident_bytes=(\d+)\b", line)] if match]
    require(len(vision) == 1 and vision[0] > 0, "uploaded vision projector absent")
    return {**result, "loaded_draft_limit": 1, "loaded_thinking": thinking[0],
            "vision_resident_bytes": vision[0],
            "draft_coverage": "NOT_YET_MEASURED"}


def memory_snapshot(pid, proc_root=None, sys_root=None):
    """Read-only Linux metrics; explicit roots support CPU fixture checks."""
    row = {"timestamp": time.time(), "monotonic_s": time.monotonic(), "pid": pid}
    if proc_root is None and not sys.platform.startswith("linux"):
        return {**row, "status": "UNSUPPORTED_PLATFORM", "rss_bytes": None,
                "mem_available_bytes": None, "gtt_used_bytes": None, "temperatures_c": {}}
    proc_root, sys_root = Path(proc_root or "/proc"), Path(sys_root or "/sys")
    def kib(path, field):
        found = re.search(r"^" + field + r":\s+(\d+)\s+kB$", path.read_text(), re.MULTILINE)
        require(found is not None, "missing memory metric: " + field)
        return int(found.group(1)) * 1024
    devices = sorted(set(path.resolve() for path in
                         sys_root.glob("class/drm/card[0-9]*/device/mem_info_gtt_used")))
    require(devices, "GTT memory metrics unavailable")
    gtt = [int(path.read_text().strip()) for path in devices]
    require(all(value >= 0 for value in gtt), "invalid GTT memory metric")
    temperatures = {}
    for path in sys_root.glob("class/drm/card[0-9]*/device/hwmon/hwmon*/temp*_input"):
        try:
            temperatures[str(path)] = int(path.read_text().strip()) / 1000
        except (OSError, ValueError):
            pass  # Temperature sensors are diagnostic; RAM/GTT are mandatory guards.
    return {**row, "status": "MEASURED", "rss_bytes": kib(proc_root / str(pid) / "status", "VmRSS"),
            "mem_available_bytes": kib(proc_root / "meminfo", "MemAvailable"),
            "gtt_used_bytes": sum(gtt), "temperatures_c": temperatures}


class ResourceMonitor:
    """Retain a sample every two seconds and stop only the owned process on risk."""
    def __init__(self, process, path):
        self.process, self.path = process, Path(path)
        self.error = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, name="faex2-memory-monitor", daemon=True)

    def start(self):
        self.thread.start()

    def run(self):
        try:
            with self.path.open("x") as output:
                while not self.stopped.is_set() and self.process.poll() is None:
                    started = time.monotonic()
                    row = memory_snapshot(self.process.pid)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    if row.get("status") != "UNSUPPORTED_PLATFORM":
                        require(row["mem_available_bytes"] >= 8*1024**3, "MemAvailable below 8 GiB")
                        require(row["gtt_used_bytes"] <= 105*1024**3, "GTT usage above 105 GiB")
                    self.stopped.wait(max(0, 2 - (time.monotonic() - started)))
        except Exception as error:
            if self.stopped.is_set() or self.process.poll() is not None:
                return
            self.error = type(error).__name__ + ": " + str(error)
            try:
                with self.path.open("a") as output:
                    output.write(json.dumps({"timestamp": time.time(), "pid": self.process.pid,
                                             "status": "FAIL", "error": self.error}) + "\n")
            except OSError:
                pass  # The parent retains self.error even when the diagnostic disk is unavailable.
            stop_owned(self.process)

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=7)


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
    settings = profile(config)
    advanced = config["stage"] != "B1"
    output = Path(output)
    if advanced:
        owned_directory(output, config["run_root"])
        if config["cache_disk_bytes"]:
            cache = Path(config["cache_disk"]).resolve()
            require(not cache.is_relative_to(output.resolve()) and not output.resolve().is_relative_to(cache),
                    "disk cache must be separate from attempt output")
    output.mkdir(exist_ok=False)  # Also rejects existing/dangling symlinks.
    output = output.resolve()
    report = {"status": "STARTING", "config": config, "started_at": time.time()}
    owned = []
    reservation = None
    monitor = None
    handlers = {}
    def interrupted(signum, _frame):
        raise InterruptedError("signal " + str(signum))
    try:
        handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        write_json(output / "server-result.json", report)
        with hard_timeout(settings["lifetime_s"]):
            reservation = socket.socket()
            # Permit our restart after TIME_WAIT, never an existing listener.
            reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            reservation.bind(("127.0.0.1", config["port"]))
            for name in (["binary", "model", "mmproj"] if advanced else ["binary", "model"]):
                require(file_hash(config[name]) == config[name + "_sha256"], name + " hash mismatch")
            for path, digest in config.get("libraries", {}).items():
                require(file_hash(path) == digest, "library hash mismatch: " + path)
            require(Path(config["library_path"]).is_dir(), "library directory absent")
            for name in ["tmp", "cache"]: (output / name).mkdir()
            environment = {"HOME": str(Path.home()), "PATH": "/usr/bin:/bin", "LC_ALL": "C",
                "TMPDIR": str(output / "tmp"), "XDG_CACHE_HOME": str(output / "cache"),
                "HSA_ENABLE_SDMA": "0", "LD_LIBRARY_PATH": config["library_path"]}
            if advanced:
                environment.update(GUFO_MTP_MOE_MAX_COHORT="4")
                if config["cache_disk_bytes"]:
                    cache = owned_directory(config["cache_disk"], config["run_root"], inspect_contents=True)
                    cache.mkdir(parents=True, exist_ok=True)
            with (output / "version.stdout").open("w") as stdout, (output / "version.stderr").open("w") as stderr:
                process = subprocess.Popen([config["binary"], "--version"], stdout=stdout,
                    stderr=stderr, env=environment, start_new_session=True)
                owned.append(process)
                require(process.wait(timeout=15) == 0, "version command failed")
            require((output / "version.stdout").read_text() ==
                    binary_version(config),
                    "wrong binary version")
            require(not (output / "version.stderr").read_bytes(), "unexpected version stderr")
            command = [config["binary"], "serve", "--host", "127.0.0.1", "--port", str(config["port"]),
                "--sessions", str(settings["sessions"]), "--max-request-bytes", str(settings["max_request_bytes"]),
                "llm", "--model", config["model"], "--served-model-name", ALIAS,
                "--context", str(settings["context"]), "--speculative", settings["mode"], "--think", settings["think"],
                "--cache-ram-bytes", str(settings["cache_ram_bytes"]), "--cache-disk-bytes", str(settings["cache_disk_bytes"]),
                "--stall-timeout-ms", "120000"]
            if advanced:
                command += ["--mmproj", config["mmproj"], "--draft-tokens", "1", "--reasoning-budget", "512"]
                if settings["mode"] == "mtp": command += ["--mtp-model", "embedded"]
                if config["cache_disk_bytes"]: command += ["--cache-disk", config["cache_disk"]]
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
                if advanced:
                    report["monitor_file"] = "monitor.jsonl"
                    monitor = ResourceMonitor(process, output / "monitor.jsonl")
                    monitor.start()
                startup = time.monotonic()
                with hard_timeout(config.get("startup_timeout_s", 240)):
                    until = startup + config.get("startup_timeout_s", 240)
                    while True:
                        if monitor: require(not monitor.error, monitor.error)
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
                    assert_model_listing(listing, ALIAS, settings["context"], ["text", "image"] if settings["vision"] else ["text"])
                    report.update(validate_loaded_profile((output / "server.log").read_text(), config))
                    if monitor: require(not monitor.error, monitor.error)
                    require(process.poll() is None, "owned server exited before readiness")
                    require(listener_owned(process.pid, config["port"]), "listener belongs to another PID")
                    report.update(status="READY", props=props, models=listing,
                        startup_ms=(time.monotonic() - startup) * 1000,
                        expires_at=report["started_at"] + settings["lifetime_s"])
                    write_json(output / "ready.json", report)
                    write_json(output / "server-result.json", report)
                while not (output / "stop-request").exists():
                    if monitor: require(not monitor.error, monitor.error)
                    require(process.poll() is None, "owned server exited " + str(process.returncode))
                    time.sleep(.1)
                report["status"] = "STOPPED"
    except BaseException as error:
        report.update(status="FAIL", error=type(error).__name__ + ": " + str(error))
    finally:
        if reservation is not None: reservation.close()
        if monitor:
            monitor.close()
            if monitor.error: report.update(status="FAIL", error=monitor.error)
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
    server = sub.add_parser("serve", help="owned bounded B1/B2/B3/C server")
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
