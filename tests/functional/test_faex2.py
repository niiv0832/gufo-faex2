"""CPU-only contracts for FAEX2; never launches the real inference binary."""

import copy
import json
import hashlib
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import faex2


def response(text="391"):
    return {
        "id": "chat-test", "object": "chat.completion", "created": 1,
        "model": "qwen3.6-35b-a3b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22,
                  "cached_tokens": 0, "draft_tokens": 0, "draft_tokens_accepted": 0,
                  "gufo": {"prefill_tokens": 20, "prefill_ms": 2.5, "decode_ms": 3.0,
                           "queue_ms": 0, "ttft_ms": 3, "cache_restore_ms": 0,
                           "cache_snapshot_ms": 0, "cache_disk_enqueue_ms": 0}},
    }


class ExchangeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(getattr(faex2, "validate_exchange", None)),
                        "B1 exchange validator is not implemented")
        self.case = {"id": "arithmetic", "expect": {"text": "391"}}
        self.body = {"model": "qwen3.6-35b-a3b", "messages": [],
                     "max_completion_tokens": 64, "stream": False}

    def validate(self, value, case=None, body=None):
        return faex2.validate_exchange(case or self.case, body or self.body,
                                       json.dumps(value).encode(), 200)

    def test_valid_answer_keeps_measured_metrics(self):
        result = self.validate(response())
        self.assertEqual(result["text"], "391")
        self.assertEqual(result["metrics"]["prefill_ms"], 2.5)

    def test_empty_truncated_or_foreign_marker_cannot_pass(self):
        for text in ["", "390", "391 чужой-маркер"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.validate(response(text))
        value = response(); value["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ValueError): self.validate(value)

    def test_corrupt_response_envelope_cannot_pass(self):
        for field, value in [("object", "garbage"), ("created", None), ("created", True)]:
            broken = response(); broken[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.validate(broken)

    def test_missing_metric_or_bad_usage_cannot_pass(self):
        for mutation in ["timing", "count", "draft"]:
            value = response()
            if mutation == "timing": del value["usage"]["gufo"]["prefill_ms"]
            elif mutation == "count": value["usage"]["total_tokens"] = 99
            else: del value["usage"]["draft_tokens"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(value)

    def test_off_rejects_any_draft_work(self):
        value = response(); value["usage"]["draft_tokens"] = 1
        with self.assertRaises(ValueError): self.validate(value)

    def test_schema_rejects_extra_or_wrong_fields(self):
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}},
                  "required": ["answer"], "additionalProperties": False}
        body = {**self.body, "response_format": {"type": "json_schema", "json_schema": {
            "name": "answer", "strict": True, "schema": schema}}}
        case = {"id": "json", "expect": {"json": {"answer": 391}}}
        self.assertEqual(self.validate(response('{"answer":391}'), case, body)["text"],
                         '{"answer":391}')
        for text in ['{"answer":"391"}', '{"answer":391,"extra":1}', '{"answer":390}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.validate(response(text), case, body)

    def test_tool_requires_exact_name_arguments_and_id(self):
        definition = {"name": "lookup", "strict": True, "parameters": {
            "type": "object", "properties": {"topic": {"type": "string"}},
            "required": ["topic"], "additionalProperties": False}}
        body = {**self.body, "tools": [{"type": "function", "function": definition}],
                "tool_choice": {"type": "function", "function": {"name": "lookup"}}}
        case = {"id": "tool", "expect": {"tool": "lookup", "arguments": {"topic": "школа"}}}
        value = response(None); value["choices"][0]["finish_reason"] = "tool_calls"
        value["choices"][0]["message"]["tool_calls"] = [{"id": "call_1", "type": "function",
            "function": {"name": "lookup", "arguments": '{"topic":"школа"}'}}]
        self.assertEqual(self.validate(value, case, body)["tools"][0]["id"], "call_1")
        for change in ["wrong_name", "bad_json", "wrong_args", "no_id", "wrong_type"]:
            broken = copy.deepcopy(value); call = broken["choices"][0]["message"]["tool_calls"][0]
            if change == "wrong_name": call["function"]["name"] = "execute"
            elif change == "bad_json": call["function"]["arguments"] = "{"
            elif change == "wrong_args": call["function"]["arguments"] = '{"topic":"почта"}'
            elif change == "no_id": call["id"] = ""
            else: call["type"] = "invalid"
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(broken, case, body)

    def test_stream_requires_done_terminal_usage_and_consistent_choice(self):
        value = response(); choice = value["choices"][0]
        event = {**value, "object": "chat.completion.chunk", "choices": [{
            "index": 0, "delta": choice["message"], "finish_reason": "stop"}]}
        body = {**self.body, "stream": True}
        raw = b"data: " + json.dumps(event).encode() + b"\n\ndata: [DONE]\n\n"
        self.assertEqual(faex2.validate_exchange(self.case, body, raw, 200)["text"], "391")
        for broken in [raw.split(b"data: [DONE]")[0], raw + b"data: {}\n\n"]:
            with self.assertRaises(ValueError):
                faex2.validate_exchange(self.case, body, broken, 200)

    def test_mtp_coverage_is_never_inferred_from_flag(self):
        self.assertTrue(callable(getattr(faex2, "validate_mode", None)))
        log = "event=load_completed kind=text speculative=mtp\n"
        for p, a in [(0, 0), (5, 0), (2, 3)]:
            with self.assertRaises(ValueError):
                faex2.validate_mode(log, "mtp", [{"draft_tokens": p, "draft_tokens_accepted": a}])
        faex2.validate_mode(log, "mtp", [{"draft_tokens": 5, "draft_tokens_accepted": 2}])
        with self.assertRaises(ValueError):
            faex2.validate_mode(log, "off", [])


class FakeHost(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(getattr(faex2, "serve", None)), "owned server not implemented")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "gufo"
        shutil.copyfile(Path(__file__).parent / "fixtures/faex2_fake_server.py", self.binary)
        # Use the test environment's exact interpreter, without requiring PATH setup.
        self.binary.write_text("#!" + sys.executable + "\n" + self.binary.read_text())
        self.binary.chmod(0o755)
        self.model = self.root / "fake.gguf"; self.model.write_bytes(b"test-only-not-weights")
        with socket.socket() as port:
            port.bind(("127.0.0.1", 0)); self.port = port.getsockname()[1]
        self.config = {"stage": "B1", "source_sha": "1" * 40,
            "binary": str(self.binary), "binary_sha256": hashlib.sha256(self.binary.read_bytes()).hexdigest(),
            "model": str(self.model), "model_sha256": hashlib.sha256(self.model.read_bytes()).hexdigest(),
            "library_path": str(self.root), "port": self.port,
            "startup_timeout_s": 2, "lifetime_s": 8}
        self.output = self.root / "run"

    def launch(self, behavior="ok"):
        (self.root / "behavior.json").write_text(json.dumps({"mode": behavior, "port": self.config["port"]}))
        config = self.root / "config.json"; config.write_text(json.dumps(self.config))
        process = subprocess.Popen([sys.executable, str(Path(faex2.__file__)), "serve",
            "--config", str(config), "--output", str(self.output)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.finish, process)
        return process

    def finish(self, process):
        if process.poll() is None:
            if self.output.is_dir(): (self.output / "stop-request").touch()
            try: process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.communicate(timeout=2)
        else:
            process.communicate()

    def ready(self, process):
        until = time.monotonic() + 5
        while time.monotonic() < until and process.poll() is None:
            if (self.output / "ready.json").exists(): return
            time.sleep(.02)
        self.fail("fake server did not become ready: " + process.communicate(timeout=2)[1])


class OwnershipTests(FakeHost):
    def test_linux_listener_parser_requires_exact_loopback_port_and_listen_state(self):
        self.assertTrue(callable(getattr(faex2, "linux_listener_inodes", None)))
        data = "header\n" + "\n".join([
            "0: 0100007F:341A 00000000:0000 0A 0:0 00:0 0 1000 0 4242",
            "1: 00000000:341A 00000000:0000 0A 0:0 00:0 0 1000 0 7777",
            "2: 0100007F:341B 00000000:0000 0A 0:0 00:0 0 1000 0 8888",
            "3: 0100007F:341A 00000000:0000 01 0:0 00:0 0 1000 0 9999"])
        self.assertEqual(faex2.linux_listener_inodes(data, 13338), {"4242"})

    def test_nested_deadline_cannot_extend_parent_limit(self):
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            with faex2.hard_timeout(.1):
                with faex2.hard_timeout(2): time.sleep(.5)
        self.assertLess(time.monotonic() - started, .4)

    def test_port_is_reserved_during_version_check(self):
        process = self.launch("inspect-port"); self.ready(process)
        self.assertEqual((self.root / "port-reservation").read_text(), "reserved")

    def test_listener_must_belong_to_owned_pid(self):
        process = self.launch("borrow-ready")
        try: result = process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.finish(process)
            self.fail("accepted another PID's listener as ready")
        self.assertNotEqual(result, 0)
        self.assertFalse((self.output / "ready.json").exists())

    def test_existing_output_and_symlink_are_not_overwritten(self):
        self.output.mkdir(); marker = self.output / "keep"; marker.write_text("untouched")
        process = self.launch(); self.assertNotEqual(process.wait(timeout=4), 0)
        self.assertEqual(marker.read_text(), "untouched")
        self.assertEqual(list(self.output.iterdir()), [marker])
        self.output = self.root / "link"; self.output.symlink_to(self.root / "run")
        process = self.launch(); self.assertNotEqual(process.wait(timeout=4), 0)
        self.assertEqual(marker.read_text(), "untouched")

    def test_occupied_port_does_not_launch_or_stop_other_process(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", self.port)); listener.listen()
            process = self.launch(); self.assertNotEqual(process.wait(timeout=4), 0)
            self.assertFalse((self.root / "invoked").exists())
            self.assertEqual(listener.getsockname()[1], self.port)

    def test_wrong_hash_version_and_skip_exit_never_pass(self):
        for behavior in ["wrong-version", "exit77", "wrong-mode"]:
            with self.subTest(behavior=behavior):
                with socket.socket() as port:
                    port.bind(("127.0.0.1", 0)); self.config["port"] = port.getsockname()[1]
                self.output = self.root / behavior
                process = self.launch(behavior); self.assertNotEqual(process.wait(timeout=5), 0)
                self.assertEqual(json.loads((self.output / "server-result.json").read_text())["status"], "FAIL")

                self.assertFalse((self.output / "ready.json").exists())
        self.output = self.root / "bad-hash"; self.model.write_bytes(b"changed")
        with socket.socket() as port:
            port.bind(("127.0.0.1", 0)); self.config["port"] = port.getsockname()[1]
        process = self.launch(); self.assertNotEqual(process.wait(timeout=4), 0)
        self.assertIn("hash", (self.output / "server-result.json").read_text())

    def test_stop_only_own_process_and_preserve_cache(self):
        outsider = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(15)"])
        self.addCleanup(lambda: (outsider.terminate(), outsider.wait(timeout=2)))
        process = self.launch(); self.ready(process)
        ready = json.loads((self.output / "ready.json").read_text())
        (self.output / "stop-request").touch()
        self.assertEqual(process.wait(timeout=5), 0)
        self.assertIsNone(outsider.poll())
        self.assertEqual((self.output / "cache/keep").read_text(), "cache-evidence")
        with self.assertRaises(ProcessLookupError): os.kill(ready["pid"], 0)
        self.assertEqual(json.loads((self.output / "server-result.json").read_text())["status"], "STOPPED")

    def test_deadline_and_startup_failure_retain_output(self):
        for behavior in ["no-ready", "ok"]:
            with self.subTest(behavior=behavior):
                self.output = self.root / behavior
                self.config["lifetime_s"] = .8; self.config["startup_timeout_s"] = .4
                process = self.launch(behavior); self.assertNotEqual(process.wait(timeout=5), 0)
                self.assertTrue((self.output / "server.log").is_file())
                self.assertEqual((self.output / "cache/keep").read_text(), "cache-evidence")
                self.assertEqual(json.loads((self.output / "server-result.json").read_text())["status"], "FAIL")

    def test_startup_deadline_also_bounds_dripping_http_headers(self):
        self.config["startup_timeout_s"] = .25
        self.config["lifetime_s"] = 5
        started = time.monotonic()
        process = self.launch("drip-health")
        self.assertNotEqual(process.wait(timeout=7), 0)
        self.assertLess(time.monotonic() - started, 2)
        self.assertTrue((self.output / "cache/keep").is_file())


class ClientTests(FakeHost):
    def setUp(self):
        super().setUp()
        self.assertTrue(callable(getattr(faex2, "check", None)), "retained B1 client not implemented")
        self.client_output = self.root / "client"

    def client(self, record=None, url=None):
        return subprocess.run([sys.executable, faex2.__file__, "check",
            "--server-record", str(record or (self.output / "ready.json")),
            "--base-url", url or f"http://127.0.0.1:{self.port}",
            "--output", str(self.client_output), "--request-timeout", ".4", "--timeout", "5"],
            capture_output=True, text=True, timeout=7)

    def test_full_fake_b1_retains_requests_and_responses(self):
        process = self.launch(); self.ready(process)
        result = self.client(); self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.client_output / "report.json").read_text())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["timing_status"], "RECORDED_NOT_COMPARED")
        self.assertGreaterEqual(len(report["requests"]), 8)
        for row in report["requests"]:
            self.assertEqual(row["status"], "PASS")
            self.assertGreater(row["metrics"]["wall_ms"], 0)
            self.assertTrue((self.client_output / row["response_file"]).is_file())
        self.assertIn("tool-followup-sse", [row["id"] for row in report["requests"]])
        self.assertTrue((self.output / "cache/keep").is_file())

    def test_client_failure_and_timeout_retain_partial_evidence(self):
        for behavior in ["empty-answer", "missing-metric", "slow-body", "wrong-content-type"]:
            with self.subTest(behavior=behavior):
                with socket.socket() as port:
                    port.bind(("127.0.0.1", 0)); self.port = port.getsockname()[1]
                self.config["port"] = self.port
                self.output = self.root / (behavior + "-server")
                self.client_output = self.root / (behavior + "-client")
                process = self.launch(behavior); self.ready(process)
                result = self.client(); self.assertNotEqual(result.returncode, 0)
                report = json.loads((self.client_output / "report.json").read_text())
                self.assertEqual(report["status"], "FAIL")
                self.assertTrue(report["requests"])
                raw = self.client_output / report["requests"][-1]["response_file"]
                self.assertTrue(raw.read_bytes())
                self.assertTrue((self.output / "cache/keep").is_file())
                self.finish(process)

    def test_wrong_record_and_existing_output_fail_without_generation(self):
        process = self.launch(); self.ready(process)
        record = json.loads((self.output / "ready.json").read_text())
        record["config"]["source_sha"] = "2" * 40
        bad = self.root / "wrong.json"; bad.write_text(json.dumps(record))
        self.assertNotEqual(self.client(record=bad).returncode, 0)
        self.assertFalse((self.root / "posted").exists())
        self.client_output = self.root / "occupied"; self.client_output.mkdir()
        marker = self.client_output / "keep"; marker.write_text("untouched")
        self.assertNotEqual(self.client().returncode, 0)
        self.assertEqual(list(self.client_output.iterdir()), [marker])

    def test_non_loopback_url_is_rejected(self):
        process = self.launch(); self.ready(process)
        self.assertNotEqual(self.client(url="http://example.invalid:80").returncode, 0)
        self.assertFalse((self.root / "posted").exists())

    def test_cannot_replace_b1_with_partial_cases(self):
        process = self.launch(); self.ready(process)
        subset = self.root / "subset.jsonl"
        first = (Path(faex2.__file__).with_name("faex2_cases.jsonl")).read_text().splitlines()[0]
        subset.write_text(first + "\n")
        result = subprocess.run([sys.executable, faex2.__file__, "check",
            "--server-record", str(self.output / "ready.json"),
            "--base-url", f"http://127.0.0.1:{self.port}", "--output", str(self.client_output),
            "--cases", str(subset)], capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "posted").exists())


if __name__ == "__main__":
    unittest.main()
