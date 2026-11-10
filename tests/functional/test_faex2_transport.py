"""CPU-проверки wire transport: настоящий loopback, без модели и SDK."""

import copy
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
if importlib.util.find_spec("faex2_transport"):
    import faex2_transport as transport
else:
    transport = None

MODEL = "qwen3.6-35b-a3b"


def usage():
    return {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24,
            "cached_tokens": 0, "draft_tokens": 4, "draft_tokens_accepted": 2,
            "gufo": {"prefill_tokens": 20, "prefill_ms": 2.5, "decode_ms": 3,
                     "queue_ms": 0, "ttft_ms": 3, "cache_restore_ms": 0,
                     "cache_snapshot_ms": 0, "cache_disk_enqueue_ms": 0}}


def chat(text="Ответ", reasoning="Проверка"):
    return {"id": "chat-test", "object": "chat.completion", "created": 1,
            "model": MODEL, "choices": [{"index": 0, "message": {
                "role": "assistant", "content": text, "reasoning_content": reasoning},
                "finish_reason": "stop"}], "usage": usage()}


def chat_stream():
    envelope = {"id": "chat-test", "object": "chat.completion.chunk", "created": 1,
                "model": MODEL}
    return [{**envelope, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            for delta, finish in [({"role": "assistant", "reasoning_content": "Проверка"}, None),
                                  ({"content": "Ответ"}, None), ({}, "stop")]] + [
        {**envelope, "choices": [], "usage": usage()}]


def response():
    return {"id": "resp-test", "object": "response", "created_at": 1,
            "model": MODEL, "status": "completed", "error": None,
            "incomplete_details": None,
            "output": [{"id": "rs-test", "type": "reasoning", "status": "completed",
                        "summary": [{"type": "summary_text", "text": "Проверка"}]},
                       {"id": "msg-test", "type": "message", "status": "completed",
                        "role": "assistant", "content": [
                            {"type": "output_text", "text": "Ответ", "annotations": []}]}],
            "usage": {"input_tokens": 20, "output_tokens": 4, "total_tokens": 24,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 3}},
            "timings": {"prompt_n": 20, "prompt_ms": 2.5, "predicted_ms": 3,
                        "cache_restore_ms": 0, "cache_snapshot_ms": 0,
                        "cache_disk_enqueue_ms": 0, "draft_n": 4, "draft_n_accepted": 2}}


def responses_stream():
    final = response()
    initial = {**final, "status": "in_progress", "output": [], "usage": None}
    events = [{"type": "response.created", "response": initial}]
    for index, item in enumerate(final["output"]):
        reasoning = item["type"] == "reasoning"
        field = "summary" if reasoning else "content"
        stem = "response.reasoning_summary_text" if reasoning else "response.output_text"
        loc = {"output_index": index, "item_id": item["id"],
               "summary_index" if reasoning else "content_index": 0}
        events.extend([
            {"type": "response.output_item.added", "output_index": index,
             "item": {**item, "status": "in_progress", field: []}},
            {"type": stem + ".delta", **loc, "delta": item[field][0]["text"]},
            {"type": stem + ".done", **loc, "text": item[field][0]["text"]},
            {"type": "response.output_item.done", "output_index": index, "item": item}])
    events.append({"type": "response.completed", "response": final})
    return [{**event, "sequence_number": index} for index, event in enumerate(events)]


def sse(events, done=False):
    return [b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n\n"
            for event in events] + ([b"data: [DONE]\n\n"] if done else [])


def request(path="/v1/chat/completions", stream=False):
    return {"method": "POST", "path": path, "body": {
        "model": MODEL, "messages": [], "stream": stream, "max_completion_tokens": 64}}


class ParsingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(transport, "ограниченный transport ещё не реализован")

    def parse(self, value, req=None, status=200):
        raw = json.dumps(value, ensure_ascii=False).encode()
        return transport.parse_response(req or request(), raw, [(23, raw)], status)

    def test_buffered_chat_preserves_reasoning_usage_and_metrics(self):
        row = self.parse(chat())
        self.assertEqual((row["status"], row["text"], row["reasoning"]),
                         ("PASS", "Ответ", "Проверка"))
        self.assertEqual(row["usage"], usage())
        self.assertEqual(row["metrics"]["draft_tokens_accepted"], 2)
        self.assertEqual(row["first_visible_content_ms"], 23)
        self.assertEqual(len(row["output_sha256"]), 64)

    def test_sse_ttfa_is_not_first_reasoning_delta(self):
        chunks = sse(chat_stream(), done=True)
        parts = list(zip([10, 70, 80, 90, 100], chunks))
        row = transport.parse_response(request(stream=True), b"".join(chunks), parts, 200)
        self.assertEqual(row["reasoning"], "Проверка")
        self.assertEqual(row["text"], "Ответ")
        self.assertEqual(row["metrics"]["client_ttft_ms"], 10)
        self.assertEqual(row["first_visible_content_ms"], 70)
        self.assertEqual(row["metrics"]["client_ttfa_ms"], 70)

    def test_responses_preserves_actual_reasoning_token_counter(self):
        row = self.parse(response(), request("/v1/responses"))
        self.assertEqual((row["text"], row["reasoning"], row["finish"]),
                         ("Ответ", "Проверка", "stop"))
        self.assertEqual(row["response_status"], "completed")
        self.assertEqual(row["usage"]["output_tokens_details"]["reasoning_tokens"], 3)
        self.assertNotIn("gufo", row["usage"])

    def test_responses_sse_keeps_visible_delta_timing(self):
        chunks = sse(responses_stream())
        parts = [(10 * i, chunk) for i, chunk in enumerate(chunks)]
        row = transport.parse_response(request("/v1/responses", True), b"".join(chunks), parts, 200)
        self.assertEqual(row["text"], "Ответ")
        self.assertEqual(row["reasoning"], "Проверка")
        self.assertEqual(row["first_visible_content_ms"], 60)

    def test_http_error_is_distinct_from_success_even_for_stream_request(self):
        error = {"error": {"type": "invalid_request_error", "message": "bad image"}}
        row = self.parse(error, request(stream=True), 400)
        self.assertEqual(row["status"], "HTTP_ERROR")
        self.assertEqual(row["error"], error["error"])
        with self.assertRaises(ValueError):
            self.parse({"error": "bad image"}, status=400)

    def test_chat_bad_envelopes_and_types_cannot_pass(self):
        mutations = [lambda v: v.update(object="response"), lambda v: v.update(id=123),
                     lambda v: v.update(created=True), lambda v: v.update(model="foreign"),
                     lambda v: v["choices"][0].update(index=True),
                     lambda v: v["choices"][0].update(index=1),
                     lambda v: v["choices"][0]["message"].update(content=False),
                     lambda v: v["choices"][0]["message"].update(reasoning_content=[]),
                     lambda v: v["usage"].update(total_tokens=999)]
        for change in mutations:
            value = chat(); change(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse(value)

    def test_chat_sse_rejects_missing_done_mixed_ids_and_events_after_terminal(self):
        original = chat_stream()
        variants = [sse(original), sse(original, True) + sse([original[0]])]
        corrupt = copy.deepcopy(original); corrupt[1]["id"] = "foreign"
        variants.append(sse(corrupt, True))
        corrupt = copy.deepcopy(original); corrupt.insert(3, original[1])
        variants.append(sse(corrupt, True))
        variants.append(sse(original, True) + [b"data: [DONE]\n\n"])
        variants.append(sse(original, True) + [b"data: {"])
        for chunks in variants:
            with self.subTest(chunks=chunks), self.assertRaises(ValueError):
                transport.parse_response(request(stream=True), b"".join(chunks),
                                         [(0, chunk) for chunk in chunks], 200)

    def test_responses_rejects_envelope_usage_and_changed_stream_text(self):
        for field, value in [("object", "wrong"), ("id", []), ("created_at", True)]:
            broken = response(); broken[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.parse(broken, request("/v1/responses"))
        for count in [True, -1, 5, "3"]:
            broken = response(); broken["usage"]["output_tokens_details"]["reasoning_tokens"] = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.parse(broken, request("/v1/responses"))
        for field, value in [("delta", "Подмена"), ("item_id", "foreign")]:
            events = responses_stream(); events[6][field] = value
            chunks = sse(events)
            with self.subTest(field=field), self.assertRaises(ValueError):
                transport.parse_response(request("/v1/responses", True), b"".join(chunks),
                                         [(0, chunk) for chunk in chunks], 200)

    def test_discovery_and_tokenize_return_original_json(self):
        row = self.parse({"status": "ok"}, {"method": "GET", "path": "/health", "body": None})
        self.assertEqual(row["response"], {"status": "ok"})
        row = self.parse({"tokens": [1, 2, 3]}, {"method": "POST", "path": "/tokenize",
                                               "body": {"content": "example"}})
        self.assertEqual(row["response"], {"tokens": [1, 2, 3]})

    def test_tool_call_keeps_id_name_and_arguments_in_both_protocols(self):
        definition = {"name": "lookup", "strict": True, "parameters": {
            "type": "object", "properties": {"topic": {"type": "string"}},
            "required": ["topic"], "additionalProperties": False}}
        req = request(); req["body"]["tools"] = [{"type": "function", "function": definition}]
        value = chat(None, None)
        value["choices"][0].update(finish_reason="tool_calls", message={"role": "assistant",
            "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {
                "name": "lookup", "arguments": '{"topic":"школа"}'}}]})
        row = self.parse(value, req)
        self.assertEqual(row["tools"][0]["id"], "call-1")
        self.assertEqual(row["tools"][0]["function"]["arguments"], '{"topic":"школа"}')
        req = request("/v1/responses")
        req["body"]["tools"] = [{"type": "function", **definition}]
        value = response(); value["output"] = [{"type": "function_call", "id": "fc-1",
            "call_id": "call-1", "name": "lookup", "arguments": '{"topic":"школа"}', "status": "completed"}]
        row = self.parse(value, req)
        self.assertEqual(row["finish"], "tool_calls")
        self.assertEqual(row["tools"][0]["call_id"], "call-1")
        value["output"][0]["arguments"] = '{"topic":7}'
        with self.assertRaises(Exception): self.parse(value, req)

    def test_corrupt_stream_text_done_cannot_be_hidden_by_final_response(self):
        events = responses_stream()
        events.insert(8, {**events[6], "delta": " лишнее"})
        for index, event in enumerate(events): event["sequence_number"] = index
        chunks = sse(events)
        with self.assertRaises(ValueError):
            transport.parse_response(request("/v1/responses", True), b"".join(chunks),
                                     [(0, chunk) for chunk in chunks], 200)

    def test_unknown_responses_event_cannot_be_silently_ignored(self):
        events = responses_stream(); events.insert(1, {"type": "response.malformed"})
        for index, event in enumerate(events): event["sequence_number"] = index
        chunks = sse(events)
        with self.assertRaises(ValueError):
            transport.parse_response(request("/v1/responses", True), b"".join(chunks),
                                     [(0, chunk) for chunk in chunks], 200)

    def test_json_duplicates_and_nonfinite_values_are_not_normalized(self):
        for raw in [json.dumps(chat()).encode().replace(b'"created": 1', b'"created": 1, "created": 2'),
                    json.dumps(chat()).encode().replace(b'"created": 1', b'"created": NaN')]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                transport.parse_response(request(), raw, [(0, raw)], 200)


class LoopbackTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(transport, "ограниченный transport ещё не реализован")
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def start_server(self, chunks, status=200, content_type="application/json", delay=0,
                     declared_length=None, headers_drip=False, chunked=False):
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                parent.received = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                try:
                    if headers_drip:
                        self.connection.sendall(b"HTTP/1.1 200 OK\r\nX-Drip: ")
                        for _ in range(50):
                            self.connection.sendall(b"a"); time.sleep(.03)
                        return
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    if chunked: self.send_header("Transfer-Encoding", "chunked")
                    if declared_length is not None:
                        self.send_header("Content-Length", str(declared_length))
                    self.end_headers()
                    for chunk in chunks:
                        self.wfile.write(chunk); self.wfile.flush()
                        if delay: time.sleep(delay)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            do_GET = do_POST

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.handle_request, daemon=True); thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 3)
        return "http://127.0.0.1:" + str(server.server_address[1])

    def launch(self, base, req=None, extra=(), raw_request=None, output=None):
        path = self.root / "input.json"
        path.write_bytes(raw_request if raw_request is not None else
                         json.dumps(req or request(), ensure_ascii=False).encode())
        destination = output or self.root / "result"
        started = time.monotonic()
        process = subprocess.run([sys.executable, str(Path(transport.__file__)), "request",
                                  "--base-url", base, "--request", str(path), "--output",
                                  str(destination), *extra], capture_output=True, text=True, timeout=5)
        elapsed = time.monotonic() - started
        record = json.loads((destination / "result.json").read_text())
        return process, record, destination, elapsed

    def test_cli_retains_wire_request_raw_response_and_timed_parts(self):
        wire = json.dumps(chat(), ensure_ascii=False).encode()
        base = self.start_server([wire])
        process, row, output, _ = self.launch(base)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(row["status"], "PASS")
        self.assertEqual((output / "response.raw").read_bytes(), wire)
        self.assertEqual((output / "request.body.raw").read_bytes(), self.received)
        self.assertEqual(json.loads((output / "request.json").read_text()), request())
        parts = json.loads((output / "parts.json").read_text())
        self.assertEqual(sum(part["size"] for part in parts), len(wire))

    def test_wall_ends_at_last_wire_byte_not_after_local_validation(self):
        """Разбор и проверки ответа на стороне клиента не входят в wall_ms.

        Регрессия C timing: +19 мс wall короткого JSON возникали после получения
        полного тела. Время локальной обработки сохраняется отдельно.
        """
        wire = json.dumps(chat(), ensure_ascii=False).encode()
        base = self.start_server([wire], declared_length=len(wire))
        path = self.root / "input.json"
        path.write_text(json.dumps(request(), ensure_ascii=False))
        original = transport._parse_response

        def slow_parse(*args, **kwargs):
            time.sleep(.3)
            return original(*args, **kwargs)

        transport._parse_response = slow_parse
        self.addCleanup(setattr, transport, "_parse_response", original)
        row = transport.run_request(base, path, self.root / "result")
        self.assertEqual(row["status"], "PASS")
        parts = json.loads((self.root / "result" / "parts.json").read_text())
        self.assertLess(row["wall_ms"] - parts[-1]["elapsed_ms"], 100,
                        "wall_ms включает локальную обработку после последнего байта")
        self.assertGreaterEqual(row["client_total_ms"], row["wall_ms"] + 300)
        self.assertGreaterEqual(row["client_post_wire_ms"], 300)

    def test_http400_returns_zero_but_never_pass(self):
        base = self.start_server([b'{"error":{"type":"invalid_request_error","message":"bad"}}'], 400)
        process, row, _, _ = self.launch(base, request(stream=True))
        self.assertEqual(process.returncode, 0)
        self.assertEqual((row["status"], row["http_status"]), ("HTTP_ERROR", 400))

    def test_malformed_and_wrong_content_type_retain_raw_before_failure(self):
        for i, (wire, kind) in enumerate([(b'{"bad":', "application/json"),
                                         (json.dumps(chat()).encode(), "text/html")]):
            base = self.start_server([wire], content_type=kind)
            process, row, output, _ = self.launch(base, output=self.root / str(i))
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(row["status"], "FAIL")
            self.assertEqual((output / "response.raw").read_bytes(), wire)

    def test_wall_timeout_retains_dripping_body(self):
        base = self.start_server([b"x"] * 50, delay=.03)
        process, row, output, elapsed = self.launch(base, extra=("--timeout", ".2"))
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(row["status"], "FAIL")
        self.assertIn("deadline", row["error"])
        self.assertTrue((output / "response.raw").read_bytes())
        self.assertLess(elapsed, 1.5)

    def test_header_drip_cannot_extend_wall_timeout(self):
        base = self.start_server([], headers_drip=True)
        process, row, _, elapsed = self.launch(base, extra=("--timeout", ".2"))
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(row["status"], "FAIL")
        self.assertLess(elapsed, 1.5)

    def test_intentional_cancel_keeps_partial_text_and_is_distinct_from_abort(self):
        chunks = sse(chat_stream(), done=True)
        base = self.start_server(chunks, content_type="text/event-stream", delay=.12)
        process, row, output, elapsed = self.launch(base, request(stream=True),
                                                   extra=("--cancel-after-content", "1"))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual((row["status"], row["text"], row["reasoning"]),
                         ("CANCELLED", "Ответ", "Проверка"))
        self.assertTrue(row["intentional_cancel"])
        self.assertNotIn(b"[DONE]", (output / "response.raw").read_bytes())
        self.assertLess(elapsed, .9)
        base = self.start_server(chunks[:2], content_type="text/event-stream")
        process, row, output, _ = self.launch(base, request(stream=True), output=self.root / "aborted")
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(row["status"], "FAIL")
        self.assertEqual((output / "response.raw").read_bytes(), b"".join(chunks[:2]))

    def test_coalesced_chat_completion_is_not_evidence_of_cancellation(self):
        # Один read1 получает content вместе с finish; наличие DONE ничего не меняет.
        for index, chunks in enumerate([sse(chat_stream(), done=True), sse(chat_stream()[:3])]):
            wire = b"".join(chunks)
            base = self.start_server([wire], content_type="text/event-stream")
            process, row, output, _ = self.launch(base, request(stream=True),
                extra=("--cancel-after-content", "1"), output=self.root / str(index))
            self.assertEqual((output / "response.raw").read_bytes(), wire)
            self.assertEqual(len(json.loads((output / "parts.json").read_text())), 1)
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(row["status"], "FAIL")
            self.assertEqual(row["error_code"], "COMPLETE_BEFORE_CANCEL")
            self.assertFalse(row["intentional_cancel"])

    def test_coalesced_responses_terminal_is_not_evidence_of_cancellation(self):
        for status in ["completed", "incomplete"]:
            events = responses_stream()
            events[-1]["type"] = "response." + status
            events[-1]["response"]["status"] = status
            if status == "incomplete":
                events[-1]["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
            wire = b"".join(sse(events))
            base = self.start_server([wire], content_type="text/event-stream")
            process, row, output, _ = self.launch(base, request("/v1/responses", True),
                extra=("--cancel-after-content", "1"), output=self.root / status)
            self.assertEqual((output / "response.raw").read_bytes(), wire)
            self.assertEqual(len(json.loads((output / "parts.json").read_text())), 1)
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(row["status"], "FAIL")
            self.assertEqual(row["error_code"], "COMPLETE_BEFORE_CANCEL")
            self.assertFalse(row["intentional_cancel"])

    def test_responses_can_still_cancel_before_terminal_event(self):
        chunks = sse(responses_stream())
        base = self.start_server(chunks, content_type="text/event-stream", delay=.04)
        process, row, output, _ = self.launch(base, request("/v1/responses", True),
                                             extra=("--cancel-after-content", "1"))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(row["status"], "CANCELLED")
        self.assertTrue(row["intentional_cancel"])
        self.assertEqual(row["text"], "Ответ")
        self.assertNotIn(b'"type": "response.completed"', (output / "response.raw").read_bytes())

    def test_short_content_length_retains_all_partial_bytes(self):
        base = self.start_server([b'{"partial":'], declared_length=1000)
        process, row, output, _ = self.launch(base)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(row["status"], "FAIL")
        self.assertEqual((output / "response.raw").read_bytes(), b'{"partial":')

    def test_truncated_chunked_transfer_retains_decoded_partial_body(self):
        base = self.start_server([b'20\r\n{"partial":'], chunked=True)
        process, row, output, _ = self.launch(base)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(row["status"], "FAIL")
        self.assertEqual((output / "response.raw").read_bytes(), b'{"partial":')

    def test_existing_output_and_symlink_cannot_be_overwritten(self):
        existing = self.root / "existing"; existing.mkdir()
        (existing / "keep").write_text("untouched")
        link = self.root / "link"; link.symlink_to(existing, target_is_directory=True)
        source = self.root / "request.json"; source.write_text(json.dumps(request()))
        for output in [existing, link]:
            result = subprocess.run([sys.executable, str(Path(transport.__file__)), "request",
                "--base-url", "http://127.0.0.1:9", "--request", str(source), "--output", str(output)],
                capture_output=True, text=True, timeout=3)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list(existing.iterdir()), [existing / "keep"])
            self.assertEqual((existing / "keep").read_text(), "untouched")

    def test_oversized_body_fails_and_retains_bounded_raw(self):
        base = self.start_server([b"x" * (17 * 1024 * 1024)])
        process, row, output, _ = self.launch(base)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(row["status"], "FAIL")
        self.assertEqual((output / "response.raw").stat().st_size, 16 * 1024 * 1024)

    def test_rejected_inputs_leave_result_without_making_connection(self):
        cases = [("http://example.com:80", request(), ()),
                 ("http://127.0.0.1:9/ignored", request(), ()),
                 ("http://127.0.0.1:9", {**request(), "path": "//example.com"}, ()),
                 ("http://127.0.0.1:9", {**request(), "env": {}}, ()),
                 ("http://127.0.0.1:9", request(), ("--timeout", "601")),
                 ("http://127.0.0.1:9", request(), ("--cancel-after-content", "1"))]
        for i, (base, req, extra) in enumerate(cases):
            process, row, _, _ = self.launch(base, req, extra, output=self.root / str(i))
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(row["status"], "FAIL")
            self.assertIsNone(row["http_status"])


if __name__ == "__main__":
    unittest.main()
