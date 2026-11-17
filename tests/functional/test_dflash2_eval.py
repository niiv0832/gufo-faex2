import http.server
import json
from pathlib import Path
import tempfile
import threading
import unittest

import dflash2_eval


class MockHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/props":
            body = json.dumps({"model_alias": "qwen3.6-35b-a3b"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(length))
        streaming = bool(req.get("stream"))
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cached_tokens": 0,
            "prompt_tokens_per_second": 500.0,
            "completion_tokens_per_second": 80.0,
            "draft_tokens": 10,
            "draft_tokens_accepted": 7,
            "completion_tokens_details": {"reasoning_tokens": 8},
            "gufo": {
                "prefill_tokens": 100,
                "prefill_ms": 200.0,
                "decode_ms": 250.0,
                "ttft_ms": 210.0,
                "queue_ms": 0.5,
                "cache_restore_ms": 0.0,
                "cache_snapshot_ms": 0.0,
                "cache_disk_enqueue_ms": 0.0,
                "draft_rounds": 5,
                "cache_hit": False,
                "cache_disk_hit": False,
                "execution_plan": "test",
            },
        }
        self.send_response(200)
        self.send_header("X-Request-ID", "req-mock-1")
        if streaming:
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            is_prose = "Объясни" in req["messages"][-1]["content"] or "Перескажи" in req["messages"][-1]["content"]
            chunks = [
                {
                    "id": "chat-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": req["model"],
                    "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "Думаю..."}}],
                },
            ]
            if is_prose:
                chunks.append({
                    "id": "chat-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": req["model"],
                    "choices": [{"index": 0, "delta": {"content": "Ответ."}}],
                })
                finish = "stop"
            else:
                chunks.append({
                    "id": "chat-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": req["model"],
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "web_search", "arguments": "{\"query\": \"test\"}"},
                            }]
                        },
                    }],
                })
                finish = "tool_calls"
            chunks.append({
                "id": "chat-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": req["model"],
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "timings": {"draft_rounds": 5, "draft_n": 10, "draft_n_accepted": 7},
            })
            chunks.append({
                "id": "chat-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": req["model"],
                "choices": [],
                "usage": usage,
            })
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            is_aux = "response_format" in req
            content = '{"title": "План подготовки"}' if is_aux else "56"
            body = json.dumps({
                "id": "chat-2",
                "object": "chat.completion",
                "created": 1,
                "model": req["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": usage,
            }).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


class DFlash2EvalTest(unittest.TestCase):
    def test_mock_end_to_end_and_log_join(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                log_path = tmp_path / "server.log"
                log_path.write_text(
                    "INFO loader event=load_completed kind=text elapsed_ms=100 speculative=dflash2 draft_limit=7\n"
                    "INFO http request=req-mock-1 event=completed method=POST path=/v1/chat/completions status=200 duration_ms=460.0 outcome=complete cache=miss cached_tokens=0 ttft_ms=210.0 prefill_tps=500.0 decode_tps=80.0 draft_accepted=7 draft_proposed=10 draft_rounds=5 acceptance_pct=70.0 plan=test\n",
                    encoding="utf-8",
                )
                out_dir = tmp_path / "out"
                rc = dflash2_eval.run_eval(
                    dflash2_eval.argparse.Namespace(
                        base_url=f"http://127.0.0.1:{port}",
                        mode="dflash2",
                        output=str(out_dir),
                        server_log=str(log_path),
                        passes=1,
                        temperature=0.0,
                        seed=1,
                        request_timeout=10.0,
                        no_warmup=False,
                        include_aux_every_pass=False,
                    )
                )
                self.assertEqual(rc, 0)
                report = json.loads((out_dir / "report.json").read_text())
                self.assertEqual(report["server_loads"][0]["speculative"], "dflash2")
                self.assertEqual(len(report["records"]), 18)  # 1 warmup + 15 cases + 2 aux
                aux = [r for r in report["records"] if r["class"] == "aux_title"]
                self.assertTrue(all(r["valid"] for r in aux))
                self.assertEqual(aux[0]["server_log"]["draft_accepted"], 7)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
