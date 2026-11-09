"""Test-only process double. No weights, GPU, network except local HTTP."""
import json
import os
from pathlib import Path
import sys
import time
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer

root = Path(__file__).parent
mode = json.loads((root / "behavior.json").read_text())["mode"]
(root / "invoked").touch()
revision = "1" * 40
if sys.argv[1:] == ["--version"]:
    if mode == "inspect-port":
        with socket.socket() as reserved:
            try:
                reserved.bind(("127.0.0.1", json.loads((root / "behavior.json").read_text())["port"]))
                observed = "free"
            except OSError: observed = "reserved"
        (root / "port-reservation").write_text(observed)
    print("gufo version development (" + ("2" * 40 if mode == "wrong-version" else revision) + ")")
    sys.exit(0)

def option(name):
    return sys.argv[sys.argv.index(name) + 1]

cache = Path(os.environ["XDG_CACHE_HOME"])
(cache / "keep").write_text("cache-evidence")
if mode == "exit77": sys.exit(77)
if mode == "no-ready": time.sleep(30)
speculative = option("--speculative")
loaded = ("off" if speculative == "mtp" else "mtp") if mode == "wrong-mode" else speculative
draft = "2" if mode == "wrong-draft" else option("--draft-tokens") if "--draft-tokens" in sys.argv else "7"
thinking = ("on" if option("--think") == "off" else "off") if mode == "wrong-thinking" else option("--think")
print("event=defaults thinking=" + thinking, flush=True)
print("event=load_completed kind=text speculative=" + loaded + " draft_limit=" + draft, flush=True)
if mode == "wrong-budget": print("reasoning_budget=513", flush=True)
vision = "--mmproj" in sys.argv
if vision: print("event=vision_prepared resident_bytes=8192", flush=True)
if mode == "wrong-vision": vision = not vision

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_POST(self):
        (root / "posted").touch()
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = body["messages"]
        prompt = messages[0]["content"]
        tools = None
        finish = "stop"
        if messages[-1]["role"] == "tool":
            if messages[-1]["tool_call_id"] != "call_fake": self.send_error(400); return
            text = json.loads(messages[-1]["content"])["answer"]
        elif body.get("tools"):
            text = None; finish = "tool_calls"
            tools = [{"id": "call_fake", "type": "function", "function": {
                "name": "lookup", "arguments": '{"topic":"астрономия"}'}}]
        elif "response_format" in body: text = '{"answer":391,"bird":"сова"}'
        elif "СОВА_142" in prompt: text = "СОВА_142"
        elif "ЛУНА_853" in prompt: text = "ЛУНА_853"
        elif "Птица:" in prompt: text = "сова"
        elif "17 × 23" in prompt: text = "391"
        else: self.send_error(400); return
        if mode == "empty-answer": text = ""
        usage = {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22,
                 "cached_tokens": 0, "draft_tokens": 0, "draft_tokens_accepted": 0,
                 "gufo": {"prefill_tokens": 20, "prefill_ms": 2.5, "decode_ms": 3.0,
                          "queue_ms": 0, "ttft_ms": 3, "cache_restore_ms": 0,
                          "cache_snapshot_ms": 0, "cache_disk_enqueue_ms": 0}}
        if mode == "missing-metric": del usage["gufo"]["decode_ms"]
        message = {"role": "assistant", "content": text}
        if tools: message["tool_calls"] = tools
        response = {"id": "chat_fake", "created": 1, "model": "qwen3.6-35b-a3b",
                    "object": "chat.completion", "usage": usage,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}]}
        if body.get("stream"):
            response["object"] = "chat.completion.chunk"
            response["choices"][0]["delta"] = response["choices"][0].pop("message")
            data = b"data: " + json.dumps(response).encode() + b"\n\ndata: [DONE]\n\n"
        else: data = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain" if mode == "wrong-content-type" else
                         "text/event-stream" if body.get("stream") else "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if mode == "slow-body":
            self.wfile.write(data[:10]); self.wfile.flush(); time.sleep(10)
        else: self.wfile.write(data)

    def do_GET(self):
        if mode == "drip-health":
            for byte in b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n":
                self.wfile.write(bytes([byte])); self.wfile.flush(); time.sleep(.1)
            time.sleep(10)
            return
        alias = "qwen3.6-35b-a3b"
        if self.path == "/health": payload = {"status": "ok"}
        elif self.path == "/ready": payload = {"status": "ready", "model": alias}
        elif self.path == "/props": payload = {
            "model": alias, "model_alias": alias, "model_path": option("--model"),
            "build_info": "development (" + revision + ")", "total_slots": int(option("--sessions")),
            "modalities": {"vision": vision, "video": False, "audio": False},
            "default_generation_settings": {"n_ctx": int(option("--context")), "params": {
                "temperature": 1.0, "top_k": 20, "top_p": .95, "min_p": .05,
                "presence_penalty": 0.0, "frequency_penalty": 0.0,
                "repeat_penalty": 1.0, "repeat_last_n": 64}}}
        elif self.path == "/v1/models": payload = {"object": "list", "data": [{
            "object": "model", "id": alias, "owned_by": "gufo", "created": 1,
            "context_length": int(option("--context")),
            "architecture": {"input_modalities": ["text", "image"] if vision else ["text"]}}]}
        else: self.send_error(404); return
        data = json.dumps(payload).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

if mode == "borrow-ready" and os.fork():
    time.sleep(30)
else:
    HTTPServer(("127.0.0.1", int(option("--port"))), Handler).serve_forever()
