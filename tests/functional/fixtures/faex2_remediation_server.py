#!/usr/bin/env python3
"""Подставной сервер для CPU-тестов приёмки доработок. Без весов, GPU и внешней сети.

В отличие от faex2_fake_server.py (двойник запуска бинарника с фиксированным usage)
моделирует то, что проверяют gates: счёт токенов и кэш RAM/диск с политикой
сервера (стабильная граница хода, полный prompt только в RAM, сетка 2048, шаг
диска 2048), бюджет рассуждений, черновики, ширину пакета, отмену, журнал
`request=… event=completed`, поздний общий префикс. Дефекты включаются именами в
`defects`: по ним тесты доказывают, что каждый gate действительно отказывает.

Работает как модуль (`Fake(...).start()`) и как исполняемый двойник `gufo`:
`faex2_remediation_server.py serve --port N … llm … --cache-disk DIR` (журнал в
stdout; дефекты — FAEX2_FAKE_DEFECTS через запятую).
"""

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import faex2_remediation_images as fixture_images
except ImportError:  # Исполняемый двойник может лежать отдельно от клиента.
    fixture_images = None

ALIAS = "qwen3.6-35b-a3b"
GRID, TAIL, STEP = 2048, 128, 2048
COLOURS = {(255, 0, 0): "red", (0, 0, 255): "blue", (0, 160, 0): "green"}
RUSSIAN = {"red": "красный", "blue": "синий", "green": "зелёный"}


def first_pixel(url):
    data = base64.b64decode(url.split(",", 1)[1])
    position, idat = 8, b""
    while position < len(data):
        length = int.from_bytes(data[position:position + 4], "big")
        if data[position + 4:position + 8] == b"IDAT":
            idat += data[position + 8:position + 8 + length]
        position += 12 + length
    return tuple(zlib.decompress(idat)[1:4])


class Fake:
    def __init__(self, cache_dir=None, log_path=None, defects=(), speculative="mtp", pace=1.0):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.log_path, self.defects, self.speculative, self.pace = log_path, set(defects), speculative, pace
        self.lock = threading.Lock()
        self.ram, self.inflight, self.readers, self.counter = [], {}, [], 0
        self.budget_leak = None
        self.peak = {}
        self.server = None
        self.answers = {}
        if fixture_images:
            for probe in fixture_images.complex_probes():
                self.answers[(fixture_images.image_sha(probe["content"]), probe["question"])] = probe["expected"]

    # --- журнал и кэш ---------------------------------------------------
    def log(self, line):
        # Формат настоящего журнала: время, уровень и подсистема перед полями key=value.
        line = time.strftime("%Y-%m-%d %H:%M:%S") + " [INFO] [fake] " + line
        with self.lock:
            if self.log_path:
                with open(self.log_path, "a") as output:
                    output.write(line + "\n")
            else:
                print(line, flush=True)

    def disk_entries(self):
        if not self.cache_dir:
            return []
        return [tuple(json.loads(path.read_text())) for path in sorted(self.cache_dir.glob("*.snapshot"))]

    def disk_store(self, tokens):
        if not self.cache_dir or not tokens:
            return
        tokens = tuple(tokens)
        entries = self.disk_entries()
        if tokens in entries:
            return
        base = max((len(entry) for entry in entries if tokens[:len(entry)] == entry), default=None)
        if base is not None and len(tokens) - base < STEP:
            self.log(f"event=disk_cache action=skipped reason=min_step tokens={len(tokens)}")
            return
        name = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:24] + ".snapshot"
        (self.cache_dir / name).write_text(json.dumps(tokens))
        self.log(f"event=disk_cache action=stored reason=saved file_bytes=1 payload_bytes=1 tokens={len(tokens)}")

    def lookup(self, prompt):
        best, source = 0, "miss"
        with self.lock:
            ram = list(self.ram)
        for entries, name in ((ram, "memory"), (self.disk_entries(), "disk")):
            for entry in entries:
                if len(entry) > best and len(entry) <= len(prompt) and tuple(prompt[:len(entry)]) == entry:
                    best, source = len(entry), name
        return best, source

    def publish(self, prompt, stable, grid=True):
        prompt = tuple(prompt)
        points = []
        if grid and len(prompt) > GRID + TAIL:
            points.append(prompt[:GRID])
        points.append(prompt[:stable])
        with self.lock:
            self.ram.extend(points)
            if "ram_remainder" not in self.defects:
                self.ram.append(prompt)
        for point in (*points, prompt):
            self.disk_store(point)

    # --- токены и ответы ------------------------------------------------
    @staticmethod
    def render(messages):
        tokens, texts, pictures = [], [], []
        for message in messages:
            tokens.append("<" + message["role"] + ">")
            content = message.get("content")
            parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
            text, urls = "", []
            for part in parts:
                if part.get("type") in ("text", "input_text", "output_text"):
                    text += part["text"] + " "
                    # Около трёх токенов на слово, как у настоящего токенизатора на русском тексте.
                    tokens.extend(piece for word in part["text"].split() for piece in (word, word + "+", word + "++"))
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    urls.append(url)
                    tokens.extend(f"<img:{hashlib.sha256(url.encode()).hexdigest()[:12]}:{index}>"
                                  for index in range(16))
            for call in message.get("tool_calls") or []:
                tokens.extend(["<call>", call["function"]["name"]])
            tokens.append("<end>")
            texts.append((message["role"], text))
            pictures.append(urls)
        return tokens, texts, pictures

    def answer(self, body, texts, pictures):
        system = " ".join(text for role, text in texts if role == "system")
        role, last = texts[-1]
        urls = pictures[-1]
        if role == "tool":
            return json.loads(last)["answer"]
        if urls:
            sha = hashlib.sha256(urls[0].encode()).hexdigest()
            for (known, question), expected in self.answers.items():
                if known == sha and question in last:
                    return "NO" if "image_wrong" in self.defects else expected
            codes = re.findall(r"(?:ответь только кодом|ровно одно слово:) (\S+?)\.", last, re.IGNORECASE)
            if codes:
                return codes[-1]
            names = [COLOURS.get(first_pixel(url), "white") for url in urls]
            if "по-русски" in last and "Первым словом" not in last:
                return RUSSIAN.get(names[0], "белый")
            key = re.search(r"ключ: (\S+?)\.", last)
            if key:
                return key.group(1) + "," + names[0]
            if "Первым словом" in last:
                return names[0] + " " + " ".join(f"слово{index}" for index in range(60))
            return ",".join(names)
        factor = re.search(r"произведения числа (\d+)", last)
        if factor:
            marker = re.search(r"Маркер беседы: (\w+)", system).group(1)
            if "cross_talk" in self.defects:
                marker = "MOON_853" if marker != "MOON_853" else "OWL_142"
            products = [int(factor.group(1)) * value for value in range(20, 28)]
            if "zero_products_during_cancel" in self.defects and "ses-c" in system:
                products = [0] * 8
            # Как настоящая модель: JSON с отступами, поле marker дословно.
            return json.dumps({"marker": marker, "products": products}, indent=2)
        if "ключ" in last.lower() and "JSON" in last:
            history = " ".join(text for _, text in texts)
            return json.dumps({name: re.search(rf"Ключ {name} = (\S+)", history).group(1)
                               for name in ("start", "middle", "end")}, ensure_ascii=False)
        codes = re.findall(r"(?:ответь только (?:кодом|словом)|ровно одно слово:) (\S+?)\.", last,
                           re.IGNORECASE)
        if codes:
            return codes[-1]
        if "простых чисел" in last:
            return "15" if "reasoning_wrong_answer" in self.defects else "16"
        if "17 × 23" in last:
            return "391"
        if "7 умножить на 8" in last:
            return "56"
        if "Напиши" in last:
            word = re.search(r"со слова (\w+)", last)
            return " ".join((word.group(1) if word else "маяк") + str(index) for index in range(60))
        return "ОК"

    @staticmethod
    def arguments(parameters):
        result = {}
        for name, schema in parameters.get("properties", {}).items():
            result[name] = schema["enum"][0] if "enum" in schema else "Казань"
        return result

    # --- запуск ---------------------------------------------------------
    def start(self, port=0):
        self.server = ThreadingHTTPServer(("127.0.0.1", port), handler(self))
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.log("event=defaults thinking=on")
        self.log("event=load_completed kind=text speculative=" + self.speculative + " draft_limit=1")
        return self.server.server_address[1]

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()


def handler(fake):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *_):
            pass

        def reply(self, payload, identifier=None, status=200):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if identifier:
                self.send_header("X-Request-ID", identifier)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            payload = {"/health": {"status": "ok"}, "/ready": {"status": "ready", "model": ALIAS},
                       "/v1/models": {"object": "list", "data": [{"object": "model", "id": ALIAS}]}}.get(self.path)
            if payload is None:
                return self.send_error(404)
            self.reply(payload)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with fake.lock:
                fake.counter += 1
                identifier = self.identifier = f"r{fake.counter}"
                fake.inflight[identifier] = 1
                for other in fake.inflight:  # Наибольшая ширина за время жизни запроса.
                    fake.peak[other] = max(fake.peak.get(other, 1), len(fake.inflight))
            try:
                if self.path == "/v1/completions":
                    self.serve_completion(body, identifier)
                elif self.path == "/v1/responses":
                    self.serve_responses(body, identifier)
                elif self.path == "/v1/chat/completions":
                    self.serve_chat(body, identifier)
                else:
                    self.send_error(404)
            finally:
                with fake.lock:
                    fake.inflight.pop(identifier, None)

        # --- общая работа запроса --------------------------------------
        def prefill(self, prompt, stable, cache, progress=None):
            """Вернуть (cached, source, groups); чтение длинного prompt занимает время."""
            cached, source = fake.lookup(prompt) if cache else (0, "miss")
            with fake.lock:
                groups = 8 if any(value == 2 for value in fake.inflight.values()) else 1
                for reader in fake.readers:  # Поздний общий префикс внутри чужого прохода.
                    common = next((index for index, (a, b) in enumerate(zip(reader["prompt"], prompt)) if a != b),
                                  min(len(reader["prompt"]), len(prompt)))
                    if TAIL <= common < GRID:
                        reader["late"] = True
            grid = True
            if len(prompt) > GRID + TAIL and cached < GRID:
                reader = {"prompt": prompt, "late": False}
                with fake.lock:
                    fake.readers.append(reader)
                time.sleep(.5 * fake.pace)
                with fake.lock:
                    fake.readers.remove(reader)
                grid = not (reader["late"] and "late_share_loses_grid" in fake.defects)
                if grid:
                    with fake.lock:
                        fake.ram.append(tuple(prompt[:GRID]))
                if progress:
                    progress(GRID, len(prompt), cached)
                time.sleep(.2 * fake.pace)
            if progress:
                progress(len(prompt), len(prompt), cached)
            fake.publish(prompt, stable, grid)
            return cached, source, groups

        def usage(self, prompt, cached, source, text, reasoning, groups, follower=False):
            completion = max(1, len(text.split())) + reasoning
            prefill = len(prompt) - cached
            if "accounting" in fake.defects or follower and "follower_accounting" in fake.defects:
                prefill += 1
            proposed = max(2, completion // 2) if fake.speculative == "mtp" else 0
            accepted = max(1, proposed - 1) if proposed else 0
            if "no_drafts" in fake.defects or "no_long_drafts" in fake.defects and len(prompt) > 100000:
                proposed = accepted = 0
            with fake.lock:
                width = max(1, len(fake.inflight), fake.peak.pop(self.identifier, 1))
            if "width1" in fake.defects:
                width = 1
            disk = source == "disk" and "no_disk_flag" not in fake.defects
            usage = {"prompt_tokens": len(prompt), "completion_tokens": completion,
                     "total_tokens": len(prompt) + completion, "cached_tokens": cached,
                     "prompt_tokens_details": {"cached_tokens": cached},
                     "completion_tokens_details": {"reasoning_tokens": reasoning},
                     "draft_tokens": proposed, "draft_tokens_accepted": accepted,
                     "gufo": {"prefill_tokens": prefill, "prefill_ms": prefill * .01, "decode_ms": 3.0,
                              "queue_ms": 0.0, "ttft_ms": 3.0, "cache_restore_ms": 0, "cache_snapshot_ms": 0,
                              "cache_disk_enqueue_ms": 0, "cache_hit": cached > 0, "cache_disk_hit": disk,
                              "physical_execution_width": width, "max_prefill_layer_groups": groups,
                              "prefill_chunks": 2 if len(prompt) > GRID else 1, "shared_prefix_wait_ms": 0.0}}
            if "missing_drafts" in fake.defects:
                del usage["draft_tokens"], usage["draft_tokens_accepted"]
            return usage

        def completed(self, identifier, usage, finish, source):
            cache = "memory" if source == "disk" and "no_disk_flag" in fake.defects else source
            detail = usage["gufo"]
            fake.log(f"request={identifier} event=completed method=POST path={self.path} status=200 "
                     f"duration_ms=1.0 outcome=ok prompt_tokens={usage['prompt_tokens']} "
                     f"prefill_tokens={detail['prefill_tokens']} generated_tokens={usage['completion_tokens']} "
                     f"finish={finish} cache={cache} cached_tokens={usage['cached_tokens']} "
                     f"batch_width={detail['physical_execution_width']} "
                     f"draft_accepted={usage.get('draft_tokens_accepted', 0)} "
                     f"draft_proposed={usage.get('draft_tokens', 0)}")

        def reasoning_tokens(self, body):
            if not body.get("chat_template_kwargs", {}).get("enable_thinking"):
                return 0
            budget = body.get("reasoning_budget_tokens")
            if budget is not None and "budget_leak" in fake.defects:
                fake.budget_leak = budget
            limit = budget if budget is not None else fake.budget_leak or 512
            natural = 300 if "reasoning_short" in fake.defects else 900
            return min(natural, limit) + (5 if "reasoning_over" in fake.defects else 0)

        # --- Chat --------------------------------------------------------
        def serve_chat(self, body, identifier):
            tokens, texts, pictures = fake.render(body["messages"])
            prompt, stable = tokens + ["<assistant>", "<think>"], len(tokens)
            stream = body.get("stream")
            envelope = {"id": "chat-" + identifier, "created": 1, "model": body["model"],
                        "object": "chat.completion.chunk" if stream else "chat.completion"}
            connected = True

            def write(event):
                nonlocal connected
                if connected:
                    try:
                        self.wfile.write(b"data: " + (event if isinstance(event, bytes) else json.dumps(
                            event, ensure_ascii=False).encode()) + b"\n\n")
                        self.wfile.flush()
                    except OSError:
                        connected = False
                return connected

            def chunk(delta, finish=None, **extra):
                return write({**envelope, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                              **extra})
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("X-Request-ID", identifier)
                self.end_headers()
            progress = None
            if stream and body.get("return_progress"):
                chunk({"role": "assistant"})
                progress = lambda processed, total, cache: chunk({}, prompt_progress={
                    "total": total, "cache": cache, "processed": processed, "time_ms": 1})
            follower = any(TAIL <= len(reader["prompt"]) for reader in fake.readers)
            cached, source, groups = self.prefill(prompt, stable, body.get("cache_prompt", True), progress)
            named = isinstance(body.get("tool_choice"), dict)
            calls = None
            if body.get("tools") and named:
                function = body["tools"][0]["function"]
                arguments = fake.arguments(function["parameters"])
                if "tool_args" in fake.defects:
                    arguments = {name: "ошибка" for name in arguments}
                calls = [{"id": "call_" + identifier, "type": "function", "function": {
                    "name": function["name"], "arguments": json.dumps(arguments, ensure_ascii=False)}}]
                text = ""
            else:
                text = fake.answer(body, texts, pictures)
                if follower and "follower_code_block" in fake.defects:
                    # Поведение из первого окна: блок кода вместо одного слова, обрыв по лимиту.
                    text = "```python\n" + text + ' = "011, 048, 085, 122, 159, 196, 233, 270, 307, 344, 381"'
            reasoning = self.reasoning_tokens(body)
            if reasoning and "empty_after_reasoning" in fake.defects:
                text = ""
            limit = body.get("max_completion_tokens", 64)
            words = text.split(" ")
            finish = "tool_calls" if calls else "stop"
            if len(words) + reasoning > limit:
                words, finish = words[:max(1, limit - reasoning)], "length"
            text = " ".join(words)
            with fake.lock:
                fake.inflight[identifier] = 2  # Декодирует: соседнее чтение идёт слоями.
            usage = None
            long = len(words) > 20
            if not stream:
                time.sleep((.8 if "произведения" in texts[-1][1] else .02) * fake.pace)
                usage = self.usage(prompt, cached, source, text, reasoning, groups, follower)
                message = {"role": "assistant", "content": text or None}
                if reasoning:
                    message["reasoning_content"] = "шаг " * reasoning
                if calls:
                    message["tool_calls"] = calls
                self.completed(identifier, usage, finish, source)
                return self.reply({**envelope, "usage": usage, "choices": [
                    {"index": 0, "message": message, "finish_reason": finish}]}, identifier)
            if not body.get("return_progress"):
                chunk({"role": "assistant"})
            if reasoning:
                chunk({"reasoning_content": "шаг " * reasoning})
            if calls:
                call = calls[0]
                chunk({"tool_calls": [{"index": 0, "id": call["id"], "type": "function", "function": {
                    "name": call["function"]["name"], "arguments": ""}}]})
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": call["function"]["arguments"]}}]})
            pause = (.01 if long else .8 / max(1, len(words)) if "произведения" in texts[-1][1] else .002) * fake.pace
            for index, word in enumerate(words if text else []):
                if not chunk({"content": word + (" " if index + 1 < len(words) else "")}):
                    break
                time.sleep(pause)
            usage = self.usage(prompt, cached, source, text, reasoning, groups, follower)
            if not connected and "no_cancel_log" not in fake.defects:
                return self.completed(identifier, usage, "cancelled", source)
            chunk({}, finish)
            write({**envelope, "choices": [], "usage": usage})
            write(b"[DONE]")
            self.completed(identifier, usage, finish, source)

        # --- Legacy Completions -------------------------------------------
        def serve_completion(self, body, identifier):
            prompt = body["prompt"].split()
            cached, source, groups = self.prefill(prompt, len(prompt), body.get("cache_prompt", True))
            text = " запись " + hashlib.sha256(body["prompt"].encode()).hexdigest()[:8]
            usage = self.usage(prompt, cached, source, text, 0, groups)
            self.completed(identifier, usage, "stop", source)
            self.reply({"id": "cmpl-" + identifier, "object": "text_completion", "created": 1,
                        "model": body["model"], "usage": usage, "choices": [
                            {"text": text, "index": 0, "logprobs": None, "finish_reason": "stop"}]}, identifier)

        # --- Responses ------------------------------------------------------
        def serve_responses(self, body, identifier):
            messages = [{"role": item.get("role", "tool"), "content": item.get("content", item.get("output", ""))}
                        for item in body["input"] if item.get("type") != "function_call"]
            tokens, texts, pictures = fake.render(messages)
            prompt = tokens + ["<assistant>", "<think>"]
            cached, source, groups = self.prefill(prompt, len(tokens), body.get("cache_prompt", True))
            if body.get("tools") and isinstance(body.get("tool_choice"), dict):
                tool = body["tools"][0]
                arguments = json.dumps(fake.arguments(tool["parameters"]), ensure_ascii=False)
                item = {"id": "fc_" + identifier, "type": "function_call", "status": "completed",
                        "call_id": "call_" + identifier, "name": tool["name"], "arguments": arguments}
                text, middle = arguments, [
                    {"type": "response.function_call_arguments.delta", "item_id": item["id"], "delta": arguments},
                    {"type": "response.function_call_arguments.done", "item_id": item["id"],
                     "arguments": arguments}]
                opener = {**item, "status": "in_progress", "arguments": ""}
            else:
                text = fake.answer(body, texts, pictures)
                item = {"id": "msg_" + identifier, "type": "message", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]}
                middle = [{"type": "response.output_text.delta", "item_id": item["id"], "content_index": 0,
                           "delta": text},
                          {"type": "response.output_text.done", "item_id": item["id"], "content_index": 0,
                           "text": text}]
                opener = {**item, "status": "in_progress", "content": []}
            chat_usage = self.usage(prompt, cached, source, text, 0, groups)
            detail = chat_usage["gufo"]
            final = {"id": "resp_" + identifier, "object": "response", "created_at": 1, "model": body["model"],
                     "status": "completed", "error": None, "incomplete_details": None, "output": [item],
                     "usage": {"input_tokens": len(prompt), "output_tokens": chat_usage["completion_tokens"],
                               "total_tokens": chat_usage["total_tokens"],
                               "input_tokens_details": {"cached_tokens": cached},
                               "output_tokens_details": {"reasoning_tokens": 0}},
                     "timings": {"prompt_n": detail["prefill_tokens"], "prompt_ms": detail["prefill_ms"],
                                 "predicted_ms": 3.0, "cache_restore_ms": 0, "cache_snapshot_ms": 0,
                                 "cache_disk_enqueue_ms": 0, "draft_n": chat_usage.get("draft_tokens", 0),
                                 "draft_n_accepted": chat_usage.get("draft_tokens_accepted", 0)}}
            events = [{"type": "response.created", "response": {**final, "status": "in_progress", "output": [],
                                                                "usage": None}},
                      {"type": "response.output_item.added", "output_index": 0, "item": opener},
                      *[{**event, "output_index": 0} for event in middle],
                      {"type": "response.output_item.done", "output_index": 0, "item": item},
                      {"type": "response.completed", "response": final}]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("X-Request-ID", identifier)
            self.end_headers()
            for index, event in enumerate(events):
                self.wfile.write(b"data: " + json.dumps({**event, "sequence_number": index},
                                                        ensure_ascii=False).encode() + b"\n\n")
            self.completed(identifier, chat_usage, "stop", source)
    return Handler


def main():
    if sys.argv[1:] == ["--version"]:
        print("gufo version development (" + "1" * 40 + ")")
        return 0

    def option(name, default=None):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
    cache = option("--cache-disk")
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
    defects = [name for name in os.environ.get("FAEX2_FAKE_DEFECTS", "").split(",") if name]
    fake = Fake(cache, None, defects, option("--speculative", "mtp"),
                float(os.environ.get("FAEX2_FAKE_PACE", "1")))
    if "exit_at_start" in defects:
        return 77
    fake.start(int(option("--port")))
    if "--mmproj" in sys.argv:
        fake.log("event=vision_prepared resident_bytes=8192")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
