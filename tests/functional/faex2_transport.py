#!/usr/bin/env python3
"""Один ограниченный loopback HTTP-запрос с сохранением wire-доказательств."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import ipaddress
import json
import math
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit

from faex2 import hard_timeout, require
from metrics import digest, integer_number, summarize

MAX_RESPONSE_BYTES = 16 * 1024 * 1024
GET_PATHS = {"/health", "/v1/health", "/ready", "/v1/ready", "/v1/models", "/props"}
POST_PATHS = {"/v1/chat/completions", "/v1/responses", "/tokenize"}
FRAME_END = re.compile(br"\r?\n\r?\n")
RESPONSE_EVENTS = {"response." + suffix for suffix in (
    "created", "in_progress", "completed", "incomplete", "output_item.added", "output_item.done",
    "content_part.added", "content_part.done", "reasoning_summary_part.added", "reasoning_summary_part.done",
    "output_text.delta", "output_text.done", "reasoning_summary_text.delta", "reasoning_summary_text.done",
    "function_call_arguments.delta", "function_call_arguments.done")}


def decode_json(raw):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("non-finite JSON number: " + value)

    return json.loads(raw, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def validate_request(request):
    require(isinstance(request, dict) and set(request) == {"method", "path", "body"},
            "request must contain exactly method/path/body")
    method, path, body = request["method"], request["path"], request["body"]
    require(isinstance(path, str), "path must be a string")
    require(method == "GET" and path in GET_PATHS or method == "POST" and path in POST_PATHS,
            "method/path outside the local whitelist")
    require(isinstance(body, dict) or method == "GET" and body is None, "body must be an object")
    if method == "GET":
        require(body in (None, {}), "GET body must be null or empty")
    if isinstance(body, dict) and "stream" in body:
        require(type(body["stream"]) is bool, "stream must be a boolean")
    return request


def loopback_address(base):
    url = urlsplit(base)
    require(url.scheme == "http" and not url.username and not url.password and
            not url.path and not url.query and not url.fragment,
            "base URL must be an explicit http loopback origin without a path")
    require(url.hostname and ipaddress.ip_address(url.hostname).is_loopback,
            "only a numeric loopback address is allowed")
    require(url.port is not None and 1 <= url.port <= 65535, "explicit port required")
    return url.hostname, url.port


def sse_events(raw, ended):
    """Сохраняет смещение конца кадра; незавершённый кадр допустим только при отмене."""
    events, position, done = [], 0, False
    for boundary in FRAME_END.finditer(raw):
        frame = raw[position:boundary.start()]
        position = boundary.end()
        data, label = [], None
        for line in frame.splitlines():
            if not line or line.startswith(b":"):
                continue
            field, separator, value = line.partition(b":")
            require(separator and field in (b"data", b"event", b"id", b"retry"),
                    "invalid SSE field")
            value = value.removeprefix(b" ")
            if field == b"data":
                data.append(value)
            elif field == b"event":
                require(label is None, "duplicate SSE event label")
                label = value.decode("utf-8")
        if not data:
            continue
        require(not done, "events after terminal DONE")
        # Gufo emits one JSON data line per event; shared metrics has this contract.
        require(len(data) == 1, "multiple SSE data lines are not supported by Gufo contract")
        if data[0] == b"[DONE]":
            done = True
            continue
        event = decode_json(data[0])
        require(isinstance(event, dict), "SSE data must be an object")
        require(label is None or label == event.get("type"), "SSE event label differs from type")
        events.append((event, position))
    require(not ended or not raw[position:].strip(), "incomplete SSE frame")
    return events, done


def visible_delta(event, responses):
    if responses:
        return event.get("delta") if event.get("type") == "response.output_text.delta" else None
    choices = event.get("choices", [])
    return next((choice.get("delta", {}).get("content") for choice in choices
                 if isinstance(choice, dict) and isinstance(choice.get("delta"), dict)
                 and choice["delta"].get("content")), None) if isinstance(choices, list) else None


def envelope(value, body, responses, streaming=False):
    require(isinstance(value, dict), "response envelope must be an object")
    expected = "response" if responses else "chat.completion.chunk" if streaming else "chat.completion"
    require(value.get("object") == expected, "wrong response object")
    require(isinstance(value.get("id"), str) and value["id"], "missing/invalid response id")
    created = value.get("created_at" if responses else "created")
    require(integer_number(created) and created > 0, "missing/invalid created time")
    require(value.get("model") == body.get("model") and isinstance(value.get("model"), str),
            "wrong response model")


def optional_text(value, field):
    require(value is None or isinstance(value, str), field + " must be text or null")
    return value or ""


def parse_chat(events, body, streaming, ended):
    text, reasoning, tools, finish, identifier, usage = "", "", {}, None, None, {}
    for event in events:
        envelope(event, body, False, streaming)
        require(identifier is None or identifier == event["id"], "mixed response ids")
        identifier = event["id"]
        require(isinstance(event.get("choices"), list) and len(event["choices"]) <= 1,
                "expected one choice or a terminal usage chunk")
        if event.get("usage") is not None:
            require(isinstance(event["usage"], dict), "usage must be an object")
            usage = event["usage"]
        for choice in event["choices"]:
            require(isinstance(choice, dict) and type(choice.get("index")) is int
                    and choice["index"] == 0, "unexpected choice index")
            require(finish is None, "choice after finish")
            message = choice.get("delta" if streaming else "message")
            require(isinstance(message, dict), "missing message/delta")
            require(message.get("role", "assistant" if streaming else None) == "assistant",
                    "wrong assistant role")
            text += optional_text(message.get("content"), "content")
            reasoning += optional_text(message.get("reasoning_content"), "reasoning_content")
            calls = message.get("tool_calls")
            require(calls is None or isinstance(calls, list), "tool_calls must be an array")
            for index, call in enumerate(calls or []):
                require(isinstance(call, dict), "tool call must be an object")
                slot = call.get("index") if streaming else index
                require(type(slot) is int and 0 <= slot <= len(tools), "invalid tool index")
                initial = slot not in tools
                require(call.get("type", None if initial else "function") == "function", "wrong tool type")
                target = tools.setdefault(slot, {"id": "", "type": "function",
                                                "function": {"name": "", "arguments": ""}})
                if "id" in call:
                    require(isinstance(call["id"], str) and call["id"] and not target["id"],
                            "missing/duplicate/invalid tool id")
                    target["id"] = call["id"]
                function = call.get("function")
                require(isinstance(function, dict), "tool function must be an object")
                for field in ("name", "arguments"):
                    require(field not in function or isinstance(function[field], str),
                            "tool " + field + " must be a string")
                    target["function"][field] += function.get(field, "")
            finish = choice.get("finish_reason")
    require(not ended or finish is not None, "missing terminal choice")
    for call in tools.values():
        require(call["id"] and call["function"]["name"], "missing tool id or name")
    return text, reasoning, list(tools.values()), finish, usage


def response_items(items):
    require(isinstance(items, list), "Responses output must be an array")
    text, reasoning, tools, ids = "", "", [], set()
    for item in items:
        require(isinstance(item, dict) and isinstance(item.get("id"), str)
                and item["id"] and item["id"] not in ids, "missing/duplicate Responses item id")
        ids.add(item["id"])
        require(item.get("status") in ("completed", "incomplete"), "invalid Responses item status")
        kind = item.get("type")
        if kind == "function_call":
            require(all(isinstance(item.get(key), str) and item[key] for key in ("call_id", "name"))
                    and isinstance(item.get("arguments"), str), "invalid Responses function call")
            tools.append(item)
            continue
        require(kind in ("reasoning", "message"), "unexpected Responses item type")
        require(kind != "message" or item.get("role") == "assistant", "wrong Responses role")
        parts = item.get("summary" if kind == "reasoning" else "content")
        require(isinstance(parts, list), "missing Responses item content")
        for part in parts:
            require(isinstance(part, dict) and part.get("type") == (
                "summary_text" if kind == "reasoning" else "output_text")
                and isinstance(part.get("text"), str), "invalid Responses text part")
            if kind == "reasoning": reasoning += part["text"]
            else: text += part["text"]
    return text, reasoning, tools


def parse_responses(events, body, streaming, ended):
    final, identifier, items, completed, deltas = None, None, {}, {}, {}
    for event in events:
        if streaming:
            require(type(event.get("sequence_number")) is int, "invalid sequence number")
            kind = event.get("type")
            require(isinstance(kind, str) and kind in RESPONSE_EVENTS, "invalid Responses event type")
            value = event.get("response")
        else:
            value, kind = event, None
        require(final is None, "events after terminal Responses output")
        if value is not None:
            envelope(value, body, True)
            require(identifier is None or identifier == value["id"], "mixed Responses ids")
            identifier = value["id"]
            require(value.get("status") in ("in_progress", "completed", "incomplete"),
                    "failed or invalid Responses status")
            if value["status"] in ("completed", "incomplete"):
                require(not streaming or kind == "response." + value["status"], "terminal status/type mismatch")
                final = value
            elif streaming:
                require(kind in ("response.created", "response.in_progress"), "unexpected lifecycle event")
        if not streaming:
            continue
        index = event.get("output_index")
        if kind == "response.output_item.added":
            item = event.get("item")
            require(type(index) is int and index == len(items) and isinstance(item, dict),
                    "invalid Responses item opener")
            require(isinstance(item.get("id"), str) and item["id"] and
                    item["id"] not in [entry["id"] for entry in items.values()], "invalid Responses item id")
            require(item.get("type") in ("message", "reasoning", "function_call"), "invalid item type")
            items[index] = item
        elif kind in ("response.output_text.delta", "response.reasoning_summary_text.delta",
                      "response.output_text.done", "response.reasoning_summary_text.done"):
            reasoning = "reasoning_summary" in kind
            item = items.get(index)
            require(type(index) is int and item is not None and index not in completed
                    and event.get("item_id") == item["id"] and item.get("type") == (
                        "reasoning" if reasoning else "message"), "text event has no matching active item")
            part_index = event.get("summary_index" if reasoning else "content_index")
            require(type(part_index) is int and part_index == 0, "unexpected Responses content index")
            if kind.endswith(".delta"):
                require(isinstance(event.get("delta"), str), "text delta must be a string")
                deltas[index] = deltas.get(index, "") + event["delta"]
            else:
                require(event.get("text") == deltas.get(index, ""), "text.done differs from deltas")
        elif kind == "response.output_item.done":
            item = event.get("item")
            require(type(index) is int and index in items and index not in completed
                    and isinstance(item, dict) and all(item.get(key) == items[index].get(key)
                                                       for key in ("id", "type")), "item.done differs from opener")
            text, reasoning, _ = response_items([item])
            if item["type"] != "function_call":
                require(text + reasoning == deltas.get(index, ""), "completed text differs from deltas")
            completed[index] = item
    require(not ended or final is not None, "missing terminal Responses output")
    if final is None:
        text = "".join(deltas.get(index, "") for index, item in items.items() if item["type"] == "message")
        reasoning = "".join(deltas.get(index, "") for index, item in items.items() if item["type"] == "reasoning")
        return text, reasoning, [], None, {}
    text, reasoning, tools = response_items(final.get("output"))
    if streaming:
        require(len(items) == len(completed) == len(final["output"]) and
                [completed[index] for index in range(len(completed))] == final["output"],
                "terminal output differs from streamed items")
    usage = final.get("usage")
    require(isinstance(usage, dict), "Responses usage must be an object")
    details = usage.get("output_tokens_details")
    require(isinstance(details, dict), "missing output_tokens_details")
    count = details.get("reasoning_tokens")
    total = usage.get("output_tokens")
    require(integer_number(count) and integer_number(total) and 0 <= count <= total,
            "invalid reasoning_tokens")
    return text, reasoning, tools, final["status"], usage


def _parse_response(request, raw, parts, status, ended):
    validate_request(request)
    require(type(status) is int and 200 <= status <= 599, "invalid HTTP status")
    require(parts and b"".join(chunk for _, chunk in parts) == raw, "parts do not match raw response")
    previous = 0
    for elapsed, chunk in parts:
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= previous,
                "invalid chunk timestamp")
        previous = elapsed
    body, path = request["body"] or {}, request["path"]
    streaming = bool(body.get("stream")) and status < 400
    if streaming:
        records, done = sse_events(raw, ended)
        require(path == "/v1/responses" and not done or path == "/v1/chat/completions" and (done or not ended),
                "missing/unexpected terminal DONE")
    else:
        value = decode_json(raw)
        require(isinstance(value, dict), "response must be an object")
        records = [(value, len(raw))]
    events = [event for event, _ in records]
    require(events, "empty response")
    text, reasoning, tools, finish, usage = "", "", [], None, {}
    if status >= 400:
        require(not streaming, "HTTP error must be JSON")
    elif path == "/v1/chat/completions":
        text, reasoning, tools, finish, usage = parse_chat(events, body, streaming, ended)
    elif path == "/v1/responses":
        text, reasoning, tools, finish, usage = parse_responses(events, body, streaming, ended)
    else:
        require(status == 200 and not streaming, "unexpected discovery/tokenize status or stream")
    measured, fingerprint = summarize(parts, streaming, ended, (path, body, status))
    first_visible, first_reasoning = None, None
    offset, part_index = len(parts[0][1]), 0
    for event, consumed in records:
        while offset < consumed:
            part_index += 1
            offset += len(parts[part_index][1])
        timestamp = parts[part_index][0]
        if streaming:
            visible = visible_delta(event, path == "/v1/responses")
            thought = event.get("delta") if event.get("type") == "response.reasoning_summary_text.delta" else any(
                choice.get("delta", {}).get("reasoning_content") for choice in event.get("choices", []))
        else:
            visible, thought = text, reasoning
        if first_visible is None and visible: first_visible = timestamp
        if first_reasoning is None and thought: first_reasoning = timestamp
    if first_visible is not None: measured["client_ttfa_ms"] = first_visible
    result = {"status": "HTTP_ERROR" if status >= 400 else "PASS", "http_status": status,
              "text": text, "reasoning": reasoning, "tools": tools, "finish": finish,
              "usage": usage, "metrics": measured, "first_visible_content_ms": first_visible,
              "first_reasoning_ms": first_reasoning, "output_sha256": fingerprint}
    if path == "/v1/responses" and status == 200:
        result["response_status"] = finish
        result["finish"] = ("tool_calls" if tools else "stop") if finish == "completed" else finish
    if status >= 400: result["error"] = events[0]["error"]
    if path not in ("/v1/chat/completions", "/v1/responses"):
        result["response"] = events[0]
        result["output_sha256"] = digest(events[0])
    return result


def parse_response(request, raw, parts, status):
    """Чистая проверка завершённого ответа; parts — (elapsed_ms, bytes)."""
    try:
        return _parse_response(request, raw, parts, status, True)
    except (AssertionError, KeyError, TypeError, AttributeError, UnicodeError) as error:
        raise ValueError(str(error)) from error


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run_request(base_url, request_path, output, timeout=120, cancel_after_content=None):
    """Запускается в главном потоке отдельного процесса; таймер охватывает весь HTTP."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = {name: str(output / name) for name in
             ("request.json", "request.body.raw", "response.raw", "parts.json", "parts.jsonl")}
    result = {"status": "FAIL", "http_status": None, "text": "", "reasoning": "", "tools": [],
              "finish": None, "usage": {}, "metrics": {}, "output_sha256": None,
              "first_visible_content_ms": None, "first_reasoning_ms": None,
              "intentional_cancel": False, "raw_paths": paths}
    parts, metadata, connection, response = [], [], None, None
    started = time.monotonic()
    wire_done = None  # Конец передачи: EOF тела либо намеренная отмена.
    raw_path = output / "response.raw"
    raw_path.touch()
    try:
        require(not sys.flags.optimize, "run without -O: shared validators use assertions")
        require(type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= 600,
                "timeout must be greater than zero and at most 600 seconds")
        with hard_timeout(timeout):
            original = Path(request_path).read_bytes()
            (output / "request.json").write_bytes(original)
            request = validate_request(decode_json(original))
            host, port = loopback_address(base_url)
            body = request["body"] or {}
            if cancel_after_content is not None:
                require(type(cancel_after_content) is int and cancel_after_content > 0 and
                        body.get("stream") is True and request["path"] in (
                            "/v1/chat/completions", "/v1/responses"),
                        "cancellation requires a streaming generation request and a positive delta count")
            payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode() if request["method"] == "POST" else b""
            (output / "request.body.raw").write_bytes(payload)
            connection = http.client.HTTPConnection(host, port, timeout=timeout)
            connection.request(request["method"], request["path"], body=payload,
                               headers={"Content-Type": "application/json", "Accept-Encoding": "identity"})
            response = connection.getresponse()
            result["http_status"] = response.status
            result["headers"] = response.getheaders()
            result["request_id"] = response.getheader("x-request-id")
            streaming = body.get("stream") is True and response.status < 400
            size, pending, seen_content = 0, b"", 0
            with raw_path.open("wb", buffering=0) as destination:
                while True:
                    block = response.read1(min(65536, MAX_RESPONSE_BYTES - size + 1))
                    if not block:
                        require(response.length in (None, 0), "response ended before Content-Length")
                        break
                    kept = block[:MAX_RESPONSE_BYTES - size]
                    if kept:
                        destination.write(kept)
                        elapsed = (time.monotonic() - started) * 1000
                        parts.append((elapsed, kept))
                        metadata.append({"elapsed_ms": elapsed, "offset": size, "size": len(kept)})
                        size += len(kept)
                        # Сохранять тайминги одновременно с байтами, включая неуспешный запрос.
                        with (output / "parts.jsonl").open("a") as timings:
                            timings.write(json.dumps(metadata[-1]) + "\n")
                    require(len(block) == len(kept), "response exceeds 16 MiB limit")
                    if cancel_after_content is not None and streaming:
                        pending += block
                        last, terminal_received = 0, False
                        for boundary in FRAME_END.finditer(pending):
                            events, done = sse_events(pending[last:boundary.end()], True)
                            last = boundary.end()
                            seen_content += sum(bool(visible_delta(event, request["path"] == "/v1/responses"))
                                                for event, _ in events)
                            terminal_received |= done or any(
                                event.get("type") in ("response.completed", "response.incomplete") or
                                any(choice.get("finish_reason") is not None
                                    for choice in event.get("choices", []))
                                for event, _ in events)
                        pending = pending[last:]
                        result["cancel_after_content"] = cancel_after_content
                        result["received_content_deltas"] = seen_content
                        if terminal_received:
                            # read1 может вернуть content и terminal вместе: это уже не отмена.
                            result["error_code"] = "COMPLETE_BEFORE_CANCEL"
                            raise ValueError("COMPLETE_BEFORE_CANCEL: terminal output arrived before cancellation")
                        if seen_content >= cancel_after_content:
                            result["intentional_cancel"] = True
                            response.close()
                            connection.close()
                            break
            wire_done = time.monotonic()
            kind = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
            require(kind == ("text/event-stream" if streaming else "application/json"),
                    "unexpected response Content-Type: " + kind)
            require(response.getheader("Content-Encoding", "identity").lower() == "identity",
                    "compressed wire responses are not supported")
            raw = b"".join(block for _, block in parts)
            parsed = _parse_response(request, raw, parts, response.status, not result["intentional_cancel"])
            result.update(parsed)
            if result["intentional_cancel"]:
                result["status"] = "CANCELLED"
                result["cancel_after_content"] = cancel_after_content
                result["received_content_deltas"] = seen_content
    except Exception as error:
        result.update(status="FAIL", error=str(error), error_type=type(error).__name__)
    finally:
        if response is not None: response.close()
        if connection is not None: connection.close()
        # wall_ms — время HTTP-обмена до конца тела. Локальные разбор, проверки и
        # close не являются задержкой сервера; при ошибке до конца тела границей
        # остаётся момент отказа.
        finished = time.monotonic()
        result["wall_ms"] = ((wire_done if wire_done is not None else finished) - started) * 1000
        result["client_total_ms"] = (finished - started) * 1000
        result["client_post_wire_ms"] = result["client_total_ms"] - result["wall_ms"]
        result["response_bytes"] = sum(len(block) for _, block in parts)
        result["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        write_json(output / "parts.json", metadata)
        write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    request = sub.add_parser("request")
    request.add_argument("--base-url", required=True)
    request.add_argument("--request", required=True)
    request.add_argument("--output", required=True)
    request.add_argument("--timeout", type=float, default=120)
    request.add_argument("--cancel-after-content", type=int)
    args = parser.parse_args()
    try:
        result = run_request(args.base_url, args.request, args.output, args.timeout, args.cancel_after_content)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "output": args.output}, ensure_ascii=False))
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
