#!/usr/bin/env python3
"""Раннер прикладного набора FAEX2 (Б2): run / export-blind / unblind.

Клиент сервер не запускает и не останавливает. Каждый HTTP-обмен исполняет
faex2_transport в отдельном процессе; запрос, сырой ответ и времена
сохраняются до любой проверки. Статусы задач: PASS, FAIL, NEEDS_REVIEW
(ручная рубрика), KNOWN_LIMIT_FAIL (известное свойство крупного текста),
NOT_EXERCISED (не выбран фильтром).

    faex2_appset_runner.py run --base-url http://127.0.0.1:13345 \
        --output NEW --pass-name greedy --seed 0
    faex2_appset_runner.py export-blind --run-a DIR_A --run-b DIR_B \
        --out NEW --seed 20261119
    faex2_appset_runner.py unblind --blind DIR --reviews blind_reviews.jsonl

Политика thinking/cache_prompt — из дизайн-документа §3.4: точечные задачи с
эталоном идут без рассуждений, многошаговые агентские сценарии — с рабочим
профилем Hermes; на ходах 2+ включается cache_prompt.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

from faex2 import ALIAS, file_hash, require
import faex2_appset as appset
import hermes_synth
from run import write_json

BLIND_SEED_DEFAULT = 20261119
RUBRIC_FILE = "faex2_quality.jsonl"
APPSET_FILE = "faex2_appset.jsonl"


def fixture_hashes() -> dict:
    root = Path(__file__).parent
    files = (APPSET_FILE, RUBRIC_FILE, "faex2_appset.py", "faex2_appset_runner.py",
             "faex2_remediation_images.py", "faex2_vision_probes.py", "faex2_images.py",
             "hermes_synth.py", "fixtures/hermes_synth/base_request.json",
             "fixtures/hermes_synth/cases.jsonl")
    return {name: file_hash(root / name) for name in files}

# ---------------------------------------------------------------------------
# Нормализация и автопроверки (чистые функции, без модели)
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u2212", "-").replace("\u2013", "-")).strip()


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", normalize(text).rstrip("."))


def _numbers(text: str) -> list:
    return [int(value) for value in re.findall(r"-?\d+", normalize(text))]


def _same_json(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_json(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_json(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _sentences(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+(?:\s|$)", normalize(text)) if part])


def _bullets(text: str) -> int:
    return len([line for line in text.splitlines()
                if re.match(r"\s*([-*•]|\d+[.)])\s+\S", line)])


def _valid_tool_call(result: dict) -> bool:
    if result.get("finish") != "tool_calls" or len(result.get("tools") or []) != 1:
        return False
    call = result["tools"][0]
    if call.get("type") != "function" or not call.get("id"):
        return False
    try:
        return isinstance(json.loads(call["function"]["arguments"]), dict)
    except (ValueError, TypeError):
        return False


def grade_long_facts(task: dict, text: str) -> list:
    """Две независимые проверки значений в заданном порядке, без поблажек типам."""
    expected = task["autocheck"]["value"]
    require(task["domain"] == "long-facts" and len(expected) == 2,
            "expected long-facts with two values")
    try:
        actual = json.loads(text)
    except (ValueError, TypeError):
        actual = []
    if not isinstance(actual, list) or len(actual) > 2:
        actual = []
    return [index < len(actual) and type(actual[index]) is int and actual[index] == value
            for index, value in enumerate(expected)]


def summarize_long_facts(rows: list, tasks: list, pass_name: str = "greedy") -> dict:
    """Отдельный описательный gate 22/24; неполные данные никогда не проходят."""
    if pass_name != "greedy":
        return {"status": "NOT_EXERCISED", "correct": None, "total": 24}
    expected = {task["id"] for task in tasks if task["domain"] == "long-facts"}
    selected = [row for row in rows if row.get("domain") == "long-facts"]
    if len(expected) != 12 or len(selected) != 12 or {r["id"] for r in selected} != expected:
        return {"status": "NOT_EXERCISED", "correct": None, "total": 24}
    checks = [row.get("fact_checks") for row in selected]
    if any(not isinstance(check, list) or len(check) != 2 or
           any(type(value) is not bool for value in check) for check in checks):
        return {"status": "NOT_EXERCISED", "correct": None, "total": 24}
    correct = sum(sum(check) for check in checks)
    return {"status": "PASS" if correct >= 22 else "FAIL",
            "correct": correct, "total": 24}


def grade_autocheck(result: dict, autocheck: dict):
    """Оценить ответ по спецификации автопроверки; вернуть (ok, detail)."""
    kind = autocheck["type"]
    if kind == "all":
        for check in autocheck["checks"]:
            ok, detail = grade_autocheck(result, check)
            if not ok:
                return False, detail
        return True, "all"
    text = result.get("text") or ""
    if kind == "exact":
        return normalize(text) == normalize(str(autocheck["value"])), "exact"
    if kind == "compact":
        return _compact(text) == _compact(str(autocheck["value"])), "compact"
    if kind == "contains_all":
        haystack = normalize(text).lower()
        missing = [value for value in autocheck["values"]
                   if normalize(value).lower() not in haystack]
        return not missing, "missing " + ";".join(missing) if missing else "contains_all"
    if kind == "excludes_all":
        haystack = normalize(text).lower()
        present = [value for value in autocheck["values"]
                   if normalize(value).lower() in haystack]
        return not present, "forbidden " + ";".join(present) if present else "excludes_all"
    if kind == "numeric":
        return normalize(text).rstrip(".") == str(autocheck["value"]), "numeric"
    if kind == "numbers_all":
        found = set(_numbers(text))
        missing = [value for value in autocheck["values"] if value not in found]
        return not missing, "missing " + str(missing) if missing else "numbers_all"
    if kind == "json":
        try:
            return _same_json(json.loads(text), autocheck["value"]), "json"
        except ValueError:
            return False, "not json"
    if kind == "sentence_count":
        return _sentences(text) == autocheck["value"], "sentence_count"
    if kind == "bullet_count":
        return _bullets(text) == autocheck["value"], "bullet_count"
    if kind == "line_match":
        actual = [re.sub(r"\s+", " ", line.strip().rstrip(".")) for line in
                  text.splitlines() if line.strip()]
        return actual == [re.sub(r"\s+", " ", line) for line in autocheck["value"]], "line_match"
    if kind == "valid_tool_call":
        if not _valid_tool_call(result):
            return False, "invalid tool identity or JSON"
        function = result["tools"][0]["function"]
        if "name" in autocheck and function.get("name") != autocheck["name"]:
            return False, "wrong tool name"
        arguments = json.loads(function["arguments"])
        field = autocheck.get("required_text")
        if field is not None:
            value = arguments.get(field)
            if not isinstance(value, str) or not value.strip():
                return False, "missing text argument"
            if "contains" in autocheck and autocheck["contains"] not in value:
                return False, "wrong text argument"
        if function.get("name") in ("web_search", "read_file") and field:
            definitions = {t["function"]["name"]: t["function"]
                           for t in hermes_synth.load_base()["tools"]}
            schema = definitions[function["name"]]["parameters"]
            properties = schema["properties"]
            for required_name in schema.get("required", []):
                if required_name not in arguments:
                    return False, "missing required tool argument: " + required_name
            for key, value in arguments.items():
                if key not in properties:
                    return False, "unknown tool argument"
                constraint = properties[key]
                kind = constraint.get("type")
                if kind == "string" and not isinstance(value, str):
                    return False, "invalid string argument"
                if kind == "integer" and (type(value) is not int or
                        ("minimum" in constraint and value < constraint["minimum"]) or
                        ("maximum" in constraint and value > constraint["maximum"])):
                    return False, "invalid integer argument"
        return True, "valid_tool_call"
    raise ValueError("Неизвестная автопроверка: " + kind)


# ---------------------------------------------------------------------------
# Построение запросов (политика §3.4 дизайн-документа)
# ---------------------------------------------------------------------------

def _body(base_updates: dict, thinking: str, pass_name: str, seed: int) -> dict:
    body = {"model": ALIAS, "cache_prompt": False, "stream": False, **base_updates}
    if pass_name == "greedy":
        body.update({"temperature": 0, "seed": 0})
    elif pass_name == "sampled":
        body.update({"temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0.05,
                     "seed": seed})
    else:
        raise ValueError("pass_name должен быть greedy или sampled")
    if thinking == "off":
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    elif thinking == "hermes":
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = True
        body["reasoning_budget_tokens"] = 512
    else:
        raise ValueError("thinking должен быть off или hermes")
    return body


def build_requests(task: dict, pass_name: str, seed: int) -> list:
    """Список шагов задачи: [(identifier, request, kind)].

    kind: "main" — единственный ход; "turn1"/"turn2" — двухходовые
    tool-цепочки (ход 2 строится по фактическому ответу хода 1).
    """
    thinking = task["thinking"]
    limit = task.get("max_completion_tokens")
    if task["domain"] == "hermes-reuse":
        body = hermes_synth.build_request(
            next(case for case in hermes_synth.load_cases()
                 if case["id"] == task["hermes_case"]))
        body = {**body, "cache_prompt": False}
        if pass_name == "greedy":
            body.update({"temperature": 0, "seed": 0})
        else:
            body.update({"temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0.05,
                         "seed": seed})
        return [(task["id"], {"method": "POST", "path": "/v1/chat/completions",
                              "body": body}, "main")]
    if "tools" in task:
        first = _body({"messages": [{"role": "user", "content": task["first_prompt"]}],
                       "tools": task["tools"], "parallel_tool_calls": False,
                       "tool_choice": {"type": "function", "function": {
                           "name": task["tool_sequence"][0]["name"]}},
                       "max_completion_tokens": limit},
                      thinking, pass_name, seed)
        return [(task["id"] + "-turn1", {"method": "POST",
                                         "path": "/v1/chat/completions", "body": first},
                 "turn1")]
    if "messages" in task:
        updates = {"messages": task["messages"], "max_completion_tokens": limit}
        if "historical_tools" in task:
            updates["tools"] = task["historical_tools"]
            updates["tool_choice"] = "none"
        body = _body(updates, thinking, pass_name, seed)
        return [(task["id"], {"method": "POST", "path": "/v1/chat/completions",
                              "body": body}, "main")]
    content: list = []
    for name in task.get("images", []):
        if name in task.get("image_lines", {}):
            block = appset._text_canvas(task["image_lines"][name])
        else:
            block = appset.image(name)
        content.append(block)
    content.append({"type": "text", "text": task["prompt"]})
    updates = {"messages": [{"role": "user", "content": content}],
               "max_completion_tokens": limit}
    if "response_format" in task:
        updates["response_format"] = task["response_format"]
    body = _body(updates, thinking, pass_name, seed)
    return [(task["id"], {"method": "POST", "path": "/v1/chat/completions",
                          "body": body}, "main")]


def turn2_request(task: dict, turn1_result: dict, pass_name: str, seed: int) -> dict:
    """Второй ход: первый результат и требование вызвать confirm_record."""
    body = {"model": ALIAS, "cache_prompt": True, "stream": False,
            "tools": task["tools"], "parallel_tool_calls": False,
            "tool_choice": {"type": "function", "function": {
                "name": task["tool_sequence"][1]["name"]}},
            "max_completion_tokens": task.get("max_completion_tokens"),
            "messages": [{"role": "user", "content": task["first_prompt"]},
                         {"role": "assistant", "content": turn1_result.get("text") or None,
                          "tool_calls": turn1_result.get("tools")},
                         {"role": "tool", "tool_call_id": turn1_result["tools"][0]["id"],
                          "content": json.dumps(task["tool_result"], ensure_ascii=False)},
                         {"role": "user", "content": task["second_prompt"]}]}
    if task["thinking"] == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif task["thinking"] == "hermes":
        body["chat_template_kwargs"] = {"enable_thinking": True}
        body["reasoning_budget_tokens"] = 512
    if pass_name == "greedy":
        body.update({"temperature": 0, "seed": 0})
    else:
        body.update({"temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0.05,
                     "seed": seed})
    return {"method": "POST", "path": "/v1/chat/completions", "body": body}


def turn3_request(task: dict, previous: dict, turn2_result: dict) -> dict:
    """Третий ход после второго проверенного вызова инструмента."""
    body = dict(previous["body"])
    body["tool_choice"] = "none"
    body["messages"] = [*body["messages"],
                        {"role": "assistant", "content": turn2_result.get("text") or None,
                         "tool_calls": turn2_result["tools"]},
                        {"role": "tool", "tool_call_id": turn2_result["tools"][0]["id"],
                         "content": json.dumps(task["tool_sequence"][1]["result"], ensure_ascii=False)},
                        {"role": "user", "content": task["prompt"]}]
    return {"method": "POST", "path": "/v1/chat/completions", "body": body}


# ---------------------------------------------------------------------------
# Прогон
# ---------------------------------------------------------------------------

class SetRunner:
    """Каждый запрос — отдельный процесс transport с собственным дедлайном."""

    def __init__(self, base_url: str, output: Path, pass_name: str, seed: int,
                 timeout: int = 300):
        require(pass_name in ("greedy", "sampled"), "unsupported pass")
        self.base_url, self.pass_name, self.seed, self.timeout = base_url, pass_name, seed, timeout
        self.output = Path(output)
        require(not self.output.exists(), "output must be a new directory")
        (self.output / "inputs").mkdir(parents=True)
        (self.output / "requests").mkdir()
        self.report = {"status": "RUNNING", "pass_name": pass_name, "seed": seed,
                       "base_url": base_url, "results": [],
                       "qualification": "NOT_QUALIFIED",
                       "provenance_status": "UNVERIFIED",
                       "fixture_sha256": fixture_hashes()}
        self.save()

    def save(self):
        write_json(self.output / "report.json", self.report)

    def _transport(self, identifier: str, request: dict) -> dict:
        input_path = self.output / "inputs" / (identifier + ".json")
        with input_path.open("x", encoding="utf-8") as out:
            json.dump(request, out, ensure_ascii=False, indent=2)
        outdir = self.output / "requests" / identifier
        command = [sys.executable, str(Path(__file__).with_name("faex2_transport.py")),
                   "request", "--base-url", self.base_url, "--request", str(input_path),
                   "--output", str(outdir), "--timeout", str(self.timeout)]
        with (self.output / (identifier + ".stdout")).open("xb") as stdout, \
             (self.output / (identifier + ".stderr")).open("xb") as stderr:
            proc = subprocess.run(command, stdout=stdout, stderr=stderr,
                                  timeout=self.timeout + 30)
        require(proc.returncode == 0, "transport process failed: " + identifier)
        return json.loads((outdir / "result.json").read_text(encoding="utf-8"))

    def verify_server(self) -> dict:
        """Снять провенанс живого сервера (/health, /ready, /props, /v1/models)."""
        provenance = {}
        for key, path in (("health", "/health"), ("ready", "/ready"),
                          ("props", "/props"), ("models", "/v1/models")):
            result = self._transport("discovery-" + key,
                                     {"method": "GET", "path": path, "body": None})
            require(result.get("status") == "PASS" and result.get("http_status") == 200,
                    f"discovery {path} failed")
            provenance[key] = result.get("response")
        require(provenance["health"] == {"status": "ok"}, "bad health")
        require(provenance["ready"] == {"status": "ready", "model": ALIAS}, "bad ready")
        props = provenance["props"]
        require(isinstance(props, dict) and props.get("model_alias") == ALIAS,
                "unexpected props model_alias")
        models = provenance["models"]
        listed = [item.get("id") for item in (models or {}).get("data", [])
                  if isinstance(item, dict)]
        require(ALIAS in listed, "model alias missing in /v1/models")
        self.report["server_provenance"] = provenance
        self.report["provenance_status"] = "VERIFIED"
        self.save()
        return provenance

    def _verify_tool_turn(self, result: dict, task: dict, index: int = 0) -> None:
        require(result.get("finish") == "tool_calls" and len(result.get("tools") or []) == 1,
                task["id"] + ": missing/extra tool call")
        call = result["tools"][0]
        require(call.get("type") == "function" and
                isinstance(call.get("id"), str) and bool(call["id"]),
                task["id"] + ": invalid tool identity")
        expected = task["tool_sequence"][index]
        require(call["function"]["name"] == expected["name"], task["id"] + ": wrong tool")
        require(json.loads(call["function"]["arguments"]) == expected["arguments"],
                task["id"] + ": wrong tool arguments")

    def run_task(self, task: dict) -> dict:
        row = {"id": task["id"], "domain": task["domain"], "bucket": task["bucket"],
               "evaluation": task["evaluation"], "status": "RUNNING"}
        try:
            require(isinstance(task["id"], str) and
                    re.fullmatch(r"[a-z0-9][a-z0-9а-яё-]{0,95}", task["id"]) is not None,
                    "invalid task id")
            steps = build_requests(task, self.pass_name, self.seed)
            results = []
            for identifier, request, kind in steps:
                result = self._transport(identifier, request)
                require(result.get("status") == "PASS",
                        identifier + ": transport status " + str(result.get("status")))
                results.append(result)
                if kind == "turn1":
                    self._verify_tool_turn(result, task)
                    request = turn2_request(task, result, self.pass_name, self.seed)
                    steps.append((task["id"] + "-turn2", request, "turn2"))
                elif kind == "turn2":
                    self._verify_tool_turn(result, task, 1)
                    request = turn3_request(task, request, result)
                    steps.append((task["id"] + "-turn3", request, "turn3"))
            final = results[-1]
            row["finish"] = final.get("finish")
            row["text"] = final.get("text") or ""
            row["tools"] = final.get("tools") or []
            row["tool_trace"] = [{"tools": result.get("tools", []),
                                  "finish": result.get("finish")}
                                 for result in results if result.get("tools")]
            row["final_result"] = final
            row["usage"] = final.get("usage", {})
            row["metrics"] = final.get("metrics", {})
            if "target_range" in task:
                count = row["metrics"].get("prompt_tokens")
                low, high = task["target_range"]
                require(type(count) is int and low <= count <= high,
                        task["id"] + ": длина истории не подтверждена в целевом диапазоне")
            if task.get("autocheck", {}).get("type") != "valid_tool_call":
                require(final.get("finish") == "stop" and
                        bool((final.get("text") or "").strip()) and
                        not final.get("tools"),
                        task["id"] + ": пустой, обрезанный или неверный ответ")
            if task["domain"] == "long-facts":
                row["fact_checks"] = grade_long_facts(task, final.get("text", ""))
            if task["evaluation"] == "known_limit":
                ok, detail = grade_autocheck(final, task["autocheck"])
                row.update(status="PASS" if ok else "KNOWN_LIMIT_FAIL",
                           detail=detail if not ok else "unexpected_probe_pass")
                if ok:
                    self.report.setdefault("unexpected_probe_passes", []).append(task["id"])
            elif task["evaluation"] == "auto":
                ok, detail = grade_autocheck(final, task["autocheck"])
                row.update(status="PASS" if ok else "FAIL", detail=detail)
            else:  # manual — оценивается слепой рубрикой
                text = (final.get("text") or "").strip()
                require(text and final.get("finish") == "stop",
                        task["id"] + ": empty or incomplete manual answer")
                row.update(status="NEEDS_REVIEW", detail="manual rubric")
        except Exception as error:
            row.update(status="FAIL", error=type(error).__name__ + ": " + str(error))
        self.report["results"].append(row)
        if row["status"] == "FAIL":
            self.report["status"] = "FAIL"
        self.save()
        return row

    def run(self, tasks: list, only: str | None = None) -> None:
        selected = [task for task in tasks
                    if only is None or only in task["id"] or only in task["domain"]]
        require(selected, "no tasks selected")
        for task in selected:
            self.run_task(task)
        self.report["long_facts"] = summarize_long_facts(self.report["results"], tasks,
                                                        self.pass_name)
        statuses = {row["status"] for row in self.report["results"]}
        if self.report["status"] != "FAIL":
            self.report["status"] = ("KNOWN_FAILURES" if "KNOWN_LIMIT_FAIL" in statuses else
                                     "NEEDS_REVIEW" if "NEEDS_REVIEW" in statuses
                                     else "RECORDED_NOT_QUALIFIED")
        self.save()


# ---------------------------------------------------------------------------
# Слепые пары и раскрытие
# ---------------------------------------------------------------------------

def _load_quality_rubric() -> tuple:
    quality = Path(__file__).with_name(RUBRIC_FILE)
    first = json.loads(quality.read_text(encoding="utf-8").splitlines()[0])
    return first["rubric"], first["pass_criteria"]


def _task_result(report: dict, identifier: str) -> dict:
    for row in report["results"]:
        if row["id"] == identifier:
            return row
    raise ValueError("В прогоне нет задачи " + identifier)


def export_blind(run_a: Path, run_b: Path, out: Path, seed: int) -> None:
    """Каталог рецензента и отдельный приватный ключ рядом с ним."""
    require(bool(out.name) and out.name not in (".", ".."),
            "blind output must name a directory")
    key_path = out.parent / (out.name + ".key.json")
    require(not out.exists(), "blind output must be a new directory")
    require(not key_path.exists() and not key_path.is_symlink(), "blind key already exists")
    report_a = json.loads((run_a / "report.json").read_text(encoding="utf-8"))
    report_b = json.loads((run_b / "report.json").read_text(encoding="utf-8"))
    require(report_a.get("pass_name") in ("greedy", "sampled") and
            report_a.get("pass_name") == report_b.get("pass_name") and
            report_a.get("seed") == report_b.get("seed"), "Несовместимые прогоны")
    expected_hashes = fixture_hashes()
    require(report_a.get("fixture_sha256") == report_b.get("fixture_sha256") == expected_hashes,
            "Разные или отсутствующие хеши набора")
    tasks = appset.all_tasks()
    pairs_raw = []
    for task in tasks:
        result_a, result_b = _task_result(report_a, task["id"]), _task_result(report_b, task["id"])
        require(not result_a.get("error") and not result_b.get("error"),
                task["id"] + ": транспорт или условия сценария не выполнены")
        pairs_raw.append((task, result_a, result_b))
    rubric, pass_criteria = _load_quality_rubric()
    images = appset.image_blocks()
    image_paths = {}
    out.mkdir(parents=True)
    (out / "images").mkdir()
    for name, block in images.items():
        url = block["image_url"]["url"]
        require(url.startswith("data:image/png;base64,"), "Неизвестный формат изображения")
        content = base64.b64decode(url.split(",", 1)[1], validate=True)
        digest = hashlib.sha256(content).hexdigest()
        relative = "images/" + digest + ".png"
        path = out / relative
        if not path.exists():
            path.write_bytes(content)
        image_paths[name] = relative
    rows, key = [], {}
    counters: dict = {}
    for task, result_a, result_b in pairs_raw:
        counters[task["domain"]] = counters.get(task["domain"], 0) + 1
        pair_id = f"{task['domain']}/{counters[task['domain']]:03d}"
        pair = {"pair_id": pair_id, "task_class": task["domain"] + "/" + task["bucket"],
                "evaluation": task["evaluation"],
                "response_a": {"text": result_a.get("text") or "",
                               "finish": result_a.get("finish"),
                               "tools": result_a.get("tools"),
                               "tool_trace": result_a.get("tool_trace", [])},
                "response_b": {"text": result_b.get("text") or "",
                               "finish": result_b.get("finish"),
                               "tools": result_b.get("tools"),
                               "tool_trace": result_b.get("tool_trace", [])}}
        auto = {}
        if task["evaluation"] == "auto":
            autocheck = task.get("autocheck") or {"type": "valid_tool_call"}
            auto = {"run_a": result_a.get("status") == "PASS" and
                    grade_autocheck(result_a, autocheck)[0],
                    "run_b": result_b.get("status") == "PASS" and
                    grade_autocheck(result_b, autocheck)[0]}
        pair["prompt"] = _task_prompt(task)
        pair["images"] = [image_paths[name] for name in task.get("images", [])]
        pair["messages"] = _review_messages(task, image_paths)
        pair["expected_facts"] = task.get("expected_facts", [])
        pair["prohibited_errors"] = task.get("prohibited_errors", [])
        pair["expected_values"] = task.get("expected_values", {})
        pair["rubric"] = task.get("rubric", rubric)
        pair["pass_criteria"] = task.get("pass_criteria", pass_criteria)
        pair["critical_failures"] = task.get("critical_failures", [])
        swap = random.Random(f"{seed}:{task['id']}").random() < 0.5
        if swap:
            pair["response_a"], pair["response_b"] = pair["response_b"], pair["response_a"]
        key[pair_id] = {"task": task["id"], "a_is": "run_b" if swap else "run_a",
                        "auto": auto}
        rows.append(pair)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(key_path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump({"seed": seed, "pairs": key, "run_a": str(run_a),
                   "run_b": str(run_b)}, output, ensure_ascii=False, indent=2)
    (out / "blind_pairs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    write_json(out / "manifest.json", {
        "seed": seed, "pairs": len(rows), "pass_name": report_a["pass_name"],
        "fixture_sha256": expected_hashes})


def _task_prompt(task: dict) -> str:
    if "hermes_case" in task:
        case = next(case for case in hermes_synth.load_cases()
                    if case["id"] == task["hermes_case"])
        return "\n".join(str(message.get("content") or "") for message in case["messages"])
    if "first_prompt" in task:
        return task["first_prompt"] + "\n" + task["prompt"]
    if "prompt" in task:
        return task["prompt"]
    parts = []
    for message in task.get("messages", []):
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts += [item.get("text", "") for item in content if item.get("type") == "text"]
    return "\n".join(parts)


def _review_messages(task: dict, image_paths: dict) -> list:
    """История для рецензента; ссылки на PNG не раскрывают вариант."""
    if "hermes_case" in task:
        case = next(case for case in hermes_synth.load_cases()
                    if case["id"] == task["hermes_case"])
        messages = copy.deepcopy(hermes_synth.build_request(case)["messages"])
    elif "messages" in task:
        messages = copy.deepcopy(task["messages"])
    elif "first_prompt" in task:
        messages = [{"role": "user", "content": task["first_prompt"]},
                    {"role": "tool", "content": json.dumps(task["tool_result"],
                                                           ensure_ascii=False)},
                    {"role": "user", "content": task["second_prompt"]},
                    {"role": "tool", "content": json.dumps(task["tool_sequence"][1]["result"],
                                                           ensure_ascii=False)},
                    {"role": "user", "content": task["prompt"]}]
    else:
        messages = [{"role": "user", "content": task["prompt"]}]
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue
            url = part["image_url"]["url"]
            require(url.startswith("data:image/png;base64,"), "Неизвестный формат изображения")
            raw = base64.b64decode(url.split(",", 1)[1], validate=True)
            part["image_url"]["url"] = "images/" + hashlib.sha256(raw).hexdigest() + ".png"
    return messages


def unblind(blind_dir: Path, reviews_path: Path | None) -> dict:
    """Соединить отзывы с blinding_key и вывести парные исходы."""
    key_path = blind_dir.parent / (blind_dir.name + ".key.json")
    key = json.loads(key_path.read_text(encoding="utf-8"))["pairs"]
    reviews = {}
    manifest = json.loads((blind_dir / "manifest.json").read_text(encoding="utf-8"))
    sampled = manifest["pass_name"] == "sampled"
    if reviews_path is not None:
        for line in Path(reviews_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                review = json.loads(line)
                reviews[review["pair_id"]] = review
    _, pass_criteria = _load_quality_rubric()
    outcomes = {"both_ok": 0, "only_a": 0, "only_b": 0, "both_fail": 0}
    per_variant = {"a": {"ok": 0, "wrong": 0, "not_graded": 0}, "b": {"ok": 0, "wrong": 0, "not_graded": 0}}
    by_domain: dict = {}
    critical_counts = {"a": 0, "b": 0}
    pairs = {json.loads(line)["pair_id"]: json.loads(line) for line in
             (blind_dir / "blind_pairs.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()}
    for pair_id, entry in key.items():
        pair = pairs[pair_id]
        domain = pair["task_class"].split("/")[0]
        graded = {"a": False, "b": False}
        ok = {"a": None, "b": None}
        for side in ("a", "b"):
            response = pair["response_" + side]
            if pair["evaluation"] == "known_limit":
                continue
            if pair["evaluation"] == "auto" and not sampled:
                original = entry["a_is"] if side == "a" else (
                    "run_b" if entry["a_is"] == "run_a" else "run_a")
                ok[side], graded[side] = entry["auto"][original], True
            else:
                review = reviews.get(pair_id)
                require(review is not None, "Нет отзыва для пары " + pair_id)
                assessment = review[side]
                scores, critical = assessment["scores"], assessment["critical_failures"]
                require(set(scores) == {"correctness", "instruction", "russian_clarity",
                                        "explanation"} and
                        all(type(value) is int and 0 <= value <= 2
                            for value in scores.values()) and
                        isinstance(critical, list) and
                        isinstance(assessment.get("rationale"), str) and
                        bool(assessment["rationale"].strip()), "Некорректная оценка")
                ok[side] = (not critical and scores.get("correctness") == 2
                            and min(scores.values()) >= 1 and sum(scores.values()) >= 6)
                original_side = side if entry["a_is"] == "run_a" else (
                    "b" if side == "a" else "a")
                critical_counts[original_side] += bool(critical)
                graded[side] = True
        if entry["a_is"] == "run_b":
            ok["a"], ok["b"] = ok["b"], ok["a"]
            graded["a"], graded["b"] = graded["b"], graded["a"]
        else:
            require(entry["a_is"] == "run_a", "Некорректный ключ")
        if pair["evaluation"] == "known_limit":
            per_variant["a"]["not_graded"] += 1
            per_variant["b"]["not_graded"] += 1
            continue
        for side in ("a", "b"):
            if graded[side]:
                per_variant[side]["ok" if ok[side] else "wrong"] += 1
        if ok["a"] and ok["b"]:
            outcomes["both_ok"] += 1
        elif ok["a"] and not ok["b"]:
            outcomes["only_b"] += 1
        elif ok["b"] and not ok["a"]:
            outcomes["only_a"] += 1
        else:
            outcomes["both_fail"] += 1
        domain_row = by_domain.setdefault(domain, {"both_ok": 0, "only_a": 0,
                                                   "only_b": 0, "both_fail": 0})
        domain_row["both_ok" if (ok["a"] and ok["b"])
                   else "only_b" if (ok["a"] and not ok["b"])
                   else "only_a" if (ok["b"] and not ok["a"]) else "both_fail"] += 1
    total = sum(outcomes.values())
    a_wrong, b_wrong = per_variant["a"]["wrong"], per_variant["b"]["wrong"]
    summary = {"outcomes": outcomes, "total_graded_pairs": total,
               "per_variant": per_variant, "critical_counts": critical_counts,
               "by_domain": by_domain,
               "paired_difference": {"a_minus_b": (per_variant["a"]["wrong"] - b_wrong) / max(1, total),
                                     "b_minus_a": (b_wrong - per_variant["a"]["wrong"]) / max(1, total)},
               "critical_note": "Критические отказы оценены только по переданным отзывам."}
    write_json(blind_dir / "unblinded.json", summary)
    return summary


def _task_text_for_tokenize(task: dict) -> str:
    """Собрать текстовое содержимое задачи для грубой калибровки через POST /tokenize."""
    if "first_prompt" in task:
        return task["first_prompt"] + "\n" + task.get("second_prompt", "") + "\n" + task.get("prompt", "")
    if "prompt" in task:
        return task["prompt"]
    parts = []
    for message in task.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts += [item.get("text", "") for item in content if item.get("type") == "text"]
    return "\n".join(parts)


def calibrate_lengths(base_url: str, tasks: list, output: Path,
                      timeout: int = 60, verify: bool = True) -> dict:
    """Проверить фактическую длину текстов задач с target_range через POST /tokenize."""
    runner = SetRunner(base_url, output, "greedy", 0, timeout=timeout)
    if verify:
        runner.verify_server()
    records = []
    all_ok = True
    for task in tasks:
        if "target_range" not in task:
            continue
        text = _task_text_for_tokenize(task)
        res = runner._transport(
            "tokenize-" + task["id"],
            {"method": "POST", "path": "/tokenize",
             "body": {"content": text, "add_special": False}},
        )
        require(res.get("status") == "PASS" and res.get("http_status") == 200,
                task["id"] + ": /tokenize failed")
        tokens = (res.get("response") or {}).get("tokens")
        require(isinstance(tokens, list), task["id"] + ": invalid /tokenize payload")
        count = len(tokens)
        low, high = task["target_range"]
        # Текстовая оценка не включает chat framing / image tokens; на прогоне действует строгий prompt_tokens.
        in_range = low - 512 <= count <= high
        above_sparse = task["bucket"] != "long" or count > appset.SPARSE_THRESHOLD
        ok = in_range and above_sparse
        if not ok:
            all_ok = False
        records.append({
            "id": task["id"],
            "domain": task["domain"],
            "bucket": task["bucket"],
            "text_tokens": count,
            "target_range": [low, high],
            "above_sparse_threshold": above_sparse,
            "ok": ok,
        })
    summary = {
        "status": "CALIBRATED" if all_ok else "OUT_OF_RANGE",
        "provenance_status": runner.report["provenance_status"],
        "sparse_threshold": appset.SPARSE_THRESHOLD,
        "tasks": records,
    }
    write_json(output / "calibration.json", summary)
    return summary


def _check_domain_non_regression(by_domain: dict) -> list:
    regressed = []
    for domain, counts in sorted(by_domain.items()):
        if counts.get("only_b", 0) > counts.get("only_a", 0):
            regressed.append({"domain": domain, **counts})
    return regressed


def build_appset_report(greedy_a: Path, greedy_b: Path, output: Path,
                        greedy_unblind: Path | None = None,
                        sampled_unblind: Path | None = None) -> dict:
    """Свести барьеры прикладного набора (без подмены численного среза Q1)."""
    report_a = json.loads((greedy_a / "report.json").read_text(encoding="utf-8"))
    report_b = json.loads((greedy_b / "report.json").read_text(encoding="utf-8"))
    require(report_a.get("pass_name") == "greedy" and report_b.get("pass_name") == "greedy",
            "greedy reports required")
    require(report_a.get("fixture_sha256") == report_b.get("fixture_sha256") == fixture_hashes(),
            "fixture_sha256 mismatch")

    tasks = appset.all_tasks()
    prov_ok = (report_a.get("provenance_status") == "VERIFIED" and
               report_b.get("provenance_status") == "VERIFIED")

    # 1. Детерминированные auto-задачи (66) + transport/critical по всем 94 оценочным.
    auto_tasks = [t for t in tasks if t["evaluation"] == "auto"]
    auto_outcomes = {"both_ok": 0, "only_a": 0, "only_b": 0, "both_fail": 0}
    auto_by_domain: dict = {}
    greedy_critical_b = 0
    for task in tasks:
        if task["evaluation"] == "known_limit":
            continue
        ra = _task_result(report_a, task["id"])
        rb = _task_result(report_b, task["id"])
        if rb.get("status") == "FAIL" and rb.get("error"):
            greedy_critical_b += 1
        if task["evaluation"] != "auto":
            continue
        ok_a = ra.get("status") == "PASS"
        ok_b = rb.get("status") == "PASS"
        key = ("both_ok" if (ok_a and ok_b) else "only_b" if (ok_a and not ok_b)
               else "only_a" if (ok_b and not ok_a) else "both_fail")
        auto_outcomes[key] += 1
        drow = auto_by_domain.setdefault(task["domain"], {"both_ok": 0, "only_a": 0,
                                                          "only_b": 0, "both_fail": 0})
        drow[key] += 1

    # 2. long-facts 22/24 с разбивкой по позициям.
    long_facts_b = summarize_long_facts(report_b.get("results", []), tasks, "greedy")
    by_position = {"begin": {"correct": 0, "total": 0},
                   "mid": {"correct": 0, "total": 0},
                   "end": {"correct": 0, "total": 0}}
    for task in tasks:
        if task["domain"] != "long-facts":
            continue
        pos = task.get("fact_position")
        if pos not in by_position:
            continue
        by_position[pos]["total"] += 2
        rb = next((r for r in report_b.get("results", []) if r.get("id") == task["id"]), None)
        if rb and isinstance(rb.get("fact_checks"), list):
            by_position[pos]["correct"] += sum(bool(x) for x in rb["fact_checks"])

    # 3. known_limit (big-text) — строго отдельно.
    known_limit_rows = []
    for task in tasks:
        if task["evaluation"] != "known_limit":
            continue
        ra = _task_result(report_a, task["id"])
        rb = _task_result(report_b, task["id"])
        known_limit_rows.append({"id": task["id"], "a_status": ra.get("status"),
                                 "b_status": rb.get("status")})

    # 4. Ручные / слепые итоги (опционально).
    greedy_blind_summary = (json.loads((greedy_unblind / "unblinded.json").read_text(encoding="utf-8"))
                            if greedy_unblind else None)
    sampled_summary = (json.loads((sampled_unblind / "unblinded.json").read_text(encoding="utf-8"))
                       if sampled_unblind else None)

    greedy_auto_ok = (auto_outcomes["only_b"] <= 1 and greedy_critical_b == 0 and
                      not _check_domain_non_regression(auto_by_domain))
    long_facts_ok = long_facts_b.get("status") == "PASS"

    sampled_ok = None
    if sampled_summary is not None:
        diff = sampled_summary["paired_difference"]["b_minus_a"]
        crit_b = sampled_summary["critical_counts"]["b"]
        sampled_reg = _check_domain_non_regression(sampled_summary["by_domain"])
        sampled_ok = (diff <= 0.05 and crit_b == 0 and not sampled_reg)

    greedy_blind_ok = None
    if greedy_blind_summary is not None:
        crit_b_g = greedy_blind_summary["critical_counts"]["b"]
        greedy_blind_reg = _check_domain_non_regression(greedy_blind_summary["by_domain"])
        greedy_blind_ok = (crit_b_g == 0 and not greedy_blind_reg)

    if not prov_ok:
        overall = "INCOMPLETE_PROVENANCE"
    elif not (greedy_auto_ok and long_facts_ok) or (sampled_ok is False) or (greedy_blind_ok is False):
        overall = "FAIL"
    elif sampled_summary is None:
        overall = "NOT_EXERCISED"
    else:
        overall = "DESCRIPTIVE_TASK_GATE_PASS"

    summary = {
        "status": overall,
        "q1_qualification": "REQUIRES_NUMERICAL_SLICE",
        "provenance": {
            "a": report_a.get("provenance_status", "UNVERIFIED"),
            "b": report_b.get("provenance_status", "UNVERIFIED"),
            "verified": prov_ok,
        },
        "greedy_auto": {
            "total": len(auto_tasks),
            "outcomes": auto_outcomes,
            "by_domain": auto_by_domain,
            "domain_regressions": _check_domain_non_regression(auto_by_domain),
            "critical_transport_failures_b": greedy_critical_b,
            "pass": greedy_auto_ok,
        },
        "long_facts": {
            **long_facts_b,
            "by_position": by_position,
        },
        "known_limit_isolated": known_limit_rows,
        "greedy_blind": greedy_blind_summary,
        "sampled_blind": sampled_summary,
        "sampled_gate_pass": sampled_ok,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--base-url", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--pass-name", choices=("greedy", "sampled"), default="greedy")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--timeout", type=int, default=300)
    run.add_argument("--only", help="подстрока id или домена для частичного прогона")
    run.add_argument("--skip-server-check", action="store_true",
                     help="пропустить снятие /health,/ready,/props,/v1/models (оставляет UNVERIFIED)")
    cal = sub.add_parser("calibrate")
    cal.add_argument("--base-url", required=True)
    cal.add_argument("--output", type=Path, required=True)
    cal.add_argument("--timeout", type=int, default=60)
    cal.add_argument("--skip-server-check", action="store_true")
    blind = sub.add_parser("export-blind")
    blind.add_argument("--run-a", type=Path, required=True)
    blind.add_argument("--run-b", type=Path, required=True)
    blind.add_argument("--out", type=Path, required=True)
    blind.add_argument("--seed", type=int, default=BLIND_SEED_DEFAULT)
    unblind_parser = sub.add_parser("unblind")
    unblind_parser.add_argument("--blind", type=Path, required=True)
    unblind_parser.add_argument("--reviews", type=Path)
    rep = sub.add_parser("report")
    rep.add_argument("--greedy-a", type=Path, required=True)
    rep.add_argument("--greedy-b", type=Path, required=True)
    rep.add_argument("--greedy-unblind", type=Path)
    rep.add_argument("--sampled-unblind", type=Path)
    rep.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        runner = SetRunner(args.base_url, args.output, args.pass_name, args.seed,
                           args.timeout)
        if not args.skip_server_check:
            runner.verify_server()
        runner.run(appset.all_tasks(), args.only)
        print("report:", runner.output / "report.json", "status:", runner.report["status"])
    elif args.command == "calibrate":
        summary = calibrate_lengths(args.base_url, appset.all_tasks(), args.output,
                                    args.timeout, verify=not args.skip_server_check)
        print("calibration:", args.output / "calibration.json", "status:", summary["status"])
    elif args.command == "export-blind":
        export_blind(args.run_a, args.run_b, args.out, args.seed)
        print("blind pairs:", args.out / "blind_pairs.jsonl")
    elif args.command == "unblind":
        print(json.dumps(unblind(args.blind, args.reviews), ensure_ascii=False, indent=1))
    else:
        summary = build_appset_report(args.greedy_a, args.greedy_b, args.output,
                                      args.greedy_unblind, args.sampled_unblind)
        print("appset report:", args.output, "status:", summary["status"])


if __name__ == "__main__":
    main()
