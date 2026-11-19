"""Проверки прикладного набора FAEX2 (Б2) без модели, сети и сторонних библиотек.

Покрывают: реестр (84 + 12 = 96, суммы дизайн-документа §3.1), детерминизм
генераторов и экспорта, длины домена D, изображения, автопроверки, политику
thinking/cache_prompt и слепой экспорт/раскрытие на сфабрикованных прогонах.
"""

from __future__ import annotations

import json
import base64
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import faex2_appset as appset
import faex2_appset_runner as runner
from test_faex2_fixtures import decode_png


def _counts(tasks, key):
    result = {}
    for task in tasks:
        result[task[key]] = result.get(task[key], 0) + 1
    return result


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.new = appset.new_tasks()
        self.all_tasks = appset.all_tasks()
        self.by_id = {task["id"]: task for task in self.all_tasks}

    def test_counts_match_the_design_document(self):
        self.assertEqual(len(self.new), 84)
        self.assertEqual(len(self.all_tasks), 96)
        domains = _counts(self.new, "domain")
        self.assertEqual(domains["long-facts"], 12)
        self.assertEqual(domains["long-retell"], 8)
        self.assertEqual(domains["long-tool-dialog"], 4)
        self.assertEqual(domains["long-vision-dialog"], 4)
        self.assertEqual(sum(domains[name] for name in
                             ("ocr-text", "ocr-table", "ocr-receipt", "ocr-formula",
                              "vision-multi", "big-text")), 24)
        self.assertEqual(sum(domains[name] for name in
                             ("code-fix", "code-explain", "tool-chain", "json-schema",
                              "hermes-reuse")), 20)
        buckets = _counts(self.all_tasks, "bucket")
        self.assertEqual(buckets, {"short": 60, "medium": 8, "long": 28})
        evaluations = _counts(self.all_tasks, "evaluation")
        self.assertEqual(evaluations, {"auto": 66, "manual": 28, "known_limit": 2})

    def test_ids_are_unique_and_safe(self):
        ids = [task["id"] for task in self.all_tasks]
        self.assertEqual(len(set(ids)), len(ids))
        for task in self.all_tasks:
            self.assertRegex(task["id"], r"^[a-z0-9][a-z0-9а-яё-]*$", task["id"])
            self.assertIn(task["thinking"], ("off", "hermes"))
            self.assertTrue(set(appset.CRITICAL) <= set(task["critical_failures"]))

    def test_every_task_supplies_grounding_to_the_reviewer(self):
        for task in self.all_tasks:
            with self.subTest(task=task["id"]):
                self.assertTrue(task.get("expected_facts"))
                self.assertTrue(task.get("prohibited_errors"))
                self.assertIn("rubric", task)
                self.assertIn("pass_criteria", task)

    def test_subgroup_coverage_cannot_silently_shrink(self):
        new_ids = {task["id"] for task in self.new}
        for expected in ("form-three-sentences", "form-four-bullets", "spelling-park",
                         "spelling-forest", "physics-train", "physics-bike",
                         "dialog-corr-notebooks", "dialog-corr-pens",
                         "retell-garden", "retell-trip",
                         "source-absent-library", "source-absent-captain",
                         "tool-chain-lookup-астрономия", "tool-chain-weather-сочи",
                         "tool-chain-lesson-monday", "json-schema-tula",
                         "hermes-reuse-tool-01", "hermes-reuse-mixed-02",
                         "big-text-gufo-518", "big-text-owl-274",
                         "vision-multi-1", "vision-multi-2",
                         "long-vision-dialog-1", "long-vision-dialog-2"):
            self.assertIn(expected, new_ids)

    def test_exported_registry_file_matches_the_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "appset.jsonl"
            appset.export(path)
            committed = Path(__file__).with_name("faex2_appset.jsonl").read_bytes()
            self.assertEqual(path.read_bytes(), committed)
            rows = [json.loads(line) for line in committed.decode("utf-8").splitlines()]
            self.assertEqual(len(rows), 84)
            self.assertEqual([row["id"] for row in rows],
                             [task["id"] for task in self.new])

    def test_long_tasks_estimates_and_sparse_threshold(self):
        def prompt_of(task):
            if "prompt" in task:
                return task["prompt"]
            parts = []
            for message in task.get("messages", []):
                content = message["content"]
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    parts += [item.get("text", "") for item in content
                              if item.get("type") == "text"]
            return "\n".join(parts)

        estimates = {}
        for task in self.all_tasks:
            if task["bucket"] != "long":
                continue
            estimate = len(prompt_of(task)) / appset.CHARS_PER_TOKEN
            self.assertGreater(estimate, appset.SPARSE_THRESHOLD, task["id"])
            estimates[task["id"]] = estimate
        self.assertEqual(len(estimates), 28)
        self.assertEqual(sum(value > 40_000 for value in estimates.values()), 14)
        self.assertEqual(sum(value < 40_000 for value in estimates.values()), 14)

    def test_long_tasks_declare_two_actual_token_buckets(self):
        tasks = [t for t in self.new if t["bucket"] == "long"]
        self.assertEqual(sum(t["target_range"] == [20000, 32000] for t in tasks), 14)
        self.assertEqual(sum(t["target_range"] == [44000, 64000] for t in tasks), 14)

    def test_wrong_actual_long_token_bucket_cannot_pass(self):
        task = next(t for t in appset.new_tasks()
                    if t["domain"] == "long-facts" and t["id"] == "long-facts-1-begin")
        answer = json.dumps(task["autocheck"]["value"])
        with tempfile.TemporaryDirectory() as tmp:
            instance = runner.SetRunner("http://127.0.0.1:1", Path(tmp) / "run", "greedy", 0)
            response = {"status": "PASS", "text": answer, "finish": "stop",
                        "metrics": {"prompt_tokens": 45000}}
            with patch.object(instance, "_transport", return_value=response):
                row = instance.run_task(task)
        self.assertNotEqual(row["status"], "PASS")

    def test_long_documents_are_deterministic_and_facts_match(self):
        first, facts = appset.long_document(1, 24_000)
        again, facts_again = appset.long_document(1, 24_000)
        self.assertEqual(first, again)
        self.assertEqual(facts, facts_again)
        day = appset._fact_day(1, "mid", 24_000)
        self.assertIn(day, facts)
        prompt, values = appset.fact_question(1, "mid", 24_000)
        self.assertEqual(values["temp"], facts[values["day"]]["temp"])
        self.assertIn(str(values["day"]), prompt)

    def test_long_facts_cover_24_independent_values(self):
        cases = [t for t in self.new if t["domain"] == "long-facts"]
        checks = [(t["id"], day, value)
                  for t in cases
                  for day, value in t["expected_values"]["fact_checks"]]
        self.assertEqual(len(checks), 24)
        for task_id, day, value in checks:
            task = next(t for t in cases if t["id"] == task_id)
            self.assertIn(f"День {day}: температура {value},", task["prompt"])

    def test_long_vision_answers_are_backed_by_actual_document(self):
        seen_signatures = set()
        for task in self.new:
            if task["domain"] != "long-vision-dialog":
                continue
            content = task["messages"][0]["content"]
            self.assertEqual([item["type"] for item in content],
                             ["text", "image_url", "text"], task["id"])
            self.assertTrue(content[0]["text"].strip() and content[2]["text"].strip(),
                            task["id"])
            text = content[0]["text"] + content[2]["text"]
            expected = task["expected_values"]
            day = expected["day"]
            seen_signatures.add((tuple(task["images"]), task["messages"][-1]["content"],
                                 content[0]["text"][:64]))
            with self.subTest(task=task["id"]):
                temp_key = f"temp_day_{day}"
                if temp_key in expected:
                    self.assertIn(f"День {day}: температура {expected[temp_key]},", text)
                else:
                    precip_key = f"precip_day_{day}"
                    line = next(line for line in text.split("День ")
                                if line.startswith(f"{day}:"))
                    self.assertIn("осадки — " + expected[precip_key] + ".", line)
        self.assertEqual(len(seen_signatures), 4)

    def test_long_tool_answer_is_contained_in_history(self):
        seen_needles = set()
        for task in self.new:
            if task["domain"] != "long-tool-dialog":
                continue
            history = "\n".join(m.get("content") or "" for m in task["messages"][:-1])
            seen_needles.add(task["expected_values"]["needle"])
            estimate = len(history) / appset.CHARS_PER_TOKEN
            low, high = task["target_range"]
            self.assertGreater(estimate, low + 2000, task["id"])
            self.assertLess(estimate, high - 2000, task["id"])
            with self.subTest(task=task["id"]):
                for value in task["expected_values"]["needle"].split():
                    self.assertIn(value, history)
        self.assertEqual(len(seen_needles), 4)

    def test_goldens_are_independently_verified(self):
        for name, (lines, total) in appset._RECEIPTS.items():
            with self.subTest(receipt=name):
                item_sum = sum(int(line.rsplit(" ", 1)[1]) for line in lines[1:])
                self.assertEqual(item_sum, total)
        for name, (expr, x_val) in appset._FORMULAS.items():
            with self.subTest(formula=name):
                rhs = expr.split("=", 1)[1].strip()
                self.assertEqual(eval(rhs, {"__builtins__": {}}, {}), x_val)
        self.assertEqual(appset._TABLES["table-a"][1][2], 11)
        self.assertEqual(sum(v for row in appset._TABLES["table-b"] for v in row), 225)
        self.assertEqual(appset._TABLES["table-c"][2][0], 49)
        self.assertEqual(sum(v > 10 for row in appset._TABLES["table-d"] for v in row), 5)
        self.assertEqual(sum(v for row in appset._TABLES["table-e"] for v in row), 2100)
        self.assertEqual(appset._TABLES["table-f"][0][1], 8)
        self.assertEqual(self.by_id["dialog-corr-notebooks"]["expected_values"],
                         {"initial": 3 * 60 + 50, "corrected": 5 * 60 + 50})
        self.assertEqual(self.by_id["dialog-corr-pens"]["expected_values"],
                         {"initial": 4 * 25 + 20, "corrected": 6 * 25 + 20})
        self.assertEqual(self.by_id["physics-train"]["expected_values"], {"speed_km_h": 240 // 4})
        self.assertEqual(self.by_id["physics-bike"]["expected_values"], {"speed_km_h": 36 // 2})
        # Независимая проверка Python-задач code-fix и code-explain.
        self.assertEqual(sum(range(1, 6)), self.by_id["code-fix-sum-range"]["expected_values"]["answer"])
        self.assertEqual(str(sorted([3, 1, 2])), self.by_id["code-fix-sort-order"]["expected_values"]["answer"])
        self.assertEqual(sum(x > 10 for x in [5, 12, 8, 20]),
                         self.by_id["code-fix-loop-bound"]["expected_values"]["answer"])
        self.assertEqual([7, 4, 9][0], self.by_id["code-fix-index"]["expected_values"]["answer"])
        self.assertEqual(2 * 3 * 4, self.by_id["code-fix-product"]["expected_values"]["answer"])
        self.assertEqual("АБВ"[:], self.by_id["code-fix-slice"]["expected_values"]["answer"])
        self.assertEqual([1, 5, 9, 3][-2], self.by_id["code-explain-last"]["expected_values"]["answer"])
        self.assertEqual(len(set([1, 2, 2, 3])), self.by_id["code-explain-set"]["expected_values"]["answer"])
        self.assertEqual(sum([2, 4, 6]) / len([2, 4, 6]),
                         self.by_id["code-explain-words-1"]["expected_values"]["answer"])
        self.assertEqual(max([5, 12, 3]), self.by_id["code-explain-words-2"]["expected_values"]["answer"])

    def test_long_tool_dialog_contains_two_valid_tool_results(self):
        for task in self.new:
            if task["domain"] != "long-tool-dialog":
                continue
            messages = task["messages"]
            tool_messages = [m for m in messages if m["role"] == "tool"]
            calls = [call for m in messages if m["role"] == "assistant"
                     for call in m.get("tool_calls", [])]
            with self.subTest(task=task["id"]):
                self.assertEqual(len(tool_messages), 2)
                self.assertEqual({m["tool_call_id"] for m in tool_messages},
                                 {c["id"] for c in calls})
                self.assertEqual(len(task["historical_tools"]), 1)

    def test_table_count_golden_matches_the_rendered_table(self):
        task = next(t for t in self.new if t["id"] == "ocr-table-d")
        actual = sum(value > 10 for row in appset._TABLES["table-d"] for value in row)
        self.assertEqual(task["expected_values"]["answer"], actual)

    def test_numeric_answer_must_not_accept_wrong_extra_values(self):
        ok, _ = runner.grade_autocheck({"text": "Ответ 11 или 12"},
                                       {"type": "numeric", "value": 11})
        self.assertFalse(ok)

    def test_every_referenced_image_exists_and_decodes(self):
        images = appset.image_blocks()
        for task in self.all_tasks:
            for name in task.get("images", []):
                self.assertIn(name, images, task["id"])
        for name, block in images.items():
            width, height, pixels = decode_png(block)
            self.assertGreater(width * height, 0, name)
            self.assertLessEqual(len({pixels[i:i + 3] for i in range(0, len(pixels), 9)}),
                                 4, name)

    def test_big_text_letters_are_in_the_known_limit_band(self):
        for name, scale in (("big-gufo-518", 12), ("big-owl-274", 10)):
            width, height, pixels = decode_png(appset.image(name))
            dark_rows = [y for y in range(height)
                         if any(pixels[(y * width + x) * 3:(y * width + x) * 3 + 3] == b"\0\0\0"
                                for x in range(0, width, 3))]
            self.assertEqual(min(dark_rows), scale, name)
            self.assertEqual(max(dark_rows) - min(dark_rows) + 1, 7 * scale, name)


class AutocheckTests(unittest.TestCase):
    def test_four_bullets_must_name_the_four_organs(self):
        task = next(t for t in appset.new_tasks() if t["id"] == "form-four-bullets")
        spec = task["autocheck"]
        good = "- корень\n- стебель\n- лист\n- цветок"
        bad = "- один\n- два\n- три\n- четыре"
        self.assertTrue(runner.grade_autocheck({"text": good}, spec)[0])
        self.assertFalse(runner.grade_autocheck({"text": bad}, spec)[0])

    def test_long_facts_are_scored_individually_and_missing_is_not_success(self):
        task = next(t for t in appset.all_tasks() if t["domain"] == "long-facts")
        expected = task["autocheck"]["value"]
        self.assertEqual(runner.grade_long_facts(task, json.dumps(expected)), [True, True])
        self.assertEqual(runner.grade_long_facts(task, json.dumps([expected[0], 999])),
                         [True, False])
        self.assertEqual(runner.grade_long_facts(task, "bad json"), [False, False])
        self.assertEqual(runner.grade_long_facts(task, json.dumps([expected[0]])),
                         [True, False])

    def test_long_facts_gate_requires_all_24_checks(self):
        tasks = [t for t in appset.all_tasks() if t["domain"] == "long-facts"]
        rows = [{"id": t["id"], "domain": "long-facts", "fact_checks": [True, True]}
                for t in tasks]
        self.assertEqual(runner.summarize_long_facts(rows, tasks)["status"], "PASS")
        rows[0]["fact_checks"] = [False, False]
        self.assertEqual(runner.summarize_long_facts(rows, tasks)["status"], "PASS")
        rows[1]["fact_checks"] = [False, True]
        self.assertEqual(runner.summarize_long_facts(rows, tasks)["status"], "FAIL")
        self.assertEqual(runner.summarize_long_facts(rows[:-1], tasks)["status"],
                         "NOT_EXERCISED")

    def test_sampled_run_does_not_claim_greedy_long_gate(self):
        tasks = [t for t in appset.all_tasks() if t["domain"] == "long-facts"]
        rows = [{"id": t["id"], "domain": "long-facts", "fact_checks": [True, True]}
                for t in tasks]
        self.assertEqual(runner.summarize_long_facts(rows, tasks, "sampled")["status"],
                         "NOT_EXERCISED")

    def grade(self, text, spec, **extra):
        result = {"text": text, "finish": extra.get("finish", "stop"),
                  "tools": extra.get("tools", [])}
        return runner.grade_autocheck(result, spec)[0]

    def test_numeric_normalizes_unicode_minus_and_whitespace(self):
        self.assertTrue(self.grade(" −5 ", {"type": "numeric", "value": -5}))
        self.assertFalse(self.grade("Температура была −5 градусов", {"type": "numeric", "value": -5}))
        self.assertFalse(self.grade("Температура была 5", {"type": "numeric", "value": -5}))
        self.assertFalse(self.grade("15", {"type": "numeric", "value": 5}))

    def test_numbers_all_requires_every_value(self):
        spec = {"type": "numbers_all", "values": [11, 5]}
        self.assertTrue(self.grade("11 и 5", spec))
        self.assertFalse(self.grade("11 и 15", spec))

    def test_compact_ignores_whitespace_and_trailing_dot(self):
        self.assertTrue(self.grade("[1,  2, 3].", {"type": "compact", "value": "[1, 2, 3]"}))
        self.assertFalse(self.grade("[1, 3, 2]", {"type": "compact", "value": "[1, 2, 3]"}))

    def test_contains_and_excludes(self):
        spec = {"type": "all", "checks": [
            {"type": "contains_all", "values": ["пришли", "ярче, чем"]},
            {"type": "excludes_all", "values": ["пришло", "?"]}]}
        self.assertTrue(self.grade("Мы пришли в парк. Солнце светило ярче, чем вчера.", spec))
        self.assertFalse(self.grade("Мы пришло в парк. Солнце светило ярче, чем вчера?", spec))

    def test_structural_checks(self):
        self.assertTrue(self.grade("Один. Два. Три.", {"type": "sentence_count", "value": 3}))
        self.assertFalse(self.grade("Один. Два.", {"type": "sentence_count", "value": 3}))
        self.assertTrue(self.grade("- а\n- б\n- в\n- г", {"type": "bullet_count", "value": 4}))
        self.assertFalse(self.grade("- а\n- б", {"type": "bullet_count", "value": 4}))
        self.assertTrue(self.grade("АБВ\nГДЕ", {"type": "line_match", "value": ["АБВ", "ГДЕ"]}))
        self.assertFalse(self.grade("АБВ ГДЕ", {"type": "line_match", "value": ["АБВ", "ГДЕ"]}))

    def test_json_and_valid_tool_call(self):
        self.assertTrue(self.grade('{"answer": 391}', {"type": "json", "value": {"answer": 391}}))
        self.assertFalse(self.grade("не json", {"type": "json", "value": {}}))
        self.assertFalse(self.grade('{"answer": true}', {"type": "json", "value": {"answer": 1}}))
        good = {"finish": "tool_calls", "tools": [{"id": "1", "type": "function",
                                                   "function": {"name": "a", "arguments": "{}"}}]}
        self.assertTrue(runner.grade_autocheck(good, {"type": "valid_tool_call"}))
        bad = {"finish": "stop", "tools": []}
        self.assertFalse(runner.grade_autocheck(bad, {"type": "valid_tool_call"})[0])

    def test_hermes_tool_name_and_arguments_are_checked(self):
        task = next(t for t in appset.new_tasks() if t["id"] == "hermes-reuse-tool-01")
        call = {"id": "1", "type": "function", "function": {
            "name": "web_search", "arguments": '{"query":"солнечное затмение"}'}}
        result = {"finish": "tool_calls", "tools": [call]}
        self.assertTrue(runner.grade_autocheck(result, task["autocheck"])[0])
        call["function"]["name"] = "terminal"
        self.assertFalse(runner.grade_autocheck(result, task["autocheck"])[0])
        call["function"]["name"] = "web_search"
        call["function"]["arguments"] = '{"query":3}'
        self.assertFalse(runner.grade_autocheck(result, task["autocheck"])[0])
        call["function"]["arguments"] = '{"query":"затмение","limit":101}'
        self.assertFalse(runner.grade_autocheck(result, task["autocheck"])[0])
        call["function"]["arguments"] = '{"limit":5}'
        self.assertFalse(runner.grade_autocheck(result, task["autocheck"])[0])

    def test_hermes_parallel_tool_calls_checked_strictly(self):
        task = next(t for t in appset.new_tasks() if t["id"] == "hermes-reuse-mixed-02")
        good_calls = [
            {"id": "1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path":"/home/user/work/schedule.csv"}'}},
            {"id": "2", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path":"/home/user/work/notes.md"}'}},
        ]
        self.assertTrue(runner.grade_autocheck(
            {"finish": "tool_calls", "tools": good_calls}, task["autocheck"])[0])
        bad_second = [
            good_calls[0],
            {"id": "2", "type": "function",
             "function": {"name": "read_file", "arguments": '{"unknown_arg":1}'}},
        ]
        self.assertFalse(runner.grade_autocheck(
            {"finish": "tool_calls", "tools": bad_second}, task["autocheck"])[0])

    def test_every_auto_spec_is_satisfiable_by_a_correct_answer(self):
        """Каждая автопроверка набора должна пропускать эталонный ответ."""
        for task in appset.all_tasks():
            spec = task.get("autocheck")
            if task["evaluation"] != "auto" or spec is None:
                continue
            with self.subTest(task=task["id"]):
                answer = self._reference_answer(task)
                ok, detail = runner.grade_autocheck({"text": answer["text"],
                                                     "finish": answer.get("finish", "stop"),
                                                     "tools": answer.get("tools", [])}, spec)
                self.assertTrue(ok, task["id"] + ": " + str(detail))

    def _reference_answer(self, task):
        spec = task["autocheck"]
        if spec["type"] == "valid_tool_call":
            arguments = {spec["required_text"]: spec.get("contains", "пример")} \
                if "required_text" in spec else {}
            return {"text": "", "tools": [{"id": "x", "type": "function",
                                           "function": {"name": spec.get("name", "any"),
                                                        "arguments": json.dumps(arguments)}}],
                    "finish": "tool_calls"}
        if spec["type"] == "all":
            return {"text": self._reference_all(spec["checks"], task),
                    "finish": "stop", "tools": []}
        return {"text": self._reference_text(spec), "finish": "stop", "tools": []}

    def _reference_all(self, checks, task):
        if task["id"] == "spelling-park":
            return "Мы пришли в парк. Трава была высокая. Солнце светило ярче, чем вчера."
        if task["id"] == "spelling-forest":
            return "Дети пошли в лес. В густом лесу росли грибы. Мы взяли корзину и ножницы."
        if task["id"] == "form-three-sentences":
            return ("Периметр многоугольника — это сумма длин всех его сторон. "
                    "Его находят сложением. Единицы длины — сантиметры.")
        if task["id"] == "form-four-bullets":
            return "- корень\n- стебель\n- лист\n- цветок"
        raise ValueError(task["id"])

    def _reference_text(self, spec):
        return {"numeric": lambda: str(spec["value"]),
                "numbers_all": lambda: " ".join(str(value) for value in spec["values"]),
                "compact": lambda: str(spec["value"]),
                "contains_all": lambda: " ".join(str(value) for value in spec["values"]),
                "json": lambda: json.dumps(spec["value"], ensure_ascii=False),
                "sentence_count": lambda: ". ".join(["Слово"] * spec["value"]) + ".",
                "bullet_count": lambda: "\n".join("- пункт %d" % (i + 1)
                                                  for i in range(spec["value"])),
                "line_match": lambda: "\n".join(spec["value"])}[spec["type"]]()


class RequestPolicyTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "нужен jsonschema")
    def test_real_loopback_transport_replays_two_validated_tool_calls(self):
        task = next(t for t in appset.all_tasks() if t["id"] == "tool-chain-weather-сочи")
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                index = len(seen)
                seen.append(body)
                message = {"role": "assistant", "content": "68"}
                finish = "stop"
                if index < 2:
                    step = task["tool_sequence"][index]
                    message = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": f"call_{index}", "type": "function",
                        "function": {"name": step["name"],
                                     "arguments": json.dumps(step["arguments"], ensure_ascii=False)}}]}
                    finish = "tool_calls"
                payload = {"id": f"fake_{index}", "created": 1, "model": body["model"],
                           "object": "chat.completion",
                           "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                           "usage": {"prompt_tokens": 5000, "completion_tokens": 2,
                                     "total_tokens": 5002, "cached_tokens": 0,
                                     "draft_tokens": 0, "draft_tokens_accepted": 0,
                                     "gufo": {"prefill_tokens": 5000, "prefill_ms": 2.5,
                                              "decode_ms": 3.0, "queue_ms": 0, "ttft_ms": 3,
                                              "cache_restore_ms": 0, "cache_snapshot_ms": 0,
                                              "cache_disk_enqueue_ms": 0}}}
                raw = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                instance = runner.SetRunner(f"http://127.0.0.1:{server.server_port}",
                                            Path(tmp) / "run", "greedy", 0)
                row = instance.run_task(task)
                self.assertEqual(row["status"], "PASS", row.get("error"))
                self.assertEqual(len(seen), 3)
                self.assertEqual([body["cache_prompt"] for body in seen], [False, True, True])
                self.assertEqual(len(row["tool_trace"]), 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_tool_chain_has_two_distinct_calls(self):
        tasks = [t for t in appset.new_tasks() if t["domain"] == "tool-chain"]
        self.assertEqual(len(tasks), 6)
        for task in tasks:
            with self.subTest(task=task["id"]):
                self.assertEqual(len(task["tool_sequence"]), 2)
                self.assertNotEqual(task["tool_sequence"][0]["name"],
                                    task["tool_sequence"][1]["name"])
                self.assertEqual(task["target_range"], [2000, 18000])
                self.assertGreater(len(task["first_prompt"]) / appset.CHARS_PER_TOKEN, 2000)

    def test_runner_executes_two_tool_rounds_before_final_answer(self):
        task = next(t for t in appset.new_tasks() if t["id"] == "tool-chain-weather-сочи")
        calls = []
        for index, step in enumerate(task["tool_sequence"]):
            calls.append({"status": "PASS", "text": "", "finish": "tool_calls",
                          "tools": [{"id": f"call_{index}", "type": "function",
                                     "function": {"name": step["name"],
                                                  "arguments": json.dumps(step["arguments"])}}]})
        calls.append({"status": "PASS", "text": "68", "finish": "stop",
                      "metrics": {"prompt_tokens": 5000}})
        with tempfile.TemporaryDirectory() as tmp:
            instance = runner.SetRunner("http://127.0.0.1:1", Path(tmp) / "run", "greedy", 0)
            with patch.object(instance, "_transport", side_effect=calls) as transport:
                row = instance.run_task(task)
            self.assertEqual(transport.call_count, 3)
            self.assertEqual(row["status"], "PASS", row)
            self.assertEqual(len(row["tool_trace"]), 2)
            self.assertEqual([item["tools"][0]["function"]["name"] for item in row["tool_trace"]],
                             [step["name"] for step in task["tool_sequence"]])
    def test_partial_run_cannot_claim_overall_pass(self):
        task = next(t for t in appset.all_tasks() if t["id"] == "ocr-table-a")
        with tempfile.TemporaryDirectory() as tmp:
            instance = runner.SetRunner("http://127.0.0.1:1", Path(tmp) / "run",
                                        "greedy", 0)
            response = {"status": "PASS", "text": "11", "finish": "stop", "metrics": {}}
            with patch.object(instance, "_transport", return_value=response):
                instance.run([task])
            self.assertNotEqual(instance.report["status"], "PASS")
    def test_reasoning_tasks_allow_budget_and_visible_answer(self):
        for task in appset.new_tasks():
            if task["thinking"] != "hermes" or task["domain"] == "hermes-reuse":
                continue
            with self.subTest(task=task["id"]):
                self.assertGreater(task["max_completion_tokens"], 512)

    def test_real_loopback_transport_records_response_and_grades_it(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payload = {
                    "id": "fake", "created": 1, "model": body["model"],
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "11"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 2,
                              "total_tokens": 22, "cached_tokens": 0,
                              "draft_tokens": 0, "draft_tokens_accepted": 0,
                              "gufo": {"prefill_tokens": 20, "prefill_ms": 2.5,
                                       "decode_ms": 3.0, "queue_ms": 0, "ttft_ms": 3,
                                       "cache_restore_ms": 0, "cache_snapshot_ms": 0,
                                       "cache_disk_enqueue_ms": 0}}}
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        task = next(t for t in appset.all_tasks() if t["id"] == "ocr-table-a")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                instance = runner.SetRunner(f"http://127.0.0.1:{server.server_port}",
                                            Path(tmp) / "run", "greedy", 0)
                row = instance.run_task(task)
                self.assertEqual(row["status"], "PASS", row.get("error"))
                self.assertTrue((instance.output / "requests" / task["id"] /
                                 "response.raw").is_file())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_long_result_without_verified_token_count_is_not_pass(self):
        task = next(t for t in appset.all_tasks() if t["id"] == "long-facts-1-begin")
        answer = str(task["expected_values"]["temp"])
        with tempfile.TemporaryDirectory() as tmp:
            instance = runner.SetRunner("http://127.0.0.1:1", Path(tmp) / "run",
                                        "greedy", 0)
            response = {"status": "PASS", "text": answer, "finish": "stop", "metrics": {}}
            with patch.object(instance, "_transport", return_value=response):
                row = instance.run_task(task)
        self.assertNotEqual(row["status"], "PASS")

    def test_truncated_autocheck_result_is_not_pass(self):
        task = next(t for t in appset.all_tasks() if t["id"] == "ocr-table-a")
        with tempfile.TemporaryDirectory() as tmp:
            instance = runner.SetRunner("http://127.0.0.1:1", Path(tmp) / "run",
                                        "greedy", 0)
            response = {"status": "PASS", "text": "11", "finish": "length", "metrics": {}}
            with patch.object(instance, "_transport", return_value=response):
                row = instance.run_task(task)
        self.assertEqual(row["status"], "FAIL")

    def test_tool_call_must_match_its_declared_schema(self):
        task = next(t for t in appset.all_tasks() if t["id"] == "tool-chain-weather-сочи")
        with tempfile.TemporaryDirectory() as tmp:
            instance = runner.SetRunner("http://127.0.0.1:1", Path(tmp) / "run",
                                        "greedy", 0)
            call = {"id": "", "type": "not-a-function",
                    "function": {"name": task["tool_name"],
                                 "arguments": json.dumps(task["tool_arguments"])}}
            with self.assertRaises(Exception):
                instance._verify_tool_turn({"finish": "tool_calls", "tools": [call]}, task)

    def test_every_task_builds_a_request_without_network(self):
        for task in appset.all_tasks():
            with self.subTest(task=task["id"]):
                steps = runner.build_requests(task, "greedy", 0)
                self.assertTrue(steps)
                body = steps[0][1]["body"]
                if "response_format" in task:
                    self.assertEqual(body.get("response_format"), task["response_format"])

    def test_point_tasks_disable_thinking_and_cache(self):
        task = next(t for t in appset.all_tasks() if t["id"] == "physics-train")
        for pass_name, seed in (("greedy", 0), ("sampled", 7)):
            steps = runner.build_requests(task, pass_name, seed)
            self.assertEqual(len(steps), 1)
            body = steps[0][1]["body"]
            self.assertEqual(body["chat_template_kwargs"]["enable_thinking"], False)
            self.assertFalse(body["cache_prompt"])
            self.assertEqual(body["temperature"], 0 if pass_name == "greedy" else 1)
            if pass_name == "sampled":
                self.assertEqual((body["top_k"], body["top_p"], body["min_p"], body["seed"]),
                                 (20, 0.95, 0.05, 7))

    def test_agent_tasks_keep_hermes_profile_and_turn2_caches(self):
        task = next(t for t in appset.all_tasks() if t["id"] == "tool-chain-weather-сочи")
        steps = runner.build_requests(task, "greedy", 0)
        body = steps[0][1]["body"]
        self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(body["reasoning_budget_tokens"], 512)
        self.assertIn("tools", body)
        fake_turn1 = {"text": None, "finish": "tool_calls", "tools": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "сочи"}'}}]}
        turn2 = runner.turn2_request(task, fake_turn1, "greedy", 0)["body"]
        self.assertTrue(turn2["cache_prompt"])
        self.assertTrue(turn2["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(turn2["reasoning_budget_tokens"], 512)
        self.assertEqual([m["role"] for m in turn2["messages"]],
                         ["user", "assistant", "tool", "user"])
        self.assertEqual(json.loads(turn2["messages"][2]["content"]), task["tool_result"])

    def test_hermes_reuse_uses_the_pinned_fixture_without_overrides(self):
        task = next(t for t in appset.all_tasks() if t["id"] == "hermes-reuse-tool-01")
        body = runner.build_requests(task, "greedy", 0)[0][1]["body"]
        self.assertNotIn("chat_template_kwargs", body)
        self.assertEqual(len(body["tools"]), 20)
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})

    def test_runner_rejects_existing_output_and_bad_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "run"
            existing.mkdir()
            with self.assertRaises(Exception):
                runner.SetRunner("http://127.0.0.1:1", existing, "greedy", 0)
            with self.assertRaises(Exception):
                runner.SetRunner("http://127.0.0.1:1", Path(tmp) / "new", "bogus", 0)

    def test_runner_rejects_unsafe_task_id_before_writing(self):
        task = dict(next(t for t in appset.all_tasks() if t["id"] == "ocr-table-a"))
        task["id"] = "../../escape"
        with tempfile.TemporaryDirectory() as tmp:
            instance = runner.SetRunner("http://127.0.0.1:1", Path(tmp) / "run", "greedy", 0)
            row = instance.run_task(task)
            self.assertEqual(row["status"], "FAIL")
            self.assertFalse((Path(tmp) / "escape.json").exists())


class BlindProtocolTests(unittest.TestCase):
    """Экспорт и раскрытие на сфабрикованных прогонах (без HTTP и модели)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.runs = {}
        for name in ("a", "b"):
            self.runs[name] = self._fabricate_run(self.tmp / ("run-" + name), None)
        self.run_a, self.run_b = self.runs["a"], self.runs["b"]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _correct_result(self, task):
        """Эталонный ответ задачи: автопроверка обязана его пропустить."""
        if task["evaluation"] == "auto":
            spec = task.get("autocheck") or {"type": "valid_tool_call"}
            if spec["type"] == "valid_tool_call":
                arguments = {spec["required_text"]: spec.get("contains", "пример")} \
                    if "required_text" in spec else {}
                return {"text": "", "finish": "tool_calls", "tools": [
                    {"id": "x", "type": "function",
                     "function": {"name": spec.get("name", "any"),
                                  "arguments": json.dumps(arguments)}}]}
            if spec["type"] == "all":
                probe = AutocheckTests("test_json_and_valid_tool_call")
                return {"text": probe._reference_all(spec["checks"], task),
                        "finish": "stop", "tools": []}
            probe = AutocheckTests("test_json_and_valid_tool_call")
            return {"text": probe._reference_text(spec), "finish": "stop", "tools": []}
        return {"text": "Эталонный ответ по рубрике.", "finish": "stop", "tools": []}

    def _fabricate_run(self, path: Path, break_id: str | None) -> Path:
        rows = []
        for task in appset.all_tasks():
            result = self._correct_result(task)
            status = "PASS"
            if task["evaluation"] == "auto" and break_id == task["id"]:
                result["text"] = "заведомо неверный ответ"
            rows.append({"id": task["id"], "domain": task["domain"],
                         "bucket": task["bucket"], "evaluation": task["evaluation"],
                         "status": status, "finish": result["finish"],
                         "text": result.get("text", ""), "tools": result.get("tools", [])})
        path.mkdir(parents=True)
        hashes = runner.fixture_hashes()
        (path / "report.json").write_text(json.dumps(
            {"status": "PASS", "pass_name": "sampled", "seed": 1, "results": rows,
             "fixture_sha256": hashes},
            ensure_ascii=False), encoding="utf-8")
        return path

    def _blind(self, name="blind"):
        out = self.tmp / name
        runner.export_blind(self.run_a, self.run_b, out, 20261119)
        return out

    def _key_path(self, out):
        return out.parent / (out.name + ".key.json")

    def test_export_hides_identity_and_keeps_review_material(self):
        out = self._blind()
        pairs = [json.loads(line) for line in
                 (out / "blind_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
        key = json.loads(self._key_path(out).read_text(encoding="utf-8"))
        self.assertEqual(len(pairs), 96)
        self.assertEqual(len(key["pairs"]), 96)
        self.assertFalse((out / "blinding_key.json").exists())
        self.assertNotIn("run_a", (out / "manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn("run_b", (out / "manifest.json").read_text(encoding="utf-8"))
        for pair in pairs:
            self.assertNotIn("id", pair)
            self.assertNotIn("run_a", json.dumps(pair))
            self.assertNotIn("run_b", json.dumps(pair))
            self.assertTrue(pair["prompt"].strip())
            self.assertTrue(pair["messages"])
            self.assertIn("expected_facts", pair)
            self.assertIn("prohibited_errors", pair)
            for side in ("response_a", "response_b"):
                self.assertNotIn("auto", pair[side])
                self.assertNotIn("status", pair[side])
        manual = [pair for pair in pairs if pair["evaluation"] == "manual"]
        self.assertEqual(len(manual), 28)
        for pair in manual:
            self.assertTrue(pair["prompt"].strip())
            self.assertIn("rubric", pair)
            self.assertIn("pass_criteria", pair)
        big = [pair for pair in pairs if pair["task_class"].startswith("big-text")]
        self.assertEqual(len(big), 2)
        for pair in big:
            self.assertTrue(pair["images"], "known_limit pairs keep their images")
            for image_path in pair["images"]:
                self.assertIsInstance(image_path, str)
                self.assertTrue((out / image_path).read_bytes().startswith(b"\x89PNG"))
        long_tool = next(pair for pair in pairs if pair["task_class"].startswith("long-tool-dialog"))
        self.assertEqual(sum(m["role"] == "tool" for m in long_tool["messages"]), 2)
        vision = next(pair for pair in pairs if pair["task_class"].startswith("long-vision-dialog"))
        self.assertTrue(vision["images"])
        self.assertNotIn("data:image", json.dumps(vision["messages"]))

    def test_unblind_on_identical_runs_shows_zero_difference(self):
        out = self._blind()
        reviews_path = self.tmp / "reviews.jsonl"
        with reviews_path.open("w", encoding="utf-8") as file:
            for line in (out / "blind_pairs.jsonl").read_text(encoding="utf-8").splitlines():
                pair = json.loads(line)
                if pair["evaluation"] != "known_limit":
                    grade = {"scores": {"correctness": 2, "instruction": 2,
                                        "russian_clarity": 2, "explanation": 2},
                             "critical_failures": [], "rationale": "Проверено"}
                    file.write(json.dumps({"pair_id": pair["pair_id"], "a": grade,
                                           "b": grade}, ensure_ascii=False) + "\n")
        summary = runner.unblind(out, reviews_path)
        self.assertEqual(summary["total_graded_pairs"], 94)
        self.assertEqual(summary["outcomes"], {"both_ok": 94, "only_a": 0,
                                               "only_b": 0, "both_fail": 0})
        self.assertEqual(summary["per_variant"], {
            "a": {"ok": 94, "wrong": 0, "not_graded": 2},
            "b": {"ok": 94, "wrong": 0, "not_graded": 2}})

    def test_single_candidate_only_loss_is_visible_after_unblind(self):
        wrong = next(task["id"] for task in appset.all_tasks()
                     if task["evaluation"] == "auto")
        self.run_b = self._fabricate_run(self.tmp / "run-c", wrong)
        out = self.tmp / "blind2"
        runner.export_blind(self.run_a, self.run_b, out, 20261119)
        reviews_path = self.tmp / "reviews2.jsonl"
        with reviews_path.open("w", encoding="utf-8") as file:
            for line in (out / "blind_pairs.jsonl").read_text(encoding="utf-8").splitlines():
                pair = json.loads(line)
                if pair["evaluation"] != "known_limit":
                    grade = {"scores": {"correctness": 2, "instruction": 2,
                                        "russian_clarity": 2, "explanation": 2},
                             "critical_failures": [], "rationale": "Проверено"}
                    bad = {"scores": {"correctness": 0, "instruction": 2,
                                      "russian_clarity": 2, "explanation": 2},
                           "critical_failures": [], "rationale": "Ошибка"}
                    key = json.loads(self._key_path(out).read_text(encoding="utf-8"))
                    entry = key["pairs"][pair["pair_id"]]
                    a = bad if entry["task"] == wrong and entry["a_is"] == "run_b" else grade
                    b = bad if entry["task"] == wrong and entry["a_is"] == "run_a" else grade
                    file.write(json.dumps({"pair_id": pair["pair_id"], "a": a,
                                           "b": b}, ensure_ascii=False) + "\n")
        summary = runner.unblind(out, reviews_path)
        self.assertEqual(summary["outcomes"]["only_a"], 0)
        self.assertEqual(summary["outcomes"]["only_b"], 1)
        self.assertEqual(summary["outcomes"]["both_ok"], 93)

    def test_unblind_requires_reviews_for_every_manual_pair(self):
        out = self._blind()
        with self.assertRaises(Exception):
            runner.unblind(out, None)

    def test_export_rejects_failed_transport_instead_of_reviewing_it_as_success(self):
        report_path = self.run_b / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["results"][0].update(status="FAIL", error="timeout")
        report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(Exception):
            runner.export_blind(self.run_a, self.run_b, self.tmp / "bad", 20261119)

    def test_export_rejects_mismatched_fixture_hashes(self):
        report_path = self.run_b / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["fixture_sha256"][runner.APPSET_FILE] = "different"
        report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(Exception):
            runner.export_blind(self.run_a, self.run_b, self.tmp / "bad", 20261119)

    def test_unblind_tracks_independent_manual_review_per_original_side(self):
        out = self._blind()
        key = json.loads(self._key_path(out).read_text(encoding="utf-8"))["pairs"]
        pairs = [json.loads(line) for line in
                 (out / "blind_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
        chosen = next(pair for pair in pairs if pair["evaluation"] == "manual"
                      and key[pair["pair_id"]]["a_is"] == "run_b")
        good = {"scores": {"correctness": 2, "instruction": 2,
                           "russian_clarity": 2, "explanation": 2},
                "critical_failures": [], "rationale": "Проверено"}
        bad = {"scores": {"correctness": 0, "instruction": 2,
                          "russian_clarity": 2, "explanation": 2},
               "critical_failures": ["incorrect_fact"], "rationale": "Неверный факт"}
        reviews = self.tmp / "reviews-independent.jsonl"
        with reviews.open("w", encoding="utf-8") as file:
            for pair in pairs:
                if pair["evaluation"] != "known_limit":
                    record = {"pair_id": pair["pair_id"],
                              "a": bad if pair is chosen else good, "b": good}
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = runner.unblind(out, reviews)
        self.assertEqual(summary["outcomes"]["only_b"], 1)
        self.assertEqual(summary["critical_counts"]["b"], 1)
        self.assertEqual(summary["critical_counts"]["a"], 0)
        self.assertEqual(summary["outcomes"]["only_a"], 0)

    def test_sampled_auto_answer_is_also_reviewed_blindly(self):
        out = self._blind()
        key = json.loads(self._key_path(out).read_text(encoding="utf-8"))["pairs"]
        pairs = [json.loads(line) for line in
                 (out / "blind_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
        chosen = next(pair for pair in pairs if pair["evaluation"] == "auto"
                      and key[pair["pair_id"]]["a_is"] == "run_b")
        good = {"scores": {"correctness": 2, "instruction": 2,
                           "russian_clarity": 2, "explanation": 2},
                "critical_failures": [], "rationale": "Проверено"}
        bad = {"scores": {"correctness": 0, "instruction": 2,
                          "russian_clarity": 2, "explanation": 2},
               "critical_failures": ["incorrect_fact"], "rationale": "Неверный факт"}
        reviews = self.tmp / "reviews-all.jsonl"
        with reviews.open("w", encoding="utf-8") as file:
            for pair in pairs:
                if pair["evaluation"] != "known_limit":
                    file.write(json.dumps({"pair_id": pair["pair_id"],
                                           "a": bad if pair is chosen else good, "b": good},
                                          ensure_ascii=False) + "\n")
        summary = runner.unblind(out, reviews)
        self.assertEqual(summary["outcomes"]["only_b"], 1)


class ProvenanceCalibrateAndReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_verify_server_and_calibrate_via_loopback_transport(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                routes = {
                    "/health": {"status": "ok"},
                    "/ready": {"status": "ready", "model": runner.ALIAS},
                    "/props": {"model_alias": runner.ALIAS, "model": runner.ALIAS, "total_slots": 4},
                    "/v1/models": {"object": "list", "data": [{"id": runner.ALIAS}]},
                }
                payload = json.dumps(routes[self.path]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                approx = max(1, int(len(body["content"]) / appset.CHARS_PER_TOKEN))
                payload = json.dumps({"tokens": list(range(approx))}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            set_runner = runner.SetRunner(url, self.tmp / "run", "greedy", 0)
            prov = set_runner.verify_server()
            self.assertEqual(set_runner.report["provenance_status"], "VERIFIED")
            self.assertEqual(prov["props"]["model_alias"], runner.ALIAS)
            long_tasks = [t for t in appset.new_tasks() if t["domain"] in ("long-facts", "tool-chain")][:4]
            cal = runner.calibrate_lengths(url, long_tasks, self.tmp / "cal", verify=True)
            self.assertEqual(cal["status"], "CALIBRATED")
            self.assertEqual(cal["provenance_status"], "VERIFIED")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def _make_greedy_run(self, name: str, provenance: str = "VERIFIED",
                         wrong_auto_ids=(), long_facts_correct: int = 24):
        path = self.tmp / name
        path.mkdir(parents=True)
        rows = []
        lf_remaining = long_facts_correct
        for task in appset.all_tasks():
            if task["evaluation"] == "known_limit":
                rows.append({"id": task["id"], "domain": task["domain"],
                             "bucket": task["bucket"], "evaluation": "known_limit",
                             "status": "KNOWN_LIMIT_FAIL", "finish": "stop", "text": "WRONG"})
                continue
            status = "FAIL" if task["id"] in wrong_auto_ids else "PASS"
            row = {"id": task["id"], "domain": task["domain"], "bucket": task["bucket"],
                   "evaluation": task["evaluation"], "status": status,
                   "finish": "stop", "text": "ok"}
            if task["domain"] == "long-facts":
                c0 = lf_remaining > 0
                lf_remaining -= 1
                c1 = lf_remaining > 0
                lf_remaining -= 1
                row["fact_checks"] = [c0, c1]
            rows.append(row)
        (path / "report.json").write_text(json.dumps({
            "status": "PASS", "pass_name": "greedy", "seed": 0,
            "provenance_status": provenance, "fixture_sha256": runner.fixture_hashes(),
            "results": rows,
        }, ensure_ascii=False), encoding="utf-8")
        return path

    def _make_sampled_unblind(self, name: str, a_wrong: int, b_wrong: int, crit_b: int = 0):
        path = self.tmp / name
        path.mkdir(parents=True)
        (path / "unblinded.json").write_text(json.dumps({
            "outcomes": {"both_ok": 94 - a_wrong - b_wrong, "only_a": a_wrong,
                         "only_b": b_wrong, "both_fail": 0},
            "total_graded_pairs": 94,
            "per_variant": {"a": {"ok": 94 - a_wrong, "wrong": a_wrong, "not_graded": 2},
                            "b": {"ok": 94 - b_wrong, "wrong": b_wrong, "not_graded": 2}},
            "critical_counts": {"a": 0, "b": crit_b},
            "by_domain": {"ocr-text": {"both_ok": 8, "only_a": 0, "only_b": 0, "both_fail": 0}},
            "paired_difference": {"a_minus_b": (a_wrong - b_wrong) / 94,
                                  "b_minus_a": (b_wrong - a_wrong) / 94},
        }), encoding="utf-8")
        return path

    def test_report_enforces_provenance_and_barriers(self):
        ga = self._make_greedy_run("ga", provenance="VERIFIED")
        gb_unverified = self._make_greedy_run("gb-u", provenance="UNVERIFIED")
        sampled_tie = self._make_sampled_unblind("sb-tie", a_wrong=2, b_wrong=2)
        rep1 = runner.build_appset_report(ga, gb_unverified, self.tmp / "r1.json",
                                          sampled_unblind=sampled_tie)
        self.assertEqual(rep1["status"], "INCOMPLETE_PROVENANCE")

        gb_ok = self._make_greedy_run("gb-ok", provenance="VERIFIED", long_facts_correct=22)
        rep_no_sampled = runner.build_appset_report(ga, gb_ok, self.tmp / "r2.json")
        self.assertEqual(rep_no_sampled["status"], "NOT_EXERCISED")

        rep_pass = runner.build_appset_report(ga, gb_ok, self.tmp / "r3.json",
                                              sampled_unblind=sampled_tie)
        self.assertEqual(rep_pass["status"], "DESCRIPTIVE_TASK_GATE_PASS")
        self.assertEqual(rep_pass["q1_qualification"], "REQUIRES_NUMERICAL_SLICE")
        self.assertEqual(len(rep_pass["known_limit_isolated"]), 2)

        gb_bad_lf = self._make_greedy_run("gb-lf", provenance="VERIFIED", long_facts_correct=21)
        rep_fail_lf = runner.build_appset_report(ga, gb_bad_lf, self.tmp / "r4.json",
                                                 sampled_unblind=sampled_tie)
        self.assertEqual(rep_fail_lf["status"], "FAIL")

        gb_bad_auto = self._make_greedy_run("gb-a", provenance="VERIFIED",
                                            wrong_auto_ids=("ocr-table-a", "ocr-table-b"))
        rep_fail_auto = runner.build_appset_report(ga, gb_bad_auto, self.tmp / "r5.json",
                                                   sampled_unblind=sampled_tie)
        self.assertEqual(rep_fail_auto["status"], "FAIL")

        sampled_bad = self._make_sampled_unblind("sb-bad", a_wrong=0, b_wrong=5)
        rep_fail_sampled = runner.build_appset_report(ga, gb_ok, self.tmp / "r6.json",
                                                      sampled_unblind=sampled_bad)
        self.assertEqual(rep_fail_sampled["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
