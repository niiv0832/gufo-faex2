"""CPU-проверки сценария B3: запросы, границы доказательств и restart."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from faex2_acceptance import Runner, grade


class RecordingRunner:
    """Подменяет только внешний модельный HTTP; grade остаётся настоящим."""

    mode = "mtp"

    def __init__(self, damage=None):
        self.calls, self.batches = [], []
        self.damage = damage

    def run_case(self, identifier, request, expected, timeout=120, cancel=None):
        self.calls.append((identifier, copy.deepcopy(request), copy.deepcopy(expected), timeout, cancel))
        body = request["body"]
        messages = body.get("messages", body.get("input", []))
        contents = "\n".join(message["content"] for message in messages)
        # Независимая модель tokenizer: реальный клиент обязан пользоваться usage.
        count = contents.count("Архивная запись:")
        total = 240 + count * 31
        values = {"OWL_142": [340, 357, 374, 391, 408, 425, 442, 459],
                  "MOON_853": [460, 483, 506, 529, 552, 575, 598, 621],
                  "PINE_627": [580, 609, 638, 667, 696, 725, 754, 783],
                  "LAKE_394": [620, 651, 682, 713, 744, 775, 806, 837]}
        if identifier.startswith("family-transition"):
            marker = next(marker for marker in values if marker in contents)
            answer = {"session": marker, "products": values[marker]}
        elif "calibrate" in identifier:
            answer = "ГОТОВО"
        elif "reasoning" in identifier:
            answer = {"answer": 7}
        elif "cancel" in identifier or "recover" in identifier:
            answer = {"squares": [n * n for n in range(1, 21)]}
        else:
            answer = {"start": "ЛИПА_274", "middle": "РЕКА_583",
                      "end": "ЗВЕЗДА_408" if "ЗВЕЗДА_408" in contents else "СОВА_916"}
        cached = 0 if body.get("cache_prompt") is False else max(0, total - 8)
        reasoning = body.get("reasoning", {}).get("effort") == "low"
        result = {"status": "CANCELLED" if cancel else "PASS", "http_status": 200,
                  "text": answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False),
                  "reasoning": "Решаю уравнение." if reasoning else "", "tools": [],
                  "finish": None if cancel else "stop",
                  "usage": {"prompt_tokens": total, "completion_tokens": 100,
                            "output_tokens_details": {"reasoning_tokens": 50 if reasoning else 0},
                            "gufo": {"cache_disk_hit": "restart" in identifier}},
                  "metrics": {"prompt_tokens": total, "cached_tokens": cached,
                              "prefill_tokens": total - cached, "completion_tokens": 100,
                              "draft_tokens": 50, "draft_tokens_accepted": 40}}
        if self.damage:
            self.damage(identifier, result)
        grade(result, expected, self.mode)
        return result

    def run_batch(self, cases, timeout=120):
        self.batches.append([case[0] for case in cases])
        return [self.run_case(*case, timeout=timeout) for case in cases]


class FamilyTest(unittest.TestCase):
    def api(self):
        path = Path(__file__).with_name("faex2_family.py")
        self.assertTrue(path.is_file(), "B3 family client is not implemented")
        spec = importlib.util.spec_from_file_location("faex2_family", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def run_initial(self, root, runner=None):
        runner = runner or RecordingRunner()
        state = Path(root) / "family-state.json"
        result = self.api().run_family(runner, state_path=state)
        return runner, state, result

    def test_transitions_keep_four_independent_answers_and_return_to_real_mtp(self):
        with tempfile.TemporaryDirectory() as root:
            runner, _, result = self.run_initial(root)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual([len(group) for group in runner.batches], [1, 2, 4, 1])
        transition = [call for call in runner.calls if call[0].startswith("family-transition")]
        self.assertEqual(len(transition), 8)
        self.assertTrue(transition[0][2]["draft_work"])
        self.assertTrue(transition[-1][2]["draft_work"])
        middle = transition[3:7]
        self.assertEqual({call[2]["json"]["session"] for call in middle},
                         {"OWL_142", "MOON_853", "PINE_627", "LAKE_394"})
        for _, request, expected, _, _ in middle:
            self.assertEqual(len(expected["excludes"]), 3)
            self.assertNotIn("draft_work", expected)
            self.assertEqual(request["body"]["temperature"], 1)
            self.assertNotIn("enum", json.dumps(request["body"]["response_format"]))

    def test_calibration_uses_two_actual_depths_and_retains_every_probe(self):
        with tempfile.TemporaryDirectory() as root:
            runner, state, _ = self.run_initial(root)
            saved = json.loads(state.read_text())
            original_bytes = state.read_bytes()
            probes = [call for call in runner.calls if "calibrate" in call[0]]
            self.assertGreaterEqual(len(probes), 3)
            self.assertEqual(len(saved["calibrations"]), len(probes))
            self.assertGreaterEqual(saved["prompt_tokens"], 110000)
            self.assertLess(saved["prompt_tokens"], 130560)
            self.assertLess(saved["prompt_tokens"], 121000)
            self.assertTrue(all(call[3] == 600 for call in probes))
            self.assertTrue(all(call[1]["body"]["max_completion_tokens"] <= 16 for call in probes))
            self.assertTrue(all(call[1]["body"]["cache_prompt"] is False for call in probes))
            rerun = RecordingRunner()
            with self.assertRaises((ValueError, FileExistsError)):
                self.api().run_family(rerun, state_path=state)
            self.assertEqual(rerun.calls, [])
            self.assertEqual(state.read_bytes(), original_bytes)

    def test_three_keys_are_only_in_archive_and_edit_drops_stale_assistant_answer(self):
        with tempfile.TemporaryDirectory() as root:
            runner, _, _ = self.run_initial(root)
        cold = next(call for call in runner.calls if call[0] == "family-long-cold")
        edit = next(call for call in runner.calls if call[0] == "family-long-edit")
        expected = {"start": "ЛИПА_274", "middle": "РЕКА_583", "end": "СОВА_916"}
        self.assertEqual(cold[2]["json"], expected)
        self.assertEqual(edit[2]["json"], {**expected, "end": "ЗВЕЗДА_408"})
        for call in (cold, edit):
            body = call[1]["body"]
            self.assertNotIn("ЛИПА_274", body["messages"][-1]["content"])
            self.assertNotIn("ЛИПА_274", json.dumps(body["response_format"]))
            self.assertEqual(call[2]["min_prompt_tokens"], 110000)
            self.assertLess(call[2]["max_prompt_tokens"], 130560)
        self.assertNotIn("СОВА_916", json.dumps(edit[1], ensure_ascii=False))
        self.assertIn("СОВА_916", edit[2]["excludes"])
        archive = cold[1]["body"]["messages"]
        positions = [next(i for i, item in enumerate(archive) if key in item["content"])
                     for key in ("ЛИПА_274", "РЕКА_583", "СОВА_916")]
        self.assertLess(positions[0], positions[1])
        self.assertLess(positions[1], positions[2])

    def test_reuse_checks_run_before_any_cold_control(self):
        with tempfile.TemporaryDirectory() as root:
            runner, _, _ = self.run_initial(root)
        ids = [call[0] for call in runner.calls]
        window = runner.calls[ids.index("family-long-cold") + 1:ids.index("family-long-edit-repeat") + 1]
        self.assertEqual([call[0] for call in window], ["family-long-repeat", "family-long-continue",
                                                     "family-long-edit", "family-long-edit-repeat"])
        for _, request, expected, _, _ in window:
            self.assertTrue(request["body"]["cache_prompt"])
            self.assertGreater(expected["min_cached_tokens"], 100000)
            self.assertLessEqual(expected["max_prefill_tokens"], 256)

    def test_bad_cache_accounting_fails_even_when_text_and_depth_are_right(self):
        def damage(identifier, result):
            if identifier == "family-long-repeat":
                result["metrics"]["cached_tokens"] = 0
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                self.run_initial(root, RecordingRunner(damage))
            self.assertFalse((Path(root) / "family-state.json").exists())

    def test_non_linear_or_invalid_calibration_never_becomes_qualified_depth(self):
        for broken in (None, True, 0, 8000):
            def damage(identifier, result):
                if "calibrate" in identifier:
                    result["metrics"]["prompt_tokens"] = broken
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as root:
                runner = RecordingRunner(damage)
                with self.assertRaises(ValueError):
                    self.run_initial(root, runner)
                self.assertFalse(any(call[0] == "family-long-cold" for call in runner.calls))

    def test_depth_gate_checks_actual_extraction_not_only_calibration(self):
        for depth in (109999, 130560):
            def damage(identifier, result):
                if identifier == "family-long-cold":
                    result["metrics"].update(prompt_tokens=depth, cached_tokens=0, prefill_tokens=depth)
            with self.subTest(depth=depth), tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ValueError):
                    self.run_initial(root, RecordingRunner(damage))
                self.assertFalse((Path(root) / "family-state.json").exists())

    def test_inconsistent_cache_sum_cannot_pass_individual_bounds(self):
        def damage(identifier, result):
            if identifier == "family-long-repeat":
                result["metrics"]["prefill_tokens"] = 7
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "accounting"):
                self.run_initial(root, RecordingRunner(damage))

    def test_post_request_accounting_failure_survives_real_runner_finish(self):
        def damage(identifier, result):
            if identifier == "family-long-repeat":
                result["metrics"]["prefill_tokens"] = 7
        fixture = RecordingRunner(damage)

        class LocalRunner(Runner):
            def exchange(self, identifier, request, expected, timeout=120, cancel=None):
                result = fixture.run_case(identifier, request, expected, timeout, cancel)
                return {"id": identifier, "status": "PASS", "result": result}

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "runner"
            runner = LocalRunner("http://127.0.0.1:13346", output, "mtp")
            with self.assertRaisesRegex(ValueError, "accounting"):
                self.run_initial(root, runner)
            self.assertTrue(all(row["status"] == "PASS" for row in runner.report["requests"]))
            runner.finish()
            persisted = json.loads((output / "report.json").read_text())
            self.assertEqual(persisted["status"], "FAIL")
            self.assertIn("accounting", persisted["error"])

    def test_calibration_overflow_stops_without_unsafe_followup(self):
        def damage(identifier, result):
            if identifier == "family-calibrate-1":
                result["metrics"].update(prompt_tokens=130560, cached_tokens=0, prefill_tokens=130560)
        with tempfile.TemporaryDirectory() as root:
            runner = RecordingRunner(damage)
            with self.assertRaises(ValueError):
                self.run_initial(root, runner)
            self.assertEqual([call[0] for call in runner.calls if "calibrate" in call[0]],
                             ["family-calibrate-0", "family-calibrate-1"])
            events = Path(root) / "family-state.initial.events.jsonl"
            self.assertEqual(json.loads(events.read_text().splitlines()[-1])["event"], "failed")

    def test_cancel_recovery_and_reasoning_server_budget_are_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            runner, _, _ = self.run_initial(root)
        calls = {call[0]: call for call in runner.calls}
        self.assertEqual(calls["family-cancel"][4], 1)
        self.assertTrue(calls["family-cancel"][2]["cancelled"])
        self.assertEqual(calls["family-recover"][2]["json"]["squares"][-1], 400)
        self.assertEqual(calls["family-cancel"][1], calls["family-recover"][1])
        on = calls["family-reasoning-on"]
        self.assertEqual(on[1]["path"], "/v1/responses")
        self.assertEqual(on[1]["body"]["reasoning"], {"effort": "low"})
        self.assertEqual(on[2]["json"], {"answer": 7})
        self.assertTrue(on[2]["reasoning"])
        self.assertEqual(on[2]["reasoning_budget"], 512)
        for field in ("reasoning_budget_tokens", "thinking_budget_tokens"):
            self.assertNotIn(field, on[1]["body"])
        self.assertGreater(on[1]["body"]["max_output_tokens"], 512)

    def test_restart_uses_exact_saved_request_and_requires_disk_and_reuse(self):
        with tempfile.TemporaryDirectory() as root:
            _, state, _ = self.run_initial(root)
            before = state.read_bytes()
            saved = json.loads(before)
            runner = RecordingRunner()
            result = self.api().run_family(runner, phase="restart", state_path=state)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(runner.calls[0][1], saved["request"])
            self.assertTrue(runner.calls[0][2]["disk_hit"])
            self.assertGreater(runner.calls[0][2]["min_cached_tokens"], 100000)
            self.assertEqual(state.read_bytes(), before)
            with self.assertRaises((ValueError, FileExistsError)):
                self.api().run_family(RecordingRunner(), phase="restart", state_path=state)

    def test_restart_refuses_tampered_state_before_http(self):
        with tempfile.TemporaryDirectory() as root:
            _, state, _ = self.run_initial(root)
            saved = json.loads(state.read_text())
            saved["request"]["body"]["messages"][-1]["content"] = "Другой запрос"
            state.write_text(json.dumps(saved))
            runner = RecordingRunner()
            with self.assertRaises(ValueError):
                self.api().run_family(runner, phase="restart", state_path=state)
            self.assertEqual(runner.calls, [])

    def test_restart_requires_persisted_cache_not_merely_correct_answer(self):
        with tempfile.TemporaryDirectory() as root:
            _, state, _ = self.run_initial(root)
            def damage(identifier, result):
                result["usage"]["gufo"]["cache_disk_hit"] = False
            with self.assertRaisesRegex(ValueError, "disk cache"):
                self.api().run_family(RecordingRunner(damage), phase="restart", state_path=state)

    def test_workflow_result_records_preservation_exception_for_parent_log_check(self):
        # Prefill 1..16 может быть допустим только с доказанным отказом полного snapshot.
        with tempfile.TemporaryDirectory() as root:
            _, state, result = self.run_initial(root)
            saved = json.loads(state.read_text())
            self.assertIn("cache_retry_log_checks", list(result))
            self.assertEqual(result["cache_retry_log_checks"], saved["cache_retry_log_checks"])
            self.assertEqual({item["id"] for item in saved["cache_retry_log_checks"]},
                             {"family-long-repeat", "family-long-edit-repeat", "family-snapshot"})
            self.assertTrue(all(item["prefill_tokens"] == 8 for item in saved["cache_retry_log_checks"]))
            self.assertTrue(all("capture_request_id" in item for item in saved["cache_retry_log_checks"]),
                            "server-log checks must identify the preceding capture request")
            self.assertEqual({item["id"]: item["capture_request_id"]
                              for item in saved["cache_retry_log_checks"]},
                             {"family-long-repeat": "family-long-cold",
                              "family-long-edit-repeat": "family-long-edit",
                              "family-snapshot": "family-long-edit-repeat"})


if __name__ == "__main__":
    unittest.main()
