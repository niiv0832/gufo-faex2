"""CPU-контракты C: сопоставимые истории, отдельные фазы и честные сроки soak."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from faex2_acceptance import grade


def conditions():
    return {"model_sha256": "1" * 64, "mmproj_sha256": "2" * 64,
            "harness_sha256": "3" * 64, "runtime_sha256": "4" * 64,
            "profile": {"mode": "mtp", "sessions": 4, "context": 131072,
                        "vision": True, "cache_ram_bytes": 16 * 1024**3,
                        "cache_disk_bytes": 64 * 1024**3, "sampler": {"temperature": 1}},
            "cache_start": "fresh-separated", "host_conditions": {"manifest_sha256": "5" * 64}}


class Clock:
    def __init__(self): self.now = 0
    def monotonic(self): return self.now
    def time(self): return 1800000000 + self.now


class RecordingRunner:
    """Заменяется только inference; grade проверяет реальные запросы/результаты."""
    mode = "mtp"

    def __init__(self, root, clock=None, damage=None):
        self.output, self.clock, self.damage = Path(root), clock, damage
        self.calls, self.batches = [], []
        self.report = {"status": "RUNNING", "requests": [], "mode": "mtp",
                       "comparison_conditions": conditions()}

    def save(self):
        (self.output / "report.json").write_text(json.dumps(self.report, ensure_ascii=False))

    def result(self, identifier, request):
        body = request["body"]
        contents = json.dumps(body.get("messages", body.get("input", [])), ensure_ascii=False)
        if "Архивная запись" in contents:
            answer = {"start": "ЛИПА_274", "middle": "РЕКА_583",
                      "end": "ЗВЕЗДА_408" if "ЗВЕЗДА_408" in contents else
                      "КЕДР_205" if "КЕДР_205" in contents else "СОВА_916"}
        elif "C1_MTP_742" in contents:
            answer = {"session": "C1_MTP_742", "squares": [1, 4, 9, 16, 25, 36, 49, 64, 81, 100,
                121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625,
                676, 729, 784, 841, 900, 961, 1024, 1089, 1156, 1225, 1296, 1369, 1444, 1521, 1600]}
        elif "image_url" in contents:
            answer = "red"
        elif "reasoning" in body:
            answer = {"answer": 7}
        elif "OWL_142" in contents:
            answer = {"session": "OWL_142", "products": [340, 357, 374, 391, 408, 425, 442, 459]}
        elif "MOON_853" in contents:
            answer = {"session": "MOON_853", "products": [460, 483, 506, 529, 552, 575, 598, 621]}
        elif "PINE_627" in contents:
            answer = {"session": "PINE_627", "products": [580, 609, 638, 667, 696, 725, 754, 783]}
        elif "LAKE_394" in contents:
            answer = {"session": "LAKE_394", "products": [620, 651, 682, 713, 744, 775, 806, 837]}
        else:
            answer = {"answer": 391}
        long = "Архивная запись" in contents
        prompt = 118040 if long and "Повтори" in contents else 118000 if long else 100
        cached = 0 if body.get("cache_prompt") is False else (
            117900 if long and "ЗВЕЗДА_408" in contents and "repeat" not in identifier else
            118000 if long else 90)
        thinking = body.get("reasoning", {}).get("effort") == "low"
        return {"status": "PASS", "http_status": 200,
                "text": answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False),
                "reasoning": "Проверяю решение" if thinking else "", "tools": [], "finish": "stop",
                "output_sha256": "a" * 64, "wall_ms": 100,
                "first_visible_content_ms": 40,
                "usage": {"output_tokens_details": {"reasoning_tokens": 20 if thinking else 0}},
                "metrics": {"prompt_tokens": prompt, "cached_tokens": cached,
                            "prefill_tokens": prompt - cached, "completion_tokens": 80,
                            "prefill_ms": 20 if prompt > cached else 0, "decode_ms": 50,
                            "ttft_ms": 30, "client_ttfa_ms": 40,
                            "draft_tokens": 40, "draft_tokens_accepted": 30}}

    def run_case(self, identifier, request, expected, timeout=120, cancel=None):
        self.calls.append((identifier, deepcopy(request), deepcopy(expected), timeout))
        result = self.result(identifier, request)
        if self.damage: self.damage(identifier, result)
        row = {"id": identifier, "status": "PASS", "expected": expected, "result": result,
               "started_at": self.clock.time() if self.clock else 1,
               "finished_at": self.clock.time() + 60 if self.clock else 2}
        self.report["requests"].append(row)
        if self.clock: self.clock.now += 60
        try: grade(result, expected, self.mode)
        except ValueError:
            row["status"] = self.report["status"] = "FAIL"
            self.save()
            raise
        self.save()
        return result

    def run_batch(self, cases, timeout=120):
        self.batches.append([identifier for identifier, _, _ in cases])
        return [self.run_case(*case, timeout=timeout) for case in cases]


class StabilityTests(unittest.TestCase):
    def api(self):
        path = Path(__file__).with_name("faex2_stability.py")
        self.assertTrue(path.is_file(), "C control/soak client is not implemented")
        spec = importlib.util.spec_from_file_location("faex2_stability", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def control(self, root, **kwargs):
        runner = RecordingRunner(root)
        self.api().run_control(runner, 3800, **kwargs)
        runner.report["status"] = "PASS"
        return runner

    def test_control_covers_exact_short_vision_concurrency_and_long_cache(self):
        with tempfile.TemporaryDirectory() as root:
            runner = self.control(root)
            cases = runner.calls[1:]
            self.assertEqual([len(batch) for batch in runner.batches], [2, 4])
            self.assertEqual([case[0] for case in cases[:3]],
                             ["control-short-json", "control-short-sse", "control-vision-red"])
            self.assertEqual([case[1]["body"]["stream"] for case in cases[:2]], [False, True])
            self.assertEqual(cases[0][2]["json"], {"answer": 391})
            for _, req, expected, _ in cases[3:9]:
                self.assertEqual(req["body"]["temperature"], 1)
                self.assertEqual(len(expected["excludes"]), 3)
            longs = cases[9:]
            self.assertEqual([case[0] for case in longs], ["control-long-cold", "control-long-repeat",
                "control-long-continue", "control-long-branch-warmup", "control-long-edit",
                "control-long-edit-repeat"])
            self.assertTrue(all(case[3] == 600 for case in longs))
            self.assertTrue(all(case[2]["min_prompt_tokens"] == 110000 for case in longs))
            self.assertFalse(longs[0][1]["body"]["cache_prompt"])
            self.assertTrue(all(case[2]["min_cached_tokens"] >= 117744 for case in longs[1:]
                                if case[0] != "control-long-branch-warmup"))
            self.assertEqual(longs[-2][2]["json"]["end"], "ЗВЕЗДА_408")

    def test_control_edit_prepares_shared_checkpoint_and_compares_warmup_cost(self):
        from test_faex2_family import SharedBranchCache
        cache = SharedBranchCache()
        with tempfile.TemporaryDirectory() as root:
            runner = RecordingRunner(root, damage=lambda _, result: cache.apply(runner.calls[-1][1], result))
            try:
                self.api().run_control(runner, 3800, ["long_cache"])
            except ValueError as error:
                self.fail("control edit lacks a prepared recurrent checkpoint: " + str(error))
            runner.report["status"] = "PASS"
            comparison = self.api().compare_reports(runner.report, deepcopy(runner.report))
            warm = next(row for row in runner.report["requests"] if row["id"] == "control-long-branch-warmup")
            metrics = warm["result"]["metrics"]
            self.assertEqual(metrics["cached_tokens"], 0)
            self.assertEqual(metrics["prefill_tokens"], metrics["prompt_tokens"])
            self.assertEqual(warm["expected"]["json"]["end"], "КЕДР_205")
            self.assertEqual(len([row for row in comparison["measurements"]
                                  if row["case"] == warm["id"]]), 5)
            edit = next(row for row in runner.report["requests"] if row["id"] == "control-long-edit")
            self.assertGreater(edit["result"]["metrics"]["cached_tokens"], 110000)
            self.assertEqual(edit["expected"]["max_prefill_tokens"], 256)

    def test_control_prepared_branch_does_not_allow_zero_reuse_on_measured_edit(self):
        def damage(identifier, result):
            if identifier == "control-long-edit":
                total = result["metrics"]["prompt_tokens"]
                result["metrics"].update(cached_tokens=0, prefill_tokens=total)
        with tempfile.TemporaryDirectory() as root:
            runner = RecordingRunner(root, damage=damage)
            with self.assertRaisesRegex(ValueError, "insufficient cached_tokens"):
                self.api().run_control(runner, 3800, ["long_cache"])
            self.assertEqual(runner.report["control"]["status"], "FAIL")
            self.assertIn("control-long-branch-warmup", [call[0] for call in runner.calls])

    def test_every_filtered_run_begins_with_a_bounded_sampled_c1_gate(self):
        for groups in (None, ["short"], ["vision"], ["batch"], ["long_cache"]):
            with self.subTest(groups=groups), tempfile.TemporaryDirectory() as root:
                runner = self.control(root, groups=groups)
                identifier, request, expected, timeout = runner.calls[0]
                self.assertEqual(identifier, "control-mtp-c1")
                self.assertEqual(runner.report["control"]["cases"][0]["group"], "mtp_c1")
                self.assertEqual(runner.report["control"]["groups"][0], "mtp_c1")
                self.assertTrue(expected["draft_work"])
                self.assertEqual(expected["json"]["session"], "C1_MTP_742")
                self.assertEqual(len(expected["json"]["squares"]), 40)
                self.assertEqual(expected["json"]["squares"][-1], 1600)
                self.assertEqual(expected["json"]["squares"], [number * number for number in range(1, 41)])
                self.assertEqual(request["body"]["temperature"], 1)
                self.assertEqual(request["body"]["max_completion_tokens"], 384)
                self.assertEqual(timeout, 120)
                self.assertFalse(request["body"]["cache_prompt"])
                self.assertNotIn("enum", json.dumps(request["body"]["response_format"]))

    def test_zero_proposed_or_zero_accepted_stops_control_at_c1_gate(self):
        for proposed, accepted in [(0, 0), (10, 0)]:
            def damage(_identifier, result):
                result["metrics"].update(draft_tokens=proposed, draft_tokens_accepted=accepted)
            with self.subTest(proposed=proposed), tempfile.TemporaryDirectory() as root:
                runner = RecordingRunner(root, damage=damage)
                with self.assertRaises(ValueError): self.api().run_control(runner, 3800, ["short"])
                self.assertEqual(len(runner.calls), 1)
                self.assertEqual(runner.report["control"]["status"], "FAIL")

    def test_comparison_rechecks_mtp_gate_but_does_not_require_equal_positive_drafts(self):
        with tempfile.TemporaryDirectory() as root:
            base = self.control(root, groups=["short"]).report
        changed = deepcopy(base)
        changed["requests"][0]["result"]["metrics"].update(draft_tokens=0, draft_tokens_accepted=0)
        with self.assertRaises(ValueError): self.api().compare_reports(base, changed)
        changed["requests"][0]["result"]["metrics"].update(draft_tokens=70, draft_tokens_accepted=60)
        self.assertEqual(self.api().compare_reports(base, changed)["status"], "PASS")

    def test_c2_c4_may_gate_drafts_after_c1_proved_mtp(self):
        def damage(identifier, result):
            if identifier.startswith("control-batch"):
                result["metrics"].update(draft_tokens=0, draft_tokens_accepted=0)
        with tempfile.TemporaryDirectory() as root:
            runner = RecordingRunner(root, damage=damage)
            self.api().run_control(runner, 3800, ["batch"])
            self.assertEqual(runner.report["control"]["status"], "PASS")

    def test_comparison_rejects_legacy_control_without_explicit_c1_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            report = self.control(root, groups=["short"]).report
        report["control"]["version"] = 1
        with self.assertRaises(ValueError): self.api().compare_reports(report, deepcopy(report))

    def test_group_repeat_keeps_full_history_and_original_inputs(self):
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            original = self.control(left)
            repeated = self.control(right, groups=["long_cache"])
            self.assertEqual(original.calls[0], repeated.calls[0])
            self.assertEqual(original.calls[10:], repeated.calls[1:])
            first = [case for case in original.report["control"]["cases"] if case["group"] == "long_cache"]
            self.assertEqual(first, repeated.report["control"]["cases"][1:])

    def test_control_rejects_missing_conditions_unknown_groups_and_bad_depth(self):
        for records, groups, delete in [(3800, ["arbitrary"], False), (7, None, False), (3800, None, True)]:
            with tempfile.TemporaryDirectory() as root:
                runner = RecordingRunner(root)
                if delete: del runner.report["comparison_conditions"]["model_sha256"]
                with self.assertRaises(ValueError): self.api().run_control(runner, records, groups)
                self.assertEqual(runner.calls, [])

    def test_bad_cache_work_stops_control_and_preserves_failure(self):
        def damage(identifier, result):
            if identifier == "control-long-repeat": result["metrics"]["cached_tokens"] = 0
        with tempfile.TemporaryDirectory() as root:
            runner = RecordingRunner(root, damage=damage)
            with self.assertRaises(ValueError): self.api().run_control(runner, 3800, ["long_cache"])
            self.assertEqual(len(runner.calls), 3)
            saved = json.loads((Path(root) / "report.json").read_text())
            self.assertEqual(saved["control"]["status"], "FAIL")
            self.assertEqual(saved["requests"][-1]["status"], "FAIL")

    def test_comparison_flags_each_request_phase_without_averaging(self):
        with tempfile.TemporaryDirectory() as root:
            original = self.control(root, groups=["short"]).report
        changed = deepcopy(original)
        changed["requests"][1]["result"]["metrics"]["decode_ms"] = 54
        changed["requests"][2]["result"]["metrics"]["decode_ms"] = 1
        report = self.api().compare_reports(original, changed)
        self.assertEqual(report["status"], "NEEDS_INVESTIGATION")
        flags = [row for row in report["measurements"] if row["status"] == "NEEDS_INVESTIGATION"]
        self.assertEqual([(row["case"], row["metric"]) for row in flags],
                         [("control-short-json", "decode_ms")])
        self.assertEqual(len(report["measurements"]), 15)

    def admission(self, report, order, ttft=(665, 665, 754, 874)):
        """Задать порядок допуска C4 и серверный TTFT, определяемый этим порядком."""
        rows = {row["id"].rsplit("-", 1)[-1]: row for row in report["requests"] if "-c4-" in row["id"]}
        self.assertEqual(sorted(rows), sorted(order))
        for rank, marker in enumerate(order, 1):
            metrics = rows[marker]["result"]["metrics"]
            metrics["resident_requests_at_admission"] = rank
            metrics["ttft_ms"] = ttft[rank - 1]
            metrics["client_ttfa_ms"] = ttft[rank - 1] + 12

    def test_concurrent_batch_is_compared_by_admission_rank_not_by_racy_name(self):
        """Регрессия C timing: перестановка допуска не является замедлением запроса.

        В реальных прогонах OWL был допущен первым в одном прогоне и четвёртым в другом;
        сравнение по имени дало +30 % TTFT при совпадении времён каждого места до 0,7 %.
        """
        with tempfile.TemporaryDirectory() as root:
            base = self.control(root, groups=["batch"]).report
        left, right = deepcopy(base), deepcopy(base)
        self.admission(left, ["OWL_142", "MOON_853", "PINE_627", "LAKE_394"])
        self.admission(right, ["PINE_627", "MOON_853", "LAKE_394", "OWL_142"], ttft=(663, 663, 746, 865))
        report = self.api().compare_reports(left, right)
        by_name = [row for row in report["measurements"] if row["case"].endswith("-c4-OWL_142")
                   and row["metric"] in ("ttft_ms", "client_ttfa_ms")]
        self.assertEqual({row["status"] for row in by_name}, {"UNMATCHED_ADMISSION"})
        self.assertEqual({row["baseline_admission_rank"] for row in by_name}, {1})
        self.assertEqual({row["candidate_admission_rank"] for row in by_name}, {4})
        ranked = [row for row in report["measurements"] if "#admission-" in row["case"]]
        self.assertEqual(sorted({row["case"] for row in ranked}),
                         [f"control-batch-c4#admission-{rank}" for rank in (1, 2, 3, 4)])
        self.assertEqual({row["status"] for row in ranked}, {"PASS"})
        self.assertEqual(report["status"], "PASS")
        # Работа запроса не зависит от места допуска и по-прежнему сравнивается по имени.
        self.assertTrue(all(row["status"] == "PASS" for row in report["measurements"]
                            if row["case"].endswith("-c4-OWL_142") and row["metric"] in ("prefill_ms", "decode_ms")))

    def test_slow_admission_rank_is_still_flagged_after_permutation(self):
        with tempfile.TemporaryDirectory() as root:
            base = self.control(root, groups=["batch"]).report
        left, right = deepcopy(base), deepcopy(base)
        self.admission(left, ["OWL_142", "MOON_853", "PINE_627", "LAKE_394"])
        self.admission(right, ["PINE_627", "MOON_853", "LAKE_394", "OWL_142"], ttft=(663, 663, 746, 930))
        report = self.api().compare_reports(left, right)
        flags = [(row["case"], row["metric"]) for row in report["measurements"]
                 if row["status"] == "NEEDS_INVESTIGATION"]
        self.assertEqual(flags, [("control-batch-c4#admission-4", "ttft_ms"),
                                 ("control-batch-c4#admission-4", "client_ttfa_ms")])
        self.assertEqual(report["status"], "NEEDS_INVESTIGATION")

    def test_same_admission_order_keeps_strict_per_request_comparison(self):
        with tempfile.TemporaryDirectory() as root:
            base = self.control(root, groups=["batch"]).report
        left, right = deepcopy(base), deepcopy(base)
        order = ["OWL_142", "MOON_853", "PINE_627", "LAKE_394"]
        self.admission(left, order)
        self.admission(right, order, ttft=(665, 665, 754, 930))
        report = self.api().compare_reports(left, right)
        flags = [(row["case"], row["metric"]) for row in report["measurements"]
                 if row["status"] == "NEEDS_INVESTIGATION"]
        self.assertEqual(flags, [("control-batch-c4-LAKE_394", "ttft_ms"),
                                 ("control-batch-c4-LAKE_394", "client_ttfa_ms")])
        self.assertFalse([row for row in report["measurements"] if "#admission-" in row["case"]])

    def test_ambiguous_admission_ranks_cannot_qualify_a_permuted_batch(self):
        with tempfile.TemporaryDirectory() as root:
            base = self.control(root, groups=["batch"]).report
        left, right = deepcopy(base), deepcopy(base)
        self.admission(left, ["OWL_142", "MOON_853", "PINE_627", "LAKE_394"])
        self.admission(right, ["PINE_627", "MOON_853", "LAKE_394", "OWL_142"])
        for row in right["requests"]:
            if row["id"].endswith("-c4-LAKE_394"):
                row["result"]["metrics"]["resident_requests_at_admission"] = 1  # дубликат места
        report = self.api().compare_reports(left, right)
        self.assertEqual(report["status"], "NEEDS_INVESTIGATION")
        self.assertFalse([row for row in report["measurements"] if "#admission-" in row["case"]])

    def test_comparison_requires_both_five_percent_and_three_ms(self):
        with tempfile.TemporaryDirectory() as root:
            base = self.control(root, groups=["short"]).report
        for baseline, candidate, expected in [(10, 13, "PASS"), (100, 105, "PASS"),
                                               (10, 13.1, "NEEDS_INVESTIGATION")]:
            left, right = deepcopy(base), deepcopy(base)
            for report, value in [(left, baseline), (right, candidate)]:
                report["requests"][1]["result"]["metrics"]["prefill_ms"] = value
            self.assertEqual(self.api().compare_reports(left, right)["status"], expected)

    def test_comparison_rejects_unmatched_inputs_conditions_or_history(self):
        with tempfile.TemporaryDirectory() as root:
            base = self.control(root, groups=["short"]).report
        for field in ["conditions", "input", "history", "quality"]:
            changed = deepcopy(base)
            if field == "conditions": changed["comparison_conditions"]["harness_sha256"] = "9" * 64
            elif field == "input": changed["control"]["cases"][0]["request_sha256"] = "9" * 64
            elif field == "history": changed["control"]["cases"][0]["history_sha256"] = "9" * 64
            else: changed["requests"][0]["status"] = "FAIL"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.api().compare_reports(base, changed)

    def test_missing_timings_are_na_only_for_zero_work(self):
        with tempfile.TemporaryDirectory() as root:
            base = self.control(root, groups=["long_cache"]).report
        left, right = deepcopy(base), deepcopy(base)
        for report in (left, right):
            del report["requests"][2]["result"]["metrics"]["prefill_ms"]
        compared = self.api().compare_reports(left, right)
        na = [row for row in compared["measurements"] if row["status"] == "N/A"]
        self.assertEqual([(row["case"], row["metric"]) for row in na], [("control-long-repeat", "prefill_ms")])
        del right["requests"][1]["result"]["metrics"]["decode_ms"]
        with self.assertRaises(ValueError): self.api().compare_reports(left, right)

    def test_mismatched_token_work_cannot_produce_qualified_timings(self):
        with tempfile.TemporaryDirectory() as root:
            base = self.control(root, groups=["short"]).report
        changed = deepcopy(base); changed["requests"][1]["result"]["metrics"]["completion_tokens"] = 81
        result = self.api().compare_reports(base, changed)
        self.assertEqual(result["status"], "NEEDS_INVESTIGATION")
        self.assertTrue(result["work_changes"])
        self.assertTrue(all(row["status"] == "UNMATCHED_WORK" for row in result["measurements"]
                            if row["case"] == "control-short-json"))

    def test_na_does_not_hide_invalid_metric_on_the_other_side(self):
        with tempfile.TemporaryDirectory() as root:
            base = self.control(root, groups=["long_cache"]).report
        left, right = deepcopy(base), deepcopy(base)
        del left["requests"][2]["result"]["metrics"]["prefill_ms"]
        right["requests"][2]["result"]["metrics"]["prefill_ms"] = "invalid"
        with self.assertRaises(ValueError): self.api().compare_reports(left, right)

    def test_interrupt_is_persisted_as_failed_control_or_soak(self):
        def interrupt(_identifier, _result): raise KeyboardInterrupt()
        for mode in ("control", "soak"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                runner = RecordingRunner(root, Clock(), interrupt)
                with self.assertRaises(KeyboardInterrupt):
                    if mode == "control": self.api().run_control(runner, 3800, ["short"])
                    else: self.api().run_soak(runner, clock=runner.clock)
                saved = json.loads((Path(root) / "report.json").read_text())
                self.assertEqual(saved["status"], "FAIL")
                self.assertEqual(saved["control" if mode == "control" else "stability"]["status"], "FAIL")

    def test_soak_cannot_shorten_warmup_or_measurement(self):
        for warm, duration in [(599, 1800), (600, 1799)]:
            with tempfile.TemporaryDirectory() as root:
                runner = RecordingRunner(root, Clock())
                with self.assertRaises(ValueError):
                    self.api().run_soak(runner, warm, duration, clock=runner.clock)
                self.assertEqual(runner.calls, [])

    def test_soak_keeps_fixed_cache_histories_and_full_phase_durations(self):
        with tempfile.TemporaryDirectory() as root:
            clock = Clock(); runner = RecordingRunner(root, clock)
            self.api().run_soak(runner, clock=clock)
            record = runner.report["stability"]
            self.assertEqual(record["status"], "WORKLOAD_PASS_REQUIRES_MONITOR_REVIEW")
            self.assertGreaterEqual(record["warmup"]["elapsed_s"], 600)
            self.assertGreaterEqual(record["measurement"]["elapsed_s"], 1800)
            self.assertGreaterEqual(record["measurement"]["started_at"], record["warmup"]["finished_at"])
            widths = [len(batch) for batch in runner.batches]
            self.assertTrue(all(width in (1, 2, 4) for width in widths))
            self.assertEqual(widths[:3], [1, 2, 4])
            marked = [call for call in runner.calls if "OWL_142" in json.dumps(call[1])]
            self.assertTrue(all(call[1] == marked[0][1] for call in marked))
            self.assertTrue(all(call[1]["body"]["cache_prompt"] for call in marked))
            self.assertEqual(len([call for call in runner.calls if "reasoning" in call[1]["body"]]), 2)
            self.assertEqual(len([call for call in runner.calls if "image_url" in json.dumps(call[1])]), 1)
            self.assertEqual(len(set(call[0] for call in runner.calls)), len(runner.calls))

    def test_soak_stops_on_wrong_answer_retaining_failure_and_phase(self):
        def damage(identifier, result):
            if "c2" in identifier: result["text"] = '{"session":"foreign","products":[]}'
        with tempfile.TemporaryDirectory() as root:
            clock = Clock(); runner = RecordingRunner(root, clock, damage)
            with self.assertRaises(ValueError): self.api().run_soak(runner, clock=clock)
            self.assertEqual(runner.report["stability"]["status"], "FAIL")
            self.assertEqual(runner.report["requests"][-1]["status"], "FAIL")
            self.assertNotIn("measurement", runner.report["stability"])


if __name__ == "__main__": unittest.main()
