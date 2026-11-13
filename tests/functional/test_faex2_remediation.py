"""CPU-проверки клиента приёмки доработок: настоящий loopback, подставной сервер, без модели.

Каждый gate проверен в обе стороны: на исправном двойнике проходит, на двойнике с
одним включённым дефектом отказывает либо остаётся NOT_EXERCISED.
"""

import base64
import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import unittest.mock
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

import faex2_remediation as remediation
import faex2_remediation_images as images
import faex2_remediation_server as double
import faex2_transport as transport


def measured_never(*_):
    raise AssertionError("memory must not be sampled without --pid")


class Bench:
    """Двойник сервера с журналом и каталогом кэша; restart() сохраняет только диск."""
    def __init__(self, root, defects=(), pace=1.0):
        self.root, self.defects, self.pace = Path(root), defects, pace
        self.cache = self.root / "cache"
        self.cache.mkdir(exist_ok=True)
        self.runs = 0
        self.fake = None
        self.restart()

    def restart(self, defects=None):
        if self.fake:
            self.fake.stop()
        self.runs += 1
        self.log = self.root / f"server-{self.runs}.log"
        self.log.touch()
        self.fake = double.Fake(self.cache, self.log, self.defects if defects is None else defects,
                                pace=self.pace)
        self.url = f"http://127.0.0.1:{self.fake.start()}"

    def check(self, only=None, *, log=True, phase="before", restore=None, extra=()):
        output = self.root / f"out-{self.runs}-{phase}-{len(list(self.root.glob('out-*')))}"
        argv = ["check", "--base-url", self.url, "--output", str(output), "--phase", phase,
                "--nonce", f"t{self.runs}", "--disk-settle", "0.2", *extra]
        if only:
            argv += ["--only", only]
        if log:
            argv += ["--server-log", str(self.log)]
        if restore:
            argv += ["--restore", str(restore)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = remediation.main(argv)
        report = json.loads((output / "report.json").read_text())
        return code, report, output


class GateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.benches = []

    def bench(self, defects=(), pace=1.0):
        bench = Bench(tempfile.mkdtemp(dir=self.directory.name), defects, pace)
        self.benches.append(bench)
        self.addCleanup(lambda: bench.fake.stop())
        return bench

    def statuses(self, report):
        return {name: row["status"] for name, row in report["gates"].items()}

    def assert_all(self, report, expected="PASS"):
        wrong = {name: (row["status"], row.get("error") or row.get("reason"))
                 for name, row in report["gates"].items() if row["status"] != expected}
        self.assertEqual(wrong, {})

    # --- исправный сервер ---------------------------------------------------
    def test_zero_prefill_reasoning_images_tools_pass_on_a_healthy_server(self):
        code, report, output = self.bench(pace=.3).check("zero_prefill,reasoning,images,images_complex,tools")
        self.assert_all(report)
        self.assertEqual((code, report["status"]), (0, "PASS"))
        exact = report["gates"]["zero_prefill_ram_text"]["evidence"]
        self.assertEqual(exact["repeat"]["prefill"], 0)
        self.assertEqual(exact["repeat"]["cached"], exact["repeat"]["prompt"])
        self.assertEqual(exact["server_log"]["cache"], "memory")
        self.assertEqual(report["gates"]["reasoning_512_sampled_sse"]["evidence"]["reasoning_tokens"], 512)
        self.assertEqual(report["gates"]["reasoning_512_sampled_sse_answer"]["evidence"]["text"], "16")
        wide = report["gates"]["reasoning_budget_isolation"]["evidence"]
        self.assertEqual((wide["with_field_2048"]["reasoning_tokens"], wide["neighbour"]["reasoning_tokens"]),
                         (900, 512))
        # Запрос и сырой ответ каждого обмена сохранены до проверки.
        for row in report["requests"]:
            self.assertTrue((output / row["request_file"]).is_file())
            self.assertTrue((output / row["result_file"]).with_name("response.raw").is_file())

    def test_drafts_and_sessions_pass_with_real_counters_and_width(self):
        code, report, _ = self.bench().check("drafts,sessions")
        self.assert_all(report)
        self.assertEqual(code, 0)
        cohort = report["gates"]["drafts_c4_sampled"]["evidence"]["requests"]
        self.assertEqual(len(cohort), 4)
        self.assertTrue(all(row["draft_tokens_accepted"] > 0 for row in cohort.values()))
        self.assertEqual(max(row["physical_execution_width"] for row in cohort.values()), 4)

    def test_late_shared_prefix_passes_only_when_the_race_is_exercised(self):
        code, report, _ = self.bench(pace=2).check("late_prefix")
        gate = report["gates"]["late_shared_prefix"]
        self.assertEqual((gate["status"], code), ("PASS", 0), gate)
        evidence = gate["evidence"]
        self.assertGreaterEqual(evidence["control_without_follower"]["edited"]["cached"], 2048)
        attempt = evidence["attempts"][-1]
        self.assertTrue(attempt["exercised"])
        self.assertGreaterEqual(attempt["leader"]["prompt"], 3500)
        self.assertEqual(attempt["leader"]["cached"], 0)
        self.assertEqual(attempt["edited"]["cached"], 2048)
        self.assertEqual(attempt["follower"]["cached"] + attempt["follower"]["prefill"],
                         attempt["follower"]["prompt"])
        self.assertGreater(attempt["timing"]["follower_admitted_before_grid_s"], 0)
        self.assertEqual(report["gates"]["late_shared_prefix_answers"]["status"], "PASS")
        self.assertEqual(set(attempt["answers"]), {"leader", "follower", "edited"})

    def test_malformed_follower_answer_does_not_stop_the_checkpoint_scenario(self):
        # Первое окно: последователь ответил блоком кода и оборвался по лимиту — сценарий не дошёл
        # до редактированного запроса. Теперь формат ответа — отдельный gate.
        code, report, _ = self.bench(["follower_code_block"], pace=2).check("late_prefix")
        gate = report["gates"]["late_shared_prefix"]
        self.assertEqual(gate["status"], "PASS", gate)
        attempt = gate["evidence"]["attempts"][-1]
        self.assertEqual(attempt["edited"]["cached"], 2048)
        self.assertFalse(attempt["answers"]["follower"]["ok"])
        answers = report["gates"]["late_shared_prefix_answers"]
        self.assertEqual((answers["status"], code), ("FAIL", 1))
        self.assertIn("late-1-follower", answers["error"])

    def test_late_shared_prefix_without_the_race_is_not_exercised(self):
        # Чтение лидера короче запуска последователя: условие гонки не возникает.
        code, report, _ = self.bench(pace=.01).check("late_prefix", extra=["--late-attempts", "2"])
        gate = report["gates"]["late_shared_prefix"]
        self.assertEqual((gate["status"], report["status"], code), ("NOT_EXERCISED", "INCOMPLETE", 2), gate)
        self.assertEqual(len(gate["evidence"]["attempts"]), 2)

    def test_disk_restore_through_restart(self):
        bench = self.bench(pace=.2)
        code, before, output = bench.check("disk")
        self.assert_all(before)
        bench.restart()
        code, after, _ = bench.check(phase="after", restore=output)
        self.assert_all(after)
        self.assertEqual(code, 0)
        raw = after["gates"]["zero_prefill_disk_raw_completion"]["evidence"]
        self.assertEqual((raw["prefill"], raw["cached"]), (0, raw["prompt"]))
        self.assertIs(raw["cache_disk_hit"], True)
        # Запрос чата после перезапуска — иное свидетельство: остаток по устройству сервера.
        for name in ("text", "image", "continuation"):
            evidence = after["gates"]["disk_restore_boundary_" + name]["evidence"]
            self.assertEqual(evidence["observed"], "RESTORE_WITH_REMAINDER_BY_DESIGN")
            self.assertGreater(evidence["prefill"], 0)
            self.assertGreater(evidence["prompt"], 2048 + 256)   # Длиннее шага дисковых checkpoint.
            self.assertEqual(evidence["cached"],
                             max(value for value in evidence["stored_tokens"] if value <= evidence["prompt"]))
            self.assertEqual(evidence["server_log"]["cache"], "disk")

    def test_after_phase_without_prepared_state_is_not_exercised(self):
        bench = self.bench()
        state = bench.root / "empty"
        state.mkdir()
        (state / "restart-state.json").write_text(json.dumps({"cases": {}}))
        code, report, _ = bench.check(phase="after", restore=state)
        statuses = self.statuses(report)
        self.assertEqual(statuses.pop("disk_restore_state"), "PASS")
        self.assertEqual(set(statuses.values()), {"NOT_EXERCISED"})
        self.assertEqual((code, report["status"]), (2, "INCOMPLETE"))

    # --- дефекты: каждый gate обязан отказать -------------------------------
    def test_restore_with_a_remainder_is_not_a_zero_prefill_pass(self):
        code, report, _ = self.bench(["ram_remainder"]).check("zero_prefill")
        self.assertEqual(code, 1)
        for name in ("zero_prefill_ram_text", "zero_prefill_ram_image", "zero_prefill_ram_continuation"):
            gate = report["gates"][name]
            self.assertEqual(gate["status"], "FAIL", gate)
        evidence = report["gates"]["zero_prefill_ram_text"]["evidence"]
        self.assertEqual(evidence["observed"], "RESTORE_WITH_REMAINDER")
        self.assertEqual(evidence["repeat"]["prefill"], 2)

    def test_incomplete_token_accounting_fails(self):
        _, report, _ = self.bench(["accounting"]).check("zero_prefill")
        self.assertEqual(report["gates"]["zero_prefill_ram_text"]["status"], "FAIL")
        self.assertIn("cached + prefill", report["gates"]["zero_prefill_ram_text"]["error"])

    def test_server_log_disagreeing_with_usage_fails(self):
        bench = self.bench()
        original = bench.fake.log
        bench.fake.log = lambda line: original(line.replace("cache=memory", "cache=miss"))
        _, report, _ = bench.check("zero_prefill")
        self.assertEqual(report["gates"]["zero_prefill_ram_text"]["status"], "FAIL")
        self.assertIn("server log cache=miss", report["gates"]["zero_prefill_ram_text"]["error"])

    def test_disk_restore_without_disk_evidence_fails(self):
        bench = self.bench(pace=.2)
        _, _, output = bench.check("disk")
        bench.restart(["no_disk_flag"])
        code, after, _ = bench.check(phase="after", restore=output)
        self.assertEqual(code, 1)
        statuses = self.statuses(after)
        self.assertEqual(statuses.pop("disk_restore_state"), "PASS")
        self.assertEqual(set(statuses.values()), {"FAIL"})
        self.assertIn("disk-hit evidence", after["gates"]["zero_prefill_disk_raw_completion"]["error"])

    def test_disk_restore_after_cache_loss_fails(self):
        bench = self.bench(pace=.2)
        _, _, output = bench.check("disk")
        for path in bench.cache.glob("*.snapshot"):
            path.unlink()
        bench.restart()
        _, after, _ = bench.check(phase="after", restore=output)
        statuses = self.statuses(after)
        self.assertEqual(statuses.pop("disk_restore_state"), "PASS")
        self.assertEqual(set(statuses.values()), {"FAIL"})

    def test_reasoning_boundary_defects(self):
        for defect, status, text in (("reasoning_over", "FAIL", "exceeded"),
                                     ("reasoning_short", "NOT_EXERCISED", "before the 512-token budget"),
                                     ("empty_after_reasoning", "FAIL", "empty visible answer")):
            with self.subTest(defect):
                code, report, _ = self.bench([defect], pace=.3).check("reasoning")
                gate = report["gates"]["reasoning_512_greedy_sse"]
                self.assertEqual(gate["status"], status, gate)
                self.assertIn(text, gate.get("error") or gate.get("reason"))
                self.assertNotEqual(code, 0)

    def test_wrong_answer_after_cut_reasoning_is_a_model_limit_not_a_boundary_failure(self):
        code, report, _ = self.bench(["reasoning_wrong_answer"], pace=.3).check("reasoning")
        statuses = self.statuses(report)
        for name in ("reasoning_512_greedy_json", "reasoning_512_sampled_sse", "reasoning_budget_isolation"):
            self.assertEqual(statuses[name], "PASS")          # Граница соблюдена…
        for name in ("reasoning_512_greedy_json_answer", "reasoning_512_sampled_sse_answer"):
            self.assertEqual(statuses[name], "MODEL_LIMIT")   # …а неверный ответ не смешан с ней и не PASS.
        self.assertEqual((code, report["status"]), (2, "KNOWN_FAILURES"))
        # Рассуждение не обрезано, а ответ неверен: это уже не ограничение обрезки.
        _, report, _ = self.bench(["reasoning_wrong_answer", "reasoning_short"], pace=.3).check("reasoning")
        statuses = self.statuses(report)
        self.assertEqual((statuses["reasoning_512_greedy_sse"], statuses["reasoning_512_greedy_sse_answer"]),
                         ("NOT_EXERCISED", "FAIL"))
        # Запрос не завершился: ответ оценивать не по чему.
        _, report, _ = self.bench(["empty_after_reasoning"], pace=.3).check("reasoning")
        self.assertEqual(self.statuses(report)["reasoning_512_greedy_sse_answer"], "NOT_EXERCISED")

    def test_request_budget_leaking_into_a_neighbour_fails(self):
        _, report, _ = self.bench(["budget_leak"], pace=.3).check("reasoning")
        gate = report["gates"]["reasoning_budget_isolation"]
        # Сосед без поля получил чужой лимит.
        self.assertEqual(gate["status"], "FAIL", gate)
        self.assertIn("budget was exceeded", gate["error"])
        self.assertGreater(gate["evidence"]["after"]["reasoning_tokens"], 512)

    def test_draft_defects(self):
        for defect, expected in (("no_drafts", {"drafts_c1_sampled": "FAIL", "drafts_c4_greedy": "FAIL"}),
                                 ("missing_drafts", {"drafts_c1_greedy": "FAIL", "drafts_c2_sampled": "FAIL"}),
                                 ("width1", {"drafts_c1_sampled": "PASS", "drafts_c2_sampled": "NOT_EXERCISED",
                                             "drafts_c4_greedy": "NOT_EXERCISED"})):
            with self.subTest(defect):
                _, report, _ = self.bench([defect]).check("drafts")
                statuses = self.statuses(report)
                self.assertEqual({name: statuses[name] for name in expected}, expected)

    def test_foreign_session_marker_fails(self):
        _, report, _ = self.bench(["cross_talk"]).check("sessions")
        gate = report["gates"]["sessions_independent_c4"]
        self.assertEqual(gate["status"], "FAIL", gate)

    def test_cancellation_without_server_confirmation_is_not_exercised(self):
        for defects, log in (((), False), (["no_cancel_log"], True)):
            with self.subTest(defects=defects, log=log):
                _, report, _ = self.bench(defects, pace=.5).check("images,sessions", log=log)
                statuses = self.statuses(report)
                self.assertEqual((statuses["image_cancel_retry"], statuses["sessions_after_cancel"]),
                                 ("NOT_EXERCISED", "NOT_EXERCISED"))
                self.assertEqual(statuses["image_multiple"], "PASS")
                self.assertEqual(report["status"], "INCOMPLETE")

    def test_late_shared_prefix_defects(self):
        for defect, text in (("late_share_loses_grid", "did not reuse the 2048-token grid checkpoint"),
                             ("follower_accounting", "cached + prefill")):
            with self.subTest(defect):
                code, report, _ = self.bench([defect], pace=2).check("late_prefix")
                gate = report["gates"]["late_shared_prefix"]
                self.assertEqual((gate["status"], code), ("FAIL", 1), gate)
                self.assertIn(text, gate["error"])

    def test_complex_image_failure_is_inherited_only_with_the_same_control_answer(self):
        code, control, control_output = self.bench(["image_wrong"]).check("images_complex")
        self.assertEqual(set(self.statuses(control).values()), {"FAIL"})
        self.assertEqual(code, 1)
        code, report, _ = self.bench(["image_wrong"]).check(
            "images_complex", extra=["--control-report", str(control_output)])
        self.assertEqual(set(self.statuses(report).values()), {"INHERITED_FAIL"})
        self.assertEqual((code, report["status"]), (2, "KNOWN_FAILURES"))

    def test_wrong_tool_arguments_fail(self):
        _, report, _ = self.bench(["tool_args"]).check("tools")
        self.assertEqual(report["gates"]["tools_chat_sse"]["status"], "FAIL")
        self.assertEqual(report["gates"]["tools_responses_sse"]["status"], "PASS")

    def test_unreachable_server_fails_every_gate_without_raising(self):
        bench = self.bench()
        bench.fake.stop()
        code, report, _ = bench.check("zero_prefill,tools")
        self.assertEqual(set(self.statuses(report).values()), {"FAIL"})
        self.assertEqual(code, 1)

    # --- отчёт никогда не PASS без выполненных проверок ----------------------
    def test_missing_or_corrupt_restart_state_is_a_recorded_failure(self):
        for label, content in (("missing", None), ("corrupt", "{not json"), ("wrong-shape", "[]")):
            with self.subTest(label):
                bench = self.bench()
                state = bench.root / "state"
                state.mkdir()
                if content is not None:
                    (state / "restart-state.json").write_text(content)
                code, report, _ = bench.check(phase="after", restore=state)
                self.assertEqual((code, report["status"]), (1, "FAIL"), report.get("error"))
                self.assertEqual(report["gates"]["disk_restore_state"]["status"], "FAIL")
                self.assertNotIn("PASS", self.statuses(report).values())

    def test_exception_before_the_first_gate_is_a_recorded_failure(self):
        bench = self.bench()
        with unittest.mock.patch.object(remediation.Checks, "zero_prefill",
                                        side_effect=RuntimeError("boom before gates")):
            code, report, _ = bench.check("zero_prefill,tools")
        self.assertEqual((code, report["status"], report["gates"]), (1, "FAIL", {}))
        self.assertIn("boom before gates", report["error"])
        self.assertEqual(report["stages"], {"zero_prefill": "FAILED", "tools": "NOT_RUN"})

    def test_exception_after_partial_gates_is_a_recorded_failure(self):
        bench = self.bench()
        with unittest.mock.patch.object(remediation.Checks, "tools", side_effect=RuntimeError("boom later")):
            code, report, _ = bench.check("zero_prefill,tools")
        self.assertEqual(set(self.statuses(report).values()), {"PASS"})  # Выполненные gates прошли…
        self.assertEqual((code, report["status"]), (1, "FAIL"))           # …но отчёт не PASS.
        self.assertIn("boom later", report["error"])
        self.assertEqual(report["stages"], {"zero_prefill": "DONE", "tools": "FAILED"})

    def test_a_stage_that_ran_no_gate_is_not_a_pass(self):
        bench = self.bench()
        with unittest.mock.patch.object(remediation.Checks, "tools", return_value=None):
            code, report, _ = bench.check("tools")
        self.assertEqual((code, report["status"], report["gates"]), (1, "FAIL", {}))
        self.assertEqual(remediation.overall({}), "FAIL")

    def test_wrong_peer_products_during_cancellation_fail(self):
        # Сосед отвечает своим session без чужих маркеров, но с нулями вместо произведений.
        _, report, _ = self.bench(["zero_products_during_cancel"], pace=.5).check("sessions")
        self.assertEqual(report["gates"]["sessions_independent_c4"]["status"], "PASS")
        gate = report["gates"]["sessions_after_cancel"]
        self.assertEqual(gate["status"], "FAIL", gate)
        self.assertIn("peer answer is wrong during the cancellation", gate["error"])

    # --- журнал сервера обязателен там, где gate на него опирается ------------
    def test_gates_that_need_the_server_log_are_not_exercised_without_it(self):
        bench = self.bench(pace=.2)
        code, report, output = bench.check("zero_prefill,disk", log=False)
        statuses = self.statuses(report)
        self.assertEqual(set(statuses.values()), {"NOT_EXERCISED"}, statuses)
        self.assertEqual((code, report["status"]), (2, "INCOMPLETE"))
        evidence = report["gates"]["zero_prefill_ram_text"]["evidence"]
        self.assertEqual((evidence["repeat"]["prefill"], evidence["server_log"]), (0, "NOT_PROVIDED"))
        # Состояние для фазы after всё равно записано; сама она без журнала тоже не PASS.
        self.assertEqual(set(json.loads((output / "restart-state.json").read_text())["cases"]),
                         {"dkt", "dki", "dkc", "dkr"})
        bench.restart()
        code, after, _ = bench.check(phase="after", restore=output, log=False)
        statuses = self.statuses(after)
        self.assertEqual(statuses.pop("disk_restore_state"), "PASS")
        self.assertEqual(set(statuses.values()), {"NOT_EXERCISED"})
        self.assertEqual(code, 2)

    def test_a_real_failure_wins_over_the_missing_server_log(self):
        _, report, _ = self.bench(["ram_remainder"]).check("zero_prefill", log=False)
        self.assertEqual(report["gates"]["zero_prefill_ram_text"]["status"], "FAIL")

    # --- запись ответов контроля -----------------------------------------------
    def test_record_only_keeps_control_answers_without_failing(self):
        for defects, graded in (((), True), (["image_wrong"], False)):
            with self.subTest(defects=defects):
                code, report, output = self.bench(defects).check("images_complex", extra=["--record-only"])
                self.assertEqual(set(self.statuses(report).values()), {"RECORDED"})
                self.assertEqual((code, report["status"]), (0, "RECORDED"))
                gate = report["gates"]["image_complex_table_cell"]
                self.assertIs(gate["evidence"]["graded_correct"], graded)
                self.assertEqual(len(remediation.control_answers(output)), 5)
        # Кандидат: INHERITED_FAIL только при том же неверном ответе записанного контроля.
        _, _, wrong_control = self.bench(["image_wrong"]).check("images_complex", extra=["--record-only"])
        code, report, _ = self.bench(["image_wrong"]).check(
            "images_complex", extra=["--control-report", str(wrong_control)])
        self.assertEqual((set(self.statuses(report).values()), code), ({"INHERITED_FAIL"}, 2))
        _, _, right_control = self.bench().check("images_complex", extra=["--record-only"])
        code, report, _ = self.bench(["image_wrong"]).check(
            "images_complex", extra=["--control-report", str(right_control)])
        self.assertEqual((set(self.statuses(report).values()), code), ({"FAIL"}, 1))

    def test_record_only_is_limited_and_transport_failures_still_fail(self):
        bench = self.bench()
        with self.assertRaisesRegex(ValueError, "images_complex"):
            bench.check("tools", extra=["--record-only"])
        bench.fake.stop()
        code, report, _ = bench.check("images_complex", extra=["--record-only"])
        self.assertEqual((set(self.statuses(report).values()), code), ({"FAIL"}, 1))
        self.assertEqual(remediation.overall({"a": {"status": "RECORDED"}, "b": {"status": "PASS"}}), "FAIL")

    # --- профиль и композиция ------------------------------------------------
    def run_command(self, bench, name, *extra):
        output = bench.root / ("out-" + name)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = remediation.main([name, "--base-url", bench.url, "--output", str(output),
                                     "--nonce", "c1", *extra])
        return code, output

    def test_profile_records_every_phase_of_every_request(self):
        measured = {"status": "MEASURED", "rss_bytes": 40 << 30, "gtt_used_bytes": 45 << 30}
        with unittest.mock.patch.object(remediation, "memory_snapshot", return_value=measured):
            code, output = self.run_command(self.bench(pace=.3), "profile", "--pid", "4242")
        profile = json.loads((output / "profile.json").read_text())
        self.assertEqual((code, profile["status"], profile["memory_status"]), (0, "RECORDED", "MEASURED"))
        self.assertEqual([mark["stage"] for mark in profile["memory"]],
                         ["ready", "after_single_session", "after_c1", "after_c2", "after_c4"])
        self.assertEqual(len(profile["rows"]), 12)
        for row in profile["rows"]:
            self.assertEqual(set(row["phases"]), set(remediation.PHASES))
            self.assertIsNotNone(row["phases"]["wall_ms"])
        repeat = next(row for row in profile["rows"] if row["case"] == "profile-repeat-4k")
        self.assertEqual(repeat["work"]["prefill_tokens"], 0)

    def test_profile_without_memory_is_marked_and_not_a_clean_record(self):
        unsupported = {"status": "UNSUPPORTED_PLATFORM", "rss_bytes": None, "gtt_used_bytes": None}
        for label, extra, patch, expected in (
                ("no pid", (), measured_never, "MISSING_NO_PID"),
                ("no metrics", ("--pid", "4242"), lambda *_: unsupported, "MISSING_METRICS")):
            with self.subTest(label), unittest.mock.patch.object(remediation, "memory_snapshot", patch):
                bench = self.bench(pace=.3)
                bench.root = bench.root / label.replace(" ", "-")
                bench.root.mkdir()
                code, output = self.run_command(bench, "profile", *extra)
                profile = json.loads((output / "profile.json").read_text())
                self.assertEqual((code, profile["status"], profile["memory_status"]),
                                 (2, "RECORDED_MEMORY_MISSING", expected))
                self.assertEqual(len(profile["rows"]), 12)   # Времена при этом записаны.

    def compose(self, defects=(), log=True, extra=()):
        bench = self.bench(defects, pace=.05)
        argv = ["--duration", "1", "--allow-short", "--records", "2136", *extra]
        if log:
            argv += ["--server-log", str(bench.log)]
        code, output = self.run_command(bench, "compose", *argv)
        return code, json.loads((output / "composition.json").read_text())

    def test_composition_fills_four_long_sessions_and_mixes_operations(self):
        bench = self.bench(pace=.05)
        code, output = self.run_command(bench, "compose", "--duration", "1", "--allow-short",
                                        "--server-log", str(bench.log))
        state = json.loads((output / "composition.json").read_text())
        self.assertEqual((code, state["status"], state["failures"]),
                         (0, "WORKLOAD_PASS_REQUIRES_MONITOR_VERDICT", []))
        self.assertEqual(len(state["depths"]), 4)
        self.assertTrue(all(remediation.MIN_DEPTH <= depth <= remediation.MAX_DEPTH
                            for depth in state["depths"].values()))
        kinds = {name.split("-")[0] for name in state["operations"]}
        self.assertLessEqual({"fill", "continue", "new", "cancel", "image"}, kinds)
        self.assertLess(state["timeline"]["fill_finished"], state["timeline"]["warmup_finished"])
        self.assertLessEqual(state["timeline"]["warmup_finished"], state["timeline"]["load_finished"])
        gate = state["gates"]["mtp_long_c4_continuations"]
        self.assertEqual(gate["status"], "PASS", gate)
        qualified = gate["evidence"]["qualified_rounds"][0]
        self.assertEqual(len(qualified["requests"]), 4)
        self.assertTrue(all(row["prompt_tokens"] >= remediation.MIN_DEPTH and row["draft_tokens"] > 0
                            and row["draft_tokens_accepted"] > 0 for row in qualified["requests"].values()))
        self.assertEqual(qualified["width"], 4)
        self.assertEqual(state["gates"]["cancellation_confirmed"]["status"], "PASS")

    def test_composition_reports_foreign_data_and_missing_drafts(self):
        for defect, text in (("cross_talk", "foreign session data"), ("no_drafts", "MTP drafts")):
            with self.subTest(defect):
                code, state = self.compose([defect])
                self.assertEqual((code, state["status"]), (1, "FAIL"))
                self.assertTrue(any(text in failure["error"] for failure in state["failures"]), state["failures"])

    def test_composition_refill_keeps_the_json_exchange_in_the_next_continuation(self):
        bench = self.bench(pace=.01)
        evidence = remediation.Evidence(bench.url, bench.root / "out-refill")
        composition = remediation.Composition(evidence, "refill")
        composition.fill(2136)
        composition.continue_all()
        for session in composition.sessions:
            composition.reset(session, 2136)
        composition.continue_all()
        self.assertEqual(composition.state["failures"], [])
        for session in composition.sessions:
            marker = session["marker"]
            rows = evidence.report["requests"]
            refill = next(row for row in rows if row["id"].endswith("refill-" + marker))
            continuation = [row for row in rows if row["id"].endswith("continue-" + marker)][-1]
            refill_messages = json.loads((evidence.output / refill["request_file"]).read_text())["body"]["messages"]
            continued = json.loads((evidence.output / continuation["request_file"]).read_text())["body"]["messages"]
            # The next "same JSON format" request must include the successful
            # refill question and its real answer, just as after the first fill.
            expected_tail = [refill_messages[-1],
                             {"role": "assistant", "content": refill["result"]["text"]},
                             {"role": "user", "content": "Повтори те же три ключа в том же JSON-формате."}]
            with self.subTest(marker=marker):
                self.assertEqual(continued[-3:], expected_tail)
                self.assertEqual(continued[:-2], refill_messages)
                self.assertEqual(json.loads(continued[-2]["content"]), session["keys"])

    def test_long_c4_continuations_without_drafts_fail(self):
        # Короткие C1 показывают черновики, длинные продолжения — нет.
        code, state = self.compose(["no_long_drafts"])
        gate = state["gates"]["mtp_long_c4_continuations"]
        self.assertEqual((gate["status"], state["status"], code), ("FAIL", "FAIL", 1), gate)
        self.assertGreater(state["mtp_sampled"], 0)

    def test_long_c4_continuations_without_a_physical_cohort_are_not_exercised(self):
        code, state = self.compose(["width1"])
        self.assertEqual(state["gates"]["mtp_long_c4_continuations"]["status"], "NOT_EXERCISED")
        self.assertEqual((state["status"], code), ("INCOMPLETE", 2))

    def test_composition_cancellation_needs_the_server_log(self):
        for defects, log in (((), False), (["no_cancel_log"], True)):
            with self.subTest(defects=defects, log=log):
                code, state = self.compose(defects, log)
                self.assertEqual(state["gates"]["cancellation_confirmed"]["status"], "NOT_EXERCISED")
                self.assertEqual((state["status"], code), ("INCOMPLETE", 2))

    def test_registered_load_duration_cannot_be_shortened_silently(self):
        with self.assertRaisesRegex(ValueError, "3600"):
            self.run_command(self.bench(), "compose", "--duration", "60")


class PureTests(unittest.TestCase):
    def test_tasks_state_one_unambiguous_answer(self):
        request, expected = remediation.product_case("n1", "drafts-c1", "OWL_142", 17, 0, turns=2)
        text = json.dumps(request["body"]["messages"], ensure_ascii=False)
        self.assertEqual(expected, {"marker": "OWL_142", "products": [340, 357, 374, 391, 408, 425, 442, 459]})
        # В первом окне модель возвращала то имя беседы, то номер сеанса из системного сообщения.
        self.assertNotIn("идентификатор", text)
        self.assertNotIn("сеанс", text.lower())
        self.assertIn('marker — строка \\"OWL_142\\" дословно', text)
        self.assertEqual(remediation.plain(' "CODE_X". '), "CODE_X")
        self.assertNotEqual(remediation.plain('```python\nCODE_X = 1'), "CODE_X")
        self.assertIn("ровно одно слово", remediation.ASK)
        self.assertEqual(remediation.REASONING_ANSWER, str(sum(
            all(number % divisor for divisor in range(2, 32)) for number in range(1000, 1101))))

    def test_overall_status_never_promotes_unexercised_or_inherited(self):
        def gates(*statuses):
            return {str(index): {"status": status} for index, status in enumerate(statuses)}
        self.assertEqual(remediation.overall(gates("PASS", "PASS")), "PASS")
        self.assertEqual(remediation.overall(gates("PASS", "NOT_EXERCISED")), "INCOMPLETE")
        self.assertEqual(remediation.overall(gates("PASS", "INHERITED_FAIL")), "KNOWN_FAILURES")
        self.assertEqual(remediation.overall(gates("PASS", "MODEL_LIMIT")), "KNOWN_FAILURES")
        self.assertEqual(remediation.overall(gates("FAIL", "MODEL_LIMIT")), "FAIL")
        self.assertEqual(remediation.overall(gates("NOT_EXERCISED", "INHERITED_FAIL")), "INCOMPLETE")
        self.assertEqual(remediation.overall(gates("PASS", "NOT_EXERCISED", "FAIL")), "FAIL")
        self.assertEqual(remediation.overall(gates("PASS", "RUNNING")), "FAIL")

    def test_missing_counters_are_failures_not_zeros(self):
        with self.assertRaisesRegex(remediation.GateFail, "missing prompt/cache/prefill"):
            remediation.work({"metrics": {"prompt_tokens": 10, "cached_tokens": 10}})
        with self.assertRaisesRegex(remediation.GateFail, "draft counters"):
            remediation.drafts({"metrics": {"draft_tokens": 3}})
        with self.assertRaisesRegex(remediation.GateFail, "reasoning token counter"):
            remediation.reasoning_tokens({"usage": {"completion_tokens_details": {}}})

    def test_server_log_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(
                "I http request=r7 event=received method=POST\n"
                "I http request=r7 event=completed method=POST path=/v1/chat/completions status=200 "
                "prompt_tokens=272 prefill_tokens=13 finish=stop cache=disk cached_tokens=259\n"
                "I cache event=disk_cache action=skipped reason=min_step file_bytes=0 tokens=272\n"
                "I cache event=disk_cache action=stored reason=saved file_bytes=9 tokens=259 write_ms=1\n")
            log = remediation.ServerLog(path)
            row = log.completed("r7", timeout=0)
            self.assertEqual((row["cache"], row["cached_tokens"], row["prefill_tokens"]), ("disk", "259", "13"))
            self.assertIsNone(log.completed("r70", timeout=0))
            self.assertEqual(log.stored(0), [259])
            self.assertEqual(log.stored(path.stat().st_size), [])
            # Строки настоящего журнала первого окна (сокращены только хвосты).
            path.write_text(
                "2026-11-13 20:22:34 [INFO] [http] request=r20 event=received method=POST "
                "path=/v1/chat/completions body_bytes=1272\n"
                "2026-11-13 20:22:34 [WARN] [http] request=r20 event=completed method=POST "
                "path=/v1/chat/completions status=200 duration_ms=259.2 outcome=disconnected prompt_tokens=146 "
                "prefill_tokens=146 generated_tokens=5 finish=cancelled cache=miss cached_tokens=0 "
                "cache_restore_ms=0.0 queue_depth=1 resident_at_admission=1 queue_ms=4.3 batch_width=1\n"
                "2026-11-13 20:21:29 [INFO] [http] request=r3 event=completed method=POST "
                "path=/v1/chat/completions status=200 duration_ms=128.7 outcome=completed prompt_tokens=300 "
                "prefill_tokens=0 generated_tokens=11 finish=stop cache=memory cached_tokens=300 batch_width=1\n"
                "2026-11-13 20:21:28 [INFO] [cache] event=disk_cache_configured capacity_bytes=68719476736\n"
                "2026-11-13 20:21:28 [INFO] [cache] event=disk_cache action=stored reason=saved "
                "file_bytes=74449391 payload_bytes=74447141 tokens=293 retained_bytes=74449391 write_ms=58.2901\n"
                "2026-11-13 20:21:28 [INFO] [cache] event=disk_cache action=skipped reason=min_step file_bytes=0 "
                "payload_bytes=0 tokens=300 retained_bytes=74449391\n")
            cancelled, repeated = log.completed("r20", timeout=0), log.completed("r3", timeout=0)
            self.assertEqual((cancelled["finish"], cancelled["outcome"]), ("cancelled", "disconnected"))
            self.assertEqual((repeated["cache"], repeated["cached_tokens"], repeated["prefill_tokens"]),
                             ("memory", "300", "0"))
            self.assertIsNone(log.completed("r2", timeout=0))
            self.assertEqual(log.stored(0), [293])
        self.assertIsNone(remediation.ServerLog(None).completed("r7", timeout=0))

    def test_complex_images_are_deterministic_png_with_exact_goldens(self):
        probes = images.complex_probes()
        self.assertEqual(len({probe["id"] for probe in probes}), 5)
        for probe in probes:
            data = base64.b64decode(probe["content"]["image_url"]["url"].split(",", 1)[1])
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            width, height = struct.unpack(">II", data[16:24])
            self.assertEqual(width, 448)
            self.assertIn(height, (96, 160, 192))
            self.assertTrue(probe["grade"](probe["expected"]))
            self.assertFalse(probe["grade"]("unknown"))
            self.assertEqual(images.image_sha(probe["content"]),
                             images.image_sha(next(p for p in images.complex_probes()
                                                   if p["id"] == probe["id"])["content"]))
        by_id = {probe["id"]: probe for probe in probes}
        self.assertTrue(by_id["paragraph-lines"]["grade"]("river port 583\n Nine owls watch.\nTHE NORTH GATE\n"))
        self.assertFalse(by_id["paragraph-lines"]["grade"]("RIVER PORT 583\nTHE NORTH GATE"))
        self.assertTrue(by_id["formula-transcribe"]["grade"]("x=(48+6)/9"))
        self.assertTrue(by_id["table-second-row"]["grade"]("25, 604, 71"))
        self.assertFalse(by_id["table-second-row"]["grade"]("417, 82, 936"))

    def test_rendered_text_has_the_expected_ink(self):
        """Каждая буква рисуется из шрифта: число чёрных пикселей равно сумме битов глифов."""
        data = base64.b64decode(images.formula_image()["image_url"]["url"].split(",", 1)[1])
        position, idat = 8, b""
        while position < len(data):
            length = int.from_bytes(data[position:position + 4], "big")
            if data[position + 4:position + 8] == b"IDAT":
                idat += data[position + 8:position + 8 + length]
            position += 12 + length
        raw = zlib.decompress(idat)
        rows = [raw[index * (448 * 3 + 1) + 1:(index + 1) * (448 * 3 + 1)] for index in range(96)]
        black = sum(row[x * 3:x * 3 + 3] == b"\0\0\0" for row in rows for x in range(448))
        bits = sum(pattern.count("1") for character in images.FORMULA for pattern in images.FONT[character])
        self.assertEqual(black, bits * images.SCALE ** 2)

    def test_legacy_completion_transport(self):
        usage = {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22, "cached_tokens": 20,
                 "gufo": {"prefill_tokens": 0, "prefill_ms": 0, "decode_ms": 3, "cache_restore_ms": 1,
                          "cache_snapshot_ms": 0, "cache_disk_enqueue_ms": 0, "cache_disk_hit": True}}
        value = {"id": "cmpl-1", "object": "text_completion", "created": 1, "model": "qwen3.6-35b-a3b",
                 "usage": usage, "choices": [{"text": " ok", "index": 0, "finish_reason": "stop"}]}
        request = {"method": "POST", "path": "/v1/completions", "body": {
            "model": "qwen3.6-35b-a3b", "prompt": "a", "max_tokens": 4, "stream": False}}
        raw = json.dumps(value).encode()
        row = transport.parse_response(request, raw, [(5, raw)], 200)
        self.assertEqual((row["status"], row["text"], row["finish"]), ("PASS", " ok", "stop"))
        self.assertEqual((row["metrics"]["cached_tokens"], row["metrics"]["prefill_tokens"]), (20, 0))
        for broken in ({**value, "object": "chat.completion"}, {**value, "choices": []},
                       {**value, "model": "other"}):
            raw = json.dumps(broken).encode()
            with self.assertRaises(ValueError):
                transport.parse_response(request, raw, [(5, raw)], 200)


if __name__ == "__main__":
    unittest.main()
