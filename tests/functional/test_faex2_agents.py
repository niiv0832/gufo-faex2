"""CPU-проверки разбора результатов агентских сценариев: без агента, модели и сети.

Проверяется, что harness требует НАСТОЯЩИЙ результат инструмента (то, что агент
сам вернул модели после исполнения), а не только итоговое состояние файлов или
слова модели о выполненной задаче.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import opencode_agent
import pi_agent


def chat_body(*steps):
    """steps: (tool, arguments, result|None) — история, как её присылает OpenCode."""
    messages = [{"role": "user", "content": "task"}]
    for index, (name, arguments, result) in enumerate(steps):
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call{index}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments)}}]})
        if result is not None:
            messages.append({"role": "tool", "tool_call_id": f"call{index}", "content": result})
    return {"messages": messages, "tools": [{"type": "function", "function": {"name": "bash"}}]}


class OpenCodeResults(unittest.TestCase):
    def history(self, *steps):
        return opencode_agent.tool_history([chat_body(*steps)])

    def test_history_joins_calls_with_results_across_growing_requests(self):
        first = chat_body(("read", {"filePath": "calc.py"}, None))
        second = chat_body(("read", {"filePath": "calc.py"}, "return a - b"),
                           ("bash", {"command": "python3 check.py"}, [{"type": "text", "text": "ok"}]))
        history = opencode_agent.tool_history([first, second])
        self.assertEqual([(call["name"], call["result"]) for call in history],
                         [("read", "return a - b"), ("bash", "ok")])

    def test_history_reads_responses_items(self):
        body = {"input": [{"role": "user", "content": "task"},
                          {"type": "function_call", "call_id": "c1", "name": "bash",
                           "arguments": '{"command":"cat hello.txt"}'},
                          {"type": "function_call_output", "call_id": "c1", "output": "bonjour\n"}]}
        self.assertEqual(opencode_agent.tool_history([body])[0]["result"], "bonjour\n")

    def test_tools_task_needs_real_cat_and_read_output(self):
        good = self.history(("write", {"filePath": "hello.txt", "content": "bonjour"}, "ok"),
                            ("bash", {"command": "cat hello.txt", "timeout": 5000}, "bonjour\n"),
                            ("read", {"filePath": "/w/hello.txt"}, "<file>\n00001| bonjour\n</file>"))
        self.assertEqual(opencode_agent.validate_tool_results("tools", good), 3)
        for label, steps in (
                ("no read", good[:2]),
                ("cat printed nothing", [good[0], {**good[1], "result": ""}, good[2]]),
                ("call never executed", [good[0], {**good[1], "result": None}, good[2]])):
            with self.subTest(label), self.assertRaises(AssertionError):
                opencode_agent.validate_tool_results("tools", steps)

    def test_no_tool_result_at_all_is_a_failure(self):
        with self.assertRaisesRegex(AssertionError, "no tool result reached the model"):
            opencode_agent.validate_tool_results("tools", self.history(("bash", {"command": "cat hello.txt"}, None)))

    def test_edit_task_needs_original_source_and_passing_run(self):
        good = self.history(("read", {"filePath": "calc.py"}, "00001| def add(a, b):\n00002|     return a - b"),
                            ("edit", {"filePath": "calc.py", "oldString": "-", "newString": "+"}, "done"),
                            ("bash", {"command": "python3 -c 'import calc; print(\"ok\")'"}, "ok\n"))
        opencode_agent.validate_tool_results("edit", good)
        failing = [*good[:2], {**good[2], "result": "Traceback (most recent call last):\nAssertionError\nok"}]
        with self.assertRaisesRegex(AssertionError, "result lacks 'ok'"):
            opencode_agent.validate_tool_results("edit", failing)
        with self.assertRaisesRegex(AssertionError, "no executed read"):
            opencode_agent.validate_tool_results("edit", good[1:])

    def test_mcp_task_needs_catalog_results(self):
        good = self.history(("catalog_lookup_record", {"id": 42}, '{"id": 42, "title": "Native framing record"}'),
                            ("catalog_lookup_date", {"date": "2026-10-05"}, '{"events": ["release"]}'))
        opencode_agent.validate_tool_results("mcp", good)
        with self.assertRaises(AssertionError):
            opencode_agent.validate_tool_results("mcp", good[:1])

    def test_verify_fixture_passes_only_after_a_real_fix_and_a_real_run(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            fixture = opencode_agent.verify_fixture(cwd)
            run = lambda: subprocess.run([sys.executable, "check.py"], cwd=cwd, capture_output=True, text=True)
            broken = run()
            self.assertNotEqual(broken.returncode, 0)
            self.assertNotIn(fixture["expected"], broken.stdout)
            self.assertNotIn(fixture["expected"], (cwd / "check.py").read_text())  # Не читается из файла.
            source = (cwd / "calc.py").read_text()
            self.assertIn(fixture["sentinel"], source)
            (cwd / "calc.py").write_text(source.replace("a - b", "a + b"))
            fixed = run()
            self.assertEqual(fixed.stdout.strip(), fixture["expected"])
            opencode_agent.validate_task("verify", cwd, cwd / "absent.jsonl")
            history = self.history(("read", {"filePath": "calc.py"}, source),
                                   ("edit", {"filePath": "calc.py"}, "Edit applied"),
                                   ("bash", {"command": "python3 check.py"}, fixed.stdout))
            self.assertEqual(opencode_agent.validate_tool_results("verify", history, fixture), 3)
            for label, steps in (
                    # Модель «знает» ответ, но инструмент его не возвращал.
                    ("guessed output", [*history[:2], {**history[2], "result": "verified-0000000000000000"}]),
                    ("output of the unfixed module", [*history[:2], {**history[2], "result": broken.stderr}]),
                    ("file was not really read", [{**history[0], "result": "def add(a, b):\n    return a - b"},
                                                  *history[1:]]),
                    ("no edit call", [history[0], history[2]])):
                with self.subTest(label), self.assertRaises(AssertionError):
                    opencode_agent.validate_tool_results("verify", steps, fixture)
            another = opencode_agent.verify_fixture(cwd)
            self.assertNotEqual(another["expected"], fixture["expected"])  # Секрет свой у каждого прогона.

    def test_isolated_environment_hides_user_home_and_variables(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            with unittest.mock.patch.dict(os.environ, {"OPENAI_API_KEY": "secret", "HOME": "/Users/someone",
                                                       "PATH": "/usr/bin"}, clear=True):
                isolated = opencode_agent.case_environment(case, True)
                inherited = opencode_agent.case_environment(case, False)
            self.assertNotIn("OPENAI_API_KEY", isolated)
            self.assertEqual(isolated["HOME"], str(case / "home"))
            self.assertTrue((case / "home").is_dir())
            self.assertEqual(inherited["HOME"], "/Users/someone")
            for environment in (isolated, inherited):
                for name in ("OPENCODE_CONFIG", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
                             "XDG_CACHE_HOME"):
                    self.assertTrue(environment[name].startswith(str(case)))

    def test_output_inside_a_git_tree_is_detected(self):
        self.assertTrue(opencode_agent.inside_git_tree(Path(__file__).parent))
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(opencode_agent.inside_git_tree(Path(directory) / "new" / "output"))
            (Path(directory) / ".git").mkdir()
            self.assertTrue(opencode_agent.inside_git_tree(Path(directory) / "new" / "output"))


def call(identifier, name, arguments):
    return {"type": "toolCall", "id": identifier, "name": name, "arguments": arguments}


def assistant(*parts, stop="toolUse"):
    return {"role": "assistant", "content": list(parts), "stopReason": stop}


def result(identifier, text, error=False):
    return {"role": "toolResult", "toolCallId": identifier, "isError": error,
            "content": [{"type": "text", "text": text}]}


class PiResults(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cwd = Path(self.directory.name)

    def tools_history(self, shown="bonjour"):
        return [assistant(call("a", "write", {"path": "hello.txt", "content": "bonjour"})), result("a", "written"),
                assistant(call("b", "bash", {"command": "cat hello.txt"})), result("b", shown),
                assistant(call("c", "read", {"path": "hello.txt"})), result("c", shown),
                assistant({"type": "text", "text": "Le fichier contient bonjour."}, stop="stop")]

    def test_session_parsing_ignores_an_unfinished_last_line(self):
        session = self.cwd / "session.jsonl"
        lines = [json.dumps({"type": "session"}), json.dumps({"type": "message", "message": result("a", "x")}),
                 '{"type": "message", "message": {"role": "assist']
        session.write_text("\n".join(lines))
        self.assertEqual(pi_agent.messages(session), [result("a", "x")])
        self.assertEqual(pi_agent.messages(self.cwd / "absent.jsonl"), [])

    def test_repeated_identical_action_is_counted_with_its_result(self):
        history = []
        for index in range(4):
            history += [assistant(call(f"c{index}", "bash", {"command": "node test.js"})),
                        result(f"c{index}", "fail")]
        history += [assistant(call("z", "bash", {"command": "node test.js"})), result("z", "ok")]
        self.assertEqual(pi_agent.repeated_actions(history), 4)
        self.assertEqual(len(pi_agent.calls_from(history)), 5)

    def test_tools_task_requires_the_file_and_a_real_tool_output(self):
        (self.cwd / "hello.txt").write_bytes(b"bonjour")
        pi_agent.validate_task("tools", self.cwd, self.tools_history())
        # Файл на месте и модель утверждает успех, но инструменты вернули ошибку либо пустоту.
        failed = self.tools_history()
        failed[3], failed[5] = result("b", "bonjour", error=True), result("c", "bonjour", error=True)
        for label, history in (("empty tool output", self.tools_history("")), ("only failed results", failed)):
            with self.subTest(label), self.assertRaisesRegex(AssertionError, "no tool returned the file"):
                pi_agent.validate_task("tools", self.cwd, history)
        (self.cwd / "hello.txt").write_bytes(b"bonjour\n")
        with self.assertRaises(AssertionError):
            pi_agent.validate_task("tools", self.cwd, self.tools_history())

    def test_edit_task_requires_the_fix_and_the_original_fixture_output(self):
        (self.cwd / "src").mkdir()
        target = self.cwd / "src/calc.py"
        target.write_text("def add(a, b):\n    return a + b\n")

        def history(shown):
            return [assistant(call("a", "read", {"path": "src/calc.py"})), result("a", shown),
                    assistant(call("b", "edit", {"path": "src/calc.py", "edits": [
                        {"oldText": "a - b", "newText": "a + b"}]})), result("b", "edited"),
                    assistant(call("c", "bash", {"command": "cat src/calc.py"})), result("c", "return a + b"),
                    assistant({"type": "text", "text": "Corrigé."}, stop="stop")]
        pi_agent.validate_task("edit", self.cwd, history("def add(a, b):\n    return a - b\n"))
        with self.assertRaisesRegex(AssertionError, "original fixture"):
            pi_agent.validate_task("edit", self.cwd, history(""))
        target.write_text("def add(a, b):\n    return a - b\n")
        with self.assertRaises(AssertionError):
            pi_agent.validate_task("edit", self.cwd, history("def add(a, b):\n    return a - b\n"))

    def test_leaked_framing_and_unfinished_turn_fail(self):
        (self.cwd / "hello.txt").write_bytes(b"bonjour")
        leaked = self.tools_history()
        leaked[-1] = assistant({"type": "text", "text": "ok</tool_call>"}, stop="stop")
        unfinished = self.tools_history()
        unfinished[-1] = assistant({"type": "text", "text": "ok"}, stop="length")
        for history in (leaked, unfinished):
            with self.assertRaises(AssertionError):
                pi_agent.validate_task("tools", self.cwd, history)

    def rows(self, cached=90, prefill=10):
        def row(identifier, cached_tokens, prefill_tokens):
            return {"status": 200, "request_id": identifier, "wall_ms": 50.0,
                    "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
                              "cached_tokens": cached_tokens},
                    "metrics": {"prefill_tokens": prefill_tokens, "prefill_ms": 1.0, "decode_ms": 2.0,
                                "cache_restore_ms": 0, "cache_snapshot_ms": 0, "cache_disk_enqueue_ms": 0}}
        return [row("r1", 0, 100), row("r2", cached, prefill)]

    def server_log(self, requests=("r1", "r2")):
        path = self.cwd / "server.log"
        path.write_text("".join(f"I http request={name} event=completed status=200 duration_ms=40.0 "
                                f"queue_ms=1.0 ttft_ms=12.0\n" for name in requests))
        return path

    def test_request_validation_joins_the_server_log_and_requires_cache_reuse(self):
        rows = self.rows()
        pi_agent.validate_requests(rows, self.server_log())
        self.assertEqual((rows[1]["metrics"]["ttft_ms"], rows[1]["metrics"]["queue_ms"]), (12.0, 1.0))
        with self.assertRaises(AssertionError):  # Продолжение прочитало историю заново.
            pi_agent.validate_requests(self.rows(cached=0, prefill=100), self.server_log())
        with self.assertRaises(KeyError):        # Запроса нет в журнале сервера.
            pi_agent.validate_requests(self.rows(), self.server_log(("r1",)))
        with self.assertRaisesRegex(AssertionError, "did not send"):
            pi_agent.validate_requests([], self.server_log())


if __name__ == "__main__":
    unittest.main()
