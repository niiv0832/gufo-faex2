"""CPU-проверки агентских сценариев: без модели и внешней сети.

Проверяется, что harness требует НАСТОЯЩИЙ результат инструмента (то, что агент
сам вернул модели после исполнения), а не только итоговое состояние файлов или
слова модели о выполненной задаче.
"""

import json
import http.client
import http.server
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
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
    return {"stream": True, "messages": messages,
            "tools": [{"type": "function", "function": {"name": "bash"}}]}


STREAM_USAGE = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
                "cached_tokens": 5}


def complete_stream(content="ok"):
    events = [{"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}]},
              {"choices": [], "usage": STREAM_USAGE}]
    return "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"


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

    def test_shell_tool_results_are_checked_like_the_older_bash_tool(self):
        good = self.history(("shell", {"command": "cat hello.txt"}, "bonjour\n"),
                            ("read", {"filePath": "hello.txt"}, "bonjour\n"))
        self.assertEqual(opencode_agent.validate_tool_results("tools", good), 2)
        with self.assertRaises(AssertionError):
            opencode_agent.validate_tool_results("tools", [{**good[0], "result": ""}, good[1]])

    def test_cancelled_background_title_does_not_hide_agent_or_server_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            title = {"messages": [{"role": "system", "content":
                     "You name conversations so the user can find them later in a list of past sessions."}]}
            agent = chat_body(("shell", {"command": "cat hello.txt"}, "bonjour"))
            for index, body in enumerate((title, agent)):
                (output / f"request-{index:04d}.json").write_text(json.dumps(body))
                (output / f"request-{index:04d}.sse").write_text(complete_stream())
            cancelled = {"index": 0, "status": 200, "finalized": True,
                         "error": "BrokenPipeError(32, 'Broken pipe')"}
            completed = {"index": 1, "status": 200, "finalized": True, "usage": STREAM_USAGE}
            self.assertEqual(opencode_agent.validate_requests([cancelled, completed], output), [completed])
            for rows in ([{**cancelled, "status": 500}, completed],
                         [{**cancelled, "error": "TimeoutError()"}, completed],
                         [{**completed, "error": cancelled["error"]}]):
                with self.subTest(rows=rows), self.assertRaises(AssertionError):
                    opencode_agent.validate_requests(rows, output)

    def test_no_tools_foreground_is_not_mistaken_for_background_title(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            native = {"stream": True, "messages": [{"role": "user", "content": "task"}], "tools": [
                {"type": "function", "function": {"name": "catalog_" + tool["name"],
                 "parameters": tool["inputSchema"]}} for tool in opencode_agent.MCP_TOOLS]}
            missing_tools = {"stream": True, "messages": [{"role": "user", "content": "finish task"}]}
            for index, body in enumerate((native, missing_tools)):
                (output / f"request-{index:04d}.json").write_text(json.dumps(body))
                (output / f"request-{index:04d}.sse").write_text(complete_stream())
            rows = [{"index": index, "status": 200, "finalized": True,
                     "usage": STREAM_USAGE,
                     "metrics": {"prefill_tokens": 5}} for index in range(2)]
            self.assertEqual(opencode_agent.validate_requests(rows[:1], output, native_mcp=True), rows[:1])
            with self.assertRaisesRegex(AssertionError, "native MCP tool missing"):
                opencode_agent.validate_requests(rows, output, native_mcp=True)
            (output / "request-0001.sse").write_text(complete_stream("<invoke forbidden framing"))
            with self.assertRaisesRegex(AssertionError, "framing reached content"):
                opencode_agent.validate_requests(rows, output)

    def test_pending_http_200_cannot_pass_before_its_late_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            body = chat_body(("shell", {"command": "cat hello.txt"}, "bonjour"))
            (output / "request-0000.json").write_text(json.dumps(body))
            (output / "request-0000.sse").write_text("data: [DONE]\n")
            row = {"index": 0, "status": 200}
            with self.assertRaisesRegex(AssertionError, "not finalized"):
                opencode_agent.validate_requests([row], output)
            row.update(finalized=True, wall_ms=10, error="TimeoutError('late upstream error')")
            with self.assertRaises(AssertionError):
                opencode_agent.validate_requests([row], output)

    def test_completed_first_turn_requires_done_terminal_choice_and_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "request-0000.json").write_text(json.dumps(chat_body()))
            stream = output / "request-0000.sse"
            row = {"index": 0, "status": 200, "finalized": True, "usage": STREAM_USAGE}
            good = complete_stream()
            stream.write_text(good)
            self.assertEqual(opencode_agent.validate_requests([row], output), [row])
            for name, raw in (
                    ("empty", ""),
                    ("missing DONE", good.replace("data: [DONE]\n\n", "")),
                    ("duplicate DONE", good + "data: [DONE]\n\n"),
                    ("event after DONE", good + 'data: {}\n\n'),
                    ("missing usage", good.split('data: {"choices": [],')[0] + "data: [DONE]\n\n"),
                    ("missing terminal choice", good.replace('"finish_reason": "stop"', '"finish_reason": null')),
                    ("server error", good.replace("data: [DONE]", 'data: {"error":{"message":"late error"}}\n\ndata: [DONE]'))):
                with self.subTest(name=name), self.assertRaises(AssertionError):
                    stream.write_text(raw)
                    opencode_agent.validate_requests([row], output)
            stream.write_text(good)
            for usage in (None, {}, {**STREAM_USAGE, "completion_tokens": 1}):
                with self.subTest(usage=usage), self.assertRaises(AssertionError):
                    opencode_agent.validate_requests([{**row, "usage": usage}], output)

    def test_shared_stream_completion_distinguishes_responses_from_chat(self):
        usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        encode = lambda event: ("data: " + json.dumps(event) + "\n\n").encode()
        for status in ("completed", "incomplete"):
            terminal = {"type": "response." + status, "response": {"status": status, "usage": usage}}
            if status == "incomplete":
                terminal["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
            with self.subTest(status=status):
                self.assertEqual(pi_agent.validate_stream_completion(
                    "/v1/responses", {"stream": True}, encode(terminal)), usage)
        for label, broken in (
                ("unexpected DONE", encode(terminal) + b"data: [DONE]\n\n"),
                ("no terminal", encode({"type": "response.created"})),
                ("failed", encode({"type": "response.failed", "response": {"status": "failed", "usage": usage}})),
                ("status mismatch", encode({"type": "response.completed", "response": {"status": "incomplete", "usage": usage}})),
                ("missing usage", encode({"type": "response.completed", "response": {"status": "completed"}}))):
            with self.subTest(label=label), self.assertRaises(AssertionError):
                pi_agent.validate_stream_completion("/v1/responses", {"stream": True}, broken)

    def test_edit_task_needs_original_source_and_passing_run(self):
        good = self.history(("read", {"filePath": "calc.py"}, "00001| def add(a, b):\n00002|     return a - b"),
                            ("edit", {"filePath": "calc.py", "oldString": "-", "newString": "+"}, "done"),
                            ("bash", {"command": "python3 -c 'import calc; assert calc.add(2, 3) == 5; print(\"ok\")'"},
                             "ok\n"))
        opencode_agent.validate_tool_results("edit", good)
        failing = [*good[:2], {**good[2], "result": "Traceback (most recent call last):\nAssertionError\nok"}]
        with self.assertRaisesRegex(AssertionError, "result lacks"):
            opencode_agent.validate_tool_results("edit", failing)
        # «ok» должен быть отдельной строкой вывода настоящей проверки, а не частью другого текста.
        for label, steps in (
                ("word inside a sentence", [*good[:2], {**good[2], "result": "this does not look ok to me"}]),
                ("token", [*good[:2], {**good[2], "result": "tokens: 12"}]),
                ("not ok", [*good[:2], {**good[2], "result": "not ok"}]),
                ("command without the assertion", [*good[:2], {
                    **good[2], "arguments": '{"command": "echo ok # calc"}'}])):
            with self.subTest(label), self.assertRaises(AssertionError):
                opencode_agent.validate_tool_results("edit", steps)
        wrapped = [*good[:2], {**good[2], "result": "<stdout>\nok\n</stdout>"}]
        opencode_agent.validate_tool_results("edit", wrapped)
        with self.assertRaisesRegex(AssertionError, "no executed read"):
            opencode_agent.validate_tool_results("edit", good[1:])

    def test_mcp_task_needs_catalog_results(self):
        good = self.history(("catalog_lookup_record", {"id": 42}, '{"id": 42, "title": "Native framing record"}'),
                            ("catalog_lookup_date", {"date": "2026-10-05"}, '{"events": ["release"]}'))
        opencode_agent.validate_tool_results("mcp", good)
        with self.assertRaises(AssertionError):
            opencode_agent.validate_tool_results("mcp", good[:1])

    def stale_calc_bytecode(self, cwd, *, cached_fixed):
        # Apple Python otherwise redirects py_compile outside the fixture to
        # ~/Library/Caches, which is both shared state and sandbox-inaccessible.
        cache = unittest.mock.patch.dict(os.environ, {"PYTHONPYCACHEPREFIX": str(cwd / "bytecode")})
        cache.start()
        self.addCleanup(cache.stop)
        fixture = opencode_agent.verify_fixture(cwd)
        path = cwd / "calc.py"
        broken = path.read_text()
        fixed = broken.replace("a - b", "a + b")
        timestamp = 1700000000
        path.write_text(fixed if cached_fixed else broken)
        os.utime(path, (timestamp, timestamp))
        # Compile explicitly: the caller may disable bytecode writes or use
        # PYTHONPYCACHEPREFIX. Timestamp invalidation reproduces same-tick edits.
        subprocess.run([sys.executable, "-c",
                        "import py_compile; py_compile.compile('calc.py', doraise=True, "
                        "invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)"],
                       cwd=cwd, check=True, capture_output=True, timeout=10)
        path.write_text(broken if cached_fixed else fixed)
        os.utime(path, (timestamp, timestamp))
        cached = subprocess.run([sys.executable, "-c", "import calc; print(calc.add(2, 3))"],
                                cwd=cwd, check=True, capture_output=True, text=True, timeout=10)
        self.assertEqual(cached.stdout.strip(), "5" if cached_fixed else "-1")
        return fixture

    def test_validate_task_accepts_fixed_source_despite_stale_broken_bytecode(self):
        for name in ("edit", "verify"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                cwd = Path(directory)
                self.stale_calc_bytecode(cwd, cached_fixed=False)
                try:
                    opencode_agent.validate_task(name, cwd, cwd / "absent.jsonl")
                except subprocess.CalledProcessError as error:
                    self.fail(f"fixed calc.py was rejected because of stale bytecode: {error.stderr!r}")

    def test_validate_task_rejects_broken_source_despite_stale_fixed_bytecode(self):
        for name in ("edit", "verify"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                cwd = Path(directory)
                self.stale_calc_bytecode(cwd, cached_fixed=True)
                with self.assertRaises(subprocess.CalledProcessError):
                    opencode_agent.validate_task(name, cwd, cwd / "absent.jsonl")

    def test_fixture_check_accepts_fixed_source_despite_stale_broken_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            fixture = self.stale_calc_bytecode(cwd, cached_fixed=False)
            result = subprocess.run([sys.executable, "check.py"], cwd=cwd,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), fixture["expected"])

    def test_fixture_check_rejects_broken_source_despite_stale_fixed_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            fixture = self.stale_calc_bytecode(cwd, cached_fixed=True)
            result = subprocess.run([sys.executable, "check.py"], cwd=cwd,
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn(fixture["expected"], result.stdout)

    def test_verify_fixture_passes_only_after_a_real_fix_and_a_real_run(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            fixture = opencode_agent.verify_fixture(cwd)
            # Same-size edits within a filesystem timestamp tick must not load
            # the broken module from Python's bytecode cache after the fix.
            timestamp = 1700000000
            os.utime(cwd / "calc.py", (timestamp, timestamp))
            run = lambda: subprocess.run([sys.executable, "check.py"], cwd=cwd, capture_output=True, text=True)
            broken = run()
            self.assertNotEqual(broken.returncode, 0)
            self.assertNotIn(fixture["expected"], broken.stdout)
            self.assertNotIn(fixture["expected"], (cwd / "check.py").read_text())  # Не читается из файла.
            source = (cwd / "calc.py").read_text()
            self.assertIn(fixture["sentinel"], source)
            (cwd / "calc.py").write_text(source.replace("a - b", "a + b"))
            os.utime(cwd / "calc.py", (timestamp, timestamp))
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
            self.assertEqual(isolated.get("OPENCODE_CONFIG_PROJECT_DISABLE"), "1")
            self.assertEqual(isolated.get("OPENCODE_DISABLE_CLAUDE_CODE"), "1")
            self.assertEqual(isolated.get("OPENCODE_CONFIG_DIR"), str(case / "xdg-config" / "opencode"))
            for environment in (isolated, inherited):
                for name in ("OPENCODE_CONFIG", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
                             "XDG_CACHE_HOME"):
                    self.assertTrue(environment[name].startswith(str(case)))

    def test_native_mcp_requires_the_actual_schemas_not_a_code_mode_wrapper(self):
        body = {"tools": [{"type": "function", "function": {
            "name": "catalog_" + tool["name"], "parameters": tool["inputSchema"]}}
            for tool in opencode_agent.MCP_TOOLS]}
        opencode_agent.validate_native_mcp(body)
        with self.assertRaises(AssertionError):
            opencode_agent.validate_native_mcp({"tools": [{"type": "function", "function": {
                "name": "execute", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}}]})
        broken = json.loads(json.dumps(body))
        broken["tools"][1]["function"]["parameters"]["properties"]["date"].pop("pattern")
        with self.assertRaises(AssertionError):
            opencode_agent.validate_native_mcp(broken)

    def test_isolation_gate_checks_user_messages_and_system_messages(self):
        clean = {"messages": [{"role": "system", "content": "OpenCode"},
                              {"role": "user", "content": "Read calc.py"}]}
        opencode_agent.validate_isolation([clean])
        for role in ("system", "user"):
            with self.subTest(role=role), self.assertRaises(AssertionError):
                opencode_agent.validate_isolation([{"messages": [{"role": role,
                    "content": "You have superpowers. Load using-superpowers."}]}])

    def test_output_inside_a_git_tree_is_detected(self):
        self.assertTrue(opencode_agent.inside_git_tree(Path(__file__).parent))
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(opencode_agent.inside_git_tree(Path(directory) / "new" / "output"))
            (Path(directory) / ".git").mkdir()
            self.assertTrue(opencode_agent.inside_git_tree(Path(directory) / "new" / "output"))


class OpenCodeVersion(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "results"
        self.executable = self.root / "fake-opencode"

    def executable_body(self, body):
        self.executable.write_text(f"#!{sys.executable}\n" + body)
        self.executable.chmod(0o755)

    def run_main(self, passes=1):
        argv = ["opencode_agent.py", "--base-url", "http://127.0.0.1:1", "--model", "test",
                "--opencode", str(self.executable), "--output", str(self.output),
                "--isolated", "--passes", str(passes), "--case", "tools"]
        with unittest.mock.patch.object(sys, "argv", argv):
            return opencode_agent.main()

    def test_main_probes_version_with_private_config_home_and_cwd(self):
        (self.root / "canary").write_text("parent home must stay hidden")
        (self.root / "opencode.json").write_text('{"plugin":["parent-canary"]}')
        self.executable_body("""import json, os
from pathlib import Path
cwd = Path.cwd()
assert cwd.name == 'version'
assert Path(os.environ['HOME']) == cwd / 'home'
assert not (Path(os.environ['HOME']) / 'canary').exists()
assert os.environ['OPENCODE_CONFIG_PROJECT_DISABLE'] == '1'
assert os.environ['OPENCODE_DISABLE_CLAUDE_CODE'] == '1'
assert 'OPENAI_API_KEY' not in os.environ
assert json.loads(Path(os.environ['OPENCODE_CONFIG']).read_text()) == {'autoupdate': False}
assert all(Path(os.environ[name]).is_relative_to(cwd) for name in
           ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME', 'XDG_CACHE_HOME'))
print('2.2.5')
""")
        with unittest.mock.patch.dict(os.environ, {"HOME": str(self.root), "OPENAI_API_KEY": "canary"}), \
                unittest.mock.patch.object(opencode_agent, "Recorder") as recorder, \
                unittest.mock.patch.object(opencode_agent, "run_case", return_value={"case": "00-tools", "status": "pass"}):
            with self.assertRaises(SystemExit) as exited:
                self.run_main()
        self.assertEqual(exited.exception.code, 0)
        metadata = json.loads((self.output / "opencode-version.json").read_text())
        self.assertEqual(metadata["version"], "2.2.5")
        self.assertEqual(metadata["returncode"], 0)
        recorder.assert_called_once()

    def test_main_rejects_nonpositive_passes_before_any_probe_or_recorder(self):
        for passes in (0, -1):
            self.output = self.root / f"results-{passes}"
            with self.subTest(passes=passes), \
                    unittest.mock.patch.object(opencode_agent, "probe_version") as version, \
                    unittest.mock.patch.object(opencode_agent, "Recorder") as recorder:
                with self.assertRaises(SystemExit) as exited:
                    self.run_main(passes)
                self.assertEqual(exited.exception.code, 2)
                version.assert_not_called()
                recorder.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_main_rejects_empty_or_failed_version_before_starting_recorder(self):
        for body, code in (("", 0), ("print('2.2.5')\nraise SystemExit(23)\n", 23)):
            with self.subTest(code=code):
                self.output = self.root / f"results-{code}"
                self.executable_body(body)
                with unittest.mock.patch.object(opencode_agent, "Recorder") as recorder:
                    with self.assertRaisesRegex(RuntimeError, "version probe failed"):
                        self.run_main()
                    recorder.assert_not_called()
                metadata = json.loads((self.output / "opencode-version.json").read_text())
                self.assertEqual(metadata["returncode"], code)

    def test_pi_main_rejects_nonpositive_passes_with_explicit_case(self):
        for passes in (0, -1):
            output = self.root / f"pi-results-{passes}"
            argv = ["pi_agent.py", "--base-url", "http://127.0.0.1:1", "--pi", "/unused/pi",
                    "--model", "test", "--output", str(output), "--server-log", str(self.root / "server.log"),
                    "--case", "simple", "--passes", str(passes)]
            with self.subTest(passes=passes), unittest.mock.patch.object(sys, "argv", argv), \
                    unittest.mock.patch.object(pi_agent.subprocess, "check_output", return_value="0.87.0") as version, \
                    unittest.mock.patch.object(pi_agent, "Recorder") as recorder:
                with self.assertRaises(SystemExit) as exited:
                    pi_agent.main()
                self.assertEqual(exited.exception.code, 2)
                version.assert_not_called()
                recorder.assert_not_called()
                self.assertFalse(output.exists())

    def test_version_timeout_is_bounded_and_retains_failure_artifacts(self):
        self.output.mkdir()
        self.executable_body("import time\ntime.sleep(30)\n")
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "version probe failed"):
            opencode_agent.probe_version(str(self.executable), self.output, True, timeout=0.1)
        self.assertLess(time.monotonic() - start, 3)
        metadata = json.loads((self.output / "opencode-version.json").read_text())
        self.assertIn("TimeoutExpired", metadata["error"])
        self.assertIsNotNone(metadata["returncode"])
        self.assertNotEqual(metadata["returncode"], 0)
        self.assertTrue((self.output / "version" / "stderr.txt").exists())


class RecorderCompletion(unittest.TestCase):
    """Задержанный loopback upstream воспроизводит позднюю ошибку после HTTP 200."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.release = threading.Event()
        release = self.release

        class PausedUpstream(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(getattr(self.server, "payload",
                                        b'data: {"choices":[{"delta":{"content":"title"}}]}\n\n'))
                self.wfile.flush()
                release.wait(3)

        self.upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), PausedUpstream)
        self.upstream.daemon_threads = True
        self.recorder = pi_agent.Recorder(f"http://127.0.0.1:{self.upstream.server_port}", self.output)
        self.clients = []
        for server in (self.upstream, self.recorder):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(self.stop)

    def stop(self):
        self.release.set()
        for thread in self.clients:
            thread.join(timeout=2)
        for server in (self.recorder, self.upstream):
            server.shutdown()
            server.server_close()

    def post(self, case):
        self.recorder.case = case
        index = len(self.recorder.requests)
        title = {"stream": True, "messages": [{"role": "system", "content":
                 "You name conversations so the user can find them later in a list of past sessions."}]}

        def send():
            conn = http.client.HTTPConnection("127.0.0.1", self.recorder.server_port, timeout=3)
            try:
                conn.request("POST", "/v1/chat/completions", json.dumps(title), {"Content-Type": "application/json"})
                conn.getresponse().read()
            except (OSError, http.client.HTTPException):
                pass  # Forced drain intentionally closes this downstream connection.
            finally:
                conn.close()

        thread = threading.Thread(target=send, daemon=True)
        self.clients.append(thread)
        thread.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if len(self.recorder.requests) > index:
                row = self.recorder.requests[index]
                if row.get("status") == 200 or row.get("finalized"):
                    return row
            time.sleep(0.005)
        self.fail("proxy did not reach response headers")

    def test_drain_waits_for_late_timeout_and_preserves_raw(self):
        self.recorder.request_timeout = 0.15
        live = self.post("one")
        self.assertNotIn("wall_ms", live)
        self.assertTrue(callable(getattr(self.recorder, "drain_case", None)), "Recorder needs a completion barrier")
        rows, report = self.recorder.drain_case("one", timeout=1)
        self.assertTrue(report["complete"])
        self.assertEqual(report["timed_out"], [])
        self.assertTrue(rows[0]["finalized"])
        self.assertRegex(rows[0]["error"], r"(?:TimeoutError|timeout)\(")
        self.assertGreater(rows[0]["wall_ms"], 100)
        self.assertTrue((self.output / "request-0000.sse").read_bytes())
        self.assertEqual(json.loads((self.output / "request-0000.timing.json").read_text()), rows[0])
        with self.assertRaises(AssertionError):
            opencode_agent.validate_requests(rows, self.output)
        timeout_error = rows[0]["error"]
        live["error"] = "later mutation"
        self.assertEqual(rows[0]["error"], timeout_error)  # Report owns an immutable snapshot.

    def test_proxy_rejects_http_eof_without_done_even_after_terminal_usage(self):
        self.upstream.payload = complete_stream().replace("data: [DONE]\n\n", "").encode()
        self.release.set()
        self.post("eof")
        rows, report = self.recorder.drain_case("eof", timeout=1)
        self.assertTrue(report["complete"])
        self.assertEqual(rows[0]["status"], 200)
        self.assertTrue(rows[0]["finalized"])
        self.assertEqual(rows[0]["usage"], STREAM_USAGE)
        self.assertIn("missing terminal DONE", rows[0]["error"])
        self.assertEqual((self.output / "request-0000.sse").read_bytes(), self.upstream.payload)

    def test_bounded_drain_aborts_only_the_current_case_and_cannot_pass(self):
        one = self.post("one")
        two = self.post("two")
        self.assertTrue(callable(getattr(self.recorder, "drain_case", None)), "Recorder needs a completion barrier")
        start = time.monotonic()
        rows, report = self.recorder.drain_case("one", timeout=0.05, abort_timeout=1)
        self.assertLess(time.monotonic() - start, 1.5)
        self.assertTrue(report["complete"])
        self.assertEqual(report["timed_out"], [one["index"]])
        self.assertTrue(rows[0]["finalized"])
        with self.assertRaises(AssertionError):
            opencode_agent.validate_requests(rows, self.output)
        self.assertFalse(two.get("finalized", False))
        self.assertNotIn("drain_timeout", two)
        self.release.set()
        _, other = self.recorder.drain_case("two", timeout=1)
        self.assertTrue(other["complete"])
        self.assertEqual(other["timed_out"], [])

    def test_body_capture_error_is_finalized_instead_of_hanging(self):
        with unittest.mock.patch.object(Path, "write_bytes", side_effect=OSError("capture disk failure")):
            row = self.post("one")
        self.assertTrue(row.get("finalized"), "body capture errors must reach the completion barrier")
        self.assertIn("capture disk failure", row["error"])
        self.assertTrue((self.output / "request-0000.timing.json").is_file())


class OpenCodeLifecycle(unittest.TestCase):
    """Настоящие дочерние процессы проверяют ожидание и cleanup, без модели/сети."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.case = Path(self.directory.name)
        self.cwd = self.case / "work"
        self.cwd.mkdir()
        self.executable = self.case / "fake-opencode"
        self.executable.write_text(f"#!{sys.executable}\n" + r'''
import json, os, pathlib, signal, subprocess, sys, time
root = pathlib.Path(__file__).parent
mode = (root / 'mode').read_text()
if sys.argv[1] == 'serve':
    (root / 'server.pid').write_text(str(os.getpid()))
    if mode == 'child':
        child = subprocess.Popen([sys.executable, '-c',
            "import pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "p=pathlib.Path('heartbeat'); "
            "exec('while True:\\n p.write_text(str(time.monotonic()))\\n time.sleep(.02)')"], cwd=root)
        (root / 'child.pid').write_text(str(child.pid))
    if mode == 'partial':
        sys.stdout.write('{"url":'); sys.stdout.flush()
    else:
        print('{"url":"http://127.0.0.1:12345"}', flush=True)
    sys.stdin.read()
elif sys.argv[1:3] == ['api', 'mcp.list']:
    assert sys.argv[3:] == ['--server', 'http://127.0.0.1:12345']
    if mode == 'query-hangs':
        time.sleep(30)
    count = root / 'queries'
    n = int(count.read_text()) + 1 if count.exists() else 1
    count.write_text(str(n))
    status = 'failed' if mode == 'failed' else 'pending' if n == 1 else 'connected'
    if status == 'connected' and not (root / 'registry-ready-at').exists():
        (root / 'registry-ready-at').write_text(str(time.time() + .1))
    print(json.dumps({'location': {'directory': os.getcwd()},
                      'data': [{'name': 'catalog', 'status': {'status': status}}]}))
    sys.stdout.flush()
    os._exit(0)
else:
    raise SystemExit('unexpected fake CLI arguments')
''')
        self.executable.chmod(0o755)

    def server(self, mode, seconds=3):
        self.assertTrue(callable(getattr(opencode_agent, "private_opencode", None)),
                        "harness must wait for MCP on an owned private server")
        (self.case / "mode").write_text(mode)
        return opencode_agent.private_opencode(str(self.executable), self.case, self.cwd,
                                               dict(os.environ), time.monotonic() + seconds)

    def assert_server_stopped(self):
        pid = int((self.case / "server.pid").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertIsNotNone(json.loads((self.case / "server-exit.json").read_text())["returncode"])

    def test_first_prompt_waits_for_connected_mcp(self):
        with self.server("pending") as (endpoint, env):
            self.assertEqual(endpoint, "http://127.0.0.1:12345")
            self.assertGreaterEqual(int((self.case / "queries").read_text()), 2)
            self.assertGreaterEqual(time.time(), float((self.case / "registry-ready-at").read_text()),
                                    "connected precedes the asynchronous tool registry update")
            self.assertTrue(env.get("OPENCODE_PASSWORD"))
        self.assert_server_stopped()

    def test_mcp_failure_prevents_prompt_and_stops_server(self):
        with self.assertRaisesRegex(AssertionError, "MCP.*failed"):
            with self.server("failed"):
                self.fail("failed MCP must not reach the agent prompt")
        self.assert_server_stopped()

    def test_case_keeps_readiness_failure_as_the_primary_error(self):
        (self.case / "mode").write_text("failed")
        args = SimpleNamespace(output=self.case / "results", model="unused", thinking=False,
                               isolated=True, opencode=str(self.executable), timeout=3)
        recorder = SimpleNamespace(server_port=1, requests=[],
                                   drain_case=lambda _: ([], {"complete": True, "timed_out": []}))
        with unittest.mock.patch("builtins.print"):
            row = opencode_agent.run_case(args, recorder, 0, "tools")
        self.assertEqual(row["status"], "fail")
        self.assertIn("MCP catalog failed", row["error"])
        self.assertIn("http_validation_error", row)
        self.assertFalse((args.output / "00-tools" / "command.json").exists())
        with self.assertRaises(ProcessLookupError):
            os.kill(int((self.case / "server.pid").read_text()), 0)

    def test_partial_readiness_and_hanging_query_obey_deadline(self):
        for mode in ("partial", "query-hangs"):
            with self.subTest(mode=mode):
                start = time.monotonic()
                with self.assertRaisesRegex(AssertionError, "timeout"):
                    with self.server(mode, seconds=0.5):
                        self.fail("incomplete readiness must not reach the agent prompt")
                self.assertLess(time.monotonic() - start, 3)
                self.assert_server_stopped()

    def test_body_exception_kills_owned_descendants_only(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            with self.assertRaisesRegex(RuntimeError, "agent failed"):
                with self.server("child"):
                    deadline = time.monotonic() + 2
                    while not (self.case / "heartbeat").exists() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue((self.case / "heartbeat").exists())
                    raise RuntimeError("agent failed")
            self.assert_server_stopped()
            heartbeat = (self.case / "heartbeat").read_text()
            time.sleep(0.1)
            self.assertEqual((self.case / "heartbeat").read_text(), heartbeat)
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)


class OptimizedPython(unittest.TestCase):
    """python -O и PYTHONOPTIMIZE удаляют assert: проверки обязаны отказать, а не пройти."""
    SNIPPETS = {
        "opencode results": "import opencode_agent as m; m.validate_tool_results('tools', [])",
        "opencode requests": "import opencode_agent as m, pathlib; m.validate_requests([], pathlib.Path('.'))",
        "opencode task": "import opencode_agent as m, pathlib; m.validate_task('tools', pathlib.Path('/nonexistent'), "
                         "pathlib.Path('/nonexistent/x'))",
        "pi task": "import pi_agent as m, pathlib; m.validate_task('tools', pathlib.Path('.'), [])",
        "pi requests": "import pi_agent as m; m.validate_requests([])",
        "pi outputs": "import pi_agent as m; m.validate_real_outputs('tools', [])",
    }

    def run_python(self, arguments, environment=None):
        return subprocess.run([sys.executable, *arguments], cwd=Path(__file__).parent, capture_output=True,
                              text=True, timeout=60, env={**os.environ, **(environment or {})})

    def test_library_checks_refuse_to_run_optimized(self):
        for label, code in self.SNIPPETS.items():
            for mode, arguments, environment in (("-O", ["-O", "-c", code], None),
                                                 ("PYTHONOPTIMIZE", ["-c", code], {"PYTHONOPTIMIZE": "1"})):
                with self.subTest(label=label, mode=mode):
                    result = self.run_python(arguments, environment)
                    self.assertNotEqual(result.returncode, 0, "optimized run silently passed")
                    self.assertIn("without -O", result.stderr)

    def test_command_line_refuses_to_run_optimized(self):
        for script in ("opencode_agent.py", "pi_agent.py"):
            with self.subTest(script), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "out"
                result = self.run_python(["-O", script, "--base-url", "http://127.0.0.1:1", "--model", "m",
                                          "--output", str(output), "--pi", "/nonexistent",
                                          "--server-log", "/nonexistent"][:8 if script.startswith("open") else 12])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("without -O", result.stderr)
                self.assertFalse(output.exists())            # Отказ до любых действий.


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

    def test_bugfix_and_creation_need_a_real_successful_test_run(self):
        def history(command, shown, error=False):
            return [assistant(call("a", "bash", {"command": command})), result("a", shown, error),
                    assistant({"type": "text", "text": "C'est corrigé."}, stop="stop")]
        pi_agent.validate_real_outputs("bugfix", history("node slugify.test.js", "ok\n"))
        pi_agent.validate_real_outputs("creation", history("node test.js", "4 cas vérifiés\n"))
        failures = {
            "bugfix: run failed": ("bugfix", history("node slugify.test.js", "AssertionError", error=True)),
            "bugfix: other output": ("bugfix", history("node slugify.test.js", "looks ok to me")),
            "bugfix: other command": ("bugfix", history("echo ok", "ok\n")),
            "bugfix: never run": ("bugfix", [assistant({"type": "text", "text": "ok"}, stop="stop")]),
            "creation: run failed": ("creation", history("node test.js", "boom", error=True)),
            "creation: never run": ("creation", history("ls", "stats.js test.js")),
        }
        for label, (name, messages) in failures.items():
            with self.subTest(label), self.assertRaises(AssertionError):
                pi_agent.validate_real_outputs(name, messages)

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
