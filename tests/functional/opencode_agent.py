#!/usr/bin/env python3
"""Real opencode agent regression for #438; executes tasks in disposable fixtures.

opencode's own tools are plain, but the MCP tools agents load often carry
schemas native tags cannot enforce exactly (pattern, oneOf, allOf, open
objects). Those used to switch every tool of the request to a JSON envelope,
and long sessions then mixed syntaxes and leaked `<invoke>`/JSON framing into
content. This runner starts a small MCP server with such tools beside
opencode's built-ins and checks that every turn stays native: framing never
reaches content, MCP arguments keep their JSON types, tasks complete and each
next request reuses the previous one from cache.

Requires opencode and an already-running local Gufo server. Keeps exact HTTP
bodies/SSE, opencode JSON events, MCP calls and per-request timings.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import select
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit

from pi_agent import Recorder, save, validate_stream_completion
from tool_native import NEIGHBORS

# Tags that are only framing. A turn that leaks one also stores it in the
# client's history, where later turns copy it (#383, #438).
FRAMING = ("<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter",
           "</parameter>", "<invoke", "</invoke>", '{"name":', "<｜DSML｜")

MCP_TOOLS = [
    {"name": "lookup_record", "description": "Look up a catalog record by id.",
     "inputSchema": {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
                     "properties": {"id": {"oneOf": [{"type": "integer"}, {"type": "string"}],
                                           "description": "Numeric or textual record id"}},
                     "required": ["id"]}},
    {"name": "lookup_date", "description": "Return events on a calendar date.",
     "inputSchema": {"type": "object", "properties": {
         "date": {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
                  "description": "Date as YYYY-MM-DD"}},
         "required": ["date"], "additionalProperties": False}},
    {"name": "tag_record", "description": "Attach a tag to a record.",
     "inputSchema": {"type": "object", "properties": {
         "name": {"allOf": [{"type": "string"}, {"minLength": 1}]},
         "weight": {"type": "integer", "minimum": 1, "maximum": 10}},
         "required": ["name"], "additionalProperties": True}},
]

# Every schema family of tool-native-schemas, loaded as further MCP tools, so
# each opencode turn declares all of them beside the ones the tasks call.
MCP_TOOLS += [{"name": f"zoo_{name}", "description": "Catalog extension; not needed.",
               "inputSchema": {"type": "object", **schema}}
              for name, schema in NEIGHBORS.items()]

PROMPTS = {
    "tools": ("Create a file hello.txt containing exactly the word bonjour, show it with "
              "cat using the shell tool with a timeout of 5000, then read it back with the "
              "read tool and confirm its content in one sentence."),
    "edit": ("Read calc.py, fix the obvious bug with the edit tool, then run "
             "`python3 -c 'import calc; assert calc.add(2, 3) == 5; print(\"ok\")'` "
             "with the shell tool and report the result."),
    "mcp": ("Use the catalog MCP tools: look up record 42 (the integer id 42), look up "
            "the date 2026-10-05, and tag record 42 with the tag name urgent and weight 3. "
            "Then write the record title returned by the lookup into result.txt and reply "
            "with that title."),
    # The bash result depends on a per-run secret, so it cannot be produced
    # without really executing the fixture's check after a real fix.
    "verify": ("Read calc.py, fix the obvious bug with the edit tool, then run "
               "`python3 check.py` with the shell tool and reply with the exact line it prints."),
}


def require_assertions():
    """Every check here is an assert statement: an optimized run would pass everything."""
    if sys.flags.optimize:
        raise SystemExit("run without -O and PYTHONOPTIMIZE: the checks are assert statements")


def serve_mcp(log):
    """A minimal stdio MCP server: newline-delimited JSON-RPC 2.0."""
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        method, ident = message.get("method"), message.get("id")
        if ident is None:
            continue  # notifications
        if method == "initialize":
            result = {"protocolVersion": message["params"].get("protocolVersion", "2025-06-18"),
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "gufo-catalog", "version": "1.0.0"}}
        elif method == "tools/list":
            result = {"tools": MCP_TOOLS}
        elif method == "tools/call":
            params = message["params"]
            with open(log, "a") as out:
                out.write(json.dumps({"name": params["name"],
                                      "arguments": params.get("arguments")}) + "\n")
            arguments = params.get("arguments") or {}
            if params["name"] == "lookup_record":
                text = json.dumps({"id": arguments.get("id"), "title": "Native framing record"})
            elif params["name"] == "lookup_date":
                text = json.dumps({"date": arguments.get("date"), "events": ["release"]})
            else:
                text = json.dumps({"tagged": True, **arguments})
            result = {"content": [{"type": "text", "text": text}], "isError": False}
        elif method == "ping":
            result = {}
        else:
            print(json.dumps({"jsonrpc": "2.0", "id": ident,
                              "error": {"code": -32601, "message": method}}), flush=True)
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": ident, "result": result}), flush=True)


def sse_output(path, *, reasoning=None):
    """Content and tool calls a recorded streamed response returned."""
    content, calls = "", {}
    for line in path.read_text().splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[6:])
        if reasoning is not None and event.get("type") == "response.reasoning_summary_text.delta":
            reasoning.append(event["delta"])
        if event.get("type") == "response.output_text.delta":
            content += event["delta"]
        if event.get("type") == "response.output_item.done":
            item = event.get("item", {})
            if item.get("type") == "function_call":
                calls[item["id"]] = {"name": item["name"], "arguments": item["arguments"]}
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if reasoning is not None:
                reasoning.append(delta.get("reasoning_content") or "")
            content += delta.get("content") or ""
            for call in delta.get("tool_calls") or []:
                entry = calls.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                function = call.get("function", {})
                entry["name"] += function.get("name") or ""
                entry["arguments"] += function.get("arguments") or ""
    return content, list(calls.values())


def validate_native_mcp(body):
    """The native-schema regression must expose MCP schemas to the model."""
    require_assertions()
    declared = {tool["function"]["name"]: tool["function"].get("parameters", {})
                for tool in body.get("tools", []) if tool.get("type") == "function"}

    def contains(actual, expected):
        if isinstance(expected, dict):
            return isinstance(actual, dict) and all(
                key in actual and contains(actual[key], value)
                for key, value in expected.items() if key != "$schema")
        return actual == expected

    for tool in MCP_TOOLS:
        name = "catalog_" + tool["name"]
        assert name in declared, ("native MCP tool missing (Code Mode is a different test)", name)
        assert contains(declared[name], tool["inputSchema"]), ("MCP schema was weakened", name)


REFERENCES = Path(__file__).resolve().parent / "fixtures" / "opencode_clean"


def reference_version(reported):
    """`opencode --version` prints "opencode v2.3.0" or the bare number."""
    match = re.search(r"\d+\.\d+\.\d+", reported)
    return match.group(0) if match else reported.strip()


def load_reference(version):
    """What a clean OpenCode of this version sends before anything of the user is added."""
    path = REFERENCES / f"{version}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no clean-run reference for OpenCode {version} at {path}; record one with "
            "--record-reference in an isolated run, review it and add it to the repository")
    return json.loads(path.read_text())


def normalize_run_text(text, cwd, model):
    """Replace what legitimately differs between two clean runs with fixed marks."""
    text = text.replace(str(cwd), "{WORKDIR}")
    text = re.sub(r"(?m)^(- (?:Name|Model ID): )" + re.escape(model) + "$", r"\1{MODEL}", text)
    text = re.sub(r"(?m)^(Today's date: ).*$", r"\1{DATE}", text)
    text = re.sub(r"(?m)^(  Platform: ).*$", r"\1{PLATFORM}", text)
    text = re.sub(r"(?m)^(  Prefer )\S+( over generic system temporary directories)", r"\1{TMP}\2", text)
    return re.sub(r"(The current year is )\d{4}", r"\1{YEAR}", text)


def message_text(message):
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


def leading_messages(body):
    """The system messages and the first user message: what a continued dialogue keeps."""
    lead = []
    for message in body.get("messages", []):
        lead.append((message.get("role"), message_text(message)))
        if message.get("role") == "user":
            break
    return lead


def classify_requests(bodies):
    """`task`, `background` or `other` for each request body of one case.

    The task chain is the first request with tools and every request that
    starts with the same system and first user message. A request with no
    tools outside that chain is a background service request, such as the
    conversation title; its wording differs between OpenCode versions.
    """
    head = next((leading_messages(body) for body in bodies if body.get("tools")), None)
    kinds = []
    for body in bodies:
        if head is not None and leading_messages(body) == head:
            kinds.append("task")
        else:
            kinds.append("other" if body.get("tools") else "background")
    return kinds


def builtin_tools(body, cwd, model):
    """Tool definitions that are not the test's own MCP catalog, in a stable form."""
    tools = [tool for tool in body.get("tools", [])
             if not tool.get("function", {}).get("name", "").startswith("catalog_")]
    return json.loads(normalize_run_text(json.dumps(tools, sort_keys=True, ensure_ascii=False), cwd, model))


def build_reference(bodies, *, cwd, model, prompt):
    """Reference of a clean run, from its recorded title and first task request."""
    kinds = classify_requests(bodies)
    task = next(body for body, kind in zip(bodies, kinds) if kind == "task")
    title = next(body for body, kind in zip(bodies, kinds) if kind == "background")
    title_user = next(message_text(m) for m in title["messages"] if m.get("role") == "user")
    assert prompt in title_user, "the background request does not carry the task prompt"
    return {
        "agent_system": normalize_run_text(message_text(task["messages"][0]), cwd, model),
        "title_system": normalize_run_text(message_text(title["messages"][0]), cwd, model),
        "title_user": normalize_run_text(title_user.replace(prompt, "{PROMPT}"), cwd, model),
        "tools": builtin_tools(task, cwd, model),
    }


def first_difference(actual, expected):
    index = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                 min(len(actual), len(expected)))
    return {"offset": index, "sent": actual[max(0, index - 80):index + 200],
            "clean": expected[max(0, index - 80):index + 200]}


def validate_isolation(bodies, reference, *, cwd, model, prompt):
    """Nothing but what a clean OpenCode sends may reach the model.

    Instructions from the user's directories arrive as extra system text,
    extra user messages or extra tools, in any wording. So every system
    message, every user message and every built-in tool definition must equal
    the clean-run reference of this OpenCode version; only the working
    directory, model name, date, platform and temporary directory may differ.
    """
    require_assertions()
    kinds = classify_requests(bodies)
    for index, (body, kind) in enumerate(zip(bodies, kinds)):
        assert kind != "other", ("request with tools outside the task chain", index,
                                 [text[:200] for _, text in leading_messages(body)])
        messages = body.get("messages", [])
        systems = [normalize_run_text(message_text(m), cwd, model) for m in messages if m.get("role") == "system"]
        users = [message_text(m) for m in messages if m.get("role") == "user"]
        expected_system = reference["agent_system" if kind == "task" else "title_system"]
        assert systems == [expected_system], (
            "system message differs from the clean reference", index, kind, len(systems),
            first_difference("\n".join(systems), expected_system))
        if kind == "task":
            assert users == [prompt], ("user message is not exactly the task prompt", index, len(users),
                                       [text[:200] for text in users])
            sent = builtin_tools(body, cwd, model)
            if sent != reference["tools"]:
                names = lambda tools: sorted(tool["function"]["name"] for tool in tools)
                changed = [a["function"]["name"] for a in sent for b in reference["tools"]
                           if a["function"]["name"] == b["function"]["name"] and a != b]
                raise AssertionError(("tool set differs from the clean reference", index,
                                      {"sent": names(sent), "clean": names(reference["tools"]),
                                       "changed": changed}))
            catalog = {"catalog_" + tool["name"] for tool in MCP_TOOLS}
            extra = {tool["function"]["name"] for tool in body.get("tools", [])
                     if tool["function"]["name"].startswith("catalog_")} - catalog
            assert not extra, ("tool set differs from the clean reference", index, sorted(extra))
        else:
            expected_user = reference["title_user"].replace("{PROMPT}", prompt)
            assert [normalize_run_text(text, cwd, model) for text in users] == [expected_user], (
                "background request differs from the clean reference", index,
                [text[:300] for text in users])
            assert not body.get("tools"), ("background request carries tools", index)
    return {"task": kinds.count("task"), "background": kinds.count("background")}


def validate_requests(rows, output, *, native_mcp=False):
    """Every agent turn stays native and reuses the previous one."""
    require_assertions()
    agent = []
    bodies = [json.loads((output / f"request-{row['index']:04d}.json").read_text()) for row in rows]
    for row, body, kind in zip(rows, bodies, classify_requests(bodies)):
        assert row.get("finalized") is True, ("HTTP request not finalized", row)
        assert not row.get("drain_timeout"), ("HTTP request exceeded recorder drain timeout", row)
        # OpenCode cancels its background conversation-title request when the
        # foreground task finishes. Keep its raw failure, but do not mistake
        # that client cancellation for an agent/tool failure. Server errors,
        # timeouts and disconnected foreground requests remain failures.
        title = kind == "background"
        cancelled_title = title and str(row.get("error", "")).startswith(
            ("BrokenPipeError(", "ConnectionResetError("))
        assert row.get("status") == 200 and (not row.get("error") or cancelled_title), row
        if title:
            continue  # counted by the caller; it has its own prompt and no tools
        usage = validate_stream_completion(row.get("path", "/v1/chat/completions"), body,
                                           (output / f"request-{row['index']:04d}.sse").read_bytes())
        assert usage and row.get("usage") == usage, ("missing/inconsistent recorded terminal usage", row)
        if native_mcp:
            validate_native_mcp(body)
        content, calls = sse_output(output / f"request-{row['index']:04d}.sse")
        leaked = [tag for tag in FRAMING if tag in content]
        assert not leaked, ("framing reached content", leaked, row["index"], content)
        for call in calls:
            assert call["name"], (row["index"], calls)
            json.loads(call["arguments"])
        # History sent back must not carry framing either.
        for message in body["messages"]:
            text = message.get("content")
            if message.get("role") == "assistant" and isinstance(text, str):
                assert not any(tag in text for tag in FRAMING), (row["index"], text)
        agent.append(row)
    assert agent, "opencode sent no tool request"
    for row in agent[1:]:
        usage = row["usage"]
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        cached = usage.get("cached_tokens",
                           (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        assert cached > 0 and row["metrics"]["prefill_tokens"] < prompt, (
            "a continued agent turn re-prefilled its history", row)
    return agent


def validate_task(name, cwd, calls_log):
    require_assertions()
    mcp = [json.loads(line) for line in calls_log.read_text().splitlines()] \
        if calls_log.exists() else []
    if name == "tools":
        assert (cwd / "hello.txt").read_text().strip() == "bonjour"
    elif name in ("edit", "verify"):
        # Execute the current source; same-size edits within one timestamp tick
        # can leave a valid-looking but stale calc.pyc after an agent's test run.
        subprocess.run([sys.executable, "-c",
                        "import runpy; calc = runpy.run_path('calc.py'); "
                        "assert calc['add'](2, 3) == 5 and calc['add'](-4, 1) == -3"],
                       cwd=cwd, check=True, capture_output=True, timeout=10)
    elif name == "mcp":
        by_name = {call["name"]: call["arguments"] for call in mcp}
        assert by_name.get("lookup_record") == {"id": 42}, mcp
        assert by_name.get("lookup_date") == {"date": "2026-10-05"}, mcp
        tag = by_name.get("tag_record")
        assert tag and tag.get("name") == "urgent" and tag.get("weight") == 3, mcp
        assert "Native framing record" in (cwd / "result.txt").read_text()
    return mcp


def tool_history(bodies):
    """Calls with the results the client really sent back, in request order.

    Tool results in a request body come from the agent's own execution: the
    recorder forwards bodies unchanged and never writes one. Chat bodies carry
    them as `tool` messages, Responses bodies as `function_call_output` items.
    """
    calls, order = {}, []

    def call(identifier, name, arguments):
        if identifier not in calls:
            calls[identifier] = {"name": name, "arguments": arguments, "result": None}
            order.append(identifier)

    def text(content):
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content if isinstance(content, str) else json.dumps(content)

    for body in bodies:
        for message in body.get("messages", []):
            for item in message.get("tool_calls") or []:
                call(item["id"], item["function"]["name"], item["function"]["arguments"])
            if message.get("role") == "tool" and message.get("tool_call_id") in calls:
                calls[message["tool_call_id"]]["result"] = text(message.get("content"))
        for item in body.get("input", []) if isinstance(body.get("input"), list) else []:
            if item.get("type") == "function_call":
                call(item["call_id"], item["name"], item["arguments"])
            elif item.get("type") == "function_call_output" and item.get("call_id") in calls:
                calls[item["call_id"]]["result"] = text(item.get("output"))
    return [calls[identifier] for identifier in order]


def validate_tool_results(name, history, fixture=None):
    """The model was shown real tool output, not a claim that the task is done."""
    require_assertions()
    executed = [call for call in history if call["result"] is not None]
    assert executed, "no tool result reached the model"

    def ran(tools, argument, result, forbidden=()):
        found = [call for call in executed
                 if any(call["name"] == tool or call["name"].endswith("_" + tool) for tool in tools)
                 and all(part in call["arguments"] for part in argument)]
        assert found, (f"no executed {'/'.join(tools)} call mentioning {argument}",
                       [(call["name"], call["arguments"]) for call in executed])
        good = [call for call in found
                if (result.search(call["result"]) if isinstance(result, re.Pattern) else result in call["result"])
                and not any(mark in call["result"] for mark in forbidden)]
        assert good, (f"{'/'.join(tools)} {argument}: result lacks {getattr(result, 'pattern', result)!r}",
                      [call["result"][-400:] for call in found])
        return good[-1]

    failed = ("Traceback", "AssertionError")
    if name == "tools":
        ran(("shell", "bash"), ("cat", "hello.txt"), "bonjour")
        ran(("read",), ("hello.txt",), "bonjour")
    elif name == "edit":
        ran(("read",), ("calc.py",), "return a - b")
        # The run that asserts the fix printed "ok" on a line of its own; the
        # word inside other text, or a command without the assertion, is not it.
        ran(("shell", "bash"), ("calc", "assert"), re.compile(r"(?m)^\s*ok\s*$"), failed)
    elif name == "verify":
        ran(("read",), ("calc.py",), fixture["sentinel"])
        ran(("edit", "patch", "multiedit", "write"), ("calc.py",), "")
        ran(("shell", "bash"), ("check.py",), fixture["expected"], failed)
    elif name == "mcp":
        ran(("lookup_record",), ("42",), "Native framing record")
        ran(("lookup_date",), ("2026-10-05",), "release")
    return len(executed)


def verify_fixture(cwd):
    """A buggy module, a marker only a real read returns and a check only a real run passes."""
    nonce, sentinel = secrets.token_hex(8), "fixture-" + secrets.token_hex(6)
    (cwd / "calc.py").write_text(f"# {sentinel}\ndef add(a, b):\n    return a - b\n")
    (cwd / ".nonce").write_text(nonce)
    (cwd / "check.py").write_text(
        "import hashlib\nimport runpy\ncalc = runpy.run_path('calc.py')\n"
        "assert calc['add'](2, 3) == 5 and calc['add'](-4, 1) == -3\n"
        "print('verified-' + hashlib.sha256(open('.nonce', 'rb').read()).hexdigest()[:16])\n")
    return {"sentinel": sentinel,
            "expected": "verified-" + hashlib.sha256(nonce.encode()).hexdigest()[:16]}


def inside_git_tree(path):
    """An agent started under a repository reads that repository's instructions."""
    path = Path(path).resolve()
    return any((parent / ".git").exists() for parent in (path, *path.parents))


def case_environment(case, isolated):
    """Per-case opencode directories; `isolated` also drops the caller's HOME and variables."""
    source = os.environ
    if isolated:
        (case / "home").mkdir(exist_ok=True)
        source = {key: value for key, value in os.environ.items()
                  if key in {"PATH", "USER", "LANG", "LC_ALL", "TMPDIR", "SHELL", "SSL_CERT_FILE"}}
        source["HOME"] = str(case / "home")
        # V2 searches all cwd ancestors for opencode.json(c), even outside a
        # Git tree. HOME/XDG alone does not isolate a fixture under /Users/name.
        source["OPENCODE_CONFIG_PROJECT_DISABLE"] = "1"
        source["OPENCODE_DISABLE_CLAUDE_CODE"] = "1"
        source["OPENCODE_CONFIG_DIR"] = str(case / "xdg-config" / "opencode")
    return {**source, "OPENCODE_CONFIG": str(case / "opencode.json"),
            "XDG_CONFIG_HOME": str(case / "xdg-config"), "XDG_DATA_HOME": str(case / "xdg-data"),
            "XDG_STATE_HOME": str(case / "xdg-state"), "XDG_CACHE_HOME": str(case / "xdg-cache")}


def stop_process_group(proc):
    """Stop only a process group created here with start_new_session=True."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # The group can still contain MCP/tool children after its leader exits.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=2)


def probe_version(opencode, output, isolated, timeout=10):
    """Collect provenance without exposing the isolated run to caller config."""
    case = (output / "version").resolve()
    case.mkdir()
    save(case / "opencode.json", {"autoupdate": False})
    env = case_environment(case, isolated)
    result = {"version": "", "returncode": None}
    proc = None
    try:
        with (case / "stdout.txt").open("w") as stdout, (case / "stderr.txt").open("w") as stderr:
            proc = subprocess.Popen([opencode, "--version"], cwd=case, env=env,
                                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                    start_new_session=True)
            try:
                proc.wait(timeout=timeout)
            finally:
                stop_process_group(proc)
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        if proc is not None:
            result["returncode"] = proc.returncode
        stdout_path = case / "stdout.txt"
        if stdout_path.exists():
            result["version"] = stdout_path.read_text().strip()
        save(output / "opencode-version.json", result)
    if result.get("error") or result["returncode"] != 0 or not result["version"]:
        raise RuntimeError(f"OpenCode version probe failed; see {output / 'opencode-version.json'}")
    return result


@contextmanager
def private_opencode(opencode, case, cwd, env, deadline):
    """Own one V2 server and wait for its MCP catalog before the first prompt.

    A cold `run --standalone` races asynchronous MCP startup (seen in OpenCode
    2.2.5; the same private-server sequence is used with 2.3.0).
    Readiness and the run must use this same server and canonical directory.
    The random loopback password is passed through env only, never saved.
    """
    require_assertions()
    cwd = cwd.resolve()
    env = {**env, "OPENCODE_PASSWORD": secrets.token_urlsafe(32)}
    command = [opencode, "serve", "--stdio", "--hostname", "127.0.0.1", "--port", "0", "--print-logs"]
    save(case / "server-command.json", command)
    server = None
    try:
        with (case / "server.log").open("w") as log, (case / "server.stdout").open("wb") as output:
            server = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=log, start_new_session=True)
            line = b""
            while b"\n" not in line:
                remaining = deadline - time.monotonic()
                assert remaining > 0, "private OpenCode readiness timeout"
                assert select.select([server.stdout], [], [], remaining)[0], "private OpenCode readiness timeout"
                chunk = os.read(server.stdout.fileno(), 4096)
                assert chunk, "private OpenCode exited before readiness; see server.log"
                output.write(chunk)
                output.flush()
                line += chunk
                assert len(line) <= 65536, "invalid private OpenCode readiness output"
            endpoint = json.loads(line.split(b"\n", 1)[0])["url"]
            parsed = urlsplit(endpoint)
            assert parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port, (
                "private OpenCode returned a non-loopback endpoint")
            with (case / "mcp-readiness.jsonl").open("w") as history:
                connected_since = None
                while True:
                    remaining = deadline - time.monotonic()
                    assert remaining > 0, "MCP catalog readiness timeout"
                    assert server.poll() is None, "private OpenCode exited while waiting for MCP; see server.log"
                    query = subprocess.Popen([opencode, "api", "mcp.list", "--server", endpoint],
                                             cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                             stdin=subprocess.DEVNULL, text=True, start_new_session=True)
                    try:
                        try:
                            stdout, stderr = query.communicate(timeout=remaining)
                        except subprocess.TimeoutExpired as error:
                            history.write(json.dumps({"error": "MCP readiness query timeout"}) + "\n")
                            raise AssertionError("MCP readiness query timeout") from error
                    finally:
                        stop_process_group(query)
                        query.stdout.close()
                        query.stderr.close()
                    history.write(json.dumps({"returncode": query.returncode,
                                              "stdout": stdout, "stderr": stderr}) + "\n")
                    history.flush()
                    assert query.returncode == 0, "MCP readiness query failed; see mcp-readiness.jsonl"
                    state = json.loads(stdout)
                    assert state["location"]["directory"] == str(cwd), "MCP readiness used a different directory"
                    catalog = next((row for row in state["data"] if row["name"] == "catalog"), None)
                    status = catalog["status"]["status"] if catalog else "pending"
                    if status == "connected":
                        # V2.2.5 updates the tool registry through a 100 ms
                        # debounce after publishing `connected`. Observe a
                        # stable connection for two polling intervals; actual
                        # outgoing schemas remain a separate strict gate.
                        if connected_since is None:
                            connected_since = time.monotonic()
                        if time.monotonic() - connected_since >= 0.2:
                            break
                    else:
                        connected_since = None
                        assert status == "pending", f"MCP catalog {status}; see mcp-readiness.jsonl"
                    time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            yield endpoint, env
    finally:
        if server is not None:
            # Closing stdin lets `serve --stdio` shut down normally. Always
            # clean the owned group as well, including children of a dead leader.
            if server.stdin:
                server.stdin.close()
            try:
                server.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            stop_process_group(server)
            if server.stdout:
                server.stdout.close()
            save(case / "server-exit.json", {"pid": server.pid, "returncode": server.returncode})


def run_case(args, recorder, index, name):
    case = (args.output / f"{index:02d}-{name}").resolve()
    cwd = case / "work"
    cwd.mkdir(parents=True)
    fixture = None
    if name == "edit":
        (cwd / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    elif name == "verify":
        fixture = verify_fixture(cwd)
        save(case / "fixture.json", fixture)
    calls_log = case / "mcp-calls.jsonl"
    config = {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "provider": {"gufo": {
            "npm": "@ai-sdk/openai-compatible", "name": "gufo",
            "options": {"baseURL": f"http://127.0.0.1:{recorder.server_port}/v1"},
            "models": {args.model: {"name": args.model, "tool_call": True,
                                    "reasoning": args.thinking,
                                    "limit": {"context": 262144, "output": 16384}}}}},
        "mcp": {"catalog": {"type": "local", "enabled": True, "codemode": False,
                            "command": [sys.executable, str(Path(__file__).resolve()),
                                        "--mcp", str(calls_log)]}},
        "permission": {"*": "allow"},
    }
    save(case / "opencode.json", config)
    env = case_environment(case, args.isolated)
    recorder.case = case.name
    row = {"case": case.name, "start_request": len(recorder.requests)}
    start = time.monotonic()
    proc = None
    try:
        with private_opencode(args.opencode, case, cwd, env, start + min(args.timeout, 30)) as (endpoint, run_env):
            command = [args.opencode, "run", "--server", endpoint, "--auto", "--format", "json",
                       "-m", f"gufo/{args.model}", PROMPTS[name]]
            save(case / "command.json", command)
            with (case / "events.jsonl").open("w") as out, (case / "stderr.log").open("w") as err:
                proc = subprocess.Popen(command, cwd=cwd.resolve(), env=run_env, stdout=out, stderr=err,
                                        start_new_session=True)
                try:
                    proc.wait(timeout=max(0.001, args.timeout - (time.monotonic() - start)))
                except subprocess.TimeoutExpired as error:
                    raise AssertionError("opencode task timeout") from error
                finally:
                    stop_process_group(proc)
                assert proc.returncode == 0, (case / "stderr.log").read_text()[-4000:]
        row["mcp_calls"] = validate_task(name, cwd, calls_log)
        row["status"] = "pass"
    except Exception as error:
        row.update(status="fail", error=repr(error))
    row["requests"], row["http_drain"] = recorder.drain_case(case.name)
    save(case / "http-drain.json", row["http_drain"])
    row["wall_ms"] = (time.monotonic() - start) * 1000
    try:
        assert row["http_drain"]["complete"] and not row["http_drain"]["timed_out"], (
            "HTTP recorder did not finish within the drain deadline", row["http_drain"])
        row["agent_requests"] = len(validate_requests(row["requests"], args.output, native_mcp=True))
    except Exception as error:
        if row["status"] == "pass":
            row.update(status="fail", error=f"HTTP/framing/cache validation: {error!r}")
        else:
            row["http_validation_error"] = repr(error)
    try:
        bodies = [json.loads((args.output / f"request-{item['index']:04d}.json").read_text())
                  for item in row["requests"]]
        row["background_requests"] = classify_requests(bodies).count("background")
        if getattr(args, "record_reference", False):
            if not (args.output / "opencode-clean-reference.json").exists():
                save(args.output / "opencode-clean-reference.json",
                     build_reference(bodies, cwd=cwd.resolve(), model=args.model, prompt=PROMPTS[name]))
        elif args.isolated:
            row["isolation"] = validate_isolation(bodies, args.reference, cwd=cwd.resolve(),
                                                  model=args.model, prompt=PROMPTS[name])
        row["tool_results"] = validate_tool_results(name, tool_history(bodies), fixture)
    except Exception as error:
        if row["status"] == "pass":
            row.update(status="fail", error=f"real tool results: {error!r}")
        else:
            row["tool_results_error"] = repr(error)
    save(case / "result.json", row)
    print(f"{case.name}: {row['status']} {row['wall_ms']/1000:.1f}s, "
          f"{row.get('agent_requests', 0)} agent requests", flush=True)
    return row


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--mcp":
        serve_mcp(sys.argv[2])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--opencode", default="opencode")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--case", action="append", choices=tuple(PROMPTS),
                        help="Repeat to select tasks; default runs all")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--thinking", action="store_true",
                        help="Declare the model as reasoning, as opencode users do")
    parser.add_argument("--shared-environment", action="store_true",
                        help="Run in the caller's HOME, configuration and variables instead of the "
                             "default isolated per-case environment. The report is marked "
                             "NOT_FOR_ACCEPTANCE: the user's instructions and plugins reach the model")
    parser.add_argument("--record-reference", action="store_true",
                        help="Write <output>/opencode-clean-reference.json from this isolated run "
                             "instead of checking it against the stored reference. For review "
                             "before adding a new OpenCode version; NOT_FOR_ACCEPTANCE")
    args = parser.parse_args()
    require_assertions()
    args.isolated = not args.shared_environment
    if args.passes <= 0:
        parser.error("--passes must be positive: an empty acceptance run cannot pass")
    if args.record_reference and not args.isolated:
        parser.error("--record-reference needs the isolated environment")
    if args.isolated and inside_git_tree(args.output.resolve().parent):
        parser.error("the isolated run needs --output outside a git work tree: opencode would load "
                     "that repository's instructions and configuration")
    args.output.mkdir(parents=True, exist_ok=False)
    version = reference_version(
        probe_version(args.opencode, args.output, args.isolated, timeout=min(10, args.timeout))["version"])
    args.reference = load_reference(version) if args.isolated and not args.record_reference else None
    recorder = Recorder(args.base_url, args.output)
    import threading
    threading.Thread(target=recorder.serve_forever, daemon=True).start()
    rows = []
    try:
        for attempt in range(args.passes):
            for name in args.case or tuple(PROMPTS):
                rows.append(run_case(args, recorder, len(rows), name))
    finally:
        recorder.shutdown()
        report = {"opencode_version": version, "isolated": args.isolated,
                  "acceptance": "ELIGIBLE" if args.reference is not None else "NOT_FOR_ACCEPTANCE",
                  "background_requests": sum(row.get("background_requests", 0) for row in rows),
                  "cases": rows}
        if args.reference is None:
            report["note"] = ("не для приёмки: " + (
                "записан эталон чистого запуска, сверка с эталоном не выполнялась" if args.record_reference
                else "запуск в общем окружении пользователя, его инструкции и плагины могли попасть в запросы"))
        save(args.output / "report.json", report)
    failed = [row["case"] for row in rows if row["status"] != "pass"]
    print(json.dumps({"passed": len(rows) - len(failed), "failed": failed,
                      "acceptance": report["acceptance"],
                      "background_requests": report["background_requests"]}, ensure_ascii=False))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
