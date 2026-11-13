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
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time

from pi_agent import Recorder, save
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
              "cat using the bash tool with a timeout of 5000, then read it back with the "
              "read tool and confirm its content in one sentence."),
    "edit": ("Read calc.py, fix the obvious bug with the edit tool, then run "
             "`python3 -c 'import calc; assert calc.add(2, 3) == 5; print(\"ok\")'` "
             "with the bash tool and report the result."),
    "mcp": ("Use the catalog MCP tools: look up record 42 (the integer id 42), look up "
            "the date 2026-10-05, and tag record 42 with the tag name urgent and weight 3. "
            "Then write the record title returned by the lookup into result.txt and reply "
            "with that title."),
    # The bash result depends on a per-run secret, so it cannot be produced
    # without really executing the fixture's check after a real fix.
    "verify": ("Read calc.py, fix the obvious bug with the edit tool, then run "
               "`python3 check.py` with the bash tool and reply with the exact line it prints."),
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


def validate_requests(rows, output):
    """Every agent turn stays native and reuses the previous one."""
    require_assertions()
    agent = []
    for row in rows:
        assert row.get("status") == 200 and not row.get("error"), row
        body = json.loads((output / f"request-{row['index']:04d}.json").read_text())
        if not body.get("tools"):
            continue  # opencode's title request has its own prompt
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
        subprocess.run([sys.executable, "-c",
                        "import calc; assert calc.add(2, 3) == 5 and calc.add(-4, 1) == -3"],
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
        ran(("bash",), ("cat", "hello.txt"), "bonjour")
        ran(("read",), ("hello.txt",), "bonjour")
    elif name == "edit":
        ran(("read",), ("calc.py",), "return a - b")
        # The run that asserts the fix printed "ok" on a line of its own; the
        # word inside other text, or a command without the assertion, is not it.
        ran(("bash",), ("calc", "assert"), re.compile(r"(?m)^\s*ok\s*$"), failed)
    elif name == "verify":
        ran(("read",), ("calc.py",), fixture["sentinel"])
        ran(("edit", "patch", "multiedit", "write"), ("calc.py",), "")
        ran(("bash",), ("check.py",), fixture["expected"], failed)
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
        "import hashlib\nimport calc\n"
        "assert calc.add(2, 3) == 5 and calc.add(-4, 1) == -3\n"
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
    return {**source, "OPENCODE_CONFIG": str(case / "opencode.json"),
            "XDG_CONFIG_HOME": str(case / "xdg-config"), "XDG_DATA_HOME": str(case / "xdg-data"),
            "XDG_STATE_HOME": str(case / "xdg-state"), "XDG_CACHE_HOME": str(case / "xdg-cache")}


def run_case(args, recorder, index, name):
    case = args.output / f"{index:02d}-{name}"
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
        "mcp": {"catalog": {"type": "local", "enabled": True,
                            "command": [sys.executable, str(Path(__file__).resolve()),
                                        "--mcp", str(calls_log)]}},
        "permission": {"*": "allow"},
    }
    save(case / "opencode.json", config)
    env = case_environment(case, args.isolated)
    # A private server: the default background service may belong to the
    # user's own opencode and carry their configuration.
    command = [args.opencode, "run", *(["--standalone"] if args.isolated else []),
               "--pure", "--auto", "--format", "json", "-m", f"gufo/{args.model}", PROMPTS[name]]
    save(case / "command.json", command)
    recorder.case = case.name
    row = {"case": case.name, "start_request": len(recorder.requests)}
    start = time.monotonic()
    proc = None
    try:
        with (case / "events.jsonl").open("w") as out, (case / "stderr.log").open("w") as err:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err,
                                    start_new_session=True)
            while proc.poll() is None:
                if time.monotonic() - start > args.timeout:
                    raise AssertionError("opencode task timeout")
                time.sleep(0.5)
            assert proc.returncode == 0, (case / "stderr.log").read_text()[-4000:]
        row["mcp_calls"] = validate_task(name, cwd, calls_log)
        row["status"] = "pass"
    except Exception as error:
        row.update(status="fail", error=repr(error))
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    row["wall_ms"] = (time.monotonic() - start) * 1000
    row["requests"] = recorder.requests[row["start_request"]:]
    try:
        row["agent_requests"] = len(validate_requests(row["requests"], args.output))
    except Exception as error:
        row.update(status="fail", error=f"HTTP/framing/cache validation: {error!r}")
    try:
        bodies = [json.loads((args.output / f"request-{item['index']:04d}.json").read_text())
                  for item in row["requests"]]
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
    parser.add_argument("--isolated", action="store_true",
                        help="Private opencode server, per-case HOME and a minimal environment; "
                             "requires --output outside any git work tree")
    args = parser.parse_args()
    require_assertions()
    if args.isolated and inside_git_tree(args.output.resolve().parent):
        parser.error("--isolated needs --output outside a git work tree: opencode would load "
                     "that repository's instructions and configuration")
    args.output.mkdir(parents=True, exist_ok=False)
    version = subprocess.run([args.opencode, "--version"], capture_output=True, text=True)
    save(args.output / "opencode-version.json", {"version": version.stdout.strip()})
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
        save(args.output / "report.json", {"cases": rows})
    failed = [row["case"] for row in rows if row["status"] != "pass"]
    print(json.dumps({"passed": len(rows) - len(failed), "failed": failed}))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
