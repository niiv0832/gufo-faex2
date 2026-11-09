"""Acceptance gates must reject plausible but wrong or incomplete evidence."""
import copy
import importlib.util
import unittest
import tempfile
import sys
import json
from pathlib import Path


class AcceptanceTest(unittest.TestCase):
    def api(self):
        path = Path(__file__).with_name("faex2_acceptance.py")
        self.assertTrue(path.exists(), "acceptance gates are not implemented")
        spec = importlib.util.spec_from_file_location("faex2_acceptance", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def result(self, text="391"):
        return {"status": "PASS", "http_status": 200, "text": text,
                "reasoning": "", "tools": [], "finish": "stop",
                "usage": {"prompt_tokens": 50, "completion_tokens": 10},
                "metrics": {"prompt_tokens": 50, "completion_tokens": 10,
                            "draft_tokens": 6, "draft_tokens_accepted": 4,
                            "prefill_tokens": 50, "cached_tokens": 0}}

    def test_wrong_answer_or_length_is_not_quality_pass(self):
        api = self.api()
        api.grade(self.result(), {"text": "391"}, "mtp")
        for patch in ({"text": "392"}, {"finish": "length"}, {"text": ""}):
            bad = {**self.result(), **patch}
            with self.assertRaises(ValueError):
                api.grade(bad, {"text": "391"}, "mtp")

    def test_mtp_return_requires_real_drafts_and_acceptance(self):
        api = self.api()
        for p, a in ((0, 0), (4, 0), (4, 5)):
            bad = self.result()
            bad["metrics"].update(draft_tokens=p, draft_tokens_accepted=a)
            with self.assertRaises(ValueError):
                api.grade(bad, {"text": "391", "draft_work": True}, "mtp")
        api.grade(self.result(), {"draft_work": True}, "mtp")

    def test_off_never_accepts_executed_drafts(self):
        with self.assertRaises(ValueError):
            self.api().grade(self.result(), {"text": "391"}, "off")

    def test_concurrent_marker_cannot_include_foreign_dialog(self):
        api = self.api()
        expect = {"contains": ["OWL_142"], "excludes": ["MOON_853"]}
        api.grade(self.result("OWL_142: короткое объяснение"), expect, "mtp")
        with self.assertRaises(ValueError):
            api.grade(self.result("OWL_142 MOON_853"), expect, "mtp")

    def test_long_context_and_cache_need_usage_not_string_length(self):
        api = self.api()
        r = self.result("KEY_A KEY_B KEY_C")
        with self.assertRaises(ValueError):
            api.grade(r, {"min_prompt_tokens": 110000}, "mtp")
        r["metrics"].update(prompt_tokens=115000, prefill_tokens=115000)
        api.grade(r, {"min_prompt_tokens": 110000, "max_prompt_tokens": 130560}, "mtp")
        with self.assertRaises(ValueError):
            api.grade(r, {"min_cached_tokens": 100000}, "mtp")
        r["metrics"].update(cached_tokens=110000, prefill_tokens=5000)
        api.grade(r, {"min_cached_tokens": 100000, "max_prefill_tokens": 5100}, "mtp")

    def test_visual_request_requires_full_fresh_prefill(self):
        api = self.api()
        api.grade(self.result("red"), {"text": "red", "fresh_prefill": True}, "mtp")
        for patch in ({"cached_tokens": 1}, {"prefill_tokens": 0}, {"prompt_tokens": 0}):
            bad = self.result("red")
            bad["metrics"].update(patch)
            with self.assertRaises(ValueError):
                api.grade(bad, {"text": "red", "fresh_prefill": True}, "mtp")

    def test_reasoning_budget_uses_response_usage(self):
        api = self.api()
        r = self.result("Ответ 7")
        r["reasoning"] = "Проверю уравнение."
        for count in (None, 513):
            r["usage"] = {"output_tokens_details": {"reasoning_tokens": count}}
            with self.assertRaises(ValueError):
                api.grade(r, {"reasoning": True, "reasoning_budget": 512}, "mtp")
        r["usage"]["output_tokens_details"]["reasoning_tokens"] = 128
        api.grade(r, {"reasoning": True, "reasoning_budget": 512}, "mtp")

    def test_expected_error_and_cancellation_are_explicit(self):
        api = self.api()
        r = {"status": "HTTP_ERROR", "http_status": 400,
             "error": {"code": "invalid_request", "message": "bad base64"}}
        api.grade(r, {"http_status": 400, "error_contains": "base64"}, "mtp")
        with self.assertRaises(ValueError):
            api.grade(r, {"text": "391"}, "mtp")
        api.grade({"status": "CANCELLED", "text": "начало", "http_status": 200},
                  {"cancelled": True}, "mtp")
        with self.assertRaises(ValueError):
            api.grade({"status": "FAIL", "text": "начало"}, {"cancelled": True}, "mtp")

    def test_rubric_cannot_hide_wrong_facts_with_high_total(self):
        api = self.api()
        good = {"correctness": 2, "instruction": 2, "russian_clarity": 1, "explanation": 1}
        self.assertTrue(api.rubric_pass(good, []))
        self.assertFalse(api.rubric_pass({**good, "correctness": 1, "explanation": 2}, []))
        self.assertFalse(api.rubric_pass(good, ["incorrect_fact"]))
        self.assertFalse(api.rubric_pass({**good, "explanation": 0}, []))

    def test_discovery_checks_body_without_generation_counters(self):
        api = self.api()
        result = {"status": "PASS", "http_status": 200, "response": {"status": "ok"}}
        api.grade(result, {"discovery": "health"}, "mtp")
        with self.assertRaises(ValueError):
            api.grade({**result, "response": {"status": "loading"}}, {"discovery": "health"}, "mtp")
        with self.assertRaises(ValueError):
            api.grade(result, {"discovery": "unknown"}, "mtp")

    def fake_transport(self, directory):
        script = Path(directory) / "transport.py"
        script.write_text('''import sys,json,time
from pathlib import Path
args=sys.argv
request=json.loads(Path(args[args.index('--request')+1]).read_text())
out=Path(args[args.index('--output')+1]);out.mkdir()
body=request['body'];text=body['messages'][0]['content']
if body.get('barrier'):
 gate=Path(body['barrier']);(gate/text).write_text('entered')
 end=time.monotonic()+3
 while len(list(gate.iterdir()))<2:
  if time.monotonic()>end:raise RuntimeError('second request never ran concurrently')
  time.sleep(.01)
result={'status':'PASS','http_status':200,'text':text,'reasoning':'','tools':[],
 'finish':'stop','usage':{},'metrics':{'draft_tokens':5,'draft_tokens_accepted':3}}
(out/'response.raw').write_text(text)
(out/'result.json').write_text(json.dumps(result))
''')
        return [sys.executable, str(script)]

    def test_parallel_runner_retains_distinct_results_and_real_overlap(self):
        api = self.api()
        self.assertTrue(hasattr(api, "Runner"), "bounded request runner is absent")
        with tempfile.TemporaryDirectory() as tmp:
            gate = Path(tmp) / "barrier"
            gate.mkdir()
            runner = api.Runner("http://127.0.0.1:13346", Path(tmp) / "out", "mtp",
                                transport_command=self.fake_transport(tmp))
            cases = [(name, {"method": "POST", "path": "/v1/chat/completions",
                             "body": {"messages": [{"role": "user", "content": name}],
                                      "barrier": str(gate)}}, {"text": name})
                     for name in ("OWL_142", "MOON_853")]
            results = runner.run_batch(cases)
            self.assertEqual([r["text"] for r in results], ["OWL_142", "MOON_853"])
            report = json.loads((Path(tmp) / "out/report.json").read_text())
            self.assertEqual({r["id"] for r in report["requests"]}, {"OWL_142", "MOON_853"})
            self.assertTrue(all(r["status"] == "PASS" for r in report["requests"]))

    def test_runner_failure_is_saved_and_duplicate_never_overwrites(self):
        api = self.api()
        self.assertTrue(hasattr(api, "Runner"), "bounded request runner is absent")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            runner = api.Runner("http://127.0.0.1:13346", out, "mtp",
                                transport_command=self.fake_transport(tmp))
            request = {"method": "POST", "path": "/v1/chat/completions",
                       "body": {"messages": [{"role": "user", "content": "392"}]}}
            with self.assertRaises(ValueError):
                runner.run_case("arithmetic", request, {"text": "391"})
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["requests"][0]["status"], "FAIL")
            self.assertEqual((out / "requests/arithmetic/response.raw").read_text(), "392")
            with self.assertRaises(ValueError):
                runner.run_case("arithmetic", request, {"text": "392"})
            self.assertEqual((out / "requests/arithmetic/response.raw").read_text(), "392")

    def test_followup_uses_actual_tool_identity_and_retains_history(self):
        api = self.api()
        self.assertTrue(hasattr(api, "tool_followup"), "tool continuation builder absent")
        body = {"messages": [{"role": "user", "content": "lookup"}], "model": "qwen3.6-35b-a3b"}
        result = {"text": "", "tools": [{"id": "call_actual_93", "type": "function",
                   "function": {"name": "lookup", "arguments": '{"topic":"астрономия"}'}}]}
        followup = api.tool_followup(body, result, {"answer": "ПОЛЯРИС_742"})
        self.assertEqual(len(body["messages"]), 1)
        self.assertEqual(followup["messages"][-1]["tool_call_id"], "call_actual_93")
        self.assertEqual(json.loads(followup["messages"][-1]["content"]), {"answer": "ПОЛЯРИС_742"})
        self.assertTrue(followup["stream"])

    def test_b2_cases_include_independent_visual_inputs_and_expected_error(self):
        api = self.api()
        self.assertTrue(hasattr(api, "b2_visual_cases"), "B2 visual requests absent")
        cases = {name: (request, expected) for name, request, expected in api.b2_visual_cases()}
        for name, mime in (("vision-png", "image/png"), ("vision-jpeg", "image/jpeg"),
                           ("vision-webp_lossless", "image/webp"), ("vision-webp_lossy", "image/webp")):
            req, expect = cases[name]
            self.assertEqual(expect["text"], "red")
            self.assertTrue(req["body"]["messages"][0]["content"][0]["image_url"]["url"].startswith("data:"+mime))
        self.assertEqual(cases["vision-text"][1]["text"], "GUFO 742")
        for name in ("vision-two-forward", "vision-two-reverse"):
            parts = cases[name][0]["body"]["messages"][0]["content"]
            self.assertNotEqual(parts[0]["image_url"]["url"], parts[1]["image_url"]["url"])
        self.assertEqual(cases["vision-bad-base64"][1]["http_status"], 400)
        self.assertEqual(cases["vision-recovery"][1]["text"], "blue")


if __name__ == "__main__":
    unittest.main()
