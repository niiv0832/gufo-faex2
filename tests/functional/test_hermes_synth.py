import json
import re
import unittest

import hermes_synth


class HermesSynthTest(unittest.TestCase):
    def setUp(self):
        self.base = hermes_synth.load_base()
        self.cases = hermes_synth.load_cases()
        self.names = {tool["function"]["name"] for tool in self.base["tools"]}

    def test_shape(self):
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 12)
        for name in hermes_synth.CLASSES:
            self.assertGreaterEqual(sum(c["class"] == name for c in self.cases), 4, name)
        self.assertEqual(len(self.base["tools"]), 20)
        self.assertTrue(self.base["system"].startswith("You are Hermes Agent"))

    def test_requests_are_well_formed(self):
        for case in self.cases:
            body = hermes_synth.build_request(case, self.base)
            messages = body["messages"]
            self.assertEqual(messages[0]["role"], "system")
            self.assertIn(messages[-1]["role"], ("user", "tool"), case["id"])
            pending = set()
            for message in messages:
                for call in message.get("tool_calls", []):
                    self.assertIn(call["function"]["name"], self.names, case["id"])
                    json.loads(call["function"]["arguments"])
                    pending.add(call["id"])
                if message["role"] == "tool":
                    self.assertIn(message["tool_call_id"], pending, case["id"])
                    pending.discard(message["tool_call_id"])
            self.assertFalse(pending, case["id"])

    def test_no_machine_or_personal_markers(self):
        text = json.dumps([self.base, self.cases, hermes_synth.load_aux_title()],
                          ensure_ascii=False)
        self.assertIsNone(re.search(r"/Users/|/private/tmp|claude-\d+|@[\w.-]+\.\w+", text))

    def test_aux_title_is_constrained_json(self):
        aux = hermes_synth.load_aux_title()
        self.assertEqual(aux["response_format"]["type"], "json_schema")
        self.assertNotIn("tools", aux)


if __name__ == "__main__":
    unittest.main()
