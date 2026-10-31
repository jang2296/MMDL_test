"""Native verifier-to-synthesis integration; authored fixtures are not model results."""
import copy
import hashlib
import json
import unittest

from fine_tuning.notebook_rl import notebook_synthesis as synthesis
from fine_tuning.notebook_rl.episode_notebook import solution_samples
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures


class TraceEvidenceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.graph_native("bfs")
        self.wrong = {"kind": "bfs", "steps": [{"node": "A", "new": ["C", "B"]},
                      {"node": "C", "new": []}, {"node": "B", "new": ["D"]}, {"node": "D", "new": []}]}
        right = {"kind": "bfs", "steps": [{"node": "A", "new": ["B", "C"]},
                 {"node": "B", "new": ["D"]}, {"node": "C", "new": []}, {"node": "D", "new": []}]}
        self.wrong_text = fixtures.graph_response(self.item.slots, self.wrong, "A,C,B,D")
        self.right_text = fixtures.graph_response(self.item.slots, right, "A,B,C,D")
        self.line = "Trace: " + json.dumps(self.wrong, separators=(",", ":"))

    def audit(self, text, contract=synthesis.BOUNDED_VERSION):
        tree = fixtures.tree_from(self.actor, self.verifier, [text] * 4 + [self.right_text] * 2)
        original = to_json(tree)
        audit = synthesis.audit_episode(tree, self.actor, self.verifier, contract=contract)
        self.assertEqual(to_json(tree), original)
        self.assertEqual(audit["attempts"][0]["trace"],
                         self.verifier.trace(self.actor.problem_id, text, "stop"))
        return tree, audit

    def test_horizontal_padding_reaches_synthesis_with_exact_evidence_and_original_issues(self):
        for contract in (synthesis.VERSION, synthesis.BOUNDED_VERSION):
            _, baseline = self.audit(self.wrong_text, contract)
            expected_checks = [{key: value for key, value in row.items() if key != "refs"}
                               for row in baseline["attempts"][0]["checks"]]
            for ending in ("\n", "\r\n"):
                for prefix, delimiter, suffix in (("", " ", ""), (" \t", "", ""),
                                                   ("", "\t \t", ""), ("", "", " \t"),
                                                   (" \t", "\t ", " \t")):
                    with self.subTest(contract=contract, ending=repr(ending), prefix=repr(prefix),
                                      delimiter=repr(delimiter), suffix=repr(suffix)):
                        line = prefix + "Trace:" + delimiter + self.line[len("Trace: "):] + suffix
                        text = self.wrong_text.replace(self.line, line)
                        text = text.replace("Observed facts:\n", "Observed facts: \t\n")
                        text = text.replace("Solution:\n", "Solution: \t\n \t\n").replace("\n", ending)
                        tree, audit = self.audit(text, contract)
                        first = audit["attempts"][0]
                        start = text.index(line)
                        evidence = {"line": line, "span": [start, start + len(line)],
                                    "sha256": hashlib.sha256(line.encode()).hexdigest()}
                        self.assertEqual(first["trace"]["trace"], evidence)
                        self.assertEqual(first["trace"]["text_sha256"], hashlib.sha256(text.encode()).hexdigest())
                        self.assertEqual([{key: value for key, value in row.items() if key != "refs"}
                                          for row in first["checks"]], expected_checks)
                        self.assertIn("BFS_NEW_NEIGHBORS", [row["rule"] for row in first["issues"]])
                        for row in first["checks"]:
                            if row["kind"] == "TRACE":
                                self.assertIn({key: evidence[key] for key in ("span", "sha256")}, row["refs"])
                        note = fixtures.note_for(tree, audit, self.item)
                        checked = synthesis.grade_note(json.dumps(note), "stop", tree, audit, self.verifier,
                                                       actor=self.actor, contract=contract)
                        self.assertEqual(checked["reward"], 1. if contract == synthesis.BOUNDED_VERSION else 0.)

    def test_invalid_trace_inputs_stay_unverified_in_synthesis(self):
        payload = self.line[len("Trace: "):]
        cases = (("\tTrace: {", "TRACE_FORMAT"),
                 ('\tTrace:{"kind":"bfs","kind":"bfs","steps":[]}', "TRACE_FORMAT"),
                 ('\tTrace:{"kind":"bfs","steps":NaN}', "TRACE_FORMAT"),
                 ('\tTrace:{"kind":"bfs","steps":7}', "TRACE_SCHEMA"),
                 ("\tTrace:" + payload + "\nTrace: " + payload, "TRACE_FORMAT"),
                 ("\vTrace:" + payload, "TRACE_FORMAT"),
                 ("\u2028Trace:" + payload, "TRACE_FORMAT"),
                 ("\tTrace:" + payload + " hidden prose", "TRACE_FORMAT"),
                 ("\tTrace:" + payload + "\nAlways reverse arrows.", "RESPONSE_SCOPE"))
        for line, rule in cases:
            for contract in (synthesis.VERSION, synthesis.BOUNDED_VERSION):
                with self.subTest(line=repr(line), contract=contract):
                    text = self.wrong_text.replace(self.line, line)
                    _, audit = self.audit(text, contract)
                    first = audit["attempts"][0]
                    self.assertIn(rule, first["trace"]["coverage"]["unverified"])
                    self.assertTrue(any(row["rule"] == rule for row in first["unverified"]))
                    self.assertFalse(first["trace_complete"])
                    self.assertFalse(audit["solution_trace_complete"])
                    if contract == synthesis.BOUNDED_VERSION:
                        self.assertTrue(audit["unknown_regions_accountable"])
                    else:
                        self.assertFalse(audit["dependency_regions_accountable"])

    def test_direct_consumer_rejects_tampered_report_and_evidence(self):
        text = self.wrong_text.replace(self.line, "\t" + self.line + " \t")
        tree, audit = self.audit(text)
        sample = solution_samples(tree)[0]
        report = audit["attempts"][0]["trace"]
        original = copy.deepcopy(report)
        synthesis._trace_checks(sample, report)
        self.assertEqual(report, original)
        for change in ("trim", "hash", "finish", "span_type", "checks"):
            with self.subTest(change=change):
                bad = copy.deepcopy(report)
                if change == "trim":
                    start, end = bad["trace"]["span"]
                    line = text[start + 1:end - 1]
                    bad["trace"] = {"line": line, "span": [start + 1, end - 1],
                                    "sha256": hashlib.sha256(line.encode()).hexdigest()}
                elif change == "hash":
                    bad["trace"]["sha256"] = "0" * 64
                elif change == "finish":
                    bad["finish_reason"] = "length"
                elif change == "span_type":
                    bad["trace"]["span"][0] = False
                else:
                    bad["checks"] = []
                with self.assertRaises(ValueError):
                    synthesis._trace_checks(sample, bad)


if __name__ == "__main__":
    unittest.main()
