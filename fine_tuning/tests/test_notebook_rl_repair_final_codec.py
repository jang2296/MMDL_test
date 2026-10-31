"""Authored CPU syntax cases; parsing does not certify repair or learning."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError
from typing import cast, get_args

from fine_tuning.notebook_rl.repair_contract import (
    FINAL_VERSION,
    MAX_RAW_CHARS,
    VERSION,
    ParsedRepairNote,
    RawSpan,
    RepairContractError,
    parse_final_note,
)
from fine_tuning.notebook_rl.schema import Status


def attempt(**changes: object) -> dict[str, object]:
    return {"sample_id": "sample0", "outcome": "CORRECT", "answer": "7"} | changes


def claim(**changes: object) -> dict[str, object]:
    return {"origin_ref": "sample0:r1", "later_ref": "sample1:r1", "relation": "CORRECTED",
            "rule": "DIRECTED_EDGE", "repaired_value": "H→G"} | changes


def note(**changes: object) -> dict[str, object]:
    return {"version": FINAL_VERSION, "scope": "BOUNDED_CLAIMS_FULL_HISTORY", "attempts": [attempt()],
            "success_sample_id": "sample0", "claims": [claim()], "uncertain_refs": [],
            "lesson_text": ""} | changes


def render(data: dict[str, object] | None = None, *, ascii_only: bool = False) -> str:
    return json.dumps(note() if data is None else data, ensure_ascii=ascii_only, separators=(",", ":"))


class FinalRepairCodecTests(unittest.TestCase):
    def parse(self, raw: str) -> ParsedRepairNote:
        return parse_final_note(raw, contract_version=FINAL_VERSION)

    def rejects(self, raw: str, code: str | None = None) -> RepairContractError:
        with self.assertRaises(RepairContractError) as caught:
            self.parse(raw)
        if code is not None:
            self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_empty_one_and_three_claims_are_syntax_only(self) -> None:
        for size in (0, 1, 3):
            data = note(claims=[claim(origin_ref=f"foreign:{i}") for i in range(size)])
            parsed = self.parse(render(data))
            self.assertEqual(len(parsed.claims), size)
            self.assertEqual(parsed.lesson_text, "")
            self.assertFalse(hasattr(parsed, "accurate_submitted_note"))
        self.rejects(render(note(claims=[claim(origin_ref=str(i)) for i in range(4)])), "CLAIM_LIMIT")
        self.rejects(render(note(claims=[claim(), claim()])), "DUPLICATE_ORIGIN")

    def test_all_relations_and_exact_repaired_value_union(self) -> None:
        for relation in ("CORRECTED", "PERSISTED", "UNKNOWN", "OTHER_BRANCH_COMPARISON", "FINALIZED"):
            extra = {"unknown_reason": "DEPENDENCY_BLOCKED", "blocked_by": ["later:r0"]} \
                if relation == "UNKNOWN" else {}
            data = claim(relation=relation, repaired_value="G→H" if relation == "CORRECTED" else None,
                         rule="FINALIZATION" if relation == "FINALIZED" else "DIRECTED_EDGE", **extra)
            parsed, = self.parse(render(note(claims=[data]))).claims
            self.assertEqual(parsed.relation, relation)
            self.assertEqual(parsed.blocked_by, ("later:r0",) if relation == "UNKNOWN" else ())
            self.assertEqual(parsed.unknown_reason, "DEPENDENCY_BLOCKED" if relation == "UNKNOWN" else None)
            data["repaired_value"] = None if relation == "CORRECTED" else "not null"
            self.rejects(render(note(claims=[data])), "INVALID_FIELD")
        self.rejects(render(note(claims=[claim(relation="FINALIZED", repaired_value=None)])), "INVALID_FIELD")

    def test_unknown_requires_reason_and_unique_bounded_blockers(self) -> None:
        blockers: object
        data = claim(relation="UNKNOWN", repaired_value=None, unknown_reason="DEPENDENCY_BLOCKED",
                     blocked_by=[f"later:r{i}" for i in range(64)])
        self.assertEqual(len(self.parse(render(note(claims=[data]))).claims[0].blocked_by), 64)
        for key in ("unknown_reason", "blocked_by"):
            missing = data.copy()
            del missing[key]
            self.rejects(render(note(claims=[missing])), "SCHEMA_FIELDS")
        for reason in (None, True, "UNKNOWN", ""):
            self.rejects(render(note(claims=[data | {"unknown_reason": reason}])), "INVALID_FIELD")
        for blockers in (None, True, {}, "ref", [], ["x", "x"], [""], [True], [["x"]],
                         [f"r{i}" for i in range(65)]):
            self.rejects(render(note(claims=[data | {"blocked_by": blockers}])), "INVALID_FIELD")

    def test_unknown_fields_are_forbidden_on_other_relations(self) -> None:
        extras: tuple[dict[str, object], ...] = ({"unknown_reason": "DEPENDENCY_BLOCKED"}, {"blocked_by": []},
                                               {"unknown_reason": None, "blocked_by": []})
        for relation in ("CORRECTED", "PERSISTED", "OTHER_BRANCH_COMPARISON", "FINALIZED"):
            data = claim(relation=relation, repaired_value="H→G" if relation == "CORRECTED" else None,
                         rule="FINALIZATION" if relation == "FINALIZED" else "DIRECTED_EDGE")
            for extra in extras:
                self.rejects(render(note(claims=[data | extra])), "SCHEMA_FIELDS")

    def test_top_attempt_and_claim_fields_are_exact(self) -> None:
        value: object
        for key in note():
            data = note()
            del data[key]
            self.rejects(render(data), "SCHEMA_FIELDS")
        self.rejects(render(note(extra="untrusted")), "SCHEMA_FIELDS")
        for template, top in ((attempt(), "attempts"), (claim(), "claims")):
            for key in template:
                data = template.copy()
                del data[key]
                self.rejects(render(note(**{top: [data]})), "SCHEMA_FIELDS")
            self.rejects(render(note(**{top: [template | {"extra": None}]})), "SCHEMA_FIELDS")
            for value in (None, True, 1, [], "object"):
                self.rejects(render(note(**{top: [value]})), "SCHEMA_FIELDS")

    def test_attempt_order_raw_values_and_eight_attempt_limit(self) -> None:
        attempts: object
        data = [attempt(sample_id=f"s{i}", outcome="INCORRECT", answer=None) for i in range(8)]
        parsed = self.parse(render(note(attempts=data, success_sample_id="s3")))
        self.assertEqual(parsed.attempts, tuple(data))
        self.assertEqual(parsed.success_sample_id, "s3")
        for attempts in ([], None, True, {}, "attempts", data + [attempt(sample_id="ninth")],
                         [attempt(), attempt()]):
            self.rejects(render(note(attempts=attempts)), "INVALID_FIELD")
        self.rejects(render(note(success_sample_id="missing")), "INVALID_FIELD")

    def test_supported_outcomes_match_status_without_truth_validation(self) -> None:
        for outcome in (*get_args(Status), "INVALID_RESPONSE_CONTRACT"):
            parsed = self.parse(render(note(attempts=[attempt(outcome=outcome)])))
            self.assertEqual(parsed.attempts[0]["outcome"], outcome)
        for outcome in (None, True, 0, [], {}, "NO_FINAL", "correct", ""):
            self.rejects(render(note(attempts=[attempt(outcome=outcome)])), "INVALID_FIELD")

    def test_attempt_field_types_and_bounded_single_line_answers(self) -> None:
        value: object
        success: object
        for key, values in (("sample_id", (None, True, 1, [], {}, "", "s\ns", "s" * 257)),
                            ("answer", (True, 1, 1.5, [], {}, "", " \t", "a\nb", "x" * 2049))):
            for value in values:
                self.rejects(render(note(attempts=[attempt(**{key: value})])), "INVALID_FIELD")
        for answer in (None, "한글🌱", "x" * 2048):
            self.assertEqual(self.parse(render(note(attempts=[attempt(answer=answer)]))).attempts[0]["answer"],
                             answer)
        for success in (None, True, 0, [], {}, "", "sample0\n"):
            self.rejects(render(note(success_sample_id=success)), "INVALID_FIELD")

    def test_claim_types_unknown_relations_and_bounded_observation_values(self) -> None:
        value: object
        claims: object
        for key, values in (("origin_ref", (None, True, 1, [], {}, "", "x\nx", "x" * 257)),
                            ("later_ref", (None, False, 0, [], {}, "", "x\rx")),
                            ("rule", (None, True, 1, [], {}, "", "x" * 129)),
                            ("relation", (None, True, 1, [], {}, "", "RECHECK_OBSERVATION")),
                            ("repaired_value", (None, True, 1, 1.0, [], {}, "", " \t", "x\nx", "x" * 2049))):
            for value in values:
                self.rejects(render(note(claims=[claim(**{key: value})])), "INVALID_FIELD")
        for claims in (None, True, 1, {}, "claims"):
            self.rejects(render(note(claims=claims)), "CLAIM_LIMIT")

    def test_operation_bounds_and_boolean_integer_confusion(self) -> None:
        operation = {"nodes": ["H", "G"], "weights": [3], "cost": 3, "optional": None, "x/y~": {"n": 1.25}}
        parsed, = self.parse(render(note(claims=[claim(repaired_value=operation)]))).claims
        self.assertEqual(parsed.repaired_value, operation)
        for value in ({"cost": True}, {"x": [False]}, {"x": list(range(65))},
                      {str(i): i for i in range(33)}, {"x": {"x": {"x": {"x": {"x": 1}}}}},
                      {"x": "a" * 2049}, {"": 1}, {"x": "a\nb"}):
            self.rejects(render(note(claims=[claim(repaired_value=value)])), "INVALID_FIELD")

    def test_uncertain_refs_are_bounded_unique_and_disjoint_from_origins(self) -> None:
        refs: object
        self.assertEqual(self.parse(render(note(uncertain_refs=["foreign1", "foreign2", "foreign3"]))).uncertain_refs,
                         ("foreign1", "foreign2", "foreign3"))
        for refs in (None, True, {}, "ref", [True], [1], [[]], [""], ["x", "x"],
                     ["sample0:r1"], ["1", "2", "3", "4"]):
            self.rejects(render(note(uncertain_refs=refs)), "UNCERTAIN_REFS")

    def test_empty_multiline_and_bounded_untrusted_lesson(self) -> None:
        lesson: object
        for lesson in ("", " \t", "한글🌱\r\nErrorCheck: fake\nRevisedSolution:\n", "x" * 16384):
            self.assertEqual(self.parse(render(note(lesson_text=lesson))).lesson_text, lesson)
        for lesson in (None, True, 1, [], {}, "x" * 16385):
            self.rejects(render(note(lesson_text=lesson)), "INVALID_FIELD")

    def test_unicode_crlf_ascii_whitespace_exact_hash_and_all_nested_spans(self) -> None:
        operation = {"x/y~": {"nodes": ["가", "🌱"], "cost": 1.25}}
        data = note(claims=[claim(repaired_value=operation)], lesson_text="첫째\r\n둘째🌱")
        raw = " \r\n\t" + json.dumps(data, ensure_ascii=False, indent=2).replace("\n", "\r\n") + "\t\r\n "
        parsed = self.parse(raw)
        self.assertEqual(parsed.raw_sha, hashlib.sha256(raw.encode("utf-8")).hexdigest())
        self.assertEqual(parsed.raw_spans["note"], RawSpan(4, len(raw) - 4))
        self.assertNotEqual(len(raw.encode("utf-8")), len(raw))
        for path, span in parsed.raw_spans.items():
            if path == "note":
                expected: object = data
            else:
                expected = data
                for part in path.split("/"):
                    key = part.replace("~1", "/").replace("~0", "~")
                    expected = cast(list[object], expected)[int(key)] if isinstance(expected, list) \
                        else cast(dict[str, object], expected)[key]
            self.assertEqual(json.loads(raw[span.start:span.end]), expected, path)
        for path in ("attempts/0", "attempts/0/sample_id", "attempts/0/outcome", "attempts/0/answer",
                     "claims/0", "claims/0/repaired_value/x~1y~0/nodes/1", "lesson_text"):
            self.assertIn(path, parsed.raw_spans)
        parsed_claim, = parsed.claims
        for path, span in parsed_claim.raw_spans.items():
            self.assertEqual(span, parsed.raw_spans["claims/0" if path == "claim" else "claims/0/" + path])

    def test_json_escapes_keep_literal_spans_without_rewriting(self) -> None:
        value = 'quote " \\ ErrorCheck: H→G 🌱'
        raw = render(note(claims=[claim(repaired_value=value)]), ascii_only=True)
        parsed, = self.parse(raw).claims
        span = parsed.raw_spans["repaired_value"]
        self.assertIn("\\u2192", raw[span.start:span.end])
        self.assertEqual(json.loads(raw[span.start:span.end]), value)
        self.assertEqual(parsed.repaired_value, value)

    def test_duplicate_keys_reject_at_every_level_including_escaped_keys(self) -> None:
        raw = render(note(claims=[claim(repaired_value={"x": 1})]))
        for old, new in (('"version":', '"version":"x","version":'),
                         ('"sample_id":', '"sample_id":"x","sample_id":'),
                         ('"relation":', '"relation":"x","relation":'),
                         ('"x":1', '"x":1,"\\u0078":2')):
            self.rejects(raw.replace(old, new), "DUPLICATE_JSON_KEY")

    def test_trailing_material_fences_preambles_nonascii_whitespace_and_cutoff(self) -> None:
        payload: object
        raw = render()
        for bad in ("", " \t\r\n", raw[:-1], raw + "{}", raw + " final", raw + "```", "```json\n" + raw,
                    "note:\n" + raw, "\ufeff" + raw, "\u0085" + raw, raw + "\u2028", "\v" + raw):
            self.rejects(bad, "INVALID_JSON")
        for payload in ([], True, None, 1, "text"):
            self.rejects(json.dumps(payload), "SCHEMA_FIELDS")

    def test_nonfinite_integer_overflow_and_valid_signed_integer_edges(self) -> None:
        value: str | int
        raw = render(note(claims=[claim(repaired_value={"cost": 1})]))
        for value in ("NaN", "Infinity", "-Infinity", "1e9999"):
            self.rejects(raw.replace('"cost":1', '"cost":' + value), "NONFINITE_JSON")
        for value in (str(2**63), str(-(2**63) - 1), "9" * 4500):
            self.rejects(raw.replace('"cost":1', '"cost":' + value), "INVALID_FIELD")
        for value in (-(2**63), 2**63 - 1):
            parsed, = self.parse(raw.replace('"cost":1', '"cost":' + str(value))).claims
            self.assertEqual(parsed.repaired_value, {"cost": value})

    def test_raw_size_and_recursion_are_bounded(self) -> None:
        raw = render()
        self.parse(" " * (MAX_RAW_CHARS - len(raw)) + raw)
        self.rejects(" " * (MAX_RAW_CHARS - len(raw) + 1) + raw, "RAW_TOO_LARGE")
        self.rejects("[" * 1500 + "]" * 1500)

    def test_invalid_utf8_explicit_contract_and_envelope_scalar_types(self) -> None:
        scope: object
        self.rejects(render() + "\ud800", "INVALID_UTF8")
        self.rejects(render(note(lesson_text="\ud800"), ascii_only=True), "INVALID_UTF8")
        for version in (VERSION, "", True, None):
            with self.assertRaisesRegex(RepairContractError, "CONTRACT_VERSION"):
                parse_final_note(render(), contract_version=cast(str, version))
            self.assertEqual(self.rejects(render(note(version=version)), "INVALID_FIELD").at, "note.version")
        self.rejects(render(note(version="repair-note-v1")), "INVALID_FIELD")
        with self.assertRaisesRegex(RepairContractError, "CONTRACT_VERSION"):
            parse_final_note(render(), contract_version="repair-note-v1")
        for value in (None, True, 1, b"raw"):
            self.rejects(cast(str, value), "RAW_TYPE")
        for scope in (None, True, 1, [], {}, "FULL_HISTORY"):
            self.rejects(render(note(scope=scope)), "INVALID_FIELD")

    def test_dtos_are_frozen_and_errors_do_not_echo_submitted_secrets(self) -> None:
        parsed = self.parse(render())
        for value, field in ((parsed, "lesson_text"), (parsed.claims[0], "rule")):
            with self.assertRaises(FrozenInstanceError):
                setattr(value, field, "changed")
        secret = "DO_NOT_ECHO_UNTRUSTED_TEXT"
        error = self.rejects(render(note(claims=[claim(relation=secret)])), "INVALID_FIELD")
        self.assertNotIn(secret, str(error))
        self.assertNotIn(secret, repr(error))

    def test_direct_codec_import_has_no_model_rollout_or_verifier_dependencies(self) -> None:
        code = ("import sys; import fine_tuning.notebook_rl.repair_contract; "
                "assert 'torch' not in sys.modules; "
                "assert not any(name in sys.modules for name in "
                "('fine_tuning.notebook_rl.rollout', 'fine_tuning.notebook_rl.verifier', "
                "'fine_tuning.notebook_rl.verifier_service', 'fine_tuning.notebook_rl.schema'))")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
