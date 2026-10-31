"""Authored CPU codec counterexamples; no model, gold, or benchmark inputs."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from fine_tuning.notebook_rl.repair_contract import (
    BOUNDARY_POLICY,
    MAX_CHECK_CHARS,
    MAX_CLAIM_CHARS,
    MAX_RAW_CHARS,
    ORIGIN_VERSION,
    VERSION,
    RawSpan,
    RepairContractError,
    claim_token_ownership,
    parse_error_check,
    parse_retry_sections,
    project_solution,
    public_attempt_outcome,
    public_origin_inventory,
    public_origins,
)
from fine_tuning.notebook_rl.repair_tokens import emitted_patch_spans, span_token_ownership
from fine_tuning.notebook_rl.schema import VerifierResult


def claim(**changes: object) -> dict:
    return {"origin_ref": "sample0/obs/1", "intent": "RECHECK_OBSERVATION",
            "rule": "DIRECTED_EDGE", "proposed_value": "H→G"} | changes


def payload(claims: object = None, **changes: object) -> dict:
    return {"version": VERSION, "claims": [claim()] if claims is None else claims,
            "uncertain_refs": []} | changes


def retry(data: dict | None = None, *, body: str = "Solution:\nFinal answer: 7\n",
          newline: str = "\n") -> str:
    text = json.dumps(payload() if data is None else data, ensure_ascii=False, separators=(",", ":"))
    return "ErrorCheck: " + text + newline + "RevisedSolution:" + newline + body


class ChunkTokenizer:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool,
               clean_up_tokenization_spaces: bool) -> str:
        return "".join(self.chunks[index] for index in token_ids)


class UnstableTokenizer(ChunkTokenizer):
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool,
               clean_up_tokenization_spaces: bool) -> str:
        return self.chunks[0] if len(token_ids) == 2 else "�"


class PublicOnly:
    """Any accidental result/reward/private-field read fails the test immediately."""

    def __init__(self, sample_id: str, text: object, parent: str | None = None, finish: str = "stop"):
        object.__setattr__(self, "_public", {"sample_id": sample_id, "text": text,
                                           "parent_sample_id": parent, "finish_reason": finish})

    def __getattribute__(self, name: str):
        values = object.__getattribute__(self, "_public")
        if name not in values:
            raise AssertionError("nonpublic attribute access: " + name)
        return values[name]


def native_body(observations: str = "dir[e4] = H->G\nweight[e4] = 3\n", trace: dict | None = None) -> str:
    trace = ({"kind": "shortest", "nodes": ["H", "G"], "edge_ids": ["e4"],
              "weights": [3], "cost": 3} if trace is None else trace)
    return ("Observed facts:\n" + observations + "Solution:\nTrace: "
            + json.dumps(trace, separators=(",", ":")) + "\nFinal answer: 3\n")


class RepairContractTests(unittest.TestCase):
    def sections(self, raw: str):
        return parse_retry_sections(raw, contract_version=VERSION)

    def rejects(self, raw: str, code: str | None = None) -> RepairContractError:
        with self.assertRaises(RepairContractError) as caught:
            self.sections(raw)
        if code is not None:
            self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_unicode_crlf_raw_hash_offsets_and_unmodified_projection(self):
        body = "Observed facts:\r\n  표[가] = H→G 🌱\r\nSolution:\r\nFinal answer: 7\r\n  \t"
        raw = retry(newline="\r\n", body=body)
        sections = self.sections(raw)
        self.assertEqual(sections.raw_sha, hashlib.sha256(raw.encode("utf-8")).hexdigest())
        self.assertEqual(raw[sections.check_span.start:sections.check_span.end],
                         raw.removeprefix("ErrorCheck: ").split("\r\nRevisedSolution:", 1)[0])
        self.assertEqual(sections.solution_span, RawSpan(raw.index(body), len(raw)))
        self.assertNotEqual(len(raw[:sections.solution_span.start].encode()), sections.solution_span.start)
        self.assertEqual(project_solution(raw, sections), body)
        parsed, = parse_error_check(raw, sections)
        self.assertEqual(parsed.proposed_value, "H→G")
        self.assertEqual(sections.claim_spans, (parsed.raw_spans["claim"],))
        for field in ("origin_ref", "intent", "rule", "proposed_value"):
            span = parsed.raw_spans[field]
            self.assertEqual(json.loads(raw[span.start:span.end]), getattr(parsed, field))

    def test_pretty_json_escaped_literals_and_fake_headers_are_not_boundaries(self):
        value = 'quoted "ErrorCheck:" and RevisedSolution: \\ Final answer: 99'
        data = payload([claim(proposed_value=value)])
        text = json.dumps(data, ensure_ascii=True, indent=2)
        raw = "ErrorCheck:\t" + text + " \t\r\nRevisedSolution:\t\r\nSolution:\nFinal answer: 7"
        sections = self.sections(raw)
        parsed, = parse_error_check(raw, sections)
        self.assertEqual(parsed.proposed_value, value)
        span = parsed.raw_spans["proposed_value"]
        self.assertIn('\\"ErrorCheck:', raw[span.start:span.end])
        self.assertEqual(project_solution(raw, sections), "Solution:\nFinal answer: 7")

    def test_ascii_blank_lines_keep_raw_check_spans_and_shift_native_token_ownership(self):
        body = native_body("dir[e4] = H→G\nweight[e4] = 3\n") + " \t\r\n"
        original = retry(body=body)
        baseline = self.sections(original)
        claims = parse_error_check(original, baseline)
        for separator in ("\n", "\r\n", "\n\n", "\r\n\r\n", " \t\n \t\r\n\t\n"):
            with self.subTest(separator=separator):
                raw = original.replace("\nRevisedSolution:", separator + "RevisedSolution:", 1)
                sections = self.sections(raw)
                delta = len(raw) - len(original)
                self.assertEqual(sections.raw_sha, hashlib.sha256(raw.encode()).hexdigest())
                self.assertEqual((sections.check_span, sections.claim_spans),
                                 (baseline.check_span, baseline.claim_spans))
                self.assertEqual(parse_error_check(raw, sections), claims)
                self.assertEqual(sections.solution_span,
                                 RawSpan(baseline.solution_span.start + delta, len(raw)))
                self.assertEqual(project_solution(raw, sections), body)
                tokenizer, ids = ChunkTokenizer(list(raw)), list(range(len(raw)))
                owned, = claim_token_ownership(raw, sections, tokenizer, ids)
                self.assertEqual(owned["token_indices"], list(range(claims[0].raw_spans["claim"].start,
                                                                    claims[0].raw_spans["claim"].end)))
                for kind, rule, locus in (("OBSERVATION", "OBSERVATION_MATCH", "dir[e4]"),
                                         ("TRACE", "COST_SUM", "cost")):
                    before = emitted_patch_spans(original, baseline, kind=kind, rule=rule, locus=locus)
                    after = emitted_patch_spans(raw, sections, kind=kind, rule=rule, locus=locus)
                    assert before is not None and after is not None
                    self.assertEqual(after, tuple(RawSpan(span.start + delta, span.end + delta) for span in before))
                    ownership = span_token_ownership(raw, ids, tokenizer, after)
                    self.assertEqual(ownership["token_indices"],
                                     [index for span in after for index in range(span.start, span.end)])
                    self.assertEqual(ownership["excluded_crossing_token_indices"], [])

    def test_boundary_policy_is_explicitly_bound_by_repair_notebook_contract(self):
        from fine_tuning.notebook_rl.repair_notebook import contract

        self.assertEqual(BOUNDARY_POLICY, "ascii-blank-lines-before-native-header-v2")
        self.assertEqual(contract()["retry_boundary_policy"], BOUNDARY_POLICY)
        self.assertEqual(contract()["retry_contract"], VERSION)

    def test_empty_and_three_claims_unknown_and_uncertain_refs(self):
        data = payload([], uncertain_refs=["foreign/ref", "future/ref", "other/ref"])
        raw = retry(data)
        sections = self.sections(raw)
        self.assertEqual(sections.uncertain_refs, tuple(data["uncertain_refs"]))
        self.assertEqual(parse_error_check(raw, sections), ())
        data = payload([claim(), claim(origin_ref="op", intent="RECHECK_OPERATION",
                                       rule="COST_SUM", proposed_value={"cost": 3}),
                        claim(origin_ref="unknown", intent="UNKNOWN", proposed_value=None)])
        raw = retry(data)
        self.assertEqual(len(parse_error_check(raw, self.sections(raw))), 3)

    def test_operation_scalar_and_container_spans_are_original_json(self):
        operation = {"nodes": ["A", "B"], "weights": [3], "cost": 3,
                     "optional": None, "x/y~": {"number": 1.25}}
        raw = retry(payload([claim(intent="RECHECK_OPERATION", proposed_value=operation)]))
        parsed, = parse_error_check(raw, self.sections(raw))
        for path, expected in (("proposed_value/nodes/0", "A"), ("proposed_value/cost", 3),
                               ("proposed_value/optional", None), ("proposed_value/x~1y~0/number", 1.25)):
            span = parsed.raw_spans[path]
            self.assertEqual(json.loads(raw[span.start:span.end]), expected)
        self.assertEqual(parsed.proposed_value, operation)

    def test_duplicate_actual_headers_preambles_fences_and_bad_newlines(self):
        raw = retry()
        for bad in (" " + raw, "\ufeff" + raw, "```\n" + raw, raw + "ErrorCheck: {}\n",
                    raw + "  RevisedSolution:\n", raw.replace("\nRevisedSolution:", " RevisedSolution:"),
                    raw.replace("\nRevisedSolution:", "\u2028RevisedSolution:"),
                    raw.replace("RevisedSolution:\n", "RevisedSolution: inline body\n"),
                    raw.replace("ErrorCheck: {", "ErrorCheck:\n{"), retry(body=" \t\r\n"),
                    raw + "\u2028ErrorCheck: {}", raw + "\rRevisedSolution:"):
            with self.subTest(bad=bad[:25]):
                self.rejects(bad, "SECTION_HEADER")
        for separator in ("", "\r", "\nprose\n", "\n\tprose\n", "\n ", "\n\t", "\n\v\n",
                          "\n\f\n", "\n\x85\n", "\n\u2028\n", "\n\u2029\n", "\n\u00a0\n"):
            with self.subTest(separator=separator):
                self.rejects(raw.replace("\nRevisedSolution:", separator + "RevisedSolution:"), "SECTION_HEADER")

    def test_json_cutoff_trailing_value_and_duplicate_keys_at_every_level(self):
        raw = retry()
        for bad in (raw.replace('"uncertain_refs":[]}', '"uncertain_refs":[]'),
                    raw.replace('\nRevisedSolution:', '{}\nRevisedSolution:'),
                    raw.replace('"version":', '"version":"repair-check-v1","version":'),
                    raw.replace('"rule":', '"rule":"X","rule":'),
                    retry(payload([claim(intent="RECHECK_OPERATION", proposed_value={"x": 1})])).replace(
                        '"x":1', '"x":1,"\\u0078":2')):
            with self.subTest(bad=bad[:30]):
                self.rejects(bad)

    def test_schema_fields_missing_extra_and_invalid_envelope_types(self):
        value: Any
        for key in payload():
            data = payload()
            del data[key]
            self.rejects(retry(data), "SCHEMA_FIELDS")
        self.rejects(retry(payload(extra="secret")), "SCHEMA_FIELDS")
        for key in claim():
            value = claim()
            del value[key]
            self.rejects(retry(payload([value])), "SCHEMA_FIELDS")
        self.rejects(retry(payload([claim(extra=0)])), "SCHEMA_FIELDS")
        for value in (True, 1, {}, "claims"):
            self.rejects(retry(payload(claims=value)), "CLAIM_LIMIT")
        for value in (None, 1, True, [], "claim"):
            self.rejects(retry(payload([value])), "SCHEMA_FIELDS")

    def test_invalid_claim_types_and_intents(self):
        value: Any
        for key, values in (("origin_ref", (None, True, 1, [], {}, "", "\nref")),
                            ("rule", (None, False, 0, {}, [], "", "A\u2028B")),
                            ("intent", (None, True, 1, {}, [], "CORRECTED")),
                            ("proposed_value", (None, True, 3, {}, [], "", " \t", "a\nb"))):
            for value in values:
                with self.subTest(key=key, value=value):
                    self.rejects(retry(payload([claim(**{key: value})])), "INVALID_FIELD")
        self.rejects(retry(payload([claim(intent="UNKNOWN", proposed_value="value")])), "INVALID_FIELD")
        for value in (None, True, 3, [], "operation", {}, {"cost": True}, {"x": [False]}):
            self.rejects(retry(payload([claim(intent="RECHECK_OPERATION", proposed_value=value)])),
                         "INVALID_FIELD")

    def test_claim_and_uncertain_uniqueness_limits_and_disjointness(self):
        refs: Any
        self.rejects(retry(payload([claim()] * 2)), "DUPLICATE_ORIGIN")
        self.rejects(retry(payload([claim(origin_ref=str(i)) for i in range(4)])), "CLAIM_LIMIT")
        for refs in (None, "ref", {}, [True], [1], [""], ["x", "x"], ["sample0/obs/1"],
                     ["1", "2", "3", "4"]):
            self.rejects(retry(payload(uncertain_refs=refs)), "UNCERTAIN_REFS")

    def test_nonfinite_numbers_integer_bounds_and_boolean_integer_confusion(self):
        value: Any
        raw = retry(payload([claim(intent="RECHECK_OPERATION", proposed_value={"cost": 1})]))
        for value in ("NaN", "Infinity", "-Infinity", "1e9999"):
            self.rejects(raw.replace('"cost":1', '"cost":' + value), "NONFINITE_JSON")
        for value in (str(2**63), str(-(2**63) - 1), "9" * 4500):
            self.rejects(raw.replace('"cost":1', '"cost":' + value), "INVALID_FIELD")
        for value in (-(2**63), 2**63 - 1):
            text = raw.replace('"cost":1', '"cost":' + str(value))
            parsed, = parse_error_check(text, self.sections(text))
            self.assertEqual(parsed.proposed_value, {"cost": value})
        invalid_bounds: tuple[tuple[Any, Any], ...] = ((True, 1), (0, False), (1.0, 2), (-1, 2), (2, 1))
        for start, end in invalid_bounds:
            with self.assertRaises(RepairContractError):
                RawSpan(start, end)

    def test_bounded_payloads_containers_and_recursion(self):
        value: Any
        self.rejects(retry(body="x" * MAX_RAW_CHARS), "RAW_TOO_LARGE")
        raw = retry()
        raw = raw.replace("\nRevisedSolution:", "\n" * (MAX_RAW_CHARS - len(raw) + 1) + "RevisedSolution:")
        self.assertEqual(len(raw), MAX_RAW_CHARS)
        self.assertEqual(project_solution(raw, self.sections(raw)), project_solution(retry(), self.sections(retry())))
        self.rejects(raw + "\n", "RAW_TOO_LARGE")
        self.rejects("ErrorCheck: {" + " " * MAX_CHECK_CHARS, "INVALID_JSON")
        self.rejects(retry(payload([claim(proposed_value="x" * 2049)])), "INVALID_FIELD")
        value = {"x": list(range(65))}
        self.rejects(retry(payload([claim(intent="RECHECK_OPERATION", proposed_value=value)])), "INVALID_FIELD")
        value = {str(i): i for i in range(33)}
        self.rejects(retry(payload([claim(intent="RECHECK_OPERATION", proposed_value=value)])), "INVALID_FIELD")
        value = {"x": {"x": {"x": {"x": {"x": 1}}}}}
        self.rejects(retry(payload([claim(intent="RECHECK_OPERATION", proposed_value=value)])), "INVALID_FIELD")
        value = {"x": ["x" * 1000] * 5}
        self.assertGreater(len(json.dumps(claim(proposed_value=value))), MAX_CLAIM_CHARS)
        self.rejects(retry(payload([claim(intent="RECHECK_OPERATION", proposed_value=value)])), "CLAIM_LIMIT")
        self.rejects(retry().replace('"H→G"', '"' + "x" * (MAX_CHECK_CHARS + 1) + '"'), "INVALID_JSON")

    def test_raw_metadata_tampering_invalid_utf8_and_explicit_contract(self):
        version: Any
        value: Any
        raw = retry()
        sections = self.sections(raw)
        for forged in (replace(sections, raw_sha="0" * 64),
                       replace(sections, check_span=RawSpan(0, 1)),
                       replace(sections, solution_span=RawSpan(0, len(raw))),
                       replace(sections, claim_spans=()), replace(sections, uncertain_refs=("x",))):
            for function in (project_solution, parse_error_check):
                with self.assertRaisesRegex(RepairContractError, "SECTION_MISMATCH"):
                    function(raw, forged)
        with self.assertRaisesRegex(RepairContractError, "SECTION_MISMATCH"):
            project_solution(raw.replace("answer: 7", "answer: 8"), sections)
        self.rejects(retry(body="\ud800"), "INVALID_UTF8")
        for version in ("legacy", "", True, None):
            with self.assertRaisesRegex(RepairContractError, "CONTRACT_VERSION"):
                parse_retry_sections(raw, contract_version=version)
        error = self.rejects(retry(payload(version="legacy")), "INVALID_FIELD")
        self.assertEqual(error.at, "check.version")
        for value in (None, True, 3, b"text"):
            with self.assertRaisesRegex(RepairContractError, "RAW_TYPE"):
                parse_retry_sections(value, contract_version=VERSION)

    def test_public_outcome_reproduces_grammar_failure_without_native_fallback(self):
        contract: Any
        grade = VerifierResult("INCORRECT", "7", 0., None, "fixture-v1")
        malformed = ("ErrorCheck: {", retry(payload(version="legacy")),
                     retry(payload(uncertain_refs=["same"] * 2)))
        for raw in malformed:
            self.assertEqual(public_attempt_outcome(raw, response_contract=VERSION, result=None),
                             ("INVALID_RESPONSE_CONTRACT", None))
            with self.assertRaisesRegex(ValueError, "acquired a native result"):
                public_attempt_outcome(raw, response_contract=VERSION, result=grade)
            self.assertEqual(public_attempt_outcome(raw, response_contract=None, result=None), (None, None))
        self.assertEqual(public_attempt_outcome(retry(), response_contract=VERSION, result=None), (None, None))
        for contract in (None, VERSION):
            self.assertEqual(public_attempt_outcome(retry(), response_contract=contract, result=grade),
                             ("INCORRECT", "7"))
        for contract in ("legacy", "repair-check-v2", True):
            with self.assertRaisesRegex(RepairContractError, "CONTRACT_VERSION"):
                public_attempt_outcome(malformed[0], response_contract=contract, result=None)

    def test_error_context_never_echoes_submitted_text(self):
        secret = "DO_NOT_OUTPUT_SUBMITTED_SECRET"
        error = self.rejects(retry(payload([claim(intent=secret)])), "INVALID_FIELD")
        self.assertNotIn(secret, str(error))
        self.assertNotIn(secret, repr(error))
        self.assertEqual(error.at, "check.claims[0]")
        self.assertIsInstance(error.offset, int)
        error = self.rejects(retry().replace('"rule":', '"' + secret + '":0,"' + secret + '":1,"rule":'))
        self.assertNotIn(secret, str(error))

    def test_original_token_indices_crossing_unstable_prefix_and_mismatch(self):
        ids: Any
        raw = retry()
        sections = self.sections(raw)
        span, = sections.claim_spans
        tokenizer = ChunkTokenizer(list(raw) + [""])
        result, = claim_token_ownership(raw, sections, tokenizer, list(range(len(raw) + 1)))
        self.assertEqual(result, {"claim_index": 0, "status": "MAPPED",
                                  "token_indices": list(range(span.start, span.end)),
                                  "excluded_crossing_token_indices": []})
        tokenizer = ChunkTokenizer([raw[:span.start + 1], raw[span.start + 1:]])
        result, = claim_token_ownership(raw, sections, tokenizer, [0, 1])
        self.assertEqual((result["status"], result["token_indices"]), ("NO_OWNED_TOKENS", []))
        self.assertEqual(result["excluded_crossing_token_indices"], [0, 1])
        tokenizer = ChunkTokenizer([raw[:span.start + 1], raw[span.start + 1:span.end - 1],
                                    raw[span.end - 1:]])
        result, = claim_token_ownership(raw, sections, tokenizer, [0, 1, 2])
        self.assertEqual((result["status"], result["token_indices"]),
                         ("MAPPED_WITH_EXCLUDED_BOUNDARIES", [1]))
        self.assertEqual(result["excluded_crossing_token_indices"], [0, 2])
        tokenizer = UnstableTokenizer([raw, ""])
        result, = claim_token_ownership(raw, sections, tokenizer, [0, 1])
        self.assertEqual((result["status"], result["token_indices"]), ("UNSTABLE_TOKEN_PREFIX", []))
        with self.assertRaisesRegex(RepairContractError, "TOKEN_TEXT_MISMATCH"):
            claim_token_ownership(raw, sections, ChunkTokenizer(["different"]), [0])
        for ids in ([True], [-1], [1.0], (0,)):
            with self.assertRaisesRegex(RepairContractError, "TOKEN_IDS"):
                claim_token_ownership(raw, sections, ChunkTokenizer([raw]), ids)

    def test_codec_import_does_not_load_legacy_or_model_modules(self):
        code = ("import sys; import fine_tuning.notebook_rl.repair_contract; "
                "assert 'torch' not in sys.modules; "
                "assert 'fine_tuning.notebook_rl.fact_credit' not in sys.modules; "
                "assert 'fine_tuning.notebook_rl.local_eval' not in sys.modules")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


class PublicOriginTests(unittest.TestCase):
    def inventory(self, samples: list[Any], contracts: dict[str, str | None], parent: str | None = None):
        return public_origin_inventory(samples, contract_version=ORIGIN_VERSION,
                                       response_contracts=contracts, parent_sample_id=parent)

    def test_public_only_fields_and_source_hashes_without_private_judgments(self):
        raw = native_body()
        samples: list[Any] = [PublicOnly("native", raw)]
        inventory = self.inventory(samples, {"native": None})
        source, = inventory.sources
        self.assertEqual((source.raw_sha, source.solution_sha), (hashlib.sha256(raw.encode()).hexdigest(),) * 2)
        self.assertEqual(source.solution_span, RawSpan(0, len(raw)))
        self.assertIsNone(source.projection_error_code)
        self.assertEqual(inventory.fixed_units, len(inventory.origins))
        self.assertEqual(public_origins(samples, contract_version=ORIGIN_VERSION,
                                       response_contracts={"native": None}), inventory.origins)
        self.assertEqual(inventory, self.inventory(samples, {"native": None}))
        wire = json.dumps(asdict(inventory))
        for private in ("gold", "result", "reward", "VIOLATED", "COMPLIANT"):
            self.assertNotIn(private, wire)
        observed = [origin for origin in inventory.origins if origin.kind == "OBSERVATION"]
        self.assertEqual([origin.rule for origin in observed], ["OBSERVATION_MATCH"] * 2)
        self.assertEqual([origin.locus for origin in observed], ["dir[e4]", "weight[e4]"])

    def test_stable_source_order_and_refs_survive_full_history_and_ancestry_subsets(self):
        samples: list[Any] = [PublicOnly("a", native_body("root = A\n")),
            PublicOnly("b", native_body("weight[e4] = 3\nroot = A\n")),
            PublicOnly("c", retry(body=native_body()), "a"),
            PublicOnly("d", retry(body=native_body()), "a"),
            PublicOnly("e", retry(body=native_body()), "c"),
            PublicOnly("f", retry(body=native_body()), "b")]
        contracts = {sample.sample_id: None if index < 2 else VERSION for index, sample in enumerate(samples)}
        full = self.inventory(samples, contracts)
        selected = self.inventory(samples, contracts, "e")
        self.assertEqual([source.sample_id for source in full.sources], list("abcdef"))
        self.assertEqual([source.sample_id for source in selected.sources], list("ace"))
        self.assertEqual(selected.origins, tuple(origin for origin in full.origins if origin.sample_id in "ace"))
        subset = [samples[index] for index in (0, 2, 4)]
        self.assertEqual(selected, self.inventory(subset, {key: contracts[key] for key in "ace"}, "e"))
        reordered = self.inventory([samples[1], samples[0], *samples[2:]], contracts)
        self.assertNotEqual(full.sha256, reordered.sha256)
        self.assertEqual({origin.ref_id: origin for origin in full.origins},
                         {origin.ref_id: origin for origin in reordered.origins})
        b_observed = [origin.locus for origin in full.origins
                      if origin.sample_id == "b" and origin.kind == "OBSERVATION"]
        self.assertEqual(b_observed, ["weight[e4]", "root"])

    def test_retry_projection_lifts_unicode_crlf_and_excludes_errorcheck_quotations(self):
        body = native_body("root = 🌱\ndir[e4] = H→G\n").replace("\n", "\r\n")
        raw = retry(payload([claim(proposed_value='quoted Trace: {} Final answer: 99 Observed facts: root = X')]),
                    body=body, newline="\r\n")
        raw = raw.replace("\r\nRevisedSolution:", "\r\n \t\r\n\r\nRevisedSolution:", 1)
        inventory = self.inventory([PublicOnly("retry", raw)], {"retry": VERSION})
        source, = inventory.sources
        sections = parse_retry_sections(raw, contract_version=VERSION)
        self.assertEqual(source.solution_span, sections.solution_span)
        self.assertEqual(source.raw_sha, hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(source.solution_sha, hashlib.sha256(body.encode()).hexdigest())
        self.assertNotEqual(source.raw_sha, source.solution_sha)
        for origin in inventory.origins:
            self.assertEqual(origin.raw_sha, source.raw_sha)
            for span in origin.occurrence_spans:
                self.assertGreaterEqual(span.start, sections.solution_span.start)
                self.assertLessEqual(span.end, len(raw))
                self.assertEqual(raw[span.start:span.end],
                    body[span.start - sections.solution_span.start:span.end - sections.solution_span.start])
        root = next(origin for origin in inventory.origins if origin.kind == "OBSERVATION" and origin.locus == "root")
        self.assertEqual(raw[root.occurrence_spans[0].start:root.occurrence_spans[0].end], "root = 🌱\r\n")

    def test_duplicate_occurrences_and_raw_unknown_regions_remain_in_scope(self):
        trace = {"kind": "dfa", "states": ["q0"], "decision": "reject"}
        body = native_body("DIR[e4] = H->G\ndir[e4] = ???\nunparsed raw line\n", trace)
        body += "Trace: " + json.dumps(trace) + "\nTrace: {broken}\n"
        inventory = self.inventory([PublicOnly("s", body)], {"s": None})
        observed = next(origin for origin in inventory.origins if origin.kind == "OBSERVATION")
        self.assertEqual(observed.locus, "dir[e4]")
        self.assertEqual([body[span.start:span.end] for span in observed.occurrence_spans],
                         ["DIR[e4] = H->G\n", "dir[e4] = ???\n"])
        operation = next(origin for origin in inventory.origins if origin.rule == "DFA_ACCEPTANCE")
        self.assertEqual(len(operation.occurrence_spans), 2)
        unknown = next(origin for origin in inventory.origins if origin.rule == "TRACE_FORMAT")
        self.assertEqual(unknown.occurrence_spans, (RawSpan(0, len(body)),))
        self.assertIn("unparsed raw line", body[unknown.occurrence_spans[0].start:unknown.occurrence_spans[0].end])
        self.assertFalse(any(origin.kind == "ABSENCE" for origin in inventory.origins))

    def test_malformed_retry_retains_raw_failure_and_only_existing_scope_refs(self):
        body = native_body()
        broken = retry(body=body).replace('"claims":', '"claims":NaN,"unused":')
        for raw in (broken, body, "", retry(body=body).replace('"version":"repair-check-v1"',
                                                              '"version":"wrong-version"')):
            inventory = self.inventory([PublicOnly("bad", raw)], {"bad": VERSION})
            source, = inventory.sources
            self.assertIsNone(source.solution_span)
            self.assertIsNone(source.solution_sha)
            self.assertIsNotNone(source.projection_error_code)
            self.assertEqual(source.raw_sha, hashlib.sha256(raw.encode()).hexdigest())
            self.assertEqual([(origin.kind, origin.rule) for origin in inventory.origins],
                             [("TRACE", "TRACE_FORMAT"), ("TRACE", "TRACE_SCHEMA"),
                              ("TRACE", "RESPONSE_SCOPE"), ("TRACE", "NATIVE_CONTRACT"),
                              ("OUTPUT", "FINALIZATION")])
            self.assertTrue(all(origin.occurrence_spans == ((RawSpan(0, len(raw)),) if raw else ())
                                for origin in inventory.origins))
        native = self.inventory([PublicOnly("bad", body)], {"bad": None})
        self.assertGreater(native.fixed_units, 5)
        self.assertIsNone(native.sources[0].projection_error_code)

    def test_empty_original_unclosed_observations_and_length_are_not_dropped(self):
        for raw in ("", "Observed facts:\nroot = A", native_body()):
            inventory = self.inventory([PublicOnly("s", raw, finish="length")], {"s": None})
            self.assertEqual(inventory.sources[0].finish_reason, "length")
            self.assertIsNone(inventory.sources[0].projection_error_code)
            self.assertGreaterEqual(inventory.fixed_units, 5)
            self.assertTrue(any(origin.rule == "FINALIZATION" for origin in inventory.origins))
        empty = self.inventory([], {})
        self.assertEqual((empty.origins, empty.sources, empty.fixed_units), ((), (), 0))
        self.assertEqual(empty.sha256, self.inventory([], {}).sha256)

    def test_all_native_trace_families_are_public_inventory_without_truth_filtering(self):
        traces: list[dict] = [{"kind": "shortest", "nodes": ["A", "B"], "edge_ids": ["e1"], "weights": [1], "cost": 2},
            {"kind": "bfs", "steps": [{"node": "A", "new": ["B"]}]},
            {"kind": "tree", "traversal": "preorder", "events": [{"op": "enter", "node": "A"}]},
            {"kind": "dfa", "states": ["q0"], "decision": "accept"},
            {"kind": "circuit_ladder", "nodes": [["N0", "0"], ["N1", "1"]],
             "ohm": [["R1", "N1", "N0", "1", "1", None]], "kcl": [],
             "input": ["1", None, None], "query": ["current", "R1", None, "A"]}]
        expected = ["COST_SUM", "BFS_FIFO", "TREE_EVENT_ORDER", "DFA_ACCEPTANCE", "CIRCUIT_OHM"]
        for trace, rule in zip(traces, expected, strict=True):
            inventory = self.inventory([PublicOnly("s", native_body(trace=trace))], {"s": None})
            self.assertTrue(any(origin.rule == rule for origin in inventory.origins))
            self.assertEqual(len({origin.ref_id for origin in inventory.origins}), inventory.fixed_units)
            self.assertTrue(all(origin.ref_id.startswith("s:r") for origin in inventory.origins))

    def test_digest_binds_raw_projection_finish_contract_and_scope(self):
        raw = native_body()
        baseline = self.inventory([PublicOnly("s", raw)], {"s": None})
        declared_retry = self.inventory([PublicOnly("s", raw)], {"s": VERSION})
        self.assertEqual(baseline.origins[0].ref_id, declared_retry.origins[0].ref_id)
        self.assertEqual(baseline.sources[0].raw_sha, declared_retry.sources[0].raw_sha)
        self.assertNotEqual(baseline.origins[0].rule, declared_retry.origins[0].rule)
        for candidate in (self.inventory([PublicOnly("s", raw.replace("answer: 3", "answer: 4"))], {"s": None}),
                          self.inventory([PublicOnly("s", raw, finish="length")], {"s": None}),
                          declared_retry,
                          self.inventory([PublicOnly("s", raw)], {"s": None}, "s")):
            self.assertNotEqual(baseline.sha256, candidate.sha256)
        binding = {key: value for key, value in asdict(baseline).items() if key != "sha256"}
        binding["fixed_units"] = baseline.fixed_units
        self.assertEqual(baseline.sha256, hashlib.sha256(json.dumps(
            binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())

    def test_integrity_errors_are_rejected_before_model_syntax_parsing(self):
        contracts: Any
        good = PublicOnly("a", retry())
        bad_histories: list[list[Any]] = [[good, PublicOnly("a", "")], [good, PublicOnly("b", "", "future")],
                         [PublicOnly("self", "", "self")], [PublicOnly("x", "", finish="exception")],
                         [PublicOnly("x", None)], [PublicOnly("x", "\ud800")],
                         [PublicOnly("x", "x" * (MAX_RAW_CHARS + 1))],
                         [SimpleNamespace(sample_id="x", parent_sample_id=None, finish_reason="stop")]]
        for samples in bad_histories:
            with patch("fine_tuning.notebook_rl.repair_contract.parse_retry_sections") as parser:
                with self.assertRaises(RepairContractError):
                    self.inventory(samples, {"a": VERSION, "b": VERSION})
                parser.assert_not_called()
        for contracts in ({}, {"a": "legacy"}, {"a": True}, {"a": None, "extra": VERSION}):
            with self.assertRaisesRegex(RepairContractError, "ORIGIN_CONTRACTS"):
                self.inventory([good], contracts)
        with self.assertRaisesRegex(RepairContractError, "ORIGIN_LINEAGE"):
            self.inventory([good], {"a": VERSION}, "future")
        with self.assertRaisesRegex(RepairContractError, "CONTRACT_VERSION"):
            public_origin_inventory([good], contract_version=VERSION, response_contracts={"a": VERSION})

    def test_reference_limit_and_legacy_bookkeeper_errors_are_not_hidden(self):
        with self.assertRaisesRegex(RepairContractError, "ORIGIN_REFERENCE"):
            self.inventory([PublicOnly("s" * 255, "")], {"s" * 255: None})
        with patch("fine_tuning.notebook_rl.whole_reference_notebook.public_inventory",
                   side_effect=RuntimeError("bookkeeping defect")):
            with self.assertRaisesRegex(RuntimeError, "bookkeeping defect"):
                self.inventory([PublicOnly("s", "")], {"s": None})


if __name__ == "__main__":
    unittest.main()
