"""Explicit synthetic CPU/tokenizer and HTTP mocks; never actual GPU evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest import mock

from experiments.gemma_verifier import preflight as p
from experiments.gemma_verifier import runtime as r


class FakeTokenizer:
    chat_template = "synthetic template"

    def __init__(self, count: int = 100):
        self.count = count
        self.calls: list[Any] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
        self.calls.append((messages, kwargs))
        return list(range(self.count))


class PreflightTests(unittest.TestCase):
    def test_runtime_public_version_and_exact_cuda_wheel_are_distinct(self) -> None:
        spec = {"vllm": {"version": "0.24.0+cu129", "wheel_sha256": "a" * 64}}
        identity = {"dtype": "bfloat16", "vllm_version": "0.24.0", "wheel_sha256": "a" * 64}
        p.check_deployment_identity(identity, spec)
        for field, value in (("dtype", "float16"), ("vllm_version", "0.24.1"),
                             ("vllm_version", "0.24.0+cu129"), ("wheel_sha256", "b" * 64)):
            with self.assertRaisesRegex(ValueError, "runtime and wheel"):
                p.check_deployment_identity(identity | {field: value}, spec)
        with self.assertRaisesRegex(ValueError, "runtime and wheel"):
            p.check_deployment_identity({k: v for k, v in identity.items() if k != "wheel_sha256"}, spec)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.jsonl"
        self.rows = [{"id": f"synthetic_{i:03d}", "subject": f"Subject_{i // 30}",
                      "question": "Choose one color.", "question_type": "mcq",
                      "options": ["red", "blue", "green"], "raw_response": "Final Answer: B",
                      "parsed_answer": "B", "correct": True, "answer": "B", "finish_reason": "stop"}
                     for i in range(900)]
        self.source.write_text("".join(r.canonical(row) + "\n" for row in self.rows))
        self.spec = r.json_object(p.SPEC.read_text())
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            data = ("synthetic " + name).encode()
            (self.snapshot / name).write_bytes(data)
            self.spec["expected_model_files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.identity = {"model_id": self.spec["model_id"], "revision": self.spec["model_revision"],
                         "tokenizer_revision": self.spec["model_revision"], "frozen_weights": True,
                         "weights_manifest_sha256": "a" * 64,
                         "chat_template_sha256": self.spec["expected_model_files"]["chat_template.jinja"]["sha256"],
                         "environment_lock_sha256": "b" * 64, "max_model_len": 131072,
                         "dtype": "bfloat16", "vllm_version": "synthetic-test"}
        self.identity_path = self.root / "identity.json"
        self.identity_path.write_text(r.canonical(self.identity))
        self.cpu = self.root / "cpu"
        self.http_calls: list[str] = []

    def prepare(self, **kwargs: Any) -> dict[str, Any]:
        return p.cpu_preflight(self.source, self.snapshot, self.cpu, tokenizer=FakeTokenizer(),
                               spec=self.spec, deadline_utc=kwargs.pop("deadline_utc", "2099-01-01T00:00:00Z"), **kwargs)

    def transport(self, endpoint: str, path: str, payload: Any = None, **kwargs: Any) -> Any:
        self.http_calls.append(path)
        if path == "/version":
            return {"version": "synthetic-test"}
        if path == "/v1/models":
            return {"data": [{"id": self.spec["model_id"]}]}
        if path == "/tokenize":
            return {"count": 100, "tokens": list(range(100)), "max_model_len": 131072}
        if path != "/v1/chat/completions":
            raise AssertionError(path)
        user = json.loads(payload["messages"][1]["content"])
        if "frozen_submitted_answer" in user:
            mapped = {"a bluish color": "B", "rotating anticlockwise": "C",
                      "a triangular figure": "B", "there is an increase": "B",
                      "a reddish shade": "C"}.get(user["frozen_submitted_answer"])
            if user["frozen_submitted_answer"] == "a triangular shape" and user["choices"]["B"] == "triangle":
                mapped = "B"
            result: Any = {"choice_assessment": {label: ("SUPPORTED" if label == mapped else "EXCLUDED")
                                                if mapped else "UNCERTAIN" for label in user["choices"]},
                           "mapping_status": "MATCHED" if mapped else "UNRESOLVED",
                           "mapped_choice": mapped, "reason": "Synthetic frozen choice."}
        elif "frozen_extracted_answer" in user:
            answer = user["frozen_extracted_answer"]
            verdict = "EQUIVALENT" if answer == "$2,960 unfavorable" and user["reference_aliases"] == ["2960 unfavorable"] else "NOT_EQUIVALENT"
            result = {"verdict": verdict, "reason": "Synthetic controlled pair."}
        else:
            raw = user["raw_response"]
            status, answer, reason, evidence, revision = "CLEAR", "B", "EXPLICIT_FINAL_ANSWER", ["Final Answer: B"], []
            if "Correction: B" in raw:
                reason, evidence, revision = "COMPLETED_CORRECTION", ["Correction: B"], ["I retract my answer."]
            elif "I withdraw" in raw:
                status, answer, reason, evidence, revision = "AMBIGUOUS", None, "RETRACTED_OR_UNRESOLVED", [], [raw.splitlines()[-1]]
            elif "A or B" in raw:
                status, answer, reason, evidence, revision = "AMBIGUOUS", None, "CONFLICTING_ANSWERS", [], [raw]
            elif "No final answer" in raw or "**Maximize**:" in raw:
                status, answer, reason, evidence = "NO_FINAL_ANSWER", None, "NO_COMMITTED_ANSWER", []
            elif raw == "Final Answer: blue":
                answer, evidence = "blue", [raw]
            elif "8 m" in raw or "unfavorable" in raw:
                answer, evidence = raw.split(": ", 1)[1], [raw]
            result = {"answer_status": status, "extracted_answer": answer,
                      "answer_evidence": evidence, "revision_evidence": revision, "reason_code": reason}
        return {"model": self.spec["model_id"], "usage": {"prompt_tokens": 100, "completion_tokens": 60},
                "choices": [{"finish_reason": "stop", "message": {"content": r.canonical(result)}}]}

    def smoke(self, **kwargs: Any) -> dict[str, Any]:
        return p.smoke(self.source, self.cpu, self.root / "smoke", self.identity_path,
                       "http://127.0.0.1:8000/v1", deadline_utc="2099-01-01T00:00:00Z",
                       transport=kwargs.pop("transport", self.transport), **kwargs)

    def test_pure_full_payload_and_gold_independence(self) -> None:
        row = copy.deepcopy(self.rows[0])
        row["raw_response"] = "full raw " * 32768 + "Final Answer: B"
        tokenizer = FakeTokenizer()
        first = p.count_input(row, tokenizer, self.spec["model_id"])
        row.update(answer="A", correct=False, parsed_answer="A", gold_explanation="secret")
        second = p.count_input(row, tokenizer, self.spec["model_id"])
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first["payload"]["messages"][1]["content"])["raw_response"], row["raw_response"])
        self.assertEqual(tokenizer.calls[0][1], {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False, "return_dict": False})

    def test_all900_ledger_and_longest_no_model_loading(self) -> None:
        summary = self.prepare()
        self.assertEqual(summary["counted"], 900)
        self.assertEqual(summary["status"], "CPU_TEST_ONLY_PASS")
        self.assertFalse(summary["full_audit_allowed"])
        self.assertEqual(len((self.cpu / "token_budget.jsonl").read_text().splitlines()), 900)
        self.assertEqual(summary["max_input_plus_output"], 1124)
        self.assertEqual(summary["top10"][0]["id"], "synthetic_000")
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_context_overflow_is_not_truncated(self) -> None:
        counted = p.count_input(self.rows[0], FakeTokenizer(131072), self.spec["model_id"])
        self.assertFalse(counted["fits_context"])
        self.assertEqual(counted["input_tokens"], 131072)

    def test_complete900_and_pinned_bytes_required(self) -> None:
        self.source.write_text(r.canonical(self.rows[0]) + "\n")
        with self.assertRaisesRegex(ValueError, "complete 900"):
            self.prepare()
        self.source.write_text("".join(r.canonical(row) + "\n" for row in self.rows))
        (self.snapshot / "tokenizer.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Pinned"):
            self.prepare()

    def test_deadline_preserves900_stopped_cells(self) -> None:
        summary = self.prepare(deadline_utc="2000-01-01T00:00:00Z")
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertEqual(summary["counted"], 0)
        self.assertEqual(len((self.cpu / "token_budget.jsonl").read_text().splitlines()), 900)
        with self.assertRaises(ValueError):
            p._deadline("2099-01-01T00:00:00")

    def test_dry_smoke_never_calls_endpoint(self) -> None:
        self.prepare()
        summary = self.smoke()
        self.assertEqual(self.http_calls, [])
        self.assertEqual(summary["status"], "DRY_RUN")
        self.assertFalse(summary["full_audit_allowed"])

    def test_synthetic_mock_pass_never_authorizes_full900(self) -> None:
        self.prepare()
        summary = self.smoke(run=True)
        self.assertEqual(summary["status"], "MOCK_SMOKE_PASS")
        self.assertEqual(summary["statuses"], {"PASS": 29})
        self.assertEqual(summary["cases"], 29)
        self.assertFalse(summary["full_audit_allowed"])
        self.assertFalse(summary["longest_semantic_correctness_checked"])
        ledger = [json.loads(line) for line in (self.root / "smoke/smoke.jsonl").read_text().splitlines()]
        self.assertEqual(ledger[-1]["stage"], "real_longest")
        self.assertNotIn("answer", ledger[-1]["row"])
        with mock.patch.object(r, "execute", side_effect=AssertionError("No full calls permitted")):
            with self.assertRaises(ValueError):
                p.gated_audit(self.source, self.cpu, self.root / "smoke", self.root / "audit",
                              self.identity_path, "http://127.0.0.1:8000/v1", run=True)

    def test_forced_mapping_of_a_new_negative_probe_fails_all_case_gate(self) -> None:
        self.prepare()
        def forced(endpoint: str, path: str, payload: Any = None, **kwargs: Any) -> Any:
            result = self.transport(endpoint, path, payload, **kwargs)
            if path == "/v1/chat/completions" and json.loads(payload["messages"][1]["content"]).get("frozen_submitted_answer") == "not blue":
                result["choices"][0]["message"]["content"] = r.canonical({
                    "choice_assessment": {"A": "SUPPORTED", "B": "EXCLUDED", "C": "POSSIBLE"},
                    "mapping_status": "MATCHED", "mapped_choice": "A", "reason": "Forced alternative."})
            return result
        summary = self.smoke(run=True, transport=forced)
        self.assertEqual(summary["status"], "GPU_SMOKE_FAILED")
        self.assertEqual(summary["statuses"], {"PASS": 28, "FAIL": 1})
        self.assertFalse(summary["full_audit_allowed"])
        records = [json.loads(line) for line in (self.root / "smoke/smoke.jsonl").read_text().splitlines()]
        self.assertEqual([record["id"] for record in records if record["status"] == "FAIL"],
                         ["synthetic_mapping_negation"])

    def test_usage_mismatch_and_equivalence_direction_fail_gate(self) -> None:
        self.prepare()
        def bad(endpoint: str, path: str, payload: Any = None, **kwargs: Any) -> Any:
            result = self.transport(endpoint, path, payload, **kwargs)
            if path == "/v1/chat/completions":
                if "frozen_extracted_answer" in json.loads(payload["messages"][1]["content"]):
                    result["choices"][0]["message"]["content"] = r.canonical({"verdict": "EQUIVALENT", "reason": "Wrong synthetic judge"})
                else:
                    result["usage"]["prompt_tokens"] = 99
            return result
        summary = self.smoke(run=True, transport=bad)
        self.assertEqual(summary["status"], "GPU_SMOKE_FAILED")
        self.assertFalse(summary["full_audit_allowed"])

    def test_real_longest_unsafe_grounding_fails_gate(self) -> None:
        self.rows[0]["raw_response"] = "An introduction without a submitted answer."
        self.source.write_text("".join(r.canonical(row) + "\n" for row in self.rows))
        self.prepare()
        def unbound(endpoint: str, path: str, payload: Any = None, **kwargs: Any) -> Any:
            result = self.transport(endpoint, path, payload, **kwargs)
            if path == "/v1/chat/completions" and json.loads(payload["messages"][1]["content"]).get("raw_response") == self.rows[0]["raw_response"]:
                result["choices"][0]["message"]["content"] = r.canonical({
                    "answer_status": "CLEAR", "extracted_answer": "A", "answer_evidence": [self.rows[0]["raw_response"]],
                    "revision_evidence": [], "reason_code": "EXPLICIT_FINAL_ANSWER"})
            return result
        summary = self.smoke(run=True, transport=unbound)
        self.assertEqual(summary["status"], "GPU_SMOKE_FAILED")
        last = json.loads((self.root / "smoke/smoke.jsonl").read_text().splitlines()[-1])
        self.assertEqual(last["status"], "FAIL")
        self.assertIn("grounded evidence", last["error"])

    def test_model_template_or_ledger_mismatch_blocks_before_http(self) -> None:
        self.prepare()
        path = self.cpu / "token_budget.jsonl"
        path.write_text(path.read_text().replace('"input_tokens":100', '"input_tokens":99', 1))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.smoke(run=True)
        self.assertEqual(self.http_calls, [])

    def test_nonloopback_rejected_without_calls(self) -> None:
        self.prepare()
        with self.assertRaises(ValueError):
            p.smoke(self.source, self.cpu, self.root / "smoke", self.identity_path,
                    "https://commercial.example/v1", run=True, transport=self.transport)
        self.assertEqual(self.http_calls, [])


if __name__ == "__main__":
    unittest.main()
