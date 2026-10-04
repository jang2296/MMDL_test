"""Private hand-calculated post-run checks; never copied into the public checkout."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from experiments.bottleneck_diagnostic import export_results as exporter
from experiments.bottleneck_diagnostic.export_results import (
    compatible_union,
    cost_rows,
    decision_envelope,
    gemma_export,
    structure_summary,
)
from experiments.bottleneck_diagnostic.metrics import summarize


class ExportResultsTests(unittest.TestCase):
    def gemma_mapping_fixture(self, root, extraction_mode="literal"):
        """Explicit synthetic stored responses; no endpoint or model invocation."""
        from experiments.gemma_verifier import runtime as g

        rows = [{"id": f"synthetic_{i}", "subject": f"Synthetic_{i // 30}",
                 "question": "Choose the submitted shape.", "question_type": "mcq",
                 "options": ["circle", "triangle", "square"],
                 "raw_response": "Final Answer: a triangular shape", "finish_reason": "stop",
                 "parsed_answer": "B" if i < 558 else None, "correct": i < 558,
                 "answer": "B"} for i in range(900)]
        source = root / "synthetic_source.jsonl"
        source.write_text("".join(g.canonical(row) + "\n" for row in rows))
        directory = root / "synthetic_audit"
        directory.mkdir()
        manifest = {"source_sha256": g.file_hash(source), "source_count": 900,
                    "execute": True, "semantic_open": True, "model": "google/gemma-4-E4B-it",
                    "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
                    "endpoint": "http://127.0.0.1:8000/v1",
                    "model_manifest": {"synthetic_fixture": True}, "protocol_sha256": "synthetic"}
        manifest["model_manifest_sha256"] = g.digest(manifest["model_manifest"])
        if extraction_mode != "literal":
            manifest.update(extraction_mode=extraction_mode, protocol=g.protocol_id(extraction_mode))
        (directory / "run_manifest.json").write_text(g.canonical(manifest))
        judge = g.canonical({"answer_status": "CLEAR", "extracted_answer": "a triangular shape",
                             "answer_evidence": [rows[0]["raw_response"]], "revision_evidence": [],
                             "reason_code": "EXPLICIT_FINAL_ANSWER"})
        if extraction_mode == "segments":
            judge = g.canonical({"answer_status": "CLEAR", "extracted_answer": "a triangular shape",
                                 "answer_evidence_ids": [0], "revision_evidence_ids": [],
                                 "reason_code": "EXPLICIT_FINAL_ANSWER"})
        mapping = g.canonical({"choice_assessment": {"A": "EXCLUDED", "B": "SUPPORTED", "C": "EXCLUDED"},
                               "mapping_status": "MATCHED", "mapped_choice": "B", "reason": "Shape category."})
        entry = g.audit_rows([rows[0]], {rows[0]["id"]: judge},
                             mapping_outputs={rows[0]["id"]: mapping}, extraction_mode=extraction_mode)[0]
        def response(raw, inputs, outputs):
            return {"model": manifest["model"], "choices": [{"finish_reason": "stop",
                    "message": {"content": raw}}],
                    "usage": {"prompt_tokens": inputs, "completion_tokens": outputs}}
        entry.update({key: manifest[key] for key in
                      ("model", "revision", "model_manifest_sha256", "protocol_sha256")})
        entry.update(requests={"extraction": g.request_payload(rows[0], manifest["model"], extraction_mode),
                               "mapping": g.mapping_request(rows[0], "a triangular shape", manifest["model"])},
                     http_responses={"extraction": response(judge, 10, 2),
                                     "mapping": response(mapping, 20, 3)}, elapsed_seconds=1)
        summary = g.summarize([entry], extraction_mode) | {"execute": True, "dry_run": False}
        (directory / "summaries.jsonl").write_text(g.canonical(summary) + "\n")
        return directory, source, entry

    def test_segment_mode_replays_pointers_mapping_and_missing_input_hashes(self):
        from experiments.gemma_verifier import runtime as g

        with tempfile.TemporaryDirectory() as temporary:
            directory, source, entry = self.gemma_mapping_fixture(Path(temporary), "segments")
            (directory / "ledger.jsonl").write_text(g.canonical(entry | {"entry_sha256": g.digest(entry)}) + "\n")
            result, queue = gemma_export(directory, source)
            self.assertEqual(result["replayed_terminal_rows"], 1)
            self.assertEqual(result["validation"]["protocol"], g.protocol_id("segments"))
            self.assertIsNone(result["validation"]["assisted_accuracy"])
            self.assertEqual(len(queue), 900)
            self.assertEqual(result["stage_costs"]["mapping"]["response_calls"], 1)
            by_id = {item["id"]: item for item in queue}
            for row in g.load_rows(source):
                self.assertEqual(by_id[row["id"]]["input_sha256"],
                                 g.digest(g.extraction_input(row, "segments")))
            self.assertEqual(by_id[entry["id"]]["validated_extraction"]["source_pointer_json"],
                             json.loads(entry["judge_raw"]))

    def test_segment_mode_pointer_payload_and_summary_tampering_rejected(self):
        from experiments.gemma_verifier import runtime as g

        with tempfile.TemporaryDirectory() as temporary:
            directory, source, original = self.gemma_mapping_fixture(Path(temporary), "segments")
            manifest_path = directory / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            for mutation in ("mode", "payload", "pointers", "offsets", "summary"):
                entry = deepcopy(original)
                changed_manifest = deepcopy(manifest)
                summary = g.summarize([entry], "segments") | {"execute": True, "dry_run": False}
                if mutation == "mode":
                    changed_manifest["extraction_mode"] = "literal"
                elif mutation == "payload":
                    data = json.loads(entry["requests"]["extraction"]["messages"][1]["content"])
                    data["raw_response"][0]["text"] = "Final Answer: a circular shape"
                    entry["requests"]["extraction"]["messages"][1]["content"] = g.canonical(data)
                elif mutation == "pointers":
                    data = json.loads(entry["judge_raw"])
                    data["answer_evidence_ids"] = [99]
                    entry["judge_raw"] = g.canonical(data)
                    entry["http_responses"]["extraction"]["choices"][0]["message"]["content"] = entry["judge_raw"]
                elif mutation == "offsets":
                    entry["validated_extraction"]["evidence_offsets"]["answer"][0]["positions"][0]["start"] = 1
                else:
                    summary["protocol"] = g.protocol_id("literal")
                manifest_path.write_text(g.canonical(changed_manifest))
                (directory / "summaries.jsonl").write_text(g.canonical(summary) + "\n")
                (directory / "ledger.jsonl").write_text(g.canonical(entry | {"entry_sha256": g.digest(entry)}) + "\n")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    gemma_export(directory, source)

    def test_cpu_mode_identity_binds_segment_helper_and_preserves_legacy_closure(self):
        from experiments.bottleneck_diagnostic import finish_runpod as f
        from experiments.gemma_verifier import runtime as g

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "experiments/gemma_verifier"
            directory.mkdir(parents=True)
            source = root / "analysis_exports/bottleneck_20261003/gemma_source_v8.jsonl"
            source.parent.mkdir(parents=True)
            source.write_text("explicit synthetic CPU source")
            preflight = root / "preflight.py"
            preflight.write_text("explicit synthetic CPU preflight")
            for name in ("runtime.py", "start_server.sh", "source_segments.py"):
                (directory / name).write_text("explicit synthetic " + name)
            names = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
            spec = {"model_id": "synthetic-model", "model_revision": "synthetic-revision",
                    "expected_model_files": {name: {"sha256": "a" * 64} for name in names}}
            (directory / "deployment.json").write_text(g.canonical(spec))
            cpu = root / "cpu"
            cpu.mkdir()
            (cpu / "token_budget.jsonl").write_text("explicit synthetic CPU ledger")
            for mode in (None, "literal", "segments"):
                paths = [preflight, *(directory / name for name in
                         ("runtime.py", "deployment.json", "start_server.sh"))]
                if mode is not None:
                    paths.append(directory / "source_segments.py")
                manifest = {"code_sha256": f.digest({p.name: f.sha(p) for p in paths}),
                            "source_sha256": f.sha(source), "model": spec["model_id"],
                            "revision": spec["model_revision"], "kind": "ACTUAL_CPU_TOKENIZER",
                            "source_count": 900, "input_truncation": False, "max_model_len": 131072,
                            "output_allowance": 1024, "runtime_chat_template_sha256": "a" * 64,
                            "tokenizer_files_sha256": {name: "a" * 64 for name in names}}
                if mode is not None:
                    manifest.update(extraction_mode=mode, protocol=g.protocol_id(mode))
                (cpu / "manifest.json").write_text(g.canonical(manifest))
                summary = {"source_sha256": manifest["source_sha256"],
                           "manifest_sha256": f.sha(cpu / "manifest.json"),
                           "token_budget_sha256": f.sha(cpu / "token_budget.jsonl"),
                           "status": "CPU_TOKENIZER_PASS", "counted": 900, "context_overflows": 0,
                           "truncation": False, "max_input_plus_output": 51003,
                           "full_audit_allowed": False, "judge_calls": 0}
                if mode is not None:
                    summary.update(extraction_mode=mode, protocol=g.protocol_id(mode))
                (cpu / "summary.json").write_text(g.canonical(summary))
                with self.subTest(mode=mode):
                    self.assertEqual(f.verify_cpu_identity(root, preflight, cpu)["code_sha256"],
                                     manifest["code_sha256"])
                    helper = directory / "source_segments.py"
                    original = helper.read_bytes()
                    helper.write_text("changed synthetic helper")
                    if mode is None:
                        f.verify_cpu_identity(root, preflight, cpu)
                    else:
                        with self.assertRaisesRegex(ValueError, "identity mismatch"):
                            f.verify_cpu_identity(root, preflight, cpu)
                    helper.write_bytes(original)
                    summary["extraction_mode"] = "literal" if mode == "segments" else "segments"
                    (cpu / "summary.json").write_text(g.canonical(summary))
                    with self.assertRaisesRegex(ValueError, "identity mismatch"):
                        f.verify_cpu_identity(root, preflight, cpu)

    def test_gemma_mapping_replay_and_costs_preserve_review_denominator(self):
        from experiments.gemma_verifier import runtime as g

        with tempfile.TemporaryDirectory() as temporary:
            directory, source, entry = self.gemma_mapping_fixture(Path(temporary))
            (directory / "ledger.jsonl").write_text(g.canonical(entry | {"entry_sha256": g.digest(entry)}) + "\n")
            result, queue = gemma_export(directory, source)
            self.assertEqual(result["replayed_terminal_rows"], 1)
            self.assertEqual(result["stored_v8_correct"], 558)
            self.assertEqual(result["decision_envelope"]["denominator"], 900)
            self.assertEqual(result["decision_envelope"]["resolved_true"], 0)
            self.assertEqual(result["decision_envelope"]["unresolved"], 900)
            self.assertEqual(len(queue), 900)
            self.assertEqual(result["stage_costs"]["mapping"]["response_calls"], 1)
            self.assertEqual(result["stage_costs"]["mapping"]["input_tokens"], 20)
            self.assertEqual(result["stage_costs"]["mapping"]["output_tokens"], 3)
            validation = result["validation"]
            self.assertIsNone(validation["assisted_accuracy"])
            self.assertFalse(validation["controller_validation"]["final_score_allowed"])
            self.assertEqual(validation["controller_validation"]["actual_completion_responses"], 2)
            self.assertEqual(validation["controller_validation"]["input_tokens"], 30)

    def test_gemma_mapping_payload_and_raw_mismatches_rejected(self):
        from experiments.gemma_verifier import runtime as g

        with tempfile.TemporaryDirectory() as temporary:
            directory, source, original = self.gemma_mapping_fixture(Path(temporary))
            for mutation in ("payload", "http_raw", "saved_raw"):
                entry = deepcopy(original)
                if mutation == "payload":
                    data = json.loads(entry["requests"]["mapping"]["messages"][1]["content"])
                    data["frozen_submitted_answer"] = "a circular shape"
                    entry["requests"]["mapping"]["messages"][1]["content"] = g.canonical(data)
                elif mutation == "http_raw":
                    entry["http_responses"]["mapping"]["choices"][0]["message"]["content"] = "{}"
                else:
                    entry["mapping_raw"] = "{}"
                (directory / "ledger.jsonl").write_text(g.canonical(entry | {"entry_sha256": g.digest(entry)}) + "\n")
                with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "mapping (request|response) differs"):
                    gemma_export(directory, source)

    def test_gemma_source_digest_reused_in_all_queue_rows(self):
        from experiments.gemma_verifier import runtime as g

        with tempfile.TemporaryDirectory() as temporary:
            directory, source, entry = self.gemma_mapping_fixture(Path(temporary))
            (directory / "ledger.jsonl").write_text(g.canonical(entry | {"entry_sha256": g.digest(entry)}) + "\n")
            expected = exporter.sha(source)
            with patch.object(exporter, "sha", wraps=exporter.sha) as hashing:
                result, queue = gemma_export(directory, source)
            self.assertEqual(sum(call.args[0] == source for call in hashing.call_args_list), 3)
            self.assertEqual(result["source_sha256"], expected)
            self.assertEqual({row["source_sha256"] for row in queue}, {expected})
            self.assertEqual(len(queue), 900)

    def test_gemma_source_final_identity_and_frozen_mutation_guards_retained(self):
        from experiments.gemma_verifier import runtime as g

        with tempfile.TemporaryDirectory() as temporary:
            directory, source, entry = self.gemma_mapping_fixture(Path(temporary))
            (directory / "ledger.jsonl").write_text(g.canonical(entry | {"entry_sha256": g.digest(entry)}) + "\n")
            original_sha = exporter.sha
            for fail_at, message in ((2, "source changed"), (3, "ledger/manifest changed")):
                calls = 0
                def changed_digest(path):
                    nonlocal calls
                    if path == source:
                        calls += 1
                        if calls == fail_at:
                            return "0" * 64
                    return original_sha(path)
                with self.subTest(fail_at=fail_at), patch.object(exporter, "sha", side_effect=changed_digest):
                    with self.assertRaisesRegex(ValueError, message):
                        gemma_export(directory, source)

    def test_compatible_union_and_overlap_rejection(self):
        identity = {"protocol_hash": "p", "code_hash": "c", "execution_kind": "ACTUAL_INFERENCE",
                    "repetitions": [3407]}
        runs = []
        for sid, suite, before, after in [("a", "pilot", False, True),
                                          ("b", "coverage", True, False)]:
            sample = {"sample_id": sid, "subject": "Math", "family_id": "shared",
                      "skill_tags": ["Diagram"]}
            rows = [{"sample_id": sid, "mode": mode, "repetition": 3407,
                     "runtime_status": "COMPLETED", "diagnostic_score": correct,
                     "run_kind": "ACTUAL_INFERENCE"}
                    for mode, correct in [("I", before), ("O0", after)]]
            runs.append({"suite": suite, "identity": identity, "samples": [sample],
                         "scores": rows, "attempts": rows})
        samples, scores, attempts = compatible_union(runs)
        pair = summarize(scores, samples, attempts)["groups"]["overall_selected"]["paired"][0]
        self.assertEqual((pair["recovered"], pair["regressed"], pair["paired_questions"]), (1, 1, 2))
        self.assertEqual(pair["question_mean_delta"], 0)
        self.assertEqual(pair["shared_image_text_families"], 1)
        self.assertIsNone(pair["family_bootstrap_95_interval"])
        overlap = deepcopy(runs)
        overlap[1]["samples"][0]["sample_id"] = "a"
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            compatible_union(overlap)
        incompatible = deepcopy(runs)
        incompatible[1]["identity"] = identity | {"code_hash": "other"}
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            compatible_union(incompatible)
        standalone = deepcopy(runs)
        standalone[0]["suite"] = "stress"
        with self.assertRaisesRegex(ValueError, "standalone"):
            compatible_union(standalone)

    def test_p_costs_shared_once_and_attempts_retained(self):
        rows = [{"sample_id": "a", "mode": mode, "repetition": 3407,
                 "runtime_status": "COMPLETED", "run_kind": "ACTUAL_INFERENCE",
                 "attempt_id": mode,
                 "input_tokens": inputs, "output_tokens": output, "elapsed_seconds": 1,
                 "backend_audit": {"image_tokens": images, "observed_device_memory_used_bytes": 100}}
                for mode, inputs, images, output in [("P", 10, 4, 2), ("M", 20, 0, 3),
                                                    ("MI", 30, 5, 4)]]
        rows.insert(0, rows[1] | {"runtime_status": "SYSTEM_ERROR", "input_tokens": 1,
                                 "output_tokens": 0, "attempt_id": "precall-error"})
        cost = cost_rows(rows)
        self.assertEqual(cost["actual_total"]["input_tokens"], 61)
        self.assertEqual(cost["actual_total"]["input_text_tokens"], 52)
        self.assertEqual(cost["actual_total"]["recorded_execution_path_attempts"], 4)
        self.assertEqual(cost["actual_total"]["completed_calls"], 3)
        self.assertIsNone(cost["actual_total"]["actual_call_count"])
        self.assertEqual(cost["logical_matched_pipelines"]["P+M"]["input_tokens"], 31)
        self.assertEqual(cost["logical_matched_pipelines"]["P+MI"]["input_tokens"], 40)
        self.assertEqual(cost["actual_total"]["memory_measurement"], "snapshot_not_true_peak")

    def test_uncertain_execution_failures_and_prebudget_stop(self):
        base = {"sample_id": "a", "mode": "P", "repetition": 3407,
                "run_kind": "ACTUAL_INFERENCE", "input_tokens": 0, "output_tokens": 0}
        rows = [base | {"runtime_status": "BUDGET_STOP", "attempt_id": "attempted-stop",
                        "charged_output_tokens": 100, "elapsed_seconds": 2},
                base | {"runtime_status": "SYSTEM_ERROR", "attempt_id": "precall-error",
                        "charged_output_tokens": 100, "elapsed_seconds": 3},
                base | {"runtime_status": "BUDGET_STOP"}]
        cost = cost_rows(rows)["actual_total"]
        self.assertEqual(cost["ledger_attempt_records"], 3)
        self.assertEqual(cost["recorded_execution_path_attempts"], 2)
        self.assertEqual(cost["unresolved_unsuccessful_execution_attempts"], 2)
        self.assertEqual(cost["completed_calls"], 0)
        self.assertIsNone(cost["actual_call_count"])
        self.assertEqual(cost["actual_call_count_status"], "UNCERTAIN")
        self.assertEqual(cost["budget_reserved_output_tokens"], 200)
        self.assertEqual(cost["output_tokens"], 0)
        self.assertEqual(cost["request_seconds"], 5)
        unattempted = cost_rows(rows[-1:])["actual_total"]
        self.assertEqual(unattempted["actual_call_count"], 0)
        self.assertEqual(unattempted["recorded_execution_path_attempts"], 0)
        completed = cost_rows([base | {"runtime_status": "COMPLETED", "output_tokens": 7,
                                      "attempt_id": "completed"}])["actual_total"]
        self.assertEqual(completed["actual_call_count"], 1)
        self.assertEqual(completed["budget_reserved_output_tokens"], 0)

    def test_review_bounds_fixed_denominator_and_partial(self):
        rows = [{"status": "RESOLVED", "candidate_correct": True},
                {"status": "RESOLVED", "candidate_correct": False},
                {"status": "REVIEW", "candidate_correct": True},
                {"status": "SYSTEM_ERROR", "candidate_correct": None}]
        envelope = decision_envelope(rows, 900)
        self.assertEqual(envelope["resolved_true"], 1)
        self.assertEqual(envelope["unresolved"], 898)
        self.assertEqual(envelope["missing"], 896)
        self.assertEqual(envelope["conditional_lower"], 1 / 900)
        self.assertEqual(envelope["conditional_upper"], 899 / 900)
        with self.assertRaises(ValueError):
            decision_envelope([{"status": "RESOLVED", "candidate_correct": None}], 30)

    def test_structure_exact_not_semantic(self):
        row = {"mode": "P", "structure_status": "SCORED_EXACT_JSON", "structure_exact": False,
               "schema_parse_status": "VALID", "missing_fields": 1, "unmatched_list_items": 2,
               "relation_precision": .5, "relation_recall": None,
               "fields": [{"field": "edges", "exact": False, "missing": True},
                          {"field": "nodes", "exact": True, "missing": False}]}
        result = structure_summary([row])
        self.assertEqual(result["field_exact_rate"], .5)
        self.assertEqual(result["relations"]["relation_precision"]["macro_mean"], .5)
        self.assertIsNone(result["relations"]["relation_recall"]["macro_mean"])
        self.assertIsNone(result["semantic_structure_exact_rate"])


if __name__ == "__main__":
    unittest.main()
