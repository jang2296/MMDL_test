"""Authored evidence only. Mocked full index/FROZEN protocol are NOT real approvals."""
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from fine_tuning.notebook_rl import approval as a, contamination as c
from fine_tuning.notebook_rl.dataset import build_splits, freeze_selection
from fine_tuning.notebook_rl.generators.common import GeneratedProblem
from fine_tuning.notebook_rl.schema import ActorItem, load_protocol
from fine_tuning.tests.test_notebook_rl_contamination import index_fixture
from fine_tuning.tests.test_notebook_rl_schema import records


def write(path, value):
    raw = a._json(value) + b"\n"
    path.write_bytes(raw)
    return a._sha(raw)


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "build"
        actor, verifier, lineage, *_ = records()
        self.pools = {}
        for si, split in enumerate(a._SPLITS):
            self.pools[split] = []
            for j in range(13):
                i = si * 13 + j
                stream = io.BytesIO()
                Image.fromarray(np.random.default_rng(i + 100).integers(0, 256, (18, 18, 3), dtype=np.uint8)).save(stream, format="PNG")
                raw, pid = stream.getvalue(), f"fixture:{split}:{j:02d}"
                item = replace(actor, problem_id=pid, image_paths=(f"{i}.png",), image_sha256=(a._sha(raw),))
                self.pools[split].append(GeneratedProblem(item, replace(verifier, problem_id=pid),
                    replace(lineage, problem_id=pid, split=split, structure_hash=a._sha(str(i).encode())), (raw,)))
        self.manifest = build_splits(self.root, self.pools)
        self.protocol = load_protocol(a._HERE / "protocol_v1.yaml")
        self.registry = {"sources": {"fake": {"rights_approved": True, "source_known": True,
            "benchmark_derived": False, "model_touched": False, "label_provenance": "PROGRAM",
            "source_revision": "fixture-v1", "license": "INTERNAL_FIXTURE", "upstream_origin": "procedural",
            "parents": [], "upstream_split": "generated"}}, "upstream_origin_blocklist": [],
            "approved_human_reviewers": {"authored-attestation": {"kind": "HUMAN", "approved": True}}}
        self.index_path = self.base / "index.json"
        self.index_data = index_fixture()
        index_sha = write(self.index_path, self.index_data)
        self.index = c._prepare(self.index_data, "authored-test-only")
        self.index.update(full_coverage=True, limitations=[], partial_name=None,
                          path=str(self.index_path), file_sha256=index_sha)
        self.descriptor = {"path": str(self.index_path), "file_sha256": index_sha}
        items, self.input_files, images = c._snapshot({s: self.root / s for s in a._SPLITS}, self.root / "images")
        self.items = items
        calibration = {"frozen_before_scan": True, "protocol_sha256": self.protocol["sha256"],
            "index_sha256": self.index["sha256"], "positive_count": 4, "negative_count": 4, "tpr": .75, "fpr": 0.,
            "per_engine": {k: {"positive_count": 4, "negative_count": 4, "tpr": .75, "fpr": 0., "examples_sha256": "d" * 64}
                           for k in ("dhash", "tiles", "sscd", "siglip")},
            "thresholds": {"perceptual_hamming": 3, "sscd_exclude": .95, "sscd_review": .75, "siglip_review": .9}}
        self.evidence = {"calibration": calibration}
        self.registry["scanner_policy"] = {"status": "FROZEN_BEFORE_INPUT_SCAN", "protocol_sha256": self.protocol["sha256"],
                                          "calibration_sha256": a._hash(calibration), "engines": {}}
        for name, engine in (("ocr", "tesseract"), ("sscd", "sscd_disc_mixup"), ("siglip", "google/siglip-fixture")):
            identity = {"engine_id": engine, "revision": "authored-fixture-v1", "artifact_sha256": "e" * 64}
            self.registry["scanner_policy"]["engines"][name] = identity.copy()
            self.evidence[name] = identity | {"index_sha256": self.index["sha256"], "calibration_sha256": a._hash(calibration)}
        self.evidence["ocr"].update(benchmark={sha: {"image_sha256": sha, "text": "authored unrelated diagram label", "status": "COMPLETE"}
                                                   for sha in self.index["images"]}, inputs={})
        def vectors(bindings, vector):
            return {key: {"sha256": sha, "vector": vector} for key, sha in bindings.items()}
        for name in ("sscd", "siglip"):
            self.evidence[name].update(benchmark_images=vectors({sha: sha for sha in self.index["images"]}, [1., 0.]),
                                      input_images=vectors({sha: sha for sha in images}, [0., 1.]))
        text_refs = {entry["key"]: a._sha(c.normalize_text(entry["text"]).encode()) for entry in self.index["text_cache"]["entries"]}
        self.evidence["siglip"].update(benchmark_texts=vectors(text_refs, [1., 0.]),
            input_texts=vectors({pid: item["text_sha256"] for pid, item in items.items()}, [0., 1.]))
        # The scanner's approval vocabulary is isolated with an authored index;
        # no test represents these 52 images as native benchmark coverage.
        with patch.object(c, "_prepare", return_value=self.index):
            self.receipt = c.scan({s: self.root / s for s in a._SPLITS}, self.root / "images",
                                  self.index, self.registry, self.evidence)
        self.assertEqual(self.receipt["status"], "EXCLUSION_CHECKED_WITH_LIMITS", self.receipt["blockers"])
        self.manifest["exclusion_receipt_sha256"] = write(self.root / "contamination" / "receipt_v2.json", self.receipt)
        self.build_sha = write(self.root / "manifest.yaml", self.manifest)
        self.phase_path, self.selection_path = self.base / "phase.json", self.base / "selection.json"
        rows = [{"problem_id": row.actor.problem_id, "family": row.actor.family, "correct": 2, "n": 8,
                 "blind_correct": 0, "blind_n": 4, "truncated": 0, "n_choices": 0, "context_ok": True}
                for row in self.pools["train"]]
        self.phase = {"status": "PHASE_A_COMPLETED", "policy_version": 0, "split": "train", "build_id": self.build_sha,
                      "rows": rows, "max_new_tokens": {"1": 3072, "2": 3072, "3": 3072},
                      "candidate_ids_sha256": a._hash(sorted(r["problem_id"] for r in rows))}
        phase_sha = write(self.phase_path, self.phase)
        self.selection = freeze_selection(self.phase_path, {"max_pass_rate": .75, "max_zero_pass_share": .5,
            "blind_floor": .25, "truncation_family_max": .1, "max_selected": 384, "min_open_answer_fraction": .6}, output=self.selection_path)
        self.binding = {"build_manifest_sha256": self.build_sha, "selection_sha256": a._sha(self.selection_path.read_bytes()),
            "phase_a_report_sha256": phase_sha, "exclusion_receipt_sha256": self.manifest["exclusion_receipt_sha256"],
            "input_files_sha256": self.receipt["input_files_sha256"], "protocol_sha256": self.protocol["sha256"],
            "index_sha256": self.index["sha256"], "index_file_sha256": self.index["file_sha256"],
            "source_registry_sha256": a._hash(self.registry), "evidence_sha256": a._hash(self.evidence)}
        producer = self.base / "authored_producer.py"
        producer.write_text("# Authored unit-test evidence producer, not actual GPU execution.\n")
        producer_descriptor = {"path": str(producer), "file_sha256": a._sha(producer.read_bytes())}
        baseline_path = a._PROJECT / self.protocol["initial_checkpoint"]["manifest"]
        baseline = json.loads(baseline_path.read_bytes())
        self.scope = {"protocol_sha256": self.protocol["sha256"], "code_sha256": a.code_hashes(),
            "input_files_sha256": self.receipt["input_files_sha256"], "runtime_id": "authored-not-a-real-runtime",
            "arm": "MOD-FML", "environment_lock_sha256": a._sha((a._PROJECT / self.protocol["environment"]["lock"]).read_bytes()),
            "model": {"model_id": baseline["repo_id"], "revision": baseline["revision"], "manifest_sha256": a._sha(baseline_path.read_bytes()),
                      "files_sha256": {f["path"]: f["sha256"] for f in baseline["files"]}},
            "device": {"type": "cuda", "uuid": "GPU-AUTHORED-FIXTURE", "name": "Authored fixture", "index": 0,
                       "total_memory_bytes": 1000000, "driver_version": "fixture"},
            "probes": [{"id": "single", "image_count": 1, "input_sha256": "1" * 64},
                       {"id": "multi", "image_count": 2, "input_sha256": "2" * 64}],
            "producers": {"data": producer_descriptor, "gpu": producer_descriptor}}
        self.proofs = {}
        self.data_report = self.report("D", range(1, 9), self.data_measurements())
        self.gpu_report = self.report("G", range(14), self.gpu_measurements())
        self.payload = {"build_root": str(self.root), "selection_path": str(self.selection_path), "phase_a_report": str(self.phase_path),
            "index": self.descriptor, "source_registry": self.registry, "evidence": self.evidence,
            "data_report": self.data_report, "gpu_report": self.gpu_report, "expected_gpu_scope": self.scope}

    def data_measurements(self):
        manifests = {s: json.loads((self.root / s / "manifest.yaml").read_bytes()) for s in a._SPLITS}
        return {"D1": {"checked_ids_sha256": a._hash(sorted(self.items)), "items_checked": 52, "repeat_mismatches": 0},
            "D2": {"checked_ids_sha256": a._hash(sorted(self.items)), "items_checked": 52, "solver_mismatches": 0, "independent_solvers": True},
            "D3": {"sampling_seed": 3407, "samples": [{"problem_id": pid, "image_sha256": self.items[pid]["image_sha256"],
                    "printed_text_slots_consistent": True, "mismatches": 0}
                    for pid in sorted(self.items, key=lambda pid: a._sha(f"3407:{pid}".encode()))[:50]]},
            "D4": {"templates": {"fake:read-cell": {"pairs": 100, "answer_changed": 95,
                    "question_unchanged": True, "nonimage_inputs_unchanged": True}}},
            "D5": {"structure_cross_split": 0, "image_cross_split": 0},
            "D6": {"exclusion_receipt_sha256": self.binding["exclusion_receipt_sha256"]},
            "D7": {"reviewed_actor_files": {p: sha for p, sha in self.input_files.items() if Path(p).name == "actor_inputs.jsonl"},
                   "warnings_sha256": a._hash({s: m["leak_review"] for s, m in manifests.items()}), "unresolved": 0},
            "D8": {"separate_process": True, "private_open_denied": True, "benchmark_open_denied": True,
                   "teacher_import_denied": True, "python_guard_not_os_sandbox": True}}

    def gpu_measurements(self):
        return {"G0": {"lock_match": True, "cuda": True, "bf16_matmul": True, "model_executor": True, "vllm_multiprocessing": False},
            "G1": {"trainable_counts": {"L": 33030144, "M": 109105152, "V": 0}, "mergers": 4, "master_dtype": "float32", "base_dtype": "bfloat16"},
            "G2": {"images_checked": 3, "grid_mismatches": 0, "token_feature_mismatches": 0},
            "G3": {"initial_max_abs": 0., "post_sync_max_abs": 0., "frozen_sha_unchanged": True, "requires_grad_restored": True},
            "G4": {"merger_changed": True, "frozen_unchanged": True},
            "G5": {"all_tensors_match": True, "all_four_mergers": True, "hf_base_unchanged": True},
            "G6": {"fresh_engine_match": True, "default_cache_comparison_measured": True},
            "G7": {"trajectories": 32, "synthetic_0_1_masked": True, "delta_seq": .05, "step0_mean_abs": .01, "post_sync_mean_abs": .01},
            "G8": {"tokens_exact": True, "generated_only_mask": True, "eos_checked": True},
            "G9": {"vision_off_grad_none": True, "nonvision_grad_finite_nonzero": True, "hard_freeze_gradient_match": True,
                "vision_parameter_count": 0, "vision_on_grad_nonzero": None, "mixed_on_only": None,
                "vision_applicability": "NOT_APPLICABLE_MOD_FML", "vision_on_status": "NOT_APPLICABLE_MOD_FML",
                "mixed_status": "NOT_APPLICABLE_MOD_FML", "vision_na_reason": "NO_INSTALLED_VISION_ADAPTERS"},
            "G10": {"numeric_tests_match": True, "trl_formula_match": True},
            "G11": {"longest_sample_step_complete": True, "no_oom": True, "peak_memory_bytes": 100, "step_seconds": 1.5},
            "G12": {"fresh_process_reload": True, "policy_logits_equal": True, "reference_logits_equal": True, "next_loss_equal": True},
            "G13": {"exclusion_allowlist_only": True, "ce_patch_raises": True, "teacher_import_denied": True}}

    def report(self, prefix, indices, measurements):
        report = {"format": f"notebook-rl-{'data' if prefix == 'D' else 'gpu'}-gates-v1", "status": "COMPLETE",
                  "binding": copy.deepcopy(self.binding), "scope": copy.deepcopy(self.scope), "gates": {}}
        for index in indices:
            gate = f"{prefix}{index}"
            proof = {"gate": gate, "status": "PASS", "binding_sha256": a._hash(self.binding), "scope_sha256": a._hash(self.scope),
                     "producer_sha256": self.scope["producers"]["data" if prefix == "D" else "gpu"]["file_sha256"],
                     "measurements": measurements[gate]}
            if gate in ("D3", "D7"):
                proof["provenance"] = {"reviewer_id": "authored-attestation", "kind": "HUMAN",
                    "method": "DIRECT_ORIGINAL_INPUT_REVIEW", "reviewed_at_utc": "2026-10-01T00:00:00+00:00"}
            path = self.base / (gate + ".json")
            report["gates"][gate] = {"status": "PASS", "artifact": {"path": str(path), "file_sha256": write(path, proof)}}
            self.proofs[gate] = proof
        return report

    def change_proof(self, gate, mutate):
        proof = self.proofs[gate]
        mutate(proof)
        report = self.data_report if gate.startswith("D") else self.gpu_report
        entry = report["gates"][gate]["artifact"]
        entry["file_sha256"] = write(Path(entry["path"]), proof)

    def validate(self, **changes):
        frozen = self.protocol | {"status": "FROZEN"}
        # Freeze the PUBLIC code fixture while disjoint collaborators may edit
        # unrelated production modules. Runtime code_hashes remains fail-closed.
        with patch.object(c, "load_index", return_value=self.index), patch.object(a, "code_hashes", return_value=self.scope["code_sha256"]), \
                patch("fine_tuning.notebook_rl.schema.load_protocol", return_value=frozen):
            return a._validate(self.payload | changes)

    def assert_blocked(self, result, code=None):
        self.assertEqual(result["status"], "BLOCKED_NOT_CHECKED")
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["approval_created"])
        self.assertEqual((result["actors"], result["structures"], result["selected_ids"]), ([], {}, []))
        if code:
            self.assertIn(code, result["blockers"])

    def test_authored_complete_contract_returns_only_actor_structure_and_bindings(self):
        before = {str(p): a._sha(p.read_bytes()) for p in self.base.rglob("*") if p.is_file()}
        result = self.validate()
        self.assertEqual(result["status"], "VALIDATED_FOR_TRAINING", result["blockers"])
        self.assertEqual(len(result["actors"]), 13)
        self.assertEqual(set(result["structures"]), set(self.selection["selected_train"]))
        self.assertFalse(result["approval_created"])
        for forbidden in ('"gold"', '"slots"', '"oracle_fact_lines"', '"printed_text"', '"human_verified"'):
            self.assertNotIn(forbidden, json.dumps(result))
        self.assertEqual(before, {str(p): a._sha(p.read_bytes()) for p in self.base.rglob("*") if p.is_file()})

    def test_actual_current_protocol_and_partial_index_never_approve(self):
        self.assert_blocked(a._validate(self.payload), "PROTOCOL_NOT_FROZEN")
        self.index["full_coverage"] = False
        self.assert_blocked(self.validate(), "FULL_INDEX_BINDING")

    def test_missing_partial_or_naked_pass_gate_reports_rejected(self):
        for changes in ({"data_report": None}, {"gpu_report": None}, {"expected_gpu_scope": None}):
            self.assert_blocked(self.validate(**changes))
        self.gpu_report["gates"]["G3"] = {"status": "PASS"}
        self.assert_blocked(self.validate())

    def test_selected_id_must_be_allowlisted_and_phase_a_recomputed(self):
        self.selection["selected_train"] = ["fixture:sealed:00"]
        write(self.selection_path, self.selection)
        self.assert_blocked(self.validate(), "SELECTION_RECOMPUTATION_MISMATCH")
        self.selection["selected_train"] = sorted(row.actor.problem_id for row in self.pools["train"])
        write(self.selection_path, self.selection)
        self.receipt["candidates"].append({"problem_id": self.selection["selected_train"][0], "stage": "text_ngrams", "reason": "AUTHORED_TEST", "action": "REVIEW"})
        family = c.family_quarantine(self.items, self.receipt["candidates"])
        self.receipt.update(allowlist=family["allowlist"], quarantined_ids=family["quarantined_ids"])
        self.receipt["receipt_sha256"] = a._hash({k: v for k, v in self.receipt.items() if k != "receipt_sha256"})
        self.manifest["exclusion_receipt_sha256"] = write(self.root / "contamination" / "receipt_v2.json", self.receipt)
        build_sha = write(self.root / "manifest.yaml", self.manifest)
        # Even a newly bound selection cannot train an explicitly quarantined ID.
        self.phase["build_id"] = build_sha
        self.selection["build_id"] = build_sha
        self.selection["phase_a_report_sha256"] = write(self.phase_path, self.phase)
        write(self.selection_path, self.selection)
        self.assert_blocked(self.validate(), "SELECTED_ID_NOT_EXCLUSION_ALLOWLISTED")

    def test_input_registry_evidence_and_code_mutations_rejected(self):
        image = self.root / "images" / "0.png"
        raw = image.read_bytes()
        image.write_bytes(raw + b"changed")
        self.assert_blocked(self.validate(), "IMAGE_CHANGED")
        image.write_bytes(raw)
        registry = copy.deepcopy(self.registry)
        registry["sources"]["fake"]["rights_approved"] = False
        self.assert_blocked(self.validate(source_registry=registry), "EXCLUSION_EVIDENCE_BINDING")
        evidence = copy.deepcopy(self.evidence)
        evidence["sscd"]["input_images"].clear()
        self.assert_blocked(self.validate(evidence=evidence), "EXCLUSION_EVIDENCE_BINDING")
        scope = copy.deepcopy(self.scope)
        scope["code_sha256"]["approval.py"] = "0" * 64
        self.assert_blocked(self.validate(expected_gpu_scope=scope), "GPU_CURRENT_SCOPE_MISMATCH")

    def test_human_review_requires_registered_human_provenance_and_exact_images(self):
        original = copy.deepcopy(self.proofs["D3"])
        self.change_proof("D3", lambda p: p["provenance"].update(kind="AI_REVIEWED"))
        self.assert_blocked(self.validate(), "HUMAN_REVIEW_PROVENANCE")
        self.proofs["D3"] = copy.deepcopy(original)
        self.change_proof("D3", lambda p: p["measurements"]["samples"][0].update(image_sha256=["f" * 64]))
        self.assert_blocked(self.validate(), "D3_INPUT_BINDING_OR_MISMATCH")
        self.proofs["D3"] = copy.deepcopy(original)
        self.change_proof("D3", lambda p: p["measurements"]["samples"].pop())
        self.assert_blocked(self.validate(), "D3_FIFTY_PER_SOURCE_REQUIRED")

    def test_leak_review_and_image_dependence_require_full_exact_scope(self):
        self.change_proof("D7", lambda p: p["measurements"]["reviewed_actor_files"].clear())
        self.assert_blocked(self.validate(), "D7_LEAK_REVIEW_BINDING")
        self.change_proof("D7", lambda p: p["measurements"].update(self.data_measurements()["D7"]))
        self.change_proof("D4", lambda p: p["measurements"]["templates"]["fake:read-cell"].update(answer_changed=94))
        self.assert_blocked(self.validate(), "D4_IMAGE_DEPENDENCE")

    def test_gpu_runtime_model_device_probe_and_producer_scope(self):
        for mutate in (lambda s: s["model"].update(revision="main"),
                       lambda s: s["probes"].pop(), lambda s: s["device"].update(type="cpu"),
                       lambda s: s["producers"]["gpu"].update(file_sha256="0" * 64)):
            scope = copy.deepcopy(self.scope)
            mutate(scope)
            self.assert_blocked(self.validate(expected_gpu_scope=scope))
        report = copy.deepcopy(self.gpu_report)
        report["scope"]["device"]["uuid"] = "GPU-OTHER"
        self.assert_blocked(self.validate(gpu_report=report), "G_REPORT_BINDING")

    def test_gpu_metrics_and_artifact_tampering_are_not_naked_pass(self):
        self.change_proof("G7", lambda p: p["measurements"].update(trajectories=31))
        self.assert_blocked(self.validate(), "G7_TRAJECTORIES")
        self.change_proof("G7", lambda p: p["measurements"].update(trajectories=32))
        self.change_proof("G11", lambda p: p["measurements"].update(peak_memory_bytes=2000000))
        self.assert_blocked(self.validate(), "G11_ACTUAL_MEASUREMENTS")
        self.change_proof("G11", lambda p: p["measurements"].update(peak_memory_bytes=100))
        Path(self.gpu_report["gates"]["G0"]["artifact"]["path"]).write_text("changed")
        self.assert_blocked(self.validate())

    def test_fml_requires_explicit_vision_na_but_actual_nonvision_measurements(self):
        self.assertEqual(self.validate()["status"], "VALIDATED_FOR_TRAINING")
        for key, value in (("mixed_on_only", True), ("vision_on_grad_nonzero", False),
                           ("mixed_status", "MEASURED"), ("vision_na_reason", None),
                           ("vision_parameter_count", False), ("vision_parameter_count", 1),
                           ("hard_freeze_gradient_match", None), ("vision_off_grad_none", None)):
            proofs = copy.deepcopy(self.proofs)
            proofs["G9"]["measurements"][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(a._Blocked, "G9"):
                a._gpu_gates(proofs, self.scope, self.protocol)
        for key in ("mixed_on_only", "vision_on_grad_nonzero", "vision_parameter_count", "mixed_status"):
            proofs = copy.deepcopy(self.proofs)
            proofs["G9"]["measurements"].pop(key)
            with self.subTest(missing=key), self.assertRaisesRegex(a._Blocked, "G9"):
                a._gpu_gates(proofs, self.scope, self.protocol)

    def test_installed_vision_cannot_use_fml_na(self):
        proofs, scope = copy.deepcopy(self.proofs), copy.deepcopy(self.scope)
        scope["arm"] = "MOD-RV"
        count = self.protocol["expected_trainable"]["V"]
        proofs["G1"]["measurements"]["trainable_counts"]["V"] = count
        g9 = proofs["G9"]["measurements"]
        g9["vision_parameter_count"] = count
        with self.assertRaisesRegex(a._Blocked, "G9_VISION_ON"):
            a._gpu_gates(proofs, scope, self.protocol)
        g9.update(vision_on_grad_nonzero=True, mixed_on_only=True, vision_applicability="MEASURED",
                  vision_on_status="MEASURED", mixed_status="MEASURED", vision_na_reason=None)
        a._gpu_gates(proofs, scope, self.protocol)
        for key, value in (("mixed_on_only", None), ("vision_on_grad_nonzero", False),
                           ("vision_parameter_count", 0), ("vision_on_status", "NOT_APPLICABLE_MOD_FML")):
            invalid = copy.deepcopy(proofs)
            invalid["G9"]["measurements"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(a._Blocked, "G9"):
                a._gpu_gates(invalid, scope, self.protocol)

    def test_rebound_zero_norm_or_missing_calibration_still_cannot_pass(self):
        next(iter(self.evidence["sscd"]["input_images"].values()))["vector"] = [0., 0.]
        self.receipt["evidence_sha256"] = a._hash(self.evidence)
        self.receipt["receipt_sha256"] = a._hash({k: v for k, v in self.receipt.items() if k != "receipt_sha256"})
        self.manifest["exclusion_receipt_sha256"] = write(self.root / "contamination" / "receipt_v2.json", self.receipt)
        build_sha = write(self.root / "manifest.yaml", self.manifest)
        self.phase["build_id"] = build_sha
        self.selection["build_id"] = build_sha
        self.selection["phase_a_report_sha256"] = write(self.phase_path, self.phase)
        write(self.selection_path, self.selection)
        self.assert_blocked(self.validate(), "MISSING_MALFORMED_OR_CHANGED_PREREQUISITE")
        self.evidence.pop("calibration")
        self.receipt["evidence_sha256"] = a._hash(self.evidence)
        self.receipt["receipt_sha256"] = a._hash({k: v for k, v in self.receipt.items() if k != "receipt_sha256"})
        self.manifest["exclusion_receipt_sha256"] = write(self.root / "contamination" / "receipt_v2.json", self.receipt)
        build_sha = write(self.root / "manifest.yaml", self.manifest)
        self.phase["build_id"] = build_sha
        self.selection["build_id"] = build_sha
        self.selection["phase_a_report_sha256"] = write(self.phase_path, self.phase)
        write(self.selection_path, self.selection)
        self.assert_blocked(self.validate(), "MISSING_MALFORMED_OR_CHANGED_PREREQUISITE")

    def test_public_success_decodes_actor_allowlist_without_private_payload(self):
        result = self.validate()
        self.assertEqual(result["status"], "VALIDATED_FOR_TRAINING")
        kwargs = {key: self.payload[key] for key in ("index", "source_registry", "evidence", "data_report", "gpu_report", "expected_gpu_scope")}
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=a._json(result))):
            public = a.validate_for_training(self.root, self.selection_path, phase_a_report=self.phase_path, **kwargs)
        self.assertEqual(public["status"], "VALIDATED_FOR_TRAINING")
        self.assertTrue(all(type(actor) is ActorItem for actor in public["actors"]))
        self.assertFalse(public["approval_created"])
        self.assertEqual(set(public["structures"]), set(self.selection["selected_train"]))

    def test_public_facade_spawns_and_never_opens_private_files_in_actor(self):
        script = """
import pathlib,sys,json
from fine_tuning.notebook_rl.approval import validate_for_training
from fine_tuning.notebook_rl.guards import install_guards
root=pathlib.Path(sys.argv[1])
files=[root/s/name for s in ('train','dev','sealed','style_sealed') for name in ('lineage.jsonl','verifier_private.jsonl')]
install_guards(forbidden_roots=[root.parent/'index.json'],vault_files=files,allow_vault=False,allow_benchmark=False)
result=validate_for_training(root,root.parent/'selection.json',phase_a_report=root.parent/'phase.json',
 index={'path':str(root.parent/'index.json'),'file_sha256':'0'*64},source_registry={},evidence={},
 data_report=None,gpu_report=None,expected_gpu_scope=None)
assert result['status']=='BLOCKED_NOT_CHECKED' and not result['actors']
print(json.dumps(result))
"""
        process = subprocess.run([sys.executable, "-c", script, str(self.root)], capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["blockers"], ["PROTOCOL_NOT_FROZEN"])

    def test_public_timeout_and_poisoned_response_fail_closed(self):
        kwargs = {key: self.payload[key] for key in ("index", "source_registry", "evidence", "data_report", "gpu_report", "expected_gpu_scope")}
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("fixture", .1)):
            self.assert_blocked(a.validate_for_training(self.root, self.selection_path, phase_a_report=self.phase_path, **kwargs), "PREFLIGHT_TIMEOUT")
        poisoned = self.validate() | {"gold": "PRIVATE-NOT-ALLOWED"}
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=a._json(poisoned))):
            self.assert_blocked(a.validate_for_training(self.root, self.selection_path, phase_a_report=self.phase_path, **kwargs), "PREFLIGHT_RESPONSE_INVALID")

    def measurement_fixture(self):
        """Separate authored D evidence; never re-label existing training proofs."""
        binding = {k: v for k, v in self.binding.items() if k not in ("selection_sha256", "phase_a_report_sha256")}
        scope = {"purpose": "FROZEN_WEIGHT_MEASUREMENT_ONLY", "protocol_sha256": self.protocol["sha256"],
            "code_sha256": self.scope["code_sha256"], "input_files_sha256": self.binding["input_files_sha256"],
            "producer": self.scope["producers"]["data"]}
        report = copy.deepcopy(self.data_report)
        report.update(binding=binding, scope=scope)
        for gate, entry in report["gates"].items():
            proof = copy.deepcopy(self.proofs[gate])
            proof.update(binding_sha256=a._hash(binding), scope_sha256=a._hash(scope))
            path = self.base / ("measurement-" + gate + ".json")
            entry["artifact"] = {"path": str(path), "file_sha256": write(path, proof)}
        return {"stage": "measurement", "build_root": str(self.root), "index": self.descriptor,
            "source_registry": self.registry, "evidence": self.evidence, "data_report": report,
            "expected_measurement_scope": scope}

    def measurement_validate(self, payload):
        with patch.object(c, "load_index", return_value=self.index), \
                patch.object(a, "code_hashes", return_value=self.scope["code_sha256"]), \
                patch("fine_tuning.notebook_rl.schema.load_protocol", return_value=self.protocol):
            return a._validate(payload)

    def test_designed_measurement_requires_no_future_phase_or_gpu_and_never_trains(self):
        payload = self.measurement_fixture()
        # These deliberately do not exist: prerequisite validation must not open them.
        payload.update(selection_path=str(self.base/"not-yet-selection.json"),
                       phase_a_report=str(self.base/"not-yet-phase.json"))
        before = {str(p): a._sha(p.read_bytes()) for p in self.base.rglob("*") if p.is_file()}
        result = self.measurement_validate(payload)
        self.assertEqual(result["status"], "VALIDATED_FOR_MEASUREMENT", result["blockers"])
        self.assertTrue(result["measurement_allowed"])
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["approval_created"])
        self.assertEqual(result["selected_ids"], sorted(r.actor.problem_id for r in self.pools["train"]))
        self.assertNotIn("selection_sha256", result)
        for forbidden in ('"gold"', '"slots"', '"oracle_fact_lines"', '"printed_text"', '"human_verified"'):
            self.assertNotIn(forbidden, json.dumps(result))
        self.assertEqual(before, {str(p): a._sha(p.read_bytes()) for p in self.base.rglob("*") if p.is_file()})
        self.assert_blocked(a._validate(self.payload), "PROTOCOL_NOT_FROZEN")

    def test_measurement_full_index_human_review_scope_and_producer_remain_mandatory(self):
        payload = self.measurement_fixture()
        self.index["full_coverage"] = False
        self.assert_blocked(self.measurement_validate(payload), "FULL_INDEX_BINDING")
        self.index["full_coverage"] = True
        for change in ({"purpose": "TRAINING"}, {"protocol_sha256": "0"*64},
                       {"code_sha256": {}}, {"input_files_sha256": "0"*64},
                       {"producer": {"path": str(self.base/"missing.py"), "file_sha256": "0"*64}}):
            invalid = copy.deepcopy(payload)
            invalid["expected_measurement_scope"].update(change)
            self.assert_blocked(self.measurement_validate(invalid))
        entry = payload["data_report"]["gates"]["D3"]["artifact"]
        proof = json.loads(Path(entry["path"]).read_text())
        proof["provenance"]["kind"] = "AI_REVIEWED"
        entry["file_sha256"] = write(Path(entry["path"]), proof)
        self.assert_blocked(self.measurement_validate(payload), "HUMAN_REVIEW_PROVENANCE")

    def test_measurement_quarantined_candidate_blocks_whole_pool_not_subset(self):
        payload = self.measurement_fixture()
        self.receipt["candidates"].append({"problem_id": self.pools["train"][0].actor.problem_id,
            "stage": "text_ngrams", "reason": "AUTHORED_TEST", "action": "REVIEW"})
        self.receipt.update(c.family_quarantine(self.items, self.receipt["candidates"]))
        self.receipt["receipt_sha256"] = a._hash({k: v for k, v in self.receipt.items() if k != "receipt_sha256"})
        self.manifest["exclusion_receipt_sha256"] = write(self.root/"contamination/receipt_v2.json", self.receipt)
        write(self.root/"manifest.yaml", self.manifest)
        self.assert_blocked(self.measurement_validate(payload), "SELECTED_ID_NOT_EXCLUSION_ALLOWLISTED")

    def test_unchanged_prelearning_data_proofs_can_be_reused_but_gpu_gates_still_required(self):
        payload = self.measurement_fixture()
        before = {e["artifact"]["path"]: Path(e["artifact"]["path"]).read_bytes()
                  for e in payload["data_report"]["gates"].values()}
        result = self.validate(data_report=payload["data_report"])
        self.assertEqual(result["status"], "VALIDATED_FOR_TRAINING", result["blockers"])
        self.assertEqual(before, {p: Path(p).read_bytes() for p in before})
        self.assert_blocked(self.validate(data_report=payload["data_report"], gpu_report=None))
        invalid = copy.deepcopy(payload["data_report"])
        invalid["binding"]["protocol_sha256"] = "0"*64
        self.assert_blocked(self.validate(data_report=invalid), "D_REPORT_BINDING")

    def test_measurement_public_facade_separates_status_permission_and_private_fields(self):
        payload = self.measurement_fixture()
        result = self.measurement_validate(payload)
        self.assertEqual(result["status"], "VALIDATED_FOR_MEASUREMENT")
        kwargs = {k: payload[k] for k in ("index", "source_registry", "evidence", "data_report", "expected_measurement_scope")}
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=a._json(result))):
            checked = a.validate_for_measurement(self.root, **kwargs)
        self.assertTrue(all(type(actor) is ActorItem for actor in checked["actors"]))
        self.assertFalse(checked["training_allowed"])
        for change in ({"status": "VALIDATED_FOR_TRAINING"}, {"training_allowed": True},
                       {"measurement_allowed": False}, {"gold": "PRIVATE-KEY"}):
            with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=a._json(result | change))):
                self.assert_blocked(a.validate_for_measurement(self.root, **kwargs), "PREFLIGHT_RESPONSE_INVALID")
        # A measurement response cannot impersonate final training approval either.
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=a._json(result))):
            self.assert_blocked(a._public_validate(self.payload, 30), "PREFLIGHT_RESPONSE_INVALID")

    def test_measurement_changed_input_or_protocol_invalidates_prior_evidence(self):
        payload = self.measurement_fixture()
        original = self.protocol
        self.protocol = copy.deepcopy(original)
        self.protocol["sha256"] = "0"*64
        self.assert_blocked(self.measurement_validate(payload), "EXCLUSION_EVIDENCE_BINDING")
        self.protocol = original
        (self.root/"images/0.png").write_bytes(b"changed")
        self.assert_blocked(self.measurement_validate(payload), "IMAGE_CHANGED")

    def limited_fixture(self):
        """Authored review provenance only, not real AI or full-index evidence."""
        self.registry["limited_smoke_reviewers"] = {"fixture-ai": {
            "kind": "AI", "approved_for": a.SMOKE_REVIEW_POLICY, "human_verified": False}}
        self.receipt["source_registry_sha256"] = a._hash(self.registry)
        self.receipt["receipt_sha256"] = a._hash({k: v for k, v in self.receipt.items() if k != "receipt_sha256"})
        receipt_sha = write(self.root/"contamination/receipt_v2.json", self.receipt)
        self.manifest["exclusion_receipt_sha256"] = receipt_sha
        self.binding.update(build_manifest_sha256=write(self.root/"manifest.yaml", self.manifest),
                            exclusion_receipt_sha256=receipt_sha, source_registry_sha256=a._hash(self.registry))
        payload = self.measurement_fixture()
        payload["stage"] = "limited_smoke"
        scope = payload["expected_measurement_scope"]
        scope.update(purpose=a.SMOKE_PURPOSE, authorization={
            "approval_reference": "AUTHORED_UNIT_TEST_NOT_USER_APPROVAL",
            "approved_at_utc": "2026-10-17T00:00:00Z", "review_policy": a.SMOKE_REVIEW_POLICY,
            "max_pool_items": 64, "max_train_items": 24, "max_optimizer_updates": 2,
            "training_campaign_allowed": False})
        payload["data_report"]["scope"] = copy.deepcopy(scope)
        for gate, entry in payload["data_report"]["gates"].items():
            path = Path(entry["artifact"]["path"])
            proof = json.loads(path.read_bytes())
            proof["scope_sha256"] = a._hash(scope)
            if gate in ("D3", "D7"):
                proof["provenance"] = {"reviewer_id": "fixture-ai", "kind": "AI_REVIEWED_PROGRAM_VERIFIED",
                    "method": "DIRECT_ORIGINAL_INPUT_AND_PROGRAM_REVIEW", "reviewed_at_utc": "2026-10-17T00:00:00Z",
                    "human_verified": False, "labels_modified": False}
            if gate == "D3":
                proof["measurements"]["samples"] = [{"problem_id": pid, "image_sha256": item["image_sha256"],
                    "printed_text_slots_consistent": True, "mismatches": 0} for pid, item in self.items.items()]
            if gate == "D6":
                proof["measurements"] = {"exclusion_receipt_sha256": receipt_sha}
            entry["artifact"]["file_sha256"] = write(path, proof)
        return payload

    def test_limited_smoke_is_explicit_bounded_not_human_or_campaign_approval(self):
        payload = self.limited_fixture()
        result = self.measurement_validate(payload)
        self.assertEqual(result["status"], "VALIDATED_FOR_MEASUREMENT", result["blockers"])
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["approval_created"])
        self.assertEqual(result["input_binding"]["limited_smoke_authorization_sha256"],
                         a._hash(payload["expected_measurement_scope"]["authorization"]))
        self.assertTrue(any("human_verified=false" in v for v in result["limitations"]))
        # The same evidence cannot enter the ordinary frozen-weight/full-data route.
        self.assert_blocked(self.measurement_validate(payload | {"stage": "measurement"}), "MEASUREMENT_SCOPE_REQUIRED")

    def test_limited_smoke_keeps_full_index_and_all_record_review_requirements(self):
        payload = self.limited_fixture()
        self.index["full_coverage"] = False
        self.assert_blocked(self.measurement_validate(payload), "FULL_INDEX_BINDING")
        self.index["full_coverage"] = True
        entry = payload["data_report"]["gates"]["D3"]["artifact"]
        path = Path(entry["path"])
        proof = json.loads(path.read_bytes())
        proof["measurements"]["samples"].pop()
        entry["file_sha256"] = write(path, proof)
        self.assert_blocked(self.measurement_validate(payload), "D3_PREREGISTERED_RANDOM_SAMPLE")

    def test_limited_smoke_rejects_pool_expansion_and_mislabeled_review(self):
        payload = self.limited_fixture()
        invalid = copy.deepcopy(payload)
        invalid["expected_measurement_scope"]["authorization"]["max_pool_items"] = 51
        self.assert_blocked(self.measurement_validate(invalid), "SMOKE_POOL_LIMIT")
        invalid = copy.deepcopy(payload)
        invalid["expected_measurement_scope"]["authorization"]["max_optimizer_updates"] = 3
        self.assert_blocked(self.measurement_validate(invalid), "SMOKE_AUTHORIZATION_REQUIRED")
        entry = payload["data_report"]["gates"]["D7"]["artifact"]
        proof = json.loads(Path(entry["path"]).read_bytes())
        proof["provenance"]["human_verified"] = True
        entry["file_sha256"] = write(Path(entry["path"]), proof)
        self.assert_blocked(self.measurement_validate(payload), "AI_PROGRAM_REVIEW_PROVENANCE")

    def test_gate_dispatch_does_not_change_default_measurement_route(self):
        with patch.object(a, "validate_for_measurement", return_value={"route": "strict"}) as strict, \
                patch.object(a, "validate_for_limited_smoke", return_value={"route": "limited"}) as limited:
            self.assertEqual(a.validate_for_gate_data(self.root, expected_measurement_scope={}), {"route": "strict"})
            self.assertEqual(a.validate_for_gate_data(self.root, expected_measurement_scope={"purpose": a.SMOKE_PURPOSE}),
                             {"route": "limited"})
            self.assertEqual((strict.call_count, limited.call_count), (1, 1))

    def test_limited_smoke_skips_quarantined_ids_before_returning_actors(self):
        removed = sorted(pid for pid, item in self.items.items() if item["split"] == "train")[0]
        self.receipt["candidates"] = [{"problem_id": removed, "stage": "text_ngrams",
                                        "reason": "AUTHORED_UNRESOLVED_OVERLAP", "action": "QUARANTINE"}]
        quarantine = c.family_quarantine(self.items, self.receipt["candidates"])
        self.receipt.update(quarantined_ids=quarantine["quarantined_ids"], allowlist=quarantine["allowlist"],
                            quarantined_families=quarantine["families"])
        result = self.measurement_validate(self.limited_fixture())
        self.assertEqual(result["status"], "VALIDATED_FOR_MEASUREMENT", result["blockers"])
        self.assertNotIn(removed, result["selected_ids"])
        self.assertNotIn(removed, {item["problem_id"] for item in result["actors"]})
        self.assertEqual(len(result["selected_ids"]), 12)

    def test_limited_smoke_does_not_start_when_all_train_ids_are_blacklisted(self):
        self.receipt["candidates"] = [{"problem_id": pid, "stage": "image_exact",
                                       "reason": "AUTHORED_DUPLICATE_IMAGE", "action": "QUARANTINE"}
                                      for pid, item in self.items.items() if item["split"] == "train"]
        quarantine = c.family_quarantine(self.items, self.receipt["candidates"])
        self.receipt.update(quarantined_ids=quarantine["quarantined_ids"], allowlist=quarantine["allowlist"],
                            quarantined_families=quarantine["families"])
        self.assert_blocked(self.measurement_validate(self.limited_fixture()), "NO_ELIGIBLE_TRAIN_ITEMS")


if __name__ == "__main__":
    unittest.main()
