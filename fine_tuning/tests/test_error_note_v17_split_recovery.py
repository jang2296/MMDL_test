"""CPU recovery fixtures for dynamic graph execution; no model files are archived."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fine_tuning.error_note import v17_eval as ev
from fine_tuning.error_note import v17_eval_engine as engine
from fine_tuning.notebook_rl import proof_eval as pe
from fine_tuning.notebook_rl.pod import collect
from fine_tuning.tests.test_error_note_v17_eval import final_value
from fine_tuning.tests import test_error_note_v17_split_artifacts as artifact_fixtures
from fine_tuning.tests import test_error_note_v17_split_engine as engine_fixtures
from mmdl.evaluation.writer import RunWriter
from mmdl.runtime.artifacts import digest, read_json, sha256_file, write_json


class SplitRecovery(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        training = artifact_fixtures.SplitArtifacts()
        training.setUp()
        self.addCleanup(training.doCleanups)
        self.seal = training.create()
        self.seal_path = training.args.artifact_output
        self.model_ref = training.export / "model_ref.json"
        self.run_dir = self.root / "artifacts/runs/run"
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        observed = engine.observe_engine(engine_fixtures.backend_value(mode="cuda_graph", slots=9, blocks=27776),
                                         "cuda_graph", 9)
        historical, probabilities = engine_fixtures.lengths(), engine_fixtures.columns()
        quantiles = engine.quantile_lengths(historical, [row["sample_id"] for row in probabilities])
        actual, long = engine_fixtures.probes()
        actual["fixed_lengths"] = {row["sample_id"]: row["max_tokens"] for row in quantiles["items"]}
        actual["generated_tokens"] = sum(actual["fixed_lengths"].values())
        actual = ev.seal(actual)
        calibration = engine.calibration_receipt(actual, long, historical, load_seconds=0.,
                                                  long_tail_seconds=0., long_tail_basis="CPU fixture")
        pf5 = {"pass": True, "model_ref_sha256": sha256_file(self.model_ref), "files": {"pass": True},
               "engine": {"pass": True, "pf5": {"pass": True, "power": True}},
               "decode_engine": {"pass": True, "power": True, "raw_logprobs": True, "graph_replay_deltas": [31] * 10}}
        selected = ev.seal({"status": "PASS", "engine_mode": "cuda_graph", "observed_engine": observed,
            "calibration": calibration, "actual_length_probe": actual, "long_context_probe": long,
            "quantiles": quantiles, "probabilities": probabilities, "eager_reference_probabilities": probabilities,
            "eager_comparison": engine.compare_probabilities(probabilities, probabilities),
            "preflight_evidence": ev.seal({"files_pass": True, "historical_engine_pass": True,
                "historical_pf5": pf5["engine"]["pf5"], "decode_pf5": pf5["decode_engine"]})})
        attempt_path, pf5_path = self.inputs / "attempt.json", self.inputs / "pf5.json"
        write_json(attempt_path, selected)
        write_json(pf5_path, pf5)
        self.validation = ev.seal({"baseline_generated_tokens": historical, "attempts": [{"path": str(attempt_path), "sha256": sha256_file(attempt_path)}],
            "preflight_report": str(pf5_path), "preflight_file_sha256": sha256_file(pf5_path), "version": engine.VERSION, "status": "PASS", "engine_mode": "cuda_graph",
            "selected_mode": "cuda_graph", "slots": 9, "safe_slots": 9, "observed_engine": observed,
            "actual_kv_token_capacity": 444416, "calibration": calibration, "selected_attempt": selected,
            "training_artifact_sha256": sha256_file(self.seal_path),
            "training_artifact_file_sha256": sha256_file(self.seal_path),
            "training_artifact_content_sha256": self.seal["sha256"], "model_ref_sha256": sha256_file(self.model_ref),
            "source_sha256": "b" * 64, "config_sha256": "c" * 64, "hardware_sha256": "d" * 64,
            "environment_sha256": "e" * 64, "code_commit": "f" * 40, "evaluation_commit": "f" * 40,
            "estimated_evaluation_seconds": calibration["estimated_evaluation_seconds"],
            "evaluation_watchdog_seconds": calibration["evaluation_watchdog_seconds"]})
        self.validation_path = self.inputs / "validation.json"
        write_json(self.validation_path, self.validation)
        self.prereg_path, self.ledger = self.inputs / "prereg.json", self.inputs / "ledger.jsonl"
        self.contract = ev.split_execution_contract(self.validation, ["one", "two"])
        self.prereg = {"format": pe.FORMAT_V12, "calibration_sha256": sha256_file(self.validation_path),
            "engine_validation": self.validation, "engine_validation_sha256": sha256_file(self.validation_path),
            "training_artifacts_file_sha256": sha256_file(self.seal_path), "execution_contract": self.contract,
            "arms": {pe.ARM_SINGLE: {"model_ref_sha256": sha256_file(self.model_ref)}},
            "evaluation_subsets": {"all": {"ids": ["one", "two"]}}}
        write_json(self.prereg_path, self.prereg)
        self.prereg_path.with_name("prereg.json.sha256").write_text(sha256_file(self.prereg_path))
        self.prereg_sha = sha256_file(self.prereg_path)
        self.events = [{"event": "START", "run_id": "run", "preregistration_sha256": self.prereg_sha}]
        self.ledger.write_text(json.dumps(self.events[0]) + "\n")
        self.registration = {"slots": 9, "expected_ids": ["one", "two"], "serving_kind": pe.LORA_KIND,
            "execution_contract": self.contract, "observed_engine": observed, "model_sha256": "a" * 64,
            "preregistration_sha256": self.prereg_sha, "model_ref_sha256": sha256_file(self.model_ref),
            "calibration_sha256": sha256_file(self.validation_path), "preflight_sha256": sha256_file(self.validation_path),
            "engine_validation_sha256": sha256_file(self.validation_path), "evaluation_commit": "f" * 40,
            "training_artifacts_path": str(self.seal_path), "training_artifacts_file_sha256": sha256_file(self.seal_path)}
        self.identity = {"mode": "partial", "model_sha256": "a" * 64}
        first = ev.submission_order(["one", "two"])[0]
        self.first = first
        with RunWriter(self.run_dir, self.identity, ["one", "two"]) as writer:
            final = final_value(first, self.registration, self.identity, "run")
            ev.exclusive(self.run_dir / "final_raw" / (first + ".json"), final)
            writer.save(ev.record_from_final(final))
        for name in ("environment.json", "code_files.json", "locations.json"):
            write_json(self.run_dir / name, {})
        write_json(self.run_dir / "backend.json", {"v17_actual_engine": observed})
        for name in ("resolved_eval_config.yaml", "resolved_hardware_config.yaml"):
            (self.run_dir / name).write_text("fixture: true\n")
        write_json(self.run_dir / pe.REGISTRATION_FILE, self.registration)
        write_json(self.run_dir / "inference_invocations/current.json", {"id": "original-invocation", "attempted_ids": [first]})
        self.append(event="BEGIN", invocation_id="original-invocation", completed_ids=[],
                    pending_ids=ev.submission_order(["one", "two"]))
        self.append(event="INTENT", invocation_id="original-invocation", id=first)
        self.append(event="ACCEPTED", invocation_id="original-invocation", id=first)
        self.patches = [patch.object(pe, "load_preregistration", return_value=(self.prereg, self.prereg_sha)),
            patch.object(pe, "read_ledger", return_value=(self.events, "a" * 64)),
            patch.dict("os.environ", {"MMDL_ARTIFACT_ROOT": str(self.root)})]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def append(self, **fields):
        # This fixture already has durable output, so write the real chained event shape directly.
        path = self.run_dir / "submission_events.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        value = ev.seal({"version": "v17-submission-event-v1", "registration_sha256": digest(self.registration),
                         "previous": records[-1]["sha256"] if records else None, **fields})
        ev.append_event(path, value)

    def pack(self, name="resume"):
        return collect.pack_eval_resume(self.run_dir, self.prereg_path, self.ledger, self.model_ref,
            self.validation_path, self.validation_path, self.root / "out", name)

    def roundtrip(self):
        manifest = self.pack()
        receipt = collect.verify_archive(self.root / "out/resume.tar.gz", self.root / "out/resume.manifest.json",
                                         self.root / "restored", self.root / "receipt.json")
        return manifest, receipt

    def test_nine_slot_graph_roundtrip_includes_seal_and_derived_audit_without_weights(self):
        manifest, receipt = self.roundtrip()
        self.assertEqual(receipt["completed_samples"], 1)
        self.assertEqual(receipt["resume_capability"], "EVALUATION_ONLY")
        self.assertFalse(receipt["training_resume_certified"])
        names = {r["path"] for r in manifest["files"]}
        self.assertIn("eval/inputs/training-artifacts.json", names)
        self.assertIn("eval/runs/run/submission_events.jsonl", names)
        self.assertIn("eval/runs/run/submission_audit.json", names)
        self.assertIn("eval/runs/run/engine_evidence_mapping.json", names)
        self.assertEqual(len([p for p in names if p.startswith("eval/inputs/engine-evidence/")]), 2)
        self.assertFalse(any(p.endswith((".pt", ".safetensors")) for p in names))
        restored = read_json(self.root / "restored/eval/runs/run/variant_registration.json")
        self.assertEqual(restored["slots"], 9)
        self.assertEqual(restored["execution_contract"]["engine_mode"], "cuda_graph")

    def test_mode_slots_or_seal_mutation_rejected_after_extract(self):
        manifest, _ = self.roundtrip()
        base = self.root / "restored/eval/runs/run"
        path = base / pe.REGISTRATION_FILE
        for key, value in (("slots", 8), ("execution_contract", self.contract | {"engine_mode": "eager"})):
            changed = deepcopy(self.registration)
            changed[key] = value
            write_json(path, changed)
            with self.subTest(key=key), self.assertRaises(collect.CollectError):
                collect.verify_eval_resume_tree(self.root / "restored", manifest)
            write_json(path, self.registration)
        seal = self.root / "restored/eval/inputs/training-artifacts.json"
        seal.write_bytes(seal.read_bytes() + b" ")
        with self.assertRaises(collect.CollectError):
            collect.verify_eval_resume_tree(self.root / "restored", manifest)

    def test_training_tree_tamper_rejected_before_pack(self):
        path = Path(self.seal["checkpoint"]["root"]) / "weights.pt"
        path.write_bytes(path.read_bytes() + b"x")
        with self.assertRaisesRegex(collect.CollectError, "inventory changed"):
            self.pack()

    def test_offline_verification_preserves_measured_engine_evidence(self):
        manifest = self.pack()
        for path in (self.inputs / "attempt.json", self.inputs / "pf5.json"):
            path.unlink()
        collect.verify_archive(self.root / "out/resume.tar.gz", self.root / "out/resume.manifest.json",
                               self.root / "restored", self.root / "receipt.json")
        base = self.root / "restored/eval/runs/run"
        mapping = read_json(base / "engine_evidence_mapping.json")
        self.assertEqual(len(mapping["files"]), 2)
        record = mapping["files"][0]
        path = self.root / "restored" / record["path"]
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(collect.CollectError, "measured engine evidence"):
            collect.verify_eval_resume_tree(self.root / "restored", manifest)

    def test_event_hash_order_and_missing_final_acceptance_are_rejected(self):
        path = self.run_dir / "submission_events.jsonl"
        original = path.read_text()
        for change in ("hash", "order", "missing"):
            events = [json.loads(line) for line in original.splitlines()]
            if change == "hash":
                events[-1]["previous"] = "0" * 64
            elif change == "order":
                events[0]["pending_ids"] = list(reversed(events[0]["pending_ids"]))
            else:
                events.pop()
            previous = None
            for event in events:
                if change != "hash":
                    event["previous"] = previous
                event["sha256"] = digest({k: v for k, v in event.items() if k != "sha256"})
                previous = event["sha256"]
            path.write_text("".join(json.dumps(e) + "\n" for e in events))
            with self.subTest(change=change), self.assertRaises(collect.CollectError):
                self.pack("bad-" + change)
            path.write_text(original)

    def test_pack_refreshes_only_valid_stale_derived_audit(self):
        audit = ev.submission_audit(self.run_dir, self.registration)
        write_json(self.run_dir / "submission_audit.json", audit)
        other = next(sid for sid in ["one", "two"] if sid != self.first)
        self.append(event="INTENT", invocation_id="original-invocation", id=other)
        manifest, _ = self.roundtrip()
        current = read_json(self.run_dir / "submission_audit.json")
        self.assertEqual(current["events"], 4)
        path = self.root / "restored/eval/runs/run/submission_audit.json"
        current["actual_submitted_ids"] = []
        current["sha256"] = digest({k: v for k, v in current.items() if k != "sha256"})
        write_json(path, current)
        with self.assertRaisesRegex(collect.CollectError, "derived submission audit"):
            collect.verify_eval_resume_tree(self.root / "restored", manifest)


if __name__ == "__main__":
    unittest.main()
