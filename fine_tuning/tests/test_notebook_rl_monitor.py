"""Authored CPU engine fixtures with a real, separate private verifier process."""
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from fine_tuning.notebook_rl.engines import Completion, Prepared, QWEN_OFFICIAL, RolloutResult
from fine_tuning.notebook_rl.monitor import (
    _Incomplete, _hash, _measure, _reference, _validate_reference, measure_dev, validate_f0_reference,
)
from fine_tuning.notebook_rl.schema import load_protocol
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier_service import VerifierService
from fine_tuning.tests.test_notebook_rl_schema import records


class Characters:
    def decode(self, ids, **kwargs):
        return "".join(chr(t - 256) for t in ids if t != 151645)


class FixtureEngine:
    supported_sampling = {"qwen_official"}

    def __init__(self, step=0, behavior="normal"):
        self.policy_version, self.behavior = step, behavior
        self.tokenizer, self.calls = Characters(), []
        self.after_sample = lambda: None

    def prepare(self, request):
        if self.behavior == "context":
            raise ValueError("CONTEXT_BLOCKED")
        text = json.dumps(request.messages, sort_keys=True, ensure_ascii=False)
        ids = [ord(c) + 256 for c in text]
        if self.behavior == "different_prompt":
            ids.append(999)
        return Prepared(ids, {}, request.images, "fixture")

    def sample_with_settings(self, requests, settings):
        assert settings == QWEN_OFFICIAL and len(requests) == 1
        request = requests[0]
        assert request.n == 1 and len(request.seeds) == 1
        self.calls.append(request)
        if self.behavior == "error":
            raise RuntimeError("fixture error")
        correct = request.request_id.rsplit(":", 2)[1] == "0" or self.behavior == "improved"
        answer = "7" if correct else "8"
        text = (f"Observed facts:\ncell[a][b] = {answer}\nSolution:\n" if request.request_id.endswith(":obs") else "")
        text += "Final answer: " + answer
        ids = [ord(c) + 256 for c in text] + [151645]
        completion = Completion(ids, [-.1] * len(ids), "stop", text)
        count = {"empty": 0, "extra": 2}.get(self.behavior, 1)
        result = RolloutResult(request.request_id, self.prepare(request).token_ids, [completion] * count)
        if self.behavior == "version":
            self.policy_version += 1
        self.after_sample()
        return [result]


class MonitorTests(unittest.TestCase):
    def test_private_vault_path_is_rejected_before_reference_file_open(self):
        with patch("fine_tuning.notebook_rl.monitor._read") as read, \
                self.assertRaisesRegex(_Incomplete, "F0_REFERENCE_PATH"):
            _reference(Path("/private/verifier_private.jsonl"), 96)
        read.assert_not_called()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.images = self.base / "images"
        self.images.mkdir()
        self.split = self.base / "dev"
        self.protocol = load_protocol(Path(__file__).resolve().parents[1] / "notebook_rl/protocol_v1.yaml")
        self.binding = {"protocol_sha256": self.protocol["sha256"],
                        "prompts_sha256": self.protocol["prompts_sha256"],
                        "data_sha256": "d" * 64, "code_sha256": "c" * 64}
        self.secret = "PRIVATE-ORACLE-UNEXPOSED-83923"
        self.make_split(2)

    def make_split(self, n, *, no_slots=False):
        actors, verifiers, lineages = [], [], []
        for i in range(n):
            actor, verifier, lineage, *_ = records()
            name = f"{i}.png"
            with Image.new("RGB", (9, 7), (i % 256, 23, 71)) as image:
                image.save(self.images / name)
            digest = hashlib.sha256((self.images / name).read_bytes()).hexdigest()
            actor = replace(actor, problem_id=f"fixture:family:dev:{i}", image_paths=(name,),
                            image_sha256=(digest,), question=f"Read the displayed cell, item {i}.")
            verifier = replace(verifier, problem_id=actor.problem_id, oracle_fact_lines=(self.secret,),
                               accept={"private_annotation": self.secret}, required_slots=() if no_slots else verifier.required_slots)
            lineage = replace(lineage, problem_id=actor.problem_id, split="dev")
            actors.append(actor)
            verifiers.append(verifier)
            lineages.append(lineage)
        write_split(self.split, actors, verifiers, lineages)

    def run_measure(self, name="f0", *, engine=None, step=0, f0_report=None, clock=time.time, **kwargs):
        engine = engine or FixtureEngine(step)
        result = _measure(self.split, self.images, self.base / name, engine=engine, protocol=self.protocol,
                          step=step, deadline_timestamp=time.time() + 600, input_binding=self.binding,
                          f0_report=f0_report, seed=3407, forbidden_roots=[self.base / "benchmark"],
                          clock=clock, _expected_n=2, **kwargs)
        return result, engine

    def test_real_verifier_complete_raw_tokens_independent_grounding_and_no_gold(self):
        report, engine = self.run_measure()
        self.assertEqual(report["status"], "COMPLETED", report["errors"])
        self.assertEqual((report["dev_first_attempt_acc"], report["grounding_probe"]), (.5, .5))
        self.assertEqual(report["grounding_counts"], {"required_correct": 1, "required_total": 2,
                         "items_with_required_slots": 2, "independent_obs_requests": 2})
        self.assertEqual(report["sampling_cost_counts"]["verifier_calls"], 4)
        self.assertEqual(report["sampling_cost_counts"]["returned_completions"], 4)
        self.assertIn("not_phase_a_train", report["len_median_source"])
        self.assertFalse(report["gpu_execution_certified"])
        ledger = [json.loads(line) for line in (self.base / "f0/records.jsonl").read_text().splitlines()]
        self.assertEqual(len(ledger), 4)
        for row in ledger:
            self.assertEqual(engine.tokenizer.decode(row["completion_token_ids"]), row["raw_text"])
            self.assertEqual(len(row["completion_token_ids"]), len(row["logprobs"]))
            self.assertTrue(all(row[k] >= 0 for k in ("prepare_seconds", "sample_seconds", "grade_seconds")))
        self.assertNotEqual(ledger[0]["seed"], ledger[1]["seed"])
        self.assertEqual(ledger[0]["request"]["images"], ledger[1]["request"]["images"])
        public = "".join(p.read_text() for p in (self.base / "f0").iterdir())
        self.assertNotIn(self.secret, public)
        self.assertNotIn('"gold"', public)
        self.assertEqual(report["summary_sha256"], _hash({k: v for k, v in report.items() if k != "summary_sha256"}))

    def test_current_matches_f0_inputs_seed_sampler_but_not_weights_or_step(self):
        f0, baseline = self.run_measure()
        current, live = self.run_measure("current", step=12, engine=FixtureEngine(12, "improved"),
                                        f0_report=self.base / "f0/summary.json")
        self.assertEqual(current["status"], "COMPLETED", current["errors"])
        self.assertEqual(current["dev_first_attempt_acc"], 1)
        self.assertEqual(current["grounding_probe"], 1)
        self.assertTrue(current["matched_f0"])
        self.assertEqual(f0["measurement_sha256"], current["measurement_sha256"])
        self.assertEqual([r.seeds for r in baseline.calls], [r.seeds for r in live.calls])
        self.assertNotEqual([r.request_id for r in baseline.calls], [r.request_id for r in live.calls])

    def test_current_missing_or_different_binding_f0_does_not_sample(self):
        report, engine = self.run_measure("missing", step=12)
        self.assertEqual(report["errors"], ["F0_REFERENCE_REQUIRED"])
        self.assertFalse(engine.calls)
        self.run_measure()
        self.binding["data_sha256"] = "e" * 64
        report, engine = self.run_measure("different", step=12, f0_report=self.base / "f0/summary.json")
        self.assertEqual(report["errors"], ["F0_CONFIGURATION_MISMATCH"])
        self.assertFalse(engine.calls)

    def test_actual_prompt_change_rejected_before_current_sampler(self):
        self.run_measure()
        report, engine = self.run_measure("changed", engine=FixtureEngine(12, "different_prompt"), step=12,
                                         f0_report=self.base / "f0/summary.json")
        self.assertEqual(report["errors"], ["F0_ACTUAL_INPUT_MISMATCH"])
        self.assertFalse(engine.calls)
        self.assertIsNone(report["dev_first_attempt_acc"])

    def test_failed_sampler_or_context_and_bad_counts_never_health_measurements(self):
        for behavior in ("error", "context", "empty", "extra", "version"):
            with self.subTest(behavior=behavior):
                report, engine = self.run_measure(behavior, engine=FixtureEngine(behavior=behavior))
                self.assertEqual(report["status"], "INCOMPLETE")
                for key in ("dev_first_attempt_acc", "grounding_probe", "grounding_counts", "len_median"):
                    self.assertIsNone(report[key])
                self.assertEqual(report["completed_obs"], 0)
                self.assertFalse(report["matched_f0"])
                ledger = [json.loads(line) for line in (self.base / behavior / "records.jsonl").read_text().splitlines()]
                self.assertEqual(len(ledger), 1)
                self.assertIsNotNone(ledger[0]["error"])
                if behavior in ("empty", "extra", "version"):
                    raw = json.loads((self.base / behavior / "sampler_calls.jsonl").read_text())
                    self.assertEqual(len(raw["results"][0]["completions"]), {"empty": 0, "extra": 2, "version": 1}[behavior])

    def test_deadline_before_call_and_after_sampler_keeps_raw_but_null_metrics(self):
        report, engine = self.run_measure("expired", clock=lambda: time.time() + 1000)
        self.assertEqual(report["errors"], ["DEADLINE_REACHED"])
        self.assertFalse(engine.calls)
        now = [time.time()]
        engine = FixtureEngine()
        engine.after_sample = lambda: now.__setitem__(0, time.time() + 1000)
        report, engine = self.run_measure("crossed", engine=engine, clock=lambda: now[0])
        self.assertEqual(report["errors"], ["DEADLINE_REACHED"])
        self.assertEqual(report["sampling_cost_counts"]["returned_completions"], 1)
        self.assertEqual(report["sampling_cost_counts"]["verifier_calls"], 0)
        row = json.loads((self.base / "crossed/records.jsonl").read_text())
        self.assertIsNotNone(row["raw_text"])
        self.assertIsNone(report["grounding_probe"])

    def test_f0_tampered_raw_or_rehashed_metrics_rejected(self):
        self.run_measure()
        path = self.base / "f0/summary.json"
        summary = json.loads(path.read_text())
        summary["dev_first_attempt_acc"] = 1
        summary["summary_sha256"] = _hash({k: v for k, v in summary.items() if k != "summary_sha256"})
        path.write_text(json.dumps(summary))
        report, engine = self.run_measure("forged", step=12, f0_report=path)
        self.assertEqual(report["errors"], ["F0_METRIC_RAW_MISMATCH"])
        self.assertFalse(engine.calls)
        raw = self.base / "f0/records.jsonl"
        raw.write_bytes(raw.read_bytes() + b" ")
        report, engine = self.run_measure("tampered", step=12, f0_report=path)
        self.assertEqual(report["errors"], ["F0_RAW_FILE_CHANGED"])
        self.assertFalse(engine.calls)

    def test_no_required_slots_cannot_claim_grounded_health(self):
        self.split = self.base / "other/dev"
        self.make_split(2, no_slots=True)
        report, engine = self.run_measure()
        self.assertEqual(len(engine.calls), 4)
        self.assertEqual(report["errors"], ["NO_REQUIRED_GROUNDING_SLOTS"])
        self.assertIsNone(report["dev_first_attempt_acc"])

    def test_public_fixed96_expired_run_plans192_not_small_fixture_success(self):
        with self.assertRaisesRegex(ValueError, "96 dev"):
            measure_dev(self.split, self.images, self.base / "wrongcount", engine=FixtureEngine(),
                        protocol=self.protocol, step=0, deadline_timestamp=time.time() + 600, input_binding=self.binding)
        self.split = self.base / "full/dev"
        self.make_split(96)
        engine = FixtureEngine()
        report = measure_dev(self.split, self.images, self.base / "public", engine=engine,
                             protocol=self.protocol, step=0, deadline_timestamp=time.time() - 1, input_binding=self.binding)
        self.assertEqual((report["status"], report["n"], report["samples_per_problem"]), ("INCOMPLETE", 96, 1))
        self.assertEqual(report["sampling_cost_counts"]["planned_sampler_calls"], 192)
        self.assertFalse(engine.calls)
        completed = measure_dev(self.split, self.images, self.base / "public-cpu", engine=engine,
                                protocol=self.protocol, step=0, deadline_timestamp=time.time() + 600,
                                input_binding=self.binding)
        self.assertEqual(completed["status"], "COMPLETED", completed["errors"])
        self.assertEqual(completed["sampling_cost_counts"]["verifier_calls"], 192)
        self.assertEqual(len(engine.calls), 192)
        self.assertEqual(completed["completed_qa"], 96)
        self.assertEqual(completed["completed_obs"], 96)
        self.assertFalse(completed["gpu_execution_certified"])
        checked = validate_f0_reference(self.base / "public-cpu/summary.json", self.split, self.images,
                                        protocol=self.protocol, input_binding=self.binding)
        self.assertEqual(checked["status"], "VALIDATED_F0_REFERENCE")
        self.assertEqual(checked["measurement_sha256"], completed["measurement_sha256"])
        self.assertFalse(checked["approval_created"])
        self.assertEqual(len(engine.calls), 192)  # Read-only validation made no model/verifier requests.
        (self.images / "0.png").write_bytes(b"tampered fixture PNG")
        with self.assertRaisesRegex(ValueError, "invalid F0 reference"):
            validate_f0_reference(self.base / "public-cpu/summary.json", self.split, self.images,
                                  protocol=self.protocol, input_binding=self.binding)

    def test_f0_rehashed_sampler_correspondence_is_not_just_summary_pass(self):
        self.run_measure()
        path = self.base / "f0/summary.json"
        raw_path = self.base / "f0/sampler_calls.jsonl"
        calls = [json.loads(line) for line in raw_path.read_text().splitlines()]
        calls[0]["results"][0]["completions"][0]["text"] = "Final answer: 123"
        raw_path.write_text("".join(json.dumps(call) + "\n" for call in calls))
        summary = json.loads(path.read_text())
        summary["files"][raw_path.name] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        summary["summary_sha256"] = _hash({k: v for k, v in summary.items() if k != "summary_sha256"})
        path.write_text(json.dumps(summary))
        report, engine = self.run_measure("ledger-forged", step=12, f0_report=path)
        self.assertEqual(report["errors"], ["F0_RAW_LEDGER_MISMATCH"])
        self.assertFalse(engine.calls)

    def test_preallocation_validator_rejects_changed_checkpoint_without_sampling(self):
        self.run_measure()
        checked = _validate_reference(self.base / "f0/summary.json", self.split, self.images,
                                      protocol=self.protocol, input_binding=self.binding, seed=3407, _expected_n=2)
        self.assertEqual(checked["status"], "VALIDATED_F0_REFERENCE")
        with self.assertRaisesRegex(ValueError, "F0 reference"):
            validate_f0_reference(self.base / "f0/summary.json", self.split, self.images,
                                  protocol=self.protocol, input_binding=self.binding)
        self.binding["code_sha256"] = "f" * 64
        with self.assertRaisesRegex(Exception, "F0_CONFIGURATION_MISMATCH"):
            _validate_reference(self.base / "f0/summary.json", self.split, self.images,
                                protocol=self.protocol, input_binding=self.binding, seed=3407, _expected_n=2)

    def test_cleanup_exception_after_last_result_cannot_retain_completed(self):
        class CleanupFailure:
            def __init__(self, *args, **kwargs):
                self.service = VerifierService(*args, **kwargs)

            def __enter__(self):
                return self.service

            def __exit__(self, *args):
                self.service.close()
                raise RuntimeError("fixture cleanup failure")

        with patch("fine_tuning.notebook_rl.monitor.VerifierService", CleanupFailure):
            report, engine = self.run_measure()
        self.assertEqual(len(engine.calls), 4)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["errors"], ["RuntimeError"])
        self.assertIsNone(report["grounding_probe"])
        self.assertIsNone(report["dev_first_attempt_acc"])

    def test_guarded_actor_does_not_open_verifier_or_lineage(self):
        script = """
import json,pathlib,sys,time
from fine_tuning.notebook_rl.guards import install_guards
from fine_tuning.notebook_rl.monitor import _measure
from fine_tuning.notebook_rl.schema import load_protocol
from fine_tuning.tests.test_notebook_rl_monitor import FixtureEngine
root=pathlib.Path(sys.argv[1]); private=[root/'dev/verifier_private.jsonl',root/'dev/lineage.jsonl']
protocol=load_protocol(pathlib.Path(sys.argv[2]))
binding={'protocol_sha256':protocol['sha256'],'prompts_sha256':protocol['prompts_sha256'],
         'data_sha256':'d'*64,'code_sha256':'c'*64}
install_guards(forbidden_roots=[root/'benchmark'],vault_files=private,allow_vault=False,allow_benchmark=False)
opened=[]
def audit(event,args):
    if event=='open' and isinstance(args[0],str): opened.append(args[0])
sys.addaudithook(audit)
report=_measure(root/'dev',root/'images',root/'guarded',engine=FixtureEngine(),protocol=protocol,
    step=0,deadline_timestamp=time.time()+600,input_binding=binding,f0_report=None,seed=3407,
    forbidden_roots=[root/'benchmark'],clock=time.time,_expected_n=2)
assert report['status']=='COMPLETED',report['errors']
assert not any(str(p) in opened for p in private)
print(json.dumps({'status':report['status'],'actor_never_opened_gold':True}))
"""
        protocol_path = Path(__file__).resolve().parents[1] / "notebook_rl/protocol_v1.yaml"
        result = subprocess.run([sys.executable, "-c", script, str(self.base), str(protocol_path)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["actor_never_opened_gold"])
        self.assertNotIn(self.secret, result.stdout + result.stderr)

    def test_output_cannot_write_source_image_or_model_inputs(self):
        engine = FixtureEngine()
        engine.model_dir = self.base / "model"
        engine.model_dir.mkdir()
        for output in (self.images / "nested/report", engine.model_dir / "nested/report"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "external non-source"):
                _measure(self.split, self.images, output, engine=engine, protocol=self.protocol, step=0,
                         deadline_timestamp=time.time() + 600, input_binding=self.binding, f0_report=None,
                         seed=3407, forbidden_roots=(), clock=time.time, _expected_n=2)
            self.assertFalse(output.parent.exists())

    def test_write_once_and_frozen_schedule_sampler_protocol(self):
        self.run_measure()
        before = (self.base / "f0/summary.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_measure()
        self.assertEqual((self.base / "f0/summary.json").read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "frozen schedule"):
            self.run_measure("badstep", step=1, engine=FixtureEngine(1))
        engine = FixtureEngine()
        engine.supported_sampling = {"temperature_1"}
        with self.assertRaisesRegex(ValueError, "official-sampler"):
            self.run_measure("sampler", engine=engine)
        self.protocol = dict(self.protocol, context_limit=100000)
        with self.assertRaisesRegex(ValueError, "actual protocol"):
            self.run_measure("mutated")


if __name__ == "__main__":
    unittest.main()
