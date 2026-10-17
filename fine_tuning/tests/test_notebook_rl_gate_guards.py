"""Fresh actors, authored exclusion declarations and actual tiny-Qwen RL math.

The private data worker simulates full-index/human declarations solely for CPU
fixtures. Neither that seam nor these tests approves actual training data/GPU.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from fine_tuning.notebook_rl import approval, contamination, gate_guards as gate, guards, updates
from fine_tuning.notebook_rl.checkpoint import make_optimizer
from fine_tuning.notebook_rl.dataset import load_images
from fine_tuning.notebook_rl.gate_update import _measure_update
from fine_tuning.tests import test_notebook_rl_approval as approval_fixture
from fine_tuning.tests.test_notebook_rl_policy import bundle, config
from fine_tuning.tests.test_notebook_rl_updates import Preparation, tree

_WORKER_MODULE = "fine_tuning.tests.test_notebook_rl_gate_guards"


def authored_validator(root, **request):
    process = subprocess.run([sys.executable, "-m", _WORKER_MODULE, "--data-worker", str(root)],
        input=json.dumps(request), capture_output=True, text=True, timeout=60)
    if process.returncode:
        raise RuntimeError("authored data worker failed: " + process.stderr[-2000:])
    return json.loads(process.stdout)


def data_worker(root):
    request = json.loads(sys.stdin.read())
    index_path = Path(request["index"]["path"])
    index = contamination._prepare(json.loads(index_path.read_text()), "authored-test-only")
    index.update(full_coverage=True, limitations=[], partial_name=None, path=str(index_path),
                 file_sha256=approval._sha(index_path.read_bytes()))
    # Same explicit authored seam as ApprovalTests, not an actual full index.
    with patch.object(contamination, "load_index", return_value=index), \
            patch.object(approval, "code_hashes", return_value=request["expected_measurement_scope"]["code_sha256"]):
        guards.install_guards(forbidden_roots=[], vault_files=[], allow_vault=True, allow_benchmark=True)
        result = approval._validate({"stage": "measurement", "build_root": str(root), **request})
    print(json.dumps(result))


def schema_fixture():
    class Sync:
        def __init__(self):
            raise AssertionError("provider constructor reached")
        def request(self):
            raise AssertionError("network request reached")
    class Async(Sync):
        pass
    names = ("OpenAI", "AsyncOpenAI", "AzureOpenAI", "AsyncAzureOpenAI", "BedrockOpenAI", "AsyncBedrockOpenAI")
    sdk = SimpleNamespace(__version__="3.16.2", _client=None,
                          **{n: type(n, (Sync,), {}) for n in names})
    base = SimpleNamespace(SyncAPIClient=Sync, AsyncAPIClient=Async)
    def dependency(name):
        value = sdk if name == "openai" else base
        sys.modules[name] = value
        return value
    with patch.object(guards, "version", side_effect=lambda n: "0.11.0" if n == "vllm" else "3.16.2"), \
            patch.object(guards, "import_module", side_effect=dependency):
        guards.prepare_vllm_schema_dependency()


def actor_worker(root, request_path, case):
    torch.set_num_threads(2)
    request = json.loads(request_path.read_text())
    eligibility = authored_validator(root, **request)
    b, protocol, preparation = bundle(), config(), Preparation()
    optimizer, scheduler = make_optimizer(b, protocol)
    pid = eligibility.get("selected_ids", ["not-approved"])[0] if eligibility.get("selected_ids") else "not-approved"
    row = tree(b, pid=pid, route="VISION_OFF")
    images = {pid: []}
    if eligibility.get("actors"):
        actor = gate.from_json(gate.ActorItem, json.dumps(eligibility["actors"][0]))
        images[pid] = load_images(actor, root / "images")
    if case == "bad-image":
        images[pid][0] = Image.new("RGB", images[pid][0].size)
    if case == "unapproved-id":
        row = tree(b, pid="not-allowlisted", route="VISION_OFF")
        images = {row.problem_id: images[pid]}
    if case in ("schema", "caught-api"):
        schema_fixture()
    journal, report, error = [], {}, None
    kwargs = dict(build_root=root, data_request=request, forbidden_roots=[root.parent / "benchmark"],
                  algorithm="ALG-T1", step=0, seed=3407, record=journal.append)
    if case != "public-cpu":
        kwargs["_test_data_validator"] = authored_validator
    try:
        manager = gate.measure_guards if case == "public-cpu" else gate._measure_guards
        with manager(b, optimizer, scheduler, preparation, protocol, **kwargs) as report:
            if case == "fake-pass":
                report.update(exclusion_allowlist_only=True, ce_patch_raises=True, teacher_import_denied=True)
            elif case == "teacher":
                importlib.import_module("anthropic")
            elif case in ("caught-teacher", "caught-api"):
                try:
                    if case == "caught-api":
                        importlib.import_module("openai").OpenAI()
                    else:
                        importlib.import_module("anthropic")
                except PermissionError:
                    pass
                _measure_update(b, optimizer, scheduler, [row], images, preparation, protocol,
                                "ALG-T1", step=0, seed=3407)
            elif case == "legacy-sft":
                try:
                    gate.train.optimizer_update()
                except PermissionError:
                    pass  # Even a swallowed forbidden call must fail the measurement.
                _measure_update(b, optimizer, scheduler, [row], images, preparation, protocol,
                                "ALG-T1", step=0, seed=3407)
            else:
                def forbidden_loss(*args, **kwargs):
                    return torch.nn.functional.cross_entropy(torch.zeros((2, 2)), torch.tensor([0, 1]))
                with patch.object(updates, "group_loss", side_effect=forbidden_loss) if case == "ce-loss" else gate.ExitStack():
                    _measure_update(b, optimizer, scheduler, [row], images, preparation, protocol,
                                    "ALG-T1", step=0, seed=3407)
    except (ValueError, PermissionError) as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    print(json.dumps({"report": report, "error": error, "journal": journal,
        "step": scheduler.last_epoch, "has_optimizer_state": bool(optimizer.state),
        "gradients_clear": all(p.grad is None for p in b.model.parameters()),
        "actor_pid": os.getpid()}))


class GuardGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This harness owns authored gold; guarded actor children do not read it.
        cls.fixture = approval_fixture.ApprovalTests()
        cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.doCleanups)
        cls.root = cls.fixture.root
        (cls.root.parent / "benchmark").mkdir()
        payload = cls.fixture.measurement_fixture()
        cls.request = {k: payload[k] for k in gate._DATA_KEYS}
        for key in ("source_registry", "evidence", "data_report"):
            path = cls.root.parent / (key + ".json")
            cls.request[key] = {"path": str(path), "file_sha256": approval_fixture.write(path, cls.request[key])}
        cls.request_path = cls.root.parent / "request.json"
        cls.request_path.write_text(json.dumps(cls.request))

    def run_actor(self, case, request_path=None):
        # Three bounded data validations plus cold imports can exceed 90s on Pod storage.
        process = subprocess.run([sys.executable, "-m", _WORKER_MODULE, "--actor-worker", str(self.root),
            str(request_path or self.request_path), case], capture_output=True, text=True, timeout=300)
        self.assertEqual(process.returncode, 0, process.stderr[-4000:])
        self.assertNotIn('"gold"', process.stdout)
        self.assertNotIn('"oracle_fact_lines"', process.stdout)
        return json.loads(process.stdout)

    def test_discovery_module_identity_does_not_change_worker_commands(self):
        process = SimpleNamespace(returncode=0, stdout="{}", stderr="")
        with patch.dict(authored_validator.__globals__,
                        {"__name__": "test_notebook_rl_gate_guards", "__spec__": None}), \
                patch.object(subprocess, "run", return_value=process) as run:
            authored_validator(self.root, **self.request)
            self.run_actor("positive")
        self.assertEqual(run.call_count, 2)
        for call, flag in zip(run.call_args_list, ("--data-worker", "--actor-worker"), strict=True):
            self.assertEqual(call.args[0][:4], [sys.executable, "-m", _WORKER_MODULE, flag])
        self.assertEqual([call.kwargs["timeout"] for call in run.call_args_list], [60, 300])

    def test_actual_reward_update_with_all_guards_and_shared_g4(self):
        value = self.run_actor("positive")
        self.assertIsNone(value["error"])
        report = value["report"]
        self.assertEqual(report["status"], "MEASURED_ONE_RL_UPDATE")
        for key in ("exclusion_allowlist_only", "ce_patch_raises", "teacher_import_denied", "update_return_requires_sampler_sync"):
            self.assertTrue(report[key])
        self.assertFalse(report["sampler_sync_observed"])
        for key in ("native_execution", "training_allowed", "gpu_gates_certified", "data_eligibility_certified"):
            self.assertFalse(report[key])
        self.assertTrue(report["changed_parameters"])
        self.assertTrue(any(v == 1 for v in report["adam_step_deltas"].values()))
        self.assertEqual(value["step"], 1)
        self.assertTrue(value["gradients_clear"])
        self.assertNotEqual(value["actor_pid"], os.getpid())
        self.assertEqual(report["input_binding"]["exclusion_receipt_sha256"],
                         self.fixture.manifest["exclusion_receipt_sha256"])

    def test_schema_only_dependency_preserves_all_client_denials(self):
        value = self.run_actor("schema")
        self.assertIsNone(value["error"])
        self.assertTrue(value["report"]["schema_only_exception"])
        names = [p["path"] for p in value["report"]["denied_probes"]]
        self.assertTrue(any(n.startswith("schema-client:") and n.endswith("request") for n in names))
        self.assertNotIn("import:openai", names)

    def test_ce_loss_abort_and_swallowed_sft_attempt_cannot_pass(self):
        ce = self.run_actor("ce-loss")
        self.assertIn("CE/SFT path blocked", ce["error"]["message"])
        self.assertEqual(ce["step"], 0)
        self.assertFalse(ce["has_optimizer_state"])
        self.assertTrue(ce["gradients_clear"])
        sft = self.run_actor("legacy-sft")
        self.assertIn("forbidden CE/SFT attempted", sft["error"]["message"])
        self.assertEqual(sft["step"], 1)  # Actual update is preserved; no rollback claim.
        self.assertEqual(sft["report"]["status"], "FAILED")

    def test_teacher_fake_pass_and_public_cpu_cannot_pass(self):
        for case, text in (("teacher", "teacher import blocked"), ("fake-pass", "one observed"),
                           ("public-cpu", "CUDA")):
            with self.subTest(case=case):
                result = self.run_actor(case)
                self.assertIn(text, result["error"]["message"])
                self.assertEqual(result["step"], 0)

    def test_swallowed_teacher_and_schema_api_attempts_fail_after_actual_update(self):
        for case in ("caught-teacher", "caught-api"):
            with self.subTest(case=case):
                result = self.run_actor(case)
                self.assertIn("forbidden teacher/API use attempted", result["error"]["message"])
                self.assertEqual(result["step"], 1)

    def test_actual_ids_and_original_image_binding_before_update(self):
        for case, text in (("unapproved-id", "allowlisted"), ("bad-image", "approved actor")):
            with self.subTest(case=case):
                result = self.run_actor(case)
                self.assertIn(text, result["error"]["message"])
                self.assertEqual(result["step"], 0)

    def test_binding_closes_only_owned_originals_on_success_and_failure(self):
        actor = self.fixture.pools["train"][0].actor
        row = tree(bundle(), pid=actor.problem_id)
        for mismatch in (False, True):
            original = Image.new("RGB", (5, 5), (31, 63, 127))
            actual = Image.new("RGB", (5, 5), (0, 0, 0) if mismatch else (31, 63, 127))
            with patch.object(gate, "load_images", return_value=[original]), \
                    patch.object(original, "close", wraps=original.close) as closed:
                arguments = ([row], {actor.problem_id: [actual]}, {"actors": [actor], "input_binding": {}},
                             self.root, Preparation(), config(), "ALG-T1", False)
                if mismatch:
                    with self.assertRaisesRegex(ValueError, "approved actor"):
                        gate._bind_batch(*arguments)
                else:
                    gate._bind_batch(*arguments)
                closed.assert_called_once_with()
            self.assertTrue(actual.tobytes())
            actual.close()

    def test_actual_public_worker_rejects_partial_index(self):
        with self.assertRaisesRegex(ValueError, "data prerequisites rejected"):
            gate._eligibility(self.root, self.request, config())

    def test_human_provenance_remains_mandatory_in_real_validator(self):
        request = deepcopy(self.request)
        report = json.loads(Path(request["data_report"]["path"]).read_text())
        proof = deepcopy(self.fixture.proofs["D3"])
        original_proof = json.loads(Path(report["gates"]["D3"]["artifact"]["path"]).read_text())
        proof.update(original_proof)
        proof["provenance"]["kind"] = "AI"
        path = self.root.parent / "bad-human-proof.json"
        report["gates"]["D3"]["artifact"] = {"path": str(path), "file_sha256": approval_fixture.write(path, proof)}
        path = self.root.parent / "bad-human-report.json"
        request["data_report"] = {"path": str(path), "file_sha256": approval_fixture.write(path, report)}
        path = self.root.parent / "bad-human-request.json"
        path.write_text(json.dumps(request))
        result = self.run_actor("bad-human", path)
        self.assertIn("HUMAN_REVIEW_PROVENANCE", result["error"]["message"])
        self.assertEqual(result["step"], 0)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--data-worker":
        data_worker(Path(sys.argv[2]))
    elif len(sys.argv) == 5 and sys.argv[1] == "--actor-worker":
        actor_worker(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
    else:
        unittest.main()
