"""CPU fixtures exercise contracts and cleanup, never actual NF4 feasibility."""
import copy
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from fine_tuning.notebook_rl import checkpoint, fresh_observation, local_calibrate as calibrate
from fine_tuning.notebook_rl.engines import Completion, Prepared, RolloutRequest, RolloutResult
from fine_tuning.notebook_rl.local_eval import read, save
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
from fine_tuning.notebook_rl.local_train import TRAINING_BUDGET_VERSION, training_binding
from fine_tuning.notebook_rl.prompts import for_actor, retry_text
from fine_tuning.notebook_rl.rollout import _hash, continuation_ids
from fine_tuning.notebook_rl.schema import ActorItem, Group, ProblemTree, Sample, VerifierItem, to_json
from fine_tuning.notebook_rl.verifier import Verifier


class Tokenizer:
    unk_token_id = -1

    def encode(self, text, **_):
        return list(text.encode())

    def decode(self, ids, **_):
        return bytes(ids).decode()

    def convert_tokens_to_ids(self, _):
        return 1000


class Engine:
    tokenizer = Tokenizer()
    policy_version = 0

    def prepare(self, _):
        return Prepared([900, 901], {}, [], "original-pixels")

    def sample(self, requests):
        return [RolloutResult(r.request_id, [900, 901], [Completion([65, 66], [-.1, -.2], "stop", "AB")])
                for r in requests]

    def model_inputs(self, request, completion, device):
        ids = torch.tensor([[900, 901, *completion]], device=device)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def toy_bundle():
    model = torch.nn.Module()
    for group in ("L", "M", "V"):
        model.register_parameter(group, torch.nn.Parameter(torch.ones(1)))
    model.register_parameter("base", torch.nn.Parameter(torch.ones(1), requires_grad=False))
    return SimpleNamespace(model=model, params={g: [getattr(model, g)] for g in ("L", "M", "V")},
                           route=SimpleNamespace(apply=Mock()), base_sha256={"base": "unchanged"})


def toy_logprobs(bundle, inputs, count, **_):
    total = sum(bundle.model.parameters())
    return total * torch.arange(1, count + 1, dtype=torch.float32).reshape(1, -1)


class LongHistoryCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = build_protocol(verified_note_fact_credit=True, context_limit=24576)
        self.prompts = local_prompts(self.protocol)
        self.actors = [ActorItem(f"fixture{i}", "gen-graph-v1", "graph", "obs", "Read graph?", (),
            "int", (f"{i}.png",), ("a" * 64,)) for i in range(10)]
        self.verifier = object.__new__(Verifier)
        self.verifier.items = {a.problem_id: VerifierItem(a.problem_id, "int", "2", {}, (),
            {"edge[a,b]": "1"}, {"edge[a,b]": "number"}, ("edge[a,b]",), (), ()) for a in self.actors}
        self.engine = Engine()
        self.trees = []
        for index, actor in enumerate(self.actors):
            groups, parent = [], None
            for turn in range(1, 4 if index == 7 else 2):
                kind, gid = f"T{turn}", f"{actor.problem_id}:g{turn}"
                prompt = ([900, 901] if parent is None else continuation_ids(parent,
                    retry_text(parent.result.status, parent.result.extracted, actor.mode, actor.family,
                               for_actor(actor, self.prompts)), self.engine.tokenizer))
                samples = []
                for number in range(4 if turn == 1 else 2):
                    text = f"Observed facts:\r\nedge[a,b] = 1\r\nSolution:\r\n🙂 {turn}/{number}\r\nFinal answer: {2 if number % 2 else 3}"
                    ids = self.engine.tokenizer.encode(text)
                    samples.append(Sample(f"{gid}:s{number}", actor.problem_id, gid, kind, turn,
                        parent.sample_id if parent else None, list(prompt), ids, [-.1] * len(ids),
                        "stop", text, number, 0, self.verifier.grade(actor.problem_id, text, "stop", "obs")))
                groups.append(Group(gid, kind, actor.problem_id, _hash(prompt), samples, True, None))
                parent = samples[0]
            self.trees.append(ProblemTree(actor.problem_id, 0, groups, "VISION_OFF", {}, {}))
        actor, tree = self.actors[7], self.trees[7]
        schema = for_actor(actor, self.prompts)["observation_instruction"][actor.family]
        tree.notebook = {"private_audit": "PRIVATE_NOT_FOR_MODEL", "rewarded_notebook": {
            "fresh_observation": fresh_observation.generate(actor, schema, [object()], self.engine,
                master_seed=3407, context_limit=24576)}}

    def parent(self, root):
        parent, model = root / "parent", root / "model"
        parent.mkdir()
        actor_file = root / "actors.json"
        save(actor_file, [asdict(a) for a in self.actors])
        historical = build_protocol(verified_note_fact_credit=True)
        preparation = root / "preparation.json"
        prepared = {"format": "notebook-rl-local-data-v1", "status": "LOCAL_DATA_REVALIDATED",
            "profile": historical["profile"], "cloud_allowed": False,
            "local_protocol_sha256": historical["sha256"], "execution_actors": {
                "path": str(actor_file), "file_sha256": checkpoint.file_hash(actor_file)}}
        save(preparation, prepared)
        contract = {"stage": "FIXED_TEN_FROM_FRESH_F0", "arm": "notebook",
            "training_budget": {"version": TRAINING_BUDGET_VERSION, "max_seconds": 72000},
            "notebook_reward_enabled": True, "max_optimizer_updates": 5,
            "pair_ids": [[a.problem_id for a in self.actors[i:i+2]] for i in range(0, 10, 2)]}
        historical_code = {"local_train.py": checkpoint.digest("historical fixed-ten producer"),
                           "local_run.py": checkpoint.digest("historical preparation reader")}
        with patch("fine_tuning.notebook_rl.approval.code_hashes", return_value=historical_code):
            binding = training_binding(prepared, historical, contract)
        launch = {"pid": 987654321, "protocol": historical, "problem_ids": [a.problem_id for a in self.actors],
            "model_path": str(model), "standalone_base": True, "preparation": str(preparation),
            "preparation_sha256": checkpoint.digest(prepared), "execution_contract": contract,
            "execution_contract_sha256": checkpoint.digest(contract), "binding": binding}
        result = {"status": "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD", "attempted_ids": launch["problem_ids"]}
        save(parent / "launch.json", launch)
        save(parent / "result.json", result)
        records = []
        for tree in self.trees:
            records.extend({"kind": "GROUP", "raw": json.loads(to_json(group))} for group in tree.groups)
            observation = tree.notebook.get("rewarded_notebook", {}).get("fresh_observation")
            if observation:
                records.append({"kind": "OBSERVATION", "raw": observation | {"phase": "COMPLETED"}})
            records.append({"kind": "TREE", "raw": json.loads(to_json(tree))})
        for index, record in enumerate(records):
            save(parent / f"raw-{index:04d}.json", record)
        proof = root / "audit.json"
        save(proof, {"status": "PASS_TERMINAL_TEN_CPU_SAVED_RESPONSE_REPLAY", "capture": str(parent),
            "capture_pid": launch["pid"], "capture_pid_absent": True, "capture_result": result,
            "fixed_ten_training_complete": True, "exact_replayed_trees": 10, "unclosed_groups": 0,
            "protocol_sha256": historical["sha256"], "read_metadata_sha256": {
                str(p): checkpoint.file_hash(p) for p in [*parent.glob("*.json"), preparation, actor_file]}})
        return parent, model, proof

    def test_historical_protocol_parent_is_explicitly_bound_without_changing_minimal_gate(self):
        with tempfile.TemporaryDirectory() as temporary, patch("psutil.pid_exists", return_value=False):
            parent, model, proof = self.parent(Path(temporary))
            launch = read(parent / "launch.json")
            prepared = read(Path(launch["preparation"]))
            current = training_binding(prepared, launch["protocol"], launch["execution_contract"])
            self.assertNotEqual(launch["binding"]["code_sha256"], current["code_sha256"])
            self.assertNotEqual(launch["binding"]["data_sha256"], checkpoint.digest(prepared))
            tree, evidence = calibrate.history_parent(parent, proof, checkpoint.file_hash(proof),
                self.actors, model, self.actors[7].problem_id)
            self.assertEqual(to_json(tree), to_json(self.trees[7]))
            self.assertNotEqual(evidence["protocol_sha256"], self.protocol["sha256"])
            self.assertIn(str(proof), evidence["files"])

    def test_reaudited_preparation_contract_and_binding_mutations_rejected(self):
        mutations = ("preparation_sha256", "execution_contract_sha256", "preparation", "contract",
                     "protocol_sha256", "prompts_sha256", "data_sha256", "extra_binding")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary, \
                    patch("psutil.pid_exists", return_value=False):
                parent, model, proof = self.parent(Path(temporary))
                launch = read(parent / "launch.json")
                preparation = Path(launch["preparation"])
                if mutation in ("preparation_sha256", "execution_contract_sha256"):
                    launch[mutation] = "0" * 64
                elif mutation == "preparation":
                    prepared = read(preparation)
                    prepared["cloud_allowed"] = True
                    preparation.write_text(json.dumps(prepared))
                    launch["preparation_sha256"] = checkpoint.digest(prepared)
                elif mutation == "contract":
                    launch["execution_contract"]["training_budget"]["max_seconds"] = 60000
                    launch["execution_contract_sha256"] = checkpoint.digest(launch["execution_contract"])
                else:
                    launch["binding"][mutation] = "0" * 64
                (parent / "launch.json").write_text(json.dumps(launch))
                audit = read(proof)
                for path in (parent / "launch.json", preparation):
                    audit["read_metadata_sha256"][str(path)] = checkpoint.file_hash(path)
                proof.write_text(json.dumps(audit))
                with self.assertRaisesRegex(ValueError, "parent actor/input binding differs"):
                    calibrate.history_parent(parent, proof, checkpoint.file_hash(proof), self.actors,
                                             model, self.actors[7].problem_id)

    def test_reaudited_invalid_training_budget_rejected(self):
        invalid = [{"version": TRAINING_BUDGET_VERSION, "max_seconds": seconds}
                   for seconds in (0, -1, 86401, True, 72000., "72000", None)]
        invalid += [None, {}, {"version": "unknown", "max_seconds": 72000},
                    {"version": TRAINING_BUDGET_VERSION, "max_seconds": 72000, "extend": True}]
        for budget in invalid:
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as temporary, \
                    patch("psutil.pid_exists", return_value=False):
                parent, model, proof = self.parent(Path(temporary))
                launch = read(parent / "launch.json")
                contract = launch["execution_contract"]
                contract["training_budget"] = budget
                launch["execution_contract_sha256"] = checkpoint.digest(contract)
                launch["binding"]["data_sha256"] = checkpoint.digest({
                    "preparation_sha256": launch["preparation_sha256"],
                    "execution_contract_sha256": launch["execution_contract_sha256"]})
                (parent / "launch.json").write_text(json.dumps(launch))
                audit = read(proof)
                audit["read_metadata_sha256"][str(parent / "launch.json")] = checkpoint.file_hash(parent / "launch.json")
                proof.write_text(json.dumps(audit))
                with self.assertRaisesRegex(ValueError, "versioned training budget"):
                    calibrate.history_parent(parent, proof, checkpoint.file_hash(proof), self.actors,
                                             model, self.actors[7].problem_id)

    def test_live_prefix_changed_raw_unbound_file_and_wrong_actor_rejected(self):
        for mutation in ("live", "prefix", "hash", "raw", "unbound", "actor", "missing", "wrong_id"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                parent, model, proof = self.parent(Path(temporary))
                if mutation in ("prefix", "unbound"):
                    value = read(proof)
                    if mutation == "prefix":
                        value["status"] = "PASS_EXPLICIT_CLOSED_PREFIX_CPU_REPLAY"
                    else:
                        value["read_metadata_sha256"].pop(str(parent / "raw-0000.json"))
                    proof.write_text(json.dumps(value))
                if mutation == "raw":
                    path = parent / "raw-0000.json"
                    value = read(path)
                    value["raw"]["samples"][0]["text"] += "changed"
                    path.write_text(json.dumps(value))
                if mutation == "missing":
                    (parent / "raw-0001.json").unlink()
                actors = list(reversed(self.actors)) if mutation == "actor" else self.actors
                with patch("psutil.pid_exists", return_value=mutation == "live"), self.assertRaises(ValueError):
                    calibrate.history_parent(parent, proof, "0" * 64 if mutation == "hash" else checkpoint.file_hash(proof),
                        actors, model, "missing" if mutation == "wrong_id" else self.actors[7].problem_id)

    def test_minimal_run_status_cannot_replace_actual_fixed_ten_terminal_status(self):
        with tempfile.TemporaryDirectory() as temporary, patch("psutil.pid_exists", return_value=False):
            parent, model, proof = self.parent(Path(temporary))
            result = read(parent / "result.json")
            result["status"] = "UPDATED_SAVED_PENDING_FRESH_RELOAD"
            (parent / "result.json").write_text(json.dumps(result))
            audit = read(proof)
            audit["capture_result"] = result
            audit["read_metadata_sha256"][str(parent / "result.json")] = checkpoint.file_hash(parent / "result.json")
            proof.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "closed audited fixed-ten parent"):
                calibrate.history_parent(parent, proof, checkpoint.file_hash(proof), self.actors,
                                         model, self.actors[7].problem_id)

    def test_all_eight_raw_solutions_success_and_unverified_reread_in_prompt(self):
        actor, tree = self.actors[7], self.trees[7]
        request, audit = calibrate.history_request(actor, tree, [object()], self.prompts, self.engine, self.verifier)
        text = request.messages[0]["content"][-1]["text"]
        self.assertEqual((request.n, request.max_new_tokens, audit["attempt_count"], audit["success_attempt"]),
                         (1, 2048, 8, 8))
        self.assertIn("UNVERIFIED", text)
        self.assertIn('"raw_response": "AB"', text)
        self.assertNotIn("PRIVATE_NOT_FOR_MODEL", text)
        self.assertNotIn("episode_audit", text)
        for group in tree.groups:
            for sample in group.samples:
                for line in sample.text.splitlines(keepends=True):
                    self.assertIn(json.dumps(line, ensure_ascii=False), text)

    def test_original_completion_branch_and_image_binding_rejected(self):
        for mutation in ("completion", "branch", "image"):
            with self.subTest(mutation=mutation):
                tree = copy.deepcopy(self.trees[7])
                if mutation == "completion":
                    tree.groups[0].samples[0].completion_token_ids[0] = 65
                elif mutation == "branch":
                    tree.groups[1].samples[0].prompt_token_ids[-1] += 1
                else:
                    tree.notebook["rewarded_notebook"]["fresh_observation"]["image_key"] = "changed"
                with self.assertRaises(ValueError):
                    calibrate.history_request(self.actors[7], tree, [object()], self.prompts, self.engine, self.verifier)

    def test_generated_action_backward_changes_no_values_and_clears_gradients(self):
        bundle = toy_bundle()
        before = {n: p.detach().clone() for n, p in bundle.model.named_parameters()}
        with patch("fine_tuning.notebook_rl.policy.completion_logprobs", side_effect=toy_logprobs) as calculate:
            result = calibrate.history_backward(bundle, {}, 2)
        self.assertEqual(calculate.call_args.kwargs, {"chunk": 64})
        self.assertTrue(all(row["all_present"] and row["norm"] > 0 for row in result["gradients"].values()))
        self.assertTrue(all(p.grad is None and torch.equal(before[n], p) for n, p in bundle.model.named_parameters()))
        bundle.route.apply.assert_called_once_with(True)

    def test_backward_nonfinite_zero_frozen_or_exception_clears_temporary_gradients(self):
        for failure in ("nonfinite", "zero", "frozen", "oom"):
            with self.subTest(failure=failure):
                bundle = toy_bundle()
                if failure == "frozen":
                    bundle.model.base.requires_grad_(True)
                def values(*args, **kwargs):
                    if failure == "oom":
                        bundle.model.L.grad = torch.ones_like(bundle.model.L)
                        raise torch.OutOfMemoryError("fixture only")
                    actual = toy_logprobs(*args, **kwargs)
                    return actual * (float("nan") if failure == "nonfinite" else 0 if failure == "zero" else 1)
                with patch("fine_tuning.notebook_rl.policy.completion_logprobs", side_effect=values), self.assertRaises(
                        (ValueError, torch.OutOfMemoryError)):
                    calibrate.history_backward(bundle, {}, 2)
                self.assertTrue(all(p.grad is None for p in bundle.model.parameters()))

    def test_existing_gradients_are_rejected_without_erasing_them(self):
        bundle = toy_bundle()
        bundle.model.L.grad = torch.ones_like(bundle.model.L)
        with self.assertRaises(ValueError):
            calibrate.history_backward(bundle, {}, 2)
        self.assertIsNotNone(bundle.model.L.grad)

    def test_raw_generation_survives_failed_backward_and_result_records_failure(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            preparation, output = root / "preparation.json", root / "output"
            data = root / "data"
            data.mkdir()
            save(preparation, {"build_root": str(data)})
            save(data / "manifest.yaml", {"split_manifests": {"train": "vault"}})
            bundle = toy_bundle()
            request = RolloutRequest("fixture", None, [], [], 1, 2048, [3407])
            def mock(target, **kwargs):
                return stack.enter_context(patch(target, **kwargs))
            stack.enter_context(patch.dict(os.environ, MMDL_CALIBRATION_SUPERVISOR=str(os.getppid())))
            mock("fine_tuning.notebook_rl.local_run.read_preparation", return_value=({"build_root": str(data)}, {}, self.actors))
            mock("fine_tuning.notebook_rl.local_run.cache_readiness", return_value={"original_model_root": str(root / "original")})
            mock("fine_tuning.notebook_rl.local_run._install_actor_guards", return_value=[])
            mock("fine_tuning.notebook_rl.local_run.parameter_hashes", return_value={})
            mock("fine_tuning.notebook_rl.local_run.make_probes", return_value=[])
            mock("fine_tuning.notebook_rl.local_calibrate.history_parent", return_value=(self.trees[7],
                {"files": {}, "protocol_sha256": "historical"}))
            mock("fine_tuning.notebook_rl.local_calibrate.history_request", return_value=(request, {}))
            mock("fine_tuning.notebook_rl.local_calibrate.history_backward", side_effect=torch.OutOfMemoryError("fixture only"))
            mock("fine_tuning.notebook_rl.policy.enable_deterministic_execution", return_value={})
            mock("fine_tuning.notebook_rl.quantized_cache.block_original_access", return_value={"denied_original_access_attempts": 0})
            mock("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy", return_value=bundle)
            mock("fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine", return_value=self.engine)
            mock("fine_tuning.notebook_rl.dataset.load_images", return_value=[Mock()])
            mock("fine_tuning.notebook_rl.trainer.synchronize", return_value=[])
            service = mock("fine_tuning.notebook_rl.verifier_service.VerifierService")
            service.return_value.__enter__.return_value.metadata = {"vault_manifest_sha256": "vault"}
            for name in ("synchronize", "reset_peak_memory_stats"):
                mock("torch.cuda." + name)
            mock("torch.cuda.is_initialized", return_value=False)
            with self.assertRaises(torch.OutOfMemoryError):
                calibrate.long_history_probe(preparation, root / "model", root / "cache", root / "parent",
                    root / "audit.json", "0" * 64, self.actors[7].problem_id, output)
            self.assertEqual(read(output / "generation.json")[0]["completions"][0]["text"], "AB")
            result = read(output / "result.json")
            self.assertEqual((result["status"], result["optimizer_updates"]), ("FAILED", 0))
            for key in ("full_24k_sequence_verified", "optimizer_fit_verified", "notebook_quality_verified",
                        "self_improvement_verified"):
                self.assertIs(result[key], False)
            self.assertTrue(all(p.grad is None for p in bundle.model.parameters()))

    def test_supervisor_owns_worker_fixed_timeout_and_cleanup(self):
        command = ["long-history", "--preparation", "/prep", "--model-path", "/model", "--cache-check", "/cache",
            "--parent", "/parent", "--parent-audit", "/audit", "--parent-audit-sha256", "a" * 64,
            "--problem-id", "fixture", "--output", "/output"]
        with patch.object(calibrate.subprocess, "Popen") as popen, patch("fine_tuning.budget.terminate_group") as stop:
            process = popen.return_value
            process.wait.side_effect = subprocess.TimeoutExpired("fixture", 3570)
            self.assertEqual(calibrate.main(command), 124)
            self.assertIn("--worker", popen.call_args.args[0])
            self.assertIn("-B", popen.call_args.args[0])
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            process.wait.assert_called_once_with(timeout=3570)
            stop.assert_called_once_with(process, grace_seconds=10)
        with patch.object(calibrate, "long_history_probe") as capture:
            self.assertEqual(calibrate.main(command + ["--worker"]), 0)
            self.assertEqual(capture.call_args.kwargs["problem_id"], "fixture")
        with patch.object(calibrate.subprocess, "Popen") as popen, self.assertRaises(SystemExit):
            calibrate.main(["long-history", "--output", "/output"])
        popen.assert_not_called()

    def test_legacy_kernel_and_policy_cli_arguments_still_dispatch(self):
        for stage in ("kernel", "policy"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                output = Path(temporary) / "probe"
                kernel = stack.enter_context(patch.object(calibrate, "kernel_probe", return_value={"status": "KERNEL"}))
                policy = stack.enter_context(patch.object(calibrate, "policy_probe", return_value={"status": "POLICY"}))
                stack.enter_context(patch.object(calibrate, "version", return_value="fixture"))
                for name, value in (("get_device_name", "CPU fixture"), ("get_device_capability", (0, 0)),
                                    ("max_memory_allocated", 0), ("max_memory_reserved", 0)):
                    stack.enter_context(patch("torch.cuda." + name, return_value=value))
                stack.enter_context(patch("torch.cuda.reset_peak_memory_stats"))
                self.assertEqual(calibrate.main([stage, "--output", str(output), "--model-path", "/model"]), 0)
                self.assertEqual((kernel.call_count, policy.call_count), (int(stage == "kernel"), int(stage == "policy")))


if __name__ == "__main__":
    unittest.main()
