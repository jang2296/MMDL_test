"""CPU integration fixtures; whole-note model quality remains unmeasured here."""
import copy
import os
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fine_tuning.notebook_rl import checkpoint, fact_context_probe as probe
from fine_tuning.notebook_rl.local_profile import build_protocol
from fine_tuning.tests import test_notebook_rl_pair_probe_integration as pair_tests


class WholeProbeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.legacy = pair_tests.PairProbeIntegrationTests()
        self.legacy.setUp()
        self.f = self.legacy.f
        self.f.protocol = build_protocol(pair_draft_synthesis=True,
            verified_note_synthesis_credit=True, context_limit=24576)

    def fixture(self, root):
        parent, model, source = self.legacy.fixture(root)
        parent = parent.rename(root / "minimal")
        for name in ("raw-0001.json", "raw-0003.json"):
            path = parent / name
            tree = probe.read(path)
            tree["raw"]["notebook"].setdefault("rewarded_notebook", {})["notebook_draft_policy"] = (
                self.f.protocol["notebook_draft_policy"])
            path.write_bytes(probe.wire(tree))
        call = {"request_id": "fixture-draft", "target_index": 1, "prompt_token_ids": [1, 2],
                "completions": [{"token_ids": [3], "logprobs": [-.1],
                                 "finish_reason": "stop", "text": "unverified draft"}]}
        drafts = {"policy": self.f.protocol["notebook_draft_policy"], "calls": [call]}
        path = parent / "raw-0003.json"
        tree = probe.read(path)
        tree["raw"]["notebook"]["rewarded_notebook"]["pair_drafts"] = drafts
        path.write_bytes(probe.wire(tree))
        problem_id = self.f.actors[1].problem_id
        events = [
            {"problem_id": problem_id, "phase": "STARTED",
             "call": {k: v for k, v in call.items() if k != "completions"}},
            {"problem_id": problem_id, "phase": "GENERATED", "request_id": call["request_id"],
             "target_index": 1, "outputs": [{"request_id": call["request_id"],
                 "prompt_token_ids": call["prompt_token_ids"], "completions": call["completions"]}]},
            {"problem_id": problem_id, "phase": "COMPLETED", "call": call},
            {"problem_id": problem_id, "phase": "DRAFTS_COMPLETED", "record": drafts}]
        for index, event in enumerate(events, 6):
            probe.save(parent / f"raw-{index:04d}.json", {"kind": "DRAFT", "raw": event})
        self.seal(parent)
        return parent, model, source

    def seal(self, parent):
        launch = probe.read(parent / "launch.json")
        preparation = Path(launch["preparation"])
        prepared = probe.read(preparation)
        metadata = [*parent.iterdir(), preparation,
                    *(Path(prepared[k]["path"]) for k in
                      ("actors", "execution_actors", "selection", "validated_data", "data_request"))]
        audit = {"capture": str(parent), "capture_pid": launch["pid"], "capture_pid_absent": True,
            "capture_result": probe.read(parent / "result.json"), "protocol_sha256": self.f.protocol["sha256"],
            "source_manifest_sha256": checkpoint.file_hash(parent.parent / "sources.json"),
            "preparation_sha256": checkpoint.file_hash(preparation), "read_files_unchanged": True,
            "source_files_unchanged": True, "closed_trees": 2, "exact_replayed_trees": 2,
            "unclosed_groups": 0, "unclosed_observation_calls": 0, "status": "DIAGNOSTIC_NOT_PASS",
            "read_metadata_sha256": {str(p): checkpoint.file_hash(p) for p in metadata}}
        path = parent.parent / "minimal-audit/audit-terminal-v1.json"
        if path.exists():
            path.write_bytes(probe.wire(audit))
        else:
            probe.save(path, audit)

    def pins(self, parent):
        return patch.multiple(probe, WHOLE_SYNTHESIS_ROOT=parent.parent,
            WHOLE_SYNTHESIS_SOURCES=checkpoint.file_hash(parent.parent / "sources.json"),
            WHOLE_SYNTHESIS_AUDIT=checkpoint.file_hash(parent.parent / "minimal-audit/audit-terminal-v1.json"),
            WHOLE_SYNTHESIS_RESULT=checkpoint.file_hash(parent / "result.json"))

    def test_fixed_historical_protocol_and_v45_identity_remain_explicit(self):
        self.assertEqual(self.f.protocol["sha256"], probe.WHOLE_SYNTHESIS_PROTOCOL)
        self.assertEqual(probe.WHOLE_SYNTHESIS_ROOT.name, "verified-relation-semantics-v45-checked")
        self.assertEqual(probe.WHOLE_SYNTHESIS_SOURCES,
                         "9dd471a3b8bc48e9f31fa701fab69345157ae8524e3e4ad6d18eb068918458fc")
        self.assertEqual(probe.WHOLE_SYNTHESIS_RESULT,
                         "5034aee23ed46377af3da7f288d83ead57c80399d09504cdffb78d1318e68810")
        self.assertEqual(probe.WHOLE_SYNTHESIS_AUDIT,
                         "b36e372ed997730b9d47538a6b1790f4db439b12f5d4379bbac0659124da7857")
        with tempfile.TemporaryDirectory() as temp, patch("psutil.pid_exists", return_value=False):
            parent, model, _ = self.fixture(Path(temp))
            with self.pins(parent):
                trees, evidence = probe.whole_synthesis_parent(parent, self.f.actors, self.f.protocol, model)
                self.assertEqual(len(trees), 2)
                self.assertFalse(evidence["historical_approval_relabelled"])
                self.assertIn(str(parent / "raw-0009.json"), evidence["files"])
                self.assertIn(str(parent.parent / "validated_data.json"), evidence["files"])
                with patch.object(probe, "WHOLE_SYNTHESIS_ROOT", parent.parent / "other"), self.assertRaises(ValueError):
                    probe.whole_synthesis_parent(parent, self.f.actors, self.f.protocol, model)

    def test_live_updates_protocol_execution_and_incomplete_trees_rejected(self):
        for mutation in ("live", "update", "bool-update", "status", "protocol", "execution", "missing-tree"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                parent, model, _ = self.fixture(Path(temp))
                launch, result = probe.read(parent / "launch.json"), probe.read(parent / "result.json")
                if mutation in ("update", "bool-update"):
                    result["optimizer_updates"] = 1 if mutation == "update" else False
                elif mutation == "status":
                    result["status"] = "NO_NOTEBOOK_LEARNING_SIGNAL"
                elif mutation == "protocol":
                    launch["protocol"]["notebook_prompt_sha256"] = "changed"
                elif mutation == "execution":
                    launch["execution_contract"] = {}
                elif mutation == "missing-tree":
                    (parent / "raw-0003.json").unlink()
                (parent / "launch.json").write_bytes(probe.wire(launch))
                (parent / "result.json").write_bytes(probe.wire(result))
                self.seal(parent)
                with self.pins(parent), patch("psutil.pid_exists", return_value=mutation == "live"), self.assertRaises(ValueError):
                    probe.whole_synthesis_parent(parent, self.f.actors, self.f.protocol, model)

    def test_source_data_draft_raw_and_audit_tamper_rejected(self):
        for mutation in ("source", "mode", "data", "draft", "reread", "audit", "manifest", "extra-raw", "result"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                parent, model, source = self.fixture(Path(temp))
                with self.pins(parent), patch("psutil.pid_exists", return_value=False):
                    if mutation == "source":
                        source.write_text("changed source")
                    elif mutation == "mode":
                        source.chmod(0o400)
                    elif mutation == "data":
                        (parent.parent / "validated_data.json").write_text("changed data")
                    elif mutation in ("draft", "reread"):
                        path = parent / ("raw-0008.json" if mutation == "draft" else "raw-0005.json")
                        value = probe.read(path)
                        value["raw"]["call" if mutation == "draft" else "completion"] = {"text": "changed"}
                        path.write_bytes(probe.wire(value))
                    elif mutation == "audit":
                        (parent.parent / "minimal-audit/audit-terminal-v1.json").write_text("changed audit")
                    elif mutation == "manifest":
                        (parent.parent / "sources.json").write_text("changed manifest")
                    elif mutation == "result":
                        path = parent / "result.json"
                        path.write_bytes(path.read_bytes() + b"\n")
                    else:
                        probe.save(parent / "raw-0010.json", {"kind": "EXTRA", "raw": {}})
                    with self.assertRaises((ValueError, KeyError)):
                        probe.whole_synthesis_parent(parent, self.f.actors, self.f.protocol, model)

    def test_cli_forwarding_exclusivity_and_timeout_cleanup(self):
        base = ["--preparation", "/prep", "--model-path", "/model", "--cache-check", "/cache",
                "--parent", "/parent", "--output", "/output", "--notebook-whole-synthesis"]
        with patch.object(probe, "capture") as capture:
            self.assertEqual(probe.main([*base, "--worker"]), 0)
            self.assertEqual(capture.call_args.kwargs, {"notebook_whole_synthesis": True})
        for timeout in (False, True):
            with self.subTest(timeout=timeout), patch.object(probe.subprocess, "Popen") as popen, \
                    patch("fine_tuning.budget.terminate_group") as stop:
                popen.return_value.wait.return_value = 0
                if timeout:
                    popen.return_value.wait.side_effect = subprocess.TimeoutExpired("fixture", 3570)
                self.assertEqual(probe.main(base), 124 if timeout else 0)
                self.assertIn("--notebook-whole-synthesis", popen.call_args.args[0])
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                popen.return_value.wait.assert_called_once_with(timeout=3570)
                stop.assert_called_once_with(popen.return_value, grace_seconds=10)
        for flag in ("--notebook-layout", "--notebook-drafts", "--notebook-relationships",
                     "--notebook-relationship-comparisons", "--notebook-pair-classification", "--trained-notebook-reload"):
            extra = [flag, "/reload"] if flag == "--trained-notebook-reload" else [flag]
            with self.subTest(flag=flag), patch("sys.stderr"), self.assertRaises(SystemExit):
                probe.main([*base, *extra])
        with self.assertRaisesRegex(ValueError, "one diagnostic intervention"):
            probe.capture(*([Path("unused")] * 5), notebook_whole_synthesis=True, notebook_pair_classification=True)

    def test_zero_update_capture_lifecycle_checks_current_approval_and_preservation(self):
        for mutation in (None, "weights", "parent", "source", "deadline"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root, actors, trees = Path(temp), self.f.actors, self.f.trees
                data = root / "data"
                probe.save(data / "manifest.yaml", {"split_manifests": {"train": "vault"}})
                parent_file = root / "parent.json"
                probe.save(parent_file, {"history": "original"})
                bundle = SimpleNamespace(base_sha256="base")
                image = SimpleNamespace(close=Mock())
                verifier = SimpleNamespace(metadata={"vault_manifest_sha256": "vault"})

                def mock(target, **kwargs):
                    return stack.enter_context(patch(target, **kwargs))

                stack.enter_context(patch.dict(os.environ, MMDL_FACT_PROBE_SUPERVISOR=str(os.getppid())))
                mock("fine_tuning.notebook_rl.policy.enable_deterministic_execution", return_value={})
                cache = mock("fine_tuning.notebook_rl.local_run.cache_readiness", return_value={"original_model_root": str(root / "original")})
                prepared = {"build_root": str(data)}
                read = mock("fine_tuning.notebook_rl.local_run.read_preparation", return_value=(prepared, {}, actors))
                mock("fine_tuning.notebook_rl.local_run.minimal_actors", return_value=actors)
                historical = mock("fine_tuning.notebook_rl.fact_context_probe.whole_synthesis_parent",
                    return_value=(trees, {"files": {str(parent_file): checkpoint.file_hash(parent_file)}}))
                sources = mock("fine_tuning.notebook_rl.approval.code_hashes",
                    side_effect=[{"source": "same"}, {"source": "changed" if mutation == "source" else "same"}])
                guards = mock("fine_tuning.notebook_rl.local_run._install_actor_guards", return_value=[])
                mock("fine_tuning.notebook_rl.quantized_cache.block_original_access", return_value={"denied_original_access_attempts": 0})
                mock("fine_tuning.notebook_rl.verifier_service.VerifierService", return_value=nullcontext(verifier))
                mock("fine_tuning.notebook_rl.episode_notebook.audit_episode", return_value={"complete_episode": True})
                load = mock("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy", return_value=bundle)
                trained = mock("fine_tuning.notebook_rl.checkpoint.load_eval")
                mock("fine_tuning.notebook_rl.local_run.parameter_hashes",
                    side_effect=[{"a": "before"}, {"a": "changed" if mutation == "weights" else "before"}])
                engine = mock("fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine")
                mock("fine_tuning.notebook_rl.dataset.load_images", return_value=[image])
                mock("fine_tuning.notebook_rl.local_run.make_probes", return_value=[])
                bind = mock("fine_tuning.notebook_rl.fact_context_probe.bind_history_tokens")
                sync = mock("fine_tuning.notebook_rl.trainer.synchronize", return_value={})

                def generate(*args):
                    if mutation == "deadline":
                        raise TimeoutError("fixture deadline")
                    if mutation == "parent":
                        parent_file.write_text("changed")
                    return [{"samples": [{}] * 4}], [{"reason": "NO_FIRST_DIAG"}]

                collect = mock("fine_tuning.notebook_rl.notebook_whole_synthesis_probe.collect", side_effect=generate)
                unchanged = mock("fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged")
                args = [root / name for name in ("prep", "model", "cache", "parent", "output")]
                original_trees = copy.deepcopy(trees)
                if mutation:
                    with self.assertRaises(TimeoutError if mutation == "deadline" else ValueError):
                        probe.capture(*args, notebook_whole_synthesis=True)
                    self.assertEqual(probe.read(root / "output/result.json")["status"], "FAILED")
                else:
                    result = probe.capture(*args, notebook_whole_synthesis=True)
                    self.assertEqual(result["status"], "WHOLE_SYNTHESIS_DIAGNOSTIC_CAPTURED")
                    self.assertEqual((result["calls"], result["samples"], result["optimizer_updates"]), (1, 4, 0))
                    self.assertFalse(result["training_allowed"])
                    self.assertTrue(result["trainables_unchanged"])
                    unchanged.assert_called_once_with(bundle)
                self.assertEqual(trees, original_trees)
                self.assertEqual(cache.call_args.args[2], self.f.protocol)
                self.assertEqual(read.call_args.args[1], self.f.protocol)
                self.assertEqual(historical.call_args.args[2], self.f.protocol)
                self.assertEqual(engine.call_args.args[2], 24576)
                self.assertEqual(engine.call_args.kwargs["policy_version"], 0)
                self.assertEqual(sync.call_args.args[5], 0)
                self.assertEqual(collect.call_args.args[:2], (actors, trees))
                self.assertIs(collect.call_args.args[5], verifier)
                self.assertGreater(collect.call_args.args[7], 0)
                self.assertEqual(sources.call_count, 1 if mutation in ("weights", "deadline") else 2)
                guards.assert_called_once_with(prepared, {})
                load.assert_called_once()
                trained.assert_not_called()
                bind.assert_called_once()
                self.assertEqual(image.close.call_count, 2)


if __name__ == "__main__":
    unittest.main()
