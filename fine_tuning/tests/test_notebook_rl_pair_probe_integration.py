"""CPU fixtures for the named failed-parent diagnostic; no actual GPU execution."""
import copy
import os
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.notebook_rl import checkpoint, fact_context_probe as probe
from fine_tuning.notebook_rl.local_data import ACTOR_MODE_POLICY
from fine_tuning.notebook_rl.local_profile import PROFILE, build_protocol
from fine_tuning.tests import test_notebook_rl_fact_context_probe as facts


class PairProbeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.f = facts.FactProbeTests()
        self.f.setUp()
        self.f.protocol = build_protocol(verified_note_synthesis_credit=True, paired_notebook=True, context_limit=24576)

    def fixture(self, root):
        parent, model = self.f.parent(root)
        source = root / "package-root/fine_tuning/notebook_rl/example.py"
        source.parent.mkdir(parents=True)
        source.write_text("fixture source")
        source.chmod(0o600)
        inventory = {"fine_tuning/notebook_rl/example.py": {"sha256": checkpoint.file_hash(source), "mode": 0o600}}
        probe.save(root / "sources.json", inventory)
        codes = {"example.py": checkpoint.file_hash(source)}

        def descriptor(name, value):
            path = root / name
            probe.save(path, value)
            return {"path": str(path), "file_sha256": checkpoint.file_hash(path)}

        launch = probe.read(parent / "launch.json")
        preparation_path = Path(launch["preparation"])
        prepared = probe.read(preparation_path)
        prepared.update(format="notebook-rl-local-data-v1", profile=PROFILE, status="LOCAL_DATA_REVALIDATED",
            local_protocol_sha256=self.f.protocol["sha256"], cloud_allowed=False,
            build_root=str(root / "data"), image_root=str(root / "data/images"),
            actor_projection={"policy": ACTOR_MODE_POLICY, "changed_ids": []},
            actors=prepared["execution_actors"],
            validated_data=descriptor("validated_data.json", {"status": "VALIDATED_FOR_MEASUREMENT",
                "blockers": [], "code_sha256": codes, "measurement_allowed": True, "training_allowed": False}),
            data_request=descriptor("data_request.json", {"build_root": str(root / "data")}),
            selection=descriptor("selection.json", {"problem_ids": [a.problem_id for a in self.f.actors]}))
        preparation_path.write_bytes(probe.wire(prepared))
        launch.update(profile=PROFILE, purpose="MINIMAL_TWO_PROBLEM_ONE_UPDATE")
        launch["binding"].update(data_sha256=checkpoint.digest(prepared), code_sha256=checkpoint.digest(codes))
        (parent / "launch.json").write_bytes(probe.wire(launch))
        result = probe.read(parent / "result.json") | {"status": "NO_ERROR_RELATIONSHIP_LEARNING_SIGNAL"}
        (parent / "result.json").write_bytes(probe.wire(result))
        tree_path = parent / "raw-0003.json"
        tree = probe.read(tree_path)
        diag = copy.deepcopy(tree["raw"]["groups"][0])
        diag.update(group_id="fixture-diag", kind="DIAG")
        for index, sample in enumerate(diag["samples"]):
            sample.update(sample_id=f"diag{index}", group_id="fixture-diag", kind="DIAG", diag_reward=0.)
        tree["raw"]["groups"].append(diag)
        observation = {"request_id": "fixture-reread", "completion": {"text": "raw reread"}}
        tree["raw"]["notebook"]["rewarded_notebook"] = {"fresh_observation": observation}
        tree_path.write_bytes(probe.wire(tree))
        probe.save(parent / "raw-0004.json", {"kind": "GROUP", "raw": diag})
        probe.save(parent / "raw-0005.json", {"kind": "OBSERVATION", "raw": observation | {"phase": "COMPLETED"}})
        self.seal(parent)
        return parent, model, source

    @staticmethod
    def seal(parent):
        launch = probe.read(parent / "launch.json")
        snapshot = {"capture": {p.name: {"sha256": checkpoint.file_hash(p), "mode": p.stat().st_mode & 0o777,
            "bytes": p.stat().st_size} for p in parent.iterdir()},
            "preparation_sha256": checkpoint.file_hash(Path(launch["preparation"]))}
        path = parent.parent / "minimal-audit/terminal-input-snapshot-v1.json"
        if path.exists():
            path.write_bytes(probe.wire(snapshot))
        else:
            probe.save(path, snapshot)

    def pins(self, parent):
        return patch.multiple(probe,
            PAIR_CLASSIFICATION_SOURCES=checkpoint.file_hash(parent.parent / "sources.json"),
            PAIR_CLASSIFICATION_SNAPSHOT=checkpoint.file_hash(parent.parent / "minimal-audit/terminal-input-snapshot-v1.json"))

    def test_named_failure_is_accepted_only_by_explicit_parent_path(self):
        with tempfile.TemporaryDirectory() as temp, patch("psutil.pid_exists", return_value=False):
            parent, model, _ = self.fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, "closed matching minimal parent"):
                probe.closed_parent(parent, self.f.actors, self.f.protocol, model, include_notebooks=True)
            with self.pins(parent):
                trees, evidence = probe.pair_classification_parent(parent, self.f.actors, self.f.protocol, model)
            self.assertEqual(len(trees), 2)
            self.assertFalse(evidence["historical_approval_relabelled"])
            self.assertEqual(evidence["historical_protocol_sha256"], self.f.protocol["sha256"])
            self.assertIn(str(parent / "raw-0005.json"), evidence["files"])
            self.assertTrue(evidence["modes"])

    def test_live_updated_wrong_protocol_or_exploratory_parent_is_rejected(self):
        for mutation in ("live", "update", "bool-update", "status", "protocol", "execution"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                parent, model, _ = self.fixture(Path(temp))
                launch, result = probe.read(parent / "launch.json"), probe.read(parent / "result.json")
                if mutation == "update":
                    result["optimizer_updates"] = 1
                elif mutation == "bool-update":
                    result["optimizer_updates"] = False
                elif mutation == "status":
                    result["status"] = "NO_NOTEBOOK_LEARNING_SIGNAL"
                elif mutation == "protocol":
                    launch["protocol"]["notebook_prompt_sha256"] = "changed"
                elif mutation == "execution":
                    launch["execution_contract"] = {}
                (parent / "launch.json").write_bytes(probe.wire(launch))
                (parent / "result.json").write_bytes(probe.wire(result))
                self.seal(parent)
                with self.pins(parent), patch("psutil.pid_exists", return_value=mutation == "live"), self.assertRaises(ValueError):
                    probe.pair_classification_parent(parent, self.f.actors, self.f.protocol, model)

    def test_source_capture_observation_preparation_and_manifest_tampering_rejected(self):
        for mutation in ("source", "mode", "raw", "observation", "preparation", "manifest", "snapshot", "extra-raw"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                parent, model, source = self.fixture(Path(temp))
                with self.pins(parent), patch("psutil.pid_exists", return_value=False):
                    if mutation == "source":
                        source.write_text("changed")
                    elif mutation == "mode":
                        source.chmod(0o400)
                    elif mutation in ("raw", "observation"):
                        path = parent / ("raw-0000.json" if mutation == "raw" else "raw-0005.json")
                        value = probe.read(path)
                        if mutation == "raw":
                            value["raw"]["samples"][0]["text"] = "changed"
                        else:
                            value["raw"]["completion"]["text"] = "changed"
                        path.write_bytes(probe.wire(value))
                    elif mutation == "preparation":
                        path = Path(probe.read(parent / "launch.json")["preparation"])
                        path.write_bytes(path.read_bytes() + b"\n")
                    elif mutation == "manifest":
                        (parent.parent / "sources.json").write_text("changed")
                    elif mutation == "snapshot":
                        (parent.parent / "minimal-audit/terminal-input-snapshot-v1.json").write_text("changed")
                    else:
                        probe.save(parent / "raw-0006.json", {"kind": "UNEXPECTED", "raw": {}})
                    with self.assertRaises(ValueError):
                        probe.pair_classification_parent(parent, self.f.actors, self.f.protocol, model)

    def test_resealed_invalid_source_code_or_projection_binding_is_rejected(self):
        for mutation in ("code", "actor", "training"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                parent, model, _ = self.fixture(Path(temp))
                launch = probe.read(parent / "launch.json")
                path = Path(launch["preparation"])
                prepared = probe.read(path)
                if mutation == "code":
                    launch["binding"]["code_sha256"] = "changed"
                elif mutation == "actor":
                    prepared["actor_projection"]["changed_ids"] = ["unbound-actor"]
                else:
                    checked_path = Path(prepared["validated_data"]["path"])
                    checked = probe.read(checked_path)
                    checked["training_allowed"] = True
                    checked_path.write_bytes(probe.wire(checked))
                    prepared["validated_data"]["file_sha256"] = checkpoint.file_hash(checked_path)
                path.write_bytes(probe.wire(prepared))
                launch["binding"]["data_sha256"] = checkpoint.digest(prepared)
                (parent / "launch.json").write_bytes(probe.wire(launch))
                self.seal(parent)
                with self.pins(parent), patch("psutil.pid_exists", return_value=False), self.assertRaises(ValueError):
                    probe.pair_classification_parent(parent, self.f.actors, self.f.protocol, model)

    def test_cli_old_modes_unchanged_new_mode_forwarded_and_exclusive(self):
        base = ["--preparation", "/prep", "--model-path", "/model", "--cache-check", "/cache",
                "--parent", "/parent", "--output", "/output"]
        for flags, expected in (([], {"notebook_layout": False}),
                (["--notebook-layout"], {"notebook_layout": True}),
                (["--notebook-drafts"], {"notebook_drafts": True}),
                (["--notebook-relationships"], {"notebook_relationships": True}),
                (["--notebook-relationship-comparisons"], {"notebook_relationship_comparisons": True}),
                (["--trained-notebook-reload", "/reload"], {"trained_notebook_reload": Path("/reload")}),
                (["--notebook-pair-classification"], {"notebook_pair_classification": True})):
            with self.subTest(flags=flags), patch.object(probe, "capture") as capture:
                self.assertEqual(probe.main([*base, *flags, "--worker"]), 0)
                self.assertEqual(capture.call_args.kwargs, expected)
        command = [*base, "--notebook-pair-classification"]
        with patch.object(probe.subprocess, "Popen") as popen, patch("fine_tuning.budget.terminate_group") as stop:
            popen.return_value.wait.return_value = 0
            self.assertEqual(probe.main(command), 0)
            self.assertIn("--notebook-pair-classification", popen.call_args.args[0])
            popen.return_value.wait.assert_called_once_with(timeout=3570)
            stop.assert_called_once_with(popen.return_value, grace_seconds=10)
        for extra in (["--notebook-layout"], ["--notebook-drafts"], ["--notebook-relationships"],
                      ["--notebook-relationship-comparisons"], ["--trained-notebook-reload", "/reload"]):
            with self.subTest(extra=extra), patch("sys.stderr"), self.assertRaises(SystemExit):
                probe.main([*command, *extra])
        with self.assertRaisesRegex(ValueError, "one diagnostic intervention"):
            probe.capture(*([Path("unused")] * 5), notebook_pair_classification=True, notebook_relationships=True)

    def test_capture_reuses_current_preparation_fresh_f0_guards_and_unchanged_checks(self):
        for changed_weights in (False, True):
            with self.subTest(changed_weights=changed_weights), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root, actors, trees = Path(temp), self.f.actors, self.f.trees
                data = root / "data"
                probe.save(data / "manifest.yaml", {"split_manifests": {"train": "vault"}})
                bundle = SimpleNamespace(base_sha256="base")

                def mock(target, **kwargs):
                    return stack.enter_context(patch(target, **kwargs))

                stack.enter_context(patch.dict(os.environ, MMDL_FACT_PROBE_SUPERVISOR=str(os.getppid())))
                mock("fine_tuning.notebook_rl.policy.enable_deterministic_execution", return_value={})
                cache = mock("fine_tuning.notebook_rl.local_run.cache_readiness", return_value={"original_model_root": str(root / "original")})
                read = mock("fine_tuning.notebook_rl.local_run.read_preparation", return_value=({"build_root": str(data)}, {}, actors))
                mock("fine_tuning.notebook_rl.local_run.minimal_actors", return_value=actors)
                historical = mock("fine_tuning.notebook_rl.fact_context_probe.pair_classification_parent", return_value=(trees, {"files": {}}))
                old_parent = mock("fine_tuning.notebook_rl.fact_context_probe.closed_parent")
                mock("fine_tuning.notebook_rl.approval.code_hashes", return_value={"source": "same"})
                guards = mock("fine_tuning.notebook_rl.local_run._install_actor_guards", return_value=[])
                mock("fine_tuning.notebook_rl.quantized_cache.block_original_access", return_value={"denied_original_access_attempts": 0})
                mock("fine_tuning.notebook_rl.verifier_service.VerifierService", return_value=nullcontext(SimpleNamespace(metadata={"vault_manifest_sha256": "vault"})))
                mock("fine_tuning.notebook_rl.episode_notebook.audit_episode", return_value={"complete_episode": True})
                load = mock("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy", return_value=bundle)
                trained = mock("fine_tuning.notebook_rl.checkpoint.load_eval")
                mock("fine_tuning.notebook_rl.local_run.parameter_hashes", side_effect=[{"a": "before"}, {"a": "after" if changed_weights else "before"}])
                engine = mock("fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine")
                mock("fine_tuning.notebook_rl.dataset.load_images", return_value=[])
                mock("fine_tuning.notebook_rl.local_run.make_probes", return_value=[])
                bind = mock("fine_tuning.notebook_rl.fact_context_probe.bind_history_tokens")
                sync = mock("fine_tuning.notebook_rl.trainer.synchronize", return_value={})
                collect = mock("fine_tuning.notebook_rl.notebook_pair_probe.collect", return_value=([{"samples": [{}] * 4}], []))
                unchanged = mock("fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged")
                args = [root / name for name in ("prep", "model", "cache", "parent", "output")]
                if changed_weights:
                    with self.assertRaisesRegex(ValueError, "source, parent or model changed"):
                        probe.capture(*args, notebook_pair_classification=True)
                else:
                    result = probe.capture(*args, notebook_pair_classification=True)
                    self.assertEqual((result["calls"], result["samples"], result["optimizer_updates"]), (1, 4, 0))
                    self.assertEqual(result["status"], "PAIR_CLASSIFICATION_DIAGNOSTIC_CAPTURED")
                    self.assertTrue(result["trainables_unchanged"])
                self.assertEqual(cache.call_args.args[2], self.f.protocol)
                self.assertEqual(read.call_args.args[1], self.f.protocol)
                self.assertEqual(historical.call_args.args[2], self.f.protocol)
                self.assertEqual(engine.call_args.args[2], 24576)
                self.assertEqual(engine.call_args.kwargs["policy_version"], 0)
                load.assert_called_once()
                trained.assert_not_called()
                old_parent.assert_not_called()
                guards.assert_called_once()
                unchanged.assert_called_once_with(bundle)
                bind.assert_called_once()
                collect.assert_called_once()
                self.assertEqual(sync.call_args.args[5], 0)


if __name__ == "__main__":
    unittest.main()
