"""CPU admission and runtime fixtures, not model-performance evidence."""
import copy
import json
import os
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.notebook_rl import checkpoint, fact_context_probe as probe
from fine_tuning.notebook_rl.local_profile import build_protocol
from fine_tuning.tests import test_notebook_rl_fact_context_probe as facts


class RelationshipRuntimeTests(unittest.TestCase):
    def protocols(self):
        current = build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
        historical = json.loads(json.dumps(current))
        historical.pop("notebook_repair_parent_policy")
        historical.pop("sha256")
        historical["sha256"] = checkpoint.digest(historical)
        self.assertEqual(historical["sha256"], probe.HISTORICAL_PROTOCOL)
        return historical, current

    def test_bridge_accepts_only_named_historical_policy_delta(self):
        historical, current = self.protocols()
        bridge = probe.relationship_protocol_bridge(historical, current)
        self.assertFalse(bridge["historical_approval_relabelled"])
        for key in ("context_limit", "prompts_sha256", "notebook_feedback_policy", "sha256"):
            changed = copy.deepcopy(historical)
            changed[key] = "changed"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "policy-only bridge"):
                probe.relationship_protocol_bridge(changed, current)
        with self.assertRaises(ValueError):
            probe.relationship_protocol_bridge(current, current)
        changed = copy.deepcopy(current)
        changed["rollout"]["diag"]["max_new_tokens"] = 2048
        with self.assertRaises(ValueError):
            probe.relationship_protocol_bridge(historical, changed)

    def test_historical_source_manifest_and_frozen_payload_are_not_trusted_by_path(self):
        historical, current = self.protocols()
        for mutation in ("manifest", "bytes", "mode", "escape"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                parent, model = root / "minimal", root / "model"
                probe.save(parent / "launch.json", {"protocol": historical})
                source = root / "package-root/fine_tuning/notebook_rl/example.py"
                source.parent.mkdir(parents=True)
                source.write_text("fixture source")
                source.chmod(0o600)
                inventory = {"fine_tuning/notebook_rl/example.py": {
                    "sha256": checkpoint.file_hash(source), "mode": 0o600}}
                if mutation == "escape":
                    inventory = {"../outside.py": next(iter(inventory.values()))}
                manifest = root / "sources.json"
                probe.save(manifest, inventory)
                pin = checkpoint.file_hash(manifest)
                if mutation == "manifest":
                    manifest.write_text("changed")
                if mutation == "bytes":
                    source.write_text("changed")
                if mutation == "mode":
                    source.chmod(0o400)
                with patch.object(probe, "HISTORICAL_SOURCES", pin), patch.object(
                        probe, "closed_parent", return_value=([], {"files": {}})):
                    with self.assertRaisesRegex(ValueError, "historical .*source"):
                        probe.relationship_parent(parent, [], current, model)

    def test_cli_exclusive_forwarding_and_owned_supervisor(self):
        for flag in ("--notebook-relationships", "--notebook-relationship-comparisons"):
            command = ["--preparation", "/prep", "--model-path", "/model", "--cache-check", "/cache",
                       "--parent", "/parent", "--output", "/output", flag]
            with patch.object(probe.subprocess, "Popen") as popen, patch("fine_tuning.budget.terminate_group") as stop:
                popen.return_value.wait.return_value = 0
                self.assertEqual(probe.main(command), 0)
                self.assertIn(flag, popen.call_args.args[0])
                popen.return_value.wait.assert_called_once_with(timeout=3570)
                stop.assert_called_once_with(popen.return_value, grace_seconds=10)
            with patch.object(probe, "capture") as capture:
                self.assertEqual(probe.main(command + ["--worker"]), 0)
                self.assertTrue(capture.call_args.kwargs[flag.removeprefix("--").replace("-", "_")])
            other = "--notebook-relationships" if flag.endswith("comparisons") else "--notebook-relationship-comparisons"
            for option in (["--notebook-layout"], ["--notebook-drafts"], ["--trained-notebook-reload", "/reload"], [other]):
                with patch("sys.stderr"), self.assertRaises(SystemExit):
                    probe.main(command + option)

    def test_capture_rejects_both_relationship_interventions(self):
        with self.assertRaisesRegex(ValueError, "one diagnostic intervention"):
            probe.capture(*([Path("unused")] * 5), notebook_relationships=True, notebook_relationship_comparisons=True)

    def test_capture_reuses_fresh_f0_guards_sync_and_unchanged_weight_checks(self):
        fixture = facts.FactProbeTests()
        fixture.setUp()
        actors, trees = fixture.actors, fixture.trees
        for changed_weights, comparisons in ((False, False), (True, False), (False, True)):
            with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root = Path(temp)
                data = root / "data"
                probe.save(data / "manifest.yaml", {"split_manifests": {"train": "vault"}})
                bundle = SimpleNamespace(base_sha256="base")

                def mock(target, **kwargs):
                    return stack.enter_context(patch(target, **kwargs))

                stack.enter_context(patch.dict(os.environ, MMDL_FACT_PROBE_SUPERVISOR=str(os.getppid())))
                mock("fine_tuning.notebook_rl.policy.enable_deterministic_execution", return_value={})
                mock("fine_tuning.notebook_rl.local_run.cache_readiness", return_value={"original_model_root": str(root / "original")})
                mock("fine_tuning.notebook_rl.local_run.read_preparation", return_value=({"build_root": str(data)}, {}, actors))
                mock("fine_tuning.notebook_rl.local_run.minimal_actors", return_value=actors)
                mock("fine_tuning.notebook_rl.fact_context_probe.relationship_parent", return_value=(trees, {"files": {}}))
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
                collect = mock("fine_tuning.notebook_rl.notebook_relationship_probe.collect", return_value=([{"samples": [{}] * 4}], []))
                unchanged = mock("fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged")
                args = [root / name for name in ("prep", "model", "cache", "parent", "output")]
                if changed_weights:
                    with self.assertRaisesRegex(ValueError, "source, parent or model changed"):
                        probe.capture(*args, notebook_relationships=not comparisons, notebook_relationship_comparisons=comparisons)
                else:
                    result = probe.capture(*args, notebook_relationships=not comparisons, notebook_relationship_comparisons=comparisons)
                    self.assertEqual((result["calls"], result["samples"], result["optimizer_updates"]), (1, 4, 0))
                    self.assertTrue(result["trainables_unchanged"])
                self.assertEqual(engine.call_args.args[2], 24576)
                self.assertEqual(engine.call_args.kwargs["policy_version"], 0)
                load.assert_called_once()
                trained.assert_not_called()
                guards.assert_called_once()
                unchanged.assert_called_once_with(bundle)
                bind.assert_called_once()
                self.assertEqual(collect.call_args.kwargs["comparisons"], comparisons)
                self.assertEqual(sync.call_args.args[5], 0)
