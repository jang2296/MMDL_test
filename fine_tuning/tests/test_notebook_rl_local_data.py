"""Authored proof fixtures test rebind plumbing, not actual exclusion eligibility."""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fine_tuning.notebook_rl import local_data as local
from fine_tuning.notebook_rl import smoke_data_evidence as evidence
from fine_tuning.notebook_rl.dataset import SPLITS, _bytes, _write
from fine_tuning.notebook_rl.schema import load_protocol
from fine_tuning.notebook_rl.vault import read_actors, read_lineage, write_split
from fine_tuning.tests.test_notebook_rl_schema import records


class LocalDataTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / "build"
        actor, verifier, lineage, *_ = records()
        self.actors = {}
        for split in SPLITS:
            actors, verifiers, lineages = [], [], []
            for i in range(16 if split == "train" else 8):
                pid = f"fixture-{split}-{i}"
                multiple = i % 2 == 1
                source = "gen-graph-pair-v1" if multiple else "gen-graph-v1"
                paths = ("a.png", "b.png") if multiple else ("a.png",)
                actors.append(replace(actor, problem_id=pid, source=source, family="graph",
                                      mode="plain" if i % 3 == 0 else "obs",
                                      image_paths=paths, image_sha256=("a"*64,) * len(paths)))
                verifiers.append(replace(verifier, problem_id=pid))
                lineages.append(replace(lineage, problem_id=pid, source=source, split=split,
                                        structure_hash=evidence._hash(pid)[:16]))
            write_split(self.root / split, actors, verifiers, lineages)
            self.actors[split] = actors
        self.receipt = self.root / "contamination/receipt_v2.json"
        _write(self.receipt, _bytes({"fixture_only": True}))
        self.old = self.base / "old"
        self.old.mkdir()
        protocol = load_protocol(Path(local.__file__).with_name("protocol_v1.yaml"))
        scope = {"purpose": local.approval.SMOKE_PURPOSE, "protocol_sha256": protocol["sha256"],
                 "code_sha256": {"historical.py": "0"*64}, "input_files_sha256": "1"*64,
                 "producer": evidence._descriptor(Path(evidence.__file__)), "authorization": {}}
        binding = {"protocol_sha256": protocol["sha256"]}
        gates = {}
        for i in range(1, 9):
            path = self.old / f"D{i}.json"
            _write(path, _bytes({"gate": f"D{i}", "status": "PASS", "measurements": {"fixture_only": True},
                "binding_sha256": evidence._hash(binding), "scope_sha256": evidence._hash(scope),
                "producer_sha256": scope["producer"]["file_sha256"],
                "components": {"path": "authored-components", "file_sha256": "2"*64},
                "metadata_rebind": None}))
            gates[f"D{i}"] = {"artifact": evidence._descriptor(path), "status": "PASS"}
        report = self.old / "data_report.json"
        _write(report, _bytes({"format": "notebook-rl-data-gates-v1", "status": "COMPLETE",
                              "scope": scope, "gates": gates, "binding": binding}))
        self.request = self.old / "data_request.json"
        _write(self.request, _bytes({"build_root": str(self.root), "expected_measurement_scope": scope,
            "data_report": evidence._descriptor(report), "index": {"fixture": "same-index"},
            "source_registry": {"fixture": "same-registry"}, "evidence": {"fixture": "same-evidence"}}))
        self.selection = self.old / "selection.json"
        chosen = [a.problem_id for a in self.actors["train"][:10]]
        structures = read_lineage(self.root / "train")
        _write(self.selection, _bytes({"problem_ids": chosen,
            "structures": {pid: structures[pid].structure_hash for pid in chosen},
            "image_counts": {a.problem_id: len(a.image_paths) for a in self.actors["train"][:10]}}))
        self.protocol = {"profile": local.PROFILE, "quantization": "authored-test-placeholder",
            "actor_mode_policy": local.ACTOR_MODE_POLICY,
            "local_authorization": {"quote": local.AUTHORIZATION_QUOTE, "reference": "test-session",
                                    "cloud_allowed": False}}

    def assemble(self, components, root, receipt, binding, scope, output, **kwargs):
        self.assertEqual(evidence._bound_json(receipt), {"fixture_only": True})
        self.assertEqual(scope["code_sha256"], local.approval.code_hashes())
        self.assertIn(local.AUTHORIZATION_QUOTE, scope["authorization"]["approval_reference"])
        self.assertIn(local.STAGED_AUTHORIZATION_QUOTE, scope["authorization"]["approval_reference"])
        output.mkdir(parents=True)
        _write(output / "data_report.json", _bytes({"scope": scope, "binding": binding}))

    def checked(self, root, **request):
        self.assertEqual(request["index"], {"fixture": "same-index"})
        self.assertEqual(request["evidence"], {"fixture": "same-evidence"})
        self.assertEqual(request["source_registry"], {"fixture": "same-registry"})
        actors, lineages = read_actors(root / "train"), read_lineage(root / "train")
        return {"status": "VALIDATED_FOR_MEASUREMENT", "measurement_allowed": True,
                "training_allowed": False, "approval_created": False, "blockers": [],
                "actors": actors, "structures": {pid: row.structure_hash for pid, row in lineages.items()},
                "selected_ids": [a.problem_id for a in actors], "limitations": ["authored test only"],
                "code_sha256": request["expected_measurement_scope"]["code_sha256"]}

    def prepare(self, *, protocol=None, out=None):
        return local.prepare_local_data(self.request, self.selection, protocol or self.protocol,
                                        out or self.base / "prepared")

    def test_new_bindings_preserve_originals_and_emit_only_selected_training_actors(self):
        before = {str(p): p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        with patch.object(evidence, "assemble", side_effect=self.assemble) as assembly, patch.object(
                local.approval, "validate_for_gate_data", side_effect=self.checked) as validator:
            result = self.prepare()
        self.assertEqual(assembly.call_count, 1)
        self.assertEqual(validator.call_count, 1)
        self.assertEqual(validator.call_args.kwargs["timeout"], 600)
        self.assertEqual(result["status"], "LOCAL_DATA_REVALIDATED")
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["gpu_verified"])
        self.assertFalse(result["cloud_allowed"])
        self.assertFalse(result["new_exclusion_scan_performed"])
        self.assertFalse(result["new_human_review_performed"])
        self.assertEqual(result["local_protocol_sha256"], evidence._hash(self.protocol))
        selected = evidence._bound_json(result["selection"])
        actors = evidence._bound_json(result["actors"])
        self.assertEqual([a["problem_id"] for a in actors], selected["problem_ids"])
        self.assertEqual(len(actors), 10)
        from fine_tuning.notebook_rl.local_run import read_preparation
        with patch.object(local.approval, "validate_for_gate_data", side_effect=self.checked) as validator:
            loaded, request, selected_actors = read_preparation(
                self.base / "prepared/preparation.json", self.protocol | {"sha256": result["local_protocol_sha256"]})
        validator.assert_called_once()
        self.assertEqual(validator.call_args.kwargs["timeout"], 600)
        self.assertEqual(loaded, result)
        self.assertEqual(request, evidence._bound_json(result["data_request"]))
        self.assertEqual([a.problem_id for a in selected_actors], selected["problem_ids"])
        self.assertEqual(selected_actors, [replace(a, mode="obs") for a in self.actors["train"][:10]])
        self.assertEqual(actors, [json.loads(local.to_json(a)) for a in self.actors["train"][:10]])
        self.assertEqual(evidence._bound_json(result["execution_actors"]),
                         [json.loads(local.to_json(a)) for a in selected_actors])
        self.assertEqual(result["actor_projection"], {"policy": local.ACTOR_MODE_POLICY,
            "changed_ids": [a.problem_id for a in self.actors["train"][:10] if a.mode == "plain"]})
        self.assertTrue(all("train" in a["problem_id"] and "gold" not in a for a in actors))
        self.assertEqual(len(evidence._bound_json(result["validated_data"])["actors"]), 16)
        for path, raw in before.items():
            self.assertEqual(Path(path).read_bytes(), raw)

    def test_heldout_structure_separation_does_not_claim_source_separation(self):
        summary = local._split_summary(self.root, [a.problem_id for a in self.actors["train"][:10]])
        for split in SPLITS[1:]:
            self.assertEqual(summary[split]["count"], 8)
            self.assertEqual(summary[split]["selected_structure_overlap"], 0)
            self.assertEqual(summary[split]["selected_source_overlap"], ["gen-graph-pair-v1", "gen-graph-v1"])
            self.assertEqual(summary[split]["selected_family_overlap"], ["graph"])
            self.assertFalse(summary[split]["qualifies_source_and_structure_disjoint_evaluation"])
            self.assertFalse(summary[split]["promoted_to_training"])

    def test_projection_rejects_rehashed_non_mode_edits_and_downgrades(self):
        from fine_tuning.notebook_rl.local_run import read_execution_actors
        originals = self.actors["train"][:10]
        projected = local.project_observation_actors(originals, self.protocol)
        path = self.base / "execution.json"
        metadata = {"policy": local.ACTOR_MODE_POLICY,
                    "changed_ids": [a.problem_id for a in originals if a.mode == "plain"]}
        for changed in (list(reversed(projected)),
                        [replace(projected[0], question="changed"), *projected[1:]],
                        [replace(projected[0], mode="plain"), *projected[1:]],
                        [replace(projected[0], image_sha256=("b"*64,)), *projected[1:]]):
            path.write_text(json.dumps([json.loads(local.to_json(a)) for a in changed]))
            prepared = {"actor_projection": metadata, "execution_actors": evidence._descriptor(path)}
            with self.subTest(changed=changed[0]), self.assertRaisesRegex(ValueError, "mode-only"):
                read_execution_actors(prepared, originals, self.protocol)
        path.write_text(json.dumps([json.loads(local.to_json(a)) for a in projected]))
        prepared = {"actor_projection": metadata, "execution_actors": evidence._descriptor(path)}
        self.assertEqual(read_execution_actors(prepared, originals, self.protocol), projected)
        with self.assertRaisesRegex(ValueError, "projection required"):
            read_execution_actors(prepared | {"actor_projection": metadata | {"changed_ids": []}},
                                  originals, self.protocol)
        with self.assertRaisesRegex(ValueError, "observation-elicitation"):
            read_execution_actors(prepared, originals, self.protocol | {"actor_mode_policy": None})

    def test_original_actor_membership_guard_is_not_replaced_by_projection_check(self):
        from fine_tuning.notebook_rl.local_run import read_preparation
        with patch.object(evidence, "assemble", side_effect=self.assemble), patch.object(
                local.approval, "validate_for_gate_data", side_effect=self.checked):
            result = self.prepare()
        actors = evidence._bound_json(result["actors"])
        actors[0]["question"] = "changed source question"
        path = Path(result["actors"]["path"])
        path.write_text(json.dumps(actors))
        result["actors"] = evidence._descriptor(path)
        preparation = self.base / "prepared/preparation.json"
        preparation.write_text(json.dumps(result))
        with patch.object(local.approval, "validate_for_gate_data", side_effect=self.checked):
            with self.assertRaisesRegex(ValueError, "fixed eligible ten changed"):
                read_preparation(preparation, self.protocol | {"sha256": result["local_protocol_sha256"]})

    def test_observations_become_checkable_without_turning_absence_into_correctness(self):
        from fine_tuning.notebook_rl.verifier import Verifier
        actor, item, *_ = records()
        original = replace(actor, mode="plain")
        projected, = local.project_observation_actors([original], self.protocol)
        verifier = Verifier.__new__(Verifier)
        verifier.items = {item.problem_id: item}
        raw = "Observed facts:\ncell[a][b] = 8\nSolution:\nRead the cell.\nFinal answer: 8"
        self.assertIsNone(verifier.grade(item.problem_id, raw, "stop", original.mode).slots)
        checked = verifier.grade(item.problem_id, raw, "stop", projected.mode)
        self.assertEqual(checked.slots.wrong_required, ("cell[a][b]",))
        missing = verifier.grade(item.problem_id, "Final answer: 7", "stop", projected.mode)
        self.assertEqual(missing.slots.required, 1)
        self.assertEqual(missing.slots.required_correct, 0)
        verifier.items[item.problem_id] = replace(item, gold="12345", slots={"cell[a][b]": "12345"})
        self.assertEqual(local.project_observation_actors([original], self.protocol), [projected])
        self.assertEqual(original.mode, "plain")

    def test_projection_requires_explicit_policy_before_preparing_any_output(self):
        with self.assertRaisesRegex(ValueError, "observation-elicitation"):
            self.prepare(protocol=self.protocol | {"actor_mode_policy": None})
        self.assertFalse((self.base / "prepared").exists())
        for actors in ([], self.actors["train"][:1] * 2):
            with self.assertRaisesRegex(ValueError, "unique actors"):
                local.project_observation_actors(actors, self.protocol)

    def test_execution_revalidates_current_vault_and_rejects_retargeted_root(self):
        from fine_tuning.notebook_rl.local_run import read_preparation
        with patch.object(evidence, "assemble", side_effect=self.assemble), patch.object(
                local.approval, "validate_for_gate_data", side_effect=self.checked):
            result = self.prepare()
        path = self.base / "prepared/preparation.json"
        protocol = self.protocol | {"sha256": result["local_protocol_sha256"]}
        for blocker in ("STALE_VAULT", "PREFLIGHT_TIMEOUT"):
            with self.subTest(blocker=blocker), patch.object(local.approval, "validate_for_gate_data",
                    return_value={"status": "BLOCKED", "actors": [], "blockers": [blocker]}) as validator:
                with self.assertRaisesRegex(ValueError, "current input/vault"):
                    read_preparation(path, protocol)
                validator.assert_called_once()
                self.assertEqual(validator.call_args.kwargs["timeout"], 600)
        path.write_text(json.dumps(result | {"build_root": str(self.base / "different")}))
        with patch.object(local.approval, "validate_for_gate_data") as validator:
            with self.assertRaisesRegex(ValueError, "root changed"):
                read_preparation(path, protocol)
            validator.assert_not_called()

    def test_failed_isolated_validation_is_preserved_without_actor_selection_outputs(self):
        for blocker in ("UNRESOLVED_PREREQUISITE", "PREFLIGHT_TIMEOUT"):
            blocked = {"status": "BLOCKED_NOT_CHECKED", "training_allowed": False, "actors": [],
                       "blockers": [blocker], "selected_ids": [], "structures": {}}
            output = self.base / blocker
            with self.subTest(blocker=blocker), patch.object(evidence, "assemble", side_effect=self.assemble), patch.object(
                    local.approval, "validate_for_gate_data", return_value=blocked):
                result = self.prepare(out=output)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual(result["blockers"], blocked["blockers"])
            for name in ("actors", "execution_actors", "selection"):
                self.assertNotIn(name, result)
                self.assertFalse((output / f"{name}.json").exists())
            self.assertEqual(evidence._bound_json(result["validated_data"]), blocked)

    def test_no_local_quote_or_cloud_authority_or_different_profile_is_rejected(self):
        for change in ({"profile": "old-bf16"}, {"local_authorization": {}},
                       {"local_authorization": self.protocol["local_authorization"] | {"cloud_allowed": True}},
                       {"local_authorization": self.protocol["local_authorization"] | {"quote": "paraphrase"}}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "authorization"):
                self.prepare(protocol=self.protocol | change)
        self.assertFalse((self.base / "prepared").exists())

    def test_execution_protocol_hash_excludes_its_own_hash_and_detects_tampering(self):
        protocol = self.protocol | {"sha256": evidence._hash(self.protocol)}
        with patch.object(evidence, "assemble", side_effect=self.assemble), patch.object(
                local.approval, "validate_for_gate_data", side_effect=self.checked):
            result = self.prepare(protocol=protocol)
        self.assertEqual(result["local_protocol_sha256"], protocol["sha256"])
        with self.assertRaisesRegex(ValueError, "protocol hash"):
            self.prepare(protocol=protocol | {"sha256": "0"*64}, out=self.base / "bad-protocol")

    def test_heldout_promotion_changed_structures_and_duplicate_ids_are_rejected(self):
        original = json.loads(self.selection.read_bytes())
        for value in (original | {"problem_ids": [a.problem_id for a in self.actors["dev"]] + original["problem_ids"][:2]},
                      original | {"structures": {k: "0"*16 for k in original["problem_ids"]}},
                      original | {"problem_ids": original["problem_ids"][:9] + original["problem_ids"][:1]}):
            self.selection.write_bytes(_bytes(value))
            with self.assertRaises(ValueError):
                self.prepare()
        self.assertFalse((self.base / "prepared").exists())

    def test_revalidation_does_not_override_existing_output(self):
        output = self.base / "prepared"
        output.mkdir()
        sentinel = output / "previous.txt"
        sentinel.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.prepare(out=output)
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_different_components_or_changed_descriptor_fails_before_assembly(self):
        request = json.loads(self.request.read_bytes())
        report = evidence._bound_json(request["data_report"])
        bad = copy.deepcopy(report)
        bad["gates"]["D2"]["artifact"]["file_sha256"] = "0"*64
        report_path = Path(request["data_report"]["path"])
        report_path.write_bytes(_bytes(bad))
        request["data_report"] = evidence._descriptor(report_path)
        self.request.write_bytes(_bytes(request))
        with patch.object(evidence, "assemble") as assembly:
            with self.assertRaisesRegex(ValueError, "prior data proof invalid"):
                self.prepare()
        assembly.assert_not_called()


if __name__ == "__main__":
    unittest.main()
