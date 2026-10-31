"""Authored CPU replay boundaries; scripted calls are not model or update proof."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from fine_tuning.notebook_rl import checkpoint, local_profile, repair_notebook, repair_replay as replay
from fine_tuning.notebook_rl.schema import ProblemTree, to_json
from fine_tuning.tests import test_notebook_rl_repair_training as training_fixture
from fine_tuning.tests.test_notebook_rl_schema import records


def probe_fixture(problem_ids: list[str]) -> dict:
    inputs, generated = [], []
    for pid, count in zip(problem_ids, (1, 2), strict=True):
        request_id, prompt = "local-sync:" + pid, list(range(8 + count))
        inputs.append({"request_id": request_id, "input_tokens": len(prompt), "image_tokens": count,
            "image_grid_thw": [[1, 2, 2]] * count, "processed_image_sizes": [[28, 28]] * count,
            "pixel_tensors": {"pixel_values": {"shape": [4 * count, 12], "dtype": "torch.float32", "sha256": "a" * 64},
                "image_grid_thw": {"shape": [count, 3], "dtype": "torch.int64", "sha256": "b" * 64}}})
        generated.append({"request_id": request_id, "prompt_token_ids": prompt,
            "completions": [{"token_ids": [1, 2], "logprobs": [-.1, -.2], "finish_reason": "stop", "text": "fixture"}]})
    return {"inputs": inputs, "reference_logprobs": [[-.1, -.2], [-.1, -.2]],
            "generated": generated, "task_accuracy_measured": False}


class RepairReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.protocol = local_profile.build_protocol(repair_notebook=True, context_limit=24576)

    @staticmethod
    def write(root: Path, name: str, value) -> str:
        (root / name).write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return checkpoint.file_hash(root / name)

    def journal(self, rows: list[dict], root: Path | None = None) -> list[dict]:
        root = root or self.root
        return [{"file": f"raw-{index:04d}.json", "kind": row["kind"],
                 "sha256": self.write(root, f"raw-{index:04d}.json", row)} for index, row in enumerate(rows)]

    def collect(self, *, final: bool = True, malformed: bool = False):
        fixture = training_fixture.RepairTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.fixture.cfg = replace(fixture.fixture.cfg, master_seed=3407)
        fixture.fixture.prompts = local_profile.local_prompts(self.protocol)
        rows = []
        original = training_fixture.build_tree

        def collect(*args, **kwargs):
            kwargs["on_raw"] = lambda event: rows.append({"kind": "REPAIR_RAW", "raw": event})
            kwargs["on_group"] = lambda group: rows.append({"kind": "GROUP", "raw": json.loads(to_json(group))})
            return original(*args, **kwargs)

        with patch.object(training_fixture, "build_tree", side_effect=collect):
            tree, engine = fixture.online(wrong_answer=not final, malformed=malformed)
        if final:
            engine.replies.append([fixture.fixture.final(tree), "{}", fixture.fixture.final(tree), "{}"])
        fixture.fixture.append(tree, engine,
            on_raw=lambda event: rows.append({"kind": "REPAIR_RAW", "raw": event}),
            on_group=lambda group: rows.append({"kind": "GROUP", "raw": json.loads(to_json(group))}))
        rows.append({"kind": "TREE", "raw": json.loads(to_json(tree))})
        return fixture, tree, engine, rows

    def test_actual_native_and_final_events_match_distinct_original_completion_shapes(self) -> None:
        fixture, tree, engine, rows = self.collect()
        records = self.journal(rows)
        paths = replay.validate_journal(self.root, records, [tree])
        self.assertEqual(len(paths), len(rows))
        native = [row["raw"] for row in rows if row["kind"] == "REPAIR_RAW"
                  and row["raw"]["purpose"] == "NATIVE_SOLUTION_COMPLETION"]
        finals = [row["raw"] for row in rows if row["kind"] == "REPAIR_RAW"
                  and row["raw"]["purpose"] == "REPAIR_FINAL_COMPLETION"]
        self.assertTrue(native)
        self.assertTrue(all(row["completions"] == [] for row in native))
        self.assertEqual([len(row["call"]["completions"]) for row in finals], [1, 2, 3, 4])
        self.assertEqual(len(tree.groups[-1].samples), 4)
        self.assertTrue(any("syntax_failure" in row for row in tree.notebook["rewarded_notebook"]["grades"]))

    def test_raw_missing_extra_foreign_rehashed_or_typed_changes_reject(self) -> None:
        _, tree, _, rows = self.collect()
        cases = ("missing", "extra", "foreign", "text", "seed_bool", "prefix", "group", "tree", "name")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root, changed = Path(directory), copy.deepcopy(rows)
                if case == "foreign":
                    changed[0]["raw"]["problem_id"] = "foreign"
                elif case == "text":
                    changed[0]["raw"]["completion"]["text"] += " changed"
                elif case == "seed_bool":
                    changed[0]["raw"]["index"] = False
                elif case == "prefix":
                    next(row["raw"] for row in changed if row["kind"] == "REPAIR_RAW"
                         and row["raw"]["purpose"] == "REPAIR_FINAL_COMPLETION")["call"]["completions"] = []
                elif case == "group":
                    next(row["raw"] for row in changed if row["kind"] == "GROUP")["samples"].pop()
                elif case == "tree":
                    changed[-1]["raw"]["policy_version"] = True
                raw_records = self.journal(changed, root)
                if case == "missing":
                    (root / "raw-0000.json").unlink()
                elif case == "extra":
                    self.write(root, "raw-9999.json", {})
                elif case == "name":
                    raw_records[0]["file"] = "../raw-0000.json"
                with self.assertRaises(ValueError):
                    replay.validate_journal(root, raw_records, [tree])

    def test_raw_journal_beyond_9999_orders_numerically(self) -> None:
        tree = ProblemTree("a", 0, [], "VISION_OFF", {}, {})
        event = {"kind": "TREE", "raw": {"fixture": 1}}
        names = [f"raw-{index:04d}.json" for index in range(10002)]
        self.assertEqual((names[9998], names[10001]), ("raw-9998.json", "raw-10001.json"))
        self.assertNotEqual(sorted(names), names)  # lexicographic order puts raw-10000 before raw-1001
        records = [{"file": name, "kind": "TREE", "sha256": self.write(self.root, name, event)} for name in names]
        self.assertEqual([path.name for path in replay.raw_journal_paths(self.root)], names)
        with patch.object(replay, "_expected_journal", return_value=[event] * len(names)):
            self.assertEqual(list(replay.validate_journal(self.root, records, [tree])), names)
            # resume prefix: the 10001st record's file (raw-10000) is the last of the contiguous start
            with patch.object(replay, "_expected_journal", return_value=[event] * 10001):
                self.assertEqual(len(replay.validate_journal(self.root, records[:10001], [tree], complete=False)), 10001)
            (self.root / "raw-10001.json").unlink()
            with self.assertRaisesRegex(ValueError, "contiguous complete raw journal"):
                replay.validate_journal(self.root, records, [tree])

    def test_online_only_and_intact_malformed_sources_keep_original_members(self) -> None:
        for malformed in (False, True):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as directory:
                _, tree, _, rows = self.collect(final=False, malformed=malformed)
                replay.validate_journal(Path(directory), self.journal(rows, Path(directory)), [tree])
                self.assertFalse(any(group.kind == "DIAG" for group in tree.groups))
                self.assertEqual(tree.notebook["rewarded_notebook"]["status"], "NO_CORRECT_ANSWER")
                self.assertEqual(tree.notebook["rewarded_notebook"]["source_audit"]["unavailable_result_ids"], [])
                self.assertTrue(tree.groups[1].usable)
                self.assertEqual(len(tree.groups[1].samples), 2)

    def test_cpu_factories_use_exact_local_processor_configuration_and_fresh_semantics(self) -> None:
        fixture, tree, engine, _ = self.collect(final=False)
        build = self.root / "build"
        build.mkdir()
        self.write(build, "manifest.yaml", {"split_manifests": {"train": "fixture"}})
        prepared = {"build_root": str(build), "image_root": str(build / "images"), "data_request": {}}
        fixture.verifier.metadata = {"vault_manifest_sha256": "fixture"}
        processor = SimpleNamespace(tokenizer=engine.tokenizer, image_processor=SimpleNamespace(size={"original": 1}))
        with ExitStack() as stack:
            stack.enter_context(patch("fine_tuning.notebook_rl.launch.read_descriptor",
                return_value={"index": {"path": "/private/index.json"}}))
            load = stack.enter_context(patch("fine_tuning.notebook_rl.dataset.load_images",
                side_effect=lambda *args: [image.copy() for image in fixture.images]))
            factory = stack.enter_context(patch("fine_tuning.notebook_rl.engines.InputPreparation", return_value=engine))
            stack.enter_context(patch("fine_tuning.notebook_rl.verifier_service.VerifierService",
                return_value=nullcontext(fixture.verifier)))
            auto_processor = stack.enter_context(patch("transformers.AutoProcessor.from_pretrained", return_value=processor))
            auto_config = stack.enter_context(patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace()))
            before = to_json(tree)
            first = replay.replay_trees([tree], [fixture.actor], self.protocol, prepared, self.root / "model")
            control = replay.replay_trees([tree], [fixture.actor], self.protocol, prepared, self.root / "model", control=True)
        receipt = first[tree.problem_id]
        self.assertTrue(receipt["admitted"])
        self.assertEqual(receipt["signal_components"], {"answer": False, "online": True, "final": False})
        self.assertFalse(control[tree.problem_id]["has_signal"])
        self.assertEqual(to_json(tree), before)
        self.assertEqual(processor.image_processor.size, {"original": 1})
        auto_processor.assert_called_with(str(self.root / "model"), local_files_only=True, trust_remote_code=False,
            min_pixels=self.protocol["image"]["min_pixels"], max_pixels=self.protocol["image"]["max_pixels"])
        auto_config.assert_called_with(str(self.root / "model"), local_files_only=True, trust_remote_code=False)
        self.assertEqual(factory.call_args.args[2:], (self.protocol["image"], 24576))
        self.assertEqual(load.call_count, 2)

    def test_actual_tiny_CPU_update_uses_fresh_original_receipt_and_canonical_evidence(self) -> None:
        from fine_tuning.notebook_rl.repair_evidence import repair_update_evidence
        from fine_tuning.tests.test_notebook_rl_repair_update import RepairUpdateTests

        fixture, tree, engine, rows = self.collect(final=False)
        tiny = RepairUpdateTests()
        tiny.setUp()
        self.addCleanup(tiny.doCleanups)
        inventory = [{"group": name, "name": name, "shape": list(parameter.shape),
                      "numel": parameter.numel(), "dtype": str(parameter.dtype)}
                     for name, parameter in tiny.model.named_parameters() if parameter.requires_grad]
        update, _, _ = tiny.step(tree, engine)
        receipt = fixture.prepare(tree, engine)
        summary = repair_update_evidence(update, self.protocol, trees=[tree],
            preparations={tree.problem_id: receipt}, parameter_inventory=inventory)
        replay.validate_journal(self.root, self.journal(rows), [tree])
        self.assertEqual(summary["update_sha256"], checkpoint.digest(update))
        self.assertIs(summary["S_gradient_verified"], update["repair_aggregate_gradient"]["S"])
        self.assertFalse(summary["repair_learning_verified"])

    def test_saved_probe_validator_requires_exact_images_tokens_and_finite_probabilities(self) -> None:
        ids = ["single", "multi"]
        original = probe_fixture(ids)
        replay.validate_probe_evidence(original, ids, [1, 2])
        changes = (lambda value: value["inputs"].pop(),
            lambda value: value["inputs"][0].update(request_id="foreign"),
            lambda value: value["inputs"][0].update(input_tokens=True),
            lambda value: value["inputs"][1]["image_grid_thw"].pop(),
            lambda value: value["inputs"][0]["image_grid_thw"][0].__setitem__(0, True),
            lambda value: value["inputs"][1]["processed_image_sizes"].pop(),
            lambda value: value["inputs"][0]["pixel_tensors"]["pixel_values"].update(shape=[99, 12]),
            lambda value: value["inputs"][0]["pixel_tensors"]["image_grid_thw"].update(dtype="torch.float32"),
            lambda value: value["reference_logprobs"][0].__setitem__(0, float("nan")),
            lambda value: value["reference_logprobs"][0].__setitem__(0, False),
            lambda value: value["generated"][0]["prompt_token_ids"].__setitem__(0, 0.),
            lambda value: value["generated"][0]["completions"].pop(),
            lambda value: value["generated"][0]["completions"][0].update(token_ids=[1] * 9, logprobs=[-.1] * 9),
            lambda value: value["generated"][0]["completions"][0]["logprobs"].__setitem__(0, False),
            lambda value: value["generated"][0]["completions"][0]["token_ids"].__setitem__(0, True))
        for ordinal, change in enumerate(changes):
            value = copy.deepcopy(original)
            change(value)
            with self.subTest(ordinal=ordinal), self.assertRaises(ValueError):
                replay.validate_probe_evidence(value, ids, [1, 2])

    def compose(self):
        actor = records()[0]
        actors = [actor, replace(actor, problem_id=actor.problem_id + ":multi",
            image_paths=("images/a.png", "images/b.png"), image_sha256=("a" * 64, "b" * 64))]
        trees, rows, receipts, links = [], [], {}, []
        for ordinal, current in enumerate(actors):
            record: dict[str, Any] = {"notebook_generation_policy": repair_notebook.VERSION, "status": "NO_CORRECT_ANSWER",
                      "source_audit": {"sha256": "source"}, "call": None, "group_ids": []}
            tree = ProblemTree(current.problem_id, 0, [], "VISION_OFF", {}, {
                "repair_collector": {"calls": [], "fresh_observation": None, "fresh_observation_calls": 0,
                    "fresh_observation_events": []}, "rewarded_notebook": record})
            trees.append(tree)
            rows.append({"kind": "REPAIR_RAW", "raw": {"version": record["notebook_generation_policy"],
                "problem_id": tree.problem_id, "policy_version": 0, "source_audit_sha256": "source",
                "purpose": "REPAIR_FINAL_CLOSED", "status": "NO_CORRECT_ANSWER", "call": None}})
            tree_file = f"raw-{len(rows):04d}.json"
            rows.append({"kind": "TREE", "raw": json.loads(to_json(tree))})
            receipt = {"admitted": True, "has_signal": True, "sha256": f"receipt{ordinal}"}
            receipts[current.problem_id] = receipt
            prep_file = f"repair-preparation-{ordinal:04d}.json"
            links.append({"problem_id": current.problem_id, "tree_file": tree_file,
                "tree_file_sha256": self.write(self.root, tree_file, rows[-1]), "preparation_file": prep_file,
                "preparation_file_sha256": self.write(self.root, prep_file, receipt), "preparation_receipt_sha256": receipt["sha256"]})
        raw_records = self.journal(rows)
        binding = {"protocol_sha256": self.protocol["sha256"], "prompts_sha256": self.protocol["prompts_sha256"],
                   "data_sha256": "a" * 64, "code_sha256": "b" * 64}
        prepared = {"build_root": "/fixture"}
        index = {"version": "repair-local-minimum-preparation-v1", "protocol_sha256": self.protocol["sha256"],
                 "capture_binding_sha256": checkpoint.digest(binding), "trees": links, "raw_records": raw_records}
        index_sha = self.write(self.root, "repair_preparation_index.json", index)
        update = {"status": "UPDATED_UNSYNCHRONIZED", "retained_repair_signal": {}, "repair_aggregate_gradient": {"S": False},
            "routes": {actor.problem_id: False for actor in actors}, "grad_norm": {"L": 1., "M": 1., "V": None}}
        update_sha = self.write(self.root, "update.json", update)
        summary = {"S_gradient_verified": False, "update_sha256": checkpoint.digest(update), "vjp_remeasured": False}
        envelope = {"version": "repair-local-minimum-update-v1", "protocol_sha256": self.protocol["sha256"],
            "capture_binding_sha256": checkpoint.digest(binding), "preparation_index_sha256": index_sha,
            "update_file_sha256": update_sha, "retained_repair_signal_sha256": checkpoint.digest({}),
            "repair_aggregate_gradient_sha256": checkpoint.digest(update["repair_aggregate_gradient"]), "validation": summary}
        envelope_sha = self.write(self.root, "repair_update_evidence.json", envelope)
        self.write(self.root, "launch.json", {"pid": 123, "protocol": self.protocol, "binding": binding,
            "problem_ids": [actor.problem_id for actor in actors], "model_path": "/fixture/model"})
        changes = {"changed_tensor_counts": {"L": 1, "M": 1, "V": 0}, "vision_on_observed": False,
                   "vision_off_only_observed": True, "both_routes_verified": False}
        before = {"L": {"l": "old"}, "M": {"m": "old"}, "V": {"v": "frozen"}}
        after = {"L": {"l": "new"}, "M": {"m": "new"}, "V": {"v": "frozen"}}
        self.write(self.root, "result.json", {"status": "UPDATED_SAVED_PENDING_FRESH_RELOAD", "optimizer_updates": 1,
            "attempted_ids": [actor.problem_id for actor in actors], "discarded_ids": [], "fresh_reload_verified": False,
            "repair_functional_verified": False, "repair_learning_verified": False, "S_gradient_verified": False,
            "repair_preparation_index_sha256": index_sha, "repair_update_evidence_sha256": envelope_sha, **changes})
        probes = probe_fixture([actor.problem_id for actor in actors])
        self.write(self.root, "reload_expected.json", {"step": 1, "repair_protocol_sha256": self.protocol["sha256"],
            "repair_update_evidence_sha256": envelope_sha, "trainable_sha256": after, "forward_probes": probes,
            "frozen_base_sha256": {}, "quantization": {"type": "fixture"}})
        inventory = {"trainable": [], "quantization": {"type": "fixture"}}
        self.write(self.root, "inventory.json", inventory)
        ckpt = self.root / "checkpoint"
        ckpt.mkdir()
        # Controller-only payloads; physical tensor parsing is covered separately by the tiny checkpoint test.
        for name in ("adapter.safetensors", "mergers.safetensors", "runtime.pt"):
            (ckpt / name).write_bytes(name.encode())
        self.write(ckpt, "metadata.json", {"format": checkpoint.FORMAT, "base_id": checkpoint.BASE_ID,
            "base_revision": checkpoint.BASE_REVISION, "arm": "MOD-RV", "protocol": self.protocol,
            "binding": binding, "resolved_protocol_sha256": checkpoint.digest(self.protocol), "inventory": inventory,
            "base_sha256": {}, "state_sha256": {n: sha for group in after.values() for n, sha in group.items()},
            "optimizer_names": [], "step": 1, "vision_on": False,
            "cursor": {"epoch": 0, "position": 2, "discarded_problem_ids": [],
                       "problem_ids": [a.problem_id for a in actors], "sync_pending": False},
            "files": {name: checkpoint.file_hash(ckpt / name)
                      for name in ("adapter.safetensors", "mergers.safetensors", "runtime.pt")}})
        (ckpt / "metadata.sha256").write_text(checkpoint.file_hash(ckpt / "metadata.json") + "\n")
        self.write(self.root, "parameter_changes.json", {"before": before, "after": after, **changes})
        self.write(self.root, "updated_forward_probes.json", probes)
        self.write(self.root, "updated_sync.json", [{"request_id": "local-sync:" + actor.problem_id,
            "mean_abs_delta": 0., "max_abs_delta": 0., "policy_version": 1} for actor in actors])
        evidence = Mock(return_value=summary)
        module = types.ModuleType("fine_tuning.notebook_rl.repair_evidence")
        module.repair_update_evidence = evidence  # type: ignore[attr-defined]
        return actors, trees, receipts, prepared, binding, evidence, module

    def test_capture_composes_exact_pair_original_preparations_and_canonical_update(self) -> None:
        actors, trees, receipts, prepared, binding, evidence, module = self.compose()
        with patch.dict(sys.modules, {module.__name__: module}), \
                patch("fine_tuning.notebook_rl.local_run.minimum_binding", return_value=binding), \
                patch.object(replay, "replay_trees", return_value=receipts) as physical:
            report = replay.replay_capture(self.root, protocol=self.protocol, prepared=prepared, actors=actors,
                parameter_inventory=[], require_s=False)
        self.assertEqual(report["problem_ids"], [tree.problem_id for tree in trees])
        self.assertEqual(report["eligible_problem_ids"], report["problem_ids"])
        self.assertFalse(report["repair_functional_verified"])
        self.assertFalse(report["repair_learning_verified"])
        self.assertFalse(report["vjp_remeasured"])
        self.assertEqual(report["sha256"], checkpoint.digest({k: v for k, v in report.items() if k != "sha256"}))
        self.assertEqual(evidence.call_args.kwargs["preparations"], receipts)
        self.assertEqual(evidence.call_args.kwargs["trees"], trees)
        self.assertIs(evidence.call_args.kwargs["control"], False)
        self.assertEqual(physical.call_args.args[:2], (trees, actors))
        self.assertEqual(len([name for name in report["paths"] if name.startswith("checkpoint/")]), 5)

    def test_minimum_checkpoint_state_mismatch_and_runtime_replacement_change_proof(self) -> None:
        actors, _, receipts, prepared, binding, _, module = self.compose()
        ckpt = self.root / "checkpoint"

        def rewrite(value):
            value["files"] = {name: checkpoint.file_hash(ckpt / name) for name in value["files"]}
            self.write(ckpt, "metadata.json", value)
            (ckpt / "metadata.sha256").write_text(checkpoint.file_hash(ckpt / "metadata.json") + "\n")

        def run():
            return replay.replay_capture(self.root, protocol=self.protocol, prepared=prepared,
                                         actors=actors, parameter_inventory=[])

        with patch.dict(sys.modules, {module.__name__: module}), \
                patch("fine_tuning.notebook_rl.local_run.minimum_binding", return_value=binding), \
                patch.object(replay, "replay_trees", return_value=receipts):
            original = run()
            metadata = json.loads((ckpt / "metadata.json").read_text())
            (ckpt / "runtime.pt").write_bytes(b"different saved optimizer")
            rewrite(metadata)
            replaced = run()
            self.assertNotEqual(original["sha256"], replaced["sha256"])
            self.assertNotEqual(original["paths"]["checkpoint/runtime.pt"], replaced["paths"]["checkpoint/runtime.pt"])
            for field, value in (("state_sha256", {"l": "old", "m": "new", "v": "frozen"}),
                                 ("cursor", metadata["cursor"] | {"position": 1}), ("step", 2),
                                 ("base_sha256", {"foreign": "sha"})):
                bad = copy.deepcopy(metadata)
                bad[field] = value
                rewrite(bad)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "checkpoint differs"):
                    run()

    def test_real_tiny_checkpoint_snapshot_all_payloads_and_read_race(self) -> None:
        from fine_tuning.tests.test_notebook_rl_checkpoint import CheckpointTests

        fixture = CheckpointTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        ckpt, saved = fixture.checkpoint(1)
        actual, paths = replay.checkpoint_snapshot(fixture.root, ckpt.name, binding=fixture.binding)
        self.assertEqual(actual, saved)
        self.assertEqual(len(paths), 5)
        self.assertTrue(all(checkpoint.file_hash(fixture.root / name) == sha for name, sha in paths.items()))
        original_reader = checkpoint.read_metadata

        def changed(*args, **kwargs):
            value = original_reader(*args, **kwargs)
            with (ckpt / "runtime.pt").open("ab") as stream:
                stream.write(b"mutation during snapshot")
            return value

        with patch.object(checkpoint, "read_metadata", side_effect=changed), \
                self.assertRaisesRegex(ValueError, "changed during snapshot"):
            replay.checkpoint_snapshot(fixture.root, ckpt.name, binding=fixture.binding)
        for name in ("../checkpoint", "/checkpoint", ".", "..", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                replay.checkpoint_snapshot(fixture.root, name, binding=fixture.binding)

    def test_capture_rejects_bool_counts_before_physical_replay(self) -> None:
        actors, _, receipts, prepared, binding, _, module = self.compose()
        result = json.loads((self.root / "result.json").read_text())
        result["optimizer_updates"] = True
        self.write(self.root, "result.json", result)
        with patch.dict(sys.modules, {module.__name__: module}), \
                patch("fine_tuning.notebook_rl.local_run.minimum_binding", return_value=binding), \
                patch.object(replay, "replay_trees", return_value=receipts) as physical, self.assertRaises(ValueError):
            replay.replay_capture(self.root, protocol=self.protocol, prepared=prepared, actors=actors, parameter_inventory=[])
        physical.assert_not_called()

    def test_actual_synchronize_list_shape_and_malformed_reports(self) -> None:
        actors, _, receipts, prepared, binding, _, module = self.compose()
        sync = json.loads((self.root / "updated_sync.json").read_text())
        bad: tuple[Any, ...] = ({}, [], [{}], [sync[0], sync[1] | {"policy_version": True}],
               [sync[0], sync[1] | {"request_id": "foreign"}],
               [sync[0], sync[1] | {"mean_abs_delta": -1.}],
               [sync[0], sync[1] | {"max_abs_delta": False}])
        for value in bad:
            self.write(self.root, "updated_sync.json", value)
            with self.subTest(value=value), patch.dict(sys.modules, {module.__name__: module}), \
                    patch("fine_tuning.notebook_rl.local_run.minimum_binding", return_value=binding), \
                    patch.object(replay, "replay_trees", return_value=receipts) as physical, self.assertRaises(ValueError):
                replay.replay_capture(self.root, protocol=self.protocol, prepared=prepared, actors=actors, parameter_inventory=[])
            physical.assert_not_called()

    def test_guarded_snapshot_hashes_exact_parsed_bytes_and_rejects_invalid_JSON(self) -> None:
        path = self.root / "snapshot.json"
        raw = b'{"original":true}\n'
        path.write_bytes(raw)
        guarded_read = replay._read

        def changed(value):
            result = guarded_read(value)
            path.write_text('{"changed":true}')
            return result

        with patch.object(replay, "_read", side_effect=changed):
            value, sha = replay.read_snapshot(path)
        self.assertEqual(value, {"original": True})
        self.assertEqual(sha, hashlib.sha256(raw).hexdigest())
        self.assertNotEqual(sha, checkpoint.file_hash(path))
        for invalid in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '{} trailing'):
            path.write_text(invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replay.read_snapshot(path)

    def test_top_tree_and_preparation_files_cannot_change_between_parse_and_path_binding(self) -> None:
        actors, _, receipts, prepared, binding, _, module = self.compose()
        for name in ("updated_sync.json", "raw-0001.json", "repair-preparation-0000.json"):
            raw = (self.root / name).read_bytes()
            reader = replay.read_snapshot
            mutated = False

            def changed(path):
                nonlocal mutated
                parsed, sha = reader(path)
                if path.name == name and not mutated:
                    mutated = True
                    path.write_text("{}")
                return parsed, sha

            with self.subTest(name=name), patch.dict(sys.modules, {module.__name__: module}), \
                    patch("fine_tuning.notebook_rl.local_run.minimum_binding", return_value=binding), \
                    patch.object(replay, "replay_trees", return_value=receipts), \
                    patch.object(replay, "read_snapshot", side_effect=changed), self.assertRaises(ValueError):
                replay.replay_capture(self.root, protocol=self.protocol, prepared=prepared, actors=actors, parameter_inventory=[])
            self.assertTrue(mutated)
            (self.root / name).write_bytes(raw)


if __name__ == "__main__":
    unittest.main()
