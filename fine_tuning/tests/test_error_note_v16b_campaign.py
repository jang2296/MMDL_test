"""CPU campaign contracts: complete schedule, local coverage, rollback and export boundaries."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fine_tuning.error_note import campaign_v16b as campaign
from fine_tuning.error_note.prep_v16b import read_sealed, seal
from fine_tuning.tests.test_error_note_v16b_prep import item


def arguments(root, smoke=False):
    return argparse.Namespace(data_root=root / "data", prep_root=root / "prep", model_path=root / "model",
        output=root / "campaign", precision="nf4" if smoke else "bf16", engine="hf" if smoke else "vllm",
        local_sequential=smoke, nf4_cache=root / "existing-cache" if smoke else None,
        kv_cache_gib=24., gpu_memory_utilization=.65, token_budget=12000,
        allow_model_run=False, allow_update=False, allow_save=False)


def prepare_fixture(root, smoke=False):
    args = arguments(root, smoke)
    args.data_root.mkdir()
    for name in ("payload-receipt.json", "order.json"):
        (args.data_root / name).write_text("{}\n")
    selection = seal(args.prep_root / "selection.json", {"prep_ids": ["prep0"], "monitor_ids": ["monitor0"]})
    generated = seal(args.prep_root / "f0-generations.json", {"monitor": [], "monitor_summary": {"median_tokens": 20}})
    row = {"row_id": "prep0:PREP_HUMAN", "kind": "PREP_HUMAN", "problem_id": "prep0"}
    prep = seal(args.prep_root / "prep-rows.json", {"selection_sha256": selection["sha256"],
        "f0_generations_sha256": generated["sha256"], "rows": [row], "batches": [[row]],
        "prep_updates": 1, "local_sequential": smoke})
    seal(args.prep_root / "baseline-init.json", {"prep_rows_sha256": prep["sha256"], "overall": .5})
    items = [item(f"train{i}") for i in range(3000)]
    for i, layer in zip((10, 20, 30), ("geometry", "graph", "scene")):
        items[i]["layer"] = layer
    with patch("fine_tuning.error_note.data_v16b.validate"), patch(
            "fine_tuning.error_note.data_v16b.load_items", return_value=items), patch(
            "fine_tuning.error_note.prep_v16b.select", return_value=selection), patch.object(
            campaign.single, "source_manifest", return_value={"actual_files": {"fixture.py": "c" * 64}}), patch.object(
            campaign, "preflight_items", side_effect=lambda entries: {"count": len(entries), "items": [
                {"problem_id": entry["problem_id"], "observation_eligibility": {
                    "problem_id": entry["problem_id"], "eligible": bool(entry.get("layer")),
                    "reason": None if entry.get("layer") else "NO_NATIVE_OBSERVATION"}} for entry in entries]}):
        plan = campaign.prepare(args)
    return args, plan


class CampaignTest(unittest.TestCase):
    def test_full_schedule_keeps_last_eight_and_three_flush_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = prepare_fixture(Path(directory))
            self.assertEqual(len(plan["blocks"]), 188)
            self.assertEqual(len(plan["blocks"][-1]), 8)
            self.assertEqual(plan["total_slots"], 1 + 188 + 3)
            self.assertEqual(plan["protocol"]["optimizer"]["total_steps"], plan["total_slots"])
            self.assertEqual(plan["capacity"]["max_num_seqs"], 32)
            self.assertIsNone(plan["time_cut"])
            self.assertFalse(plan["auto_evaluate"])
            old = (args.output / "campaign-plan.json").read_bytes()
            (args.output / "eta.json").write_text('{"done":4}')
            self.assertEqual(old, (args.output / "campaign-plan.json").read_bytes())

    def test_local_schedule_covers_supported_observation_layers_in_frozen_relative_order(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan = prepare_fixture(Path(directory), smoke=True)
            self.assertEqual([len(block) for block in plan["blocks"]], [2] * 4)
            self.assertEqual({item.get("layer") for item in plan["items"]} - {None}, {"geometry", "graph", "scene"})
            indices = [int(entry["problem_id"].removeprefix("train")) for entry in plan["items"]]
            self.assertEqual(indices, sorted(indices))
            self.assertFalse(plan["protocol"]["training_contract"]["cloud_allowed"])

    def test_plan_rejects_deadline_order_and_schedule_changes_even_if_rehashed(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan = prepare_fixture(Path(directory))
            for key, value in (("time_cut", 3600), ("total_slots", 4), ("blocks", plan["blocks"][:-1])):
                changed = copy.deepcopy(plan)
                changed[key] = value
                changed["sha256"] = campaign.digest({k: v for k, v in changed.items() if k != "sha256"})
                with self.assertRaisesRegex(ValueError, "immutable"):
                    campaign._plan_check(changed)

    def test_stop_and_watchdog_never_open_automatic_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for status in ("STOPPED", "WATCHDOG_STOP", "PREP_LENGTH_STOP"):
                body = {"status": status, "stop_reasons": [status]}
                (root / "campaign-state.json").write_text(json.dumps(body | {"sha256": campaign.digest(body)}))
                with self.assertRaisesRegex(ValueError, "explicit evaluation decision"):
                    campaign.require_complete(root)

    def test_rollback_moves_only_uncommitted_transaction_and_later_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("transactions", "update-results"):
                (root / name).mkdir()
                for slot in (7, 8, 9):
                    (root / name / f"slot-{slot}.json").write_text(json.dumps({"slot": slot}))
            archived = root / "rolled-back" / "previous" / "update-results" / "slot-9.json"
            archived.parent.mkdir(parents=True)
            archived.write_text("previous failure")
            for slot in (1, 8, 9):
                (root / f"checkpoint-{slot}-rl").mkdir()
            campaign._rollback(root, 8)
            self.assertTrue((root / "transactions" / "slot-7.json").exists())
            self.assertFalse((root / "transactions" / "slot-8.json").exists())
            self.assertTrue((root / "checkpoint-8-rl").exists())
            self.assertFalse((root / "checkpoint-9-rl").exists())
            self.assertEqual(len(list((root / "rolled-back").glob("recovery-*"))), 1)
            self.assertEqual(archived.read_text(), "previous failure")
            self.assertTrue((root / "update-results" / "slot-7.json").exists())
            for slot in (8, 9):
                self.assertFalse((root / "update-results" / f"slot-{slot}.json").exists())
                recovered = list((root / "rolled-back").glob(f"recovery-*/update-results/slot-{slot}.json"))
                self.assertEqual(len(recovered), 1)
                self.assertEqual(json.loads(recovered[0].read_text()), {"slot": slot})

    def test_export_requires_latest_committed_phase_and_rejects_local_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = prepare_fixture(Path(directory))
            path = args.output / "checkpoint-192-rl"
            path.mkdir()
            cursor = {"phase": "final", "slot": 192, "updates": 190}
            metadata = {"protocol": plan["protocol"], "cursor": {"single": cursor}, "step": 192}
            (path / "metadata.json").write_text(json.dumps(metadata))
            seal(args.output / "campaign-state.json", {"plan_sha256": plan["sha256"], "cursor": cursor,
                "checkpoint": campaign.descriptor(path / "metadata.json"), "status": "COMPLETE"})
            exported = campaign.export_metadata(metadata, plan, path)
            self.assertEqual(exported["stage"], "final")
            tampered = copy.deepcopy(metadata)
            tampered["step"] -= 1
            with self.assertRaisesRegex(ValueError, "latest committed"):
                campaign.export_metadata(tampered, plan, path)
            plan["local_sequential"] = True
            with patch.object(campaign, "_plan_check"):
                with self.assertRaisesRegex(ValueError, "NF4 smoke"):
                    campaign.export_metadata(metadata, plan, path)

    def test_dispatch_reaches_v16b_before_legacy_parser(self):
        from fine_tuning.error_note import single_campaign
        with patch.object(campaign, "main", return_value=17) as main:
            self.assertEqual(single_campaign.main(["prepare", "--data-root", "fixture"]), 17)
        main.assert_called_once_with(["prepare", "--data-root", "fixture"])

    def test_eta_file_does_not_enter_source_or_plan_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = prepare_fixture(Path(directory))
            old = campaign._binding(plan)
            (args.output / "eta.json").write_text('{"remaining_seconds": 1}')
            self.assertEqual(old, campaign._binding(read_sealed(args.output / "campaign-plan.json")))


    def _run_fixture_campaign(self, scenario="complete"):
        import contextlib
        from types import SimpleNamespace
        import torch
        from fine_tuning.tests.test_error_note_v16b_prep import Sampler

        class FixtureSampler(Sampler):
            def release_idle_versions(self):
                self.registry.clear()

            def live_registry(self):  # real samplers report refcounts; drain restore checks them
                return {version: dict(entry, refcount=0) for version, entry in self.registry.items()}

            def prefill_barrier(self):
                return True

            def release(self):
                self.assert_drained = not self.pending

            def step(self):
                request = self.pending.pop(0)
                self.engine_steps += 1
                kind = request["kind"]
                text = (json.dumps({"observed_facts": [], "error": None, "lesson": {"text": "Check the arithmetic."}})
                        if kind == "NOTE" else "Final answer: " + ("B" if kind == "T1" else "A"))
                generation = {key: request[key] for key in ("kind", "policy_sha256", "policy_version", "seed", "request_sha256")}
                tokens = 3 if scenario == "collapse-recovery" and request["problem_id"].startswith("monitor") else 20
                generation.update(text=text, completion_token_ids=list(range(1, tokens + 1)), finish_reason="stop")
                if kind == "T1":
                    generation["b2_tail_flags"] = [True, None, False]  # None: unresolved boundary tie
                for key in ("baseline_receipt", "obs_baseline_receipt"):
                    if key in request:
                        generation[key] = request[key]
                return [(request, {"text": text, "generation": generation, "finish_reason": "stop"})]

        with tempfile.TemporaryDirectory() as directory:
            args, plan = prepare_fixture(Path(directory), smoke=True)
            args.allow_model_run = args.allow_update = args.allow_save = True
            model = torch.nn.Linear(1, 1, bias=False)
            bundle = SimpleNamespace(model=model, processor=SimpleNamespace(tokenizer=object()),
                params={"L": list(model.parameters()), "M": [], "V": []}, route=SimpleNamespace(apply=lambda value: None))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
            fixture_sampler = FixtureSampler()
            native = plan["items"] + [item("prep0"), item("monitor0")]
            calls = []

            def grade(parent, output):
                return (("INCORRECT", "B") if output["generation"]["kind"] == "T1" else ("CORRECT", "A"))

            def backward(policy, encoded, contract, token_budget, capacity=None):
                calls.append((contract["phase"], contract["global_slot"], [row["kind"] for row in encoded]))
                if scenario.startswith("stop") and contract["global_slot"] == (0 if scenario.endswith("-prep") else 1):
                    (args.output / "STOP_REQUESTED").write_text("fixture funding stop")
                active = bool(encoded)
                if active:
                    model.weight.grad = torch.ones_like(model.weight)
                return {"has_gradient": active, "task_signal": active}

            interrupted = []
            def save(policy, opt, lr, *, cursor, path, protocol, binding, allow_save):
                if scenario == "collapse-recovery" and path.name.endswith("-stopped") and not interrupted:
                    interrupted.append(True)
                    raise InterruptedError("fixture crash after sealed monitor")
                path.mkdir()
                (path / "metadata.json").write_text(json.dumps({"protocol": protocol, "cursor": {"single": cursor}, "step": cursor["slot"]}))

            def probe(policy, rows, path):
                (path / "probe.json").write_text('{}')
                (path / "probe_inputs.safetensors").write_bytes(b"fixture")

            hash_callers = []
            def policy_hash(policy):
                frame = sys._getframe(1)
                while frame is not None and not frame.f_code.co_filename.endswith("campaign_v16b.py"):
                    frame = frame.f_back
                hash_callers.append(frame.f_code.co_name if frame is not None else None)
                return "a" * 64

            over_bound = []
            def note_tokens(policy, request, image):
                if scenario == "note-bound" and not over_bound:
                    over_bound.append(request["problem_id"])
                    return campaign.NOTE_PROMPT_BOUND + 1
                return campaign.NOTE_PROMPT_BOUND

            patches = [
                patch.object(campaign, "_factory", return_value=lambda *args: bundle),
                patch.object(campaign, "make_sampler", return_value=fixture_sampler),
                patch.object(campaign, "_encode", side_effect=lambda bundle, rows, items, image: rows),
                patch.object(campaign, "images", return_value=[]),
                patch.object(campaign, "grader", return_value=grade),
                patch("fine_tuning.error_note.prep_v16b.grader", return_value=grade),
                patch.object(campaign.single, "source_manifest", return_value=plan["source"]),
                patch.object(campaign.single, "save_single_continuation", side_effect=save),
                patch.object(campaign.single, "load_single_continuation", side_effect=lambda *a, **k:
                    (bundle, optimizer, scheduler, copy.deepcopy(read_sealed(args.output / "campaign-state.json")["cursor"]))),
                patch("fine_tuning.error_note.data_v16b.validate"),
                patch("fine_tuning.error_note.data_v16b.load_items", side_effect=lambda root, split: [x for x in native
                    if (x["problem_id"].startswith("train") if split == "train" else x["problem_id"].startswith(split))]),
                patch("fine_tuning.error_note.rl.policy_sha256", side_effect=policy_hash),
                patch.object(campaign, "prompt_tokens", side_effect=note_tokens),
                patch("fine_tuning.error_note.run_v16._setup_execution"),
                patch("fine_tuning.notebook_rl.checkpoint.make_optimizer", return_value=(optimizer, scheduler)),
                patch("fine_tuning.error_note.train._write_probe", side_effect=probe),
                patch("fine_tuning.error_note.train_v4.backward_single_block", side_effect=backward),
                patch("fine_tuning.error_note.v16b_rows.problem_contract_v16b", return_value={"fact_catalog": [], "rules": {}}),
                patch("fine_tuning.error_note.v16b_rows.build_obs_prompt_v16b", return_value="Observe the image."),
                patch("fine_tuning.error_note.v16b_rows.observation_eligibility_v16b",
                      side_effect=lambda entry: plan["observation_eligibility"][entry["problem_id"]]),
            ]
            with contextlib.ExitStack() as stack:
                for patched in patches:
                    stack.enter_context(patched)
                if scenario == "collapse-recovery":
                    with self.assertRaisesRegex(InterruptedError, "sealed monitor"):
                        campaign.run_campaign(plan, output=args.output, args=args)
                    self.assertTrue((args.output / "monitor-post-prep.json").exists())
                    self.assertEqual(read_sealed(args.output / "campaign-state.json")["status"], "READY")
                result = campaign.run_campaign(plan, output=args.output, args=args)
                if scenario.startswith("stop-resume"):
                    state_path = args.output / "campaign-state.json"
                    sealed = state_path.read_bytes()
                    self.assertEqual(result["stop_reasons"], ["STOP_REQUESTED"])
                    self.assertEqual(campaign.run_campaign(plan, output=args.output, args=args)["sha256"], result["sha256"])
                    args.resume_funding_stop = True
                    with self.assertRaisesRegex(ValueError, "archive the STOP_REQUESTED"):
                        campaign.run_campaign(plan, output=args.output, args=args)
                    self.assertEqual(state_path.read_bytes(), sealed)
                    (args.output / "STOP_REQUESTED").rename(args.output / "STOP_REQUESTED.archived")
                    result = campaign.run_campaign(plan, output=args.output, args=args)
                    self.assertEqual(result["status"], "COMPLETE")
                    self.assertEqual(result["cursor"]["slot"], plan["total_slots"])
                    self.assertEqual(len(result["guard"]["history"]), len(plan["blocks"]))
                    self.assertEqual(campaign.require_complete(args.output)["status"], "COMPLETE")
                    with self.assertRaisesRegex(ValueError, "status COMPLETE"):
                        campaign.run_campaign(plan, output=args.output, args=args)
                    return
            if scenario == "stop":
                self.assertEqual(result["status"], "STOPPED")
                self.assertEqual(result["cursor"]["slot"], 2)
                envelope = json.loads(Path(result["runtime"]["path"]).read_text())
                self.assertTrue(envelope["state"]["pending"])
                self.assertFalse((args.output / "export-ready.json").exists())
                return
            if scenario == "collapse-recovery":
                self.assertEqual(result["status"], "PREP_LENGTH_STOP")
                self.assertEqual([slot for _, slot, _ in calls], [0])
                self.assertFalse((args.output / "runtime").exists())
                self.assertFalse((args.output / "export-ready.json").exists())
                return
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["cursor"]["slot"], 8)
            self.assertEqual(scheduler.last_epoch, 8)
            self.assertEqual([slot for _, slot, _ in calls], list(range(8)))
            self.assertEqual(calls[0][0], "prep")
            receipts = [read_sealed(path) for path in sorted((args.output / "update-results").glob("slot-*.json"))]
            self.assertEqual(len(receipts), 8)
            for receipt, (phase, slot, kinds) in zip(receipts, calls, strict=True):
                transaction = read_sealed(args.output / "transactions" / f"slot-{slot}.json")
                self.assertEqual(receipt["plan_sha256"], plan["sha256"])
                self.assertEqual(receipt["transaction_sha256"], transaction["sha256"])
                self.assertEqual((receipt["phase"], receipt["global_slot"]), (phase, slot))
                self.assertEqual(receipt["row_counts"], {kind: kinds.count(kind) for kind in set(kinds)})
                self.assertEqual(receipt["result"]["cursor"]["slot"], slot + 1)
                self.assertEqual(receipt["result"]["scheduled_slots"], 1)
                self.assertEqual(receipt["result"]["optimizer_steps"], int(bool(kinds)))
                self.assertEqual(receipt["result"]["stats"]["task_signal"], bool(kinds))
                self.assertEqual(receipt["result"]["stats"]["has_gradient"], bool(kinds))
                self.assertIn("update_gradient_norms", receipt["result"]["stats"])
                self.assertGreaterEqual(receipt["wall_seconds"], 0)
                self.assertIsNone(receipt["cuda_peak_allocated_bytes"])
                self.assertIsNone(receipt["cuda_peak_reserved_bytes"])
            self.assertEqual(receipts[0]["row_counts"], {"PREP_HUMAN": 1})
            self.assertEqual(receipts[-2]["row_counts"], {})
            self.assertTrue((args.output / "prep-checkpoint.json").is_file())
            self.assertEqual(len(result["cursor"]["consumed_problem_ids"]), 8)
            self.assertTrue((args.output / "monitor-post-prep.json").is_file())
            self.assertTrue((args.output / "monitor-final.json").is_file())
            self.assertTrue((args.output / "export-ready.json").is_file())
            (args.output / "export-ready.json").unlink()
            self.assertEqual(campaign.require_complete(args.output)["status"], "COMPLETE")
            self.assertFalse((Path(result["checkpoint"]["path"]).parent / "probe.json").exists())
            self.assertTrue(fixture_sampler.assert_drained)
            journal = [json.loads(line) for line in (args.output / "runtime" / "requests.jsonl").read_text().splitlines()]
            self.assertTrue(any(row.get("kind") == "NOTE" for row in journal))
            self.assertTrue(any(row.get("kind") == "RESOLVE" for row in journal))
            records = [json.loads(path.read_text()) for path in (args.output / "runtime" / "items").glob("*.json")]
            self.assertEqual(receipts[0]["generation_counts"], {})
            self.assertNotIn("request", hash_callers)
            self.assertEqual(json.loads((args.output / "guard.json").read_text())["tail_flags"],
                             {"resolved": 4, "unresolved": 2})
            self.assertEqual([row["tail_rate"] for row in result["guard"]["history"]], [.5] * len(plan["blocks"]))
            saved = json.loads(Path(result["runtime"]["path"]).read_text())["state"]["records"]
            if scenario == "note-bound":
                skipped = saved[over_bound[0]]
                self.assertEqual(skipped["note_skip"], {"reason": "NOTE_PROMPT_OVER_BOUND",
                    "input_tokens": campaign.NOTE_PROMPT_BOUND + 1, "bound": campaign.NOTE_PROMPT_BOUND})
                for key in ("note", "note_prompt", "note_receipt", "resolve"):
                    self.assertNotIn(key, skipped)
                self.assertTrue(any("note" in record for record in saved.values()))
            else:
                self.assertFalse(any("note_skip" in record for record in saved.values()))
            for kind in ("NOTE", "RESOLVE"):
                generated = [record for record in records if kind.lower() in record]
                self.assertTrue(all(kind.lower() + "_receipt" in record for record in generated))
                counts = [receipt["generation_counts"].get(kind, {}) for receipt in receipts]
                self.assertEqual(sum(count.get("generated", 0) for count in counts), len(generated))
                for label, accepted in (("accepted", True), ("rejected", False)):
                    self.assertEqual(sum(count.get(label, 0) for count in counts),
                        sum(record[kind.lower() + "_receipt"]["accepted"] is accepted for record in generated))


    def test_real_scheduler_runs_prep_then_four_blocks_and_all_flushes_with_shared_optimizer(self):
        self._run_fixture_campaign()

    def test_drained_checkpoint_preserves_completed_unconsumed_future_outputs(self):
        self._run_fixture_campaign("stop")

    def test_existing_collapsed_monitor_still_stops_after_crash_and_resume(self):
        self._run_fixture_campaign("collapse-recovery")

    def test_funding_stop_resumes_only_with_explicit_flag_after_marker_archive(self):
        self._run_fixture_campaign("stop-resume")

    def test_funding_stop_during_preparation_records_reason_and_resumes(self):
        self._run_fixture_campaign("stop-resume-prep")

    def test_note_prompt_over_bound_is_skipped_without_note_or_resolve(self):
        self._run_fixture_campaign("note-bound")

    def test_resume_flag_is_rejected_outside_advance(self):
        with self.assertRaisesRegex(ValueError, "advance only"):
            campaign.main(["reload", "--output", "fixture", "--resume-funding-stop"])

    def test_rollback_archives_same_slot_orphans_but_preserves_committed_checkpoint_and_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            committed = root / "checkpoint-1-prep-end"
            orphan = root / "checkpoint-1-stopped"
            probe = root / "probes" / "slot-1-prep-end"
            orphan_probe = root / "probes" / "slot-1-stopped"
            for path in (committed, orphan, probe, orphan_probe):
                path.mkdir(parents=True)
            campaign._rollback(root, 1, committed_checkpoint=committed, committed_probe=probe)
            self.assertTrue(committed.exists())
            self.assertTrue(probe.exists())
            self.assertFalse(orphan.exists())
            self.assertFalse(orphan_probe.exists())


if __name__ == "__main__":
    unittest.main()
