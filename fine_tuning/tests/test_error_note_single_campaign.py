"""Authored CPU campaign integration; never pretrained Qwen, CUDA, or real data approval."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fine_tuning.error_note import audit, references, rl, single, single_campaign as campaign, train_v4
from fine_tuning.error_note.contract import IMAGE, REFERENCE_HEADER
from fine_tuning.error_note.protocol import (SINGLE_IMAGE_V14, VERSION_SINGLE_V14, _digest, plain_protocol_single,
                                            single_blocks)
from fine_tuning.error_note.single_grading import SCORER_V2
from fine_tuning.notebook_rl import checkpoint
from fine_tuning.tests import test_error_note_single_runtime as runtime_fixture
from fine_tuning.tests.test_error_note_single_loss import (as_v10, bind_reward_fixture, block, perturbed_factory,
                                                     pin_reward_fixture)
from fine_tuning.tests.test_error_note_train import tiny_factory
from fine_tuning.tests.test_error_note_v3_audit import PROVENANCE



_SHORT = None


def setUpModule():
    """Tiny-model fixtures use short targets; the 32-token solution rule is tested in single_runtime/loss."""
    global _SHORT
    from fine_tuning.error_note import single as _single

    _SHORT = patch.object(_single, "SOLUTION_MIN_TOKENS", 1)
    _SHORT.start()


def tearDownModule():
    _SHORT.stop()

def plan_for(count=32, mode="proof", execution_max_blocks=None):
    """A v11 fixture plan; with ``execution_max_blocks`` a v14 plan whose LR schedule spans exactly that many blocks."""
    items = [{"problem_id": f"p{i}", "subject": "Biology", "source": "authored-test"} for i in range(count)]
    blocks = single_blocks([x["problem_id"] for x in items])
    v14 = {} if execution_max_blocks is None else {"version": VERSION_SINGLE_V14, "schedule_steps": execution_max_blocks}
    protocol = plain_protocol_single("bf16", len(blocks),
        contracts={key: "a" * 64 for key in ("source", "dataset", "order", "data_admission")}, token_scale=1024., **v14)
    body = {"version": campaign.VERSION, "mode": mode, "items": items, "blocks": blocks, "protocol": protocol,
            "source": {"authored_fixture": True}, "generation": {"model_files_sha256": "b" * 64},
            "capacity": {"max_num_seqs": 1, "max_num_batched_tokens": 24576, "token_budget": 0, "context_limit": 24576, "engine": "hf"},
            "gate_rule": campaign.GATE_RULE, "dev_ids_sha256": "c" * 64, "retention_ids_sha256": "d" * 64}
    if execution_max_blocks is not None:
        body["execution_max_blocks"] = execution_max_blocks
    return body | {"sha256": audit.digest(body)}


def reseal(plan):
    plan["sha256"] = audit.digest({key: value for key, value in plan.items() if key != "sha256"})
    return plan


class CampaignCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def linear_factory(self, model_path, arm, protocol):
        model = torch.nn.Linear(2, 2)
        return SimpleNamespace(model=model, params={"L": list(model.parameters())})

    def execute(self, plan, **kwargs):
        count = len(plan["items"])
        return campaign.run_campaign(plan, output=kwargs.pop("output", self.out), model_path=Path("unused"),
            actors={f"p{i}": object() for i in range(count)}, provenance={f"p{i}": PROVENANCE for i in range(count)},
            gold=lambda _: "B", grade=lambda *args: ("CORRECT", "B"), load_images=lambda _: [],
            policy_factory=self.linear_factory, encoder=lambda owner, row: row, **kwargs)

    def authored_closed(self, bundle, **kwargs):
        ids = [item["problem_id"] for item in kwargs["items"]]
        contract = kwargs["protocol"]["training_contract"] | {"expected_problem_ids": ids,
                    "policy_sha256": rl.policy_sha256(bundle)}
        path = kwargs["output"] / "rows.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"problem_id": pid, "kind": "RL"} for pid in ids]
        path.write_text(json.dumps({"status": "ROWS_READY", "rows": rows, "rows_sha256": audit.digest(rows),
                                    "contract": contract}))
        return single._state(kwargs["output"] / "phase.json", {"status": "ROWS_READY", "rows_path": path.name})


class CampaignTests(CampaignCase):
    def test_concurrent_campaign_writer_holds_before_model_construction(self):
        with single._runtime_lock(self.out, ".campaign.lock"):
            with self.assertRaisesRegex(ValueError, "concurrent writer"):
                self.execute(plan_for(2))

    def test_main_gate_preload_exact_thresholds_and_immutable_receipt(self):
        plan = plan_for(3500, "train")
        state = {"checkpoint": {"path": "authored-pilot-metadata", "file_sha256": "a" * 64}}
        gate = {"version": campaign.GATE_VERSION, "campaign_sha256": plan["sha256"], "rule": campaign.GATE_RULE,
            "pilot_checkpoint": state["checkpoint"], "dev_ids_sha256": "c" * 64, "retention_ids_sha256": "d" * 64,
            "functional_checks": {name: True for name in ("data", "probability", "gradients", "function")},
            "system_failures": 0, "missing": 0, "remaining_cost": 1., "remaining_budget": 2.,
            "metrics": {"f0": {"correct": 1, "retention_correct": 1, "no_final": 0, "token_cap": 0},
                        "hybrid": {"correct": 2, "retention_correct": 1, "no_final": 0, "token_cap": 0}}}
        path = self.out / "gate.json"
        for change in (lambda g: g["functional_checks"].update(data=1),
                       lambda g: g["metrics"]["hybrid"].update(correct=1),
                       lambda g: g["metrics"]["hybrid"].update(retention_correct=0),
                       lambda g: g["metrics"]["hybrid"].update(no_final=1),
                       lambda g: g["metrics"]["hybrid"].update(token_cap=1),
                       lambda g: g.update(remaining_cost=3.)):
            altered = copy.deepcopy(gate)
            change(altered)
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "failed or stale"):
                campaign._gate(plan, copy.deepcopy(state), campaign._descriptor(path), True)
        path.write_text(json.dumps(gate))
        descriptor = campaign._descriptor(path) | {"raw": path.read_bytes()}
        path.unlink()  # Trusted controller already read the private bytes before installing actor guards.
        campaign._gate(plan, state, descriptor, True)
        self.assertIn("main_gate", state)
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            campaign._gate(plan, state, descriptor | {"raw": b"{}"}, True)
        path.write_text(json.dumps(gate) + " ")
        with self.assertRaisesRegex(ValueError, "cannot be replaced"):
            campaign._gate(plan, state, campaign._descriptor(path), True)


    def test_all3500_proof_slots_boundary_and_no_optimizer_or_checkpoint(self):
        plan = plan_for(3500)
        seen = []

        def collect(owner, **kwargs):
            seen.append([x["problem_id"] for x in kwargs["items"]])
            return self.authored_closed(owner, **kwargs)

        proof = {"status": "HOLD", "all_parameters_unchanged": True, "gradients_cleared": True,
                 "components": {}, "missing": ["A2", "B"], "paired_incorrect_a2_b": []}
        with patch.object(single, "run_single_block", side_effect=collect), \
                patch.object(single, "prove_single_backward", return_value=proof), \
                patch.object(checkpoint, "make_optimizer", side_effect=AssertionError("optimizer in proof")), \
                patch.object(single, "save_single_continuation", side_effect=AssertionError("save in proof")):
            pilot = self.execute(plan, max_blocks=219)
            self.assertEqual((pilot["status"], pilot["cursor"]["slot"]), ("PILOT_COMPLETE", 19))
            self.assertEqual(len(pilot["cursor"]["consumed_problem_ids"]), 300)
            final = self.execute(plan, max_blocks=219, continue_main=True)
        self.assertEqual(final["status"], "COMPLETE")
        self.assertEqual(len(seen), 219)
        self.assertEqual(len(seen[18]), 12)
        self.assertEqual(seen[19][0], "p300")
        self.assertEqual([pid for group in seen for pid in group], [f"p{i}" for i in range(3500)])
        self.assertEqual(final["cursor"]["updates"], 0)
        self.assertEqual(final["proof_summary"]["verified_slots"], 219)
        self.assertEqual(final["proof_summary"]["required_component_status"], "HOLD")
        self.assertEqual(final["proof_summary"]["optimizer_steps"], 0)
        self.assertFalse(list(self.out.glob("checkpoint-*")))

    def test_mode_order_source_profile_and_prefix_changes_reject_before_work(self):
        plan = plan_for(2)
        with patch.object(single, "run_single_block", side_effect=self.authored_closed), \
                patch.object(single, "prove_single_backward", return_value={"status": "HOLD", "all_parameters_unchanged": True}):
            self.execute(plan)
        for key, value in (("mode", "train"), ("source", {"changed": True}),
                           ("capacity", plan["capacity"] | {"token_budget": 12})):
            changed = reseal(copy.deepcopy(plan) | {key: value})
            with self.assertRaisesRegex(ValueError, "binding changed|mode/source"):
                self.execute(changed, allow_update=value == "train", allow_save=value == "train")
        state = single._state(self.out / "campaign-state.json")
        state["cursor"]["consumed_problem_ids"].reverse()
        single._state(self.out / "campaign-state.json", state)
        with self.assertRaisesRegex(ValueError, "prefix"):
            self.execute(plan)

    def test_real_collector_program_gate_completes_and_rerun_reuses_outputs(self):
        fixture = runtime_fixture.RuntimeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.out = self.out / "block-0"
        plan = plan_for(2)
        fixture.protocol = plan["protocol"]

        def factory(*_):
            fixture.owner = self.linear_factory(None, None, None)
            return fixture.owner

        def run(**kwargs):
            return campaign.run_campaign(plan, output=self.out, model_path=Path("unused"),
                actors=fixture.actors, provenance={pid: PROVENANCE for pid in fixture.actors}, gold=lambda _: "B",
                grade=lambda pid, text, finish, mode: ("CORRECT" if text.endswith("B") else "INCORRECT", text[-1]),
                load_images=lambda _: [runtime_fixture.Image.new("RGB", (8, 8))], policy_factory=factory,
                stream=fixture.stream, encoder=lambda owner, row: row, **kwargs)

        with patch.object(single, "prove_single_backward", return_value={"status": "HOLD", "all_parameters_unchanged": True}), \
                patch.object(checkpoint, "make_optimizer", side_effect=AssertionError("unexpected optimizer")):
            first = run()
            self.assertEqual(first["status"], "COMPLETE")
            self.assertEqual([r["stage"] for r in fixture.seen], ["t1", "t1", "note", "resolve"])
            again = run()
            self.assertEqual(again["status"], "COMPLETE")
            self.assertEqual(len(fixture.seen), 4)

    def collect_with_reference(self, reference):
        fixture = runtime_fixture.RuntimeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        output, plan = Path(tempfile.mkdtemp(dir=self.out)), plan_for(2)
        fixture.protocol = plan["protocol"]

        def factory(*_):
            fixture.owner = self.linear_factory(None, None, None)
            return fixture.owner

        with patch.object(single, "prove_single_backward", return_value={"status": "HOLD", "all_parameters_unchanged": True}):
            state = campaign.run_campaign(plan, output=output, model_path=Path("unused"), actors=fixture.actors,
                provenance={pid: PROVENANCE for pid in fixture.actors}, gold=lambda _: "B",
                grade=lambda pid, text, finish, mode: ("CORRECT" if text.endswith("B") else "INCORRECT", text[-1]),
                load_images=lambda _: [runtime_fixture.Image.new("RGB", (8, 8))], policy_factory=factory,
                stream=fixture.stream, encoder=lambda owner, row: row, reference=reference)
        self.assertEqual(state["status"], "COMPLETE")
        note = single._state(output / "block-0" / "phase.json")["records"]["p1"]["note"]["prompt"]
        rows = single._read(Path(single._read(output / "transaction-0.json")["rows"]["path"]))["rows"]
        return fixture.seen, note, rows

    def test_campaign_reference_reaches_only_the_stored_note_prompt(self):
        text = "Solution: the dataset explains item p1."
        shown = f"{REFERENCE_HEADER}\n{text}\n\n"
        seen, note, rows = self.collect_with_reference({"p1": text}.get)
        plain_seen, plain_note, plain_rows = self.collect_with_reference({}.get)  # no text for p1: exact old prompt
        self.assertEqual([r["stage"] for r in seen], ["t1", "t1", "note", "resolve"])
        for request in seen:
            self.assertEqual(shown in request["messages"][0]["content"][-1]["text"], request["stage"] == "note", request["stage"])
        self.assertTrue(all(REFERENCE_HEADER not in r["messages"][0]["content"][-1]["text"] for r in plain_seen))
        self.assertIn(shown, note)
        self.assertNotIn(REFERENCE_HEADER, plain_note)
        self.assertEqual(note.replace(shown, ""), plain_note)
        # SFT rows A1/A2 and the RL rows keep their P0 prompts; only the note row B carries the reference.
        self.assertEqual({r["kind"]: REFERENCE_HEADER in r["prompt"] for r in rows},
                         {"RL": False, "A1": False, "A2": False, "B": True})
        self.assertTrue(all(REFERENCE_HEADER not in r["prompt"] for r in plain_rows))
        self.assertTrue(all(REFERENCE_HEADER not in r["target"] for r in rows))

    def test_pending_collector_request_never_resamples(self):
        fixture = runtime_fixture.RuntimeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        plan = plan_for(2)
        fixture.protocol = plan["protocol"]

        def factory(*_):
            fixture.owner = self.linear_factory(None, None, None)
            return fixture.owner

        def interrupted(requests, stats, stop_after):
            for index, value in enumerate(fixture.stream(requests, stats, stop_after)):
                if index:
                    raise RuntimeError("authored interruption")
                yield value

        kwargs = dict(output=self.out, model_path=Path("unused"), actors=fixture.actors,
            provenance={pid: PROVENANCE for pid in fixture.actors}, gold=lambda _: "B",
            grade=lambda *args: ("CORRECT", "B"), load_images=lambda _: [runtime_fixture.Image.new("RGB", (8, 8))],
            policy_factory=factory, stream=interrupted)
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            campaign.run_campaign(plan, **kwargs)
        count = len(fixture.seen)
        ledger = json.loads((self.out / "block-0/single-requests.json").read_text())
        self.assertIn("RuntimeError", ledger["error"])
        self.assertTrue(any(entry["state"] == "STARTED" for entry in ledger["requests"].values()))
        with self.assertRaisesRegex(ValueError, "STARTED|uncertain|resampl|incomplete|terminal|stale"):
            campaign.run_campaign(plan, **kwargs)
        self.assertEqual(len(fixture.seen), count)

    def test_train_optin_nf4_hold_and_failed_proof_do_not_advance(self):
        plan = plan_for(2, "train")
        with self.assertRaisesRegex(ValueError, "opt-in"):
            self.execute(plan)
        plan["protocol"]["precision"] = "nf4"
        plan["protocol"]["sha256"] = _digest({k: v for k, v in plan["protocol"].items() if k != "sha256"})
        reseal(plan)
        with self.assertRaisesRegex(ValueError, "HOLD NF4"):
            self.execute(plan, allow_update=True, allow_save=True)
        proof_plan = plan_for(2)
        with patch.object(single, "run_single_block", side_effect=self.authored_closed), \
                patch.object(single, "prove_single_backward", return_value={"status": "FAIL", "all_parameters_unchanged": False}):
            state = self.execute(proof_plan)
        self.assertEqual((state["status"], state["cursor"]["slot"]), ("PROOF_FAILED", 0))
        with self.assertRaisesRegex(ValueError, "production model requires"):
            campaign.run_campaign(proof_plan, output=self.out, model_path=Path("unused"), actors={}, provenance={},
                gold=lambda _: "B", grade=lambda *args: None, load_images=lambda _: [])


class PrepareTests(CampaignCase):
    def test_prepare_checks_all3500_and_pinned_rules_images_without_model_loading(self):
        from contextlib import ExitStack
        from fine_tuning.error_note import answer_contract, frozen_data, order, source_evidence
        from fine_tuning.notebook_rl import vault
        from fine_tuning.notebook_rl.schema import ActorItem, to_json

        root = self.out / "vault"
        (root / "train").mkdir(parents=True)
        (root / "images").mkdir()
        (root / "images/shared.png").write_bytes(b"authored CPU fixture image")
        (root / "train/manifest.yaml").write_text("authored CPU vault")
        (root / "manifest.yaml").write_text("authored CPU union")
        image_sha = single.file_sha(root / "images/shared.png")
        train_sha = single.file_sha(root / "train/manifest.yaml")
        actors = [ActorItem(f"p{i}", "authored-test", "fixture", "plain", f"Question{i}", ("red", "blue"),
                            "mcq", ("shared.png",), (image_sha,)) for i in range(3500)]
        private = {actor.problem_id: SimpleNamespace(gold="B", answer_type="mcq") for actor in actors}
        lineages = {actor.problem_id: SimpleNamespace(source_revision="e" * 40) for actor in actors}
        sources = {actor.problem_id: {"source": actor.source, "source_revision": "e" * 40,
            "actor_sha256": audit.text_sha256(to_json(actor)), "image_sha256": [image_sha], "split": "train",
            "vault_manifest_sha256": train_sha, "label": {"value": "B", "sha256": audit.text_sha256("B")},
            "annotations": [], "unit_policy": "MCQ_UNCHANGED"} for actor in actors}
        order_path = self.out / "order.json"
        ids = [actor.problem_id for actor in actors]
        order_path.write_text(json.dumps({"version": order.VERSION, "count": len(ids), "ids_sha256": order._ids_sha256(ids),
            "items": [{"problem_id": pid, "source": "authored-test", "subject": "Biology"} for pid in ids]}))
        approval_path = self.out / "approval.json"
        approval_path.write_text("{}")
        source_path = self.out / "sources.json"
        source_path.write_text(json.dumps({"version": source_evidence.VERSION, "items": sources}))
        rules_path = self.out / "rules.json"
        rules_path.write_text(json.dumps({"version": answer_contract.VERSION, "vault_manifest_sha256": train_sha, "items": {}}))
        supplement_path = self.out / "supplement.json"
        supplement_path.write_text(json.dumps({"version": references.SUPPLEMENT_VERSION, "fields_sha256": references.fields_sha256(),
            "missing": [], "items": {"p1": {"source": "authored-test", "annotations": []},
                                      "p7": {"source": "authored-test", "annotations": []}}}))
        model = self.out / "raw-model"
        model.mkdir()
        (model / "config.json").write_text('{"torch_dtype":"bfloat16"}')
        args = SimpleNamespace(build_root=root, model_path=model, output=self.out / "campaign", order=order_path,
            frozen_approval=approval_path, source_provenance=source_path, training_rules=rules_path,
            note_reference_supplement=supplement_path, campaign_mode="proof", precision="bf16", token_scale=1024., max_blocks=1, execution_max_blocks=1,
            pilot_ids_sha256=order._ids_sha256(ids[:300]), dev_ids_sha256="c" * 64, retention_ids_sha256="d" * 64,
            capacity_profile=None, capacity_profile_sha256=None, max_num_seqs=None, max_num_batched_tokens=None, token_budget=None)
        for name in campaign.CAMPAIGN_INPUTS:
            setattr(args, name + "_sha256", single.file_sha(getattr(args, name)))
        with ExitStack() as stack:
            checked = stack.enter_context(patch.object(frozen_data, "verify_frozen_data", return_value={"authored_cpu_fixture": True}))
            stack.enter_context(patch.object(vault, "read_actors", return_value=actors))
            stack.enter_context(patch.object(vault, "read_verifier", return_value=private))
            stack.enter_context(patch.object(vault, "read_lineage", return_value=lineages))
            stack.enter_context(patch.object(single, "source_manifest", return_value={"base_commit": "a" * 40, "actual_files": {}}))
            stack.enter_context(patch.object(campaign, "generation_identity", return_value={"model_files_sha256": "b" * 64}))
            stack.enter_context(patch("fine_tuning.notebook_rl.policy.build_policy", side_effect=AssertionError("CPU prepare loaded model")))
            # The authored source offers no real reference fields; a supplement entry stands for "a reference exists".
            stack.enter_context(patch.object(references, "note_reference", side_effect=lambda item, extra=None: "ref" if extra else None))
            prepared = campaign.prepare(args)
            self.assertEqual(len(prepared["blocks"]), 219)
            # A new plan is SINGLE v14: the LR schedule spans the plan's execution_max_blocks, not all 219 slots.
            self.assertEqual(prepared["protocol"]["version"], VERSION_SINGLE_V14)
            self.assertEqual(prepared["protocol"]["optimizer"]["total_steps"], args.execution_max_blocks)
            self.assertEqual(prepared["protocol"]["training_contract"]["schedule_steps"], args.execution_max_blocks)
            self.assertEqual(prepared["protocol"]["training_contract"]["total_slots"], 219)
            self.assertEqual(prepared["protocol"]["image"], SINGLE_IMAGE_V14)
            self.assertEqual(prepared["protocol"]["training_contract"]["note_reference"],
                {"version": references.REFERENCE_VERSION, "fields_sha256": references.fields_sha256(),
                 "supplement_sha256": args.note_reference_supplement_sha256, "max_chars": references.REFERENCE_MAX_CHARS,
                 "items_with_reference": 2})
            self.assertEqual(prepared["inputs"]["note_reference_supplement"], campaign._descriptor(supplement_path))
            original_supplement, original_sha = supplement_path.read_text(), args.note_reference_supplement_sha256
            args.note_reference_supplement_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "identity changed"):
                campaign.prepare(args)
            foreign = json.loads(original_supplement)
            foreign["items"]["p9999"] = foreign["items"]["p1"]
            supplement_path.write_text(json.dumps(foreign))
            args.note_reference_supplement_sha256 = single.file_sha(supplement_path)
            with self.assertRaisesRegex(ValueError, "outside the order"):
                campaign.prepare(args)
            changed = json.loads(original_supplement)
            del changed["items"]["p7"]  # a different supplement is a different sealed protocol
            supplement_path.write_text(json.dumps(changed))
            args.note_reference_supplement_sha256 = single.file_sha(supplement_path)
            with self.assertRaisesRegex(ValueError, "immutable binding changed"):
                campaign.prepare(args)
            supplement_path.write_text(original_supplement)
            args.note_reference_supplement_sha256 = original_sha
            self.assertEqual(campaign.prepare(args), prepared)
            self.assertEqual(prepared["blocks"][19][0], "p300")
            self.assertEqual(prepared["capacity"]["max_num_seqs"], 1)
            self.assertEqual(checked.call_args.kwargs["expected_pilot_sha256"], args.pilot_ids_sha256)
            sources["p3499"]["image_sha256"] = ["0" * 64]
            source_path.write_text(json.dumps({"version": source_evidence.VERSION, "items": sources}))
            args.source_provenance_sha256 = single.file_sha(source_path)
            with self.assertRaisesRegex(ValueError, "full3500"):
                campaign.prepare(args)
            sources["p3499"]["image_sha256"] = [image_sha]
            sources["p3499"]["unit_policy"] = "invented-relaxation"
            source_path.write_text(json.dumps({"version": source_evidence.VERSION, "items": sources}))
            args.source_provenance_sha256 = single.file_sha(source_path)
            with self.assertRaisesRegex(ValueError, "unit policy"):
                campaign.prepare(args)
            args.frozen_approval_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "identity changed"):
                campaign.prepare(args)


class TrainCampaignTests(CampaignCase):
    """Real tiny CPU Qwen loss/update/checkpoint; collector rows are authored fixtures."""

    def raw_rows(self, owner, ids):
        rows, contract = block(owner, count=len(ids), reward=getattr(self, "reward", 1))
        if getattr(self, "spike", False):
            rows[0]["generation"]["sampling_logprobs"][0] -= 3.5  # one replay-failing row; v9 drops it
        contract["total_slots"] = len(self.plan["blocks"])
        for row, pid in zip(rows, ids, strict=True):
            row.update(problem_id=pid, row_id=pid + ":RL")
            bind_reward_fixture(row)
            for name in ("input_ids", "pixel_values", "image_grid_thw"):
                row[name] = row[name].tolist()
        pin_reward_fixture(rows, contract)
        if getattr(self, "v10", False):
            contract = as_v10(rows, contract)
        contract["expected_problem_ids"] = ids
        return rows, contract

    def real_closed(self, owner, **kwargs):
        ids = [item["problem_id"] for item in kwargs["items"]]
        rows, contract = self.raw_rows(owner, ids)
        path = kwargs["output"] / "rows.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "ROWS_READY", "rows": rows, "rows_sha256": audit.digest(rows), "contract": contract}))
        return single._state(kwargs["output"] / "phase.json", {"status": "ROWS_READY", "rows_path": path.name})

    def encode(self, owner, row):
        result = copy.deepcopy(row)
        for name in ("input_ids", "image_grid_thw", "pixel_values"):
            result[name] = torch.tensor(result[name], dtype=torch.float32 if name == "pixel_values" else torch.long)
        return result

    def execute_train(self, **kwargs):
        return campaign.run_campaign(self.plan, output=kwargs.pop("output", self.out), model_path=Path("unused"), actors={},
            provenance={}, gold=lambda _: "B", grade=lambda *args: ("CORRECT", "B"), load_images=lambda _: [],
            policy_factory=kwargs.pop("policy_factory", tiny_factory), encoder=self.encode,
            allow_update=True, allow_save=True, **kwargs)

    def test_real_joint_update_checkpoint_reload_and_reconcile_completed_commit(self):
        self.plan = plan_for(2, "train")
        with patch.object(single, "run_single_block", side_effect=self.real_closed) as collect:
            first = self.execute_train()
            self.assertEqual((first["cursor"]["slot"], first["cursor"]["updates"]), (1, 1))
            restored = self.execute_train(reload_only=True)
        self.assertEqual(restored["status"], "RELOADED")
        self.assertEqual(collect.call_count, 1)
        script = """
import json
from pathlib import Path
import torch
from fine_tuning.error_note import single_campaign as campaign
from fine_tuning.tests.test_error_note_train import tiny_factory
torch.set_num_threads(2)
torch.cuda._lazy_init = lambda: (_ for _ in ()).throw(AssertionError('CUDA initialized'))
output=Path(__import__('sys').argv[1])
plan=json.loads((output/'campaign-plan.json').read_text())
result=campaign.run_campaign(plan,output=output,model_path=Path('unused'),actors={},provenance={},gold=lambda _: 'B',
    grade=lambda *args: None,load_images=lambda _: [],policy_factory=tiny_factory,reload_only=True)
assert not torch.cuda.is_initialized()
print(json.dumps({'status':result['status'],'cursor':result['cursor']}))
"""
        child = subprocess.run([sys.executable, "-c", script, str(self.out)], check=True, capture_output=True, text=True,
                               env=os.environ | {"CUDA_VISIBLE_DEVICES": ""}, timeout=45)
        fresh = json.loads(child.stdout)
        self.assertEqual(fresh["status"], "RELOADED")
        self.assertEqual(fresh["cursor"], first["cursor"])
        child = checkpoint.read_metadata(self.out / "checkpoint-1", binding=campaign._binding(self.plan))
        self.assertEqual(child["cursor"]["single"], first["cursor"])
        # Emulate death after checkpoint publication but before the state pointer commit.
        parent = checkpoint.read_metadata(self.out / "checkpoint-0", binding=campaign._binding(self.plan))
        stale = copy.deepcopy(first)
        stale.update(cursor=parent["cursor"]["single"], checkpoint=campaign._descriptor(self.out / "checkpoint-0/metadata.json"),
            pending=campaign._descriptor(self.out / "transaction-0.json"), rng=None)
        single._state(self.out / "campaign-state.json", stale)
        with patch.object(single, "run_single_block", side_effect=AssertionError("regenerated committed block")), \
                patch.object(single, "run_joint_single_update", side_effect=AssertionError("duplicate optimizer step")):
            from unittest.mock import Mock

            factory = Mock(wraps=tiny_factory)
            recovered = self.execute_train(policy_factory=factory, reload_only=True)
        self.assertEqual(recovered["cursor"], first["cursor"])
        self.assertEqual(factory.call_count, 1)

    def test_initial_checkpoint_recovery_loads_only_one_policy(self):
        from unittest.mock import Mock

        self.plan = plan_for(2, "train")
        actual_state = single._state

        def interrupted(path, value=None):
            if path.name == "campaign-state.json" and value is not None:
                raise RuntimeError("authored initial pointer crash")
            return actual_state(path, value)

        with patch.object(single, "_state", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "initial pointer"):
                self.execute_train()
        self.assertTrue((self.out / "checkpoint-0/metadata.json").exists())
        self.assertFalse((self.out / "campaign-state.json").exists())
        factory = Mock(wraps=tiny_factory)
        with patch.object(single, "save_single_continuation", side_effect=AssertionError("duplicate checkpoint")):
            result = self.execute_train(policy_factory=factory, reload_only=True)
        self.assertEqual((result["status"], result["cursor"]["slot"]), ("RELOADED", 0))
        self.assertEqual(factory.call_count, 1)

    def test_zero_signal_saves_slot_and_scheduler_without_adam_decay(self):
        self.plan, self.reward = plan_for(2, "train"), 0

        def forbidden_step(*_args, **_kwargs):
            raise AssertionError("zero-signal campaign called AdamW.step")

        with patch.object(single, "run_single_block", side_effect=self.real_closed), \
                patch.object(torch.optim.AdamW, "step", forbidden_step):
            result = self.execute_train()
        before = checkpoint.read_metadata(self.out / "checkpoint-0", binding=campaign._binding(self.plan))
        after = checkpoint.read_metadata(self.out / "checkpoint-1", binding=campaign._binding(self.plan))
        self.assertEqual(result["cursor"]["slot"], 1)
        self.assertEqual(result["cursor"]["updates"], 0)
        self.assertEqual(before["state_sha256"], after["state_sha256"])
        runtime = torch.load(self.out / "checkpoint-1/runtime.pt", weights_only=True)
        self.assertEqual(runtime["scheduler"]["last_epoch"], 1)
        self.assertFalse(runtime["optimizer"]["state"])

    def test_crash_after_step_restarts_from_parent_without_resampling(self):
        self.plan = plan_for(2, "train")
        real_save = single.save_single_continuation

        def interrupted(*args, **kwargs):
            if kwargs["path"].name == "checkpoint-1":
                raise RuntimeError("authored post-step crash")
            return real_save(*args, **kwargs)

        with patch.object(single, "run_single_block", side_effect=self.real_closed), \
                patch.object(single, "save_single_continuation", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "post-step"):
                self.execute_train()
        before = single._state(self.out / "campaign-state.json")
        self.assertEqual(before["cursor"]["slot"], 0)
        self.assertIsNotNone(before["pending"])
        with patch.object(single, "run_single_block", side_effect=AssertionError("resampling pending rows")), \
                patch.object(single, "run_joint_single_update", wraps=single.run_joint_single_update) as update:
            recovered = self.execute_train()
        self.assertEqual((recovered["cursor"]["slot"], recovered["cursor"]["updates"]), (1, 1))
        self.assertEqual(update.call_count, 1)

    def test_v10_excluded_row_anchor_and_advantage_stats_reconcile_on_crash_restart_and_v8_v9_cannot_resume(self):
        self.plan, self.spike, self.v10 = plan_for(2, "train"), True, True
        real_save = single.save_single_continuation

        def interrupted(*args, **kwargs):
            if kwargs["path"].name == "checkpoint-1":
                raise RuntimeError("authored post-step crash")
            return real_save(*args, **kwargs)

        sealed_path = self.out / "block-0" / "update-stats.json"
        with patch.object(train_v4, "SINGLE_REPLAY_BLOCK_MEAN_ATOL", 10.):
            with patch.object(single, "run_single_block", side_effect=self.real_closed), \
                    patch.object(single, "save_single_continuation", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "post-step"):
                    self.execute_train(policy_factory=perturbed_factory)
            sealed = json.loads(sealed_path.read_text())
            with patch.object(single, "run_single_block", side_effect=AssertionError("resampling pending rows")), \
                    patch.object(single, "run_joint_single_update", wraps=single.run_joint_single_update) as update:
                recovered = self.execute_train(policy_factory=perturbed_factory)
        self.assertEqual(update.call_count, 1)  # the recomputed stats matched the sealed receipt
        self.assertEqual((recovered["cursor"]["slot"], recovered["cursor"]["updates"]), (1, 1))
        self.assertEqual(json.loads(sealed_path.read_text()), sealed)
        stats = sealed["stats"]
        self.assertEqual([(e["problem_id"], e["tokens"]) for e in stats["replay"]["excluded"]], [("p0", 3)])
        self.assertGreater(stats["rl"]["anchor"], 0)
        self.assertAlmostEqual(stats["loss"], stats["rl"]["pg"] + stats["rl"]["kl"] + stats["rl"]["anchor"]
                               + sum(v["weighted_loss"] for v in stats["sft"].values()), places=12)
        # v10: one PG row per non-excluded RL row; authored-test is not in the table, so b = global_mean 0.3002.
        self.assertEqual(stats["rl"]["pg_rows"], stats["rl"]["rows"] - len(stats["replay"]["excluded"]))
        self.assertGreater(stats["rl"]["advantage_abs_mean"], 0)
        self.assertLessEqual(abs(stats["rl"]["advantage_mean"]), stats["rl"]["advantage_abs_mean"])
        closed = json.loads((self.out / "block-0" / "rows.json").read_text())
        self.assertEqual(closed["contract"]["version"], "error-note-single-response-v10")
        for row in (r for r in closed["rows"] if r["kind"] == "RL"):
            self.assertEqual((row["baseline"], row["advantage"]), (.3002, row["reward"] - .3002))
        for version in ("error-note-single-response-v8", "error-note-single-response-v9"):
            old = copy.deepcopy(self.plan)
            old["protocol"]["version"] = old["protocol"]["training_contract"]["version"] = version
            old["protocol"]["sha256"] = _digest({k: v for k, v in old["protocol"].items() if k != "sha256"})
            self.plan = reseal(old)
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "invalid single-response protocol"):
                self.execute_train(output=self.out / version)

    def test_partial_checkpoint_is_preserved_and_reports_recovery_required(self):
        self.plan = plan_for(2, "train")
        with patch.object(single, "run_single_block", side_effect=self.real_closed), \
                patch.object(single, "run_joint_single_update", side_effect=RuntimeError("authored before step")):
            with self.assertRaises(RuntimeError):
                self.execute_train()
        partial = self.out / "checkpoint-1.tmp"
        partial.mkdir()
        (partial / "incomplete").write_text("preserve")
        with patch.object(single, "run_joint_single_update", side_effect=AssertionError("unsafe replay")):
            result = self.execute_train()
        self.assertEqual(result["status"], "RECOVERY_REQUIRED")
        self.assertEqual((partial / "incomplete").read_text(), "preserve")

    def test_live_capacity_is_shared_with_generation_and_actual_backward_optimizer(self):
        from fine_tuning.error_note import capacity

        self.plan = plan_for(18, "train")
        profile = self.out / "profile.json"
        profile.write_text('{"authored_cpu_fixture":true}')
        self.plan["capacity"]["profile"] = campaign._descriptor(profile)
        reseal(self.plan)
        sessions, moments_seen = [], []

        class Session:
            def __init__(self, bundle, profile, identity, optimizer=None):
                self.bundle, self.optimizer = bundle, optimizer
                self.calls = []
                sessions.append(self)

            def partition(self, rows, layout, phase):
                self.calls.append(phase)
                return layout

            def check(self, rows, phase):
                self.calls.append(phase)

        def collect(owner, **kwargs):
            session = kwargs["stream_args"].capacity_session
            self.assertIs(session, sessions[-1])
            self.assertIs(session.bundle, owner)
            self.assertIsNotNone(session.optimizer)
            moments_seen.append(bool(session.optimizer.state))
            return self.real_closed(owner, **kwargs)

        with patch.object(capacity, "load_profile", return_value={"authored_cpu_fixture": True}), \
                patch.object(capacity, "Capacity", Session), patch.object(single, "run_single_block", side_effect=collect):
            result = self.execute_train(max_blocks=2)
        self.assertEqual(result["cursor"]["slot"], 2)
        self.assertEqual(moments_seen, [False, True])
        self.assertIn("rl_kl", sessions[0].calls)

    def test_pilot_boundary_preserves_actual_weights_moments_lr_rng_and_prefix(self):
        self._pilot_boundary(direct=False)

    def test_direct_main_continues_past_the_pilot_on_the_recorded_user_decision(self):
        self._pilot_boundary(direct=True)
        state = single._state(self.out / "campaign-state.json")
        self.assertEqual(state["main_gate"]["version"], campaign.DIRECT_GATE_VERSION)
        self.assertEqual(state["main_gate"]["decision"], campaign.DIRECT_DECISION)
        self.assertEqual(state["main_gate"]["pilot_checkpoint"], state["pilot_checkpoint"])
        self.assertEqual(state["main_gate"]["campaign_sha256"], self.plan["sha256"])

    def _pilot_boundary(self, *, direct: bool) -> None:
        import warnings

        self.plan = plan_for(316, "train")
        protocol = self.plan["protocol"]
        torch.manual_seed(3407)
        owner = tiny_factory(Path("unused"), "MOD-RV", protocol)
        opt, sched = checkpoint.make_optimizer(owner, protocol)
        # Authored zero-signal history puts the real scheduler at the pilot boundary without weight updates.
        with warnings.catch_warnings(record=True):
            for _ in range(17):
                sched.step()
        policy = rl.policy_sha256(owner)
        cursor = {"slot": 17, "updates": 0, "consumed_problem_ids": [f"p{i}" for i in range(272)],
                  "policy_sha256": policy, "last_transaction_sha256": None, "parent_checkpoint": None}
        binding = campaign._binding(self.plan)
        single._sealed(self.out / "campaign-plan.json", self.plan)
        single.save_single_continuation(owner, opt, sched, cursor=cursor, path=self.out / "checkpoint-17",
                                       protocol=protocol, binding=binding, allow_save=True)
        parent = campaign._descriptor(self.out / "checkpoint-17/metadata.json")
        rows_path = self.out / "history-17.json"
        rows_path.write_text('{"contract":{}}')
        transaction = {"version": campaign.VERSION, "plan_sha256": self.plan["sha256"], "slot": 17,
            "ids": self.plan["blocks"][17], "policy_sha256": policy, "parent_checkpoint": parent,
            "rows": campaign._descriptor(rows_path), "contract_sha256": audit.digest({})}
        single._sealed(self.out / "transaction-17.json", transaction)
        sched.step()
        cursor.update(slot=18, consumed_problem_ids=[f"p{i}" for i in range(288)],
                      last_transaction_sha256=audit.digest(transaction), parent_checkpoint=parent)
        single.save_single_continuation(owner, opt, sched, cursor=cursor, path=self.out / "checkpoint-18",
                                       protocol=protocol, binding=binding, allow_save=True)
        state = {"version": campaign.VERSION, "plan_sha256": self.plan["sha256"], "cursor": cursor,
            "checkpoint": campaign._descriptor(self.out / "checkpoint-18/metadata.json"), "pending": None,
            "rng": None, "proofs": [], "status": "READY"}
        single._state(self.out / "campaign-state.json", state)
        reference, reference_opt, reference_sched, reference_cursor = single.load_single_continuation(
            Path("unused"), self.out / "checkpoint-18", protocol, binding=binding, policy_factory=tiny_factory)
        for ids in self.plan["blocks"][18:]:
            rows, contract = self.raw_rows(reference, ids)
            single.run_joint_single_update(reference, reference_opt, reference_sched,
                [self.encode(reference, row) for row in rows], contract, cursor=reference_cursor,
                block_ids=ids, allow_update=True)
        reference_rng = checkpoint.capture_rng()
        with patch.object(single, "run_single_block", side_effect=self.real_closed), \
                patch.object(single, "run_joint_single_update", wraps=single.run_joint_single_update) as update:
            pilot = self.execute_train(max_blocks=2)
            self.assertEqual((pilot["status"], pilot["cursor"]["slot"]), ("PILOT_COMPLETE", 19))
            self.assertEqual(len(pilot["cursor"]["consumed_problem_ids"]), 300)
            with self.assertRaisesRegex(ValueError, "gate required"):
                self.execute_train(continue_main=True)
            with self.assertRaisesRegex(ValueError, "direct main requires --continue-main"):
                self.execute_train(direct_main=True)
            if direct:
                with self.assertRaisesRegex(ValueError, "no pilot/dev gate file"):
                    self.execute_train(continue_main=True, direct_main=True, pilot_gate={"file_sha256": "0" * 64})
                final = self.execute_train(continue_main=True, direct_main=True)
            gate = {"version": campaign.GATE_VERSION, "campaign_sha256": self.plan["sha256"], "rule": campaign.GATE_RULE,
                "pilot_checkpoint": pilot["checkpoint"], "dev_ids_sha256": "c" * 64, "retention_ids_sha256": "d" * 64,
                "functional_checks": {name: True for name in ("data", "probability", "gradients", "function")},
                "system_failures": 0, "missing": 0, "remaining_cost": 1., "remaining_budget": 2.,
                "metrics": {"f0": {"correct": 1, "retention_correct": 1, "no_final": 0, "token_cap": 0},
                            "hybrid": {"correct": 2, "retention_correct": 1, "no_final": 0, "token_cap": 0}}}
            gate_path = self.out / "gate.json"
            gate_path.write_text(json.dumps(gate))
            if not direct:
                final = self.execute_train(continue_main=True, pilot_gate=campaign._descriptor(gate_path))
        self.assertEqual(update.call_count, 2)
        self.assertEqual(final["cursor"]["consumed_problem_ids"], [f"p{i}" for i in range(316)])
        # Retention: only the pilot checkpoint (19, also the parent) and the newest child survive.
        self.assertEqual(sorted(p.name for p in self.out.glob("checkpoint-*")), ["checkpoint-19", "checkpoint-20"])
        loaded, loaded_opt, loaded_sched, loaded_cursor = single.load_single_continuation(
            Path("unused"), self.out / "checkpoint-20", protocol, binding=binding, policy_factory=tiny_factory)
        self.assertEqual(rl.policy_sha256(loaded), rl.policy_sha256(reference))
        self.assertEqual(loaded_sched.state_dict(), reference_sched.state_dict())
        self.assertEqual(loaded_cursor["updates"], reference_cursor["updates"])
        self.assertEqual(loaded_cursor["slot"], 20)
        for original, restored in zip(reference_opt.state.values(), loaded_opt.state.values(), strict=True):
            for key in original:
                torch.testing.assert_close(restored[key], original[key], rtol=0, atol=0)
        self.assertTrue(torch.equal(checkpoint.capture_rng()["torch"], reference_rng["torch"]))

    # v14: the LR schedule spans execution_max_blocks, the campaign never passes that slot, images follow the plan.
    def test_plan_check_requires_schedule_steps_to_equal_the_execution_block_limit(self):
        plan = plan_for(34, "train", execution_max_blocks=2)
        campaign._plan_check(plan)
        self.assertEqual((plan["protocol"]["optimizer"]["total_steps"],
                          plan["protocol"]["training_contract"]["schedule_steps"], len(plan["blocks"])), (2, 2, 3))
        for limit in (1, 3, None, True):
            changed = copy.deepcopy(plan)
            changed["execution_max_blocks"] = limit
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "invalid immutable campaign plan"):
                campaign._plan_check(reseal(changed))
        del plan["execution_max_blocks"]
        with self.assertRaisesRegex(ValueError, "invalid immutable campaign plan"):
            campaign._plan_check(reseal(plan))
        campaign._plan_check(plan_for(34, "train"))  # a v11 plan carries no schedule_steps and is checked as before

    def test_campaign_stops_at_the_execution_limit_and_a_further_advance_runs_no_block(self):
        self.plan = plan_for(34, "train", execution_max_blocks=2)
        with patch.object(single, "run_single_block", side_effect=self.real_closed) as collect:
            first = self.execute_train()
            self.assertEqual((first["status"], first["cursor"]["slot"], first["cursor"]["updates"]), ("BOUNDED_STOP", 1, 1))
            last = self.execute_train()
            self.assertEqual((last["status"], last["cursor"]["slot"], last["cursor"]["updates"]), ("COMPLETE", 2, 2))
            with patch.object(single, "run_joint_single_update", side_effect=AssertionError("block past the schedule")):
                again = self.execute_train()
        self.assertEqual((again["status"], again["cursor"]["slot"]), ("COMPLETE", 2))
        self.assertEqual(collect.call_count, 2)  # slot 2 (the third block) was never generated
        self.assertEqual(again["cursor"]["consumed_problem_ids"], [f"p{i}" for i in range(32)])
        self.assertFalse((self.out / "block-2").exists())
        runtime = torch.load(self.out / "checkpoint-2/runtime.pt", weights_only=True)
        self.assertEqual(runtime["scheduler"]["last_epoch"], 2)  # the scheduler took exactly its schedule_steps

    def test_plan_image_config_reaches_generation_and_training_inputs(self):
        for limit, expected in ((1, SINGLE_IMAGE_V14), (None, IMAGE)):
            with self.subTest(v14=limit is not None):
                self.plan = plan_for(2, "train", execution_max_blocks=limit)
                self.assertEqual(self.plan["protocol"]["image"], expected)
                seen = {}

                def collect(owner, **kwargs):
                    seen["image_cfg"] = kwargs["stream_args"].image_cfg
                    return self.real_closed(owner, **kwargs)

                def encode(processor, model_config, row, images, *, image=None):
                    seen["encode_image"] = image
                    return self.encode(None, row)

                output = self.out / f"image-{limit}"
                with patch.object(single, "run_single_block", side_effect=collect), \
                        patch.object(train_v4, "encode_single_row", side_effect=encode):
                    campaign.run_campaign(self.plan, output=output, model_path=Path("unused"), actors={"p0": object(),
                        "p1": object()}, provenance={}, gold=lambda _: "B", grade=lambda *args: ("CORRECT", "B"),
                        load_images=lambda _: [], policy_factory=tiny_factory, allow_update=True, allow_save=True)
                self.assertEqual(seen, {"image_cfg": expected, "encode_image": expected})

    def test_vllm_sampler_is_built_with_the_plan_image_config(self):
        from fine_tuning.error_note import single_vllm

        self.plan = plan_for(2, "train", execution_max_blocks=1)
        self.plan["capacity"] |= {"engine": "vllm", "kv_cache_gib": 1., "gpu_memory_utilization": .5, "vllm_sleep": False}
        reseal(self.plan)
        with patch.object(single, "run_single_block", side_effect=self.real_closed), \
                patch("transformers.GenerationConfig.from_pretrained", return_value=SimpleNamespace(eos_token_id=[7])), \
                patch.object(single_vllm, "SingleVllmSampler") as sampler:
            self.execute_train()
        self.assertEqual(sampler.call_args.kwargs["image"], SINGLE_IMAGE_V14)


class UnionGradingTests(unittest.TestCase):
    """The campaign/CLI grade closure: one response, V8 first, training-answer-v2 only to turn it CORRECT."""

    def grade(self, contract, v8, training):
        calls = []

        class Verifier:
            def grade(self, pid, text, finish, mode, *, answer_contract):
                calls.append(answer_contract)
                status = {"p0-eval-aligned-v1": v8, "training-answer-v2": training}[answer_contract]
                return SimpleNamespace(status=status, extracted=answer_contract)

        return single.grade_answer(Verifier(), contract, "p", "3.84 billion euros", "stop", "plain"), calls

    def test_open_union_mcq_v8_only(self):
        union = "p0-eval-aligned-v1|training-answer-v2"
        self.assertEqual(self.grade(union, "CORRECT", "INCORRECT"),
                         (("CORRECT", "p0-eval-aligned-v1"), ["p0-eval-aligned-v1"]))
        self.assertEqual(self.grade(union, "INCORRECT", "CORRECT"),
                         (("CORRECT", "training-answer-v2"), ["p0-eval-aligned-v1", "training-answer-v2"]))
        self.assertEqual(self.grade(union, "INVALID_NO_FINAL", "INCORRECT")[0], ("INVALID_NO_FINAL", "p0-eval-aligned-v1"))
        self.assertEqual(self.grade(union, "UNVERIFIABLE", "INVALID_NO_FINAL")[0], ("UNVERIFIABLE", "p0-eval-aligned-v1"))
        self.assertEqual(self.grade("p0-eval-aligned-v1", "INCORRECT", "CORRECT"),
                         (("INCORRECT", "p0-eval-aligned-v1"), ["p0-eval-aligned-v1"]))
        protocol = plain_protocol_single("bf16", 1, contracts={k: "a" * 64 for k in ("source", "dataset", "order",
                                                                                      "data_admission")}, token_scale=1024.)
        self.assertEqual(protocol["training_contract"]["answer_contracts"]["open"],
                         "v8-extract+normalize+training-compare-v1")

    def test_v8_open_scorer_delegates_to_single_grading_with_the_actor(self):
        scorer = "v8-extract+normalize+training-compare-v1"
        with patch("fine_tuning.error_note.single_grading.grade_single", return_value=("CORRECT", "3.84")) as graded:
            self.assertEqual(single.grade_answer("verifier", scorer, "p", "t", "stop", "plain", actor="actor"),
                             ("CORRECT", "3.84"))
        # grade_answer now passes the unit switch explicitly: off for the v8 scorer, on for v14's scorer v2.
        graded.assert_called_once_with("verifier", "actor", "p", "t", "stop", "plain", units=False)
        with patch("fine_tuning.error_note.single_grading.grade_single", return_value=("CORRECT", "3.84")) as graded_v2:
            single.grade_answer("verifier", SCORER_V2, "p", "t", "stop", "plain", actor="actor")
        graded_v2.assert_called_once_with("verifier", "actor", "p", "t", "stop", "plain", units=True)
        with self.assertRaisesRegex(ValueError, "needs the item actor"):
            single.grade_answer("verifier", scorer, "p", "t", "stop", "plain")


class ParserDefaultTests(unittest.TestCase):
    def test_cli_token_scale_defaults_build_the_v11_protocol(self):
        """v11 pins the PG token scale; both CLIs must default to it or every default run fails at prepare."""
        contracts = {key: "a" * 64 for key in ("source", "dataset", "order", "data_admission")}
        for parser in (campaign.build_parser(), single.build_parser()):
            default = parser.get_default("token_scale")
            protocol = plain_protocol_single("bf16", 1, contracts=contracts, token_scale=default)
            self.assertEqual(protocol["training_contract"]["rl"]["token_scale"], default)


if __name__ == "__main__":
    unittest.main()
