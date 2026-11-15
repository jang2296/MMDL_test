"""CPU v16 routing, continuation and unmerged export; no pretrained model or effect claim."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import torch

from fine_tuning.error_note import audit, plan as planner, protocol as proto, rl, single, single_campaign as campaign
from fine_tuning.error_note import train_v4
from fine_tuning.notebook_rl import checkpoint, proof_eval
from fine_tuning.tests.test_notebook_rl_checkpoint import fixture, fresh_tiny


def content_plan(manifest: Path, *, mode: str = "proof", slots: int = 2) -> dict:
    items = [{"problem_id": f"p{i}", "parent_id": f"p{i}", "split": "train"} for i in range(slots)]
    ids = [item["problem_id"] for item in items]
    source = {"actual_files": {"authored-fixture.py": "a" * 64}, "base_commit": "b" * 40}
    admission = {"train_count": slots, "manifest_sha256": campaign.file_sha(manifest)}
    pins = {"source": audit.digest(source["actual_files"]), "dataset": campaign.file_sha(manifest),
            "order": audit.digest(ids), "data_admission": audit.digest(admission)}
    protocol = proto.plain_protocol_content_v16("bf16", slots, contracts=pins, source_weights={"graph": 1.0})
    body = {"version": campaign.CONTENT_CAMPAIGN_VERSION, "mode": mode, "items": items,
            "blocks": [[pid] for pid in ids], "block_size": 1, "protocol": protocol, "source": source,
            "generation": {"model_files_sha256": "c" * 64}, "data_admission": admission,
            "inputs": {"content_manifest": campaign._descriptor(manifest)}, "content_rows": [],
            "capacity": {"engine": "hf", "context_limit": 24576, "token_budget": 0},
            "gate_rule": campaign.CONTENT_GATE_RULE, "execution_max_blocks": slots, "functional_gate": None}
    return body | {"sha256": audit.digest(body)}


class V16IntegrationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="v16-integration-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.manifest = self.root / "content.json"
        self.manifest.write_text("{}\n")
        self.plan = content_plan(self.manifest)

    def test_protocol_is_content_only_and_legacy_hashes_stay_fixed(self):
        protocol = self.plan["protocol"]
        single._protocol_check(protocol)
        self.assertTrue(proof_eval._lora_protocol(protocol))
        self.assertEqual(protocol["training_contract"]["generation_counts"], {"t1": 1, "note": 1, "resolve": 1})
        self.assertEqual(protocol["optimizer"]["lr"], {"L": 5e-5, "M": 1e-5, "V": 1e-5})
        self.assertEqual(protocol["optimizer"]["warmup_steps"], 4)
        self.assertNotIn("rl", protocol["training_contract"])
        expected = {
            proto.VERSION_SINGLE: "e12ab3bc40d7924be71cc0f4fefdebabaf7a048c25ecc6fa886616cbdc04a648",
            proto.VERSION_SINGLE_V12: "b0e2db5cce14a6c26366b9a7403b0a0b0757ebf3d55594e401e4510df67b4b8d",
            proto.VERSION_SINGLE_V13: "60c48ddbb926e446285cce4ae847cae3ba2fa74cdffc46ff85c01ae9454e3ef8",
            proto.VERSION_SINGLE_V14: "0eef0d7130948df0cc47a994ea66530cafdba6fcb33ae36baa2a0bc4bdfc6101",
            proto.VERSION_SINGLE_V15: "4fbffb0b2893dccb104c87fbe4b3a71af279364c2b562c93de9a1dd745a837ff"}
        pins = {key: "a" * 64 for key in proto.SINGLE_CONTRACT_HASHES}
        for version, digest in expected.items():
            extra = {"schedule_steps": 64} if version in (proto.VERSION_SINGLE_V14, proto.VERSION_SINGLE_V15) else {}
            self.assertEqual(proto.plain_protocol_single("bf16", 219, contracts=pins, version=version, **extra)["sha256"], digest)

    def test_resealed_generation_pg_or_adapter_mismatch_fails_closed(self):
        for mutate in (lambda p: p.pop("rollout_adapter"),
                       lambda p: p["training_contract"].update(pg_weight=1.0),
                       lambda p: p["training_contract"]["generation_counts"].update(resolve=3),
                       lambda p: p.update(prompts_sha256="a" * 64)):
            candidate = copy.deepcopy(self.plan["protocol"])
            mutate(candidate)
            candidate["sha256"] = proto._digest({k: v for k, v in candidate.items() if k != "sha256"})
            with self.assertRaisesRegex(ValueError, "v16 protocol"):
                single._protocol_check(candidate)

    def test_row_and_backward_dispatch_before_legacy_rl_requirements(self):
        with patch("fine_tuning.error_note.train_v16.encode_content_row", return_value={"encoded": True}) as encode:
            self.assertEqual(train_v4.encode_single_row(None, None, {"schema": "v16-content-row-v1"}, []), {"encoded": True})
            self.assertEqual(encode.call_args.kwargs["max_target_tokens"], 8192)
        with patch("fine_tuning.error_note.train_v16.backward_content_block", return_value={"task_signal": True}) as backward:
            self.assertEqual(train_v4.backward_single_block(None, [], self.plan["protocol"]["training_contract"]),
                             {"task_signal": True})
            self.assertIsNone(backward.call_args.kwargs["components"])

    def test_optimizer_needs_task_signal_and_gradient_but_always_schedules(self):
        bundle = SimpleNamespace(model=torch.nn.Linear(1, 1), params={})
        bundle.params = {"L": list(bundle.model.parameters())}
        for task_signal, gradient, expected in ((False, True, 0), (True, False, 0), (True, True, 1)):
            optimizer, scheduler = Mock(), Mock()
            policy = rl.policy_sha256(bundle)
            contract = proto.content_contract_for_block(self.plan["protocol"], 0) | {
                "policy_sha256": policy, "expected_problem_ids": ["p0"]}
            cursor = {"slot": 0, "updates": 0, "consumed_problem_ids": [], "policy_sha256": policy}
            stats = {"task_signal": task_signal, "has_gradient": gradient}
            with patch.object(train_v4, "backward_single_block", return_value=stats):
                update = single.run_joint_single_update(bundle, optimizer, scheduler, [], contract,
                    cursor=cursor, block_ids=["p0"], allow_update=True)
            self.assertEqual(optimizer.step.call_count, expected)
            self.assertEqual(update["optimizer_steps"], expected)
            self.assertEqual(cursor["slot"], 1)
            self.assertEqual(cursor["consumed_problem_ids"], ["p0"])
            scheduler.step.assert_called_once()


    def test_empty_admitted_block_advances_without_parameter_update(self):
        bundle = SimpleNamespace(model=torch.nn.Linear(1, 1), params={})
        bundle.params = {"L": list(bundle.model.parameters())}
        optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        policy = rl.policy_sha256(bundle)
        contract = proto.content_contract_for_block(self.plan["protocol"], 0) | {
            "policy_sha256": policy, "expected_problem_ids": ["p0"],
            "excluded_problem_ids": {"p0": "NO_TASK_SIGNAL"}}
        cursor = {"slot": 0, "updates": 0, "consumed_problem_ids": [], "policy_sha256": policy}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            update = single.run_joint_single_update(bundle, optimizer, scheduler, [], contract,
                cursor=cursor, block_ids=["p0"], allow_update=True)
        self.assertEqual(update["optimizer_steps"], 0)
        self.assertEqual((cursor["slot"], cursor["updates"], scheduler.last_epoch), (1, 0, 1))
        self.assertEqual(policy, rl.policy_sha256(bundle))

    def test_authored_observation_does_not_generate_or_read_gold(self):
        bundle = SimpleNamespace(model=torch.nn.Linear(1, 1), params={})
        bundle.params = {"L": list(bundle.model.parameters())}
        gold, stream = Mock(side_effect=AssertionError("gold must stay private")), Mock()
        rows = [{"schema": "v16-content-row-v1", "problem_id": "p0", "parent_id": "p0", "kind": "O",
                 "layer": "graph", "prompt": "authored observation", "target": "authored annotation"}]
        output = self.root / "block"
        state = single.run_single_block(bundle, actors={"p0": object()}, items=[{"problem_id": "p0"}],
            provenance={}, gold=gold, grade=Mock(), load_images=Mock(), protocol=self.plan["protocol"],
            content_rows=rows, output=output, stream=stream)
        closed = single._read(output / state["rows_path"])
        self.assertEqual(closed["rows"], rows)
        self.assertEqual(closed["contract"]["expected_problem_ids"], ["p0"])
        self.assertEqual(state["status"], "ROWS_READY")
        gold.assert_not_called()
        stream.assert_not_called()

    def test_modified_rows_cannot_bypass_the_pinned_manifest(self):
        candidate = copy.deepcopy(self.plan)
        candidate["content_rows"] = [{"problem_id": "p0", "target": "injected"}]
        candidate["sha256"] = audit.digest({key: value for key, value in candidate.items() if key != "sha256"})
        factory = Mock()
        with patch("fine_tuning.error_note.content_v16.load_content_manifest",
                   return_value={"items": candidate["items"], "rows": []}):
            with self.assertRaisesRegex(ValueError, "admitted content manifest differs"):
                campaign.run_campaign(candidate, output=self.root / "campaign", model_path=self.root,
                    actors={}, provenance={}, gold=Mock(), grade=Mock(), load_images=Mock(), policy_factory=factory)
        factory.assert_not_called()

    def test_cloud_and_phase2_gate_stop_before_policy_factory(self):
        candidate = content_plan(self.manifest, mode="train")
        admitted = {"items": candidate["items"], "rows": []}
        factory = Mock(side_effect=AssertionError("policy must not load"))
        with patch("fine_tuning.error_note.content_v16.load_content_manifest", return_value=admitted):
            with self.assertRaisesRegex(ValueError, "FUNCTIONAL_NOTRUN"):
                campaign.run_campaign(candidate, output=self.root / "campaign", model_path=self.root,
                    actors={}, provenance={}, gold=lambda _: "B", grade=lambda *args: None,
                    load_images=lambda _: [], policy_factory=factory, allow_update=True, allow_save=True)
        factory.assert_not_called()
        descriptor = self.root / "gate.json"
        descriptor.write_text("{}")
        candidate["functional_gate"] = campaign._descriptor(descriptor)
        with patch("fine_tuning.error_note.run_v16.validate_bf16_effect_receipt", return_value={"passed": False}, create=True):
            with self.assertRaisesRegex(ValueError, "EFFECT_STOP"):
                campaign._content_runtime_gate(candidate, campaign._descriptor(descriptor), checkpoint_sha256="e" * 64)

    def test_campaign_sampler_enables_unmerged_lora(self):
        self.plan["capacity"] |= {"engine": "vllm", "kv_cache_gib": 1.0,
                                  "max_num_seqs": 1, "gpu_memory_utilization": .5, "vllm_sleep": True}
        self.plan["sha256"] = audit.digest({k: v for k, v in self.plan.items() if k != "sha256"})
        admitted = {"items": self.plan["items"], "rows": []}
        with patch("fine_tuning.error_note.content_v16.load_content_manifest", return_value=admitted), \
             patch("transformers.GenerationConfig.from_pretrained", return_value=SimpleNamespace(eos_token_id=2)), \
             patch("fine_tuning.error_note.single_vllm.SingleVllmSampler") as sampler:
            with self.assertRaisesRegex(RuntimeError, "authored stop"):
                campaign.run_campaign(self.plan, output=self.root / "campaign", model_path=self.root,
                    actors={}, provenance={}, gold=lambda _: "B", grade=lambda *args: None,
                    load_images=lambda _: [], policy_factory=Mock(side_effect=RuntimeError("authored stop")))
            self.assertIs(sampler.call_args.kwargs["lora"], True)

    def test_actual_counts_define_final_or_stopped_without_pilot(self):
        self.assertEqual(proof_eval._single_stage(1, 8, 2, content_v16=True), "stopped")
        self.assertEqual(proof_eval._single_stage(19, 300, 20, content_v16=True), "stopped")
        self.assertEqual(proof_eval._single_stage(2, 9, 2, content_v16=True), "final")
        with self.assertRaises(ValueError):
            proof_eval._single_stage(0, 0, 2, content_v16=True)


    def test_stage_boundary_changes_weights_without_changing_protocol(self):
        weights = {"O": .25, "J": .45, "C": .20, "N": .10, "A1": .05}
        protocol = proto.plain_protocol_content_v16("bf16", 3,
            contracts=self.plan["protocol"]["training_contract"]["hashes"], source_weights={"graph": 1.0},
            weights=weights, stage_a_blocks=1)
        before = copy.deepcopy(protocol)
        first, second = [proto.content_contract_for_block(protocol, slot) for slot in (0, 1)]
        self.assertEqual(first["active_stage"], "A")
        self.assertEqual(first["weights"], proto.CONTENT_WEIGHTS_V16)
        self.assertEqual(second["active_stage"], "B")
        self.assertEqual(second["weights"], weights)
        self.assertEqual(first["generation_counts"], {"t1": 1, "note": 1, "resolve": 1})
        self.assertEqual(protocol, before)
        with self.assertRaisesRegex(ValueError, "block outside"):
            proto.content_contract_for_block(protocol, 3)


    def test_stage_a_collects_only_authored_rows_and_stage_b_remains_gated(self):
        protocol = proto.plain_protocol_content_v16("bf16", 2,
            contracts=self.plan["protocol"]["training_contract"]["hashes"], source_weights={"graph": 1.0},
            weights={"O": .25, "J": 0.0, "C": .20, "N": .10, "A1": .05}, stage_a_blocks=1)
        bundle = SimpleNamespace(model=torch.nn.Linear(1, 1), params={})
        bundle.params = {"L": list(bundle.model.parameters())}
        stream, gold = Mock(), Mock()
        kwargs = {"actors": {"p0": object()}, "items": [{"problem_id": "p0"}], "provenance": {},
                  "gold": gold, "grade": Mock(), "load_images": Mock(), "protocol": protocol,
                  "content_rows": [{"kind": "O", "problem_id": "p0", "layer": "graph"}], "stream": stream}
        state = single.run_single_block(bundle, **kwargs, block_id=0, output=self.root / "stage-a")
        closed = single._read(self.root / "stage-a" / state["rows_path"])
        self.assertEqual(closed["contract"]["active_stage"], "A")
        self.assertEqual(closed["contract"]["weights"], proto.CONTENT_WEIGHTS_V16)
        stream.assert_not_called()
        gold.assert_not_called()
        with self.assertRaisesRegex(ValueError, "EFFECT_NOTRUN_OR_STOP"):
            single.run_single_block(bundle, **kwargs, block_id=1, output=self.root / "stage-b")
        stream.assert_not_called()


    def test_first_note_yield_denominator_includes_all_shared_wrong_statuses(self):
        states = ["INCORRECT", "TRUNCATED", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "CORRECT", "UNVERIFIABLE"]
        ids = [f"p{i}" for i in range(len(states))]
        items = [{"problem_id": pid, "parent_id": pid, "annotation": {}} for pid in ids]
        actors = {pid: SimpleNamespace(question=pid + "?", choices=("one", "two"), image_paths=("image.png",),
                                      image_sha256=("a" * 64,), source="authored", mode="plain") for pid in ids}
        rows = [{"kind": "O", "problem_id": pid, "layer": "graph"} for pid in ids]
        bundle = SimpleNamespace(model=torch.nn.Linear(1, 1), params={}, processor=SimpleNamespace(tokenizer=Mock()))
        bundle.params = {"L": list(bundle.model.parameters())}
        protocol = proto.plain_protocol_content_v16("bf16", 1,
            contracts=self.plan["protocol"]["training_contract"]["hashes"], source_weights={"graph": 1.0},
            weights={"O": 1.0, "J": 0.0, "C": .2, "N": .1, "A1": 0.0})
        gold = Mock(return_value="A")

        def generate(requests, stream, stats, output, **kwargs):
            gold.assert_not_called()
            self.assertEqual([request["stage"] for request in requests], ["t1"] * len(ids))
            return {request["request_id"]: {
                "text": "Original reasoning.", "finish_reason": "stop", "prompt_tokens": 1,
                "completion_tokens": 1, "completion_count": 1,
                "generation": {"text": "Original reasoning.", "completion_token_ids": [1], "finish_reason": "stop",
                               "completion_count": 1,
                               "policy_sha256": rl.policy_sha256(bundle), "protocol_sha256": protocol["sha256"]}}
                    for request in requests}

        with patch("fine_tuning.error_note.run_v16.validate_bf16_effect_receipt", return_value={"passed": True}), \
             patch("fine_tuning.error_note.self_note_v16.build_problem_contract", return_value={"supported": False}), \
             patch.object(single, "run_single_requests", side_effect=generate):
            state = single.run_single_block(bundle, actors=actors, items=items,
                provenance={pid: items[index] for index, pid in enumerate(ids)}, gold=gold,
                grade=lambda pid, *args: (states[ids.index(pid)], None), load_images=lambda actor: [],
                protocol=protocol, content_rows=rows, effect_gate={}, output=self.root / "yield", stream=Mock())
        self.assertEqual(state["note_yield_gate"]["wrong_t1_count"], 4)
        self.assertEqual(state["note_yield_gate"]["verified_note_count"], 0)
        self.assertEqual([record["t1"]["status"] for record in state["note_yield_records"]], states[:4])


    def test_effect_sentinel_stop_at_eight_blocks_holds_on_resume(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, old_threads)
        base, _, _, _ = fixture(self.root)
        candidate = content_plan(self.manifest, mode="train", slots=9)
        candidate["content_rows"] = [{"problem_id": f"p{i}", "row_id": f"O:p{i}", "kind": "O"}
                                     for i in range(9)]
        candidate["sha256"] = audit.digest({k: v for k, v in candidate.items() if k != "sha256"})
        admitted = {"items": candidate["items"], "rows": candidate["content_rows"]}
        registration = self.root / "effect-prereg.json"
        registration.write_text("{}")
        observed = []

        def effect(bundle, pre, manifest, manifest_path, *, stage, output, checkpoint):
            observed.append(stage)
            if stage == "8":
                self.assertIn("file_sha256", checkpoint)
                self.assertNotIn("sha256", checkpoint)
            return {"stage": stage, "stop_reasons": ["SENTINEL_COLLAPSE"] if stage == "8" else []}

        def backward(bundle, encoded, contract, *args, **kwargs):
            sum(parameters[0].sum() * 1e-6 for parameters in bundle.params.values()).backward()
            return {"loss": 0.0, "task_signal": True, "has_gradient": True}

        kwargs = {"output": self.root / "campaign", "model_path": base,
                  "actors": {f"p{i}": object() for i in range(9)}, "provenance": {},
                  "gold": Mock(), "grade": Mock(), "load_images": Mock(), "policy_factory": fresh_tiny,
                  "encoder": lambda bundle, row: row, "max_blocks": 9, "allow_update": True, "allow_save": True,
                  "effect_prereg": campaign._descriptor(registration), "effect_evaluator": effect}
        with patch("fine_tuning.error_note.content_v16.load_content_manifest", return_value=admitted), \
             patch.object(campaign, "_content_functional_gate", return_value={"ready": True}), \
             patch.object(train_v4, "backward_single_block", side_effect=backward):
            result = campaign.run_campaign(candidate, **kwargs)
            self.assertEqual((result["status"], result["cursor"]["slot"]), ("CONTENT_EFFECT_STOP", 8))
            self.assertFalse((self.root / "campaign/block-8").exists())
            repeated = campaign.run_campaign(candidate, **kwargs)
        self.assertEqual((repeated["status"], repeated["cursor"]["slot"]), ("CONTENT_EFFECT_STOP", 8))
        self.assertEqual(observed, ["0", "8"])
        omitted = dict(kwargs)
        omitted.pop("effect_prereg")
        with patch("fine_tuning.error_note.content_v16.load_content_manifest", return_value=admitted), \
             patch.object(campaign, "_content_functional_gate", return_value={"ready": True}):
            with self.assertRaisesRegex(ValueError, "same preregistration"):
                campaign.run_campaign(candidate, **omitted)

    def test_plan_has_one_actual_campaign_phase(self):
        p = {"version": planner.PARAMS_VERSION_CONTENT_V16, "commit": "a" * 40,
             "python": "/venv/bin/python", "repo_root": "/repo", "model_path": "/models/" + planner.BASE_REVISION,
             "out_root": "/private/output", "content_manifest_path": "/private/content.json",
             "content_manifest_sha256": "b" * 64, "functional_gate_path": "/private/l0.json", "functional_gate_sha256": "c" * 64,
             "total_blocks": 3, "block_size": 8, "timeouts": {key: 1 for key in ("cpu", "advance", "reload", "export")}}
        steps = planner.plan("content-v16", p)["steps"]
        self.assertEqual([step["name"] for step in steps][-3:], ["advance-content", "reload-content", "export-content"])
        self.assertIn("checkpoint-3/metadata.json", steps[-3]["expect_files"][0])
        self.assertFalse(any("pilot" in step["name"] for step in steps))

    def test_protocol_to_real_tiny_unmerged_export(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, old_threads)
        base, base_manifest, _, _ = fixture(self.root)
        candidate = content_plan(self.manifest, mode="train", slots=1)
        protocol = candidate["protocol"]
        campaign._plan_check(candidate)
        out = self.root / "campaign"
        out.mkdir()
        single._sealed(out / "campaign-plan.json", candidate)
        bundle = fresh_tiny(base, proto.ARM, protocol)
        optimizer, scheduler = checkpoint.make_optimizer(bundle, protocol)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scheduler.step()
        cursor = {"slot": 1, "updates": 0, "consumed_problem_ids": ["p0"],
                  "policy_sha256": rl.policy_sha256(bundle), "last_transaction_sha256": None, "parent_checkpoint": None}
        checkpoint_path = out / "checkpoint-1"
        single.save_single_continuation(bundle, optimizer, scheduler, cursor=cursor, path=checkpoint_path,
            protocol=protocol, binding=campaign._binding(candidate), allow_save=True)
        ref = proof_eval.export_for_eval(checkpoint_path, out / "campaign-plan.json", base, self.root / "export", "f" * 40,
            _test_policy_factory=fresh_tiny, _test_base_manifest=base_manifest)
        self.assertEqual(ref["kind"], "unmerged-lora")
        self.assertEqual(ref["export"]["campaign"]["stage"], "final")
        self.assertEqual(ref["export"]["campaign"]["updates"], 0)
        self.assertEqual(ref["llm_lora"], proof_eval.lora_section())
        proof_eval._single_model_ref(self.root / "export/model_ref.json", lora=True)


if __name__ == "__main__":
    unittest.main()
