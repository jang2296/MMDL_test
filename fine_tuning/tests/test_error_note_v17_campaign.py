"""CPU caller-graph checks; generated fixtures are not real model or repair evidence."""
from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fine_tuning.error_note import campaign_v16b as campaign, prep_v16b as prep, v17_rows
from fine_tuning.error_note.collect import single_request_sha256
from fine_tuning.error_note.v16b_runtime import reserve_request_v17
from fine_tuning.tests.test_error_note_v16b_campaign import arguments
from fine_tuning.tests.test_error_note_v16b_prep import item
from fine_tuning.tests.test_error_note_v17_rows import CATALOG, TOKENIZER, output, verifier


class CpuEngine:
    """Deterministic synthetic completions, with actual waiting reservations and byte tokens."""

    max_num_seqs = 2

    def __init__(self, bad_note=False):
        self.pending, self.registry, self.admissions = [], {}, []
        self.engine_steps, self.bad_note = 0, bad_note
        self.preflight_stage = "f0"
        self.capacity_waits = 0
        self.max_pending = 0

    def live_registry(self):
        return copy.deepcopy(self.registry)

    def add_version(self, bundle, version):
        if str(version) in self.registry:  # the real sampler refuses an already registered version
            raise RuntimeError("v16b vLLM: registry, fresh version and visual-prefill barrier required")
        self.release_idle_versions()
        self.registry[str(version)] = {"policy_sha256": "a" * 64, "refcount": 0}

    def release_idle_versions(self):
        self.registry = {v: e for v, e in self.registry.items() if e["refcount"]}

    def set_preflight_stage_v17(self, stage):
        assert not self.pending
        self.preflight_stage = stage

    def prepare_request_v17(self, request):
        return {"prompt_tokens": 1, "max_tokens": request["max_tokens"],
                "request_sha256": single_request_sha256(request)}

    def can_submit_v17(self, prepared):
        reservations = [r["max_tokens"] + 1 for r in self.pending]
        fits = reserve_request_v17(prepared, 60000, 1, reservations, active=len(reservations),
                                   ceiling=self.max_num_seqs)["status"] == "ADMIT"
        self.capacity_waits += int(not fits)
        return fits

    def submit(self, request, *, prepared=None):
        if prepared is not None:
            assert prepared["request_sha256"] == single_request_sha256(request)
            assert self.can_submit_v17(prepared)
        self.pending.append(request)
        self.max_pending = max(self.max_pending, len(self.pending))
        self.registry[str(request["policy_version"])]["refcount"] += 1
        self.admissions.append((request["problem_id"], request["kind"], request["policy_version"]))

    def has_unfinished_requests(self):
        return bool(self.pending)

    def prefill_barrier(self):
        return True

    def release(self):
        assert not self.pending

    def step(self):
        request = self.pending.pop(0)
        self.engine_steps += 1
        pid, kind = request["problem_id"], request["kind"]
        correct = not pid.startswith("train") or int(pid.removeprefix("train")) % 2 == 0
        if kind == "T1":
            text = ("AB = 7\nUse the printed length in the calculation.\nFinal answer: A" if correct else
                    "AB = 9\nUse the printed length in the calculation.\nFinal answer: B")
        elif kind == "OBS":
            text = json.dumps({"claims": [{"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": "7"}]})
        elif kind == "NOTE":
            text = json.dumps({"repairs": [{"claim_id": "t0", "corrected_value": "8" if self.bad_note else "7"}],
                               "lesson": "Inspect each printed label before computing."})
        else:
            text = "AB = 7\nFinal answer: A"
        result = output(text)
        generation = result["generation"]
        generation.update({k: request[k] for k in ("policy_sha256", "policy_version", "request_sha256",
                                                   "protocol_sha256", "seed", "kind")})
        generation.update(mixed_version=False, b2_tail_flags=[False] * len(generation["completion_token_ids"]))
        for key in ("baseline_receipt", "obs_baseline_receipt"):
            if key in request:
                generation[key] = request[key]
        self.registry[str(request["policy_version"])]["refcount"] -= 1
        return [(request, result)]


def fixture(root: Path, *, smoke=True):
    args = arguments(root, smoke)
    args.version = "error-note-v17"
    args.allow_model_run = args.allow_update = args.allow_save = True
    args.data_root.mkdir()
    for name in ("payload-receipt.json", "order.json"):
        (args.data_root / name).write_text("{}\n")
    native = [item(f"train{i}") | {"layer": "geometry"} for i in range(3000)]
    native[20]["layer"], native[30]["layer"] = "graph", "scene"
    native += [item("prep0"), item("monitor0"), item("effect0")]
    selection = prep.seal(args.prep_root / "selection.json", {"prep_ids": ["prep0"], "monitor_ids": ["monitor0"]})
    def f0_record(pid):
        return {"problem_id": pid, "output": output("AB = 7\nFinal answer: A")}
    generated = prep.seal(args.prep_root / "f0-generations.json", {
        "monitor": [f0_record("monitor0")], "effect": [f0_record("effect0")],
        "monitor_summary": {"median_tokens": 23}})
    row = {"row_id": "prep0:PREP_HUMAN", "kind": "PREP_HUMAN", "problem_id": "prep0"}
    rows = prep.seal(args.prep_root / "prep-rows.json", {
        "selection_sha256": selection["sha256"], "f0_generations_sha256": generated["sha256"],
        "rows": [row], "batches": [[row]], "prep_updates": 1, "local_sequential": smoke})
    prep.seal(args.prep_root / "baseline-init.json", {"prep_rows_sha256": rows["sha256"], "overall": .5})
    def load(root, split):
        return [p for p in native if p["problem_id"].startswith(split)]
    def preflight(items, **kwargs):
        return {"items": [{"problem_id": p["problem_id"], "repair_supported": True,
                            "observation_eligibility": eligibility(p)} for p in items]}
    def eligibility(parent):
        return {"problem_id": parent["problem_id"], "eligible": True, "reason": None}
    patches = [patch("fine_tuning.error_note.v17_data.validate", return_value={"effect_eligible": True, "effect_status": "CPU_FIXTURE"}),
               patch("fine_tuning.error_note.v17_data.load_items", side_effect=load),
               patch.object(prep, "select", return_value=selection),
               patch.object(campaign, "preflight_items", side_effect=preflight),
               patch.object(campaign.single, "source_manifest", return_value={"actual_files": {"fixture": "c" * 64}}),
               patch.object(v17_rows, "observation_eligibility_v17", side_effect=eligibility)]
    return args, native, patches


class CampaignIntegration(unittest.TestCase):
    def test_public_campaign_dispatch_keeps_explicit_v17_namespace(self):
        from fine_tuning.error_note import single_campaign

        for argv in (["prepare", "--version", "error-note-v17"],
                     ["advance", "--version=error-note-v17"]):
            with self.subTest(argv=argv), patch.object(campaign, "main", return_value=17) as route:
                self.assertEqual(single_campaign.main(argv), 17)
                route.assert_called_once_with(argv)

    def test_v17_collection_reserves_native_waiting_requests_and_reuses_durable_results(self):
        from fine_tuning.error_note.protocol import EVAL_SAMPLING_V16B

        engine = CpuEngine()
        engine.max_num_seqs = 16
        parents = [item(f"monitor{i}") for i in range(13)]
        def grade(parent, result):
            return "CORRECT", "A"
        with tempfile.TemporaryDirectory() as tmp, patch.object(prep, "images", return_value=[]), patch.object(
                prep, "grader", return_value=grade), patch(
                "fine_tuning.error_note.rl.policy_sha256", return_value="a" * 64):
            kwargs = dict(data_root=Path(tmp), split="monitor", out=Path(tmp) / "monitor",
                          stage="v17-monitor", protocol_sha256="d" * 64,
                          sampling=EVAL_SAMPLING_V16B, cap=8192, algorithm_version="error-note-v17")
            first = prep.collect(None, engine, parents, **kwargs)
            admissions = len(engine.admissions)
            second = prep.collect(None, engine, parents, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(admissions, len(parents))
        self.assertEqual(len(engine.admissions), admissions)
        self.assertGreater(engine.capacity_waits, 0)
        self.assertEqual(engine.max_pending, 6)
        self.assertFalse(engine.pending)
        self.assertEqual([r["problem_id"] for r in first], [p["problem_id"] for p in parents])

    def run_fixture(self, root, *, smoke=True, bad_note=False, funding_resume=False, gate_crash=None):
        args, native, patches = fixture(root, smoke=smoke)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            plan = campaign.prepare(args)
            model = torch.nn.Linear(1, 1, bias=False)
            bundle = SimpleNamespace(model=model, processor=SimpleNamespace(tokenizer=TOKENIZER),
                                     params={"L": list(model.parameters()), "M": [], "V": []},
                                     route=SimpleNamespace(apply=lambda _: None))
            opt = torch.optim.AdamW(model.parameters(), lr=.01)
            lr = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
            engine, calls = CpuEngine(bad_note), []
            def grade(parent, result):
                return ("INCORRECT", "B") if result["text"].endswith("B") else ("CORRECT", "A")
            def backward(policy, encoded, contract, token_budget, capacity=None):
                calls.append((contract["phase"], contract["global_slot"], copy.deepcopy(encoded), copy.deepcopy(contract)))
                if funding_resume and contract["global_slot"] == 2:
                    (args.output / "STOP_REQUESTED").write_text("CPU fixture funding stop")
                if encoded:
                    model.weight.grad = torch.ones_like(model.weight)
                nrows = [r for r in encoded if r["kind"] == "N"]
                return {"has_gradient": bool(encoded), "task_signal": bool(encoded),
                        "verified_repair_parent_ids": [r["problem_id"] for r in nrows],
                        "n_verified_content_tokens": sum(sum(r["verified_content_token_mask"]) for r in nrows),
                        "n_content_gradient_finite_nonzero": bool(nrows), "note_gradient_retained": bool(nrows)}
            def save(bundle, optimizer, scheduler, *, cursor, path, protocol, **kwargs):
                path.mkdir()
                (path / "metadata.json").write_text(json.dumps({"protocol": protocol, "cursor": {"single": cursor}, "step": cursor["slot"]}))
            def probe(bundle, rows, path):
                (path / "probe.json").write_text("{}")
                (path / "probe_inputs.safetensors").write_bytes(b"fixture")
            def contract(parent):
                return {"supported": True, "verifier": verifier(parent), "fact_catalog": CATALOG, "rules": {}}
            more = [patch.object(campaign, "_factory", return_value=lambda *a: bundle),
                    patch.object(campaign, "make_sampler", return_value=engine),
                    patch.object(campaign, "_encode", side_effect=lambda bundle, rows, items, image: rows),
                    patch.object(campaign, "images", return_value=[]),
                    patch.object(prep, "images", return_value=[]),
                    patch.object(campaign, "grader", return_value=grade), patch.object(prep, "grader", return_value=grade),
                    patch.object(campaign.single, "save_single_continuation", side_effect=save),
                    patch.object(campaign.single, "load_single_continuation", side_effect=lambda *a, **k:
                        (bundle, opt, lr, copy.deepcopy(prep.read_sealed(args.output / "campaign-state.json")["cursor"]))),
                    patch("fine_tuning.error_note.rl.policy_sha256", return_value="a" * 64),
                    patch.object(campaign, "prompt_tokens", return_value=100),
                    patch("fine_tuning.error_note.run_v16._setup_execution"),
                    patch("fine_tuning.notebook_rl.checkpoint.make_optimizer", return_value=(opt, lr)),
                    patch("fine_tuning.error_note.train._write_probe", side_effect=probe),
                    patch("fine_tuning.error_note.train_v4.backward_single_block", side_effect=backward),
                    patch.object(v17_rows, "problem_contract_v17", side_effect=contract),
                    patch.object(model, "to", return_value=model)]
            for p in more:
                stack.enter_context(p)
            if not smoke:
                stack.enter_context(patch("fine_tuning.error_note.v17_preflight.bind_execution",
                                          return_value={"status": "CPU_FIXTURE_NOT_PRODUCTION_ADMITTED"}))
                prep.seal(args.output / "capacity.json", {"shape_bounds": {}})
            crash_state, resumed = [], []
            actual_write, actual_collect = campaign.write_json, campaign.collect
            from fine_tuning.error_note import v16b_runtime
            actual_run = v16b_runtime.run_v16b
            class GateResumeVerified(Exception):
                pass
            def write_with_crash(path, value):
                actual_write(path, value)
                if (gate_crash == "before-evaluate" and not crash_state
                        and Path(path) == args.output / "campaign-state.json"
                        and value.get("runtime") is not None
                        and (value.get("signal_gate") or {}).get("status") == "WARMUP"
                        and value["cursor"]["slot"] == plan["protocol"]["training_contract"]["prep_updates"] + plan["signal_gate"]["due_slot"] + 1):
                    crash_state.append(copy.deepcopy(value))
                    raise InterruptedError("CPU gate checkpoint crash")
            def collect_with_crash(*a, **kwargs):
                result = actual_collect(*a, **kwargs)
                if gate_crash in ("during-monitor-fail", "during-monitor-pass") and not crash_state and Path(kwargs["out"]).name == "monitor-signal-gate":
                    durable = prep.read_sealed(args.output / "campaign-state.json")
                    self.assertTrue(durable["signal_gate_monitor_pending"])
                    self.assertEqual(durable["signal_gate"]["status"], "PASS" if gate_crash.endswith("pass") else "FAIL")
                    metadata = json.loads(Path(durable["checkpoint"]["path"]).read_text())
                    self.assertEqual(metadata["cursor"]["single"], durable["cursor"])
                    crash_state.append(durable)
                    raise InterruptedError("CPU gate monitor crash")
                return result
            def run_after_resolution(*a, **kwargs):
                if resumed and gate_crash == "during-monitor-pass":
                    durable = prep.read_sealed(args.output / "campaign-state.json")
                    self.assertEqual(durable["signal_gate"]["status"], "PASS")
                    self.assertFalse(durable["signal_gate_monitor_pending"])
                    self.assertEqual(len(calls), resumed[0][0])
                    self.assertEqual([entry for entry in engine.admissions if entry[0].startswith("train")], resumed[0][1])
                    # run_v16b registers the next version at once: the gate monitor must not leave it registered.
                    self.assertEqual({v: e for v, e in engine.registry.items() if not e["refcount"]}, {})
                    raise GateResumeVerified()
                return actual_run(*a, **kwargs)
            if gate_crash:
                stack.enter_context(patch.object(campaign, "write_json", side_effect=write_with_crash))
                stack.enter_context(patch.object(campaign, "collect", side_effect=collect_with_crash))
                stack.enter_context(patch.object(v16b_runtime, "run_v16b", side_effect=run_after_resolution))
                with self.assertRaisesRegex(InterruptedError, "CPU gate"):
                    campaign.run_campaign(plan, output=args.output, args=args)
                self.assertEqual(len(crash_state), 1)
                self.assertEqual(crash_state[0]["cursor"]["slot"], 6)
                self.assertFalse(engine.pending)
                resumed.append((len(calls), [entry for entry in engine.admissions if entry[0].startswith("train")]))
                if gate_crash == "during-monitor-pass":
                    with self.assertRaises(GateResumeVerified):
                        campaign.run_campaign(plan, output=args.output, args=args)
                    result = prep.read_sealed(args.output / "campaign-state.json")
                    return args, plan, result, calls, engine
                result = campaign.run_campaign(plan, output=args.output, args=args)
                self.assertEqual(len(calls), resumed[0][0])
                self.assertEqual([entry for entry in engine.admissions if entry[0].startswith("train")], resumed[0][1])
                self.assertEqual(result["signal_gate"]["status"], "FAIL")
                metadata = json.loads(Path(result["checkpoint"]["path"]).read_text())
                self.assertEqual(metadata["cursor"]["single"], result["cursor"])
            else:
                result = campaign.run_campaign(plan, output=args.output, args=args)
            if funding_resume:
                self.assertEqual(result["status"], "STOPPED")
                self.assertEqual(result["cursor"]["slot"], 3)
                saved = json.loads(Path(result["runtime"]["path"]).read_text())["state"]
                self.assertTrue(saved["pending"])
                self.assertEqual(saved["version"], 2)
                (args.output / "STOP_REQUESTED").rename(args.output / "STOP_REQUESTED.archived")
                args.resume_funding_stop = True
                result = campaign.run_campaign(plan, output=args.output, args=args)
                args.resume_funding_stop = False
            replay_calls, replay_admissions = len(calls), len(engine.admissions)
            again = campaign.run_campaign(plan, output=args.output, args=args)
            self.assertEqual((len(calls), len(engine.admissions)), (replay_calls, replay_admissions))
            self.assertEqual(again["sha256"], result["sha256"])
            return args, plan, result, calls, engine

    def test_local_full_caller_graph_consumes_a1_n_c_and_three_flushes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, plan, result, calls, engine = self.run_fixture(Path(tmp))
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual([slot for _, slot, _, _ in calls], list(range(8)))
            self.assertEqual(result["cursor"]["slot"], plan["total_slots"])
            kinds = {r["kind"] for _, _, rows, _ in calls for r in rows}
            self.assertTrue({"RL", "A1", "N", "C", "PREP_HUMAN"} <= kinds)
            self.assertEqual(result["signal_gate"]["status"], "LOCAL_SMOKE_UNVERIFIED")
            self.assertEqual(result["guard"]["version"], "error-note-v17-guard-v1")
            self.assertEqual(engine.preflight_stage, "trained")
            self.assertTrue((args.output / "evaluation-admission.json").exists())
            self.assertFalse(engine.pending)
            for _, _, rows, contract in calls[1:]:
                for r in rows:
                    self.assertEqual(r["schema"], "error-note-v17-row-v1")
                    if r["kind"] == "A1":
                        self.assertIn(r["problem_id"], contract["expected_problem_ids"])
                    elif r["kind"] == "N":
                        self.assertIn(r["problem_id"], contract["note_problem_ids"])
                    elif r["kind"] == "C":
                        self.assertIn(r["problem_id"], contract["resolve_problem_ids"])

    def test_checkpoint_resume_keeps_consumed_rows_guard_windows_and_future_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, plan, result, calls, engine = self.run_fixture(Path(tmp), funding_resume=True)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual([slot for _, slot, _, _ in calls], list(range(8)))
            self.assertEqual(len(result["guard"]["history"]), len(plan["blocks"]))
            admitted = [(pid, kind) for pid, kind, _ in engine.admissions if pid.startswith("train")]
            self.assertEqual(len(admitted), len(set(admitted)))
            self.assertEqual(len(result["cursor"]["consumed_problem_ids"]), 8)
            self.assertEqual(result["signal_gate"]["status"], "LOCAL_SMOKE_UNVERIFIED")

    def test_final_checkpoint_bindings_and_stage_reports_preserve_actual_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, plan, result, calls, _ = self.run_fixture(Path(tmp))
            ready = campaign.require_complete(args.output, check_evaluation=False)
            self.assertEqual(ready["metadata_descriptor"], result["checkpoint"])
            self.assertEqual(ready["plan_sha256"], plan["sha256"])
            reports = [prep.read_sealed(path) for path in sorted((args.output / "signal-reports").glob("slot-*.json"))]
            self.assertEqual(len(reports), 7)
            for sealed in reports:
                phase, slot, rows, contract = next(c for c in calls if c[1] == sealed["global_slot"])
                self.assertEqual(phase, "rl")
                self.assertEqual(sealed["plan_sha256"], plan["sha256"])
                report = sealed["report"]
                self.assertEqual(report["parents_planned"], len(contract["expected_problem_ids"]))
                self.assertEqual(report["row_counts"], {kind: sum(r["kind"] == kind for r in rows)
                                                         for kind in {r["kind"] for r in rows}})
                self.assertEqual(report["stages"]["T1"]["completed"], len(contract["expected_problem_ids"]))
                for stage in ("T1", "OBS", "NOTE", "RESOLVE"):
                    self.assertEqual(sum(source["stages"][stage]["completed"] for source in report["sources"].values()),
                                     report["stages"][stage]["completed"])
                self.assertEqual(sum(source["stages"]["T1"]["length"]["n"] for source in report["sources"].values()),
                                 report["stages"]["T1"]["completed"])
                self.assertEqual(report["n_verified_content_tokens"],
                                 sum(sum(r.get("verified_content_token_mask", [])) for r in rows if r["kind"] == "N"))
            self.assertEqual(sum(r["report"]["stages"]["T1"]["completed"] for r in reports), 8)
            self.assertGreater(sum(r["report"]["stages"]["NOTE"]["verified_repairs"] for r in reports), 0)
            final_path = args.output / "monitor-final.json"
            final = prep.read_sealed(final_path)
            final["checkpoint"] = final["checkpoint"] | {"file_sha256": "0" * 64}
            final.pop("sha256")
            campaign.write_json(final_path, final | {"sha256": campaign.digest(final)})
            with self.assertRaisesRegex(ValueError, "final monitor/checkpoint binding"):
                campaign.require_complete(args.output, check_evaluation=False)

    def test_crash_after_due_checkpoint_resolves_gate_before_any_resumed_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, result, calls, _ = self.run_fixture(Path(tmp), smoke=False, bad_note=True,
                                                       gate_crash="before-evaluate")
            self.assertEqual(result["status"], "STOPPED")
            self.assertEqual(result["stop_reasons"], ["SIGNAL_GATE_FAIL"])
            self.assertEqual([slot for _, slot, _, _ in calls], list(range(6)))
            self.assertTrue((args.output / "signal-gate.json").exists())

    def test_crash_during_failed_gate_monitor_keeps_stopped_checkpoint_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, result, calls, _ = self.run_fixture(Path(tmp), smoke=False, bad_note=True,
                                                     gate_crash="during-monitor-fail")
            self.assertEqual(result["status"], "STOPPED")
            self.assertEqual(result["stop_reasons"], ["SIGNAL_GATE_FAIL"])
            self.assertEqual([slot for _, slot, _, _ in calls], list(range(6)))
            self.assertTrue(result["signal_gate_monitor_pending"])

    def test_crash_during_passed_gate_monitor_resolves_pending_monitor_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, result, calls, _ = self.run_fixture(Path(tmp), smoke=False, bad_note=False,
                                                       gate_crash="during-monitor-pass")
            self.assertEqual(result["status"], "READY")
            self.assertEqual(result["signal_gate"]["status"], "PASS")
            self.assertFalse(result["signal_gate_monitor_pending"])
            self.assertEqual([slot for _, slot, _, _ in calls], list(range(6)))
            self.assertTrue((args.output / "signal-gate.json").exists())

    def test_production_gate_checkpoints_at_fixed_closure_and_stays_stopped_on_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, plan, result, calls, engine = self.run_fixture(Path(tmp), smoke=False, bad_note=True)
            self.assertEqual(plan["signal_gate"]["due_slot"], 4)
            self.assertEqual(result["status"], "STOPPED")
            self.assertEqual(result["stop_reasons"], ["SIGNAL_GATE_FAIL"])
            self.assertEqual(result["signal_gate"]["status"], "FAIL")
            self.assertEqual([slot for _, slot, _, _ in calls], list(range(6)))
            self.assertTrue((args.output / "monitor-signal-gate.json").exists())
            self.assertTrue((args.output / "signal-gate.json").exists())
            self.assertFalse((args.output / "export-ready.json").exists())
            self.assertFalse(any(version > 4 for pid, _, version in engine.admissions if pid.startswith("train")))
            envelope = json.loads(Path(result["runtime"]["path"]).read_text())
            self.assertEqual(set(envelope["state"]["requests"]), set(envelope["state"]["closed"]))


if __name__ == "__main__":
    unittest.main()
