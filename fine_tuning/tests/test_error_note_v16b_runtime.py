"""CPU evidence for fixed admission, immutable policies and v16b runtime barriers."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from fine_tuning.error_note import llm_lora
from fine_tuning.error_note.collect import _SingleActionSampler
from fine_tuning.error_note.single_vllm import SingleVllmSampler, b2_tail_flag, sampled_logprob
from fine_tuning.error_note.v16b_runtime import RuntimeCallbacks, V16bScheduler, run_v16b


SAMPLING = {"temperature": .7, "top_p": 1., "top_k": 0, "presence_penalty": 1.5}


class FakeSampler:
    """One completed request per step, optionally reverse completion order."""

    def __init__(self, reverse=False):
        self.versions, self.pending, self.admissions = {}, [], []
        self.engine_steps = 0
        self.reverse, self.barrier = reverse, True
        self.saved_live = []

    def add_version(self, bundle, version):
        for old, entry in list(self.versions.items()):
            if not entry["refcount"]:
                del self.versions[old]
        if len(self.versions) >= 2:
            raise RuntimeError("more than two live versions")
        sha = f"{version:064x}"
        self.versions[version] = {"policy_sha256": sha, "refcount": 0}
        self.saved_live.append(sorted(self.versions))
        return sha

    def live_registry(self):
        return {str(v): dict(entry) for v, entry in self.versions.items()}

    def submit(self, request):
        self.pending.append(request)
        self.versions[request["policy_version"]]["refcount"] += 1
        self.admissions.append((request["policy_version"], request["block_id"], request["kind"]))

    def has_unfinished_requests(self):
        return bool(self.pending)

    def prefill_barrier(self):
        return self.barrier

    def step(self):
        if not self.pending:
            return []
        request = self.pending.pop(-1 if self.reverse else 0)
        self.engine_steps += 1
        self.versions[request["policy_version"]]["refcount"] -= 1
        generation = {k: request[k] for k in ("request_id", "policy_sha256", "request_sha256",
                                              "policy_version", "seed", "kind")}
        generation.update(completion_token_ids=[4, 9], prompt_token_ids=[1], sampling_logprobs=[-.3, -.2],
                          visual_version=request["policy_version"], mixed_version=False, preempted=False)
        return [(request, {"generation": generation, "text": "answer", "finish_reason": "stop",
                           "completion_tokens": 2})]


def fixture(count=4, reverse=False, seed_offset=0):
    sampler = FakeSampler(reverse)
    items = [{"problem_id": f"p{i}", "source": "source", "layer": "scene"} for i in range(count)]
    plan = {"blocks": [[item["problem_id"]] for item in items]}
    baselines, checks, saved = [], [], []

    def make_request(item, kind, version, record):
        return {"request_id": f"{item['problem_id']}:{kind}", "stage": "v16b-" + kind.lower(),
                "seed": int(item["problem_id"][1:]) + seed_offset, "max_tokens": 8,
                "messages": [{"role": "user", "content": [{"type": "text", "text": kind}]}],
                "images": [], "sampling": SAMPLING}

    def baseline(item, kind, history, cutoff):
        baselines.append((item["problem_id"], kind, cutoff, copy.deepcopy(history)))
        return {"baseline": .25, "history_cutoff": cutoff, "history_sha256": "a" * 64}

    def complete(item, kind, output, record):
        return {"reward": 1.0} if kind == "T1" else {"obs_score": {"score": .5}} if kind == "OBS" else {}

    def rows(kind, item, record):
        output = record.get(kind.lower())
        return [] if output is None else [{"problem_id": item["problem_id"], "kind": kind,
                                           "generation": output["generation"]}]

    def update(v, rows, contract):
        checks.append((v, copy.deepcopy(rows), contract, torch.are_deterministic_algorithms_enabled()))
        return {"optimizer_step": True, "task_signal": True}

    def checkpoint(v, state, pending):
        saved.append((v, copy.deepcopy(state), pending, sampler.has_unfinished_requests()))
        return {}

    callbacks = RuntimeCallbacks(make_request, rows, complete, baseline, update, checkpoint)
    return sampler, items, plan, callbacks, baselines, checks, saved


class RuntimeTests(unittest.TestCase):
    def test_host_timing_includes_drain_and_checkpoint_and_distinguishes_overhead(self):
        sampler, items, plan, callbacks, *_ = fixture(count=2)
        clock = SimpleNamespace(now=0.0)
        def measured(callback, seconds):
            def call(*args, **kwargs):
                result = callback(*args, **kwargs)
                clock.now += seconds
                return result
            return call
        sampler.step = measured(sampler.step, 2.0)
        callbacks.make_request = measured(callbacks.make_request, .25)
        callbacks.complete = measured(callbacks.complete, .5)
        callbacks.update = measured(callbacks.update, 3.0)
        checkpoint = callbacks.checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def save(v, state, pending):
                self.assertFalse((root / f"update-timing-{v}.json").exists())
                checkpoint(v, state, pending)
                clock.now += 5.0
            callbacks.checkpoint = save
            with patch("fine_tuning.error_note.v16b_runtime.time.perf_counter", side_effect=lambda: clock.now):
                result = run_v16b(None, Mock(), None, sampler, plan, out=root, protocol={"prep_updates": 7},
                                  items=items, callbacks=callbacks)
            reports = [json.loads((root / f"update-timing-{v}.json").read_text()) for v in range(5)]
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(reports[0]["timing"]["drain_generation_seconds"], 4.0)
        self.assertEqual([r["timing"]["checkpoint_seconds"] for r in reports], [5., 0., 0., 0., 5.])
        self.assertEqual(sum(r["timing"]["generation_seconds"] for r in reports), 16.0)
        for version, report in enumerate(reports):
            timing = report["timing"]
            self.assertEqual(len(report["sha256"]), 64)
            self.assertEqual(timing["version"], version)
            self.assertEqual(timing["update_seconds"], 3.0)
            self.assertEqual(timing["idle_wait_seconds"], 0.0)
            self.assertIn("no standalone idle wait", timing["idle_semantics"])
            self.assertGreater(timing["other_overhead_seconds"], 0.0)
            self.assertTrue(all(value >= 0 for key, value in timing.items() if key.endswith("_seconds")))
            self.assertEqual(timing["wall_seconds"], sum(timing[key] for key in
                ("generation_seconds", "update_seconds", "checkpoint_seconds", "other_overhead_seconds", "idle_wait_seconds")))

    def test_restore_keeps_checkpoint_timing_and_archives_later_receipts_without_overwrite(self):
        sampler, items, plan, callbacks, *_ = fixture(count=2)
        callbacks.guard = lambda v, stats: ["test stop"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_v16b(None, Mock(), None, sampler, plan, out=root, protocol={}, items=items, callbacks=callbacks)
            committed = (root / "update-timing-0.json").read_bytes()
            for attempt in range(2):
                stale = root / "update-timing-1.json"
                stale.write_text(json.dumps({"attempt": attempt}))
                stale_bytes = stale.read_bytes()
                fresh, fresh_items, fresh_plan, fresh_callbacks, *_ = fixture(count=2)
                runtime = V16bScheduler(fresh, plan=fresh_plan, protocol={}, items=fresh_items,
                    grade=None, verify=None, out=root, callbacks=fresh_callbacks, restore=True)
                runtime.restore_drain_state(root / "drain-state.json")
                suffix = "" if attempt == 0 else "-2"
                self.assertEqual((root / "rolled-back" / ("after-1" + suffix) / stale.name).read_bytes(), stale_bytes)
                self.assertFalse(stale.exists())
                self.assertEqual((root / "update-timing-0.json").read_bytes(), committed)
            self.assertEqual(json.loads((root / "rolled-back" / "after-1" / stale.name).read_text()), {"attempt": 0})

    def test_checkpoint_cadence_includes_preparation_slots_and_preserves_future_results(self):
        sampler, items, plan, callbacks, _, _, saved = fixture(count=10)
        with tempfile.TemporaryDirectory() as directory:
            result = run_v16b(None, Mock(), None, sampler, plan, out=Path(directory),
                              protocol={"prep_updates": 3}, items=items, callbacks=callbacks)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual([version for version, *_ in saved], [4, 12])
        self.assertTrue(saved[0][2])
        self.assertFalse(saved[0][3])
        self.assertTrue(all(record["consume_version"] >= 5 for record in saved[0][2]))

    def run_fixture(self, count=4, reverse=False):
        sampler, items, plan, callbacks, baselines, checks, saved = fixture(count, reverse)
        with tempfile.TemporaryDirectory() as directory:
            result = run_v16b(None, Mock(), None, sampler, plan, out=Path(directory), protocol={"version": "test"},
                              items=items, callbacks=callbacks)
        return result, sampler, baselines, checks, saved

    def test_eta_one_schedule_flush_and_update_determinism(self):
        previous = torch.are_deterministic_algorithms_enabled()
        result, sampler, baselines, checks, saved = self.run_fixture()
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual([v for v, *_ in checks], list(range(7)))
        self.assertTrue(all(deterministic for *_, deterministic in checks))
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), previous)
        expected = [(0, 0, "T1"), (0, 0, "OBS"), (0, 1, "T1"), (0, 1, "OBS"),
                    (1, 2, "T1"), (1, 2, "OBS"), (1, 0, "NOTE"),
                    (2, 3, "T1"), (2, 3, "OBS"), (2, 1, "NOTE"), (2, 0, "RESOLVE"),
                    (3, 2, "NOTE"), (3, 1, "RESOLVE"), (4, 3, "NOTE"),
                    (4, 2, "RESOLVE"), (5, 3, "RESOLVE")]
        self.assertEqual(sampler.admissions, expected)
        self.assertTrue(all(len(live) <= 2 for live in sampler.saved_live))
        self.assertFalse(saved[-1][-1])
        self.assertEqual([r["kind"] for r in checks[3][1]], ["T1", "OBS", "NOTE", "RESOLVE"])
        self.assertEqual(checks[3][2]["expected_problem_ids"], ["p3"])
        self.assertEqual(checks[3][2]["resolve_problem_ids"], ["p0"])
        for pid, kind, cutoff, history in baselines:
            self.assertEqual(cutoff, int(pid[1:]) - 2)
            self.assertTrue(all(entry["committed"] for entry in history))

    def test_completed_future_blocks_do_not_enter_baseline_history(self):
        _, _, baselines, _, _ = self.run_fixture(reverse=True)
        for pid, kind, cutoff, history in baselines:
            block = int(pid[1:])
            self.assertTrue(all(entry["block"] <= block - 2 for entry in history))
        p2 = next(entry for entry in baselines if entry[0:2] == ("p2", "T1"))
        self.assertEqual({entry["block"] for entry in p2[3]}, {0})

    def test_completion_order_does_not_change_sealed_rows(self):
        _, _, _, forward, _ = self.run_fixture()
        _, _, _, reverse, _ = self.run_fixture(reverse=True)
        self.assertEqual([rows for _, rows, *_ in forward], [rows for _, rows, *_ in reverse])

    def test_real_baseline_receipts_are_frozen_before_generation(self):
        from fine_tuning.error_note.v16b_rows import baseline_receipt_v16b, obs_baseline_v16b

        sampler, items, plan, callbacks, *_ = fixture()
        def baseline(item, kind, history, cutoff):
            if kind == "OBS":
                return obs_baseline_v16b(item["layer"], history, history_cutoff=cutoff)
            return baseline_receipt_v16b(item, history, prior={"source_rates": {"source": .25}, "overall": .25},
                                          history_cutoff=cutoff)
        callbacks.baseline = baseline
        with tempfile.TemporaryDirectory() as directory:
            runtime = V16bScheduler(sampler, plan=plan, protocol={}, items=items, grade=None, verify=None,
                                    out=Path(directory), callbacks=callbacks)
            sampler.add_version(None, 0)
            runtime.submit_for_version(0)
            frozen = copy.deepcopy(runtime.requests["p0:T1"]["baseline_receipt"])
            cold = runtime.requests["p0:OBS"]["obs_baseline_receipt"]
            self.assertIsNone(cold["baseline"])
            self.assertTrue(cold["zero_advantage"])
            self.assertEqual(cold["history_cutoff"], -2)
            runtime.drain()
            runtime.commit_history(0)
            self.assertEqual(runtime.requests["p0:T1"]["baseline_receipt"], frozen)
            runtime.version = 1
            sampler.add_version(None, 1)
            runtime.submit_for_version(1)
            warmed = runtime.requests["p2:T1"]["baseline_receipt"]
            self.assertEqual(warmed["history_cutoff"], 0)
            self.assertEqual(warmed["window_count"], 1)
            self.assertAlmostEqual(warmed["baseline"], (8 * .25 + 1) / 9)
            self.assertEqual(runtime.requests["p1:T1"]["baseline_receipt"]["window_count"], 0)

    def test_waiting_or_unfinished_prefill_blocks_update(self):
        sampler, items, plan, callbacks, *_ = fixture(1)
        with tempfile.TemporaryDirectory() as directory:
            scheduler = V16bScheduler(sampler, plan=plan, protocol={}, items=items, grade=None, verify=None,
                                      out=Path(directory), callbacks=callbacks)
            sampler.add_version(None, 0)
            scheduler.submit_for_version(0)
            scheduler.drain()
            sampler.barrier = False
            self.assertFalse(scheduler.ready_for_update(0))
            sampler.barrier = True
            self.assertTrue(scheduler.ready_for_update(0))

    def regenerating_fixture(self, directory, *, text, policy_tag, seed_offset=0, restore=False):
        """Content a nondeterministic engine regenerates after resume: T1 text and log-probs, NOTE prompts that embed
        the T1 text, history-bound baseline receipts and the policy hashes trained on those rows."""
        sampler, items, plan, callbacks, *_ = fixture(seed_offset=seed_offset)
        step, add = sampler.step, sampler.add_version
        def regenerated_step():
            out = step()
            for request, output in out:
                if request["kind"] == "T1":
                    output["text"] = text
                    output["generation"]["sampling_logprobs"] = [-.3 - len(text) * 1e-3, -.2]
            return out
        def regenerated_add(bundle, version):
            sha = add(bundle, version)
            if version >= 2:
                sha = hashlib.sha256((sha + policy_tag).encode()).hexdigest()
                sampler.versions[version]["policy_sha256"] = sha
            return sha
        sampler.step, sampler.add_version = regenerated_step, regenerated_add
        make_request = callbacks.make_request
        def note_embeds_t1(item, kind, version, record):
            request = make_request(item, kind, version, record)
            if kind == "NOTE":
                request["messages"] = [{"role": "user", "content": [{"type": "text", "text": record["t1"]["text"]}]}]
            return request
        callbacks.make_request = note_embeds_t1
        callbacks.baseline = lambda item, kind, history, cutoff: {"baseline": .25, "history_cutoff": cutoff,
            "history_sha256": hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest()}
        runtime = V16bScheduler(sampler, plan=plan, protocol={}, items=items, grade=None, verify=None,
                                out=Path(directory), callbacks=callbacks, restore=restore)
        return runtime, sampler, callbacks, items, plan

    def test_crash_two_updates_after_drain_checkpoint_resumes_regenerated_content_but_not_new_seed(self):
        for seed_offset in (0, 100):
            with tempfile.TemporaryDirectory() as directory:
                runtime, sampler, callbacks, items, plan = self.regenerating_fixture(
                    directory, text="first run", policy_tag="a")
                callbacks.guard = lambda v, stats: ["test stop"] if v == 0 else []
                result = run_v16b(None, Mock(), None, sampler, plan, out=Path(directory), protocol={}, items=items,
                                  callbacks=callbacks, runtime=runtime)
                self.assertEqual(result["status"], "STOPPED")
                for v in (1, 2):  # the process keeps training past the checkpoint, then dies after admitting v3
                    sampler.add_version(None, v)
                    runtime.submit_for_version(v)
                    while not runtime.ready_for_update(v):
                        runtime.step()
                    runtime.rows_for_update(v)
                    runtime.commit_history(v)
                    runtime.version += 1
                sampler.add_version(None, 3)
                runtime.submit_for_version(3)
                crashed = dict(runtime.requests)
                restored, fresh, callbacks, items, plan = self.regenerating_fixture(
                    directory, text="replay differs in one token", policy_tag="b", seed_offset=seed_offset, restore=True)
                restored.restore_drain_state(Path(directory) / "drain-state.json")
                if seed_offset:
                    with self.assertRaisesRegex(RuntimeError, "seed"):
                        run_v16b(None, Mock(), None, fresh, plan, out=Path(directory), protocol={}, items=items,
                                 callbacks=callbacks, runtime=restored)
                    continue
                result = run_v16b(None, Mock(), None, fresh, plan, out=Path(directory), protocol={}, items=items,
                                  callbacks=callbacks, runtime=restored)
                self.assertEqual(result["status"], "COMPLETE")
                for rid in ("p2:NOTE", "p3:T1"):
                    self.assertNotEqual(restored.requests[rid]["request_sha256"], crashed[rid]["request_sha256"])
                self.assertNotEqual(restored.requests["p3:T1"]["policy_sha256"], crashed["p3:T1"]["policy_sha256"])
                self.assertEqual(restored.requests["p3:T1"]["seed"], crashed["p3:T1"]["seed"])

    def test_checkpoint_drains_before_next_version_admission(self):
        _, sampler, _, _, saved = self.run_fixture(count=10)
        self.assertEqual(saved[0][0], 7)
        state = saved[0][1]["state"]
        self.assertEqual(state["version"], 8)
        self.assertTrue(all(entry["refcount"] == 0 for entry in state["registry"].values()))
        self.assertTrue(all(entry["policy_version"] <= 7 for entry in state["requests"].values()))
        self.assertTrue(any(version == 8 for version, *_ in sampler.admissions))


class BackendTests(unittest.TestCase):
    def test_selected_probability_and_b2_tail_use_full_vocab_draw(self):
        row = {i: SimpleNamespace(logprob=float(torch.log_softmax(torch.arange(25, dtype=torch.float), 0)[i]))
               for i in range(25)}
        self.assertEqual(sampled_logprob(0, row), row[0].logprob)
        self.assertTrue(b2_tail_flag(0, row, vocab_size=25))
        self.assertFalse(b2_tail_flag(24, row, vocab_size=25))
        with self.assertRaises(RuntimeError):
            sampled_logprob(26, row)
        with self.assertRaises(RuntimeError):
            sampled_logprob(1, {1: SimpleNamespace(logprob=float("nan"))})

    def test_tail_diagnostic_matches_native_topk_threshold_ties_and_rejects_ambiguous_nucleus(self):
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        values = torch.arange(256, dtype=torch.float32) / 50
        values[235] = values[236]  # Native top-k retains both at its twentieth threshold.
        mu = values.log_softmax(0)
        indices = torch.topk(values, 128).indices.tolist()
        rows = {i: SimpleNamespace(logprob=float(mu[i])) for i in indices}
        native = apply_top_k_top_p(values[None].clone(), torch.tensor([20]), torch.tensor([.8]))[0]
        for token in (235, 236, 239, 250, 255):
            self.assertEqual(b2_tail_flag(token, rows, vocab_size=256), not bool(torch.isfinite(native[token])))
        tied = {i: SimpleNamespace(logprob=-2.) for i in range(128)}
        self.assertIsNone(b2_tail_flag(1, tied, vocab_size=256))
        tied = {i: SimpleNamespace(logprob=-2.) for i in range(20)}
        self.assertIsNone(b2_tail_flag(1, tied, vocab_size=20))

    @staticmethod
    def vllm_row(probabilities, vocab_size=151936, sampled=None):
        """top-20 of a full-vocabulary processed distribution plus the sampled action, as vLLM logprobs=20 returns."""
        row = {i: SimpleNamespace(logprob=float(torch.tensor(p, dtype=torch.float64).log()))
               for i, p in enumerate(probabilities)}
        if sampled is not None:
            row[sampled[0]] = SimpleNamespace(logprob=float(torch.tensor(sampled[1], dtype=torch.float64).log()))
        return row

    def test_tail_diagnostic_marks_top_p_boundary_tie_ambiguous_without_raising(self):
        # Ascending: 12 x .005 (.077 of the candidates), then six tied .02 straddling the .2 cut, then two .3.
        row = self.vllm_row([.3] * 2 + [.02] * 6 + [.005] * 12)
        self.assertIsNone(b2_tail_flag(3, row, vocab_size=151936))
        self.assertFalse(b2_tail_flag(0, row, vocab_size=151936))
        # Unreturned tokens may tie the twentieth value, so its membership depends on unseen mass.
        self.assertIsNone(b2_tail_flag(19, row, vocab_size=151936))

    def test_tail_diagnostic_uses_twenty_rows_plus_sampled_action(self):
        from fine_tuning.error_note import single_vllm

        self.assertEqual(single_vllm.TAIL_LOGPROBS, 20)
        top = [.3] * 2 + [.02] * 6 + [.005] * 12
        below = self.vllm_row(top, sampled=(500, .001))
        self.assertEqual(len(below), 21)
        self.assertIs(b2_tail_flag(500, below, vocab_size=151936), True)
        self.assertEqual(sampled_logprob(500, below), float(torch.tensor(.001, dtype=torch.float64).log()))
        self.assertIs(b2_tail_flag(1, below, vocab_size=151936), False)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            b2_tail_flag(0, self.vllm_row(top[:19]), vocab_size=151936)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            b2_tail_flag(600, self.vllm_row(top), vocab_size=151936)

    def coexistence_sampler(self, alone=-.2, mixed=-.2, mixed_greedy_offset=0):
        sampler = SingleVllmSampler.__new__(SingleVllmSampler)
        sampler.pending, sampler._ready_outputs, sampler.visual_version = {}, [], 1
        sampler.image = {"min_pixels": 1024, "max_pixels": 1024}
        sampler.versions = {0: {"request": SimpleNamespace(lora_int_id=10)},
                            1: {"request": SimpleNamespace(lora_int_id=11)}}
        core = SimpleNamespace(has_unfinished_requests=lambda: False)
        calls = []
        def generate(prompts, params, lora_request, **kwargs):
            calls.append((len(lora_request), params.prompt_logprobs))
            out = []
            for prompt, request in zip(prompts, lora_request, strict=True):
                ids = prompt["prompt_token_ids"]
                if params.prompt_logprobs is None:
                    token = request.lora_int_id + (mixed_greedy_offset if len(lora_request) == 2 else 0)
                    out.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=[token] * 64)]))
                    continue
                value = mixed if len(lora_request) == 2 else alone
                rows = [None] + [{t: SimpleNamespace(logprob=value)} for t in ids[1:]]
                out.append(SimpleNamespace(prompt_token_ids=list(ids), prompt_logprobs=rows, outputs=[]))
            return out
        sampler.engine = SimpleNamespace(llm=SimpleNamespace(llm_engine=core, generate=generate),
            prepare=lambda request: SimpleNamespace(token_ids=[1]),
            _prompt=lambda prepared, ids: {"prompt_token_ids": list(ids)})
        sampler._report = Mock()
        return sampler, calls

    def test_coexistence_probe_compares_individual_and_mixed_outputs(self):
        sampler, calls = self.coexistence_sampler()
        report = sampler.verify_coexistence(0, 1)
        self.assertTrue(report["pass"])
        self.assertEqual(calls, [(1, None), (1, None), (2, None), (1, 0), (1, 0), (2, 0)])
        self.assertIn("limitation", report)
        sampler, _ = self.coexistence_sampler(mixed=-.3)
        with self.assertRaisesRegex(RuntimeError, "coexistence"):
            sampler.verify_coexistence(0, 1)
        self.assertFalse(sampler._report.call_args.args[0]["pass"])

    def test_coexistence_greedy_token_flip_is_diagnostic_only(self):
        sampler, _ = self.coexistence_sampler(mixed=-.23, mixed_greedy_offset=1)
        report = sampler.verify_coexistence(0, 1)
        self.assertTrue(report["pass"])
        self.assertEqual([v["same_token_ids"] for v in report["versions"]], [False, False])
        self.assertAlmostEqual(report["versions"][0]["mean_logprob_delta"], .03)

    def test_local_full_vocabulary_records_mu_without_support_arrays(self):
        request = {"seed": 3, "request_id": "test", "max_tokens": 2, "sampling": SAMPLING}
        sampler = _SingleActionSampler([request], 1, {99}, 0, "cpu", loop_stop=False, stop_at=None)
        scores = torch.arange(30, dtype=torch.float).unsqueeze(0)
        emitted = sampler(torch.tensor([[1]]), scores)
        chosen = int(emitted.argmax())
        expected = torch.log_softmax(scores / .7, -1)[0, chosen]
        self.assertAlmostEqual(sampler.logprobs[0][0], float(expected), places=6)
        self.assertEqual(sampler.support_ids, [[]])
        self.assertEqual(len(sampler.tail_flags[0]), 1)

    def test_image_tensor_memo_reuses_processing_and_binds_image_bytes(self):
        from PIL import Image
        class ImageProcessor:
            merge_size = 1
            def __init__(self):
                self.calls = 0
            def __call__(self, images, **kwargs):
                self.calls += 1
                return {"pixel_values": torch.ones(1, 3), "image_grid_thw": torch.ones(1, 3, dtype=torch.long)}
        class Processor:
            def __init__(self):
                self.image_processor = ImageProcessor()
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"][-1]["text"]
            def __call__(self, text, images, **kwargs):
                image_inputs = self.image_processor(images, **kwargs)
                return dict(image_inputs, input_ids=torch.tensor([[98, 97, 99, len(text[0])]]),
                            attention_mask=torch.ones(1, 4, dtype=torch.long))
        sampler = SingleVllmSampler.__new__(SingleVllmSampler)
        sampler._image_memo, sampler._input_memo = {}, {}
        processor = Processor()
        original = processor.image_processor
        sampler.engine = SimpleNamespace(processor=processor, model_config=SimpleNamespace(image_token_id=97))
        sampler.image = {"preprocessing": "hf_processor"}
        image = Image.new("RGB", (1, 1))
        request = {"problem_id": "p", "images": [image], "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "T1"}]}]}
        first, _ = sampler._prepared_input(request, [{"size": [1, 1], "sha256": "a" * 64}])
        request["messages"][0]["content"][-1]["text"] = "NOTE"
        second, _ = sampler._prepared_input(request, [{"size": [1, 1], "sha256": "a" * 64}])
        self.assertIs(processor.image_processor, original)
        self.assertEqual(original.calls, 1)
        self.assertNotEqual(first["input_sha256"], second["input_sha256"])
        with self.assertRaisesRegex(RuntimeError, "image bytes"):
            sampler._prepared_input(request, [{"size": [1, 1], "sha256": "b" * 64}])
        sampler.forget_item("p")
        self.assertEqual(sampler._image_memo, {})

    def test_physical_slot_guard_prevents_add_lora(self):
        sampler = SingleVllmSampler.__new__(SingleVllmSampler)
        sampler.version_registry, sampler.versions = True, {}
        sampler.pending, sampler._preflight_done = {}, True
        sampler.visual_version, sampler._ready_outputs = None, []
        sampler.engine = SimpleNamespace(llm=SimpleNamespace(llm_engine=SimpleNamespace(add_lora=Mock())))
        sampler.prefill_barrier = lambda: True
        with patch("fine_tuning.error_note.rl.policy_sha256", return_value="a" * 64), \
                patch.object(llm_lora, "free_lora_slots", return_value=0):
            with self.assertRaisesRegex(RuntimeError, "safe LoRA slot"):
                sampler.add_version(None, 2)
        sampler.engine.llm.llm_engine.add_lora.assert_not_called()

    def test_two_live_versions_limit_precedes_add_lora(self):
        sampler = SingleVllmSampler.__new__(SingleVllmSampler)
        sampler.version_registry, sampler.visual_version = True, None
        sampler.versions = {0: {"refcount": 1}, 1: {"refcount": 1}}
        sampler.prefill_barrier = lambda: True
        sampler.engine = SimpleNamespace(llm=SimpleNamespace(llm_engine=SimpleNamespace(add_lora=Mock())))
        with patch("fine_tuning.error_note.rl.policy_sha256", return_value="a" * 64):
            with self.assertRaisesRegex(RuntimeError, "safe LoRA slot"):
                sampler.add_version(None, 3)
        sampler.engine.llm.llm_engine.add_lora.assert_not_called()

    def test_all_live_identity_separates_old_llm_from_current_visual_weights(self):
        old = torch.tensor([[1.]], dtype=torch.bfloat16)
        new = torch.tensor([[2.]], dtype=torch.bfloat16)
        a = torch.stack([old, new, torch.zeros_like(old)]).unsqueeze(1)
        module = SimpleNamespace(lora_a_stacked=[a.clone()], lora_b_stacked=[a.clone()])
        manager = SimpleNamespace(lora_index_to_id=[10, 11, None], modules={"packed": module})
        merger = torch.tensor([7.], dtype=torch.bfloat16)
        model = SimpleNamespace(lora_manager=manager, named_parameters=lambda: [("visual.merger.weight", merger)])
        llm = SimpleNamespace(apply_model=lambda callback: [callback(model)])
        hooks = SimpleNamespace(on=True, buffers={"vision": (new.clone(), new.clone())})
        versions = {0: {"lora_int_id": 10, "slots": {"packed": [(old, old)]}},
                    1: {"lora_int_id": 11, "slots": {"packed": [(new, new)]}}}
        with (patch.object(llm_lora, "merger_bf16", return_value={"visual.merger.weight": merger.clone()}),
              patch.object(llm_lora, "vision_lora", return_value={"vision": (new, new)})):
            self.assertTrue(llm_lora.check_all_live_identity(llm, versions, bundle=None, hooks=hooks)["pass"])
            module.lora_a_stacked[0][0].zero_()
            self.assertFalse(llm_lora.check_all_live_identity(llm, versions, bundle=None, hooks=hooks)["pass"])

    def test_local_snapshot_replays_admission_policy_then_restores_training_policy(self):
        from fine_tuning.error_note.rl import policy_sha256
        from fine_tuning.error_note.v16b_runtime import LocalSequentialSampler
        from fine_tuning.tests.test_error_note_single_collect import bundle, request

        owner = bundle()
        sampler = LocalSequentialSampler(owner, protocol_sha256="a" * 64,
            image={"preprocessing": "hf_processor"}, sampling=SAMPLING)
        old = sampler.add_version(owner, 0)
        row = request(cap=3)
        row.update(kind="T1", policy_version=0, policy_sha256=old, request_sha256="b" * 64, admit_step=0,
                   sampling=SAMPLING)
        sampler.submit(row)
        with torch.no_grad():
            next(owner.model.parameters()).add_(.1)
        new = sampler.add_version(owner, 1)
        self.assertNotEqual(old, new)
        output = sampler.step()[0][1]["generation"]
        self.assertEqual(output["policy_sha256"], old)
        self.assertEqual((output["policy_version"], output["visual_version"]), (0, 0))
        self.assertEqual(output["probability_space"], "processed-full-vocab-v1")
        self.assertNotIn("support_token_ids", output)
        self.assertEqual(policy_sha256(owner), new)

    def test_referenced_version_cannot_be_removed(self):
        sampler = SingleVllmSampler.__new__(SingleVllmSampler)
        sampler.versions = {0: {"refcount": 1}}
        with self.assertRaisesRegex(RuntimeError, "referenced"):
            sampler.release_version(0)

    def test_preempted_generation_keeps_admission_policy_and_marks_visual_mix(self):
        sampler = SingleVllmSampler.__new__(SingleVllmSampler)
        sampler.engine_steps, sampler.visual_version, sampler._ready_outputs = 3, 1, []
        sampler.sampling = SAMPLING
        sampler.eos_token_ids = [9]
        request = {"request_id": "p:T1", "problem_id": "p", "block_id": 0, "stage": "v16b-t1",
                   "seed": 7, "max_tokens": 8, "kind": "T1", "policy_version": 0,
                   "policy_sha256": "0" * 64, "protocol_sha256": "a" * 64, "request_sha256": "b" * 64,
                   "sampling": SAMPLING, "messages": [], "images": []}
        live = SimpleNamespace(num_computed_tokens=0, status=SimpleNamespace(name="PREEMPTED"))
        sampler.pending = {"p:T1": {"request": request, "live": live, "item": {
            "prompt_token_ids": [1, 2], "input_sha256": "c" * 64}, "last_computed": 8,
            "visual_version": 0, "mixed_version": False, "preempted": False, "admit_step": 1}}
        sampler.versions = {0: {"policy_sha256": "0" * 64, "refcount": 1, "lora_int_id": 7}}
        diagnostic = {i: SimpleNamespace(logprob=float(torch.log_softmax(torch.arange(32, dtype=torch.float), 0)[i]))
                      for i in range(32)}
        result = SimpleNamespace(token_ids=[4, 9], logprobs=[diagnostic, diagnostic],
                                 finish_reason="stop")
        output = SimpleNamespace(finished=True, request_id="p:T1", prompt_token_ids=[1, 2], outputs=[result])
        core = SimpleNamespace(step=Mock(side_effect=[[], [output]]))
        sampler.engine = SimpleNamespace(llm=SimpleNamespace(llm_engine=core),
                                          model_config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=32)),
                                          tokenizer=SimpleNamespace(decode=lambda *a, **kw: "answer"))
        sampler.release_idle_versions = lambda **kwargs: None
        self.assertEqual(sampler.step(), [])
        self.assertTrue(sampler.pending["p:T1"]["mixed_version"])
        generation = sampler.step()[0][1]["generation"]
        self.assertTrue(generation["preempted"])
        self.assertTrue(generation["mixed_version"])
        self.assertEqual((generation["policy_version"], generation["visual_version"]), (0, 1))
        self.assertEqual(generation["policy_sha256"], "0" * 64)
        self.assertNotIn("support_token_ids", generation)

    def test_push_marks_outputless_partially_prefilled_request_mixed(self):
        sampler = SingleVllmSampler.__new__(SingleVllmSampler)
        sampler.version_registry, sampler.visual_version, sampler._preflight_done = True, 2, True
        sampler._lora_id = 5
        sampler.versions = {2: {"refcount": 1}}
        sampler.pending = {"running": {"live": SimpleNamespace(output_token_ids=[], num_computed_tokens=4),
                                       "visual_version": 2, "mixed_version": False}}
        sampler.prefill_barrier = lambda: True
        sampler.vision_hooks = SimpleNamespace(load=Mock())
        sampler.engine = SimpleNamespace(llm=SimpleNamespace(llm_engine=SimpleNamespace(add_lora=Mock(return_value=True))))
        with tempfile.TemporaryDirectory() as directory:
            sampler.lora_dir = Path(directory)
            with (patch("fine_tuning.error_note.rl.policy_sha256", return_value="3" * 64),
                  patch.object(llm_lora, "free_lora_slots", return_value=2),
                  patch.object(llm_lora, "write_llm_adapter"),
                  patch.object(llm_lora, "llm_slots", return_value={}),
                  patch.object(llm_lora, "vision_lora", return_value={}),
                  patch.object(llm_lora, "push_mergers") as push,
                  patch.object(llm_lora, "check_all_live_identity", return_value={"pass": True})):
                sampler.add_version(None, 3)
        self.assertTrue(sampler.pending["running"]["mixed_version"])
        self.assertEqual(sampler.pending["running"]["visual_version"], 3)
        self.assertTrue(push.call_args.kwargs["inflight"])

    def test_inflight_merger_push_does_not_flush_encoder_or_prefix_cache(self):
        from fine_tuning.tests.test_error_note_single_vllm import PushMergerTests

        case, loaded = PushMergerTests(), []
        names = ["visual.merger.linear_fc1.weight", "visual.deepstack_merger_list.0.linear_fc1.weight"]
        engine = case.engine(names, loaded)
        core = engine.llm.llm_engine
        scheduler = core.engine_core.engine_core.scheduler
        scheduler.waiting, scheduler.running = [], [SimpleNamespace(num_computed_tokens=8, prompt_token_ids=[1] * 4)]
        scheduler.has_unfinished_requests = lambda: True
        scheduler.reset_prefix_cache = Mock(side_effect=AssertionError("cache reset forbidden"))
        engine.llm.collective_rpc = Mock(side_effect=AssertionError("encoder reset forbidden"))
        core.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=False),
            model_config=SimpleNamespace(multimodal_config=SimpleNamespace(mm_processor_cache_gb=0)))
        manager = scheduler.encoder_cache_manager
        manager.cached = {"inflight": {1}}
        llm_lora.push_mergers(engine, case.named(), 1, inflight=True)
        self.assertIs(scheduler.encoder_cache_manager, manager)
        self.assertEqual(manager.cached, {"inflight": {1}})
        scheduler.waiting = [object()]
        with self.assertRaisesRegex(RuntimeError, "prefill"):
            llm_lora.push_mergers(engine, case.named(), 2, inflight=True)


if __name__ == "__main__":
    unittest.main()
