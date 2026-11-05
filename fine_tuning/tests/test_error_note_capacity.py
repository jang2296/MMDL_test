"""Authored CPU capacity counterexamples, not CUDA measurements or pretrained-model fit evidence."""
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
import gc
import hashlib
import io
import json
from pathlib import Path
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch

from fine_tuning.error_note import capacity as cap, train_v4 as training
from fine_tuning.error_note import collect
from fine_tuning.error_note.collect import StreamStats, _SingleActionSampler, hf_single_stream_builder
from fine_tuning.error_note.collect_v4 import SINGLE_LEDGER, run_single_requests
from fine_tuning.tests.test_error_note_single_collect import args, bundle, outputs, request
from fine_tuning.tests.test_error_note_single_loss import bind_reward_fixture, block, gradients, pin_reward_fixture
from fine_tuning.tests.test_error_note_train import tiny_factory

IDENTITY = {"source_sha256": "a" * 64, "model_sha256": "b" * 64, "precision": "bf16"}


def authored_profile(owner, max_rows=2):
    size = sum(p.numel() * p.element_size() for ps in owner.params.values() for p in ps)
    fixed = IDENTITY | {"layout_sha256": "c" * 64, "config_sha256": "d" * 64, "device": "GPU-authored, fixture-driver",
        "total_bytes": 2**30, "versions": {"authored_cpu_fixture": "not a real measurement"}, "attention": "sdpa",
        "compute_dtype": "bfloat16", "cache": "dynamic-bfloat16", "checkpointing": True,
        "execution": cap._execution(),
        "state_bytes": {phase: size * (2 if phase == "generation" else 3) for phase in cap.PHASES}}
    entries = []
    for phase, kind in (("generation", "GENERATION"), ("rl_kl", "RL"), ("sft_lm", "A1"), ("sft_lm", "B"), ("sft_full", "A2")):
        route = cap.PHASES[phase]
        baseline = size * 3 + 10000
        geometry = cap.shape([{"problem_id": f"shape-{i}", "prompt_tokens": 256, "target_tokens": 256,
                               "image_grids": [[1, 32, 32], [1, 32, 32]]} for i in range(max_rows)],
                             generation=phase == "generation")
        entries.append({"phase": phase, "route": route,
            "shape": geometry,
            "extra_bytes": 10000, "resident_state_bytes": fixed["state_bytes"][phase],
            "measurement": "cuda_allocator_actual_work", "seconds": 1., "full_kl": phase == "rl_kl",
            "kind": kind, "pg_rows": 1 if kind == "RL" else 0, "work_ids": [f"authored-{kind}-{i}" for i in range(max_rows)],
            "minimum_moment_bytes": 0, "state_baseline_allocated_bytes": baseline - fixed["state_bytes"][phase],
            "state_overhead_bytes": 0,
            "workload_sha256": "e" * 64,
            "baseline": {"free": 2**30 - baseline, "total": 2**30, "allocated": baseline, "reserved": baseline},
            "peak_allocated_bytes": baseline + 10000, "peak_reserved_bytes": baseline + 10000})
    return {"version": cap.VERSION, "status": "MEASURED", "binding": fixed, "reserve_bytes": 10000,
            "calibration_policy_sha256": stream_policy(owner), "entries": entries}


def available_memory():
    return {"free": 2**29, "total": 2**30, "allocated": 10000, "reserved": 10000}


class CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def session(self, owner, profile):
        self.enterContext(patch.object(cap, "binding", return_value=copy.deepcopy(profile["binding"])))
        self.enterContext(patch.object(cap, "memory", side_effect=lambda owner: available_memory()))
        return cap.Capacity(owner, profile, IDENTITY)

    def cpu_capture(self, owner):
        """Authored allocator metadata only; actual math and Capture transaction code run on CPU."""
        profile = authored_profile(owner)
        capture = cap.Capture.__new__(cap.Capture)
        capture.bundle, capture.identity, capture.optimizer = owner, IDENTITY, None
        capture.binding, capture.policy_sha256 = profile["binding"], stream_policy(owner)
        capture.reserve_bytes, capture.entries, capture.failed, capture._training_state = 10000, [], False, None
        size = profile["binding"]["state_bytes"]["rl_kl"]
        before = {"free": 2**29, "total": 2**30, "allocated": 10000, "reserved": 10000}
        after = before | {"allocated": 10000 + size, "reserved": 10000 + size}
        self.enterContext(patch.object(cap, "binding", return_value=profile["binding"]))
        self.enterContext(patch.object(cap, "memory", side_effect=lambda owner: dict(next(readings))))
        readings = iter([value for _ in range(20) for value in (before, after)])
        for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
            self.enterContext(patch.object(torch.cuda, name))
        for name in ("max_memory_allocated", "max_memory_reserved"):
            self.enterContext(patch.object(torch.cuda, name, return_value=after["allocated"] + 10000))
        capture.before_batches, capture.after_batches, capture.final_gradients = [], [], []
        original_batch, original_state = capture.batch, capture.training_state

        @contextmanager
        def observe_batch(*args, **kwargs):
            capture.before_batches.append(gradients(owner))
            with original_batch(*args, **kwargs):
                yield
            capture.after_batches.append(gradients(owner))

        @contextmanager
        def observe_state():
            with original_state():
                try:
                    yield
                finally:
                    capture.final_gradients.append(gradients(owner))

        capture.batch, capture.training_state = observe_batch, observe_state
        return capture

    def test_profile_hash_binding_state_and_full_kl_are_required(self):
        owner = tiny_factory()
        profile = authored_profile(owner)
        path = self.root / "profile.json"
        path.write_text(json.dumps(profile))
        pin = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(cap.load_profile(path, pin), profile)
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(cap.CapacityHold, "SHA256"):
            cap.load_profile(path, pin)
        for mutate in (lambda p: p["entries"].pop(1), lambda p: p["entries"][1].update(full_kl=False),
                       lambda p: p["entries"][1].update(resident_state_bytes=0),
                       lambda p: p["entries"][1].update(minimum_moment_bytes=1),
                       lambda p: p["entries"][1].update(extra_bytes=0), lambda p: p.update(status="ESTIMATED")):
            changed = copy.deepcopy(profile)
            mutate(changed)
            with self.assertRaises(cap.CapacityHold):
                cap.validate_profile(changed)
        with patch.object(cap, "binding", return_value=profile["binding"] | {"precision": "nf4"}):
            with self.assertRaisesRegex(cap.CapacityHold, "binding|changed"):
                cap.Capacity(owner, profile, IDENTITY)

    def test_partial_capture_stays_uncalibrated_and_cannot_be_admitted(self):
        owner = tiny_factory()
        profile = authored_profile(owner)
        capture = cap.Capture.__new__(cap.Capture)  # Authored writer fixture; actual Capture rejects this CPU runtime.
        capture.binding, capture.reserve_bytes = profile["binding"], profile["reserve_bytes"]
        capture.policy_sha256 = profile["calibration_policy_sha256"]
        capture.entries = profile["entries"][:1]
        path = self.root / "partial-capture.json"
        result = capture.write(path)
        self.assertEqual(result["status"], "UNCALIBRATED")
        with self.assertRaises(cap.CapacityHold):
            cap.load_profile(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_profile_publication_has_one_owner_and_never_overwrites(self):
        owner = tiny_factory()
        profile = authored_profile(owner)
        capture = cap.Capture.__new__(cap.Capture)
        capture.binding, capture.reserve_bytes, capture.entries = profile["binding"], profile["reserve_bytes"], []
        capture.policy_sha256 = profile["calibration_policy_sha256"]
        path = self.root / "concurrent-profile.json"
        entered, release = Event(), Event()
        original = collect.write_json

        def blocked_write(destination, value):
            entered.set()
            if not release.wait(10):
                raise AssertionError("publication race test timed out")
            original(destination, value)

        with ThreadPoolExecutor(2) as workers, patch.object(collect, "write_json", side_effect=blocked_write):
            first = workers.submit(capture.write, path)
            self.assertTrue(entered.wait(10))
            second = workers.submit(capture.write, path)
            try:
                with self.assertRaisesRegex(ValueError, "concurrent writer"):
                    second.result(timeout=10)
            finally:
                release.set()
            expected = first.result(timeout=10)
        self.assertEqual(json.loads(path.read_text()), expected)
        before = path.read_bytes()
        with self.assertRaisesRegex(cap.CapacityHold, "already exists"):
            capture.write(path)
        self.assertEqual(path.read_bytes(), before)

    def test_pinned_partial_assembly_preserves_missing_evidence_and_rejects_overlap_or_drift(self):
        profile = authored_profile(tiny_factory())

        def partial(name, entries, **changes):
            value = copy.deepcopy(profile) | {"status": "UNCALIBRATED", "entries": copy.deepcopy(entries)} | changes
            path = self.root / name
            path.write_text(json.dumps(value))
            return path, hashlib.sha256(path.read_bytes()).hexdigest()

        first, h1 = partial("generation.json", profile["entries"][:1])
        second, h2 = partial("backward.json", profile["entries"][1:])
        with redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(cap.main(["assemble", "--input", str(first), "--input-sha256", h1,
                                       "--output", str(self.root / "incomplete.json")]), 2)
        incomplete = json.loads((self.root / "incomplete.json").read_text())
        self.assertEqual(json.loads(printed.getvalue())["file_sha256"], hashlib.sha256((self.root / "incomplete.json").read_bytes()).hexdigest())
        self.assertEqual(incomplete["status"], "UNCALIBRATED")
        with self.assertRaisesRegex(cap.CapacityHold, "completed measured"):
            cap.load_profile(first, h1)
        with self.assertRaisesRegex(cap.CapacityHold, "SHA256"):
            cap.assemble([first], ["f" * 64], self.root / "bad-pin.json")
        for paths, pins in (([first, first], [h1, h1]), ([first, second], [h1])):
            with self.assertRaisesRegex(cap.CapacityHold, "distinct partial"):
                cap.assemble(paths, pins, self.root / "duplicate.json")
        for key, changed in (("reserve_bytes", profile["reserve_bytes"] + 1),
                             ("calibration_policy_sha256", "f" * 64),
                             ("binding", profile["binding"] | {"source_sha256": "f" * 64})):
            other, pin = partial(f"drift-{key}.json", profile["entries"][1:], **{key: changed})
            with self.assertRaisesRegex(cap.CapacityHold, "bindings differ"):
                cap.assemble([first, other], [h1, pin], self.root / f"rejected-{key}.json")
        overlap, pin = partial("overlap.json", profile["entries"][:1] + profile["entries"][1:2])
        with self.assertRaisesRegex(cap.CapacityHold, "overlapping"):
            cap.assemble([first, overlap], [h1, pin], self.root / "rejected-overlap.json")
        complete = cap.assemble([first, second], [h1, h2], self.root / "complete.json")
        self.assertEqual(complete["status"], "MEASURED")
        self.assertEqual(complete["entries"], profile["entries"])
        self.assertEqual(complete["calibration_policy_sha256"], profile["calibration_policy_sha256"])

    def test_capture_requires_real_pg_and_all_sft_kinds_without_fabricating_coverage(self):
        profile = authored_profile(tiny_factory())
        for mutate in (lambda p: p["entries"].pop(3), lambda p: p["entries"][1].update(pg_rows=0)):
            changed = copy.deepcopy(profile)
            mutate(changed)
            with self.assertRaisesRegex(cap.CapacityHold, "A1/A2/B"):
                cap.validate_profile(changed)
            changed["status"] = "UNCALIBRATED"
            cap.validate_profile(changed, allow_partial=True)

    def test_hot_state_capture_cannot_admit_cold_state_or_omit_allocator_overhead(self):
        owner = tiny_factory()
        profile = authored_profile(owner)
        entry = profile["entries"][1]
        entry["minimum_moment_bytes"] = profile["binding"]["state_bytes"]["generation"]
        entry["state_baseline_allocated_bytes"] += entry["minimum_moment_bytes"] - 512
        entry["state_overhead_bytes"] = 512
        entry["extra_bytes"] += 512
        session = self.session(owner, profile)
        rows = [{"problem_id": "p", "prompt_tokens": 4, "target_tokens": 3, "image_grids": [[1, 4, 4]]}]
        with self.assertRaisesRegex(cap.CapacityHold, "unsupported"):
            session.check(rows, "rl_kl")
        optimizer = SimpleNamespace(state={p: {"exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p)}
                                           for ps in owner.params.values() for p in ps})
        session.optimizer = optimizer
        result = session.check(rows, "rl_kl")
        expected = 10000 + 512 + profile["binding"]["state_bytes"]["rl_kl"] // 3 + profile["reserve_bytes"]
        self.assertEqual(result["required_bytes"], expected)

    def test_capture_backward_uses_actual_full_kl_and_sft_with_exact_batch_budget(self):
        owner = tiny_factory()
        encoded, contract = block(owner)
        encoded[1].update(reward=0, status="INCORRECT")
        bind_reward_fixture(encoded[1])
        pin_reward_fixture(encoded, contract)
        for kind, index, route in (("A1", 0, "LM"), ("A2", 1, "FULL"), ("B", 1, "FULL")):
            row = copy.deepcopy(encoded[index])
            row.update(row_id=f"{kind}-authored", kind=kind, route=route)
            encoded.append(row)
        before = stream_policy(owner)
        observe = self.cpu_capture(owner)
        with patch.object(training, "_single_raw_kl", wraps=training._single_raw_kl) as kl:
            cap.capture_backward(observe, encoded, contract, token_budget=10000)
            self.assertTrue(kl.call_count)
            self.assertTrue(all(call.kwargs["force_full"] for call in kl.call_args_list))
        for kind in ("A1", "A2", "B"):
            cap.capture_backward(observe, encoded, contract, component=kind, token_budget=10000)
        self.assertEqual([(entry["phase"], entry["kind"]) for entry in observe.entries],
                         [("rl_kl", "RL"), ("sft_lm", "A1"), ("sft_full", "A2"), ("sft_full", "B")])
        self.assertTrue(observe.entries[0]["full_kl"])
        self.assertEqual(observe.entries[0]["pg_rows"], 1)
        self.assertEqual(before, stream_policy(owner))
        self.assertTrue(all(p.grad is None for ps in owner.params.values() for p in ps))
        with patch.object(owner.model, "forward", side_effect=AssertionError("oversized batch executed")):
            with self.assertRaisesRegex(cap.CapacityHold, "token budget"):
                cap.capture_backward(observe, encoded, contract, token_budget=1)

    def test_native_capture_packing_accumulation_and_fixed_denominator_equal_regular_backward(self):
        owner = tiny_factory()
        rows, contract = block(owner, count=3)
        capture = self.cpu_capture(owner)
        lengths = [row["input_ids"].numel() for row in rows]
        for budget in (0, max(lengths) * 2):
            plan = training.plan_microbatches(lengths, [row["route"] for row in rows], budget)
            expected = training.backward_single_block(owner, rows, contract, budget, components=("PG", "KL"), force_full_kl=True)
            expected_gradients = gradients(owner)
            owner.model.zero_grad(set_to_none=True)
            start = len(capture.entries)
            initial_resets = torch.cuda.reset_peak_memory_stats.call_count
            with patch.object(owner.model, "zero_grad", wraps=owner.model.zero_grad) as clear:
                actual = cap.capture_backward(capture, rows, contract, token_budget=budget)
            self.assertEqual(clear.call_count, 1)
            self.assertEqual(torch.cuda.reset_peak_memory_stats.call_count, initial_resets + 1)
            entries = capture.entries[start:]
            self.assertEqual([entry["work_ids"] for entry in entries], [[rows[i]["row_id"] for i in batch] for batch in plan])
            self.assertEqual([entry["shape"] for entry in entries], [cap.shape(cap.encoded_shapes([rows[i] for i in batch])) for batch in plan])
            self.assertEqual(max(entry["shape"]["rows"] for entry in entries), 1 if budget == 0 else 2)
            self.assertAlmostEqual(actual["loss"], expected["loss"], places=7)
            self.assertEqual(actual["rl"]["problem_count"], 3)
            for name, value in capture.final_gradients[-1].items():
                torch.testing.assert_close(value, expected_gradients[name], atol=1e-7, rtol=1e-4)
            for previous, current in zip(capture.after_batches[start:], capture.before_batches[start + 1:]):
                for name in previous:
                    torch.testing.assert_close(previous[name], current[name])
            self.assertTrue(all(p.grad is None for ps in owner.params.values() for p in ps))

    def test_capture_mixed_sft_routes_keep_global_kind_weight_and_late_replay_fails_before_gradients(self):
        owner = tiny_factory()
        rows, contract = block(owner)
        for index, route in ((0, "LM"), (1, "FULL")):
            row = copy.deepcopy(rows[index])
            row.update(row_id=f"a2-{index}", kind="A2", route=route)
            rows.append(row)
        expected = training.backward_single_block(owner, rows, contract, 0, components=("A2",), force_full_kl=True)
        expected_gradients = gradients(owner)
        owner.model.zero_grad(set_to_none=True)
        capture = self.cpu_capture(owner)
        actual = cap.capture_backward(capture, rows, contract, component="A2", token_budget=0)
        self.assertEqual({entry["phase"] for entry in capture.entries}, {"sft_lm", "sft_full"})
        self.assertAlmostEqual(actual["loss"], expected["loss"], places=7)
        for name, value in capture.final_gradients[-1].items():
            torch.testing.assert_close(value, expected_gradients[name], atol=1e-7, rtol=1e-4)
        bad = copy.deepcopy(rows)
        bad[0]["generation"]["sampling_logprobs"][0] += 1
        bind_reward_fixture(bad[0])
        pin_reward_fixture(bad, contract)
        prior = len(capture.entries)
        with self.assertRaisesRegex(ValueError, "replay changed"):
            cap.capture_backward(capture, bad, contract)
        self.assertEqual(len(capture.entries), prior)
        self.assertTrue(capture.failed)
        self.assertTrue(all(p.grad is None for ps in owner.params.values() for p in ps))

    def test_failed_capture_keeps_completed_groups_but_never_certifies_or_assembles(self):
        owner = tiny_factory()
        rows, contract = block(owner)
        owner.model.eval()
        owner.route.apply(True)
        capture = self.cpu_capture(owner)
        original = training.single_batch_statistics
        forward_calls = 0

        def injected(*args, **kwargs):
            nonlocal forward_calls
            if torch.is_grad_enabled():
                forward_calls += 1
                if forward_calls == 2:
                    raise torch.OutOfMemoryError("authored second microbatch failure")
            return original(*args, **kwargs)

        with patch.object(training, "single_batch_statistics", side_effect=injected):
            with self.assertRaises(torch.OutOfMemoryError):
                cap.capture_backward(capture, rows, contract, token_budget=0)
        self.assertEqual(len(capture.entries), 1)
        self.assertEqual(capture.entries[0]["shape"]["rows"], 1)
        self.assertTrue(capture.failed)
        self.assertFalse(owner.model.training)
        self.assertTrue(owner.route.vision_on)
        self.assertTrue(all(p.grad is None for ps in owner.params.values() for p in ps))
        # Complete authored coverage must still be unusable once its transaction has failed.
        capture.entries = authored_profile(owner)["entries"]
        path = self.root / "failed-profile.json"
        failed = capture.write(path)
        self.assertEqual(failed["status"], "UNCALIBRATED_FAILED")
        with self.assertRaises(cap.CapacityHold):
            cap.validate_profile(failed)
        with self.assertRaisesRegex(cap.CapacityHold, "failed capture"):
            cap.assemble([path], [hashlib.sha256(path.read_bytes()).hexdigest()], self.root / "must-not-publish.json")
        self.assertFalse((self.root / "must-not-publish.json").exists())

    def test_measured_packing_preserves_fifo_and_stops_on_long_or_low_memory_rows(self):
        owner = tiny_factory()
        session = self.session(owner, authored_profile(owner, max_rows=2))
        rows = [{"problem_id": f"p{i}", "prompt_tokens": 10, "target_tokens": 8, "image_grids": [[1, 4, 4]]} for i in range(5)]
        self.assertEqual(session.partition(rows, [list(range(5))], "generation"), [[0, 1], [2, 3], [4]])
        self.assertEqual(cap.shape([rows[0] | {"target_tokens": 30}, rows[1] | {"prompt_tokens": 40}], generation=True)["total_tokens"], 70)
        with self.assertRaisesRegex(cap.CapacityHold, "singleton"):
            session.partition([rows[0] | {"target_tokens": 1000}], [[0]], "sft_full")
        with patch.object(cap, "memory", return_value=available_memory() | {"free": 0}):
            with self.assertRaisesRegex(cap.CapacityHold, "available memory"):
                session.partition(rows, [[0]], "generation")

    def test_preflight_holds_before_started_then_completed_resume_draws_nothing(self):
        owner = bundle()
        owner.model.generation_config.eos_token_id = None
        profile = authored_profile(owner)
        session = self.session(owner, profile)
        items = [request(0, cap=3), request(1, cap=3)]
        stats = StreamStats()
        stream = hf_single_stream_builder(owner, args(owner, seqs=2, capacity_session=session), stats)
        output = self.root / "ledger"
        with patch.object(cap, "memory", return_value=available_memory() | {"free": 0}), \
                patch.object(owner.model, "generate", side_effect=AssertionError("unadmitted generation")):
            with self.assertRaises(cap.CapacityHold):
                run_single_requests(items, stream, stats, output, policy_sha256=stream_policy(owner), protocol_sha256="a" * 64)
        self.assertFalse((output / SINGLE_LEDGER).exists())
        first = run_single_requests(items, stream, stats, output, policy_sha256=stream_policy(owner), protocol_sha256="a" * 64)
        with patch.object(owner.model, "generate", side_effect=AssertionError("completed output regenerated")), \
                patch.object(stream, "admit", side_effect=AssertionError("completed output readmitted")):
            second = run_single_requests(items, stream, stats, output, policy_sha256=stream_policy(owner), protocol_sha256="a" * 64)
        self.assertEqual(first, second)

    def test_oom_is_preserved_and_never_retried_at_lower_concurrency(self):
        owner = bundle()
        session = self.session(owner, authored_profile(owner))
        items, stats = [request(0, cap=3), request(1, cap=3)], StreamStats()
        stream = hf_single_stream_builder(owner, args(owner, seqs=2, capacity_session=session), stats)
        output = self.root / "oom"
        with patch.object(owner.model, "generate", side_effect=torch.OutOfMemoryError("authored CPU OOM")) as generate:
            with self.assertRaises(torch.OutOfMemoryError):
                run_single_requests(items, stream, stats, output, policy_sha256=stream_policy(owner), protocol_sha256="a" * 64)
            with self.assertRaisesRegex(ValueError, "stale|INCOMPLETE"):
                run_single_requests(items, stream, stats, output, policy_sha256=stream_policy(owner), protocol_sha256="a" * 64)
            self.assertEqual(generate.call_count, 1)
        ledger = json.loads((output / SINGLE_LEDGER).read_text())
        self.assertTrue(all(value["state"] == "STARTED" for value in ledger["requests"].values()))

    def test_memory_loss_after_admission_holds_without_execution_or_regeneration(self):
        owner = bundle()
        session = self.session(owner, authored_profile(owner))
        items, stats = [request(0, cap=3), request(1, cap=3)], StreamStats()
        stream = hf_single_stream_builder(owner, args(owner, seqs=2, capacity_session=session), stats)
        admit = stream.admit
        current = available_memory()

        def admitted_then_lost(requests):
            admit(requests)
            current["free"] = 0

        output = self.root / "memory-loss"
        with patch.object(cap, "memory", side_effect=lambda owner: dict(current)), \
                patch.object(stream, "admit", side_effect=admitted_then_lost), \
                patch.object(owner.model, "generate", side_effect=AssertionError("memory loss ignored")) as generate:
            with self.assertRaises(cap.CapacityHold):
                run_single_requests(items, stream, stats, output, policy_sha256=stream_policy(owner), protocol_sha256="a" * 64)
            generate.assert_not_called()
        self.assertFalse((output / SINGLE_LEDGER).exists())
        current.update(available_memory())
        completed = run_single_requests(items, stream, stats, output, policy_sha256=stream_policy(owner), protocol_sha256="a" * 64)
        self.assertEqual(set(completed), {item["request_id"] for item in items})

    def test_completed_batch_reused_and_untouched_batch_resumes_with_same_builder(self):
        owner = bundle()
        owner.model.generation_config.eos_token_id = None
        session = self.session(owner, authored_profile(owner, max_rows=1))
        items, stats = [request(i, cap=3) for i in range(3)], StreamStats()
        stream = hf_single_stream_builder(owner, args(owner, seqs=3, capacity_session=session), stats)
        current, executed = available_memory(), []
        original = owner.model.generate

        def first_then_memory_loss(**kwargs):
            value = original(**kwargs)
            executed.append(tuple(kwargs["input_ids"][0].tolist()))
            current["free"] = 0
            return value

        output = self.root / "batch-resume"
        with patch.object(cap, "memory", side_effect=lambda owner: dict(current)), \
                patch.object(owner.model, "generate", side_effect=first_then_memory_loss):
            with self.assertRaises(cap.CapacityHold):
                run_single_requests(items, stream, stats, output, policy_sha256=stream_policy(owner), protocol_sha256="a" * 64)
        ledger = json.loads((output / SINGLE_LEDGER).read_text())
        self.assertEqual(set(ledger["requests"]), {items[0]["request_id"]})
        self.assertEqual(ledger["requests"][items[0]["request_id"]]["state"], "COMPLETED")
        self.assertNotIn("error", ledger)
        with patch.object(owner.model, "generate", wraps=original) as generate:
            completed = run_single_requests(items, stream, stats, output, policy_sha256=stream_policy(owner), protocol_sha256="a" * 64)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(len(executed), 1)
        self.assertEqual(len(completed), 3)

    def test_profile_bounds_total_actions_and_each_rows_full_image_grid(self):
        owner = tiny_factory()
        profile = authored_profile(owner)
        measured = [{"problem_id": "r0", "prompt_tokens": 10, "target_tokens": 100, "image_grids": [[1, 4, 16]]},
                    {"problem_id": "r1", "prompt_tokens": 109, "target_tokens": 1, "image_grids": [[1, 4, 16]]}]
        for entry in profile["entries"]:
            if entry["phase"] == "sft_lm":
                entry["shape"] = cap.shape(measured)
        session = self.session(owner, profile)
        doubled = [row | {"prompt_tokens": 10, "target_tokens": 100} for row in measured]
        self.assertEqual(cap.shape(measured)["action_tokens"], 101)
        self.assertEqual(cap.shape(doubled)["action_tokens"], 200)
        with self.assertRaisesRegex(cap.CapacityHold, "unsupported"):
            session.check(doubled, "sft_lm")
        transposed = [row | {"image_grids": [[1, 16, 4]]} for row in measured]
        self.assertEqual(cap.shape(transposed)["image_cells"], cap.shape(measured)["image_cells"])
        with self.assertRaisesRegex(cap.CapacityHold, "unsupported"):
            session.check(transposed, "sft_lm")
        session.check(list(reversed(measured)), "sft_lm")  # Row matching does not reorder the actual work.

    def test_profile_batches_keep_original_question_seeds_actions_and_logprobs(self):
        owner = bundle()
        owner.model.generation_config.eos_token_id = None
        session = self.session(owner, authored_profile(owner, max_rows=2))
        items = [request(i, cap=4) for i in range(3)]
        serial = outputs(owner, items)
        batches = []
        original = owner.model.generate

        def generate(**kwargs):
            batches.append(kwargs["input_ids"].shape[0])
            return original(**kwargs)

        with patch.object(owner.model, "generate", side_effect=generate):
            actual = outputs(owner, items, seqs=3, capacity_session=session)
        self.assertEqual(batches, [2, 1])
        for first, second in zip(serial, actual, strict=True):
            for key in ("seed", "completion_token_ids", "prompt_token_ids", "support_sha256"):
                self.assertEqual(first["generation"][key], second["generation"][key])
            torch.testing.assert_close(torch.tensor(first["generation"]["sampling_logprobs"]),
                                       torch.tensor(second["generation"]["sampling_logprobs"]), atol=2e-6, rtol=0)

    def test_training_partition_keeps_pg_sft_gradients_and_fixed_problem_denominator(self):
        owner = tiny_factory()
        rows, contract = block(owner, count=3)
        for kind in ("A1", "A2", "B"):
            for original in rows[:3]:
                row = copy.deepcopy(original)
                row.update(row_id=f"{original['problem_id']}:{kind}", kind=kind, route="LM" if kind == "A1" else "FULL")
                rows.append(row)
        expected = training.backward_single_block(owner, rows, contract, token_budget=10000)
        expected_gradients = gradients(owner)
        owner.model.zero_grad(set_to_none=True)
        session = self.session(owner, authored_profile(owner, max_rows=1))
        actual = training.backward_single_block(owner, rows, contract, token_budget=10000, capacity=session)
        self.assertEqual(actual["rl"]["problem_count"], 3)
        self.assertAlmostEqual(actual["loss"], expected["loss"], places=6)
        for name, gradient in gradients(owner).items():
            torch.testing.assert_close(gradient, expected_gradients[name], atol=2e-6, rtol=2e-4)

    def test_long_full_row_and_explicit_singleton_overflow_fail_before_any_gradient(self):
        owner = tiny_factory()
        rows, contract = block(owner)
        sft = copy.deepcopy(rows[0])
        sft.update(kind="B", row_id="B", route="FULL")
        rows.append(sft)
        profile = authored_profile(owner)
        profile["entries"][-1]["shape"] = cap.shape([{"problem_id": "short", "prompt_tokens": 1,
            "target_tokens": 1, "image_grids": [[1, 32, 32]]}])
        profile["entries"][-1]["work_ids"] = profile["entries"][-1]["work_ids"][:1]
        session = self.session(owner, profile)
        for budget in (10000, 1):
            with self.assertRaises(cap.CapacityHold):
                training.backward_single_block(owner, rows, contract, token_budget=budget, capacity=session)
            self.assertTrue(all(p.grad is None for ps in owner.params.values() for p in ps))

    def test_full_kl_capture_path_is_computed_without_parameter_changes(self):
        head = torch.nn.Linear(4, 13).requires_grad_(False)
        hidden = torch.randn(3, 4, requires_grad=True)
        with patch("torch.nn.functional.linear", wraps=torch.nn.functional.linear) as linear:
            training._single_raw_kl(head, hidden, hidden.detach(), 5)
            self.assertEqual(linear.call_count, 0)
            values = training._single_raw_kl(head, hidden, hidden.detach(), 5, force_full=True)
            self.assertGreater(linear.call_count, 0)
        torch.testing.assert_close(values, torch.zeros_like(values), rtol=0, atol=3e-7)
        values.sum().backward()
        self.assertTrue(bool(torch.isfinite(hidden.grad).all()))
        owner = tiny_factory()
        before = stream_policy(owner)
        with self.assertRaisesRegex(cap.CapacityHold, "CUDA"):
            cap.Capture(owner, IDENTITY, 1024)
        self.assertEqual(before, stream_policy(owner))

    def test_device_history_detects_earlier_prefix_mutation_without_cpu_prefix_lists(self):
        sampler = _SingleActionSampler([request(cap=5)], 2, set(), 0, torch.device("cpu"), loop_stop=False, stop_at=None)
        tokens = torch.tensor([[7, 8]])
        scores = torch.arange(32, dtype=torch.float32)[None]
        for _ in range(2):
            sampler(tokens, scores)
            tokens = torch.cat([tokens, torch.tensor([[sampler.tokens[0][-1]]])], dim=1)
        tokens[0, 2] = (int(tokens[0, 2]) + 1) % 32  # Earlier than the last token: a last-token-only check would miss it.
        with self.assertRaisesRegex(RuntimeError, "recorded draws"):
            sampler(tokens, scores)

    def test_capture_starts_before_inputs_and_releases_previous_batch_tensors(self):
        owner = bundle()
        owner.model.generation_config.eos_token_id = None
        retained, entries = [], []
        active = False
        original_inputs, original_sampler = collect._single_batch_inputs, collect._SingleActionSampler

        class Observe:
            @contextmanager
            def measure(self, *args):
                nonlocal active
                gc.collect()
                if retained:
                    self_case.assertTrue(all(reference() is None for reference in retained))
                entries.append(True)
                active = True
                try:
                    yield None  # CPU lifetime observation only; never a CUDA measurement/profile.
                finally:
                    active = False

        def inputs(*args):
            self.assertTrue(active)
            value = original_inputs(*args)
            retained.append(weakref.ref(value["input_ids"]))
            return value

        def sampler(*args, **kwargs):
            self.assertTrue(active)
            value = original_sampler(*args, **kwargs)
            retained.append(weakref.ref(value.history))
            return value

        self_case = self
        with patch.object(collect, "_single_batch_inputs", side_effect=inputs), \
                patch.object(collect, "_SingleActionSampler", side_effect=sampler):
            actual = outputs(owner, [request(0, cap=3), request(1, cap=3)], seqs=1, capacity_capture=Observe())
        self.assertEqual(len(actual), 2)
        self.assertEqual(len(entries), 2)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in retained))

    def test_capture_resume_never_draws_an_extra_answer_for_completed_requests(self):
        owner = bundle()
        owner.model.generation_config.eos_token_id = None
        calls = []

        class Observe:
            @contextmanager
            def measure(self, *args):
                calls.append(args)
                yield None  # CPU orchestration observation, not a measured profile.

        items, stats = [request(0, cap=3), request(1, cap=3)], StreamStats()
        stream = hf_single_stream_builder(owner, args(owner, seqs=1, capacity_capture=Observe()), stats)
        output = self.root / "capture-resume"
        first = run_single_requests(items, stream, stats, output, policy_sha256=stream_policy(owner), protocol_sha256="a" * 64)
        self.assertEqual(len(calls), 2)
        with patch.object(owner.model, "generate", side_effect=AssertionError("capture regenerated completed answer")):
            second = run_single_requests(items, stream, stats, output, policy_sha256=stream_policy(owner), protocol_sha256="a" * 64)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 2)

    def test_execution_and_frozen_dtype_drift_are_bound_but_vision_routing_is_not(self):
        owner = tiny_factory()
        before = cap.digest(cap._layout(owner))
        owner.route.apply(False)
        self.assertEqual(before, cap.digest(cap._layout(owner)))
        parameter = next(p for p in owner.model.parameters() if not p.requires_grad and p.dtype == torch.float32)
        parameter.data = parameter.data.to(torch.float64)
        self.assertNotEqual(before, cap.digest(cap._layout(owner)))
        flags = cap._execution()
        with patch.object(torch.backends.cuda, "flash_sdp_enabled", return_value=not flags["sdpa_flash"]):
            self.assertNotEqual(flags, cap._execution())

    def test_cuda_logical_device_uses_nonstring_uuid_not_nvml_index(self):
        uuid = "12345678-1234-5678-1234-567812345678"

        class AuthoredCUuuid:
            def __str__(self):
                return uuid

        properties = SimpleNamespace(uuid=AuthoredCUuuid(), total_memory=2**30)
        with patch.object(torch.cuda, "get_device_properties", return_value=properties), \
                patch.object(cap.subprocess, "check_output", return_value=f"GPU-{uuid}, authored-driver") as query:
            actual = cap._physical_device(torch.device("cuda:1"))
        self.assertEqual(actual, (f"GPU-{uuid}, authored-driver", 2**30))
        self.assertIn(f"--id=GPU-{uuid}", query.call_args.args[0])
        self.assertNotIn("--id=1", query.call_args.args[0])


def stream_policy(owner):
    from fine_tuning.error_note.rl import policy_sha256

    return policy_sha256(owner)


if __name__ == "__main__":
    unittest.main()
