"""SINGLE collection CPU fixtures and tiny Qwen checks; no pretrained weights, GPU or network."""
import copy
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from fine_tuning.error_note.collect import (SINGLE_SAMPLING, SingleGenerationDeadline, StreamStats, hf_single_stream_builder,
                                            single_request_sha256, _SingleActionSampler)
from fine_tuning.error_note.collect_v4 import (SINGLE_LEDGER, run_single_requests, validate_single_output, validate_single_partial)
from fine_tuning.error_note.rl import policy_sha256, processed_scores, support_ids, support_sha256
from fine_tuning.notebook_rl.input_preparation import digest, prepare_protocol_inputs
from fine_tuning.tests.test_notebook_rl_shared_policy_engine import processor
from fine_tuning.tests.test_training import tiny_model


def request(i=0, *, cap=7, images=1, stage="t1"):
    prompt = " ".join(["question"] * (i+1))
    return {"request_id": f"p{i}:{stage}", "problem_id": f"p{i}", "block_id": "b0", "stage": stage,
            "seed": 31+i, "max_tokens": cap,
            "messages": [{"role": "user", "content": [{"type": "image"} for _ in range(images)]
                         + [{"type": "text", "text": prompt}]}],
            "images": [Image.new("RGB", (8, 8), (30+i+j, 90, 120)) for j in range(images)]}


def bundle():
    model = tiny_model()
    return SimpleNamespace(model=model, processor=processor(), params={"L": list(model.parameters())})


def args(owner, *, seqs=1, budget=1536, **kwargs):
    return SimpleNamespace(policy_sha256=policy_sha256(owner), protocol_sha256="a"*64, max_num_seqs=seqs,
                           max_num_batched_tokens=budget, context_limit=96,
                           image_cfg={"preprocessing": "hf_processor"}, **kwargs)


def outputs(owner, requests, *, seqs=1, budget=1536, **kwargs):
    stats = StreamStats()
    stream = hf_single_stream_builder(owner, args(owner, seqs=seqs, budget=budget, **kwargs), stats)
    return [output for _, output in stream(requests, stats, None)]


class SingleCollectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_actual_tiny_qwen_processed_replay_and_bounded_score_capture(self):
        owner, item = bundle(), request(cap=6)
        item["seed"] = 32  # Fixed CPU fixture without generated image placeholders; real placeholders are rejected at replay.
        owner.model.train()
        rng = torch.get_rng_state().clone()
        original = owner.model.generate

        def generate(**kwargs):
            cfg = kwargs["generation_config"]
            self.assertTrue(cfg.do_sample)
            self.assertEqual((cfg.temperature, cfg.top_p, cfg.top_k), (1., 1., 0))
            self.assertEqual((cfg.num_return_sequences, cfg.num_beams), (1, 1))
            self.assertFalse(cfg.output_scores)
            result = original(**kwargs)
            self.assertIsNone(result.scores)
            return result

        with patch.object(owner.model, "generate", side_effect=generate):
            output, = outputs(owner, [item])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(owner.model.training)
        generation = output["generation"]
        prepared, _ = prepare_protocol_inputs(owner.processor, owner.model.config, item["messages"], item["images"],
                                               {"preprocessing": "hf_processor"})
        self.assertEqual(generation["prompt_token_ids"], prepared["prompt_token_ids"])
        self.assertEqual(generation["input_sha256"], prepared["input_sha256"])
        self.assertEqual(generation["request_sha256"], single_request_sha256(item))
        owner.model.eval()
        action_ids = generation["completion_token_ids"]
        inputs = dict(prepared["inputs"])
        inputs["input_ids"] = torch.tensor([prepared["prompt_token_ids"] + action_ids])
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        with torch.no_grad():
            logits = owner.model(**inputs, use_cache=False).logits
        raw_difference = []
        for t, token in enumerate(action_ids):
            score = logits[:, len(prepared["prompt_token_ids"])-1+t].float()
            transformed = processed_scores(score, torch.tensor([action_ids[:t]], dtype=torch.long), **SINGLE_SAMPLING)
            expected = transformed.log_softmax(-1)[0, token].item()
            self.assertAlmostEqual(generation["sampling_logprobs"][t], expected, places=5)
            self.assertEqual(generation["support_sha256"][t], support_sha256(transformed)[0])
            self.assertEqual(generation["support_token_ids"][t], support_ids(transformed)[0])
            self.assertIn(token, generation["support_token_ids"][t])
            raw_difference.append(abs(expected-score.log_softmax(-1)[0, token].item()))
        self.assertGreater(max(raw_difference), .1)
        validate_single_output(item, output, policy_sha256=policy_sha256(owner), protocol_sha256="a"*64)

    def test_different_question_batches_padding_images_seed_and_capacity_equivalence(self):
        owner = bundle()
        owner.model.generation_config.eos_token_id = None
        items = [request(0, cap=3), request(1, cap=5, images=2), request(2, cap=7)]
        serial = outputs(owner, items)
        seen_batch_shapes = []
        original = owner.model.generate

        def generate(**kwargs):
            seen_batch_shapes.append(tuple(kwargs["input_ids"].shape))
            self.assertEqual(int(kwargs["image_grid_thw"].shape[0]), 4)
            self.assertEqual(kwargs["attention_mask"][0, 0].item(), 0)
            return original(**kwargs)

        with patch.object(owner.model, "generate", side_effect=generate):
            batched = outputs(owner, items, seqs=3)
        self.assertEqual(len(seen_batch_shapes), 1)
        bounded = outputs(owner, items, seqs=3, budget=45)
        for expected, actual, small in zip(serial, batched, bounded, strict=True):
            for candidate in (actual, small):
                for field in ("prompt_token_ids", "completion_token_ids", "support_sha256", "input_sha256", "seed"):
                    self.assertEqual(candidate["generation"][field], expected["generation"][field])
                torch.testing.assert_close(torch.tensor(candidate["generation"]["sampling_logprobs"]),
                                           torch.tensor(expected["generation"]["sampling_logprobs"]), rtol=0, atol=2e-6)
            self.assertEqual(actual["finish_reason"], "length")
            self.assertEqual(actual["completion_tokens"], actual["generation"]["max_tokens"])
        self.assertNotEqual(serial[0]["generation"]["completion_token_ids"],
                            serial[1]["generation"]["completion_token_ids"][:3])

    def test_actual_eos_is_preserved_per_row_and_length_never_adds_eos(self):
        owner = bundle()
        owner.model.generation_config.eos_token_id = None
        items = [request(i, cap=8) for i in range(3)]
        free = outputs(owner, items, seqs=3)
        eos = next(t for t in range(1, 32) if sum(t in o["generation"]["completion_token_ids"][:5] for o in free) == 1)
        owner.model.generation_config.eos_token_id = [eos]
        actual = outputs(owner, items, seqs=3)
        reasons = set()
        for first, output in zip(free, actual, strict=True):
            raw = first["generation"]["completion_token_ids"]
            expected = raw[:raw.index(eos)+1] if eos in raw else raw
            self.assertEqual(output["generation"]["completion_token_ids"], expected)
            self.assertEqual(output["finish_reason"], "stop" if eos in raw else "length")
            reasons.add(output["finish_reason"])
        self.assertEqual(reasons, {"stop", "length"})

    def test_stale_policy_duplicate_and_over_budget_fail_before_new_candidate(self):
        owner = bundle()
        cfg = args(owner, seqs=2)
        stats = StreamStats()
        stream = hf_single_stream_builder(owner, cfg, stats)
        with torch.no_grad():
            next(owner.model.parameters()).add_(.1)
        with self.assertRaisesRegex(ValueError, "stale"):
            list(stream([request()], stats, None))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            outputs(owner, [request(), request()], seqs=2)
        with self.assertRaisesRegex(ValueError, "budget"):
            outputs(owner, [request()], budget=1)
        for key, value in (("n", 1), ("seeds", [31]), ("num_return_sequences", 2)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "forbidden"):
                outputs(owner, [request() | {key: value}])

    def test_generator_missing_or_duplicate_output_shape_is_rejected(self):
        owner = bundle()
        for shape in ((0, 1), (2, 10)):
            with self.subTest(shape=shape), patch.object(owner.model, "generate",
                    return_value=SimpleNamespace(sequences=torch.zeros(shape, dtype=torch.long))), self.assertRaises(RuntimeError):
                outputs(owner, [request()])

    def test_per_row_loop_stop_and_deadline_do_not_create_eos(self):
        from fine_tuning.error_note.v4 import BlockRepetition

        items = [request(0, cap=8), request(1, cap=2)]
        sampler = _SingleActionSampler(items, 1, {3}, 0, torch.device("cpu"), loop_stop=True, stop_at=None)
        sampler.detectors = [BlockRepetition(2, 2), None]
        prompt = torch.tensor([[9], [9]])
        sequences = prompt.clone()
        for _ in range(4):
            scores = torch.tensor([[-100., 100., -100., -100.]]*2)
            forced = sampler(sequences, scores)
            sequences = torch.cat([sequences, forced.argmax(-1, keepdim=True)], dim=1)
        sampler.finish(sequences, prompt)
        self.assertEqual(sampler.reasons, ["loop_stop", "length"])
        self.assertEqual(sampler.tokens, [[1]*4, [1]*2])
        expired = _SingleActionSampler([request()], 1, {3}, 0, torch.device("cpu"), loop_stop=False, stop_at=0.)
        with self.assertRaises(TimeoutError):
            expired(torch.tensor([[9]]), torch.zeros(1, 4))
        self.assertEqual(expired.tokens, [[]])

    def deadline_fixture(self, root, *, before_draw=False):
        owner = bundle()
        owner.model.generation_config.eos_token_id = 3
        items = [request(0, cap=8), request(1, cap=8)]
        stats = StreamStats()
        stream = hf_single_stream_builder(owner, args(owner, seqs=2), stats)
        options = dict(policy_sha256=policy_sha256(owner), protocol_sha256="a"*64)

        def generate(**kwargs):
            sampler, prefix = kwargs["logits_processor"][0], kwargs["input_ids"]
            scores = torch.full((2, 32), -1000.)
            scores[0, 3], scores[1, 1] = 0., 0.
            if not before_draw:
                draw = sampler(prefix, scores).argmax(-1, keepdim=True)
                prefix = torch.cat([prefix, draw], dim=1)
            sampler.stop_at = 0.  # Controlled expiry; never wait for a wall clock in a CPU fixture.
            return sampler(prefix, scores)

        with patch.object(owner.model, "generate", side_effect=generate) as generated, \
                self.assertRaises(SingleGenerationDeadline) as failure:
            run_single_requests(items, stream, stats, root, stop_after=1000., **options)
        self.assertEqual(generated.call_count, 1)
        self.assertTrue(stats.stopped_early)
        return items, options, failure.exception

    def test_deadline_preserves_two_original_traces_including_finished_eos_and_refuses_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items, options, failure = self.deadline_fixture(root)
            saved = json.loads((root / SINGLE_LEDGER).read_text())
            self.assertIn("SingleGenerationDeadline", saved["error"])
            self.assertEqual(failure.request_ids, [item["request_id"] for item in items])
            for index, (item, partial) in enumerate(zip(items, failure.partials, strict=True)):
                record = saved["requests"][item["request_id"]]
                self.assertEqual(record["state"], "STARTED")
                self.assertNotIn("output", record)
                self.assertEqual(record["partial_output"], partial)
                self.assertEqual(record["partial_output_sha256"], digest(partial))
                trace = partial["generation"]
                self.assertEqual(trace["completion_token_ids"], [[3], [1]][index])
                self.assertEqual(trace["finish_reason"], ["stop", "deadline"][index])
                self.assertEqual(trace["sampling_logprobs"], failure.sampled["logprobs"][index])
                self.assertEqual(trace["support_sha256"], failure.sampled["supports"][index])
                validate_single_partial(item, partial, **options)
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    validate_single_output(item, partial, **options)
            with patch("fine_tuning.error_note.collect._single_generate_batch", side_effect=AssertionError("resampling")):
                with self.assertRaisesRegex(ValueError, "corrupt|INCOMPLETE"):
                    run_single_requests(items, lambda *args: iter(()), StreamStats(), root, **options)

    def test_deadline_before_first_draw_preserves_zero_actions_without_eos(self):
        with tempfile.TemporaryDirectory() as tmp:
            items, options, failure = self.deadline_fixture(Path(tmp), before_draw=True)
            for item, partial in zip(items, failure.partials, strict=True):
                self.assertEqual(partial["text"], "")
                self.assertEqual(partial["completion_tokens"], 0)
                for name in ("completion_token_ids", "sampling_logprobs", "support_sha256"):
                    self.assertEqual(partial["generation"][name], [])
                self.assertEqual(partial["finish_reason"], "deadline")
                validate_single_partial(item, partial, **options)

    def test_malformed_deadline_traces_fail_closed_without_persisting_any_partial(self):
        with tempfile.TemporaryDirectory() as fixture:
            items, options, genuine = self.deadline_fixture(Path(fixture))
        changes = (
            lambda parts: parts.pop(),
            lambda parts: parts.append(copy.deepcopy(parts[0])),
            lambda parts: parts[1]["generation"].update(request_sha256="b"*64),
            lambda parts: parts[1].update(completion_tokens=2),
            lambda parts: parts[1]["generation"].update(sampling_logprobs=[float("nan")]),
            lambda parts: parts[1]["generation"].update(support_sha256=["bad"]),
            lambda parts: parts[1]["generation"].update(prompt_token_ids=[9]),
            lambda parts: parts[1]["generation"].update(incomplete=False),
            lambda parts: (parts[1].update(finish_reason="stop"), parts[1]["generation"].update(finish_reason="stop")),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmp:
                error = SingleGenerationDeadline(copy.deepcopy(genuine.sampled), list(genuine.request_ids))
                error.partials = copy.deepcopy(genuine.partials)
                change(error.partials)

                def interrupted(requests, stats, stop_after):
                    list(requests)
                    raise error
                    yield

                root = Path(tmp)
                with self.assertRaisesRegex(ValueError, "invalid deadline trace"):
                    run_single_requests(items, interrupted, StreamStats(), root, **options)
                saved = json.loads((root / SINGLE_LEDGER).read_text())
                self.assertIn("INVALID_DEADLINE_TRACE", saved["error"])
                self.assertTrue(all(record["state"] == "STARTED" and "partial_output" not in record
                                    for record in saved["requests"].values()))

    def ledger_fixture(self):
        owner, item = bundle(), request(cap=3)
        output, = outputs(owner, [item])
        return owner, item, output

    def test_ledger_reuses_only_exact_completed_request_and_preserves_original_ids(self):
        owner, item, output = self.ledger_fixture()
        calls = []

        def stream(requests, stats, stop_after):
            for r in requests:
                calls.append(r["request_id"])
                yield r, output

        with tempfile.TemporaryDirectory() as tmp:
            options = dict(policy_sha256=policy_sha256(owner), protocol_sha256="a"*64)
            first = run_single_requests([item], stream, StreamStats(), Path(tmp), **options)
            second = run_single_requests([item], stream, StreamStats(), Path(tmp), **options)
            self.assertEqual(first, second)
            self.assertEqual(calls, [item["request_id"]])
            changed = copy.deepcopy(item)
            changed["images"][0].putpixel((0, 0), (1, 2, 3))
            with self.assertRaisesRegex(ValueError, "changed"):
                run_single_requests([changed], stream, StreamStats(), Path(tmp), **options)
            with self.assertRaisesRegex(ValueError, "second candidate"):
                run_single_requests([item | {"request_id": "new-candidate"}], stream, StreamStats(), Path(tmp), **options)
            saved = json.loads((Path(tmp)/SINGLE_LEDGER).read_text())
            saved["requests"][item["request_id"]]["output"]["generation"]["completion_token_ids"][0] += 1
            (Path(tmp)/SINGLE_LEDGER).write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "corrupt"):
                run_single_requests([item], stream, StreamStats(), Path(tmp), **options)

    def test_concurrent_collectors_cannot_both_read_absent_ledger_and_generate(self):
        owner, item, output = self.ledger_fixture()
        options = dict(policy_sha256=policy_sha256(owner), protocol_sha256="a"*64)
        entered, release = Event(), Event()
        calls = []

        def stream(requests, stats, stop_after):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test collector was not released")
            for r in requests:
                calls.append(r["request_id"])
                yield r, output

        with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(max_workers=2) as pool:
            root = Path(tmp)
            first = pool.submit(run_single_requests, [item], stream, StreamStats(), root, **options)
            try:
                self.assertTrue(entered.wait(5))
                self.assertFalse((root / SINGLE_LEDGER).exists())
                second = pool.submit(run_single_requests, [item], stream, StreamStats(), root, **options)
                with self.assertRaisesRegex(ValueError, "BUSY.*busy"):
                    second.result(timeout=2)
            finally:
                release.set()
            self.assertEqual(first.result(timeout=5), {item["request_id"]: output})
            self.assertEqual(calls, [item["request_id"]])
            lock_path = root / ".single-requests.lock"
            inode = lock_path.stat().st_ino
            self.assertEqual(run_single_requests([item], stream, StreamStats(), root, **options),
                             {item["request_id"]: output})
            self.assertEqual(lock_path.stat().st_ino, inode)
            self.assertEqual(calls, [item["request_id"]])

    def test_collector_exception_releases_lock_and_preserves_STARTED_without_resampling(self):
        import fcntl

        owner, item, _ = self.ledger_fixture()
        options = dict(policy_sha256=policy_sha256(owner), protocol_sha256="a"*64)

        def interrupted(requests, stats, stop_after):
            next(iter(requests))
            raise RuntimeError("interrupted before completion")
            yield

        def forbidden(requests, stats, stop_after):
            raise AssertionError("an uncertain STARTED request must never generate again")
            yield

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_single_requests([item], interrupted, StreamStats(), root, **options)
            saved = json.loads((root / SINGLE_LEDGER).read_text())
            self.assertEqual(saved["requests"][item["request_id"]]["state"], "STARTED")
            lock_path = root / ".single-requests.lock"
            inode = lock_path.stat().st_ino
            with lock_path.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "corrupt|INCOMPLETE"):
                run_single_requests([item], forbidden, StreamStats(), root, **options)
            self.assertEqual(lock_path.stat().st_ino, inode)

    def test_uncertain_missing_duplicate_tampered_completions_fail_and_cannot_resample(self):
        owner, item, output = self.ledger_fixture()
        options = dict(policy_sha256=policy_sha256(owner), protocol_sha256="a"*64)

        def interrupted(requests, stats, stop_after):
            next(iter(requests))
            raise RuntimeError("interrupted before commit")
            yield

        def missing(requests, stats, stop_after):
            yield from ()

        def duplicate(requests, stats, stop_after):
            r = next(iter(requests))
            yield r, output
            yield r, output

        for broken in (interrupted, missing, duplicate):
            with self.subTest(kind=broken.__name__), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises((ValueError, RuntimeError)):
                    run_single_requests([item], broken, StreamStats(), Path(tmp), **options)
                if broken is not missing:
                    with self.assertRaisesRegex(ValueError, "corrupt|INCOMPLETE"):
                        run_single_requests([item], missing, StreamStats(), Path(tmp), **options)
        for change in (lambda o: o.update(completion_count=2),
                       lambda o: o["generation"].update(sampling_logprobs=[float("nan")]),
                       lambda o: o["generation"].update(completion_token_ids=[]),
                       lambda o: o["generation"].update(seed=999),
                       lambda o: o["generation"].update(finish_reason="stop", eos_token_ids=[31])):
            bad = copy.deepcopy(output)
            change(bad)
            with self.assertRaises(ValueError):
                validate_single_output(item, bad, **options)

    def test_resume_rejects_truncated_ledger_and_config_changes(self):
        owner, item, output = self.ledger_fixture()
        options = dict(policy_sha256=policy_sha256(owner), protocol_sha256="a"*64)

        def stream(requests, stats, stop_after):
            for r in requests:
                yield r, output

        with tempfile.TemporaryDirectory() as tmp:
            run_single_requests([item], stream, StreamStats(), Path(tmp), **options)
            with self.assertRaisesRegex(ValueError, "stale"):
                run_single_requests([item], stream, StreamStats(), Path(tmp), **(options | {"protocol_sha256": "b"*64}))
            path = Path(tmp)/SINGLE_LEDGER
            path.write_bytes(path.read_bytes()[:-3])
            with self.assertRaises(ValueError):
                run_single_requests([item], stream, StreamStats(), Path(tmp), **options)


if __name__ == "__main__":
    unittest.main()
