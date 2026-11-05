"""CPU checks of the vLLM SINGLE stream with an authored engine; no GPU, model weights or real vLLM engine."""
from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from vllm.logprobs import Logprob

from fine_tuning.error_note import rl
from fine_tuning.error_note.collect import WAIT, StreamStats
from fine_tuning.error_note.collect_v4 import validate_single_output
from fine_tuning.error_note.single_vllm import SingleVllmSampler, support_from_logprobs

PROMPT = [7, 8, 9]
EOS = 2


def row(values: dict[int, float]) -> dict:
    return {token: Logprob(value) for token, value in values.items()}


class FakeEngine:
    """add_request/step like vLLM V1: every active request finishes on the next step with an authored completion."""

    def __init__(self, completions):
        self.completions, self.active, self.params = completions, [], {}

    def add_request(self, request_id, prompt, params):
        self.active.append(request_id)
        self.params[request_id] = params

    def step(self):
        done, self.active = self.active, []
        return [SimpleNamespace(request_id=key, finished=True, prompt_token_ids=PROMPT,
                                outputs=[self.completions[key]]) for key in done]

    def abort_request(self, ids):
        self.active = [key for key in self.active if key not in ids]


def sampler(engine):
    owner = SimpleNamespace(prepare=lambda request: SimpleNamespace(token_ids=PROMPT),
                            _prompt=lambda prepared, ids: {"prompt_token_ids": list(ids)},
                            processor=None, model_config=None,
                            tokenizer=SimpleNamespace(decode=lambda ids, **_: " ".join(map(str, ids))),
                            llm=SimpleNamespace(llm_engine=engine))
    value = SingleVllmSampler.__new__(SingleVllmSampler)
    value.engine, value.max_num_seqs, value.sleep, value.eos_token_ids = owner, 16, False, [EOS]
    value.loaded_policy, value.asleep = "c" * 64, False
    return value


def request(pid):
    return {"request_id": f"{pid}:t1", "problem_id": pid, "block_id": 0, "stage": "t1", "seed": 5, "max_tokens": 8,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}],
            "images": [Image.new("RGB", (4, 4))]}


class SupportTests(unittest.TestCase):
    def test_finite_processed_logprobs_are_the_support_and_include_the_action(self):
        value, support = support_from_logprobs(5, row({5: -.2, 3: -1.7, 9: -math.inf, 1: -math.inf}))
        self.assertEqual((value, support), (-.2, [3, 5]))
        with self.assertRaisesRegex(RuntimeError, "outside"):
            support_from_logprobs(9, row({5: -.2, 9: -math.inf}))
        with self.assertRaisesRegex(RuntimeError, "outside"):
            support_from_logprobs(4, row({t: -3. for t in range(21)} | {4: -1.}))


class StreamTests(unittest.TestCase):
    def build(self, completions):
        engine = FakeEngine(completions)
        value = sampler(engine)
        args = SimpleNamespace(policy_sha256="c" * 64, protocol_sha256="a" * 64)
        item = {"prompt_token_ids": PROMPT, "input_tokens": 3, "input_sha256": "b" * 64}
        with patch("fine_tuning.error_note.single_vllm.SingleVllmSampler.sync", return_value="c" * 64), \
                patch("fine_tuning.notebook_rl.input_preparation.prepare_protocol_inputs", return_value=(item, [])):
            stream = value.stream_builder(None, args, StreamStats())
            self.assertTrue(stream.continuous)
            return list(stream([request("p0"), WAIT, request("p1")], StreamStats(), None)), engine

    def test_completion_records_actions_support_and_exact_prompt_and_passes_ledger_validation(self):
        good = SimpleNamespace(token_ids=[4, EOS], finish_reason="stop",
                               logprobs=[row({4: -.1, 6: -2.5, 8: -math.inf}), row({EOS: -.05, 3: -3.})])
        outputs, engine = self.build({"p0:t1": good, "p1:t1": good})
        self.assertEqual([r["request_id"] for r, _ in outputs], ["p0:t1", "p1:t1"])
        params = engine.params["p0:t1"]
        self.assertEqual((params.temperature, params.top_p, params.top_k, params.presence_penalty, params.logprobs),
                         (.7, .8, 20, 1.5, 20))
        generation = outputs[0][1]["generation"]
        self.assertEqual(generation["support_token_ids"], [[4, 6], [EOS, 3]])
        self.assertEqual(generation["support_sha256"], [rl.digest([4, 6]), rl.digest([EOS, 3])])
        self.assertEqual(generation["sampling_logprobs"], [-.1, -.05])
        self.assertEqual(generation["completion_token_ids"], [4, EOS])
        validate_single_output(request("p0"), outputs[0][1], policy_sha256="c" * 64, protocol_sha256="a" * 64)

    def test_stop_without_eos_or_missing_logprobs_fails_closed(self):
        bad = {"p0:t1": SimpleNamespace(token_ids=[4], finish_reason="stop", logprobs=[row({4: -.1})]),
               "p1:t1": SimpleNamespace(token_ids=[4], finish_reason="length", logprobs=None)}
        with self.assertRaisesRegex(RuntimeError, "EOS|log-probability"):
            self.build(bad)


class SleepTests(unittest.TestCase):
    def test_release_frees_engine_and_sync_wakes_and_pushes_complete_weights(self):
        calls = []
        llm = SimpleNamespace(sleep=lambda level: calls.append(("sleep", level)), wake_up=lambda: calls.append(("wake",)))
        engine = SimpleNamespace(llm=llm, policy_version=3,
                                 _load_weights=lambda named, version: calls.append(("push", version, len(list(named)))))
        value = SingleVllmSampler.__new__(SingleVllmSampler)
        value.engine, value.sleep, value.asleep, value.loaded_policy = engine, True, False, "x" * 64
        with patch.object(torch.cuda, "empty_cache"):
            value.release()
            self.assertTrue(value.asleep)
            self.assertIsNone(value.loaded_policy)
            with patch("fine_tuning.notebook_rl.policy.merged_named_tensors", return_value=iter([("w", torch.ones(1))])), \
                    patch("fine_tuning.error_note.rl.policy_sha256", return_value="y" * 64):
                self.assertEqual(value.sync(object()), "y" * 64)
                value.sync(object())  # unchanged policy: no second push
        self.assertEqual(calls, [("sleep", 2), ("wake",), ("push", 4, 1)])


if __name__ == "__main__":
    unittest.main()
