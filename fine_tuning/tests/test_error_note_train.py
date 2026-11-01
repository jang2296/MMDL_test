"""Tiny real Qwen3-VL on CPU: micro-batches, Vision routing, save, resume and reload of the error-note training stage."""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

from fine_tuning.error_note import train as trainer
from fine_tuning.error_note.contract import KIND_NOTE, KIND_SOLUTION, PER_UPDATE_ITEMS, ROUTE_FULL, ROUTE_LM
from fine_tuning.error_note.order import VERSION, _ids_sha256
from fine_tuning.error_note.protocol import ARM, plain_protocol
from fine_tuning.modules import tensor_sha256
from fine_tuning.notebook_rl import checkpoint as ckpt
from fine_tuning.notebook_rl.policy import assert_base_unchanged, configure_policy
from fine_tuning.notebook_rl.quantized_policy import _validate_protocol
from fine_tuning.tests.test_training import tiny_model

COMMIT = "a" * 40
IMAGE_BLOCK = [4, 3, 3, 3, 3, 5]  # one 4x4 grid image: start, 4 merged image tokens, end
END = 2


def tiny_factory(model_path=None, arm=ARM, protocol=None):
    return configure_policy(tiny_model(), None, arm, protocol or plain_protocol("bf16"), enforce_counts=False)


def encoded(row_id="r", prompt=(7, 8), target=(9, 10), route=ROUTE_LM, kind=KIND_SOLUTION, images=1, seed=0):
    # Plain targets get the end token, as encode_row does.
    block = IMAGE_BLOCK * images
    pixels = torch.linspace(-1, 1, 16 * 24).reshape(16, 24) * (.5 + .1 * seed)
    return {"row_id": row_id, "kind": kind, "route": route, "input_ids": torch.tensor([*block, *prompt, *target, END]),
            "target_len": len(target) + 1, "pixel_values": torch.cat([pixels] * images),
            "image_grid_thw": torch.tensor([[1, 4, 4]] * images)}


def tiny_encode(row):
    return encoded(row["row_id"], [int(t) for t in row["prompt"].split()], [int(t) for t in row["target"].split()],
                   row["route"], row["kind"], seed=int(row["problem_id"].split("-")[1]) % 5)


def tiny_processor():
    words = ["<pad>", "<unk>", "<eos>", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>", "<|im_end|>",
             "a", "b", "question", "assistant"]
    raw = Tokenizer(WordLevel(dict(zip(words, range(len(words)), strict=True)), unk_token="<unk>"))
    raw.pre_tokenizer = WhitespaceSplit()
    tokenizer = Qwen2TokenizerFast(tokenizer_object=raw, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>",
                                   additional_special_tokens=words[3:8])
    template = ("{% for message in messages %}{% for block in message['content'] %}"
                "{% if block['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
                "{% else %} {{ block['text'] }} {% endif %}{% endfor %}{% endfor %} assistant")
    return Qwen3VLProcessor(image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=2, merge_size=2),
                            tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor(), chat_template=template)


def hashes(params):
    return [tensor_sha256(p) for p in params]


def rows_for(index):
    problem_id = f"item-{index}"
    common = {"problem_id": problem_id, "target_source": "T1", "image_paths": ["x.png"], "target_tokens": 2,
              "subject": "s", "source": "t"}
    # Row B reads the wrong solution "8 8" as input; it is never in a target.
    return [{**common, "row_id": f"{problem_id}:A", "kind": KIND_SOLUTION, "prompt": "7 8", "target": "9 10",
             "route": ROUTE_FULL if index % 2 else ROUTE_LM},
            {**common, "row_id": f"{problem_id}:B", "kind": KIND_NOTE, "prompt": "7 8 8 8 11", "target": "12 13 14",
             "route": ROUTE_FULL if index % 2 else ROUTE_LM}]


class RunFiles:
    """Rows and order on disk: one item per group, none in the listed empty groups; the last group of several is Vision routed."""

    def __init__(self, root, groups=2, empty_groups=()):
        self.root = Path(root)
        count = groups * PER_UPDATE_ITEMS
        ids = [f"item-{i}" for i in range(count)]
        self.order = self.root / "order.json"
        self.order.write_text(json.dumps({"version": VERSION, "items": [{"problem_id": i} for i in ids], "count": count,
                                          "ids_sha256": _ids_sha256(ids)}))
        chosen = [g * PER_UPDATE_ITEMS + int(0 < g == groups - 1) for g in range(groups) if g not in empty_groups]
        self.rows = self.root / "rows.jsonl"
        self.rows.write_text("".join(json.dumps(row) + "\n" for i in chosen for row in rows_for(i)))
        self.count = count

    def args(self, output="run", arm="note", *extra):
        return trainer.build_parser().parse_args([
            "train", "--rows", str(self.rows), "--build-root", str(self.root), "--order", str(self.order),
            "--limit", str(self.count), "--arm", arm, "--precision", "bf16", "--model-path", str(self.root),
            "--output", str(self.root / output), "--code-commit", COMMIT, *extra])

    def train(self, *args, factory=tiny_factory, **kwargs):
        return trainer.run_train(self.args(*args, **kwargs), policy_factory=factory, device="cpu",
                                 make_encoder=lambda bundle, root: tiny_encode)


class TrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_plan_microbatches_groups_by_route_and_budget(self):
        lengths = [10, 40, 30, 20, 100, 5]
        routes = [ROUTE_LM, ROUTE_LM, ROUTE_FULL, ROUTE_LM, ROUTE_LM, ROUTE_FULL]
        plan = trainer.plan_microbatches(lengths, routes, 90)
        self.assertEqual(plan, [[2, 5], [4], [1, 3], [0]])
        self.assertEqual(plan, trainer.plan_microbatches(lengths, routes, 90))
        self.assertEqual(sorted(i for batch in plan for i in batch), list(range(6)))
        for batch in plan:
            self.assertEqual(len({routes[i] for i in batch}), 1)
            self.assertTrue(len(batch) == 1 or max(lengths[i] for i in batch) * len(batch) <= 90)
        self.assertEqual(trainer.plan_microbatches(lengths, routes, 0), [[2], [5], [4], [1], [3], [0]])
        self.assertEqual(trainer.plan_microbatches(lengths, routes), trainer.plan_microbatches(lengths, routes, 0))
        self.assertEqual(trainer.plan_microbatches([50, 50], [ROUTE_LM, ROUTE_LM], 10), [[0], [1]])
        self.assertEqual(trainer.plan_microbatches([50, 50], [ROUTE_LM, ROUTE_LM], 100), [[0, 1]])
        self.assertEqual(trainer.plan_microbatches([], [], 100), [])

    def test_batch_logprobs_equals_one_at_a_time(self):
        rows = [encoded("short", (7,), (9,), seed=1), encoded("two-images", (7, 8, 9, 10), (11, 12, 13, 14), images=2, seed=2),
                encoded("long", tuple(range(7, 25)), (9, 10, 11, 12, 13), seed=3)]
        bundle = tiny_factory()
        batched = trainer.batch_logprobs(bundle, rows)
        for row, values in zip(rows, batched, strict=True):
            self.assertEqual(values.shape, (row["target_len"],))
        sum(-v.mean() for v in batched).backward()
        batched_grads = {g: [p.grad.clone() for p in bundle.params[g]] for g in ("L", "M", "V")}
        bundle.model.zero_grad(set_to_none=True)
        single = [trainer.batch_logprobs(bundle, [row])[0] for row in rows]
        sum(-v.mean() for v in single).backward()
        for one, many in zip(single, batched, strict=True):
            torch.testing.assert_close(many, one, rtol=0, atol=1e-5)
        for group, grads in batched_grads.items():
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
            for grad, parameter in zip(grads, bundle.params[group], strict=True):
                torch.testing.assert_close(grad, parameter.grad, rtol=0, atol=1e-4)

    def test_prompt_tokens_get_no_loss(self):
        bundle = tiny_factory()
        clean = encoded(prompt=(7, 8, 8, 8, 11))
        changed = encoded(prompt=(7, 8, 20, 8, 11))  # one token inside the wrong-solution span of a note prompt
        self.assertFalse(torch.equal(clean["input_ids"], changed["input_ids"]))
        self.assertEqual(clean["target_len"], changed["target_len"])
        self.assertTrue(torch.equal(clean["input_ids"][-3:], changed["input_ids"][-3:]))
        base = bundle.model.get_base_model()
        for row in (clean, changed):
            values = trainer.batch_logprobs(bundle, [row])[0]
            self.assertEqual(values.numel(), row["target_len"])
            # The same value from full logits, gathered at target positions and the end token only.
            ids = row["input_ids"].unsqueeze(0)
            with torch.no_grad():
                logits = base(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=row["pixel_values"],
                              image_grid_thw=row["image_grid_thw"], use_cache=False).logits[0, -4:-1].float()
            expected = logits.log_softmax(-1).gather(-1, ids[0, -3:].unsqueeze(-1)).squeeze(-1)
            torch.testing.assert_close(values.detach(), expected, rtol=0, atol=1e-5)
        self.assertFalse(torch.equal(trainer.batch_logprobs(bundle, [clean])[0], trainer.batch_logprobs(bundle, [changed])[0]))

    def test_encode_row_supervises_target_and_end_token_only(self):
        processor, config = tiny_processor(), SimpleNamespace(image_token_id=3)
        row = {"row_id": "x:B", "kind": KIND_NOTE, "route": ROUTE_FULL, "prompt": "question a a", "target": "b a"}
        image = Image.new("RGB", (16, 16), (30, 90, 120))
        item = trainer.encode_row(processor, config, row, [image])
        self.assertEqual(item["target_len"], 3)
        self.assertEqual(item["input_ids"][-3:].tolist(), [9, 8, 7])  # b, a, <|im_end|>
        self.assertEqual(int((item["input_ids"] == 3).sum()), int(item["image_grid_thw"].prod()) // 4)
        self.assertEqual(int((item["input_ids"][-3:] == 3).sum()), 0)
        for bad in ({"prompt": "question <|image_pad|>"}, {"target": "a <|im_end|> b"}):
            with self.assertRaises(trainer.DropRow):
                trainer.encode_row(processor, config, row | bad, [image])

    def test_route_lm_leaves_vision_grad_none(self):
        bundle = tiny_factory()
        bundle.route.apply(False)
        trainer.batch_logprobs(bundle, [encoded()])[0].sum().backward()
        self.assertTrue(all(p.grad is None for p in bundle.params["V"]))
        for group in ("L", "M"):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in bundle.params[group] if p.grad is not None), 0)

    def test_route_full_gives_vision_grad(self):
        bundle = tiny_factory()
        bundle.route.apply(True)
        trainer.batch_logprobs(bundle, [encoded(route=ROUTE_FULL)])[0].sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in bundle.params["V"] if p.grad is not None), 0)

    def test_update_changes_llm_and_merger_not_vision_when_all_lm(self):
        bundle = tiny_factory()
        optimizer, scheduler = ckpt.make_optimizer(bundle, plain_protocol("bf16"))
        before = {g: hashes(bundle.params[g]) for g in ("L", "M", "V")}
        rows = [encoded("a"), encoded("b", (7,), (9, 11), kind=KIND_NOTE, seed=1)]
        stats = trainer.run_update(bundle, optimizer, scheduler, rows, 0)
        self.assertEqual(hashes(bundle.params["V"]), before["V"])
        for group in ("L", "M"):
            self.assertNotEqual(hashes(bundle.params[group]), before[group])
        self.assertIsNone(stats["grad_norms"]["V"])
        self.assertGreater(stats["grad_norms"]["L"], 0)
        self.assertEqual(stats["rows_by_kind"], {KIND_SOLUTION: 1, KIND_NOTE: 1})
        self.assertEqual(set(stats["loss_by_kind"]), {KIND_SOLUTION, KIND_NOTE})
        self.assertEqual(stats["microbatches"], 2)
        self.assertFalse(any(p in optimizer.state for p in bundle.params["V"]))
        self.assertTrue(bundle.route.vision_on)
        self.assertTrue(all(p.grad is None for ps in bundle.params.values() for p in ps))

    def test_mixed_update_one_step_only(self):
        bundle = tiny_factory()
        optimizer, scheduler = ckpt.make_optimizer(bundle, plain_protocol("bf16"))
        rows = [encoded("a", route=ROUTE_FULL), encoded("b", (7,), (9, 11), route=ROUTE_LM, seed=1)]
        stats = trainer.run_update(bundle, optimizer, scheduler, rows, 1000)
        self.assertEqual(stats["microbatches"], 2)
        self.assertEqual(stats["routes"], {ROUTE_FULL: 1, ROUTE_LM: 1})
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertEqual({float(state["step"]) for state in optimizer.state.values()}, {1.0})
        self.assertTrue(all(p in optimizer.state for p in bundle.params["V"]))
        self.assertGreater(stats["grad_norms"]["V"], 0)
        # A later all-LM update: every step counter moves except Vision, which AdamW skips.
        trainer.run_update(bundle, optimizer, scheduler, [encoded("c")], 0)
        self.assertEqual(scheduler.last_epoch, 2)
        self.assertEqual({float(optimizer.state[p]["step"]) for p in bundle.params["V"]}, {1.0})
        self.assertEqual({float(optimizer.state[p]["step"]) for p in bundle.params["L"]}, {2.0})

    def test_batched_and_single_row_updates_end_close(self):
        rows = [encoded("a", (7, 8), (9, 10, 11), seed=1), encoded("b", (7, 9, 9, 9), (10, 11), seed=2),
                encoded("c", (8,), (10, 12, 13, 9), kind=KIND_NOTE, seed=3)]
        finals = []
        for budget in (0, 10_000):
            bundle = tiny_factory()
            optimizer, scheduler = ckpt.make_optimizer(bundle, plain_protocol("bf16"))
            stats = trainer.run_update(bundle, optimizer, scheduler, rows, budget)
            self.assertEqual(stats["microbatches"], 3 if budget == 0 else 1)
            finals.append({group: [p.detach().clone() for p in params] for group, params in bundle.params.items()})
        for group in ("L", "M"):
            for one, many in zip(finals[0][group], finals[1][group], strict=True):
                torch.testing.assert_close(many, one, rtol=0, atol=1e-5)

    def test_both_arms_same_update_count(self):
        files = RunFiles(self.tmp, groups=3, empty_groups=(1,))
        self.assertEqual(files.train("note", "note", "--token-budget", "200"), 0)
        self.assertEqual(files.train("solution", "solution", "--gradient-checkpointing", "off"), 0)
        note = json.loads((self.tmp / "note/result.json").read_text())
        solution = json.loads((self.tmp / "solution/result.json").read_text())
        for result in (note, solution):
            self.assertEqual((result["status"], result["updates"], result["skipped_updates"]), ("COMPLETE", 2, 1))
            self.assertEqual(result["changed_tensors"]["V"] > 0, result["arm"] == "note")
            self.assertGreater(result["changed_tensors"]["L"], 0)
            self.assertGreater(result["changed_tensors"]["M"], 0)
            self.assertEqual(result["total_tensors"]["M"], 24)
        self.assertEqual(note["rows_by_kind"], {KIND_SOLUTION: 2, KIND_NOTE: 2})
        self.assertEqual(solution["rows_by_kind"], {KIND_SOLUTION: 2})
        self.assertEqual(solution["route_counts"], {ROUTE_LM: 2})
        self.assertEqual(solution["gradient_checkpointing"], "off")
        # Both arms start from one adapter init.
        self.assertEqual(json.loads((self.tmp / "note/initial_state.json").read_text())["sha256"],
                         json.loads((self.tmp / "solution/initial_state.json").read_text())["sha256"])
        updates = [json.loads(line) for line in (self.tmp / "note/updates.jsonl").read_text().splitlines()]
        self.assertEqual([row["skipped"] for row in updates], [False, True, False])
        self.assertEqual([row["step"] for row in updates], [1, 1, 2])

    def test_base_unchanged_after_training(self):
        bundle = tiny_factory()
        optimizer, scheduler = ckpt.make_optimizer(bundle, plain_protocol("bf16"))
        for _ in range(2):
            trainer.run_update(bundle, optimizer, scheduler, [encoded("a", route=ROUTE_FULL)], 0)
        assert_base_unchanged(bundle)
        with torch.no_grad():
            bundle.model.get_base_model().model.visual.patch_embed.proj.weight.add_(1)
        with self.assertRaises(ValueError):
            assert_base_unchanged(bundle)

    def test_save_and_resume_same_tensors(self):
        files = RunFiles(self.tmp, groups=4, empty_groups=(1,))
        self.assertEqual(files.train("whole", "note", "--save-every", "1"), 0)
        whole = self.tmp / "whole"
        self.assertEqual(sorted(p.name for p in whole.glob("checkpoint-*")),
                         ["checkpoint-1", "checkpoint-2", "checkpoint-3", "checkpoint-final"])
        # A crash after three updates, with a half-written checkpoint and a record past the last good one.
        cut = self.tmp / "cut"
        cut.mkdir()
        for name in ("binding.json", "initial_state.json", "updates.jsonl", "drops_local.jsonl"):
            shutil.copy(whole / name, cut / name)
        with (cut / "updates.jsonl").open("a") as stream:
            stream.write('{"step": 4, "gro')
        shutil.copytree(whole / "checkpoint-2", cut / "checkpoint-2")
        (cut / "checkpoint-3.tmp").mkdir()
        self.assertEqual(ckpt.read_metadata(cut / "checkpoint-2", binding=json.loads((cut / "binding.json").read_text()))["cursor"]["position"], 3)
        with self.assertRaises(SystemExit):  # no silent restart over an existing run
            files.train("cut", "note")
        self.assertEqual(files.train("cut", "note", "--save-every", "1", "--resume"), 0)
        binding = json.loads((whole / "binding.json").read_text())
        first = ckpt.read_metadata(whole / "checkpoint-final", binding=binding)
        second = ckpt.read_metadata(cut / "checkpoint-final", binding=binding)
        self.assertEqual(second["state_sha256"], first["state_sha256"])
        self.assertEqual((second["step"], second["cursor"]), (first["step"], first["cursor"]))
        whole_result, cut_result = (json.loads((d / "result.json").read_text()) for d in (whole, cut))
        self.assertEqual((cut_result["updates"], cut_result["skipped_updates"]), (whole_result["updates"], whole_result["skipped_updates"]))
        self.assertEqual(cut_result["rows_used"], whole_result["rows_used"])
        self.assertEqual(len((cut / "updates.jsonl").read_text().splitlines()), 4)
        with self.assertRaises(SystemExit):  # a complete run has nothing to resume
            files.train("cut", "note", "--resume")
        # A different arm, rows file or commit cannot continue the run.
        (cut / "result.json").unlink()
        with self.assertRaises(SystemExit):
            files.train("cut", "solution", "--resume")

    def test_reload_probe_detects_mismatch(self):
        files = RunFiles(self.tmp, groups=1)
        self.assertEqual(files.train("run", "note"), 0)
        run = self.tmp / "run"
        args = trainer.build_parser().parse_args(["reload", "--run", str(run), "--model-path", str(self.tmp), "--precision", "bf16"])
        reload = lambda: trainer.run_reload(args, policy_factory=tiny_factory, device="cpu")  # noqa: E731

        self.assertEqual(reload(), 0)
        report = json.loads((run / "reload.json").read_text())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["step"], 1)
        self.assertEqual(report["tensors"]["M"], 24)
        self.assertLessEqual(report["max_abs_delta"], 1e-3)
        self.assertTrue(report["greedy_equal"])

        original = (run / "probe.json").read_text()
        probe = json.loads(original)
        self.assertEqual(len(probe["logprobs"]), 2)
        self.assertLessEqual(len(probe["greedy_ids"]), 16)
        probe["logprobs"][0][0] += .01
        (run / "probe.json").write_text(json.dumps(probe))
        self.assertEqual(reload(), 1)
        self.assertEqual(json.loads((run / "reload.json").read_text())["status"], "FAIL")

        probe = json.loads(original)
        probe["greedy_ids"][0] = (probe["greedy_ids"][0] + 1) % 32
        (run / "probe.json").write_text(json.dumps(probe))
        self.assertEqual(reload(), 1)
        self.assertFalse(json.loads((run / "reload.json").read_text())["greedy_equal"])
        (run / "probe.json").write_text(original)

        # A wrong precision protocol is a FAIL, not a load.
        nf4 = trainer.build_parser().parse_args(["reload", "--run", str(run), "--model-path", str(self.tmp), "--precision", "nf4"])
        self.assertEqual(trainer.run_reload(nf4, policy_factory=tiny_factory, device="cpu"), 1)
        self.assertIn("protocol", json.loads((run / "reload.json").read_text())["error"])

    def test_nonfinite_stops(self):
        bundle = tiny_factory()
        optimizer, scheduler = ckpt.make_optimizer(bundle, plain_protocol("bf16"))
        before = hashes(bundle.params["L"])
        hook = bundle.params["L"][0].register_hook(lambda grad: grad * float("inf"))
        with self.assertRaises(trainer.NonFiniteStop):
            trainer.run_update(bundle, optimizer, scheduler, [encoded()], 0)
        hook.remove()
        self.assertEqual(hashes(bundle.params["L"]), before)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertFalse(optimizer.state)

        def poisoned(model_path, arm, protocol):
            built = tiny_factory(model_path, arm, protocol)
            with torch.no_grad():
                built.params["M"][0].fill_(float("nan"))
            return built
        files = RunFiles(self.tmp, groups=2)
        self.assertEqual(files.train("run", "note", factory=poisoned), 1)
        result = json.loads((self.tmp / "run/result.json").read_text())
        self.assertEqual((result["status"], result["step"], result["group"]), ("NONFINITE", 0, 0))
        self.assertFalse((self.tmp / "run/checkpoint-final").exists())
        self.assertEqual((self.tmp / "run/updates.jsonl").read_text(), "")

    def test_no_rows_is_not_a_pass(self):
        files = RunFiles(self.tmp, groups=2, empty_groups=(0, 1))
        self.assertEqual(files.train("run", "note"), 1)
        result = json.loads((self.tmp / "run/result.json").read_text())
        self.assertEqual((result["status"], result["updates"], result["skipped_updates"]), ("NO_UPDATES", 0, 2))

    def test_max_row_tokens_drops_with_a_record(self):
        files = RunFiles(self.tmp, groups=1)
        self.assertEqual(files.train("run", "note", "--max-row-tokens", "12"), 0)
        drops = [json.loads(line) for line in (self.tmp / "run/drops_local.jsonl").read_text().splitlines()]
        self.assertEqual([(d["group"], d["row_id"], d["reason"]) for d in drops], [(0, "item-0:B", "OVER_MAX_ROW_TOKENS")])
        self.assertEqual(json.loads((self.tmp / "run/result.json").read_text())["rows_by_kind"], {KIND_SOLUTION: 1})

    def test_quantized_forbidden_accepts_self_note_contract(self):
        protocol = plain_protocol("nf4")
        self.assertEqual(protocol["forbidden"], {"additional_sft": False, "teacher_or_upper_model_labels": True,
                                                 "benchmark_training": True})
        _validate_protocol(self.tmp, ARM, protocol, 0, require_snapshot=False)
        rl = {**protocol, "forbidden": {"additional_sft": True, "teacher_or_upper_model_labels": True, "benchmark_training": True}}
        _validate_protocol(self.tmp, ARM, rl, 0, require_snapshot=False)
        for tampered in ({"teacher_or_upper_model_labels": False}, {"benchmark_training": False},
                         {"teacher_or_upper_model_labels": False, "benchmark_training": False}):
            bad = {**protocol, "forbidden": {**protocol["forbidden"], **tampered}}
            with self.assertRaisesRegex(ValueError, "forbidden"):
                _validate_protocol(self.tmp, ARM, bad, 0, require_snapshot=False)
            with self.assertRaisesRegex(ValueError, "forbidden"):
                _validate_protocol(self.tmp, ARM, {**rl, "forbidden": {**rl["forbidden"], **tampered}}, 0, require_snapshot=False)

    def test_binding_is_the_checkpoint_contract(self):
        files = RunFiles(self.tmp, groups=1)
        self.assertEqual(files.train("run", "note"), 0)
        binding = json.loads((self.tmp / "run/binding.json").read_text())
        protocol = plain_protocol("bf16")
        self.assertEqual(binding, {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
                                   "data_sha256": hashlib.sha256(files.rows.read_bytes()).hexdigest(),
                                   "code_sha256": hashlib.sha256(COMMIT.encode()).hexdigest()})
        self.assertEqual(ckpt.read_metadata(self.tmp / "run/checkpoint-final", binding=binding)["protocol"], protocol)

    def test_cli_refusals(self):
        files = RunFiles(self.tmp, groups=1)
        with self.assertRaises(SystemExit):
            trainer.build_parser().parse_args(["train", "--rows", "r", "--build-root", "b", "--order", "o", "--limit", "1",
                                               "--arm", "note", "--precision", "bf16", "--model-path", "m", "--output", "d",
                                               "--code-commit", "short"])
        args = files.args("run", "note", "--gradient-checkpointing", "off")
        args.precision = "nf4"
        with self.assertRaisesRegex(SystemExit, "bf16"):
            trainer.run_train(args, policy_factory=tiny_factory, device="cpu")
        args = files.args("run", "note")
        args.limit = files.count + 1
        with self.assertRaisesRegex(ValueError, "limit"):
            trainer.run_train(args, policy_factory=tiny_factory, device="cpu")


if __name__ == "__main__":
    unittest.main()
