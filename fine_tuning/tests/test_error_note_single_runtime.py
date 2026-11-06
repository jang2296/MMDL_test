"""Authored CPU runtime receipts; no pretrained model, GPU, or real source-review evidence."""
from __future__ import annotations

import copy
from contextlib import nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from fine_tuning.error_note import audit, rl, single
from fine_tuning.error_note.collect import WAIT, single_request_binding, single_request_sha256
from fine_tuning.error_note.protocol import SINGLE_SAMPLING, plain_protocol_single, single_blocks
from fine_tuning.tests.test_error_note_single_audit import record_for
from fine_tuning.tests.test_error_note_single_loss import bind_reward_fixture, block, pin_reward_fixture
from fine_tuning.tests.test_error_note_train import tiny_factory
from fine_tuning.tests.test_error_note_v3_audit import PROVENANCE, example


def protocol(count=219):
    return plain_protocol_single("bf16", count, contracts={k: "a"*64 for k in ("source", "dataset", "order", "data_admission")},
                                 token_scale=1024.)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        model = torch.nn.Linear(2, 2)
        self.owner = SimpleNamespace(model=model, params={"L": list(model.parameters())})
        self.protocol = protocol()
        self.actors = {f"p{i}": SimpleNamespace(problem_id=f"p{i}", source="authored-test", mode="plain",
                                               question=f"Authored question {i}", choices=("red", "blue"),
                                               image_paths=("fixture.png",), image_sha256=("a"*64,)) for i in range(10)}
        self.items = [{"problem_id": pid, "subject": "Biology", "source": "authored-test"} for pid in self.actors]
        self.seen, self.gold_requests = [], []

    def stream(self, requests, stats, stop_after):
        for request in requests:
            self.seen.append(copy.deepcopy({k: v for k, v in request.items() if k != "images"}))
            stage, pid = request["stage"], request["problem_id"]
            text = (example(misread=True)["note"]["text"] if stage == "note" else
                    " ".join(f"step{j}" for j in range(38)) + " Final answer: " +
                    ("A" if pid == "p1" and stage == "t1" else "B"))
            actions = list(range(100, 140)) + [2]
            gen = single_request_binding(request) | {
                "text": text, "policy_sha256": rl.policy_sha256(self.owner), "protocol_sha256": self.protocol["sha256"],
                "sampling": SINGLE_SAMPLING, "sampling_sha256": rl.sampling_sha256(SINGLE_SAMPLING),
                "request_sha256": single_request_sha256(request), "input_sha256": "b"*64,
                "prompt_token_ids": [4, 5], "completion_token_ids": actions,
                "sampling_logprobs": [-.5]*len(actions), "support_sha256": ["c"*64]*len(actions),
                "finish_reason": "stop", "eos_token_ids": [2], "completion_count": 1}
            yield request, {"text": text, "finish_reason": "stop", "completion_count": 1,
                            "completion_tokens": len(actions), "prompt_tokens": 2, "seconds": .1, "generation": gen}

    def run_block(self, *, limit=2, **kwargs):
        def grade(pid, text, finish, mode):
            self.assertEqual(mode, "plain")
            return ("CORRECT" if text.endswith("B") else "INCORRECT", text[-1])

        def gold(pid):
            self.gold_requests.append(pid)
            return "B"

        return single.run_single_block(self.owner, actors=self.actors, items=self.items,
                                       provenance={pid: PROVENANCE for pid in self.actors}, gold=gold, grade=grade,
                                       load_images=lambda actor: [Image.new("RGB", (8, 8))], protocol=self.protocol,
                                       output=self.out, limit=limit, **({"stream": self.stream} | kwargs))

    def leaking(self, lesson):
        original = self.stream

        def authored(requests, stats, stop_after):
            for request, output in original(requests, stats, stop_after):
                if request["stage"] == "note":
                    text = example(misread=True, lesson=lesson)["note"]["text"]
                    output["text"] = output["generation"]["text"] = text
                yield request, output
        return authored

    def test_program_gate_runs_without_pauses_keeps_originals_and_gold_only_in_note_request(self):
        result = self.run_block()
        self.assertEqual(result["status"], "ROWS_READY")
        self.assertEqual([(r["problem_id"], r["stage"]) for r in self.seen],
                         [("p0", "t1"), ("p1", "t1"), ("p1", "note"), ("p1", "resolve")])
        self.assertEqual(set(self.gold_requests), {"p1"})
        closed = json.loads((self.out / result["rows_path"]).read_text())
        self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "A1", "RL", "A2", "B"])
        for row in closed["rows"]:
            self.assertEqual(row["target"], row["generation"]["text"])
            if row["kind"] in ("RL", "A1", "A2"):
                self.assertNotIn("Lesson:", row["prompt"])
        self.assertEqual([row["route"] for row in closed["rows"] if row["kind"] in ("A2", "B")], ["FULL", "FULL"])
        original_wrong = next(row for row in closed["rows"] if row["kind"] == "RL" and row["problem_id"] == "p1")
        self.assertEqual(original_wrong["reward"], 0)
        self.assertNotIn("review", original_wrong["reward_evidence"])
        self.assertTrue(closed["note_decisions"]["p1"]["accepted"])
        self.assertTrue(closed["row_decisions"]["p1:A2"]["accepted"])
        self.assertEqual(closed["contract"]["length_policy"]["content_acceptance"], "program_checks_only_content_unverified")
        retry = next(r for r in self.seen if r["stage"] == "resolve")
        prompt = retry["messages"][0]["content"][-1]["text"]
        self.assertNotIn("Correct answer", prompt)
        self.assertNotIn("What went wrong", prompt)
        self.assertIn("Read each label", prompt)
        self.assertEqual(self.run_block()["status"], "ROWS_READY")
        self.assertEqual(len(self.seen), 4)  # a completed block never generates again

    def test_v7_note_cap_resolve_instruction_and_a2_b_program_filters(self):
        from fine_tuning.error_note.contract import resolve_prompt_single
        from fine_tuning.notebook_rl.prompts import p0_eval_text

        self.run_block()
        caps = {r["stage"]: r["max_tokens"] for r in self.seen}
        self.assertEqual(caps, {"t1": 8192, "note": 1024, "resolve": 8192})
        retry = next(r for r in self.seen if r["stage"] == "resolve")
        self.assertEqual(retry["messages"][0]["content"][-1]["text"], resolve_prompt_single(
            p0_eval_text("Authored question 1", ("red", "blue")), "Read each label before comparing the curves."))
        root, original = self.out, self.stream
        for name, stage, text, kinds, reason in (
                ("echo", "resolve", "As per the lesson, read each label. " + " ".join(f"s{j}" for j in range(38))
                 + " Final answer: B", ["RL", "A1", "RL"], "A2_RESOLVE_LEAK:lesson"),
                ("hedge", "resolve", "Based solely on the image " + " ".join(f"s{j}" for j in range(38))
                 + " Final answer: B", ["RL", "A1", "RL"], "A2_IMAGE_ONLY_HEDGE"),
                ("denial", "note", None, ["RL", "A1", "RL", "A2"], "B_ANSWER_DENIAL:mislabel")):
            self.out, self.seen = root / name, []

            def authored(requests, stats, stop_after, stage=stage, text=text):
                for request, output in original(requests, stats, stop_after):
                    if request["stage"] == stage:
                        new = text or output["text"].replace("confused two labels.", "confused two labels; a mislabel.")
                        output["text"] = output["generation"]["text"] = new
                    yield request, output

            with self.subTest(name=name), patch.object(self, "stream", new=authored):
                state = self.run_block()
                closed = json.loads((self.out / state["rows_path"]).read_text())
                self.assertEqual([r["kind"] for r in closed["rows"]], kinds)
                key = "p1:B" if stage == "note" else "p1:A2"
                self.assertEqual(closed["row_decisions"][key], {"accepted": False, "method": "program", "reason": reason})
                if stage != "note":
                    self.assertNotIn("p1:B", closed["row_decisions"])  # B follows its A2

    def test_fixed2_then10_reuses_first2_and_only_adds_next8(self):
        self.assertEqual(self.run_block()["status"], "ROWS_READY")
        result = self.run_block(limit=10)
        self.assertEqual(result["status"], "ROWS_READY")
        self.assertEqual(sum(r["stage"] == "t1" for r in self.seen), 10)
        self.assertEqual(sum(r["stage"] == "note" for r in self.seen), 1)
        self.assertEqual(sum(r["stage"] == "resolve" for r in self.seen), 1)
        self.assertEqual(len(self.seen), 12)
        with self.assertRaisesRegex(ValueError, "backwards"):
            self.run_block(limit=2)

    def test_leaking_lesson_never_admits_retry_but_keeps_original_rl_row(self):
        for lesson in ("Pick b next time.", "Choose the second choice.", "The answer was blue."):
            self.out, self.seen = Path(self.tmp.name) / str(len(lesson)), []
            with patch.object(self, "stream", new=self.leaking(lesson)):
                state = self.run_block()
            self.assertEqual(state["status"], "ROWS_READY")
            self.assertFalse(any(r["stage"] == "resolve" for r in self.seen), lesson)
            closed = json.loads((self.out / state["rows_path"]).read_text())
            self.assertFalse(closed["note_decisions"]["p1"]["accepted"])
            self.assertIn("LEAK", closed["note_decisions"]["p1"]["reason"])
            self.assertEqual([row["kind"] for row in closed["rows"]], ["RL", "A1", "RL"])

    def test_long_or_repetitive_correct_t1_stays_original_with_zero_reward(self):
        root, original_stream = self.out, self.stream
        for actions, reason in ((list(range(100, 4196)) + [2], "TOO_LONG"), ([11] * 36 + [2], "REPETITION")):
            self.out, self.seen = root / reason, []

            def authored(requests, stats, stop_after):
                for request, output in original_stream(requests, stats, stop_after):
                    if request["problem_id"] == "p0" and request["stage"] == "t1":
                        output["completion_tokens"] = len(actions)
                        output["generation"].update(completion_token_ids=actions, sampling_logprobs=[-.5] * len(actions),
                                                    support_sha256=["c" * 64] * len(actions))
                    yield request, output

            with patch.object(self, "stream", new=authored):
                state = self.run_block()
            self.assertEqual(state["status"], "ROWS_READY")
            closed = json.loads((self.out / state["rows_path"]).read_text())
            row = closed["rows"][0]
            self.assertEqual((row["kind"], row["reward"]), ("RL", 0))
            self.assertIn(reason, row["reward_evidence"]["decision"]["reason"])
            self.assertEqual(row["generation"]["completion_token_ids"], actions)
            self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "RL", "A2", "B"])
            self.assertEqual(len(self.seen), 4)

    def continuous(self, events):
        """A WAIT-aware engine like collect.continuous_stream: up to 16 active, one completion per step (FIFO)."""
        def stream(requests, stats, stop_after):
            iterator, active, done = iter(requests), [], False

            def fill():
                nonlocal done
                while not done and len(active) < 16:
                    item = next(iterator, None)
                    if item is None:
                        done = True
                    elif item is WAIT:
                        return
                    else:
                        events.append(("submit", item["request_id"]))
                        active.append(item)
            fill()
            while active:
                item = active.pop(0)
                events.append(("complete", item["request_id"]))
                yield from self.stream([item], stats, stop_after)
                fill()
        stream.continuous = True
        return stream

    def test_continuous_engine_starts_a_note_before_other_first_solves_end_with_identical_rows(self):
        staged = self.run_block(limit=10)
        staged_rows = json.loads((self.out / staged["rows_path"]).read_text())["rows"]
        self.out, self.seen, events = Path(self.tmp.name) / "pipelined", [], []
        with patch.object(self, "stream", wraps=self.stream):
            state = self.run_block(limit=10, stream=self.continuous(events))
        self.assertEqual(state["status"], "ROWS_READY")
        self.assertEqual(json.loads((self.out / state["rows_path"]).read_text())["rows"], staged_rows)
        self.assertLess(events.index(("submit", "p1:note")), events.index(("complete", "p9:t1")))
        self.assertEqual(len(self.seen), 12)
        self.assertEqual(self.run_block(limit=10, stream=self.continuous(events))["status"], "ROWS_READY")
        self.assertEqual(len(self.seen), 12)  # a rerun reuses every completion, including follow-ups

    def test_unverifiable_t1_keeps_rl_row_with_zero_reward_and_gets_no_gold_note(self):
        def grade(pid, text, finish, mode):
            return ("UNVERIFIABLE", None) if pid == "p1" else ("CORRECT", "B")

        state = single.run_single_block(self.owner, actors=self.actors, items=self.items,
                                        provenance={pid: PROVENANCE for pid in self.actors},
                                        gold=lambda pid: self.gold_requests.append(pid) or "B", grade=grade,
                                        load_images=lambda actor: [Image.new("RGB", (8, 8))], protocol=self.protocol,
                                        output=self.out, limit=2, stream=self.stream)
        self.assertEqual(state["status"], "ROWS_READY")
        self.assertEqual([r["stage"] for r in self.seen], ["t1", "t1"])
        self.assertEqual(self.gold_requests, [])
        closed = json.loads((self.out / state["rows_path"]).read_text())
        self.assertEqual([r["kind"] for r in closed["rows"]], ["RL", "A1", "RL"])
        unverifiable = closed["rows"][2]
        self.assertEqual((unverifiable["status"], unverifiable["reward"]), ("UNVERIFIABLE", 0))
        self.assertEqual(unverifiable["reward_evidence"]["decision"]["reason"], "T1_UNVERIFIABLE")
        self.assertEqual(single.validate_single_reward(unverifiable)["accepted"], False)

    def test_concurrent_runtime_writer_stops_before_generation(self):
        with single._runtime_lock(self.out, ".runtime.lock"):
            with self.assertRaisesRegex(ValueError, "HOLD concurrent"):
                self.run_block()
        self.assertEqual(self.seen, [])
        self.assertFalse((self.out / "phase.json").exists())
        self.assertEqual(self.run_block()["status"], "ROWS_READY")

    def test_existing_python_guard_denies_private_vault_and_benchmark_before_generation(self):
        root, hub = self.out / "vault", self.out / "hub"
        private = root / "train" / "verifier_private.jsonl"
        private.parent.mkdir(parents=True)
        private.write_text("authored private gold")
        rules = self.out / "rules.json"
        rules.write_text("authored private rules")
        benchmark = hub / "datasets--MMMU--MMMU" / "fixture.json"
        benchmark.parent.mkdir(parents=True)
        benchmark.write_text("authored exclusion fixture")
        script = """
import pathlib, sys, types
from unittest.mock import patch
from fine_tuning.error_note.collect import install_training_guards
from fine_tuning.notebook_rl import quantized_policy, shared_smoke
root, rules, benchmark = map(pathlib.Path, sys.argv[1:])
# Only dependency preparation is authored here; the real unchanged file guard runs.
sys.modules['bitsandbytes'] = types.ModuleType('bitsandbytes')
with patch.object(quantized_policy, '_versions', return_value={}), \\
     patch.object(shared_smoke, '_prepare_hf_dependencies', return_value={}):
    forbidden = install_training_guards(root, private_files=[rules])
assert any(benchmark.is_relative_to(path) for path in forbidden)
for path in [root/'train'/'verifier_private.jsonl', rules, benchmark]:
    try: path.read_bytes()
    except PermissionError: pass
    else: raise AssertionError('forbidden file opened')
print('guard boundary PASS; no model loaded')
"""
        result = subprocess.run([sys.executable, "-B", "-c", script, str(root), str(rules), str(benchmark)],
                                env=os.environ | {"HF_HUB_CACHE": str(hub)}, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("guard boundary PASS", result.stdout)

    def test_duplicate_actors_and_changed_policy_rejected_before_generation(self):
        self.run_block()
        self.assertEqual(len(self.seen), 4)
        self.items[1] = self.items[0]
        with self.assertRaisesRegex(ValueError, "unique"):
            self.run_block()
        self.items[1] = {"problem_id": "p1", "subject": "Biology", "source": "authored-test"}
        with torch.no_grad():
            self.owner.model.weight.add_(1)
        with self.assertRaisesRegex(ValueError, "binding changed"):
            self.run_block()
        self.assertEqual(len(self.seen), 4)

    def test_prepare_binds_source_evidence_without_loading_model_and_rejects_changed_capacity(self):
        from fine_tuning.notebook_rl.schema import ActorItem, to_json
        from fine_tuning.error_note.answer_contract import VERSION as RULES_VERSION
        from fine_tuning.error_note.source_evidence import VERSION as SOURCE_VERSION

        actors = [ActorItem(a.problem_id, a.source, "fixture", "plain", a.question, a.choices, "mcq",
                            a.image_paths, a.image_sha256) for a in self.actors.values()]
        root = self.out / "vault"
        (root / "train").mkdir(parents=True)
        (root / "train" / "manifest.yaml").write_text("authored fixture train")
        (root / "manifest.yaml").write_text("authored fixture union")
        train_sha = single.file_sha(root / "train" / "manifest.yaml")
        provenance = {a.problem_id: PROVENANCE | {"actor_sha256": audit.text_sha256(to_json(a)),
                      "image_sha256": list(a.image_sha256), "vault_manifest_sha256": train_sha} for a in actors}
        source = self.out / "source.json"
        source.write_text(json.dumps({"version": SOURCE_VERSION, "items": provenance}))
        rules = self.out / "rules.json"
        rules.write_text(json.dumps({"version": RULES_VERSION, "vault_manifest_sha256": train_sha, "items": {}}))
        model = self.out / "model"
        model.mkdir()
        config = model / "config.json"
        config.write_text(json.dumps({"model_type": "qwen3_vl", "torch_dtype": "bfloat16"}))
        args = SimpleNamespace(build_root=root, source_provenance=source, training_rules=rules, order_sha256="a"*64,
                               model_path=model, output=self.out, mode="local-smoke", token_scale=1024.,
                               max_num_seqs=1, max_num_batched_tokens=24576, token_budget=0, stop_after_seconds=None)
        with patch.object(single, "_validate_data", return_value=({"items": self.items}, {"authored_fixture": True})), \
                patch("fine_tuning.notebook_rl.vault.read_actors", return_value=actors), \
                patch.object(single, "source_manifest", return_value={"base_commit": "a"*40, "actual_files": {}}), \
                patch.object(single, "generation_identity", return_value={"authored_fixture": True, "model_files_sha256": "b" * 64}):
            plan = single.prepare(args)
            self.assertFalse(plan["actual_model_execution"])
            self.assertFalse(plan["cloud_ready"])
            self.assertEqual(len(plan["fixed10"]), 10)
            self.assertEqual(plan, single.prepare(args))
            args.max_num_seqs = 2
            with self.assertRaisesRegex(ValueError, "binding changed"):
                single.prepare(args)
            args.max_num_seqs = 1
            config.write_text(json.dumps({"quantization_config": {"load_in_4bit": True}}))
            with self.assertRaisesRegex(ValueError, "raw BF16 snapshot"):
                single.prepare(args)
            config.write_text(json.dumps({"model_type": "qwen3_vl", "torch_dtype": "bfloat16"}))
            (model / "nf4_base.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "raw BF16 snapshot"):
                single.prepare(args)
            (model / "nf4_base.json").unlink()
            from fine_tuning.tests.test_error_note_capacity import authored_profile

            profile = authored_profile(self.owner)
            profile["binding"]["source_sha256"] = audit.digest({"base_commit": "a" * 40, "actual_files": {}})
            profile_path = self.out / "capacity.json"
            profile_path.write_text(json.dumps(profile))
            args.output, args.precision = self.out / "measured", "bf16"
            args.capacity_profile, args.capacity_profile_sha256 = profile_path, single.file_sha(profile_path)
            args.max_num_seqs = args.max_num_batched_tokens = args.token_budget = None
            measured = single.prepare(args)
            self.assertEqual(measured["generation"]["precision"], "bf16")
            self.assertEqual(measured["capacity"]["max_num_seqs"], 16)
            self.assertEqual(measured["capacity"]["training_token_budget"], 16 * 24576)
            self.assertEqual(measured["capacity"]["measured_profile"]["file_sha256"], args.capacity_profile_sha256)
            self.assertFalse(measured["cloud_ready"] or measured["actual_model_execution"])
            profile_path.write_bytes(profile_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                single.prepare(args)

            args.capacity_profile = args.capacity_profile_sha256 = None
            args.output, args.capture_capacity, args.capacity_reserve_bytes = self.out / "capture", self.out / "part1.json", 12345
            captured = single.prepare(args)
            self.assertEqual(captured["capacity"]["capture"], {"enabled": True, "reserve_bytes": 12345})
            args.capture_capacity = self.out / "part2.json"
            self.assertEqual(captured, single.prepare(args))
            args.capacity_reserve_bytes += 1
            with self.assertRaisesRegex(ValueError, "binding changed"):
                single.prepare(args)
            args.capture_capacity.write_text("original completed partial")
            with self.assertRaisesRegex(ValueError, "fresh output"):
                single.prepare(args)

    def test_cli_publishes_partial_at_review_wait_and_after_backward_hold_without_extra_generation(self):
        from fine_tuning.error_note import capacity
        from fine_tuning.tests.test_error_note_capacity import authored_profile

        profile = authored_profile(self.owner)
        capture = capacity.Capture.__new__(capacity.Capture)  # Authored publication fixture, never actual CUDA evidence.
        capture.binding, capture.reserve_bytes, capture.entries = profile["binding"], profile["reserve_bytes"], []
        capture.policy_sha256 = profile["calibration_policy_sha256"]
        source = self.out / "source.json"
        source.write_text(json.dumps({"items": {}}))
        args = SimpleNamespace(command="collect", output=self.out, build_root=self.out / "build", source_provenance=source,
            training_rules=self.out / "rules.json", note_reviews=None, solution_reviews=None, model_path=self.out / "model",
            precision="nf4", limit=2, token_budget=1024, stop_after_seconds=None,
            capture_capacity=self.out / "wait-partial.json", capacity_reserve_bytes=profile["reserve_bytes"])
        plan = {"source": {}, "generation": {"model_files_sha256": "b" * 64}, "capacity": {}, "protocol": self.protocol,
                "fixed10": self.items, "rules_sha256": "a" * 64}
        rows = [{"problem_id": "p0", "kind": kind, "route": "FULL" if kind in ("A2", "B") else "LM"}
                for kind in ("RL", "A1", "A2", "B")]
        (self.out / "block-0").mkdir()
        (self.out / "block-0" / "rows.json").write_text(json.dumps({"rows": rows, "contract": {}}))
        with patch.object(single, "prepare", return_value=plan), \
                patch("fine_tuning.notebook_rl.vault.read_actors", return_value=list(self.actors.values())), \
                patch("fine_tuning.notebook_rl.vault.read_verifier", return_value={pid: SimpleNamespace(gold="B") for pid in self.actors}), \
                patch("fine_tuning.notebook_rl.policy.enable_deterministic_execution"), \
                patch("fine_tuning.notebook_rl.quantized_policy.build_quantized_policy", return_value=self.owner), \
                patch("fine_tuning.notebook_rl.verifier_service.VerifierService"), \
                patch.object(single, "install_training_guards", return_value=[]), \
                patch.object(single, "ResourceLog", side_effect=lambda *args: nullcontext()), \
                patch.object(capacity, "Capture", return_value=capture), \
                patch.object(self.owner.model, "forward", side_effect=AssertionError("unexpected extra generation")), \
                patch.object(single, "run_single_block", return_value={"status": "WAIT_NOTE_REVIEW"}) as run, \
                patch.object(single, "prove_single_backward", return_value={"status": "HOLD", "optimizer_steps": 0}) as proof, \
                patch("fine_tuning.error_note.train_v4.encode_single_row", side_effect=lambda processor, config, row, images: row), \
                patch("fine_tuning.notebook_rl.dataset.load_images", return_value=[]), \
                patch.object(capacity, "capture_backward") as backward, redirect_stdout(io.StringIO()):
            self.assertEqual(single._main(args), 2)
            proof.assert_not_called()
            self.assertEqual(json.loads(args.capture_capacity.read_text())["status"], "UNCALIBRATED")
            args.command, args.capture_capacity = "dry-run", self.out / "backward-partial.json"
            self.owner.processor, self.owner.model.config = None, None
            run.return_value = {"status": "ROWS_READY", "rows_path": "rows.json"}
            self.assertEqual(single._main(args), 2)
            self.assertEqual(run.call_count, 2)
            self.assertEqual([call.kwargs["component"] for call in backward.call_args_list], ["RL", "A1", "A2", "B"])
            self.assertEqual(json.loads(args.capture_capacity.read_text())["status"], "UNCALIBRATED")
            self.assertEqual(json.loads((self.out / "dry-run-2.json").read_text())["status"], "HOLD")
            before = (self.out / "dry-run-2.json").read_bytes()
            args.capture_capacity = self.out / "failed-capture.json"
            backward.side_effect = torch.OutOfMemoryError("authored CPU capture failure")
            with self.assertRaises(torch.OutOfMemoryError):
                single._main(args)
            self.assertEqual((self.out / "dry-run-2.json").read_bytes(), before)
            self.assertEqual(json.loads(args.capture_capacity.read_text())["status"], "UNCALIBRATED_FAILED")
            args.output, args.capture_capacity = self.out / "proof-error", self.out / "proof-error-partial.json"
            (args.output / "block-0").mkdir(parents=True)
            (args.output / "block-0/rows.json").write_bytes((self.out / "block-0/rows.json").read_bytes())
            proof.return_value = {"status": "FAIL", "error": "authored backward error", "all_parameters_unchanged": True}
            backward.reset_mock()
            self.assertEqual(single._main(args), 2)
            backward.assert_not_called()
            self.assertEqual(json.loads((args.output / "dry-run-2.json").read_text())["status"], "FAIL")

    def test_solutions_need_32_tokens_but_notes_do_not(self):
        from fine_tuning.error_note.v4 import target_problem

        self.assertEqual(single.SOLUTION_MIN_TOKENS, 32)
        self.assertEqual(target_problem("B", [11], "stop", "CORRECT"), "TOO_SHORT")
        self.assertEqual(self.protocol["training_contract"]["length_policy"]["min_target_tokens"], 32)
        self.assertEqual(self.protocol["training_contract"]["answer_contracts"],
                         {"mcq": "p0-eval-aligned-v1", "open": "p0-eval-aligned-v1|training-answer-v2"})
        for kind in ("A1", "A2", "B"):
            record = record_for(correct=kind == "A1")
            stage = {"A1": "t1", "A2": "resolve", "B": "note"}[kind]
            if kind != "B":
                record[stage]["text"] = "B"
            for output in (record["t1"], record["note"], record["resolve"]):
                output["generation"] = {"text": output["text"], "completion_token_ids": [11]}
            row = single._candidate(record, kind)
            # A bare one-token answer earns no SFT row; a one-token note is a format question for the note gate.
            self.assertEqual(row is None, kind != "B", kind)
            if kind != "B":
                record[stage]["generation"]["completion_token_ids"] = list(range(100, 132))
                self.assertIsNotNone(single._candidate(record, kind), kind)

class UpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.short = patch.object(single, "SOLUTION_MIN_TOKENS", 1)  # tiny-model fixtures use short targets
        cls.short.start()
        cls.addClassCleanup(cls.short.stop)
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def owner(self):
        torch.manual_seed(3407)
        return tiny_factory(Path("unused"), "MOD-RV", protocol())

    def test_independent_no_update_backward_missing_kinds_is_hold_and_clears(self):
        owner = self.owner()
        rows, contract = block(owner)
        before = rl.policy_sha256(owner)
        report = single.prove_single_backward(owner, rows, contract, precision="bf16")
        self.assertEqual(report["status"], "HOLD")
        self.assertIn("A2", report["missing"])
        self.assertTrue(report["all_parameters_unchanged"])
        self.assertTrue(report["gradients_cleared"])
        self.assertEqual(rl.policy_sha256(owner), before)
        self.assertTrue(report["components"]["PG"]["stats"]["has_gradient"])
        self.assertTrue(report["components"]["KL"]["zero_F0_KL_allowed"])

    def test_independent_all_components_and_same_problem_pair_pass_without_update(self):
        owner = self.owner()
        rows, contract = block(owner)
        rows[1].update(reward=0, status="INCORRECT")
        bind_reward_fixture(rows[1])
        pin_reward_fixture(rows, contract)
        for kind, original in (("A1", rows[0]), ("A2", rows[1]), ("B", rows[1])):
            row = copy.deepcopy(original)
            row.update(kind=kind, row_id=original["problem_id"]+":"+kind,
                       route="LM" if kind == "A1" else "FULL", status="CORRECT")
            rows.append(row)
        report = single.prove_single_backward(owner, rows, contract, precision="bf16", token_budget=64)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["paired_incorrect_a2_b"], ["item-1"])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertTrue(report["all_parameters_unchanged"])

    def test_joint_step_explicit_optin_one_call_and_duplicate_cursor_rejected(self):
        owner = self.owner()
        rows, contract = block(owner)
        contract["total_slots"] = 219
        optimizer = torch.optim.AdamW([p for ps in owner.params.values() for p in ps], lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        ids = contract["expected_problem_ids"]
        cursor = {"slot": 18, "updates": 0, "consumed_problem_ids": [], "policy_sha256": rl.policy_sha256(owner)}
        with self.assertRaisesRegex(ValueError, "opt-in"):
            single.run_joint_single_update(owner, optimizer, scheduler, rows, contract, cursor=cursor, block_ids=ids)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            result = single.run_joint_single_update(owner, optimizer, scheduler, rows, contract, cursor=cursor,
                                                    block_ids=ids, allow_update=True)
        self.assertEqual(step.call_count, 1)
        self.assertEqual(result["cursor"]["slot"], 19)
        self.assertEqual(result["cursor"]["updates"], 1)
        with self.assertRaisesRegex(ValueError, "stale|duplicate"):
            single.run_joint_single_update(owner, optimizer, scheduler, rows, contract, cursor=cursor,
                                           block_ids=ids, allow_update=True)

    def test_zero_signal_consumes_slot_without_optimizer_decay(self):
        owner = self.owner()
        rows, contract = block(owner, reward=0)
        contract["total_slots"] = 219
        optimizer = torch.optim.AdamW([p for ps in owner.params.values() for p in ps], lr=.01, weight_decay=.2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        before = rl.policy_sha256(owner)
        cursor = {"slot": 0, "updates": 0, "consumed_problem_ids": [], "policy_sha256": before}
        result = single.run_joint_single_update(owner, optimizer, scheduler, rows, contract, cursor=cursor,
                                                block_ids=contract["expected_problem_ids"], allow_update=True)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(cursor["slot"], 1)
        self.assertEqual(rl.policy_sha256(owner), before)
        self.assertFalse(optimizer.state)

    def test_3500_schedule_closes_300_before_main_and_has219_slots(self):
        ids = [str(i) for i in range(3500)]
        blocks = single_blocks(ids)
        self.assertEqual(len(blocks), 219)
        self.assertEqual([len(x) for x in blocks[:19]], [16]*18+[12])
        self.assertEqual(blocks[19][0], "300")
        self.assertEqual([pid for b in blocks for pid in b], ids)

    def test_cpu_checkpoint_restores_optimizer_rng_and_continuation_cursor(self):
        from fine_tuning.notebook_rl import checkpoint

        owner = self.owner()
        proto = protocol()
        rows, contract = block(owner)
        contract["total_slots"] = 219
        optimizer, scheduler = checkpoint.make_optimizer(owner, proto)
        cursor = {"slot": 0, "updates": 0, "consumed_problem_ids": [], "policy_sha256": rl.policy_sha256(owner)}
        single.run_joint_single_update(owner, optimizer, scheduler, rows, contract, cursor=cursor,
                                       block_ids=contract["expected_problem_ids"], allow_update=True)
        binding = {"protocol_sha256": proto["sha256"], "prompts_sha256": proto["prompts_sha256"],
                   "data_sha256": "c"*64, "code_sha256": "d"*64}

        def factory(model, arm, protocol):
            torch.manual_seed(3407)
            return tiny_factory(model, arm, protocol)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint"
            with self.assertRaisesRegex(ValueError, "opt-in"):
                single.save_single_continuation(owner, optimizer, scheduler, cursor=cursor, path=path,
                                                protocol=proto, binding=binding)
            single.save_single_continuation(owner, optimizer, scheduler, cursor=cursor, path=path,
                                            protocol=proto, binding=binding, allow_save=True)
            rng = torch.get_rng_state().clone()
            restored, opt, sched, restored_cursor = single.load_single_continuation(
                Path("unused"), path, proto, binding=binding, policy_factory=factory)
            self.assertEqual(restored_cursor, cursor)
            self.assertEqual(sched.state_dict(), scheduler.state_dict())
            self.assertEqual(rl.policy_sha256(restored), rl.policy_sha256(owner))
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(len(opt.state), len(optimizer.state))
            for original, loaded in zip(optimizer.state.values(), opt.state.values(), strict=True):
                for key in original:
                    torch.testing.assert_close(loaded[key], original[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
