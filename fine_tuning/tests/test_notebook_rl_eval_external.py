"""Authored fake-engine arithmetic and real isolated verifier; no GPU claims."""
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from fine_tuning.notebook_rl.engines import Completion, RolloutResult
from fine_tuning.notebook_rl.evaluate_external import evaluate, pass_at_k, summarize
from fine_tuning.notebook_rl.schema import load_protocol
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.tests.test_notebook_rl_schema import records


class Characters:
    unk_token_id = -1

    def convert_tokens_to_ids(self, text):
        return 151645

    def encode(self, text, **kwargs):
        return [ord(c)+256 for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i-256) for i in ids if i != 151645)


class Engine:
    supported_sampling = frozenset(("qwen_official", "temperature_1"))
    policy_version = 0
    tokenizer = Characters()

    def __init__(self, model_dir, *, missing=False, broken=False, mutate=False):
        self.model_dir = model_dir
        self.calls = []
        self.missing, self.broken, self.mutate = missing, broken, mutate

    def prepare(self, request):
        if request.prompt_token_ids is not None:
            ids = request.prompt_token_ids
        else:
            ids = self.tokenizer.encode(json.dumps(request.messages, ensure_ascii=False))
        return SimpleNamespace(token_ids=list(ids))

    def sample_with_settings(self, requests, sampling):
        request = requests[0]
        self.calls.append((request, sampling, [im.getextrema() for im in request.images]))
        if self.broken:
            raise RuntimeError("authored engine failure")
        if self.mutate:
            self.policy_version += 1
        pid = request.request_id.split(":")[0]
        condition = request.request_id.split(":")[4]
        n = request.n-int(self.missing)
        cs = []
        for index in range(n):
            parent_index = int(request.request_id.split(":")[-1])
            if condition == "forced_second":
                correct = ([True, True, False, True] if pid == "mcq" else [False, True, True, True])[parent_index]
            elif condition == "feedback_second":
                correct = parent_index in ((1, 2) if pid == "mcq" else (1,))
            elif condition == "blind":
                correct = False
            elif sampling.name == "temperature_1":
                correct = pid == "open" and index < 2
            else:
                correct = index == 0 if pid == "mcq" else index in (0, 2)
            answer = ("A" if correct else "B") if pid == "mcq" else ("7" if correct else "8")
            text = ("" if pid == "mcq" else "Observed facts:\ncell[a][b] = "+answer+"\nSolution:\n") + "Final answer: "+answer
            ids = self.tokenizer.encode(text)+[151645]
            cs.append(Completion(ids, [-.1]*len(ids), "stop", text))
        return [RolloutResult(request.request_id, self.prepare(request).token_ids, cs)]


def cell(summary, mode, sampler="qwen_official", kind="*"):
    return next(c for c in summary["cells"] if c["mode"] == mode and c["sampler"] == sampler
                and c["family"] == c["prompt_mode"] == "*" and c["answer_kind"] == kind)


class ExternalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="external-eval-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / "authored-model"
        self.model.mkdir()
        self.split = self.root / "dev"
        image_root = self.root / "images"
        image_root.mkdir()
        raw = io.BytesIO()
        Image.new("RGB", (9, 7), (20, 30, 40)).save(raw, format="PNG")
        (image_root / "a.png").write_bytes(raw.getvalue())
        a, v, lineage, *_ = records()
        self.actors = [replace(a, problem_id="open", image_paths=("a.png",),
            image_sha256=(hashlib.sha256(raw.getvalue()).hexdigest(),))]
        self.actors.append(replace(self.actors[0], problem_id="mcq", mode="plain", answer_format="mcq",
                                  choices=tuple(f"choice-{i}" for i in range(10))))
        verifiers = [replace(v, problem_id="open"), replace(v, problem_id="mcq", answer_type="mcq", gold="A",
            choices=self.actors[1].choices, slots={}, slot_types={}, required_slots=(), oracle_fact_lines=(), printed_text=())]
        write_split(self.split, self.actors, verifiers, [replace(lineage, problem_id=a.problem_id, split="dev") for a in self.actors])
        self.verifier = Verifier(self.split)
        self.protocol = load_protocol(Path(__file__).parents[1] / "notebook_rl/protocol_v1.yaml")

    def run_eval(self, modes=("first",), *, engine=None, name="eval", **kwargs):
        engine = engine or Engine(self.model)
        summary = evaluate(self.model, self.split, self.root / name, modes, engine=engine,
            protocol=self.protocol, _test_verifier=self.verifier, **kwargs)
        rows = [json.loads(line) for line in (self.root / name / "results.jsonl").read_text().splitlines()]
        return summary, rows, engine

    def test_pass_at_k_handwritten_values_and_invalid_counts(self):
        self.assertEqual(pass_at_k(8, 2, 1), .25)
        self.assertAlmostEqual(pass_at_k(8, 2, 4), 55/70)
        self.assertEqual(pass_at_k(8, 2, 8), 1)
        self.assertEqual(pass_at_k(8, 0, 8), 0)
        for args in ((8, 9, 1), (8, 1, 9), (8, 1, 0), (True, 1, 1)):
            with self.assertRaises(ValueError):
                pass_at_k(*args)

    def test_all_modes_first_macro_pass_at_k_pairing_and_elimination(self):
        summary, rows, engine = self.run_eval(("first", "pass_at_k", "forced_second", "feedback_second", "blind"))
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertEqual(summary["primary_first_attempt"][0]["accuracy"], .375)
        self.assertEqual(summary["primary_first_attempt"][0]["n_expected"], 8)
        self.assertEqual(cell(summary, "first", "temperature_1")["accuracy"], .25)
        passes = cell(summary, "pass_at_k", "temperature_1")["pass_at_k"]
        self.assertEqual(passes["1"]["value"], .125)
        self.assertAlmostEqual(passes["4"]["value"], 55/140)
        self.assertEqual(passes["8"]["value"], .5)
        self.assertEqual(cell(summary, "pass_at_k", "temperature_1", "mcq")["pass_at_k"]["8"]["value"], 0)
        paired = cell(summary, "forced_second")["paired"]
        self.assertAlmostEqual(paired["i_to_c"]["value"], 5/6)
        self.assertEqual(paired["i_to_c"]["denominator"], 5)
        self.assertEqual(paired["i_to_c"]["numerator"], 4)
        self.assertEqual(paired["c_to_i"]["value"], .25)
        self.assertEqual(paired["c_to_i"]["denominator"], 3)
        feedback = cell(summary, "feedback_second")
        self.assertAlmostEqual(feedback["accuracy"], 7/12)
        self.assertEqual(feedback["n_expected"], 5)
        self.assertEqual(feedback["n_inapplicable"], 3)
        self.assertAlmostEqual(feedback["paired"]["mcq_elimination_baseline"]["value"], 1/9)
        self.assertEqual(feedback["paired"]["mcq_elimination_baseline"]["denominator"], 3)
        self.assertEqual(cell(summary, "blind")["accuracy"], 0)
        self.assertEqual(cell(summary, "first")["required_slot_accuracy"]["value"], .5)
        self.assertEqual(sum(r["mode"] == "first" and r["sampler"] == "temperature_1" and "reused_from" in r for r in rows), 8)
        self.assertFalse(summary["gpu_execution_certified"])
        self.assertEqual(summary["sampling_cost_counts"]["returned_completions"], 45)
        self.assertTrue(engine.calls)

    def test_literal_history_neutral_review_no_gold_and_same_seed_blind_images(self):
        _, rows, engine = self.run_eval(("first", "forced_second", "feedback_second", "blind"))
        by_id = {r["record_id"]: r for r in rows}
        for row in rows:
            if row["mode"] == "forced_second":
                parent = by_id[row["parent_record_id"]]
                expected = parent["prompt_token_ids"]+parent["completion_token_ids"]
                self.assertEqual(row["prompt_token_ids"][:len(expected)], expected)
                suffix = engine.tokenizer.decode(row["prompt_token_ids"][len(expected):])
                self.assertIn("Review your solution and give a final answer", suffix)
                self.assertNotIn("INCORRECT", suffix)
                self.assertNotIn("CORRECT", suffix)
            if row["mode"] == "first" and row["sampler"] == "qwen_official":
                blind = next(r for r in rows if r["mode"] == "blind" and r["problem_id"] == row["problem_id"] and r["k"] == row["k"])
                self.assertEqual(blind["seed"], row["seed"])
        blind_calls = [r for r, _, _ in engine.calls if ":blind:" in r.request_id]
        for r in blind_calls:
            # Images are closed after use; bounded metadata was recorded before closing.
            self.assertEqual(r.images[0].size, (9, 7))
        self.assertTrue(all(extrema == [((128, 128),)*3] for r, _, extrema in engine.calls if ":blind:" in r.request_id))
        serialized = json.dumps([r["request"] for r in rows], ensure_ascii=False)
        for private in ("required_slots", "oracle_fact_lines", "verifier_private", '"gold"', '"accept"'):
            self.assertNotIn(private, serialized)

    def test_missing_system_and_unverifiable_are_incomplete_not_zero(self):
        for kind in ("missing", "broken", "unverifiable"):
            engine = Engine(self.model, missing=kind == "missing", broken=kind == "broken")
            original = self.verifier.grade
            if kind == "unverifiable":
                self.verifier.grade = lambda pid, text, finish, mode: replace(
                    original(pid, text, finish, mode), status="UNVERIFIABLE", reward_answer=0.)
            summary, rows, _ = self.run_eval(engine=engine, name=kind)
            self.verifier.grade = original
            self.assertEqual(summary["status"], "INCOMPLETE")
            self.assertIsNone(cell(summary, "first")["accuracy"])
            self.assertTrue(any(r["correct"] is None and r["applicable"] for r in rows))
            self.assertEqual(cell(summary, "first")["n_expected"], 8)

    def test_incomplete_pass_at_k_does_not_guess_missing_outcomes(self):
        summary, _, _ = self.run_eval(("pass_at_k",), engine=Engine(self.model, missing=True))
        metric = cell(summary, "pass_at_k", "temperature_1")["pass_at_k"]["4"]
        self.assertIsNone(metric["value"])
        self.assertEqual(metric["n_complete_trials"], 0)
        self.assertEqual(metric["n_expected_trials"], 2)

    def test_invalid_finalization_is_scored_but_does_not_eliminate_an_mcq_choice(self):
        engine = Engine(self.model)
        original = engine.sample_with_settings
        def invalid_first(requests, sampling):
            results = original(requests, sampling)
            if ":first:" in requests[0].request_id:
                text = "I have not submitted a final answer."
                ids = engine.tokenizer.encode(text)+[151645]
                results[0].completions[:] = [Completion(ids, [-.1]*len(ids), "stop", text) for _ in results[0].completions]
            return results
        setattr(engine, "sample_with_settings", invalid_first)
        summary, _, _ = self.run_eval(("first", "feedback_second"), engine=engine)
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertEqual(cell(summary, "first")["accuracy"], 0)
        baseline = cell(summary, "feedback_second")["paired"]["mcq_elimination_baseline"]
        self.assertEqual(baseline["value"], .1)
        self.assertEqual(baseline["denominator"], 4)

    def test_policy_change_stops_further_sampling_and_extra_outputs_are_preserved(self):
        engine = Engine(self.model, mutate=True)
        summary, rows, _ = self.run_eval(engine=engine, name="changed")
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertTrue(all(r["correct"] is None for r in rows))
        raw = [json.loads(line) for line in (self.root / "changed/sampler_calls.jsonl").read_text().splitlines()]
        self.assertEqual(len(raw), 1)
        self.assertEqual(len(raw[0]["results"][0]["completions"]), 4)
        engine = Engine(self.model)
        original = engine.sample_with_settings
        def extra(requests, sampling):
            results = original(requests, sampling)
            results[0].completions.append(results[0].completions[0])
            return results
        setattr(engine, "sample_with_settings", extra)
        summary, rows, _ = self.run_eval(engine=engine, name="extra")
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertEqual(cell(summary, "first")["n_expected"], 8)
        self.assertEqual(cell(summary, "first")["n_scored"], 0)
        self.assertEqual(sum(not r["planned"] for r in rows), 4)

    def test_truncated_completion_is_zero_not_a_missing_system_failure(self):
        engine = Engine(self.model)
        original = engine.sample_with_settings
        def truncated(requests, sampling):
            results = original(requests, sampling)
            results[0].completions[:] = [replace(c, finish_reason="length") for c in results[0].completions]
            return results
        setattr(engine, "sample_with_settings", truncated)
        summary, _, _ = self.run_eval(engine=engine)
        self.assertEqual(summary["status"], "COMPLETE")
        first = cell(summary, "first")
        self.assertEqual(first["accuracy"], 0)
        self.assertEqual(first["truncation_rate"], 1)

    def test_item_weighting_not_response_pooling_and_duplicate_rejection(self):
        _, rows, _ = self.run_eval()
        official = [r for r in rows if r["sampler"] == "qwen_official"]
        # Four MCQ answers at .25, one open correct answer at1 =>(.25+1)/2, not2/5.
        selected = [r for r in official if r["problem_id"] == "mcq"] + [next(r for r in official if r["problem_id"] == "open")]
        summary = summarize(selected)
        self.assertEqual(cell(summary, "first")["observed_accuracy"], .625)
        self.assertIsNone(cell(summary, "first")["accuracy"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize(selected+selected[:1])

    def test_fixed_seed_replay_output_write_once_and_adapter_refusal(self):
        one, rows1, engine1 = self.run_eval(seeds=(3407, 3408))
        two, rows2, engine2 = self.run_eval(name="again", seeds=(3407, 3408))
        self.assertEqual(one, two)
        self.assertEqual(rows1, rows2)
        self.assertEqual([(r.seeds, s.name) for r, s, _ in engine1.calls], [(r.seeds, s.name) for r, s, _ in engine2.calls])
        with self.assertRaises(FileExistsError):
            self.run_eval()
        engine = Engine(self.model)
        engine.supported_sampling = frozenset(("temperature_1",))
        with self.assertRaisesRegex(ValueError, "sampling-capable"):
            self.run_eval(engine=engine, name="unsupported")
        self.assertFalse((self.root / "unsupported").exists())
        with self.assertRaises(ValueError):
            self.run_eval(name="train-claimed", split="train")

    def test_real_fresh_verifier_service_smoke_without_private_actor_reads(self):
        summary = evaluate(self.model, self.split, self.root / "service-eval", ("first",),
            engine=Engine(self.model), protocol=self.protocol)
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertEqual(summary["primary_first_attempt"][0]["accuracy"], .375)
        metadata = json.loads((self.root / "service-eval/verifier_manifest.json").read_text())
        self.assertTrue(metadata["python_guards_installed"])
        self.assertFalse(metadata["os_sandbox"])


if __name__ == "__main__":
    unittest.main()
