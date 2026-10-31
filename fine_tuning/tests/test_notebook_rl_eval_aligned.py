"""Opt-in p0-eval-aligned-v1 T1 contract; CPU fixtures, not Qwen behaviour."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import io
import json
import os
import re
import tempfile
import unittest
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from fine_tuning.notebook_rl import (answers, checkpoint, compact_note, local_profile, local_run, notebook_synthesis,
                                     prompts as builders, repair_evidence, rollout, v8_parser, verifier_service)
from fine_tuning.notebook_rl.generators import graph
from fine_tuning.notebook_rl.repair_contract import ORIGIN_VERSION, VERSION as RETRY_VERSION, public_origin_inventory
from fine_tuning.notebook_rl.repair_training import finalization_reward
from fine_tuning.notebook_rl.schema import ActorItem, Lineage, Sample, VerifierItem
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier import VERSION as VERIFIER_VERSION, Verifier
from fine_tuning.notebook_rl.verifier_service import VerifierService
from fine_tuning.tests.test_notebook_rl_scale import HISTORICAL_OPTIONS, HISTORICAL_SHA256
from mmdl.evaluation import final_answer_parser as src_base, final_answer_parser_v8 as src_v8
from mmdl.evaluation.prompt import build_messages, build_prompt

ROOT = Path(__file__).resolve().parents[2]
EVAL = answers.TURN1_EVAL_CONTRACT
SOURCE = "test-eval-aligned-v1"
PATHS, HASHES = ("img/0.png", "img/1.png"), ("a" * 64, "b" * 64)
B2 = ROOT / "analysis_exports/mmmu_20260924/team_final_v8/official_b2/predictions.jsonl"
UNIT_V8 = ROOT / "tests/unit/test_final_answer_parser_v8.py"
SNAPSHOT = (Path(os.environ.get("HF_HOME") or Path.home() / "mmdl-cache/huggingface")
            / "hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17")
ASSISTANT_START = "<|im_start|>assistant\n"


def flagged(**changes) -> dict:
    return local_profile.build_protocol(**({"repair_notebook": True, "context_limit": 24576,
        "note_contract": compact_note.VERSION, "answer_verified_route": True, "finalization_reward": True,
        "slotless_sources": (SOURCE,)} | changes))


def eval_flagged(**changes) -> dict:
    return flagged(turn1_contract=EVAL, **changes)


def make_actor(pid: str, question: str, choices: tuple[str, ...], answer_format: str, n_images: int = 1,
               mode: str = "obs", source: str = SOURCE) -> ActorItem:
    return ActorItem(pid, source, "chart_bar", mode, question, choices, answer_format,
                     PATHS[:n_images], HASHES[:n_images])


# name, question, choices, answer_format, image count: synthetic text only.
CASES = (
    ("mcq-two", "Which bar is taller?", ("left", "right"), "mcq", 1),
    ("mcq-ten", "Pick the printed label.", tuple(f"label {i}" for i in range(10)), "mcq", 1),
    ("mcq-braces", "Which value does <image 1> show for {x}?", ("{x}", "{y}", "{x}{y}", "{0}"), "mcq", 1),
    ("mcq-unicode", "Quelle valeur de α verifie β ≥ γ ? 日本語",
     ("α", "β ≤ γ", "日本語 ✓"), "mcq", 1),
    ("mcq-multiline", "First line\nSecond <image 1> line\n\nThird <image 2> line", ("a", "b", "c"), "mcq", 2),
    ("open-trailing", "How many bars are taller than 5?  \n", (), "int", 1),
    ("open-image", "Read <image 1> and give the total.", (), "int", 1),
    ("open-multiline", "Name the shape.\nBe brief.", (), "short_text", 2),
)

MCQ = make_actor(f"{SOURCE}:mcq:1", "How many bars are taller than 5?", ("2", "3", "4", "5"), "mcq")
INT = replace(MCQ, problem_id=f"{SOURCE}:int:2", answer_format="int", choices=())
BAD_GOLD = replace(INT, problem_id=f"{SOURCE}:int:3")
ITEMS = (VerifierItem(MCQ.problem_id, "mcq", "B", {}, MCQ.choices, {}, {}, (), (), ()),
         VerifierItem(INT.problem_id, "int", "12", {}, (), {}, {}, (), (), ()),
         VerifierItem(BAD_GOLD.problem_id, "int", "abc", {}, (), {}, {}, (), (), ()))


def eval_verifier() -> Verifier:
    verifier = Verifier.__new__(Verifier)
    verifier.actors = {actor.problem_id: actor for actor in (MCQ, INT, BAD_GOLD)}
    verifier.items = {item.problem_id: item for item in ITEMS}
    return verifier


VERIFIER = eval_verifier()
CONTRACT = {"answer_contract": EVAL}


def grade(actor: ActorItem, text: str, finish: str = "stop", **options):
    return VERIFIER.grade(actor.problem_id, text, finish, "obs", **options)


class PromptBytesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prompts = local_profile.local_prompts(eval_flagged())
        cls.processor = None
        if SNAPSHOT.is_dir():
            from transformers import AutoProcessor

            cls.processor = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True)

    def test_text_messages_and_chat_template_equal_the_evaluation_build(self) -> None:
        for name, question, choices, answer_format, n_images in CASES:
            sample = {"question": question, "question_type": "multiple-choice" if choices else "open",
                      "options": list(choices)}
            expected = build_prompt(sample, ROOT)
            evaluation = [{**message, "content": [{"type": "image"} if part["type"] == "image" else part
                                                  for part in message["content"]]}
                          for message in build_messages(sample, [object()] * n_images, ROOT)]
            for mode in ("plain", "obs"):
                with self.subTest(case=name, mode=mode):
                    actor = make_actor(f"{SOURCE}:{name}:{mode}", question, choices, answer_format, n_images, mode)
                    text = builders.turn1_text(actor, self.prompts)
                    self.assertEqual(text.encode(), expected.encode())
                    self.assertNotIn("Final answer:", text)
                    training = [builders.user_message(text, n_images)]
                    self.assertEqual(training, evaluation)
                    self.assertEqual([part["type"] for part in training[0]["content"]],
                                     ["image"] * n_images + ["text"])
                    if self.processor is not None:
                        rendered = self.processor.apply_chat_template(training, tokenize=False,
                                                                      add_generation_prompt=True)
                        self.assertEqual(rendered, self.processor.apply_chat_template(
                            evaluation, tokenize=False, add_generation_prompt=True))
                        self.assertTrue(rendered.endswith(ASSISTANT_START))


class SlotActorsUnaffectedTests(unittest.TestCase):
    def test_native_slot_actor_keeps_its_prompt_and_has_no_eval_contract(self) -> None:
        native = ActorItem("gen-graph-v1:graph:0001", "gen-graph-v1", "graph", "obs",
                           graph._question("shortest", "postorder"), (), "int", PATHS[:1], HASHES[:1])
        without, with_contract = local_profile.local_prompts(flagged()), local_profile.local_prompts(eval_flagged())
        text = builders.turn1_text(native, with_contract)
        self.assertEqual(text, builders.turn1_text(native, without))
        self.assertIn("Trace", text)
        self.assertIsNone(builders.turn1_contract(native, with_contract))
        self.assertIsNone(builders.turn1_contract(native, without))

    def test_only_slotless_t1_changes_and_retries_stay_byte_identical(self) -> None:
        without, with_contract = local_profile.local_prompts(flagged()), local_profile.local_prompts(eval_flagged())
        self.assertEqual({key: value for key, value in with_contract.items() if key != "repair_turn1_contract"},
                         without)
        self.assertEqual(builders.turn1_contract(MCQ, with_contract)["version"], EVAL)
        self.assertIsNone(builders.turn1_contract(MCQ, without))
        self.assertIn("Final answer:", builders.turn1_text(MCQ, without))
        self.assertNotEqual(builders.turn1_text(MCQ, with_contract), builders.turn1_text(MCQ, without))
        parent = Sample("p:T1:0", MCQ.problem_id, "p:T1", "T1", 1, None, [1], [2], [-.1], "stop",
                        "Solution:\nI cannot decide.", 0, 0)
        inventory = public_origin_inventory([parent], contract_version=ORIGIN_VERSION,
            response_contracts={parent.sample_id: None}, parent_sample_id=parent.sample_id)
        for mode in ("obs", "plain"):
            actor = replace(MCQ, mode=mode)
            for status, previous in (("INCORRECT", "C"), ("INVALID_NO_FINAL", None), ("TRUNCATED", None)):
                with self.subTest(mode=mode, status=status):
                    texts = [builders.retry_text(status, previous, mode, actor.family,
                                                 builders.for_actor(actor, prompts))
                             for prompts in (without, with_contract)]
                    self.assertEqual(texts[0], texts[1])
                    self.assertIn("Final answer", texts[0])
                    repairs = [builders.repair_retry_text(status, previous, actor, prompts, inventory)
                               for prompts in (without, with_contract)]
                    self.assertEqual(repairs[0], repairs[1])


class V8ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = None
        if B2.is_file():
            with B2.open(encoding="utf-8") as handle:  # read-only
                cls.records = [json.loads(line) for line in handle]

    def test_vendored_definitions_equal_their_sources(self) -> None:
        functions = {name: value for name, value in vars(v8_parser).items()
                     if inspect.isfunction(value) and value.__module__ == v8_parser.__name__}
        self.assertTrue({"extract_mcq", "extract_open", "_extract", "_result", "_text_key"} <= set(functions))
        self.assertGreater(len(functions), 20)
        for name, function in functions.items():
            with self.subTest(function=name):
                sources = {inspect.getsource(getattr(module, name)) for module in (src_v8, src_base)
                           if inspect.isfunction(getattr(module, name, None))
                           and getattr(module, name).__module__ == module.__name__}
                self.assertTrue(sources, "no source definition")
                self.assertIn(inspect.getsource(function), sources)

    def test_vendored_constants_equal_their_sources(self) -> None:
        constants = {name: value for name, value in vars(v8_parser).items()
                     if not name.startswith("__") and name != "SOURCE_SHA256"
                     and isinstance(value, (re.Pattern, str, int, dict, set, frozenset, tuple, list))}
        self.assertIn("VERSION", constants)
        self.assertTrue(any(isinstance(value, re.Pattern) for value in constants.values()))

        def key(value):
            return (value.pattern, value.flags) if isinstance(value, re.Pattern) else value

        for name, value in constants.items():
            with self.subTest(constant=name):
                sources = [key(getattr(module, name)) for module in (src_v8, src_base) if hasattr(module, name)]
                self.assertTrue(sources, "no source constant")
                self.assertIn(key(value), sources)
        self.assertEqual(v8_parser.VERSION, src_v8.VERSION)
        self.assertEqual(set(v8_parser.SOURCE_SHA256), {"src/mmdl/evaluation/final_answer_parser.py",
                                                         "src/mmdl/evaluation/final_answer_parser_v8.py"})
        for relative, digest in v8_parser.SOURCE_SHA256.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest, relative)

    def test_existing_v8_unit_suite_passes_on_the_vendored_extractors(self) -> None:
        spec = importlib.util.spec_from_file_location("v8_unit_suite_for_parity", UNIT_V8)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        stream = io.StringIO()
        with patch.object(src_v8, "extract_mcq", wraps=v8_parser.extract_mcq) as mcq, \
                patch.object(src_v8, "extract_open", wraps=v8_parser.extract_open) as open_:
            result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
        self.assertTrue(result.wasSuccessful(), stream.getvalue())
        self.assertEqual(result.testsRun, 20)
        self.assertTrue(mcq.called and open_.called)

    def test_vendored_extraction_equals_v8_and_the_stored_b2_parse_for_all_900(self) -> None:
        if self.records is None:  # analysis_exports is not part of a clean checkout
            return
        self.assertEqual(len(self.records), 900)
        differ, stored = [], []
        for record in self.records:
            args = (record["raw_response"], record["finish_reason"])
            if record["question_type"] == "multiple-choice":
                found = v8_parser.extract_mcq(args[0], record["options"], args[1], question=record["question"])
                reference = src_v8.extract_mcq(args[0], record["options"], args[1], question=record["question"])
            else:
                found = v8_parser.extract_open(*args, question=record["question"])
                reference = src_v8.extract_open(*args, question=record["question"])
            differ += [record["id"]] if found != reference else []
            stored += [record["id"]] if (found["answer"], found["status"]) != (
                record["parsed_answer"], record["parse_status"]) else []
        self.assertEqual(differ, [])
        self.assertEqual(stored, [])


class StatusMappingTests(unittest.TestCase):
    def test_v8_extraction_then_private_gold_comparison_sets_the_status(self) -> None:
        cases = (
            (MCQ, "Reasoning. The answer is B.", "stop", "CORRECT", "B", 1.0),
            (MCQ, "So the answer is (C)", "stop", "INCORRECT", "C", 0.0),
            (MCQ, "Let me think.\nStep one compares the bars.\nStep two counts them.\nI am unsure.",
             "stop", "INVALID_NO_FINAL", None, 0.0),
            (MCQ, "The answer is B.\n\nThe answer is C.", "stop", "INVALID_NO_FINAL", None, 0.0),
            (MCQ, "The answer is B", "length", "TRUNCATED", None, 0.0),
            (INT, "Therefore the answer is 12.", "stop", "CORRECT", "12", 1.0),
            (INT, "Answer: twelve", "stop", "INCORRECT", "twelve", 0.0),
            (BAD_GOLD, "The answer is 5.", "stop", "UNVERIFIABLE", "5", 0.0))
        for actor, text, finish, status, extracted, reward in cases:
            with self.subTest(text=text, finish=finish):
                result = grade(actor, text, finish, **CONTRACT)
                self.assertEqual((result.status, result.extracted, result.reward_answer, result.slots),
                                 (status, extracted, reward, None))
                self.assertTrue(result.verifier_version.endswith("+" + EVAL))

    def test_without_the_contract_the_strict_final_answer_result_is_kept(self) -> None:
        strict = grade(MCQ, "Reasoning. The answer is B.")
        self.assertEqual((strict.status, strict.extracted), ("INVALID_NO_FINAL", None))
        self.assertNotIn(EVAL, strict.verifier_version)
        self.assertEqual(grade(INT, "Therefore the answer is 12.").status, "INVALID_NO_FINAL")
        for contract in (None, EVAL):
            options = {} if contract is None else CONTRACT
            self.assertEqual(grade(MCQ, "Reasoning.\nFinal answer: B", **options).status, "CORRECT")

    def test_contract_misuse_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            grade(MCQ, "Final answer: B", answer_contract=EVAL, response_contract="repair-check-v1")
        with self.assertRaises(ValueError):
            grade(MCQ, "Final answer: B", answer_contract="junk")

    def test_extract_eval_aligned_statuses(self) -> None:
        question = MCQ.question
        self.assertEqual(answers.extract_eval_aligned("The answer is B.", "stop", question, MCQ.choices, "mcq"),
                         answers.Extraction("OK", "B"))
        self.assertEqual(answers.extract_eval_aligned("Therefore the answer is 12.", "stop", question, (), "int"),
                         answers.Extraction("OK", "12"))
        self.assertEqual(answers.extract_eval_aligned("Hmm.\nStep one.\nStep two.\nUnsure.", "stop", question,
                                                      MCQ.choices, "mcq"), answers.Extraction("INVALID_NO_FINAL", None))
        self.assertEqual(answers.extract_eval_aligned("The answer is B.", "length", question, MCQ.choices, "mcq"),
                         answers.Extraction("TRUNCATED", None))
        for bad in (("x", "other", question, (), "int"), ("x", "stop", None, (), "int"), (None, "stop", question, (), "int")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                answers.extract_eval_aligned(*bad)


class FinalizationTests(unittest.TestCase):
    def test_t1_finalization_reward_follows_the_v8_graded_result(self) -> None:
        cases = (("The answer is B.", "stop", 1.0), ("The answer is C.", "stop", 1.0),
                 ("Unsure.\nStep one.\nStep two.\nNo conclusion.", "stop", 0.0), ("The answer is B.", "length", 0.0))
        statuses = []
        for text, finish, expected in cases:
            result = grade(MCQ, text, finish, **CONTRACT)
            statuses.append(result.status)
            sample = Sample("p:T1:0", MCQ.problem_id, "p:T1", "T1", 1, None, [1], [2], [-.1], finish, text, 0, 0,
                            result=result)
            with self.subTest(status=result.status):
                self.assertEqual(finalization_reward(sample, None), expected)
        self.assertEqual(statuses, ["CORRECT", "INCORRECT", "INVALID_NO_FINAL", "TRUNCATED"])


class ProtocolTests(unittest.TestCase):
    def test_historical_hashes_do_not_bind_the_contract(self) -> None:
        for name, options in HISTORICAL_OPTIONS.items():
            with self.subTest(name=name):
                protocol = local_profile.build_protocol(**options)
                self.assertEqual(protocol["sha256"], HISTORICAL_SHA256[name])
                self.assertNotIn("repair_turn1_contract", protocol)
                self.assertNotIn("repair_turn1_contract_sha256", protocol)
        self.assertNotIn("repair_turn1_contract", local_profile.local_prompts(flagged()))

    def test_contract_and_cap_bind_round_trip_and_leave_caps_unchanged(self) -> None:
        plain, default, capped = flagged(), eval_flagged(), eval_flagged(turn1_max_new_tokens=4096)
        for protocol in (default, capped):
            self.assertEqual(local_profile.restore_protocol(json.loads(json.dumps(protocol))), protocol)
            self.assertEqual(protocol["rollout"], plain["rollout"])
            self.assertEqual(protocol["repair_turn1_contract_sha256"],
                             local_profile.digest(protocol["repair_turn1_contract"]))
        self.assertEqual(len({plain["sha256"], default["sha256"], capped["sha256"]}), 3)
        self.assertEqual(eval_flagged(turn1_max_new_tokens=1024), default)
        self.assertEqual(default["repair_turn1_contract"]["max_new_tokens"], 1024)
        self.assertEqual(capped["repair_turn1_contract"]["max_new_tokens"], 4096)
        self.assertEqual(builders.turn1_contract(MCQ, local_profile.local_prompts(capped))["max_new_tokens"], 4096)
        self.assertEqual(builders.turn1_contract(MCQ, local_profile.local_prompts(default))["max_new_tokens"], 1024)

    def test_invalid_options_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            local_profile.build_protocol(repair_notebook=True, context_limit=24576, note_contract=compact_note.VERSION,
                                         turn1_contract=EVAL)
        with self.assertRaises(ValueError):
            flagged(turn1_max_new_tokens=4096)
        for cap in (0, 24576, "4096", True):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                eval_flagged(turn1_max_new_tokens=cap)
        with self.assertRaises(ValueError):
            flagged(turn1_contract="junk")

    def test_tampered_saved_protocol_is_rejected(self) -> None:
        def contract_field(key, value):
            return lambda protocol: protocol["repair_turn1_contract"].__setitem__(key, value)

        mutations = (contract_field("max_new_tokens", 2048), contract_field("version", "junk"),
                     lambda protocol: protocol["repair_turn1_contract"]["parser_source_sha256"].update(
                         {key: "0" * 64 for key in list(protocol["repair_turn1_contract"]["parser_source_sha256"])[:1]}),
                     lambda protocol: protocol.pop("repair_turn1_contract_sha256"),
                     lambda protocol: protocol.pop("repair_turn1_contract"))
        for mutate in mutations:
            protocol = json.loads(json.dumps(eval_flagged()))
            mutate(protocol)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                local_profile.restore_protocol(protocol)

    def test_tampered_prompts_contract_is_rejected_by_for_actor(self) -> None:
        for key, value in (("version", "junk"), ("scope", "everything"), ("message", "other"),
                           ("parser", "team-final-answer-v7"), ("vendored_parser_sha256", "0" * 64),
                           ("comparison", "contains gold"), ("finalization", "different"),
                           ("status", {}), ("p0_examples", {}), ("extra", 1), ("max_new_tokens", 0)):
            prompts = local_profile.local_prompts(eval_flagged())
            prompts["repair_turn1_contract"] = copy.deepcopy(prompts["repair_turn1_contract"]) | {key: value}
            with self.subTest(field=key), self.assertRaises(ValueError):
                builders.for_actor(MCQ, prompts)
        prompts = local_profile.local_prompts(eval_flagged())
        prompts["repair_turn1_contract"]["parser_source_sha256"] = {"x": "0" * 64}
        with self.assertRaises(ValueError):
            builders.for_actor(MCQ, prompts)
        prompts["repair_turn1_contract"] = "junk"
        with self.assertRaises(ValueError):
            builders.for_actor(MCQ, prompts)


class CliAndReadinessTests(unittest.TestCase):
    FLAGS = ["--repair-notebook", "--context-limit", "24576", "--note-contract", compact_note.VERSION,
             "--answer-verified-route", "--slotless-source", "hf-x", "--finalization-reward",
             "--turn1-contract", EVAL, "--turn1-max-new-tokens", "4096"]
    NO_CAP, NO_CONTRACT = FLAGS[:-2], FLAGS[:-4]
    FLAGGED_OPTIONS = HISTORICAL_OPTIONS["flagged"]

    @staticmethod
    def parse(flags: list[str]):
        parser = ArgumentParser()
        parser.add_argument("--repair-notebook", action="store_true")
        parser.add_argument("--context-limit", type=int, default=16384)
        parser.add_argument("--note-contract", default=local_profile.DEFAULT_NOTE_CONTRACT)
        local_profile.add_repair_cli_arguments(parser)
        return parser.parse_args(flags)

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "model").mkdir()
        (self.root / "model" / "nf4_base.json").write_text('{"fixture_only":true}')

    def cache_check(self, name: str, flags: list[str]) -> Path:
        from fine_tuning.notebook_rl import quantized_cache

        output = self.root / name
        with patch("fine_tuning.notebook_rl.policy.enable_deterministic_execution", return_value={}), \
                patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.max_memory_allocated", return_value=0), \
                patch("torch.cuda.max_memory_reserved", return_value=0), patch("traceback.print_exc"), \
                patch("builtins.print"):
            code = quantized_cache.main(["check", "--cache", str(self.root / "cache"), "--output", str(output),
                                         "--parent", str(self.root / "absent-parent"), *flags])
        self.assertEqual(code, 1)  # stops at the absent parent, after sources.json/protocol.json are written
        (output / "result.json").write_text(json.dumps({
            "status": "NF4_CACHE_FRESH_PROCESS_RELOAD_GENERATION_VERIFIED", "denied_original_access_attempts": 0,
            "cache_metadata_sha256": checkpoint.file_hash(self.root / "model" / "nf4_base.json")}))
        return output

    def test_flags_build_the_same_protocol_as_the_keywords(self) -> None:
        args = self.parse(self.FLAGS)
        self.assertIsNone(local_profile.repair_cli_error(args, repair_notebook=args.repair_notebook))
        built = local_profile.build_protocol(repair_notebook=True, context_limit=24576,
                                             **local_profile.repair_cli_keywords(args))
        self.assertEqual(built, local_profile.build_protocol(
            **self.FLAGGED_OPTIONS, turn1_contract=EVAL, turn1_max_new_tokens=4096))
        self.assertEqual(built["repair_turn1_contract"]["max_new_tokens"], 4096)
        default = local_profile.repair_cli_keywords(self.parse(self.NO_CAP))
        self.assertEqual(default["turn1_contract"], EVAL)
        self.assertNotIn("turn1_max_new_tokens", default)
        self.assertFalse({"turn1_contract", "turn1_max_new_tokens"} & set(
            local_profile.repair_cli_keywords(self.parse(self.NO_CONTRACT))))

    def test_pod_proof_flags_bind_the_fast8k_eval_cap_and_image_budget_inside_the_24576_context(self) -> None:
        """User decision 2026-10-31: the trained eval-aligned T1 uses the statistical eval arms' 8192 cap and
        262144..1003520 pixels; retries keep 1024 and the final note keeps its 8192 reservation."""
        import yaml

        from fine_tuning.notebook_rl import proof_eval
        from fine_tuning.notebook_rl.pod import chain

        args = self.parse(["--repair-notebook", "--context-limit", "24576", "--note-contract", compact_note.VERSION,
                           *chain.PROOF_FLAGS])
        self.assertIsNone(local_profile.repair_cli_error(args, repair_notebook=args.repair_notebook))
        built = local_profile.build_protocol(repair_notebook=True, context_limit=24576,
                                             **local_profile.repair_cli_keywords(args))
        cap = built["repair_turn1_contract"]["max_new_tokens"]
        self.assertEqual((cap, built["context_limit"]), (8192, 24576))
        self.assertEqual({k: built["image"][k] for k in proof_eval.FAST_PIXELS}, proof_eval.FAST_PIXELS)
        self.assertEqual({k: v for k, v in built["rollout"]["max_new_tokens"].items() if k != 1}, {2: 1024, 3: 1024})
        for vid, (path, _, _, role) in proof_eval.VARIANTS.items():
            if role == "statistical":
                cfg = yaml.safe_load(path.read_text())
                self.assertEqual((cfg["generation"]["max_new_tokens"], cfg["image"]["min_pixels"],
                                  cfg["image"]["max_pixels"]), (cap, built["image"]["min_pixels"],
                                                                built["image"]["max_pixels"]), vid)

    def test_misuse_is_reported_and_unknown_contract_is_refused(self) -> None:
        without_route = ["--repair-notebook", "--context-limit", "24576", "--note-contract", compact_note.VERSION,
                         "--turn1-contract", EVAL]
        cap_only = [*self.NO_CONTRACT, "--turn1-max-new-tokens", "4096"]
        for flags in (without_route, cap_only):
            with self.subTest(flags=flags):
                message = local_profile.repair_cli_error(self.parse(flags), repair_notebook=True)
                self.assertIn("--turn1-contract requires --answer-verified-route", message)
                self.assertIn("--turn1-max-new-tokens requires --turn1-contract", message)
        self.assertIsNone(local_profile.repair_cli_error(self.parse(self.NO_CAP), repair_notebook=True))
        self.assertIsNone(local_profile.repair_cli_error(self.parse(self.NO_CONTRACT), repair_notebook=True))
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            self.parse([*self.NO_CONTRACT, "--turn1-contract", "junk"])

    def test_cache_readiness_binds_the_contract_and_its_cap(self) -> None:
        expected = local_profile.build_protocol(**self.FLAGGED_OPTIONS, turn1_contract=EVAL, turn1_max_new_tokens=4096)
        check = self.cache_check("turn1", self.FLAGS)
        self.assertEqual(json.loads((check / "protocol.json").read_text())["sha256"], expected["sha256"])
        result = local_run.cache_readiness(check, self.root / "model", expected)
        self.assertEqual(result["status"], "NF4_CACHE_FRESH_PROCESS_RELOAD_GENERATION_VERIFIED")
        for mismatch in (local_profile.build_protocol(**self.FLAGGED_OPTIONS),
                         local_profile.build_protocol(**self.FLAGGED_OPTIONS, turn1_contract=EVAL),
                         local_profile.build_protocol(**self.FLAGGED_OPTIONS, turn1_contract=EVAL,
                                                      turn1_max_new_tokens=2048)):
            with self.subTest(sha=mismatch["sha256"]), self.assertRaises(ValueError):
                local_run.cache_readiness(check, self.root / "model", mismatch)


    def test_v8_parser_bytes_bind_through_the_contract_protocol_not_the_sources_list(self) -> None:
        from fine_tuning.notebook_rl import quantized_cache

        self.assertNotIn("notebook_rl/v8_parser.py", quantized_cache._sources(repair=True))
        contract = local_profile.build_protocol(**self.FLAGGED_OPTIONS, turn1_contract=EVAL, turn1_max_new_tokens=4096)
        default = local_profile.build_protocol(**self.FLAGGED_OPTIONS)
        checks = {"turn1": self.cache_check("turn1", self.FLAGS), "default": self.cache_check("plain", self.NO_CONTRACT)}
        self.assertEqual(*[json.loads((check / "sources.json").read_text()) for check in checks.values()])
        changed = self.root / "v8_parser.py"
        changed.write_bytes(Path(v8_parser.__file__).read_bytes() + b"# changed\n")
        with patch.object(v8_parser, "__file__", str(changed)):
            moved = local_profile.build_protocol(**self.FLAGGED_OPTIONS, turn1_contract=EVAL, turn1_max_new_tokens=4096)
            self.assertEqual(local_profile.build_protocol(**self.FLAGGED_OPTIONS), default)
        self.assertNotEqual(moved["sha256"], contract["sha256"])
        with self.assertRaises(ValueError):
            local_run.cache_readiness(checks["turn1"], self.root / "model", moved)
        for name, protocol in (("turn1", contract), ("default", default)):
            with self.subTest(name=name):
                self.assertEqual(local_run.cache_readiness(checks[name], self.root / "model", protocol)["status"],
                                 "NF4_CACHE_FRESH_PROCESS_RELOAD_GENERATION_VERIFIED")


LONG_RIGHT = "Counting the bars one by one. " * 50 + "\nThree bars are taller than 5, so the answer is B."
P0_REPLIES = [LONG_RIGHT, "So the answer is (C)",
              "Let me think.\nStep one compares the bars.\nStep two counts them.\nI am unsure.",
              "Three bars are taller than 5. The answer is B."]


class EndToEndTests(unittest.TestCase):
    """T1 P0 -> V8 grade -> retry -> compact note -> credit -> tiny update, through the fresh-process service."""

    def setUp(self) -> None:
        from fine_tuning.tests.test_notebook_rl_route_finalization import RIGHT, Pipeline, note, retry

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        lineage = Lineage(MCQ.problem_id, SOURCE, "fixture-v1", "INTERNAL_FIXTURE", "PROGRAM", "rendered:fixture",
                          "procedural", "fixture-v1", 3407, "count-bars", "c" * 16, "train", ())
        write_split(base / "vault", [MCQ], [ITEMS[0]], [lineage])
        self.service = VerifierService(base / "vault", forbidden_roots=[base / "benchmark"])
        self.addCleanup(self.service.close)
        self.p = Pipeline(self, eval_flagged(turn1_max_new_tokens=4096), slotless=True, actor=MCQ, item=ITEMS[0])
        self.p.f.verifier = self.p.f.fixture.verifier = self.service
        # Every fresh native audit (collector, source re-audit, compact natives, online credit) is observed.
        self.audits: list[tuple[int, str | None]] = []
        original = notebook_synthesis.audit_native_sample

        def spy(sample, *args, **kwargs):
            self.audits.append((sample.turn, kwargs.get("answer_contract")))
            return original(sample, *args, **kwargs)

        patcher = patch.object(notebook_synthesis, "audit_native_sample", side_effect=spy)
        patcher.start()
        self.addCleanup(patcher.stop)
        # The retry abstains on one public origin of its actual parent, so online credit re-audits that P0 parent.
        group = f"{MCQ.problem_id}:v0:T1:{rollout._hash('root')[:16]}"
        self.parent = min((f"{group}:{index}" for index in (1, 2)),
                          key=lambda sid: hashlib.sha256(f"{MCQ.problem_id}:0:{sid}".encode()).digest())
        parent = Sample(self.parent, MCQ.problem_id, group, "T1", 1, None, [1], [2], [-.1], "stop",
                        P0_REPLIES[int(self.parent[-1])], 0, 0)
        origin = public_origin_inventory([parent], contract_version=ORIGIN_VERSION,
                                         response_contracts={self.parent: None}).origins[0]
        claim = {"origin_ref": origin.ref_id, "intent": "UNKNOWN", "rule": origin.rule, "proposed_value": None}
        abstain = ("ErrorCheck: " + json.dumps({"version": RETRY_VERSION, "claims": [claim], "uncertain_refs": []})
                   + "\nRevisedSolution:\n" + RIGHT)
        self.tree, self.engine = self.p.f.collect([P0_REPLIES, [abstain, retry(RIGHT)]])
        self.p.f.attach(self.tree, self.engine, [note([{"attempt": 3, "type": "NO_FINAL_ANSWER"}]),
                                                 note([{"attempt": 1, "type": "NO_FINAL_ANSWER"}]), note([]), "{"])

    def test_p0_attempt_is_v8_graded_by_the_service_and_audited_without_finalization_checks(self) -> None:
        self.assertIn("v8_parser.py", self.service.metadata["code_sha256"])
        first, second = self.engine.requests[:2]
        self.assertEqual(first.messages[0]["content"][-1]["text"], builders.p0_eval_text(MCQ.question, MCQ.choices))
        self.assertEqual((first.max_new_tokens, second.max_new_tokens), (4096, 1024))
        t1, t2 = self.tree.groups[:2]
        self.assertEqual([group.kind for group in self.tree.groups], ["T1", "T2", "DIAG"])
        self.assertGreater(len(t1.samples[0].completion_token_ids), 1024)  # admitted only by the bound T1 cap
        self.assertEqual([(row.result.status, row.result.extracted) for row in t1.samples],
                         [("CORRECT", "B"), ("INCORRECT", "C"), ("INVALID_NO_FINAL", None), ("CORRECT", "B")])
        self.assertEqual({row.result.verifier_version for row in t1.samples}, {VERIFIER_VERSION + "+" + EVAL})
        self.assertEqual({(row.result.status, row.result.verifier_version) for row in t2.samples},
                         {("CORRECT", VERIFIER_VERSION + "+" + RETRY_VERSION)})
        audits = self.tree.notebook["repair_collector"]["native_audits"]
        self.assertEqual([audits[row.sample_id]["output_checks"] for row in t1.samples], [[]] * 4)
        self.assertEqual([[check["status"] for check in audits[row.sample_id]["output_checks"]] for row in t2.samples],
                         [["COMPLIANT"]] * 2)
        # The strict Final-answer grade would have rejected every P0 attempt.
        self.assertEqual([self.service.grade(MCQ.problem_id, row.text, row.finish_reason, "obs").status
                          for row in t1.samples], ["INVALID_NO_FINAL"] * 4)
        record = self.tree.notebook["rewarded_notebook"]
        self.assertEqual((record["status"], record["source_audit"]["turn1_answer_contract"]), ("GENERATED", EVAL))
        self.assertEqual(record["source_audit"]["routing"]["admission"], "ANSWER_VERIFIED")
        # NO_FINAL_ANSWER is judged by the V8 outcome: attempt 1 ends "the answer is B" without a Final line.
        self.assertEqual([row["audit"]["entries"][0]["reason"] for row in record["grades"][:2]],
                         ["NO_FINAL_CONFIRMED", "ATTEMPT_HAD_FINAL"])
        self.assertEqual(set(self.audits), {(1, EVAL), (2, None)})

    def test_credit_cap_v8_finalization_and_a_real_tiny_update_with_evidence(self) -> None:
        receipt = self.p.f.prepare(self.tree, self.engine)
        self.assertEqual((receipt["admission"], receipt["admitted"]), ("ANSWER_VERIFIED", True))
        t1 = receipt["groups"][0]
        self.assertEqual((t1["kind"], t1["original_group_size"], t1["completion_cap"]), ("T1", 4, 4096))
        self.assertEqual([row["answer_reward"] for row in t1["samples"]], [1., 0., 0., 1.])
        self.assertEqual([row["finalization_reward"] for row in t1["samples"]], [1., 1., 0., 1.])
        result, _, _ = self.p.u.step(self.tree, self.engine, require_retained_learning_signal=True)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        aggregate = result["repair_aggregate_gradient"]
        inventory = [{key: value for key, value in row.items() if key != "index"}
                     for row in aggregate["parameter_inventory"]]
        evidence = repair_evidence.repair_update_evidence(result, self.p.u.protocol, trees=[self.tree],
            preparations={MCQ.problem_id: self.p.f.prepare(self.tree, self.engine)}, parameter_inventory=inventory)
        self.assertTrue(evidence["S_text_gradient_verified"])
        online = receipt["groups"][1]["samples"][0]["semantic_audit"]["source_bindings"]["native_audits"]
        self.assertEqual(online[self.parent]["result_verifier_version"], VERIFIER_VERSION + "+" + EVAL)
        self.assertEqual(set(self.audits), {(1, EVAL), (2, None)})


class BindingTests(unittest.TestCase):
    def sample(self, text: str, turn: int = 1) -> Sample:
        return Sample(f"p:T{turn}:0", MCQ.problem_id, f"p:T{turn}", f"T{turn}", turn, None if turn == 1 else "p:T1:0",
                      [1], [2], [-.1], "stop", text, 0, 0)

    def test_native_audit_binds_the_v8_answer_and_refuses_misuse(self) -> None:
        audit = notebook_synthesis.audit_native_sample(self.sample("The answer is B."), MCQ, VERIFIER, **CONTRACT)
        self.assertEqual((audit["result"]["status"], audit["output_checks"]), ("CORRECT", []))
        strict = notebook_synthesis.audit_native_sample(self.sample("Solution:\nx\nFinal answer: B"), MCQ, VERIFIER)
        self.assertEqual([row["status"] for row in strict["output_checks"]], ["COMPLIANT"])
        other = MagicMock(wraps=VERIFIER)
        other.grade.side_effect = lambda *args, **kwargs: replace(VERIFIER.grade(*args, **kwargs), extracted="C")
        with self.assertRaisesRegex(ValueError, "not bound"):
            notebook_synthesis.audit_native_sample(self.sample("The answer is B."), MCQ, other, **CONTRACT)
        other.grade.side_effect = lambda *args, **kwargs: VERIFIER.grade(*args[:4])
        with self.assertRaisesRegex(ValueError, "not bound"):  # strict-version result under the contract
            notebook_synthesis.audit_native_sample(self.sample("Final answer: B"), MCQ, other, **CONTRACT)
        for sample, options in ((self.sample("The answer is B.", 2), CONTRACT),
                                (self.sample("The answer is B."), CONTRACT | {"response_contract": RETRY_VERSION})):
            with self.subTest(turn=sample.turn), self.assertRaises(ValueError):
                notebook_synthesis.audit_native_sample(sample, MCQ, VERIFIER, **options)

    def test_worker_dispatch_reextracts_with_v8_and_rejects_vault_answers(self) -> None:
        def request(params: dict) -> dict:
            return {"id": 1, "method": "grade", "problem_id": MCQ.problem_id,
                    "params": {"text": "So the answer is (C)", "finish_reason": "stop", "mode": "obs"} | params}

        value = verifier_service._dispatch(VERIFIER, verifier_service._Issued(), request(CONTRACT))
        self.assertEqual((value["status"], value["extracted"]), ("INCORRECT", "C"))
        leaky = MagicMock(wraps=VERIFIER)
        leaky.items, leaky.actors = VERIFIER.items, VERIFIER.actors
        leaky.grade.side_effect = lambda *args, **kwargs: replace(VERIFIER.grade(*args, **kwargs), extracted="B")
        for verifier, params, code in ((leaky, CONTRACT, "VERIFIER_ERROR"),
                                       (VERIFIER, CONTRACT | {"response_contract": RETRY_VERSION}, "INVALID_REQUEST")):
            with self.subTest(code=code), self.assertRaisesRegex(verifier_service._Rejected, code):
                verifier_service._dispatch(verifier, verifier_service._Issued(), request(params))

    def test_loss_and_evidence_accept_the_bound_t1_cap_only_for_t1(self) -> None:
        from fine_tuning.notebook_rl.losses import repair_component_losses
        import torch

        current = torch.tensor([[-.3, -.2]], dtype=torch.float64, requires_grad=True)
        zero, mask, keep = torch.zeros(1, 2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.bool), torch.tensor([True])

        def run(kind: str, size: int, cap: int):
            return repair_component_losses(current, current.detach(), current.detach(), mask, keep, answer=zero + .5,
                online=zero, final=zero, kind=kind, original_group_size=size, completion_cap=cap)

        self.assertAlmostEqual(run("T1", 4, 4096)["answer"].item() * 4096, run("T1", 4, 1024)["answer"].item() * 1024)
        for kind, size, cap in (("T2", 2, 4096), ("T1", 2, 4096), ("T1", 4, 24576), ("T1", 4, 0)):
            with self.subTest(kind=kind, cap=cap), self.assertRaisesRegex(ValueError, "group size/cap"):
                run(kind, size, cap)


if __name__ == "__main__":
    unittest.main()
