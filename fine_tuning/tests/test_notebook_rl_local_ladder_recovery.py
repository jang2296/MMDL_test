"""Program-authored CPU fixtures, never evidence of model learning or inference."""
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import local_ladder_generation as generation
from fine_tuning.notebook_rl import local_ladder_recovery as recovery
from fine_tuning.notebook_rl.cli import write_pool
from fine_tuning.notebook_rl.dataset import _bytes, _sha, _write
from fine_tuning.notebook_rl.engines import Completion, Prepared, RolloutResult
from fine_tuning.notebook_rl.generators import circuit
from fine_tuning.notebook_rl.local_ladder import INDICES, SEED
from fine_tuning.notebook_rl.observations import observation_spans, slot_id
from fine_tuning.notebook_rl.prompts import load_prompts


class FixtureTokenizer:
    unk_token_id = -1

    def __init__(self):
        self.pieces = {}

    def convert_tokens_to_ids(self, text):
        return 103 if text == "<|im_end|>" else -1

    def encode(self, text, **kwargs):
        ids = []
        for piece in re.findall(r"[\s\S]{1,8}", text):
            token = 1000 + int.from_bytes(hashlib.sha256(piece.encode()).digest()[:6], "big")
            self.pieces[token] = piece
            ids.append(token)
        return ids

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[token] for token in ids)


class FixtureEngine:
    context_limit = 16384
    model_config = SimpleNamespace(image_token_id=99)

    def __init__(self, rows, *, version=0, script=None, bad_note=None):
        self.rows = {row.actor.problem_id: row for row in rows}
        self.policy_version, self.tokenizer, self.calls = version, FixtureTokenizer(), []
        self.script, self.bad_note = script or {}, bad_note
        self.after_sample = None

    def prepare(self, request):
        tokens = list(request.prompt_token_ids) if request.prompt_token_ids is not None else (
            [101, 99, 102] + self.tokenizer.encode(json.dumps(request.messages, ensure_ascii=False)))
        identity = _sha(request.images[0].tobytes())
        return Prepared(tokens, {"pixel_values": torch.tensor([[int(identity[:3], 16), 1., 2.]]),
            "image_grid_thw": torch.tensor([[1, 2, 2]])}, request.images, identity, ((1, 2),))

    def _answer(self, pid, outcome):
        row = self.rows[pid]
        observed = dict(row.verifier.slots)
        if outcome == "visual":
            key = next(key for key in observed if key.startswith("comp["))
            observed[key] = "999999 V"
        facts = "Observed facts:\r\n" + "\r\n".join(f"{key} = {value}" for key, value in observed.items())
        final = row.verifier.gold if outcome in ("correct", "truncated") else "999999 A"
        text = facts + "\r\nSolution:\r\nFinal answer: " + final
        if outcome == "nofinal":
            text = facts + "\r\nSolution:\r\nUnfinished reasoning.\r\n"
        if outcome == "multiple":
            text += "\r\nFinal answer: 1 A"
        return text, "length" if outcome == "truncated" else "stop"

    def _note(self, request, pid):
        body = request.messages[0]["content"][-1]["text"]
        rows = json.loads(body.split("<read_only_attempts>\n", 1)[1].split("\n</read_only_attempts>", 1)[0])
        claims = []
        registered = {slot_id(key): value for key, value in self.rows[pid].verifier.slots.items()}
        for row in rows:
            text = "".join(line for _, line in row["source_lines"])
            wrong = sorted({key for _, _, key, value in observation_spans(text)
                            if key in registered and value != registered[key]})
            claims.append({"attempt": row["attempt"], "line_ranges": [[1, row["line_count"]]],
                "answer": row["answer"], "cause": "ANSWER_CORRECT" if row["status"] == "CORRECT" else
                    ("VISUAL_MISREAD" if wrong else "NON_VISUAL") if row["status"] == "INCORRECT" else "FINALIZATION", "wrong_slots": wrong})
        note = {"corrected_facts": self.rows[pid].verifier.slots, "attempts": claims,
                "success_attempt": rows[-1]["attempt"], "corrected_answer": self.rows[pid].verifier.gold,
                "lesson": "Read the circuit and finish one checked answer."}
        if self.bad_note == pid:
            note["corrected_facts"] = {}
        return json.dumps(note), "stop"

    def sample(self, requests):
        request = requests[0]
        self.calls.append(request)
        pid, stage = request.request_id.rsplit(":", 1)
        text, finish = self._note(request, pid) if stage == "NOTE" else self._answer(pid, self.script.get((pid, stage), "correct"))
        tokens = self.tokenizer.encode(text)
        result = [RolloutResult(request.request_id, self.prepare(request).token_ids,
            [Completion(tokens, [-0.5] * len(tokens), finish, text)])]
        if self.after_sample:
            self.after_sample(self, request)
        return result


class NativeRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [circuit.generate(index, SEED, "sealed", "obs") for index in INDICES]
        cls.templates = load_prompts(Path(__file__).parents[1] / "notebook_rl/prompts_v1.yaml")

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cohort = self.root / "cohort"
        write_pool(self.cohort, self.rows, recipe={"fixture_only": True})
        prep = {"format": "notebook-rl-local-ladder-preparation-v1",
            "status": "CPU_COHORT_PREPARED_PENDING_GATES", "training_allowed": False,
            "evaluation_allowed": False, "gpu_evaluation_completed": False, "count": 10,
            "records_manifest": generation._descriptor(self.cohort / "records/manifest.yaml"),
            "problem_ids": [row.actor.problem_id for row in self.rows]}
        _write(self.cohort / "preparation.json", _bytes(prep))
        self.sha, self.ids = _sha(_bytes(prep)), prep["problem_ids"]
        self.engine = FixtureEngine(self.rows)

    def run_fixture(self, name="out", *, deadline=None, context_limit=16384):
        return recovery.diagnose(self.cohort, self.root / name, self.engine, copy.deepcopy(self.templates),
            expected_preparation_sha256=self.sha, local_protocol_sha256="a" * 64,
            deadline=time.monotonic() + 120 if deadline is None else deadline, context_limit=context_limit)

    def test_24k_recovery_replay_rejects_row_nested_and_cross_context_tampering(self):
        self.mixed()
        self.engine.context_limit = 24576
        record = self.run_fixture("wide", context_limit=24576)
        self.assertEqual(record["status"], "COMPLETE_NATIVE_LADDER_RECOVERY", record.get("reason"))
        self.assertEqual(recovery.replay(self.cohort, record,
            expected_preparation_sha256=self.sha)["note_verified_count"], 3)
        self.assertEqual(record["protocol"]["context_limit"], 24576)
        self.assertEqual(record["generation"]["protocol"]["context_limit"], 24576)
        self.assertTrue(all(row["context_limit"] == 24576 for row in record["calls"]))
        for mutation in ("row", "nested"):
            changed = copy.deepcopy(record)
            if mutation == "row":
                changed["calls"][0]["context_limit"] = 16384
                self.rehash(changed["calls"][0])
            else:
                first = changed["generation"]
                first["protocol"]["context_limit"] = 16384
                first["protocol_sha256"] = generation._digest(first["protocol"])
                for row in first["rows"]:
                    row["context_limit"] = 16384
                    self.rehash(row)
            self.rehash(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                recovery.replay(self.cohort, changed, expected_preparation_sha256=self.sha)
        self.engine = FixtureEngine(self.rows, version=1)
        legacy = self.run_fixture("legacy")
        with self.assertRaisesRegex(ValueError, "paired protocol"):
            recovery.validate_paired_records(record, legacy)

    def test_note_reserves_full_2048_tokens_at_each_explicit_context_boundary(self):
        for limit in (16384, 24576):
            for extra in (0, 1):
                self.engine = FixtureEngine(self.rows, script={(self.ids[0], "T1"): "wrong"})
                self.engine.context_limit = limit
                original = self.engine.prepare
                def padded(request):
                    prepared = original(request)
                    if request.request_id.endswith(":NOTE"):
                        prepared.token_ids += [1000] * (limit - 2048 + extra - len(prepared.token_ids))
                    return prepared
                self.engine.prepare = padded
                result = self.run_fixture(f"note-{limit}-{extra}", context_limit=limit)
                with self.subTest(limit=limit, extra=extra):
                    self.assertEqual(result["status"], "INCOMPLETE" if extra else "COMPLETE_NATIVE_LADDER_RECOVERY")
                    if extra:
                        self.assertIn("CONTEXT_BLOCKED", result["reason"])
                        self.assertFalse(any(call.request_id.endswith(":NOTE") for call in self.engine.calls))
                    else:
                        self.assertEqual(result["calls"][-1]["input"]["input_tokens"], limit - 2048)
                        self.assertEqual(result["calls"][-1]["max_new_tokens"], 2048)

    def test_recovery_protocol_context_choice_is_explicit_and_exact(self):
        default = recovery.protocol(self.templates, "a" * 64)
        self.assertEqual(default, recovery.protocol(self.templates, "a" * 64, context_limit=16384))
        self.assertEqual(default | {"context_limit": 24576},
                         recovery.protocol(self.templates, "a" * 64, context_limit=24576))
        for value in (True, 24576., 8192, 32768, "24576", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                recovery.protocol(self.templates, "a" * 64, context_limit=value)

    def test_generic_policy_recovery_replays_with_explicit_legacy_note_metric(self):
        from fine_tuning.notebook_rl import episode_notebook
        from fine_tuning.notebook_rl.local_ladder_gate import _validate_protocol

        self.templates = generation.circuit_evaluation_prompts()
        self.mixed()
        record = self.run_fixture("generic-recovery")
        self.assertEqual(record["status"], "COMPLETE_NATIVE_LADDER_RECOVERY", record.get("reason"))
        _validate_protocol(record["protocol"])
        self.assertEqual(record["protocol"]["notebook"]["metric_identity"], episode_notebook.VERSION)
        self.assertEqual(record["protocol"]["notebook"]["grader"], "episode_notebook.grade_note")
        self.assertFalse(record["protocol"]["notebook"]["bounded_synthesis_verified"])
        replayed = recovery.replay(self.cohort, record, expected_preparation_sha256=self.sha)
        self.assertEqual(replayed["note_verified_count"], 3)
        changed = copy.deepcopy(record)
        row = changed["generation"]["rows"][-1]
        row["input"]["messages"][0]["content"][-1]["text"] = "Noncanonical legacy prompt"
        row["input_sha256"] = generation._digest(row["input"])
        self.rehash(row)
        self.rehash(changed)
        with self.assertRaisesRegex(ValueError, "literal canonical"):
            recovery.replay(self.cohort, changed, expected_preparation_sha256=self.sha)
        changed = copy.deepcopy(record)
        changed["prompt_templates"]["solution_trace_contract"] = "native-reasoning-trace-v3"
        self.rehash(changed)
        with self.assertRaisesRegex(ValueError, "canonical circuit"):
            recovery.replay(self.cohort, changed, expected_preparation_sha256=self.sha)

    def mixed(self, version=0):
        a, b, c, d, e = self.ids[:5]
        script = {(a, "T1"): "visual", (b, "T1"): "wrong", (b, "T2"): "wrong", (b, "T3"): "wrong",
            (c, "T1"): "nofinal", (c, "T2"): "wrong", (d, "T1"): "truncated", (e, "T1"): "multiple"}
        self.engine = FixtureEngine(self.rows, version=version, script=script, bad_note=e)

    def test_real_fresh_verifier_recovery_notes_history_and_denominators(self):
        self.mixed()
        before = {path: path.read_bytes() for path in self.cohort.rglob("*") if path.is_file()}
        result = self.run_fixture()
        self.assertEqual(result["status"], "COMPLETE_NATIVE_LADDER_RECOVERY", result.get("reason"))
        self.assertEqual((result["first_attempt_accuracy"], result["final_accuracy"]), (.5, .9))
        self.assertEqual((result["initially_failed"], result["recovered_count"], result["recovered_fraction"]), (5, 4, .8))
        self.assertEqual(result["initially_incorrect"], 2)
        self.assertEqual(result["first_status_counts"], {"INCORRECT": 2, "INVALID_NO_FINAL": 1,
            "TRUNCATED": 1, "INVALID_MULTIPLE": 1, "CORRECT": 5})
        self.assertEqual((result["note_count"], result["note_verified_count"], result["note_verified_fraction"]), (4, 3, .75))
        self.assertEqual(result["first_statuses"][self.ids[2]], "INVALID_NO_FINAL")
        self.assertEqual(result["first_statuses"][self.ids[3]], "TRUNCATED")
        self.assertEqual(result["first_statuses"][self.ids[4]], "INVALID_MULTIPLE")
        self.assertEqual(len(self.engine.calls), 21)
        self.assertEqual(result["trees"][0]["tree"]["route"], "VISION_ON")
        self.assertEqual(result["trees"][0]["episode_audit"]["visual_attempts"], [1])
        self.assertTrue(all(call.request_id.endswith(":T1") for call in self.engine.calls[:10]))
        self.assertFalse(any(call.request_id.startswith(self.ids[-1] + ":T") for call in self.engine.calls[10:]))
        for entry in result["trees"]:
            audit = entry["episode_audit"]
            self.assertTrue(audit["complete_episode"])
            self.assertFalse(audit["training_complete_episode"])
            self.assertFalse(audit["training_admission"])
            for group in entry["tree"]["groups"]:
                self.assertFalse(group["usable"])
                self.assertEqual(group["reason"], "EVALUATION_SINGLE_SAMPLE")
                self.assertEqual(len(group["samples"]), 1)
        replayed = recovery.replay(self.cohort, result, expected_preparation_sha256=self.sha)
        self.assertEqual(replayed["status"], "CPU_NATIVE_LADDER_RECOVERY_REPLAYED")
        self.assertEqual(replayed["note_verified_count"], 3)
        self.assertNotEqual(replayed["verifier_process"]["pid"], result["verifier_process"]["pid"])
        self.assertFalse(replayed["model_generation_provenance_verified"])
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(result, json.loads((self.root / "out/result.json").read_bytes()))

    def test_all_correct_retained_no_retries_and_unchanged_native_T1(self):
        result = self.run_fixture()
        self.assertEqual(result["status"], "COMPLETE_NATIVE_LADDER_RECOVERY", result.get("reason"))
        self.assertEqual((result["first_attempt_accuracy"], result["final_accuracy"]), (1., 1.))
        self.assertEqual(result["note_count"], 0)
        self.assertIsNone(result["note_verified_fraction"])
        self.assertIsNone(result["recovered_fraction"])
        self.assertEqual(len(self.engine.calls), 10)
        engine = FixtureEngine(self.rows)
        first = generation.generate_ladder(self.cohort, self.root / "only-first", engine, self.templates,
            expected_preparation_sha256=self.sha, local_protocol_sha256="a" * 64)
        for actual, expected in zip(result["generation"]["rows"], first["rows"], strict=True):
            self.assertEqual({k: v for k, v in actual.items() if k not in ("elapsed_seconds", "record_sha256")},
                             {k: v for k, v in expected.items() if k not in ("elapsed_seconds", "record_sha256")})

    def test_all_wrong_finish_three_attempts_without_notes(self):
        self.engine = FixtureEngine(self.rows, script={(pid, stage): "wrong" for pid in self.ids for stage in ("T1", "T2", "T3")})
        result = self.run_fixture()
        self.assertEqual(result["status"], "COMPLETE_NATIVE_LADDER_RECOVERY", result.get("reason"))
        self.assertEqual((result["final_accuracy"], result["recovered_count"], result["recovered_fraction"]), (0., 0, 0.))
        self.assertEqual(result["note_count"], 0)
        self.assertEqual(len(self.engine.calls), 30)

    def test_payload_contains_only_own_status_answer_public_schema_and_literal_history(self):
        self.mixed()
        result = self.run_fixture()
        for call in result["calls"]:
            if call["stage"] != "NOTE":
                self.assertNotIn("gold", call["feedback"])
                self.assertNotIn("wrong_slots", call["feedback"])
                self.assertNotIn('"cause"', call["feedback"])
                self.assertIsNone(call["input"]["messages"])
                continue
            entry = next(e for e in result["trees"] if e["problem_id"] == call["problem_id"])
            text = call["input"]["messages"][0]["content"][-1]["text"]
            transcript = json.loads(text.split("<read_only_attempts>\n", 1)[1].split("\n</read_only_attempts>", 1)[0])
            samples = [group["samples"][0] for group in entry["tree"]["groups"]]
            for row, sample in zip(transcript, samples, strict=True):
                self.assertEqual("".join(line for _, line in row["source_lines"]), sample["text"])
                self.assertEqual(set(row), {"attempt", "parent", "turn", "status", "finish_reason", "answer", "line_count", "source_lines"})
            self.assertNotIn('"episode_audit"', text)
            self.assertNotIn('"required_slots"', text)

    def test_deadline_failure_preserves_partial_artifacts_and_null_metrics(self):
        self.engine.after_sample = lambda engine, request: (_ for _ in ()).throw(RuntimeError("fixture sampler failed")) if len(engine.calls) == 2 else None
        result = self.run_fixture()
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["partial_first_attempt_count"], 1)
        self.assertTrue((self.root / "out/first/response-0000.json").is_file())
        self.assertTrue(all(result[key] is None for key in recovery._METRICS))
        self.engine = FixtureEngine(self.rows)
        expired = self.run_fixture("expired", deadline=time.monotonic() - 1)
        self.assertEqual(expired["status"], "INCOMPLETE")
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(expired["generation_calls"], 0)

    def test_deadline_before_retry_no_overshoot_or_partial_credit(self):
        self.mixed()
        original = generation.generate_ladder
        late = False
        def first(*args, **kwargs):
            nonlocal late
            value = original(*args, **kwargs)
            late = True
            return value
        with patch.object(generation, "generate_ladder", side_effect=first), patch.object(recovery.time, "monotonic", side_effect=lambda: 1000. if late else 0.):
            result = self.run_fixture(deadline=100)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["error_type"], "TimeoutError")
        self.assertEqual(len(self.engine.calls), 10)
        self.assertEqual(result["usage"], result["generation"]["usage"])
        self.assertTrue(all(result[key] is None for key in recovery._METRICS))

    def test_nooverwrite_wrong_pin_and_source_mutation(self):
        result = self.run_fixture()
        original = (self.root / "out/result.json").read_bytes()
        for path in (self.root / "out", self.cohort / "out"):
            with self.assertRaisesRegex(ValueError, "preserve"):
                recovery.diagnose(self.cohort, path, self.engine, self.templates, expected_preparation_sha256=self.sha,
                    local_protocol_sha256="a" * 64, deadline=time.monotonic() + 30)
        self.assertEqual((self.root / "out/result.json").read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "pinned"):
            recovery.diagnose(self.cohort, self.root / "wrong", self.engine, self.templates,
                expected_preparation_sha256="0" * 64, local_protocol_sha256="a" * 64, deadline=time.monotonic() + 30)
        self.engine = FixtureEngine(self.rows)
        sources = recovery.code_hashes()
        with patch.object(recovery, "code_hashes", side_effect=[sources, sources | {"changed": "a" * 64}]):
            changed = self.run_fixture("changed")
        self.assertEqual(changed["status"], "INCOMPLETE")
        self.assertIsNone(changed["final_accuracy"])
        with patch.object(recovery, "code_hashes", return_value=sources | {"changed": "a" * 64}):
            with self.assertRaisesRegex(ValueError, "pinned source"):
                recovery.replay(self.cohort, result, expected_preparation_sha256=self.sha)

    def test_paired_and_tamper_rejections(self):
        self.mixed()
        left = self.run_fixture("f0")
        self.engine = FixtureEngine(self.rows, version=4)
        right = self.run_fixture("ft")
        self.assertEqual(left["status"], "COMPLETE_NATIVE_LADDER_RECOVERY", left.get("reason"))
        self.assertEqual(right["status"], "COMPLETE_NATIVE_LADDER_RECOVERY", right.get("reason"))
        paired = recovery.validate_paired_records(left, right)
        self.assertEqual(len(paired["first_gains"]), 5)
        self.assertEqual(paired["first_losses"], [])
        self.assertEqual(paired["first_attempt_accuracy_delta"], .5)
        self.assertFalse(paired["model_generation_provenance_verified"])
        for field, value in (("final_accuracy", 1), ("note_verified_count", 4), ("status", "INCOMPLETE")):
            changed = copy.deepcopy(left)
            changed[field] = value
            self.rehash(changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                recovery.validate_paired_records(changed, right)
        for field, value in (("seed", 3), ("parent_sample_id", "invented"), ("feedback", "correct answer is private")):
            changed = copy.deepcopy(left)
            changed["calls"][0][field] = value
            self.rehash(changed["calls"][0])
            self.rehash(changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                recovery.replay(self.cohort, changed, expected_preparation_sha256=self.sha)
        changed = copy.deepcopy(left)
        changed["calls"][0]["input"]["prompt_token_ids"][0] += 1
        changed["calls"][0]["input_sha256"] = generation._digest(changed["calls"][0]["input"])
        self.rehash(changed["calls"][0])
        self.rehash(changed)
        with self.assertRaises(ValueError):
            recovery.replay(self.cohort, changed, expected_preparation_sha256=self.sha)

    def test_unverifiable_skipped_and_mixed_episodes_rejected(self):
        self.mixed()
        result = self.run_fixture()
        for mutation in ("skip", "mixed", "unverifiable", "training", "omit"):
            changed = copy.deepcopy(result)
            tree = changed["trees"][0]["tree"]
            if mutation == "skip":
                tree["notebook"]["skipped_branches"] = [{"kind": "T2", "status": "BUDGET_STOP"}]
            elif mutation == "mixed":
                tree["groups"][0]["samples"][0]["policy_version"] = 2
            elif mutation == "unverifiable":
                tree["groups"][0]["samples"][0]["result"]["status"] = "UNVERIFIABLE"
            elif mutation == "training":
                tree["groups"][0]["usable"] = True
            else:
                changed["trees"].pop()
            self.rehash(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                recovery.replay(self.cohort, changed, expected_preparation_sha256=self.sha)
        self.assertFalse(torch.cuda.is_initialized())

    def test_final_note_crossing_deadline_is_saved_but_incomplete(self):
        pid = self.ids[0]
        self.engine = FixtureEngine(self.rows, script={(pid, "T1"): "wrong"})
        late = False
        def finish(engine, request):
            nonlocal late
            late = request.request_id.endswith(":NOTE")
        self.engine.after_sample = finish
        with patch.object(recovery.time, "monotonic", side_effect=lambda: 1000. if late else 0.):
            result = self.run_fixture(deadline=100)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["error_type"], "TimeoutError")
        self.assertEqual(result["calls"][-1]["stage"], "NOTE")
        self.assertTrue((self.root / "out/call-0001.json").is_file())
        self.assertTrue(all(result[key] is None for key in recovery._METRICS))
        self.assertEqual(len(self.engine.calls), 12)

    def test_invalid_retry_completion_raw_survives_and_no_note_is_called(self):
        self.engine = FixtureEngine(self.rows, script={(self.ids[0], "T1"): "wrong"})
        original = self.engine.sample
        def malformed(requests):
            results = original(requests)
            if requests[0].request_id.endswith(":T2"):
                results[0].completions[0].logprobs.pop()
            return results
        self.engine.sample = malformed
        result = self.run_fixture()
        self.assertEqual(result["status"], "INCOMPLETE")
        raw = json.loads((self.root / "out/call-0000.json").read_bytes())
        self.assertEqual(len(raw["logprobs"]) + 1, len(raw["completion_token_ids"]))
        self.assertEqual(len(self.engine.calls), 11)

    def test_protocol_boolean_or_float_budgets_rejected_even_with_rehashed_record(self):
        result = self.run_fixture()
        for key, value in (("n", True), ("max_attempts", 3.), ("max_new_tokens", 1024.)):
            changed = copy.deepcopy(result)
            changed["protocol"][key] = value
            changed["protocol_sha256"] = generation._digest(changed["protocol"])
            self.rehash(changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "protocol changed"):
                recovery.replay(self.cohort, changed, expected_preparation_sha256=self.sha)

    def test_pair_fresh_replay_rejects_rehashed_false_note_success(self):
        self.mixed()
        left = self.run_fixture("f0")
        self.engine = FixtureEngine(self.rows, version=1)
        right = self.run_fixture("ft")
        for mutation in ("summary", "invalid_json", "length", "attempt_checks", "bool_reward", "integer_checks"):
            changed = copy.deepcopy(left)
            if mutation == "summary":
                note = next(entry["note"] for entry in changed["trees"]
                    if entry["note"] is not None and entry["note"]["checked"]["reward"] == 0.)
                checked = note["checked"]
                checked.update(reward=1., failure_code="NONE", checks={key: True for key in checked["checks"]})
                changed.update(note_verified_count=4, note_verified_fraction=1.)
            else:
                note = next(entry["note"] for entry in changed["trees"]
                    if entry["note"] is not None and entry["note"]["checked"]["reward"] == 1.)
                call = next(row for row in changed["calls"] if row["request_id"] == note["request_id"])
                if mutation == "bool_reward":
                    note["checked"]["reward"] = True
                elif mutation == "integer_checks":
                    note["checked"]["checks"] = {key: 1 for key in note["checked"]["checks"]}
                elif mutation == "invalid_json":
                    call["text"] = "{invalid JSON"
                elif mutation == "length":
                    call["finish_reason"] = "length"
                    changed["length_terminated_calls"] += 1
                else:
                    note["checked"]["attempt_checks"][0]["history"] = False
                self.rehash(call)
            self.rehash(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                recovery.validate_paired_records(changed, right)

    @staticmethod
    def rehash(value):
        value["record_sha256"] = generation._digest({k: v for k, v in value.items() if k != "record_sha256"})


if __name__ == "__main__":
    unittest.main()
