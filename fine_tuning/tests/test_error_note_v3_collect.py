"""Authored CPU integration cases for audited lessons and immutable collection inputs."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from fine_tuning.error_note import collect, collect_v3
from fine_tuning.error_note.audit import validate_row_audits
from fine_tuning.error_note.contract import NOTE_INSTRUCTION_V3, note_prompt_v3
from fine_tuning.notebook_rl.prompts import p0_eval_text
from fine_tuning.modules import BASE_REVISION
from fine_tuning.tests.test_error_note_v3_audit import PROVENANCE, example, review_for, tokenize
from fine_tuning.tests.test_error_note_v2_collect import FakeStream, KWARGS, load


class CollectionFixture:
    def __init__(self, root, *, correct=True, reviewed=True):
        self.out = Path(root)
        self.record = example(correct=correct)
        self.pid = self.record["problem_id"]
        self.actor = SimpleNamespace(problem_id=self.pid, question="Compare these curves.", choices=("red", "blue"),
                                     image_paths=("fixture.png",), mode="plain")
        self.record["p0_text"] = p0_eval_text(self.actor.question, self.actor.choices)
        self.record["note"]["prompt"] = note_prompt_v3(self.record["p0_text"], self.record["t1"]["text"],
                                                      self.record["t1"]["status"], "B")
        self.items = [{k: self.record[k] for k in ("problem_id", "subject", "source")}]
        order = {"version": "error-note-order-v1", "master_seed": 3407, "count": 1, "items": self.items,
                 "ids_sha256": collect_v3.text_sha256(self.pid)}
        collect.write_json(self.out / "order.json", order)
        model = self.out / "models" / BASE_REVISION
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"cpu identity fixture only")
        self.args = SimpleNamespace(build_root=self.out / "vault", order=self.out / "order.json", limit=1,
                                    output=self.out, answer_contract="training-answer-v2",
                                    engine="hf", model_path=model, code_commit="a" * 40,
                                    training_rules=self.out / "rules.json", training_rules_sha256="1" * 64,
                                    source_provenance=self.out / "source.json", reviews=self.out / "reviews.json",
                                    audit=self.out / "audit.json")
        collect_v3.bind_collection(self.args, self.items)
        collect.append_jsonl(self.out / "t1.jsonl", {"problem_id": self.pid, **self.record["t1"]})
        collect.append_jsonl(self.out / "note.jsonl", {"problem_id": self.pid, "note_version": "v2",
                                                       "note_instruction_sha256": collect_v3.text_sha256(NOTE_INSTRUCTION_V3),
                                                       **self.record["note"]})
        collect.write_json(self.args.source_provenance, {"version": "note-source-provenance-v1",
                                                        "items": {self.pid: PROVENANCE}})
        collect.write_json(self.args.reviews, {"version": "note-reviews-v1",
                                               "items": {self.pid: review_for(self.record)} if reviewed else {}})

    @contextlib.contextmanager
    def vault(self):
        with mock.patch("fine_tuning.notebook_rl.vault.read_actors", return_value=[self.actor]), \
             mock.patch("fine_tuning.notebook_rl.vault.read_verifier",
                        return_value={self.pid: SimpleNamespace(gold="B")}):
            yield


class AuditedCollectTests(unittest.TestCase):
    def test_cli_rejects_old_training_contract_instead_of_reinterpreting_it(self):
        common = ["rows", "--build-root", "/vault", "--order", "/order.json", "--limit", "1",
                  "--output", "/collect", "--tokenizer-path", "/model", "--answer-contract"]
        self.assertEqual(collect._parser().parse_args(common + ["training-answer-v2"]).answer_contract,
                         "training-answer-v2")
        error = io.StringIO()
        with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
            collect._parser().parse_args(common + ["training-answer-v1"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("invalid choice: 'training-answer-v1'", error.getvalue())

    def test_reviewed_note_transfer_and_rows_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CollectionFixture(tmp)
            with f.vault():
                result = collect_v3.audit_collection(f.args)
                self.assertEqual(result["usable_lessons"], 1)
                lessons = collect_v3.verify_lessons(f.args)
                self.assertEqual(set(lessons[f.pid]), {"lesson", "skip"})
                rows, drops, summary = collect_v3.collect_rows_v3(f.args, tokenize)
            self.assertEqual([r["kind"] for r in rows], ["A", "B"])
            self.assertEqual(drops, [])
            self.assertEqual(summary["rows_version"], "v3")
            self.assertEqual(set(summary["length_finishes"]), {"t1", "note", "resolve"})
            validate_row_audits(rows, f.args.audit)

    def test_unreviewed_note_does_not_enable_resolve_or_B(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CollectionFixture(tmp, reviewed=False)
            with f.vault():
                collect_v3.audit_collection(f.args)
                lessons = collect_v3.verify_lessons(f.args)
                rows, drops, _ = collect_v3.collect_rows_v3(f.args, tokenize)
            self.assertIsNone(lessons[f.pid]["lesson"])
            self.assertEqual(lessons[f.pid]["skip"], "CONTENT_UNKNOWN")
            self.assertEqual([r["kind"] for r in rows], ["A"])
            self.assertTrue(any(r["kind"] == "B" for r in drops))

    def test_same_size_note_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CollectionFixture(tmp)
            with f.vault():
                collect_v3.audit_collection(f.args)
                path = f.out / "note.jsonl"
                before = path.read_bytes()
                after = before.replace(b"fact0", b"fake0")
                self.assertEqual(len(before), len(after))
                self.assertNotEqual(before, after)
                path.write_bytes(after)
                with self.assertRaises(ValueError):
                    collect_v3.verify_lessons(f.args)

    def test_changed_lesson_cannot_reuse_original_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CollectionFixture(tmp)
            with f.vault():
                collect_v3.audit_collection(f.args)
                path = f.out / "lessons-v3.json"
                data = json.loads(path.read_text())
                data["items"][f.pid]["lesson"] = "Choose blue."
                collect.write_json(path, data)
                with self.assertRaisesRegex(ValueError, "stale, altered"):
                    collect_v3.verify_lessons(f.args)

    def test_resolve_requires_closed_row_and_preserves_generated_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CollectionFixture(tmp, correct=False)
            with f.vault():
                collect_v3.audit_collection(f.args)
                lessons = collect_v3.verify_lessons(f.args)
                with self.assertRaisesRegex(ValueError, "terminal resolve"):
                    collect_v3.collect_rows_v3(f.args, tokenize)
                response = " ".join(f"step{i}" for i in range(40)) + " Final answer: B"
                stream = FakeStream(lambda request: {"text": response, "finish_reason": "stop",
                                                     "prompt_tokens": 20, "completion_tokens": 45, "seconds": 0.1})
                collect.run_resolve(f.items, {f.pid: f.actor}, {f.pid: f.record["t1"]}, lessons,
                                    load, stream, collect.StreamStats(), lambda *a: ("CORRECT", "B"), f.out, **KWARGS)
                rows, _, _ = collect_v3.collect_rows_v3(f.args, tokenize)
            solution = next(r for r in rows if r["kind"] == "A")
            self.assertEqual(solution["target"], response)
            self.assertEqual(solution["target_source"], "RESOLVE")
            self.assertEqual(solution["prompt"], f.record["p0_text"])
            self.assertNotIn("Lesson from", solution["prompt"])
            validate_row_audits(rows, f.args.audit)

    def test_collection_rejects_changed_rules_and_legacy_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CollectionFixture(tmp)
            args = copy.copy(f.args)
            args.training_rules_sha256 = "2" * 64
            with self.assertRaisesRegex(ValueError, "immutable evidence"):
                collect_v3.bind_collection(args, f.items)
            (f.out / "collection.json").unlink()
            with self.assertRaisesRegex(ValueError, "legacy collection"):
                collect_v3.bind_collection(f.args, f.items)

    def test_context_blocked_resolve_records_drop_and_keeps_reviewed_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CollectionFixture(tmp, correct=False)
            with f.vault():
                collect_v3.audit_collection(f.args)
                lessons = collect_v3.verify_lessons(f.args)
                stream = FakeStream(lambda request: {"blocked": True, "error": "context too long"})
                collect.run_resolve(f.items, {f.pid: f.actor}, {f.pid: f.record["t1"]}, lessons,
                                    load, stream, collect.StreamStats(), lambda *a: self.fail("blocked input graded"),
                                    f.out, **KWARGS)
                rows, drops, _ = collect_v3.collect_rows_v3(f.args, tokenize)
            self.assertEqual([r["kind"] for r in rows], ["B"])
            self.assertTrue(any(r["kind"] == "A" and r["reason"] == "NOT_STOPPED" for r in drops))
            validate_row_audits(rows, f.args.audit)

    def test_changed_engine_model_or_source_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CollectionFixture(tmp)
            for key, value in (("engine", "vllm"), ("code_commit", "b" * 40)):
                args = copy.copy(f.args)
                setattr(args, key, value)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "immutable evidence"):
                    collect_v3.bind_collection(args, f.items)
            with mock.patch.object(collect_v3, "source_identity", return_value="2" * 64), \
                 self.assertRaisesRegex(ValueError, "immutable evidence"):
                collect_v3.bind_collection(f.args, f.items)
            (f.args.model_path / "model.safetensors").write_bytes(b"changed snapshot weights")
            with self.assertRaisesRegex(ValueError, "immutable evidence"):
                collect_v3.bind_collection(f.args, f.items)

    def test_audit_requires_and_reuses_original_generation_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CollectionFixture(tmp)
            args = copy.copy(f.args)
            del args.engine
            with mock.patch.object(collect_v3, "generation_identity", side_effect=AssertionError("model read")):
                value = collect_v3.bind_collection(args, f.items)
            self.assertEqual(value["generation"]["precision"], "nf4")
            (f.out / "collection.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing generation binding"):
                collect_v3.bind_collection(args, f.items)

    def test_private_checker_returns_only_lessons_and_hides_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CollectionFixture(tmp)
            allowed = {f.pid: {"lesson": "Read the axes.", "skip": None}}
            result = SimpleNamespace(returncode=0, stdout=json.dumps(allowed), stderr="")
            with mock.patch.object(collect_v3.subprocess, "run", return_value=result) as run:
                self.assertEqual(collect_v3.verified_lessons_for_actor(f.args), allowed)
                self.assertIn("verify-lessons", run.call_args.args[0])
            result.stdout = json.dumps({f.pid: {"lesson": "x", "skip": None, "gold": "B"}})
            with mock.patch.object(collect_v3.subprocess, "run", return_value=result), self.assertRaises(ValueError):
                collect_v3.verified_lessons_for_actor(f.args)
            result.returncode, result.stderr = 1, "private gold B"
            with mock.patch.object(collect_v3.subprocess, "run", return_value=result), \
                 self.assertRaisesRegex(ValueError, "^v3 private lesson verification failed$"):
                collect_v3.verified_lessons_for_actor(f.args)


if __name__ == "__main__":
    unittest.main()
