"""Authored CPU fixtures exercise evaluation contracts, never model performance."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fine_tuning.notebook_rl import circuit_trace, episode_notebook, local_eval, prompts
from fine_tuning.notebook_rl import local_ladder_generation as generation
from fine_tuning.notebook_rl import local_ladder_recovery as recovery
from fine_tuning.notebook_rl import notebook_synthesis as synthesis, whole_reference_notebook as reference
from fine_tuning.notebook_rl.cli import write_pool
from fine_tuning.notebook_rl.dataset import _bytes, _sha, _write
from fine_tuning.notebook_rl.generators import circuit, graph
from fine_tuning.notebook_rl.generators.common import GeneratedProblem
from fine_tuning.notebook_rl.local_ladder import INDICES, SEED
from fine_tuning.notebook_rl.local_ladder_gate import _validate_protocol
from fine_tuning.notebook_rl.schema import ProblemTree, from_json
from fine_tuning.tests import test_notebook_rl_circuit_trace as trace_fixture
from fine_tuning.tests import test_notebook_rl_local_eval as runtime_fixture
from fine_tuning.tests import test_notebook_rl_local_ladder_recovery as legacy_fixture


class ReferenceEngine(legacy_fixture.FixtureEngine):
    context_limit = 24576

    def _answer(self, pid, outcome):
        problem = self.rows[pid]
        text = trace_fixture.response(problem, trace_fixture.fixture_trace(problem))
        if outcome == "wrong":
            text = text.rsplit("Final answer:", 1)[0] + "Final answer: 999999 A"
        return text, "stop"

    def _note(self, request, pid):
        body = request.messages[0]["content"][-1]["text"]
        originals = json.loads(body.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
        inventory = json.loads(body.split("<public_inventory>\n")[1].split("\n</public_inventory>")[0])
        attempts = []
        for original in originals:
            text = "".join(line for _, line in original["source_lines"])
            answer = text.rsplit("Final answer: ", 1)[1]
            attempts.append([original["attempt"], "CORRECT" if answer == self.rows[pid].verifier.gold
                             else "INCORRECT", answer])
        note = {"corrected_facts": self.rows[pid].verifier.slots, "attempts": attempts,
                "assessments": [[ref, "N", [], None] for ref in inventory["assessment_refs"]],
                "success_attempt": originals[-1]["attempt"], "corrected_answer": self.rows[pid].verifier.gold}
        if pid == self.bad_note:
            note["attempts"].pop()
        return json.dumps(note), "stop"


class ReferenceLadderEvaluationTests(unittest.TestCase):
    rows: list[GeneratedProblem]

    @classmethod
    def setUpClass(cls):
        cls.rows = [circuit.generate(index, SEED, "sealed", "obs") for index in INDICES]

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cohort = self.root / "cohort"
        write_pool(self.cohort, self.rows, recipe={"fixture_only": True})
        preparation = {"format": "notebook-rl-local-ladder-preparation-v1",
            "status": "CPU_COHORT_PREPARED_PENDING_GATES", "training_allowed": False,
            "evaluation_allowed": False, "gpu_evaluation_completed": False, "count": 10,
            "records_manifest": generation._descriptor(self.cohort / "records/manifest.yaml"),
            "problem_ids": [row.actor.problem_id for row in self.rows]}
        _write(self.cohort / "preparation.json", _bytes(preparation))
        self.sha = _sha(_bytes(preparation))
        self.templates = generation.circuit_evaluation_prompts(generation.CIRCUIT_REFERENCE_PROMPT_POLICY)

    def test_reference_generation_strict_replay_and_tampering(self):
        ids = [row.actor.problem_id for row in self.rows]
        engine = ReferenceEngine(self.rows, script={(pid, "T1"): "wrong" for pid in ids[:2]}, bad_note=ids[1])
        record = recovery.diagnose(self.cohort, self.root / "evaluation", engine, self.templates,
            expected_preparation_sha256=self.sha, local_protocol_sha256="a" * 64,
            deadline=time.monotonic() + 180, context_limit=24576)
        self.assertEqual(record["status"], "COMPLETE_NATIVE_LADDER_RECOVERY", record.get("reason"))
        self.assertEqual((record["note_count"], record["note_verified_count"]), (2, 1))
        _validate_protocol(record["protocol"])
        checked = recovery.replay(self.cohort, record, expected_preparation_sha256=self.sha)
        self.assertEqual(checked["note_verified_count"], 1)
        self.assertNotEqual(checked["verifier_process"]["pid"], record["verifier_process"]["pid"])
        note_calls = [row for row in record["calls"] if row["stage"] == "NOTE"]
        self.assertTrue(all(row["max_new_tokens"] == 8192 and row["n"] == 1 for row in note_calls))
        for call in note_calls:
            body = call["input"]["messages"][0]["content"][-1]["text"]
            self.assertIn("<public_inventory>", body)
            self.assertIn("<read_only_solution_contract>", body)
            self.assertNotIn("<unverified_fresh_observation>", body)
            self.assertNotIn("<unverified_pair_drafts>", body)
            self.assertEqual(call["input"]["messages"][0]["content"][0], {"type": "image"})
        for entry in record["trees"]:
            self.assertTrue(all(not group["usable"] for group in entry["tree"]["groups"]))
        tree = from_json(ProblemTree, json.dumps(record["trees"][0]["tree"]))
        with recovery._verifier(self.cohort) as verifier:
            training = synthesis.audit_episode(tree, self.rows[0].actor, verifier, contract=synthesis.BOUNDED_VERSION)
            self.assertFalse(training["complete_episode"])
            audit = record["trees"][0]["episode_audit"]
            with self.assertRaises(ValueError):
                reference.grade_note(note_calls[0]["text"], "stop", tree, audit, verifier,
                    actor=self.rows[0].actor, contract=synthesis.BOUNDED_VERSION)
        for mutation in ("history", "image", "protocol", "raw", "metric", "training"):
            changed = copy.deepcopy(record)
            row = next(row for row in changed["calls"] if row["stage"] == "NOTE")
            if mutation == "history":
                row["input"]["messages"][0]["content"][-1]["text"] += "removed original history"
            elif mutation == "image":
                row["input"]["pixel_tensors"] = {}
            elif mutation == "protocol":
                changed["protocol"]["notebook"]["max_new_tokens"] = 2048
                changed["protocol_sha256"] = generation._digest(changed["protocol"])
            elif mutation == "raw":
                row["text"] = "{}"
            elif mutation == "metric":
                changed["trees"][0]["note"]["checked"]["reward"] = 0.0
            else:
                changed["trees"][0]["tree"]["groups"][0]["usable"] = True
            row["input_sha256"] = generation._digest(row["input"])
            row["record_sha256"] = generation._digest({key: value for key, value in row.items() if key != "record_sha256"})
            changed["record_sha256"] = generation._digest({key: value for key, value in changed.items() if key != "record_sha256"})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                recovery.replay(self.cohort, changed, expected_preparation_sha256=self.sha)

    def test_protocol_budgets_public_grammar_and_legacy_are_separate(self):
        contract = recovery.protocol(self.templates, "a" * 64, context_limit=24576)
        note = contract["notebook"]
        self.assertEqual(note["metric_identity"], {"reference": reference.VERSION,
            "bounded": synthesis.BOUNDED_VERSION, "circuit": circuit_trace.VERSION})
        self.assertEqual(note["generation_contract"], reference.evaluation_generation_contract())
        self.assertEqual(contract["solution_prompt_contract"], prompts.circuit_solution_prompt_contract())
        for key in ("fresh_observation_calls", "unverified_draft_calls", "repair_calls"):
            self.assertEqual(note[key], 0)
        self.assertFalse(note["training_assistance_equivalent"])
        for limit in (16384, 8192, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                recovery.protocol(self.templates, "a" * 64, context_limit=limit)
        legacy = recovery.protocol(generation.circuit_evaluation_prompts(), "a" * 64)
        self.assertEqual(legacy["notebook"]["max_new_tokens"], 2048)
        self.assertEqual(legacy["notebook"]["metric_identity"], episode_notebook.VERSION)
        self.assertNotIn("solution_prompt_contract", legacy)
        actor = self.rows[0].actor
        body = prompts.turn1_text(actor, self.templates)
        self.assertIn(circuit_trace.VERSION, body)
        self.assertIn(prompts.p0_body(actor.question, ()), body)
        for invalid in (replace(actor, mode="plain"), replace(actor, source=graph.GENERATOR_VERSION),
                        replace(actor, family="graph"), replace(actor, question=circuit._question(("voltage", "N2")))):
            with self.subTest(actor=invalid), self.assertRaises(ValueError):
                prompts.for_actor(invalid, self.templates)
        for key in ("turn1", "retry", "instructions", "scope"):
            changed = copy.deepcopy(contract)
            changed["solution_prompt_contract"][key] = "mutated"
            with self.subTest(field=key), self.assertRaises(ValueError):
                _validate_protocol(changed)

    def test_cli_requires_recovery_and_dispatches_explicit_policy(self):
        policy = generation.CIRCUIT_REFERENCE_PROMPT_POLICY
        with self.assertRaises(ValueError):
            local_eval._evaluate(self.root, self.root, self.root, condition="F0", reload_result=self.root,
                minimal_capture=self.root, minimal_reload=self.root, ladder_cohort=self.root,
                evaluation_prompt_policy=policy)
        args = ["evaluate-ladder", "--capture", str(self.root / "capture"), "--cohort", str(self.cohort),
                "--data-gate", str(self.root / "gate"), "--reload-result", str(self.root / "reload"),
                "--minimal-capture", str(self.root / "minimum"), "--minimal-reload", str(self.root / "minimum-reload"),
                "--output", str(self.root / "out"), "--condition", "F0", "--arm", "notebook",
                "--with-recovery", "--evaluation-prompt-policy", policy]
        with patch.object(local_eval, "evaluate_ladder", return_value={"status": "COMPLETE_NATIVE_LADDER_RECOVERY"}) as evaluate:
            self.assertEqual(local_eval.main(args), 0)
        self.assertEqual(evaluate.call_args.kwargs["evaluation_prompt_policy"], policy)
        self.assertTrue(evaluate.call_args.kwargs["with_recovery"])

    def test_mocked_runtime_and_paired_comparison_keep_reference_budget(self):
        from fine_tuning.notebook_rl import local_train

        runtime = runtime_fixture.NativeLadderRuntimeTests()
        self.addCleanup(runtime.doCleanups)
        runtime.setUp()
        parent = runtime.exploratory_contract(context_limit=24576)
        runtime.templates = self.templates
        runtime.generation["protocol"] = generation.protocol(self.templates, runtime.protocol["sha256"],
                                                             context_limit=24576)
        record = runtime.recovery_record()
        with patch.object(local_train, "training_parent_gate", return_value=parent), \
                patch.object(recovery, "diagnose", return_value=record) as diagnose:
            after = local_eval.evaluate_ladder(runtime.capture, runtime.cohort, runtime.receipt,
                runtime.root / "reference-runtime", condition="FT", arm="notebook", reload_result=runtime.reload,
                minimal_capture=runtime.root / "minimum", minimal_reload=runtime.root / "minimum-reload",
                with_recovery=True, evaluation_prompt_policy=generation.CIRCUIT_REFERENCE_PROMPT_POLICY)
        self.assertEqual(after["status"], "COMPLETE_NATIVE_LADDER_RECOVERY")
        self.assertEqual(after["conditions"]["notebook_output_tokens"], 8192)
        self.assertEqual(diagnose.call_args.args[3], self.templates)
        self.assertEqual(diagnose.call_args.kwargs["context_limit"], 24576)
        before = copy.deepcopy(after)
        before.update(condition="F0", initial_trainable_sha256=runtime.initial, checkpoint_training_updates=0)
        before["recovery"]["policy_version"] = before["generation"]["policy_version"] = 0
        with patch.object(recovery, "validate_paired_records", return_value={"n_pairs": 10}):
            self.assertEqual(local_eval.compare(before, after)["status"], "COMPLETE_PAIRED_NATIVE_LADDER_RECOVERY")
            before["conditions"]["notebook_output_tokens"] = 2048
            after["conditions"]["notebook_output_tokens"] = 2048
            with self.assertRaisesRegex(ValueError, "canonical recorded execution"):
                local_eval.compare(before, after)
