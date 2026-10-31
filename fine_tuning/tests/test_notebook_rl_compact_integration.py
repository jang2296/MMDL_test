"""Compact final-note contract wired through profile, generation, credit, update and evidence.

Authored CPU fixtures with a scripted engine and tiny parameter vectors; not Qwen behaviour.
"""
from __future__ import annotations

import copy
import json
import time
import unittest

from fine_tuning.notebook_rl import (compact_note, episode_notebook as episode, local_ladder_recovery as recovery,
                                     local_profile, repair_evidence, repair_notebook as notes)
from fine_tuning.notebook_rl.checkpoint import make_optimizer
from fine_tuning.notebook_rl.repair_contract import FINAL_VERSION
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.tests import test_notebook_rl_repair_update as update_fixture

# Frozen default repair protocol; any change here means v2 behaviour or its hash moved.
DEFAULT_REPAIR_PROTOCOL_SHA256 = "89181dfc7e8e4d306c03d0b340f198a6d2d59acee5b988eaa6ab86eb36dbad33"


def compact_protocol() -> dict:
    return local_profile.build_protocol(repair_notebook=True, context_limit=24576,
                                        note_contract=compact_note.VERSION)


class CompactProfileTests(unittest.TestCase):
    def test_default_contract_and_protocol_hash_are_unchanged(self) -> None:
        default = local_profile.build_protocol(repair_notebook=True, context_limit=24576)
        self.assertEqual(default["sha256"], DEFAULT_REPAIR_PROTOCOL_SHA256)
        self.assertEqual(notes.contract(), notes.contract(FINAL_VERSION))
        self.assertEqual(notes.note_contract(default), FINAL_VERSION)
        self.assertNotIn("s_definition", local_profile.repair_objective_contract())

    def test_compact_protocol_restores_threads_prompts_and_objective(self) -> None:
        protocol = compact_protocol()
        self.assertNotEqual(protocol["sha256"], DEFAULT_REPAIR_PROTOCOL_SHA256)
        self.assertEqual(local_profile.restore_protocol(json.loads(json.dumps(protocol))), protocol)
        prompts = local_profile.local_prompts(protocol)
        notes._prompts(prompts)
        self.assertEqual(notes.note_contract(prompts), compact_note.VERSION)
        selected = prompts["repair_notebook_contract"]
        self.assertEqual((selected["version"], selected["response_contract"], selected["note_renderer"]),
                         (notes.COMPACT_VERSION, compact_note.VERSION, "compact-history-v1"))
        self.assertEqual((selected["n"], selected["max_new_tokens"], selected["context_limit"]), (4, 8192, 24576))
        self.assertEqual(selected["prompt_sha256"], _hash(compact_note.PROMPT))
        self.assertEqual(protocol["repair_final_contract"], compact_note.VERSION)
        self.assertEqual(protocol["repair_objective"]["s_definition"], "verified-note-claim-v1")

    def test_mixed_or_unknown_contracts_fail_closed(self) -> None:
        protocol = compact_protocol()
        for mutate in (lambda p: p["repair_notebook_contract"].update(response_contract="junk"),
                       lambda p: p["repair_notebook_contract"].update(prompt_sha256="0" * 64),
                       lambda p: p.update(repair_final_contract=FINAL_VERSION)):
            changed = copy.deepcopy(protocol)
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                local_profile.restore_protocol(changed)
        prompts = local_profile.local_prompts(protocol)
        prompts["repair_notebook_version"] = notes.VERSION
        with self.assertRaises(ValueError):
            notes._prompts(prompts)
        with self.assertRaises(ValueError):
            local_profile.build_protocol(note_contract=compact_note.VERSION)
        with self.assertRaises(ValueError):
            notes.contract("repair-note-v3")


class CompactPipelineTests(unittest.TestCase):
    """Generation -> fresh grading -> group credit -> real tiny optimizer update -> evidence."""

    def setUp(self) -> None:
        self.u = update_fixture.RepairUpdateTests()
        self.u.setUp()
        self.addCleanup(self.u.doCleanups)
        self.u.protocol = compact_protocol()
        self.u.optimizer, self.u.scheduler = make_optimizer(self.u.bundle, self.u.protocol)
        self.f = self.u.f
        self.f.fixture.prompts = local_profile.local_prompts(self.u.protocol)
        self.tree, self.engine = self.f.online(wrong_answer=False, visual=True)
        value = next(line for line in self.f.fixture.right.splitlines()
                     if line.startswith("edge[A,B]")).split(" = ", 1)[1]
        self.notes = [
            json.dumps({"errors": [{"attempt": 1, "type": "OBSERVATION", "item": "edge[A,B]", "correct": value}],
                        "vision_misread": "YES", "lesson": "FREE_LESSON"}),
            json.dumps({"errors": [{"attempt": 1, "type": "OBSERVATION", "item": "edge[A,G]", "correct": "x"}],
                        "vision_misread": "NO"}),
            json.dumps({"errors": [], "vision_misread": "UNSURE"}), "{"]
        self.f.attach(self.tree, self.engine, list(self.notes))

    def test_note_request_is_readable_full_history_without_v2_payload(self) -> None:
        record = self.tree.notebook["rewarded_notebook"]
        self.assertEqual((record["status"], record["notebook_generation_policy"]), ("GENERATED", notes.COMPACT_VERSION))
        request = self.engine.requests[-1]
        body = request.messages[0]["content"][-1]["text"]
        attempts = episode.solution_samples(self.tree)
        self.assertEqual((request.n, request.max_new_tokens), (4, 8192))
        self.assertTrue(body.endswith(compact_note.PROMPT))
        self.assertNotIn("<read_only_history>", body)
        self.assertEqual(body.count("=== Attempt "), len(attempts))
        self.assertTrue(all(row.text in body for row in attempts))
        observation = self.tree.notebook["repair_collector"]["fresh_observation"]
        self.assertIn(observation["completion"]["text"], body)

    def test_replay_journal_expects_the_compact_generation_version(self) -> None:
        from fine_tuning.notebook_rl.repair_replay import _expected_journal

        expected = [row["raw"] for row in _expected_journal([self.tree])
                    if row["kind"] == "REPAIR_RAW" and row["raw"]["purpose"].startswith("REPAIR_FINAL")]
        emitted = [event for event in self.f.fixture.events if event["purpose"].startswith("REPAIR_FINAL")]
        self.assertEqual(_hash(expected), _hash(emitted))
        self.assertTrue(all(event["version"] == notes.COMPACT_VERSION for event in emitted))

    def test_fresh_grades_regrade_and_byte_identical_recomputation(self) -> None:
        grades = self.tree.notebook["rewarded_notebook"]["grades"]
        self.assertEqual([row["audit"]["reward"] if row["audit"] else None for row in grades], [.4, -.4, 0., None])
        self.assertEqual(grades[0]["audit"]["entries"][0]["reason"], "MISREAD_AND_CORRECT_VALUE")
        self.assertEqual(grades[1]["audit"]["entries"][0]["reason"], "FACT_WAS_CORRECT")
        self.assertEqual(grades[3]["syntax_failure"], {"code": "INVALID_JSON"})
        self.assertEqual(notes.regrade(self.tree, self.f.actor, self.f.verifier, self.engine.tokenizer)["grades"], grades)
        changed = copy.deepcopy(self.tree)
        record = changed.notebook["rewarded_notebook"]
        record["grades"][1]["audit"]["reward"] = .4
        record["grades_sha256"] = _hash(record["grades"])
        changed.notebook["model_notes"] = copy.deepcopy(record["grades"])
        with self.assertRaisesRegex(ValueError, "fresh final semantics"):
            self.f.prepare(changed, self.engine)

    def test_credit_is_centered_on_note_tokens_and_lesson_is_unrewarded(self) -> None:
        receipt = self.f.prepare(self.tree, self.engine)
        self.assertTrue(receipt["admitted"])
        self.assertTrue(receipt["signal_components"]["final"])
        self.assertEqual(receipt["source_bindings"]["final_note_contract"], compact_note.VERSION)
        rows = receipt["groups"][-1]["samples"]
        units = [row["component_credit"]["unit_rows"][0] for row in rows]
        self.assertEqual([unit["advantage"] for unit in units], [.4, -.4, 0., 0.])
        self.assertEqual([unit["positive_verified_note"] for unit in units], [True, False, False, False])
        self.assertTrue(all(value >= 0 for value in rows[0]["final_coefficients"]))
        self.assertAlmostEqual(sum(rows[0]["final_coefficients"]), 8192 * .0625 * .4)
        self.assertAlmostEqual(sum(rows[1]["final_coefficients"]), -8192 * .0625 * .4)
        self.assertFalse(any(rows[2]["final_coefficients"]) or any(rows[3]["final_coefficients"]))
        self.assertEqual(units[3]["token_indices"], list(range(len(self.notes[3]))))
        start = self.notes[0].index("FREE_LESSON")
        self.assertFalse(any(rows[0]["final_coefficients"][start:start + len("FREE_LESSON")]))
        self.assertFalse(any(row["answer_coefficients"] != [0.] * len(row["answer_coefficients"]) for row in rows))

    def test_real_tiny_update_s_definition_and_evidence(self) -> None:
        result, _, _ = self.u.step(self.tree, self.engine, require_retained_learning_signal=True)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        aggregate = result["repair_aggregate_gradient"]
        self.assertEqual(aggregate["s_definition"], "verified-note-claim-v1")
        finals = [row for row in aggregate["members"] if row["component"] == "final"]
        self.assertEqual([row["positive_verified_tokens"] > 0 for row in finals], [True, False, False, False])
        self.assertGreater(sum(aggregate["channels"]["final"]["positive"].values()), 0)
        receipts = {self.tree.problem_id: self.f.prepare(self.tree, self.engine)}
        inventory = [{key: value for key, value in row.items() if key != "index"}
                     for row in aggregate["parameter_inventory"]]

        def verify(update, protocol):
            return repair_evidence.repair_update_evidence(update, protocol, trees=[self.tree],
                preparations=receipts, parameter_inventory=inventory, require_s=True)

        self.assertTrue(verify(result, self.u.protocol)["S_gradient_verified"])
        tampered = copy.deepcopy(result)
        tampered["repair_aggregate_gradient"]["s_definition"] = "native-corrected-v1"
        with self.assertRaisesRegex(ValueError, "aggregate required"):
            verify(tampered, self.u.protocol)
        default = local_profile.build_protocol(repair_notebook=True, context_limit=24576)
        with self.assertRaisesRegex(ValueError, "credit version differs"):
            verify(tampered | {"repair_execution": result["repair_execution"] | {"protocol_sha256": default["sha256"]}},
                   default)


class CompactRecoveryProtocolTests(unittest.TestCase):
    def test_compact_evaluator_protocol_and_metrics(self) -> None:
        from fine_tuning.notebook_rl import local_ladder_gate
        from fine_tuning.notebook_rl.local_ladder_generation import (CIRCUIT_REPAIR_PROMPT_POLICY, _digest,
                                                                     circuit_evaluation_prompts)

        templates = circuit_evaluation_prompts(CIRCUIT_REPAIR_PROMPT_POLICY)
        base = recovery.protocol(templates, "a" * 64, context_limit=24576)
        compact = recovery.protocol(templates, "a" * 64, context_limit=24576, note_contract=compact_note.VERSION)
        self.assertEqual(recovery.protocol(templates, "a" * 64, context_limit=24576, note_contract=FINAL_VERSION), base)
        self.assertEqual({key for key in base if base[key] != compact[key]}, {"notebook"})
        self.assertEqual((recovery._note_contract(base), recovery._note_contract(compact)),
                         (FINAL_VERSION, compact_note.VERSION))
        self.assertEqual(compact["notebook"]["prompt_sha256"], _digest(compact_note.PROMPT))
        local_ladder_gate._validate_protocol(compact)
        with self.assertRaises(ValueError):
            local_ladder_gate._validate_protocol(compact | {"notebook": compact["notebook"] | {"grader": "x"}})
        with self.assertRaises(ValueError):
            recovery.protocol(circuit_evaluation_prompts(), "a" * 64, context_limit=24576,
                              note_contract=compact_note.VERSION)
        summary = recovery._compact_note_metrics([
            {"problem_id": "p1", "note": {"checked": {"syntax_failure": None, "audit": {
                "reward": .4, "verified": 1, "refuted": 0,
                "vision": {"claim": "YES", "route": "VISION_ON", "score": 1}}}}},
            {"problem_id": "p2", "note": {"checked": {"syntax_failure": {"code": "INVALID_JSON"}, "audit": None}}},
            {"problem_id": "p3", "note": None}])
        self.assertEqual((summary["note_count"], summary["parsed_note_count"], summary["verified_entry_count"],
                          summary["refuted_entry_count"], summary["vision_agreement_count"], summary["mean_reward"]),
                         (2, 1, 1, 0, 1, .2))
        self.assertFalse(summary["lesson_rewarded"])


class CompactGuardTests(unittest.TestCase):
    def test_cache_sources_and_dispatch_versions(self) -> None:
        from fine_tuning.notebook_rl import quantized_cache

        self.assertIn("notebook_rl/compact_note.py", quantized_cache._sources(repair=True))
        self.assertNotIn("notebook_rl/compact_note.py", quantized_cache._sources())
        self.assertEqual(notes.VERSIONS, (notes.VERSION, notes.COMPACT_VERSION))


if __name__ == "__main__":
    unittest.main()


class CompactRecoveryRunTests(unittest.TestCase):
    """Scripted n=1 evaluator run; the NOTE reply is authored, not generated by a model."""

    @classmethod
    def setUpClass(cls):
        from fine_tuning.tests import test_notebook_rl_repair_ladder as ladder
        ladder.RepairLadderTests.setUpClass.__func__(cls)

    def setUp(self):
        from fine_tuning.tests import test_notebook_rl_repair_ladder as ladder
        ladder.RepairLadderTests.setUp(self)

    def test_compact_note_diagnose_validate_and_replay(self) -> None:
        from fine_tuning.notebook_rl.engines import Completion, RolloutResult
        from fine_tuning.notebook_rl.local_ladder_gate import _validate_protocol
        from fine_tuning.notebook_rl.local_ladder_generation import _digest
        from fine_tuning.tests import test_notebook_rl_repair_ladder as ladder

        bodies: list[str] = []

        class Engine(ladder.RepairEngine):
            def sample(self, requests):
                request = requests[0]
                if not request.request_id.endswith(":NOTE"):
                    return super().sample(requests)
                self.calls.append(request)
                bodies.append(request.messages[0]["content"][-1]["text"])
                text = json.dumps({"errors": [{"attempt": 1, "type": "WRONG_STEP"}], "vision_misread": "UNSURE",
                                   "lesson": "untrusted"})
                tokens = self.tokenizer.encode(text)
                return [RolloutResult(request.request_id, self.prepare(request).token_ids,
                                      [Completion(tokens, [-.5] * len(tokens), "stop", text)])]

        pid = self.rows[0].actor.problem_id
        result = recovery.diagnose(self.cohort, self.root / "compact", Engine(self.rows, script={(pid, "T1"): "native_issue"}),
            self.templates, expected_preparation_sha256=self.sha, local_protocol_sha256="a" * 64,
            deadline=time.monotonic() + 180, context_limit=24576, note_contract=compact_note.VERSION)
        self.assertEqual(result["status"], "COMPLETE_NATIVE_LADDER_RECOVERY", result.get("reason"))
        self.assertEqual(result["protocol"]["notebook"]["response_contract"], compact_note.VERSION)
        self.assertTrue(bodies and all(body.endswith(compact_note.PROMPT) and "<read_only_history>" not in body
                                       for body in bodies))
        metrics = result["repair_metrics"]
        self.assertIsNone(metrics["final_claims"])
        compact = metrics["compact_notes"]
        self.assertEqual(compact["note_count"], len(bodies))
        self.assertEqual(compact["verified_entry_count"], 1)
        self.assertEqual(metrics["final_false_accusation_count"], compact["refuted_entry_count"])
        _validate_protocol(result["protocol"])
        replayed = recovery.replay(self.cohort, result, expected_preparation_sha256=self.sha)
        self.assertEqual(replayed["repair_metrics"], metrics)
        changed = copy.deepcopy(result)
        note_row = next(row for row in changed["calls"] if row["stage"] == "NOTE")
        note_row["response_contract"] = FINAL_VERSION
        note_row["record_sha256"] = _digest({key: value for key, value in note_row.items() if key != "record_sha256"})
        changed["record_sha256"] = _digest({key: value for key, value in changed.items() if key != "record_sha256"})
        with self.assertRaises(ValueError):
            recovery.replay(self.cohort, changed, expected_preparation_sha256=self.sha)


class CompactLocalEvalTests(unittest.TestCase):
    """The separate evaluator restores the trained final-note contract from the saved protocol."""

    def fixture(self, contract: str):
        from unittest.mock import patch

        from fine_tuning.notebook_rl import local_eval as runner
        from fine_tuning.tests import test_notebook_rl_repair_local_eval as repair_eval

        case = repair_eval.RepairLocalEvalTests()
        real = local_profile.build_protocol

        def build(**options):
            if options.get("repair_notebook"):
                options.setdefault("note_contract", contract)
            return real(**options)

        with patch.object(local_profile, "build_protocol", side_effect=build):
            case.setUp()
        self.addCleanup(case.doCleanups)
        case.protocol = recovery.protocol(case.templates, case.fixture.protocol["sha256"], context_limit=24576,
            **({"note_contract": contract} if contract != FINAL_VERSION else {}))
        case.fixture.gate["protocol_sha256"] = runner.digest(case.protocol)
        case.record.update(protocol=case.protocol, protocol_sha256=runner.digest(case.protocol))
        return case

    def test_evaluator_receives_the_trained_note_contract(self) -> None:
        for contract in (FINAL_VERSION, compact_note.VERSION):
            with self.subTest(contract=contract):
                case = self.fixture(contract)
                case.evaluate("evaluation-" + contract)
                options = case.diagnose.call_args.kwargs
                if contract == FINAL_VERSION:
                    self.assertNotIn("note_contract", options)
                else:
                    self.assertEqual(options["note_contract"], compact_note.VERSION)


class CompactCliTests(unittest.TestCase):
    """--note-contract only changes build_protocol's argument; default calls are unchanged."""

    def test_local_run_prepare_and_minimal_thread_the_selection(self) -> None:
        import contextlib
        import io
        from pathlib import Path
        from unittest.mock import patch

        from fine_tuning.notebook_rl import local_run

        repair = ["--repair-notebook", "--context-limit", "24576"]
        prepare = ["prepare", "--previous-request", "/r", "--selection", "/s", "--output", "/o"]
        minimal = ["minimal", "--preparation", "/p", "--model-path", "/m", "--output", "/o"]
        for extra, expected in (([], FINAL_VERSION), (["--note-contract", compact_note.VERSION], compact_note.VERSION)):
            with self.subTest(extra=extra), patch("fine_tuning.notebook_rl.local_data.prepare_local_data",
                    return_value={"status": "LOCAL_DATA_REVALIDATED"}) as called, patch("builtins.print"):
                self.assertEqual(local_run.main(prepare + repair + extra), 0)
                self.assertEqual(notes.note_contract(called.call_args.args[2]), expected)
            with self.subTest(extra=extra), patch.object(local_run, "capture") as called:
                self.assertEqual(local_run.main(minimal + repair + extra), 0)
                self.assertEqual(called.call_args.kwargs.get("note_contract", FINAL_VERSION), expected)
                self.assertEqual("note_contract" in called.call_args.kwargs, bool(extra))
            with self.subTest(extra=extra), patch.object(local_run, "read_preparation",
                    side_effect=ValueError("CPU boundary")) as prepared:
                with self.assertRaisesRegex(ValueError, "CPU boundary"):
                    local_run.capture(Path("/p"), Path("/m"), Path("/o"), standalone=False, repair_notebook=True,
                                      context_limit=24576, note_contract=expected)
                self.assertEqual(notes.note_contract(prepared.call_args.args[1]), expected)
        for argv in (prepare + ["--note-contract", compact_note.VERSION],
                     minimal + ["--note-contract", compact_note.VERSION]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                local_run.main(argv)

    def test_local_train_and_cache_cli_require_explicit_matching_selection(self) -> None:
        import contextlib
        import io
        from unittest.mock import patch

        from fine_tuning.notebook_rl import local_train, quantized_cache
        from fine_tuning.tests.test_notebook_rl_local_train import IntermediateCheckpointTests

        argv = ["train", "--preparation", "/p", "--model-path", "/m", "--cache-check", "/c", "--output", "/o",
                "--arm", "notebook", "--minimal", "/min", "--minimal-reload", "/rel"]
        for extra in ([], ["--note-contract", compact_note.VERSION]):
            with self.subTest(extra=extra), patch.object(local_train, "train",
                    return_value={"status": "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD"}) as called, \
                    patch("builtins.print"):
                self.assertEqual(local_train.main(argv + extra), 0)
                self.assertEqual(called.call_args.kwargs.get("note_contract", FINAL_VERSION),
                                 extra[-1] if extra else FINAL_VERSION)
        fixture = IntermediateCheckpointTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with self.assertRaisesRegex(ValueError, "--note-contract must match"):
            fixture.run_training(note_contract=compact_note.VERSION)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            quantized_cache.main(["save", "--cache", "/c", "--output", "/o", "--note-contract", compact_note.VERSION])
