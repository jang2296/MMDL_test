"""Opt-in answer-verified (slotless) route and finalization reward.

Authored CPU fixtures with scripted engines and tiny parameter vectors; not Qwen behaviour.
"""
from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace

import torch
from PIL import Image

from fine_tuning.notebook_rl import (compact_note, episode_notebook as episode, local_profile, prompts as builders,
                                     repair_evidence, repair_training)
from fine_tuning.notebook_rl.checkpoint import make_optimizer
from fine_tuning.notebook_rl.local_run import verify_changes
from fine_tuning.notebook_rl.losses import repair_component_losses
from fine_tuning.notebook_rl.notebook_synthesis import audit_native_sample
from fine_tuning.notebook_rl.reasoning_trace import validate_trace_report
from fine_tuning.notebook_rl.repair_contract import ORIGIN_VERSION, VERSION as RETRY_VERSION, public_origin_inventory
from fine_tuning.notebook_rl.schema import ActorItem, Sample, VerifierItem
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.tests import test_notebook_rl_repair_update as update_fixture
from fine_tuning.tests.test_notebook_rl_compact_integration import DEFAULT_REPAIR_PROTOCOL_SHA256

COMPACT_REPAIR_PROTOCOL_SHA256 = "e87c4ab05cce479c173b0c94eb287d49f2050bad99320d955794425b09aeb730"
SOURCE = "test-slotless-mcq-v1"
ACTOR = ActorItem(f"{SOURCE}:mcq:0001", SOURCE, "chart_bar", "obs", "How many bars are taller than 5?",
                  ("2", "3", "4", "5"), "mcq", ("img/0.png",), ("a" * 64,))
ITEM = VerifierItem(ACTOR.problem_id, "mcq", "B", {}, ACTOR.choices, {}, {}, (), (), ())
RIGHT = "Solution:\nThree bars are taller than 5.\nFinal answer: B"
WRONG = "Solution:\nFour bars are taller than 5.\nFinal answer: C"
NO_FINAL = "Solution:\nI cannot decide from the image."


def retry(body: str) -> str:
    return 'ErrorCheck: {"version":"repair-check-v1","claims":[],"uncertain_refs":[]}\nRevisedSolution:\n' + body


def note(errors: list, vision: str = "UNSURE") -> str:
    return json.dumps({"errors": errors, "vision_misread": vision, "lesson": "FREE_LESSON"})


def flagged(**changes) -> dict:
    return local_profile.build_protocol(**({"repair_notebook": True, "context_limit": 24576,
        "note_contract": compact_note.VERSION, "answer_verified_route": True, "finalization_reward": True,
        "slotless_sources": (SOURCE,)} | changes))


def compact() -> dict:
    return local_profile.build_protocol(repair_notebook=True, context_limit=24576, note_contract=compact_note.VERSION)


def slotless_verifier(actor: ActorItem = ACTOR, item: VerifierItem = ITEM) -> Verifier:
    verifier = Verifier.__new__(Verifier)
    verifier.actors, verifier.items = {actor.problem_id: actor}, {actor.problem_id: item}
    return verifier


class Pipeline:
    """The shared tiny-update fixture, rebound to an explicit protocol and optionally the slotless item."""

    def __init__(self, case: unittest.TestCase, protocol: dict, *, slotless: bool,
                 actor: ActorItem = ACTOR, item: VerifierItem = ITEM) -> None:
        self.u = update_fixture.RepairUpdateTests()
        self.u.setUp()
        case.addCleanup(self.u.doCleanups)
        self.u.protocol = protocol
        self.u.optimizer, self.u.scheduler = make_optimizer(self.u.bundle, protocol)
        self.f = self.u.f
        fixture = self.f.fixture
        if slotless:
            images = [Image.new("RGB", (16, 16), "white")]
            case.addCleanup(images[0].close)
            self.f.actor = fixture.actor = actor
            self.f.verifier = fixture.verifier = slotless_verifier(actor, item)
            self.f.images, fixture.images = images, {actor.problem_id: images}
        fixture.prompts = local_profile.local_prompts(protocol)

    def hashes(self) -> dict:
        return {group: {group: self.u.model[group].detach().clone()} for group in ("L", "M", "V")}


def changed(before: dict, model) -> tuple[dict, dict]:
    hashes = {group: {group: str(values[group].tolist())} for group, values in before.items()}
    after = {group: {group: str(model[group].detach().tolist())} for group in before}
    return hashes, after


class ProfileTests(unittest.TestCase):
    def test_default_and_compact_hashes_are_unchanged_and_flags_round_trip(self) -> None:
        self.assertEqual(local_profile.build_protocol(repair_notebook=True, context_limit=24576)["sha256"],
                         DEFAULT_REPAIR_PROTOCOL_SHA256)
        self.assertEqual(compact()["sha256"], COMPACT_REPAIR_PROTOCOL_SHA256)
        self.assertNotIn("finalization", local_profile.repair_objective_contract(compact_note.VERSION)["component_weights"])
        protocols = [flagged(), flagged(finalization_reward=False), flagged(answer_verified_route=False, slotless_sources=())]
        digests = {protocol["sha256"] for protocol in protocols[:3]}
        self.assertEqual(len(digests | {DEFAULT_REPAIR_PROTOCOL_SHA256, COMPACT_REPAIR_PROTOCOL_SHA256}), 5)
        generators = ("gen-statics-v1", "gen-music-v1", "gen-chart-v1", "gen-thermo-v1", "gen-materials-v1")
        protocols.append(flagged(slotless_sources=(SOURCE, *generators, "hf-scienceqa-v1")))
        for protocol in protocols:
            self.assertEqual(local_profile.restore_protocol(json.loads(json.dumps(protocol))), protocol)
        protocol = protocols[0]
        self.assertEqual(protocol["repair_slotless_sources"], [SOURCE])
        self.assertEqual(protocol["repair_slotless_prompt_sha256"],
                         local_profile.digest(builders.slotless_solution_prompt_contract()))
        self.assertEqual(protocol["repair_objective"]["component_weights"]["finalization"], .25)
        self.assertEqual(protocol["repair_objective"]["text_s_definition"], "verified-note-text-claim-v1")
        prompts = local_profile.local_prompts(protocol)
        self.assertEqual(prompts["repair_route_policy"], episode.ANSWER_ROUTE_RULE)
        self.assertEqual(prompts["repair_slotless_observation_instruction"], builders.SLOTLESS_REREAD_INSTRUCTION)

    def test_tampered_or_invalid_flags_fail_closed(self) -> None:
        for mutate in (lambda p: p.update(repair_route_policy="junk"),
                       lambda p: p.pop("repair_slotless_sources"),
                       lambda p: p.update(repair_slotless_sources=[SOURCE, "other-v1"]),
                       lambda p: p.update(repair_finalization_weight=.5),
                       lambda p: p.update(repair_finalization_contract="junk"),
                       lambda p: p.pop("repair_finalization_contract"),
                       lambda p: p["repair_objective"]["component_weights"].update(finalization=1.0)):
            changed_protocol = copy.deepcopy(flagged())
            mutate(changed_protocol)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                local_profile.restore_protocol(json.loads(json.dumps(changed_protocol)))
        for options in ({"note_contract": "repair-note-v2"}, {"slotless_sources": ()},
                        {"slotless_sources": ("gen-graph-v1",)}, {"slotless_sources": (SOURCE, SOURCE)},
                        {"answer_verified_route": False}, {"repair_notebook": False, "context_limit": 16384}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                flagged(**options)


class SlotlessTextTests(unittest.TestCase):
    """r2 risk: public origins and the trace report on slotless solution text."""

    def test_public_origin_inventory_and_trace_report_accept_slotless_text(self) -> None:
        verifier = slotless_verifier()
        parent = Sample("p:T1:0", ACTOR.problem_id, "p:T1", "T1", 1, None, [1], [2], [-.1], "stop", WRONG, 0, 0)
        child = Sample("p:T2:0", ACTOR.problem_id, "p:T2", "T2", 2, parent.sample_id, [3], [4], [-.1], "stop",
                       retry(RIGHT), 0, 0)
        inventory = public_origin_inventory([parent, child], contract_version=ORIGIN_VERSION,
                                            response_contracts={parent.sample_id: None, child.sample_id: RETRY_VERSION})
        self.assertTrue(inventory.origins)
        self.assertEqual({origin.kind for origin in inventory.origins} - {"TRACE", "OUTPUT"}, set())
        for sample, contract in ((parent, None), (child, RETRY_VERSION)):
            options = {} if contract is None else {"response_contract": contract}
            report = verifier.trace(ACTOR.problem_id, sample.text, "stop", **options)
            self.assertEqual([(row["rule"], row["status"]) for row in report["checks"]],
                             [("NATIVE_CONTRACT", "UNVERIFIABLE")])
            validate_trace_report(report, sample.text, "stop", **options)
            audit = audit_native_sample(sample, ACTOR, verifier, response_contract=contract)
            self.assertIsNone(audit["result"]["slots"])
            self.assertFalse(audit["issues"])

    def test_slotless_prompts_have_no_trace_or_slot_schema_and_slotted_sources_are_unchanged(self) -> None:
        prompts = local_profile.local_prompts(flagged())
        text = builders.turn1_text(ACTOR, prompts)
        self.assertEqual(text, builders._render(builders.SLOTLESS_TURN1, P0_BODY=builders.p0_body(ACTOR.question, ACTOR.choices),
                                                FINAL_LINE=builders.SLOTLESS_FINAL_LINES["mcq"]))
        self.assertNotIn("Trace", text)
        selected = builders.for_actor(ACTOR, prompts)
        self.assertEqual(selected["observation_instruction"][ACTOR.family], builders.SLOTLESS_REREAD_INSTRUCTION)
        retry_text = builders.retry_text("INCORRECT", "C", "obs", ACTOR.family, selected)
        self.assertTrue(retry_text.startswith('Verifier result for your previous answer "C": INCORRECT.'))
        self.assertNotIn("Trace", retry_text)
        bad = prompts | {"repair_slotless_observation_instruction": ""}
        with self.assertRaises(ValueError):
            builders.for_actor(ACTOR, bad)
        # Without the flag the slotless actor still has no native contract (unchanged hard block).
        with self.assertRaises(ValueError):
            builders.for_actor(ACTOR, local_profile.local_prompts(compact()))


class SlotlessAnswerFormatTests(unittest.TestCase):
    def test_each_answer_format_gets_its_bare_final_line_in_every_slotless_prompt(self) -> None:
        number = "Final answer: <a number only, with no units, percent sign or other words>"
        expected = {"mcq": "Final answer: <the letter of the correct option>", "int": number, "decimal": number,
                    "fraction": "Final answer: <the exact value as a fraction or whole number, with no units or words>",
                    "unit_number": "Final answer: <answer>",
                    "short_text": "Final answer: <a short word or phrase only>",
                    "list": "Final answer: <the items separated by commas>"}
        self.assertEqual(builders.SLOTLESS_FINAL_LINES, expected)
        self.assertEqual(builders.slotless_solution_prompt_contract()["final_lines"], expected)
        prompts = local_profile.local_prompts(flagged())
        for answer_format, line in expected.items():
            for mode in ("obs", "plain"):
                actor = replace(ACTOR, mode=mode, answer_format=answer_format,
                                choices=ACTOR.choices if answer_format == "mcq" else ())
                with self.subTest(answer_format=answer_format, mode=mode):
                    turn1 = builders.turn1_text(actor, prompts)
                    self.assertTrue(turn1.endswith("end your response with exactly one line: " + line + "\n"))
                    selected = builders.for_actor(actor, prompts)
                    self.assertTrue(builders.retry_text("TRUNCATED", None, mode, actor.family, selected)
                                    .endswith("exactly one line: " + line + "\n"))
                    parent = Sample("p:T1:0", actor.problem_id, "p:T1", "T1", 1, None, [1], [2], [-.1], "stop",
                                    NO_FINAL, 0, 0)
                    inventory = public_origin_inventory([parent], contract_version=ORIGIN_VERSION,
                        response_contracts={parent.sample_id: None}, parent_sample_id=parent.sample_id)
                    repair = builders.repair_retry_text("INVALID_NO_FINAL", None, actor, prompts, inventory)
                    self.assertIn("Within RevisedSolution write your reasoning under Solution:, then exactly one line: "
                                  + line + "\n", repair)
                    others = set(expected.values()) - {line}
                    self.assertFalse(any(other in text for other in others for text in (turn1, repair)))


class PlainHiddenSlotTests(unittest.TestCase):
    def test_plain_generator_actor_with_private_slots_routes_answer_verified(self) -> None:
        source = "gen-statics-v1"
        actor = ActorItem(f"{source}:statics:0001", source, "statics", "plain", "What is the reaction at A in N?",
                          (), "int", ("img/0.png",), ("b" * 64,))
        item = VerifierItem(actor.problem_id, "int", "12", {}, (), {"span": "4"}, {"span": "number"}, ("span",), (), ())
        p = Pipeline(self, flagged(slotless_sources=(source,)), slotless=True, actor=actor, item=item)
        right, wrong = "Solution:\nspan = 4\nFinal answer: 12", "Solution:\nspan = 5\nFinal answer: 13"
        tree, engine = p.f.collect([[right, wrong, right, right], [retry(right)] * 2])
        p.f.attach(tree, engine, [note([])] * 3 + ["{"])
        results = [sample.result for sample in episode.solution_samples(tree) if sample.result is not None]
        self.assertTrue(results and all(result.slots is None for result in results))
        self.assertIn(builders.SLOTLESS_FINAL_LINES["int"], engine.requests[0].messages[0]["content"][-1]["text"])
        receipt = p.f.prepare(tree, engine)
        self.assertEqual((receipt["admission"], receipt["admitted"]), ("ANSWER_VERIFIED", True))
        self.assertEqual(tree.notebook["rewarded_notebook"]["status"], "GENERATED")


class FinalizationUnitTests(unittest.TestCase):
    def sample(self, text: str, finish: str = "stop", contract: str | None = None) -> Sample:
        kind, turn = ("T1", 1) if contract is None else ("T2", 2)
        row = Sample("s", ACTOR.problem_id, "g", kind, turn, None if turn == 1 else "p", [1], [2], [-.1],
                     finish, text, 0, 0)
        try:
            row.result = slotless_verifier().grade(ACTOR.problem_id, text, finish, "obs",
                                                   **({"response_contract": contract} if contract else {}))
        except ValueError:
            row.result = None
        return row

    def test_reward_table_centering_and_equal_rewards(self) -> None:
        table = {"CORRECT": (RIGHT, "stop", None), "INCORRECT": (WRONG, "stop", None),
                 "TRUNCATED": (RIGHT, "length", None), "INVALID_NO_FINAL": (NO_FINAL, "stop", None),
                 "INVALID_MULTIPLE": (RIGHT + "\nFinal answer: C", "stop", None),
                 "INVALID_RESPONSE_CONTRACT": ("ErrorCheck: {", "stop", RETRY_VERSION),
                 "RETRY_CORRECT": (retry(RIGHT), "stop", RETRY_VERSION)}
        rewards = {}
        for name, (text, finish, contract) in table.items():
            row = self.sample(text, finish, contract)
            if name != "INVALID_RESPONSE_CONTRACT" and not name.startswith("RETRY"):
                self.assertEqual(row.result.status, name)
            rewards[name] = repair_training.finalization_reward(row, contract)
        self.assertEqual(rewards, {"CORRECT": 1., "INCORRECT": 1., "TRUNCATED": 0., "INVALID_NO_FINAL": 0.,
                                   "INVALID_MULTIPLE": 0., "INVALID_RESPONSE_CONTRACT": 0., "RETRY_CORRECT": 1.})
        vector = repair_training.finalization_vector
        self.assertEqual(vector([1., 1., 0., 1.], 0, 3, weight=.25, n=4), [.25 * .25] * 3)
        self.assertEqual(vector([1., 1., 0., 1.], 2, 2, weight=.25, n=4), [.25 * -.75] * 2)
        self.assertEqual(vector([1., 1.], 1, 4, weight=.25, n=2), [0.] * 4)
        self.assertEqual(vector([0.] * 4, 3, 1, weight=.25, n=4), [0.])
        for args, options in (([[1., 0.], 0, 1], {"weight": .25, "n": 4}), ([[1., 0.], 0, 1], {"weight": .5, "n": 2}),
                              ([[1., .5], 0, 1], {"weight": .25, "n": 2})):
            with self.subTest(args=args, options=options), self.assertRaises(ValueError):
                vector(*args, **options)

    def test_loss_channel_is_optional_kl_once_and_forbidden_on_diag(self) -> None:
        current = torch.tensor([[-.3, -.2]], dtype=torch.float64, requires_grad=True)
        old = current.detach().clone()
        reference = old - .1
        mask, keep = torch.tensor([[True, True]]), torch.tensor([True])
        zero, one = torch.zeros(1, 2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.float64)

        def run(kind, size, cap, **extra):
            return repair_component_losses(current, old, reference, mask, keep, answer=zero, online=zero, final=zero,
                                           kind=kind, original_group_size=size, completion_cap=cap, **extra)

        base = run("T1", 4, 1024)
        self.assertNotIn("finalization", base)
        with_channel = run("T1", 4, 1024, finalization=one)
        self.assertEqual(set(with_channel), set(base) | {"finalization"})
        torch.testing.assert_close(with_channel["kl"], base["kl"])
        torch.testing.assert_close(with_channel["total"], base["total"] + with_channel["finalization"])
        self.assertNotEqual(float(with_channel["finalization"].detach()), 0.)
        self.assertIn("finalization", run("DIAG", 4, 8192, finalization=zero))
        with self.assertRaisesRegex(ValueError, "inapplicable channel"):
            run("DIAG", 4, 8192, finalization=one)


class AnswerVerifiedPipelineTests(unittest.TestCase):
    """T1 -> retry -> compact note on a slotless item, real parsers and a real tiny optimizer step."""

    def setUp(self) -> None:
        self.p = Pipeline(self, flagged(), slotless=True)
        self.notes = [note([{"attempt": 3, "type": "NO_FINAL_ANSWER"}]),
                      note([{"attempt": 1, "type": "NO_FINAL_ANSWER"}]),
                      note([{"attempt": 1, "type": "OBSERVATION", "item": "bar[x]", "correct": "3"}], "YES"), "{"]
        self.tree, self.engine = self.p.f.collect([[RIGHT, WRONG, NO_FINAL, RIGHT], [retry(RIGHT)] * 2])
        self.p.f.attach(self.tree, self.engine, list(self.notes))

    def test_generation_routes_answer_verified_without_vision_off_claim(self) -> None:
        record = self.tree.notebook["rewarded_notebook"]
        routing = record["source_audit"]["routing"]
        self.assertEqual(record["status"], "GENERATED")
        self.assertEqual((routing["admission"], routing["route"], routing["routing_rule"]),
                         ("ANSWER_VERIFIED", "ANSWER_VERIFIED", episode.ANSWER_ROUTE_RULE))
        self.assertEqual((routing["strict_admission"], routing["module_causality_certified"]), ("UNRESOLVED", False))
        self.assertEqual(self.tree.route, "VISION_OFF")  # schema wire value only; Vision frozen
        self.assertEqual([len(group.samples) for group in self.tree.groups], [4, 2, 4])
        first = self.engine.requests[0].messages[0]["content"][-1]["text"]
        self.assertEqual(first, builders.turn1_text(ACTOR, self.p.f.fixture.prompts))
        call = self.tree.notebook["repair_collector"]["calls"][1]
        self.assertIn("RevisedSolution:", call["feedback"])
        self.assertNotIn("Trace", call["feedback"])
        self.assertEqual(self.tree.notebook["repair_collector"]["fresh_observation"]["schema"],
                         builders.SLOTLESS_REREAD_INSTRUCTION)
        grades = record["grades"]
        self.assertEqual([row["audit"]["reward"] if row["audit"] else None for row in grades], [.2, -.2, 0., None])
        self.assertEqual(grades[2]["audit"]["entries"][0]["reason"], "ROUTE_TEXT_ONLY")
        self.assertEqual(grades[2]["audit"]["vision"]["reason"], "ROUTE_TEXT_ONLY")

    def test_credit_finalization_and_online_zeroing(self) -> None:
        receipt = self.p.f.prepare(self.tree, self.engine)
        self.assertEqual((receipt["route"], receipt["admission"], receipt["admitted"]),
                         ("ANSWER_VERIFIED", "ANSWER_VERIFIED", True))
        self.assertEqual(receipt["signal_components"], {"answer": True, "online": False, "final": True,
                                                        "finalization": True})
        t1, t2, notes_group = receipt["groups"]
        self.assertEqual([row["finalization_reward"] for row in t1["samples"]], [1., 1., 0., 1.])
        self.assertEqual([row["applied_finalization_coefficients"][0] for row in t1["samples"]],
                         [.25 * .25, .25 * .25, .25 * -.75, .25 * .25])
        self.assertTrue(all(value == 0 for row in t2["samples"] for value in row["applied_finalization_coefficients"]))
        self.assertTrue(all(value == 0 for row in notes_group["samples"] for value in row["applied_finalization_coefficients"]))
        self.assertTrue(all(value == 0 for row in t2["samples"] for value in row["applied_online_coefficients"]))
        units = [row["component_credit"]["unit_rows"][0] for row in notes_group["samples"]]
        self.assertEqual([unit["positive_verified_note"] for unit in units], [True, False, False, False])
        self.assertEqual(notes_group["samples"][0]["component_credit"]["source_bindings"]["route_admission"],
                         "ANSWER_VERIFIED")

    def test_real_tiny_update_freezes_vision_labels_route_and_reports_s_text(self) -> None:
        before = self.p.hashes()
        result, _, _ = self.p.u.step(self.tree, self.engine, require_retained_learning_signal=True)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        pid = ACTOR.problem_id
        self.assertEqual((result["routes"], result["route_labels"]), ({pid: False}, {pid: "ANSWER_VERIFIED"}))
        model = self.p.u.model
        torch.testing.assert_close(model["V"], before["V"]["V"], atol=0, rtol=0)
        self.assertFalse(torch.equal(model["M"], before["M"]["M"]))
        self.assertFalse(torch.equal(model["L"], before["L"]["L"]))
        changes = verify_changes(*changed(before, model), result)
        self.assertTrue(changes["answer_verified_observed"])
        self.assertFalse(changes["vision_off_only_observed"] or changes["vision_on_observed"])
        signal = result["retained_repair_signal"]
        self.assertTrue(signal["channel_signal"]["finalization"] and signal["channel_signal"]["answer"])
        self.assertFalse(signal["finalization_only_signal"])
        aggregate = result["repair_aggregate_gradient"]
        self.assertFalse(aggregate["S"])
        self.assertEqual(aggregate["retained_positive_verified_tokens"], 0)
        self.assertTrue(aggregate["S_text"]["S"])
        self.assertEqual(aggregate["S_text"]["group_norms"]["V"], 0.)
        receipts = {pid: self.p.f.prepare(self.tree, self.engine)}
        inventory = [{key: value for key, value in row.items() if key != "index"}
                     for row in aggregate["parameter_inventory"]]

        def verify(update, **options):
            return repair_evidence.repair_update_evidence(update, self.p.u.protocol, trees=[self.tree],
                preparations=receipts, parameter_inventory=inventory, **options)

        evidence = verify(result)
        self.assertEqual((evidence["S_gradient_verified"], evidence["S_text_gradient_verified"]), (False, True))
        with self.assertRaisesRegex(ValueError, "positive aggregate S"):
            verify(result, require_s=True)
        for mutate in (lambda u: u.pop("route_labels"), lambda u: u["route_labels"].update({pid: "VISION_OFF"}),
                       lambda u: u["repair_aggregate_gradient"]["S_text"].update(S=False),
                       lambda u: u["repair_aggregate_gradient"].update(S=True)):
            tampered = copy.deepcopy(result)
            mutate(tampered)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                verify(tampered)

    def test_finalization_only_signal_is_labelled(self) -> None:
        p = Pipeline(self, flagged(), slotless=True)
        tree, engine = p.f.collect([[WRONG, WRONG, NO_FINAL, WRONG], [retry(WRONG)] * 2, [retry(WRONG)] * 2])
        p.f.attach(tree, engine)
        self.assertEqual(tree.notebook["rewarded_notebook"]["status"], "NO_CORRECT_ANSWER")
        result, _, _ = p.u.step(tree, engine, require_retained_learning_signal=True)
        self.assertEqual(result["status"], "UPDATED_UNSYNCHRONIZED")
        signal = result["retained_repair_signal"]
        self.assertEqual(signal["channel_signal"], {"answer": False, "online": False, "final": False,
                                                    "finalization": True})
        self.assertTrue(signal["finalization_only_signal"])
        self.assertIn("finalization", signal["basis"])

    def test_unverifiable_retry_group_is_excluded_not_credited(self) -> None:
        """local-test-v2 minimal (2026-10-31): a strict-graded retry answering outside the options is UNVERIFIABLE,
        so its T2 group is unusable and the episode closes INCOMPLETE; preparation must refuse admission
        instead of raising the strict online membership check."""
        from fine_tuning.notebook_rl.answers import TURN1_EVAL_CONTRACT
        from fine_tuning.notebook_rl.repair_credit import repair_group_credit

        p = Pipeline(self, flagged(turn1_contract=TURN1_EVAL_CONTRACT), slotless=True)
        outside = retry("Solution:\nNo listed count fits.\nFinal answer: None of the options are correct.")
        tree, engine = p.f.collect([["So the answer is (C)"] * 4, [retry(WRONG), outside]])
        p.f.attach(tree, engine)
        t2 = tree.groups[1]
        self.assertEqual([group.kind for group in tree.groups], ["T1", "T2"])
        self.assertEqual((t2.usable, t2.reason, [row.result.status for row in t2.samples]),
                         (False, "UNVERIFIABLE", ["INCORRECT", "UNVERIFIABLE"]))
        self.assertEqual(tree.notebook["rewarded_notebook"]["status"], "INCOMPLETE_EPISODE")
        # The strict check itself is unchanged: this exact group still fails it.
        contracts = tree.notebook["rewarded_notebook"]["source_audit"]["response_contracts"]
        inventory = public_origin_inventory(tree.groups[0].samples, contract_version=ORIGIN_VERSION,
            response_contracts={row.sample_id: contracts[row.sample_id] for row in tree.groups[0].samples},
            parent_sample_id=t2.samples[0].parent_sample_id)
        with self.assertRaisesRegex(ValueError, "original group, cap, or audit membership differs"):
            repair_group_credit(t2, {row.sample_id: None for row in t2.samples}, engine.tokenizer,
                                inventory=inventory, completion_cap=1024, component="online")
        # ...and `usable` is the only membership condition it fails (the same group marked usable passes it).
        with self.assertRaisesRegex(ValueError, "parsed candidate requires its actual audit"):
            repair_group_credit(replace(t2, usable=True, reason=None), {row.sample_id: None for row in t2.samples},
                                engine.tokenizer, inventory=inventory, completion_cap=1024, component="online")
        receipt = p.f.prepare(tree, engine)
        self.assertFalse(receipt["admitted"] or receipt["has_signal"])
        self.assertIn("UNUSABLE_SOLUTION_GROUP", receipt["admission_reasons"])
        rows = receipt["groups"][1]["samples"]
        self.assertEqual([row["component_credit"] for row in rows], [None, None])
        self.assertTrue(all(value == 0 for group in receipt["groups"] for row in group["samples"]
                            for key, values in row.items() if key.startswith("applied_") for value in values))


class SlottedUnchangedTests(unittest.TestCase):
    def test_slotted_tree_under_the_route_flag_keeps_strict_routing_and_credit(self) -> None:
        receipts = []
        for protocol in (compact(), flagged(finalization_reward=False)):
            p = Pipeline(self, protocol, slotless=False)
            tree, engine = p.f.online(wrong_answer=False, visual=True)
            value = next(line for line in p.f.fixture.right.splitlines()
                         if line.startswith("edge[A,B]")).split(" = ", 1)[1]
            p.f.attach(tree, engine, [
                note([{"attempt": 1, "type": "OBSERVATION", "item": "edge[A,B]", "correct": value}], "YES"),
                note([{"attempt": 1, "type": "OBSERVATION", "item": "edge[A,G]", "correct": "x"}], "NO"),
                note([]), "{"])
            receipts.append(p.f.prepare(tree, engine))
        base, routed = receipts
        self.assertEqual(base["admission"], "VISION_ON")
        for key in ("route", "admission", "admitted", "signal_components", "groups"):
            self.assertEqual(routed[key], base[key], key)
        self.assertEqual(routed["source_bindings"]["source_audit"]["routing"],
                         base["source_bindings"]["source_audit"]["routing"])
        self.assertNotIn("finalization", base["signal_components"])
        self.assertFalse(any("applied_finalization_coefficients" in row
                             for group in base["groups"] for row in group["samples"]))


if __name__ == "__main__":
    unittest.main()
