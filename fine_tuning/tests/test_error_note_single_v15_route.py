"""SINGLE v15 route CPU checks (wave 2): every training row of an image item takes the FULL route (vision adapter +
Merger + LLM adapter), RL and A1 included; v11-v14 keep their route rule. Tiny random models only; these are not
pretrained Qwen learning evidence."""
import copy
import unittest

from fine_tuning.error_note import single
from fine_tuning.error_note import train_v4 as train
from fine_tuning.error_note.capacity import CapacityHold
from fine_tuning.error_note.contract import ROUTE_FULL, ROUTE_LM
from fine_tuning.error_note.protocol import VERSION_SINGLE_V15
from fine_tuning.error_note.train import _gradient_norms
from fine_tuning.tests import test_error_note_single_v12 as v12
from fine_tuning.tests.test_error_note_single_loss import block
from fine_tuning.tests.test_error_note_single_v14 import as_v14
from fine_tuning.tests.test_error_note_single_v15 import protocol
from fine_tuning.tests.test_error_note_train import encoded, tiny_factory
from fine_tuning.tests.test_error_note_v3_audit import example

SOLUTION = " ".join(f"step{i}" for i in range(38)) + " Final answer: B"
ROUTE_ERROR = "single train: invalid kind, problem or route"


def record(*, correct, misread, image=True):
    """A collected record whose T1, note and re-solve all pass the program row checks."""
    value = example(correct=correct, misread=misread)
    value.update(image_paths=["fixture.png"] if image else [], image_sha256=["a" * 64] if image else [])
    value["t1"]["prompt"] = value["p0_text"]
    value["resolve"] = {"text": SOLUTION, "finish_reason": "stop", "status": "CORRECT", "prompt": "resolve input"}
    for stage in ("t1", "note", "resolve"):
        value[stage]["generation"] = {"text": value[stage]["text"], "completion_token_ids": list(range(100, 140))}
    return value


def routes(value, image_route):
    kinds = ("RL", "A1") if value["t1"]["status"] == "CORRECT" else ("RL", "A2", "B")
    return {kind: single._candidate(value, kind, image_route=image_route)["route"] for kind in kinds}


def as_v15(rows, base):
    """The tiny v14 block contract (RL rows rebound under gate v9) with the v15 version, routes and attempts."""
    contract = as_v14(rows, base)
    v15 = protocol(VERSION_SINGLE_V15)["training_contract"]
    contract.update({key: v15[key] for key in ("version", "routes", "resolve_attempts", "generation_counts")})
    return contract


def set_route(rows, route):
    for row in rows:
        row["route"] = route
    return rows


def a1_row():
    row = encoded("item-0:A1", (7, 8), (12, 13), ROUTE_FULL, "A1")
    row.update(problem_id="item-0", image_sha256=["a" * 64])
    return row


class CandidateRoutes(unittest.TestCase):
    def test_v15_image_item_rows_are_all_full(self):
        for misread in (False, True):
            self.assertEqual(routes(record(correct=True, misread=misread), True), {"RL": ROUTE_FULL, "A1": ROUTE_FULL})
            self.assertEqual(routes(record(correct=False, misread=misread), True),
                             {"RL": ROUTE_FULL, "A2": ROUTE_FULL, "B": ROUTE_FULL})

    def test_v15_text_only_item_rows_are_lm_even_with_a_misread_note(self):
        self.assertEqual(routes(record(correct=True, misread=True, image=False), True), {"RL": ROUTE_LM, "A1": ROUTE_LM})
        self.assertEqual(routes(record(correct=False, misread=True, image=False), True),
                         {"RL": ROUTE_LM, "A2": ROUTE_LM, "B": ROUTE_LM})

    def test_v14_keeps_misread_only_full_and_lm_rl(self):
        self.assertEqual(routes(record(correct=False, misread=True), False),
                         {"RL": ROUTE_LM, "A2": ROUTE_FULL, "B": ROUTE_FULL})
        self.assertEqual(routes(record(correct=False, misread=False), False),
                         {"RL": ROUTE_LM, "A2": ROUTE_LM, "B": ROUTE_LM})
        self.assertEqual(routes(record(correct=True, misread=True), False), {"RL": ROUTE_LM, "A1": ROUTE_LM})

    def test_a2_reads_the_given_resolve_stage(self):
        value = record(correct=False, misread=False)
        other = SOLUTION.replace("step0", "first")
        value["resolve2"] = copy.deepcopy(value["resolve"]) | {"text": other}
        value["resolve2"]["generation"]["text"] = other
        self.assertEqual(single._candidate(value, "A2")["target"], SOLUTION)
        self.assertEqual(single._candidate(value, "A2", stage="resolve2")["target"], other)
        self.assertIsNone(single._candidate(value, "A2", stage="resolve3"))


class BlockRouteRule(v12.TinyThreads):
    def setUp(self):
        self.bundle = tiny_factory()

    def validate(self, rows, contract):
        return train._validate_single_block(self.bundle, rows, contract)

    def test_v15_image_rl_row_must_be_full(self):
        rows, base = block(self.bundle, count=1, reward=0)
        contract = as_v15(rows, base)
        with self.assertRaisesRegex(ValueError, ROUTE_ERROR):
            self.validate(rows, contract)
        self.assertEqual(len(self.validate(set_route(rows, ROUTE_FULL), contract)), 1)
        with self.assertRaisesRegex(ValueError, ROUTE_ERROR):  # an image A1 row on LM
            self.validate(rows + [a1_row() | {"route": ROUTE_LM}], contract)

    def test_v14_rl_row_with_full_is_refused(self):
        rows, base = block(self.bundle, count=1, reward=0)
        contract = as_v14(rows, base)
        self.assertEqual(len(self.validate(rows, contract)), 1)
        with self.assertRaisesRegex(ValueError, ROUTE_ERROR):
            self.validate(set_route(rows, ROUTE_FULL), contract)

    def test_v15_text_only_rl_row_must_be_lm(self):
        """The route rule accepts LM for a text-only row; the reward source binding then still requires image hashes
        (audit._single_source_bindings), so a text-only RL row never reaches training in any version."""
        rows, base = block(self.bundle, count=1, reward=0)
        contract = as_v15(rows, base)
        rows[0].update(image_paths=[], image_sha256=[])
        with self.assertRaisesRegex(ValueError, "image-file hashes required") as caught:
            self.validate(rows, contract)
        self.assertNotIn(ROUTE_ERROR, str(caught.exception))
        with self.assertRaisesRegex(ValueError, ROUTE_ERROR):
            self.validate(set_route(rows, ROUTE_FULL), contract)


class VisionGradient(v12.TinyThreads):
    def setUp(self):
        self.bundle = tiny_factory()

    def backward(self, rows, contract, components, **kwargs):
        self.bundle.model.zero_grad(set_to_none=True)
        stats = train.backward_single_block(self.bundle, rows, contract, components=components, **kwargs)
        return stats, _gradient_norms(self.bundle)

    def test_v15_pg_and_a1_alone_give_vision_gradients(self):
        rows, base = block(self.bundle, count=2, reward=0)
        contract = as_v15(rows, base)
        set_route(rows, ROUTE_FULL)
        stats, norms = self.backward(rows, contract, ("PG",))
        self.assertEqual(stats["rl"]["pg_rows_negative"], 2)
        for group in ("L", "M", "V"):
            self.assertGreater(norms[group] or 0, 0, group)
        self.assertFalse(self.bundle.route.vision_on)  # the initial route is restored
        rows.append(a1_row())
        stats, norms = self.backward(rows, contract, ("A1",))
        self.assertEqual(stats["sft"]["A1"]["rows"], 1)
        for group in ("L", "M", "V"):
            self.assertGreater(norms[group] or 0, 0, group)
        self.bundle.model.zero_grad(set_to_none=True)
        report = single.prove_single_backward(self.bundle, rows, contract, precision="bf16")
        self.assertNotIn("error", report)
        for name in ("PG", "A1"):
            self.assertTrue(report["components"][name]["route_valid"], name)
            self.assertGreater(report["components"][name]["gradient_norms"]["V"] or 0, 0, name)
        self.assertTrue(report["all_parameters_unchanged"] and report["gradients_cleared"])

    def test_v14_pg_gives_no_vision_gradient(self):
        rows, base = block(self.bundle, count=2, reward=0)
        stats, norms = self.backward(rows, as_v14(rows, base), ("PG",))
        self.assertEqual(stats["rl"]["pg_rows_negative"], 2)
        self.assertGreater(norms["L"] or 0, 0)
        self.assertIsNone(norms["V"])

    def test_a_capacity_profile_refuses_full_rl_rows(self):
        rows, base = block(self.bundle, count=1, reward=0)
        contract = as_v15(rows, base)
        set_route(rows, ROUTE_FULL)
        with self.assertRaisesRegex(CapacityHold, "capacity HOLD: the measured profile has no FULL RL phase"):
            self.backward(rows, contract, ("PG",), capacity=object())
        self.assertTrue(all(p.grad is None for ps in self.bundle.params.values() for p in ps))


if __name__ == "__main__":
    unittest.main()
