"""Legacy f360427 prompt identities, paired with the recovered seventh v18 campaign."""
import inspect
import unittest
from unittest.mock import patch

from fine_tuning.error_note import v18_rows as old, v18b_rows as current
from fine_tuning.error_note.audit import digest, text_sha256

LEGACY = "7c9820bc5f8aec2f16e59323c566b836132f3eceb571b9c138520b26e39d39c7"


class PromptIdentity(unittest.TestCase):
    def test_legacy_identity_fails_closed_after_source_or_instruction_changes(self):
        self.assertEqual(old.prompts_sha256(), LEGACY)
        source = inspect.getsource
        for builder in (old._query, old.build_note_request_v18):
            with patch.object(inspect, "getsource", side_effect=lambda fn: source(fn) + "# changed\n"
                              if fn is builder else source(fn)):
                self.assertNotEqual(old.prompts_sha256(), LEGACY)
        with patch.object(old, "NOTE_INSTRUCTION_V18", "changed instruction"):
            self.assertNotEqual(old.prompts_sha256(), LEGACY)

    def test_v18b_binds_shared_sources_even_when_legacy_identity_is_fixed(self):
        source = inspect.getsource
        with patch.object(old, "prompts_sha256", return_value=LEGACY):
            baseline = current.prompts_sha256()
            for builder in (old._query, old.build_note_request_v18, old.build_obs_request_v18,
                            current.bound_observation_request_v18b, current.checked.balanced_query,
                            current.render_note_target_last):
                with patch.object(inspect, "getsource", side_effect=lambda fn: source(fn) + "# changed\n"
                                  if fn is builder else source(fn)):
                    self.assertNotEqual(current.prompts_sha256(), baseline)

    def test_default_note_prompts_match_published_f360427_golden_outputs(self):
        golden = (
            ("self_found", False, "27ddd000865287fe0634c245bbfbbb53d9fb33f05f1565d504f55b60293c658e"),
            ("self_found", True, "aafc0caffcdb1e190cb393f1c5f9351fdec5963fc2ca6602822ae34808691c96"),
            ("location_given", False, "c1c75c52fe2f5c848368a1f61533d03a1a67730e76dffef1cf8594357bbbbe43"),
            ("location_given", True, "0b3772a22375a6739e67bfc33ac5e2c10a86fde2e8c18917bbc5d12f9fda2780"),
        )
        for condition, observed, expected in golden:
            identity = next("fixture" + str(n) for n in range(100)
                            if old.note_condition_v18("fixture" + str(n)) == condition)
            item = {"problem_id": identity, "p0_text": "Question: Double the printed segment AB.",
                    "image_sha256": [], "t1": {"text": "AB = 9\nFinal answer: B", "finish_reason": "stop",
                    "status": "INCORRECT", "generation": {"completion_token_ids": [1, 2, 3]}}}
            ledger = old._seal({"problem_id": identity, "stage": "T1", "image_sha256": [],
                "generation_sha256": digest(item["t1"]["generation"]), "claims": [{"verdict": "CONTRADICTED",
                "assertion": "ASSERTED", "original_byte_spans": [[0, 6]], "original_quote": "AB = 9"}]})
            if observed:
                item["obs"] = {"text": '{"value":"9","unit":"length"}'}
            for formatted in (False, True):
                if formatted:
                    item["t1_policy_receipt"] = {"format_receipt": old._seal(
                        {"accepted": True, "output_sha256": digest(item["t1"])})}
                request = old.build_note_request_v18(item, ledgers=[ledger], gold="14")
                self.assertEqual(text_sha256(request["prompt"]), expected)

    def test_default_observation_prompts_match_published_f360427_golden_outputs(self):
        cases = (
            ("geometry", "Use twice the printed segment AB; multiply by 2.",
             {"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": "7", "unit": "length"},
             "2a8b3d06592b63ed416ec3e41a64e41f08ef5ebe282502036fcf3cb71c61a134"),
            ("scene", "Count the objects.",
             {"kind": "OBJECT_ATTRIBUTE", "entity": [1, "color"], "value": "red", "unit": ""},
             "5ac76ef441d399a7585bd129625bf432dabddec57201ea530043cb76a0a43dfc"),
            ("scene", "What color is the cube?",
             {"kind": "OBJECT_ATTRIBUTE", "entity": [1, "material"], "value": "metal", "unit": ""},
             "8ae481583ab80041d3e106bd3d6488c1670dff4023666746c6ff9dfe0195e095"),
            ("graph", "Find the shortest path.",
             {"kind": "EDGE_PRESENT", "entity": [0, 1], "value": True, "unit": ""},
             "55cb9ab8e3062c2301f440cce30da95fa19472d98723f7a135bd44d48163c7c5"),
        )
        for layer, question, fact, expected in cases:
            item = {"problem_id": "fixture", "layer": layer, "question": question, "image_sha256": []}
            item["v17_private_verifier"] = old._seal({"problem_id": "fixture", "image_sha256": [],
                "facts": [fact], "closed_world_kinds": ["EDGE_PRESENT"],
                "closed_world_scopes": {"visible_nodes": [0, 1]}})
            request = old.build_obs_request_v18(item)
            self.assertEqual(text_sha256(request["prompt"]), expected)
