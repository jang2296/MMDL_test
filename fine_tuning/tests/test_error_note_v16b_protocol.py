"""Sealed v16b schedule and legacy protocol regression checks, without model access."""
from __future__ import annotations

import copy
import unittest

from fine_tuning.error_note import protocol as proto, single
from fine_tuning.error_note.contract import sample_seed


def protocol(*, prep_updates=1, local_smoke=False):
    return proto.plain_protocol_v16b("nf4" if local_smoke else "bf16",
        rl_blocks=4 if local_smoke else 188, prep_updates=prep_updates,
        contracts={key: "0" * 64 for key in proto.V16B_CONTRACT_HASHES}, local_smoke=local_smoke)


class V16bProtocolTests(unittest.TestCase):
    def test_full_protocol_preserves_complete_rl_and_b2_evaluation(self):
        sealed = protocol(prep_updates=12)
        proto.validate_protocol_v16b(sealed)
        single._protocol_check(sealed)
        contract = sealed["training_contract"]
        self.assertEqual((contract["rl_items"], contract["rl_blocks"], contract["items_per_block"]), (3000, 188, 16))
        self.assertEqual(contract["schedule"]["last_block_items"], 8)
        self.assertEqual(contract["total_slots"], 12 + 188 + 3)
        self.assertEqual(sealed["optimizer"]["total_steps"], contract["total_slots"])
        self.assertIsNone(contract["schedule"]["time_cut"])
        self.assertEqual(sealed["sampling"], {"temperature": .7, "top_p": 1., "top_k": 0, "presence_penalty": 1.5})
        self.assertEqual((sealed["eval_sampling"]["top_p"], sealed["eval_sampling"]["top_k"]), (.8, 20))
        self.assertEqual(sealed["image"], {"min_pixels": 1003520, "max_pixels": 4014080,
                                           "preprocessing": "qwen_vl_utils_0_0_14"})
        self.assertEqual(contract["generation_counts"], {"t1": 1, "obs": 1, "note": 1, "resolve": 1})
        self.assertEqual(contract["max_tokens"]["t1"], 8192)
        self.assertEqual(contract["probability_space"], "processed-full-vocab-v1")
        self.assertEqual(contract["sft_weights"], {"PREP_SELF": 1., "PREP_HUMAN": 1., "C": .35, "N": .05})
        self.assertFalse(contract["prep"]["overlap_with_rl"])
        self.assertFalse(contract["prep"]["overlap_with_monitor"])
        self.assertTrue(contract["cloud_allowed"])
        self.assertFalse(contract["local_smoke"])

    def test_nf4_smoke_is_explicit_bounded_and_cannot_launch_cloud(self):
        sealed = protocol(local_smoke=True)
        proto.validate_protocol_v16b(sealed)
        contract = sealed["training_contract"]
        self.assertEqual((sealed["precision"], contract["rl_items"], contract["rl_blocks"]), ("nf4", 8, 4))
        self.assertEqual(contract["total_slots"], 8)
        self.assertEqual(set(contract["max_tokens"].values()), {256})
        self.assertEqual(sealed["image"]["max_pixels"], 262144)
        self.assertFalse(contract["cloud_allowed"])
        self.assertTrue(contract["local_smoke"])

    def test_rejects_unbounded_or_mismatched_modes_and_unpinned_inputs(self):
        pins = {key: "0" * 64 for key in proto.V16B_CONTRACT_HASHES}
        for extra in ({"precision": "nf4"}, {"rl_blocks": 187}, {"prep_updates": -1},
                      {"prep_updates": 19}, {"prep_updates": True}, {"local_smoke": 1},
                      {"image": {"min_pixels": 1, "max_pixels": 1}},
                      {"contracts": {key: value for key, value in pins.items() if key != "monitor"}},
                      {"contracts": pins | {"extra": "0" * 64}},
                      {"contracts": pins | {"source": "A" * 64}}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                kwargs = {"precision": "bf16", "rl_blocks": 188, "prep_updates": 1, "contracts": pins} | extra
                proto.plain_protocol_v16b(**kwargs)
        for prep in (0, 2):
            with self.subTest(smoke_prep=prep), self.assertRaises(ValueError):
                protocol(prep_updates=prep, local_smoke=True)

    def test_resealed_relaxations_still_fail_validation(self):
        mutations = (
            ("sampling", "top_p", .8), ("sampling", "top_k", 20),
            ("training_contract", "rl_items", 2999), ("training_contract", "finish_updates", 2),
            ("training_contract", "full_kl_beta", .01),
            ("training_contract", "cloud_allowed", False),
            ("training_contract.schedule", "time_cut", 3600),
            ("training_contract.generation_counts", "resolve", 3),
            ("training_contract.max_tokens", "t1", 4096),
            ("training_contract.routes", "image", "LM"),
            ("training_contract.prep", "overlap_with_rl", True),
            ("training_contract.reward", "CORRECT", 2.),
            ("optimizer.lr", "L", .001),
        )
        for path, field, value in mutations:
            candidate = copy.deepcopy(protocol())
            container = candidate
            for component in path.split("."):
                container = container[component]
            container[field] = value
            candidate["sha256"] = proto._digest({key: value for key, value in candidate.items() if key != "sha256"})
            with self.subTest(path=path, field=field), self.assertRaisesRegex(ValueError, "invalid sealed"):
                single._protocol_check(candidate)

    def test_update_helper_separates_global_lr_slots_from_rl_versions_and_pins_rewards(self):
        sealed = protocol(prep_updates=2)
        rewards = {"RL:p0": {"reward": 0., "pg": False}}
        baseline = {"RL:p0": {"baseline": .4, "sha256": "b" * 64}}
        prep = proto.v16b_contract_for_update(sealed, 1, phase="prep", policy_sha256="a" * 64,
                                             expected_problem_ids=["prep-parent"])
        rl = proto.v16b_contract_for_update(sealed, 2, phase="rl", policy_sha256="a" * 64,
            expected_problem_ids=["p0"], baseline_receipts=baseline, reward_values=rewards)
        self.assertEqual((prep["global_slot"], prep["block_id"], prep["policy_version"]), (1, 1, 0))
        self.assertEqual((rl["global_slot"], rl["block_id"], rl["policy_version"]), (2, 0, 0))
        self.assertEqual(rl["reward_values"], rewards)
        self.assertEqual(rl["protocol_sha256"], sealed["sha256"])
        rewards["RL:p0"]["reward"] = 1.
        baseline["RL:p0"]["baseline"] = 1.
        self.assertEqual(rl["reward_values"]["RL:p0"], {"reward": 0., "pg": False})
        self.assertEqual(rl["baseline_receipts"]["RL:p0"]["baseline"], .4)
        for slot, phase in ((0, "rl"), (2, "prep"), (-1, "prep"), (193, "rl"), (True, "prep")):
            with self.subTest(slot=slot, phase=phase), self.assertRaises(ValueError):
                proto.v16b_contract_for_update(sealed, slot, phase=phase, policy_sha256="a" * 64,
                                               expected_problem_ids=[])

    def test_builders_do_not_share_mutable_nested_contracts(self):
        first = protocol()
        first["training_contract"]["pg"]["is_cap"] = 99
        first["training_contract"]["schedule"]["time_cut"] = 1
        second = protocol()
        self.assertEqual(second["training_contract"]["pg"]["is_cap"], 2.)
        self.assertIsNone(second["training_contract"]["schedule"]["time_cut"])
        proto.validate_protocol_v16b(second)

    def test_legacy_protocol_golden_hashes_remain_unchanged(self):
        # Saved before the v16b edits in implementation-20261115/legacy-protocol-hashes.json.
        expected = {
            proto.VERSION_SINGLE: "b98882d2cba4dc79455173d52ef69b0695ebda62a083d00aaff6e618222acc7a",
            proto.VERSION_SINGLE_V12: "46269db2e0a8083dbdc4fae86056c87931a364a92cf146ed97dd673d26c3480d",
            proto.VERSION_SINGLE_V13: "34121b802c936410271cd3b974e453b2a627f4cdae2420e5df7d2eab6ee0cc19",
            proto.VERSION_SINGLE_V14: "4091a611a1e51a7ce2f39f791dd1a06684a68bf9cb0c887f4afbcdce17aed726",
            proto.VERSION_SINGLE_V15: "4b5ed9fb7d524b795c21198bada8eee8dffd88f7299bb6c7a446e674e0232c75",
            proto.VERSION_CONTENT_V16: "00e3cee4c69bf935259a7f0be8870626a7b92368a34835c92fe6485694517afd",
            proto.VERSION_V2: "b5c949ed5c930e38d0b35f26349af1d577fd31c130941da8dd7cf6f1548f0452",
        }
        pins = {key: "0" * 64 for key in proto.SINGLE_CONTRACT_HASHES}
        actual = {}
        for version in (proto.VERSION_SINGLE, proto.VERSION_SINGLE_V12, proto.VERSION_SINGLE_V13,
                        proto.VERSION_SINGLE_V14, proto.VERSION_SINGLE_V15):
            extra = {"schedule_steps": 64} if version in (proto.VERSION_SINGLE_V14, proto.VERSION_SINGLE_V15) else {}
            actual[version] = proto.plain_protocol_single("bf16", 219, contracts=pins, version=version, **extra)["sha256"]
        actual[proto.VERSION_CONTENT_V16] = proto.plain_protocol_content_v16(
            "bf16", 16, contracts=pins, source_weights={"graph": 1.})["sha256"]
        actual[proto.VERSION_V2] = proto.plain_protocol_v2("bf16", 10)["sha256"]
        self.assertEqual(actual, expected)

    def test_stage_seeds_preserve_legacy_and_separate_v16b_requests(self):
        expected = {
            "t1": 1134404534, "note": 685958697, "resolve": 1711512893,
            "resolve2": 1390238722, "resolve3": 235976557, "control": 418691227,
            "v16b-t1": 284040443, "v16b-obs": 1083765923, "v16b-note": 1577712980,
            "v16b-resolve": 2056519179, "v16b-prep": 472290066, "v16b-monitor": 1469250892,
        }
        self.assertEqual({stage: sample_seed("fixture-problem", stage) for stage in expected}, expected)
        self.assertEqual(len(set(expected.values())), len(expected))
        with self.assertRaises(ValueError):
            sample_seed("fixture-problem", "unknown-stage")


if __name__ == "__main__":
    unittest.main()
