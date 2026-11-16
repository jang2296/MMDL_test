"""Version boundaries and fixed v17 loss/schedule identifiers are sealed together."""
import copy
import hashlib
import json
import unittest

from fine_tuning.error_note import protocol as proto


def protocol(*, smoke=False, prep_updates=1):
    return proto.plain_protocol_v17("nf4" if smoke else "bf16", rl_blocks=4 if smoke else 188,
        prep_updates=prep_updates, contracts={key: "0" * 64 for key in proto.V17_CONTRACT_HASHES},
        local_smoke=smoke)


class V17ProtocolTests(unittest.TestCase):
    def test_complete_schedule_and_explicit_loss_contract(self):
        sealed = protocol(prep_updates=12)
        proto.validate_protocol_v17(sealed)
        contract = sealed["training_contract"]
        self.assertEqual(contract | proto.V17_IDENTIFIERS, contract)
        self.assertEqual((contract["rl_items"], contract["prep"]["items"], contract["monitor_items"]),
                         (3000, 512, 200))
        self.assertEqual((contract["rl_blocks"], contract["schedule"]["last_block_items"],
                          contract["total_slots"]), (188, 8, 203))
        self.assertEqual(contract["sft_weights"],
                         {"PREP_SELF": 1., "PREP_HUMAN": 1., "A1": .2, "C": .35, "N": .05})
        self.assertEqual(contract["prep"]["self_tokens"], [1, 8192])
        self.assertEqual(contract["anchor"]["positions"], 64)
        self.assertEqual(contract["anchor"]["prep_positions"], 4)
        self.assertEqual(contract["pg"]["hold_mixed_fraction"], 0.)
        self.assertEqual(contract["master_seed"], 3407)
        self.assertEqual(contract["f0_position_salt"], "v17-f0-stratified-64-v1")
        self.assertFalse(contract["measure_component_gradients"])
        self.assertTrue(contract["reward"]["LOOP_CONFIRMED"]["pg"])
        self.assertFalse(contract["reward"]["LENGTH"]["pg"])
        self.assertEqual(sealed["eval_sampling"], proto.EVAL_SAMPLING_V16B)

    def test_resealed_changes_cannot_relax_v17(self):
        for key, value in (("probability_space_sft", "processed-full-vocab-v1"),
                           ("main_parent_denominator", 8), ("eval_concurrency", 32),
                           ("training_elapsed_cutoff", 54000), ("mixed_policy_action", "skip")):
            candidate = protocol()
            candidate["training_contract"][key] = value
            candidate["sha256"] = proto._digest({k: v for k, v in candidate.items() if k != "sha256"})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "invalid sealed"):
                proto.validate_protocol_v17(candidate)

    def test_update_contract_preserves_slots_and_detaches_receipts(self):
        sealed = protocol(prep_updates=2)
        baselines = {"RL:p": {"baseline": .4}}
        result = proto.v17_contract_for_update(sealed, 189, phase="rl", policy_sha256="a" * 64,
            expected_problem_ids=["p"], expected_obs_problem_ids=["p"], note_problem_ids=["older"],
            baseline_receipts=baselines)
        baselines["RL:p"]["baseline"] = .9
        self.assertEqual((result["global_slot"], result["block_id"], result["policy_version"]), (189, 187, 187))
        self.assertEqual(result["main_parent_denominator"], 16)
        self.assertEqual(result["baseline_receipts"]["RL:p"]["baseline"], .4)
        self.assertEqual(result["protocol_sha256"], sealed["sha256"])
        for slot, phase in ((0, "rl"), (2, "prep"), (True, "prep"), (193, "rl")):
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                proto.v17_contract_for_update(sealed, slot, phase=phase,
                    policy_sha256="a" * 64, expected_problem_ids=[])

    def test_bounded_nf4_mode_and_mutable_defaults_are_separate(self):
        smoke = protocol(smoke=True)
        proto.validate_protocol_v17(smoke)
        self.assertEqual((smoke["training_contract"]["rl_items"],
                          smoke["training_contract"]["total_slots"]), (8, 8))
        self.assertFalse(smoke["training_contract"]["cloud_allowed"])
        changed = copy.deepcopy(protocol())
        changed["training_contract"]["anchor"]["beta"] = 99
        self.assertEqual(protocol()["training_contract"]["anchor"]["beta"], .02)

    def test_legacy_v16b_output_bytes_are_unchanged(self):
        legacy = proto.plain_protocol_v16b("bf16", rl_blocks=188, prep_updates=1,
            contracts={key: "0" * 64 for key in proto.V16B_CONTRACT_HASHES})
        actual = hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(actual, "7ccc45c632c545ea9892255c13f65b38c4cbac72c16d5a74805c4b1eada8f7a1")
