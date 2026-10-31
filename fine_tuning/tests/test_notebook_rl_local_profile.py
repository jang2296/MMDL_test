"""The local experiment must not silently inherit the historical BF16 identity."""
import copy
import unittest

from fine_tuning.notebook_rl.checkpoint import digest
from fine_tuning.notebook_rl.local_profile import (ACTOR_MODE_POLICY, ATOMIC_PROTOCOL_ID, HERE,
    PROTOCOL_ID, build_protocol, local_prompts, restore_protocol)
from fine_tuning.notebook_rl.schema import load_protocol


class LocalProfileTests(unittest.TestCase):
    def test_all_pairs_revision_requires_native_update_and_rejects_old_identity(self):
        from fine_tuning.notebook_rl.updates import NATIVE_RELATIONSHIP_REQUIREMENT

        current = build_protocol(pair_action_notebook=True, verified_note_synthesis_credit=True,
                                 context_limit=24576)
        self.assertEqual(current["protocol_id"], "local-nf4-pair-action-notebook-synthesis-credit-24k-v2")
        self.assertEqual(current["notebook_generation_policy"], "verified-pair-action-notebook-v2")
        self.assertEqual(current["notebook_update_requirement"], NATIVE_RELATIONSHIP_REQUIREMENT)
        self.assertEqual(restore_protocol(current), current)
        for field, replacement in (("protocol_id", current["protocol_id"].replace("-v2", "-v1")),
                                   ("notebook_update_requirement", None)):
            changed = copy.deepcopy(current)
            changed[field] = replacement
            changed["sha256"] = digest({key: value for key, value in changed.items() if key != "sha256"})
            with self.subTest(field=field), self.assertRaises(ValueError):
                restore_protocol(changed)

    def test_verifier_semantics_are_bound_and_missing_or_old_versions_cannot_be_rehashed(self):
        from fine_tuning.notebook_rl.verifier import VERSION

        for kwargs in ({}, {"verified_note_fact_credit": True},
                       {"pair_draft_synthesis": True, "verified_note_synthesis_credit": True,
                        "context_limit": 24576}):
            current = build_protocol(**kwargs)
            self.assertEqual(current["verifier_version"], VERSION)
            self.assertEqual(restore_protocol(current), current)
            for version in (None, VERSION.replace("relations-v3", "relations-v2"), "unknown"):
                stale = copy.deepcopy(current)
                if version is None:
                    stale.pop("verifier_version")
                else:
                    stale["verifier_version"] = version
                stale["sha256"] = digest({k: v for k, v in stale.items() if k != "sha256"})
                with self.subTest(kwargs=kwargs, version=version):
                    with self.assertRaisesRegex(ValueError, "saved protocol changed"):
                        restore_protocol(stale)
                    with self.assertRaisesRegex(ValueError, "verifier version"):
                        local_prompts(stale)

    def test_prior_draft_protocol_requires_a_new_preparation(self):
        current = build_protocol(pair_draft_synthesis=True, verified_note_synthesis_credit=True,
                                 context_limit=24576)
        self.assertEqual(current["protocol_id"], "local-nf4-pair-draft-synthesis-credit-24k-v2")
        old = current | {"protocol_id": "local-nf4-pair-draft-synthesis-credit-24k-v1"}
        old["sha256"] = digest({k: v for k, v in old.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "unknown saved protocol"):
            restore_protocol(old)

    def test_24k_requires_explicit_fact_credit_and_only_changes_identity_and_context(self):
        import json
        from fine_tuning.notebook_rl.local_profile import FACT_CREDIT_24K_PROTOCOL_ID

        original = build_protocol(verified_note_fact_credit=True)
        expanded = build_protocol(verified_note_fact_credit=True, context_limit=24576)
        self.assertEqual(FACT_CREDIT_24K_PROTOCOL_ID, "local-nf4-verified-note-fact-credit-24k-v19")
        self.assertEqual(expanded["protocol_id"], FACT_CREDIT_24K_PROTOCOL_ID)
        self.assertEqual(expanded["context_limit"], 24576)
        self.assertNotEqual(original["sha256"], expanded["sha256"])
        self.assertEqual(restore_protocol(json.loads(json.dumps(expanded))), expanded)
        self.assertEqual(local_prompts(original), local_prompts(expanded))
        for key in original.keys() - {"sha256", "protocol_id", "context_limit"}:
            self.assertEqual(original[key], expanded[key])
        for kwargs in ({}, {"verified_note_atoms": True}, {"verified_note_components": True},
                       {"verified_note_atoms": True, "verified_note_fact_credit": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_protocol(context_limit=24576, **kwargs)
        for limit in (True, False, "24576", 24576.0, None, 8192, 32768, 0, -1):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "context limit"):
                build_protocol(verified_note_fact_credit=True, context_limit=limit)

    def test_context_and_objective_are_bound_to_canonical_id_even_after_rehash(self):
        choices = ({}, {"verified_note_atoms": True}, {"verified_note_components": True},
                   {"verified_note_fact_credit": True},
                   {"verified_note_fact_credit": True, "context_limit": 24576})
        protocols = [build_protocol(**kwargs) for kwargs in choices]
        for current in protocols:
            with self.subTest(protocol=current["protocol_id"]):
                for context in (16384 if current["context_limit"] == 24576 else 24576,
                                float(current["context_limit"]), True, "16384", None, 32768):
                    changed = current | {"context_limit": context}
                    changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
                    with self.assertRaisesRegex(ValueError, "saved protocol changed"):
                        restore_protocol(changed)
                    with self.assertRaisesRegex(ValueError, "canonical protocol"):
                        local_prompts(changed)
                for other in protocols:
                    if other["protocol_id"] == current["protocol_id"]:
                        continue
                    changed = current | {"protocol_id": other["protocol_id"]}
                    changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
                    with self.assertRaisesRegex(ValueError, "saved protocol changed"):
                        restore_protocol(changed)
                    with self.assertRaises(ValueError):
                        local_prompts(changed)

    def test_stale_reward_and_wrong_slot_contracts_are_rejected_after_rehash(self):
        choices: tuple[dict[str, bool], ...] = ({}, {"verified_note_atoms": True}, {"verified_note_components": True},
                   {"verified_note_fact_credit": True})
        for kwargs in choices:
            with self.subTest(kwargs=kwargs):
                current = build_protocol(**kwargs)
                stale = copy.deepcopy(current)
                stale["notebook_reward"] = "local-all-attempt-notebook-v4"
                stale["sha256"] = digest({k: v for k, v in stale.items() if k != "sha256"})
                self.assertNotEqual(stale["sha256"], current["sha256"])
                with self.assertRaisesRegex(ValueError, "saved protocol changed"):
                    restore_protocol(stale)
                with self.assertRaisesRegex(ValueError, "prompt/reward mismatch"):
                    local_prompts(stale)
                if kwargs:
                    objective = current["notebook_objective"]
                    self.assertEqual(objective["strict_reward_version"], current["notebook_reward"])
                    stale = copy.deepcopy(current)
                    stale["notebook_objective"].pop("wrong_slot_ids")
                    stale["sha256"] = digest({k: v for k, v in stale.items() if k != "sha256"})
                    with self.assertRaisesRegex(ValueError, "saved protocol changed"):
                        restore_protocol(stale)
                    with self.assertRaisesRegex(ValueError, "contract changed"):
                        local_prompts(stale)

    def test_fact_credit_is_opt_in_frozen_and_keeps_generation_and_route_contract(self):
        import json
        from fine_tuning.notebook_rl.local_profile import FACT_CREDIT_PROTOCOL_ID
        from fine_tuning.notebook_rl.notebook_objective import FACT_CREDIT_VERSION, contract

        original, scoped = build_protocol(), build_protocol(verified_note_fact_credit=True)
        self.assertEqual(scoped["protocol_id"], FACT_CREDIT_PROTOCOL_ID)
        self.assertEqual(FACT_CREDIT_VERSION, "verified-note-fact-token-credit-v2")
        self.assertEqual(scoped["notebook_objective"], contract(FACT_CREDIT_VERSION))
        self.assertEqual(restore_protocol(json.loads(json.dumps(scoped))), scoped)
        self.assertEqual(local_prompts(original), local_prompts(scoped))
        for key in original.keys() - {"sha256", "protocol_id"}:
            self.assertEqual(original[key], scoped[key])
        for extra in ({"verified_note_atoms": True}, {"verified_note_components": True}):
            with self.assertRaises(ValueError):
                build_protocol(verified_note_fact_credit=True, **extra)
        with self.assertRaises(ValueError):
            build_protocol(verified_note_fact_credit="true")
        with self.assertRaisesRegex(ValueError, "contract changed"):
            local_prompts(scoped | {"notebook_objective": contract()})
        stale = copy.deepcopy(scoped)
        stale["notebook_objective"].update(objective_version="verified-note-fact-token-credit-v1",
            fact_guard="canonical IDs; no unknown keys, conflicts, false or unverified relations")
        stale["notebook_objective"].pop("corrected_fact_ids")
        stale["notebook_objective"].pop("fact_id_collisions")
        stale["sha256"] = digest({k: v for k, v in stale.items() if k != "sha256"})
        self.assertNotEqual(stale["sha256"], scoped["sha256"])
        with self.assertRaisesRegex(ValueError, "saved protocol changed"):
            restore_protocol(stale)
        with self.assertRaisesRegex(ValueError, "contract changed"):
            local_prompts(stale)

    def test_components_are_explicit_frozen_and_incompatible_with_atomic_selection(self):
        import json
        from fine_tuning.notebook_rl.local_profile import COMPONENT_PROTOCOL_ID
        from fine_tuning.notebook_rl.notebook_objective import COMPONENT_VERSION, contract

        original, components = build_protocol(), build_protocol(verified_note_components=True)
        self.assertEqual(components["protocol_id"], COMPONENT_PROTOCOL_ID)
        self.assertEqual(components["notebook_objective"], contract(COMPONENT_VERSION))
        self.assertEqual(restore_protocol(json.loads(json.dumps(components))), components)
        self.assertEqual(local_prompts(components), local_prompts(original))
        for key in original.keys() - {"sha256", "protocol_id"}:
            self.assertEqual(original[key], components[key])
        with self.assertRaises(ValueError):
            build_protocol(verified_note_atoms=True, verified_note_components=True)
        with self.assertRaises(ValueError):
            build_protocol(verified_note_components="yes")
        with self.assertRaisesRegex(ValueError, "contract changed"):
            local_prompts(components | {"notebook_objective": contract()})

    def test_atomic_objective_is_explicit_separate_and_survives_json_reload(self):
        import json
        from fine_tuning.notebook_rl.notebook_objective import contract

        original, atomic = build_protocol(), build_protocol(verified_note_atoms=True)
        self.assertNotIn("notebook_objective", original)
        self.assertEqual(atomic["protocol_id"], ATOMIC_PROTOCOL_ID)
        self.assertEqual(atomic["notebook_objective"], contract())
        self.assertNotEqual(original["sha256"], atomic["sha256"])
        for key in original.keys() - {"sha256", "protocol_id"}:
            self.assertEqual(original[key], atomic[key])
        for protocol in (original, atomic):
            self.assertEqual(restore_protocol(json.loads(json.dumps(protocol))), protocol)
            self.assertEqual(local_prompts(protocol), local_prompts(original))
            altered = copy.deepcopy(protocol)
            altered["context_limit"] -= 1
            with self.assertRaisesRegex(ValueError, "saved protocol changed"):
                restore_protocol(altered)
        with self.assertRaisesRegex(ValueError, "own protocol"):
            local_prompts(original | {"notebook_objective": contract()})
        with self.assertRaisesRegex(ValueError, "contract changed"):
            local_prompts(atomic | {"notebook_objective": contract() | {"atomic_weight": .5}})
        with self.assertRaisesRegex(ValueError, "unknown saved protocol"):
            restore_protocol(atomic | {"protocol_id": "unknown"})
        with self.assertRaises(ValueError):
            build_protocol(verified_note_atoms="true")

    def test_separate_reproducible_profile_keeps_all_module_groups(self):
        historical = load_protocol(HERE / "protocol_v1.yaml")
        local = build_protocol()
        self.assertEqual(local, build_protocol())
        self.assertEqual(local["protocol_id"], PROTOCOL_ID)
        self.assertEqual(PROTOCOL_ID, "local-nf4-full-history-notebook-v13")
        self.assertEqual(local["notebook_input_policy"], "question-choices-images-and-full-transcript-v1")
        with self.assertRaisesRegex(ValueError, "notebook input policy"):
            local_prompts(local | {"notebook_input_policy": None})
        self.assertEqual(local["actor_mode_policy"], ACTOR_MODE_POLICY)
        with self.assertRaisesRegex(ValueError, "observation-elicitation"):
            local_prompts(local | {"actor_mode_policy": None})
        self.assertNotEqual(local["sha256"], historical["sha256"])
        self.assertEqual(local["parent_protocol_sha256"], historical["sha256"])
        for key in ("expected_trainable", "forbidden", "route_rule", "initial_checkpoint"):
            self.assertEqual(local[key], historical[key])
        self.assertEqual(local["model"]["lora"], historical["model"]["lora"])
        self.assertEqual(local["model"]["modules_to_save"], historical["model"]["modules_to_save"])
        self.assertEqual(local["sha256"], digest({k: v for k, v in local.items() if k != "sha256"}))
        self.assertIn("rewarded_notebook", local_prompts(local))
        self.assertEqual(local["notebook_observation_n"], 1)
        self.assertEqual(local["notebook_observation_max_new_tokens"], 1024)
        for key in ("notebook_observation_policy", "notebook_observation_prompt_sha256",
                    "notebook_observation_max_new_tokens", "notebook_observation_n",
                    "notebook_observation_seed_rule"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "fresh observation"):
                local_prompts(local | {key: None})
        self.assertEqual(local["notebook_reward"], "local-all-attempt-notebook-v5")
        self.assertEqual(local["context_limit"], 16384)
        self.assertEqual(local["rollout"]["diag"]["max_new_tokens"], 2048)
        self.assertEqual(local["notebook_max_rounds"], 3)
        for key in ("notebook_max_rounds", "notebook_retry_instruction_sha256"):
            altered = copy.deepcopy(local)
            altered[key] = None
            with self.assertRaisesRegex(ValueError, "retry budget/instruction"):
                local_prompts(altered)
        self.assertNotEqual(local["sha256"],
            "57efc3fbc17e93d75e4253ce6c8cc8345c918f6a1476e8cbd1208f411b876843")
        self.assertNotEqual(local["sha256"],
            "4f2b91786a04e545c6f493109ccb82c466c88fc200b13c6883288727961fe810")
        self.assertNotEqual(local["sha256"],
            "62e3f98404d3e77e8d29e89c0b12ee261690b5a24cffc1cf8b192eb1a28dbff0")
        altered = copy.deepcopy(local)
        altered["notebook_prompt_sha256"] = "0"*64
        with self.assertRaises(ValueError):
            local_prompts(altered)


if __name__ == "__main__":
    unittest.main()
