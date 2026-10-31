"""CPU protocol metadata checks; no optimizer, CLI or real-model readiness claim."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from typing import Any
from unittest.mock import patch

from fine_tuning.notebook_rl import local_profile as profile, notebook_synthesis, reasoning_trace
from fine_tuning.notebook_rl import repair_contract, repair_credit, repair_notebook
from fine_tuning.notebook_rl.checkpoint import digest
from fine_tuning.notebook_rl.prompts import (REPAIR_RETRY, load_prompts, native_solution_prompt_contract,
                                            repair_retry_prompt_contract)
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash
from fine_tuning.notebook_rl.verifier import VERSION as VERIFIER_VERSION


_LEGACY_FLAGS: tuple[dict[str, Any], ...] = (
    {}, {"verified_note_atoms": True}, {"verified_note_components": True}, {"verified_note_fact_credit": True},
    {"verified_note_fact_credit": True, "context_limit": 24576}, {"verified_note_synthesis": True},
    {"verified_note_synthesis": True, "context_limit": 24576},
    {"verified_note_synthesis_credit": True, "context_limit": 24576},
    *({"verified_note_synthesis_credit": True, "context_limit": 24576, key: True} for key in (
        "partial_notebook", "referenced_notebook", "paired_notebook", "pair_action_notebook", "pair_draft_synthesis")),
    {"verified_note_synthesis_credit": True, "pair_draft_synthesis": True, "whole_reference_notebook": True, "context_limit": 24576},
    {"verified_note_synthesis_credit": True, "pair_draft_synthesis": True, "whole_reference_pages": True, "context_limit": 24576},
    {"verified_note_synthesis_credit": True, "pair_draft_synthesis": True, "whole_reference_notebook": True,
     "verified_note_assessment_credit": True, "context_limit": 24576},
)
# Captured from the unedited profile before adding the repair branch.
_LEGACY_PROTOCOL_SHA = (
    "cf9146ed9a09119941bff30b2b12481208f38a08b6eaddb33f85a0ff2931b336",
    "5772699a0f906cd358c5022f6102572d9df2e16a2486888a258a5894bfb22105",
    "8400fb4ab0f9aa66e25d586c220831d6a6c1eefbf1cdec0922860de06a127478",
    "5dc9ce849aab1cc69c96d734d767c584384f67e0fc8ab600c3fd0edc829487d5",
    "a5c80162a73fe5ce42e11e1a0c42b609e525d2702f6f88e004737b0cd87b39f5",
    "35b4e0717dcfb6bad9f848cf570361566194e9a93af6f9461836a997e2611043",
    "9636d913e5be7d8d95995e8731eba78764c9f97bf8093e8db49f5d085424e35e",
    "d3f3e583e350d6f9ff7f38a7929191197b1093193bcbb0aafe3b302e055ef391",
    "aaecc9a45c639ee0ae238a3d77ce8470ccbb97177c71522fbb07009857aa3e11",
    "08ac5fd77c64811db3fc77b7a34b5ea58fa4c948579aab77069d9eb660bc7f50",
    "8302f4f1ef69db18f5c2fe2607ec4973232d99f843c27d091ac650d5fcabd2b9",
    "394e73c32188acfaa5eaaa6980eec44e7c36fe6e4c633d313c18cd146cb4a956",
    "ed5c2e0686681acca1f68b8ae33d5a49b80cca5f23a26f83ae31cbcafbdb16a1",
    "58ffc03e768d9f257daedea024acbcb40f9adac3d4a948f6f34d0bdbc3c8b23d",
    "01812f47b6cb822d0bcb69a0d6166236adc983d94d2bb415c748d8cdedb9c16b",
    "6f8a0bfb74af68e602012c7d144ce31660af92704fd56262e9d003dca44a2b7f",
)
_LEGACY_PROMPT_SHA = (
    *("6aa99da45718564fa477df1404d183d26d0b00b2bd375b79d3356e657598c6a5",) * 5,
    *("9d42cba246676a013e5f9ae23477e77b2bf638b8b086bc02951e74fa0d4d2935",) * 3,
    "384ad8ee5792fbbc72d736e8c4346f99bc64f265a7848f5b7c4df70a03189e62",
    "a20e324699de936a2a84cff688b2634d909e7c1993ca24b42fefc4e98c7f7a5c",
    "d169bf1195cc5b6b4254c2e99a57334e8675d882e8885db2543f7eb4e6541445",
    "e05c4127d52af3382785af5b9fa33ffc49d98de47c65b5c40f739f7e492a1314",
    "b4ff34cae2194adf6a1ddf9868ecc40bc3b0370678621e5efc165b5f8ff5aed0",
    "1eba30ef78d65defbdfc255d535b8a14a63aadd0838b5cb24afc838ce3327f59",
    "af919da932f1fd2bdaf746b2f7e4be395172d012f26454e178fa4a7f236b24f7",
    "1eba30ef78d65defbdfc255d535b8a14a63aadd0838b5cb24afc838ce3327f59",
)


def protocol(**changes: Any) -> dict:
    return profile.build_protocol(**({"repair_notebook": True, "context_limit": 24576} | changes))


def reseal(value: dict) -> dict:
    value["sha256"] = digest({key: item for key, item in value.items() if key != "sha256"})
    return value


class RepairProfileTests(unittest.TestCase):
    def test_all_sixteen_legacy_protocol_prompt_digests_and_roundtrips_are_preserved(self) -> None:
        self.assertEqual(len(_LEGACY_FLAGS), 16)
        ids = set()
        for flags, protocol_sha, prompt_sha in zip(_LEGACY_FLAGS, _LEGACY_PROTOCOL_SHA, _LEGACY_PROMPT_SHA, strict=True):
            with self.subTest(flags=flags):
                current = profile.build_protocol(**flags)
                ids.add(current["protocol_id"])
                self.assertEqual(current["sha256"], protocol_sha)
                self.assertEqual(digest(profile.local_prompts(current)), prompt_sha)
                self.assertEqual(profile.restore_protocol(json.loads(json.dumps(current))), current)
                self.assertEqual(profile.build_protocol(**(flags | {"repair_notebook": False})), current)
        self.assertEqual(len(ids), 16)

    def test_new_identity_keeps_model_images_sampling_seed_budget_and_fixed_rollout(self) -> None:
        old, current = profile.build_protocol(), protocol()
        self.assertEqual(current["protocol_id"], "local-nf4-verified-repair-notebook-v1")
        self.assertEqual(current["profile"], profile.PROFILE)
        for key in ("initial_checkpoint", "quantization", "model", "image", "optimizer", "engine", "seeds",
                    "batch", "local_budget", "loss", "reward", "local_authorization", "actor_mode_policy",
                    "source_observation_instruction", "route_rule", "prompts_sha256", "parent_protocol_sha256"):
            self.assertEqual(current[key], old[key], key)
        self.assertEqual(current["rollout"]["sampler"], old["rollout"]["sampler"])
        self.assertEqual(current["context_limit"], 24576)
        self.assertEqual(current["rollout"]["retry_contract"], repair_contract.VERSION)
        self.assertEqual(current["rollout"]["n"], {1: 4, 2: 2, 3: 2})
        self.assertEqual(current["rollout"]["branch"], {1: 1, 2: 1})
        self.assertEqual(current["rollout"]["max_new_tokens"], {1: 1024, 2: 1024, 3: 1024})
        self.assertEqual(current["rollout"]["diag"], {"enabled_in": [], "n": 4, "max_new_tokens": 8192, "class_cap": None})
        self.assertIs(current["rollout"]["final_notebook"], False)
        self.assertFalse(any(key.startswith("notebook_") for key in current))
        self.assertFalse(any("ready" in key for key in current))
        cfg = RolloutConfig.from_protocol(current, "ALG-NB3")
        self.assertEqual((cfg.retry_contract, cfg.master_seed, cfg.diag, cfg.n_diag, cfg.diag_max_new_tokens),
                         (repair_contract.VERSION, 3407, False, 4, 8192))
        with self.assertRaises(ValueError):
            RolloutConfig.from_protocol(current, "ALG-T1")

    def test_repair_selection_is_exclusive_and_requires_explicit_integer_24k(self) -> None:
        value: Any
        legacy = ("verified_note_atoms", "verified_note_components", "verified_note_fact_credit", "verified_note_synthesis",
            "verified_note_synthesis_credit", "verified_note_assessment_credit", "partial_notebook", "referenced_notebook",
            "paired_notebook", "pair_action_notebook", "pair_draft_synthesis", "whole_reference_notebook", "whole_reference_pages")
        for key in legacy:
            for value in (True, 0, None, "false"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    protocol(**{key: value})
        for value in (0, 1, None, "true"):
            with self.subTest(repair=value), self.assertRaises(ValueError):
                protocol(repair_notebook=value)
        for value in (16384, 8192, 24576.0, True, "24576"):
            with self.subTest(context=value), self.assertRaises(ValueError):
                protocol(context_limit=value)
        with self.assertRaises(ValueError):
            profile.build_protocol(repair_notebook=True)

    def test_objective_has_explicit_masks_components_and_separate_f_s_requirements(self) -> None:
        objective = profile.repair_objective_contract()
        self.assertEqual(objective["objective_version"], "verified-native-repair-token-credit-v1")
        self.assertEqual(objective["component_weights"], {"answer": 1.0, "online_repair": .0625, "final_note": .0625,
            "legacy_strict": 0.0, "legacy_fact": 0.0, "legacy_assessment": 0.0})
        self.assertEqual(objective["answer_mask"], {"T1": "full_original_completion", "T2_T3": "revised_solution_body_only", "DIAG": "zero"})
        self.assertEqual(objective["group_mean"], "original_n_before_mis_no_recenter")
        self.assertIs(objective["lesson_pg"], False)
        self.assertEqual(objective["credit_version"], "repair-group-credit-v1")
        self.assertEqual(objective["preparation_version"], "repair-tree-credit-v1")
        self.assertEqual(objective["loss_version"], "separate-answer-online-final-kl-v1")
        self.assertEqual(objective["vjp_version"], "native-corrected-signed-and-positive-v1")
        self.assertEqual(objective["F_requires"], "retained_policy_gradient_not_accuracy_or_complete_note")
        self.assertEqual(objective["S_requires"], "positive_verified_corrected_tokens_and_nonzero_signed_and_positive_aggregate_vjp")
        self.assertEqual(protocol()["repair_objective_sha256"], digest(objective))
        objective["component_weights"]["online_repair"] = 9
        self.assertEqual(protocol()["repair_objective"]["component_weights"]["online_repair"], .0625)

    def test_existing_grammar_native_verifier_retry_and_final_contracts_are_bound(self) -> None:
        current = protocol()
        expected = {"repair_retry_contract": repair_contract.VERSION, "repair_origin_contract": repair_contract.ORIGIN_VERSION,
            "repair_final_contract": repair_contract.FINAL_VERSION, "verifier_version": VERIFIER_VERSION,
            "repair_native_audit_version": notebook_synthesis.NATIVE_SAMPLE_AUDIT_VERSION,
            "repair_online_audit_version": repair_credit.AUDIT_VERSION, "repair_final_audit_version": repair_credit.FINAL_AUDIT_VERSION,
            "solution_trace_contract": reasoning_trace.VERSION, "solution_prompt_sha256": digest(native_solution_prompt_contract()),
            "repair_retry_prompt_sha256": _hash(repair_retry_prompt_contract()),
            "repair_notebook_version": repair_notebook.VERSION, "repair_notebook_contract": repair_notebook.contract(),
            "repair_notebook_contract_sha256": _hash(repair_notebook.contract()), "repair_route_rule": profile.STRICT_ROUTE_RULE}
        for key, value in expected.items():
            self.assertEqual(current[key], value, key)

    def test_local_prompts_have_only_base_native_source_and_three_repair_keys(self) -> None:
        current = protocol()
        prompts = profile.local_prompts(current)
        base = load_prompts(profile.HERE / "prompts_v1.yaml")
        extra = {"solution_trace_contract", "source_observation_instruction", "repair_notebook_version",
                 "repair_notebook_contract", "repair_notebook_contract_sha256"}
        self.assertEqual(set(prompts), set(base) | extra)
        for key, value in base.items():
            self.assertEqual(prompts[key], value, key)
        self.assertFalse(any(key.startswith("notebook_") or key == "rewarded_notebook" for key in prompts))
        repair_notebook._prompts(prompts)
        self.assertEqual(prompts, profile.local_prompts(profile.restore_protocol(json.loads(json.dumps(current)))))

    def test_exact_protocol_and_prompt_roundtrip_reject_resealed_metadata_and_numeric_types(self) -> None:
        original = protocol()
        self.assertEqual(original, protocol())
        self.assertEqual(original, profile.restore_protocol(json.loads(json.dumps(original))))
        self.assertEqual(original["sha256"], digest({key: value for key, value in original.items() if key != "sha256"}))
        mutations = [lambda row: row.__setitem__("context_limit", 24576.0),
            lambda row: row["rollout"]["n"].__setitem__(1, 4.0),
            lambda row: row["rollout"]["diag"].__setitem__("n", True),
            lambda row: row["repair_objective"]["component_weights"].__setitem__("online_repair", .125),
            lambda row: row["repair_objective"]["component_weights"].__setitem__("legacy_strict", 1.),
            lambda row: row["repair_objective"]["answer_mask"].__setitem__("DIAG", "full"),
            lambda row: row["repair_objective"].__setitem__("S_requires", "full_note_accuracy"),
            lambda row: row["repair_objective"].__setitem__("loss_version", "different"),
            lambda row: row["loss"]["kl"].__setitem__("beta", .02),
            lambda row: row["image"].__setitem__("max_pixels", 1),
            lambda row: row.__setitem__("notebook_objective", {"objective_version": "legacy"}),
            lambda row: row.__setitem__("notebook_observation_policy", "legacy-reread"),
            lambda row: row.__setitem__("repair_notebook_version", "wrong")]
        for mutate in mutations:
            changed = copy.deepcopy(original)
            mutate(changed)
            reseal(changed)
            with self.subTest(mutation=mutate):
                for check in (profile.restore_protocol, profile.local_prompts):
                    with self.assertRaises(ValueError):
                        check(changed)

    def test_every_bound_repair_field_and_legacy_identity_mixing_fail_closed(self) -> None:
        original = protocol()
        for key in (key for key in original if key.startswith("repair_") or key in ("solution_trace_contract", "solution_prompt_sha256")):
            for missing in (False, True):
                changed = copy.deepcopy(original)
                if missing:
                    del changed[key]
                else:
                    changed[key] = "different"
                reseal(changed)
                with self.subTest(key=key, missing=missing), self.assertRaises(ValueError):
                    profile.local_prompts(changed)
        for old in (profile.build_protocol(), profile.build_protocol(verified_note_fact_credit=True)):
            changed = reseal(old | {"repair_notebook_version": repair_notebook.VERSION})
            with self.assertRaises(ValueError):
                profile.local_prompts(changed)
        changed = reseal(original | {"protocol_id": profile.PROTOCOL_ID})
        with self.assertRaises(ValueError):
            profile.restore_protocol(changed)

    def test_current_producer_template_changes_cannot_restore_a_frozen_old_binding(self) -> None:
        original = protocol()
        with patch("fine_tuning.notebook_rl.prompts.REPAIR_RETRY", REPAIR_RETRY + "\nchanged"):
            self.assertNotEqual(protocol()["repair_retry_prompt_sha256"], original["repair_retry_prompt_sha256"])
            with self.assertRaises(ValueError):
                profile.restore_protocol(original)
        with patch.object(repair_notebook, "contract", return_value=repair_notebook.contract() | {"max_calls": 2}):
            with self.assertRaises(ValueError):
                profile.local_prompts(original)
        with patch("fine_tuning.notebook_rl.prompts.REPAIR_MALFORMED_FEEDBACK", "different grammar feedback"):
            self.assertNotEqual(protocol()["repair_retry_prompt_sha256"], original["repair_retry_prompt_sha256"])
            with self.assertRaises(ValueError):
                profile.restore_protocol(original)

    def test_old_collector_and_feedback_binding_are_not_automatically_certified(self) -> None:
        original = protocol()
        self.assertEqual(original["repair_notebook_contract"]["collector_version"], "repair-retry-collector-v3")
        self.assertEqual(original["repair_notebook_contract"]["retry_feedback_version"], "repair-retry-feedback-v3")
        changed = copy.deepcopy(original)
        contract = changed["repair_notebook_contract"]
        for key in ("collector_version", "retry_selection_policy", "retry_feedback_version", "retry_prompt_contract_sha256"):
            del contract[key]
        changed["repair_notebook_contract_sha256"] = _hash(contract)
        changed["repair_retry_prompt_sha256"] = _hash({"response_contract": repair_contract.VERSION, "template": REPAIR_RETRY})
        reseal(changed)
        with self.assertRaises(ValueError):
            profile.restore_protocol(changed)

    def test_new_profile_does_not_import_training_helpers_or_initialize_cuda(self) -> None:
        code = ("import sys,torch; from fine_tuning.notebook_rl.local_profile import build_protocol; "
                "build_protocol(repair_notebook=True,context_limit=24576); "
                "assert 'fine_tuning.notebook_rl.repair_training' not in sys.modules; "
                "assert 'fine_tuning.notebook_rl.repair_loss' not in sys.modules; "
                "assert not torch.cuda.is_initialized()")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
