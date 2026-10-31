"""Paged profile and admission boundaries; CPU fixtures are not model performance."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import local_profile, local_run, local_train, quantized_cache, updates
from fine_tuning.notebook_rl import whole_reference_notebook as reference, whole_reference_pages as pages
from fine_tuning.notebook_rl.checkpoint import digest
from fine_tuning.notebook_rl.schema import to_json


def protocol(**changes):
    return local_profile.build_protocol(**({"verified_note_synthesis_credit": True,
        "pair_draft_synthesis": True, "whole_reference_pages": True,
        "context_limit": 24576} | changes))


class PagesProfileTests(unittest.TestCase):
    def test_page_prompt_revision_cannot_restore_a_prior_protocol_as_current(self):
        current = protocol()
        self.assertEqual(current["protocol_id"], "local-nf4-whole-reference-pages-synthesis-credit-24k-v2")
        self.assertEqual(current["notebook_generation_policy"], "whole-history-reference-pages-v2")
        historical = copy.deepcopy(current)
        historical["protocol_id"] = "local-nf4-whole-reference-pages-synthesis-credit-24k-v1"
        historical["sha256"] = digest({key: value for key, value in historical.items() if key != "sha256"})
        with self.assertRaisesRegex(ValueError, "unknown saved protocol"):
            local_profile.restore_protocol(historical)

    def learning_fixture(self):
        from fine_tuning.tests.test_notebook_rl_pages_collection import PageCollectionTests

        case = PageCollectionTests()
        fixture = case.fixture()
        self.addCleanup(case.doCleanups)
        page_count = len(pages.plan([s for g in fixture.tree.groups for s in g.samples])["pages"])
        for replies in fixture.engine.replies[-page_count:]:
            payload = json.loads(replies[0])
            if "assessments" in payload:
                for row in payload["assessments"]:
                    row[3] = None
                for ordinal in (1, 3):
                    replies[ordinal] = json.dumps(payload)
        case.append(fixture)
        updates.bind_notebook_objective(fixture.tree, fixture.actor, fixture.verifier,
                                       fixture.protocol["notebook_objective"])
        return fixture

    def test_other_historical_protocol_hashes_are_unchanged(self):
        # Pins independently rechecked in the unchanged V49 package before paging.
        cases: list[tuple[dict[str, Any], str]] = [
            ({}, "cf9146ed9a09119941bff30b2b12481208f38a08b6eaddb33f85a0ff2931b336"),
            ({"verified_note_atoms": True}, "5772699a0f906cd358c5022f6102572d9df2e16a2486888a258a5894bfb22105"),
            ({"verified_note_components": True}, "8400fb4ab0f9aa66e25d586c220831d6a6c1eefbf1cdec0922860de06a127478"),
            ({"verified_note_fact_credit": True}, "5dc9ce849aab1cc69c96d734d767c584384f67e0fc8ab600c3fd0edc829487d5"),
            ({"verified_note_fact_credit": True, "context_limit": 24576},
             "a5c80162a73fe5ce42e11e1a0c42b609e525d2702f6f88e004737b0cd87b39f5"),
            ({"verified_note_synthesis": True}, "35b4e0717dcfb6bad9f848cf570361566194e9a93af6f9461836a997e2611043"),
            ({"verified_note_synthesis": True, "context_limit": 24576},
             "9636d913e5be7d8d95995e8731eba78764c9f97bf8093e8db49f5d085424e35e"),
            ({"verified_note_synthesis_credit": True, "context_limit": 24576},
             "d3f3e583e350d6f9ff7f38a7929191197b1093193bcbb0aafe3b302e055ef391"),
            ({"verified_note_synthesis_credit": True, "context_limit": 24576, "partial_notebook": True},
             "aaecc9a45c639ee0ae238a3d77ce8470ccbb97177c71522fbb07009857aa3e11"),
            ({"verified_note_synthesis_credit": True, "context_limit": 24576, "referenced_notebook": True},
             "08ac5fd77c64811db3fc77b7a34b5ea58fa4c948579aab77069d9eb660bc7f50"),
            ({"verified_note_synthesis_credit": True, "context_limit": 24576, "paired_notebook": True},
             "8302f4f1ef69db18f5c2fe2607ec4973232d99f843c27d091ac650d5fcabd2b9"),
            ({"verified_note_synthesis_credit": True, "context_limit": 24576, "pair_draft_synthesis": True},
             "ed5c2e0686681acca1f68b8ae33d5a49b80cca5f23a26f83ae31cbcafbdb16a1"),
            ({"verified_note_synthesis_credit": True, "context_limit": 24576, "pair_draft_synthesis": True,
              "whole_reference_notebook": True}, "58ffc03e768d9f257daedea024acbcb40f9adac3d4a948f6f34d0bdbc3c8b23d")]
        for flags, sha in cases:
            old = local_profile.build_protocol(**flags)
            self.assertEqual(old["sha256"], sha)
            self.assertEqual(local_profile.restore_protocol(json.loads(json.dumps(old))), old)

    def test_explicit_pages_profile_binds_public_prompt_budget_and_native_commit(self):
        actual = protocol()
        old = protocol(whole_reference_pages=False, whole_reference_notebook=True)
        changed = {key for key in actual if actual.get(key) != old.get(key)}
        self.assertEqual(changed, {"protocol_id", "sha256", "notebook_generation_policy",
            "notebook_page_contract", "notebook_page_contract_sha256", "notebook_prompt_sha256"})
        self.assertEqual(actual["protocol_id"], local_profile.WHOLE_REFERENCE_PAGES_PROTOCOL_ID)
        self.assertEqual(actual["notebook_generation_policy"], pages.VERSION)
        self.assertEqual(actual["notebook_page_contract"], pages.contract())
        self.assertEqual(actual["notebook_page_contract_sha256"], digest(pages.contract()))
        self.assertEqual(actual["notebook_reference_contract"], reference.contract())
        self.assertEqual(actual["notebook_update_requirement"], updates.NATIVE_RELATIONSHIP_REQUIREMENT)
        self.assertEqual((actual["rollout"]["diag"]["n"], actual["rollout"]["diag"]["max_new_tokens"],
                          actual["context_limit"], actual["notebook_max_rounds"]), (4, 8192, 24576, 1))
        self.assertEqual((actual["notebook_page_contract"]["header_tokens"],
                          actual["notebook_page_contract"]["assessment_tokens"]), (2048, 6144))
        self.assertEqual(local_profile.restore_protocol(json.loads(json.dumps(actual))), actual)
        prompts = local_profile.local_prompts(actual)
        self.assertEqual(prompts["rewarded_notebook"], pages.prompt_contract())
        self.assertEqual(actual["notebook_prompt_sha256"], digest(prompts["rewarded_notebook"]))
        for key in ("notebook_page_contract", "notebook_page_contract_sha256", "notebook_draft_contract",
                    "notebook_reference_contract", "notebook_generation_policy"):
            self.assertEqual(prompts[key], actual[key])
        self.assertIs(prompts["notebook_strict_feedback_repairs"], False)
        self.assertIn("complete-original-history-and-images-every-call", str(pages.contract()))
        self.assertIn("no-previous-page-outputs", str(pages.contract()))

    def test_pages_require_exclusive_explicit_synthesis_credit_and_24k(self):
        for value in (None, 0, 1, "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol(whole_reference_pages=cast(Any, value))
        for changes in ({"whole_reference_notebook": True}, {"pair_draft_synthesis": False},
                {"verified_note_synthesis_credit": False}, {"context_limit": 16384},
                {"partial_notebook": True}, {"referenced_notebook": True}, {"paired_notebook": True},
                {"pair_action_notebook": True}, {"verified_note_synthesis": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                protocol(**changes)
        self.assertEqual(local_train.execution_admission(
            {"admission_policy": local_train.SYNTHESIS_CREDIT_ADMISSION}, protocol()),
            local_train.SYNTHESIS_CREDIT_ADMISSION)
        with self.assertRaises(ValueError):
            local_run.minimum_execution_contract(local_run.VERIFIED_UPDATE_EXPLORATION, protocol())

    def test_resealed_page_contract_prompt_and_budget_tampering_are_rejected(self):
        changes = {"notebook_page_contract": {}, "notebook_page_contract_sha256": "0" * 64,
            "notebook_prompt_sha256": digest(reference.prompt_contract()), "notebook_generation_policy": reference.VERSION,
            "notebook_update_requirement": None, "notebook_max_rounds": True,
            "notebook_strict_feedback_repairs": 0, "notebook_reference_contract": {},
            "notebook_draft_contract": {}, "context_limit": 16384}
        for key, value in changes.items():
            for missing in (False, True):
                changed = copy.deepcopy(protocol())
                if missing:
                    del changed[key]
                else:
                    changed[key] = value
                changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
                for check in (local_profile.restore_protocol, local_profile.local_prompts):
                    with self.subTest(key=key, missing=missing, check=check.__name__), self.assertRaises(ValueError):
                        check(changed)
        forged = protocol()
        forged["rollout"]["diag"]["max_new_tokens"] = 6144
        forged["sha256"] = digest({k: v for k, v in forged.items() if k != "sha256"})
        with self.assertRaises(ValueError):
            local_profile.restore_protocol(forged)
        old = local_profile.build_protocol()
        for key in ("notebook_page_contract", "notebook_page_contract_sha256"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                local_profile.local_prompts(old | {key: protocol()[key]})

    def test_prepare_minimal_and_both_cache_stages_forward_explicit_pages(self):
        flags = ["--whole-reference-pages", "--pair-draft-synthesis",
                 "--verified-note-synthesis-credit", "--context-limit", "24576"]
        prepare = ["prepare", "--previous-request", "/unused/request", "--selection", "/unused/selection",
                   "--output", "/unused/new-data"]
        minimal = ["minimal", "--preparation", "/unused/preparation", "--model-path", "/unused/base",
                   "--output", "/unused/minimal"]
        with patch("fine_tuning.notebook_rl.local_data.prepare_local_data",
                   return_value={"status": "LOCAL_DATA_REVALIDATED"}) as called, patch("builtins.print"):
            self.assertEqual(local_run.main(prepare + flags), 0)
            self.assertEqual(called.call_args.args[2], protocol())
        with patch.object(local_run, "capture") as called:
            self.assertEqual(local_run.main(minimal + flags), 0)
            self.assertIs(called.call_args.kwargs["whole_reference_pages"], True)
            self.assertIs(called.call_args.kwargs["whole_reference_notebook"], False)
        for stage in ("save", "check"):
            with tempfile.TemporaryDirectory() as directory, patch.object(quantized_cache, "_sources", return_value={}), \
                    patch.object(quantized_cache, "build_protocol", side_effect=ValueError("CPU boundary")) as build, \
                    patch("fine_tuning.notebook_rl.policy.enable_deterministic_execution"), \
                    patch.multiple(torch.cuda, reset_peak_memory_stats=lambda: None,
                        max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0), \
                    patch.object(quantized_cache, "build_quantized_policy") as save, \
                    patch.object(quantized_cache, "load_quantized_policy") as load, \
                    patch("traceback.print_exc"), patch("builtins.print"):
                args = [stage, "--cache", "/unused/cache", "--model-path", "/unused/base", "--parent", "/unused/parent",
                        "--output", str(Path(directory) / "out")]
                self.assertEqual(quantized_cache.main(args + flags), 1)
                self.assertIs(build.call_args.kwargs["whole_reference_pages"], True)
                save.assert_not_called()
                load.assert_not_called()

    def test_invalid_cli_pages_fail_before_any_io(self):
        commands = [(local_run.main, ["prepare", "--previous-request", "/unused/request",
            "--selection", "/unused/selection", "--output", "/unused/data"]),
            (local_run.main, ["minimal", "--preparation", "/unused/preparation",
                "--model-path", "/unused/base", "--output", "/unused/capture"]),
            (quantized_cache.main, ["save", "--model-path", "/unused/base", "--cache", "/unused/cache",
                "--output", "/unused/cache-save"])]
        invalid = [["--whole-reference-pages"],
            ["--whole-reference-pages", "--pair-draft-synthesis"],
            ["--whole-reference-pages", "--pair-draft-synthesis", "--verified-note-synthesis-credit"],
            ["--whole-reference-pages", "--whole-reference-notebook", "--pair-draft-synthesis",
             "--verified-note-synthesis-credit", "--context-limit", "24576"]]
        for main, args in commands:
            for flags in invalid:
                with self.subTest(main=main.__module__, flags=flags), patch("sys.stderr"), \
                        patch.object(Path, "mkdir") as mkdir, self.assertRaises(SystemExit):
                    main(args + flags)
                mkdir.assert_not_called()

    def test_native_admission_uses_shared_bundle_mis_and_rejects_forged_membership(self):
        from fine_tuning.notebook_rl import synthesis_credit

        fixture = self.learning_fixture()
        optimizer = fixture.optimizer_fixture()
        scores = updates.notebook_objective_scores(fixture.tree, fixture.protocol["notebook_objective"])
        physical_signals = sum(g.kind == "DIAG" and g.usable and updates.group_has_signal(g, scores)
                               for g in fixture.tree.groups)
        self.assertGreater(physical_signals, 1)
        self.assertEqual(updates.notebook_signal_groups(fixture.tree, scores), 1)
        credits = {sample_id: credit for group in fixture.tree.groups if group.kind == "DIAG"
                   for sample_id, credit in synthesis_credit.group_credit(group.samples, scores,
                                                                          fixture.engine.tokenizer).items()}
        sid = next(sample_id for sample_id, credit in credits.items()
                   if any(local_run._positive_error_relationship(entry, native_only=True)
                          for entry in credit["relationships"]))
        threshold = fixture.protocol["loss"]["mismatch"]["delta_seq"]

        def shifted(bundle, inputs, completion_len, *, chunk):
            original = optimizer[4](bundle, inputs, completion_len, chunk=chunk)
            return original + (2 * threshold if inputs["sample"].sample_id == sid else 0.)

        update = fixture.step((*optimizer[:4], shifted, optimizer[5]))
        self.assertEqual(update["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertGreater(abs(update["mean_log_ratio_by_sample"][sid]), threshold)
        self.assertLess(abs(update["effective_mean_log_ratio_by_sample"][sid]), threshold)
        self.assertTrue(local_run.native_relationship_update_evidence(update, fixture.protocol)[
            "native_relationship_only_gradient_verified"])
        result = {"notebook_signal_groups": 1, "notebook_gradient_contribution_verified": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "raw-0000.json"
            path.write_text(json.dumps({"kind": "TREE", "raw": json.loads(to_json(fixture.tree))}))
            kwargs = {"actors": {fixture.actor.problem_id: fixture.actor}, "verifier": fixture.verifier,
                      "tokenizer": fixture.engine.tokenizer}
            self.assertEqual(local_train.fact_credit_learning_gate(root, result, update, fixture.protocol, **kwargs), [path])
            for mutation in ("missing", "missing_effective", "physical", "bundle", "counts"):
                forged = copy.deepcopy(update)
                if mutation == "missing":
                    forged.pop("bundle_mis")
                elif mutation == "missing_effective":
                    forged.pop("effective_mean_log_ratio_by_sample")
                elif mutation == "physical":
                    forged["effective_mean_log_ratio_by_sample"][sid] = update["mean_log_ratio_by_sample"][sid]
                elif mutation == "bundle":
                    forged["bundle_mis"]["bundles"][0]["sample_ids"].reverse()
                    forged["bundle_mis"]["sha256"] = digest({key: value for key, value in forged["bundle_mis"].items()
                                                           if key != "sha256"})
                else:
                    forged["action_token_counts_by_sample"][sid] += 1
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    local_train.fact_credit_learning_gate(root, result, forged, fixture.protocol, **kwargs)
            with self.assertRaises(ValueError):
                local_train.fact_credit_learning_gate(root, result | {"notebook_signal_groups": physical_signals},
                                                     update, fixture.protocol, **kwargs)


if __name__ == "__main__":
    unittest.main()
