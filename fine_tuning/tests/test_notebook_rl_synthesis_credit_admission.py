"""CPU-authored partial-credit admission evidence; never pretrained model success."""
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.notebook_rl import local_profile, local_run, local_train, quantized_cache, updates
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash
from fine_tuning.notebook_rl.schema import Lineage, to_json
from fine_tuning.tests import test_notebook_rl_synthesis_integration as integration


def partial_fixture(*, facts_only=False):
    fixture = integration.SynthesisIntegrationTests()
    fixture.setUp()
    fixture.protocol = local_profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
    fixture.prompts = local_profile.local_prompts(fixture.protocol)
    fixture.cfg = RolloutConfig.from_protocol(fixture.protocol, "ALG-NB3", master_seed=3407)
    note = copy.deepcopy(fixture.note)
    note["attempts"] = []
    if facts_only:
        note["lessons"] = []
    else:
        note["corrected_facts"] = {}
    fixture.append([[json.dumps(note, separators=(",", ":")), "{}", "{}", "{}"]] * 3)
    updates.bind_notebook_objective(fixture.tree, fixture.actor, fixture.verifier,
                                   fixture.protocol["notebook_objective"])
    return fixture


def output_fixture(*, mixed=False, uniform=False, zero=False):
    from fine_tuning.tests import test_notebook_rl_synthesis_output as output_tests

    fixture = integration.SynthesisIntegrationTests()
    fixture.setUp()
    fixture.protocol = local_profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
    fixture.prompts = local_profile.local_prompts(fixture.protocol)
    fixture.cfg = RolloutConfig.from_protocol(fixture.protocol, "ALG-NB3", master_seed=3407)
    samples = integration.episode.solution_samples(fixture.tree)
    samples[0].text = samples[0].text.rsplit("\nFinal answer:", 1)[0]
    for index, sample in enumerate(samples):
        parent = samples[0] if index in (2, 3) else samples[2] if index in (4, 5) else None
        sample.prompt_token_ids = ([1, 2] if parent is None else
            parent.prompt_token_ids + parent.completion_token_ids + [900 + index // 2])
        sample.completion_token_ids = fixture.engine.tokenizer.encode(sample.text) + [2]
        sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
        sample.result = fixture.verifier.grade(fixture.actor.problem_id, sample.text, sample.finish_reason, fixture.actor.mode)
        sample.reward = sample.result.reward_answer
    audit = integration.synthesis.audit_episode(fixture.tree, fixture.actor, fixture.verifier,
                                                contract=integration.synthesis.BOUNDED_VERSION)
    note = integration.note_for(fixture.tree, audit, fixture.item)
    lesson = output_tests.output_lesson(samples, earlier=1, later=3)
    note["lessons"] = [*note["lessons"], lesson] if mixed else [lesson]
    note["attempts"], note["corrected_facts"] = [], {}
    text = json.dumps(note, separators=(",", ":"))
    replies = ["{}"] * 4 if zero else [text] * 4 if uniform else [text, "{}", "{}", "{}"]
    fixture.append([replies] * 3)
    updates.bind_notebook_objective(fixture.tree, fixture.actor, fixture.verifier,
                                   fixture.protocol["notebook_objective"])
    return fixture


class SynthesisCreditAdmissionTests(unittest.TestCase):
    def fixture(self, *, facts_only=False, output=False, mixed=False):
        fixture = output_fixture(mixed=mixed) if output else partial_fixture(facts_only=facts_only)
        self.addCleanup(fixture.doCleanups)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "raw-0000.json").write_text(json.dumps({"kind": "TREE", "raw": json.loads(to_json(fixture.tree))}))
        update = fixture.step(fixture.optimizer_fixture())
        scores = updates.notebook_objective_scores(fixture.tree, fixture.protocol["notebook_objective"])
        result = {"notebook_signal_groups": sum(g.kind == "DIAG" and g.usable and updates.group_has_signal(g, scores)
                                                for g in fixture.tree.groups),
                  "notebook_gradient_contribution_verified": any(row["changed_groups"] and row["reward_gradient_isolated"]
                                                                 for row in update["notebook_backward"])}
        return fixture, root, update, result

    def gate(self, fixture, root, update, result):
        return local_train.fact_credit_learning_gate(root, result, update, fixture.protocol,
            actors={fixture.actor.problem_id: fixture.actor}, verifier=fixture.verifier,
            tokenizer=fixture.engine.tokenizer)

    def test_profile_is_explicit_24k_and_preserves_strict_generation(self):
        from fine_tuning.notebook_rl import synthesis_credit

        strict = local_profile.build_protocol(verified_note_synthesis=True, context_limit=24576)
        current = local_profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
        self.assertEqual(current["protocol_id"], local_profile.SYNTHESIS_CREDIT_PROTOCOL_ID)
        self.assertEqual(current["notebook_objective"], synthesis_credit.contract())
        self.assertEqual(local_profile.restore_protocol(json.loads(json.dumps(current))), current)
        self.assertEqual(local_profile.local_prompts(strict), local_profile.local_prompts(current))
        self.assertEqual({k: v for k, v in current.items() if k not in ("protocol_id", "sha256", "notebook_objective")},
                         {k: v for k, v in strict.items() if k not in ("protocol_id", "sha256")})
        self.assertEqual(current["rollout"]["diag"]["max_new_tokens"], 8192)
        for changes in ({"verified_note_synthesis": True}, {"verified_note_fact_credit": True},
                        {"verified_note_atoms": True}, {"verified_note_components": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                local_profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576, **changes)
        with self.assertRaises(ValueError):
            local_profile.build_protocol(verified_note_synthesis_credit=True)
        forged = current | {"notebook_reward": strict["notebook_reward"] + "-forged"}
        forged["sha256"] = _hash({k: v for k, v in forged.items() if k != "sha256"})
        with self.assertRaises(ValueError):
            local_profile.restore_protocol(forged)
        self.assertEqual(local_train.execution_admission(
            {"admission_policy": local_train.SYNTHESIS_CREDIT_ADMISSION}, current), local_train.SYNTHESIS_CREDIT_ADMISSION)
        with self.assertRaises(ValueError):
            local_train.execution_admission({"admission_policy": local_train.SYNTHESIS_CREDIT_ADMISSION}, strict)
        self.assertTrue(current["protocol_id"].endswith("-v31"))
        for version in ("v29", "v30"):
            old = copy.deepcopy(current)
            old["protocol_id"] = old["protocol_id"].removesuffix("v31") + version
            old["sha256"] = _hash({k: v for k, v in old.items() if k != "sha256"})
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "unknown saved protocol"):
                local_profile.restore_protocol(old)
        old = copy.deepcopy(current)
        old["notebook_objective"]["objective_version"] = "verified-synthesis-unit-token-credit-v1"
        old["sha256"] = _hash({k: v for k, v in old.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "saved protocol changed"):
            local_profile.restore_protocol(old)
        with self.assertRaises(ValueError):
            local_train.execution_admission({"admission_policy": "verified-synthesis-credit-exploration-v1"}, current)

    def test_cli_prepare_minimal_and_cache_bind_the_new_flag_before_model_work(self):
        flag = ["--verified-note-synthesis-credit", "--context-limit", "24576"]
        prepare = ["prepare", "--previous-request", "/private/request", "--selection", "/private/selection",
                   "--output", "/private/new-data"]
        minimal = ["minimal", "--preparation", "/private/preparation", "--model-path", "/private/model",
                   "--output", "/private/new-minimal"]
        with patch("fine_tuning.notebook_rl.local_data.prepare_local_data", return_value={"status": "LOCAL_DATA_REVALIDATED"}) as called:
            self.assertEqual(local_run.main(prepare + flag), 0)
            self.assertEqual(called.call_args.args[2]["protocol_id"], local_profile.SYNTHESIS_CREDIT_PROTOCOL_ID)
        with patch.object(local_run, "capture") as called:
            self.assertEqual(local_run.main(minimal + flag), 0)
            self.assertTrue(called.call_args.kwargs["verified_note_synthesis_credit"])
        for arguments in (prepare, minimal):
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                local_run.main(arguments + ["--verified-note-synthesis-credit"])
        for stage in ("save", "check"):
            with tempfile.TemporaryDirectory() as directory, patch.object(quantized_cache, "build_protocol",
                    side_effect=ValueError("CPU protocol boundary")) as called, \
                    patch("fine_tuning.notebook_rl.policy.enable_deterministic_execution"), \
                    patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.max_memory_allocated", return_value=0), \
                    patch("torch.cuda.max_memory_reserved", return_value=0), patch("traceback.print_exc"):
                self.assertEqual(quantized_cache.main([stage, "--cache", "/private/cache", "--parent", "/private/parent",
                    "--model-path", "/private/model", "--output", str(Path(directory) / "out"), *flag]), 1)
                self.assertTrue(called.call_args.kwargs["verified_note_synthesis_credit"])

    def test_strict_zero_relationship_credit_passes_only_separate_partial_admission(self):
        fixture, root, update, result = self.fixture()
        self.assertTrue(all(s.reward == 0 for g in fixture.tree.groups if g.kind == "DIAG" for s in g.samples))
        self.assertTrue(local_run.error_relationship_signal([fixture.tree], fixture.protocol["notebook_objective"], fixture.engine.tokenizer))
        self.assertEqual(self.gate(fixture, root, update, result), [root / "raw-0000.json"])
        self.assertEqual(self.gate(fixture, root, update, result | {"notebook_gradient_contribution_verified": False}),
                         [root / "raw-0000.json"])
        with self.assertRaises(ValueError):
            local_train.notebook_learning_gate(root, result, update, fixture.protocol)

    def test_facts_only_and_no_relationship_variance_cannot_satisfy_minimum_or_admission(self):
        fixture, root, update, result = self.fixture(facts_only=True)
        self.assertFalse(local_run.error_relationship_signal([fixture.tree], fixture.protocol["notebook_objective"], fixture.engine.tokenizer))
        with self.assertRaisesRegex(ValueError, "relationship-token"):
            self.gate(fixture, root, update, result)
        fixture.tree.notebook.pop("learning_objective")
        build = root / "capture-build"
        build.mkdir()
        (build / "manifest.yaml").write_text(json.dumps({"split_manifests": {"train": "fixture"}}))
        fixture.verifier.metadata = {"vault_manifest_sha256": "fixture"}
        returns = {
            "fine_tuning.notebook_rl.local_run.read_preparation": ({"build_root": str(build)}, {}, [fixture.actor]),
            "fine_tuning.notebook_rl.local_run.minimal_actors": [fixture.actor],
            "fine_tuning.notebook_rl.local_run._binding": {},
            "fine_tuning.notebook_rl.local_run._install_actor_guards": [],
            "fine_tuning.notebook_rl.local_run.load_images": fixture.images[fixture.actor.problem_id],
            "fine_tuning.notebook_rl.local_run.make_probes": [],
            "fine_tuning.notebook_rl.local_calibrate.verify_original": {},
            "fine_tuning.notebook_rl.quantized_policy.build_quantized_policy": SimpleNamespace(inventory={"fixture_only": True}),
            "fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine": fixture.engine,
            "fine_tuning.notebook_rl.checkpoint.make_optimizer": (object(), object()),
            "fine_tuning.notebook_rl.verifier_service.VerifierService": nullcontext(fixture.verifier),
            "fine_tuning.notebook_rl.trainer.synchronize": {},
            "fine_tuning.notebook_rl.rollout.build_tree": fixture.tree,
            "fine_tuning.notebook_rl.notebook_learning.append_rewarded_notebooks": None,
            "torch.cuda.reset_peak_memory_stats": None,
            "torch.cuda.max_memory_allocated": 0,
            "torch.cuda.max_memory_reserved": 0,
        }
        with ExitStack() as stack:
            for target, value in returns.items():
                stack.enter_context(patch(target, return_value=value))
            optimizer = stack.enter_context(patch("fine_tuning.notebook_rl.updates.optimizer_step"))
            captured = local_run.capture(root / "preparation.json", root / "unused-model", root / "capture",
                standalone=False, verified_note_synthesis_credit=True, context_limit=24576)
        self.assertEqual(captured["status"], "NO_ERROR_RELATIONSHIP_LEARNING_SIGNAL")
        self.assertEqual(captured["optimizer_updates"], 0)
        self.assertGreater(captured["notebook_signal_groups"], 0)
        self.assertFalse((root / "capture/update.json").exists())
        optimizer.assert_not_called()

    def test_kl_only_negative_forged_credit_and_mis_rejection_do_not_admit(self):
        fixture, root, update, result = self.fixture()
        for kind in ("kl", "zero", "negative", "hash", "token_count", "nonfinite", "vision", "mis", "credit"):
            changed = copy.deepcopy(update)
            positives = [row for row in changed["notebook_backward"] if row["relationship_gradient"].get("measured")]
            for row in positives:
                evidence = row["relationship_gradient"]
                if kind == "kl":
                    row["relationship_gradient"] = {"scope": "relationship_policy_loss_only", "measured": False}
                elif kind == "zero":
                    evidence.update(nonzero_groups=[], group_norms={"L": 0., "M": 0., "V": 0.})
                elif kind == "negative":
                    evidence["positive_tokens"] = 0
                elif kind == "hash":
                    evidence["credit_sha256"] = "0" * 64
                elif kind == "token_count":
                    evidence["token_count"] += 1
                elif kind == "nonfinite":
                    evidence["group_norms"]["L"] = float("nan")
                elif kind == "vision":
                    evidence["group_norms"]["V"] = 1.
                    evidence["nonzero_groups"] = [g for g in ("L", "M", "V") if evidence["group_norms"][g] > 0]
                elif kind == "mis":
                    changed["mean_log_ratio_by_sample"][row["sample_id"]] = 999.
                    changed["notebook_backward"].remove(row)
                else:
                    changed["synthesis_token_credit"][row["sample_id"]]["relationship_token_advantages"][0] += 1
            self.assertTrue(positives)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.gate(fixture, root, changed, result)

    def test_real_parent_gate_is_mandatory_before_any_partial_regrade(self):
        with patch.object(local_train, "parent_gate", side_effect=ValueError("closed real update and reload required")) as parent:
            with self.assertRaisesRegex(ValueError, "closed real"):
                local_train.training_parent_gate(Path("/private/minimal"), Path("/private/reload"),
                                                  admission=local_train.SYNTHESIS_CREDIT_ADMISSION)
            parent.assert_called_once_with(Path("/private/minimal"), Path("/private/reload"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = local_profile.build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
            for name, value in (("launch.json", {"protocol": json.loads(json.dumps(protocol))}),
                                ("result.json", {"status": "NO_ERROR_RELATIONSHIP_LEARNING_SIGNAL"}),
                                ("reload/result.json", {}), ("reload_expected.json", {}),
                                ("update.json", {}), ("parameter_changes.json", {})):
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "completed real minimal update and fresh reload"):
                local_train.training_parent_gate(root, root / "reload", admission=local_train.SYNTHESIS_CREDIT_ADMISSION)

    def test_verified_finalization_admits_only_typed_output_signal_with_wrong_answer_separate(self):
        fixture, root, update, result = self.fixture(output=True)
        self.assertTrue(local_run.error_relationship_signal([fixture.tree], fixture.protocol["notebook_objective"], fixture.engine.tokenizer))
        self.assertEqual(self.gate(fixture, root, update, result), [root / "raw-0000.json"])
        summary = local_run.relationship_learning_summary(update)
        self.assertEqual(summary["learned_signal_types"], ["OUTPUT_FINALIZATION"])
        self.assertTrue(summary["output_finalization_only_gradient_verified"])
        self.assertFalse(summary["native_relationship_only_gradient_verified"])
        self.assertEqual(summary["native_relationship_gradient_sample_ids"], [])
        self.assertFalse(summary["visual_improvement_verified"])
        self.assertFalse(summary["strict_notebook_correctness_verified"])
        positive = [entry for credit in update["synthesis_token_credit"].values() for entry in credit["relationships"]
                    if local_run._positive_error_relationship(entry)]
        self.assertTrue(positive)
        self.assertTrue(all(entry["credit_type"] == "OUTPUT_FINALIZATION" and entry["relation"] == "FINALIZED"
                            and entry["later_answer_correct"] is False for entry in positive))
        for wrong_type in (None, "NATIVE_ERROR_RELATIONSHIP", "VISUAL_IMPROVEMENT"):
            forged = copy.deepcopy(update)
            for row in forged["notebook_backward"]:
                if row["relationship_gradient"].get("measured") is not True:
                    continue
                credit = forged["synthesis_token_credit"][row["sample_id"]]
                for entry in credit["relationships"]:
                    if entry["relation"] == "FINALIZED":
                        if wrong_type is None:
                            entry.pop("credit_type")
                        else:
                            entry["credit_type"] = wrong_type
                row["synthesis_credit_sha256"] = row["relationship_gradient"]["credit_sha256"] = _hash(credit)
            with self.subTest(wrong_type=wrong_type), self.assertRaises(ValueError):
                self.gate(fixture, root, forged, result)

    def test_mixed_relationship_vjp_does_not_certify_output_only_and_zero_groups_do_not_force_signal(self):
        fixture, root, update, result = self.fixture(output=True, mixed=True)
        self.assertEqual(self.gate(fixture, root, update, result), [root / "raw-0000.json"])
        summary = local_run.relationship_learning_summary(update)
        self.assertEqual(summary["learned_signal_types"], ["COMBINED_RELATIONSHIP_POLICY_LOSS"])
        self.assertFalse(summary["output_finalization_only_gradient_verified"])
        self.assertTrue(summary["native_relationship_only_gradient_verified"])
        self.assertTrue(summary["native_relationship_gradient_sample_ids"])
        forged = copy.deepcopy(update)
        for row in forged["notebook_backward"]:
            gradient = row["relationship_gradient"]
            if gradient.get("measured"):
                gradient.update(effect_scope="OUTPUT_FINALIZATION_ONLY", credit_types=["OUTPUT_FINALIZATION"])
        with self.assertRaises(ValueError):
            self.gate(fixture, root, forged, result)
        for kwargs in ({"uniform": True}, {"zero": True}):
            other = output_fixture(**kwargs)
            self.addCleanup(other.doCleanups)
            with self.subTest(kwargs=kwargs):
                self.assertFalse(local_run.error_relationship_signal(
                    [other.tree], other.protocol["notebook_objective"], other.engine.tokenizer))

    def test_native_summary_requires_separate_finite_positive_gradient_evidence(self):
        _, _, update, _ = self.fixture()
        self.assertTrue(local_run.relationship_learning_summary(update)["native_relationship_only_gradient_verified"])
        for mutation in ("missing", "scope", "zero", "masked", "nonfinite", "binding"):
            changed = copy.deepcopy(update)
            for row in changed["notebook_backward"]:
                native = row.get("native_relationship_gradient", {})
                if mutation == "missing":
                    row.pop("native_relationship_gradient", None)
                elif native.get("measured"):
                    if mutation == "scope":
                        native["scope"] = "relationship_policy_loss_only"
                    elif mutation == "zero":
                        native["group_norms"] = dict.fromkeys(("L", "M", "V"), 0.)
                        native["nonzero_groups"] = []
                    elif mutation == "masked":
                        native["positive_tokens"] = 0
                    elif mutation == "nonfinite":
                        native["group_norms"]["L"] = float("nan")
                    else:
                        native["credit_sha256"] = "different credit"
            with self.subTest(mutation=mutation):
                summary = local_run.relationship_learning_summary(changed)
                self.assertFalse(summary["native_relationship_only_gradient_verified"])
                self.assertEqual(summary["native_relationship_gradient_sample_ids"], [])
                self.assertFalse(summary["strict_notebook_correctness_verified"])
                self.assertFalse(summary["visual_improvement_verified"])

    def test_native_evaluation_accepts_new_admission_without_claiming_strict_success(self):
        from fine_tuning.notebook_rl import checkpoint, local_eval, local_ladder_generation as generation
        from fine_tuning.tests import test_notebook_rl_local_eval as evaluation

        fixture = evaluation.NativeLadderRuntimeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.protocol.clear()
        fixture.protocol.update(json.loads(json.dumps(local_profile.build_protocol(
            verified_note_synthesis_credit=True, context_limit=24576))))
        fixture.launch["binding"]["protocol_sha256"] = fixture.protocol["sha256"]
        parent = {"launch": {"protocol": fixture.protocol}, "strict_notebook_correctness_verified": False,
                  "notebook_reward_learning_verified": False}
        fixture.contract.update(admission_policy=local_train.SYNTHESIS_CREDIT_ADMISSION, parent_gate=parent)
        fixture.launch["execution_contract_sha256"] = checkpoint.digest(fixture.contract)
        templates = generation.circuit_evaluation_prompts()
        contract = generation.protocol(templates, fixture.protocol["sha256"], context_limit=24576)
        fixture.gate["protocol_sha256"] = local_eval.digest(contract)
        fixture.generation["protocol"] = contract
        for row in fixture.generation["rows"]:
            row["context_limit"] = 24576
        with patch.object(local_train, "training_parent_gate", return_value=parent) as checked:
            result = local_eval.evaluate_ladder(fixture.capture, fixture.cohort, fixture.receipt, fixture.root / "v29",
                condition="FT", arm="notebook", reload_result=fixture.reload,
                minimal_capture=fixture.root / "minimal", minimal_reload=fixture.root / "minimal-reload",
                evaluation_prompt_policy=generation.CIRCUIT_PROMPT_POLICY)
        checked.assert_called_once_with(fixture.root / "minimal", fixture.root / "minimal-reload",
                                        admission=local_train.SYNTHESIS_CREDIT_ADMISSION)
        self.assertEqual(fixture.generate.call_args.args[3], templates)
        self.assertEqual(result["conditions"]["protocol"], contract)
        launch = json.loads((fixture.root / "v29/launch.json").read_text())
        self.assertFalse(launch["minimal_parent_gate"]["strict_notebook_correctness_verified"])

    def test_parent_uses_fresh_verifier_and_keeps_strict_success_false(self):
        from fine_tuning.notebook_rl import checkpoint
        from fine_tuning.notebook_rl.local_data import ACTOR_MODE_POLICY
        from fine_tuning.notebook_rl.vault import write_split
        from fine_tuning.notebook_rl.verifier_service import VerifierService

        fixture, root, update, result = self.fixture()
        actor = fixture.actor
        lineage = Lineage(actor.problem_id, actor.source, "fixture", "CC0", "PROGRAM", "fixture", "fixture",
                          actor.source, 3407, "fixture", "a" * 64, "train", ())
        build = root / "build"
        write_split(build / "train", [actor], [fixture.item], [lineage])
        def save(name, value):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            return {"path": str(path), "file_sha256": checkpoint.file_hash(path)}
        manifest = save("build/manifest.yaml", {"split_manifests": {"train": checkpoint.file_hash(build / "train/manifest.yaml")}})
        actors = save("actors.json", [json.loads(to_json(actor))])
        checked = save("checked.json", {"status": "VALIDATED_FOR_MEASUREMENT", "blockers": [],
            "measurement_allowed": True, "training_allowed": False,
            "input_binding": {"build_manifest_sha256": manifest["file_sha256"]}, "actors": [json.loads(to_json(actor))]})
        request = save("request.json", {"build_root": str(build), "index": {"path": str(root / "benchmark/index.json")}})
        prepared = {"format": "notebook-rl-local-data-v1", "status": "LOCAL_DATA_REVALIDATED",
            "local_protocol_sha256": fixture.protocol["sha256"], "build_root": str(build), "actors": actors,
            "execution_actors": actors, "actor_projection": {"policy": ACTOR_MODE_POLICY, "changed_ids": []},
            "data_request": request, "validated_data": checked}
        preparation = save("preparation.json", prepared)
        save("result.json", result)
        save("update.json", update)
        prerequisite = {"launch": {"protocol": json.loads(json.dumps(fixture.protocol)), "preparation": preparation["path"],
            "problem_ids": [actor.problem_id], "model_path": str(root / "unused-model"),
            "binding": {"data_sha256": checkpoint.digest(prepared), "protocol_sha256": fixture.protocol["sha256"],
                        "prompts_sha256": fixture.protocol["prompts_sha256"]}}, "files": [], "notebook_reward_learning_verified": False}
        services = []
        def service(*args, **kwargs):
            created = VerifierService(*args, **kwargs)
            services.append(created)
            return created
        with patch.object(local_train, "parent_gate", return_value=prerequisite), \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=fixture.engine.tokenizer), \
                patch("fine_tuning.notebook_rl.verifier_service.VerifierService", side_effect=service):
            gate = local_train.training_parent_gate(root, root / "reload", admission=local_train.SYNTHESIS_CREDIT_ADMISSION)
        self.assertTrue(gate["positive_relationship_credit_and_isolated_relationship_gradient_verified"])
        self.assertFalse(gate["strict_notebook_correctness_verified"])
        self.assertFalse(gate["notebook_reward_learning_verified"])
        self.assertTrue(services[0].metadata["python_guards_installed"])
        self.assertIsNotNone(services[0]._process.poll())


if __name__ == "__main__":
    unittest.main()
