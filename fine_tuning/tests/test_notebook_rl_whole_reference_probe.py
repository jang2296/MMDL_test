"""Scripted whole-reference CPU plumbing, never actual model-learning evidence."""
import copy
import json
import os
import tempfile
import time
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.notebook_rl import episode_notebook as episode, fact_context_probe as runner
from fine_tuning.notebook_rl import fresh_observation, notebook_pair_drafts as drafts
from fine_tuning.notebook_rl import notebook_whole_reference_probe as probe
from fine_tuning.notebook_rl import whole_reference_notebook as reference
from fine_tuning.notebook_rl.engines import Completion
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_rl_whole_synthesis_probe as original


class WholeReferenceProbeTests(unittest.TestCase):
    def fixture(self):
        fixture, actors, trees, good = original.WholeSynthesisProbeTests.fixture(self)
        originals = episode.solution_samples(fixture.tree)
        inventory = reference.public_inventory(originals)
        native = json.loads(good)
        audit = probe.synthesis.audit_episode(fixture.tree, fixture.actor, fixture.verifier,
                                             contract=probe.synthesis.BOUNDED_VERSION)
        refs = inventory["references"]

        def find(endpoint):
            return next(ref for ref in inventory["assessment_refs"] if all(
                refs[ref][key] == endpoint[key] for key in ("attempt", "kind", "rule", "locus")))

        judgments = []
        for ref in inventory["assessment_refs"]:
            endpoint = refs[ref]
            checked = audit["attempts"][endpoint["attempt"] - 1]
            key = tuple(endpoint[name] for name in ("kind", "rule", "locus"))
            issue = any(tuple(row[name] for name in ("kind", "rule", "locus")) == key for row in checked["issues"])
            unknown = next((row for row in checked["unverified"]
                if tuple(row[name] for name in ("kind", "rule", "locus")) == key), None)
            blocked = [find({"attempt": endpoint["attempt"], "kind": "TRACE", **dependency})
                       for dependency in unknown["blocked_by"]] if unknown else []
            lesson = next((row for row in native["lessons"] if find(row["earlier"]) == ref), None)
            decision = None if lesson is None else [lesson["relation"],
                inventory["absence_refs"][str(endpoint["attempt"])] if lesson["later"]["kind"] == "ABSENCE"
                else find(lesson["later"])]
            judgments.append([ref, ("I" if issue else "") + ("U" if unknown else "") or "N", blocked, decision])
        compact = {"corrected_facts": native["corrected_facts"],
            "attempts": [[row["attempt"], row["outcome"], row["answer"]] for row in native["attempts"]],
            "assessments": judgments, "success_attempt": native["success_attempt"],
            "corrected_answer": native["corrected_answer"]}
        return fixture, actors, trees, json.dumps(compact)

    def collect(self, fixture, actors, trees, replies, *, deadline=None, record_completion=None):
        fixture.engine.replies = [replies]
        journal = []
        rows, skipped = probe.collect(actors, trees, fixture.images, fixture.prompts, fixture.engine,
            fixture.verifier, lambda *row: journal.append(row),
            time.monotonic() + 60 if deadline is None else deadline,
            record_completion=record_completion)
        return rows, skipped, journal

    def test_completed_sidecars_preserve_raw_group_and_do_not_replace_group_scoring(self):
        fixture, actors, trees, good = self.fixture()
        sample = fixture.engine.sample
        captured: list[dict[str, object]] = []

        def observed(requests, *, on_completion):
            outputs = sample(requests)
            for index, value in enumerate(outputs[0].completions):
                on_completion(requests[0].request_id, index, requests[0].seeds[index], value)
            return outputs

        with patch.object(fixture.engine, "sample", side_effect=observed):
            rows, _, journal = self.collect(fixture, actors, trees, [good, "invalid", good, good],
                                           record_completion=captured.append)
        self.assertEqual([event for event, _ in journal], ["CALL_STARTED", "CALL_GENERATED", "CALL_COMPLETED"])
        generated = journal[1][1]
        self.assertEqual([row["completion"] for row in captured], generated["completions"])
        self.assertEqual([row["index"] for row in captured], list(range(4)))
        self.assertEqual([row["seed"] for row in captured], journal[0][1]["seeds"])
        for row in captured:
            self.assertEqual(row["prompt_sha256"], journal[0][1]["changed_prompt_sha256"])
            self.assertEqual(row["messages_sha256"], journal[0][1]["changed_messages_sha256"])
            self.assertIs(row["group_complete"], False)
            self.assertIs(row["eligible_for_rl_loss"], False)
        self.assertEqual(len(rows[0]["diagnostic_group_credit"]), 4)

    def test_completed_sidecar_survives_later_failure_without_group_score(self):
        fixture, actors, trees, good = self.fixture()
        fixture.engine.replies = [[good] * 4]
        sample = fixture.engine.sample
        journal = []

        def interrupted(requests, *, on_completion):
            output = sample(requests)[0]
            on_completion(requests[0].request_id, 0, requests[0].seeds[0], output.completions[0])
            raise TimeoutError("later completion interrupted")

        with tempfile.TemporaryDirectory() as directory, patch.object(
                fixture.engine, "sample", side_effect=interrupted), patch.object(probe.credit, "check") as grade:
            path = Path(directory) / "completion-0000.json"
            with self.assertRaisesRegex(TimeoutError, "later completion"):
                probe.collect(actors, trees, fixture.images, fixture.prompts, fixture.engine,
                    fixture.verifier, lambda *row: journal.append(row), time.monotonic() + 60,
                    record_completion=lambda row: runner.save(path, row))
            self.assertEqual(runner.read(path)["completion"]["text"], good)
            self.assertIs(runner.read(path)["eligible_for_rl_loss"], False)
            grade.assert_not_called()
        self.assertEqual([event for event, _ in journal], ["CALL_STARTED"])

    def test_sidecar_identity_decode_and_final_group_parity_are_required(self):
        for mutation in ("index", "seed", "decode", "missing", "returned"):
            with self.subTest(mutation=mutation):
                fixture, actors, trees, good = self.fixture()
                fixture.engine.replies = [[good] * 4]
                sample = fixture.engine.sample
                journal = []
                captured: list[dict[str, object]] = []

                def changed(requests, *, on_completion):
                    outputs = sample(requests)
                    for index, value in enumerate(outputs[0].completions):
                        if mutation == "missing":
                            continue
                        if mutation == "decode":
                            value = Completion(value.token_ids, value.logprobs, value.finish_reason, value.text + "x")
                        on_completion(requests[0].request_id, index + int(mutation == "index"),
                                      requests[0].seeds[index] + int(mutation == "seed"), value)
                    if mutation == "returned":
                        value = outputs[0].completions[0]
                        outputs[0].completions[0] = Completion(value.token_ids, value.logprobs, value.finish_reason, "changed")
                    return outputs

                with patch.object(fixture.engine, "sample", side_effect=changed), patch.object(
                        probe.credit, "check") as grade, self.assertRaises(ValueError):
                    probe.collect(actors, trees, fixture.images, fixture.prompts, fixture.engine,
                        fixture.verifier, lambda *row: journal.append(row), time.monotonic() + 60,
                        record_completion=captured.append)
                grade.assert_not_called()
                self.assertEqual([event for event, _ in journal], ["CALL_STARTED"])

    def test_explicit_contract_keeps_runtime_and_original_training_prompt_unchanged(self):
        fixture, _, _, _ = self.fixture()
        before = copy.deepcopy(fixture.protocol)
        original_prompt = probe.synthesis.prompt_contract(probe.synthesis.BOUNDED_VERSION)
        actual = probe.contract(fixture.protocol)
        self.assertNotEqual(actual["protocol_id"], probe.baseline.VERSION)
        self.assertEqual(actual["reference_contract"], reference.contract())
        self.assertEqual((actual["n"], actual["max_new_tokens"], actual["context_limit"]), (4, 8192, 24576))
        self.assertEqual(actual["optimizer_updates"], 0)
        for key in ("training_allowed", "feedback", "retries", "draft_regeneration", "actual_learning_measured",
                    "self_improvement_verified", "module_causality_certified"):
            self.assertIs(actual[key], False)
        self.assertEqual(fixture.protocol, before)
        self.assertEqual(probe.synthesis.prompt_contract(probe.synthesis.BOUNDED_VERSION), original_prompt)
        with self.assertRaises(ValueError):
            probe.contract(fixture.protocol | {"context_limit": 16384})

    def test_all_histories_images_reread_drafts_raw_outputs_and_seeds_remain_bound(self):
        fixture, actors, trees, good = self.fixture()
        before = [to_json(tree) for tree in trees]
        replies = [good, "not JSON", (good, "length"), ""]
        rows, skipped, journal = self.collect(fixture, actors, trees, replies)
        self.assertEqual([to_json(tree) for tree in trees], before)
        self.assertEqual(skipped, [{"problem_id": actors[0].problem_id, "reason": "NO_VERIFIED_RECOVERY"}])
        self.assertEqual([event for event, _ in journal], ["CALL_STARTED", "CALL_GENERATED", "CALL_COMPLETED"])
        self.assertEqual(len(fixture.engine.requests), 1)
        request = fixture.engine.requests[0]
        self.assertIs(request.images, fixture.images[fixture.actor.problem_id])
        self.assertEqual((len(request.images), request.n, request.max_new_tokens), (2, 4, 8192))
        self.assertEqual(request.seeds, [sample.seed for sample in fixture.tree.groups[-1].samples])
        started = journal[0][1]
        self.assertEqual(started["original_messages"], fixture.tree.notebook["rewarded_notebook"]["notebook_messages"])
        self.assertEqual(request.messages[0]["content"][:-1], started["original_messages"][0]["content"][:-1])
        self.assertEqual(started["original_messages_sha256"], _hash(started["original_messages"]))
        self.assertEqual(started["changed_messages_sha256"], _hash(request.messages))
        self.assertEqual(started["original_prompt_sha256"], fixture.tree.groups[-1].state_sha256)
        self.assertEqual(started["public_inventory_sha256"], _hash(started["public_inventory"]))
        self.assertEqual(started["audit_coverage"]["status"], "AUDIT_COVERAGE_COMPLETE")
        text = request.messages[0]["content"][-1]["text"]
        transcript = json.loads(text.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
        self.assertEqual(len(transcript), 6)
        for row, sample in zip(transcript, episode.solution_samples(fixture.tree), strict=True):
            self.assertEqual("".join(line for _, line in row["source_lines"]), sample.text)
        note = fixture.tree.notebook["rewarded_notebook"]
        for block in (fresh_observation.quoted_context(note["fresh_observation"]), drafts.quoted_context(note["pair_drafts"])):
            self.assertEqual(text.count(block), 1)
            self.assertEqual(started["original_messages"][0]["content"][-1]["text"].count(block), 1)
        self.assertNotIn(probe.synthesis.PUBLIC_JSON_SHAPE, text)
        self.assertNotIn(probe.synthesis.BOUNDED_PROCEDURE, text)
        self.assertNotIn('"checks":', text)
        self.assertNotIn('"expected"', text)
        row = rows[0]
        self.assertEqual([sample["text"] for sample in row["samples"]], [good, "not JSON", good, ""])
        self.assertEqual([sample["finish_reason"] for sample in row["samples"]], ["stop", "stop", "length", "stop"])
        self.assertEqual(row["baseline_totals"]["strict_verified_notes"], 0)
        self.assertEqual(row["expanded_grading_representations"][0]["status"], reference.CAPTURED)
        self.assertEqual(row["expanded_grading_representations"][1]["status"], reference.INVALID)
        self.assertEqual(row["expanded_grading_representations"][2]["status"], reference.TRUNCATED)
        self.assertEqual(row["reference_totals"]["program_rendered_strict_verified_notes"], 1)
        self.assertEqual(len(row["diagnostic_group_credit"]), 4)
        self.assertEqual(row["optimizer_updates"], 0)
        for key in ("actual_learning_measured", "self_improvement_verified", "eligible_for_rl_loss", "used_for_routing"):
            self.assertIs(row[key], False)

    def test_active_prompt_reads_no_private_audit_or_reference_values(self):
        fixture, _, _, _ = self.fixture()
        note = fixture.tree.notebook["rewarded_notebook"]
        samples = episode.solution_samples(fixture.tree)
        schema = probe.for_actor(fixture.actor, fixture.prompts)["observation_instruction"][fixture.actor.family]
        before = probe._messages(note["notebook_messages"], fixture.actor, schema, samples, note, 6)
        fixture.tree.notebook["private_audit"] = "PRIVATE_AUDIT_SENTINEL"
        fixture.item.slots["start"] = "SECRET_REFERENCE_VALUE"
        after = probe._messages(note["notebook_messages"], fixture.actor, schema, samples, note, 6)
        self.assertEqual(after, before)
        text = after[0][0]["content"][-1]["text"]
        self.assertNotIn("PRIVATE_AUDIT_SENTINEL", text)
        self.assertNotIn("SECRET_REFERENCE_VALUE", text)

    def test_unmappable_private_check_stops_before_generation_and_never_enters_prompt(self):
        fixture, actors, trees, good = self.fixture()
        validate = reference.validate_audit_coverage

        def missing(inventory, audit):
            changed = copy.deepcopy(audit)
            changed["attempts"][0]["checks"].append({"kind": "TRACE", "rule": "PRIVATE_ONLY", "locus": "SECRET_VALUE"})
            return validate(inventory, changed)

        with patch.object(reference, "validate_audit_coverage", side_effect=missing), self.assertRaisesRegex(
                ValueError, "private check absent"):
            self.collect(fixture, actors, trees, [good] * 4)
        self.assertEqual(fixture.engine.requests, [])

    def test_context_and_changed_pixels_reject_without_dropping_any_history(self):
        for mutation in ("context", "pixels"):
            with self.subTest(mutation=mutation):
                fixture, actors, trees, good = self.fixture()
                before = [to_json(tree) for tree in trees]
                prepare = fixture.engine.prepare

                def changed(request):
                    prepared = prepare(request)
                    if probe.VERSION in request.request_id:
                        if mutation == "context":
                            prepared.token_ids = [1] * (24576 - 8192 + 1)
                        else:
                            prepared.pixels["pixel_values"] += 1
                    return prepared

                with patch.object(fixture.engine, "prepare", side_effect=changed), self.assertRaisesRegex(
                        ValueError, "CONTEXT_BLOCKED" if mutation == "context" else "image tensors"):
                    self.collect(fixture, actors, trees, [good] * 4)
                self.assertEqual(fixture.engine.requests, [])
                self.assertEqual([to_json(tree) for tree in trees], before)

    def test_altered_saved_seed_or_text_and_expired_budget_never_sample(self):
        for mutation in ("seed", "text", "deadline"):
            with self.subTest(mutation=mutation):
                fixture, actors, trees, good = self.fixture()
                if mutation == "seed":
                    fixture.tree.groups[-1].samples[0].seed += 1
                if mutation == "text":
                    episode.solution_samples(fixture.tree)[0].text += " changed"
                with self.assertRaises(TimeoutError if mutation == "deadline" else ValueError):
                    self.collect(fixture, actors, trees, [good] * 4,
                        deadline=time.monotonic() - 1 if mutation == "deadline" else None)
                self.assertEqual(fixture.engine.requests, [])

    def test_incomplete_output_or_wrong_decode_cannot_be_recorded_as_complete(self):
        for mutation in ("count", "decode"):
            with self.subTest(mutation=mutation):
                fixture, actors, trees, good = self.fixture()
                sample = fixture.engine.sample

                def changed(requests):
                    outputs = sample(requests)
                    if mutation == "count":
                        outputs[0].completions.pop()
                    else:
                        value = outputs[0].completions[0]
                        outputs[0].completions[0] = Completion(value.token_ids, value.logprobs,
                            value.finish_reason, value.text + "x")
                    return outputs

                with patch.object(fixture.engine, "sample", side_effect=changed), self.assertRaisesRegex(
                        ValueError, "incomplete" if mutation == "count" else "decode"):
                    self.collect(fixture, actors, trees, [good] * 4)

    def test_cli_forwards_new_mode_and_rejects_every_competing_intervention(self):
        command = ["--preparation", "/prep", "--model-path", "/model", "--cache-check", "/cache",
                   "--parent", "/parent", "--output", "/output", "--notebook-whole-reference"]
        with patch.object(runner.subprocess, "Popen") as popen, patch("fine_tuning.budget.terminate_group") as stop:
            popen.return_value.wait.return_value = 0
            self.assertEqual(runner.main(command), 0)
            self.assertIn("--notebook-whole-reference", popen.call_args.args[0])
            popen.return_value.wait.assert_called_once_with(timeout=3570)
            stop.assert_called_once_with(popen.return_value, grace_seconds=10)
        with patch.object(runner, "capture") as capture:
            self.assertEqual(runner.main(command + ["--worker"]), 0)
            capture.assert_called_once_with(Path("/prep"), Path("/model"), Path("/cache"),
                Path("/parent"), Path("/output"), notebook_whole_reference=True)
        alternatives = {"notebook_layout": [], "notebook_drafts": [], "notebook_relationships": [],
            "notebook_relationship_comparisons": [], "notebook_pair_classification": [],
            "notebook_whole_synthesis": [], "trained_notebook_reload": ["/reload"]}
        for name, value in alternatives.items():
            with self.subTest(name=name), patch.object(runner.subprocess, "Popen") as popen, patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    runner.main(command + ["--" + name.replace("_", "-"), *value])
                popen.assert_not_called()
                with self.assertRaisesRegex(ValueError, "one diagnostic"):
                    runner.capture(*(Path("/unused") for _ in range(5)), notebook_whole_reference=True,
                        notebook_layout=name == "notebook_layout", notebook_drafts=name == "notebook_drafts",
                        notebook_relationships=name == "notebook_relationships",
                        notebook_relationship_comparisons=name == "notebook_relationship_comparisons",
                        notebook_pair_classification=name == "notebook_pair_classification",
                        notebook_whole_synthesis=name == "notebook_whole_synthesis",
                        trained_notebook_reload=Path("/reload") if name == "trained_notebook_reload" else None)

    def test_capture_reuses_v45_parent_nf4_lifecycle_and_never_loads_trained_reload(self):
        fixture, actors, trees, _ = self.fixture()
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            parent, data, model, cache, output, prep = [root / name for name in
                ("parent", "data", "model", "cache", "output", "preparation.json")]
            runner.save(data / "manifest.yaml", {"split_manifests": {"train": "vault"}})
            bundle = SimpleNamespace(base_sha256="base")

            def patched(target, **kwargs):
                return stack.enter_context(patch(target, **kwargs))

            stack.enter_context(patch.dict(os.environ, MMDL_FACT_PROBE_SUPERVISOR=str(os.getppid())))
            patched("fine_tuning.notebook_rl.policy.enable_deterministic_execution", return_value={})
            readiness = patched("fine_tuning.notebook_rl.local_run.cache_readiness",
                return_value={"original_model_root": str(root / "original")})
            preparation = patched("fine_tuning.notebook_rl.local_run.read_preparation",
                return_value=({"build_root": str(data)}, {}, actors))
            patched("fine_tuning.notebook_rl.local_run.minimal_actors", return_value=actors)
            closed = patched("fine_tuning.notebook_rl.fact_context_probe.whole_synthesis_parent",
                return_value=(trees, {"files": {}}))
            patched("fine_tuning.notebook_rl.approval.code_hashes", return_value={"source": "unchanged"})
            patched("fine_tuning.notebook_rl.local_run._install_actor_guards", return_value=[])
            patched("fine_tuning.notebook_rl.quantized_cache.block_original_access",
                return_value={"denied_original_access_attempts": 0})
            patched("fine_tuning.notebook_rl.verifier_service.VerifierService",
                return_value=nullcontext(SimpleNamespace(metadata={"vault_manifest_sha256": "vault"})))
            patched("fine_tuning.notebook_rl.episode_notebook.audit_episode", return_value={"complete_episode": True})
            load = patched("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy", return_value=bundle)
            reload = patched("fine_tuning.notebook_rl.checkpoint.load_eval")
            patched("fine_tuning.notebook_rl.local_run.parameter_hashes", return_value={"unchanged": "weights"})
            engine = patched("fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine")
            patched("fine_tuning.notebook_rl.dataset.load_images", return_value=[])
            patched("fine_tuning.notebook_rl.local_run.make_probes", return_value=[])
            patched("fine_tuning.notebook_rl.fact_context_probe.bind_history_tokens")
            sync = patched("fine_tuning.notebook_rl.trainer.synchronize", return_value={})
            collect = stack.enter_context(patch.object(probe, "collect", return_value=([{"samples": [{}] * 4}], [])))
            unchanged = patched("fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged")
            result = runner.capture(prep, model, cache, parent, output, notebook_whole_reference=True)
            self.assertEqual((result["status"], result["calls"], result["samples"], result["optimizer_updates"]),
                             ("WHOLE_REFERENCE_DIAGNOSTIC_CAPTURED", 1, 4, 0))
            self.assertFalse(result["training_allowed"])
            self.assertFalse(result["actual_learning_measured"])
            closed.assert_called_once_with(parent, actors, fixture.protocol, model)
            readiness.assert_called_once_with(cache, model, fixture.protocol)
            preparation.assert_called_once_with(prep, fixture.protocol)
            load.assert_called_once_with(model, "MOD-RV", fixture.protocol)
            reload.assert_not_called()
            self.assertEqual(engine.call_args.args[2], 24576)
            self.assertEqual(engine.call_args.kwargs["policy_version"], 0)
            self.assertEqual(sync.call_args.args[5], 0)
            collect.assert_called_once()
            sidecar = {"index": 0, "completion": {"text": "raw diagnostic output"},
                       "group_complete": False, "eligible_for_rl_loss": False}
            collect.call_args.kwargs["record_completion"](sidecar)
            self.assertEqual(runner.read(output / "completion-0000.json"), sidecar)
            unchanged.assert_called_once_with(bundle)
            self.assertEqual(runner.read(output / "launch.json")["diagnostic"], probe.contract(fixture.protocol))


if __name__ == "__main__":
    unittest.main()
