"""Scripted diagnostic plumbing; fixtures are not actual native learning evidence."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fine_tuning.notebook_rl import episode_notebook as episode, fact_context_probe as runner
from fine_tuning.notebook_rl import notebook_pair_to_whole_probe as probe, notebook_whole_trained_probe as old
from fine_tuning.notebook_rl import local_profile as profile, pair_action_notebook as action
from fine_tuning.notebook_rl.engines import Completion
from fine_tuning.notebook_rl.rollout import RolloutConfig
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests import test_notebook_rl_whole_reference_probe as reference_tests
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as native_tests


class PairToWholeProbeTests(unittest.TestCase):
    def fixture(self):
        f, actors, trees, good = reference_tests.WholeReferenceProbeTests.fixture(self)
        f.protocol = profile.build_protocol(verified_note_synthesis_credit=True,
            pair_action_notebook=True, context_limit=24576)
        f.prompts = profile.local_prompts(f.protocol)
        f.cfg = RolloutConfig.from_protocol(f.protocol, "ALG-NB3", master_seed=3407)
        f.engine.context_limit = 24576
        prepare = f.engine.prepare

        def with_spans(request):
            actual = prepare(request)
            actual.image_spans = ((1, 2),)
            return actual

        f.engine.prepare = with_spans
        f.images[actors[0].problem_id] = f.images[f.actor.problem_id]
        for actor, tree in zip(actors, trees, strict=True):
            tree.groups = [group for group in tree.groups if group.kind != "DIAG"]
            tree.notebook = {}
            samples = episode.solution_samples(tree)
            for index, sample in enumerate(samples):
                sample.completion_token_ids = f.engine.tokenizer.encode(sample.text) + [2]
                sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
                if sample.parent_sample_id is not None:
                    parent = next(value for value in samples if value.sample_id == sample.parent_sample_id)
                    sample.prompt_token_ids = parent.prompt_token_ids + parent.completion_token_ids + [900 + index // 2]
            f.engine.replies = [["Observed facts:\n" + "\n".join(
                f"{key} = {value}" for key, value in f.item.slots.items()) + "\nSolution:"], ["PERSISTED"] * 4]
            action.append([tree], {actor.problem_id: f.images[actor.problem_id]}, f.engine, f.verifier, f.prompts, f.cfg,
                actors={actor.problem_id: actor},
                should_stop=lambda: any(group.kind == "DIAG" for group in tree.groups))
        f.engine.requests.clear()
        f.engine.context_limit = 24576
        return f, actors, trees, good

    def collect(self, f, actors, trees, replies, *, arm="baseline", baseline=None, mutation=None):
        f.engine.policy_version = 0 if arm == "baseline" else 1
        f.engine.replies = [replies]
        events = []
        completed: list[dict] = []
        sample = f.engine.sample

        def observed(requests, *, on_completion):
            outputs = sample(requests)
            if mutation is not None:
                outputs = mutation(outputs)
            for index, value in enumerate(outputs[0].completions):
                on_completion(requests[0].request_id, index, requests[0].seeds[index], value)
            return outputs

        with patch.object(f.engine, "sample", side_effect=observed):
            rows, skipped = probe.collect(actors, trees, f.images, f.prompts, f.engine, f.verifier,
                lambda *value: events.append(value), time.monotonic() + 60,
                protocol=f.protocol, arm=arm, baseline=baseline, record_completion=completed.append)
        return rows, skipped, events, completed

    def test_separate_contract_preserves_old_exact_replay_gate(self):
        protocol = profile.build_protocol(verified_note_synthesis_credit=True,
            pair_action_notebook=True, context_limit=24576)
        contract = probe.contract(protocol)
        self.assertEqual((contract["n"], contract["max_new_tokens"], contract["context_limit"]), (4, 8192, 24576))
        self.assertFalse(contract["training_allowed"])
        self.assertFalse(contract["lesson_semantics_verified"])
        with self.assertRaises(ValueError):
            old.contract(protocol)
        with self.assertRaises(ValueError):
            probe.contract(profile.build_protocol())

    def test_new_f0_and_ft_share_all_history_pixels_prompt_and_seeds_and_strict_grader(self):
        f, actors, trees, good = self.fixture()
        original = [to_json(tree) for tree in trees]
        rows, skipped, events, completed = self.collect(f, actors, trees, [good, "PERSISTED", "not JSON", (good, "length")])
        self.assertEqual(skipped, [{"problem_id": actors[0].problem_id, "reason": "NO_VERIFIED_RECOVERY"}])
        self.assertEqual([value[0] for value in events], ["CALL_STARTED", "CALL_GENERATED", "CALL_COMPLETED"])
        self.assertEqual(rows[0]["strict_verified_notes"], 1)
        self.assertEqual([value["completion"] for value in completed], events[1][1]["completions"])
        started = events[0][1]
        text = started["messages"][0]["content"][-1]["text"]
        transcript = json.loads(text.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
        self.assertEqual(["".join(line for _, line in value["source_lines"]) for value in transcript],
                         [sample.text for sample in episode.solution_samples(f.tree)])
        self.assertEqual(len(transcript), 6)
        self.assertIn("UNVERIFIED", text)
        self.assertNotIn("unverified_pair_drafts", text)
        self.assertNotIn('"expected"', text)
        self.assertNotIn('"checks":', text)
        self.assertIs(f.engine.requests[-1].images, f.images[f.actor.problem_id])
        baseline = json.loads(json.dumps({"started": started, "row": rows[0]}))
        self.assertIs(type(started["images"]["image_spans"]), tuple)
        self.assertIs(type(baseline["started"]["images"]["image_spans"]), list)
        self.assertEqual(baseline["started"]["images"]["image_spans"], [[1, 2]])
        after, _, trained_events, _ = self.collect(f, actors, trees, [good] * 4, arm="trained", baseline=baseline)
        self.assertEqual({k: v for k, v in trained_events[0][1].items() if k != "generation_policy_version"},
                         {k: v for k, v in started.items() if k != "generation_policy_version"})
        self.assertEqual(trained_events[0][1]["generation_policy_version"], 1)
        self.assertEqual(runner.trained_note_counts(after)["recovered"], 3)
        self.assertEqual([to_json(tree) for tree in trees], original)
        self.assertTrue(all(value["eligible_for_rl_loss"] is False and value["used_for_routing"] is False
                            for value in after[0]["samples"]))

    def test_old_label_baseline_missing_baseline_and_changed_new_input_reject(self):
        f, actors, trees, good = self.fixture()
        with self.assertRaises(ValueError):
            self.collect(f, actors, trees, [good] * 4, arm="trained")
        rows, _, events, _ = self.collect(f, actors, trees, [good] * 4)
        for key in ("messages", "seeds", "images", "history_sha256"):
            baseline = {"started": copy.deepcopy(events[0][1]), "row": rows[0]}
            baseline["started"][key] = "different"
            f.engine.requests.clear()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.collect(f, actors, trees, [good] * 4, arm="trained", baseline=baseline)
            self.assertEqual(f.engine.requests, [])

    def test_original_text_pair_seed_and_image_mutations_reject_before_generation(self):
        for mutation in ("text", "seed", "pixels", "logprobs"):
            with self.subTest(mutation=mutation):
                f, actors, trees, good = self.fixture()
                if mutation == "text":
                    episode.solution_samples(f.tree)[0].text += "changed"
                elif mutation == "seed":
                    f.tree.groups[-1].samples[0].seed += 1
                elif mutation == "logprobs":
                    episode.solution_samples(f.tree)[0].sampler_logprobs[0] = float("nan")
                else:
                    prepare = f.engine.prepare

                    def changed(request):
                        actual = prepare(request)
                        actual.pixels["pixel_values"] += 1
                        return actual

                    f.engine.prepare = changed
                with self.assertRaises((ValueError, RuntimeError)):
                    self.collect(f, actors, trees, [good] * 4)
                self.assertEqual(f.engine.requests, [])

    def test_bad_actual_decode_prompt_history_context_and_partial_group_reject(self):
        for mutation in ("decode", "prompt", "history", "context", "n", "messages"):
            with self.subTest(mutation=mutation):
                f, actors, trees, good = self.fixture()

                def changed(outputs):
                    if mutation == "decode":
                        value = outputs[0].completions[0]
                        outputs[0].completions[0] = Completion(value.token_ids, value.logprobs, value.finish_reason, "changed")
                    elif mutation == "prompt":
                        outputs[0] = replace(outputs[0], prompt_token_ids=[123])
                    elif mutation == "history":
                        f.tree.notebook["changed"] = True
                    elif mutation == "n":
                        outputs[0].completions.pop()
                    elif mutation == "messages":
                        f.engine.requests[-1].messages[0]["ignored_metadata"] = "changed"
                    return outputs

                if mutation == "context":
                    prepare = f.engine.prepare

                    def oversized(request):
                        actual = prepare(request)
                        if probe.VERSION in request.request_id:
                            actual.token_ids = [1] * (24576 - 8192 + 1)
                        return actual

                    f.engine.prepare = oversized
                elif mutation == "messages":
                    prepare = f.engine.prepare

                    def ignore_metadata(request):
                        messages = copy.deepcopy(request.messages)
                        messages[0].pop("ignored_metadata", None)
                        return prepare(replace(request, messages=messages))

                    f.engine.prepare = ignore_metadata
                with self.assertRaises((ValueError, RuntimeError)):
                    self.collect(f, actors, trees, [good] * 4, mutation=changed)
                if mutation == "context":
                    self.assertEqual(f.engine.requests, [])

    def test_reload_required_for_both_cli_arms_and_missing_reload_is_not_provenance(self):
        flags = ["--preparation", "/prep", "--model-path", "/model", "--cache-check", "/cache",
                 "--parent", "/parent", "--output", "/output"]
        for arm in ("baseline", "trained"):
            with self.subTest(arm=arm), patch("sys.stderr"), self.assertRaises(SystemExit):
                runner.main([*flags, "--pair-to-whole", arm])
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            runner.trained_parent(Path(directory) / "missing-parent", Path(directory) / "missing-reload",
                                  Path("/model"), Path("/cache"))
        with patch.dict(os.environ, MMDL_FACT_PROBE_SUPERVISOR=str(os.getppid())), \
                patch.object(runner, "capture") as capture:
            self.assertEqual(runner.main([*flags, "--worker", "--pair-to-whole", "baseline",
                                         "--trained-notebook-reload", "/reload"]), 0)
            self.assertEqual(capture.call_args.kwargs["pair_to_whole"], "baseline")

    def test_new_baseline_requires_closed_whole_capture_and_exact_raw_completion_journal(self):
        f, actors, trees, good = self.fixture()
        rows, _, events, completed = self.collect(f, actors, trees, [good] * 4)
        sources = {"current": "fixture"}
        evidence: dict = {"files": {}, "modes": {}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof = root / "parent"
            runner.save(proof / "parameter_changes.json", {"before": {"L": "initial"}, "after": {"L": "updated"}})
            runner.save(proof / "reload_expected.json", {"frozen_base_sha256": "base"})
            trained = {"files": [{"path": str(proof / name)} for name in ("parameter_changes.json", "reload_expected.json")]}
            current = {"preparation_sha256": "current-prep", "cache_readiness_sha256": "current-cache",
                       "model_path": "/model", "problem_ids": [actor.problem_id for actor in actors]}
            audits = [episode.audit_episode(tree, actor, f.verifier, routing_rule=f.protocol["notebook_route_rule"])
                      for tree, actor in zip(trees, actors, strict=True)]
            launch = {"pid": 999999999, "diagnostic": probe.contract(f.protocol),
                "pair_to_whole_arm": "baseline", "trained_parent_gate": trained, "parent": evidence, "sources": sources,
                "protocol": json.loads(json.dumps(f.protocol)), "diagnostic_sha256": probe.checkpoint.digest(probe.contract(f.protocol)), **current}
            runner.save(root / "launch.json", launch)
            runner.save(root / "result.json", {"status": "PAIR_TO_WHOLE_DIAGNOSTIC_CAPTURED",
                "pair_to_whole_arm": "baseline", "generation_policy_version": 0, "optimizer_updates": 0,
                "training_allowed": False, "source_unchanged": True, "trainables_unchanged": True,
                "base_unchanged": True, "calls": 1, "samples": 4,
                "trainable_sha256": {"L": "initial"}, "frozen_base_sha256": "base",
                "original_access_guard": {"denied_original_access_attempts": 0}})
            runner.save(root / "initial_sync.json", [{"request_id": "local-sync:" + actor.problem_id,
                "mean_abs_delta": 0., "max_abs_delta": 0., "policy_version": 0} for actor in actors])
            runner.save(root / "parent-audit.json", audits)
            for index, (kind, value) in enumerate(events):
                runner.save(root / f"raw-{index:04d}.json", {"kind": kind, "raw": value})
            for index, value in enumerate(completed):
                runner.save(root / f"completion-{index:04d}.json", value)
            actual = probe.baseline_capture(root, f.protocol, trained, evidence, sources, current=current)
            self.assertEqual(actual["row"], rows[0])
            self.assertEqual(actual["audits"], audits)
            self.assertEqual(len(actual["files"]), 11)
            for path, mutate in ((root / "launch.json", lambda value: value.update(diagnostic={"policy": action.VERSION})),
                    (root / "launch.json", lambda value: value.update(pid=os.getpid())),
                    (root / "result.json", lambda value: value.update(trainable_sha256={"L": "updated"})),
                    (root / "initial_sync.json", lambda value: value[0].update(policy_version=1)),
                    (root / "raw-0001.json", lambda value: value["raw"]["completions"][0].update(text="PERSISTED")),
                    (root / "completion-0000.json", lambda value: value.update(seed=123)),
                    (root / "completion-0000.json", lambda value: value.update(request_id="different")),
                    (root / "completion-0000.json", lambda value: value.update(prompt_sha256="different")),
                    (root / "completion-0000.json", lambda value: value.update(eligible_for_rl_loss=True))):
                original = path.read_bytes()
                value = json.loads(original)
                mutate(value)
                path.write_text(json.dumps(value))
                with self.subTest(path=path.name), self.assertRaises(ValueError):
                    probe.baseline_capture(root, f.protocol, trained, evidence, sources, current=current)
                path.write_bytes(original)
            for key in ("protocol", "diagnostic_sha256", *current):
                path = root / "launch.json"
                original = path.read_bytes()
                value = json.loads(original)
                value[key] = "contradictory"
                path.write_text(json.dumps(value))
                with self.subTest(binding=key), self.assertRaises(ValueError):
                    probe.baseline_capture(root, f.protocol, trained, evidence, sources, current=current)
                path.write_bytes(original)

    def test_historical_source_mapping_is_separate_but_hashes_and_modes_are_exact(self):
        protocol = profile.build_protocol(verified_note_synthesis_credit=True,
            pair_action_notebook=True, context_limit=24576)
        with tempfile.TemporaryDirectory() as directory:
            root, parent = Path(directory), Path(directory) / "minimal"
            source = root / "package-root/fine_tuning/notebook_rl/__init__.py"
            source.parent.mkdir(parents=True)
            source.write_text("# historical CPU fixture\n")
            codes = {"__init__.py": probe.checkpoint.file_hash(source)}
            runner.save(root / "sources.json", {"fine_tuning/notebook_rl/__init__.py": {
                "sha256": codes["__init__.py"], "mode": source.stat().st_mode & 0o777}})

            def descriptor(name, value):
                path = root / name
                runner.save(path, value)
                return {"path": str(path), "file_sha256": probe.checkpoint.file_hash(path)}

            actor = replace(native_tests.native()[0], mode="obs")
            actors = descriptor("actors.json", [json.loads(to_json(actor))])
            build = root / "build"
            prepared = {"format": "notebook-rl-local-data-v1", "profile": profile.PROFILE,
                "status": "LOCAL_DATA_REVALIDATED", "cloud_allowed": False,
                "local_protocol_sha256": protocol["sha256"], "build_root": str(build), "image_root": str(build / "images"),
                "actors": actors, "execution_actors": actors,
                "actor_projection": {"policy": "all-fixed-ten-observations-v1", "changed_ids": []},
                "selection": descriptor("selection.json", {"problem_ids": [actor.problem_id]}),
                "validated_data": descriptor("checked.json", {"code_sha256": codes,
                    "status": "VALIDATED_FOR_MEASUREMENT", "blockers": [],
                    "measurement_allowed": True, "training_allowed": False}),
                "data_request": descriptor("request.json", {"build_root": str(build)})}
            from fine_tuning.notebook_rl.local_data import ACTOR_MODE_POLICY
            prepared["actor_projection"]["policy"] = ACTOR_MODE_POLICY
            runner.save(root / "preparation.json", prepared)
            launch = {"preparation": str(root / "preparation.json"), "binding": {"code_sha256": probe.checkpoint.digest(codes)}}
            runner.save(parent / "launch.json", launch)
            runner.save(parent / "update.json", {})
            trained = {"status": "REAL_MINIMAL_UPDATE_AND_FRESH_RELOAD_VERIFIED", "launch": launch, "files": []}
            with patch.object(runner, "closed_parent", return_value=([], {"files": {}})), \
                    patch("fine_tuning.notebook_rl.local_run.native_relationship_update_evidence", return_value={"fixture": True}):
                _, evidence = probe.historical_parent(parent, [], protocol, root / "model", trained)
                self.assertFalse(evidence["historical_approval_relabelled"])
                self.assertIn(str(source), evidence["files"])
                source.chmod(0o600 if source.stat().st_mode & 0o777 != 0o600 else 0o644)
                with self.assertRaises(ValueError):
                    probe.historical_parent(parent, [], protocol, root / "model", trained)

    def test_capture_loads_original_f0_or_verified_ft_and_requires_native_gate_for_both(self):
        f, actors, trees, _ = self.fixture()
        initial = {group: {group: "initial"} for group in ("L", "M", "V")}
        updated = {group: {group: "updated"} for group in ("L", "M", "V")}
        for arm, audit_valid in (("baseline", True), ("trained", True), ("trained", False)):
            with self.subTest(arm=arm, audit_valid=audit_valid), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                parent, data = root / "parent", root / "data"
                model, cache, reload, output = (root / name for name in ("model", "cache", "reload", "output"))
                runner.save(data / "manifest.yaml", {"split_manifests": {"train": "vault"}})
                runner.save(parent / "reload_expected.json", {"trainable_sha256": updated,
                    "step": 1, "frozen_base_sha256": "base", "quantization": {"fixture": True}})
                runner.save(parent / "parameter_changes.json", {"before": initial, "after": updated})
                for name in ("result.json", "update.json"):
                    runner.save(parent / name, {})
                f.verifier.metadata = {"vault_manifest_sha256": "vault"}
                bundle = SimpleNamespace(base_sha256="base", inventory={"quantization": {"fixture": True}})

                def mocked(target, **kwargs):
                    return stack.enter_context(patch(target, **kwargs))

                stack.enter_context(patch.dict(os.environ, MMDL_FACT_PROBE_SUPERVISOR=str(os.getppid())))
                mocked("fine_tuning.notebook_rl.policy.enable_deterministic_execution", return_value={})
                mocked("fine_tuning.notebook_rl.fact_context_probe.trained_parent", return_value={
                    "launch": {"protocol": f.protocol, "binding": {"historical": "fixture"}}, "files": []})
                mocked("fine_tuning.notebook_rl.local_run.cache_readiness", return_value={"original_model_root": str(root / "original")})
                mocked("fine_tuning.notebook_rl.local_run.read_preparation", return_value=({"build_root": str(data)}, {}, actors))
                mocked("fine_tuning.notebook_rl.local_run.minimal_actors", return_value=actors)
                mocked("fine_tuning.notebook_rl.notebook_pair_to_whole_probe.historical_parent",
                       return_value=(trees, {"files": {}, "modes": {}}))
                mocked("fine_tuning.notebook_rl.approval.code_hashes", return_value={"current": "fixture"})
                mocked("fine_tuning.notebook_rl.local_run._install_actor_guards", return_value=[])
                mocked("fine_tuning.notebook_rl.quantized_cache.block_original_access", return_value={"denied_original_access_attempts": 0})
                mocked("fine_tuning.notebook_rl.verifier_service.VerifierService", return_value=nullcontext(f.verifier))
                load_ft = mocked("fine_tuning.notebook_rl.checkpoint.load_eval", return_value=bundle)
                load_f0 = mocked("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy", return_value=bundle)
                mocked("fine_tuning.notebook_rl.local_run.parameter_hashes", return_value=initial if arm == "baseline" else updated)
                engine = mocked("fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine", return_value=f.engine)
                mocked("fine_tuning.notebook_rl.dataset.load_images", return_value=f.images[f.actor.problem_id])
                mocked("fine_tuning.notebook_rl.local_run.make_probes", return_value=[])
                mocked("fine_tuning.notebook_rl.fact_context_probe.bind_history_tokens")
                native = mocked("fine_tuning.notebook_rl.local_train.fact_credit_learning_gate", return_value=[])
                mocked("fine_tuning.notebook_rl.trainer.synchronize", return_value={})
                mocked("fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged")
                mocked("fine_tuning.notebook_rl.notebook_pair_to_whole_probe.baseline_capture",
                       return_value={"files": {}, "modes": {}, "audits": [episode.audit_episode(tree, actor,
                           f.verifier, routing_rule=f.protocol["notebook_route_rule"])
                           for tree, actor in zip(trees, actors, strict=True)] if audit_valid else []})
                sample = {"baseline_notebook_check": {"reward": 0}, "notebook_check": {"reward": 1}}
                collect = mocked("fine_tuning.notebook_rl.notebook_pair_to_whole_probe.collect",
                                 return_value=([{"samples": [sample] * 4}], []))
                if not audit_valid:
                    with self.assertRaisesRegex(ValueError, "parent audit differs"):
                        runner.capture(root / "prep", model, cache, parent, output,
                            trained_notebook_reload=reload, pair_to_whole=arm,
                            pair_to_whole_baseline=root / "new-baseline")
                    load_ft.assert_not_called()
                    load_f0.assert_not_called()
                    native.assert_not_called()
                    continue
                result = runner.capture(root / "prep", model, cache, parent, output,
                    trained_notebook_reload=reload, pair_to_whole=arm,
                    pair_to_whole_baseline=root / "new-baseline" if arm == "trained" else None)
                self.assertEqual(result["status"], "PAIR_TO_WHOLE_DIAGNOSTIC_CAPTURED")
                self.assertEqual(result["generation_policy_version"], 0 if arm == "baseline" else 1)
                self.assertEqual(engine.call_args.kwargs["policy_version"], result["generation_policy_version"])
                self.assertEqual(engine.call_args.args[2], 24576)
                self.assertEqual(collect.call_args.kwargs["arm"], arm)
                self.assertEqual(load_f0.call_count, int(arm == "baseline"))
                self.assertEqual(load_ft.call_count, int(arm == "trained"))
                native.assert_called_once()


if __name__ == "__main__":
    unittest.main()
