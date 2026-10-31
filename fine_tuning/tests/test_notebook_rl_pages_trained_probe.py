"""Authored CPU page replay proves input/action plumbing, never model improvement."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import time
import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.notebook_rl import episode_notebook as episode, fact_context_probe as capture
from fine_tuning.notebook_rl import local_profile, notebook_whole_trained_probe as probe
from fine_tuning.notebook_rl import whole_reference_pages as pages
from fine_tuning.notebook_rl.engines import Completion
from fine_tuning.notebook_rl.notebook_learning import append_rewarded_notebooks
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash
from fine_tuning.notebook_rl.schema import Sample, to_json
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as native
from fine_tuning.tests import test_notebook_rl_whole_reference_training_episode as training


class PagesTrainedProbeTests(unittest.TestCase):
    def fixture(self, *, journal=False):
        authored = training.WholeReferenceTrainingEpisodeTests()
        authored.setUp()
        self.addCleanup(authored.doCleanups)
        f = authored.f
        if journal:
            second = f.images[f.actor.problem_id][0].copy()
            f.images[f.actor.problem_id].append(second)
            self.addCleanup(second.close)
            f.actor = replace(f.actor, image_paths=(*f.actor.image_paths, "images/fixture-second.png"),
                              image_sha256=(*f.actor.image_sha256, "b" * 64))
            f.verifier.actors[f.actor.problem_id] = f.actor
        f.protocol = local_profile.build_protocol(verified_note_synthesis_credit=True,
            pair_draft_synthesis=True, whole_reference_pages=True, context_limit=24576)
        f.prompts = local_profile.local_prompts(f.protocol)
        f.cfg = RolloutConfig.from_protocol(f.protocol, "ALG-NB3", master_seed=3407)
        compact = json.loads(authored.good)
        page_plan = pages.plan(episode.solution_samples(f.tree))
        good = [json.dumps({key: value for key, value in compact.items() if key != "assessments"})]
        good.extend(json.dumps({"assessments": [row for row in compact["assessments"]
            if row[0] in page["assessment_refs"]]}) for page in page_plan["pages"][1:])
        observation = "Observed facts:\n" + "\n".join(
            f"{key} = {value}" for key, value in f.item.slots.items()) + "\nSolution:"
        sample = f.engine.sample

        def streaming(requests, *, on_completion=None):
            outputs = sample(requests)
            if on_completion is not None:
                for ordinal, completion in enumerate(outputs[0].completions):
                    on_completion(requests[0].request_id, ordinal, requests[0].seeds[ordinal], completion)
            return outputs

        f.engine.sample = streaming
        f.engine.replies = [[observation], ["CORRECTED", "{broken", ("PERSISTED", "length"), "UNRESOLVED"],
                            *[["{}"] * 4 for _ in good]]
        f.page_journal = []
        callbacks = {name: (lambda value, kind=kind: f.page_journal.append((kind,
            asdict(value) if kind == "GROUP" else copy.deepcopy(value))))
            for name, kind in (("on_group", "GROUP"), ("on_observation", "OBSERVATION"),
                               ("on_draft", "DRAFT"), ("on_page", "PAGE"))} if journal else {}
        append_rewarded_notebooks([f.tree], f.images, f.engine, f.verifier, f.prompts, f.cfg,
                                  actors={f.actor.problem_id: f.actor}, on_group=callbacks.get("on_group"),
                                  on_observation=callbacks.get("on_observation"), on_draft=callbacks.get("on_draft"),
                                  on_page=callbacks.get("on_page"))
        self.assertEqual(len([group for group in f.tree.groups if group.kind == "DIAG"]), len(good))
        f.engine.policy_version, f.engine.context_limit = 1, 24576
        f.engine.requests.clear()
        first_actor = replace(f.actor, problem_id="fixed-first-no-recovery",
            image_paths=f.actor.image_paths[:1], image_sha256=f.actor.image_sha256[:1])
        f.verifier.actors[first_actor.problem_id] = first_actor
        f.verifier.items[first_actor.problem_id] = replace(f.item, problem_id=first_actor.problem_id)
        first = native.tree_from(first_actor, f.verifier, [native.response(f.item.slots)] * 6)
        if journal:
            prior: dict[str, Sample] = {}
            for original in episode.solution_samples(first):
                parent = prior.get(original.parent_sample_id)
                if parent is not None:
                    original.prompt_token_ids = parent.prompt_token_ids + parent.completion_token_ids + [901]
                original.completion_token_ids = f.engine.tokenizer.encode(original.text) + [f.engine.tokenizer.end]
                original.sampler_logprobs = [-.1] * len(original.completion_token_ids)
                prior[original.sample_id] = original
            for group in first.groups:
                group.group_id = f"{first_actor.problem_id}:{group.group_id}"
                for original in group.samples:
                    original.group_id = group.group_id
            f.engine.policy_version = 0
            append_rewarded_notebooks([first], {first_actor.problem_id: f.images[f.actor.problem_id][:1]},
                f.engine, f.verifier, f.prompts, f.cfg, actors={first_actor.problem_id: first_actor})
            f.engine.policy_version = 1
        return f, [first_actor, f.actor], [first, f.tree], good

    def collect(self, f, actors, trees, replies, *, record=None):
        f.engine.replies = replies
        events = []
        rows, skipped = probe.collect(actors, trees, f.images, f.prompts, f.engine, f.verifier,
            record or (lambda *row: events.append(row)), time.monotonic() + 60, protocol=f.protocol)
        return rows, skipped, events

    def test_explicit_paged_contract_preserves_legacy_identity_and_disallows_changed_profile(self):
        protocol = local_profile.build_protocol(verified_note_synthesis_credit=True,
            pair_draft_synthesis=True, whole_reference_pages=True, context_limit=24576)
        policy = probe.contract(protocol)
        self.assertEqual(policy["protocol_id"], probe.PAGES_VERSION)
        self.assertEqual(policy["notebook_page_contract"], pages.contract())
        self.assertEqual(policy["notebook_page_contract_sha256"], _hash(pages.contract()))
        self.assertEqual((policy["n"], policy["max_new_tokens"], policy["optimizer_updates"]), (4, 8192, 0))
        for key in ("training_allowed", "feedback", "retries", "heldout", "draft_regeneration"):
            self.assertIs(policy[key], False)
        historical = local_profile.build_protocol(verified_note_synthesis_credit=True,
            pair_draft_synthesis=True, whole_reference_notebook=True, context_limit=24576)
        self.assertEqual(probe.contract(historical)["protocol_id"], probe.REFERENCE_VERSION)
        for key in ("protocol_id", "notebook_generation_policy", "notebook_page_contract"):
            changed = copy.deepcopy(protocol)
            changed[key] = "changed"
            changed["sha256"] = _hash({key: value for key, value in changed.items() if key != "sha256"})
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.contract(changed)

    def test_exact_page_inputs_and_actual_policy_one_actions_form_four_logical_comparisons(self):
        f, actors, trees, good = self.fixture()
        before = [to_json(tree) for tree in trees]
        note = f.tree.notebook["rewarded_notebook"]
        with patch.object(probe.credit, "score_page_bundles", wraps=probe.credit.score_page_bundles) as scored, \
                patch.object(probe, "Sample", wraps=Sample) as constructed:
            rows, skipped, events = self.collect(f, actors, trees, [[text] * 4 for text in good])
        self.assertEqual([to_json(tree) for tree in trees], before)
        self.assertEqual(skipped, [{"problem_id": actors[0].problem_id, "reason": "NO_VERIFIED_RECOVERY"}])
        row = rows[0]
        self.assertEqual((row["physical_call_count"], row["physical_sample_count"]), (len(good), len(good) * 4))
        self.assertEqual(len(row["samples"]), 4)
        self.assertEqual(constructed.call_count, len(good) * 4)
        self.assertEqual((row["before_totals"]["strict_verified_notes"], row["after_totals"]["strict_verified_notes"]), (0, 4))
        self.assertEqual(capture.trained_note_counts(rows)["recovered"], 4)
        self.assertIs(scored.call_args_list[0].args[0], f.tree)
        observed = scored.call_args.kwargs["observed_groups"]
        saved_ids = {sample.sample_id for group in f.tree.groups for sample in group.samples}
        for index, (request, call, group) in enumerate(zip(f.engine.requests, note["page_calls"], observed, strict=True)):
            self.assertEqual(request.messages, call["messages"])
            self.assertIs(request.images, f.images[f.actor.problem_id])
            self.assertEqual(request.seeds, call["seeds"])
            self.assertEqual(request.max_new_tokens, note["page_plan"]["pages"][index]["max_new_tokens"])
            for sample in group.samples:
                self.assertNotIn(sample.sample_id, saved_ids)
                self.assertEqual(sample.policy_version, 1)
                self.assertEqual(f.engine.tokenizer.decode(sample.completion_token_ids), sample.text)
                self.assertIn(sample.text, good)
        for physical in row["physical_samples"]:
            self.assertTrue(physical["score"]["diagnostic_only"])
            self.assertFalse(physical["score"]["training_allowed"])
            self.assertEqual(physical["score"]["raw_text_sha256"], probe.reference._sha(physical["text"]))
        for logical in row["samples"]:
            self.assertNotIn("completion_token_ids", logical)
            self.assertNotIn("sampler_logprobs", logical)
            self.assertTrue(logical["score"]["program_rendered_only"])
            self.assertFalse(logical["score"]["training_allowed"])
            self.assertNotIn("score_sha256", logical["score"])
            self.assertEqual(len(logical["member_sample_ids"]), len(good))
        captured = [value for kind, value in events if kind == "COMPLETION_CAPTURED"]
        self.assertEqual(len(captured), len(good) * 4)
        self.assertTrue(all(value["bundle_complete"] is False for value in captured))

    def test_missing_page_changed_prompt_pixels_seed_cap_and_owner_fail_before_sampling(self):
        for mutation in ("missing", "messages", "pixels", "seed", "cap", "owner"):
            with self.subTest(mutation=mutation):
                f, actors, trees, good = self.fixture()
                note = f.tree.notebook["rewarded_notebook"]
                groups = [group for group in f.tree.groups if group.kind == "DIAG"]
                if mutation == "missing":
                    f.tree.groups.remove(groups[-1])
                elif mutation == "messages":
                    note["page_calls"][1]["messages"][0]["content"][-1]["text"] += "changed"
                elif mutation == "pixels":
                    prepare = f.engine.prepare

                    def changed(request):
                        prepared = prepare(request)
                        prepared.pixels["pixel_values"] += 1
                        return prepared

                    f.engine.prepare = changed
                elif mutation == "seed":
                    groups[-1].samples[0].seed += 1
                elif mutation == "cap":
                    note["page_calls"][-1]["max_new_tokens"] += 1
                else:
                    groups[-1].samples[0].group_id = groups[0].group_id
                with self.assertRaises(ValueError):
                    self.collect(f, actors, trees, [[text] * 4 for text in good])
                self.assertEqual(f.engine.requests, [])

    def test_partial_completions_are_preserved_without_observed_scoring_or_completed_bundle(self):
        f, actors, trees, good = self.fixture()
        sample = f.engine.sample
        events = []

        def interrupted(requests, *, on_completion):
            outputs = sample(requests)
            for ordinal, completion in enumerate(outputs[0].completions[:2]):
                on_completion(requests[0].request_id, ordinal, requests[0].seeds[ordinal], completion)
            raise TimeoutError("authored interruption")

        with patch.object(f.engine, "sample", side_effect=interrupted), \
                patch.object(probe.credit, "score_page_bundles", wraps=probe.credit.score_page_bundles) as scored:
            with self.assertRaises(TimeoutError):
                self.collect(f, actors, trees, [[text] * 4 for text in good], record=lambda *row: events.append(row))
        self.assertEqual(scored.call_count, 1)
        self.assertNotIn("observed_groups", scored.call_args.kwargs)
        self.assertEqual([kind for kind, _ in events], ["CALL_STARTED", "COMPLETION_CAPTURED", "COMPLETION_CAPTURED"])

    def test_missing_callback_changed_decode_and_returned_group_fail_closed(self):
        for mutation in ("callback", "decode", "returned"):
            with self.subTest(mutation=mutation):
                f, actors, trees, good = self.fixture()
                sample = f.engine.sample
                events = []

                def changed(requests, *, on_completion):
                    outputs = sample(requests)
                    for ordinal, completion in enumerate(outputs[0].completions):
                        if mutation == "decode" and ordinal == 0:
                            completion = Completion(completion.token_ids, completion.logprobs,
                                completion.finish_reason, completion.text + "changed")
                        if mutation != "callback":
                            on_completion(requests[0].request_id, ordinal, requests[0].seeds[ordinal], completion)
                    if mutation == "returned":
                        outputs[0].completions.pop()
                    return outputs

                with patch.object(f.engine, "sample", side_effect=changed), self.assertRaises(ValueError):
                    self.collect(f, actors, trees, [[text] * 4 for text in good], record=lambda *row: events.append(row))
                self.assertNotIn("CALL_COMPLETED", [kind for kind, _ in events])

    def test_malformed_or_truncated_page_never_passes_a_partial_bundle(self):
        f, actors, trees, good = self.fixture()
        replies = [[text] * 4 for text in good]
        replies[1][1] = "{}"
        replies[-1][2] = (good[-1], "length")
        rows, _, _ = self.collect(f, actors, trees, replies)
        self.assertEqual([row["strict_check"]["reward"] for row in rows[0]["samples"]], [1., 0., 0., 1.])
        physical = rows[0]["physical_samples"]
        for ordinal in (1, 2):
            self.assertTrue(all(row["score"]["atomic_reward"] == 0 for row in physical[ordinal::4]))

    def test_capture_restores_paged_profile_and_reports_physical_calls_and_actions(self):
        f, actors, trees, good = self.fixture()
        hashes = {"L": {"l": "l"}, "M": {"m": "m"}, "V": {"v": "v"}}
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            data, parent = root / "data", root / "parent"
            model, cache, reload, output = (root / name for name in ("model", "cache", "reload", "output"))
            capture.save(data / "manifest.yaml", {"split_manifests": {"train": "vault"}})
            capture.save(parent / "reload_expected.json", {"trainable_sha256": hashes, "step": 1,
                "frozen_base_sha256": "base", "quantization": {"kind": "fixture-nf4"}})
            f.verifier.metadata = {"vault_manifest_sha256": "vault"}
            bundle = SimpleNamespace(base_sha256="base", inventory={"quantization": {"kind": "fixture-nf4"}})

            def mocked(target, **kwargs):
                return stack.enter_context(patch(target, **kwargs))

            stack.enter_context(patch.dict(os.environ, MMDL_FACT_PROBE_SUPERVISOR=str(os.getppid())))
            mocked("fine_tuning.notebook_rl.policy.enable_deterministic_execution", return_value={})
            mocked("fine_tuning.notebook_rl.fact_context_probe.trained_parent", return_value={
                "launch": {"protocol": f.protocol, "binding": {"history": "original"}}, "files": []})
            mocked("fine_tuning.notebook_rl.local_run.cache_readiness",
                return_value={"original_model_root": str(root / "original")})
            mocked("fine_tuning.notebook_rl.local_run.read_preparation",
                return_value=({"build_root": str(data)}, {}, actors))
            mocked("fine_tuning.notebook_rl.local_run.minimal_actors", return_value=actors)
            mocked("fine_tuning.notebook_rl.fact_context_probe.closed_parent", return_value=(trees, {"files": {}}))
            mocked("fine_tuning.notebook_rl.approval.code_hashes", return_value={"source": "unchanged"})
            mocked("fine_tuning.notebook_rl.local_run._install_actor_guards", return_value=[])
            mocked("fine_tuning.notebook_rl.quantized_cache.block_original_access",
                return_value={"denied_original_access_attempts": 0})
            mocked("fine_tuning.notebook_rl.verifier_service.VerifierService", return_value=nullcontext(f.verifier))
            mocked("fine_tuning.notebook_rl.checkpoint.load_eval", return_value=bundle)
            mocked("fine_tuning.notebook_rl.local_run.parameter_hashes", return_value=hashes)
            engine = mocked("fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine", return_value=f.engine)
            mocked("fine_tuning.notebook_rl.dataset.load_images", return_value=f.images[f.actor.problem_id])
            mocked("fine_tuning.notebook_rl.local_run.make_probes", return_value=[])
            mocked("fine_tuning.notebook_rl.fact_context_probe.bind_history_tokens")
            mocked("fine_tuning.notebook_rl.trainer.synchronize", return_value={})
            mocked("fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged")
            legacy = mocked("fine_tuning.notebook_rl.notebook_layout_probe.collect")
            f.engine.replies = [[text] * 4 for text in good]
            result = capture.capture(root / "prep", model, cache, parent, output, trained_notebook_reload=reload)
            self.assertEqual((result["calls"], result["samples"], result["optimizer_updates"]), (len(good), len(good) * 4, 0))
            self.assertEqual(result["same_history_note_comparison"]["notes"], 4)
            self.assertEqual(result["same_history_note_comparison"]["after_strict_valid"], 4)
            self.assertEqual(engine.call_args.args[2], 24576)
            self.assertEqual(engine.call_args.kwargs["policy_version"], 1)
            self.assertEqual(capture.read(output / "launch.json")["diagnostic"], probe.contract(f.protocol))
            saved = [capture.read(path) for path in sorted(output.glob("raw-*.json"))]
            self.assertEqual(sum(row["kind"] == "COMPLETION_CAPTURED" for row in saved), len(good) * 4)
            legacy.assert_not_called()

    def parent(self, root, f, actors, trees, records):
        parent, model = root / "parent", root / "model"
        actors_path = root / "actors.json"
        capture.save(actors_path, [asdict(actor) for actor in actors])
        preparation = {"execution_actors": {"path": str(actors_path),
            "file_sha256": capture.checkpoint.file_hash(actors_path)}}
        preparation_path = root / "preparation.json"
        capture.save(preparation_path, preparation)
        ids = [actor.problem_id for actor in actors]
        capture.save(parent / "launch.json", {"pid": 987654321, "protocol": f.protocol,
            "problem_ids": ids, "model_path": str(model), "standalone_base": True,
            "preparation": str(preparation_path), "binding": {"protocol_sha256": f.protocol["sha256"],
                "prompts_sha256": f.protocol["prompts_sha256"], "data_sha256": capture.checkpoint.digest(preparation)}})
        capture.save(parent / "result.json", {"status": "NO_LEARNING_SIGNAL", "seconds": 1.,
            "attempted_ids": ids, "optimizer_updates": 0})
        for index, (kind, value) in enumerate(records):
            capture.save(parent / f"raw-{index:04d}.json", {"kind": kind, "raw": value})
        return parent, model

    def journal(self, f, trees):
        records = [("GROUP", asdict(group)) for group in trees[0].groups]
        records.append(("TREE", json.loads(to_json(trees[0]))))
        records.extend(("GROUP", asdict(group)) for group in trees[1].groups if group.kind != "DIAG")
        records.extend(copy.deepcopy(f.page_journal))
        records.append(("TREE", json.loads(to_json(trees[1]))))
        return records

    def test_closed_parent_binds_real_page_callbacks_and_rejects_changed_missing_duplicate_or_late_events(self):
        f, actors, trees, _ = self.fixture(journal=True)
        records = self.journal(f, trees)
        page_positions = [index for index, (kind, _) in enumerate(records) if kind == "PAGE"]
        cases = (None, "missing", "duplicate", "reordered", "group-before-pages", "tree-before-pages",
                 "messages", "images", "prompt_token_ids", "seed", "completion", "group_id", "partial", "budget-stop")
        for mutation in cases:
            changed = copy.deepcopy(records)
            first, last = page_positions[0], page_positions[-1]
            if mutation == "missing":
                changed.pop(last)
            elif mutation == "duplicate":
                changed.insert(first, copy.deepcopy(changed[first]))
            elif mutation == "reordered":
                changed[first], changed[first + 1] = changed[first + 1], changed[first]
            elif mutation in ("group-before-pages", "tree-before-pages"):
                kind = "GROUP" if mutation == "group-before-pages" else "TREE"
                position = next(index for index in range(last + 1, len(changed)) if changed[index][0] == kind)
                changed.insert(last, changed.pop(position))
            elif mutation in ("messages", "images", "prompt_token_ids", "seed", "completion", "group_id"):
                event = changed[first][1]
                if mutation == "messages":
                    event["messages"][0]["content"][-1]["text"] += "changed"
                elif mutation == "images":
                    event["images"]["pixel_tensors"] = {}
                elif mutation == "prompt_token_ids":
                    event["prompt_token_ids"] = [999]
                elif mutation == "seed":
                    event["seed"] += 1
                elif mutation == "completion":
                    event["completion"]["token_ids"] = [999]
                else:
                    event["group_id"] = "other-page"
            elif mutation in ("partial", "budget-stop"):
                tree = copy.deepcopy(trees[1])
                tree.groups = [group for group in tree.groups if group.kind != "DIAG"]
                note = tree.notebook["rewarded_notebook"]
                note.update(status="BUDGET_STOP" if mutation == "budget-stop" else "FINAL_NOTE_BLOCKED",
                    bundles=[], page_calls=[] if mutation == "budget-stop" else note["page_calls"][:1])
                changed = [(kind, value) for kind, value in changed
                    if not (kind == "GROUP" and value["kind"] == "DIAG")]
                changed[-1] = ("TREE", json.loads(to_json(tree)))
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary, \
                    patch("psutil.pid_exists", return_value=False):
                parent, model = self.parent(Path(temporary), f, actors, trees, changed)
                if mutation:
                    with self.assertRaisesRegex(ValueError, "page|precede"):
                        capture.closed_parent(parent, actors, f.protocol, model, include_notebooks=True)
                else:
                    restored, evidence = capture.closed_parent(parent, actors, f.protocol, model, include_notebooks=True)
                    self.assertEqual([to_json(tree) for tree in restored], [to_json(tree) for tree in trees])
                    self.assertEqual(evidence["status"], "CLOSED")


if __name__ == "__main__":
    unittest.main()
