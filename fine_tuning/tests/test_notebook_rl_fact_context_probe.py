"""CPU functional fixtures only; these do not establish model performance."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import checkpoint
from fine_tuning.notebook_rl import fact_context_probe as probe
from fine_tuning.notebook_rl.engines import Completion, Prepared, RolloutResult
from fine_tuning.notebook_rl.episode_notebook import audit_episode, public_transcript, solution_samples
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
from fine_tuning.notebook_rl.schema import ActorItem, Group, ProblemTree, Sample, VerifierItem, to_json
from fine_tuning.notebook_rl.verifier import Verifier


class FactProbeTests(unittest.TestCase):
    def setUp(self):
        self.protocol = build_protocol()
        self.actors = [ActorItem(f"fixture{i}", "gen-graph-v1", "graph", "obs", "Read the graph?", (),
            "int", tuple(f"{j}.png" for j in range(i + 1)), ("a" * 64,) * (i + 1)) for i in range(2)]
        self.verifier = object.__new__(Verifier)
        self.verifier.items = {a.problem_id: VerifierItem(a.problem_id, "int", "2", {}, (),
            {"edge[a,b]": "1"}, {"edge[a,b]": "number"}, ("edge[a,b]",), (), ()) for a in self.actors}
        self.trees = []
        for actor in self.actors:
            samples = []
            for i in range(4):
                text = 'Observed facts:\r\nedge[a,b] = 1\r\nSolution:\r\n<|im_start|>🙂\r\n\r\nFinal answer: 2'
                sample = Sample(f"{actor.problem_id}:s{i}", actor.problem_id, f"{actor.problem_id}:g", "T1", 1,
                    None, [1, 2], [3, 4], [-.1, -.2], "stop", text, i, 0,
                    self.verifier.grade(actor.problem_id, text, "stop", "obs"))
                samples.append(sample)
            group = Group(samples[0].group_id, "T1", actor.problem_id, "a" * 64, samples, True, None)
            self.trees.append(ProblemTree(actor.problem_id, 0, [group], "VISION_OFF", {},
                {"private_audit": "SECRET_PRIVATE_AUDIT"}))
        self.images = {a.problem_id: [object() for _ in a.image_paths] for a in self.actors}

    def requests(self):
        return probe.paired_requests(self.actors, self.trees, self.images, local_prompts(self.protocol))

    def test_lossless_all_siblings_no_private_audit_or_diag_and_balanced_order(self):
        diag = copy.deepcopy(self.trees[0].groups[0])
        diag.kind = "DIAG"
        diag.samples[0].text = "SECRET_DIAG"
        self.trees[0].groups.append(diag)
        requests = self.requests()
        self.assertEqual([row[1] for row in requests], [*probe.CONDITIONS, *reversed(probe.CONDITIONS)])
        for actor, condition, request in requests:
            self.assertEqual((request.n, request.max_new_tokens), (4, 1024))
            self.assertIs(request.images, self.images[actor.problem_id])
            self.assertIsNone(request.prompt_token_ids)
            self.assertEqual(len(request.messages), 1)
            text = request.messages[0]["content"][-1]["text"]
            self.assertNotIn("SECRET_DIAG", text)
            self.assertNotIn("SECRET_PRIVATE_AUDIT", text)
            self.assertNotIn("<|im_start|>", text)
            actual = json.loads(text.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
            expected = public_transcript(solution_samples(self.trees[self.actors.index(actor)]))
            self.assertEqual(actual, expected if condition == "WITH_ALL_HISTORY" else [])
            if actual:
                self.assertEqual(len(actual), 4)
                self.assertEqual("".join(line["text"] for line in actual[0]["lines"]),
                                 self.trees[self.actors.index(actor)].groups[0].samples[0].text)
        for first, second in (requests[:2], requests[2:]):
            self.assertEqual(first[2].seeds, second[2].seeds)
            self.assertEqual(len(set(first[2].seeds)), 4)

    def parent(self, root):
        parent, model = root / "parent", root / "model"
        parent.mkdir()
        actor_file = root / "actors.json"
        actor_file.write_text(json.dumps([asdict(a) for a in self.actors]))
        preparation = {"execution_actors": {"path": str(actor_file),
            "file_sha256": checkpoint.file_hash(actor_file)}}
        preparation_path = root / "preparation.json"
        probe.save(preparation_path, preparation)
        launch = {"pid": 987654321, "protocol": self.protocol,
            "problem_ids": [a.problem_id for a in self.actors], "model_path": str(model),
            "standalone_base": True, "preparation": str(preparation_path),
            "binding": {"protocol_sha256": self.protocol["sha256"],
                "prompts_sha256": self.protocol["prompts_sha256"],
                "data_sha256": checkpoint.digest(preparation), "code_sha256": "b" * 64}}
        result = {"status": "NO_NOTEBOOK_LEARNING_SIGNAL", "seconds": 1.,
                  "attempted_ids": launch["problem_ids"], "optimizer_updates": 0}
        probe.save(parent / "launch.json", launch)
        probe.save(parent / "result.json", result)
        for i, tree in enumerate(self.trees):
            probe.save(parent / f"raw-{2*i:04d}.json", {"kind": "GROUP", "raw": asdict(tree.groups[0])})
            probe.save(parent / f"raw-{2*i+1:04d}.json", {"kind": "TREE", "raw": json.loads(to_json(tree))})
        return parent, model

    def test_closed_failed_learning_parent_accepted_and_attempts_freshly_audited(self):
        with tempfile.TemporaryDirectory() as temporary, patch("psutil.pid_exists", return_value=False):
            parent, model = self.parent(Path(temporary))
            trees, evidence = probe.closed_parent(parent, self.actors, self.protocol, model)
            self.assertEqual(evidence["status"], "CLOSED")
            self.assertEqual(len(evidence["files"]), 8)
            for tree, actor in zip(trees, self.actors, strict=True):
                self.assertTrue(audit_episode(tree, actor, self.verifier)["complete_episode"])
            trees[0].groups[0].samples[0].text = "tampered"
            with self.assertRaisesRegex(ValueError, "fresh verifier"):
                audit_episode(trees[0], self.actors[0], self.verifier)
            trees[1].groups[0].usable = False
            trees[1].groups[0].reason = "functional fixture: missing samples"
            self.assertFalse(audit_episode(trees[1], self.actors[1], self.verifier)["complete_episode"])

    def test_closed_parent_binds_minimum_execution_contract_and_preparation(self):
        from fine_tuning.notebook_rl.local_run import VERIFIED_UPDATE_EXPLORATION, minimum_execution_contract

        self.protocol = build_protocol(verified_note_synthesis_credit=True, context_limit=24576)
        # The contract is unchanged across the explicitly recorded historical parent protocol.
        self.protocol.pop("notebook_repair_parent_policy")
        self.protocol.pop("sha256")
        self.protocol["sha256"] = checkpoint.digest(self.protocol)
        for mutation in (None, "preparation-only", "contract-hash", "contract", "missing-contract", "boolean-limit"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                parent, model = self.parent(Path(temporary))
                launch = probe.read(parent / "launch.json")
                execution = minimum_execution_contract(VERIFIED_UPDATE_EXPLORATION,
                    build_protocol(verified_note_synthesis_credit=True, context_limit=24576))
                launch.update(execution_contract=execution, execution_contract_sha256=checkpoint.digest(execution))
                if mutation != "preparation-only":
                    launch["binding"]["data_sha256"] = checkpoint.digest({
                        "preparation_sha256": checkpoint.digest(probe.read(Path(launch["preparation"]))),
                        "execution_contract_sha256": checkpoint.digest(execution)})
                if mutation == "contract-hash":
                    launch["execution_contract_sha256"] = "changed"
                if mutation == "contract":
                    launch["execution_contract"]["max_optimizer_updates"] = 2
                if mutation == "missing-contract":
                    del launch["execution_contract"]
                if mutation == "boolean-limit":
                    launch["execution_contract"]["max_optimizer_updates"] = True
                    launch["execution_contract_sha256"] = checkpoint.digest(launch["execution_contract"])
                    launch["binding"]["data_sha256"] = checkpoint.digest({
                        "preparation_sha256": checkpoint.digest(probe.read(Path(launch["preparation"]))),
                        "execution_contract_sha256": launch["execution_contract_sha256"]})
                (parent / "launch.json").write_bytes(probe.wire(launch))
                with patch("psutil.pid_exists", return_value=False):
                    if mutation:
                        with self.assertRaises(ValueError):
                            probe.closed_parent(parent, self.actors, self.protocol, model)
                    else:
                        trees, _ = probe.closed_parent(parent, self.actors, self.protocol, model)
                        self.assertEqual(len(trees), 2)

    def test_live_mismatched_incomplete_and_tampered_parent_rejected(self):
        for mutation in ("live", "protocol", "ids", "missing_tree", "group", "incomplete", "actors",
                         "update_on_failed_parent", "missing_update_count", "boolean_update_count"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                parent, model = self.parent(Path(temporary))
                target = parent / "launch.json"
                launch = probe.read(target)
                if mutation == "protocol":
                    launch["protocol"]["context_limit"] = 8
                if mutation == "ids":
                    launch["problem_ids"].reverse()
                if mutation == "actors":
                    self.actors.reverse()
                if mutation == "missing_tree":
                    (parent / "raw-0003.json").unlink()
                if mutation == "group":
                    path = parent / "raw-0000.json"
                    row = probe.read(path)
                    row["raw"]["samples"][0]["text"] = "changed"
                    path.write_bytes(probe.wire(row))
                if mutation == "incomplete":
                    path = parent / "result.json"
                    row = probe.read(path) | {"status": "INCOMPLETE"}
                    path.write_bytes(probe.wire(row))
                if mutation in ("update_on_failed_parent", "missing_update_count", "boolean_update_count"):
                    path = parent / "result.json"
                    row = probe.read(path)
                    if mutation == "missing_update_count":
                        del row["optimizer_updates"]
                    else:
                        row["optimizer_updates"] = 1 if mutation == "update_on_failed_parent" else False
                    path.write_bytes(probe.wire(row))
                target.write_bytes(probe.wire(launch))
                with patch("psutil.pid_exists", return_value=mutation == "live"):
                    with self.assertRaises(ValueError):
                        probe.closed_parent(parent, self.actors, self.protocol, model)
                if mutation == "actors":
                    self.actors.reverse()

    def test_layout_parent_binds_diag_and_observation_journals_to_tree(self):
        with tempfile.TemporaryDirectory() as temporary, patch("psutil.pid_exists", return_value=False):
            parent, model = self.parent(Path(temporary))
            tree_path = parent / "raw-0001.json"
            row = probe.read(tree_path)
            diag = copy.deepcopy(row["raw"]["groups"][0])
            diag.update(group_id="fixture-diag", kind="DIAG")
            for index, sample in enumerate(diag["samples"]):
                sample.update(sample_id=f"diag{index}", group_id="fixture-diag", kind="DIAG", diag_reward=0.)
            row["raw"]["groups"].append(diag)
            observation = {"request_id": "fixture-reread", "completion": {"text": "raw reread"}}
            row["raw"]["notebook"]["rewarded_notebook"] = {"fresh_observation": observation}
            tree_path.write_bytes(probe.wire(row))
            with self.assertRaisesRegex(ValueError, "TREE/GROUP"):
                probe.closed_parent(parent, self.actors, self.protocol, model, include_notebooks=True)
            probe.save(parent / "raw-0004.json", {"kind": "GROUP", "raw": diag})
            with self.assertRaisesRegex(ValueError, "observation journal"):
                probe.closed_parent(parent, self.actors, self.protocol, model, include_notebooks=True)
            observation_path = parent / "raw-0005.json"
            probe.save(observation_path, {"kind": "OBSERVATION", "raw": observation | {"phase": "COMPLETED"}})
            probe.closed_parent(parent, self.actors, self.protocol, model, include_notebooks=True)
            changed = copy.deepcopy(diag)
            changed["samples"][0]["text"] += "tampered"
            (parent / "raw-0004.json").write_bytes(probe.wire({"kind": "GROUP", "raw": changed}))
            with self.assertRaisesRegex(ValueError, "TREE/GROUP"):
                probe.closed_parent(parent, self.actors, self.protocol, model, include_notebooks=True)
            (parent / "raw-0004.json").write_bytes(probe.wire({"kind": "GROUP", "raw": diag}))
            observation_path.write_bytes(probe.wire({"kind": "OBSERVATION", "raw": observation | {
                "phase": "COMPLETED", "completion": {"text": "tampered reread"}}}))
            with self.assertRaisesRegex(ValueError, "observation journal"):
                probe.closed_parent(parent, self.actors, self.protocol, model, include_notebooks=True)

    def test_composite_draft_raw_lifecycle_bound_separately_from_observation(self):
        self.protocol["notebook_draft_policy"] = "fixture-public-drafts"
        self.protocol["sha256"] = checkpoint.digest({k: v for k, v in self.protocol.items() if k != "sha256"})
        for tree in self.trees:
            tree.notebook["rewarded_notebook"] = {"notebook_draft_policy": "fixture-public-drafts"}
        call = {"request_id": "fixture-draft", "target_index": 1, "prompt_token_ids": [10, 11],
                "completions": [{"token_ids": [12], "logprobs": [-.1], "finish_reason": "length",
                                 "text": "unverified raw draft"}]}
        drafts = {"policy": "fixture-public-drafts", "status": "COMPLETE", "calls": [call]}
        pid = self.actors[0].problem_id
        self.trees[0].notebook["rewarded_notebook"]["pair_drafts"] = drafts
        events = [
            {"problem_id": pid, "phase": "STARTED", "call": {k: v for k, v in call.items() if k != "completions"}},
            {"problem_id": pid, "phase": "GENERATED", "request_id": "fixture-draft", "target_index": 1,
             "outputs": [{"request_id": "fixture-draft", "prompt_token_ids": [10, 11],
                          "completions": call["completions"]}]},
            {"problem_id": pid, "phase": "COMPLETED", "call": call},
            {"problem_id": pid, "phase": "DRAFTS_COMPLETED", "record": drafts}]
        for mutation in (None, "missing-root", "missing-raw", "raw-text", "order", "duplicate",
                         "actor", "policy", "extra-observation"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary, \
                    patch("psutil.pid_exists", return_value=False):
                parent, model = self.parent(Path(temporary))
                rows = [{"kind": "DRAFT", "raw": copy.deepcopy(event)} for event in events]
                if mutation == "missing-root":
                    rows.pop()
                elif mutation == "missing-raw":
                    rows.pop(1)
                elif mutation == "raw-text":
                    rows[1]["raw"]["outputs"][0]["completions"][0]["text"] = "silently fixed"
                elif mutation == "order":
                    rows[1], rows[2] = rows[2], rows[1]
                elif mutation == "duplicate":
                    rows.append(copy.deepcopy(rows[-1]))
                elif mutation == "actor":
                    rows[-1]["raw"]["problem_id"] = self.actors[1].problem_id
                elif mutation == "policy":
                    tree = probe.read(parent / "raw-0001.json")
                    tree["raw"]["notebook"]["rewarded_notebook"].pop("notebook_draft_policy")
                    (parent / "raw-0001.json").write_bytes(probe.wire(tree))
                elif mutation == "extra-observation":
                    rows[2] = {"kind": "OBSERVATION", "raw": {"phase": "COMPLETED",
                                                                          "request_id": "draft-as-reread"}}
                for index, row in enumerate(rows, 4):
                    probe.save(parent / f"raw-{index:04d}.json", row)
                if mutation is None:
                    loaded, _ = probe.closed_parent(parent, self.actors, self.protocol, model, include_notebooks=True)
                    self.assertEqual(loaded[0].notebook["rewarded_notebook"]["pair_drafts"], drafts)
                else:
                    with self.assertRaisesRegex(ValueError, "draft|observation journal"):
                        probe.closed_parent(parent, self.actors, self.protocol, model, include_notebooks=True)

    def test_unsafe_output_paths_and_collisions_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("model", "data", "parent"):
                protected = root / name
                protected.mkdir()
                for output in (protected, protected / "probe", root):
                    with self.assertRaisesRegex(ValueError, "collision|protected"):
                        probe.output_path(output, protected)
            alias = root / "alias"
            alias.symlink_to(root / "parent", target_is_directory=True)
            with self.assertRaises(ValueError):
                probe.output_path(alias / "probe", root / "parent")
            self.assertEqual(probe.output_path(root / "new", root / "parent"), root / "new")

    def engine(self, *, excess=False):
        class Engine:
            def prepare(inner, request):
                return Prepared([1] * (16384 if excess else 2),
                    {"pixel_values": torch.zeros((1, 2)), "image_grid_thw": torch.tensor([[1, 1, 1]])},
                    [SimpleNamespace(size=(28, 28))], "same-image-key")

            def sample(inner, requests):
                request = requests[0]
                return [RolloutResult(request.request_id, [1, 1], [Completion([2], [-.1],
                    "length" if i == 3 else "stop", f"Observed facts:\nedge[a,b] = {2 if i == 0 else 1}\nSolution:\n")
                    for i in range(4)])]
        return Engine()

    def test_facts_scoring_preserves_wrong_facts_and_is_not_answer_accuracy(self):
        journal = []
        rows = probe.collect(self.requests(), self.engine(), self.verifier,
                             lambda *row: journal.append(row), time.monotonic() + 10)
        self.assertEqual(len(rows), 4)
        self.assertEqual([r[0] for r in journal], ["CALL_STARTED", "CALL_COMPLETED"] * 4)
        for row in rows:
            self.assertEqual(row["samples"][0]["factual_status"], "WRONG_FACTS")
            self.assertEqual(row["samples"][0]["slots"]["required_correct"], 0)
            self.assertEqual(row["samples"][1]["factual_status"], "ALL_REQUIRED_FACTS_VERIFIED")
            self.assertEqual(row["samples"][3]["finish_reason"], "length")
            self.assertNotIn("reward_answer", row["samples"][0])
            self.assertNotIn("extracted", row["samples"][0])
        contract = probe.contract(self.protocol)
        for key in ("answer_accuracy_measured", "self_improvement_verified", "module_causality_certified", "training_allowed"):
            self.assertIs(contract[key], False)
        self.assertEqual(contract["optimizer_updates"], 0)
        self.assertIn("public outcome labels", contract["history_intervention"])
        self.assertIs(contract["history_text_only_causal_effect_certified"], False)

    def test_determinism_configured_before_readiness_or_model_work(self):
        events = []
        def readiness(*_):
            events.append("readiness")
            raise LookupError("functional fixture stops before model work")
        with patch.dict(os.environ, MMDL_FACT_PROBE_SUPERVISOR=str(os.getppid())), patch(
                "fine_tuning.notebook_rl.policy.enable_deterministic_execution",
                side_effect=lambda: events.append("deterministic")), patch(
                "fine_tuning.notebook_rl.local_run.cache_readiness", side_effect=readiness):
            with self.assertRaisesRegex(LookupError, "functional fixture"):
                probe.capture(*(Path("/unused") for _ in range(5)))
        self.assertEqual(events, ["deterministic", "readiness"])

    def test_context_and_deadline_fail_before_generation_no_truncation(self):
        for excess, deadline in ((True, time.monotonic() + 10), (False, 0)):
            engine = self.engine(excess=excess)
            with patch.object(engine, "sample") as sample:
                with self.assertRaises((ValueError, TimeoutError)):
                    probe.collect(self.requests(), engine, self.verifier, lambda *_: None, deadline)
                sample.assert_not_called()

    def test_actual_token_text_and_initial_image_prompt_binding(self):
        engine = SimpleNamespace(prepare=lambda _: SimpleNamespace(token_ids=[1, 2]),
            tokenizer=SimpleNamespace(decode=lambda *_, **__: self.trees[0].groups[0].samples[0].text))
        probes = [SimpleNamespace(request=object()) for _ in self.trees]
        probe.bind_history_tokens(self.trees, engine, probes)
        self.trees[1].groups[0].samples[0].completion_token_ids = [99]
        engine.tokenizer.decode = lambda ids, **_: "changed" if ids == [99] else self.trees[0].groups[0].samples[0].text
        with self.assertRaisesRegex(ValueError, "actual token"):
            probe.bind_history_tokens(self.trees, engine, probes)
        self.trees[1].groups[0].samples[0].completion_token_ids = [3, 4]
        self.trees[0].groups[0].samples[0].prompt_token_ids = [77]
        with self.assertRaisesRegex(ValueError, "actual token"):
            probe.bind_history_tokens(self.trees, engine, probes)

    def test_supervisor_fixed_timeout_and_owned_group_cleanup(self):
        command = ["--preparation", "/prep", "--model-path", "/model", "--cache-check", "/cache",
                   "--parent", "/parent", "--output", "/output"]
        with patch.object(probe.subprocess, "Popen") as popen, patch("fine_tuning.budget.terminate_group") as stop:
            process = popen.return_value
            process.wait.side_effect = subprocess.TimeoutExpired("fixture", 3570)
            self.assertEqual(probe.main(command), 124)
            process.wait.assert_called_once_with(timeout=3570)
            stop.assert_called_once_with(process, grace_seconds=10)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(probe.MAX_SECONDS, 3600)

    def test_notebook_layout_option_propagates_to_owned_worker(self):
        command = ["--preparation", "/prep", "--model-path", "/model", "--cache-check", "/cache",
                   "--parent", "/parent", "--output", "/output", "--notebook-layout"]
        with patch.object(probe.subprocess, "Popen") as popen, patch("fine_tuning.budget.terminate_group") as stop:
            process = popen.return_value
            process.wait.return_value = 0
            self.assertEqual(probe.main(command), 0)
            self.assertIn("--notebook-layout", popen.call_args.args[0])
            stop.assert_called_once_with(process, grace_seconds=10)
        with patch.object(probe, "capture") as capture:
            self.assertEqual(probe.main(command + ["--worker"]), 0)
            capture.assert_called_once_with(Path("/prep"), Path("/model"), Path("/cache"),
                                            Path("/parent"), Path("/output"), notebook_layout=True)

    def test_trained_notebook_option_propagates_and_cannot_mix_interventions(self):
        command = ["--preparation", "/prep", "--model-path", "/model", "--cache-check", "/cache",
                   "--parent", "/parent", "--output", "/output", "--trained-notebook-reload", "/reload"]
        with patch.object(probe.subprocess, "Popen") as popen, patch("fine_tuning.budget.terminate_group") as stop:
            popen.return_value.wait.return_value = 0
            self.assertEqual(probe.main(command), 0)
            child = popen.call_args.args[0]
            self.assertEqual(child[child.index("--trained-notebook-reload") + 1], "/reload")
            popen.return_value.wait.assert_called_once_with(timeout=3570)
            stop.assert_called_once_with(popen.return_value, grace_seconds=10)
        with patch.object(probe, "capture") as capture:
            self.assertEqual(probe.main(command + ["--worker"]), 0)
            capture.assert_called_once_with(Path("/prep"), Path("/model"), Path("/cache"),
                Path("/parent"), Path("/output"), trained_notebook_reload=Path("/reload"))
        for option in ("--notebook-layout", "--notebook-drafts"):
            with patch.object(probe.subprocess, "Popen") as popen, patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    probe.main(command + [option])
                popen.assert_not_called()
        with self.assertRaisesRegex(ValueError, "one diagnostic"):
            probe.capture(*(Path("/unused") for _ in range(5)), notebook_layout=True,
                          trained_notebook_reload=Path("/reload"))

    def test_trained_parent_requires_real_update_gate_and_closed_reload(self):
        gate = {"launch": {"model_path": "/model", "cache_check": "/cache"}}
        with patch("fine_tuning.notebook_rl.local_train.parent_gate", return_value=gate) as check, patch.object(
                probe, "read", return_value={"pid": 987654321}), patch("psutil.pid_exists", return_value=False):
            self.assertEqual(probe.trained_parent(Path("/parent"), Path("/reload"),
                                                Path("/model"), Path("/cache")), gate)
            check.assert_called_once_with(Path("/parent"), Path("/reload"), require_notebook_learning=False)
        with patch("fine_tuning.notebook_rl.local_train.parent_gate", side_effect=ValueError("not updated")), patch.object(
                probe, "read") as read:
            with self.assertRaisesRegex(ValueError, "not updated"):
                probe.trained_parent(*(Path("/unused") for _ in range(4)))
            read.assert_not_called()
        for mutation in ("live", "bad_pid", "model", "cache"):
            with self.subTest(mutation=mutation), patch(
                    "fine_tuning.notebook_rl.local_train.parent_gate", return_value=gate), patch.object(
                    probe, "read", return_value={"pid": True if mutation == "bad_pid" else 987654321}), patch(
                    "psutil.pid_exists", return_value=mutation == "live"):
                with self.assertRaisesRegex(ValueError, "closed reload"):
                    probe.trained_parent(Path("/parent"), Path("/reload"),
                        Path("/different" if mutation == "model" else "/model"),
                        Path("/different" if mutation == "cache" else "/cache"))

    def test_trained_counts_separate_recovery_regression_and_do_not_authorize_training(self):
        rows = [{"samples": [{"baseline_notebook_check": {"reward": a},
                              "notebook_check": {"reward": b}} for a, b in ((0., 1.), (1., 0.), (0., 0.), (1., 1.))]}]
        result = probe.trained_note_counts(rows)
        self.assertEqual({k: result[k] for k in ("notes", "before_strict_valid", "after_strict_valid", "recovered", "regressed")},
                         {"notes": 4, "before_strict_valid": 2, "after_strict_valid": 2, "recovered": 1, "regressed": 1})
        for name in ("heldout", "strict_training_expansion_authorized", "notebook_reward_effect_isolated"):
            self.assertIs(result[name], False)
        self.assertEqual(probe.trained_note_counts([])["notes"], 0)
        for bad in (.25, True, float("nan")):
            rows[0]["samples"][0]["notebook_check"]["reward"] = bad
            with self.assertRaisesRegex(ValueError, "strict binary"):
                probe.trained_note_counts(rows)

    def test_trained_capture_loads_verified_weights_before_same_history_generation(self):
        from fine_tuning.notebook_rl import notebook_layout_probe as layout

        protocol = build_protocol(verified_note_components=True)
        hashes = {"V": {"vision": "v"}, "M": {"merger": "m"}, "L": {"language": "l"}}
        expected = {"trainable_sha256": hashes, "step": 1,
                    "frozen_base_sha256": "base", "quantization": {"kind": "nf4-fixture"}}
        for mutation in (None, "weights", "base", "quantization", "step"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                root = Path(temporary)
                parent, data = root / "parent", root / "data"
                model, cache, reload = (root / name for name in ("model", "cache", "reload"))
                output, preparation = root / "output", root / "preparation.json"
                launch = {"protocol": protocol, "binding": {"historical": "unchanged"}}
                original = copy.deepcopy(expected)
                if mutation == "step":
                    original["step"] = 2
                probe.save(parent / "reload_expected.json", original)
                probe.save(data / "manifest.yaml", {"split_manifests": {"train": "vault"}})
                bundle = SimpleNamespace(base_sha256="wrong" if mutation == "base" else "base",
                    inventory={"quantization": {} if mutation == "quantization" else expected["quantization"]})
                old_trees = [to_json(tree) for tree in self.trees]
                request = SimpleNamespace(condition="CURRENT_JSON")
                skipped = [{"problem_id": self.actors[0].problem_id, "reason": "NO_VERIFIED_RECOVERY"}]
                rows = [{"samples": [{"baseline_notebook_check": {"reward": 0.},
                                       "notebook_check": {"reward": 1.}} for _ in range(4)]}]
                def patched(target, **kwargs):
                    return stack.enter_context(patch(target, **kwargs))
                stack.enter_context(patch.dict(os.environ, MMDL_FACT_PROBE_SUPERVISOR=str(os.getppid())))
                patched("fine_tuning.notebook_rl.policy.enable_deterministic_execution", return_value={})
                patched("fine_tuning.notebook_rl.fact_context_probe.trained_parent",
                        return_value={"launch": launch, "files": []})
                patched("fine_tuning.notebook_rl.local_run.cache_readiness",
                        return_value={"original_model_root": str(root / "original")})
                patched("fine_tuning.notebook_rl.local_run.read_preparation",
                        return_value=({"build_root": str(data)}, {}, self.actors))
                patched("fine_tuning.notebook_rl.local_run.minimal_actors", return_value=self.actors)
                closed = patched("fine_tuning.notebook_rl.fact_context_probe.closed_parent",
                        return_value=(self.trees, {"files": {}}))
                patched("fine_tuning.notebook_rl.approval.code_hashes", return_value={"source": "same"})
                patched("fine_tuning.notebook_rl.local_run._install_actor_guards", return_value=[])
                patched("fine_tuning.notebook_rl.quantized_cache.block_original_access",
                        return_value={"denied_original_access_attempts": 0})
                patched("fine_tuning.notebook_rl.verifier_service.VerifierService",
                        return_value=nullcontext(SimpleNamespace(metadata={"vault_manifest_sha256": "vault"})))
                patched("fine_tuning.notebook_rl.episode_notebook.audit_episode", return_value={"complete_episode": True})
                load = patched("fine_tuning.notebook_rl.checkpoint.load_eval", return_value=bundle)
                patched("fine_tuning.notebook_rl.local_run.parameter_hashes",
                        return_value={} if mutation == "weights" else hashes)
                engine = patched("fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine")
                patched("fine_tuning.notebook_rl.dataset.load_images", return_value=[])
                patched("fine_tuning.notebook_rl.local_run.make_probes", return_value=[])
                binding = patched("fine_tuning.notebook_rl.fact_context_probe.bind_history_tokens")
                sync = patched("fine_tuning.notebook_rl.trainer.synchronize", return_value={})
                patched("fine_tuning.notebook_rl.notebook_layout_probe.paired_requests",
                        return_value={"requests": [SimpleNamespace(condition="NUMBERED_LINES"), request], "skipped": skipped})
                collect = patched("fine_tuning.notebook_rl.notebook_layout_probe.collect", return_value=rows)
                unchanged = patched("fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged")
                if mutation:
                    with self.assertRaisesRegex(ValueError, "loaded model differs"):
                        probe.capture(preparation, model, cache, parent, output, trained_notebook_reload=reload)
                    engine.assert_not_called()
                    collect.assert_not_called()
                    self.assertEqual(probe.read(output / "result.json")["status"], "FAILED")
                else:
                    result = probe.capture(preparation, model, cache, parent, output, trained_notebook_reload=reload)
                    self.assertEqual(result["status"], "DIAGNOSTIC_CAPTURE_COMPLETE")
                    self.assertEqual((result["calls"], result["samples"], result["optimizer_updates"]), (1, 4, 0))
                    self.assertIs(result["training_allowed"], False)
                    self.assertIs(result["self_improvement_verified"], False)
                    self.assertEqual(result["skipped"], skipped)
                    self.assertEqual(engine.call_args.kwargs["policy_version"], 1)
                    self.assertEqual(sync.call_args.args[5], 1)
                    binding.assert_called_once()
                    self.assertEqual(collect.call_args.args[0], [request])
                    self.assertEqual(collect.call_args.kwargs, {"generation_policy_version": 1})
                    unchanged.assert_called_once_with(bundle)
                    self.assertEqual(probe.read(output / "launch.json")["diagnostic"], layout.trained_contract(protocol))
                load.assert_called_once()
                self.assertEqual(load.call_args.args, (model, parent / "checkpoint"))
                self.assertEqual(load.call_args.kwargs["binding"], launch["binding"])
                self.assertEqual(load.call_args.kwargs["policy_factory"].keywords, {"device_index": 0})
                closed.assert_called_once_with(parent, self.actors, protocol, model, include_notebooks=True)
                self.assertEqual([to_json(tree) for tree in self.trees], old_trees)

    def test_notebook_drafts_option_propagates_and_cannot_mix_interventions(self):
        command = ["--preparation", "/prep", "--model-path", "/model", "--cache-check", "/cache",
                   "--parent", "/parent", "--output", "/output", "--notebook-drafts"]
        with patch.object(probe.subprocess, "Popen") as popen, patch("fine_tuning.budget.terminate_group") as stop:
            process = popen.return_value
            process.wait.return_value = 0
            self.assertEqual(probe.main(command), 0)
            self.assertIn("--notebook-drafts", popen.call_args.args[0])
            self.assertNotIn("--notebook-layout", popen.call_args.args[0])
            process.wait.assert_called_once_with(timeout=3570)
            stop.assert_called_once_with(process, grace_seconds=10)
        with patch.object(probe, "capture") as capture:
            self.assertEqual(probe.main(command + ["--worker"]), 0)
            capture.assert_called_once_with(Path("/prep"), Path("/model"), Path("/cache"),
                                            Path("/parent"), Path("/output"), notebook_drafts=True)
        with patch.object(probe.subprocess, "Popen") as popen, self.assertRaises(SystemExit) as raised:
            probe.main(command + ["--notebook-layout"])
        self.assertEqual(raised.exception.code, 2)
        popen.assert_not_called()
        with self.assertRaisesRegex(ValueError, "choose one diagnostic"):
            probe.capture(*(Path("/unused") for _ in range(5)), notebook_layout=True, notebook_drafts=True)

    def test_actual_timeout_reaps_separate_verifier_session_including_startup(self):
        import psutil
        from fine_tuning.notebook_rl.vault import write_split
        from fine_tuning.tests.test_notebook_rl_schema import records

        script = '''
import json, os, selectors, signal, subprocess, sys
from contextlib import ExitStack
from pathlib import Path
from fine_tuning.notebook_rl import fact_context_probe as probe
from fine_tuning.notebook_rl.verifier_service import VerifierService

root, stage = Path(sys.argv[1]), sys.argv[2]
args = ["--preparation", str(root), "--model-path", str(root), "--cache-check", str(root),
        "--parent", str(root), "--output", str(root)]
if "--worker" in sys.argv:
    def identities(service):
        print(json.dumps({"worker": os.getpid(), "verifier": service._process.pid,
            "worker_session": os.getsid(0), "verifier_session": os.getsid(service._process.pid)}), flush=True)
    if stage == "startup":
        def wait_for_signal(service, *_):
            identities(service)
            signal.pause()
        VerifierService._response = wait_for_signal
    def fixture(*_, **__):
        with ExitStack() as stack:
            service = stack.enter_context(VerifierService(root, forbidden_roots=[], timeout=30))
            identities(service)
            signal.pause()
    probe.capture = fixture
    args.append("--worker")
else:
    real_popen = subprocess.Popen
    def child(command, **kwargs):
        process = real_popen([sys.executable, "-c", sys.argv[3], str(root), stage, sys.argv[3], "--worker"],
            stdout=subprocess.PIPE, **kwargs)
        real_wait = process.wait
        def wait_for_stage(timeout):
            process.wait = real_wait
            assert timeout == 3570
            # Arm the short test timeout only after the verifier reaches the intended phase.
            with process.stdout, selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                if not selector.select(20):
                    raise TimeoutError("fixture worker did not reach verifier stage")
                identity = process.stdout.readline()
                if not identity:
                    raise RuntimeError("fixture worker exited before verifier stage")
                print(identity.decode().rstrip(), flush=True)
            return real_wait(timeout=1)
        process.wait = wait_for_stage
        return process
    probe.subprocess.Popen = child
raise SystemExit(probe.main(args))
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "vault"
            actor, verifier, lineage, *_ = records()
            write_split(root, [actor], [verifier], [lineage])
            for stage in ("ready", "startup"):
                with self.subTest(stage=stage):
                    completed = subprocess.run([sys.executable, "-c", script, str(root), stage, script],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=50)
                    self.assertEqual(completed.returncode, 124, completed.stderr)
                    self.assertTrue(completed.stdout.splitlines(), completed.stderr)
                    identity = json.loads(completed.stdout.splitlines()[0])
                    self.assertNotEqual(identity["worker_session"], identity["verifier_session"])
                    self.assertFalse(psutil.pid_exists(identity["worker"]), completed.stderr)
                    self.assertFalse(psutil.pid_exists(identity["verifier"]), completed.stderr)


if __name__ == "__main__":
    unittest.main()
