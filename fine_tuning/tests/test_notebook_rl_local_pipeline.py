"""Local full-chain contract checks, NOT pretrained-model learning or efficacy.

Retry texts are authored branch fixtures. Log probabilities, image forwards,
RL gradients, parameter changes and checkpoint reloads use a real tiny Qwen VLM.
No fixture text is a supervised target or a model-improvement measurement.
"""
from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from fine_tuning.modules import tensor_sha256
from fine_tuning.notebook_rl.checkpoint import capture_rng, digest, load_resume, make_optimizer, save
from fine_tuning.notebook_rl.engines import Completion, HFRolloutEngine, RolloutResult
from fine_tuning.notebook_rl.generators import graph
from fine_tuning.notebook_rl.policy import assert_base_unchanged, completion_logprobs, reference_logprobs
from fine_tuning.notebook_rl.prompts import load_prompts
from fine_tuning.notebook_rl.rollout import RolloutConfig, append_diagnoses, build_tree
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.notebook_rl.updates import optimizer_step
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier_service import VerifierService
from fine_tuning.tests.test_notebook_rl_checkpoint import fixture, fresh_tiny
from fine_tuning.tests.test_notebook_rl_engines import Processor
from fine_tuning.tests.test_training import tiny_model


class Characters:
    unk_token_id = -1

    def convert_tokens_to_ids(self, text):
        return 2

    def encode(self, text, **kwargs):
        return [ord(c)+32 for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i-32) for i in ids if i >= 32)


class ControlledActions:
    """Authored actions + actual policy likelihoods; no claim of model reasoning."""
    policy_version = 0

    def __init__(self, bundle, replies):
        self.bundle, self.replies = bundle, replies
        self.tokenizer, self.requests = Characters(), []

    def prepare(self, request):
        ids = request.prompt_token_ids
        if ids is None:
            text = request.messages[0]["content"][-1]["text"]
            ids = [4, 3, 3, 3, 3, 5]+self.tokenizer.encode(text)
        return SimpleNamespace(token_ids=ids)

    def model_inputs(self, request, completion, device):
        ids = torch.tensor([self.prepare(request).token_ids+completion], device=device)
        with request.images[0].resize((8, 8)) as resized:
            values = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float()/127.5-1
        patches = [values[:, y:y+2, x:x+2].unsqueeze(1).expand(3, 2, 2, 2).reshape(24)
                   for y in range(0, 8, 2) for x in range(0, 8, 2)]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "pixel_values": torch.stack(patches).to(device),
                "image_grid_thw": torch.tensor([[1, 4, 4]], device=device)}

    def sample(self, requests):
        request = requests[0]
        kind = "FINAL_NOTE" if ":FINAL_NOTE:" in request.request_id else request.request_id.split(":")[-2]
        texts = self.replies[kind]
        assert len(texts) == request.n
        result = []
        for text in texts:
            ids = self.tokenizer.encode(text)+[2]
            assert len(ids) <= request.max_new_tokens
            with torch.no_grad():
                lp = completion_logprobs(self.bundle, self.model_inputs(request, ids, torch.device("cpu")), len(ids))
            result.append(Completion(ids, lp[0].tolist(), "stop", text))
        self.requests.append(request)
        return [RolloutResult(request.request_id, self.prepare(request).token_ids, result)]


def snapshot(bundle):
    return {n: tensor_sha256(p) for n, p in bundle.model.named_parameters()}


def reload_worker(root):
    torch.set_num_threads(2)
    payload = json.loads((root/"reload.json").read_text())
    bundle, _, _, step, cursor = load_resume(root/payload["base"], root/"checkpoint", payload["protocol"],
        binding=payload["binding"], _test_policy_factory=fresh_tiny)
    (root/"reloaded.json").write_text(json.dumps({"snapshot": snapshot(bundle), "step": step, "cursor": cursor}))


class LocalPipelineTests(unittest.TestCase):
    def test_actual_random_tiny_generation_retries_without_fabricated_success(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory(prefix="notebook-local-generated-") as temporary:
            root = Path(temporary)
            problem = graph.generate(0, 3407, "train", "obs")
            write_split(root/"vault", [problem.actor], [problem.verifier], [problem.lineage])
            processor = Processor()
            processor.tokenizer.convert_tokens_to_ids = lambda token: 2
            engine = HFRolloutEngine(tiny_model(), processor, {"preprocessing": "hf_processor"}, 4096)
            prompts = load_prompts(Path(__file__).parents[1]/"notebook_rl/prompts_v1.yaml")
            cfg = RolloutConfig({1: 2, 2: 2, 3: 2}, {1: 1, 2: 1}, 3,
                                {1: 3, 2: 3, 3: 3}, 4096, False, 4, final_notebook=True)
            with Image.open(io.BytesIO(problem.image_bytes[0])) as image, \
                    VerifierService(root/"vault", forbidden_roots=[], timeout=60) as verifier:
                tree = build_tree(problem.actor, [image], engine, verifier, prompts, cfg, 0,
                                  "verified-any-visual-v1")
            self.assertEqual([g.kind for g in tree.groups], ["T1", "T2", "T3"])
            self.assertEqual(sum(len(g.samples) for g in tree.groups), 6)
            self.assertTrue(all(s.result.status in ("TRUNCATED", "INVALID_NO_FINAL")
                                for g in tree.groups for s in g.samples))
            self.assertEqual(tree.notebook["final_notebook"]["status"], "NO_RECOVERED_PATH")
            self.assertFalse(any(s.reward for g in tree.groups for s in g.samples))

    def test_real_routed_rl_on_retry_tree_final_notebook_and_fresh_process_reload(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory(prefix="notebook-local-chain-") as temporary:
            root = Path(temporary)
            base_path, _, protocol, binding = fixture(root)
            base = tiny_model()
            base.resize_token_embeddings(256, mean_resizing=False)
            base.bfloat16().save_pretrained(base_path, safe_serialization=True)
            protocol = copy.deepcopy(protocol)
            protocol["rollout"]["max_new_tokens"] = {1: 1024, 2: 1024, 3: 1024}
            protocol["rollout"]["final_notebook"] = True
            protocol["sha256"] = digest({k: v for k, v in protocol.items() if k != "sha256"})
            binding = binding | {"protocol_sha256": protocol["sha256"]}
            bundle = fresh_tiny(base_path, "MOD-RV", protocol)
            optimizer, scheduler = make_optimizer(bundle, protocol)
            prompts = load_prompts(Path(__file__).parents[1]/"notebook_rl/prompts_v1.yaml")
            cfg = RolloutConfig.from_protocol(protocol, "ALG-NB3")
            problem = graph.generate(0, 3407, "train", "obs")
            graph.self_check(problem)
            actors = [replace(problem.actor, problem_id=problem.actor.problem_id+suffix)
                      for suffix in ("-visual", "-nonvisual")]
            write_split(root/"vault", actors,
                [replace(problem.verifier, problem_id=a.problem_id) for a in actors],
                [replace(problem.lineage, problem_id=a.problem_id) for a in actors])
            image = Image.open(io.BytesIO(problem.image_bytes[0])).convert("RGB")
            self.addCleanup(image.close)
            facts = "\n".join(f"{key} = {value}" for key, value in problem.verifier.slots.items())
            wrong_key = problem.verifier.required_slots[0]
            wrong_facts = "\n".join(f"{key} = {float(value)+1 if key == wrong_key else value}"
                                     for key, value in problem.verifier.slots.items())
            gold = problem.verifier.gold
            wrong_answer = str(float(gold)+1)
            trace = []
            with VerifierService(root/"vault", forbidden_roots=[], timeout=60) as verifier:
                for step, actor in enumerate(actors):
                    wrong = f"Observed facts:\n{wrong_facts if step == 0 else facts}\nSolution:\nFinal answer: {wrong_answer}"
                    right = f"Error check:\nRecheck values and calculation.\nObserved facts:\n{facts}\nSolution:\nFinal answer: {gold}"
                    # The note intentionally claims the opposite cause: it must not control routing.
                    note = "Cause: NON_VISUAL\nWrong item: none\nLesson: Audit text is not routing evidence."
                    engine = ControlledActions(bundle, {"T1": [wrong]*8, "T2": [wrong]*4,
                        "T3": [right, wrong, right, wrong], "FINAL_NOTE": [note]})
                    engine.policy_version = step
                    tree = build_tree(actor, [image], engine, verifier, prompts, cfg, step, protocol["route_rule"])
                    self.assertEqual([g.kind for g in tree.groups], ["T1", "T2", "T2", "T3"])
                    self.assertEqual(tree.notebook["final_notebook"]["status"], "COMPLETED")
                    self.assertEqual(tree.route, "VISION_ON" if step == 0 else "VISION_OFF")
                    self.assertFalse(tree.notebook["final_notebook"]["used_for_loss"])
                    self.assertEqual(tree.notebook["routing"]["module_causality_certified"], False)
                    before = {g: [tensor_sha256(p) for p in ps] for g, ps in bundle.params.items()}
                    probe = engine.model_inputs(engine.requests[0], [75, 76, 2], torch.device("cpu"))
                    reference = reference_logprobs(bundle, probe, 3).clone()
                    with patch("torch.nn.functional.cross_entropy", side_effect=AssertionError("SFT forbidden")), \
                            patch("fine_tuning.train.AssistantCollator", side_effect=AssertionError("SFT forbidden")):
                        updated = optimizer_step(bundle, optimizer, scheduler, [tree], {actor.problem_id: [image]},
                            engine, protocol, "ALG-NB3", step=step, seed=3407)
                    self.assertEqual(updated["status"], "UPDATED_UNSYNCHRONIZED")
                    self.assertEqual(updated["masked_fraction"], 0)
                    for group in ("M", "L"):
                        self.assertGreater(updated["grad_norm"][group], 0)
                        self.assertNotEqual(before[group], [tensor_sha256(p) for p in bundle.params[group]])
                    if step == 0:
                        self.assertGreater(updated["grad_norm"]["V"], 0)
                        self.assertNotEqual(before["V"], [tensor_sha256(p) for p in bundle.params["V"]])
                    else:
                        self.assertIsNone(updated["grad_norm"]["V"])
                        self.assertEqual(before["V"], [tensor_sha256(p) for p in bundle.params["V"]])
                    self.assertTrue(torch.equal(reference, reference_logprobs(bundle, probe, 3)))
                    assert_base_unchanged(bundle)
                    trace.append({"tree": json.loads(to_json(tree)), "update": updated})
                # Diagnosis-reward learning is separate from the one-note audit and is class balanced.
                balanced = []
                cfg_diag = RolloutConfig.from_protocol(protocol, "ALG-NB+D")
                for step, actor in enumerate(actors):
                    wrong = f"Observed facts:\n{wrong_facts if step == 0 else facts}\nSolution:\nFinal answer: {wrong_answer}"
                    right = f"Error check:\nRecheck.\nObserved facts:\n{facts}\nSolution:\nFinal answer: {gold}"
                    engine = ControlledActions(bundle, {"T1": [wrong]*8, "T2": [right, wrong]*2,
                        "FINAL_NOTE": ["Cause: UNKNOWN\nWrong item: none\nLesson: Check independently."]})
                    engine.policy_version = 2
                    balanced.append(build_tree(actor, [image], engine, verifier, prompts, cfg_diag, 2,
                                               protocol["route_rule"]))
                diagnosis = ControlledActions(bundle, {})
                diagnosis.policy_version = 2
                original_sample = diagnosis.sample
                def diagnosis_actions(requests):
                    cause = "VISUAL_MISREAD" if requests[0].request_id.startswith(actors[0].problem_id+":") else "NON_VISUAL"
                    correct = f"Cause: {cause}\nWrong item: {wrong_key if cause == 'VISUAL_MISREAD' else 'none'}\nLesson: Recheck the evidence."
                    wrong_note = "Cause: FINALIZATION\nWrong item: none\nLesson: Incorrect diagnosis fixture."
                    diagnosis.replies["DIAG"] = [correct, wrong_note]*2
                    return original_sample(requests)
                with patch.object(diagnosis, "sample", side_effect=diagnosis_actions):
                    diag_report = append_diagnoses(balanced, {a.problem_id: [image] for a in actors},
                        diagnosis, verifier, prompts, cfg_diag)
                self.assertEqual(diag_report["generated_groups"], 2)
                self.assertEqual(diag_report["class_counts"], {"VISUAL": 1, "NON_VISUAL": 1, "FINALIZATION": 0})
                for row in balanced:
                    self.assertGreater(max(s.reward for s in row.groups[-1].samples), 0)
                    self.assertEqual(min(s.reward for s in row.groups[-1].samples), 0)
                with patch("torch.nn.functional.cross_entropy", side_effect=AssertionError("SFT forbidden")), \
                        patch("fine_tuning.train.AssistantCollator", side_effect=AssertionError("SFT forbidden")):
                    diag_update = optimizer_step(bundle, optimizer, scheduler, balanced,
                        {a.problem_id: [image] for a in actors}, diagnosis, protocol, "ALG-NB+D", step=2, seed=3407)
                self.assertEqual(diag_update["status"], "UPDATED_UNSYNCHRONIZED")
                self.assertEqual(diag_update["n_groups_by_kind"]["DIAG"], 2)
                trace.append({"diagnosis_batch": diag_report, "update": diag_update,
                              "trees": [json.loads(to_json(t)) for t in balanced]})
            cursor = {"epoch": 0, "position": 2, "discarded_problem_ids": [],
                      "order_sha256": digest([a.problem_id for a in actors])}
            save(bundle, optimizer, scheduler, 3, cursor, capture_rng(), root/"checkpoint",
                 protocol=protocol, binding=binding)
            expected = snapshot(bundle)
            (root/"reload.json").write_text(json.dumps({"base": base_path.name, "protocol": protocol, "binding": binding}))
            subprocess.run([sys.executable, "-m", "fine_tuning.tests.test_notebook_rl_local_pipeline", "--reload-worker", str(root)],
                           check=True, timeout=90, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            restored = json.loads((root/"reloaded.json").read_text())
            self.assertEqual(restored, {"snapshot": expected, "step": 3, "cursor": cursor})
            if destination := os.environ.get("MMDL_LOCAL_CONTRACT_EVIDENCE"):
                out = Path(destination)
                out.mkdir(parents=True, exist_ok=False)
                (out/"verification.json").write_text(json.dumps({"status": "LOCAL_TINY_CONTRACT_PASS",
                    "actual_tiny_optimizer_updates": 3, "fresh_process_reload_exact": True,
                    "sampling": "AUTHORED_CONTROL_FLOW_FIXTURE_NOT_MODEL_REASONING",
                    "pretrained_4b_verified": False, "model_improvement_verified": False,
                    "production_protocol_changed": False, "traces": trace}, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--reload-worker":
        reload_worker(Path(sys.argv[2]))
    else:
        unittest.main()
