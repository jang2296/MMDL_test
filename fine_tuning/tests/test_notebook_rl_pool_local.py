"""Opt-in AI-reviewed local pool: approval stage, label-blind preparation and the generic pool evaluator.

Authored CPU fixtures, scripted engines and a tiny real Qwen/PEFT policy only: never Qwen-4B
behaviour, real exclusion eligibility of a pool, or evidence that a model improved.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from fine_tuning.notebook_rl import (approval as a, checkpoint, compact_note, local_data, local_eval, local_pool_eval,
                                     local_profile, local_run)
from fine_tuning.notebook_rl.dataset import SPLITS, _bytes, _write, build_splits, generate_pool
from fine_tuning.notebook_rl.engines import Completion, Prepared, RolloutResult
from fine_tuning.notebook_rl.generators.common import GeneratedProblem
from fine_tuning.notebook_rl.schema import ActorItem, Lineage, VerifierItem
from fine_tuning.notebook_rl.vault import read_actors, read_lineage

HF_TRAIN = ("hf-fixture-a", "hf-fixture-b", "hf-fixture-c", "hf-fixture-d")
HF_HELD = ("hf-fixture-e", "hf-fixture-f")
PLAIN = "gen-statics-v1"
SLOTLESS = (*HF_TRAIN, *HF_HELD, PLAIN)
REVISION = "a" * 40
CHOICES = ("red", "green", "blue", "gold")
RIGHT = "Solution:\nThe marked region is green.\nFinal answer: B"
WRONG = "Solution:\nThe marked region is blue.\nFinal answer: C"
NO_FINAL = "Solution:\nI cannot decide from the image."
CHECK = 'ErrorCheck: {"version":"repair-check-v1","claims":[],"uncertain_refs":[]}\nRevisedSolution:\n'
DEFAULT_SHA = "89181dfc7e8e4d306c03d0b340f198a6d2d59acee5b988eaa6ab86eb36dbad33"
COMPACT_SHA = "e87c4ab05cce479c173b0c94eb287d49f2050bad99320d955794425b09aeb730"
FLAGGED_SHA = "f4a1c722574ead60e9eee154fabafb2d68adf5514e1ecada035264b533a19b45"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def png(seed: int) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.random.default_rng(seed).integers(0, 256, (18, 18, 3), dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def note(errors: list, vision: str = "UNSURE") -> str:
    return json.dumps({"errors": errors, "vision_misread": vision, "lesson": "FREE_LESSON"})


def hf_problem(source: str, split: str, index: int, *, label: str = "HUMAN", gold: str = "B",
               images: int = 1) -> GeneratedProblem:
    """HF-shaped verifier-only gold item (adapters.hf_common.to_problem layout), plain mode, no slots."""
    raws = tuple(png(1000 + index + 500 * k) for k in range(images))
    digest = _sha(f"{source}:{split}:{index}".encode())[:16]
    pid = f"{source}:diagram:{split}:{index:06d}:{digest}"
    actor = ActorItem(pid, source, "diagram", "plain", f"Which colour marks region {index} in the {source} figure?",
                      CHOICES, "mcq", tuple(f"{source}/{split}/{index:06d}-{digest}-{k}.png" for k in range(images)),
                      tuple(_sha(raw) for raw in raws))
    verifier = VerifierItem(pid, "mcq", gold, {}, CHOICES, {}, {}, (), (),
                            ("region legend",) if label == "PROGRAM" else ())
    lineage = Lineage(pid, source, REVISION, "CC-BY-4.0", label, f"hf:fixture/{source}@{REVISION}",
                      "fixture annotators", None, None, f"{source}-row-v1", digest, split, ())
    return GeneratedProblem(actor, verifier, lineage, raws)


def plain_problem(split: str, index: int, *, source: str = PLAIN,
                  question: str = "What is the vertical reaction at support A in N?") -> GeneratedProblem:
    """Plain-mode program-generator record (statics layout; private slots never reach the actor)."""
    raw = png(5000 + index)
    digest = _sha(f"{source}:{split}:{index}".encode())[:16]
    pid = f"{source}:statics:{split}:3407:{index:06d}:{digest}"
    actor = ActorItem(pid, source, "statics", "plain", question, (), "int",
                      (f"{source}/{split}/3407-{index:06d}-{digest}-0.png",), (_sha(raw),))
    verifier = VerifierItem(pid, "int", "12", {}, (), {"span": "4"}, {"span": "number"}, ("span",), (), ("A", "B"))
    lineage = Lineage(pid, source, "f" * 64, "project-authored", "PROGRAM", f"rendered:{source}:default",
                      "procedural", source, 3407, "reaction-a", digest, split, ())
    return GeneratedProblem(actor, verifier, lineage, (raw,))


_OBS: dict[str, list[GeneratedProblem]] = {}


def obs_problems() -> list[GeneratedProblem]:
    if not _OBS:
        _OBS["rows"] = generate_pool("graph_pair", 2, 3407, "train", 1.0) + generate_pool("graph", 1, 3407, "train", 1.0)
    return _OBS["rows"]


def allowlist_receipt(root: Path, quarantined=()) -> str:
    """Authored stand-in for a reviewed exclusion receipt (status/allowlist only; the approval is faked here).

    Rebinds the build manifest and returns its new SHA."""
    ids = set(local_data.public_pool(root))
    receipt = {"status": "EXCLUSION_CHECKED_WITH_LIMITS", "allowlist": sorted(ids - set(quarantined)),
               "quarantined_ids": sorted(quarantined)}
    manifest = json.loads((root / "manifest.yaml").read_bytes())
    manifest["exclusion_receipt_sha256"] = _write_json(root / "contamination/receipt_v2.json", receipt)
    return _write_json(root / "manifest.yaml", manifest)


def pool_build(root: Path, *, gold: str = "B") -> None:
    pools: dict[str, list[GeneratedProblem]] = {split: [] for split in SPLITS}
    pools["train"] = [hf_problem(source, "train", 2 * i + k, gold=gold) for i, source in enumerate(HF_TRAIN) for k in (0, 1)]
    pools["train"] += obs_problems() + [plain_problem("train", 1), plain_problem("train", 2)]
    pools["dev"] = [hf_problem("hf-fixture-e", "dev", 20, gold=gold), hf_problem("hf-fixture-e", "dev", 21, gold=gold),
                    hf_problem("hf-fixture-f", "dev", 22, gold=gold), hf_problem("hf-fixture-a", "dev", 23, gold=gold)]
    pools["sealed"] = [hf_problem("hf-fixture-e", "sealed", 30, gold=gold), hf_problem("hf-fixture-f", "sealed", 31, gold=gold)]
    pools["style_sealed"] = [hf_problem("hf-fixture-f", "style_sealed", 40, gold=gold)]
    build_splits(root, pools)
    allowlist_receipt(root)


def pool_protocol() -> dict:
    return local_profile.build_protocol(repair_notebook=True, context_limit=24576, note_contract=compact_note.VERSION,
                                        answer_verified_route=True, finalization_reward=True, slotless_sources=SLOTLESS)


def protocol_args() -> list[str]:
    return ["--note-contract", compact_note.VERSION, "--answer-verified-route", "--finalization-reward",
            *[value for source in SLOTLESS for value in ("--slotless-source", source)]]


def fake_validation(root, blocked: dict | None = None, **_) -> dict:
    """Authored stand-in for the isolated ai_reviewed_local worker (approval itself is tested below)."""
    root = Path(root)
    public = local_data.public_pool(root)
    allowlist = local_data.pool_allowlist(root, public)
    blocked = blocked or {}
    eligible = sorted(pid for pid in allowlist if public[pid][0].source not in blocked)
    actors = [actor for actor in read_actors(root / "train") if actor.problem_id in eligible]
    lineage = read_lineage(root / "train")
    review = "c" * 64
    return {"status": "VALIDATED_FOR_MEASUREMENT", "training_allowed": False, "approval_created": False,
            "actors": actors, "structures": {actor.problem_id: lineage[actor.problem_id].structure_hash for actor in actors},
            "selected_ids": [actor.problem_id for actor in actors], "blockers": [],
            "limitations": ["authored fixture validation, not an exclusion approval"],
            "input_binding": {"build_manifest_sha256": _sha((root / "manifest.yaml").read_bytes()),
                              "ai_review_file_sha256": review},
            "code_sha256": a.code_hashes(), "measurement_allowed": True, "stage": a.AI_LOCAL_STAGE,
            "ai_review_file_sha256": review, "human_verified": False,
            "eligibility": {"allowlisted_by_source": dict(sorted(Counter(public[pid][0].source for pid in allowlist).items())),
                            "quarantined_count": len(public) - len(allowlist), "blocked_sources": blocked,
                            "eligible_ids": eligible}}


def write_request(base: Path, root: Path) -> Path:
    descriptor = {"path": str(base / "authored-index.json"), "file_sha256": "d" * 64}
    path = base / "ai_local_request.json"
    _write(path, _bytes({"build_root": str(root), "index": descriptor, "source_registry": descriptor,
                         "evidence": descriptor, "data_report": descriptor,
                         "expected_measurement_scope": {"purpose": a.AI_LOCAL_PURPOSE}}))
    return path


class CharTokenizer:
    """Stateless per-character tokenizer, so separate processes decode identically."""
    unk_token_id = -1

    def convert_tokens_to_ids(self, text):
        return 151645 if text == "<|im_end|>" else -1

    def encode(self, text, **kwargs):
        return [1000 + ord(ch) for ch in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(token - 1000) for token in ids if token != 151645)


class ScriptedEngine:
    """Replies chosen by (problem ID, request kind); never reads verifier gold."""
    model_config = SimpleNamespace(image_token_id=99)

    def __init__(self, replies: dict, *, policy_version: int = 0, context_limit: int = 24576):
        self.replies, self.policy_version, self.context_limit = replies, policy_version, context_limit
        self.tokenizer, self.calls = CharTokenizer(), []

    def prepare(self, request):
        n = len(request.images)
        tokens = (list(request.prompt_token_ids) if request.prompt_token_ids is not None else
                  [101, 99, 102] * n + self.tokenizer.encode(json.dumps(request.messages, sort_keys=True, ensure_ascii=False)))
        identity = _sha(b"".join(image.tobytes() for image in request.images))
        pixels = {"pixel_values": torch.tensor([[float(int(identity[:4], 16)), 1., 2.]] * n),
                  "image_grid_thw": torch.tensor([[1, 2, 2]] * n)}
        return Prepared(tokens, pixels, request.images, identity, tuple((3 * i + 1, 3 * i + 2) for i in range(n)))

    def _texts(self, request) -> list[str]:
        if request.request_id.endswith("FRESH_OBSERVATION"):
            return ["Observed: a coloured diagram with labelled regions."] * request.n
        pid = next(pid for pid in self.replies if request.request_id.startswith(pid + ":"))
        kind = next(kind for kind in ("REPAIR_NOTEBOOK", "NOTE", "T1", "T2", "T3") if ":" + kind in request.request_id)
        texts = self.replies[pid][kind]
        if len(texts) != request.n:
            raise AssertionError(f"fixture script for {kind} has {len(texts)} texts, request n={request.n}")
        return texts

    def sample(self, requests, *, on_completion=None):
        request = requests[0]
        self.calls.append(request.request_id)
        completions = []
        for index, text in enumerate(self._texts(request)):
            ids = self.tokenizer.encode(text) + [151645]
            completion = Completion(ids, [-0.1] * len(ids), "stop", text)
            completions.append(completion)
            if on_completion is not None:
                on_completion(request.request_id, index, request.seeds[index], completion)
        return [RolloutResult(request.request_id, self.prepare(request).token_ids, completions)]

    def model_inputs(self, request, completion, device):
        return {"completion_tokens": len(completion)}


class PoolFixture(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="notebook-pool-")
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / "build"
        pool_build(self.root)
        self.protocol = pool_protocol()
        self.prompts = local_profile.local_prompts(self.protocol)
        self.request = write_request(self.base, self.root)

    def propose(self, root: Path | None = None, **options) -> dict:
        return local_data.propose_pool_rule(root or self.root, self.prompts, **options)

    def prepare(self, rule: dict, name: str = "prepared", *, validation=fake_validation) -> tuple[dict, Path]:
        path = self.base / f"{name}-rule.json"
        _write(path, _bytes(rule))
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=validation):
            result = local_data.prepare_pool_data(self.root, self.request, path, self.protocol, self.base / name)
        return result, self.base / name / "preparation.json"


class PoolPreparationTests(PoolFixture):
    def test_old_and_flagged_protocol_hashes_are_unchanged(self) -> None:
        build = local_profile.build_protocol
        self.assertEqual(build(repair_notebook=True, context_limit=24576)["sha256"], DEFAULT_SHA)
        self.assertEqual(build(repair_notebook=True, context_limit=24576, note_contract=compact_note.VERSION)["sha256"],
                         COMPACT_SHA)
        self.assertEqual(build(repair_notebook=True, context_limit=24576, note_contract=compact_note.VERSION,
                               answer_verified_route=True, finalization_reward=True,
                               slotless_sources=("test-slotless-mcq-v1",))["sha256"], FLAGGED_SHA)

    def test_proposal_is_deterministic_label_blind_and_gold_invariant(self) -> None:
        rule = self.propose(evaluation_count=2)
        self.assertEqual(rule, self.propose(evaluation_count=2))
        public = local_data.public_pool(self.root)
        fixed = [public[pid] for pid in rule["fixed"]]
        self.assertEqual(rule["minimal"], rule["fixed"][:2])
        self.assertEqual([len(actor.image_paths) for actor, _ in fixed[:2]], [1, 2])
        self.assertEqual(sum(a.label_class(lin) != "GENERATOR" for _, lin in fixed), 6)
        self.assertGreaterEqual(len({actor.source for actor, lin in fixed if a.label_class(lin) != "GENERATOR"}), 4)
        self.assertEqual(sum(a.label_class(lin) == "GENERATOR" and actor.mode == "obs" for actor, lin in fixed), 2)
        self.assertEqual(sum(a.label_class(lin) == "GENERATOR" and actor.mode == "plain" for actor, lin in fixed), 2)
        self.assertEqual({public[pid][0].source for pid in rule["evaluation"]}, set(HF_HELD))
        # Different private gold with identical public inputs: the same IDs and order.
        other = self.base / "other-gold"
        pool_build(other, gold="D")
        changed = self.propose(other, evaluation_count=2)
        self.assertNotEqual(changed["build_manifest_sha256"], rule["build_manifest_sha256"])
        self.assertEqual({k: v for k, v in changed.items() if k != "build_manifest_sha256"},
                         {k: v for k, v in rule.items() if k != "build_manifest_sha256"})
        # Label blindness: the proposal still works with every verifier file removed.
        for split in SPLITS:
            (other / split / "verifier_private.jsonl").unlink()
        self.assertEqual(self.propose(other, evaluation_count=2), changed)
        self.assertNotEqual(self.propose(evaluation_count=2, seed=11)["fixed"], rule["fixed"])

    def test_prepare_preserves_modes_and_existing_consumers_read_the_binding(self) -> None:
        rule = self.propose(evaluation_count=2)
        result, path = self.prepare(rule)
        self.assertEqual(result["status"], local_data.POOL_STATUS, result.get("blockers"))
        self.assertEqual((result["format"], result["human_verified"], result["training_allowed"]),
                         (local_data.POOL_FORMAT, False, False))
        selection = json.loads(Path(result["selection"]["path"]).read_text())
        self.assertEqual(selection["pair_ids"], [rule["fixed"][i:i + 2] for i in range(0, 10, 2)])
        self.assertEqual(selection["evaluation_ids"], rule["evaluation"])
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=fake_validation) as validator:
            prepared, _, actors = local_run.read_preparation(path, self.protocol)
        self.assertEqual(validator.call_args.kwargs["timeout"], local_data.LOCAL_PREFLIGHT_SECONDS)
        public = local_data.public_pool(self.root)
        self.assertEqual(actors, [public[pid][0] for pid in rule["fixed"]])
        self.assertEqual({a_.mode for a_ in actors}, {"plain", "obs"})
        self.assertEqual([a_.problem_id for a_ in local_run.minimal_actors(actors)], rule["minimal"])
        from fine_tuning.notebook_rl.local_train import fixed_pairs
        self.assertEqual([[a_.problem_id for a_ in pair] for pair in fixed_pairs(actors)], selection["pair_ids"])
        self.assertEqual(local_run.read_execution_actors(prepared, actors, self.protocol), actors)
        # A different protocol, an edited selection, or a mode projection is rejected by the consumer.
        with self.assertRaises(ValueError):
            local_run.read_preparation(path, local_profile.build_protocol(repair_notebook=True, context_limit=24576))
        execution = Path(result["execution_actors"]["path"])
        projected = [json.loads(local_run.to_json(a_)) | {"mode": "obs"} for a_ in actors]
        execution.chmod(0o600)
        execution.write_text(json.dumps(projected))
        tampered = dict(result, execution_actors={"path": str(execution), "file_sha256": _sha(execution.read_bytes())})
        with self.assertRaisesRegex(ValueError, "unchanged built actors"):
            local_run.read_execution_actors(tampered, actors, self.protocol)
        old_style = dict(tampered, format="notebook-rl-local-data-v1")
        with self.assertRaisesRegex(ValueError, "observation projection"):
            local_run.read_execution_actors(old_style, actors, self.protocol)

    def test_rule_disjointness_minimal_and_sizes_are_enforced(self) -> None:
        rule = self.propose(evaluation_count=2)
        public = local_data.public_pool(self.root)
        same_source = next(pid for pid, (actor, lin) in public.items()
                           if lin.split == "dev" and actor.source == "hf-fixture-a")
        train_extra = next(pid for pid, (_, lin) in public.items() if lin.split == "train" and pid not in rule["fixed"])
        cases = {"evaluation source shared with training": rule | {"evaluation": [same_source]},
                 "evaluation from train": rule | {"evaluation": [train_extra]},
                 "minimal not first single/two-image": rule | {"minimal": rule["minimal"][::-1]},
                 "nine fixed": rule | {"fixed": rule["fixed"][:9]},
                 "foreign build": rule | {"build_manifest_sha256": "0" * 64},
                 "unknown key": rule | {"labels": {}}}
        for name, changed in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                local_data.check_pool_rule(changed, public, self.prompts, rule["build_manifest_sha256"])
        compact = local_profile.local_prompts(local_profile.build_protocol(
            repair_notebook=True, context_limit=24576, note_contract=compact_note.VERSION))
        with self.assertRaisesRegex(ValueError, "render"):
            local_data.check_pool_rule(rule, public, compact, rule["build_manifest_sha256"])
        self.assertFalse((self.base / "prepared").exists())

    def test_blocked_validation_writes_no_actor_outputs(self) -> None:
        def blocked(root, **_):
            return {"status": "BLOCKED_NOT_CHECKED", "training_allowed": False, "approval_created": False, "actors": [],
                    "structures": {}, "selected_ids": [], "blockers": ["AI_LOCAL_REVIEW_FAILED_SOURCE"], "limitations": []}
        result, _ = self.prepare(self.propose(evaluation_count=2), "blocked", validation=blocked)
        self.assertEqual((result["status"], result["blockers"]), ("BLOCKED", ["AI_LOCAL_REVIEW_FAILED_SOURCE"]))
        for name in ("actors", "execution_actors", "evaluation_actors", "selection"):
            self.assertNotIn(name, result)
            self.assertFalse((self.base / "blocked" / f"{name}.json").exists())

    def test_quarantined_ids_and_blocked_sources_are_never_selected(self) -> None:
        full = self.propose(evaluation_count=2)
        quarantined = {full["fixed"][2], full["evaluation"][0]}
        build_sha = allowlist_receipt(self.root, quarantined)
        public = local_data.public_pool(self.root)
        # A rule naming a quarantined ID is rejected by the rule check and by preparation (no outputs written).
        stale = full | {"build_manifest_sha256": build_sha}
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            local_data.check_pool_rule(stale, public, self.prompts, build_sha,
                                       eligible=local_data.pool_allowlist(self.root, public))
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            self.prepare(stale, "stale")
        self.assertFalse((self.base / "stale").exists())
        # The proposal draws only from allowlisted IDs; preparation records the quarantine count.
        rule = self.propose(evaluation_count=2)
        chosen = set(rule["minimal"] + rule["fixed"] + rule["evaluation"])
        self.assertFalse(chosen & quarantined)
        result, _ = self.prepare(rule)
        self.assertEqual(result["status"], local_data.POOL_STATUS, result.get("blockers"))
        self.assertEqual((result["eligibility"]["quarantined_count"], result["eligibility"]["blocked_sources"]), (2, {}))
        validated = json.loads(Path(result["validated_data"]["path"]).read_text())
        self.assertFalse(set(validated["selected_ids"]) & quarantined)
        # A source blocked by the AI review is skipped by the proposal and rejected by preparation.
        self.assertEqual({public[pid][0].source for pid in rule["evaluation"]}, set(HF_HELD))
        blocked = {"hf-fixture-e": ["FAILED_VERDICT render_ok fixture"]}
        eligible = fake_validation(self.root, blocked)["eligibility"]["eligible_ids"]
        skipped = self.propose(evaluation_count=2, eligible=eligible)
        self.assertEqual({public[pid][0].source for pid in skipped["evaluation"]}, {"hf-fixture-f"})
        self.assertFalse(set(skipped["fixed"] + skipped["evaluation"]) & quarantined)
        with self.assertRaisesRegex(ValueError, "blocked sources"):
            self.prepare(rule, "blocked-source", validation=lambda root, **_: fake_validation(root, blocked))
        self.assertFalse((self.base / "blocked-source").exists())
        self.assertEqual(self.prepare(skipped, "skipped", validation=lambda root, **_: fake_validation(
            root, blocked))[0]["eligibility"]["blocked_sources"], blocked)

    def test_subjects_are_reconstructed_only_when_hf_pool_counts_match(self) -> None:
        public = local_data.public_pool(self.root)
        self.assertEqual(set(local_data.pool_subjects(self.root, public).values()), {None})
        native = {pid: ([actor.source, pid] if lin.generator_version else [actor.source, f"row-{pid}"])
                  for pid, (actor, lin) in public.items()}
        counts: dict = {split: {} for split in SPLITS}
        for pid, (actor, lin) in public.items():
            subject = ("Physics" if pid.split(":")[3] in ("000021", "000030") else "Biology"
                       ) if actor.source == "hf-fixture-e" else "Math"
            bucket = counts[lin.split].setdefault(actor.source, {})
            bucket[subject] = bucket.get(subject, 0) + 1
        manifest = self.root / "pool_manifest.json"
        manifest.write_text(json.dumps({"counts": counts, "native_ids": native}))
        subjects = local_data.pool_subjects(self.root, public)
        held = {pid: subjects[pid] for pid, (actor, _) in public.items() if actor.source == "hf-fixture-e"}
        self.assertEqual(sorted(held.values()), ["Biology", "Physics", "Physics"])
        self.assertEqual(subjects[next(pid for pid in held if pid.split(":")[3] == "000020")], "Biology")
        # hf-fixture-a's dev index follows its train indices, which hf_pool's block order never produces: refused.
        self.assertEqual({pid for pid, value in subjects.items() if value is None},
                         {pid for pid, (actor, _) in public.items() if actor.source == "hf-fixture-a"})
        counts["dev"]["hf-fixture-e"] = {"Biology": 3}  # three dev rows claimed, two exist
        manifest.write_text(json.dumps({"counts": counts, "native_ids": native}))
        self.assertTrue(all(local_data.pool_subjects(self.root, public)[pid] is None for pid in held))

    def test_cli_proposes_and_prepares_with_flags(self) -> None:
        rule_path = self.base / "cli-rule.json"
        self.assertEqual(local_run.main(["propose-pool", "--build", str(self.root), "--evaluation-count", "2",
                                         *protocol_args(), "--output", str(rule_path)]), 0)
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=fake_validation):
            code = local_run.main(["prepare-pool", "--build", str(self.root), "--approval", str(self.request),
                                   "--rule", str(rule_path), *protocol_args(), "--output", str(self.base / "cli")])
        self.assertEqual(code, 0)
        prepared = json.loads((self.base / "cli/preparation.json").read_text())
        self.assertEqual(prepared["local_protocol_sha256"], self.protocol["sha256"])
        with self.assertRaises(SystemExit):
            local_run.main(["prepare-pool", "--build", str(self.root), "--approval", str(self.request), "--rule",
                            str(rule_path), "--answer-verified-route", "--output", str(self.base / "bad")])


EVAL_REPLIES = {"hf-fixture-e": {"T1": [WRONG], "T2": [CHECK + RIGHT],
                                 "NOTE": [note([{"attempt": 1, "type": "WRONG_STEP"}])]},
                "hf-fixture-f": {"T1": [NO_FINAL], "T2": [CHECK + NO_FINAL], "T3": [CHECK + RIGHT],
                                 "NOTE": [note([{"attempt": 1, "type": "NO_FINAL_ANSWER"},
                                                {"attempt": 2, "type": "NO_FINAL_ANSWER"}])]}}


def eval_replies(ids: list[str], replies: dict = EVAL_REPLIES) -> dict:
    return {pid: replies[next(source for source in replies if pid.startswith(source + ":"))] for pid in ids}


class PoolEvaluatorTests(PoolFixture):
    def setUp(self) -> None:
        super().setUp()
        self.rule = self.propose(evaluation_count=2)
        result, self.preparation = self.prepare(self.rule)
        self.assertEqual(result["status"], local_data.POOL_STATUS)
        self.sha = _sha(self.preparation.read_bytes())

    def evaluate(self, name: str, replies: dict, version: int = 0) -> dict:
        engine = ScriptedEngine(eval_replies(self.rule["evaluation"], replies), policy_version=version)
        return local_pool_eval.diagnose(self.preparation, self.base / name, engine, copy.deepcopy(self.prompts),
            expected_preparation_sha256=self.sha, local_protocol_sha256=self.protocol["sha256"],
            deadline=time.monotonic() + 300, context_limit=24576, note_contract=compact_note.VERSION)

    def test_metrics_breakdowns_replay_and_tamper_detection(self) -> None:
        record = self.evaluate("f0", EVAL_REPLIES)
        self.assertEqual(record["status"], local_pool_eval.COMPLETE, record.get("reason"))
        m = record["metrics"]
        self.assertEqual((m["n"], m["first_attempt_accuracy"], m["recovery_rate_diagnostic"], m["recovered_fraction"]),
                         (2, 0.0, 1.0, 1.0))
        self.assertEqual((m["attempt_count"], m["finalized_attempt_count"], m["finalization_rate"]), (5, 3, 0.6))
        self.assertEqual((m["length_terminated_attempts"], m["length_terminated_fraction"]), (0, 0.0))
        self.assertEqual((m["note_count"], m["parsed_note_count"], m["note_verified_count"], m["note_verified_fraction"]),
                         (2, 2, 1, 0.5))
        self.assertEqual(set(m["by_source"]), set(HF_HELD))
        self.assertEqual(m["by_subject"], {"UNKNOWN": local_pool_eval._summary(record["items"])})
        self.assertEqual({item["route"] for item in record["items"]}, {"ANSWER_VERIFIED"})
        self.assertTrue(all(len(item["attempt_text_sha256"]) == len(item["attempt_statuses"]) for item in record["items"]))
        self.assertEqual(json.loads((self.base / "f0/result.json").read_text()), record)
        replayed = local_pool_eval.replay(record, expected_preparation_sha256=self.sha)
        self.assertEqual(replayed["metrics"], record["metrics"])
        for mutate in (lambda r: r["calls"][0].update(text=r["calls"][0]["text"] + " "),
                       lambda r: r["metrics"].update(recovery_rate_diagnostic=0.5),
                       lambda r: r["items"][0].update(first_status="CORRECT"),
                       lambda r: r["calls"].pop()):
            tampered = copy.deepcopy(record)
            mutate(tampered)
            tampered["record_sha256"] = local_pool_eval.generation._digest(
                {k: v for k, v in tampered.items() if k != "record_sha256"})
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                local_pool_eval.replay(tampered, expected_preparation_sha256=self.sha)
        with self.assertRaises(ValueError):
            local_pool_eval.replay(record, expected_preparation_sha256="0" * 64)

    def test_recovery_rate_is_a_diagnostic_never_named_accuracy_and_legacy_records_replay(self) -> None:
        record = self.evaluate("f0", EVAL_REPLIES)
        self.assertNotIn("final_accuracy", json.dumps(record))

        def legacy(value):
            if isinstance(value, dict):
                return {("final_accuracy" if k == "recovery_rate_diagnostic" else k): legacy(v) for k, v in value.items()}
            return value

        def resign(value: dict) -> dict:
            value["record_sha256"] = local_pool_eval.generation._digest(
                {k: v for k, v in value.items() if k != "record_sha256"})
            return value

        old = copy.deepcopy(record)
        old["metrics"] = legacy(old["metrics"])
        self.assertIn("final_accuracy", json.dumps(old["metrics"]))
        resign(old)
        replayed = local_pool_eval.replay(old, expected_preparation_sha256=self.sha)
        self.assertNotIn("final_accuracy", json.dumps(replayed))
        self.assertEqual(replayed["metrics"], record["metrics"])
        table = local_pool_eval.compare({"old": old, "new": record}, expected_preparation_sha256=self.sha)
        self.assertNotIn("final_accuracy", json.dumps(table))
        self.assertEqual(table["summary"]["old"], table["summary"]["new"])
        self.assertEqual(table["by_subject"]["old"], table["by_subject"]["new"])
        self.assertEqual(table["summary"]["old"]["recovery_rate_diagnostic"], 1.0)
        both = copy.deepcopy(record)
        both["metrics"]["final_accuracy"] = both["metrics"]["recovery_rate_diagnostic"]
        with self.assertRaises(ValueError):
            local_pool_eval.replay(resign(both), expected_preparation_sha256=self.sha)

    def test_compare_pairs_same_items_only(self) -> None:
        before = self.evaluate("before", EVAL_REPLIES)
        improved = {"hf-fixture-e": {"T1": [RIGHT]}, "hf-fixture-f": EVAL_REPLIES["hf-fixture-f"]}
        after = self.evaluate("after", improved, version=1)
        table = local_pool_eval.compare({"F0": before, "FT-notebook": after}, expected_preparation_sha256=self.sha)
        self.assertEqual(table["labels"], ["F0", "FT-notebook"])
        self.assertEqual(table["n_items"], 2)
        gained = next(pid for pid in self.rule["evaluation"] if pid.startswith("hf-fixture-e:"))
        self.assertEqual(table["paired_vs_reference"]["FT-notebook"], {"first_gains": [gained], "first_losses": []})
        self.assertEqual([row["problem_id"] for row in table["table"]], self.rule["evaluation"])
        self.assertFalse(table["improvement_claimed"])
        # Same records, different item set: never pooled.
        smaller = copy.deepcopy(after)
        smaller["problem_ids"] = smaller["problem_ids"][:1]
        smaller["record_sha256"] = local_pool_eval.generation._digest({k: v for k, v in smaller.items() if k != "record_sha256"})
        with self.assertRaises(ValueError):
            local_pool_eval.compare({"F0": before, "FT": smaller}, expected_preparation_sha256=self.sha)

    def test_evaluate_pool_cli_forwards_and_pool_mode_is_exclusive(self) -> None:
        with patch.object(local_eval, "evaluate_pool", return_value={"status": local_pool_eval.COMPLETE}) as run, \
                patch("builtins.print"):
            code = local_eval.main(["evaluate-pool", "--capture", "c", "--preparation", "p", "--reload-result", "r",
                                    "--output", "o", "--condition", "F0", "--arm", "notebook"])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.args, (Path("c"), Path("p"), Path("o")))
        self.assertEqual(run.call_args.kwargs | {}, {"condition": "F0", "arm": "notebook", "reload_result": Path("r"),
                                                    "minimal_capture": None, "minimal_reload": None})
        for options in ({"ladder_cohort": Path("cohort")}, {"with_recovery": False},
                        {"evaluation_prompt_policy": "circuit-ladder-verified-repair-notebook-v1"}):
            arguments = {"with_recovery": True, "pool": True} | options
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "pool evaluation"):
                local_eval._evaluate(Path("c"), Path("p"), Path("o"), condition="F0", reload_result=Path("r"),
                                     minimal_capture=None, minimal_reload=None, arm="notebook", **arguments)

    def test_cli_replay_pool_reproduces_metrics(self) -> None:
        record = self.evaluate("cli", EVAL_REPLIES)
        output = self.base / "replayed.json"
        code = local_eval.main(["replay-pool", "--record", str(self.base / "cli/result.json"),
                                "--preparation-sha256", self.sha, "--output", str(output)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.read_text())["metrics"], record["metrics"])
        compared = self.base / "compared.json"
        self.assertEqual(local_eval.main(["compare-pool", "--record", f"F0={self.base / 'cli/result.json'}",
                                          "--record", f"again={self.base / 'cli/result.json'}",
                                          "--preparation-sha256", self.sha, "--output", str(compared)]), 0)
        self.assertEqual(json.loads(compared.read_text())["paired_vs_reference"]["again"],
                         {"first_gains": [], "first_losses": []})


def _write_json(path: Path, value) -> str:
    raw = a._json(value) + b"\n"
    path.write_bytes(raw)
    return a._sha(raw)


def source_entry(label: str, *, generated: bool = False, revision: str = REVISION) -> dict:
    return {"rights_approved": True, "source_known": True, "benchmark_derived": False, "model_touched": False,
            "label_provenance": label, "source_revision": revision,
            "license": "project-authored" if generated else "CC-BY-4.0",
            "upstream_origin": "procedural" if generated else "fixture annotators", "parents": [],
            "upstream_split": "generated" if generated else "train"}


def scanned_build(base: Path, pools: dict, sources: dict) -> dict:
    """Authored full-index scan, as in the measurement approval fixture; not a real exclusion receipt."""
    from fine_tuning.notebook_rl import contamination as c
    from fine_tuning.notebook_rl.schema import load_protocol
    from fine_tuning.tests.test_notebook_rl_contamination import index_fixture

    root = base / "build"
    manifest = build_splits(root, pools)
    protocol = load_protocol(a._HERE / "protocol_v1.yaml")
    registry = {"sources": sources, "upstream_origin_blocklist": []}
    index_path = base / "index.json"
    index_sha = _write_json(index_path, index_fixture())
    index = c._prepare(index_fixture(), "authored-test-only")
    index.update(full_coverage=True, limitations=[], partial_name=None, path=str(index_path), file_sha256=index_sha)
    items, input_files, images = c._snapshot({s: root / s for s in SPLITS}, root / "images")
    calibration = {"frozen_before_scan": True, "protocol_sha256": protocol["sha256"], "index_sha256": index["sha256"],
        "positive_count": 4, "negative_count": 4, "tpr": .75, "fpr": 0.,
        "per_engine": {k: {"positive_count": 4, "negative_count": 4, "tpr": .75, "fpr": 0., "examples_sha256": "d" * 64}
                       for k in ("dhash", "tiles", "sscd", "siglip")},
        "thresholds": {"perceptual_hamming": 3, "sscd_exclude": .95, "sscd_review": .75, "siglip_review": .9}}
    evidence: dict = {"calibration": calibration}
    registry["scanner_policy"] = {"status": "FROZEN_BEFORE_INPUT_SCAN", "protocol_sha256": protocol["sha256"],
                                  "calibration_sha256": a._hash(calibration), "engines": {}}
    for name, engine in (("ocr", "tesseract"), ("sscd", "sscd_disc_mixup"), ("siglip", "google/siglip-fixture")):
        identity = {"engine_id": engine, "revision": "authored-fixture-v1", "artifact_sha256": "e" * 64}
        registry["scanner_policy"]["engines"][name] = identity.copy()
        evidence[name] = identity | {"index_sha256": index["sha256"], "calibration_sha256": a._hash(calibration)}
    human = {sha for item in items.values() if item["lineage"].label_provenance == "HUMAN" for sha in item["image_sha256"]}
    evidence["ocr"].update(benchmark={sha: {"image_sha256": sha, "text": "authored unrelated diagram label",
                                            "status": "COMPLETE"} for sha in index["images"]},
                           inputs={sha: {"image_sha256": sha, "text": "fixture legend", "status": "COMPLETE"} for sha in human})

    def vectors(bindings, vector):
        return {key: {"sha256": sha, "vector": vector} for key, sha in bindings.items()}
    for name in ("sscd", "siglip"):
        evidence[name].update(benchmark_images=vectors({sha: sha for sha in index["images"]}, [1., 0.]),
                              input_images=vectors({sha: sha for sha in images}, [0., 1.]))
    text_refs = {entry["key"]: a._sha(c.normalize_text(entry["text"]).encode()) for entry in index["text_cache"]["entries"]}
    evidence["siglip"].update(benchmark_texts=vectors(text_refs, [1., 0.]),
                              input_texts=vectors({pid: item["text_sha256"] for pid, item in items.items()}, [0., 1.]))
    with patch.object(c, "_prepare", return_value=index):
        receipt = c.scan({s: root / s for s in SPLITS}, root / "images", index, registry, evidence)
    manifest["exclusion_receipt_sha256"] = _write_json(root / "contamination" / "receipt_v2.json", receipt)
    build_sha = _write_json(root / "manifest.yaml", manifest)
    return {"root": root, "manifest": manifest, "protocol": protocol, "registry": registry, "index": index,
            "descriptor": {"path": str(index_path), "file_sha256": index_sha}, "items": items, "evidence": evidence,
            "receipt": receipt, "input_files": input_files, "build_sha": build_sha}


def review_record(items: dict, manifests: dict, input_files_sha256: str, *, quarantined=(), **changes) -> dict:
    """Authored AI review over the allowlisted items (sample prefix in hash order, leak warnings)."""
    groups: dict[str, list[str]] = {}
    for pid, item in items.items():
        if pid not in quarantined:
            groups.setdefault(item["actor"].source, []).append(pid)
    sources = {source: [{"problem_id": pid, "image_sha256": list(items[pid]["actor"].image_sha256),
                         "render_ok": True, "leak_free": True, "gold_supported": True, "notes": "fixture review"}
                        for pid in a.ai_review_order(pids, 3407)[:min(a.AI_LOCAL_SAMPLE, len(pids))]]
               for source, pids in groups.items()}
    leaks = [{"problem_id": row["problem_id"], "field": row["field"], "leak_free": True, "notes": "fixture"}
             for manifest in manifests.values() for row in manifest["leak_review"] if row["problem_id"] not in quarantined]
    return {"format": a.AI_LOCAL_REVIEW_FORMAT, "reviewer_id": "fixture-ai-reviewer", "human_verified": False,
            "labels_modified": False, "reviewed_at_utc": "2026-10-30T00:00:00+00:00", "sampling_seed": 3407,
            "input_files_sha256": input_files_sha256, "sources": sources, "leak_warnings": leaks} | changes


class AIReviewedLocalApprovalTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="notebook-ai-local-")
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)
        self.code = a.code_hashes()
        self.cases = 0
        self.build()

    def build(self, *, claim_program: bool = False) -> None:
        self.cases += 1
        self.base = self.temp / f"case-{self.cases}"
        self.base.mkdir()
        pools: dict[str, list[GeneratedProblem]] = {split: [] for split in SPLITS}
        for si, split in enumerate(SPLITS):
            n = 100 * si
            pools[split] += [hf_problem("hf-fixture-a", split, n + j) for j in range(3)]
            pools[split] += [hf_problem("hf-fixture-b", split, n + 5),
                             hf_problem("hf-fixture-prog", split, n + 6, label="PROGRAM"),
                             plain_problem(split, n + 7, source="gen-fixture-v1")]
        # Its private gold "12" appears in the question: one manifest leak warning for the AI review.
        pools["train"].append(plain_problem("train", 8, source="gen-fixture-v1",
                                            question="Is the vertical reaction at support A 12 N or more?"))
        if claim_program:
            first = pools["train"][0]
            pools["train"][0] = GeneratedProblem(first.actor, VerifierItem(*[
                ("region legend",) if field == "printed_text" else getattr(first.verifier, field)
                for field in first.verifier.__dataclass_fields__]), Lineage(*[
                "PROGRAM" if field == "label_provenance" else getattr(first.lineage, field)
                for field in first.lineage.__dataclass_fields__]), first.image_bytes)
        sources = {"hf-fixture-a": source_entry("HUMAN"), "hf-fixture-b": source_entry("HUMAN"),
                   "hf-fixture-prog": source_entry("PROGRAM"),
                   "gen-fixture-v1": source_entry("PROGRAM", generated=True, revision="f" * 64)}
        self.s = scanned_build(self.base, pools, sources)
        self.manifests = {split: json.loads((self.s["root"] / split / "manifest.yaml").read_bytes()) for split in SPLITS}
        self.generator_ids = sorted(pid for pid, item in self.s["items"].items() if item["actor"].source == "gen-fixture-v1")
        self.review = review_record(self.s["items"], self.manifests, self.s["receipt"]["input_files_sha256"])
        self.assertEqual(len(self.review["leak_warnings"]), 1)

    def payload(self, *, review: dict | None = None, provenance: dict | None = None, measurements: dict | None = None,
                stage: str = a.AI_LOCAL_STAGE, purpose: str = a.AI_LOCAL_PURPOSE) -> dict:
        s = self.s
        review_path = self.base / f"review-{len(list(self.base.glob('review-*')))}.json"
        review_descriptor = {"path": str(review_path), "file_sha256": _write_json(review_path, review or self.review)}
        binding = {"build_manifest_sha256": s["build_sha"], "exclusion_receipt_sha256": s["manifest"]["exclusion_receipt_sha256"],
                   "input_files_sha256": s["receipt"]["input_files_sha256"], "protocol_sha256": s["protocol"]["sha256"],
                   "index_sha256": s["index"]["sha256"], "index_file_sha256": s["index"]["file_sha256"],
                   "source_registry_sha256": a._hash(s["registry"]), "evidence_sha256": a._hash(s["evidence"])}
        producer = self.base / "authored_producer.py"
        producer.write_text("# Authored unit-test evidence producer.\n")
        scope = {"purpose": purpose, "protocol_sha256": s["protocol"]["sha256"], "code_sha256": self.code,
                 "input_files_sha256": s["receipt"]["input_files_sha256"],
                 "producer": {"path": str(producer), "file_sha256": a._sha(producer.read_bytes())}}
        ids = a._hash(self.generator_ids)
        gates = {"D1": {"checked_ids_sha256": ids, "items_checked": len(self.generator_ids), "repeat_mismatches": 0},
            "D2": {"checked_ids_sha256": ids, "items_checked": len(self.generator_ids), "solver_mismatches": 0,
                   "independent_solvers": True},
            "D3": {"sampling_seed": 3407, "review": review_descriptor},
            "D4": {"templates": {"gen-fixture-v1:reaction-a": {"pairs": 100, "answer_changed": 97,
                   "question_unchanged": True, "nonimage_inputs_unchanged": True}}},
            "D5": {"structure_cross_split": 0, "image_cross_split": 0},
            "D6": {"exclusion_receipt_sha256": s["manifest"]["exclusion_receipt_sha256"]},
            "D7": {"reviewed_actor_files": {p: sha for p, sha in s["input_files"].items() if Path(p).name == "actor_inputs.jsonl"},
                   "warnings_sha256": a._hash({split: m["leak_review"] for split, m in self.manifests.items()}), "unresolved": 0},
            "D8": {"separate_process": True, "private_open_denied": True, "benchmark_open_denied": True,
                   "teacher_import_denied": True, "python_guard_not_os_sandbox": True}} | (measurements or {})
        report = {"format": "notebook-rl-data-gates-v1", "status": "COMPLETE", "binding": binding, "scope": scope, "gates": {}}
        for gate, value in gates.items():
            proof = {"gate": gate, "status": "PASS", "binding_sha256": a._hash(binding), "scope_sha256": a._hash(scope),
                     "producer_sha256": scope["producer"]["file_sha256"], "measurements": value}
            if gate in ("D3", "D7"):
                proof["provenance"] = {"reviewer_id": "fixture-ai-reviewer", "kind": a.AI_LOCAL_REVIEW_KIND,
                    "method": a.AI_LOCAL_REVIEW_METHOD, "reviewed_at_utc": "2026-10-30T00:00:00+00:00",
                    "human_verified": False, "labels_modified": False} | (provenance or {})
            path = self.base / f"{gate}-{len(list(self.base.glob(gate + '-*')))}.json"
            report["gates"][gate] = {"status": "PASS", "artifact": {"path": str(path), "file_sha256": _write_json(path, proof)}}
        return {"stage": stage, "build_root": str(s["root"]), "index": s["descriptor"], "source_registry": s["registry"],
                "evidence": s["evidence"], "data_report": report, "expected_measurement_scope": scope}

    def validate(self, payload: dict) -> dict:
        from fine_tuning.notebook_rl import contamination as c
        with patch.object(c, "load_index", return_value=self.s["index"]), patch.object(a, "code_hashes", return_value=self.code):
            return a._validate(payload)

    def blocked(self, payload: dict, code: str | None = None) -> None:
        result = self.validate(payload)
        self.assertEqual(result["status"], "BLOCKED_NOT_CHECKED")
        self.assertEqual((result["actors"], result["selected_ids"], result["training_allowed"]), ([], [], False))
        if code is not None:
            self.assertIn(code, result["blockers"])

    def test_accepts_mixed_human_and_program_build_without_human_claims(self) -> None:
        result = self.validate(self.payload())
        self.assertEqual((result["status"], result["blockers"]), ("VALIDATED_FOR_MEASUREMENT", []))
        self.assertEqual((result["stage"], result["human_verified"], result["training_allowed"]), (a.AI_LOCAL_STAGE, False, False))
        review_sha = _sha(Path(sorted(self.base.glob("review-*"))[-1]).read_bytes())
        self.assertEqual(result["ai_review_file_sha256"], review_sha)
        self.assertEqual(result["input_binding"]["ai_review_file_sha256"], review_sha)
        self.assertEqual(set(result["selected_ids"]), {pid for pid, item in self.s["items"].items() if item["split"] == "train"})
        self.assertEqual({actor["source"] for actor in result["actors"]},
                         {"hf-fixture-a", "hf-fixture-b", "hf-fixture-prog", "gen-fixture-v1"})
        self.assertTrue(any("human_verified=false" in text for text in result["limitations"]))
        self.assertEqual({a.label_class(item["lineage"]) for item in self.s["items"].values()},
                         {"HUMAN", "EXTERNAL_PROGRAM", "GENERATOR"})
        self.assertEqual(result["eligibility"], {
            "allowlisted_by_source": {"gen-fixture-v1": 5, "hf-fixture-a": 12, "hf-fixture-b": 4, "hf-fixture-prog": 4},
            "quarantined_count": 0, "blocked_sources": {}, "eligible_ids": sorted(self.s["items"])})
        # Public facade: stage, review hash and human_verified=false survive; a poisoned claim fails closed.
        kwargs = {k: v for k, v in self.payload().items() if k not in ("stage", "build_root")}
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=a._json(result))):
            checked = a.validate_for_ai_reviewed_local(self.s["root"], **kwargs)
        self.assertTrue(all(type(actor) is ActorItem for actor in checked["actors"]))
        for change in ({"human_verified": True}, {"stage": "measurement"}, {"ai_review_file_sha256": "0" * 64},
                       {"eligibility": result["eligibility"] | {"eligible_ids": []}},
                       {"eligibility": result["eligibility"] | {"blocked_sources": {"hf-fixture-a": ["x"]}}}):
            with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=a._json(result | change))):
                self.assertIn("PREFLIGHT_RESPONSE_INVALID", a.validate_for_ai_reviewed_local(self.s["root"], **kwargs)["blockers"])
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=a._json(result))):
            self.assertIn("PREFLIGHT_RESPONSE_INVALID", a.validate_for_measurement(self.s["root"], **kwargs)["blockers"])

    def test_existing_measurement_and_training_results_keep_their_exact_shape(self) -> None:
        from fine_tuning.tests.test_notebook_rl_approval import ApprovalTests
        fixture = ApprovalTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        measured = fixture.measurement_validate(fixture.measurement_fixture())
        trained = fixture.validate()
        base = {"status", "training_allowed", "approval_created", "actors", "structures", "selected_ids", "blockers",
                "limitations", "input_binding", "code_sha256"}
        self.assertEqual((measured["status"], measured["blockers"]), ("VALIDATED_FOR_MEASUREMENT", []))
        self.assertEqual(set(measured), base | {"measurement_allowed"})
        self.assertEqual((trained["status"], trained["blockers"]), ("VALIDATED_FOR_TRAINING", []))
        self.assertEqual(set(trained), base | {"selection_sha256"})
        self.assertNotIn("ai_review_file_sha256", measured["input_binding"])

    def assert_source_blocked(self, result: dict, source: str, reason: str) -> None:
        self.assertEqual((result["status"], result["blockers"]), ("VALIDATED_FOR_MEASUREMENT", []))
        self.assertEqual(list(result["eligibility"]["blocked_sources"]), [source])
        self.assertTrue(any(text.startswith(reason) for text in result["eligibility"]["blocked_sources"][source]))
        ineligible = {pid for pid, item in self.s["items"].items() if item["actor"].source == source}
        self.assertEqual(set(result["eligibility"]["eligible_ids"]), set(self.s["items"]) - ineligible)
        self.assertEqual(set(result["selected_ids"]), {pid for pid, item in self.s["items"].items()
                                                       if item["split"] == "train" and pid not in ineligible})
        self.assertNotIn(source, {actor["source"] for actor in result["actors"]})

    def test_malformed_review_blocks_all_and_short_sample_blocks_its_source(self) -> None:
        self.blocked(self.payload(measurements={"D3": {"sampling_seed": 3407}}), "AI_LOCAL_REVIEW_FILE_REQUIRED")
        missing = copy.deepcopy(self.review)
        del missing["sources"]["gen-fixture-v1"]
        self.blocked(self.payload(review=missing), "AI_LOCAL_REVIEW_SOURCE_COVERAGE")
        reordered = copy.deepcopy(self.review)
        reordered["sources"]["hf-fixture-a"].reverse()
        self.blocked(self.payload(review=reordered), "AI_LOCAL_REVIEW_PREREGISTERED_SAMPLE")
        short = copy.deepcopy(self.review)
        short["sources"]["hf-fixture-a"].pop()
        self.assert_source_blocked(self.validate(self.payload(review=short)), "hf-fixture-a",
                                   "SHORT_SAMPLE reviewed=11 required=12")

    def test_failing_verdict_blocks_only_its_source(self) -> None:
        for verdict in a.AI_LOCAL_VERDICTS:
            failing = copy.deepcopy(self.review)
            failing["sources"]["hf-fixture-prog"][0][verdict] = False
            with self.subTest(verdict=verdict):
                self.assert_source_blocked(self.validate(self.payload(review=failing)), "hf-fixture-prog",
                                           f"FAILED_VERDICT {verdict}")
        leaking = copy.deepcopy(self.review)
        leaking["leak_warnings"][0]["leak_free"] = False
        # D7 must report the review's unresolved count; the leak blocks the generator source only.
        self.blocked(self.payload(review=leaking), "D7_LEAK_REVIEW_BINDING")
        d7 = {"reviewed_actor_files": {p: sha for p, sha in self.s["input_files"].items() if Path(p).name == "actor_inputs.jsonl"},
              "warnings_sha256": a._hash({split: m["leak_review"] for split, m in self.manifests.items()}), "unresolved": 1}
        self.assert_source_blocked(self.validate(self.payload(review=leaking, measurements={"D7": d7})),
                                   "gen-fixture-v1", "LEAK_WARNING_NOT_LEAK_FREE question")

    def test_acceptance_review_v2_makes_only_failed_items_ineligible(self) -> None:
        review = copy.deepcopy(self.review) | {"format": a.AI_LOCAL_REVIEW_FORMAT_V2, "acceptance_rule": a.ACCEPTANCE_RULE}
        failed = review["sources"]["hf-fixture-a"][0]["problem_id"]
        review["sources"]["hf-fixture-a"][0]["gold_supported"] = False
        result = self.validate(self.payload(review=review))
        self.assertEqual((result["status"], result["blockers"]), ("VALIDATED_FOR_MEASUREMENT", []))
        eligibility = result["eligibility"]
        # n=12 < 60: one failure is accepted only because all 12 were reviewed.
        self.assertEqual(eligibility["quality"]["sources"]["hf-fixture-a"], {
            "reviewed": 12, "failed": 1, "observed_rate": round(1 / 12, 10), "cp_upper": 0.3386806684,
            "failed_ids": [failed], "status": "ELIGIBLE"})
        self.assertEqual((eligibility["quality"]["rule"], eligibility["blocked_sources"], eligibility["eligible_ids"]),
                         (a.ACCEPTANCE_RULE, {}, sorted(set(self.s["items"]) - {failed})))
        self.assertNotIn(failed, result["selected_ids"])
        self.assertIs(result["human_verified"], False)
        kwargs = {k: v for k, v in self.payload(review=review).items() if k not in ("stage", "build_root")}
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=a._json(result))):
            self.assertEqual(a.validate_for_ai_reviewed_local(self.s["root"], **kwargs)["blockers"], [])
        for change in ({"eligible_ids": sorted(self.s["items"])}, {"quality": None}):
            poisoned = result | {"eligibility": eligibility | change}
            with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=a._json(poisoned))):
                self.assertIn("PREFLIGHT_RESPONSE_INVALID",
                              a.validate_for_ai_reviewed_local(self.s["root"], **kwargs)["blockers"])
        short = copy.deepcopy(review)
        short["sources"]["hf-fixture-a"].pop()
        self.assert_source_blocked(self.validate(self.payload(review=short)), "hf-fixture-a",
                                   "ACCEPTANCE_SAMPLE_TOO_SMALL failed=1 reviewed=11 required=12")

    def test_every_source_blocked_fails(self) -> None:
        failing = copy.deepcopy(self.review)
        for rows in failing["sources"].values():
            rows[0]["gold_supported"] = False
        self.blocked(self.payload(review=failing), "NO_ELIGIBLE_SOURCE")

    def test_human_verified_claims_are_rejected(self) -> None:
        self.blocked(self.payload(review=self.review | {"human_verified": True}), "AI_LOCAL_REVIEW_BINDING")
        self.blocked(self.payload(provenance={"human_verified": True}), "AI_LOCAL_REVIEW_PROVENANCE")
        self.blocked(self.payload(provenance={"kind": "HUMAN"}), "AI_LOCAL_REVIEW_PROVENANCE")

    def test_missing_or_incomplete_exclusion_receipt_is_rejected(self) -> None:
        payload = self.payload()
        receipt = self.s["root"] / "contamination" / "receipt_v2.json"
        raw = receipt.read_bytes()
        partial = json.loads(raw) | {"status": "BLOCKED_NOT_CHECKED"}
        self.s["manifest"]["exclusion_receipt_sha256"] = _write_json(receipt, partial)
        _write_json(self.s["root"] / "manifest.yaml", self.s["manifest"])
        self.blocked(payload, "EXCLUSION_NOT_COMPLETE")
        receipt.unlink()
        self.blocked(payload, "MISSING_MALFORMED_OR_CHANGED_PREREQUISITE")

    def test_quarantined_families_stay_in_the_pool_but_are_never_eligible(self) -> None:
        from fine_tuning.notebook_rl import contamination as c
        items = self.s["items"]
        quarantined = sorted([next(pid for pid in sorted(items) if items[pid]["split"] == split
                                   and items[pid]["actor"].source == "hf-fixture-a") for split in ("train", "dev")]
                             + [self.generator_ids[0]])
        receipt = self.s["receipt"]
        receipt["candidates"] = [{"problem_id": pid, "stage": "text_ngrams", "reason": "AUTHORED_TEST", "action": "QUARANTINE"}
                                 for pid in quarantined]
        family = c.family_quarantine(items, receipt["candidates"])
        self.assertEqual(family["quarantined_ids"], quarantined)  # authored singleton families
        receipt.update(quarantined_ids=family["quarantined_ids"], allowlist=family["allowlist"],
                       quarantined_families=family["families"])
        receipt["receipt_sha256"] = a._hash({k: v for k, v in receipt.items() if k != "receipt_sha256"})
        self.s["manifest"]["exclusion_receipt_sha256"] = _write_json(self.s["root"] / "contamination/receipt_v2.json", receipt)
        self.s["build_sha"] = _write_json(self.s["root"] / "manifest.yaml", self.s["manifest"])
        # D1/D2 over every generator item, or the old whole-pool review sample, covers a quarantined ID.
        review = review_record(items, self.manifests, receipt["input_files_sha256"], quarantined=quarantined)
        self.assertFalse({row["problem_id"] for rows in review["sources"].values() for row in rows} & set(quarantined))
        self.blocked(self.payload(review=review), "D1_COVERAGE_OR_MISMATCH")
        self.generator_ids = [pid for pid in self.generator_ids if pid not in quarantined]
        self.blocked(self.payload(), "AI_LOCAL_REVIEW_PREREGISTERED_SAMPLE")
        result = self.validate(self.payload(review=review))
        self.assertEqual((result["status"], result["blockers"]), ("VALIDATED_FOR_MEASUREMENT", []))
        self.assertEqual(result["eligibility"]["quarantined_count"], 3)
        self.assertEqual(result["eligibility"]["allowlisted_by_source"],
                         {"gen-fixture-v1": 4, "hf-fixture-a": 10, "hf-fixture-b": 4, "hf-fixture-prog": 4})
        self.assertEqual(result["eligibility"]["eligible_ids"], sorted(set(items) - set(quarantined)))
        self.assertFalse(set(result["selected_ids"]) & set(quarantined))
        self.assertEqual(set(result["selected_ids"]), {pid for pid, item in items.items()
                                                       if item["split"] == "train" and pid not in quarantined})

    def test_human_item_claiming_program_and_program_coverage_distinction(self) -> None:
        # External PROGRAM labels have no solver: D1/D2 claiming to cover them is not the generator set.
        external = sorted(pid for pid, item in self.s["items"].items() if a.label_class(item["lineage"]) != "HUMAN")
        self.blocked(self.payload(measurements={"D1": {"checked_ids_sha256": a._hash(external),
                     "items_checked": len(external), "repeat_mismatches": 0}}), "D1_COVERAGE_OR_MISMATCH")
        self.build(claim_program=True)
        self.blocked(self.payload(), "SOURCE_OR_RECORD_IDENTITY")

    def test_stage_purpose_and_existing_stages_are_not_interchangeable(self) -> None:
        self.blocked(self.payload(stage="measurement"), "MEASUREMENT_SCOPE_REQUIRED")
        self.blocked(self.payload(purpose="FROZEN_WEIGHT_MEASUREMENT_ONLY"), "MEASUREMENT_SCOPE_REQUIRED")
        self.blocked(self.payload(stage="limited_smoke"))  # its own authorization-bound scope is required
        self.blocked(self.payload(stage="ai_reviewed"), "UNKNOWN_VALIDATION_STAGE")


class AcceptanceSamplingRuleTests(unittest.TestCase):
    """acceptance-sampling-v1 on authored item stubs (sources of 80 and 30 items): the review rule only."""

    def setUp(self) -> None:
        self.items = {f"{source}:{i:03d}": {"actor": SimpleNamespace(source=source, image_sha256=(f"{source}{i}",))}
                      for source, n in (("big", 80), ("small", 30)) for i in range(n)}
        self.order = {source: a.ai_review_order([pid for pid in self.items if pid.startswith(source)], 3407)
                      for source in ("big", "small")}
        self.leak = self.order["big"][70]  # a manifest leak warning outside every reviewed prefix

    def rule(self, big: int, small: int = 20, failed=(), leak_free: bool = True, **changes):
        sources = {source: [{"problem_id": pid, "image_sha256": list(self.items[pid]["actor"].image_sha256),
                             "render_ok": True, "leak_free": True, "gold_supported": pid not in failed, "notes": ""}
                            for pid in self.order[source][:count]] for source, count in (("big", big), ("small", small))}
        review = {"format": a.AI_LOCAL_REVIEW_FORMAT_V2, "acceptance_rule": a.ACCEPTANCE_RULE, "reviewer_id": "ai",
                  "human_verified": False, "labels_modified": False, "reviewed_at_utc": "2026-10-30T00:00:00+00:00",
                  "sampling_seed": 3407, "input_files_sha256": "f" * 64, "sources": sources,
                  "leak_warnings": [{"problem_id": self.leak, "field": "question", "leak_free": leak_free, "notes": ""}]}
        blocked, unresolved, quality = a._ai_local_review(review | changes, "ai", self.items,
            {"train": [{"problem_id": self.leak, "field": "question"}]}, "f" * 64, 3407, set(self.items))
        self.assertEqual(unresolved, int(not leak_free))
        return blocked, quality

    def eligible(self, blocked: dict, quality: dict | None) -> set[str]:
        return set(a.ai_local_eligibility({pid: i["actor"].source for pid, i in self.items.items()}, self.items,
                                          blocked, quality)["eligible_ids"])

    def test_zero_failures_pass_with_twenty_reviewed(self) -> None:
        blocked, quality = self.rule(20)
        self.assertEqual(blocked, {})
        assert quality is not None
        self.assertEqual((quality["rule"], quality["bound"], quality["confidence"]),
                         (a.ACCEPTANCE_RULE, "clopper-pearson-one-sided-upper", .95))
        self.assertEqual(quality["sources"]["big"], {"reviewed": 20, "failed": 0, "observed_rate": 0.0,
                                                     "cp_upper": 0.1391083407, "failed_ids": [], "status": "ELIGIBLE"})
        self.assertEqual(self.eligible(blocked, quality), set(self.items))

    def test_one_failure_in_twenty_blocks_until_sixty_are_reviewed(self) -> None:
        bad = self.order["big"][3]
        blocked, quality = self.rule(20, failed={bad})
        self.assertEqual(blocked, {"big": ["ACCEPTANCE_SAMPLE_TOO_SMALL failed=1 reviewed=20 required=60",
                                           f"FAILED_VERDICT gold_supported {bad}"]})
        self.assertEqual(quality["sources"]["big"]["status"], "BLOCKED")
        self.assertFalse({pid for pid in self.eligible(blocked, quality) if pid.startswith("big")})
        blocked, quality = self.rule(60, failed={bad})
        self.assertEqual(blocked, {})
        self.assertEqual(self.eligible(blocked, quality), set(self.items) - {bad})

    def test_two_failures_in_sixty_are_ineligible_and_three_block(self) -> None:
        two = set(self.order["big"][10:12])
        blocked, quality = self.rule(60, failed=two)
        self.assertEqual(blocked, {})
        self.assertEqual(quality["sources"]["big"], {"reviewed": 60, "failed": 2, "observed_rate": round(2 / 60, 10),
                                                     "cp_upper": 0.1012358949, "failed_ids": sorted(two),
                                                     "status": "ELIGIBLE"})
        self.assertEqual(self.eligible(blocked, quality), set(self.items) - two)
        three = set(self.order["big"][10:13])
        blocked, quality = self.rule(80, failed=three)
        self.assertEqual(blocked["big"][0], "ACCEPTANCE_TOO_MANY_FAILED failed=3 max=2 reviewed=80")
        self.assertEqual((quality["sources"]["big"]["failed_ids"], list(blocked)), (sorted(three), ["big"]))
        self.assertEqual(self.eligible(blocked, quality), {pid for pid in self.items if pid.startswith("small")})
        # A leak-warning row with leak_free=false is one more failure of its source.
        blocked, quality = self.rule(60, failed=two, leak_free=False)
        self.assertIn("ACCEPTANCE_TOO_MANY_FAILED failed=3 max=2 reviewed=60", blocked["big"])
        self.assertIn(f"LEAK_WARNING_NOT_LEAK_FREE question {self.leak}", blocked["big"])
        blocked, quality = self.rule(60, leak_free=False)
        self.assertEqual((blocked, quality["sources"]["big"]["failed_ids"]), ({}, [self.leak]))
        self.assertNotIn(self.leak, self.eligible(blocked, quality))

    def test_small_source_needs_every_item_reviewed(self) -> None:
        bad = self.order["small"][0]
        blocked, _ = self.rule(20, small=29, failed={bad})
        self.assertEqual(blocked["small"][0], "ACCEPTANCE_SAMPLE_TOO_SMALL failed=1 reviewed=29 required=30")
        blocked, quality = self.rule(20, small=30, failed={bad})
        self.assertEqual((blocked, quality["sources"]["small"]["failed_ids"]), ({}, [bad]))

    def test_v1_review_keeps_the_old_whole_source_block(self) -> None:
        bad = self.order["big"][3]
        review = {"format": a.AI_LOCAL_REVIEW_FORMAT}
        sources = {s: [{"problem_id": p, "image_sha256": list(self.items[p]["actor"].image_sha256), "render_ok": True,
                        "leak_free": True, "gold_supported": p != bad, "notes": ""} for p in self.order[s][:60]]
                   for s in ("big", "small")}
        blocked, unresolved, quality = a._ai_local_review(review | {
            "reviewer_id": "ai", "human_verified": False, "labels_modified": False,
            "reviewed_at_utc": "2026-10-30T00:00:00+00:00", "sampling_seed": 3407, "input_files_sha256": "f" * 64,
            "sources": sources, "leak_warnings": [{"problem_id": self.leak, "field": "question", "leak_free": True,
                                                   "notes": ""}]}, "ai", self.items,
            {"train": [{"problem_id": self.leak, "field": "question"}]}, "f" * 64, 3407, set(self.items))
        self.assertEqual((blocked, unresolved, quality), ({"big": [f"FAILED_VERDICT gold_supported {bad}"]}, 0, None))
        self.assertNotIn("quality", a.ai_local_eligibility({p: i["actor"].source for p, i in self.items.items()},
                                                           self.items, blocked))
        # The rule field selects the format: missing, unknown or attached to v1 is malformed.
        for changes in ({"acceptance_rule": "acceptance-sampling-v0"}, {"format": a.AI_LOCAL_REVIEW_FORMAT}):
            with self.subTest(changes=changes), self.assertRaisesRegex(a._Blocked, "AI_LOCAL_REVIEW_BINDING"):
                self.rule(60, **changes)

    def test_clopper_pearson_upper_bound(self) -> None:
        # Reference values: scipy.stats.beta.ppf(0.95, x + 1, n - x), rounded to 10 digits.
        for (x, n), expected in {(0, 20): 0.1391083407, (0, 60): 0.0487029133, (1, 60): 0.0766399949,
                                 (2, 60): 0.1012358949, (3, 60): 0.1241872105, (1, 12): 0.3386806684,
                                 (2, 30): 0.1953260437}.items():
            self.assertAlmostEqual(a.clopper_pearson_upper(x, n), expected, places=9, msg=(x, n))
        for n in (1, 7, 60, 400):
            self.assertAlmostEqual(a.clopper_pearson_upper(0, n), 1 - .05 ** (1 / n), places=9)
            self.assertAlmostEqual(a.clopper_pearson_upper(n - 1, n), .95 ** (1 / n), places=9)
        self.assertEqual((a.clopper_pearson_upper(5, 5), a.clopper_pearson_upper(0, 0), a.clopper_pearson_upper(9, 4)),
                         (1.0, 1.0, 1.0))


# Producer CLI with torch/pyarrow unimportable (the notebook-data env has neither).
NO_TORCH = ("import runpy, sys; sys.modules.update(torch=None, pyarrow=None); "
            "runpy.run_module('fine_tuning.notebook_rl.pool_data_evidence', run_name='__main__', alter_sys=True)")


def producer_cli(*argv: str):
    done = subprocess.run([sys.executable, "-B", "-c", NO_TORCH, *argv], capture_output=True, text=True,
                          timeout=600, check=False)
    if done.returncode:
        raise AssertionError(done.stdout[-2000:] + done.stderr[-4000:])
    return json.loads(done.stdout)


class PoolEvidenceProducerTests(unittest.TestCase):
    """Real generator regeneration and the real approval gate checks on an authored scan."""

    def test_producer_writes_d1_to_d8_that_the_approval_stage_accepts(self) -> None:
        from fine_tuning.notebook_rl import contamination as c, pool_data_evidence as producer
        temp = tempfile.TemporaryDirectory(prefix="notebook-pool-evidence-")
        self.addCleanup(temp.cleanup)
        base = Path(temp.name)
        generated = generate_pool("graph_pair", 1, 3407, "train", 1.0)
        pools = {split: [hf_problem("hf-fixture-a", split, 10 * i), hf_problem("hf-fixture-b", split, 10 * i + 1)]
                 for i, split in enumerate(SPLITS)}
        pools["train"] += generated
        s = scanned_build(base, pools, {"hf-fixture-a": source_entry("HUMAN"), "hf-fixture-b": source_entry("HUMAN"),
            "gen-graph-pair-v1": source_entry("PROGRAM", generated=True, revision=generated[0].lineage.source_revision)})
        self.assertEqual(s["receipt"]["status"], "EXCLUSION_CHECKED_WITH_LIMITS", s["receipt"]["blockers"])
        # A quarantined family stays in the build; it is never sampled, regenerated or eligible.
        held = next(pid for pid, item in s["items"].items() if item["split"] == "dev" and item["actor"].source == "hf-fixture-a")
        receipt = s["receipt"]
        receipt["candidates"] = [{"problem_id": held, "stage": "text_ngrams", "reason": "AUTHORED_TEST", "action": "QUARANTINE"}]
        family = c.family_quarantine(s["items"], receipt["candidates"])
        receipt.update(quarantined_ids=family["quarantined_ids"], allowlist=family["allowlist"],
                       quarantined_families=family["families"])
        receipt["receipt_sha256"] = a._hash({k: v for k, v in receipt.items() if k != "receipt_sha256"})
        s["manifest"]["exclusion_receipt_sha256"] = _write_json(s["root"] / "contamination/receipt_v2.json", receipt)
        _write_json(s["root"] / "manifest.yaml", s["manifest"])
        manifests = {split: json.loads((s["root"] / split / "manifest.yaml").read_bytes()) for split in SPLITS}
        sample = producer.review_sample(s["root"])
        self.assertEqual(sample, {"leak_warnings": [], "sources": {source: a.ai_review_order(
            [pid for pid, item in s["items"].items() if item["actor"].source == source and pid != held], 3407)
            for source in ("gen-graph-pair-v1", "hf-fixture-a", "hf-fixture-b")}})
        self.assertEqual(producer_cli("review-sample", "--build", str(s["root"])), sample)
        review_path = base / "review.json"
        _write_json(review_path, review_record(s["items"], manifests, s["receipt"]["input_files_sha256"], quarantined={held}))
        benchmark = base / "benchmark-probe.txt"
        benchmark.write_text("authored stand-in benchmark file for the denial probe")
        paths = {"index": Path(s["descriptor"]["path"]), "source_registry": base / "registry.json",
                 "evidence": base / "evidence.json"}
        _write_json(paths["source_registry"], s["registry"])
        _write_json(paths["evidence"], s["evidence"])
        descriptors = {name: {"path": str(path), "file_sha256": _sha(path.read_bytes())} for name, path in paths.items()}
        out = producer_cli("produce", "--build", str(s["root"]), "--review", str(review_path), "--benchmark-probe",
                           str(benchmark), "--index", str(paths["index"]), "--source-registry", str(paths["source_registry"]),
                           "--evidence", str(paths["evidence"]), "--out", str(base / "evidence-out"))
        self.assertEqual((out["status"], out["training_allowed"], out["human_verified"]),
                         ("AI_LOCAL_DATA_EVIDENCE_WRITTEN_NOT_APPROVAL", False, False))
        self.assertEqual((out["eligibility"]["quarantined_count"], out["eligibility"]["blocked_sources"]), (1, {}))
        request = json.loads(Path(out["data_request"]["path"]).read_text())
        d4 = json.loads(Path(json.loads((base / "evidence-out/data_report.json").read_text())["gates"]["D4"]["artifact"]["path"]).read_text())
        self.assertEqual(d4["measurements"]["templates"]["gen-graph-pair-v1:graph-pair-shortest"]["method"],
                         producer.INTERVENTION_METHOD)
        with patch.object(c, "load_index", return_value=s["index"]):
            result = a._validate(request | {"stage": a.AI_LOCAL_STAGE})
        self.assertEqual(result["status"], "VALIDATED_FOR_MEASUREMENT", result["blockers"])
        self.assertEqual((result["stage"], result["human_verified"]), (a.AI_LOCAL_STAGE, False))
        self.assertNotIn(held, result["eligibility"]["eligible_ids"])
        # A failing verdict blocks its source only; the producer and the approval agree.
        failing = review_record(s["items"], manifests, s["receipt"]["input_files_sha256"], quarantined={held})
        failing["sources"]["hf-fixture-b"][0]["gold_supported"] = False
        _write_json(base / "failing.json", failing)
        out = producer.produce(s["root"], {"path": str(base / "failing.json"), "file_sha256": _sha((base / "failing.json").read_bytes())},
                               benchmark, base / "failing-out", **descriptors)
        self.assertEqual(list(out["eligibility"]["blocked_sources"]), ["hf-fixture-b"])
        request = json.loads(Path(out["data_request"]["path"]).read_text())
        with patch.object(c, "load_index", return_value=s["index"]):
            result = a._validate(request | {"stage": a.AI_LOCAL_STAGE})
        self.assertEqual((result["status"], list(result["eligibility"]["blocked_sources"])),
                         ("VALIDATED_FOR_MEASUREMENT", ["hf-fixture-b"]), result["blockers"])
        self.assertNotIn("hf-fixture-b", {actor["source"] for actor in result["actors"]})
        # Every source blocked: no evidence report is written.
        for rows in failing["sources"].values():
            rows[0]["render_ok"] = False
        _write_json(base / "all-failing.json", failing)
        with self.assertRaisesRegex(a._Blocked, "NO_ELIGIBLE_SOURCE"):
            producer.produce(s["root"], {"path": str(base / "all-failing.json"),
                                         "file_sha256": _sha((base / "all-failing.json").read_bytes())},
                             benchmark, base / "all-failing-out", **descriptors)
        self.assertFalse((base / "all-failing-out/data_report.json").exists())


QUANTIZATION_FIXTURE = {"fixture": "authored tiny CPU Qwen/PEFT policy; not NF4 and not Qwen-4B"}


def tiny_policy(model_path, arm, protocol, **_):
    from fine_tuning.tests.test_notebook_rl_checkpoint import fresh_tiny
    bundle = fresh_tiny(model_path, arm, protocol)
    bundle.inventory["quantization"] = dict(QUANTIZATION_FIXTURE)
    return bundle


def tiny_logprobs(bundle, inputs, completion_len, *, chunk=None):
    """Differentiable stand-in forward: equals the sampler log-probabilities, gradient one per trainable value."""
    groups = ("L", "M", "V") if bundle.route.vision_on else ("L", "M")
    shift = sum((p - p.detach()).double().sum() for group in groups for p in bundle.params[group])
    return torch.full((1, completion_len), -0.1, dtype=torch.float64) + shift


def tiny_reference(bundle, inputs, completion_len, *, chunk=None):
    return torch.full((1, completion_len), -0.1, dtype=torch.float64)


def drive(config_path: str) -> None:
    """Fresh-process entry: the CPU fixture boundary around the real local_run CLI stages."""
    config = json.loads(Path(config_path).read_text())
    with ExitStack() as stack:
        for target, value in _boundary(config, config["minimal_ids"]).items():
            stack.enter_context(patch(target, value))
        code = local_run.main(config["argv"])
    raise SystemExit(code)


def probe_fixture(problem_ids: list[str]) -> dict:
    """repair_replay's single+two-image probe fixture, or only its single-image row for a one-probe coverage set."""
    from fine_tuning.tests.test_notebook_rl_repair_replay import probe_fixture as pair_fixture

    if len(problem_ids) == 2:
        return pair_fixture(problem_ids)
    return {key: value[:1] if type(value) is list else value
            for key, value in pair_fixture([*problem_ids, "unused-two-image-row"]).items()}


def _boundary(config: dict, ids: list[str]) -> dict:
    """CPU fixture patches around the real stages; `ids` are the synchronized probe IDs."""
    root = Path(config["build_root"])
    resume = checkpoint.load_resume

    def load_resume(model_path, ckpt, protocol, *, binding, **_):
        return resume(model_path, ckpt, protocol, binding=binding, _test_policy_factory=tiny_policy)

    return {
        "fine_tuning.notebook_rl.approval.validate_for_ai_reviewed_local": fake_validation,
        "fine_tuning.notebook_rl.local_run._install_actor_guards": lambda *args, **kwargs: [],
        "fine_tuning.notebook_rl.quantized_policy.build_quantized_policy": tiny_policy,
        "fine_tuning.notebook_rl.quantized_policy.load_quantized_policy": tiny_policy,
        "fine_tuning.notebook_rl.local_run.cache_readiness": lambda *args, **kwargs: {"original_model_root": str(root)},
        "fine_tuning.notebook_rl.quantized_cache.block_original_access":
            lambda path: {"denied_original_access_attempts": 0},
        "fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged":
            __import__("fine_tuning.notebook_rl.policy", fromlist=["x"]).assert_base_unchanged,
        "fine_tuning.notebook_rl.local_calibrate.verify_original": lambda path: {"fixture": "no BF16 original"},
        "fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine":
            lambda *args, policy_version=0, **kwargs: ScriptedEngine(config["replies"], policy_version=policy_version),
        "fine_tuning.notebook_rl.local_run.make_probes": lambda *args, **kwargs: [],
        "fine_tuning.notebook_rl.local_run.probe_evidence": lambda engine, probes: probe_fixture(list(ids)),
        "fine_tuning.notebook_rl.trainer.synchronize": lambda *args, **kwargs: [
            {"request_id": "local-sync:" + pid, "mean_abs_delta": 0.0, "max_abs_delta": 0.0, "policy_version": args[5]}
            for pid in ids],
        "fine_tuning.notebook_rl.updates.completion_logprobs": tiny_logprobs,
        "fine_tuning.notebook_rl.updates.reference_logprobs": tiny_reference,
        "fine_tuning.notebook_rl.checkpoint.load_resume": load_resume,
        "fine_tuning.notebook_rl.repair_replay.replay_trees": lambda *args, **kwargs: _replay_receipts(config, *args),
        "torch.cuda.reset_peak_memory_stats": lambda *args, **kwargs: None,
        "torch.cuda.max_memory_allocated": lambda *args, **kwargs: 0,
        "torch.cuda.max_memory_reserved": lambda *args, **kwargs: 0}


class PoolEndToEndTests(PoolFixture):
    """prepare-pool -> minimal (flags) -> real tiny update/save -> fresh-process reload -> evaluate-pool -> replay-pool."""

    def run_stage(self, name: str, argv: list[str], config: dict) -> None:
        path = self.base / f"{name}-config.json"
        path.write_text(json.dumps(config | {"argv": argv}))
        env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
        done = subprocess.run([sys.executable, "-B", "-c", "import sys; from fine_tuning.tests.test_notebook_rl_pool_local "
                               "import drive; drive(sys.argv[1])", str(path)],
                              capture_output=True, text=True, timeout=900, env=env, check=False)
        self.assertEqual(done.returncode, 0, done.stdout[-3000:] + done.stderr[-6000:])

    def test_minimal_update_reload_evaluate_and_replay(self) -> None:
        from fine_tuning.tests.test_notebook_rl_checkpoint import fixture as tiny_fixture
        from fine_tuning.tests.test_notebook_rl_reasoning_trace import fixture_trace, response
        from fine_tuning.notebook_rl.local_train import REPAIR_FUNCTIONAL_EXPLORATION, training_parent_gate

        rule_path = self.base / "rule.json"
        self.assertEqual(local_run.main(["propose-pool", "--build", str(self.root), "--evaluation-count", "2",
                                         *protocol_args(), "--output", str(rule_path)]), 0)
        rule = json.loads(rule_path.read_text())
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=fake_validation):
            self.assertEqual(local_run.main(["prepare-pool", "--build", str(self.root), "--approval", str(self.request),
                "--rule", str(rule_path), *protocol_args(), "--output", str(self.base / "prepared")]), 0)
        preparation = self.base / "prepared/preparation.json"
        model_path, *_ = tiny_fixture(self.base / "model")
        public = local_data.public_pool(self.root)
        slotless_id, obs_id = rule["minimal"]
        graph_problem = next(p for p in obs_problems() if p.actor.problem_id == obs_id)
        correct = response(graph_problem, fixture_trace(graph_problem))
        replies = {slotless_id: {"T1": [RIGHT, WRONG, NO_FINAL, RIGHT], "T2": [CHECK + RIGHT] * 2, "T3": [CHECK + RIGHT] * 2,
                                 "REPAIR_NOTEBOOK": [note([{"attempt": 3, "type": "NO_FINAL_ANSWER"}]),
                                                     note([{"attempt": 1, "type": "NO_FINAL_ANSWER"}]), note([]), "{"]},
                   obs_id: {"T1": [correct] * 4, "T2": [CHECK + correct] * 2, "T3": [CHECK + correct] * 2,
                            "REPAIR_NOTEBOOK": [note([])] * 4}}
        config = {"build_root": str(self.root), "minimal_ids": rule["minimal"], "replies": replies}
        capture, reload = self.base / "minimal", self.base / "reload"
        self.run_stage("minimal", ["minimal", "--preparation", str(preparation), "--model-path", str(model_path),
                                   "--standalone-base", "--cache-check", str(self.base / "authored-cache-check"),
                                   "--output", str(capture), "--repair-notebook", "--context-limit", "24576",
                                   *protocol_args()], config)
        result = json.loads((capture / "result.json").read_text())
        self.assertEqual(result["status"], "UPDATED_SAVED_PENDING_FRESH_RELOAD", result)
        self.assertEqual((result["optimizer_updates"], result["attempted_ids"]), (1, rule["minimal"]))
        self.assertEqual(result["answer_verified_ids"], [slotless_id])
        self.assertTrue(result["answer_verified_observed"])
        self.assertFalse(result["vision_off_only_observed"] or result["vision_on_observed"])
        counts = result["changed_tensor_counts"]
        self.assertTrue(counts["L"] > 0 and counts["M"] > 0 and counts["V"] == 0, counts)
        launch = json.loads((capture / "launch.json").read_text())
        self.assertEqual(launch["protocol"]["sha256"], self.protocol["sha256"])
        update = json.loads((capture / "update.json").read_text())
        self.assertEqual(update["route_labels"], {slotless_id: "ANSWER_VERIFIED"})
        self.assertTrue(update["retained_repair_signal"]["channel_signal"]["finalization"])
        self.run_stage("reload", ["reload", "--capture", str(capture), "--output", str(reload)], config)
        restored = json.loads((reload / "result.json").read_text())
        self.assertEqual(restored["status"], "FRESH_PROCESS_RELOAD_VERIFIED")
        self.assertTrue(restored["repair_functional_verified"] and restored["all_trainable_hashes_match"])
        self.assertNotEqual(restored["pid"], launch["pid"])
        # The existing local_train parent gate accepts the pool-bound minimum through its unchanged checks.
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=fake_validation), \
                patch("fine_tuning.notebook_rl.repair_replay.replay_trees",
                      side_effect=lambda *args, **kwargs: _replay_receipts(config, *args)):
            gate = training_parent_gate(capture, reload, admission=REPAIR_FUNCTIONAL_EXPLORATION)
        self.assertTrue(gate["repair_functional_verified"])
        # The saved FT tensors load and match before the pool evaluation.
        expected = json.loads((capture / "reload_expected.json").read_text())
        bundle = checkpoint.load_eval(model_path, capture / "checkpoint", binding=launch["binding"],
                                      _test_policy_factory=tiny_policy)
        self.assertEqual(local_run.parameter_hashes(bundle), expected["trainable_sha256"])
        engine = ScriptedEngine(eval_replies(rule["evaluation"]), policy_version=1)
        sha = _sha(preparation.read_bytes())
        record = local_pool_eval.diagnose(preparation, self.base / "evaluation", engine, self.prompts,
            expected_preparation_sha256=sha, local_protocol_sha256=self.protocol["sha256"],
            deadline=time.monotonic() + 300, context_limit=24576, note_contract=compact_note.VERSION,
            memory_probe=lambda: 0)
        self.assertEqual(record["status"], local_pool_eval.COMPLETE, record.get("reason"))
        self.assertEqual(record["problem_ids"], rule["evaluation"])
        self.assertEqual({public[pid][0].source for pid in record["problem_ids"]} & {public[pid][0].source for pid in rule["fixed"]},
                         set())
        output = self.base / "replay.json"
        self.assertEqual(local_eval.main(["replay-pool", "--record", str(self.base / "evaluation/result.json"),
                                          "--preparation-sha256", sha, "--output", str(output)]), 0)
        self.assertEqual(json.loads(output.read_text())["metrics"], record["metrics"])


def _replay_receipts(config: dict, trees, actors, protocol, prepared, model_path, **_):
    """Fresh CPU receipts with the same scripted preparation the capture used (no HF processor exists here)."""
    from fine_tuning.notebook_rl.dataset import load_images
    from fine_tuning.notebook_rl.repair_training import prepare_tree_credit
    from fine_tuning.notebook_rl.verifier_service import VerifierService

    root, receipts = Path(config["build_root"]), {}
    by_id = {actor.problem_id: actor for actor in actors}
    with VerifierService(root / "train", forbidden_roots=[], timeout=60) as verifier:
        for tree in trees:
            # Same-policy preparation, as the real replay sets per tree (all policy 0 in the minimal stage).
            engine = ScriptedEngine(config["replies"], policy_version=tree.policy_version)
            images = load_images(by_id[tree.problem_id], root / "images")
            receipts[tree.problem_id] = prepare_tree_credit(tree, by_id[tree.problem_id], verifier, engine.tokenizer,
                                                            preparation=engine, images=images, master_seed=3407)
            for image in images:
                image.close()
    return receipts



# ---------------------------------------------------------------- existing-datasets-only pool (v2)

V2_SLOTLESS = (*HF_TRAIN, *HF_HELD)
SCALE_12_4 = {"train_problems": 12, "problems_per_update": 4, "max_updates": 3}
SCALE_4_2 = {"train_problems": 4, "problems_per_update": 2, "max_updates": 2}
EXCLUDED = ("hf-chebi", "hf-ptbxl")


def v2_build(root: Path, *, gold: str = "B", paired: bool = False) -> None:
    """Existing-dataset fixtures, program-generator items and the two excluded HF sources.

    Every existing-dataset item is single-image unless `paired`, which gives hf-fixture-f's held-out items two
    images, so a proposed evaluation set (one item per held-out source) holds a two-image actor."""
    pools: dict[str, list[GeneratedProblem]] = {split: [] for split in SPLITS}
    held = partial(hf_problem, gold=gold, images=2 if paired else 1)
    pools["train"] = [hf_problem(source, "train", 10 * i + k, gold=gold) for i, source in enumerate(HF_TRAIN)
                      for k in range(4)]
    pools["train"] += [hf_problem("hf-chebi", "train", 60, gold=gold), hf_problem("hf-ptbxl", "train", 61, gold=gold)]
    pools["train"] += obs_problems() + [plain_problem("train", 1)]
    pools["dev"] = [hf_problem("hf-fixture-e", "dev", 70, gold=gold), held("hf-fixture-f", "dev", 71),
                    hf_problem("hf-ptbxl", "dev", 72, gold=gold)]
    pools["sealed"] = [hf_problem("hf-fixture-e", "sealed", 80, gold=gold), hf_problem("hf-chebi", "sealed", 81, gold=gold)]
    pools["style_sealed"] = [held("hf-fixture-f", "style_sealed", 90)]
    build_splits(root, pools)
    allowlist_receipt(root)


def v2_protocol(scale: dict | None = None) -> dict:
    return local_profile.build_protocol(repair_notebook=True, context_limit=24576, note_contract=compact_note.VERSION,
                                        answer_verified_route=True, finalization_reward=True,
                                        slotless_sources=V2_SLOTLESS, **({"scale": scale} if scale else {}))


def v2_args(scale: dict | None = None) -> list[str]:
    flags = ["--note-contract", compact_note.VERSION, "--answer-verified-route", "--finalization-reward",
             *[value for source in V2_SLOTLESS for value in ("--slotless-source", source)]]
    return flags + [value for key, number in (scale or {}).items() for value in ("--" + key.replace("_", "-"), str(number))]


def rewrite_json(path: Path, value) -> dict:
    """Overwrite an output (tamper fixture) and return its new descriptor."""
    if path.exists():
        path.chmod(0o600)
    path.write_text(json.dumps(value))
    return {"path": str(path), "file_sha256": _sha(path.read_bytes())}


class PoolV2Fixture(PoolFixture):
    """Default: a two-image held-out actor is evaluated, so both probes stay required (the pre-coverage behaviour)."""
    scale: dict | None = SCALE_12_4
    paired = True

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="notebook-pool-v2-")
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / "build"
        v2_build(self.root, paired=self.paired)
        self.protocol = v2_protocol(self.scale)
        self.prompts = local_profile.local_prompts(self.protocol)
        self.request = write_request(self.base, self.root)

    def propose_v2(self, root: Path | None = None, **options) -> dict:
        options = {"evaluation_count": 2, "sizes": local_data.pool_sizes(self.protocol)} | options
        return local_data.propose_pool_rule(root or self.root, self.prompts, version="v2", **options)


class PoolV2SelectionTests(PoolV2Fixture):
    def test_v2_proposal_is_deterministic_gold_invariant_and_existing_datasets_only(self) -> None:
        rule = self.propose_v2()
        self.assertEqual(rule, self.propose_v2())
        public = local_data.public_pool(self.root)
        self.assertEqual((rule["version"], rule["policy"], rule["excluded_sources"]),
                         (local_data.POOL_RULE_VERSION_V2, local_data.POOL_DATA_POLICY,
                          sorted(local_data.DEFAULT_EXCLUDED_SOURCES)))
        self.assertEqual((len(rule["fixed"]), rule["minimal"]), (12, rule["fixed"][:2]))
        self.assertEqual({public[pid][0].source for pid in rule["fixed"]}, set(HF_TRAIN))
        self.assertEqual({public[pid][0].source for pid in rule["evaluation"]}, set(HF_HELD))
        for pid in rule["fixed"] + rule["evaluation"]:
            actor, lineage = public[pid]
            self.assertFalse(actor.source.startswith("gen-") or actor.source in EXCLUDED, pid)
            self.assertNotEqual(a.label_class(lineage), "GENERATOR")
        # An evaluated item has two images but no training item does: the first approved two-image train actor in
        # hash order completes the covered counts as a probe only.
        self.assertEqual(sorted({len(public[pid][0].image_paths) for pid in rule["evaluation"]}), [1, 2])
        self.assertEqual(rule["probe_coverage"], {"rule": local_data.PROBE_COVERAGE_RULE, "image_counts": [1, 2]})
        train = [pid for pid in local_data._order(local_data.pool_allowlist(self.root, public), 3407)
                 if public[pid][1].split == "train"]
        self.assertEqual(rule["probe_only"], [next(pid for pid in train if len(public[pid][0].image_paths) == 2)])
        self.assertEqual(public[rule["probe_only"][0]][0].source, "gen-graph-pair-v1")
        # Different private gold with identical public inputs: the same IDs and order; verifier files are never read.
        other = self.base / "other-gold"
        v2_build(other, gold="D", paired=self.paired)
        changed = self.propose_v2(other)
        self.assertNotEqual(changed["build_manifest_sha256"], rule["build_manifest_sha256"])
        self.assertEqual({k: v for k, v in changed.items() if k != "build_manifest_sha256"},
                         {k: v for k, v in rule.items() if k != "build_manifest_sha256"})
        for split in SPLITS:
            (other / split / "verifier_private.jsonl").unlink()
        self.assertEqual(self.propose_v2(other), changed)
        self.assertNotEqual(self.propose_v2(seed=11)["fixed"], rule["fixed"])
        # Default (unscaled) sizes: N=10; an extra exclusion only adds to the defaults.
        ten = self.propose_v2(sizes=(10, 2), excluded_sources=("hf-fixture-d",))
        self.assertEqual(len(ten["fixed"]), 10)
        self.assertEqual(ten["excluded_sources"], sorted({*local_data.DEFAULT_EXCLUDED_SOURCES, "hf-fixture-d"}))
        self.assertNotIn("hf-fixture-d", {public[pid][0].source for pid in ten["fixed"] + ten["evaluation"]})
        with self.assertRaisesRegex(ValueError, "v1 proposals"):
            local_data.propose_pool_rule(self.root, self.prompts, excluded_sources=("x",))

    def test_v2_rule_rejects_excluded_generator_resized_and_unused_probe_items(self) -> None:
        rule = self.propose_v2()
        public = local_data.public_pool(self.root)
        sizes = local_data.pool_sizes(self.protocol)
        check = partial(local_data.check_pool_rule_v2, public=public, prompts=self.prompts,
                        build_sha256=rule["build_manifest_sha256"], sizes=sizes)
        summary = check(rule)
        self.assertEqual(summary["batch_ids"], [rule["fixed"][i:i + 4] for i in (0, 4, 8)])
        self.assertEqual(summary["probe_ids"], [rule["minimal"][0], rule["probe_only"][0]])
        self.assertEqual(summary["probe_only_policy"], local_data.PROBE_ONLY_POLICY)

        def find(source: str, split: str = "train") -> str:
            return next(pid for pid, (actor, lin) in public.items() if actor.source == source and lin.split == split)
        spare = next(pid for pid, (actor, lin) in public.items()
                     if lin.split == "train" and actor.source in HF_TRAIN and pid not in rule["fixed"])
        swap = rule["fixed"][2]
        cases = {"excluded source in fixed": rule | {"fixed": [find("hf-chebi") if p == swap else p for p in rule["fixed"]]},
                 "generator item in fixed": rule | {"fixed": [find(PLAIN) if p == swap else p for p in rule["fixed"]]},
                 "excluded source in evaluation": rule | {"evaluation": [find("hf-ptbxl", "dev")]},
                 "default exclusion dropped": rule | {"excluded_sources": ["gen-*", "hf-chebi"]},
                 "other policy": rule | {"policy": "any-source"},
                 "unused probe-only": rule | {"probe_only": rule["probe_only"] + [spare]},
                 "probe-only also trained": rule | {"probe_only": [rule["fixed"][5]]},
                 "missing two-image probe": rule | {"probe_only": []},
                 "coverage narrowed to single-image": rule | {"probe_only": [], "probe_coverage": {
                     "rule": local_data.PROBE_COVERAGE_RULE, "image_counts": [1]}},
                 "other coverage rule": rule | {"probe_coverage": rule["probe_coverage"] | {"rule": "any-probes"}},
                 "no coverage record": {k: v for k, v in rule.items() if k != "probe_coverage"},
                 "minimal outside fixed": rule | {"minimal": [rule["minimal"][0], spare]},
                 "N differs from the bound scale": rule | {"fixed": rule["fixed"][:10]},
                 "v1 key set": {k: v for k, v in rule.items() if k not in ("probe_only", "excluded_sources", "policy")}}
        for name, changed in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                check(changed)
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            check(rule, eligible=set(rule["fixed"] + rule["evaluation"]))  # probe-only must be approved too
        with self.assertRaises(ValueError):
            local_data.check_pool_rule(rule, public, self.prompts, rule["build_manifest_sha256"])  # v1 checker

    def test_v2_prepare_binds_n_items_probe_set_and_consumers_read_it(self) -> None:
        rule = self.propose_v2()
        result, path = self.prepare(rule)
        self.assertEqual(result["status"], local_data.POOL_STATUS, result.get("blockers"))
        self.assertEqual((result["format"], result["data_policy"], result["human_verified"]),
                         (local_data.POOL_FORMAT_V2, local_data.POOL_DATA_POLICY, False))
        self.assertTrue(any(local_data.PROBE_ONLY_POLICY == text for text in result["limitations"]))
        selection = json.loads(Path(result["selection"]["path"]).read_text())
        self.assertEqual((selection["version"], selection["batch_ids"], selection["probe_only_ids"]),
                         (local_data.POOL_RULE_VERSION_V2, [rule["fixed"][i:i + 4] for i in (0, 4, 8)], rule["probe_only"]))
        public = local_data.public_pool(self.root)
        probes = [public[pid][0] for pid in (rule["minimal"][0], rule["probe_only"][0])]
        self.assertEqual(json.loads(Path(result["probe_actors"]["path"]).read_text()),
                         [json.loads(local_run.to_json(actor)) for actor in probes])
        self.assertNotIn(rule["probe_only"][0], {row["problem_id"] for row in json.loads(
            Path(result["actors"]["path"]).read_text())})
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=fake_validation):
            prepared, _, actors = local_run.read_preparation(path, self.protocol)
            self.assertEqual([actor.problem_id for actor in actors], rule["fixed"])
            self.assertEqual(local_run.minimal_training_actors(prepared, actors), [public[pid][0] for pid in rule["minimal"]])
            self.assertEqual(local_run.probe_source(prepared, actors), probes)
            self.assertEqual(local_run.minimal_actors(local_run.probe_source(prepared, actors)), probes)
            from fine_tuning.notebook_rl import scale
            self.assertEqual([[x.problem_id for x in batch] for batch in scale.batches(actors, self.protocol)],
                             selection["batch_ids"])
            # The v2 preparation is bound to its N/K protocol and to the approval's current eligibility.
            with self.assertRaises(ValueError):
                local_run.read_preparation(path, v2_protocol())

            def narrowed(root, **_):  # same training actors; one held-out ID no longer eligible
                value = fake_validation(root)
                value["eligibility"]["eligible_ids"].remove(rule["evaluation"][0])
                return value
            with patch.object(a, "validate_for_ai_reviewed_local", side_effect=narrowed), \
                    self.assertRaisesRegex(ValueError, "allowlisted"):
                local_run.read_preparation(path, self.protocol)
            self.assertEqual((local_run.probe_set(prepared, actors), local_run.probe_counts(prepared),
                              local_run.sync_options(prepared)), (probes, (1, 2), {}))
        self.assertEqual(result["probe_coverage"], rule["probe_coverage"])
        # Re-bound to single-image coverage with a matching one-probe file: refused, the evaluated two-image actor
        # still requires the two-image probe.
        narrowed_coverage = dict(result, probe_coverage={"rule": local_data.PROBE_COVERAGE_RULE, "image_counts": [1]},
            probe_actors=rewrite_json(self.base / "one-probe.json", [json.loads(local_run.to_json(probes[0]))]))
        rewrite_json(path, narrowed_coverage)
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=fake_validation), \
                self.assertRaisesRegex(ValueError, "probe coverage"):
            local_run.read_preparation(path, self.protocol)
        rewrite_json(path, result)
        probe_file = Path(result["probe_actors"]["path"])
        probe_file.chmod(0o600)
        probe_file.write_text(json.dumps([json.loads(local_run.to_json(probes[0]))] * 2))
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=fake_validation), \
                self.assertRaises(ValueError):
            local_run.read_preparation(path, self.protocol)

    def test_v2_evaluator_accepts_the_v2_preparation(self) -> None:
        rule = self.propose_v2()
        result, path = self.prepare(rule)
        engine = ScriptedEngine(eval_replies(rule["evaluation"]))
        record = local_pool_eval.diagnose(path, self.base / "evaluation", engine, self.prompts,
            expected_preparation_sha256=_sha(path.read_bytes()), local_protocol_sha256=self.protocol["sha256"],
            deadline=time.monotonic() + 300, context_limit=24576, note_contract=compact_note.VERSION)
        self.assertEqual((record["status"], record["problem_ids"]), (local_pool_eval.COMPLETE, rule["evaluation"]))

    def test_cli_v2_proposal_requires_and_forwards_the_approval_eligibility(self) -> None:
        output = self.base / "cli-v2-rule.json"
        argv = ["propose-pool", "--build", str(self.root), "--rule-version", "v2", "--evaluation-count", "2",
                *v2_args(self.scale), "--output", str(output)]
        with self.assertRaises(SystemExit):
            local_run.main(argv)  # v2 requires --approval
        blocked = {"hf-fixture-e": ["FAILED_VERDICT render_ok fixture"]}
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=lambda root, **_: fake_validation(root, blocked)):
            self.assertEqual(local_run.main(argv + ["--approval", str(self.request)]), 0)
        rule = json.loads(output.read_text())
        public = local_data.public_pool(self.root)
        self.assertEqual({public[pid][0].source for pid in rule["evaluation"]}, {"hf-fixture-f"})
        self.assertEqual(len(rule["fixed"]), 12)
        with self.assertRaises(SystemExit):
            local_run.main(["propose-pool", "--build", str(self.root), "--exclude-source", "hf-x", *v2_args(),
                            "--output", str(self.base / "bad.json")])  # v1 takes no exclusions


def drive_v2(config_path: str) -> None:
    """Fresh-process entry for the v2 stages: v1 boundary plus probe/collector recorders and local_train fixtures."""
    from fine_tuning.notebook_rl import local_train, rollout

    config = json.loads(Path(config_path).read_text())
    ids, record = config["probe_ids"], Path(config["record"])
    build_tree = rollout.build_tree

    def log(kind: str, values: list) -> None:
        with record.open("a") as stream:
            stream.write(json.dumps({"stage": config["argv"][0], "kind": kind, "ids": values}) + "\n")

    def recorded_tree(actor, *args, **kwargs):
        log("rollout", [actor.problem_id])
        return build_tree(actor, *args, **kwargs)

    def recorded_probes(actors, *args, image_counts=(1, 2), **kwargs):
        # The real selection over the counts each consumer passes (the preparation's bound coverage).
        log("probes", [actor.problem_id for actor in local_run.covering_actors(actors, image_counts)])
        return []

    def synchronize(bundle, preparation, engine, probes, protocol, step, *, push, image_counts=(1, 2)):
        log("sync", list(image_counts))
        engine.policy_version = step  # as the real synchronization does after each update
        return [{"request_id": "local-sync:" + pid, "mean_abs_delta": 0.0, "max_abs_delta": 0.0, "policy_version": step}
                for pid in ids]

    boundary = _boundary(config, ids) | {
        "fine_tuning.notebook_rl.trainer.synchronize": synchronize,
        "fine_tuning.notebook_rl.rollout.build_tree": recorded_tree,
        "fine_tuning.notebook_rl.local_run.make_probes": recorded_probes,
        "fine_tuning.notebook_rl.local_train.make_probes": recorded_probes,
        "fine_tuning.notebook_rl.local_train.probe_evidence": lambda engine, probes: probe_fixture(list(ids)),
        "fine_tuning.notebook_rl.local_train._install_actor_guards": lambda *args, **kwargs: [],
        "fine_tuning.notebook_rl.local_train.cache_readiness":
            lambda *args, **kwargs: {"original_model_root": config["build_root"]},
        "fine_tuning.notebook_rl.policy.enable_deterministic_execution": lambda: {"fixture_only": True},
        "torch.cuda.get_device_properties": lambda *args, **kwargs: "fixture CPU (no CUDA device)",
        "torch.cuda.get_device_name": lambda *args, **kwargs: "fixture CPU"}
    with ExitStack() as stack:
        for target, value in boundary.items():
            stack.enter_context(patch(target, value))
        module = {"local_train": local_train, "local_eval": local_eval}.get(config["module"], local_run)
        code = module.main(config["argv"])
    raise SystemExit(code)


class Turn1OptionTests(unittest.TestCase):
    def test_minimal_cli_binds_turn1_options_exactly_like_build_protocol(self) -> None:
        from fine_tuning.notebook_rl.local_profile import TURN1_EVAL_CONTRACT

        built: list[dict] = []

        def spy(**kwargs):
            built.append(local_profile.build_protocol(**kwargs))
            return built[-1]
        base = ["minimal", "--preparation", "p", "--model-path", "m", "--output", "o", "--repair-notebook",
                "--context-limit", "24576", *protocol_args()]
        flagged = dict(repair_notebook=True, context_limit=24576, note_contract=compact_note.VERSION,
                       answer_verified_route=True, finalization_reward=True, slotless_sources=SLOTLESS)
        cases = {(): flagged, ("--turn1-contract", TURN1_EVAL_CONTRACT): flagged | {"turn1_contract": TURN1_EVAL_CONTRACT},
                 ("--turn1-contract", TURN1_EVAL_CONTRACT, "--turn1-max-new-tokens", "2048"):
                     flagged | {"turn1_contract": TURN1_EVAL_CONTRACT, "turn1_max_new_tokens": 2048}}
        for extra, keywords in cases.items():
            with self.subTest(extra=extra), patch.object(local_run, "build_protocol", side_effect=spy), \
                    patch.object(local_run, "read_preparation", side_effect=ValueError("CPU boundary")), \
                    patch("traceback.print_exc"):
                self.assertEqual(local_run.main(base + list(extra)), 1)
                self.assertEqual(built[-1], local_profile.build_protocol(**keywords))
                self.assertEqual("repair_turn1_contract" in built[-1], bool(extra))
        self.assertEqual(len({protocol["sha256"] for protocol in built}), 3)
        unrouted = base[:10] + ["--note-contract", compact_note.VERSION]
        for argv in (base + ["--turn1-max-new-tokens", "2048"], unrouted + ["--turn1-contract", TURN1_EVAL_CONTRACT]):
            with self.subTest(argv=argv[10:]), patch("sys.stderr"), self.assertRaises(SystemExit):
                local_run.main(argv)


class V2StageRunner(unittest.TestCase):
    base: Path

    def run_stage(self, name: str, module: str, argv: list[str], config: dict) -> None:
        path = self.base / f"{name}-config.json"
        path.write_text(json.dumps(config | {"argv": argv, "module": module}))
        env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
        done = subprocess.run([sys.executable, "-B", "-c", "import sys; from fine_tuning.tests.test_notebook_rl_pool_local "
                               "import drive_v2; drive_v2(sys.argv[1])", str(path)],
                              capture_output=True, text=True, timeout=1800, env=env, check=False)
        self.assertEqual(done.returncode, 0, done.stdout[-3000:] + done.stderr[-6000:])


SLOTLESS_SCRIPT = {"T1": [RIGHT, WRONG, NO_FINAL, RIGHT], "T2": [CHECK + RIGHT] * 2, "T3": [CHECK + RIGHT] * 2,
                   "REPAIR_NOTEBOOK": [note([{"attempt": 3, "type": "NO_FINAL_ANSWER"}]),
                                       note([{"attempt": 1, "type": "NO_FINAL_ANSWER"}]), note([]), "{"]}


class PoolV2EndToEndTests(V2StageRunner, PoolV2Fixture):
    """propose (v2, approval) -> prepare-pool N=12 -> minimal (two slotless items; probe-only paired actor)
    -> fresh reload -> local_train 3 updates of K=4 -> fresh training reload; tiny CPU policy throughout.
    An evaluated two-image item keeps the two-image probe required in every stage."""

    def test_prepare_minimal_train_three_updates_and_reload(self) -> None:
        from fine_tuning.tests.test_notebook_rl_checkpoint import fixture as tiny_fixture
        from fine_tuning.notebook_rl.local_train import REPAIR_FUNCTIONAL_EXPLORATION

        rule_path = self.base / "rule.json"
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=fake_validation):
            self.assertEqual(local_run.main(["propose-pool", "--build", str(self.root), "--rule-version", "v2",
                "--approval", str(self.request), "--evaluation-count", "2", *v2_args(self.scale),
                "--output", str(rule_path)]), 0)
            self.assertEqual(local_run.main(["prepare-pool", "--build", str(self.root), "--approval", str(self.request),
                "--rule", str(rule_path), *v2_args(self.scale), "--output", str(self.base / "prepared")]), 0)
        rule = json.loads(rule_path.read_text())
        preparation = self.base / "prepared/preparation.json"
        model_path, *_ = tiny_fixture(self.base / "model")
        (model_path / "nf4_base.json").write_text('{"fixture_only":true}')  # cache identity the F0 record hashes
        slotless = SLOTLESS_SCRIPT
        probe_ids = [rule["minimal"][0], rule["probe_only"][0]]
        record = self.base / "collector.jsonl"
        # No script exists for the probe-only actor: any rollout request for it would fail the stage.
        config = {"build_root": str(self.root), "minimal_ids": rule["minimal"], "probe_ids": probe_ids,
                  "record": str(record), "replies": {pid: slotless for pid in rule["fixed"]}}
        capture, reload, trained, restored = (self.base / name for name in ("minimal", "reload", "train", "train-reload"))
        self.run_stage("minimal", "local_run", ["minimal", "--preparation", str(preparation), "--model-path", str(model_path),
            "--standalone-base", "--cache-check", str(self.base / "authored-cache-check"), "--output", str(capture),
            "--repair-notebook", "--context-limit", "24576", *v2_args(self.scale)], config)
        result = json.loads((capture / "result.json").read_text())
        self.assertEqual((result["status"], result["attempted_ids"]), ("UPDATED_SAVED_PENDING_FRESH_RELOAD", rule["minimal"]))
        self.assertEqual(result["answer_verified_ids"], rule["minimal"])
        self.run_stage("reload", "local_run", ["reload", "--capture", str(capture), "--output", str(reload)], config)
        self.assertEqual(json.loads((reload / "result.json").read_text())["status"], "FRESH_PROCESS_RELOAD_VERIFIED")
        self.run_stage("train", "local_train", ["train", "--preparation", str(preparation), "--model-path", str(model_path),
            "--cache-check", str(self.base / "authored-cache-check"), "--minimal", str(capture),
            "--minimal-reload", str(reload), "--output", str(trained), "--arm", "notebook",
            "--admission", REPAIR_FUNCTIONAL_EXPLORATION, *v2_args()], config)
        result = json.loads((trained / "result.json").read_text())
        self.assertEqual((result["status"], result["optimizer_updates"], result["attempted_ids"]),
                         ("LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD", 3, rule["fixed"]))
        launch = json.loads((trained / "launch.json").read_text())
        self.assertEqual(launch["execution_contract"]["pair_ids"], [rule["fixed"][i:i + 4] for i in (0, 4, 8)])
        self.run_stage("train-reload", "local_train", ["reload", "--capture", str(trained), "--output", str(restored)], config)
        reloaded = json.loads((restored / "result.json").read_text())
        self.assertEqual((reloaded["status"], reloaded["step"]), ("LOCAL_TEN_FRESH_PROCESS_RELOAD_VERIFIED", 3))
        self.assertTrue(reloaded["repair_functional_verified"])
        # Collector inputs: rollouts only ever saw training items; every probe set held the probe-only actor.
        rows = [json.loads(line) for line in record.read_text().splitlines()]
        rolled = {pid for row in rows if row["kind"] == "rollout" for pid in row["ids"]}
        self.assertEqual(rolled, set(rule["fixed"]))
        self.assertEqual({row["stage"] for row in rows if row["kind"] == "rollout"}, {"minimal", "train"})
        self.assertNotIn(rule["probe_only"][0], rolled)
        self.assertEqual({tuple(row["ids"]) for row in rows if row["kind"] == "probes"}, {tuple(probe_ids)})
        self.assertEqual({row["stage"] for row in rows if row["kind"] == "probes"}, {"minimal", "reload", "train"})
        self.assertEqual({tuple(row["ids"]) for row in rows if row["kind"] == "sync"}, {(1, 2)})
        trees = [json.loads(path.read_text()) for directory in (capture, trained) for path in sorted(directory.glob("raw-*.json"))]
        self.assertNotIn(rule["probe_only"][0], {row["raw"]["problem_id"] for row in trees if row["kind"] == "TREE"})


class PoolV2SingleImageCoverageTests(PoolV2Fixture):
    """probe-coverage-trained-image-counts-v1 on a pool whose trained and evaluated actors are all single-image."""
    paired = False
    scale = SCALE_4_2

    def test_single_image_rule_binds_one_probe_and_tampered_coverage_is_refused(self) -> None:
        rule = self.propose_v2()
        public = local_data.public_pool(self.root)
        single = {"rule": local_data.PROBE_COVERAGE_RULE, "image_counts": [1]}
        paired = {"rule": local_data.PROBE_COVERAGE_RULE, "image_counts": [1, 2]}
        self.assertEqual((len(rule["fixed"]), rule["probe_only"], rule["probe_coverage"]), (4, [], single))
        self.assertEqual({len(public[pid][0].image_paths) for pid in rule["fixed"] + rule["evaluation"]}, {1})
        check = partial(local_data.check_pool_rule_v2, public=public, prompts=self.prompts,
                        build_sha256=rule["build_manifest_sha256"], sizes=local_data.pool_sizes(self.protocol))
        summary = check(rule)
        self.assertEqual((summary["probe_ids"], summary["probe_coverage"]), ([rule["minimal"][0]], single))
        pair = next(pid for pid, (actor, lin) in public.items() if lin.split == "train" and len(actor.image_paths) == 2)
        cases = {"coverage claims a two-image probe": rule | {"probe_coverage": paired},
                 "two-image probe-only with widened coverage": rule | {"probe_only": [pair], "probe_coverage": paired},
                 "unused two-image probe-only": rule | {"probe_only": [pair]},
                 "empty coverage": rule | {"probe_coverage": single | {"image_counts": []}}}
        for name, changed in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                check(changed)

        result, path = self.prepare(rule)
        self.assertEqual(result["status"], local_data.POOL_STATUS, result.get("blockers"))
        self.assertEqual(result["probe_coverage"], single)
        self.assertIn(local_data.PROBE_COVERAGE_LIMITATION, result["limitations"])
        self.assertIn("quantized_cache", local_data.PROBE_COVERAGE_LIMITATION)
        probe = public[rule["minimal"][0]][0]
        self.assertEqual(json.loads(Path(result["probe_actors"]["path"]).read_text()),
                         [json.loads(local_run.to_json(probe))])
        selection = json.loads(Path(result["selection"]["path"]).read_text())
        self.assertEqual((selection["probe_ids"], selection["probe_coverage"]), ([probe.problem_id], single))
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=fake_validation):
            prepared, _, actors = local_run.read_preparation(path, self.protocol)
        self.assertEqual((local_run.probe_set(prepared, actors), local_run.probe_counts(prepared),
                          local_run.sync_options(prepared)), ([probe], (1,), {"image_counts": (1,)}))
        # Probes are built for exactly the bound counts; a count missing from the given actors is refused, never skipped.
        images, engine = {probe.problem_id: [Image.new("RGB", (8, 8))]}, ScriptedEngine({})
        made = local_run.make_probes([probe], images, engine, self.prompts, image_counts=(1,))
        self.assertEqual([p.request.request_id for p in made], ["local-sync:" + probe.problem_id])
        with self.assertRaisesRegex(ValueError, "coverage"):
            local_run.make_probes([probe], images, engine, self.prompts)
        two = [json.loads(local_run.to_json(x)) for x in (probe, public[pair][0])]
        tampers = {"coverage widened": dict(result, probe_coverage=paired),
                   "two-image probe appended with widened coverage": dict(
                       result, probe_coverage=paired, probe_actors=rewrite_json(self.base / "two-probes.json", two)),
                   "two-image probe appended": dict(result, probe_actors=rewrite_json(self.base / "two-probes-b.json", two)),
                   "probe file emptied": dict(result, probe_actors=rewrite_json(self.base / "no-probe.json", [])),
                   "coverage record dropped": {k: v for k, v in result.items() if k != "probe_coverage"}}
        for name, changed in tampers.items():
            rewrite_json(path, changed)
            with self.subTest(name), patch.object(a, "validate_for_ai_reviewed_local", side_effect=fake_validation), \
                    self.assertRaises(ValueError):
                local_run.read_preparation(path, self.protocol)


class SyncCoverageGuardTests(unittest.TestCase):
    """trainer/engine/scale synchronization take the caller's bound counts; the default stays the two-probe pair."""

    def test_one_probe_only_with_explicit_single_image_counts(self) -> None:
        from fine_tuning.notebook_rl import engines, scale, trainer

        one_image, two_images = hf_problem("hf-fixture-a", "train", 1).actor, hf_problem("hf-fixture-b", "train", 2, images=2).actor
        images = {a.problem_id: [Image.new("RGB", (8, 8))] * len(a.image_paths) for a in (one_image, two_images)}
        engine, prompts = SimpleNamespace(tokenizer=CharTokenizer(), policy_version=-1), local_profile.local_prompts(pool_protocol())
        one = local_run.make_probes([one_image], images, engine, prompts, image_counts=(1,))
        both = local_run.make_probes([two_images, one_image], images, engine, prompts)
        for probes, counts in ((one, (1, 2)), (both, (1,)), (one, (2,)), (one + one, (1, 1)), ([], ())):
            with self.subTest(probes=len(probes), counts=counts), self.assertRaisesRegex(ValueError, "synchronization probes"):
                trainer.synchronize(None, None, engine, probes, {}, 0, push=False, image_counts=counts)
        for probes, options in ((one, {"image_counts": (1,)}), (both, {})):
            with self.subTest(probes=len(probes)), self.assertRaisesRegex(RuntimeError, "policy version"):
                trainer.synchronize(None, None, engine, probes, {}, 0, push=False, **options)  # past the probe guard
        reference = [(p.request, p.completion_ids, [-.1] * len(p.completion_ids)) for p in one]
        fake = SimpleNamespace(_push_complete=True, policy_version=3, probe_logprobs=lambda request, ids: [-.1] * len(ids))
        with self.assertRaisesRegex(ValueError, "single/multi-image"):
            engines.InputPreparation.validate_sync(fake, reference, .02)
        reports = engines.InputPreparation.validate_sync(fake, reference, .02, image_counts=(1,))
        self.assertEqual([(r["request_id"], r["policy_version"]) for r in reports], [(one[0].request.request_id, 3)])
        seen: list[dict] = []
        # The HF policy samples each sampler probe's continuation (scale.policy_sampled_probes) before vLLM is probed.
        hf = SimpleNamespace(sample=lambda requests: [RolloutResult(r.request_id, [1], [Completion([5, 6], [-.1, -.2],
                                                                                                "length", "")])
                                                      for r in requests])
        vllm = object()
        with patch.object(trainer, "synchronize", side_effect=lambda *args, **kwargs: seen.append(kwargs) or []):
            scale.synchronize(None, hf, hf, one, {}, 0, push=False, image_counts=(1,))
            scale.synchronize(None, vllm, hf, both, {}, 1, push=True)
        self.assertEqual(seen, [{"push": False, "image_counts": (1,)}, {"push": True}, {"push": True}])


class PoolV2SingleImageEndToEndTests(V2StageRunner, PoolV2Fixture):
    """A single-image-only v2 pool: propose -> prepare -> minimal (one probe) -> fresh reload -> local_train
    (N=4, K=2) -> fresh training reload -> evaluate-pool; every stage synchronizes exactly the one bound probe."""
    paired = False
    scale = SCALE_4_2

    def test_single_image_pool_trains_reloads_and_evaluates_with_one_probe(self) -> None:
        from fine_tuning.tests.test_notebook_rl_checkpoint import fixture as tiny_fixture
        from fine_tuning.notebook_rl.local_train import REPAIR_FUNCTIONAL_EXPLORATION

        rule_path = self.base / "rule.json"
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=fake_validation):
            self.assertEqual(local_run.main(["propose-pool", "--build", str(self.root), "--rule-version", "v2",
                "--approval", str(self.request), "--evaluation-count", "2", *v2_args(self.scale),
                "--output", str(rule_path)]), 0)
            self.assertEqual(local_run.main(["prepare-pool", "--build", str(self.root), "--approval", str(self.request),
                "--rule", str(rule_path), *v2_args(self.scale), "--output", str(self.base / "prepared")]), 0)
        rule = json.loads(rule_path.read_text())
        self.assertEqual((rule["probe_only"], rule["probe_coverage"]["image_counts"], len(rule["fixed"])), ([], [1], 4))
        preparation = self.base / "prepared/preparation.json"
        model_path, *_ = tiny_fixture(self.base / "model")
        (model_path / "nf4_base.json").write_text('{"fixture_only":true}')  # cache identity the F0 record hashes
        probe_ids = [rule["minimal"][0]]
        record = self.base / "collector.jsonl"
        config = {"build_root": str(self.root), "minimal_ids": rule["minimal"], "probe_ids": probe_ids,
                  "record": str(record),
                  "replies": {pid: SLOTLESS_SCRIPT for pid in rule["fixed"]} | eval_replies(rule["evaluation"])}
        capture, reload, trained, restored, evaluated = (self.base / name for name in (
            "minimal", "reload", "train", "train-reload", "evaluate"))
        self.run_stage("minimal", "local_run", ["minimal", "--preparation", str(preparation), "--model-path", str(model_path),
            "--standalone-base", "--cache-check", str(self.base / "authored-cache-check"), "--output", str(capture),
            "--repair-notebook", "--context-limit", "24576", *v2_args(self.scale)], config)
        result = json.loads((capture / "result.json").read_text())
        self.assertEqual((result["status"], result["attempted_ids"]), ("UPDATED_SAVED_PENDING_FRESH_RELOAD", rule["minimal"]))
        sync_ids = ["local-sync:" + probe_ids[0]]
        self.assertEqual([row["request_id"] for row in json.loads((capture / "updated_sync.json").read_text())], sync_ids)
        self.assertEqual(len(json.loads((capture / "updated_forward_probes.json").read_text())["inputs"]), 1)
        self.run_stage("reload", "local_run", ["reload", "--capture", str(capture), "--output", str(reload)], config)
        self.assertEqual(json.loads((reload / "result.json").read_text())["status"], "FRESH_PROCESS_RELOAD_VERIFIED")
        self.run_stage("train", "local_train", ["train", "--preparation", str(preparation), "--model-path", str(model_path),
            "--cache-check", str(self.base / "authored-cache-check"), "--minimal", str(capture),
            "--minimal-reload", str(reload), "--output", str(trained), "--arm", "notebook",
            "--admission", REPAIR_FUNCTIONAL_EXPLORATION, *v2_args()], config)
        result = json.loads((trained / "result.json").read_text())
        self.assertEqual((result["status"], result["optimizer_updates"], result["attempted_ids"]),
                         ("LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD", 2, rule["fixed"]))
        self.assertEqual(json.loads((trained / "launch.json").read_text())["execution_contract"]["pair_ids"],
                         [rule["fixed"][:2], rule["fixed"][2:]])
        for name in ("initial_sync.json", "sync-00.json", "sync-01.json"):
            self.assertEqual([row["request_id"] for row in json.loads((trained / name).read_text())], sync_ids, name)
        self.run_stage("train-reload", "local_train", ["reload", "--capture", str(trained), "--output", str(restored)], config)
        reloaded = json.loads((restored / "result.json").read_text())
        self.assertEqual((reloaded["status"], reloaded["step"], reloaded["repair_functional_verified"]),
                         ("LOCAL_TEN_FRESH_PROCESS_RELOAD_VERIFIED", 2, True))
        self.run_stage("evaluate-pool", "local_eval", ["evaluate-pool", "--capture", str(trained),
            "--preparation", str(preparation), "--reload-result", str(restored / "result.json"),
            "--minimal-capture", str(capture), "--minimal-reload", str(reload), "--condition", "FT", "--arm", "notebook",
            "--output", str(evaluated)], config)
        evaluation = json.loads((evaluated / "result.json").read_text())
        self.assertEqual((evaluation["status"], evaluation["recovery"]["problem_ids"]),
                         (local_pool_eval.COMPLETE, rule["evaluation"]))
        self.assertEqual([row["request_id"] for row in json.loads(
            (evaluated / "synchronization-forward-only.json").read_text())], sync_ids)
        rows = [json.loads(line) for line in record.read_text().splitlines()]
        self.assertEqual({pid for row in rows if row["kind"] == "rollout" and row["stage"] in ("minimal", "train")
                          for pid in row["ids"]}, set(rule["fixed"]))
        self.assertEqual({tuple(row["ids"]) for row in rows if row["kind"] == "probes"}, {tuple(probe_ids)})
        self.assertEqual({row["stage"] for row in rows if row["kind"] == "probes"},
                         {"minimal", "reload", "train", "evaluate-pool"})
        self.assertEqual({tuple(row["ids"]) for row in rows if row["kind"] == "sync"}, {(1,)})


if __name__ == "__main__":
    unittest.main()
