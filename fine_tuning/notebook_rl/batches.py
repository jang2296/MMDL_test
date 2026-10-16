"""Bounded family-balanced rollout collection; this is not a data approval gate."""
from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image

from .rollout import RolloutConfig, append_diagnoses, build_tree
from .schema import ActorItem, Group, ProblemTree, to_json
from .updates import has_learning_signal


def family_order(actors: list[ActorItem], seed: int) -> list[str]:
    if type(seed) is not int or seed < 0 or len({a.problem_id for a in actors}) != len(actors):
        raise ValueError("batch: nonnegative seed and unique actor IDs required")
    def rank(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).digest()
    families = defaultdict(list)
    for actor in actors:
        to_json(actor)
        families[actor.family].append(actor.problem_id)
    queues = {key: deque(sorted(values, key=rank)) for key, values in families.items()}
    result = []
    while any(queues.values()):
        for key in sorted(queues, key=rank):
            if queues[key]:
                result.append(queues[key].popleft())
    return result


@dataclass
class CollectedBatch:
    trees: list[ProblemTree]
    images: dict[str, list]
    cursor: int
    attempted: list[str]
    discarded: list[dict]
    diagnosis: dict | None
    seconds: float
    status: str


def collect_batch(actors: dict[str, ActorItem], structure_hashes: dict[str, str], order: list[str],
                  cursor: int, loader: Callable[[ActorItem], list], engine: Any, verifier: Any,
                  prompts: dict, protocol: dict, algorithm: str, *, should_stop: Callable[[], bool],
                  seed: int = 3407, on_tree: Callable[[ProblemTree], None] | None = None,
                  on_group: Callable[[Group], None] | None = None) -> CollectedBatch:
    """Consume at most 3B IDs once; record every skipped ID rather than retry forever."""
    if (set(actors) != set(order) or len(order) != len(set(order)) or set(structure_hashes) != set(actors)
            or any(not re.fullmatch(r"(?:[0-9a-f]{16}|[0-9a-f]{64})", value) for value in structure_hashes.values())
            or type(cursor) is not int or not 0 <= cursor <= len(order)):
        raise ValueError("batch: frozen exact actor/order/structure coverage required")
    target, maximum = protocol["batch"]["problems_per_step"], protocol["batch"]["max_attempted_problems_per_step"]
    if type(target) is not int or target < 1 or maximum != 3*target:
        raise ValueError("batch: bounded 3B attempt policy required")
    cfg = RolloutConfig.from_protocol(protocol, algorithm, master_seed=seed)
    trees: list[ProblemTree] = []
    images: dict[str, list] = {}
    attempted: list[str] = []
    discarded: list[dict] = []
    used_structures: set[str] = set()
    started, version = time.monotonic(), engine.policy_version
    status = "COLLECTED"
    while len(trees) < target and len(attempted) < maximum and cursor < len(order):
        if should_stop():
            status = "BUDGET_STOP"
            break
        pid = order[cursor]
        cursor += 1
        attempted.append(pid)
        if structure_hashes[pid] in used_structures:
            discarded.append({"problem_id": pid, "reason": "STRUCTURE_ALREADY_IN_STEP"})
            continue
        actor = actors[pid]
        loaded = []
        try:
            loaded = loader(actor)
            if algorithm == "CTL-BLIND":
                gray = [Image.new("RGB", image.size, (128, 128, 128)) for image in loaded]
                for image in loaded:
                    image.close()
                loaded = gray
            tree = build_tree(actor, loaded, engine, verifier, prompts, cfg, version, protocol["route_rule"],
                              on_group=on_group, should_stop=should_stop)
            tree.notebook["input_condition"] = "BLIND_SAMPLED" if algorithm == "CTL-BLIND" else "ORIGINAL"
            tree.notebook["structure_hash"] = structure_hashes[pid]
            if on_tree is not None:
                on_tree(tree)
        except BaseException:
            for image in loaded+[im for values in images.values() for im in values]:
                image.close()
            raise
        stopped = any(r["status"] == "BUDGET_STOP" for r in tree.notebook.get("skipped_branches", []))
        # Closed full raw trees, including discarded ones, are the caller's responsibility
        # to persist via its collector; this return value retains them in the discard ledger.
        if not has_learning_signal(tree):
            discarded.append({"problem_id": pid, "reason": "NO_SOLUTION_ADVANTAGE", "tree": to_json(tree)})
            for image in loaded:
                image.close()
            if stopped:
                status = "BUDGET_STOP"
                break
            continue
        trees.append(tree)
        images[pid] = loaded
        used_structures.add(structure_hashes[pid])
        if stopped:
            status = "BUDGET_STOP"
            break
    diagnosis = None
    try:
        if trees and cfg.diag and status != "BUDGET_STOP" and not should_stop():
            diagnosis = append_diagnoses(trees, images, engine, verifier, prompts, cfg,
                                        on_group=on_group, should_stop=should_stop)
            if diagnosis.get("budget_stopped"):
                status = "BUDGET_STOP"
        elif trees and cfg.diag:
            status = "BUDGET_STOP"
        if engine.policy_version != version:
            raise RuntimeError("batch: policy changed during rollout collection")
    except BaseException:
        for image in [im for values in images.values() for im in values]:
            image.close()
        raise
    if status != "BUDGET_STOP" and len(trees) < target:
        status = "EXHAUSTED" if cursor == len(order) else "ATTEMPT_LIMIT"
    return CollectedBatch(trees, images, cursor, attempted, discarded, diagnosis,
                          time.monotonic()-started, status)
