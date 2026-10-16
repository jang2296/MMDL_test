"""Same-state retry trees. Observed failures are data, not labels for SFT.

Only the verifier interface handles private answers. The actor/engine receives
images, question and its own actual prior tokens plus bounded status feedback.
No model-token history is re-tokenized or silently shortened.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

from . import prompts as prompt_builders
from .engines import RolloutRequest, RolloutResult
from .schema import ActorItem, Cause, DiagTarget, Group, GroupKind, ProblemTree, Sample, from_json, to_json
from .verifier import answer_reward

RETRYABLE = frozenset(("INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"))
BLOCKING = frozenset(("UNVERIFIABLE", "CONTEXT_BLOCKED", "SYSTEM_ERROR"))


@dataclass(frozen=True)
class RolloutConfig:
    n: dict[int, int]
    branch: dict[int, int]
    max_attempts: int
    max_new_tokens: dict[int, int]
    context_limit: int
    diag: bool
    n_diag: int
    oracle_counterfactual: bool = False
    self_perception_probe: bool = False
    diag_max_new_tokens: int = 256
    visual_lambda: float = 0.0
    master_seed: int = 3407

    @classmethod
    def from_protocol(cls, protocol: dict, algorithm: str, *, master_seed: int = 3407):
        if algorithm not in protocol["algorithms"]:
            raise ValueError("rollout: unknown algorithm")
        r = protocol["rollout"]
        return cls(r["n"], r["branch"], 1 if algorithm == "ALG-T1" else 3 if algorithm == "ALG-NB3" else 2,
            r["max_new_tokens"], protocol["context_limit"], algorithm in r["diag"]["enabled_in"],
            r["diag"]["n"], r["oracle_counterfactual"], r["self_perception_probe"],
            r["diag"]["max_new_tokens"], protocol["reward"]["visual_lambda"] if algorithm == "ALG-NB+D+V" else 0,
            master_seed)

    def validate(self) -> None:
        if (self.max_attempts not in (1, 2, 3) or type(self.context_limit) is not int or self.context_limit < 1
                or any(type(self.n.get(t)) is not int or self.n[t] < 2 for t in range(1, self.max_attempts+1))
                or any(type(self.max_new_tokens.get(t)) is not int or self.max_new_tokens[t] < 1
                       for t in range(1, self.max_attempts+1))
                or any(type(self.branch.get(t)) is not int or self.branch[t] < 0 for t in (1, 2))
                or self.n_diag < 2 or self.diag_max_new_tokens < 1 or not 0 <= self.visual_lambda <= .1
                or type(self.master_seed) is not int or not 0 <= self.master_seed < 2**31):
            raise ValueError("rollout: invalid explicit budgets")
        if self.oracle_counterfactual or self.self_perception_probe:
            raise ValueError("rollout: optional diagnostic probes require a separate probe runner; never use loss groups")


def seeds(problem_id: str, policy_version: int, turn: int, branch: str, n: int,
          *, master_seed: int = 3407) -> list[int]:
    if (not problem_id or policy_version < 0 or turn not in (1, 2, 3) or n < 1
            or type(master_seed) is not int or not 0 <= master_seed < 2**31):
        raise ValueError("rollout: invalid seed identity")
    return [int.from_bytes(hashlib.sha256(f"{master_seed}|{problem_id}|{policy_version}|{turn}|{branch}|{k}".encode()).digest()[:8],
                           "big") % 2**31 for k in range(n)]


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def select_failures(samples: list[Sample], k: int, key: str) -> list[Sample]:
    if type(k) is not int or k < 0:
        raise ValueError("rollout: nonnegative branch count required")
    candidates = sorted((s for s in samples if s.result is not None and s.result.status in RETRYABLE),
                        key=lambda s: hashlib.sha256(f"{key}:{s.sample_id}".encode()).digest())
    distinct, repeated, seen = [], [], set()
    for sample in candidates:
        assert sample.result is not None
        answer = sample.result.extracted
        if answer not in seen:
            distinct.append(sample)
            seen.add(answer)
        else:
            repeated.append(sample)
    return (distinct + repeated)[:k]


def continuation_ids(sample: Sample, feedback: str, tokenizer: Any) -> list[int]:
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if type(end) is not int or end < 0 or end == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("rollout: tokenizer lacks the Qwen conversation boundary")
    closure = [] if sample.completion_token_ids and sample.completion_token_ids[-1] == end else [end]
    suffix = tokenizer.encode("\n<|im_start|>user\n" + feedback + "<|im_end|>\n<|im_start|>assistant\n",
                              add_special_tokens=False)
    return sample.prompt_token_ids + sample.completion_token_ids + closure + suffix


def _sample_request(request: RolloutRequest, engine: Any, policy_version: int,
                    context_limit: int) -> RolloutResult:
    prepared = engine.prepare(request)
    if len(prepared.token_ids)+request.max_new_tokens > context_limit:
        raise ValueError("CONTEXT_BLOCKED: prompt plus completion reservation")
    if engine.policy_version != policy_version:
        raise RuntimeError("rollout: policy changed during a tree")
    outputs = engine.sample([request])
    if (len(outputs) != 1 or outputs[0].request_id != request.request_id
            or len(outputs[0].completions) != request.n
            or outputs[0].prompt_token_ids != prepared.token_ids
            or request.prompt_token_ids is not None and outputs[0].prompt_token_ids != request.prompt_token_ids
            or engine.policy_version != policy_version):
        raise RuntimeError("rollout: sampler altered the actual state, version or group size")
    return outputs[0]


def build_tree(actor: ActorItem, images: list, engine: Any, verifier: Any, prompts: dict,
               cfg: RolloutConfig, policy_version: int, route_rule: str, *,
               on_group: Callable[[Group], None] | None = None,
               should_stop: Callable[[], bool] | None = None) -> ProblemTree:
    """Solution tree first; DIAG is deferred to a class-balanced batch operation."""
    cfg.validate()
    to_json(actor)
    if len(images) != len(actor.image_paths) or engine.policy_version != policy_version:
        raise ValueError("rollout: image coverage or policy version mismatch")
    groups: list[Group] = []
    causes: dict[str, Cause] = {}
    skipped: list[dict] = []
    by_id: dict[str, Sample] = {}
    candidate = None

    def generate(turn: int, parent: Sample | None, ids: list[int] | None) -> Group | None:
        kinds: dict[int, GroupKind] = {1: "T1", 2: "T2", 3: "T3"}
        kind = kinds[turn]
        branch = "root" if parent is None else parent.sample_id
        if should_stop is not None and should_stop():
            skipped.append({"kind": kind, "parent_sample_id": branch, "status": "BUDGET_STOP"})
            return None
        group_id = f"{actor.problem_id}:v{policy_version}:{kind}:{_hash(branch)[:16]}"
        cap = cfg.max_new_tokens[turn]
        n = cfg.n[turn]
        if ids is not None and len(ids)+cap > cfg.context_limit:
            skipped.append({"kind": kind, "parent_sample_id": branch, "status": "CONTEXT_BLOCKED",
                            "input_tokens": len(ids), "reserved_output": cap})
            return None
        request = RolloutRequest(group_id, ids, None if ids is not None else [prompt_builders.user_message(
            prompt_builders.turn1_text(actor, prompts), len(images))], images, n, cap,
            seeds(actor.problem_id, policy_version, turn, kind+":"+branch, n, master_seed=cfg.master_seed))
        try:
            output = _sample_request(request, engine, policy_version, cfg.context_limit)
        except ValueError as exc:
            if not str(exc).startswith("CONTEXT_BLOCKED:"):
                raise
            skipped.append({"kind": kind, "parent_sample_id": branch, "status": "CONTEXT_BLOCKED"})
            return None
        samples = []
        for i, completion in enumerate(output.completions):
            sample = Sample(group_id+f":{i}", actor.problem_id, group_id, kind, turn,
                parent.sample_id if parent is not None else None, list(output.prompt_token_ids),
                completion.token_ids, completion.logprobs, completion.finish_reason,
                completion.text, request.seeds[i], policy_version)
            sample.result = verifier.grade(actor.problem_id, sample.text, sample.finish_reason, actor.mode)
            if sample.result.status not in BLOCKING:
                sample.reward = answer_reward(sample.result, visual_lambda=cfg.visual_lambda)
            cause = verifier.cause(actor.problem_id, sample.result)
            if cause is not None:
                causes[sample.sample_id] = cause
            to_json(sample)
            samples.append(sample)
            by_id[sample.sample_id] = sample
        bad = sorted({s.result.status for s in samples if s.result is not None and s.result.status in BLOCKING})
        group = Group(group_id, kind, actor.problem_id, _hash(output.prompt_token_ids), samples, not bad,
                      ",".join(bad) if bad else None)
        to_json(group)
        groups.append(group)
        if on_group is not None:
            on_group(group)
        return group

    first = generate(1, None, None)
    frontier = select_failures(first.samples, cfg.branch[1], f"{actor.problem_id}:{policy_version}") if (
        first is not None and first.usable) else []
    for turn in range(2, cfg.max_attempts+1):
        next_failures = []
        for parent in frontier:
            assert parent.result is not None
            feedback = prompt_builders.retry_text(parent.result.status, parent.result.extracted,
                                                  actor.mode, actor.family, prompts)
            ids = continuation_ids(parent, feedback, engine.tokenizer)
            group = generate(turn, parent, ids)
            if group is not None and group.usable:
                next_failures.extend(group.samples)
        frontier = select_failures(next_failures, cfg.branch.get(turn, 0),
                                   f"{actor.problem_id}:{policy_version}:{turn}")
    if cfg.diag:
        candidates = sorted((s for g in groups if g.usable for s in g.samples
                             if s.turn >= 2 and s.result is not None and s.result.status == "CORRECT"),
                            key=lambda s: _hash(s.sample_id))
        for success in candidates:
            earliest = success
            while earliest.parent_sample_id is not None:
                earliest = by_id[earliest.parent_sample_id]
            target = verifier.diagnosis_target(actor.problem_id, earliest.result)
            if target is not None:
                candidate = {"parent_sample_id": success.sample_id, "target": json.loads(to_json(target))}
                break
    results = [s.result for g in groups for s in g.samples if s.result is not None]
    route, evidence = verifier.route(actor.problem_id, results, route_rule)
    notebook = {"routing": evidence, "skipped_branches": skipped, "model_notes": [],
        "diagnosis_candidate": candidate, "diagnosis_target": None, "diagnosis_batch_closed": False,
        "attempts": [{"sample_id": s.sample_id, "status": s.result.status,
                      "cause": causes.get(s.sample_id), "slots": json.loads(to_json(s.result.slots))
                      if s.result.slots else None} for g in groups for s in g.samples if s.result is not None],
        "verified_internal_module_responsibility": False}
    tree = ProblemTree(actor.problem_id, policy_version, groups, route, causes, notebook)
    to_json(tree)
    return tree


def append_diagnoses(trees: list[ProblemTree], images: dict[str, list], engine: Any,
                     verifier: Any, prompts: dict, cfg: RolloutConfig, *,
                     on_group: Callable[[Group], None] | None = None,
                     should_stop: Callable[[], bool] | None = None) -> dict:
    """Choose <=50% per verified class BEFORE generating; never drop after seeing rewards."""
    cfg.validate()
    if (not cfg.diag or not trees or len({t.problem_id for t in trees}) != len(trees)
            or set(images) != {t.problem_id for t in trees}
            or any(t.policy_version != engine.policy_version for t in trees)
            or any(t.notebook.get("diagnosis_batch_closed") for t in trees)):
        raise ValueError("rollout: one fresh same-policy diagnosis batch required")
    eligible: dict[str, list] = {c: [] for c in ("VISUAL", "NON_VISUAL", "FINALIZATION")}
    for tree in trees:
        to_json(tree)
        candidate = tree.notebook.get("diagnosis_candidate")
        if candidate is None:
            continue
        target = from_json(DiagTarget, json.dumps(candidate["target"]))
        parents = [s for g in tree.groups if g.usable and g.kind != "DIAG" for s in g.samples
                   if s.sample_id == candidate["parent_sample_id"] and s.turn >= 2
                   and s.result is not None and s.result.status == "CORRECT"]
        if len(parents) != 1:
            raise ValueError("rollout: diagnosis needs an actual recovered path")
        parent = parents[0]
        ids = continuation_ids(parent, prompt_builders.diagnose_text(parent.turn, prompts), engine.tokenizer)
        gid = f"{tree.problem_id}:v{tree.policy_version}:DIAG:{_hash(parent.sample_id)[:16]}"
        request = RolloutRequest(gid, ids, None, images[tree.problem_id], cfg.n_diag,
            cfg.diag_max_new_tokens, seeds(tree.problem_id, tree.policy_version, parent.turn,
                                          "DIAG:"+parent.sample_id, cfg.n_diag, master_seed=cfg.master_seed))
        if len(ids)+request.max_new_tokens > cfg.context_limit:
            tree.notebook["skipped_branches"].append({"kind": "DIAG", "status": "CONTEXT_BLOCKED"})
            continue
        eligible[target.cause3].append((tree, target, parent, request))
    for entries in eligible.values():
        entries.sort(key=lambda entry: _hash(entry[0].problem_id))
    excluded = []
    while sum(map(len, eligible.values())):
        largest = max(eligible, key=lambda key: (len(eligible[key]), key))
        if len(eligible[largest])*2 <= sum(map(len, eligible.values())):
            break
        excluded.append(eligible[largest].pop())
    for tree, _, _, _ in excluded:
        tree.notebook["skipped_branches"].append({"kind": "DIAG", "status": "CLASS_CAP"})
    admitted = [entry for entries in eligible.values() for entry in entries]
    report: dict[str, Any] = {"class_counts": {key: len(values) for key, values in eligible.items()},
              "excluded_by_cap": len(excluded), "generated_groups": 0}
    # Mark before generation so an interrupted batch cannot silently generate duplicates.
    for tree in trees:
        tree.notebook["diagnosis_batch_closed"] = True
    for tree, target, parent, request in sorted(admitted, key=lambda entry: _hash(entry[0].problem_id)):
        if should_stop is not None and should_stop():
            report["budget_stopped"] = True
            break
        output = _sample_request(request, engine, tree.policy_version, cfg.context_limit)
        samples = []
        for i, completion in enumerate(output.completions):
            reward = verifier.grade_diagnosis(tree.problem_id, completion.text, target) if (
                completion.finish_reason == "stop") else 0.0
            sample = Sample(request.request_id+f":{i}", tree.problem_id, request.request_id, "DIAG",
                parent.turn, parent.sample_id, output.prompt_token_ids, completion.token_ids,
                completion.logprobs, completion.finish_reason, completion.text, request.seeds[i],
                tree.policy_version, None, reward, reward)
            samples.append(sample)
            tree.notebook["model_notes"].append({"sample_id": sample.sample_id, "text": sample.text,
                "reward": reward, "lesson_semantics_verified": False})
        tree.groups.append(Group(request.request_id, "DIAG", tree.problem_id,
                                 _hash(output.prompt_token_ids), samples, True, None))
        if on_group is not None:
            on_group(tree.groups[-1])
        tree.notebook["diagnosis_target"] = json.loads(to_json(target))
        report["generated_groups"] += 1
        to_json(tree)
    return report
