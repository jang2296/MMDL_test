"""Frozen rule-C training guards and funding decisions; no cloud calls or evaluation launch."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

RULES = ("ACC_DROP_SELF", "TRUNC_UP", "NOFINAL_DRIFT", "COLLAPSE_LEN", "COLLAPSE_SHORT",
         "TAIL_RATE", "IS_DRIFT", "ORDER_MISMATCH")
GUARD_VERSION = "error-note-v16b-guard-c-v1"
EVAL_SECONDS = 23400


class GuardV16B:
    """Append one completed RL block; serializable state belongs in the campaign checkpoint."""

    def __init__(self, order: list[list[str]], state: dict | None = None):
        ids = [item for block in order for item in block]
        if (not order or len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids)
                or any(not 1 <= len(block) <= 16 for block in order)):
            raise ValueError("v16b guard: unique planned IDs in blocks of at most 16 required")
        self.order = copy.deepcopy(order)
        binding = hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()
        self.state = copy.deepcopy(state) if state is not None else {
            "version": GUARD_VERSION, "order_sha256": binding, "history": [], "reasons": [],
            "collapse_windows": 0, "tail_streak": 0, "hold_streak": 0}
        if (self.state.get("version") != GUARD_VERSION or self.state.get("order_sha256") != binding
                or len(self.state["history"]) > len(order)):
            raise ValueError("v16b guard: state/order binding mismatch")

    def update(self, block: int, items: list[dict], *, tail_rate: float | None = None,
               is_hold: bool = False) -> dict:
        if type(block) is not int or block != len(self.state["history"]) + 1 or block > len(self.order):
            raise ValueError("v16b guard: append completed blocks in order, starting at 1")
        if not items:
            raise ValueError("v16b guard: empty completed block")
        if type(is_hold) is not bool or (tail_rate is not None and (
                isinstance(tail_rate, bool) or not isinstance(tail_rate, (int, float))
                or not math.isfinite(tail_rate) or not 0 <= tail_rate <= 1)):
            raise ValueError("v16b guard: finite tail proportion and boolean HOLD required")
        for item in items:
            if (any(type(item.get(k)) is not bool for k in ("correct", "truncated", "no_final"))
                    or type(item.get("tokens")) is not int or item["tokens"] < 0
                    or (item.get("f0_tokens") is not None and (
                        type(item["f0_tokens"]) is not int or item["f0_tokens"] < 0))):
                raise ValueError("v16b guard: exact T1 booleans and nonnegative lengths required")
        reasons = []
        if [i["id"] for i in items] != self.order[block - 1]:
            reasons.append("ORDER_MISMATCH")
        row = {"n": len(items), "correct": sum(i["correct"] for i in items),
               "truncated": sum(i["truncated"] for i in items), "no_final": sum(i["no_final"] for i in items),
               "short": sum(i["tokens"] < 32 for i in items), "tail_rate": tail_rate,
               "length_pairs": [[i["tokens"], i["f0_tokens"]] for i in items if i.get("f0_tokens") is not None]}
        history = self.state["history"]
        history.append(row)
        if block >= 2 and all(r["short"] >= 12 for r in history[-2:]):
            reasons.append("COLLAPSE_SHORT")
        self.state["hold_streak"] = self.state["hold_streak"] + 1 if is_hold else 0
        if self.state["hold_streak"] >= 2:
            reasons.append("IS_DRIFT")
        if block > 4 and tail_rate is not None and all(r["tail_rate"] is not None for r in history[:4]):
            baseline = statistics.mean(r["tail_rate"] for r in history[:4])
            self.state["tail_streak"] = self.state["tail_streak"] + 1 if tail_rate > 2 * baseline else 0
        else:
            self.state["tail_streak"] = 0
        if self.state["tail_streak"] >= 2:
            reasons.append("TAIL_RATE")
        if block % 6 == 0:
            pairs = [pair for r in history[-6:] for pair in r["length_pairs"]]
            collapsed = bool(pairs) and statistics.median(p[0] for p in pairs) < 0.5 * statistics.median(p[1] for p in pairs)
            self.state["collapse_windows"] = self.state["collapse_windows"] + 1 if collapsed else 0
            if block >= 12 and self.state["collapse_windows"] >= 2:
                reasons.append("COLLAPSE_LEN")
        thresholds = {}
        if block >= 36:
            reference, recent = history[:24], history[-12:]
            ref_n, recent_n = sum(r["n"] for r in reference), sum(r["n"] for r in recent)
            ref_acc = sum(r["correct"] for r in reference) / ref_n
            thresholds["accuracy"] = ref_acc - 0.15
            if sum(r["correct"] for r in recent) / recent_n < thresholds["accuracy"]:
                reasons.append("ACC_DROP_SELF")
            for field, rule in (("truncated", "TRUNC_UP"), ("no_final", "NOFINAL_DRIFT")):
                ref = sum(r[field] for r in reference) * 12 / 24
                thresholds[field] = {"ref": ref, "minimum": max(2 * ref, ref + 4 * math.sqrt(ref))}
                if sum(r[field] for r in recent) >= thresholds[field]["minimum"]:
                    reasons.append(rule)
        self.state["reasons"] = sorted(set(self.state["reasons"] + reasons))
        return {"version": GUARD_VERSION, "block": block, "status": "STOP_REQUESTED" if self.state["reasons"] else "CONTINUE",
                "reasons": self.state["reasons"][:], "thresholds": thresholds, "auto_evaluate": False}


def funding_decision(balance_usd: float, hourly_usd: float, now: float,
                     requested_at: float | None = None) -> dict:
    """Caller polls actual balance every ten minutes; recharge never authorizes a new run."""
    values = [balance_usd, hourly_usd, now] + ([] if requested_at is None else [requested_at])
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values)
            or hourly_usd <= 0 or (requested_at is not None and requested_at > now)):
        raise ValueError("v16b funding: finite actual balance, rate and monotonic timestamps required")
    protected = 5.65 * hourly_usd + 1.5
    low = balance_usd - protected < hourly_usd
    started = requested_at if requested_at is not None else now
    status = "CONTINUE" if not low else "STOP_REQUESTED" if now - started >= 1800 else "RECHARGE_REQUESTED"
    return {"status": status, "protected_usd": protected, "next_hour_usd": hourly_usd,
            "requested_at": started if low else None, "auto_evaluate": False}


def falsealarm_receipt(manifest: Path, *, trials: int = 2000, seed: int = 3407) -> dict:
    """Source-stratified F0 bootstrap, using only exact prior bindings in the frozen 3,000 pool."""
    if type(trials) is not int or trials < 1:
        raise ValueError("v16b guard: positive bootstrap trials required")
    raw = Path(manifest).read_bytes()
    items = json.loads(raw)["items"]
    if len(items) != 3000 or len({i["problem_id"] for i in items}) != 3000:
        raise ValueError("v16b guard: frozen 3,000 unique parents required")
    sources = defaultdict(list)
    for item in items:
        prior = item.get("f0_prior")
        if prior is not None:
            if prior.get("binding") != "EXACT_NATIVE_ID_IMAGE_QUESTION_CHOICES_GOLD":
                raise ValueError("v16b guard: exact F0 prior binding required")
            sources[item["source"]].append(prior)
    n = sum(map(len, sources.values()))
    if not n:
        raise ValueError("v16b guard: no F0 priors")
    rng = random.Random(seed)
    hits = dict.fromkeys(RULES, 0)
    any_hits = 0
    order = [[str(i) for i in range(j, min(j + 16, 3000))] for j in range(0, 3000, 16)]
    for _ in range(trials):
        keyed = []
        for source in sorted(sources):
            rows = sources[source]
            picks = [rng.choice(rows) for _ in range(max(1, round(3000 * len(rows) / n)))]
            phase = rng.random()
            keyed += [((i + phase) / len(picks), rng.random(), prior) for i, prior in enumerate(picks)]
        keyed.sort(key=lambda r: r[:2])
        sampled = [r[2] for r in keyed]
        while len(sampled) < 3000:
            sampled.append(rng.choice(sampled))
        guard = GuardV16B(order)
        for block, ids in enumerate(order, 1):
            batch = []
            for item_id in ids:
                prior = sampled[int(item_id)]
                batch.append({"id": item_id, "correct": prior["status"] == "CORRECT",
                              "tokens": prior["completion_tokens"], "f0_tokens": prior["completion_tokens"],
                              "truncated": prior["finish_reason"] == "length",
                              "no_final": prior["status"] == "INVALID_NO_FINAL"})
            guard.update(block, batch)
        for rule in guard.state["reasons"]:
            hits[rule] += 1
        any_hits += bool(guard.state["reasons"])
    rates = {rule: count / trials for rule, count in hits.items()}
    # F0 priors contain no sampler tail or IS statistics: their false-alarm rate cannot be certified here.
    unmeasured = ["TAIL_RATE", "IS_DRIFT"]
    measured_pass = max([any_hits / trials, *rates.values()]) <= 0.02
    return {"version": "v16b-guard-falsealarm-v2", "guard_version": GUARD_VERSION,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(), "pool_items": 3000, "prior_items": n,
            "trials": trials, "seed": seed, "method": "source-stratified-interleaved-bootstrap-v1",
            "rule_hits": {rule: None if rule in unmeasured else value for rule, value in hits.items()},
            "rule_rates": {rule: None if rule in unmeasured else value for rule, value in rates.items()},
            "overall_rate": any_hits / trials,
            "unmeasured_rules": unmeasured, "false_alarm_max": 0.02,
            "measured_status": "PASS" if measured_pass else "FAIL", "measured_pass": measured_pass,
            "status": "INCOMPLETE" if measured_pass else "FAIL", "reason": "TAIL_RATE and IS_DRIFT unavailable in frozen f0_prior; no threshold tuning",
            "auto_evaluate": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=2000)
    args = parser.parse_args(argv)
    receipt = falsealarm_receipt(args.manifest, trials=args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0 if receipt["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
