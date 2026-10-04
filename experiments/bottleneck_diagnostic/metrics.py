"""Paired diagnostic transitions; repeated seeds are not independent questions."""

from collections import Counter, defaultdict
from random import Random

CLUSTERS = {
    "A_SYMBOLIC_VISUAL": {"Music", "Electronics", "Materials"},
    "B_DIAGRAM_SPATIAL": {"Architecture_and_Engineering", "Mechanical_Engineering", "Geography"},
    "C_SCIENTIFIC_VISUAL_DIAGNOSIS": {"Diagnostics_and_Laboratory_Medicine", "Biology", "Agriculture", "Chemistry"},
    "D_MULTISTEP_QUANTITATIVE": {"Accounting", "Math", "Computer_Science", "Energy_and_Power"},
}
PAIRS = [("I", "O0"), ("O0", "O1"), ("I", "M"), ("M", "MI")]


def cluster(subject):
    return next((name for name, members in CLUSTERS.items() if subject in members), "OTHER_CONTROL")


def transition(rows, samples, before, after, resamples=2000):
    sample_map = {r["sample_id"]: r for r in samples}
    indexed = {(r["sample_id"], r["mode"], r["repetition"]): r for r in rows}
    pairs = []
    for (sid, mode, rep), row in indexed.items():
        other = indexed.get((sid, after, rep))
        if mode != before or not other:
            continue
        # Null means not executed/N/A; never count it as a model error.
        a, b = row.get("diagnostic_score"), other.get("diagnostic_score")
        if type(a) is bool and type(b) is bool:
            pairs.append((sid, a, b))
    counts = Counter((a, b) for _, a, b in pairs)
    base_wrong = counts[False, False] + counts[False, True]
    base_right = counts[True, False] + counts[True, True]
    by_item = defaultdict(list)
    for sid, a, b in pairs:
        by_item[sid].append(int(b) - int(a))
    by_family = defaultdict(list)
    for sid, deltas in by_item.items():
        by_family[sample_map[sid]["family_id"]].append(sum(deltas) / len(deltas))
    values = list(by_family.values())
    interval = None
    if len(values) >= 2:
        rng = Random(3407)
        estimates = []
        for _ in range(resamples):
            sampled = [rng.choice(values) for _ in values]
            flat = [x for group in sampled for x in group]
            estimates.append(sum(flat) / len(flat))
        estimates.sort()
        interval = [estimates[int(.025 * resamples)], estimates[min(resamples - 1, int(.975 * resamples))]]
    return {"before": before, "after": after, "paired_cells": len(pairs),
            "paired_questions": len(by_item), "shared_image_text_families": len(values),
            "both_correct": counts[True, True], "both_wrong": counts[False, False],
            "recovered": counts[False, True], "regressed": counts[True, False],
            "recovery_rate": counts[False, True] / base_wrong if base_wrong else None,
            "regression_rate": counts[True, False] / base_right if base_right else None,
            "question_mean_delta": sum(sum(x) / len(x) for x in by_item.values()) / len(by_item) if by_item else None,
            "family_bootstrap_95_interval": interval,
            "denominator_policy": "Only completed applicable matched cells; item means aggregate repeat seeds"}


def summarize(scores, samples, responses):
    sample_map = {s["sample_id"]: s for s in samples}
    # Runtime ledger can contain earlier error attempts; only latest cells count.
    latest = {(r["sample_id"], r["mode"], r["repetition"]): r for r in responses}
    valid_scores = {(r["sample_id"], r["mode"], r["repetition"]): r for r in scores}
    scored = list(valid_scores.values())
    groups = {"overall_selected": set(sample_map)}
    for sid, sample in sample_map.items():
        groups.setdefault("subject/" + sample["subject"], set()).add(sid)
        groups.setdefault("cluster/" + cluster(sample["subject"]), set()).add(sid)
        for tag in sample["skill_tags"]:
            groups.setdefault("image_type/" + tag, set()).add(sid)
    result = {}
    for name, ids in sorted(groups.items()):
        subset = [r for r in scored if r["sample_id"] in ids]
        modes = {}
        for mode in sorted({r["mode"] for r in subset}):
            cells = [r for r in subset if r["mode"] == mode]
            applicable = [r for r in cells if type(r.get("diagnostic_score")) is bool]
            modes[mode] = {"requested_cells": len(cells),
                           "statuses": dict(Counter(r["runtime_status"] for r in cells)),
                           "applicable_completed": len(applicable),
                           "correct": sum(r["diagnostic_score"] for r in applicable),
                           "accuracy": sum(r["diagnostic_score"] for r in applicable) / len(applicable) if applicable else None,
                           "no_parse": sum(r.get("diagnostic_parse_status") == "NO_PARSE" for r in cells)}
        result[name] = {"questions": len(ids), "modes": modes,
                        "paired": [transition(subset, samples, a, b) for a, b in PAIRS]}
    audit_flags = Counter(flag for row in scored for flag in row.get("audit_flags", []))
    return {"scope": "Selected diagnostic conditions, NOT MMMU population accuracy or a replacement baseline",
            "cluster_policy": "Predeclared subject-based hypotheses, NOT diagnosed failure causes",
            "groups": result, "audit_flags": dict(audit_flags),
            "runtime_statuses": dict(Counter(r["runtime_status"] for r in latest.values())),
            "actual_inference_cells": sum(r.get("run_kind") == "ACTUAL_INFERENCE" and r["runtime_status"] == "COMPLETED" for r in latest.values()),
            "oracle_coverage": {mode: dict(Counter(r["runtime_status"] for r in latest.values() if r["mode"] == mode)) for mode in ("O0", "O1", "IO0")},
            "cost": {"input_tokens": sum(r.get("input_tokens", 0) for r in responses),
                     "image_tokens": sum(r.get("backend_audit", {}).get("image_tokens", 0) for r in responses),
                     "output_tokens": sum(r.get("output_tokens", 0) for r in responses),
                     "request_seconds": sum(r.get("elapsed_seconds", 0) for r in responses)},
            "module_attribution": "Functional recovery evidence does not uniquely identify an internal module",
            "provenance": "BENCHMARK_DERIVED", "training_allowed": False}


def all_subject_first_pass(selection):
    """A frozen subset of expansion: first two remaining ranked IDs per subject."""
    pilot = set(selection["pilot_ids"])
    extras = []
    for subject in sorted({r["subject"] for r in selection["strata"]}):
        if any(r["subject"] == subject and r["sample_id"] in pilot for r in selection["strata"]):
            continue
        for correct in (True, False):
            candidates = [r for r in selection["strata"] if r["subject"] == subject and r["historical_correct"] is correct]
            extras.append(min(candidates, key=lambda r: r["rank"])["sample_id"])
    if not set(extras) <= set(selection["expansion_ids"]):
        raise ValueError("Coverage selections must be nested in predeclared expansion")
    return extras
