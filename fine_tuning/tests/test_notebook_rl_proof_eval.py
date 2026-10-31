"""CPU checks for the preregistered MMMU-val proof path (synthetic runs, tiny-model export)."""
from __future__ import annotations

import copy
import inspect
import json
import math
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

from fine_tuning.notebook_rl import proof_eval as pe
from mmdl.evaluation.parsers import score_with_parser
from mmdl.runtime.artifacts import digest, sha256_file
from mmdl.runtime.contracts import SUBJECTS, sample_seed

IDS = sorted(f"validation_{s}_{k}" for s in SUBJECTS for k in range(1, 31))
SUBSET = IDS[:342]  # the synthetic B2 below answers exactly these wrong
FAST = [f"mmmu-val-v8-fast8k-seed{s}-v1" for s in (3407, 3408, 3409)]
C16_FILES = [pe.HERE / f"eval_configs/mmmu_val_v8_c16_seed{s}.yaml" for s in (3407, 3408, 3409)]


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in flatten(item, f"{prefix}{key}.").items()}
    return {prefix[:-1]: value}


def write_run(root, name, cfg, model_sha, outcome, *, code="c" * 64, protocol=None, seed=None, registration=None,
              ids=None, mode="partial"):
    """A run directory written by the evaluator's own RunWriter + finalize (as engine.run does).

    outcome(index) -> True / False / None (None = empty, length-cut response); ids default: the subset.
    registration: None = what run-eval writes (fast8k IDs only); True/False forces presence/absence.
    """
    from mmdl.evaluation.writer import RunWriter, finalize

    ids = list(SUBSET if ids is None else ids)
    run = root / name
    seed = cfg["generation"]["seed"] if seed is None else seed
    identity = {"protocol_sha256": protocol or digest(cfg), "model_sha256": model_sha, "code_sha256": code,
                "ids_sha256": digest(ids), "mode": mode, "git_commit": "0" * 40}
    with RunWriter(run, identity, ids) as writer:
        for index, sid in enumerate(ids):
            result = outcome(index)
            raw, finish = ("", "length") if result is None else ("Answer: A", "stop")
            row = {"id": sid, "subject": pe._subject(sid), "question_type": "multiple-choice",
                   "options": ["p", "q", "r", "s"], "answer": "A" if result else "B", "question": "Q?",
                   "raw_response": raw, "finish_reason": finish, "seed": sample_seed(seed, sid),
                   "generated_tokens": 1, "generation_seconds": 0.0}
            writer.save(row | score_with_parser(raw, "multiple-choice", row["options"], row["answer"], pe.PARSER,
                                                finish, question="Q?"))
        finalize(run, ids, mode)
    (run / "resolved_eval_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if registration is None:
        registration = cfg["protocol_id"] in pe.REGISTERED
    if registration:
        pid = cfg["protocol_id"] if cfg["protocol_id"] in pe.REGISTERED else FAST[0]
        (run / pe.REGISTRATION_FILE).write_text(json.dumps(pe.variant_registration(pid)))
    return run


def write_subset(path, b2, ids=None, **changes):
    value = {"version": pe.SUBSET_VERSION, "purpose": "fixture", "source": str(b2), "source_sha256": sha256_file(b2),
             "count": len(SUBSET if ids is None else ids), "sample_ids": list(SUBSET if ids is None else ids)}
    path.write_text(json.dumps(value | changes))
    return path


def variant_cfg(vid):
    return yaml.safe_load(pe.VARIANTS[vid][0].read_text())


class ProofAnalysisTests(unittest.TestCase):
    # Over the 342 subset items: F0 even items right (171), every 30th odd item empty + length-cut (11).
    # FT seeds: 206 / 171 / 136 right.
    f0_outcome = staticmethod(lambda i: None if i % 30 == 29 else i % 2 == 0)
    ft_outcomes = {3407: lambda i: i % 2 == 0 or i % 10 == 1, 3408: lambda i: i % 2 == 0,
                   3409: lambda i: i % 2 == 0 and i % 10 != 0}

    @classmethod
    def setUpClass(cls):
        """The six runs go through the evaluator's RunWriter (one fsync per record) once; setUp copies them."""
        tmp = tempfile.TemporaryDirectory(prefix="proof-eval-class-")
        cls.addClassCleanup(tmp.cleanup)
        root = cls.template = Path(tmp.name)
        b2 = root / "b2.jsonl"
        b2.write_text("".join(json.dumps({"id": sid, "correct": i >= 342}) + "\n" for i, sid in enumerate(IDS)))
        pe.preregister(root / "prereg.json", write_subset(root / "subset.json", b2), b2_predictions=b2)
        (root / "model_ref.json").write_text(json.dumps({
            "files": [{"path": "model.safetensors", "bytes": 1, "sha256": "f" * 64}],
            "export": {"unmerged_inference_bit_equivalent": False}}))
        f0 = json.loads((root / "prereg.json").read_text())["arms"]["F0"]["model_sha256"]
        ft = digest(json.loads((root / "model_ref.json").read_text())["files"])
        for v in FAST:
            write_run(root, f"f0-{v}", variant_cfg(v), f0, cls.f0_outcome)
        for v, s in zip(FAST, (3407, 3408, 3409)):
            write_run(root, f"ft-{v}", variant_cfg(v), ft, cls.ft_outcomes[s])

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="proof-eval-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "case"
        shutil.copytree(self.template, self.root)
        self.b2, self.subset, self.prereg = self.root / "b2.jsonl", self.root / "subset.json", self.root / "prereg.json"
        self.prereg_sha = sha256_file(self.prereg)
        self.f0 = json.loads(self.prereg.read_text())["arms"]["F0"]["model_sha256"]
        self.ft_ref = self.root / "model_ref.json"
        self.ft = digest(json.loads(self.ft_ref.read_text())["files"])
        self.runs = [self.root / f"f0-{v}" for v in FAST] + [self.root / f"ft-{v}" for v in FAST]
        self.ledger = self.root / "ledger.jsonl"
        for run in self.runs:
            self.start(run)

    def start(self, run, ledger=None):
        """ledger-start the run directory under the (arm, variant) its content declares."""
        loaded = pe.load_run(run, json.loads(self.prereg.read_text()))
        return pe.ledger_start(self.prereg, loaded["arm"], loaded["variant"], Path(run).name, ledger or self.ledger)

    def analyze(self, runs, ref=None):
        return pe.analyze(self.prereg, runs, ref or self.ft_ref, self.ledger)

    def test_variants_differ_from_v8_only_in_seed_slots_length_pixels_and_context(self):
        from fine_tuning.notebook_rl import local_profile

        base = flatten(variant_cfg(pe.COURSE))
        self.assertEqual(pe.VARIANTS[pe.COURSE][0], pe.ROOT / "configs/eval/mmmu_val_v8.yaml")
        self.assertEqual([v for v, row in pe.VARIANTS.items() if row[3] == "statistical"], FAST)
        self.assertEqual(set(pe.REGISTERED), set(FAST))
        fast = {"execution.batch_size": 32, "generation.max_new_tokens": 8192, "image.min_pixels": 262144,
                "image.max_pixels": 1003520, "execution.max_model_len": 24576}
        for vid, seed in zip(FAST, (3407, 3408, 3409)):
            cfg = flatten(variant_cfg(vid))
            self.assertEqual(set(cfg), set(base))
            changed = {k for k in base if cfg[k] != base[k]}
            expected = {"protocol_id", *fast} | ({"generation.seed"} if seed != 3407 else set())
            self.assertEqual(changed, expected, vid)
            self.assertEqual({k: cfg[k] for k in fast}, fast)
            self.assertEqual((cfg["protocol_id"], cfg["generation.seed"]), (vid, seed))
        # the eval image budget is the training one (protocol image of the local/pod training profile)
        image = local_profile.build_protocol(repair_notebook=True, context_limit=24576)["image"]
        self.assertEqual({k: image[k] for k in pe.FAST_PIXELS}, pe.FAST_PIXELS)
        # the c16 files stay on disk but are no longer part of the proof
        self.assertTrue(all(path.is_file() for path in C16_FILES))
        self.assertFalse({path.resolve() for path in C16_FILES} & {row[0].resolve() for row in pe.VARIANTS.values()})

    def test_preregistration_is_frozen_hashed_and_write_once(self):
        prereg, sha = pe.load_preregistration(self.prereg)
        self.assertEqual(sha, self.prereg_sha)
        self.assertEqual(len(prereg["b2_reference"]["wrong_ids"]), 342)
        self.assertEqual(len(prereg["b2_reference"]["right_ids"]), 558)
        self.assertEqual(prereg["runs"], [{"arm": a, "variant": v, "role": "statistical"} for a in ("F0", "FT")
                                          for v in FAST])  # no course run
        self.assertEqual(set(prereg["protocol_variants"]), set(FAST))
        self.assertEqual({prereg["protocol_variants"][v]["concurrency"] for v in FAST}, {32})
        self.assertEqual(prereg["primary_endpoint"]["variants"], FAST)
        self.assertEqual(prereg["primary_endpoint"]["items"], 342)
        self.assertEqual(prereg["sensitivity"]["mcnemar_exact_two_sided"]["variant"], FAST[0])
        self.assertEqual(prereg["course_facing_run"]["status"], "DROPPED")
        self.assertEqual((prereg["mmmu_val_budget"]["checkpoints_per_arm"], prereg["mmmu_val_budget"]["runs"],
                          prereg["mmmu_val_budget"]["items_per_run"]), (1, 6, 342))
        subset = prereg["evaluation_subset"]
        self.assertEqual((subset["ids"], subset["sha256"], subset["count"]), (SUBSET, sha256_file(self.subset), 342))
        self.assertEqual(subset["ids"], prereg["b2_reference"]["wrong_ids"])
        self.assertIn("558", prereg["limitations"][0])
        self.assertIn("LLM-assisted judge", prereg["scorer"]["only_scorer"])
        with self.assertRaises(FileExistsError):
            pe.preregister(self.prereg, self.subset, b2_predictions=self.b2)
        self.prereg.write_text(self.prereg.read_text().replace('"FROZEN"', '"DRAFT"', 1))
        with self.assertRaisesRegex(ValueError, "sidecar"):
            pe.load_preregistration(self.prereg)

    def test_analysis_means_bootstrap_mcnemar_and_splits_on_the_subset(self):
        report = self.analyze([run / "predictions.jsonl" for run in self.runs[:3]] + self.runs[3:])
        p = report["primary"]
        self.assertEqual((p["items"], p["scorer"]), (342, pe.PARSER))
        self.assertEqual(p["correct_per_seed"], {"F0": {"3407": 171, "3408": 171, "3409": 171},
                                                 "FT": {"3407": 206, "3408": 171, "3409": 136}})
        for seed, count in p["correct_per_seed"]["FT"].items():  # per-seed accuracy = recovered count / 342
            self.assertAlmostEqual(p["accuracy_per_seed"]["FT"][seed], count / 342)
        self.assertAlmostEqual(p["F0_accuracy"], 0.5)
        self.assertAlmostEqual(p["FT_accuracy"], 0.5)
        self.assertAlmostEqual(p["FT_minus_F0"], 0.0)  # best-of-seeds would give +35/342
        self.assertLessEqual(p["ci95"][0], 0.0)
        self.assertGreaterEqual(p["ci95"][1], 0.0)
        m = report["sensitivity_mcnemar_seed3407"]
        self.assertEqual((m["f0_right_ft_wrong"], m["f0_wrong_ft_right"]), (0, 35))
        self.assertTrue(math.isclose(m["p_value"], 2 / 2 ** 35))
        self.assertEqual(set(report["secondary"]["F0"]), {"length_cutoff_rate", "missing_final_answer_rate"})
        self.assertAlmostEqual(report["secondary"]["F0"]["length_cutoff_rate"], 11 / 342)
        self.assertAlmostEqual(report["secondary"]["F0"]["missing_final_answer_rate"], 11 / 342)
        self.assertEqual(sum(r["n"] for r in report["per_discipline"].values()), 342)
        self.assertEqual(sum(r["n"] for r in report["per_subject"].values()), 342)
        self.assertEqual(report["course_facing"]["status"], "DROPPED")
        self.assertEqual(report["preregistration_sha256"], self.prereg_sha)
        markdown = pe.render_markdown(report)
        self.assertIn("206 (60.23%)", markdown)
        self.assertIn("Course-facing run (FT, mmmu-val-v8, c2, seed 3407): DROPPED", markdown)
        self.assertIn("Limitation: 342-only", markdown)

    def test_runs_must_be_partial_runs_on_exactly_the_preregistered_subset(self):
        cfg, out = variant_cfg(FAST[0]), self.ft_outcomes[3407]
        cases = {"missing one": dict(ids=SUBSET[:-1]), "one swapped for a B2-right item": dict(ids=SUBSET[:-1] + [IDS[400]]),
                 "full 900-item run": dict(ids=IDS, mode="full")}
        for name, changes in cases.items():
            with self.subTest(name):
                run = write_run(self.root, "ft-" + name.replace(" ", "-"), cfg, self.ft, out, **changes)
                with self.assertRaisesRegex(ValueError, "sample set differs|not an original partial"):
                    self.analyze(self.runs[:3] + [run] + self.runs[4:])
        unfinished = write_run(self.root, "ft-incomplete", cfg, self.ft, out)
        summary = json.loads((unfinished / "summary.json").read_text())
        (unfinished / "summary.json").write_text(json.dumps(summary | {"status": "INCOMPLETE"}))
        with self.assertRaisesRegex(ValueError, "is INCOMPLETE, not PARTIAL"):
            self.analyze(self.runs[:3] + [unfinished] + self.runs[4:])
        manifest = json.loads((self.runs[3] / "run_manifest.json").read_text())
        manifest["identity"]["ids_sha256"] = digest(IDS)
        (self.runs[3] / "run_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "sample set differs"):
            self.analyze(self.runs)

    def test_preregistration_refuses_a_subset_that_is_not_the_b2_wrong_set(self):
        for name, path in {"other ids": write_subset(self.root / "s1.json", self.b2, ids=SUBSET[:-1] + [IDS[400]]),
                           "unsorted": write_subset(self.root / "s2.json", self.b2, ids=SUBSET[::-1]),
                           "other source": write_subset(self.root / "s3.json", self.b2, source_sha256="0" * 64),
                           "wrong count": write_subset(self.root / "s4.json", self.b2, count=341),
                           "other version": write_subset(self.root / "s5.json", self.b2, version="v0")}.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, "B2 V8 wrong IDs"):
                pe.preregister(self.root / f"prereg-{name}.json", path, b2_predictions=self.b2)

    def test_best_of_selection_is_impossible_through_the_api(self):
        self.assertEqual(list(inspect.signature(pe.seed_mean).parameters), ["per_seed"])
        self.assertEqual(list(inspect.signature(pe.analyze).parameters), ["prereg_path", "run_paths", "ft_model_ref", "ledger"])
        self.assertEqual(pe.seed_mean([{"a": True}, {"a": False}, {"a": False}]), {"a": 1 / 3})
        best = write_run(self.root, "ft-again", variant_cfg(FAST[2]), self.ft, lambda i: i % 2 == 0 or i % 10 == 1)
        with self.assertRaisesRegex(ValueError, "duplicate run"):
            self.analyze(self.runs + [best])
        extra_cfg = variant_cfg(FAST[0])
        extra_cfg["generation"]["seed"] = 3410
        extra = write_run(self.root, "ft-3410", extra_cfg, self.ft, self.ft_outcomes[3407])
        with self.assertRaisesRegex(ValueError, "not preregistered"):
            self.analyze(self.runs + [extra])
        with self.assertRaisesRegex(ValueError, "missing"):
            self.analyze(self.runs[:-1])
        other = write_run(self.root, "ft-other-ckpt", variant_cfg(FAST[2]), "e" * 64, self.ft_outcomes[3409])
        with self.assertRaisesRegex(ValueError, "one checkpoint per arm"):
            self.analyze(self.runs[:-1] + [other])

    def test_tampered_score_wrong_seed_and_mismatched_protocol_refused(self):
        run = self.runs[3]
        rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
        rows[5] = {k: v for k, v in rows[5].items() if k != "_record_sha256"} | {"correct": not rows[5]["correct"]}
        rows[5]["_record_sha256"] = digest(rows[5])
        (run / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "re-score"):
            self.analyze(self.runs)
        wrong_seed = write_run(self.root, "ft-seed", variant_cfg(FAST[0]), self.ft, self.ft_outcomes[3407], seed=1)
        with self.assertRaisesRegex(ValueError, "seed"):
            self.analyze(self.runs[:3] + [wrong_seed] + self.runs[4:])
        unpaired = write_run(self.root, "ft-proto", variant_cfg(FAST[0]), self.ft, self.ft_outcomes[3407],
                             protocol="9" * 64)
        with self.assertRaisesRegex(ValueError, "protocol hash differs"):
            self.analyze(self.runs[:3] + [unpaired] + [self.runs[4], self.runs[5]])

    def test_run_eval_provenance_is_required_and_the_dropped_course_run_is_refused(self):
        bare = write_run(self.root, "ft-bare", variant_cfg(FAST[0]), self.ft, self.ft_outcomes[3407],
                         registration=False)
        with self.assertRaisesRegex(ValueError, pe.REGISTRATION_FILE):
            self.analyze(self.runs[:3] + [bare] + self.runs[4:])
        course = write_run(self.root, "ft-course", variant_cfg(pe.COURSE), self.ft, self.ft_outcomes[3407],
                           ids=IDS, mode="full")
        with self.assertRaisesRegex(ValueError, "protocol config is not preregistered"):  # course dropped
            self.analyze(self.runs + [course])
        registration = json.loads((self.runs[5] / pe.REGISTRATION_FILE).read_text())
        registration["proof_eval_sha256"] = "0" * 64
        (self.runs[5] / pe.REGISTRATION_FILE).write_text(json.dumps(registration))
        with self.assertRaisesRegex(ValueError, "share one registration"):
            self.analyze(self.runs)


    # ------------------------------------------------------------------ run ledger

    def ledger_events(self):
        return pe.read_ledger(self.ledger)[0]

    def test_ledger_start_refuses_a_second_start_until_the_first_is_recorded_failed(self):
        events = self.ledger_events()
        self.assertEqual([e["event"] for e in events], ["START"] * 6)
        self.assertEqual({e["run_id"] for e in events}, {r.name for r in self.runs})
        self.assertEqual({e["preregistration_sha256"] for e in events}, {self.prereg_sha})
        with self.assertRaisesRegex(ValueError, "already started as f0-" + FAST[0]):
            pe.ledger_start(self.prereg, "F0", FAST[0], "f0-again", self.ledger)
        with self.assertRaisesRegex(ValueError, "already started"):  # a resume is not a second start
            pe.ledger_start(self.prereg, "F0", FAST[0], self.runs[0].name, self.ledger)
        with self.assertRaisesRegex(ValueError, "not a preregistered run"):
            pe.ledger_start(self.prereg, "F0", pe.COURSE, "f0-course", self.ledger)
        with self.assertRaisesRegex(ValueError, "not a preregistered run"):
            pe.ledger_start(self.prereg, "FT", "mmmu-val-v8-fast8k-seed3410-v1", "ft-3410", self.ledger)
        with self.assertRaisesRegex(ValueError, "not a preregistered run"):
            pe.ledger_start(self.prereg, "FT", "mmmu-val-v8-c16-seed3407-v1", "ft-c16", self.ledger)
        with self.assertRaisesRegex(ValueError, "run_id invalid"):
            pe.ledger_start(self.prereg, "FT", FAST[1], "../escape", self.ledger)
        with self.assertRaisesRegex(ValueError, "started, not yet failed"):
            pe.ledger_fail(self.ledger, "never-started", "x", self.root)
        with self.assertRaisesRegex(ValueError, "and a reason"):
            pe.ledger_fail(self.ledger, "f0-again", " ", self.root)
        self.assertEqual(len(self.ledger_events()), 6)  # every refusal left the ledger untouched
        pe.ledger_fail(self.ledger, self.runs[0].name, "pod preempted", self.root)
        with self.assertRaisesRegex(ValueError, "started, not yet failed"):
            pe.ledger_fail(self.ledger, self.runs[0].name, "again", self.root)
        pe.ledger_start(self.prereg, "F0", FAST[0], "f0-second", self.ledger)
        with self.assertRaisesRegex(ValueError, "already started as f0-second"):
            pe.ledger_start(self.prereg, "F0", FAST[0], "f0-third", self.ledger)
        self.assertEqual([e["event"] for e in self.ledger_events()][6:], ["FAIL", "START"])
        self.assertEqual(oct(self.ledger.stat().st_mode & 0o777), "0o600")

    def test_ledger_fail_refuses_a_finished_run_and_a_tampered_chain_is_refused(self):
        with self.assertRaisesRegex(ValueError, "is PARTIAL; a finished run"):
            pe.ledger_fail(self.ledger, self.runs[0].name, "score was low", self.run_root())
        self.assertEqual(len(self.ledger_events()), 6)
        lines = self.ledger.read_bytes().splitlines(keepends=True)
        for name, edited in (("edit", [lines[0].replace(b'"F0"', b'"FT"', 1), *lines[1:]]),
                             ("delete", lines[:2] + lines[3:]), ("reorder", [lines[1], lines[0], *lines[2:]]),
                             ("partial", [*lines[:-1], lines[-1].rstrip(b"\n")])):
            with self.subTest(name):
                broken = self.root / f"broken-{name}.jsonl"
                broken.write_bytes(b"".join(edited))
                with self.assertRaisesRegex(ValueError, "chain broken|partial line"):
                    pe.read_ledger(broken)
        with self.assertRaisesRegex(ValueError, "chain broken|partial line"):
            pe.ledger_start(self.prereg, "FT", FAST[0], "ft-x", self.root / "broken-delete.jsonl")

    def run_root(self):
        """An artifact root whose runs/<run_id> holds the synthetic finished (PARTIAL) runs."""
        root = self.root / "artifacts"
        (root / "runs").mkdir(parents=True, exist_ok=True)
        for run in self.runs:
            if not (root / "runs" / run.name).exists():
                (root / "runs" / run.name).symlink_to(run)
        return root

    def test_analyze_refuses_unstarted_failed_replaced_and_omitted_runs(self):
        again = write_run(self.root, "ft-again", variant_cfg(FAST[2]), self.ft, lambda i: i % 2 == 0 or i % 10 == 1)
        runs = self.runs[:5] + [again]  # a repeat of (FT, seed 3409) that was never started in the ledger
        with self.assertRaisesRegex(ValueError, "ft-again was never started"):
            self.analyze(runs)
        # Starting it is refused while the first run is live, so the repeat cannot be slipped in silently.
        with self.assertRaisesRegex(ValueError, "already started"):
            self.start(again)
        pe.ledger_fail(self.ledger, self.runs[5].name, "operator says crashed", self.root)  # no runs/ dir: nothing COMPLETE
        self.start(again)
        with self.assertRaisesRegex(ValueError, "FAILED"):  # the replaced run is no longer analyzable
            self.analyze(self.runs)
        report = self.analyze(runs)
        self.assertEqual(report["ledger"]["failed_run_ids"], [self.runs[5].name])
        self.assertEqual(report["ledger"]["starts"], 7)
        self.assertEqual(report["ledger"]["head"], pe.read_ledger(self.ledger)[1])
        self.assertEqual(report["ledger"]["file_sha256"], sha256_file(self.ledger))
        # A run started under another (arm, variant) than its content declares.
        other = self.root / "ledger-2.jsonl"
        keys = [("F0", FAST[0]), ("F0", FAST[2]), ("F0", FAST[1]), ("FT", FAST[0]), ("FT", FAST[1]), ("FT", FAST[2])]
        for run, (arm, variant) in zip(self.runs, keys):
            pe.ledger_start(self.prereg, arm, variant, run.name, other)
        with self.assertRaisesRegex(ValueError, "another .arm, variant. or FAILED"):
            pe.analyze(self.prereg, self.runs, self.ft_ref, other)

    def test_analyze_needs_a_ledger_of_the_same_preregistration(self):
        with self.assertRaises(TypeError):
            pe.analyze(self.prereg, self.runs, self.ft_ref)  # type: ignore[call-arg]
        with self.assertRaises(FileNotFoundError):
            pe.analyze(self.prereg, self.runs, self.ft_ref, self.root / "absent.jsonl")
        other = self.root / "other.jsonl"
        prereg = json.loads(self.prereg.read_text())
        with mock.patch.object(pe, "load_preregistration", return_value=(prereg, "1" * 64)):
            for run in self.runs:
                self.start(run, other)
        with self.assertRaisesRegex(ValueError, "another preregistration"):
            pe.analyze(self.prereg, self.runs, self.ft_ref, other)

    def test_cli_requires_ledger_and_writes_it(self):
        out = self.root / "report"
        argv = ["analyze", "--prereg", str(self.prereg), "--runs", *map(str, self.runs), "--ft-model-ref",
                str(self.ft_ref), "--output", str(out)]
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            pe.main(argv)
        cli = self.root / "cli.jsonl"
        base = ["--prereg", str(self.prereg), "--ledger", str(cli)]
        with mock.patch("builtins.print") as printed:
            self.assertEqual(pe.main(["ledger-start", *base, "--arm", "FT", "--variant", FAST[0], "--run-id", "c1"]), 0)
            self.assertEqual(pe.main(["ledger-fail", "--ledger", str(cli), "--run-id", "c1", "--reason", "oom",
                                      "--artifact-root", str(self.root / "no-artifacts")]), 0)
            self.assertEqual(pe.main(["ledger-start", *base, "--arm", "FT", "--variant", FAST[0], "--run-id", "c2"]), 0)
            self.assertEqual(pe.main([*argv, "--ledger", str(self.ledger)]), 0)
        self.assertEqual([e["event"] for e in pe.read_ledger(cli)[0]], ["START", "FAIL", "START"])
        self.assertTrue(out.with_suffix(".json").is_file())
        self.assertIn("LEDGER_FAILED c1", [call.args[0] for call in printed.call_args_list])


class ProofExportTests(unittest.TestCase):
    def test_tiny_export_yields_model_ref_accepted_by_eval_verify_model(self):
        from fine_tuning.notebook_rl.checkpoint import capture_rng, make_optimizer, save
        from fine_tuning.tests.test_notebook_rl_checkpoint import action_step, fixture, fresh_tiny
        from mmdl.evaluation.engine import verify_model

        torch.set_num_threads(2)
        random.seed(3407)
        np.random.seed(3407)
        torch.manual_seed(3407)
        with tempfile.TemporaryDirectory(prefix="proof-export-") as tmp:
            root = Path(tmp)
            model_path, base_manifest, protocol, binding = fixture(root)
            bundle = fresh_tiny(model_path, "MOD-RV", protocol)
            optimizer, scheduler = make_optimizer(bundle, protocol)
            action_step(bundle, optimizer, scheduler)
            bundle.route.apply(False)
            cursor = {"epoch": 0, "position": 1, "discarded_problem_ids": []}
            save(bundle, optimizer, scheduler, 1, cursor, capture_rng(), root / "ckpt", protocol=protocol,
                 binding=binding)
            launch = root / "launch.json"
            launch.write_text(json.dumps({"binding": binding}))
            with self.assertRaisesRegex(ValueError, "40-hex"):
                pe.export_for_eval(root / "ckpt", launch, model_path, root / "bad", "HEAD")
            ref = pe.export_for_eval(root / "ckpt", launch, model_path, root / "out", "a" * 40,
                                     _test_policy_factory=fresh_tiny, _test_base_manifest=base_manifest)
            self.assertEqual(json.loads((root / "out/model_ref.json").read_text()), ref)
            self.assertEqual(ref["kind"], "merged")
            self.assertIs(ref["export"]["unmerged_inference_bit_equivalent"], False)
            self.assertEqual(ref["export"]["arm"], "MOD-RV")
            self.assertEqual(ref["data_manifest_sha256"], binding["data_sha256"])
            self.assertIn("export_manifest.json", {f["path"] for f in ref["files"]})
            verify_model(root / "out/model", ref)  # the unchanged src/ evaluator accepts the reference
            config = root / "out/model/config.json"
            config.write_bytes(config.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify_model(root / "out/model", ref)
            with self.assertRaises(FileExistsError):
                pe.export_for_eval(root / "ckpt", launch, model_path, root / "out", "a" * 40,
                                   _test_policy_factory=fresh_tiny, _test_base_manifest=base_manifest)


A40 = pe.HERE / "eval_configs/hardware/a40_48gb.yaml"


class FakeLLM:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        FakeLLM.calls.append(kwargs)

    def get_default_sampling_params(self):
        return "defaults"


def backend_mocks():
    """CPU stand-ins for CUDA, the HF processor/config and vLLM; the real VLLMBackend.__init__ runs."""
    from mmdl.evaluation.backends import vllm_backend

    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.is_bf16_supported.return_value = True
    return mock.patch.multiple(vllm_backend, torch=fake_torch, AutoProcessor=mock.MagicMock(),
                               AutoConfig=mock.MagicMock(), _vllm=lambda: (FakeLLM, object, "0.11.0"),
                               _final_only_output_kind=lambda: "FINAL_ONLY")


def validation_outcome(cfg, hw):
    from mmdl.runtime import contracts

    try:
        contracts.validate_configs(copy.deepcopy(cfg), copy.deepcopy(hw))
        return None
    except Exception as exc:  # noqa: BLE001 - the outcome itself is compared
        return type(exc).__name__, str(exc)


def changed(cfg, **paths):
    cfg = copy.deepcopy(cfg)
    for path, value in paths.items():
        *parents, leaf = path.split("__")
        node = cfg
        for key in parents:
            node = node[key]
        node[leaf] = value
    return cfg


class ProofRunEvalTests(unittest.TestCase):
    def test_a40_profile_is_rtx4090_semantics_with_48gb_caps(self):
        base = yaml.safe_load((pe.ROOT / "configs/hardware/rtx4090_24gb.yaml").read_text())
        a40 = yaml.safe_load(A40.read_text())
        self.assertEqual({k for k in base if base[k] != a40[k]}, {"name", "expected_vram_gib", "gpu_weight_cap_gib"})
        self.assertEqual((a40["placement"], a40["expected_vram_gib"], a40["gpu_weight_cap_gib"]), ("gpu_only", 48, 44.0))
        self.assertIsNone(validation_outcome(variant_cfg(pe.COURSE), a40))

    def test_registration_changes_no_other_protocol_and_is_restored(self):
        from mmdl.evaluation.backends.vllm_backend import VLLMBackend
        from mmdl.runtime import contracts

        hws = {p.name: yaml.safe_load(p.read_text())
               for p in [*sorted((pe.ROOT / "configs/hardware").glob("*.yaml")), A40]}
        cfgs = {p.name: yaml.safe_load(p.read_text()) for p in sorted((pe.ROOT / "configs/eval").glob("*.yaml"))}
        v8, fast = cfgs["mmmu_val_v8.yaml"], variant_cfg(FAST[0])
        cfgs |= {"v8_batch32": changed(v8, execution__batch_size=32),
                 "v8_seed3408": changed(v8, generation__seed=3408),
                 "v8_8k_low_res": changed(v8, generation__max_new_tokens=8192, image__min_pixels=262144,
                                          image__max_pixels=1003520, execution__max_model_len=24576),
                 "unregistered_fast": changed(fast, protocol_id="mmmu-val-v8-fast8k-seed3410-v1"),
                 "unregistered_fast_seed3410": changed(fast, protocol_id="mmmu-val-v8-fast8k-seed3410-v1",
                                                       generation__seed=3410),
                 **{f"old_{p.stem}": yaml.safe_load(p.read_text()) for p in C16_FILES}}
        bad_registered = {"wrong_seed": changed(fast, generation__seed=3408),
                          "two_slots": changed(fast, execution__batch_size=2),
                          "sixteen_slots": changed(fast, execution__batch_size=16),
                          "32k_output": changed(fast, generation__max_new_tokens=32768),
                          "4k_output": changed(fast, generation__max_new_tokens=4096),
                          "official_pixels": changed(fast, image__min_pixels=1003520, image__max_pixels=4014080),
                          "max_pixels_only": changed(fast, image__max_pixels=1310720),
                          "40960_context": changed(fast, execution__max_model_len=40960),
                          "v4_parser": changed(fast, parser="team-final-answer-v4"),
                          "official_resize": changed(fast, image__resize_owner="official_processor")}
        registered = {vid: variant_cfg(vid) for vid in FAST}
        tables = copy.deepcopy((contracts.CONTROL_IMAGES, contracts.EXECUTION_PROFILES))
        functions = (contracts.validate_configs, VLLMBackend.__init__)
        outside = {(c, h): validation_outcome(cfgs[c], hws[h]) for c in cfgs for h in hws}
        self.assertTrue(all(validation_outcome(cfg, hws["a40_48gb.yaml"]) for cfg in registered.values()))
        with pe.registered_variants():
            for key, expected in outside.items():
                self.assertEqual(validation_outcome(cfgs[key[0]], hws[key[1]]), expected, key)
            for vid, cfg in registered.items():
                for h, hw in hws.items():  # a registered ID validates exactly as its mmmu-val-v8 twin
                    self.assertEqual(validation_outcome(cfg, hw), outside[("mmmu_val_v8.yaml", h)], (vid, h))
            for name, cfg in bad_registered.items():
                self.assertIsNotNone(validation_outcome(cfg, hws["a40_48gb.yaml"]), name)
            for name in ("unregistered_fast", "old_mmmu_val_v8_c16_seed3407"):
                self.assertIn("named fixed protocol", validation_outcome(cfgs[name], hws["a40_48gb.yaml"])[1])
            cfg, _ = contracts.load_configs(pe.VARIANTS[FAST[2]][0], A40)  # the cli.py import path
            self.assertEqual((cfg["generation"]["seed"], cfg["execution"]["batch_size"],
                              cfg["generation"]["max_new_tokens"], cfg["execution"]["max_model_len"],
                              cfg["image"]["min_pixels"], cfg["image"]["max_pixels"]),
                             (3409, 32, 8192, 24576, 262144, 1003520))
            with self.assertRaisesRegex(RuntimeError, "already registered"):
                with pe.registered_variants():
                    pass
        self.assertEqual((contracts.CONTROL_IMAGES, contracts.EXECUTION_PROFILES), tables)
        self.assertEqual((contracts.validate_configs, VLLMBackend.__init__), functions)
        self.assertEqual({k: validation_outcome(cfgs[k[0]], hws[k[1]]) for k in outside}, outside)

    def test_backend_gets_thirty_two_slots_only_for_registered_ids(self):
        from mmdl.evaluation.backends.vllm_backend import VLLMBackend

        hw, fast, v8 = yaml.safe_load(A40.read_text()), variant_cfg(FAST[1]), variant_cfg(pe.COURSE)
        FakeLLM.calls.clear()
        recorded = []
        with backend_mocks():
            with self.assertRaisesRegex(ValueError, "one or two active requests"):
                VLLMBackend("model", fast, hw)
            with pe.registered_variants(on_backend=lambda: recorded.append(True)):
                wide = VLLMBackend("model", fast, hw)
                plain = VLLMBackend("model", v8, hw)
                with self.assertRaisesRegex(ValueError, "one or two active requests"):
                    VLLMBackend("model", changed(v8, execution__batch_size=32), hw)
        self.assertEqual([(c["max_num_seqs"], c["max_model_len"], c["mm_processor_kwargs"]["size"],
                           c["enable_prefix_caching"]) for c in FakeLLM.calls],
                         [(32, 24576, {"shortest_edge": 262144, "longest_edge": 1003520}, False),
                          (2, 40960, {"shortest_edge": 1003520, "longest_edge": 4014080}, False)])
        self.assertEqual((wide.batch_size, plain.batch_size), (32, 2))
        self.assertEqual(wide.max_model_len, 24576)
        self.assertIs(wide.cfg, fast)
        self.assertEqual(recorded, [True])

    def test_run_eval_calls_the_eval_sh_entry_point_and_records_registration(self):
        from mmdl.evaluation import engine
        from mmdl.evaluation.backends.vllm_backend import VLLMBackend
        from mmdl.runtime import contracts

        seen = {}

        def fake_run(args, cfg, hw, root):
            run_dir = args.artifact_root / "runs" / args.run_id
            run_dir.mkdir(parents=True, exist_ok=args.resume)
            seen.update(cfg=cfg, cwd=Path.cwd(), hw=hw["name"], model_ref=args.model_ref, mode=args.mode,
                        sample_id=args.sample_id, run_role=args.run_role)
            VLLMBackend(str(args.model_path), cfg, hw, "base", str(args.base_path))
            return {"status": "SMOKE"}

        with tempfile.TemporaryDirectory(prefix="proof-run-eval-") as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data/manifest.json").write_text("{}")
            argv = ["--protocol", "fine_tuning/notebook_rl/eval_configs/mmmu_val_v8_fast8k_seed3408.yaml",
                    "--hardware", str(A40), "--model-ref", "manifests/models/baseline.json",
                    "--model-path", str(root / "model"), "--base-path", str(root / "base"),
                    "--data-root", str(root / "data"), "--artifact-root", str(root / "art"),
                    "--job-id", "fast8k-3408", "--run-role", "standalone", "--run-id", "fast8k-3408-evaluation",
                    "--mode", "partial", *(x for sid in SUBSET[:3] for x in ("--sample-id", sid)), "--no-download"]
            cwd, saved_argv = Path.cwd(), list(sys.argv)
            FakeLLM.calls.clear()
            with backend_mocks(), mock.patch.object(engine, "run", fake_run), mock.patch.dict(os.environ):
                self.assertEqual(pe.main(["run-eval", *argv]), 0)
                run_dir = root / "art/runs/fast8k-3408-evaluation"
                registration = json.loads((run_dir / pe.REGISTRATION_FILE).read_text())
                registration_path = run_dir / pe.REGISTRATION_FILE
                registration_path.write_text(json.dumps(registration | {"proof_eval_sha256": "0" * 64}))
                with self.assertRaisesRegex(ValueError, "different variant registration"):
                    pe.run_eval(argv + ["--resume"])
            self.assertEqual((seen["cfg"]["generation"]["seed"], seen["cfg"]["execution"]["batch_size"],
                              seen["cfg"]["generation"]["max_new_tokens"], seen["cfg"]["image"]["max_pixels"]),
                             (3408, 32, 8192, 1003520))
            # --mode partial and every --sample-id reach the evaluator unchanged (the real cli validation ran)
            self.assertEqual((seen["mode"], seen["sample_id"], seen["run_role"]), ("partial", SUBSET[:3], "standalone"))
            self.assertEqual((seen["cwd"], seen["hw"], seen["model_ref"]),
                             (pe.ROOT, "a40_48gb", Path("manifests/models/baseline.json")))
            self.assertEqual([(call["max_num_seqs"], call["max_model_len"]) for call in FakeLLM.calls], [(32, 24576)])
            self.assertEqual(registration, pe.variant_registration(FAST[1]))
            self.assertEqual(set(registration["registered"]), set(FAST))
            self.assertEqual(registration["proof_eval_sha256"], sha256_file(Path(pe.__file__)))
            self.assertEqual(set(registration["src_files"]), set(pe.PATCHED_SRC))
            self.assertEqual((Path.cwd(), sys.argv), (cwd, saved_argv))
            self.assertFalse(set(FAST) & set(contracts.EXECUTION_PROFILES))
            for protocol in (pe.ROOT / "configs/eval/mmmu_val_v8.yaml", root / "edited.yaml", *C16_FILES):
                if protocol.name == "edited.yaml":
                    protocol.write_text(pe.VARIANTS[FAST[0]][0].read_text() + "# edited\n")
                with self.assertRaisesRegex(ValueError, "plain scripts/eval.sh"):
                    pe.run_eval(["--protocol", str(protocol), "--run-id", "refused"])


if __name__ == "__main__":
    unittest.main()
