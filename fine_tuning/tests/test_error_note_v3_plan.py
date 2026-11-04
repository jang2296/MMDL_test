"""CPU-only v3 stage plans: private pins, review boundary and shared admission."""
from __future__ import annotations

import contextlib
import copy
import datetime as dt
import io
import json
import tempfile
import unittest
from pathlib import Path

from fine_tuning.error_note import plan as planner
from fine_tuning.notebook_rl.pod import chain, job


BASE = {
    "version": "error-note-plan-params-v3", "commit": "a" * 40,
    "python": "/workspace/volume/venv/bin/python", "repo_root": "/workspace/volume/clone",
    "build_root": "/workspace/private/build", "out_root": "/workspace/private/note-v3",
    "volume_root": "/workspace/volume",
    "model_path": "/workspace/volume/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17",
    "order_path": "/workspace/private/order.json", "order_sha256": "b" * 64,
    "training_rules_path": "/workspace/private/training-rules.json", "training_rules_sha256": "c" * 64,
    "source_provenance_path": "/workspace/private/source-provenance.json", "source_provenance_sha256": "d" * 64,
    "limit": 300, "seqs": "auto", "sweep_seconds": 600, "kv_cache_gib": "auto",
    "max_num_batched_tokens": 16384, "token_budget": 24000, "gradient_checkpointing": "on",
    "timeouts": {"collect": 111, "train": 222, "reload": 333, "export": 444},
}
COLLECT_DIR = f"{BASE['out_root']}/collect"
RESOLVE_PINS = {"reviews_path": "/workspace/private/reviews.json", "reviews_sha256": "e" * 64,
              "t1_sha256": "f" * 64, "note_sha256": "0" * 64}
TRAIN_PINS = {"rows_path": f"{COLLECT_DIR}/rows.jsonl", "rows_sha256": "1" * 64,
              "audit_path": f"{COLLECT_DIR}/audit.json", "audit_sha256": "2" * 64,
              "solution_audit_path": "/workspace/private/solution-audit.json", "solution_audit_sha256": "3" * 64}


def params(stage: str, **changes: object) -> dict:
    value = copy.deepcopy(BASE)
    if stage == "note-v3-train":
        value.update(TRAIN_PINS)
    elif stage == "note-v3-resolve":
        value.update(RESOLVE_PINS)
    value.update(changes)
    return value


def value_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


class V3PlanTests(unittest.TestCase):
    def test_collect_stops_at_notes_for_actual_source_review(self) -> None:
        spec = planner.plan("note-v3-collect", params("note-v3-collect"))
        self.assertIs(chain.check_spec(spec), spec)
        self.assertEqual([step["name"] for step in spec["steps"]],
                         ["pin-order", "pin-training-rules", "pin-source-provenance", "sweep", "note"])
        note = spec["steps"][-1]["argv"]
        self.assertEqual(value_after(note, "--note-version"), "v2")
        self.assertEqual(value_after(note, "--max-num-seqs-from"), f"{COLLECT_DIR}/sweep.json")
        self.assertNotIn("reviews", json.dumps(spec))

    def test_resolve_pins_collected_evidence_and_stops_before_solution_review(self) -> None:
        spec = planner.plan("note-v3-resolve", params("note-v3-resolve"))
        self.assertIs(chain.check_spec(spec), spec)
        names = [step["name"] for step in spec["steps"]]
        self.assertEqual(names, ["pin-order", "pin-training-rules", "pin-source-provenance", "pin-t1", "pin-note",
                                 "pin-reviews", "audit", "resolve", "rows"])
        steps = {step["name"]: step for step in spec["steps"]}
        for name, path, digest in (("order", BASE["order_path"], BASE["order_sha256"]),
                                   ("training-rules", BASE["training_rules_path"], BASE["training_rules_sha256"]),
                                   ("source-provenance", BASE["source_provenance_path"], BASE["source_provenance_sha256"]),
                                   ("t1", f"{COLLECT_DIR}/t1.jsonl", RESOLVE_PINS["t1_sha256"]),
                                   ("note", f"{COLLECT_DIR}/note.jsonl", RESOLVE_PINS["note_sha256"]),
                                   ("reviews", RESOLVE_PINS["reviews_path"], RESOLVE_PINS["reviews_sha256"])):
            self.assertEqual(steps[f"pin-{name}"]["argv"][3:], ["pin", "--file", path, "--sha256", digest])
        self.assertEqual(steps["audit"]["expect_files"], [f"{COLLECT_DIR}/audit.json", f"{COLLECT_DIR}/lessons-v3.json"])
        self.assertEqual(value_after(steps["rows"]["argv"], "--rows-version"), "v3")
        self.assertEqual(value_after(steps["resolve"]["argv"], "--max-num-seqs"), "auto")

    def test_train_consumes_reviewed_closed_rows_without_regeneration(self) -> None:
        spec = planner.plan("note-v3-train", params("note-v3-train"))
        self.assertIs(chain.check_spec(spec), spec)
        steps = {step["name"]: step for step in spec["steps"]}
        self.assertEqual(list(steps), ["pin-order", "pin-training-rules", "pin-source-provenance", "pin-rows", "pin-audit",
                                      "pin-solution-audit", "admit", "train-note", "reload-note", "export-note",
                                      "train-no-note", "reload-no-note", "export-no-note"])
        self.assertFalse(any(step["argv"][2] == planner.COLLECT for step in steps.values()))
        for name in ("rows", "audit", "solution_audit"):
            self.assertEqual(steps[f"pin-{name.replace('_', '-')}"]["argv"][3:],
                             ["pin", "--file", TRAIN_PINS[f"{name}_path"], "--sha256", TRAIN_PINS[f"{name}_sha256"]])

    def test_common_training_contract_and_same_admission_for_both_arms(self) -> None:
        for stage in planner.V3_STAGES:
            spec = planner.plan(stage, params(stage))
            for step in spec["steps"]:
                argv = step["argv"]
                if argv[2] == planner.COLLECT:
                    for flag, value in (("--order", BASE["order_path"]), ("--answer-contract", "training-answer-v2"),
                                        ("--training-rules", BASE["training_rules_path"]),
                                        ("--training-rules-sha256", BASE["training_rules_sha256"])):
                        self.assertEqual(value_after(argv, flag), value, step["name"])
                if argv[2] == planner.TRAIN:
                    self.assertEqual(value_after(argv, "--precision"), "bf16")
                    if argv[3] != "reload":
                        for flag, value in (("--audit", f"{COLLECT_DIR}/audit.json"),
                                            ("--sidecar", BASE["training_rules_path"]), ("--order", BASE["order_path"]),
                                            ("--rows", TRAIN_PINS["rows_path"]),
                                            ("--solution-audit", TRAIN_PINS["solution_audit_path"])):
                            self.assertEqual(value_after(argv, flag), value, step["name"])
                    if argv[3] == "train":
                        self.assertEqual(value_after(argv, "--admission"), f"{BASE['out_root']}/admission.json")
                        self.assertEqual(value_after(argv, "--token-budget"), "24000")
                    if argv[3] in ("train", "reload"):
                        self.assertEqual(value_after(argv, "--protocol-version"), "v3")
        exports = [step for step in spec["steps"] if step["name"].startswith("export-")]
        self.assertEqual([value_after(step["argv"], "--output") for step in exports],
                         [f"{BASE['volume_root']}/eval-data/proof/export-{arm}-v3" for arm in ("note", "no-note")])

    def test_missing_changed_or_repository_private_pins_refused(self) -> None:
        for stage in planner.V3_STAGES:
            p = params(stage)
            extra = TRAIN_PINS if stage.endswith("train") else RESOLVE_PINS if stage.endswith("resolve") else {}
            for key in ("order_path", "order_sha256", "training_rules_path", "training_rules_sha256",
                        "source_provenance_path", "source_provenance_sha256", *extra):
                with self.subTest(stage=stage, missing=key), self.assertRaises(ValueError):
                    planner.plan(stage, {name: value for name, value in p.items() if name != key})
            for key in ("order_sha256", "training_rules_sha256", "source_provenance_sha256",
                        *(key for key in extra if key.endswith("sha256"))):
                with self.subTest(stage=stage, malformed=key), self.assertRaises(ValueError):
                    planner.plan(stage, params(stage, **{key: "bad"}))
            for key in (*planner.V3_PRIVATE_PATHS, "build_root", "out_root", *(key for key in extra if key.endswith("path"))):
                with self.subTest(stage=stage, public=key), self.assertRaisesRegex(ValueError, "outside repo_root"):
                    planner.plan(stage, params(stage, **{key: f"{BASE['repo_root']}/private.json"}))
            for changes in ({"seqs": [1]}, {"kv_cache_gib": 8}, {"token_budget": 4096}, {"reuse_t1_sha256": "f" * 64},
                            {"arms": ["solution"]}):
                with self.subTest(stage=stage, changes=changes), self.assertRaises(ValueError):
                    planner.plan(stage, params(stage, **changes))

    def test_deterministic_cli_plan_preserves_existing_output(self) -> None:
        stage = "note-v3-train"
        first, changed = planner.plan(stage, params(stage)), planner.plan(stage, params(stage, solution_audit_sha256="4" * 64))
        self.assertNotEqual(first["meta"]["params_sha256"], changed["meta"]["params_sha256"])
        self.assertEqual(first, planner.plan(stage, params(stage)))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = Path(tmp)
            source, output = root / "params.json", root / "steps.json"
            source.write_text(json.dumps(params(stage)), encoding="utf-8")
            argv = ["plan", "--stage", stage, "--params", str(source), "--output", str(output)]
            self.assertEqual(planner.main(argv), 0)
            original = output.read_bytes()
            self.assertEqual(planner.main(argv), 2)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(json.loads(original), first)

    def test_new_job_stages_require_training_data(self) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        for stage in planner.V3_STAGES:
            value = {"version": job.VERSION, "commit": "a" * 40, "volume_root": "/workspace/volume",
                     "run_dir": "/workspace/run", "stage": stage, "smoke": ["python", "smoke"], "main": ["python", "main"],
                     "max_main_seconds": 600, "min_free_gib": 1,
                     "financial": {"owned_pod_id": "fixture", "gpu_cutoff_utc": (now + dt.timedelta(hours=1)).isoformat(),
                                   "billing_start_utc": now.isoformat(), "balance_usd": 10, "reserve_usd": 1, "hourly_usd": 1},
                     "data": {"bundle_manifest": "/workspace/bundle.json", "bundle_manifest_sha256": "b" * 64,
                              "dataset_manifest": "/private/manifest.json", "dataset_manifest_sha256": "c" * 64,
                              "vault_root": "/private/vault"},
                     "approval": {"approved_by_user": True, "approved_utc": now.isoformat(), "approval_note": "fixture",
                                  "approved_sha256": ""}}
            value["approval"]["approved_sha256"] = job.content_sha256(value)
            self.assertIs(job.validate_job(value, now=now), value)
            del value["data"]
            with self.assertRaises(job.JobError):
                job.validate_job(value, now=now)

    def test_generated_collect_and_train_argv_parse_with_real_clis(self) -> None:
        from fine_tuning.error_note import collect, train

        parsers = {planner.COLLECT: collect._parser(), planner.TRAIN: train.build_parser()}
        for stage in planner.V3_STAGES:
            for step in planner.plan(stage, params(stage))["steps"]:
                argv = step["argv"]
                if argv[2] in parsers:
                    with self.subTest(stage=stage, step=step["name"]):
                        parsers[argv[2]].parse_args(argv[3:])


if __name__ == "__main__":
    unittest.main()
