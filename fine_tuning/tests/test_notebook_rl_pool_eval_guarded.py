"""Pool evaluation under the REAL actor guards (evaluate-pool installs them before local_pool_eval.diagnose).

The pool stage tests patch `_install_actor_guards` to a no-op, so a vault read in the evaluator (e.g. lineage.jsonl)
passed CPU tests and failed on the GPU. Here a fresh process installs the guards exactly as evaluate-pool does
(`local_run._install_actor_guards(prepared, request)` over the preparation's own data request) and runs the evaluator
and the regrading `replay` with a scripted engine. Guards are process-wide audit hooks, so never in the test process.
Authored CPU fixtures only: no model, no GPU, no evidence about any model's quality.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fine_tuning.notebook_rl import compact_note, local_data, local_pool_eval, local_profile, local_run
from fine_tuning.notebook_rl.launch import read_descriptor
from fine_tuning.tests import test_notebook_rl_pool_local as pool_local
from fine_tuning.tests import test_notebook_rl_proof_selection as proof

LEGACY = "legacy-cohort-reads-lineage"


def drive_guarded(config_path: str) -> None:
    """Fresh process: real guards, then diagnose + replay; the outcome (or the refusal) goes to config["out"]."""
    config = json.loads(Path(config_path).read_text())
    protocol = pool_local.pool_protocol() if config["protocol"] == "v1" else pool_local.v2_protocol(proof.PROOF)
    preparation = Path(config["preparation"])
    prepared = json.loads(preparation.read_text())
    root = Path(prepared["build_root"])
    local_run._install_actor_guards(prepared, read_descriptor(prepared["data_request"]))
    out: dict = {"blocked": {}}
    for split in ("train", "dev"):
        for name in ("lineage.jsonl", "verifier_private.jsonl"):
            try:
                (root / split / name).read_bytes()
                out["blocked"][f"{split}/{name}"] = False
            except PermissionError:
                out["blocked"][f"{split}/{name}"] = True
    real_cohort = local_pool_eval._cohort

    def legacy_cohort(*args):
        # The pre-fix _cohort cross-checked the actors against local_data.public_pool, which reads lineage.jsonl.
        local_data.public_pool(root)
        return real_cohort(*args)

    sha = config["sha256"]
    try:
        with patch.object(local_pool_eval, "_cohort", legacy_cohort if config.get(LEGACY) else real_cohort):
            engine = pool_local.ScriptedEngine(config["replies"])
            record = local_pool_eval.diagnose(preparation, Path(config["output"]), engine,
                copy.deepcopy(local_profile.local_prompts(protocol)), expected_preparation_sha256=sha,
                local_protocol_sha256=protocol["sha256"], deadline=time.monotonic() + 600, context_limit=24576,
                note_contract=compact_note.VERSION)
            replayed = local_pool_eval.replay(record, expected_preparation_sha256=sha)
        out |= {"status": record["status"], "reason": record.get("reason"), "problem_ids": record["problem_ids"],
                "metrics_equal": replayed["metrics"] == record["metrics"]}
    except Exception as exc:  # reported to the parent test, which asserts on it
        out["error"] = f"{type(exc).__name__}: {exc}"
    Path(config["out"]).write_text(json.dumps(out))


class GuardedRun:
    def own_index_directory(self) -> None:
        """The guards forbid the request index's whole directory (the benchmark index lives there), so the
        authored index must not share a directory with the fixture's build, preparation or output."""
        index = self.base / "benchmark-index"
        index.mkdir()
        self.request = pool_local.write_request(index, self.root)

    def run_guarded(self, preparation: Path, protocol: str, replies: dict, **extra) -> dict:
        out = self.base / "guarded-out.json"
        config = {"preparation": str(preparation), "protocol": protocol, "replies": replies, "out": str(out),
                  "sha256": pool_local._sha(preparation.read_bytes()), "output": str(self.base / "guarded-evaluation"),
                  **extra}
        path = self.base / "guarded-config.json"
        path.write_text(json.dumps(config))
        done = subprocess.run([sys.executable, "-B", "-c", "import sys; from fine_tuning.tests."
                               "test_notebook_rl_pool_eval_guarded import drive_guarded; drive_guarded(sys.argv[1])",
                               str(path)], capture_output=True, text=True, timeout=900, check=False,
                              env=os.environ | {"CUDA_VISIBLE_DEVICES": ""})
        self.assertEqual(done.returncode, 0, done.stdout[-3000:] + done.stderr[-6000:])
        return json.loads(out.read_text())

    def assert_guarded_complete(self, result: dict, ids: list[str]) -> None:
        self.assertTrue(result["blocked"] and all(result["blocked"].values()), result["blocked"])  # guards are active
        self.assertEqual(result.get("error"), None)
        self.assertEqual((result["status"], result["problem_ids"], result["metrics_equal"]),
                         (local_pool_eval.COMPLETE, ids, True), result)

    def assert_legacy_refused(self, result: dict) -> None:
        # The GPU symptom: the guard's PermissionError surfaces from vault.read_lineage as this ValueError.
        self.assertEqual(result.get("error"), "ValueError: vault: missing or unreadable split file", result)
        self.assertNotIn("status", result)


class GuardedPoolEvalTests(GuardedRun, pool_local.PoolFixture):
    def setUp(self) -> None:
        super().setUp()
        self.own_index_directory()
        self.rule = self.propose(evaluation_count=2)
        _, self.preparation = self.prepare(self.rule)
        self.replies = pool_local.eval_replies(self.rule["evaluation"])

    def test_diagnose_and_replay_complete_under_the_real_guards(self) -> None:
        self.assert_guarded_complete(self.run_guarded(self.preparation, "v1", self.replies),
                                     self.rule["evaluation"])

    def test_a_cohort_that_reads_lineage_is_refused_by_the_guards(self) -> None:
        self.assert_legacy_refused(self.run_guarded(self.preparation, "v1", self.replies,
                                                    **{LEGACY: True}))


class GuardedUnionPoolEvalTests(GuardedRun, proof.ProofFixture):
    """The pod shape: frozen union vault, shard roots/registries/evidence removed, per-subject sentinels."""

    def setUp(self) -> None:
        super().setUp()
        self.own_index_directory()
        rule_path = self.propose("guarded-rule", proof.PROOF)
        self.preparation = self.prepare("guarded-prepared", rule_path, proof.PROOF)
        selected = json.loads(Path(json.loads(self.preparation.read_text())["selection"]["path"]).read_text())
        self.ids = selected["evaluation_ids"]
        self.replies = pool_local.eval_replies(self.ids, proof.SENTINEL_REPLIES)

    def test_diagnose_and_replay_complete_under_the_real_guards(self) -> None:
        self.assert_guarded_complete(self.run_guarded(self.preparation, "v2", self.replies), self.ids)

    def test_a_cohort_that_reads_lineage_is_refused_by_the_guards(self) -> None:
        self.assert_legacy_refused(self.run_guarded(self.preparation, "v2", self.replies, **{LEGACY: True}))


if __name__ == "__main__":
    unittest.main()
