"""frozen-approval-verify-v1: a union-vault pool preparation checked by hashes instead of a re-run approval.

Authored CPU fixtures only: the pool_local build wrapped as a one-shard pool_freeze union vault and an authored
stand-in approval output. Not a real exclusion approval, not a pod run, not evidence about any model.
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from fine_tuning.notebook_rl import approval as a, local_data, local_run, local_train, passrate, pool_freeze as pf
from fine_tuning.notebook_rl.dataset import SPLITS
from fine_tuning.tests.test_notebook_rl_pool_local import (PoolFixture, _sha, _write_json, fake_validation, pool_build,
                                                           write_request)


def unionize(base: Path, root: Path) -> dict:
    """Wrap a built pool vault in pool_freeze's materialize output: placeholder receipt, dataset manifest of every
    record/image, freeze binding and a one-shard union receipt whose shard root/registry/evidence live outside it."""
    shard = base / "shard-0"
    (shard / "contamination").mkdir(parents=True)
    shard_receipt = _write_json(shard / "contamination" / "receipt_v2.json", {"status": "authored shard receipt"})
    shard_build = _write_json(shard / "manifest.yaml", {"format": "authored shard build"})
    inputs = {name: base / f"shard-0-{name}.json" for name in ("registry", "evidence")}
    descriptors = {name: {"path": str(path), "file_sha256": _write_json(path, {"authored": name})}
                   for name, path in inputs.items()}
    manifest = json.loads((root / "manifest.yaml").read_bytes())
    manifest["exclusion_receipt_sha256"] = _write_json(root / "contamination" / "receipt_v2.json",
                                                       {"status": "BLOCKED_NOT_CHECKED", "allowlist": []})
    vault_sha = _write_json(root / "manifest.yaml", manifest)
    status = {"train": pf.INCLUDED, "dev": pf.DEV, "sealed": pf.SEALED, "style_sealed": pf.SEALED}
    items = []
    for split in SPLITS:
        lines = {kind: pf._lines(root / split / name) for kind, name in pf._RECORDS}
        for pid, line in lines["actor"].items():
            paths = json.loads(line)["image_paths"]
            items.append({"problem_id": pid, "shard": 0, "split": split, "status": status[split],
                          "record_sha256": {kind: _sha(lines[kind][pid]) for kind in lines},
                          "image_paths": paths,
                          "image_sha256": [_sha((root / "images" / pf._rel(rel)).read_bytes()) for rel in paths]})
    meta = {"shard": 0, "build_manifest_sha256": shard_build, "exclusion_receipt_file_sha256": shard_receipt}
    dataset = base / "frozen" / "dataset_manifest.json"
    dataset.parent.mkdir()
    manifest_sha = _write_json(dataset, {"version": pf.VERSION, "shards": [meta], "items": items})
    binding_sha = _write_json(root / "freeze_binding.json", {
        "version": pf.BINDING_VERSION, "dataset_manifest_sha256": manifest_sha, "vault_manifest_sha256": vault_sha,
        "counts": {split: sum(r["split"] == split for r in items) for split in SPLITS}, "training_allowed": False})
    identity = {key: f"authored-{key}" for key in pf._IDENTITY}
    entry = {"shard": 0, "root": str(shard), "build_manifest_sha256": shard_build,
             "receipt": {"path": str(shard / "contamination" / "receipt_v2.json"), "file_sha256": shard_receipt},
             "source_registry": descriptors["registry"], "evidence": descriptors["evidence"], **identity,
             "source_registry_sha256": "1" * 64, "evidence_sha256": "2" * 64}
    union = {"version": pf.UNION_VERSION, "status": pf.UNION_STATUS, "training_allowed": False,
             "approval_granted": False, "human_verified": False,
             "dataset_manifest": {"path": str(dataset), "file_sha256": manifest_sha}, "vault_manifest_sha256": vault_sha,
             "freeze_binding_sha256": binding_sha, **identity,
             "source_registry_sha256": a._hash(["1" * 64]), "evidence_sha256": a._hash(["2" * 64]), "shards": [entry],
             "items": {r["problem_id"]: {"shard": 0, "shard_allowlisted": True} for r in items},
             "limitations": ["authored fixture"]}
    union["union_receipt_sha256"] = a._hash(union)
    _write_json(root / "contamination" / pf.UNION_FILE, union)
    pf.read_union(root)
    assert pf.verify(dataset, root) == []
    return {"shard": shard, "inputs": list(inputs.values()), "dataset": dataset}


def union_validation(root, **kwargs) -> dict:
    """Authored stand-in for a passing ai_reviewed_local worker over a union vault (every kept item allowlisted)."""
    with patch.object(local_data, "pool_allowlist", side_effect=lambda _root, public: set(public)):
        result = fake_validation(root, **kwargs)
    result["input_binding"]["exclusion_receipt_sha256"] = _sha((Path(root) / "contamination" / pf.UNION_FILE).read_bytes())
    return result


class FrozenApprovalTests(PoolFixture):
    def setUp(self) -> None:
        super().setUp()
        rule = self.propose(evaluation_count=2)          # label-blind proposal over the allowlisted build
        self.union = unionize(self.base, self.root)
        rule["build_manifest_sha256"] = _sha((self.root / "manifest.yaml").read_bytes())
        result, self.path = self.prepare(rule, validation=union_validation)
        self.assertEqual(result["status"], local_data.POOL_STATUS, result.get("blockers"))
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=union_validation) as validator:
            self.fresh = local_run.read_preparation(self.path, self.protocol)
        validator.assert_called_once()
        # Simulated pod: the approval output copied elsewhere, every shard root/registry/evidence gone.
        bound = result["validated_data"]
        pod = self.base / "pod" / "approval.json"
        pod.parent.mkdir()
        shutil.copyfile(bound["path"], pod)
        self.frozen = {"path": str(pod), "file_sha256": bound["file_sha256"]}
        shutil.rmtree(self.union["shard"])
        for path in self.union["inputs"]:
            path.unlink()

    def read(self, frozen: dict | None = None, path: Path | None = None):
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=AssertionError("approval re-run")):
            return local_run.read_frozen_preparation(path or self.path, self.protocol, frozen or self.frozen)

    def test_frozen_mode_verifies_hashes_without_shards_and_is_recorded(self) -> None:
        prepared, request, actors, record = self.read()
        self.assertEqual((prepared, request, actors), self.fresh)
        union_sha = _sha((self.root / "contamination" / pf.UNION_FILE).read_bytes())
        self.assertEqual({k: record[k] for k in ("mode", "approval_path", "approval_sha256", "manifest_sha256")},
                         {"mode": local_run.FROZEN_APPROVAL_VERIFY, "approval_path": self.frozen["path"],
                          "approval_sha256": self.frozen["file_sha256"],
                          "manifest_sha256": _sha(self.union["dataset"].read_bytes())})
        self.assertEqual((record["vault"]["root"], record["vault"]["union_receipt_file_sha256"],
                          record["vault"]["kept_items"], record["vault"]["verify_errors"]),
                         (str(self.root), union_sha, len(local_data.public_pool(self.root)), 0))
        # A run's later reads repeat the recorded mode; a changed record, or none, is not mistaken for it.
        launch = {"preparation": str(self.path), "data_verification": record}
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=AssertionError("approval re-run")):
            self.assertEqual(local_run.read_launch_preparation(launch, self.protocol), self.fresh)
            self.assertEqual(local_train._launch_preparation(launch, self.protocol), self.fresh)
            changed = dict(launch, data_verification={**record, "manifest_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "differs from the run's launch"):
                local_run.read_launch_preparation(changed, self.protocol)
        with patch.object(local_run, "read_preparation", return_value=self.fresh) as fresh, \
                patch.object(local_train, "read_preparation", return_value=self.fresh) as train_fresh:
            local_run.read_launch_preparation({"preparation": str(self.path)}, self.protocol)
            local_train._launch_preparation({"preparation": str(self.path)}, self.protocol)
        fresh.assert_called_once_with(self.path, self.protocol)
        train_fresh.assert_called_once_with(self.path, self.protocol)

    def test_refusals(self) -> None:
        with self.assertRaisesRegex(ValueError, "not this preparation's bound validated_data"):
            self.read({**self.frozen, "file_sha256": "0" * 64})
        other = self.base / "pod" / "other-approval.json"
        other.write_bytes(Path(self.frozen["path"]).read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "not this preparation's bound validated_data"):
            self.read({"path": str(other), "file_sha256": _sha(other.read_bytes())})
        with self.assertRaisesRegex(ValueError, "artifact bytes differ"):
            self.read({"path": str(other), "file_sha256": self.frozen["file_sha256"]})
        with patch.object(a, "code_hashes", return_value={**a.code_hashes(), "changed.py": "0" * 64}), \
                self.assertRaisesRegex(ValueError, "stale source"):
            self.read()
        image = next(path for path in sorted((self.root / "images").rglob("*")) if path.is_file())
        original = image.read_bytes()
        image.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "frozen vault differs from its dataset manifest"):
            self.read()
        image.write_bytes(original)
        # A re-hashed, still well-formed union receipt that the approval did not bind.
        union_path = self.root / "contamination" / pf.UNION_FILE
        union = json.loads(union_path.read_bytes())
        union["limitations"] = ["edited after approval"]
        union["union_receipt_sha256"] = a._hash({k: v for k, v in union.items() if k != "union_receipt_sha256"})
        saved = union_path.read_bytes()
        _write_json(union_path, union)
        with self.assertRaisesRegex(ValueError, "not the one the approval bound"):
            self.read()
        union_path.write_bytes(saved)
        self.assertEqual(self.read()[:3], self.fresh)
        # Non-union pool preparation and the fixed-cohort format are refused.
        plain = self.base / "plain"
        pool_build(plain)
        (self.base / "plain-request").mkdir()
        self.root, self.request = plain, write_request(self.base / "plain-request", plain)
        result, path = self.prepare(self.propose(plain, evaluation_count=2), "plain-prepared")
        self.assertEqual(result["status"], local_data.POOL_STATUS)
        with self.assertRaisesRegex(ValueError, "requires a union vault"):
            self.read(result["validated_data"], path)
        cohort = self.base / "cohort.json"
        cohort.write_text(json.dumps({"format": "notebook-rl-local-data-v1"}))
        with self.assertRaisesRegex(ValueError, "accepts only pool preparations"):
            self.read(path=cohort)

    def test_cli_flags_are_explicit_opt_in(self) -> None:
        sha = self.frozen["file_sha256"]
        base = ["minimal", "--preparation", str(self.path), "--model-path", str(self.base), "--output", str(self.base / "o"),
                "--repair-notebook", "--context-limit", "24576"]
        with patch.object(local_run, "capture") as capture:
            self.assertEqual(local_run.main(base), 0)
            self.assertEqual(local_run.main([*base, "--frozen-approval", self.frozen["path"],
                                             "--frozen-approval-sha256", sha]), 0)
        self.assertNotIn("frozen_approval", capture.call_args_list[0].kwargs)
        self.assertEqual(capture.call_args_list[1].kwargs["frozen_approval"], self.frozen)
        train = ["train", "--preparation", str(self.path), "--model-path", str(self.base), "--output", str(self.base / "t"),
                 "--arm", "notebook"]
        with patch.object(local_train, "train", return_value={"status": "x"}) as run, patch("builtins.print"):
            local_train.main(train)
            local_train.main([*train, "--frozen-approval", self.frozen["path"], "--frozen-approval-sha256", sha])
        self.assertNotIn("frozen_approval", run.call_args_list[0].kwargs)
        self.assertEqual(run.call_args_list[1].kwargs["frozen_approval"], self.frozen)
        for argv, main in (([*base, "--frozen-approval", self.frozen["path"]], local_run.main),
                           ([*train, "--frozen-approval-sha256", sha], local_train.main),
                           (["measure", "--build-root", str(self.root), "--model-path", str(self.base), "--k", "1",
                             "--seed-namespace", "s", "--output", str(self.base / "p"),
                             "--frozen-approval", self.frozen["path"], "--frozen-approval-sha256", sha], passrate.main)):
            with self.subTest(argv=argv[0]), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as stop:
                main(argv)
            self.assertEqual(stop.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
