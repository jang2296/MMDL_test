"""pool-proof-selection-v1: pass-rate-selected training set and same-source per-subject sentinels on a pod.

Authored CPU fixtures only: a one-shard union vault with authored subjects, a stand-in ai_reviewed_local approval
output, scripted engines and the tiny CPU Qwen/PEFT policy. Not a real exclusion approval, not a pod run and no
evidence about any model's quality.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fine_tuning.notebook_rl import (approval as a, local_data, local_pool_eval, local_profile, local_run, passrate,
                                     pool_freeze as pf)
from fine_tuning.notebook_rl.dataset import build_splits
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.tests import test_notebook_rl_pool_local as pool_local
from fine_tuning.tests.test_notebook_rl_frozen_approval import union_validation, unionize
from fine_tuning.tests.test_notebook_rl_passrate import ScriptedEngine as PassrateEngine
from fine_tuning.tests.test_notebook_rl_pool_local import (EVAL_REPLIES, RIGHT, SLOTLESS_SCRIPT, WRONG, _sha, _write_json,
                                                           allowlist_receipt, eval_replies, hf_problem, v2_args,
                                                           v2_protocol, write_request)

TRAIN_SOURCES = ("hf-fixture-a", "hf-fixture-b", "hf-fixture-c", "hf-fixture-d")
SUBJECTS = {"hf-fixture-a": "Biology", "hf-fixture-b": "Chemistry", "hf-fixture-c": "Biology",
            "hf-fixture-d": "Physics", "hf-fixture-e": "Geography", "hf-chebi": "Chemistry"}
BLOCKED = {"hf-fixture-d": ["FAILED_VERDICT render_ok fixture"]}  # the approval makes every hf-fixture-d item ineligible
MEASURE = {"train_problems": 12, "problems_per_update": 4, "max_updates": 3}  # all-eligible measuring preparation
PROOF = {"train_problems": 8, "problems_per_update": 4, "max_updates": 2}
SENTINEL_REPLIES = {"hf-fixture-a": EVAL_REPLIES["hf-fixture-e"], "hf-fixture-b": EVAL_REPLIES["hf-fixture-f"],
                    "hf-fixture-c": EVAL_REPLIES["hf-fixture-f"], "hf-fixture-e": EVAL_REPLIES["hf-fixture-e"]}


def proof_build(root: Path) -> None:
    """Train: four HF sources x 4, a same-structure twin of the first item and an excluded-source item.
    Held out: dev/sealed items of the training sources (same sources), of hf-fixture-e, of an excluded source and of
    the approval-blocked source; one style_sealed item (never a sentinel split). Every item is single-image."""
    train = [hf_problem(source, "train", 10 * i + k) for i, source in enumerate(TRAIN_SOURCES) for k in range(4)]
    twin = hf_problem("hf-fixture-a", "train", 9)
    train += [replace(twin, lineage=replace(twin.lineage, structure_hash=train[0].lineage.structure_hash)),
              hf_problem("hf-chebi", "train", 60)]
    build_splits(root, {
        "train": train,
        "dev": [hf_problem("hf-fixture-a", "dev", 70), hf_problem("hf-fixture-a", "dev", 71),
                hf_problem("hf-fixture-b", "dev", 72), hf_problem("hf-fixture-e", "dev", 73),
                hf_problem("hf-chebi", "dev", 74)],
        "sealed": [hf_problem("hf-fixture-c", "sealed", 80), hf_problem("hf-fixture-a", "sealed", 81),
                   hf_problem("hf-fixture-d", "sealed", 82)],
        "style_sealed": [hf_problem("hf-fixture-b", "style_sealed", 90)]})
    allowlist_receipt(root)


def add_subjects(root: Path, dataset: Path) -> None:
    """Authored subject per frozen-manifest item (pool_freeze carries one), rebinding the freeze binding and union."""
    manifest = json.loads(dataset.read_bytes())
    for item in manifest["items"]:
        item["subject"] = SUBJECTS[item["problem_id"].split(":")[0]]
    manifest_sha = _write_json(dataset, manifest)
    binding = json.loads((root / "freeze_binding.json").read_bytes()) | {"dataset_manifest_sha256": manifest_sha}
    union_path = root / "contamination" / pf.UNION_FILE
    union = json.loads(union_path.read_bytes()) | {
        "dataset_manifest": {"path": str(dataset), "file_sha256": manifest_sha},
        "freeze_binding_sha256": _write_json(root / "freeze_binding.json", binding)}
    union["union_receipt_sha256"] = a._hash({k: v for k, v in union.items() if k != "union_receipt_sha256"})
    _write_json(union_path, union)
    pf.read_union(root)
    assert pf.verify(dataset, root) == []


def validation(root, **_) -> dict:
    return union_validation(root, blocked=BLOCKED)


def refuse(*args, **kwargs):
    raise AssertionError("ai_reviewed_local approval re-run on the pod")


def drive_proof(config_path: str) -> None:
    """pool_local's v2 stage boundary, except that any approval call fails: pod stages verify hashes only."""
    with patch.object(pool_local, "fake_validation", refuse):
        pool_local.drive_v2(config_path)


class ProofFixture(unittest.TestCase):
    """Local: union vault + exported approval output. Then the 'pod': every shard root, registry and evidence gone."""

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="notebook-proof-")
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / "vault"
        proof_build(self.root)
        self.union = unionize(self.base, self.root)
        add_subjects(self.root, self.union["dataset"])
        self.request = write_request(self.base, self.root)
        self.public = local_data.public_pool(self.root)
        self.prompts = local_profile.local_prompts(v2_protocol(PROOF))
        exported = self.base / "export" / "approval.json"
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=validation), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(local_run.main(["export-approval", "--build", str(self.root), "--approval",
                                             str(self.request), "--output", str(exported)]), 0)
        self.frozen = {"path": str(exported), "file_sha256": _sha(exported.read_bytes())}
        self.assertEqual(json.loads(printed.getvalue())["file_sha256"], self.frozen["file_sha256"])
        self.frozen_args = ["--frozen-approval", self.frozen["path"], "--frozen-approval-sha256", self.frozen["file_sha256"]]
        shutil.rmtree(self.union["shard"])
        for path in self.union["inputs"]:
            path.unlink()
        self.eligible = local_run.frozen_approval_output(self.root, self.frozen)[1]

    def source(self, pid: str) -> str:
        return self.public[pid][0].source

    def pod(self, main, argv: list[str]) -> int:
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=refuse), \
                contextlib.redirect_stdout(io.StringIO()):
            return main(argv)

    def propose(self, name: str, scale: dict, *extra: str) -> Path:
        output = self.base / f"{name}.json"
        self.assertEqual(self.pod(local_run.main, ["propose-proof", "--build", str(self.root), *self.frozen_args,
                                                   "--sentinels-per-subject", "2", *v2_args(scale), *extra,
                                                   "--output", str(output)]), 0)
        return output

    def prepare(self, name: str, rule: Path, scale: dict) -> Path:
        self.assertEqual(self.pod(local_run.main, ["prepare-pool", "--build", str(self.root), "--approval",
                                                   str(self.request), *self.frozen_args, "--rule", str(rule),
                                                   *v2_args(scale), "--output", str(self.base / name)]), 0)
        return self.base / name / "preparation.json"

    def prepare_rule(self, name: str, rule: dict, scale: dict = PROOF) -> dict:
        """Library prepare-pool (frozen) of an in-memory rule; refusals raise."""
        path = self.base / f"{name}-rule.json"
        path.write_text(json.dumps(rule))
        with patch.object(a, "validate_for_ai_reviewed_local", side_effect=refuse):
            return local_data.prepare_pool_data(self.root, self.request, path, v2_protocol(scale), self.base / name,
                                                frozen_approval=self.frozen)

    def measure_report(self, name: str, ids: list[str], rights: dict[str, int], weights: dict | None = None) -> Path:
        """In-process build-root measurement with a scripted engine (rights[pid] correct of k=4, default 2)."""
        output = self.base / name

        def script(request):
            right = rights.get(request.request_id.split(":v0")[0], 2)
            return [RIGHT] * right + [WRONG] * (request.n - right)
        passrate.measure(self.root, "train", ids, protocol=v2_protocol(MEASURE), engine=PassrateEngine(script), k=4,
                         seed_namespace="pre", output=output, weights=weights)
        return output


class ProofChainTests(ProofFixture):
    """export-approval (local) -> on the pod: propose-proof all-eligible -> prepare-pool (frozen) -> passrate measure
    of that preparation (frozen) -> select -> propose-proof from the selection -> prepare-pool (frozen) -> minimal ->
    reload -> train N=8, K=4 -> reload -> evaluate-pool F0 and FT on the per-subject sentinels."""

    def run_stage(self, name: str, module: str, argv: list[str], config: dict) -> None:
        path = self.base / f"{name}-config.json"
        path.write_text(json.dumps(config | {"argv": argv, "module": module}))
        done = subprocess.run([sys.executable, "-B", "-c", "import sys; from fine_tuning.tests.test_notebook_rl_proof_selection "
                               "import drive_proof; drive_proof(sys.argv[1])", str(path)],
                              capture_output=True, text=True, timeout=1800, env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
                              check=False)
        self.assertEqual(done.returncode, 0, done.stdout[-3000:] + done.stderr[-6000:])

    def measure_cli(self, preparation: Path, rights: dict[str, int]) -> Path:
        """passrate measure --preparation (frozen) through _cli_measure with the model/engine/guards patched."""
        output = self.base / "passrate"
        manifest = json.loads((self.root / "manifest.yaml").read_text())

        class Service:
            def __init__(self, root, **_):
                self.verifier = Verifier(root)
                self.metadata = {"vault_manifest_sha256": manifest["split_manifests"]["train"]}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def grade(self, *args, **kwargs):
                return self.verifier.grade(*args, **kwargs)

        def script(request):
            right = rights.get(request.request_id.split(":v0")[0], 2)
            return [RIGHT] * right + [WRONG] * (request.n - right)
        engine = PassrateEngine(script)
        with ExitStack() as stack:
            for target, value in (("fine_tuning.notebook_rl.hf_pool_exclusion.gpu_guard", lambda *_: None),
                                  ("fine_tuning.notebook_rl.passrate._install_guards", lambda *_, **__: []),
                                  ("fine_tuning.notebook_rl.verifier_service.VerifierService", Service),
                                  ("fine_tuning.notebook_rl.quantized_policy.build_quantized_policy", lambda *_, **__: object()),
                                  ("fine_tuning.notebook_rl.scale.make_engines", lambda *_: (engine, engine))):
                stack.enter_context(patch(target, value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(self.pod(passrate.main, ["measure", "--preparation", str(preparation), "--model-path",
                                                      str(self.base / "model"), "--k", "4", "--seed-namespace", "pre",
                                                      "--output", str(output), *v2_args(MEASURE), *self.frozen_args]), 0)
        return output

    def test_pod_chain_from_passrate_selection_to_sentinel_evaluation(self) -> None:
        from fine_tuning.notebook_rl.local_train import REPAIR_FUNCTIONAL_EXPLORATION
        from fine_tuning.tests.test_notebook_rl_checkpoint import fixture as tiny_fixture

        # All-eligible measuring preparation: 12 eligible distinct training items (d blocked, chebi excluded, twin merged).
        measuring = json.loads(self.propose("measure-rule", MEASURE).read_text())
        self.assertEqual(measuring["selection_source"]["kind"], "all-eligible")
        self.assertEqual({self.source(pid) for pid in measuring["fixed"]}, set(TRAIN_SOURCES[:3]))
        p0 = self.prepare("measure-prepared", self.base / "measure-rule.json", MEASURE)
        fixed0 = measuring["fixed"]
        rights = {fixed0[2]: 4, fixed0[5]: 4, fixed0[7]: 0}  # ALL_CORRECT, ALL_CORRECT, ALL_WRONG; the rest 2 of 4
        report = self.measure_cli(p0, rights)
        self.assertEqual([json.loads(line)["problem_id"] for line in (report / "records.jsonl").read_text().splitlines()],
                         fixed0)
        selection = self.base / "selection.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(passrate.main(["select", "--report", str(report), "--min-pass", "1", "--max-pass", "3",
                                            "--output", str(selection)]), 0)
        rule_path = self.propose("proof-rule", PROOF, "--passrate-selection", str(selection),
                                 "--passrate-report", str(report))
        rule = json.loads(rule_path.read_text())
        kept = [pid for pid in fixed0 if pid not in rights]
        self.assertEqual((rule["version"], rule["fixed"], rule["minimal"]),
                         (local_data.POOL_RULE_VERSION_PROOF, kept[:8], kept[:2]))
        self.assertEqual(rule["selection_source"] | {"path": None, "report": None},
                         {"kind": "passrate-selection", "path": None, "file_sha256": _sha(selection.read_bytes()),
                          "report": None, "report_sha256": json.loads(selection.read_text())["report_sha256"],
                          "excluded_ids": [], "dropped_ids": kept[8:]})
        sentinels = rule["sentinels"]
        self.assertEqual({subject: len(ids) for subject, ids in sentinels.items()},
                         {"Biology": 2, "Chemistry": 1, "Geography": 1})
        flat = [pid for subject in sorted(sentinels) for pid in sentinels[subject]]
        self.assertTrue({self.source(pid) for pid in flat} & {self.source(pid) for pid in rule["fixed"]})  # same sources
        self.assertEqual({self.public[pid][1].split for pid in flat} - {"dev", "sealed"}, set())
        self.assertFalse({self.public[pid][1].structure_hash for pid in flat}
                         & {self.public[pid][1].structure_hash for pid in rule["fixed"]})

        preparation = self.prepare("prepared", rule_path, PROOF)
        prepared = json.loads(preparation.read_text())
        self.assertEqual((prepared["format"], prepared["status"], prepared["validated_data"]["file_sha256"]),
                         (local_data.POOL_FORMAT_PROOF, local_data.POOL_STATUS, self.frozen["file_sha256"]))
        self.assertEqual(prepared["data_verification"]["mode"], local_run.FROZEN_APPROVAL_VERIFY)
        selected = json.loads(Path(prepared["selection"]["path"]).read_text())
        self.assertEqual((selected["evaluation_ids"], selected["sentinel_ids"]), (flat, sentinels))
        self.assertEqual({selected["subjects"][pid] for pid in flat}, {"Biology", "Chemistry", "Geography"})

        model_path, *_ = tiny_fixture(self.base / "model")
        (model_path / "nf4_base.json").write_text('{"fixture_only":true}')
        probe_ids = [rule["minimal"][0]]
        config = {"build_root": str(self.root), "minimal_ids": rule["minimal"], "probe_ids": probe_ids,
                  "record": str(self.base / "collector.jsonl"),
                  "replies": {pid: SLOTLESS_SCRIPT for pid in rule["fixed"]} | eval_replies(flat, SENTINEL_REPLIES)}
        capture, reload, trained, restored = (self.base / name for name in ("minimal", "reload", "train", "train-reload"))
        cache = ["--cache-check", str(self.base / "authored-cache-check")]
        self.run_stage("minimal", "local_run", ["minimal", "--preparation", str(preparation), "--model-path",
            str(model_path), "--standalone-base", *cache, "--output", str(capture), "--repair-notebook",
            "--context-limit", "24576", *v2_args(PROOF), *self.frozen_args], config)
        result = json.loads((capture / "result.json").read_text())
        self.assertEqual((result["status"], result["attempted_ids"]), ("UPDATED_SAVED_PENDING_FRESH_RELOAD", rule["minimal"]))
        self.assertEqual(json.loads((capture / "launch.json").read_text())["data_verification"]["approval_sha256"],
                         self.frozen["file_sha256"])
        self.run_stage("reload", "local_run", ["reload", "--capture", str(capture), "--output", str(reload)], config)
        self.assertEqual(json.loads((reload / "result.json").read_text())["status"], "FRESH_PROCESS_RELOAD_VERIFIED")
        self.run_stage("train", "local_train", ["train", "--preparation", str(preparation), "--model-path", str(model_path),
            *cache, "--minimal", str(capture), "--minimal-reload", str(reload), "--output", str(trained), "--arm",
            "notebook", "--admission", REPAIR_FUNCTIONAL_EXPLORATION, *v2_args(), *self.frozen_args], config)
        result = json.loads((trained / "result.json").read_text())
        self.assertEqual((result["status"], result["optimizer_updates"], result["attempted_ids"]),
                         ("LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD", 2, rule["fixed"]))
        self.assertEqual(json.loads((trained / "launch.json").read_text())["execution_contract"]["pair_ids"],
                         [rule["fixed"][:4], rule["fixed"][4:]])
        self.run_stage("train-reload", "local_train", ["reload", "--capture", str(trained), "--output", str(restored)], config)
        reloaded = json.loads((restored / "result.json").read_text())
        self.assertEqual((reloaded["status"], reloaded["step"]), ("LOCAL_TEN_FRESH_PROCESS_RELOAD_VERIFIED", 2))
        records = {}
        for condition in ("F0", "FT"):
            evaluated = self.base / f"evaluate-{condition}"
            self.run_stage(f"evaluate-{condition}", "local_eval", ["evaluate-pool", "--capture", str(trained),
                "--preparation", str(preparation), "--reload-result", str(restored / "result.json"),
                "--minimal-capture", str(capture), "--minimal-reload", str(reload), "--condition", condition,
                "--arm", "notebook", "--output", str(evaluated)], config)
            evaluation = json.loads((evaluated / "result.json").read_text())
            self.assertEqual((evaluation["status"], evaluation["recovery"]["problem_ids"]),
                             (local_pool_eval.COMPLETE, flat), condition)
            records[condition] = evaluation["recovery"]
            self.assertEqual(set(evaluation["recovery"]["metrics"]["by_subject"]), {"Biology", "Chemistry", "Geography"})
        self.assertEqual(records["F0"]["policy_version"], 0)
        self.assertEqual(records["FT"]["policy_version"], 2)
        rows = [json.loads(line) for line in (self.base / "collector.jsonl").read_text().splitlines()]
        self.assertEqual({pid for row in rows if row["kind"] == "rollout" for pid in row["ids"]}, set(rule["fixed"]))


class ProofRefusalTests(ProofFixture):
    def all_eligible(self, scale: dict = PROOF, **options) -> dict:
        return local_data.propose_proof_rule(self.root, self.prompts, eligible=self.eligible,
                                             sizes=(scale["train_problems"], scale["problems_per_update"]),
                                             sentinels_per_subject=options.pop("sentinels_per_subject", 2), **options)

    def selection(self, weights: dict | None = None, name: str = "report") -> tuple[Path, Path, dict]:
        """Build-root measurement of every slotless-source train item (approval-blocked hf-fixture-d and the twin
        included; hf-chebi is no slotless source of the protocol, so it has no prompt) and its 1..3 of 4 selection.
        `weights`: what the measurement binds (default F0)."""
        ids = [pid for pid, (actor, lineage) in self.public.items()
               if lineage.split == "train" and actor.source in pool_local.V2_SLOTLESS]
        families = Counter(self.public[pid][1].structure_hash for pid in ids)
        plain_a = [pid for pid in ids if self.source(pid) == "hf-fixture-a" and families[self.public[pid][1].structure_hash] == 1]
        report = self.measure_report(name, ids, {local_data._order(plain_a, 3407)[0]: 4}, weights)  # one ALL_CORRECT item
        output = self.base / f"{name}-selection.json"
        return output, report, passrate.select(report, {"version": passrate.RULES_VERSION, "min_pass": 1, "max_pass": 3,
                                                        "max_blocking": 0}, output)

    def test_build_root_selection_is_intersected_with_eligibility_and_recorded(self) -> None:
        path, report, selected = self.selection()
        rule = self.all_eligible(passrate=(path, report))
        source = rule["selection_source"]
        families = Counter(lineage.structure_hash for _, lineage in self.public.values())
        twins = {pid for pid, (_, lineage) in self.public.items() if families[lineage.structure_hash] == 2}
        self.assertEqual(len(twins), 2)
        ineligible = {pid for pid in selected["selected_ids"] if self.source(pid) == "hf-fixture-d"}
        self.assertEqual(len(ineligible), 4)
        # Approval-ineligible items and the later twin in hash order are recorded, never fixed or dropped.
        self.assertTrue(ineligible < set(source["excluded_ids"]))
        self.assertEqual(len(set(source["excluded_ids"]) & twins), 1)
        self.assertEqual(len(rule["fixed"]) + len(source["dropped_ids"]) + len(source["excluded_ids"]),
                         len(selected["selected_ids"]))
        self.assertEqual(rule["fixed"] + source["dropped_ids"], local_data._order(rule["fixed"] + source["dropped_ids"], 3407))
        self.assertEqual(self.all_eligible(passrate=(path, report)), rule)  # deterministic

    def test_a_checkpoint_measured_selection_is_refused_for_training_and_f0_is_accepted(self) -> None:
        f0_path, f0_report, _ = self.selection()
        weights = {"checkpoint": "/ckpt/step-2", "trainable_sha256": "c" * 64}
        ft_path, ft_report, ft_selected = self.selection(weights, "ft-report")
        self.assertEqual(ft_selected["policy_version"], 0)  # the version alone cannot tell the measured weights apart
        with self.assertRaisesRegex(ValueError, "measured at F0.*only feed hard_negatives"):
            self.all_eligible(passrate=(ft_path, ft_report))
        self.assertEqual(self.all_eligible(passrate=(f0_path, f0_report))["selection_source"]["kind"], "passrate-selection")
        # Relabeling the report as F0 breaks the selection's report hash; re-selecting from the edited report is the
        # only way through, and that is a different (visible, hash-bound) report, not the checkpoint one.
        summary = ft_report / "summary.json"
        summary.chmod(0o600)
        summary.write_text(json.dumps(json.loads(summary.read_text()) | {"weights": dict(passrate.F0_WEIGHTS)}))
        with self.assertRaisesRegex(ValueError, "differs from the one its measurement report derives"):
            self.all_eligible(passrate=(ft_path, ft_report))
        # Checkpoint-measured reports still feed hard negatives (the post-training failures to re-train on).
        fresh_path, fresh_report, _ = self.selection(weights, "ft-report-2")
        hard = self.base / "ft-hard.json"
        self.assertEqual(passrate.hard_negatives(fresh_report, max_pass=3, output=hard)["kind"], "HARD_NEGATIVES")
        self.assertEqual(passrate.verify_selection(hard, fresh_report)[1]["weights"], weights)
        self.assertTrue(fresh_path.exists())

    def test_ineligible_ids_and_n_mismatch_are_refused(self) -> None:
        rule = self.all_eligible()
        blocked_train = next(pid for pid in local_data._order(self.public, 3407)
                             if self.source(pid) == "hf-fixture-d" and self.public[pid][1].split == "train")
        blocked_sealed = next(pid for pid in self.public if self.source(pid) == "hf-fixture-d"
                              and self.public[pid][1].split == "sealed")
        fixed = list(rule["fixed"])
        fixed[3] = blocked_train
        sentinels = {**rule["sentinels"], "Physics": [blocked_sealed]}
        for name, changed in (("fixed", rule | {"fixed": fixed}), ("sentinel", rule | {"sentinels": sentinels})):
            with self.subTest(name), self.assertRaisesRegex(ValueError, "exclusion-allowlisted"):
                self.prepare_rule(f"ineligible-{name}", changed)
        with self.assertRaisesRegex(ValueError, "fewer than train_problems 16"):
            self.all_eligible({"train_problems": 16, "problems_per_update": 4})
        with self.assertRaisesRegex(ValueError, "the bound scale train_problems is 4"):
            self.prepare_rule("n-mismatch", rule, {"train_problems": 4, "problems_per_update": 4, "max_updates": 1})
        dropped = rule["selection_source"]["dropped_ids"]
        swapped = rule | {"fixed": [*rule["fixed"][:7], dropped[0]],
                          "selection_source": {**rule["selection_source"], "dropped_ids": [rule["fixed"][7], *dropped[1:]]}}
        with self.assertRaisesRegex(ValueError, "hash order"):
            self.prepare_rule("swapped", swapped)
        style = next(pid for pid in self.public if self.public[pid][1].split == "style_sealed")
        with self.assertRaisesRegex(ValueError, "sentinels dev/sealed"):
            self.prepare_rule("style", rule | {"sentinels": {**rule["sentinels"], "Chemistry": [style]}})
        with self.assertRaisesRegex(ValueError, "own subject"):
            self.prepare_rule("subject", rule | {"sentinels": {"Physics": rule["sentinels"]["Biology"],
                                                              **{k: v for k, v in rule["sentinels"].items() if k != "Biology"}}})
        # A well-formed but different sentinel choice is not the label-blind derivation.
        fewer = rule | {"sentinels": {**rule["sentinels"], "Biology": rule["sentinels"]["Biology"][:1]}}
        with self.assertRaisesRegex(ValueError, "label-blind derivation"):
            self.prepare_rule("fewer", fewer)
        self.assertEqual(self.prepare_rule("accepted", rule)["status"], local_data.POOL_STATUS)

    def test_tampered_selection_or_report_is_refused(self) -> None:
        path, report, selected = self.selection()
        edited = self.base / "edited-selection.json"
        edited.write_text(json.dumps(selected | {"selected_ids": selected["selected_ids"][1:]}))
        with self.assertRaisesRegex(ValueError, "differs from the one its measurement report derives"):
            self.all_eligible(passrate=(edited, report))
        hard = self.base / "hard.json"
        passrate.hard_negatives(report, max_pass=3, output=hard)
        with self.assertRaisesRegex(ValueError, "PASSRATE_SELECTION of this vault"):
            self.all_eligible(passrate=(hard, report))
        rule = self.all_eligible(passrate=(path, report))
        records = report / "records.jsonl"
        saved = records.read_bytes()
        records.chmod(0o600)
        records.write_bytes(saved.replace(b'"passes":2', b'"passes":3', 1))
        with self.assertRaisesRegex(ValueError, "matching records"):
            self.all_eligible(passrate=(path, report))
        records.write_bytes(saved)
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "changed since the rule was proposed"):
            self.prepare_rule("changed-selection", rule)

    def test_pod_without_frozen_flags_needs_a_fresh_approval_and_refuses(self) -> None:
        # The real approval worker: shard roots, registries and evidence are gone on the pod.
        with contextlib.redirect_stderr(io.StringIO()):
            code = local_run.main(["propose-proof", "--build", str(self.root), "--approval", str(self.request),
                                   "--sentinels-per-subject", "2", *v2_args(PROOF), "--output", str(self.base / "fresh.json")])
        self.assertEqual(code, 1)
        self.assertFalse((self.base / "fresh.json").exists())
        rule = self.base / "rule.json"
        rule.write_text(json.dumps(self.all_eligible()))
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            code = local_run.main(["prepare-pool", "--build", str(self.root), "--approval", str(self.request),
                                   "--rule", str(rule), *v2_args(PROOF), "--output", str(self.base / "fresh-prepared")])
        self.assertNotEqual(code, 0)
        prepared = self.base / "fresh-prepared" / "preparation.json"
        self.assertTrue(not prepared.exists() or json.loads(prepared.read_text())["status"] == "BLOCKED")
        # A vault copied elsewhere is not the build the data request names.
        moved = self.base / "moved-vault"
        shutil.copytree(self.root, moved)
        with self.assertRaisesRegex(ValueError, "request for this build"), \
                patch.object(a, "validate_for_ai_reviewed_local", side_effect=refuse):
            local_data.prepare_pool_data(moved, self.request, rule, v2_protocol(PROOF), self.base / "moved",
                                         frozen_approval=self.frozen)
        # The frozen pin must name the exported bytes.
        with self.assertRaisesRegex(ValueError, "artifact bytes differ"):
            local_run.frozen_approval_output(self.root, {**self.frozen, "file_sha256": "0" * 64})
        for argv in (["propose-proof", "--build", str(self.root), "--sentinels-per-subject", "2", *v2_args(PROOF),
                      "--output", str(self.base / "x.json")],
                     ["propose-proof", "--build", str(self.root), *self.frozen_args, "--approval", str(self.request),
                      "--sentinels-per-subject", "2", *v2_args(PROOF), "--output", str(self.base / "x.json")],
                     ["propose-proof", "--build", str(self.root), *self.frozen_args, "--sentinels-per-subject", "2",
                      "--note-contract", "repair-note-compact-v1", "--output", str(self.base / "x.json")],
                     ["propose-proof", "--build", str(self.root), *self.frozen_args, "--sentinels-per-subject", "2",
                      "--passrate-selection", str(rule), *v2_args(PROOF), "--output", str(self.base / "x.json")]):
            with self.subTest(argv=argv[3:6]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                local_run.main(argv)

    def test_rule_versions_stay_separate(self) -> None:
        rule = self.all_eligible()
        sha = rule["build_manifest_sha256"]
        subjects = local_data.proof_subjects(self.root, self.public)
        with self.assertRaisesRegex(ValueError, "pool-local-selection-v2"):
            local_data.check_pool_rule_v2(rule, self.public, self.prompts, sha, sizes=(8, 4))
        with self.assertRaisesRegex(ValueError, "pool-local-selection-v1"):
            local_data.check_pool_rule(rule, self.public, self.prompts, sha)
        for version in (local_data.POOL_RULE_VERSION, local_data.POOL_RULE_VERSION_V2):
            with self.subTest(version), self.assertRaisesRegex(ValueError, "pool-proof-selection-v1"):
                local_data.check_proof_rule(rule | {"version": version}, self.public, self.prompts, sha, sizes=(8, 4),
                                            subjects=subjects)
        self.assertEqual((local_data.POOL_FORMATS[:2], local_run.pool_v2({"format": local_data.POOL_FORMAT_PROOF}),
                          local_run.pool_v2({"format": local_data.POOL_FORMAT})),
                         ((local_data.POOL_FORMAT, local_data.POOL_FORMAT_V2), True, False))


if __name__ == "__main__":
    unittest.main()
