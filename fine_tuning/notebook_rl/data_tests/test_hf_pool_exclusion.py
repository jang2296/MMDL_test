"""CPU-only tests for hf_pool_exclusion: authored fixtures, no GPU, network or benchmark data."""
import dataclasses
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from fine_tuning.notebook_rl import contamination as c
from fine_tuning.notebook_rl import hf_pool_exclusion as hx
from fine_tuning.notebook_rl.adapters import hf_common as hf
from fine_tuning.notebook_rl.data_tests.test_hf_pool import SPECS, cand, family_keys, png
from fine_tuning.notebook_rl.dataset import SPLITS, build_splits
from fine_tuning.notebook_rl.generators import common as gc
from fine_tuning.notebook_rl.schema import ActorItem, VerifierItem

GEN_REVISION = "c" * 64


def gen_problem(split: str, source: str = "gen-fake-v1", revision: str = GEN_REVISION, **lineage_changes):
    """A generators.common.build record as generate_pool leaves it (64-hex code revision), optionally corrupted."""
    style = "shift" if split == "style_sealed" else "default"
    k = 0
    while not gc.matches_split(gc.structure_hash({"k": k}), split, style):
        k += 1
    problem = gc.build(source, "fake", 0, split, "plain", "Generated question?", ("x", "y"), "mcq", "A", {}, {}, {},
                       (), (), ("printed",), {"k": k}, "tmpl", [png(f"{source}-{split}")], source, 0, style)
    lineage = dataclasses.replace(problem.lineage, source_revision=revision, **lineage_changes)
    return dataclasses.replace(problem, lineage=lineage)


def make_build(tmp: Path, spec_overrides: dict | None = None, gen=None) -> Path:
    """One hf-fake-a problem per split (plus gen(split) when given); spec_overrides replaces SourceSpec fields
    for the build only."""
    spec = dataclasses.replace(SPECS["hf-fake-a"], **(spec_overrides or {}))
    keys = family_keys("hf-fake-a", 1)
    pools = {}
    for i, split in enumerate(SPLITS):
        pools[split] = [hf.to_problem(cand("hf-fake-a", f"n-{split}", "Biology", keys[split][0]), spec, index=i)]
        if gen is not None:
            pools[split].append(gen(split))
    build_splits(tmp / "build", pools)
    return tmp / "build"


class HfPoolExclusionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name)

    def test_registry_has_exact_source_fields(self):
        build = make_build(self.tmp)
        reg = self.tmp / "reg"
        hx.run_registry(build, reg, specs={"hf-fake-a": SPECS["hf-fake-a"]})
        registry = json.loads((reg / "source_registry.json").read_bytes())
        self.assertNotIn("scanner_policy", registry)
        self.assertEqual(registry["sources"], {"hf-fake-a": {
            "rights_approved": True, "source_known": True, "benchmark_derived": False, "model_touched": False,
            "label_provenance": "HUMAN", "source_revision": "a" * 40, "license": "cc-by-4.0",
            "upstream_origin": "authored annotators", "parents": [], "upstream_split": "train"}})
        self.assertFalse(registry["training_allowed"])
        receipt = json.loads((reg / "receipt.json").read_bytes())
        self.assertEqual(receipt["outputs"]["source_registry"]["file_sha256"],
                         hashlib.sha256((reg / "source_registry.json").read_bytes()).hexdigest())

    def test_registry_rejects_spec_mismatch_and_unknown_source(self):
        build = make_build(self.tmp)
        changed = dataclasses.replace(SPECS["hf-fake-a"], license="cc-by-sa-4.0")
        with self.assertRaises(ValueError):
            hx.run_registry(build, self.tmp / "mismatch", specs={"hf-fake-a": changed})
        self.assertFalse((self.tmp / "mismatch").exists())
        with self.assertRaises(ValueError):
            hx.run_registry(build, self.tmp / "unknown", specs={})
        self.assertFalse((self.tmp / "unknown").exists())

    def test_generator_source_registered_from_its_lineage(self):
        build = make_build(self.tmp, gen=gen_problem)
        reg = self.tmp / "reg"
        receipt = hx.run_registry(build, reg, specs={"hf-fake-a": SPECS["hf-fake-a"]})
        registry = json.loads((reg / "source_registry.json").read_bytes())
        self.assertEqual(sorted(registry["sources"]), ["gen-fake-v1", "hf-fake-a"])
        self.assertEqual(registry["sources"]["gen-fake-v1"], {
            "rights_approved": True, "source_known": True, "benchmark_derived": False, "model_touched": False,
            "label_provenance": "PROGRAM", "source_revision": GEN_REVISION, "license": "project-authored",
            "upstream_origin": "procedural", "parents": [], "upstream_split": "generated"})
        self.assertEqual(registry["sources"]["hf-fake-a"]["upstream_split"], "train")
        self.assertNotIn("scanner_policy", registry)
        self.assertEqual(receipt["sources"], ["gen-fake-v1", "hf-fake-a"])
        count = 0
        for split in SPLITS:
            lineages = hx.vault.read_lineage(build / split)
            for actor in hx.vault.read_actors(build / split):
                self.assertEqual(c._source_checks(actor, lineages[actor.problem_id], registry), [])
                count += 1
        self.assertEqual(count, 2 * len(SPLITS))
        for name in ("gen_stage.py", "generators/common.py", "generators/circuit.py", "hf_pool.py"):
            self.assertIn(name, receipt["code_sha256"])
        self.assertEqual(receipt["code_sha256"]["generators/common.py"], hx._sha_file(hx._HERE / "generators/common.py"))

    def test_registry_rejects_inconsistent_or_non_generator_lineage(self):
        def revision(split):
            return gen_problem(split, revision="d" * 64 if split == "dev" else GEN_REVISION)

        cases = {
            "inconsistent source_revision": revision,
            "unpinned source_revision": lambda split: gen_problem(split, revision="gen-fake-v1"),
            "non-gen source": lambda split: gen_problem(split, source="zzz-fake-v1"),
            "license": lambda split: gen_problem(split, license="cc-by-4.0"),
            "upstream_origin": lambda split: gen_problem(split, upstream_origin="authored annotators"),
            "label_provenance": lambda split: gen_problem(split, label_provenance="HUMAN"),
        }
        for name, gen in cases.items():
            with self.subTest(name):
                build = make_build(self.tmp / name.replace(" ", "_"), gen=gen)
                out = self.tmp / f"reg-{name.replace(' ', '_')}"
                with self.assertRaises(ValueError):
                    hx.run_registry(build, out, specs={"hf-fake-a": SPECS["hf-fake-a"]})
                self.assertFalse(out.exists())

    def test_refuses_existing_output(self):
        build = make_build(self.tmp)
        out = self.tmp / "reg"
        out.mkdir()
        (out / "keep.txt").write_text("keep")
        argv = ["registry", "--build", str(build), "--output", str(out)]
        with redirect_stdout(io.StringIO()) as printed:
            code = hx.main(argv)  # real adapter registry: hf-fake-a is not a real source
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(printed.getvalue())["status"], "REFUSED")
        # Same call with hf-fake-a registered, so only the existing-output check can refuse it.
        known = {"hf-fake-a": (SPECS["hf-fake-a"], None)}
        with mock.patch("fine_tuning.notebook_rl.hf_pool.default_registry", return_value=known):
            with redirect_stdout(io.StringIO()) as printed:
                code = hx.main(argv)
        self.assertEqual(code, 2)
        self.assertIn("already exists", json.loads(printed.getvalue())["error"])
        self.assertEqual((out / "keep.txt").read_text(), "keep")
        self.assertFalse((out / "source_registry.json").exists())

    def test_gpu_stage_refused_without_allow_gpu_and_when_gpu_busy(self):
        missing = self.tmp / "missing"
        out = self.tmp / "engines"
        base = ["engines", "--build", str(missing), "--index", str(missing), "--raw-images", str(missing),
                "--calibration", str(missing), "--reuse-cache-root", str(missing), "--output", str(out),
                "--stages", "sscd", "--sscd-artifact", str(missing)]
        with mock.patch("fine_tuning.notebook_rl.hf_pool_exclusion.subprocess.run") as run:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(hx.main(base), 2)
                run.assert_not_called()
                self.assertFalse(out.exists())

                run.return_value = mock.Mock(stdout="4242, python\n")
                self.assertEqual(hx.main(base + ["--allow-gpu"]), 2)
                run.assert_called_once()
                self.assertFalse(out.exists())

            run.reset_mock()
            run.side_effect = None
            run.return_value = mock.Mock(stdout="")
            self.assertIsNone(hx.gpu_guard(True))
            run.assert_called_once()

            run.side_effect = OSError()
            with self.assertRaises(ValueError):
                hx.gpu_guard(True)

    def test_copy_build_is_byte_identical_without_root_manifest(self):
        build = make_build(self.tmp)
        fresh = self.tmp / "fresh"
        summary = hx.copy_build(build, fresh)
        self.assertEqual(summary["status"], "SAME_CONTENT_DISTINCT_PATH_BINDING_VERIFIED")
        self.assertFalse((fresh / "manifest.yaml").exists())
        self.assertFalse((fresh / "contamination").exists())
        for split in SPLITS:
            originals = [p.relative_to(build / split) for p in (build / split).rglob("*") if p.is_file()]
            self.assertTrue(originals)
            for rel in originals:
                self.assertEqual((fresh / split / rel).read_bytes(), (build / split / rel).read_bytes())
        with self.assertRaises(ValueError):
            hx.copy_build(build, fresh)

    def test_apply_rebinds_manifest_to_reviewed_receipt_once(self):
        build = make_build(self.tmp)
        asm = self.tmp / "asm"
        hx.copy_build(build, asm / "build-reviewed")
        (asm / "raw-fullscan").mkdir()
        (asm / "raw-fullscan/receipt_v2.json").write_text("{}")
        hx._save(asm / "receipt.json", {"status": "RAW_SCAN_COMPLETE_AWAITING_ADJUDICATION",
                                        "inputs": {"build_manifest": hx._desc(build / "manifest.yaml")}})
        decisions = self.tmp / "decisions.json"
        decisions.write_text("{}")
        _, files, _ = hx._snapshot(asm / "build-reviewed")
        reviewed = {"status": "EXCLUSION_CHECKED_WITH_LIMITS", "input_files": files,
                    "input_files_sha256": c._digest(files), "allowlist": ["x"],
                    "adjudication": {"counts": {"DISTINCT": 1}, "blacklisted_problem_ids": [],
                                     "unresolved_problem_ids": []},
                    **{k: "1" * 64 for k in ("index_sha256", "index_file_sha256", "source_registry_sha256",
                                             "evidence_sha256")}}
        with mock.patch.object(hx.adjudication, "apply", return_value=reviewed) as apply, \
                mock.patch.object(hx.adjudication, "validate", return_value=[]):
            result = hx.run_apply(asm, decisions, self.tmp / "out")
            self.assertTrue(apply.call_args.args[0]["path"].endswith("raw-fullscan/receipt_v2.json"))
            with self.assertRaises(ValueError):  # fresh root already bound: never rewritten
                hx.run_apply(asm, decisions, self.tmp / "out2")
        self.assertFalse((self.tmp / "out2").exists())
        receipt_raw = (asm / "build-reviewed/contamination/receipt_v2.json").read_bytes()
        manifest = json.loads((asm / "build-reviewed/manifest.yaml").read_bytes())
        original = json.loads((build / "manifest.yaml").read_bytes())
        self.assertEqual(manifest, original | {"exclusion_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest()})
        self.assertEqual(json.loads(receipt_raw), reviewed)
        self.assertEqual(result["status"], "EXCLUSION_CHECKED_WITH_LIMITS")
        rebind = json.loads((self.tmp / "out/metadata_rebind.json").read_bytes())
        self.assertEqual(rebind["kind"], "METADATA_REBIND_IDENTICAL_INPUT_BYTES")
        self.assertEqual(set(rebind["old_to_new_files"].values()), set(files))

    def test_review_packet_redacts_answers(self):
        ref = {"dataset_id": "MMMU/MMMU", "configuration": "Art", "split": "validation",
               "id": "validation_Art_1", "file_sha256": "f" * 64}
        key = "MMMU/MMMU|Art|validation|validation_Art_1"
        index_data = {"images": {"a" * 64: {}}, "records": [{
            "ref": ref,
            "texts": {"question": ["Bench question?"], "choices": ["red", "blue"],
                      "explanation": ["EXPLAIN_SECRET the answer is B"]},
            "native_texts": {"explanation": "EXPLAIN_SECRET", "options": "['red','blue']",
                             "question": "Bench question?"},
            "answer": "ANSWER_SECRET", "image_sha256": ["a" * 64], "image_fields": ["image_1"]}]}
        actor = ActorItem("pid1", "hf-fake-a", "fam", "plain", "New question?", ("x", "y"), "mcq",
                          ("hf-fake-a/train/0.png",), ("b" * 64,))
        verifier = VerifierItem("pid1", "mcq", "GOLD_SECRET", {"alt": "ACCEPT_SECRET"}, ("x", "y"),
                                {}, {}, (), (), ())
        items = {"pid1": {"split": "train", "actor": actor, "verifier": verifier}}
        text_key = c._digest(["MMMU/MMMU", "Art", "validation", "validation_Art_1", "explanation", 0])
        text = {"problem_id": "pid1", "stage": "text_ngrams", "field": "question",
                "reason": "NEAR_TEXT_5GRAM", "action": "QUARANTINE", "ref": ref}
        image = {"problem_id": "pid1", "stage": "image_exact", "reason": "EXACT_RAW_OR_PIXEL_IMAGE",
                 "action": "QUARANTINE", "ref": {"a" * 64: [{**ref, "image_field": "image_1"}]}}
        siglip = {"problem_id": "pid1", "stage": "siglip", "reason": "SIGLIP_IMAGE_TO_TEXT",
                  "action": "REVIEW", "ref": text_key, "cosine": 0.95}
        receipt = {"index_sha256": "1" * 64, "input_files_sha256": "2" * 64,
                   "candidates": [text, text, image, siglip]}
        packet = hx.review_packet(receipt, {"path": "/x/receipt_v2.json", "file_sha256": "3" * 64},
                                  index_data, items, Path("/b"), Path("/raw"))

        dumped = json.dumps(packet)
        for secret in ("EXPLAIN_SECRET", "ANSWER_SECRET", "GOLD_SECRET", "ACCEPT_SECRET"):
            self.assertNotIn(secret, dumped)
        self.assertIn("Bench question?", dumped)
        self.assertIn("New question?", dumped)
        self.assertEqual(packet["counts"]["unique_candidates"], 3)
        by_stage = {p["stage"]: p for p in packet["candidates"]}
        self.assertEqual(by_stage["text_ngrams"]["occurrences"], 2)
        self.assertTrue(by_stage["text_ngrams"]["releasable"])
        self.assertFalse(by_stage["image_exact"]["releasable"])
        self.assertEqual(by_stage["siglip"]["benchmark_records"], [key])
        self.assertEqual(by_stage["siglip"]["benchmark_text_fields"], ["explanation"])
        self.assertEqual(packet["benchmark_records"][key]["image_paths"], ["/raw/" + "a" * 64])
        self.assertTrue(packet["benchmark_records"][key]["explanation_withheld"])
        self.assertEqual(packet["items"]["pid1"]["image_paths"], ["/b/images/hf-fake-a/train/0.png"])
        self.assertEqual(packet["review_template"]["raw_receipt_sha256"], "3" * 64)
        self.assertFalse(packet["review_template"]["provenance"]["human_verified"])


if __name__ == "__main__":
    unittest.main()
