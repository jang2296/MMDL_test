"""Boundaries of the portable source cache and selective native-row reader."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from fine_tuning.error_note import v18_data as data
from fine_tuning.error_note.v18_data_convert import SourceReader


class V18DataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def spec(self, body=b"upstream data"):
        return {"provider": "url", "url": "https://example.org/pinned/data",
                "path": "source.bin", "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}

    def seed(self, spec, body):
        path = data.cache_path(self.root, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return path

    def test_duplicate_files_are_transferred_once_and_warm_cache_has_zero_requests(self):
        body = b"upstream data"
        spec = self.spec(body)
        recipe = {"files": {"a": spec, "same-source-again": dict(spec)}}
        with patch.object(data, "urlopen", return_value=io.BytesIO(body)) as request:
            first = data.fetch(recipe, self.root)
        self.assertEqual((first["requests"], first["received_bytes"], request.call_count), (1, len(body), 1))
        with patch.object(data, "urlopen", side_effect=AssertionError("warm cache retransmitted")):
            second = data.fetch(recipe, self.root)
        self.assertEqual((second["requests"], second["received_bytes"], second["cache_hits"]), (0, 0, 1))

    def test_fetch_plan_hashes_warm_objects_once_and_separates_shared_source_timing(self):
        one, shared = self.spec(b"only one"), self.spec(b"shared")
        self.seed(one, b"only one")
        self.seed(shared, b"shared")
        recipe = {"files": {"one": one, "shared": shared}, "items": [
            {"source": "source-one", "locator": {"file_key": "one", "image_file_key": "shared"}},
            {"source": "source-two", "locator": {"file_key": "shared"}}]}
        stderr = io.StringIO()
        with patch.object(data, "file_sha", wraps=data.file_sha) as hashed, \
                patch("sys.stderr", stderr), \
                patch.object(data, "urlopen", side_effect=AssertionError("warm network")):
            receipt = data.fetch(recipe, self.root)
        plan = json.loads(stderr.getvalue())
        self.assertEqual(hashed.call_count, 2)
        self.assertEqual((plan["files"], plan["verified_cached_files"], plan["verified_cached_bytes"]),
                         (2, 2, len(b"only one") + len(b"shared")))
        self.assertEqual(receipt["per_source"]["source-one"]["files"], 1)
        self.assertNotIn("source-two", receipt["per_source"])
        self.assertEqual(receipt["shared_or_unassigned_file_tasks"][0]["sources"], ["source-one", "source-two"])

    def test_corrupt_cache_is_rejected_offline_then_only_that_object_is_replaced(self):
        good, damaged = self.spec(b"good"), self.spec(b"fixed")
        preserved = self.seed(good, b"good")
        self.seed(damaged, b"wrong")
        recipe = {"files": {"good": good, "damaged": damaged}}
        with patch.object(data, "urlopen", side_effect=AssertionError("offline network")):
            with self.assertRaisesRegex(ValueError, "CACHE_MISSING_OR_CORRUPT"):
                data.fetch(recipe, self.root, offline=True)
        with patch.object(data, "urlopen", return_value=io.BytesIO(b"fixed")) as request:
            receipt = data.fetch(recipe, self.root)
        self.assertEqual((receipt["requests"], request.call_count), (1, 1))
        self.assertEqual(preserved.read_bytes(), b"good")

    def test_mid_stream_timeout_and_reset_are_refetched_not_fatal(self):
        # Local empty-cache prepare 2026-11-18 died after 8.2 GB on one read TimeoutError (no retry for socket errors).
        body = b"upstream data"
        spec = self.spec(body)

        class Dropping(io.BytesIO):
            def __init__(self, error):
                super().__init__(body)
                self.error = error

            def read(self, size=-1):
                raise self.error

        responses = [Dropping(TimeoutError("The read operation timed out")), Dropping(ConnectionResetError()),
                     io.BytesIO(body)]
        with patch.object(data, "urlopen", side_effect=responses), patch.object(data.time, "sleep"):
            result = data.fetch({"files": {"a": spec}}, self.root)
        self.assertEqual((result["requests"], data.cache_path(self.root, spec).read_bytes()), (3, body))
        self.assertEqual([p.name for p in data.cache_path(self.root, spec).parent.iterdir()],
                         [data.cache_path(self.root, spec).name])  # no partial temporary file left
        with patch.object(data, "urlopen", side_effect=[Dropping(TimeoutError()) for _ in range(data.FETCH_ATTEMPTS)]), \
                patch.object(data.time, "sleep"), self.assertRaises(TimeoutError):
            data.fetch({"files": {"b": self.spec(b"other body")}}, self.root)

    def test_bad_source_body_never_commits_and_does_not_leave_partial_file(self):
        spec = self.spec(b"expected")
        with patch.object(data, "urlopen", return_value=io.BytesIO(b"tampered")):
            with self.assertRaisesRegex(ValueError, "SOURCE_HASH_MISMATCH"):
                data.fetch({"files": {"a": spec}}, self.root)
        self.assertEqual(list((self.root / "objects").iterdir()), [])

    def test_oversized_response_is_bounded_and_uncommitted(self):
        spec = self.spec(b"small")
        response = io.BytesIO(b"x" * 100000)
        with patch.object(data, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "SIZE_EXCEEDED"):
                data.fetch({"files": {"a": spec}}, self.root)
        self.assertFalse(data.cache_path(self.root, spec).exists())

    def test_clevr_prefix_is_derived_from_pinned_archive_without_network(self):
        rows = [{"image_index": 0, "text": "a[escaped]"}, {"image_index": 2}, {"image_index": 9}]
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            zipped.writestr("questions.json", json.dumps({"info": {}, "questions": rows}))
        raw_archive = archive.getvalue()
        archive_spec = self.spec(raw_archive)
        self.seed(archive_spec, raw_archive)
        body = (json.dumps({"questions": rows[:2]}, sort_keys=True, separators=(",", ":")) + "\n").encode()
        spec = dict(self.spec(body), provider="zip-json-prefix", member="questions.json",
                    array_key="questions", stop_image_index=3, archive_bytes=len(raw_archive),
                    archive_sha256=archive_spec["sha256"])
        with patch.object(data, "urlopen", side_effect=AssertionError("local archive network")):
            receipt = data.fetch({"files": {"prefix": spec}}, self.root, offline=True)
        self.assertEqual(data.cache_path(self.root, spec).read_bytes(), body)
        self.assertEqual(receipt["received_bytes"], 0)

    def test_selected_parquet_rows_keep_native_indices_and_skip_other_groups(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet = self.root / "source.parquet"
        pq.write_table(pa.table({"native": ["zero", "one", "two", "three"]}), parquet, row_group_size=2)
        spec = self.spec(parquet.read_bytes())
        spec["path"] = "source.parquet"
        self.seed(spec, parquet.read_bytes())
        items = [{"source": "fixture", "locator": {"file_key": "f", "row_index": 3}},
                 {"source": "fixture", "locator": {"file_key": "f", "row_index": 2}}]
        reader = SourceReader({"files": {"f": spec}}, self.root)
        self.addCleanup(reader.close)
        reader.selected_rows(items)
        self.assertEqual([reader.row(x)["native"] for x in items], ["three", "two"])
        self.assertEqual(set(reader.rows), {("f", 2), ("f", 3)})
        bad = {"source": "fixture", "locator": {"file_key": "f", "row_index": 4}}
        with self.assertRaisesRegex(ValueError, "ROW_OUT_OF_RANGE"):
            reader.selected_rows([bad])

    def test_failed_prepare_preserves_attempt_then_retries_and_reuses_complete_output(self):
        recipe = {"files": {}, "conversion": {"dependencies": {}, "tokenizer": {"files": {}}}}
        output = self.root / "prepared"
        tok = SimpleNamespace(name_or_path=str(self.root))

        def interrupted(recipe, sidecars, cache, stage, **kwargs):
            stage.mkdir()
            (stage / "partial.json").write_text("preserved")
            raise ValueError("native row mismatch")

        def completed(recipe, sidecars, cache, stage, **kwargs):
            stage.mkdir()
            (stage / "complete.json").write_text("verified")

        with patch("fine_tuning.error_note.v18_recipe.load_recipe", return_value=recipe), \
                patch("fine_tuning.error_note.v18_recipe.load_sidecars", return_value={}), \
                patch.object(data, "fetch", return_value={}) as fetch, \
                patch.object(data, "validate", return_value={"status": "verified"}), \
                patch("fine_tuning.error_note.v18_data_convert.convert", side_effect=interrupted) as convert:
            with self.assertRaisesRegex(ValueError, "native row mismatch"):
                data.prepare(self.root / "recipe", self.root / "cache", output, tokenizer=tok)
            self.assertFalse(output.exists())
            failures = list(self.root.glob(".prepared.v18-*.failed"))
            self.assertEqual((failures[0] / "partial.json").read_text(), "preserved")
            convert.side_effect = completed
            self.assertEqual(data.prepare(self.root / "recipe", self.root / "cache", output, tokenizer=tok),
                             {"status": "verified"})
            self.assertEqual((output / "complete.json").read_text(), "verified")
            before = fetch.call_count
            data.prepare(self.root / "recipe", self.root / "cache", output)
            self.assertEqual(fetch.call_count, before)

    def _prepare_small_source(self):
        from dataclasses import asdict
        from PIL import Image
        import pyarrow as pa
        import pyarrow.parquet as pq
        from fine_tuning.error_note.v18_recipe import expected_digests, load_recipe
        from fine_tuning.notebook_rl.adapters.hf_common import to_problem
        from fine_tuning.notebook_rl.adapters.hf_science import SPECS, _scienceqa_candidate
        from fine_tuning.notebook_rl.contamination import image_fingerprints

        picture = io.BytesIO()
        Image.new("RGB", (16, 16), "red").save(picture, format="PNG")
        row = {"image": {"bytes": picture.getvalue(), "path": None}, "question": "Which material is solid?",
               "choices": ["wood", "water"], "answer": 0, "hint": "", "topic": "physics",
               "category": "Physical properties", "grade": "grade1", "skill": "Properties",
               "subject": "natural science", "task": "closed choice", "lecture": "", "solution": ""}
        native = _scienceqa_candidate(row, "train:0")
        made = to_problem(native, SPECS["hf-scienceqa"], index=0)
        records = {name: asdict(getattr(made, name)) for name in ("actor", "verifier", "lineage")}
        metadata = {name: row[name] for name in ("category", "grade", "skill", "subject", "task", "topic")}
        metadata["native_row_index"] = 0
        expected_item = dict(records, question=made.actor.question, choices=made.actor.choices,
            evaluation={"gold": made.verifier.gold, "answer_type": made.verifier.answer_type,
                        "accept": made.verifier.accept, "question": made.actor.question, "choices": made.actor.choices},
            annotation=None, human_solution=None,
            references={"raw_annotation": None, "native_metadata": metadata, "human_solution": None})
        parquet = self.root / "native.parquet"
        pq.write_table(pa.Table.from_pylist([row]), parquet)
        spec = dict(self.spec(parquet.read_bytes()), path="train.parquet")
        self.seed(spec, parquet.read_bytes())
        with Image.open(io.BytesIO(made.image_bytes[0])) as normalized:
            pixels = image_fingerprints(normalized)["pixels_sha256"]
        selected = {"problem_id": made.actor.problem_id, "source": native.source,
                    "source_revision": SPECS[native.source].revision, "native_id": native.native_id,
                    "ordinal": 0, "family_id": "fixture family", "role": "train", "role_index": 0,
                    "locator": {"file_key": "fixture", "row_index": 0}, "auxiliary_refs": [],
                    "images": [{"path": made.actor.image_paths[0], "sha256": made.actor.image_sha256[0],
                                "pixel_sha256": pixels}], "expected": expected_digests(expected_item)}
        canonical = load_recipe()
        recipe = {"content_sha256": "fixture", "files": {"fixture": spec},
                  "sources": {native.source: canonical["sources"][native.source]},
                  "items": [selected], "conversion": dict(canonical["conversion"],
                      tokenizer={"files": {}})}
        sidecar = {"problem_id": made.actor.problem_id, "layer": "unsupported", "rights": SPECS[native.source].license,
                   "auxiliary_refs": [], "contamination_receipt": None, "inherited_reservations": [],
                   "f0_prior": None, "human_solution_metadata": None, "references_human_metadata": None}
        sidecars = {"items": {made.actor.problem_id: sidecar}}
        tokenizer = SimpleNamespace(name_or_path=str(self.root), encode=lambda *args, **kwargs: [1, 2])
        output = self.root / "prepared-real"
        with patch("fine_tuning.error_note.v18_recipe.load_recipe", return_value=recipe), \
                patch("fine_tuning.error_note.v18_recipe.load_sidecars", return_value=sidecars):
            data.prepare(self.root / "recipe.json", self.root, output, offline=True, tokenizer=tokenizer)
        return output, recipe, sidecars, made

    def test_prepare_creates_vault_that_real_training_grader_can_read(self):
        from fine_tuning.error_note.prep_v16b import grader

        output, _, _, made = self._prepare_small_source()
        grade = grader(output, "train")
        status, _ = grade({"problem_id": made.actor.problem_id},
                          {"text": f"Final answer: {made.verifier.gold}", "finish_reason": "stop"})
        self.assertEqual(status, "CORRECT")
        for name in ("actor_inputs.jsonl", "verifier_private.jsonl", "lineage.jsonl", "manifest.yaml"):
            self.assertTrue((output / "vault/train" / name).is_file())

    def test_vault_gold_cannot_be_changed_by_resealing_both_transport_manifests(self):
        from dataclasses import replace
        from fine_tuning.notebook_rl.vault import read_verifier
        from fine_tuning.notebook_rl.schema import to_json

        root, recipe, sidecars, made = self._prepare_small_source()
        private = read_verifier(root / "vault/train")[made.actor.problem_id]
        changed = replace(private, gold="B" if private.gold == "A" else "A")
        path = root / "vault/train/verifier_private.jsonl"
        path.write_text(to_json(changed) + "\n")
        manifest_path = root / "vault/train/manifest.yaml"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["verifier_private.jsonl"]["sha256"] = data.file_sha(path)
        manifest_path.write_text(json.dumps(manifest))
        receipt_path = root / "payload-receipt.json"
        receipt = json.loads(receipt_path.read_text())
        for changed_path in (path, manifest_path):
            receipt["files"][changed_path.relative_to(root).as_posix()] = data.file_sha(changed_path)
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "VAULT_CONTENT_CHANGED"):
            data.validate(root, recipe=recipe, sidecars=sidecars)

    def test_guard_metadata_cannot_be_changed_by_resealing_transport_manifest(self):
        root, recipe, sidecars, _ = self._prepare_small_source()
        sidecars["guard"] = {"items": []}
        path = root / "guard-metadata-v18.json"
        path.write_text(json.dumps({"items": {}, "guard": {"items": ["changed"]}}))
        receipt_path = root / "payload-receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["files"][path.name] = data.file_sha(path)
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "FIXED_GUARD_METADATA_CHANGED"):
            data.validate(root, recipe=recipe, sidecars=sidecars)

    def test_trusted_recipe_rejects_changed_gold_even_after_local_receipt_is_resealed(self):
        from fine_tuning.error_note.v17_data import capability_manifest, private_verifier
        from fine_tuning.error_note.v18_recipe import expected_digests, portable_payload

        root = self.root / "portable"
        root.mkdir()
        item = {"problem_id": "fixture:one", "source": "fixture", "source_revision": "fixed",
                "native_id": "native:1", "family_id": "family:1", "group_id": "family:1",
                "actor": {"question": "Question", "choices": [], "image_paths": []},
                "verifier": {"gold": "original", "answer_type": "short_text", "accept": {}, "choices": []},
                "lineage": {"split": "train"}, "question": "Question", "choices": [],
                "evaluation": {"gold": "original", "answer_type": "short_text", "accept": {},
                               "choices": [], "question": "Question"},
                "annotation": None, "references": {"raw_annotation": None, "native_metadata": {},
                                                  "human_solution": None},
                "human_solution": None, "split": "train", "training_allowed": True,
                "benchmark_derived": False, "image_paths": [], "image_sha256": [], "pixel_sha256": [],
                "annotation_refs": [], "contamination_receipt": None, "f0_prior": None,
                "rights": "fixture", "layer": "unsupported", "inherited_reservations": []}
        item["v17_private_verifier"] = private_verifier(item)
        item["v17_capability"] = capability_manifest(item)
        recipe = {"content_sha256": "fixed recipe", "items": [{"problem_id": item["problem_id"],
            "role": "train", "family_id": item["family_id"], "images": [],
            "expected": expected_digests(item)}]}
        sidecars = {"items": {item["problem_id"]: dict(item, auxiliary_refs=[])}}
        data._write(root / "dataset-manifest.json", {"version": data.VERSION, "items": [item]})
        data._write(root / "order.json", {"items": [{"problem_id": item["problem_id"]}],
            "batches": [{"problem_ids": [item["problem_id"]]}]})
        receipt = {"version": data.VERSION, "recipe_content_sha256": recipe["content_sha256"],
            "source_sha256": data.file_sha(Path(data.__file__)),
            "converter_sha256": data.file_sha(Path(data.__file__).with_name("v18_data_convert.py")),
            "counts": {role: int(role == "train") for role in data.ROLES},
            "logical_sha256": data.canonical_digest(portable_payload([item])),
            "files": {p.name: data.file_sha(p) for p in root.iterdir()}}
        data._write(root / "payload-receipt.json", receipt)
        item["verifier"]["gold"] = "changed"
        path = root / "dataset-manifest.json"
        path.write_text(json.dumps({"version": data.VERSION, "items": [item]}))
        receipt["files"][path.name] = data.file_sha(path)
        receipt["logical_sha256"] = data.canonical_digest(portable_payload([item]))
        (root / "payload-receipt.json").write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "FROZEN_CONTENT_CHANGED"):
            data.validate(root, recipe=recipe, sidecars=sidecars)

    def test_existing_unverified_output_is_preserved_and_never_replaced(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "user-file"
        marker.write_text("keep")
        with patch("fine_tuning.error_note.v18_recipe.load_recipe", return_value={}), \
                patch("fine_tuning.error_note.v18_recipe.load_sidecars", return_value={}), \
                patch.object(data, "validate", side_effect=ValueError("changed")), \
                patch.object(data, "fetch", side_effect=AssertionError("unexpected fetch")):
            with self.assertRaisesRegex(ValueError, "changed"):
                data.prepare(self.root / "recipe", self.root / "cache", output)
        self.assertEqual(marker.read_text(), "keep")

    def test_archive_member_paths_and_hf_revisions_are_checked(self):
        reader = SourceReader({"files": {}}, self.root)
        self.addCleanup(reader.close)
        with self.assertRaisesRegex(ValueError, "UNSAFE_ARCHIVE_MEMBER"):
            reader.member("unknown", "../secret")
        with self.assertRaisesRegex(ValueError, "UNPINNED_HF_REVISION"):
            data._url({"provider": "huggingface", "repo": "org/data", "revision": "main", "path": "x"})


if __name__ == "__main__":
    unittest.main()
