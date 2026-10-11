"""Self-authored fixtures only; no downloaded benchmark or TabMWP text."""

import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fine_tuning import tabmwp
from fine_tuning.audit import check_splits, dump, read_rows, sha, write_rows


def problem(pid, split="train", *, mcq=False):
    return {"question": f"Read item {pid} from the image.", "answer": "fixture answer",
            "choices": ["other", "fixture answer"] if mcq else None,
            "ques_type": "multi_choice" if mcq else "free_text", "table_title": "Fixture title",
            "table_for_pd": {"Z header": [pid], "A header": ["fixture cell"]},
            "table": f"Z header | A header\n{pid} | fixture cell", "row_num": 2,
            "column_num": 2, "split": split, "solution": "DO NOT SERIALIZE THIS SOLUTION"}


def pool():
    result = {}
    offset = 0
    for split, count in tabmwp.COUNTS.items():
        result[split] = {str(i): problem(str(i), split, mcq=i % 3 == 0)
                         for i in range(offset, offset + count + 3)}
        offset += count + 3
    return result


def synthetic_artifacts(problems):
    bodies = {"README.md": b"Synthetic fixture dataset license: CC-BY-NC-SA-4.0",
              "LICENSE.md": b"Synthetic fixture code license: MIT"}
    bodies["data/tabmwp/splits.json"] = json.dumps(
        {split: [int(pid) for pid in rows] for split, rows in problems.items()}).encode()
    for split, rows in problems.items():
        bodies[f"data/tabmwp/problems_{split}.json"] = json.dumps(rows).encode()
    identities = {name: (len(data), hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest())
                  for name, data in bodies.items()}
    return bodies, identities


def candidate_scan(source, prepared, candidate_ids):
    """Self-authored historical scan with unchanged original proposals/index bindings."""
    manifest = json.loads((prepared / "manifest.json").read_text())
    native = read_rows(prepared / "A/train.jsonl")
    trial = source / "original-trial"
    write_rows(trial / "A/proposed.jsonl", list(reversed(native)))
    index = source / "benchmark_index.json"
    dump(index, {"status": "COMPLETE", "fixture": True})
    trial_manifest = {"tabmwp_preparation_hash": manifest["preparation_hash"],
                      "proposals": {"A": {"sha256": sha(trial / "A/proposed.jsonl")}}}
    trial_manifest["trial_hash"] = tabmwp.digest(trial_manifest)
    dump(trial / "trial_manifest.json", trial_manifest)
    paths = [source / "source_receipt.json", *[source / name for name in tabmwp.ARTIFACTS],
             prepared / "manifest.json", *[prepared / name for name in manifest["files"]],
             trial / "trial_manifest.json", trial / "A/proposed.jsonl", index, Path(tabmwp.__file__).resolve()]
    return {"status": "CANDIDATE_ONLY_SCAN_COMPLETE_NOT_APPROVAL", "source_id": tabmwp.SOURCE_ID,
            "tabmwp_revision": tabmwp.REVISION, "condition": "A", "training_allowed": False,
            "approval_executed": False, "gpu_executed": False, "processor_executed": False,
            "sample_replacements": False, "inputs_unchanged_before_after": True,
            "scanned_ids": [row["id"] for row in reversed(native)], "scanned_record_count": 1600,
            "candidates": [{"id": sid, "reasons": ["NEAR_IMAGE_REVIEW_REQUIRED"]} for sid in candidate_ids],
            "candidate_record_count": len(candidate_ids), "no_candidate_record_count": 1600 - len(candidate_ids),
            "reason_counts": {"NEAR_IMAGE_REVIEW_REQUIRED": len(candidate_ids)},
            "reason_combination_counts": {"NEAR_IMAGE_REVIEW_REQUIRED": len(candidate_ids)},
            "trial_root": str(trial), "tabmwp_trial_hash": trial_manifest["trial_hash"],
            "tabmwp_preparation_hash": manifest["preparation_hash"],
            "source_and_index_sha256": {str(path): sha(path) for path in paths}}


class TabMWPTests(unittest.TestCase):
    def test_explicit_candidate_family_exclusion_preserves_default_and_eval(self):
        samples = pool()
        first = tabmwp.plan(samples)[1]["selected_ids"]["train"][0].removeprefix("tabmwp-")
        sibling = next(pid for pid in samples["train"] if pid != first)
        samples["train"][sibling] = dict(copy.deepcopy(samples["train"][first]),
                                          question="Another question on the same fixture table.")
        bodies, identities = synthetic_artifacts(samples)
        with tempfile.TemporaryDirectory() as temp, patch.object(tabmwp, "ARTIFACTS", identities), \
                patch.object(tabmwp.urllib.request, "urlopen", side_effect=lambda url, timeout: io.BytesIO(bodies[url.removeprefix(tabmwp.RAW_URL)])):
            source = Path(temp) / "source"
            tabmwp.acquire(source)
            original_root = source / "prepared"
            old = tabmwp.prepare(source, original_root)
            original_hashes = {name: sha(original_root / name) for name in ["manifest.json", *old["files"]]}
            default = tabmwp.prepare(source, source / "default-again", None)
            self.assertEqual(old, default)
            self.assertEqual(old["preparation_hash"], default["preparation_hash"])
            receipt = source / "scan.json"
            scan = candidate_scan(source, original_root, ["tabmwp-" + first])
            # Updating this converter must not invalidate a historical producer SHA.
            scan["source_and_index_sha256"].pop(str(Path(tabmwp.__file__).resolve()))
            scan["source_and_index_sha256"][str(source / "historical-code/tabmwp.py")] = "1" * 64
            dump(receipt, scan)
            new = tabmwp.prepare(source, source / "new", receipt)
            again = tabmwp.prepare(source, source / "new-again", receipt)
            self.assertEqual(new, again)
            self.assertNotEqual(new["preparation_hash"], old["preparation_hash"])
            policy = new["selection"]["train_candidate_exclusion"]
            self.assertEqual(set(policy["excluded_train_ids"]), {"tabmwp-" + first, "tabmwp-" + sibling})
            self.assertEqual(policy["scan_sha256"], sha(receipt))
            self.assertEqual(policy["historical_converter_sha256"], "1" * 64)
            self.assertTrue(set(policy["excluded_train_ids"]).isdisjoint(new["selection"]["selected_ids"]["train"]))
            self.assertEqual({role: len(ids) for role, ids in new["selection"]["selected_ids"].items()},
                             {"train": 1600, "dev": 200, "sealed": 200})
            for variant in ("A", "B"):
                for role in ("dev", "sealed"):
                    self.assertEqual((source / "new" / f"{variant}/{role}.jsonl").read_bytes(),
                                     (original_root / f"{variant}/{role}.jsonl").read_bytes())
            pair = [read_rows(source / "new" / f"{variant}/train.jsonl") for variant in ("A", "B")]
            self.assertEqual([r["id"] for r in pair[0]], [r["id"] for r in pair[1]])
            for left, right in zip(*pair, strict=True):
                self.assertEqual({k: v for k, v in left.items() if k != "target_text"},
                                 {k: v for k, v in right.items() if k != "target_text"})
            self.assertEqual(original_hashes, {name: sha(original_root / name) for name in original_hashes})
            excluded = [row["family_id"] for row in read_rows(original_root / "A/train.jsonl")[:5]]
            with self.assertRaisesRegex(ValueError, "Insufficient clean train"):
                tabmwp.plan(samples, excluded)

    def test_candidate_scan_rejects_unknown_dev_approval_counts_and_changed_bindings(self):
        bodies, identities = synthetic_artifacts(pool())
        with tempfile.TemporaryDirectory() as temp, patch.object(tabmwp, "ARTIFACTS", identities), \
                patch.object(tabmwp.urllib.request, "urlopen", side_effect=lambda url, timeout: io.BytesIO(bodies[url.removeprefix(tabmwp.RAW_URL)])):
            source = Path(temp) / "source"
            tabmwp.acquire(source)
            prepared = source / "prepared"
            old = tabmwp.prepare(source, prepared)
            receipt = source / "scan.json"
            scan = candidate_scan(source, prepared, [old["selection"]["selected_ids"]["train"][0]])
            variants = [{"status": "APPROVED"}, {"training_allowed": True}, {"gpu_executed": True},
                        {"approval_executed": True}, {"training_executed": True}, {"tabmwp_revision": "wrong"},
                        {"candidate_record_count": 2}, {"reason_counts": {}}, {"scanned_ids": scan["scanned_ids"][:-1]},
                        {"candidates": [{"id": "tabmwp-unknown", "reasons": ["NEAR_IMAGE_REVIEW_REQUIRED"]}]},
                        {"candidates": [{"id": old["selection"]["selected_ids"]["dev"][0], "reasons": ["NEAR_IMAGE_REVIEW_REQUIRED"]}]},
                        {"candidates": [{"id": scan["candidates"][0]["id"], "reasons": ["MODEL_WRONG_ANSWER"]}]}]
            for i, changes in enumerate(variants):
                receipt.write_text(json.dumps(dict(scan, **changes)))
                output = source / f"rejected-{i}"
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    tabmwp.prepare(source, output, receipt)
                self.assertFalse(output.exists())
            receipt.write_text(json.dumps(scan))
            (source / "benchmark_index.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                tabmwp.prepare(source, source / "changed-index", receipt)

    def test_native_parity_order_and_headerless(self):
        item = problem("1")
        facts = tabmwp.table_facts(item)
        self.assertEqual(facts["headers"], ["Z header", "A header"])
        self.assertEqual(facts["rows"], [["1", "fixture cell"]])
        item.update(table_title=None, row_num=1, table="1 | fixture cell")
        self.assertIsNone(tabmwp.table_facts(item)["headers"])
        self.assertIsNone(tabmwp.table_facts(item)["title"])

    def test_malformed_tables_quarantined(self):
        mutations = [{"column_num": 4}, {"row_num": 8}, {"table": "bad parity"},
                     {"table_for_pd": {"one": ["1"], "two": ["1", "2"]}},
                     {"table_for_pd": {"one": [1]}}]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                tabmwp.table_facts(dict(problem("1"), **mutation))
        samples = pool()
        samples["train"]["0"]["table"] = "malformed"
        _, selection = tabmwp.plan(samples)
        self.assertTrue(any(row["native_id"] == "0" and "disagree" in row["reason"]
                            for row in selection["quarantine"]))
        self.assertNotIn("tabmwp-0", selection["selected_ids"]["train"])

    def test_pair_inputs_answer_and_solution_isolation(self):
        for mcq in (False, True):
            item = problem("1", mcq=mcq)
            pair = tabmwp.paired_records("1", item, tabmwp.table_facts(item), "train")
            a, b = pair["A"], pair["B"]
            self.assertEqual({k: v for k, v in a.items() if k != "target_text"},
                             {k: v for k, v in b.items() if k != "target_text"})
            self.assertEqual(json.loads(a["target_text"])["answer"], json.loads(b["target_text"])["answer"])
            self.assertNotIn("Z header", a["user_text"])
            self.assertNotIn("fixture cell", a["user_text"])
            self.assertNotIn("DO NOT SERIALIZE", json.dumps(pair))
            if mcq:
                self.assertIn("A. other\nB. fixture answer", a["user_text"])
            else:
                self.assertNotIn("fixture answer", a["user_text"])
            modified = dict(item, answer="other", solution="different hidden solution")
            other = tabmwp.paired_records("1", modified, tabmwp.table_facts(modified), "train")
            self.assertEqual(a["user_text"], other["A"]["user_text"])
            self.assertFalse(a["training_allowed"])

    def test_illegal_native_mcq(self):
        item = problem("1", mcq=True)
        for changes in ({"answer": "not a choice"}, {"choices": ["same", "same"]},
                        {"choices": None}, {"ques_type": "unrecognized"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                tabmwp.paired_records("1", dict(item, **changes), tabmwp.table_facts(item), "train")
        with self.assertRaises(ValueError):
            tabmwp.paired_records("../1", item, tabmwp.table_facts(item), "train")

    def test_deterministic_plan_family_excludes_all_splits(self):
        samples = pool()
        train_id, dev_id = next(iter(samples["train"])), next(iter(samples["dev"]))
        shared = copy.deepcopy(samples["train"][train_id])
        shared["split"], shared["table_title"] = "dev", "Different title is not a new table"
        # Even an invalid question in dev must exclude its valid train table family.
        shared["question"] = ""
        samples["dev"][dev_id] = shared
        before = copy.deepcopy(samples)
        variants, selection = tabmwp.plan(samples)
        reversed_pool = {split: dict(reversed(list(rows.items()))) for split, rows in samples.items()}
        self.assertEqual((variants, selection), tabmwp.plan(reversed_pool))
        self.assertEqual(samples, before)
        self.assertEqual({k: len(v) for k, v in selection["selected_ids"].items()},
                         {"train": 1600, "dev": 200, "sealed": 200})
        excluded = {row["native_id"] for row in selection["quarantine"]}
        self.assertTrue({train_id, dev_id}.issubset(excluded))
        check_splits([row for rows in variants["A"].values() for row in rows])
        self.assertEqual(variants["A"]["sealed"][0]["split"], "holdout")
        del samples["test"]
        with self.assertRaises(ValueError):
            tabmwp.plan(samples)

    def test_immutable_source_bundle_preparation_and_no_overwrite(self):
        bodies, identities = synthetic_artifacts(pool())
        def download(url, timeout):
            self.assertEqual(timeout, 60)
            self.assertTrue(url.startswith(tabmwp.RAW_URL))
            return io.BytesIO(bodies[url.removeprefix(tabmwp.RAW_URL)])
        with tempfile.TemporaryDirectory() as temp, patch.object(tabmwp, "ARTIFACTS", identities), \
                patch.object(tabmwp.urllib.request, "urlopen", side_effect=download):
            source, output = Path(temp) / "source", Path(temp) / "prepared"
            tabmwp.acquire(source)
            manifest = tabmwp.prepare(source, output)
            self.assertEqual(manifest["status"], "PREPARED_PENDING_GATES")
            self.assertFalse(manifest["training_allowed"])
            self.assertEqual(len(manifest["pending_gates"]), 5)
            self.assertFalse((source / "tables").exists())
            self.assertEqual(len(read_rows(output / "A/train.jsonl")), 1600)
            self.assertEqual(len(read_rows(output / "B/sealed.jsonl")), 200)
            with self.assertRaises(FileExistsError):
                tabmwp.acquire(source)
            with self.assertRaises(FileExistsError):
                tabmwp.prepare(source, output)
            receipt_path = source / "source_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["files"]["README.md"]["sha256"] = "0" * 64
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                tabmwp.load_source(source)
            (source / "README.md").write_text("modified official bytes")
            with self.assertRaises(ValueError):
                tabmwp.load_source(source)

    def test_download_hash_failure_creates_no_bundle(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(
                tabmwp.urllib.request, "urlopen", return_value=io.BytesIO(b"not official")):
            source = Path(temp) / "source"
            with self.assertRaises(ValueError):
                tabmwp.acquire(source)
            self.assertFalse(source.exists())


if __name__ == "__main__":
    unittest.main()
