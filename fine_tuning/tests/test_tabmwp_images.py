"""Tiny self-authored PNG/Git fixtures; no committed source images or questions."""

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from fine_tuning import tabmwp, tabmwp_images as images
from fine_tuning.audit import dump, sha
from fine_tuning.tests.test_tabmwp import candidate_scan, pool, synthetic_artifacts


def fixture_tree():
    pngs = {}
    for i in range(3):
        stream = io.BytesIO()
        Image.new("RGB", (2 + i, 3), (i, 80, 120)).save(stream, format="PNG")
        pngs[f"tables/{i}.png"] = stream.getvalue()
    entries = [{"path": "data/tabmwp/" + path, "mode": "100644", "type": "blob",
                "sha": images.git_hash("blob", data), "size": len(data)} for path, data in pngs.items()]
    for parent in ("data/tabmwp/tables", "data/tabmwp", "data"):
        children = [e for e in entries if e["path"].rpartition("/")[0] == parent]
        payload = b"".join(e["mode"].lstrip("0").encode() + b" " + e["path"].rsplit("/", 1)[1].encode()
                           + b"\0" + bytes.fromhex(e["sha"]) for e in sorted(children, key=lambda e: e["path"]))
        entries.append({"path": parent, "mode": "040000", "type": "tree",
                        "sha": images.git_hash("tree", payload)})
    root_sha = images.git_hash("tree", b"40000 data\0" + bytes.fromhex(entries[-1]["sha"]))
    commit = {"sha": tabmwp.REVISION, "tree": {"sha": root_sha}}
    return pngs, commit, {"sha": root_sha, "truncated": False, "tree": entries}


class TabMWPImageTests(unittest.TestCase):
    def test_selection_replays_frozen_candidate_exclusions_and_keeps_default(self):
        bodies, identities = synthetic_artifacts(pool())
        with tempfile.TemporaryDirectory() as temp, patch.object(tabmwp, "ARTIFACTS", identities), \
                patch.object(tabmwp.urllib.request, "urlopen", side_effect=lambda url, timeout: io.BytesIO(bodies[url.removeprefix(tabmwp.RAW_URL)])):
            source = Path(temp) / "source"
            tabmwp.acquire(source)
            prepared = source / "prepared"
            old = tabmwp.prepare(source, prepared)
            before = images.selection(prepared)
            receipt = source / "scan.json"
            sid = old["selection"]["selected_ids"]["train"][0]
            dump(receipt, candidate_scan(source, prepared, [sid]))
            revised = source / "revised"
            new = tabmwp.prepare(source, revised, receipt)
            paths, bound, identity = images.selection(revised)
            self.assertEqual(identity, new["preparation_hash"])
            self.assertEqual(len(paths), 2000)
            self.assertNotIn(sid, {value["id"] for value in paths.values()})
            self.assertIn(str(revised / "exclusion_receipt.json"), bound)
            self.assertEqual(images.selection(prepared), before)
            scan = json.loads((revised / "exclusion_receipt.json").read_text())
            scan["candidate_record_count"] = 5
            (revised / "exclusion_receipt.json").write_text(json.dumps(scan))
            with self.assertRaises(ValueError):
                images.selection(revised)
            (revised / "exclusion_receipt.json").write_bytes(receipt.read_bytes())
            manifest = json.loads((revised / "manifest.json").read_text())
            manifest["selection"]["train_candidate_exclusion"]["excluded_train_families"] = []
            manifest["preparation_hash"] = tabmwp.digest({k: v for k, v in manifest.items() if k != "preparation_hash"})
            (revised / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                images.selection(revised)

    def test_tree_blob_png_and_path_trust_boundaries(self):
        pngs, commit, tree = fixture_tree()
        with patch.object(images, "TREE_SHA", tree["sha"]):
            entries = images.verify_tree(commit, tree)
            data = pngs["tables/0.png"]
            entry = entries["data/tabmwp/tables/0.png"]
            proof = images.verify_png(data, entry)
            self.assertEqual(proof["size"], [2, 3])
            for changed in (dict(tree, truncated=True), dict(tree, sha="0" * 40)):
                with self.assertRaises(ValueError):
                    images.verify_tree(commit, changed)
            for path in ("../escape", "/absolute", "data//bad", "data\\bad"):
                altered = copy.deepcopy(tree)
                altered["tree"][0]["path"] = path
                with self.assertRaises(ValueError):
                    images.verify_tree(commit, altered)
            altered = copy.deepcopy(tree)
            altered["tree"][0]["sha"] = "0" * 40
            with self.assertRaises(ValueError):
                images.verify_tree(commit, altered)
            with self.assertRaises(ValueError):
                images.verify_png(data + b"changed", entry)
            with self.assertRaises(ValueError):
                images.verify_png(data, dict(entry, mode="120000"))
            bad = b"not a PNG"
            with self.assertRaises(ValueError):
                images.verify_png(bad, dict(entry, size=len(bad), sha=images.git_hash("blob", bad)))

    def test_partial_preserved_resume_complete_and_never_overwritten(self):
        pngs, commit, tree = fixture_tree()
        paths = {path: {"id": f"tabmwp-{i}", "role": "train"} for i, path in enumerate(pngs)}
        calls = []
        failures = {"tables/1.png"}
        def download(url, limit):
            calls.append(url)
            if "/commits/" in url:
                return json.dumps(commit).encode()
            if "/trees/" in url:
                return json.dumps(tree).encode()
            self.assertTrue(url.startswith(tabmwp.RAW_URL + "data/tabmwp/"))
            path = url.removeprefix(tabmwp.RAW_URL + "data/tabmwp/")
            if path in failures:
                raise TimeoutError("fixture interruption")
            self.assertEqual(limit, len(pngs[path]))
            return pngs[path]
        with tempfile.TemporaryDirectory() as temp, patch.object(images, "TREE_SHA", tree["sha"]), \
                patch.object(images, "selection", return_value=(paths, {}, "frozen")), \
                patch.object(images, "download", side_effect=download):
            root = Path(temp) / "new-images"
            first = images.acquire("fixture-prepared", root)
            self.assertEqual(first["status"], "INCOMPLETE")
            self.assertEqual(first["verified_count"], 2)
            self.assertFalse(first["rights_approved"])
            self.assertFalse(first["training_allowed"])
            saved = sha(root / "tables/0.png")
            first_receipt = sha(root / "receipt_01.json")
            failures.clear()
            final = images.acquire("fixture-prepared", root)
            self.assertEqual(final["status"], "COMPLETE")
            self.assertEqual(set(final["files"]), set(paths))
            self.assertEqual(sha(root / "tables/0.png"), saved)
            self.assertEqual(sha(root / "receipt_01.json"), first_receipt)
            self.assertEqual(sum("/trees/" in url for url in calls), 1)
            self.assertEqual(sum(url.endswith("/0.png") for url in calls), 1)
            self.assertFalse(final["human_semantic_certification"])
            with self.assertRaises(FileExistsError):
                images.acquire("fixture-prepared", root)

    def test_corrupted_existing_files_preserved_and_retry_bounded(self):
        pngs, commit, tree = fixture_tree()
        paths = {path: {"id": f"tabmwp-{i}", "role": "train"} for i, path in enumerate(pngs)}
        def download(url, limit):
            if "/commits/" in url:
                return json.dumps(commit).encode()
            if "/trees/" in url:
                return json.dumps(tree).encode()
            raise TimeoutError("fixture")
        with tempfile.TemporaryDirectory() as temp, patch.object(images, "TREE_SHA", tree["sha"]), \
                patch.object(images, "selection", return_value=(paths, {}, "frozen")), \
                patch.object(images, "download", side_effect=download):
            root = Path(temp) / "new-images"
            images.acquire("fixture-prepared", root)
            (root / "tables").mkdir()
            bad = root / "tables/0.png"
            bad.write_bytes(b"preserve interrupted data")
            second = images.acquire("fixture-prepared", root)
            self.assertEqual(second["status"], "INCOMPLETE")
            self.assertIn("tables/0.png", second["errors"])
            self.assertEqual(bad.read_bytes(), b"preserve interrupted data")
            images.acquire("fixture-prepared", root)
            with self.assertRaises(ValueError):
                images.acquire("fixture-prepared", root)
            with patch.object(images, "selection", return_value=({}, {}, "different")), self.assertRaises(ValueError):
                images.acquire("fixture-prepared", root)

    def test_native_frozen_selection_and_source_hash_binding(self):
        bodies, identities = synthetic_artifacts(pool())
        def download(url, timeout):
            return io.BytesIO(bodies[url.removeprefix(tabmwp.RAW_URL)])
        with tempfile.TemporaryDirectory() as temp, patch.object(tabmwp, "ARTIFACTS", identities), \
                patch.object(tabmwp.urllib.request, "urlopen", side_effect=download):
            source = Path(temp) / "source"
            prepared = source / "prepared"
            tabmwp.acquire(source)
            tabmwp.prepare(source, prepared)
            paths, bound, _ = images.selection(prepared)
            self.assertEqual(len(paths), 2000)
            self.assertEqual(sum(p["role"] == "sealed" for p in paths.values()), 200)
            self.assertIn(str(source / "source_receipt.json"), bound)
            with (prepared / "B/train.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaises(ValueError):
                images.selection(prepared)


if __name__ == "__main__":
    unittest.main()
