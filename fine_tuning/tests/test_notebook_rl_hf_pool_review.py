"""Authored synthetic packet fixtures; the verdicts below are test inputs, not real similarity findings."""
import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from fine_tuning.notebook_rl import adjudication as a
from fine_tuning.notebook_rl import contamination as c
from fine_tuning.notebook_rl import hf_pool_review as r

SEED = 20261030
# pid -> (source, split, one list of linked benchmark record keys per candidate)
SPEC = {"P1": ("src-a", "train", [["R1"], ["R2"], ["R1", "R2"]]),
        "P2": ("src-a", "dev", [["R1", "R3"]]),
        "P3": ("src-b", "sealed", [["R3"], []]),
        "P4": ("src-a", "sealed", [["R2"]])}
UNFLAGGED = ("P5", "src-b")  # in the vault snapshot only: no candidate, so it stays allowlisted


class Fixture:
    def __init__(self, root: Path, spec: dict, long_question: bool = False):
        self.root, self.spec = root, spec
        keys = sorted({k for _, _, cands in spec.values() for keys in cands for k in keys})
        images = root / "images"
        images.mkdir()

        def png(name: str, size: tuple[int, int]) -> str:
            path = images / f"{name}.png"
            Image.new("RGB", size, (len(name) * 20 % 256, 90, 160)).save(path)
            return str(path)

        self.records: dict = {k: {"ref": {"id": k}, "question": ["x" * 1000 if long_question and i == 0 else f"bench question {k}"],
                            "choices": ["bench c1", "bench c2"], "image_fields": ["image_1"],
                            "image_paths": [png(k + "a", (50 + i, 40)), *([png(k + "b", (900, 100))] if i % 5 == 1 else [])],
                            "explanation_withheld": True} for i, k in enumerate(keys)}
        self.items: dict = {pid: {"problem_id": pid, "split": split, "source": src, "question": f"pool question {pid}",
                            "choices": ["pool yes", "pool no"], "image_paths": [png(pid, (700, 300))],
                            "image_sha256": [c._digest(pid)]} for pid, (src, split, _) in spec.items()}
        self.raw_candidates, self.rows, self.cid = [], [], {}
        for pid, (_, _, cands) in spec.items():
            for n, keys_ in enumerate(cands):
                cand = {"problem_id": pid, "stage": "perceptual", "reason": "DHASH_CROP_TILE_NEAR", "action": "REVIEW",
                        "hamming": n, "ref": {"low_information": n == 0, "benchmark_refs": []}}
                self.cid[(pid, n)] = a.candidate_id(cand)
                self.raw_candidates.append(cand)
                self.rows.append({"candidate_id": self.cid[(pid, n)], "problem_id": pid, "stage": cand["stage"],
                                  "reason": cand["reason"], "releasable": True, "benchmark_records": keys_,
                                  "candidate": cand})
        self.vault = {pid: {"structure_hash": f"family-{pid}", "image_sha256": [c._digest(pid)],
                            "actor": SimpleNamespace(source=src)} for pid, (src, _, _) in spec.items()}
        self.vault[UNFLAGGED[0]] = {"structure_hash": "family-P5", "image_sha256": [c._digest("P5")],
                                    "actor": SimpleNamespace(source=UNFLAGGED[1])}
        files = {"vault/actor_inputs.jsonl": "0" * 64}
        raw = {"version": c.VERSION, "status": "EXCLUSION_CHECKED_WITH_LIMITS", "training_allowed": False,
               "approval_granted": False, "blockers": [], "partial_name": None, "input_count": len(self.vault),
               "index_sha256": "1" * 64, "input_files": files, "input_files_sha256": c._digest(files),
               "steps": {s: {"status": "COMPLETE", "candidate_count": 0} for s in c._STEPS},
               "candidates": self.raw_candidates, "limitations": ["Authored unit fixture"]}
        family = c.family_quarantine(self.vault, self.raw_candidates)
        raw.update(allowlist=family["allowlist"], quarantined_ids=family["quarantined_ids"],
                   quarantined_families=family["families"])
        raw["receipt_sha256"] = c._digest({k: v for k, v in raw.items() if k != "receipt_sha256"})
        self.sample_path = root / "sample.json"
        self.assemble = root / "assemble"
        (self.assemble / "raw-fullscan").mkdir(parents=True)
        raw_path = self.assemble / "raw-fullscan/receipt_v2.json"
        raw_path.write_text(json.dumps(raw))
        self.files = files
        template = {"format": a.FORMAT, "raw_receipt_sha256": r._sha_file(raw_path), "index_sha256": raw["index_sha256"],
                    "input_files_sha256": raw["input_files_sha256"],
                    "provenance": {"kind": "AI_DIRECT_SIMILARITY_REVIEW", "human_verified": False,
                                   "reviewer_id": "", "reviewed_at_utc": ""}, "decisions": []}
        self.packet: dict = {"format": r.PACKET_FORMAT, "human_verified": False, "training_allowed": False,
                       "review_template": template, "candidates": self.rows, "items": self.items,
                       "benchmark_records": self.records}
        self.packet_path = root / "packet.json"
        self.packet_path.write_text(json.dumps(self.packet))
        self.sha = r._sha_file(self.packet_path)


class PoolReviewTests(unittest.TestCase):
    def setUp(self):
        r._thumb.cache_clear()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def fixture(self, spec=SPEC, name="fx", **kwargs) -> Fixture:
        (self.root / name).mkdir()
        return Fixture(self.root / name, spec, **kwargs)

    def rendered(self, fx: Fixture, per_source=2):
        sample = r.make_sample(fx.packet, fx.sha, per_source, SEED)
        out = fx.root / "sheets"
        fx.sample_path.write_text(json.dumps(sample))
        counts = r.make_sheets(fx.packet, sample, out)
        return sample, out, json.loads((out / "index.json").read_text()), counts

    def docs(self, index, overrides=None):
        docs: dict[str, dict] = {}
        for stem, meta in index.items():
            doc = docs.setdefault(meta["problem_id"], {"problem_id": meta["problem_id"], "sheets": [], "pairs": []})
            doc["sheets"].append(stem)
            for label, key in meta["labels"].items():
                doc["pairs"].append({"label": label, "record": key, "reason": f"authored reason {label}",
                                     "verdict": (overrides or {}).get((meta["problem_id"], key), "DISTINCT")})
        return docs

    def compile(self, fx, sample, out, index, docs, reviewer="authored-test-reviewer", stamp="2026-10-30T00:00:00Z"):
        vdir = Path(tempfile.mkdtemp(dir=fx.root))
        for i, doc in enumerate(docs.values() if isinstance(docs, dict) else docs):
            (vdir / f"{i:03d}.json").write_text(json.dumps(doc))
        return r.compile_decisions(fx.packet, r._desc(fx.sample_path), sample, index, out, vdir, reviewer, stamp)

    # ------------------------------------------------------------ sample
    def test_sample_is_deterministic_hash_ordered_split_preferring_and_gold_free(self):
        spec = {**SPEC, "P6": ("src-a", "train", [["R1"]])}
        fx = self.fixture(spec)
        for item in fx.packet["items"].values():
            item["gold"] = "SECRET-GOLD-ANSWER"
        first = r.make_sample(fx.packet, fx.sha, 2, SEED)
        self.assertEqual(first, r.make_sample(copy.deepcopy(fx.packet), fx.sha, 2, SEED))
        order = sorted(["P1", "P2", "P6"], key=lambda p: hashlib.sha256(f"{SEED}:{p}".encode()).hexdigest())
        self.assertEqual(first["problem_ids"], [*order[:2], "P3"])  # sealed P4 never beats train/dev; src-b only has P3
        self.assertEqual(first["counts"], {"src-a": {"flagged": 4, "sampled": 2}, "src-b": {"flagged": 1, "sampled": 1}})
        self.assertEqual(r.make_sample(fx.packet, fx.sha, 4, SEED)["problem_ids"][:3], sorted(
            ["P1", "P2", "P6"], key=lambda p: hashlib.sha256(f"{SEED}:{p}".encode()).hexdigest()))
        self.assertNotIn("SECRET", json.dumps(first))
        self.assertEqual(set(first), {"version", "packet_sha256", "seed", "per_source", "problem_ids", "counts"})
        with self.assertRaises(ValueError):
            r.make_sample(fx.packet, fx.sha, 0, SEED)

    # ------------------------------------------------------------ sheets
    def test_sheets_split_at_twelve_records_with_matching_hashes_and_trimmed_text(self):
        keys = [f"R{i:02d}" for i in range(13)]
        fx = self.fixture({"P1": ("src-a", "train", [keys[:7], keys[5:]])}, long_question=True)
        fx.items["P1"]["gold"] = "SECRET-GOLD-ANSWER"
        sample, out, index, counts = self.rendered(fx)
        self.assertEqual(counts, {"items": 1, "sheets": 2, "pairs": 13})
        self.assertEqual(sorted(index), ["0001-a", "0001-b"])
        self.assertEqual(list(index["0001-a"]["labels"]), [f"B{i}" for i in range(1, 13)])
        self.assertEqual(index["0001-b"]["labels"], {"B13": "R12"})
        for stem, meta in index.items():
            self.assertEqual(meta["problem_id"], "P1")
            for kind in ("png", "txt"):
                self.assertEqual(meta[f"{kind}_sha256"], hashlib.sha256((out / meta[kind]).read_bytes()).hexdigest())
            with Image.open(out / meta["png"]) as image:
                self.assertEqual((image.mode, image.width), ("RGB", 1320))
        text = (out / "0001-a.txt").read_text()
        self.assertIn("x" * 600, text)
        self.assertNotIn("x" * 601, text)
        self.assertIn("pool question P1", text)
        self.assertIn("(B) bench c2", text)
        self.assertIn("--- B1 R00 ---", text)
        self.assertIn("perceptual/DHASH_CROP_TILE_NEAR hamming=0 low_information=True", text)
        self.assertNotIn("SECRET", text)
        self.assertNotIn("B13", text)
        self.assertIn("B13", (out / "0001-b.txt").read_text())
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            r.make_sheets(fx.packet, sample, out)

    def test_sheets_raise_on_missing_benchmark_or_pool_image(self):
        for which in ("record", "pool"):
            with self.subTest(which=which):
                r._thumb.cache_clear()
                fx = self.fixture(SPEC, name=f"fx-{which}")
                sample = r.make_sample(fx.packet, fx.sha, 2, SEED)
                path = fx.records["R1"]["image_paths"][0] if which == "record" else fx.items["P1"]["image_paths"][0]
                Path(path).unlink()
                with self.assertRaisesRegex(ValueError, "missing image"):
                    r.make_sheets(fx.packet, sample, fx.root / "sheets")

    # ------------------------------------------------------------ compile
    def test_compile_coverage_precedence_unreviewed_and_adjudication_schema(self):
        fx = self.fixture()
        sample, out, index, _ = self.rendered(fx)
        self.assertEqual(sorted(sample["problem_ids"]), ["P1", "P2", "P3"])
        review, summary = self.compile(fx, sample, out, index,
                                       self.docs(index, {("P1", "R2"): "DUPLICATE", ("P2", "R1"): "UNRESOLVED"}))
        self.assertEqual(review["provenance"], {"kind": "AI_DIRECT_SIMILARITY_REVIEW", "human_verified": False,
                                                "reviewer_id": "authored-test-reviewer",
                                                "reviewed_at_utc": "2026-10-30T00:00:00Z"})
        self.assertEqual({k: v for k, v in review.items() if k not in ("provenance", "decisions")},
                         {k: v for k, v in fx.packet["review_template"].items() if k not in ("provenance", "decisions")})
        got = {d["candidate_id"]: d for d in review["decisions"]}
        self.assertEqual(len(got), len(fx.cid))
        expect = {("P1", 0): "DISTINCT", ("P1", 1): "DUPLICATE", ("P1", 2): "DUPLICATE", ("P2", 0): "UNRESOLVED",
                  ("P3", 0): "DISTINCT", ("P3", 1): "UNRESOLVED", ("P4", 0): "UNRESOLVED"}
        self.assertEqual({k: got[cid]["verdict"] for k, cid in fx.cid.items()}, expect)
        self.assertEqual(got[fx.cid[("P4", 0)]]["reason"], r.NOT_REVIEWED)
        self.assertEqual(got[fx.cid[("P4", 0)]]["evidence"], [r._desc(fx.sample_path)])
        self.assertTrue(got[fx.cid[("P3", 1)]]["reason"].startswith("NO_BENCHMARK_RECORD_LINKED"))
        self.assertIn("B2=DUPLICATE", got[fx.cid[("P1", 2)]]["reason"])
        evidence = got[fx.cid[("P1", 0)]]["evidence"]
        self.assertEqual(sorted(Path(e["path"]).suffix for e in evidence), [".json", ".png", ".txt"])
        # the decision schema must be accepted by adjudication's own validator against the synthetic raw receipt
        review_path = fx.root / "decisions.json"
        review_path.write_text(json.dumps(review))
        reviewed = a.apply(r._desc(fx.assemble / "raw-fullscan/receipt_v2.json"), r._desc(review_path), fx.vault)
        adj = reviewed["adjudication"]
        self.assertEqual(adj["counts"], {"DISTINCT": 2, "DUPLICATE": 2, "UNRESOLVED": 3})
        self.assertEqual(adj["blacklisted_problem_ids"], ["P1"])
        self.assertEqual(adj["unresolved_problem_ids"], ["P2", "P3", "P4"])
        self.assertEqual(reviewed["allowlist"], ["P5"])
        self.assertFalse(adj["human_verified"])
        # summary
        self.assertEqual(summary["candidates"], {"DISTINCT": 2, "DUPLICATE": 2, "UNRESOLVED": 3})
        self.assertEqual(summary["reviewed_candidates"], 6)
        self.assertEqual(summary["reviewed_pairs"], {"DISTINCT": 3, "DUPLICATE": 1, "UNRESOLVED": 1})
        self.assertEqual(summary["per_source_items"], {"src-a": {"reviewed": 2, "DISTINCT": 0, "DUPLICATE": 1, "UNRESOLVED": 1},
                                                      "src-b": {"reviewed": 1, "DISTINCT": 0, "DUPLICATE": 0, "UNRESOLVED": 1}})
        self.assertEqual(summary["per_reason"]["perceptual/DHASH_CROP_TILE_NEAR"],
                         {"candidates": 6, "DISTINCT": 2, "DUPLICATE": 2, "UNRESOLVED": 2, "duplicate_rate": 0.3333})

    def test_compile_rejects_incomplete_or_malformed_verdicts(self):
        fx = self.fixture()
        sample, out, index, _ = self.rendered(fx)
        stems = {meta["problem_id"]: stem for stem, meta in index.items()}

        def drop_pair(docs):
            docs["P1"]["pairs"].pop()

        def set_pair(**change):
            def mutate(docs):
                docs["P1"]["pairs"][0].update(change)
            return mutate

        def duplicate_pair(docs):
            docs["P1"]["pairs"].append(dict(docs["P1"]["pairs"][0]))

        def other_item_sheet(docs):
            docs["P1"]["sheets"].append(stems["P2"])

        def unsampled_item(docs):
            docs["P1"]["problem_id"] = "P4"

        def drop_item(docs):
            del docs["P3"]

        for name, mutate in (("missing verdict", drop_pair), ("unknown label", set_pair(label="B9")),
                             ("record mismatch", set_pair(record="R3")), ("bad verdict", set_pair(verdict="MAYBE")),
                             ("lowercase verdict", set_pair(verdict="duplicate")), ("empty reason", set_pair(reason="  ")),
                             ("missing reason", set_pair(reason=None)), ("unhashable label", set_pair(label=["B1"])),
                             ("second verdict", duplicate_pair), ("foreign sheet", other_item_sheet),
                             ("unsampled item", unsampled_item), ("item without file", drop_item)):
            docs = self.docs(index)
            mutate(docs)
            with self.subTest(name), self.assertRaises(ValueError):
                self.compile(fx, sample, out, index, docs)
        with self.assertRaises(ValueError):
            self.compile(fx, sample, out, index, [])
        for stamp in ("2026-10-30T00:00:00", "2026-10-30T00:00:00+09:00", "yesterday"):
            with self.subTest(stamp), self.assertRaisesRegex(ValueError, "UTC|ISO"):
                self.compile(fx, sample, out, index, self.docs(index), stamp=stamp)
        with self.assertRaisesRegex(ValueError, "reviewer-id"):
            self.compile(fx, sample, out, index, self.docs(index), reviewer=" ")

    def test_compile_rejects_changed_sheet_or_stale_packet(self):
        fx = self.fixture()
        sample, out, index, _ = self.rendered(fx)
        (out / index["0001-a"]["txt"]).write_text("changed after rendering")
        with self.assertRaisesRegex(ValueError, "changed since rendering"):
            self.compile(fx, sample, out, index, self.docs(index))
        stale = copy.deepcopy(fx.packet)
        stale["candidates"][0]["candidate_id"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "candidate_id mismatch"):
            r.compile_decisions(stale, r._desc(fx.sample_path), sample, index, out, out, "id", "2026-10-30T00:00:00Z")
        other = dict(sample, packet_sha256="0" * 64)
        other_path = fx.root / "other_sample.json"
        other_path.write_text(json.dumps(other))
        with self.assertRaisesRegex(ValueError, "does not belong"):
            r._load_sample(other_path, fx.packet, fx.sha)

    # ------------------------------------------------------------ CLI end to end (+ check dry run)
    def test_cli_sample_sheets_compile_check_roundtrip_writes_nothing_into_build(self):
        fx = self.fixture()
        work = fx.root / "pass1"

        def run(*argv):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = r.main([str(x) for x in argv])
            return code, json.loads(buffer.getvalue())

        code, shown = run("sample", "--packet", fx.packet_path, "--per-source", 2, "--seed", SEED, "--output", work)
        self.assertEqual((code, shown["status"], shown["items"]), (0, "SAMPLE_WRITTEN", 3))
        self.assertEqual(run("sample", "--packet", fx.packet_path, "--per-source", 2, "--seed", SEED, "--output", work)[0], 2)
        code, shown = run("sheets", "--packet", fx.packet_path, "--sample", work / "sample.json", "--output", work / "sheets")
        self.assertEqual((code, shown["items"], shown["sheets"], shown["pairs"]), (0, 3, 3, 5))
        index = json.loads((work / "sheets/index.json").read_text())
        (work / "verdicts").mkdir()
        for i, doc in enumerate(self.docs(index, {("P2", "R3"): "DUPLICATE"}).values()):
            (work / f"verdicts/{i}.json").write_text(json.dumps(doc))
        decisions = work / "decisions.json"
        args = ("compile", "--packet", fx.packet_path, "--sample", work / "sample.json", "--sheets",
                work / "sheets/index.json", "--verdicts", work / "verdicts", "--output", decisions,
                "--reviewer-id", "authored-test-reviewer", "--reviewed-at-utc", "2026-10-30T00:00:00Z")
        code, shown = run(*args)
        self.assertEqual((code, shown["status"], shown["decisions"]), (0, "DECISIONS_WRITTEN", 7))
        self.assertEqual(json.loads(Path(f"{decisions}.summary.json").read_text())["candidates"],
                         {"DISTINCT": 4, "DUPLICATE": 1, "UNRESOLVED": 2})
        self.assertEqual(run(*args)[0], 2)  # refuses to overwrite
        before = sorted(p.relative_to(fx.assemble) for p in fx.assemble.rglob("*"))
        with patch.object(r, "_snapshot", return_value=(fx.vault, fx.files, {})):
            code, shown = run("check", "--decisions", decisions, "--assemble-dir", fx.assemble)
        self.assertEqual(code, 0, shown)
        self.assertEqual(shown["status"], "CHECK_PASSED_DRY_RUN")
        self.assertEqual((shown["allowlist"], shown["items"]), (2, 5))
        self.assertEqual(shown["per_source"], {"src-a": {"items": 3, "allowlisted": 1}, "src-b": {"items": 2, "allowlisted": 1}})
        self.assertEqual(before, sorted(p.relative_to(fx.assemble) for p in fx.assemble.rglob("*")))
        decisions.write_text(decisions.read_text().replace("authored-test-reviewer", "x"))
        with patch.object(r, "_snapshot", return_value=(fx.vault, fx.files, {})):
            self.assertEqual(run("check", "--decisions", decisions, "--assemble-dir", fx.assemble)[0], 0)
            bad = json.loads(decisions.read_text())
            bad["decisions"].pop()
            decisions.write_text(json.dumps(bad))
            code, shown = run("check", "--decisions", decisions, "--assemble-dir", fx.assemble)
        self.assertEqual((code, shown["status"]), (2, "REFUSED"))


if __name__ == "__main__":
    unittest.main()
