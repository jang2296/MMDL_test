"""Authored fixtures only; no real dataset rows, upstream images, network, GPU or benchmark data."""
import contextlib
import copy
import dataclasses
import functools
import hashlib
import heapq
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from fine_tuning.notebook_rl import exclusion_engines, hf_pool
from fine_tuning.notebook_rl.adapters import hf_common as hf
from fine_tuning.notebook_rl.dataset import SPLITS
from fine_tuning.notebook_rl.generators.common import GeneratedProblem
from fine_tuning.notebook_rl.vault import read_actors, read_lineage, read_verifier

SPECS = {s: hf.SourceSpec(s, f"org/{s[3:]}", "a" * 40, "cc-by-4.0", "HUMAN", "authored annotators", s[3:].replace("-", "_"))
         for s in ("hf-fake-a", "hf-fake-b")}
SPECS["hf-fake-p"] = hf.SourceSpec("hf-fake-p", "org/fake-p", "a" * 40, "cc-by-4.0", "PROGRAM", "upstream program", "fake_p")
SUBJECTS = ("Biology", "Physics")


def png(seed: str) -> bytes:
    r, g, b = hashlib.sha256(seed.encode()).digest()[:3]
    out = io.BytesIO()
    Image.new("RGB", (8, 8), (r, g, b)).save(out, format="PNG")
    return out.getvalue()


@functools.cache
def family_keys(source: str, per_split: int) -> dict[str, list[str]]:
    """Deterministic search for family keys that split_for sends to each split."""
    found: dict[str, list[str]] = {s: [] for s in SPLITS}
    i = 0
    while any(len(v) < per_split for v in found.values()):
        key = f"fam{i}"
        i += 1
        probe = hf.Candidate(source, "probe", "Biology", "q", (), "short_text", "x", family_key=key)
        split = hf.split_for(probe)
        if len(found[split]) < per_split:
            found[split].append(key)
    return found


def cand(source, native_id, subject, family_key, *, gold="A", image=None, question=None, choices=("x", "y", "z")):
    return hf.Candidate(source, native_id, subject, question or f"Question {native_id}?", choices, "mcq", gold,
                        images=(image or png(native_id),), family_key=family_key)


def supply(source, per_bucket, *, gold="A", subjects=SUBJECTS):
    keys = family_keys(source, 2 * per_bucket)
    return [cand(source, f"{split}-{subject}-{k}", subject, keys[split][si * per_bucket + k], gold=gold)
            for split in SPLITS for si, subject in enumerate(subjects) for k in range(per_bucket)]


def registry(supplies, seen=None):
    def make(source):
        def iterate(path):
            if seen is not None:
                seen.append(path)
            yield from supplies[source]
        return iterate
    return {s: (SPECS[s], make(s)) for s in supplies}


def quotas(n, sources=("hf-fake-a",), subjects=SUBJECTS):
    return {s: {sub: dict.fromkeys(SPLITS, n) for sub in subjects} for s in sources}


def plan(q, seed=7):
    return {"version": hf_pool.PLAN_VERSION, "seed": seed, "quotas": q}


def plan2(q, stage_sha256=None, seed=7):
    return {"version": hf_pool.PLAN_V2, "seed": seed, "quotas": q, "stage_sha256": stage_sha256}


GEN_SPEC = hf.SourceSpec("gen-fake", "org/gen-fake", "a" * 40, "cc-by-4.0", "HUMAN", "authored annotators", "gen_fake")


def staged_item(split, native_id, *, index, subject="Biology", image=None, question=None, choices=("x", "y", "z"), k=0):
    """A (generator, subject, GeneratedProblem) triple as gen_stage.load_stage would return it."""
    key = family_keys("gen-fake", 4)[split][k]
    c = hf.Candidate("gen-fake", native_id, subject, question or f"Gen question {native_id}?", choices, "mcq", "A",
                     images=(image or png("gen-" + native_id),), family_key=key)
    return ("gen-fake", subject, hf.to_problem(c, GEN_SPEC, index=index))


def staged_all():
    return [staged_item(split, f"g-{split}", index=100 + i) for i, split in enumerate(SPLITS)]


def ids(pools):
    return {split: [p.actor.problem_id for p in rows] for split, rows in pools.items()}


class LoadPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.reg = registry({"hf-fake-a": []})

    def load(self, value):
        path = Path(self.tmp.name) / "plan.json"
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return hf_pool.load_plan(path, self.reg)

    def test_valid_plan_is_returned(self):
        good = plan(quotas(2))
        self.assertEqual(self.load(good), good)

    def test_rejections(self):
        def mutated(fn):
            value = copy.deepcopy(plan(quotas(2)))
            fn(value)
            return value
        bad = {
            "bad version": mutated(lambda p: p.update(version="hf-pool-plan-v0")),
            "extra key": mutated(lambda p: p.update(extra=1)),
            "missing key": mutated(lambda p: p.pop("seed")),
            "negative seed": mutated(lambda p: p.update(seed=-1)),
            "bool seed": mutated(lambda p: p.update(seed=True)),
            "unknown source": mutated(lambda p: p["quotas"].update({"hf-nope": p["quotas"]["hf-fake-a"]})),
            "unknown subject": mutated(lambda p: p["quotas"]["hf-fake-a"].update({"Basketry": dict.fromkeys(SPLITS, 1)})),
            "missing split key": mutated(lambda p: p["quotas"]["hf-fake-a"]["Biology"].pop("dev")),
            "extra split key": mutated(lambda p: p["quotas"]["hf-fake-a"]["Biology"].update(test=1)),
            "negative quota": mutated(lambda p: p["quotas"]["hf-fake-a"]["Biology"].update(train=-1)),
            "float quota": mutated(lambda p: p["quotas"]["hf-fake-a"]["Biology"].update(train=1.5)),
            "split with zero total": plan(quotas(0) | {"hf-fake-a": {"Biology": {**dict.fromkeys(SPLITS, 1), "dev": 0}}}),
            "empty quotas": plan({}),
            "not an object": [1, 2],
        }
        for name, value in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.load(value)

    def test_invalid_json_rejected(self):
        with self.assertRaises(ValueError):
            self.load("{not json")

    def test_v2_valid_with_hash_or_null(self):
        for stage in (None, "0123456789abcdef" * 4):
            with self.subTest(stage=stage):
                good = plan2(quotas(2), stage)
                self.assertEqual(self.load(good), good)

    def test_v2_rejections(self):
        def mutated(fn):
            value = plan2(quotas(2), "a" * 64)
            fn(value)
            return value
        bad = {
            "missing stage_sha256": mutated(lambda p: p.pop("stage_sha256")),
            "extra key": mutated(lambda p: p.update(extra=1)),
            "short hash": mutated(lambda p: p.update(stage_sha256="a" * 63)),
            "uppercase hash": mutated(lambda p: p.update(stage_sha256="A" * 64)),
            "non-hex hash": mutated(lambda p: p.update(stage_sha256="g" * 64)),
            "int hash": mutated(lambda p: p.update(stage_sha256=5)),
            "v2 quota rules still apply": mutated(lambda p: p.update(quotas={})),
            "v1 version with stage_sha256": mutated(lambda p: p.update(version=hf_pool.PLAN_VERSION)),
        }
        for name, value in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.load(value)


class SelectTests(unittest.TestCase):
    def test_deterministic_and_seed_sensitive(self):
        reg = registry({"hf-fake-a": supply("hf-fake-a", 8)})
        first, report = hf_pool.select(plan(quotas(2)), Path("/raw"), reg)
        again, report2 = hf_pool.select(plan(quotas(2)), Path("/raw"), reg)
        self.assertEqual(ids(first), ids(again))
        self.assertEqual(report, report2)
        self.assertTrue(all(len(rows) == 4 for rows in first.values()))
        other, _ = hf_pool.select(plan(quotas(2), seed=8), Path("/raw"), reg)
        self.assertNotEqual(ids(first), ids(other))

    def test_global_index_is_contiguous_in_visit_order(self):
        reg = registry({"hf-fake-a": supply("hf-fake-a", 3)})
        pools, _ = hf_pool.select(plan(quotas(2)), Path("/raw"), reg)
        indexes = sorted(int(p.actor.problem_id.split(":")[3]) for rows in pools.values() for p in rows)
        self.assertEqual(indexes, list(range(16)))

    def test_label_blind(self):
        base = supply("hf-fake-a", 8, gold="A")
        flipped = supply("hf-fake-a", 8, gold="B")
        self.assertNotEqual(base[0].gold, flipped[0].gold)
        a, ra = hf_pool.select(plan(quotas(2)), Path("/raw"), registry({"hf-fake-a": base}))
        b, rb = hf_pool.select(plan(quotas(2)), Path("/raw"), registry({"hf-fake-a": flipped}))
        self.assertEqual(ids(a), ids(b))
        self.assertEqual(ra["native_ids"], rb["native_ids"])
        # Stream order does not matter either: only the seed-keyed (source, native_id) hash does.
        c, _ = hf_pool.select(plan(quotas(2)), Path("/raw"), registry({"hf-fake-a": base[::-1]}))
        self.assertEqual({k: sorted(v) for k, v in ids(a).items()}, {k: sorted(v) for k, v in ids(c).items()})

    def test_quota_respected_with_shortfalls_and_ignored_subjects(self):
        extra = [cand("hf-fake-a", f"math-{i}", "Math", family_keys("hf-fake-a", 4)["train"][i]) for i in range(4)]
        reg = registry({"hf-fake-a": supply("hf-fake-a", 4) + extra})
        q = {"hf-fake-a": {"Biology": dict.fromkeys(SPLITS, 2), "Physics": dict.fromkeys(SPLITS, 6),
                           "Music": dict.fromkeys(SPLITS, 1), "Art": dict.fromkeys(SPLITS, 0)}}
        pools, report = hf_pool.select(plan(q), Path("/raw"), reg)
        for split in SPLITS:
            self.assertEqual(report["counts"][split], {"hf-fake-a": {"Biology": 2, "Physics": 4}})
            self.assertEqual(len(pools[split]), 6)
        self.assertEqual(report["shortfalls"],
                         [{"source": "hf-fake-a", "subject": subject, "split": split, "quota": quota, "accepted": accepted}
                          for subject, quota, accepted in (("Music", 1, 0), ("Physics", 6, 4))
                          for split in sorted(SPLITS)])
        self.assertEqual(report["dropped"], {"duplicate_image": 0, "duplicate_text": 0, "invalid": 0})
        self.assertFalse(any("math-" in nid for _, nid in report["native_ids"].values()))
        self.assertEqual(len(report["native_ids"]), 24)

    def test_duplicates_and_invalid_dropped_across_pool(self):
        ka, kb = family_keys("hf-fake-a", 4), family_keys("hf-fake-b", 4)
        shared = png("shared")
        a = [cand("hf-fake-a", "img-1", "Biology", ka["train"][0], image=shared),
             cand("hf-fake-a", "txt-1", "Biology", ka["train"][1], question="What  IS this?", choices=("Cat", "dog")),
             cand("hf-fake-a", "bad-1", "Biology", ka["train"][2], gold="Z")]
        b = [cand("hf-fake-b", "img-2", "Physics", kb["dev"][0], image=shared),   # same pixels, other split
             cand("hf-fake-b", "txt-2", "Physics", kb["dev"][1], question="what is this?", choices=("cat", "DOG"))]
        for split in ("sealed", "style_sealed"):
            a.append(cand("hf-fake-a", f"ok-{split}", "Biology", ka[split][0]))
        reg = registry({"hf-fake-a": a, "hf-fake-b": b})
        q = {"hf-fake-a": {"Biology": dict.fromkeys(SPLITS, 5)}, "hf-fake-b": {"Physics": dict.fromkeys(SPLITS, 5)}}
        pools, report = hf_pool.select(plan(q), Path("/raw"), reg)
        self.assertEqual(report["dropped"], {"duplicate_image": 1, "duplicate_text": 0, "invalid": 1})
        problems = [p for rows in pools.values() for p in rows]
        self.assertEqual(len(problems), 7 - 2)               # 7 candidates minus invalid, image dup
        natives = {nid for _, nid in report["native_ids"].values()}
        self.assertEqual(len(natives & {"img-1", "img-2"}), 1)
        self.assertEqual(natives & {"txt-1", "txt-2"}, {"txt-1", "txt-2"})   # same text, different images: both kept
        self.assertNotIn("bad-1", natives)
        self.assertEqual(len({hf.pixel_sha256(p.image_bytes[0]) for p in problems}), len(problems))

    def test_duplicates_do_not_count_toward_quota(self):
        keys = family_keys("hf-fake-a", 4)["train"]
        shared = png("shared")
        rows = [cand("hf-fake-a", "u", "Biology", keys[0]),
                cand("hf-fake-a", "d1", "Biology", keys[1], image=shared),
                cand("hf-fake-a", "d2", "Biology", keys[2], image=shared)]
        q = {"hf-fake-a": {"Biology": {"train": 2, "dev": 0, "sealed": 0, "style_sealed": 0}}}
        pools, report = hf_pool.select(plan(q), Path("/raw"), registry({"hf-fake-a": rows}))
        self.assertEqual(len(pools["train"]), 2)
        self.assertEqual(report["shortfalls"], [])
        self.assertEqual({sp: pools[sp] for sp in SPLITS if sp != "train"}, {"dev": [], "sealed": [], "style_sealed": []})

    def test_same_text_with_different_images_are_both_selected(self):
        keys = family_keys("hf-fake-a", 4)["train"]
        question, choices = "How many rings does the molecule contain?", ("1", "2")
        rows = [cand("hf-fake-a", "vis-1", "Biology", keys[0], question=question, choices=choices),
                cand("hf-fake-a", "vis-2", "Biology", keys[1], question=question, choices=choices)]
        q = {"hf-fake-a": {"Biology": {"train": 2, "dev": 0, "sealed": 0, "style_sealed": 0}}}
        pools, report = hf_pool.select(plan(q), Path("/raw"), registry({"hf-fake-a": rows}))
        self.assertEqual(report["dropped"], {"duplicate_image": 0, "duplicate_text": 0, "invalid": 0})
        self.assertEqual(sorted(nid for _, nid in report["native_ids"].values()), ["vis-1", "vis-2"])
        self.assertEqual(len(pools["train"]), 2)

    def test_heap_memory_is_bounded_by_quota_times_oversample(self):
        key = family_keys("hf-fake-a", 1)["train"][0]
        quota, total = 2, 10 * 2 * hf_pool.OVERSAMPLE
        cands = [cand("hf-fake-a", f"n{i}", "Biology", key) for i in range(total)]
        yielded = []

        def stream(path):
            for c in cands:
                yielded.append(c)
                yield c

        reg = {"hf-fake-a": (SPECS["hf-fake-a"], stream)}
        q = {"hf-fake-a": {"Biology": {"train": quota, "dev": 0, "sealed": 0, "style_sealed": 0}}}
        sizes = []
        real_push = heapq.heappush

        def spy(heap, item):
            real_push(heap, item)
            sizes.append(len(heap))

        with mock.patch.object(heapq, "heappush", spy):
            pools, report = hf_pool.select(plan(q), Path("/raw"), reg)
        self.assertEqual(len(yielded), total)
        self.assertEqual(max(sizes), quota * hf_pool.OVERSAMPLE)
        expected = sorted(cands, key=lambda c: hf.candidate_order(c, 7))[:quota]
        self.assertEqual(sorted(nid for _, nid in report["native_ids"].values()), sorted(c.native_id for c in expected))
        self.assertEqual(len(pools["train"]), quota)

    def test_iterator_receives_repo_directory_under_raw_root(self):
        seen = []
        hf_pool.select(plan(quotas(1)), Path("/raw"), registry({"hf-fake-a": supply("hf-fake-a", 1)}, seen))
        self.assertEqual(seen, [Path("/raw/org__fake-a")])


class StagedSelectTests(unittest.TestCase):
    def setUp(self):
        self.reg = registry({"hf-fake-a": supply("hf-fake-a", 3)})
        self.plan = plan2(quotas(2))
        self.base, self.base_report = hf_pool.select(self.plan, Path("/raw"), self.reg)

    def select(self, staged):
        return hf_pool.select(self.plan, Path("/raw"), self.reg, staged)

    def test_staged_items_join_their_own_split_after_hf(self):
        staged = staged_all()
        pools, report = self.select(staged)
        for (_, _, problem), split in zip(staged, SPLITS, strict=True):
            self.assertEqual(problem.lineage.split, split)
            self.assertEqual(pools[split][:-1], self.base[split])          # HF selection unchanged
            self.assertIs(pools[split][-1], problem)
            self.assertEqual(report["counts"][split]["gen-fake"], {"Biology": 1})
            self.assertEqual(report["counts"][split]["hf-fake-a"], self.base_report["counts"][split]["hf-fake-a"])
            self.assertEqual(report["native_ids"][problem.actor.problem_id], ["gen-fake", problem.actor.problem_id])
        self.assertEqual(report["dropped"], {"duplicate_image": 0, "duplicate_text": 0, "invalid": 0})
        self.assertEqual(report["shortfalls"], self.base_report["shortfalls"])

    def test_counts_accumulate_per_subject(self):
        staged = [staged_item("train", "a", index=100), staged_item("train", "b", index=101),
                  staged_item("train", "c", index=102, subject="Physics")]
        _, report = self.select(staged)
        self.assertEqual(report["counts"]["train"]["gen-fake"], {"Biology": 2, "Physics": 1})

    def test_dedup_against_hf_and_between_staged(self):
        hf_problem = self.base["train"][0]
        staged = [
            staged_item("train", "same-image", index=100, image=hf_problem.image_bytes[0]),
            # Same text as an HF row but another picture: kept (the text key includes the image).
            staged_item("dev", "same-text", index=101, question=self.base["dev"][0].actor.question.upper(),
                        choices=self.base["dev"][0].actor.choices),
            # Templated generator text with different images: both kept (text dedup is against HF rows only).
            staged_item("sealed", "first", index=102, question="Twin?"),
            staged_item("sealed", "second", index=103, question="  twin?", k=1),
            staged_item("style_sealed", "ok", index=104),
        ]
        pools, report = self.select(staged)
        self.assertEqual(report["dropped"], {"duplicate_image": 1, "duplicate_text": 0, "invalid": 0})
        kept = {pid for _, pid in report["native_ids"].values() if pid.startswith("gen-fake:")}
        self.assertEqual(kept, {staged[i][2].actor.problem_id for i in (1, 2, 3, 4)})
        self.assertEqual(len(pools["sealed"]), len(self.base["sealed"]) + 2)
        self.assertEqual(len(pools["train"]), len(self.base["train"]))
        self.assertEqual(len(pools["dev"]), len(self.base["dev"]) + 1)
        every_image = [hf.pixel_sha256(raw) for rows in pools.values() for p in rows for raw in p.image_bytes]
        self.assertEqual(len(set(every_image)), len(every_image))

    def test_collisions_with_hf_problems_rejected(self):
        victim = self.base["train"][0]
        same_path = GeneratedProblem(dataclasses.replace(victim.actor, problem_id="gen-fake:other"),
                                     victim.verifier, victim.lineage, victim.image_bytes)
        bad_split = GeneratedProblem(victim.actor, victim.verifier,
                                     dataclasses.replace(victim.lineage, split="bogus"), victim.image_bytes)
        for name, problem in {"problem id": victim, "image path": same_path, "unknown split": bad_split}.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.select([("gen-fake", "Biology", problem)])

    def test_no_staged_items_is_unchanged(self):
        again, report = hf_pool.select(self.plan, Path("/raw"), self.reg)
        self.assertEqual(ids(again), ids(self.base))
        self.assertEqual(report, self.base_report)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text(json.dumps(plan(quotas(2))))
        self.reg = registry({"hf-fake-a": supply("hf-fake-a", 3)})

    def test_build_writes_unscanned_candidate_build(self):
        (self.raw / "download_manifest.jsonl").write_text('{"row": 1}\n')
        out = self.root / "out"
        result = hf_pool.build(self.plan_path, self.raw, out, registry=self.reg)
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
        self.assertEqual((result["version"], result["status"], result["training_allowed"]),
                         (hf_pool.POOL_VERSION, "BUILT_UNSCANNED_NOT_APPROVED", False))
        self.assertEqual(json.loads((out / "pool_manifest.json").read_text()), result)
        self.assertEqual(result["plan_sha256"], sha(self.plan_path))
        self.assertEqual(result["raw_download_manifest_sha256"], sha(self.raw / "download_manifest.jsonl"))
        self.assertEqual(result["build_manifest_sha256"], sha(out / "manifest.yaml"))
        self.assertEqual(result["seed"], 7)
        self.assertIsNone(result["stage_manifest_sha256"])
        for name in ("hf_common.py", "hf_pool.py"):
            self.assertRegex(result["adapter_code_sha256"][name], r"^[0-9a-f]{64}$")
        self.assertEqual(result["adapter_code_sha256"]["hf_pool.py"], sha(Path(hf_pool.__file__)))
        manifest = json.loads((out / "manifest.yaml").read_text())
        self.assertFalse(manifest["training_allowed"])
        self.assertEqual(manifest["counts"], dict.fromkeys(SPLITS, 4))
        receipt = json.loads((out / "contamination" / "receipt_v2.json").read_text())
        self.assertEqual(receipt["status"], "BLOCKED_NOT_CHECKED")
        vault_ids = {a.problem_id for split in SPLITS for a in read_actors(out / split)}
        self.assertEqual(vault_ids, set(result["native_ids"]))
        self.assertEqual(len(vault_ids), 16)
        self.assertEqual(result["shortfalls"], [])

    def test_build_is_deterministic_across_output_dirs(self):
        a = hf_pool.build(self.plan_path, self.raw, self.root / "out-a", registry=self.reg)
        b = hf_pool.build(self.plan_path, self.raw, self.root / "out-b", registry=self.reg)
        self.assertEqual(list(a["native_ids"]), list(b["native_ids"]))
        self.assertEqual(a["counts"], b["counts"])
        self.assertIsNone(a["raw_download_manifest_sha256"])

    def test_build_refuses_empty_split_without_padding(self):
        no_style = [c for c in supply("hf-fake-a", 3) if hf.split_for(c) != "style_sealed"]
        out = self.root / "out"
        with self.assertRaisesRegex(ValueError, "style_sealed"):
            hf_pool.build(self.plan_path, self.raw, out, registry=registry({"hf-fake-a": no_style}))
        self.assertFalse(out.exists())

    def test_build_rejects_invalid_plan(self):
        self.plan_path.write_text(json.dumps(plan(quotas(2), seed=-3)))
        with self.assertRaises(ValueError):
            hf_pool.build(self.plan_path, self.raw, self.root / "out", registry=self.reg)

    def test_build_writes_item_index(self):
        out = self.root / "out"
        result = hf_pool.build(self.plan_path, self.raw, out, registry=self.reg)
        index_text = (out / "item_index.json").read_text()
        index = json.loads(index_text)
        self.assertEqual(index["version"], hf_pool.ITEM_INDEX_VERSION)
        self.assertEqual(index["split_manifests"], json.loads((out / "manifest.yaml").read_text())["split_manifests"])
        self.assertEqual(set(index["items"]), set(result["native_ids"]))
        subjects = {c.native_id: c.subject for c in supply("hf-fake-a", 3)}
        code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        seen = set()
        for split in SPLITS:
            lineage = read_lineage(out / split)
            for actor in read_actors(out / split):
                pid = actor.problem_id
                seen.add(pid)
                item = index["items"][pid]
                self.assertEqual(item["source"], lineage[pid].source)
                self.assertEqual(item["source_revision"], lineage[pid].source_revision)
                self.assertEqual(item["source_revision"], "a" * 40)
                self.assertEqual(item["split"], split)
                self.assertEqual(item["family"], actor.family)
                self.assertEqual(item["image_sha256"], list(actor.image_sha256))
                self.assertEqual(item["native_id"], result["native_ids"][pid][1])
                self.assertEqual(item["subject"], subjects[item["native_id"]])
                self.assertEqual(item["plan_version"], hf_pool.PLAN_VERSION)
                self.assertEqual(item["adapter_code_sha256"], code_sha)
        self.assertEqual(seen, set(index["items"]))
        self.assertNotIn("subjects", result)
        self.assertNotIn("subjects", json.loads((out / "pool_manifest.json").read_text()))
        for c in supply("hf-fake-a", 3):
            self.assertNotIn(c.question, index_text)
        self.assertNotIn("gold", index_text)


class StagedBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.stage = self.root / "stage"
        self.stage.mkdir()
        (self.stage / "manifest.json").write_text('{"stage": "authored"}\n')
        self.stage_sha = hashlib.sha256((self.stage / "manifest.json").read_bytes()).hexdigest()
        self.reg = registry({"hf-fake-a": supply("hf-fake-a", 3)})
        self.loaded = []

    def loader(self, directory):
        self.loaded.append(directory)
        return staged_all()

    def write_plan(self, value):
        path = self.root / "plan.json"
        path.write_text(json.dumps(value))
        return path

    def test_build_merges_staged_items_and_records_stage_hash(self):
        out = self.root / "out"
        path = self.write_plan(plan2(quotas(2), self.stage_sha))
        result = hf_pool.build(path, self.raw, out, registry=self.reg, stage_dir=self.stage, stage_loader=self.loader)
        self.assertEqual(self.loaded, [self.stage])
        self.assertEqual(result["stage_manifest_sha256"], self.stage_sha)
        self.assertEqual(json.loads((out / "pool_manifest.json").read_text()), result)
        vault_ids = {a.problem_id for split in SPLITS for a in read_actors(out / split)}
        self.assertEqual(vault_ids, set(result["native_ids"]))
        self.assertEqual(len(vault_ids), 16 + 4)
        self.assertEqual(sum(1 for src, _ in result["native_ids"].values() if src == "gen-fake"), 4)
        self.assertEqual({split: result["counts"][split]["gen-fake"] for split in SPLITS}, dict.fromkeys(SPLITS, {"Biology": 1}))
        self.assertEqual(json.loads((out / "manifest.yaml").read_text())["counts"], dict.fromkeys(SPLITS, 5))

    def test_stage_manifest_hash_mismatch_rejected(self):
        out = self.root / "out"
        cases = {"other hash": plan2(quotas(2), "b" * 64), "plan without stage": plan2(quotas(2), None),
                 "v1 plan": plan(quotas(2))}
        for name, value in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                hf_pool.build(self.write_plan(value), self.raw, out, registry=self.reg,
                              stage_dir=self.stage, stage_loader=self.loader)
        path = self.write_plan(plan2(quotas(2), self.stage_sha))
        (self.stage / "manifest.json").write_text('{"stage": "tampered"}\n')
        with self.assertRaises(ValueError):
            hf_pool.build(path, self.raw, out, registry=self.reg, stage_dir=self.stage, stage_loader=self.loader)
        self.assertEqual(self.loaded, [])
        self.assertFalse(out.exists())

    def test_staged_id_collision_aborts_build(self):
        first = hf_pool.select(plan2(quotas(2)), self.raw, self.reg)[0]["train"][0]
        path = self.write_plan(plan2(quotas(2), self.stage_sha))
        with self.assertRaises(ValueError):
            hf_pool.build(path, self.raw, self.root / "out", registry=self.reg, stage_dir=self.stage,
                          stage_loader=lambda d: [("gen-fake", "Biology", first)])
        self.assertFalse((self.root / "out").exists())

    def test_without_stage_dir_v2_and_v1_plans_build_identically(self):
        v1 = hf_pool.build(self.write_plan(plan(quotas(2))), self.raw, self.root / "o1", registry=self.reg)
        v2 = hf_pool.build(self.write_plan(plan2(quotas(2))), self.raw, self.root / "o2", registry=self.reg)
        self.assertEqual(list(v1["native_ids"]), list(v2["native_ids"]))
        self.assertEqual(v1["counts"], v2["counts"])
        self.assertIsNone(v2["stage_manifest_sha256"])
        self.assertNotIn("gen-fake", json.dumps(v2["counts"]))

    def test_cli_passes_stage_dir(self):
        fake = {"status": "S", "counts": {}, "shortfalls": []}
        for argv, expected in ((["--stage", str(self.stage)], self.stage), ([], None)):
            with self.subTest(argv), mock.patch.object(hf_pool, "build", return_value=fake) as build, \
                    contextlib.redirect_stdout(io.StringIO()):
                hf_pool.main(["build", "--plan", "p.json", "--raw-root", "r", "--output", "o", *argv])
            self.assertEqual(build.call_args.kwargs, {"stage_dir": expected})

    def test_item_index_records_staged_subject_and_stage_adapter(self):
        out = self.root / "out"
        path = self.write_plan(plan2(quotas(2), self.stage_sha))
        hf_pool.build(path, self.raw, out, registry=self.reg, stage_dir=self.stage, stage_loader=self.loader)
        items = json.loads((out / "item_index.json").read_text())["items"]
        staged = {pid: item for pid, item in items.items() if item["source"] == "gen-fake"}
        self.assertEqual(len(staged), 4)
        stage_code = hashlib.sha256((Path(hf_pool.__file__).parent / "gen_stage.py").read_bytes()).hexdigest()
        for pid, item in staged.items():
            with self.subTest(pid):
                self.assertEqual(item["subject"], "Biology")
                self.assertEqual(item["native_id"], pid)
                self.assertEqual(item["adapter_code_sha256"], stage_code)
                self.assertEqual(item["plan_version"], hf_pool.PLAN_V2)


class FakeOCR:
    """Stands in for exclusion_engines.OCR: returns the texts in turn, one per recognized image."""
    identity = {"engine": "fake-tesseract"}

    def __init__(self, *texts):
        self.texts, self.calls = texts or ("A 12\nB",), 0

    def recognize(self, image):
        text = self.texts[self.calls % len(self.texts)]
        self.calls += 1
        return {"text": text, "status": "COMPLETE" if text.strip() else "NO_TEXT"}


class PrintedTextOcrTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.out = self.root / "out"
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text(json.dumps(plan(quotas(2, ("hf-fake-a", "hf-fake-p")))))
        self.reg = registry({"hf-fake-a": supply("hf-fake-a", 3), "hf-fake-p": supply("hf-fake-p", 3)})

    def printed(self, out=None):
        verifiers = {pid: v.printed_text for split in SPLITS for pid, v in read_verifier((out or self.out) / split).items()}
        by_source = {}
        for pid, text in verifiers.items():
            by_source.setdefault(pid.split(":")[0], {})[pid] = text
        return by_source

    def test_only_program_hf_items_are_filled(self):
        ocr = FakeOCR()
        result = hf_pool.build(self.plan_path, self.raw, self.out, registry=self.reg, ocr=ocr)
        printed = self.printed()
        self.assertEqual(len(printed["hf-fake-p"]), 8)
        self.assertEqual(set(printed["hf-fake-p"].values()), {("A 12", "B")})
        self.assertEqual(set(printed["hf-fake-a"].values()), {()})
        self.assertEqual(ocr.calls, 8)
        self.assertEqual(result["printed_text_ocr"], {"engine_identity": FakeOCR.identity,
                                                      "problem_ids": sorted(printed["hf-fake-p"])})
        self.assertEqual(json.loads((self.out / "pool_manifest.json").read_text()), result)

    def test_blank_text_gives_single_empty_line_and_lines_are_stripped(self):
        hf_pool.build(self.plan_path, self.raw, self.out, registry=self.reg, ocr=FakeOCR(""))
        self.assertEqual(set(self.printed()["hf-fake-p"].values()), {("",)})
        hf_pool.build(self.plan_path, self.raw, self.root / "out2", registry=self.reg, ocr=FakeOCR("  A 12 \n\n  B\n"))
        self.assertEqual(set(self.printed(self.root / "out2")["hf-fake-p"].values()), {("A 12", "B")})

    def test_all_images_are_read_in_order(self):
        two = [dataclasses.replace(c, images=(c.images[0], png("second-" + c.native_id))) for c in supply("hf-fake-p", 3)]
        reg = registry({"hf-fake-a": supply("hf-fake-a", 3), "hf-fake-p": two})
        ocr = FakeOCR("first 1", "second 2")
        hf_pool.build(self.plan_path, self.raw, self.out, registry=reg, ocr=ocr)
        self.assertEqual(set(self.printed()["hf-fake-p"].values()), {("first 1", "second 2")})
        self.assertEqual(ocr.calls, 16)

    def test_missing_ocr_fails_before_anything_is_written(self):
        with self.assertRaisesRegex(ValueError, "tesseract-root"):
            hf_pool.build(self.plan_path, self.raw, self.out, registry=self.reg)
        self.assertFalse(self.out.exists())

    def test_human_only_pool_needs_no_ocr(self):
        self.plan_path.write_text(json.dumps(plan(quotas(2))))
        result = hf_pool.build(self.plan_path, self.raw, self.out, registry=self.reg)
        self.assertEqual(result["printed_text_ocr"], {"engine_identity": None, "problem_ids": []})

    def test_staged_generator_items_are_never_touched(self):
        stage = self.root / "stage"
        stage.mkdir()
        (stage / "manifest.json").write_text('{"stage": "authored"}\n')
        sha = hashlib.sha256((stage / "manifest.json").read_bytes()).hexdigest()
        program_stage = [(n, sub, dataclasses.replace(p, lineage=dataclasses.replace(p.lineage, label_provenance="PROGRAM")))
                         for n, sub, p in staged_all()]
        self.plan_path.write_text(json.dumps(plan2(quotas(2, ("hf-fake-a", "hf-fake-p")), sha)))
        ocr = FakeOCR()
        result = hf_pool.build(self.plan_path, self.raw, self.out, registry=self.reg, stage_dir=stage,
                               stage_loader=lambda d: program_stage, ocr=ocr)
        printed = self.printed()
        self.assertEqual(set(printed["gen-fake"].values()), {()})
        self.assertEqual(len(printed["gen-fake"]), 4)
        self.assertEqual(ocr.calls, 8)
        self.assertEqual(result["printed_text_ocr"]["problem_ids"], sorted(printed["hf-fake-p"]))

    def test_plan_pinning_a_stage_requires_the_stage_dir(self):
        self.plan_path.write_text(json.dumps(plan2(quotas(2, ("hf-fake-a", "hf-fake-p")), "a" * 64)))
        with self.assertRaisesRegex(ValueError, "stage"):
            hf_pool.build(self.plan_path, self.raw, self.out, registry=self.reg, ocr=FakeOCR())
        self.assertFalse(self.out.exists())

    def test_adapter_code_hash_covers_stage_and_generator_files(self):
        result = hf_pool.build(self.plan_path, self.raw, self.out, registry=self.reg, ocr=FakeOCR())
        here = Path(hf_pool.__file__).parent
        hashes = result["adapter_code_sha256"]
        self.assertEqual(hashes["gen_stage.py"], hashlib.sha256((here / "gen_stage.py").read_bytes()).hexdigest())
        files = sorted((here / "generators").glob("*.py"))
        self.assertTrue(files)
        for path in files:
            self.assertEqual(hashes[f"generators/{path.name}"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_cli_builds_the_pinned_ocr_engine_from_tesseract_root(self):
        fake = {"status": "S", "counts": {}, "shortfalls": []}
        engine = object()
        with mock.patch.object(hf_pool, "build", return_value=fake) as build, \
                mock.patch.object(exclusion_engines, "OCR", return_value=engine) as ocr_class, \
                contextlib.redirect_stdout(io.StringIO()):
            hf_pool.main(["build", "--plan", "p.json", "--raw-root", "r", "--output", "o", "--tesseract-root", "tess"])
        ocr_class.assert_called_once_with(Path("tess"))
        self.assertEqual(build.call_args.kwargs, {"stage_dir": None, "ocr": engine})


if __name__ == "__main__":
    unittest.main()
