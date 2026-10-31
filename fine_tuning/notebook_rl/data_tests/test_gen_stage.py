"""CPU-only tests for gen_stage: program generators in the data env, no GPU or network."""
import copy
import shutil
import tempfile
import unittest
from pathlib import Path

from fine_tuning.notebook_rl.dataset import generate_pool
from fine_tuning.notebook_rl.gen_stage import STAGE_VERSION, load_stage, stage
from fine_tuning.notebook_rl.schema import to_json

SEED = 7
SPEC = {
    "version": "gen-stage-spec-v1",
    "generators": {
        "graph": {"subject": "Graph Theory", "seed": SEED, "mode": "obs", "counts": {"train": 2, "dev": 1}},
        "statics": {"subject": "Statics", "seed": SEED, "mode": "plain", "counts": {"train": 1}},
    },
    "exclude_structure_hashes": [],
}


def canon(problem):
    """Compare canonical wire forms, the representation that is staged and reloaded."""
    return (to_json(problem.actor), to_json(problem.verifier), to_json(problem.lineage),
            tuple(problem.image_bytes))


def spec_with(**overrides):
    spec = copy.deepcopy(SPEC)
    spec.update(overrides)
    return spec


class GenStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name) / "stage"
        cls.manifest = stage(SPEC, cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_reload_matches_direct_generate_pool(self):
        graph = generate_pool("graph", 2, SEED, "train", 1.0) + generate_pool("graph", 1, SEED, "dev", 1.0)
        statics = generate_pool("statics", 1, SEED, "train", 0.0)
        expected = [("graph", "Graph Theory", p) for p in graph] + [("statics", "Statics", p) for p in statics]
        loaded = load_stage(self.base)
        self.assertEqual([(g, s, canon(p)) for g, s, p in loaded],
                         [(g, s, canon(p)) for g, s, p in expected])
        self.assertEqual({p.actor.mode for g, _, p in loaded if g == "graph"}, {"obs"})
        self.assertEqual({p.actor.mode for g, _, p in loaded if g == "statics"}, {"plain"})
        self.assertEqual(self.manifest["version"], STAGE_VERSION)
        self.assertEqual(self.manifest["status"], "STAGED_UNSCANNED_NOT_APPROVED")
        self.assertEqual(self.manifest["excluded"], {"graph": 0, "statics": 0})

    def test_exclusion_drops_hash_and_keeps_next_structure(self):
        direct = generate_pool("graph", 3, SEED, "train", 1.0)
        spec = spec_with(
            generators={"graph": {"subject": "Graph Theory", "seed": SEED, "mode": "obs",
                                  "counts": {"train": 2}}},
            exclude_structure_hashes=[direct[0].lineage.structure_hash])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            manifest = stage(spec, out)
            loaded = load_stage(out)
        self.assertEqual([canon(p) for _, _, p in loaded], [canon(p) for p in direct[1:3]])
        self.assertEqual(manifest["excluded"], {"graph": 1})

    def test_refuses_existing_output_directory(self):
        with self.assertRaises(FileExistsError):
            stage(SPEC, self.base)

    def test_tampered_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / "copy"
            shutil.copytree(self.base, staged)
            image = next(p for p in (staged / "images").rglob("*") if p.is_file())
            image.write_bytes(image.read_bytes() + b"\0")
            with self.assertRaises(ValueError):
                load_stage(staged)

    def test_tampered_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / "copy"
            shutil.copytree(self.base, staged)
            records = staged / "records.jsonl"
            raw = records.read_bytes()
            self.assertIn(b'"train"', raw)
            records.write_bytes(raw.replace(b'"train"', b'"dev"', 1))
            with self.assertRaises(ValueError):
                load_stage(staged)

    def test_invalid_spec_writes_nothing(self):
        bad_specs = [
            spec_with(version="gen-stage-spec-v0"),
            spec_with(exclude_structure_hashes=["ABCDEF0123456789"]),
            spec_with(generators={"no_such_generator": {"subject": "X", "seed": 0, "mode": "obs",
                                                        "counts": {"train": 1}}}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            for bad in bad_specs:
                with self.subTest(spec=bad["version"] if bad["version"] != SPEC["version"] else "other"):
                    with self.assertRaises(ValueError):
                        stage(bad, out)
                    self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
