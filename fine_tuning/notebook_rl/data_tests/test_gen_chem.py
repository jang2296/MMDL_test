"""Self-authored 50-row molecule fixture; not an acquired or approved ChEBI corpus."""
import hashlib
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml
from rdkit import Chem

from fine_tuning.notebook_rl.answers import compare
from fine_tuning.notebook_rl.generators import chem
from fine_tuning.notebook_rl.generators.common import matches_split, png_chunks
from fine_tuning.notebook_rl.observations import compare_slots, parse_observations
from fine_tuning.notebook_rl.schema import ActorItem, Lineage, VerifierItem, from_json, to_json


class ChemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "authored_fixture.tsv"
        smiles = [("C" * n) + suffix for n in range(6, 16)
                  for suffix in ("", "O", "N", "C(=O)O", "c1ccccc1")]
        cls.path.write_text("chebi_id\tsmiles\n" + "".join(
            f"CHEBI:{900000 + i}\t{s}\n" for i, s in enumerate(smiles)))
        cls.env = patch.dict(os.environ, {"NOTEBOOK_RL_CHEBI_TSV": str(cls.path)})
        cls.env.start()
        cls.problems = [chem.generate(i, 3407, "train", "obs") for i in range(100)]

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.temp.cleanup()

    def test_determinism_and_modes(self):
        for i in (0, 1, 2, 3, 17, 56):
            p = self.problems[i]
            self.assertEqual(p, chem.generate(i, 3407, "train", "obs"))
            plain = chem.generate(i, 3407, "train", "plain")
            self.assertEqual(p.image_bytes, plain.image_bytes)
            self.assertEqual(p.verifier, plain.verifier)

    def test_100_selfchecks_and_real_consumer(self):
        for p in self.problems:
            chem.self_check(p)
            self.assertTrue(compare(p.verifier.gold, p.verifier))
            text = "Observed facts:\n" + "\n".join(p.verifier.oracle_fact_lines) + "\nSolution:"
            report = compare_slots(parse_observations(text), p.verifier)
            self.assertEqual(report.all_correct, len(p.verifier.slots))
            self.assertEqual(report.required_correct, len(p.verifier.required_slots))
            self.assertEqual(p.actor.answer_format == "mcq", False)

    def test_splits_and_shift(self):
        seen = {k: set() for k in ("train", "dev", "sealed", "style_sealed")}
        for split in seen:
            for i in range(12):
                p = chem.generate(i, 3407, split, "obs", "shift" if split == "style_sealed" else "default")
                seen[split].add(p.lineage.structure_hash)
                self.assertTrue(matches_split(p.lineage.structure_hash, split,
                                             "shift" if split == "style_sealed" else "default"))
                chem.self_check(p)
        for a in seen:
            for b in seen:
                if a != b:
                    self.assertTrue(seen[a].isdisjoint(seen[b]))
        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        self.assertNotEqual(chem._render(mol, "default"), chem._render(mol, "shift"))

    def test_source_binding_and_png_no_metadata(self):
        sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        for p in self.problems[:12]:
            self.assertIn("chebi_tsv_sha256:" + sha, p.lineage.parents)
            self.assertEqual(p.lineage.upstream_origin, "ChEBI")
            png_chunks(p.image_bytes[0])
            self.assertEqual(hashlib.sha256(p.image_bytes[0]).hexdigest(), p.actor.image_sha256[0])

    def test_formula_ring_and_donor_independent_known_values(self):
        p = next(p for p in self.problems if p.lineage.template_id == "formula")
        self.assertTrue(p.verifier.accept["case_sensitive"])
        self.assertFalse(compare(p.verifier.gold.lower(), p.verifier))
        examples = (("CCCCCC", "C6H14", "0", "0", "0"),
                    ("CCCCC(=O)O", "C5H10O2", "0", "1", "1"),
                    ("c1ccccc1", "C6H6", "1", "0", "0"),
                    ("NCCCCC", "C5H13N", "0", "1", "0"),
                    ("C1C2CC3CC1CC(C2)C3", "C10H16", "3", "0", "0"))
        for smiles, *values in examples:
            mol = Chem.MolFromSmiles(smiles)
            for task, value in zip(chem._TASKS, values, strict=True):
                self.assertEqual(chem._answers(mol, task), (value, value))

    def test_same_template_different_image_answer_witnesses(self):
        for task in ("formula", "donors", "carboxyl", "rings"):
            pool = [p for p in self.problems if p.lineage.template_id == task]
            pair = next(((a, b) for a in pool for b in pool if a.verifier.gold != b.verifier.gold), None)
            self.assertIsNotNone(pair)
            a, b = pair
            self.assertEqual(a.actor.question, b.actor.question)
            self.assertNotEqual(a.image_bytes, b.image_bytes)
        # Witnesses prove input dependence exists, not blind-model difficulty or approval.

    def test_roundtrip_selfcheck_and_tampering(self):
        p = self.problems[0]
        restored = replace(p, actor=from_json(ActorItem, to_json(p.actor)),
                           verifier=from_json(VerifierItem, to_json(p.verifier)),
                           lineage=from_json(Lineage, to_json(p.lineage)))
        chem.self_check(restored)
        with self.assertRaises(ValueError):
            chem.self_check(replace(p, verifier=replace(p.verifier, gold="wrong")))
        with patch.object(chem, "_answers", return_value=("1", "2")):
            with self.assertRaises(ValueError):
                chem.generate(0, 3407, "train", "obs")

    def test_source_validation_and_no_download(self):
        for raw in (b"bad\theader\n", b"chebi_id\tsmiles\nCHEBI:1\tnot_smiles\n",
                    b"chebi_id\tsmiles\nCHEBI:1\tCCCCCC\nCHEBI:1\tCCCCCCC\n",
                    b"chebi_id\tsmiles\nCHEBI:1\tCC\n"):
            with self.assertRaises(ValueError):
                chem._source(raw)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                chem.generate(0, 3407, "train", "obs")

    def test_atom_mapping_is_not_a_new_structure_family(self):
        raw = ("chebi_id\tsmiles\n" + "".join(
            f"CHEBI:{i}\t[CH3:{i}]CCCCC\n" for i in range(1, 30))).encode()
        self.assertEqual(chem._source(raw), (("CHEBI:1", "CCCCCC"),))

    def test_missing_split_does_not_borrow_structure(self):
        candidates = (("CHEBI:1", "CCCCCC"),)
        with patch.object(chem, "_load_source", return_value=(b"fixture", candidates)):
            p = chem.generate(0, 3407, "train", "obs")
            self.assertEqual(p.lineage.split, "train")
            with self.assertRaises(RuntimeError):
                chem.generate(0, 3407, "style_sealed", "obs", "shift")

    def test_prompt_contract(self):
        path = Path(chem.__file__).resolve().parents[1] / "prompts_v1.yaml"
        self.assertEqual(yaml.safe_load(path.read_text())["observation_instruction"][chem.FAMILY],
                         chem.OBSERVATION_INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
