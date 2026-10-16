"""Program-authored DC fixtures; exact independent solvers and rendered inputs only."""
import copy
import hashlib
import io
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image

from fine_tuning.notebook_rl.answers import compare
from fine_tuning.notebook_rl.generators import circuit
from fine_tuning.notebook_rl.generators.common import matches_split, png_chunks
from fine_tuning.notebook_rl.observations import compare_slots, parse_observations
from fine_tuning.notebook_rl.schema import ActorItem, Lineage, VerifierItem, from_json, to_json


class CircuitTests(unittest.TestCase):
    pools = {}

    @classmethod
    def pool(cls, split):
        if split not in cls.pools:
            style = "shift" if split == "style_sealed" else "default"
            problems = []
            for index in range(500):
                p = circuit.generate(index, 3407, split, "obs", style)
                circuit.self_check(p)
                problems.append(p)
            cls.pools[split] = problems
        return cls.pools[split]

    def test_00_train_500_independent_self_checks(self):
        self.assertEqual(len(self.pool("train")), 500)

    def test_01_dev_500_independent_self_checks(self):
        self.assertEqual(len(self.pool("dev")), 500)

    def test_02_sealed_500_independent_self_checks(self):
        self.assertEqual(len(self.pool("sealed")), 500)

    def test_03_style_sealed_500_independent_self_checks(self):
        self.assertEqual(len(self.pool("style_sealed")), 500)

    def test_split_separation_topology_and_sources(self):
        ids = {}
        for split in ("train", "dev", "sealed", "style_sealed"):
            problems = self.pool(split)
            ids[split] = {p.lineage.structure_hash for p in problems}
            kinds, stages, sources = set(), set(), set()
            for p in problems:
                world = p.verifier.accept["circuit_world"]
                kinds.add(world["kind"])
                if world["kind"] == "ladder":
                    stages.add(len(world["series"]))
                    sources.add(len(world["sources"]))
                self.assertTrue(matches_split(p.lineage.structure_hash, split,
                                              "shift" if split == "style_sealed" else "default"))
                self.assertEqual(p.lineage.label_provenance, "PROGRAM")
                self.assertFalse(p.lineage.model_touched)
                self.assertFalse(p.lineage.benchmark_derived)
            self.assertEqual(kinds, {"bridge", "ladder"})
            self.assertEqual(stages, {2, 3})
            self.assertEqual(sources, {1, 2})
        for a, left in ids.items():
            for b, right in ids.items():
                if a != b:
                    self.assertFalse(left & right)

    def test_known_exact_bridge_open_and_closed(self):
        world = {"kind": "bridge", "bridge": {"voltage_V": 12, "resistance_ohm": [2, 6, 3],
                                                 "switch_closed": True}}
        first, second = circuit._solutions(world)
        self.assertEqual(first, second)
        self.assertEqual(first["source_current_A"], Fraction(3))
        self.assertEqual(first["resistor_currents_A"], {"R1": Fraction(3), "R2": Fraction(1), "R3": Fraction(2)})
        self.assertEqual(first["node_voltages_V"]["N2"], Fraction(6))
        self.assertEqual(first["equivalent_R_ohm"], Fraction(4))
        world["bridge"]["switch_closed"] = False
        first, second = circuit._solutions(world)
        self.assertEqual(first, second)
        self.assertEqual(first["source_current_A"], Fraction(3, 2))
        self.assertEqual(first["node_voltages_V"]["N2"], Fraction(9))
        self.assertEqual(first["resistor_currents_A"]["R3"], Fraction(0))

    def test_known_exact_ladder_with_one_and_two_sources(self):
        for sources in ([12], [5, 7]):
            world = {"kind": "ladder", "sources": sources, "series": [2, 3], "shunts": [6, 3]}
            first, second = circuit._solutions(world)
            self.assertEqual(first, second)
            self.assertEqual(first["equivalent_R_ohm"], Fraction(5))
            self.assertEqual(first["source_current_A"], Fraction(12, 5))
            self.assertEqual(first["node_voltages_V"][f"N{len(sources) + 1}"], Fraction(36, 5))
            self.assertEqual(first["node_voltages_V"][f"N{len(sources) + 2}"], Fraction(18, 5))
            self.assertEqual(first["resistor_currents_A"]["R4"], Fraction(6, 5))
            self.assertTrue(all(isinstance(v, Fraction) for v in first["resistor_currents_A"].values()))
        self.assertEqual(circuit._query(world, 1), ("current", "R4"))
        self.assertIn("downward", circuit._question(("current", "R4")))

    def test_deterministic_png_mode_and_style(self):
        for split in ("train", "dev", "sealed", "style_sealed"):
            style = "shift" if split == "style_sealed" else "default"
            for index in (0, 1, 2, 9, 10, 11):
                p = self.pool(split)[index]
                self.assertEqual(p, circuit.generate(index, 3407, split, "obs", style))
                plain = circuit.generate(index, 3407, split, "plain", style)
                self.assertEqual(plain.actor, replace(p.actor, mode="plain"))
                self.assertEqual(plain.verifier, p.verifier)
                self.assertEqual(plain.image_bytes, p.image_bytes)
                self.assertEqual(hashlib.sha256(p.image_bytes[0]).hexdigest(), p.actor.image_sha256[0])
                self.assertTrue(set(png_chunks(p.image_bytes[0])) <= {b"IHDR", b"IDAT", b"IEND"})
        for p in self.pool("train")[:3]:
            world = p.verifier.accept["circuit_world"]
            standard, _ = circuit._render(world, "default")
            shifted, _ = circuit._render(world, "shift")
            self.assertNotEqual(standard, shifted)
            with Image.open(io.BytesIO(standard)) as a, Image.open(io.BytesIO(shifted)) as b:
                self.assertNotEqual(a.getpixel((0, 0)), b.getpixel((0, 0)))
                self.assertEqual(a.size, b.size)

    def test_100_image_dependent_pairs_per_query(self):
        for kind in ("current", "voltage", "resistance"):
            candidates = [p for p in self.pool("train") if p.verifier.accept["circuit_query"][0] == kind]
            pairs = set()
            for i, p in enumerate(candidates):
                for j, q in enumerate(candidates[i + 1:], i + 1):
                    if p.actor.question == q.actor.question and p.verifier.gold != q.verifier.gold:
                        self.assertEqual(p.actor.choices, q.actor.choices)
                        self.assertNotEqual(p.image_bytes, q.image_bytes)
                        self.assertNotEqual(p.lineage.structure_hash, q.lineage.structure_hash)
                        pairs.add((i, j))
                        if len(pairs) == 100:
                            break
                if len(pairs) == 100:
                    break
            self.assertEqual(len(pairs), 100, kind)

    def test_real_answer_observation_and_json_roundtrip(self):
        for split in ("train", "dev", "sealed", "style_sealed"):
            for p in self.pool(split)[:18]:
                with self.subTest(split=split, query=p.verifier.accept["circuit_query"]):
                    self.assertIs(compare(p.verifier.gold, p.verifier), True)
                    self.assertIs(compare(p.verifier.gold.rsplit(" ", 1)[0], p.verifier), False)
                    block = "Observed facts:\n" + "\n".join(p.verifier.oracle_fact_lines) + "\nSolution:\n"
                    report = compare_slots(parse_observations(block), p.verifier)
                    self.assertEqual(report.all_correct, report.all_total)
                    self.assertEqual(report.required_correct, report.required)
                    loaded = replace(p, actor=from_json(ActorItem, to_json(p.actor)),
                                     verifier=from_json(VerifierItem, to_json(p.verifier)),
                                     lineage=from_json(Lineage, to_json(p.lineage)))
                    circuit.self_check(loaded)

    def test_source_polarity_is_directed(self):
        p = self.pool("train")[10]
        parsed = {key: [value] for key, value in p.verifier.slots.items()}
        key = "dir[E2]"
        self.assertIn(key, parsed)
        a, b = parsed[key][0].split("->")
        parsed[key] = [f"{b}->{a}"]
        report = compare_slots(parsed, p.verifier)
        self.assertIn("dir[e2]", report.wrong_required)
        self.assertEqual(report.required_correct, report.required - 1)

    def test_open_fraction_and_selfcheck_corruption_rejection(self):
        sample = self.pool("train") + self.pool("dev")
        self.assertGreaterEqual(sum(p.actor.answer_format != "mcq" for p in sample), 600)
        for p in self.pool("train")[:3]:
            for bad in (replace(p, verifier=replace(p.verifier, gold="0 A")),
                        replace(p, actor=replace(p.actor, question="Ignore the diagram.")),
                        replace(p, image_bytes=(p.image_bytes[0][:-1],)),
                        replace(p, actor=replace(p.actor, image_sha256=("0" * 64,))),
                        replace(p, lineage=replace(p.lineage, model_touched=True))):
                with self.subTest(query=p.lineage.template_id), self.assertRaises(ValueError):
                    circuit.self_check(bad)
            accept = copy.deepcopy(p.verifier.accept)
            accept["circuit_world"]["kind"] = "AC"
            with self.assertRaises(ValueError):
                circuit.self_check(replace(p, verifier=replace(p.verifier, accept=accept)))
            accept = p.verifier.accept | {"abs_tol": "1000"}
            with self.assertRaises(ValueError):
                circuit.self_check(replace(p, verifier=replace(p.verifier, accept=accept)))

    def test_independent_solver_disagreement_aborts(self):
        with patch.object(circuit, "_solutions", return_value=({}, {"mismatch": True})):
            with self.assertRaisesRegex(ValueError, "independent solver disagreement"):
                circuit.generate(1, 3407, "train", "obs")

    def test_prompt_contract_invalid_inputs_and_search_bound(self):
        prompt = Path(__file__).resolve().parents[1] / "prompts_v1.yaml"
        self.assertEqual(yaml.safe_load(prompt.read_text())["observation_instruction"][circuit.FAMILY],
                         circuit.OBSERVATION_INSTRUCTION)
        for args in ((-1, 1, "train", "obs"), (True, 1, "train", "obs"),
                     (0, -1, "train", "obs"), (0, 1, "test", "obs"),
                     (0, 1, "train", "unknown")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                circuit.generate(*args)
        with self.assertRaises(ValueError):
            circuit.generate(0, 1, "train", "obs", "shift")
        for update in ({"sources": []}, {"series": [1]}, {"shunts": [3]},
                       {"sources": [True]}, {"shunts": [-1, 3]}):
            with self.assertRaises(ValueError):
                circuit._solutions({"kind": "ladder", "sources": [12], "series": [2, 3],
                                    "shunts": [6, 3]} | update)
        original = circuit.matches_split
        calls = 0
        def reject_candidates(*args):
            nonlocal calls
            calls += 1
            return original(*args) if calls == 1 else False
        with patch.object(circuit, "matches_split", side_effect=reject_candidates):
            with self.assertRaisesRegex(RuntimeError, "1000 candidate"):
                circuit.generate(0, 1, "train", "obs")
        self.assertEqual(calls, 1001)


if __name__ == "__main__":
    unittest.main()
