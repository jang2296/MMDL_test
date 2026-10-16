"""Program-authored graph fixtures; CPU generation, no benchmark/model/network access."""
import copy
import hashlib
import io
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import yaml
from PIL import Image

from fine_tuning.notebook_rl.generators import graph
from fine_tuning.notebook_rl.generators.common import matches_split, png_chunks
from fine_tuning.notebook_rl.answers import compare
from fine_tuning.notebook_rl.observations import compare_slots, parse_observations
from fine_tuning.notebook_rl.schema import ActorItem, Lineage, VerifierItem, from_json, to_json


class GraphTests(unittest.TestCase):
    pools = {}

    @classmethod
    def pool(cls, split):
        if split not in cls.pools:
            style = "shift" if split == "style_sealed" else "default"
            problems = []
            for index in range(500):
                p = graph.generate(index, 3407, split, "obs", style)
                graph.self_check(p)
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

    def test_split_structures_are_disjoint_and_all_tasks_present(self):
        ids = {}
        for split in ("train", "dev", "sealed", "style_sealed"):
            problems = self.pool(split)
            ids[split] = {p.lineage.structure_hash for p in problems}
            counts = Counter(p.verifier.accept["graph_world"]["kind"] for p in problems)
            self.assertEqual(counts, dict.fromkeys(("shortest", "bfs", "dfa", "tree"), 125))
            for p in problems:
                self.assertTrue(matches_split(p.lineage.structure_hash, split,
                                              "shift" if split == "style_sealed" else "default"))
                self.assertFalse(p.lineage.benchmark_derived)
                self.assertFalse(p.lineage.model_touched)
                self.assertEqual(p.lineage.label_provenance, "PROGRAM")
        for first, a in ids.items():
            for second, b in ids.items():
                if first != second:
                    self.assertFalse(a & b, (first, second))

    def test_deterministic_png_and_mode_independence(self):
        for split in ("train", "dev", "sealed", "style_sealed"):
            style = "shift" if split == "style_sealed" else "default"
            for index in range(4):
                with self.subTest(split=split, index=index):
                    p = self.pool(split)[index]
                    self.assertEqual(p, graph.generate(index, 3407, split, "obs", style))
                    plain = graph.generate(index, 3407, split, "plain", style)
                    self.assertEqual(plain.image_bytes, p.image_bytes)
                    self.assertEqual(plain.verifier, p.verifier)
                    self.assertEqual(plain.lineage.structure_hash, p.lineage.structure_hash)
                    self.assertEqual(plain.actor, replace(p.actor, mode="plain"))
                    for raw, expected in zip(p.image_bytes, p.actor.image_sha256, strict=True):
                        self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
                        self.assertTrue(set(png_chunks(raw)) <= {b"IHDR", b"IDAT", b"IEND"})

    def test_style_changes_font_color_layout_but_not_the_answer(self):
        for p in self.pool("train")[:4]:
            world = p.verifier.accept["graph_world"]
            standard, _ = graph._render(world, "default")
            shifted, _ = graph._render(world, "shift")
            self.assertNotEqual(standard, shifted)
            with Image.open(io.BytesIO(standard)) as a, Image.open(io.BytesIO(shifted)) as b:
                self.assertNotEqual(a.getpixel((0, 0)), b.getpixel((0, 0)))
                self.assertEqual(a.size, b.size)
            self.assertEqual(graph._answers(world)[0], graph._answers(world)[1])

    def test_100_image_dependent_pairs_per_task_same_question_and_choices(self):
        for kind in ("shortest", "bfs", "dfa", "tree"):
            candidates = [p for p in self.pool("train") if p.verifier.accept["graph_world"]["kind"] == kind]
            found = set()
            for i, p in enumerate(candidates):
                for j, q in enumerate(candidates[i + 1:], i + 1):
                    if (p.actor.question == q.actor.question and p.actor.choices == q.actor.choices
                            and p.verifier.gold != q.verifier.gold):
                        self.assertNotEqual(p.image_bytes, q.image_bytes)
                        self.assertNotEqual(p.lineage.structure_hash, q.lineage.structure_hash)
                        found.add((i, j))
                        if len(found) == 100:
                            break
                if len(found) == 100:
                    break
            self.assertEqual(len(found), 100, kind)

    def test_open_answer_fraction_over_1000(self):
        problems = self.pool("train") + self.pool("dev")
        self.assertGreaterEqual(sum(p.actor.answer_format != "mcq" for p in problems), 600)
        self.assertTrue(all(not p.actor.choices for p in problems))

    def test_real_answer_observation_contract_and_json_roundtrip(self):
        for split in ("train", "dev", "sealed", "style_sealed"):
            for p in self.pool(split)[:16]:
                with self.subTest(split=split, template=p.lineage.template_id):
                    self.assertIs(compare(p.verifier.gold, p.verifier), True)
                    block = "Observed facts:\n" + "\n".join(p.verifier.oracle_fact_lines) + "\nSolution:\n"
                    report = compare_slots(parse_observations(block), p.verifier)
                    self.assertEqual(report.required_correct, report.required)
                    self.assertEqual(report.all_correct, report.all_total)
                    self.assertEqual(report.conflicting, ())
                    loaded = replace(p, actor=from_json(ActorItem, to_json(p.actor)),
                                     verifier=from_json(VerifierItem, to_json(p.verifier)),
                                     lineage=from_json(Lineage, to_json(p.lineage)))
                    graph.self_check(loaded)

    def test_direction_and_explicit_queue_tie_order(self):
        shortest = {"kind": "shortest", "edges": [
            ["A", "B", 5], ["B", "H", 5], ["H", "C", 1], ["C", "A", 1]]}
        self.assertEqual(graph._answers(shortest), ("10", "10"))
        self.assertEqual(nx.shortest_path_length(graph._network(shortest, undirected=True),
                                               "A", "H", weight="weight"), 2)
        self.assertIsNone(graph._shortcut_reason(shortest))
        bfs = {"kind": "bfs", "edges": [["A", "D", 1], ["A", "B", 1],
                                         ["B", "C", 1], ["E", "A", 1]]}
        self.assertEqual(graph._answers(bfs), ("A,B,D,C", "A,B,D,C"))
        self.assertIsNone(graph._shortcut_reason(bfs))
        self.assertIn("ascending alphabetical order", graph._question("bfs", "postorder"))
        self.assertIn("outgoing arrows only", graph._question("bfs", "postorder"))

    def test_known_dfa_and_all_tree_traversals(self):
        dfa = {"kind": "dfa", "start": "q0", "accepting": ["q1"], "trans": {
            "q0": {"0": "q1", "1": "q0"}, "q1": {"0": "q1", "1": "q2"},
            "q2": {"0": "q3", "1": "q1"}, "q3": {"0": "q2", "1": "q0"}}}
        self.assertEqual(graph._answers(dfa), ("accept", "accept"))
        self.assertEqual(graph._answers(dfa | {"accepting": ["q3"]}), ("reject", "reject"))
        tree = {"kind": "tree", "root": "D", "children": {
            "D": {"L": "B", "R": "F"}, "B": {"L": "A", "R": "C"},
            "F": {"L": "E", "R": "G"}}}
        for kind, answer in (("preorder", "D,B,A,C,F,E,G"), ("inorder", "A,B,C,D,E,F,G"),
                             ("postorder", "A,C,B,E,G,F,D")):
            self.assertEqual(graph._answers(tree, kind), (answer, answer))
            self.assertIn("L and R label", graph._question("tree", kind))

    def test_shortcut_rejection_and_bounded_search(self):
        self.assertEqual(graph._shortcut_reason({"kind": "shortest", "edges": [["A", "H", 1]]}),
                         "ONE_EDGE_SHORTCUT")
        self.assertEqual(graph._shortcut_reason({"kind": "shortest", "edges": [
            ["A", "B", 1], ["B", "H", 1]]}), "DIRECTION_FREE_SHORTCUT")
        self.assertEqual(graph._shortcut_reason({"kind": "bfs", "edges": [["A", "B", 1]]}),
                         "ONE_STEP_OR_ALPHABETICAL_SHORTCUT")
        dfa = {"kind": "dfa", "start": "q0", "trans": {"q0": {"0": "q0", "1": "q0"}}}
        self.assertEqual(graph._shortcut_reason(dfa), "TRIVIAL_DFA_TRACE")
        with patch.object(graph, "_sample", return_value=None) as sample:
            with self.assertRaisesRegex(RuntimeError, "1000 candidate"):
                graph.generate(0, 7, "train", "plain")
            self.assertEqual(sample.call_count, 1000)
        for split in ("train", "dev", "sealed", "style_sealed"):
            for p in self.pool(split):
                self.assertIsNone(graph._shortcut_reason(p.verifier.accept["graph_world"]))

    def test_selfcheck_rejects_poisoned_answer_pixels_facts_and_structure(self):
        for p in self.pool("train")[:4]:
            bad_slots = p.verifier.slots | {"unexpected": "private"}
            changes = (replace(p, verifier=replace(p.verifier, gold="wrong")),
                       replace(p, verifier=replace(p.verifier, slots=bad_slots,
                                                  slot_types=p.verifier.slot_types | {"unexpected": "text"})),
                       replace(p, image_bytes=(p.image_bytes[0][:-1],)),
                       replace(p, actor=replace(p.actor, question="Give the fixed answer.")),
                       replace(p, actor=replace(p.actor, image_sha256=("0" * 64,))),
                       replace(p, lineage=replace(p.lineage, model_touched=True)))
            for bad in changes:
                with self.subTest(template=p.lineage.template_id), self.assertRaises(ValueError):
                    graph.self_check(bad)
            accept = copy.deepcopy(p.verifier.accept)
            accept["graph_world"]["kind"] = "unsupported"
            with self.assertRaises(ValueError):
                graph.self_check(replace(p, verifier=replace(p.verifier, accept=accept)))
        with patch.object(graph, "_answers", return_value=("1", "2")):
            with self.assertRaisesRegex(ValueError, "independent solver"):
                graph.generate(0, 3407, "train", "obs")

    def test_prompt_contract_and_argument_validation(self):
        prompt = Path(__file__).resolve().parents[1] / "prompts_v1.yaml"
        self.assertEqual(yaml.safe_load(prompt.read_text())["observation_instruction"][graph.FAMILY],
                         graph.OBSERVATION_INSTRUCTION)
        for args in ((-1, 1, "train", "obs"), (True, 1, "train", "obs"),
                     (0, -1, "train", "obs"), (0, 1, "test", "obs"),
                     (0, 1, "train", "unknown")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                graph.generate(*args)
        for split, style in (("train", "shift"), ("style_sealed", "default"), ("train", "bad")):
            with self.assertRaises(ValueError):
                graph.generate(0, 1, split, "obs", style)

    def test_selfcheck_freezes_comparison_policy_and_provenance_metadata(self):
        for index in range(12):
            p = graph.generate(index, 3407, "train", "obs")
            bad = [replace(p, verifier=replace(p.verifier, accept=p.verifier.accept | {"abs_tol": "999"})),
                   replace(p, verifier=replace(p.verifier, oracle_fact_lines=("invented = fact",))),
                   replace(p, lineage=replace(p.lineage, generator_version="unapproved")),
                   replace(p, lineage=replace(p.lineage, source_revision="unapproved")),
                   replace(p, lineage=replace(p.lineage, image_provenance="different renderer"))]
            if p.actor.answer_format == "list":
                bad.extend([replace(p, verifier=replace(p.verifier, accept=p.verifier.accept | change))
                            for change in ({"ordered": False}, {"ordered": 1}, {"element_type": "int"})])
            for item in bad:
                with self.subTest(index=index), self.assertRaises(ValueError):
                    graph.self_check(item)
            graph.self_check(p)
        tree = {"kind": "tree", "root": "A", "children": {
            "A": {"L": "B", "R": "C"}, "B": {"L": "D", "R": "E"},
            "C": {"L": "F", "R": "F"}, "F": {"R": "G"}}}
        with self.assertRaisesRegex(ValueError, "binary tree"):
            graph._validate_world(tree)


if __name__ == "__main__":
    unittest.main()
