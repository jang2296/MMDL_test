"""Real program renders and independent solvers; no model/benchmark inputs."""
import copy
import hashlib
import io
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from fine_tuning.notebook_rl.answers import compare
from fine_tuning.notebook_rl.dataset import generate_pool
from fine_tuning.notebook_rl.generators import get_generator, graph, graph_pair
from fine_tuning.notebook_rl.generators.common import matches_split, png_chunks, structure_hash
from fine_tuning.notebook_rl.observations import compare_slots, parse_observations
from fine_tuning.notebook_rl.prompts import load_prompts, turn1_text


class GraphPairTests(unittest.TestCase):
    def test_deterministic_real_images_and_matching_graph_family_splits(self):
        families = {}
        self.assertIs(get_generator("graph_pair"), graph_pair)
        for split in ("train", "dev", "sealed", "style_sealed"):
            style = "shift" if split == "style_sealed" else "default"
            families[split] = set()
            for i in range(4):
                p = graph_pair.generate(i, 3407, split, "obs", style)
                graph_pair.self_check(p)
                self.assertEqual(p, graph_pair.generate(i, 3407, split, "obs", style))
                single = graph.generate(i * 4, 3407, split, "obs", style)
                self.assertEqual(p.lineage.structure_hash, single.lineage.structure_hash)
                self.assertEqual(p.verifier.gold, single.verifier.gold)
                self.assertEqual(p.actor.family, single.actor.family)
                self.assertTrue(matches_split(p.lineage.structure_hash, split, style))
                families[split].add(p.lineage.structure_hash)
                self.assertEqual(len(p.image_bytes), 2)
                self.assertNotEqual(*p.image_bytes)
                self.assertEqual(graph_pair.generate(i, 3407, split, "plain", style),
                                 replace(p, actor=replace(p.actor, mode="plain")))
                for j, raw in enumerate(p.image_bytes):
                    self.assertEqual(p.actor.image_sha256[j], hashlib.sha256(raw).hexdigest())
                    self.assertTrue(set(png_chunks(raw)) <= {b"IHDR", b"IDAT", b"IEND"})
                    with Image.open(io.BytesIO(raw)) as image:
                        self.assertEqual(image.size, (1000, 760))
        for left in families:
            for right in families:
                if left != right:
                    self.assertFalse(families[left] & families[right])

    def test_both_images_are_needed_with_other_panel_held_exactly_fixed(self):
        world = {"kind": "shortest", "nodes": list("ABCDEFGH"),
                 "positions": {"A": 0, "B": 1, "C": 2, "D": 6, "H": 5, "E": 4, "F": 3, "G": 7},
                 "edges": [[a, b, 2] for a, b in (("A", "B"), ("B", "C"), ("C", "D"),
                                                  ("D", "H"), ("H", "E"), ("E", "A"))]}
        graph._validate_world(world)
        self.assertIsNone(graph._shortcut_reason(world))
        images, _ = graph_pair._render(world, "default")
        reweighted = copy.deepcopy(world)
        reweighted["edges"] = [[a, b, 3] for a, b, _ in world["edges"]]
        changed_weights, _ = graph_pair._render(reweighted, "default")
        self.assertEqual(images[0], changed_weights[0])
        self.assertNotEqual(images[1], changed_weights[1])
        self.assertEqual(graph._answers(world), ("8", "8"))
        self.assertEqual(graph._answers(reweighted), ("12", "12"))
        relabel = {name: {"E": "H", "H": "E"}.get(name, name) for name in world["nodes"]}
        rewired = copy.deepcopy(world)
        rewired["positions"] = {relabel[name]: pos for name, pos in world["positions"].items()}
        rewired["edges"] = [[relabel[a], relabel[b], weight] for a, b, weight in world["edges"]]
        graph._validate_world(rewired)
        self.assertIsNone(graph._shortcut_reason(rewired))
        changed_topology, _ = graph_pair._render(rewired, "default")
        self.assertNotEqual(images[0], changed_topology[0])
        self.assertEqual(images[1], changed_topology[1])
        self.assertEqual(graph._answers(rewired), ("10", "10"))

    def test_typed_direction_slots_and_existing_prompt_contract(self):
        p = graph_pair.generate(0, 3407, "train", "obs")
        prompts = load_prompts(Path(graph_pair.__file__).parents[1] / "prompts_v1.yaml")
        self.assertIn("dir[<edge ID>]", turn1_text(p.actor, prompts))
        self.assertEqual(prompts["observation_instruction"][p.actor.family], graph_pair.OBSERVATION_INSTRUCTION)
        self.assertTrue(compare(p.verifier.gold, p.verifier))
        text = "Observed facts:\n" + "\n".join(p.verifier.oracle_fact_lines) + "\nSolution:\n"
        parsed = parse_observations(text)
        report = compare_slots(parsed, p.verifier)
        self.assertEqual(report.required_correct, len(p.verifier.slots))
        for key, kind in p.verifier.slot_types.items():
            self.assertEqual(kind, "relation" if key.startswith("dir[") else "number")
        key = "dir[e0]"
        parsed[key] = ["->".join(reversed(p.verifier.slots[key].split("->")))]
        self.assertIn(key, compare_slots(parsed, p.verifier).wrong_required)

    def test_selfcheck_rejects_swapped_duplicate_corrupt_and_misbound_panels(self):
        p = graph_pair.generate(0, 3407, "train", "plain")
        changes = (
            replace(p, image_bytes=tuple(reversed(p.image_bytes))),
            replace(p, image_bytes=(p.image_bytes[0], p.image_bytes[0])),
            replace(p, image_bytes=(p.image_bytes[0], p.image_bytes[1][:-1])),
            replace(p, actor=replace(p.actor, image_sha256=tuple(reversed(p.actor.image_sha256)))),
            replace(p, actor=replace(p.actor, image_paths=tuple(reversed(p.actor.image_paths)))),
            replace(p, verifier=replace(p.verifier, gold="99999")),
            replace(p, verifier=replace(p.verifier, slots=p.verifier.slots | {"weight[e0]": "99999"})),
            replace(p, lineage=replace(p.lineage, structure_hash="0" * 16)),
            replace(p, lineage=replace(p.lineage, model_touched=True)),
        )
        for changed in changes:
            with self.subTest(change=changes.index(changed)), self.assertRaises(ValueError):
                graph_pair.self_check(changed)
        self.assertEqual(p.lineage.structure_hash,
                         structure_hash({k: v for k, v in p.verifier.accept["graph_world"].items() if k != "positions"}))

    def test_generator_revision_includes_reused_graph_source(self):
        original = generate_pool("graph_pair", 1, 3407, "train")[0]
        real_read = Path.read_bytes
        graph_path = Path(graph.__file__)
        def dependency_changed(path):
            raw = real_read(path)
            return raw + b"\n# authored dependency identity change\n" if path == graph_path else raw
        with patch.object(Path, "read_bytes", dependency_changed):
            changed = generate_pool("graph_pair", 1, 3407, "train")[0]
        self.assertNotEqual(original.lineage.source_revision, changed.lineage.source_revision)
        self.assertEqual(original.actor, changed.actor)
        self.assertEqual(original.image_bytes, changed.image_bytes)
        self.assertFalse(original.lineage.benchmark_derived)
        self.assertFalse(original.lineage.model_touched)
        self.assertEqual(original.lineage.label_provenance, "PROGRAM")


if __name__ == "__main__":
    unittest.main()
