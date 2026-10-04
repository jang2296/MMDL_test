"""Synthetic tests; no benchmark fixtures, network access, or model calls."""
import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from fine_tuning import nvw


class NVWTests(unittest.TestCase):
    def test_independent_graph_solvers_all_four_node_dags(self):
        edges = list(itertools.combinations(range(4), 2))
        for mask in range(1 << len(edges)):
            graph = [list(edge) for i, edge in enumerate(edges) if mask & (1 << i)]
            for source in range(4):
                world = {"n": 4, "edges": graph, "initial": [source], "steps": 4}
                self.assertEqual(nvw.solve_signal(world), nvw.verify_signal(world))

    def test_independent_boolean_solvers_exhaustive(self):
        for mapping in itertools.product(nvw.RULES, repeat=2):
            for sequence in itertools.product(range(2), repeat=4):
                for initial in range(2):
                    self.assertEqual(nvw.solve_fsm(list(mapping), list(sequence), initial),
                                     nvw.verify_fsm(list(mapping), list(sequence), initial))

    def test_answer_can_be_known_when_rule_is_not(self):
        world = {"support": [[0, 0, 1], [1, 0, 0], [1, 1, 1]],
                 "mapping": ["FLIP", "IDENTITY"], "sequence": [0], "initial": 0}
        quality = nvw.fsm_identifiability(world)
        self.assertTrue(quality["answer_identifiable"])
        self.assertFalse(quality["rule_identifiable"])
        self.assertEqual(quality["possible_answers"], [1])
        world["initial"] = 1
        self.assertFalse(nvw.fsm_identifiability(world)["answer_identifiable"])
        world["support"].append([0, 0, 0])
        with self.assertRaises(ValueError):
            nvw.fsm_identifiability(world)

    def test_spatial_relations_match_geometry_and_sets(self):
        for index in range(200):
            episode = nvw.build_episode("spatial_composition", index)
            world = episode["world"]
            self.assertEqual(nvw.solve_spatial(world), nvw.verify_spatial(world))
        objects = [{"id": i, "x": i % 3, "y": i // 3, "type": 0, "color": 1}
                   for i in range(6)]
        world = {"objects": objects, "filters": [["left_of", 2], ["above", 3]],
                 "operation": "INTERSECTION"}
        self.assertEqual(nvw.solve_spatial(world)[-1], [0, 1])
        self.assertEqual(nvw.verify_spatial(world)[-1], [0, 1])

    def test_deterministic_streams_and_split_before_views(self):
        for family in nvw.FAMILIES:
            episode = nvw.build_episode(family, 17)
            self.assertEqual(episode, nvw.build_episode(family, 17))
            changed = copy.deepcopy(episode)
            changed["seeds"]["layout"] += 1
            changed["options"].reverse()
            self.assertEqual(nvw.semantic_family(changed["world"], family), episode["family_id"])
            self.assertEqual(nvw.family_split(episode["family_id"]), episode["split"])
            self.assertEqual(episode["states"], changed["states"])
        self.assertEqual(len(set(nvw.seeds(3407, 17, "finite_state").values())), 8)

    def test_input_whitelist_does_not_read_hidden_gold(self):
        episode = nvw.build_episode("finite_state", 1)
        original = nvw.input_record(episode, "images/query.png", "grounded")
        changed = copy.deepcopy(episode)
        changed.update(answer="SECRET_GOLD", answer_letter="HIDDEN", states=["LATENT"])
        changed["world"]["mapping"] = ["SECRET_RULE", "SECRET_RULE"]
        self.assertEqual(original, nvw.input_record(changed, "images/query.png", "grounded"))
        self.assertEqual(set(original), {"image", "instruction"})
        with self.assertRaises(ValueError):
            nvw.input_record(episode, "images/query.png", "leak")

    def test_render_bounds_and_reject_corrupt_states(self):
        for family in nvw.FAMILIES:
            for i in range(60):
                episode = nvw.build_episode(family, i)
                image, obs = nvw.render(episode)
                self.assertEqual(image.size, (1200, 1000))
                episode["observed_structure"] = obs
                nvw.validate_episode(episode)
                changed = copy.deepcopy(episode)
                changed["states"][-1] = "CORRUPT"
                with self.assertRaises(ValueError):
                    nvw.validate_episode(changed)

    def test_spatial_render_preserves_actual_geometry_not_only_grid_labels(self):
        for i in range(60):
            episode = nvw.build_episode("spatial_composition", i)
            _, observation = nvw.render(episode)
            for a, b in itertools.product(observation["objects"], repeat=2):
                self.assertEqual(a["center"][0] < b["center"][0],
                                 a["grid_position"][0] < b["grid_position"][0])
                self.assertEqual(a["center"][1] < b["center"][1],
                                 a["grid_position"][1] < b["grid_position"][1])
                x0, _, x1, _ = a["bbox"]
                self.assertEqual(x1 < 735, a["grid_position"][0] < 2)
                self.assertEqual(x0 > 760, a["grid_position"][0] == 2)

    def test_generation_manifest_and_smoke_separation(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "dataset"
            manifest = nvw.generate_dataset(out, per_family=6, smoke_roots=9)
            self.assertEqual(nvw.verify_dataset(out)["episodes"], 27)
            episodes = [json.loads(line) for line in (out/"canonical_private.jsonl").read_text().splitlines()]
            self.assertEqual(manifest["root_count"], 27)
            self.assertFalse(manifest["training_allowed"])
            self.assertTrue(all(e["split"] == "train" for e in episodes if e["purpose"] == "smoke"))
            self.assertEqual(len({e["root_id"] for e in episodes}), len(episodes))
            self.assertTrue(all(len(e["parents"]) == 0 for e in episodes))
            with Image.open(out/episodes[0]["image"]) as image:
                self.assertEqual(image.size, nvw.OUTPUT_SIZE)
            for path in out.glob("prototype_*.jsonl"):
                for line in path.read_text().splitlines():
                    record = json.loads(line)
                    self.assertEqual(record["user_text"].count("<image>"), len(record["image_paths"]))
                    self.assertFalse(record["training_allowed"])
                    self.assertFalse(record["benchmark_derived"])
                    self.assertEqual(record["image_paths"], record["image"])
            with self.assertRaises(FileExistsError):
                nvw.generate_dataset(out)

    def test_tampered_gold_and_identifiability_fail_independent_verification(self):
        episode = nvw.build_episode("finite_state", 1)
        changed = copy.deepcopy(episode)
        changed["answer"] = "INVENTED"
        changed["options"][ord(changed["answer_letter"])-65] = "INVENTED"
        with self.assertRaisesRegex(ValueError, "answer disagrees"):
            nvw.validate_episode(changed)
        changed = copy.deepcopy(episode)
        changed["quality"]["rule_identifiable"] = not changed["quality"]["rule_identifiable"]
        with self.assertRaisesRegex(ValueError, "identifiability metadata"):
            nvw.validate_episode(changed)

    def test_isomorphic_graphs_and_renamed_operators_share_family(self):
        first = {"n": 4, "edges": [[0, 2], [1, 2], [1, 3]]}
        renamed = {"n": 4, "edges": [[3, 0], [2, 0], [2, 1]]}
        self.assertEqual(nvw.semantic_family(first, "signal_graph"),
                         nvw.semantic_family(renamed, "signal_graph"))
        first = {"mapping": ["FLIP", "SET_ONE"], "sequence": [0, 1, 0]}
        renamed = {"mapping": ["SET_ONE", "FLIP"], "sequence": [1, 0, 1]}
        self.assertEqual(nvw.semantic_family(first, "finite_state"),
                         nvw.semantic_family(renamed, "finite_state"))


if __name__ == "__main__":
    unittest.main()
