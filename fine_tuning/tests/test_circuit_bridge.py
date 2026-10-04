"""Independent circuit arithmetic, paired controls and quarantine; no model/GPU."""
import copy
import io
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from PIL import Image

from fine_tuning import audit, circuit_bridge as bridge


class CircuitBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = bridge.bundle(4,73041,1_000_000)
        cls.episodes = [json.loads(line) for line in cls.files["canonical_private.jsonl"].splitlines()]

    def test_exact_solvers_open_closed_and_invalid_values(self):
        for closed in (False,True):
            world = {"voltage_V":12,"resistance_ohm":[2,6,3],"switch_closed":closed}
            solved = bridge.solve_nodal(world)
            self.assertEqual(solved,bridge.verify_series_parallel(world))
            self.assertEqual(Fraction(3 if closed else "3/2"),Fraction(solved["source_current_A"]))
            self.assertEqual("6" if closed else "9",solved["node_voltages_V"]["N2"])
            self.assertEqual("6" if closed else "0",solved["node_voltages_V"]["N3"])
        for voltage,resistors,state in ((True,[2,6,3],True),(12,[0,6,3],True),(12,[2,6],True),(12,[2,6,3],1)):
            with self.assertRaises(ValueError):
                bridge.solve_nodal({"voltage_V":voltage,"resistance_ohm":resistors,"switch_closed":state})
        for i in range(100):
            root = bridge.root_episode(1_100_000+i,23041)
            self.assertEqual(bridge.solve_nodal(root["world"]),bridge.verify_series_parallel(root["world"]))

    def test_semantics_controls_and_rendered_component_identity(self):
        for i in range(0,len(self.episodes),4):
            group = self.episodes[i:i+4]
            for episode in group:
                for key in ("root_id","family_id","split","world","solution","options","answer_letter"):
                    self.assertEqual(group[0][key],episode[key])
                self.assertEqual(sorted(episode["component_glyph_mapping"].values()),[0,1,2])
                for path,scene,kind in zip(episode["image_paths"],episode["representation_scenes"],episode["representations"],strict=True):
                    self.assertEqual(bridge.canonical_structure(episode["world"]),scene["canonical_structure"])
                    image,expected = bridge.render(episode,kind)
                    self.assertEqual(scene,expected)
                    with Image.open(io.BytesIO(self.files[path])) as saved:
                        self.assertEqual(saved.tobytes(),image.tobytes())
                    self.assertFalse(any(k in scene for k in ("answer","solution","source_current_A")))
                if episode["bridge_enabled"]:
                    self.assertNotEqual(episode["representation_scenes"][0]["node_positions"],episode["representation_scenes"][2]["node_positions"])
                else:
                    self.assertEqual(len({self.files[p] for p in episode["image_paths"]}),1)
            for slot in (1,2):
                self.assertEqual(self.files[group[2]["image_paths"][slot]],self.files[group[3]["image_paths"][slot]])
            for a,b in ((0,2),(1,3)):
                self.assertEqual(self.files[group[a]["image_paths"][0]],self.files[group[b]["image_paths"][0]])

    def test_loader_whitelist_shared_splits_counts_and_unapproved_lineage(self):
        all_rows = []
        manifest = json.loads(self.files["manifest.json"])
        self.assertEqual((manifest["records_per_arm"],manifest["image_references_per_arm"]),(12,20))
        self.assertEqual(sum(p.endswith(".png") for p in self.files),48)
        for arm in bridge.ARMS:
            rows = [json.loads(line) for line in self.files[arm+".jsonl"].splitlines()]
            all_rows += rows
            self.assertEqual(len(rows),12)
            for row in rows:
                self.assertEqual(row["user_text"].count("<image>"),len(row["image_paths"]))
                audit.model_input_record(row)
                for key,value in bridge.FLAGS.items():
                    self.assertIs(row[key],value)
                with self.assertRaises(ValueError):
                    audit.check_lineage(row,{})
        audit.check_splits(all_rows)
        poisoned = copy.deepcopy(all_rows)
        poisoned[0]["split"] = "development" if poisoned[0]["split"] != "development" else "train"
        with self.assertRaises(ValueError):
            audit.check_splits(poisoned)

    def test_options_no_longer_encode_a_single_correct_numeric_rank(self):
        counts = dict.fromkeys(range(4),0)
        for i in range(1000):
            episode = bridge.root_episode(1_100_000+i,23041)
            answer = Fraction(episode["solution"]["source_current_A"])
            values = list(map(Fraction,episode["options"]))
            self.assertEqual(len(set(values)),4)
            self.assertTrue(all(v > 0 for v in values))
            self.assertEqual(answer,values[ord(episode["answer_letter"])-65])
            counts[sorted(values).index(answer)] += 1
        self.assertTrue(all(counts.values()))
        self.assertLess(counts[1],400)

    def test_deterministic_generation_refuses_overwrite_mutation_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"prototype"
            bridge.generate(output)
            self.assertEqual("CPU_REPRODUCED_NOT_APPROVED",bridge.verify(output)["status"])
            with self.assertRaises(FileExistsError):
                bridge.generate(output)
            (output/"extra.json").write_text("{}")
            with self.assertRaises(ValueError):
                bridge.verify(output)
        for count,seed,index in ((True,1,1_000_000),(1,-1,1_000_000),(1,1,0)):
            with self.assertRaises(ValueError):
                bridge.bundle(count,seed,index)
        world = {"voltage_V":12,"resistance_ohm":[2,6,3],"switch_closed":True,"gold":"B"}
        with self.assertRaises(ValueError):
            bridge.solve_nodal(world)


if __name__ == "__main__":
    unittest.main()
