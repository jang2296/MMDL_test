"""CPU-only synthetic graph bridge checks; persistent artifacts are not generated."""
import copy
import hashlib
import inspect
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from fine_tuning import audit, graph_bridge as bridge, nvw


def rows(files, name):
    return [json.loads(line) for line in files[name].splitlines()]


class GraphBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = bridge.bundle(4, 73041, 1_000_000)
        cls.episodes = rows(cls.files, "canonical_private.jsonl")

    def test_semantics_splits_and_new_unapproved_provenance(self):
        self.assertEqual(len(self.episodes), 16)
        manifest = json.loads(self.files["manifest.json"])
        self.assertEqual(manifest["source_ancestry"][bridge.SOURCE_ID],
                         {"parents": ["independent-nvw-v1"], "rights_approved": False})
        for i in range(0, len(self.episodes), 4):
            root = self.episodes[i]
            self.assertNotEqual(root["states"][0], root["states"][-1])
            for episode in self.episodes[i:i + 4]:
                for key in ("root_id", "family_id", "split", "world", "states", "answer",
                            "answer_letter", "options", "seeds", "quality", "node_labels"):
                    self.assertEqual(episode[key], root[key])
                self.assertEqual(nvw.solve_signal(episode["world"]),
                                 nvw.verify_signal(episode["world"]))
                self.assertEqual(episode["family_id"], nvw.semantic_family(episode["world"], "signal_graph"))
                nvw.validate_episode(episode)
                self.assertEqual(episode["source_id"], bridge.SOURCE_ID)
                self.assertEqual(episode["lineage"], [bridge.SOURCE_ID, "independent-nvw-v1"])
                self.assertEqual(episode["parents"], [])
                self.assertEqual(episode["generator_parents"], ["independent-nvw-v1"])
                for key, value in bridge.FLAGS.items():
                    self.assertEqual(episode[key], value)
                self.assertEqual(episode["quality"]["model_input"], "UNKNOWN")
        for name, expected in manifest["source_files_sha256"].items():
            self.assertEqual(hashlib.sha256(self.files[name]).hexdigest(), expected)
        self.assertEqual(manifest["render_assets"]["font_ttf_sha256"],
                         hashlib.sha256(nvw.FONT.path.getvalue()).hexdigest())
        self.assertTrue(all(not r["training_allowed"] for r in rows(self.files, "review_queue.jsonl")))

    def test_fixed_randomized_glyphs_encode_two_semantic_roles(self):
        permutations = set()
        for episode in self.episodes:
            mapping = episode["endpoint_role_mapping"]
            self.assertEqual(set(mapping), set(bridge.ROLES))
            self.assertEqual(sorted(mapping.values()), [0, 1])
            if episode["condition"] in ("A0", "A2"):
                self.assertEqual(mapping, bridge.FIXED_MAPPING)
            else:
                permutations.add(tuple(mapping[r] for r in bridge.ROLES))
            self.assertEqual(mapping, bridge.condition_episode(
                episode, episode["condition"], 73041)["endpoint_role_mapping"])
            scene = episode["observed_structure"]
            inverse = {code: role for role, code in mapping.items()}
            nodes = {o["id"]: o["center"] for o in scene["objects"]}
            inverse_labels = {label: node for node, label in enumerate(episode["node_labels"])}
            decoded = []
            for relation in scene["relations"]:
                semantic = {}
                for endpoint in relation["endpoints"]:
                    role = inverse[endpoint["glyph_code"]]
                    nearest = min(nodes, key=lambda node: math.dist(endpoint["center"], nodes[node]))
                    self.assertEqual(nearest, endpoint["node_id"])
                    semantic[role] = nearest
                decoded.append([inverse_labels[semantic["SOURCE/TAIL"]],
                                inverse_labels[semantic["TARGET/HEAD"]]])
            self.assertEqual(decoded, episode["world"]["edges"])
        self.assertEqual(permutations, {(0, 1), (1, 0)})

    def test_direction_and_multistep_are_needed_against_explicit_shortcuts(self):
        for episode in self.episodes:
            counts = bridge.shortcut_counts(episode["world"])
            self.assertEqual(counts, episode["shortcut_audit"])
            self.assertGreater(counts["directed_final"], counts["one_step_directed"])
            for key in ("one_step_undirected", "bounded_undirected", "reversed"):
                self.assertNotEqual(counts["directed_final"], counts[key])
        # This exact v1 root was solvable by an undirected one-hop degree count.
        old = nvw.build_episode("signal_graph", 1_000_000, 73041)
        self.assertEqual(bridge.exclusion_reason(old), "NO_POST_STEP_ONE_GROWTH")
        with self.assertRaises(ValueError):
            bridge.condition_episode(old, "A0", 73041)

    def test_shuffled_layouts_have_no_global_direction_shortcut_or_residual_arrows(self):
        for episode in self.episodes:
            scene = episode["observed_structure"]
            positions = [o["center"] for o in scene["objects"]]
            self.assertEqual(positions, bridge.layout(episode, "standard"))
            bridge.validate_layout(episode["world"], positions)
            displacements = [positions[b][0] - positions[a][0] for a, b in episode["world"]["edges"]]
            self.assertTrue(any(d > 0 for d in displacements))
            self.assertTrue(any(d < 0 for d in displacements))
            labels = episode["node_labels"]
            self.assertTrue(any(labels[a] < labels[b] for a, b in episode["world"]["edges"]))
            self.assertTrue(any(labels[a] > labels[b] for a, b in episode["world"]["edges"]))
            self.assertNotEqual(positions, bridge.layout(episode, "standard_relayout"))
            with patch.object(nvw, "arrow", side_effect=AssertionError("novel arrow leak")):
                image, repeated = bridge.render(episode, "novel", positions)
            self.assertEqual(repeated, scene)
            with Image.open(io.BytesIO(self.files[episode["image"]])) as stored:
                self.assertEqual(stored.tobytes(), image.tobytes())
            with self.assertRaises(ValueError):
                bridge.validate_layout(episode["world"], [[170, 450]] * episode["world"]["n"])

    def test_three_representations_preserve_graph_initial_state_and_hide_computed_results(self):
        for episode in self.episodes:
            if not episode["bridge_enabled"]:
                continue
            self.assertEqual(episode["representations"], ["novel", "standard", "standard_relayout"])
            world = episode["world"]
            labels = episode["node_labels"]
            inverse_labels = {label: node for node, label in enumerate(labels)}
            for kind, scene, path in zip(episode["representations"], episode["representation_scenes"],
                                         episode["image_paths"]):
                self.assertEqual([[inverse_labels[r["from"]], inverse_labels[r["to"]]]
                                  for r in scene["relations"]], world["edges"])
                self.assertEqual([inverse_labels[o["id"]] for o in scene["objects"] if o["initially_active"]],
                                 world["initial"])
                positions = [o["center"] for o in scene["objects"]]
                image, expected = bridge.render(episode, kind, positions)
                self.assertEqual(scene, expected)
                with Image.open(io.BytesIO(self.files[path])) as stored:
                    self.assertEqual(stored.tobytes(), image.tobytes())
                self.assertFalse(any(k in scene for k in ("answer", "answer_letter", "states")))
                self.assertTrue(all(isinstance(o["id"], str) for o in scene["objects"]))
                self.assertFalse(any("answer:" in text.casefold() or "state:" in text.casefold()
                                     for text in scene["printed_text"]))
                self.assertIn(episode["question"], scene["printed_text"])
                if kind != "novel":
                    self.assertTrue(all("endpoints" not in r for r in scene["relations"]))
            first, second, third = episode["image_paths"]
            self.assertNotEqual(self.files[first], self.files[second])
            self.assertNotEqual(self.files[second], self.files[third])
        # Canonical pixels cannot depend on the novel role permutation.
        for i in range(0, len(self.episodes), 4):
            for asset in (1, 2):
                self.assertEqual(self.files[self.episodes[i + 2]["image_paths"][asset]],
                                 self.files[self.episodes[i + 3]["image_paths"][asset]])

    def test_matched_counts_and_exact_bridge_pair_controls(self):
        manifest = json.loads(self.files["manifest.json"])
        self.assertEqual((manifest["records_per_arm"], manifest["image_files_per_arm"],
                          manifest["image_references_per_arm"]), (12, 12, 20))
        self.assertEqual(sum(n.endswith(".png") for n in self.files), 48)
        combined = []
        for arm, (_, enabled) in bridge.ARMS.items():
            records = rows(self.files, arm + ".jsonl")
            combined.extend(records)
            self.assertEqual(len(records), 12)
            self.assertEqual(sum(len(r["image_paths"]) for r in records), 20)
            for row in records:
                self.assertEqual(row["user_text"].count("<image>"), len(row["image_paths"]))
                audit.model_input_record(row)
                for key, value in bridge.FLAGS.items():
                    self.assertEqual(row[key], value)
                if row["view"] == "grounded":
                    images = [self.files[p] for p in row["image_paths"]]
                    self.assertEqual(images[0] == images[1] == images[2], not enabled)
                    episode = next(e for e in self.episodes if e["root_id"] == row["root_id"]
                                   and e["condition"] == arm)
                    target = json.loads(row["target_text"])
                    self.assertEqual([s["state"] for s in target["states"]],
                                     [sorted(episode["node_labels"][node] for node in state)
                                      for state in episode["states"]])
        audit.check_splits(combined)
        for left, right in (("A0", "A2"), ("A1", "A3")):
            for a, b in zip(rows(self.files, left + ".jsonl"), rows(self.files, right + ".jsonl")):
                for key in ("user_text", "target_text", "family_id", "root_id", "split"):
                    self.assertEqual(a[key], b[key])
                self.assertEqual(self.files[a["image_paths"][0]], self.files[b["image_paths"][0]])

    def test_whitelist_cannot_read_private_gold_and_bundle_cannot_take_external_roots(self):
        episode = self.episodes[0]
        changed = copy.deepcopy(episode)
        changed.update(answer="SECRET_GOLD", answer_letter="Z", states=["PRIVATE_TRACE"])
        changed["world"].update(initial=["SECRET_INITIAL"], edges=[["SECRET", "TARGET"]])
        for view in bridge.VIEWS:
            first = bridge.paired_record(episode, episode["image_paths"], view)
            second = bridge.paired_record(changed, episode["image_paths"], view)
            self.assertEqual(first["user_text"], second["user_text"])
            self.assertEqual(first["conversations"][0], second["conversations"][0])
            self.assertNotIn("SECRET", first["user_text"])
            self.assertNotIn("node_labels", first["user_text"])
        self.assertEqual(set(inspect.signature(bridge.bundle).parameters), {"count", "seed", "start_index"})
        with self.assertRaises(TypeError):
            bridge.bundle(1, 1, 1_000_000, episodes=[episode])

    def test_taint_and_uninformative_roots_fail_closed(self):
        episode = self.episodes[0]
        for key in ("benchmark_derived", "training_allowed", "public_upload_allowed"):
            changed = copy.deepcopy(episode)
            changed[key] = True
            with self.assertRaises(ValueError):
                bridge.condition_episode(changed, "A0", 73041)
        for mutate in (lambda e: e.update(parents=["benchmark"]),
                       lambda e: e.update(all_parents_known=False),
                       lambda e: e["contamination"].update(benchmark_derived=True)):
            changed = copy.deepcopy(episode)
            mutate(changed)
            with self.assertRaises(ValueError):
                bridge.exclusion_reason(changed)
        empty = None
        for index in range(1_000_000, 1_000_100):
            candidate = nvw.build_episode("signal_graph", index, 73041)
            if candidate["states"][0] == candidate["states"][-1]:
                empty = candidate
                break
        self.assertIsNotNone(empty)
        self.assertEqual(bridge.exclusion_reason(empty), "EMPTY_OR_UNINFORMATIVE_GRAPH")
        with self.assertRaises(ValueError):
            bridge.condition_episode(empty, "A0", 73041)
        with self.assertRaises(KeyError):
            bridge.condition_episode(episode, "UNKNOWN", 73041)

    def test_reproducibility_exclusive_output_and_tamper_rejection(self):
        self.assertEqual(self.files, bridge.bundle(4, 73041, 1_000_000))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "graph"
            bridge.generate(output, count=1)
            self.assertEqual(bridge.verify(output)["status"], "CPU_REPRODUCED_NOT_APPROVED")
            with self.assertRaises(FileExistsError):
                bridge.generate(output, count=1)
            image = next((output / "images").glob("*.png"))
            original = image.read_bytes()
            image.write_bytes(original + b"tampered")
            with self.assertRaisesRegex(ValueError, "differs"):
                bridge.verify(output)
            image.write_bytes(original)
            manifest_path = output / "manifest.json"
            original_manifest = manifest_path.read_bytes()
            records = output / "A0.jsonl"
            records.write_bytes(records.read_bytes().replace(b'"benchmark_derived": false',
                                                            b'"benchmark_derived": true'))
            manifest = json.loads(original_manifest)
            manifest["files"]["A0.jsonl"]["sha256"] = hashlib.sha256(records.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "differs"):
                bridge.verify(output)
        for args in ((0, 1, 1_000_000), (101, 1, 1_000_000), (1, -1, 1_000_000), (1, 1, 0)):
            with self.assertRaises(ValueError):
                bridge.bundle(*args)


if __name__ == "__main__":
    unittest.main()
