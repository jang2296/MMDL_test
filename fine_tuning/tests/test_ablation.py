"""CPU-only paired Boolean prototype checks; every output is a temporary fixture."""
import copy
import inspect
import hashlib
import io
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fine_tuning import ablation, nvw


def records(files, name):
    return [json.loads(line) for line in files[name].splitlines()]


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = ablation.bundle(4, 73041, 1_000_000)

    def test_semantics_and_splits_invariant_across_conditions(self):
        episodes = records(self.files, "canonical_private.jsonl")
        self.assertEqual(len(episodes), 16)
        for i in range(0, len(episodes), 4):
            base = episodes[i]
            for episode in episodes[i:i+4]:
                for key in ("world", "states", "answer", "answer_letter", "options", "root_id",
                            "family_id", "split", "seeds", "quality"):
                    self.assertEqual(base[key], episode[key])
                nvw.validate_episode(episode)
                self.assertEqual(episode["glyph_codes"], [episode["rule_glyph_mapping"][rule]
                                                       for rule in episode["world"]["mapping"]])

    def test_rule_not_operator_mapping_fixed_and_randomized_reproducible(self):
        episodes = records(self.files, "canonical_private.jsonl")
        randomized = []
        for episode in episodes:
            mapping = episode["rule_glyph_mapping"]
            self.assertEqual(sorted(mapping.values()), [0, 1, 2, 3])
            if episode["condition"] in ("A0", "A2"):
                self.assertEqual(mapping, ablation.FIXED_MAPPING)
            else:
                randomized.append(mapping)
                repeated = ablation.condition_episode(episode, episode["condition"], 73041)
                self.assertEqual(mapping, repeated["rule_glyph_mapping"])
        self.assertTrue(any(m != ablation.FIXED_MAPPING for m in randomized))
        self.assertGreater(len({nvw.digest(m) for m in randomized}), 1)
        self.assertTrue(any(episodes[i]["glyph_codes"] != episodes[i+1]["glyph_codes"]
                            for i in range(0, len(episodes), 4)))
        self.assertEqual(self.files, ablation.bundle(4, 73041, 1_000_000))

    def test_counts_bridge_pair_input_and_supervision(self):
        manifest = json.loads(self.files["manifest.json"])
        self.assertEqual(manifest["records_per_arm"], 12)
        self.assertEqual(manifest["image_files_per_arm"], 8)
        self.assertEqual(sum(name.endswith(".png") for name in self.files), 32)
        for arm, (_, bridge) in ablation.ARMS.items():
            rows = records(self.files, arm + ".jsonl")
            self.assertEqual(len(rows), 12)
            for row in rows:
                self.assertEqual(len(row["image_paths"]), 2 if row["view"] == "grounded" else 1)
                self.assertEqual(row["user_text"].count("<image>"), len(row["image_paths"]))
                for key, value in ablation.FLAGS.items():
                    self.assertEqual(row[key], value)
                if row["view"] == "grounded":
                    left, right = [self.files[name] for name in row["image_paths"]]
                    self.assertEqual(left == right, not bridge)
        for first, second in (("A0", "A2"), ("A1", "A3")):
            for a, b in zip(records(self.files, first + ".jsonl"),
                            records(self.files, second + ".jsonl")):
                self.assertEqual(a["user_text"], b["user_text"])
                self.assertEqual(a["target_text"], b["target_text"])
                self.assertEqual(self.files[a["image_paths"][0]], self.files[b["image_paths"][0]])

    def test_canonical_labels_are_above_correct_left_to_right_arrows(self):
        self.assertEqual(ablation.VERSION, "boolean-finite-state-ablation-v2")
        for episode in records(self.files, "canonical_private.jsonl"):
            if not episode["bridge_enabled"]:
                continue
            old, _ = nvw.render(episode, canonical=True)
            repaired, scene = ablation.render_canonical(episode)
            encoded = io.BytesIO()
            old.save(encoded, format="PNG")
            self.assertNotEqual(hashlib.sha256(encoded.getvalue()).hexdigest(),
                                episode["companion_sha256"])
            self.assertEqual(scene, episode["companion_scene"])
            # Support, initial bit, question/options and every pixel outside the band survive.
            self.assertEqual(old.crop((0, 0, 1200, 550)).tobytes(),
                             repaired.crop((0, 0, 1200, 550)).tobytes())
            self.assertEqual(old.crop((0, 661, 1200, 1000)).tobytes(),
                             repaired.crop((0, 661, 1200, 1000)).tobytes())
            sequence = episode["world"]["sequence"]
            self.assertEqual([label["rule"] for label in scene["query_labels"]],
                             [episode["world"]["mapping"][op] for op in sequence])
            for label in scene["query_labels"]:
                self.assertGreaterEqual(635 - label["bbox"][3], 50)
                self.assertEqual(repaired.getpixel((label["bbox"][0], 605)), (255, 255, 255))
            for i, relation in enumerate(scene["relations"]):
                self.assertEqual((relation["from"], relation["to"]), (i, i+1))
                left, right = relation["segment"]
                self.assertEqual((left[1], right[1]), (635, 635))
                self.assertLess(left[0], right[0])
                self.assertEqual(repaired.getpixel(((left[0]+right[0])//2, 635)), (0, 0, 0))
            # Layout version changes must not silently change v1's semantic-glyph permutation.
            codes = list(ablation.FIXED_MAPPING.values())
            random.Random(nvw.digest(["boolean-finite-state-ablation-v1", 73041,
                                      episode["root_id"], "rule_glyph"])).shuffle(codes)
            if episode["condition"] == "A3":
                self.assertEqual(episode["rule_glyph_mapping"], dict(zip(ablation.FIXED_MAPPING, codes)))

    def test_input_whitelist_and_external_input_absent(self):
        episode = records(self.files, "canonical_private.jsonl")[0]
        changed = copy.deepcopy(episode)
        changed.update(answer="SECRET", answer_letter="GOLD", states=["SECRET"])
        changed["world"]["mapping"] = ["HIDDEN", "GOLD"]
        for view in ablation.VIEWS:
            self.assertEqual(nvw.input_record(episode, "novel.png", view),
                             nvw.input_record(changed, "novel.png", view))
            self.assertEqual(ablation.paired_record(episode, "novel.png", "second.png", view)["user_text"],
                             ablation.paired_record(changed, "novel.png", "second.png", view)["user_text"])
        self.assertEqual(set(inspect.signature(ablation.bundle).parameters),
                         {"count", "seed", "start_index"})
        with self.assertRaises(TypeError):
            ablation.bundle(1, 1, 1_000_000, episodes=[episode])

    def test_exclusions_and_taint_fail_closed(self):
        episode = records(self.files, "canonical_private.jsonl")[0]
        for key in ("benchmark_derived", "training_allowed", "public_upload_allowed"):
            changed = copy.deepcopy(episode)
            changed[key] = True
            with self.assertRaises(ValueError):
                ablation.condition_episode(changed, "A0", 73041)
        changed = copy.deepcopy(episode)
        changed["contamination"]["benchmark_derived"] = True
        with self.assertRaises(ValueError):
            ablation.exclusion_reason(changed)
        duplicate = unknown = None
        for i in range(1_000_000, 1_000_100):
            item = nvw.build_episode("finite_state", i, 73041)
            reason = ablation.exclusion_reason(item)
            if reason == "DUPLICATE_SEMANTIC_RULES":
                duplicate = item
            if reason == "NOT_FULLY_IDENTIFIABLE":
                unknown = item
        self.assertIsNotNone(duplicate)
        self.assertIsNotNone(unknown)
        with self.assertRaisesRegex(ValueError, "DUPLICATE"):
            ablation.condition_episode(duplicate, "A0", 73041)
        with self.assertRaisesRegex(ValueError, "IDENTIFIABLE"):
            ablation.condition_episode(unknown, "A0", 73041)
        exclusions = records(ablation.bundle(1, 73041, 1_000_006), "excluded_roots.jsonl")
        self.assertTrue(exclusions)
        self.assertTrue(all(r["reason"] and not r["training_allowed"] for r in exclusions))

    def test_no_overwrite_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "paired"
            ablation.generate(out, count=1)
            self.assertEqual(ablation.verify(out)["status"], "CPU_REPRODUCED_NOT_APPROVED")
            original_manifest = (out / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                ablation.generate(out, count=1)
            self.assertEqual((out / "manifest.json").read_bytes(), original_manifest)
            image = next((out / "images").glob("*.png"))
            original_image = image.read_bytes()
            image.write_bytes(original_image + b"tampered")
            with self.assertRaisesRegex(ValueError, "differs"):
                ablation.verify(out)
            image.write_bytes(original_image)
            path = out / "A0.jsonl"
            original = path.read_bytes()
            path.write_bytes(original.replace(b'"benchmark_derived": false',
                                              b'"benchmark_derived": true'))
            # Even replacing the stored hash does not legitimize tainted records.
            manifest = json.loads((out / "manifest.json").read_bytes())
            manifest["files"]["A0.jsonl"]["sha256"] = "0" * 64
            (out / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "differs"):
                ablation.verify(out)
        for args in ((0, 1, 1_000_000), (101, 1, 1_000_000), (1, -1, 1_000_000), (1, 1, 0)):
            with self.assertRaises(ValueError):
                ablation.bundle(*args)

    def test_cli_generates_and_verifies_only_temporary_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            command = [sys.executable, "-m", "fine_tuning.ablation", "--output",
                       str(Path(temp) / "cli"), "--count", "1", "--seed", "73041",
                       "--start-index", "1000006"]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)["root_count"], 1)
            result = subprocess.run(command + ["--verify"], check=True,
                                    capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)["status"], "CPU_REPRODUCED_NOT_APPROVED")


if __name__ == "__main__":
    unittest.main()
