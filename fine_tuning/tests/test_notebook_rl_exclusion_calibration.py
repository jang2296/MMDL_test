import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from fine_tuning.notebook_rl import contamination as c
from fine_tuning.notebook_rl import exclusion_calibration as e
from fine_tuning.notebook_rl.build_exclusion_index import _write_once
from fine_tuning.notebook_rl.exclusion_calibration import TRANSFORMS, _rate, threshold_for_fpr, transform


class CalibrationTests(unittest.TestCase):
    def material_fixture(self, root):
        images, fingerprints, vectors = {}, {}, {}
        for n, color in enumerate(("green", "blue", "red", "cyan")):
            with Image.new("RGB", (12, 12), color) as image:
                stream = io.BytesIO()
                image.save(stream, format="PNG")
                raw = stream.getvalue()
                sha = hashlib.sha256(raw).hexdigest()
                path = root / f"{n}.png"
                _write_once(path, raw)
                images[sha] = str(path)
                fingerprints[sha] = c.image_fingerprints(image)
                vectors[sha] = {"sha256": sha, "vector": [1., 0.] if n < 2 else [0., 1.]}
        reference, positive, negative, validation = images
        material = {"format": "notebook-rl-copy-calibration-v1", "training_allowed": False,
                    "seed": e.SEED, "index_sha256": "i" * 64,
                    "producer_sha256": hashlib.sha256(Path(e.__file__).read_bytes()).hexdigest(),
                    "fingerprint_producer_sha256": hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(),
                    "images": images, "fingerprints": fingerprints, "reference_sha256": [reference],
                    "positives": [{"reference": reference, "query": positive, "role": role}
                                  for role in ("calibration", "validation")],
                    "negatives": [{"query": negative, "role": "calibration"},
                                  {"query": validation, "role": "validation"}],
                    "limitations": ["Authored tiny fixture, not real benchmark calibration"]}
        engines = {"sscd": {"engine_id": "sscd_disc_mixup", "artifact_sha256": "a" * 64},
                   "siglip": {"engine_id": "google/siglip-base-patch16-224", "artifact_sha256": "b" * 64}}
        return material, vectors, engines

    def test_transform_determinism_and_no_metadata(self):
        source = io.BytesIO()
        Image.new("RGB", (80, 100), (15, 39, 250)).save(source, format="PNG")
        expected = {"jpeg80": (80, 100), "resize_half": (40, 50),
                    "center80": (64, 80), "white_border25": (120, 140)}
        for name in TRANSFORMS:
            raw = transform(source.getvalue(), name)
            self.assertEqual(raw, transform(source.getvalue(), name))
            with Image.open(io.BytesIO(raw)) as image:
                self.assertEqual(image.size, expected[name])
                self.assertNotIn("exif", image.info)

    def test_fpr_threshold_uses_calibration_only_and_ties(self):
        scores = [.2] * 99 + [.8]
        threshold = threshold_for_fpr(scores)
        self.assertGreater(threshold, .2)
        self.assertEqual(sum(v >= threshold for v in scores), 1)
        self.assertGreater(threshold_for_fpr([.3] * 64), .3)
        self.assertEqual(threshold_for_fpr([-.2] * 64), .000001)

    def test_invalid_scores_rejected(self):
        for values in ([], [math.nan], [1.1], [1.]):
            with self.assertRaises(ValueError):
                threshold_for_fpr(values)
        with self.assertRaises(ValueError):
            transform(b"", "bad")

    def test_observed_rates_not_fabricated_pass(self):
        observed = _rate([.9, .4], [.8, .1], .75, {"ids": [1, 2]})
        self.assertEqual(observed["tpr"], .5)
        self.assertEqual(observed["fpr"], .5)
        self.assertEqual(observed["positive_count"], 2)
        self.assertEqual(len(observed["examples_sha256"]), 64)

    def test_measure_rejects_stale_material_before_reading_images_or_publishing(self):
        for field in ("producer_sha256", "fingerprint_producer_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                material, vectors, engines = self.material_fixture(root)
                material[field] = "0" * 64
                # A missing image must not hide the earlier stale-source rejection.
                Path(next(iter(material["images"].values()))).unlink()
                _write_once(root / "material.json", c._json(material))
                with self.assertRaisesRegex(ValueError, "material producer/fingerprint code changed"):
                    e.measure(root / "material.json", vectors, vectors, root / "output", engines=engines)
                self.assertFalse((root / "output").exists())

    def test_measure_result_binds_current_sources_and_actual_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            material, vectors, engines = self.material_fixture(root)
            _write_once(root / "material.json", c._json(material))
            result = e.measure(root / "material.json", vectors, vectors, root / "output", engines=engines)
            saved = json.loads((root / "output/calibration.json").read_bytes())
            self.assertEqual(saved, result)
            self.assertEqual(result["producer_sha256"], material["producer_sha256"])
            self.assertEqual(result["fingerprint_producer_sha256"], material["fingerprint_producer_sha256"])
            self.assertEqual(result["embedding_exports_sha256"], {"sscd": c._digest(vectors), "siglip": c._digest(vectors)})
            self.assertEqual(result["validation"]["siglip"]["negative_count"], 1)
            self.assertFalse(result["training_allowed"])


if __name__ == "__main__":
    unittest.main()
