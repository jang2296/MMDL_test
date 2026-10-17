"""Authored CPU fixtures and mocked models; actual engine execution is separate."""
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from fine_tuning.notebook_rl import contamination as c
from fine_tuning.notebook_rl import exclusion_engines as e


class FakeOCR:
    identity = {"engine_id": "tesseract", "revision": "authored-test-only", "artifact_sha256": "a" * 64}

    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def recognize(self, image):
        self.calls += 1
        if self.fail:
            raise subprocess.TimeoutExpired("authored-command", 1)
        return {"text": "Authored amber triangle", "status": "COMPLETE"}


class FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 1

    def encode(self, text, **kwargs):
        if kwargs != {"add_special_tokens": False, "truncation": False}:
            raise AssertionError("full untruncated encode required")
        return list(range(10, 150))


class EngineCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "cache"
        stream = io.BytesIO()
        Image.new("RGB", (12, 15), "orange").save(stream, format="PNG")
        self.raw = stream.getvalue()
        self.sha = hashlib.sha256(self.raw).hexdigest()
        self.image = self.root / self.sha
        self.image.write_bytes(self.raw)
        self.sources = {self.sha: str(self.image)}

    def test_cache_entry_binds_image_and_engine_and_restart_reuses(self):
        engine = FakeOCR()
        first = e.produce(engine, self.sources, self.output)
        self.assertEqual(first["new_entries"], 1)
        self.assertEqual(first["status"], "PARTIAL_CACHE_ONLY")
        self.assertFalse(first["full_coverage_claimed"])
        cache = Path(first["cache_root"])
        value = json.loads((cache / "images" / (self.sha + ".json")).read_bytes())
        self.assertEqual(value["source_sha256"], self.sha)
        self.assertEqual(value["engine_sha256"], c._digest(engine.identity))
        self.assertEqual(value["payload"]["image_sha256"], self.sha)
        self.assertEqual(value["payload_sha256"], c._digest(value["payload"]))
        again = e.produce(engine, self.sources, self.output)
        self.assertEqual(again["reused_entries"], 1)
        self.assertEqual(engine.calls, 1)
        self.assertEqual((cache / "images" / (self.sha + ".json")).stat().st_mode & 0o777, 0o600)

    def test_timeout_is_a_failure_never_no_text_or_success_cache(self):
        result = e.produce(FakeOCR(fail=True), self.sources, self.output)
        cache = Path(result["cache_root"])
        self.assertEqual(result["failed_entries"], 1)
        self.assertEqual(list((cache / "images").iterdir()), [])
        failure = json.loads(next((cache / "failures").iterdir()).read_bytes())
        self.assertEqual(failure["status"], "FAILED")
        self.assertEqual(failure["error_type"], "TimeoutExpired")
        self.assertNotIn("text", failure)
        self.assertEqual(e.produce(FakeOCR(), self.sources, self.output)["new_entries"], 1)

    def test_tampered_cache_and_source_fail_closed(self):
        result = e.produce(FakeOCR(), self.sources, self.output)
        path = Path(result["cache_root"]) / "images" / (self.sha + ".json")
        original = path.read_bytes()
        value = json.loads(original)
        value["payload"]["text"] = "tampered"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "tampered existing entry"):
            e.produce(FakeOCR(), self.sources, self.output)
        path.write_bytes(original)
        self.image.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "source image SHA256 changed"):
            e.produce(FakeOCR(), self.sources, self.output)

    def test_chunks_preserve_every_token_with_eos_and_padding(self):
        chunks, lengths = e.text_chunks(FakeTokenizer(), "authored")
        self.assertEqual(lengths, [63, 63, 14])
        self.assertEqual([len(part) for part in chunks], [64, 64, 64])
        self.assertEqual([v for part, count in zip(chunks, lengths) for v in part[:count]], list(range(10, 150)))
        for part, count in zip(chunks, lengths):
            self.assertEqual(part[count:], [1] * (64 - count))

    def test_tokenizer_empty_content_is_encoded_as_native_eos_padding(self):
        class Tokenizer(FakeTokenizer):
            def encode(self, text, **kwargs):
                self_options = {"add_special_tokens": False, "truncation": False}
                if kwargs != self_options:
                    raise AssertionError("full untruncated encode required")
                return []

        tokenizer = Tokenizer()
        self.assertEqual(e.text_chunks(tokenizer, "?"), ([[1] * 64], [0]))
        observed = []

        class Model:
            def get_text_features(self, input_ids):
                observed.extend(input_ids.tolist())
                return torch.ones((len(input_ids), 768), dtype=torch.float32)

        engine = e.SigLIP.__new__(e.SigLIP)
        engine.torch, engine.device = torch, "cpu"
        engine.processor = SimpleNamespace(tokenizer=tokenizer)
        engine.model = Model()
        payload = engine.text_vectors(["?"])[0]
        self.assertEqual(observed, [[1] * 64])
        self.assertEqual((payload["token_count"], payload["chunks"], payload["chunk_token_counts"]), (0, 1, [0]))
        self.assertAlmostEqual(sum(x*x for x in payload["vector"]), 1, places=5)

    def test_siglip_mocked_text_model_uses_all_chunks_then_unit_mean(self):
        observed = []

        class Model:
            def get_text_features(self, input_ids):
                observed.extend(input_ids.tolist())
                output = torch.zeros((len(input_ids), 768), dtype=torch.float32)
                output[:, 0] = 1
                output[:, 1] = input_ids[:, 0].float() / 100
                return output

        engine = e.SigLIP.__new__(e.SigLIP)
        engine.torch, engine.device = torch, "cpu"
        engine.processor = SimpleNamespace(tokenizer=FakeTokenizer())
        engine.model = Model()
        payload = engine.text_vectors(["authored"])[0]
        self.assertEqual(payload["chunks"], 3)
        self.assertEqual(payload["token_count"], 140)
        self.assertEqual(len(observed), 3)
        self.assertAlmostEqual(sum(x*x for x in payload["vector"]), 1, places=5)
        chunks = torch.stack([Model().get_text_features(torch.tensor([part]))[0] for part in observed[:3]])
        expected = torch.nn.functional.normalize(torch.nn.functional.normalize(chunks, dim=1).mean(0), dim=0)
        self.assertTrue(torch.allclose(torch.tensor(payload["vector"]), expected))

    def test_sscd_mocked_model_receives_square_float32_and_unit_output(self):
        engine = e.SSCD.__new__(e.SSCD)
        engine.torch, engine.device = torch, "cpu"
        engine.transform = lambda image: torch.ones((3, 320, 320), dtype=torch.float32)

        def model(batch):
            self.assertEqual(tuple(batch.shape), (1, 3, 320, 320))
            self.assertEqual(batch.dtype, torch.float32)
            self.assertFalse(torch.is_grad_enabled())
            return torch.ones((1, 512))

        engine.model = model
        result = engine.image_vectors([Image.new("RGB", (12, 15))])
        self.assertEqual(len(result[0]["vector"]), 512)
        self.assertAlmostEqual(sum(x*x for x in result[0]["vector"]), 1, places=5)
        with self.assertRaisesRegex(ValueError, "zero embedding"):
            e._unit_vectors(torch.zeros((1, 512)), 512)

    def test_siglip_batches_mixed_text_chunks_without_changing_per_text_formula(self):
        counts = {"short-first": 3, "long": 5000, "medium": 130, "short-last": 8}
        batches = []

        class Tokenizer(FakeTokenizer):
            def encode(self, text, **kwargs):
                self_options = {"add_special_tokens": False, "truncation": False}
                if kwargs != self_options:
                    raise AssertionError("full untruncated encode required")
                return list(range(10, 10 + counts[text]))

        def features(ids):
            output = torch.zeros((len(ids), 768), dtype=torch.float32)
            output[:, 0] = ids[:, 0].float() / 10000 + 1
            output[:, 1] = ids.sum(dim=1).float() / 10000
            output[:, 2] = (ids != 1).sum(dim=1).float() / 100
            return output

        class Model:
            def get_text_features(self, input_ids):
                batches.append(input_ids.tolist())
                return features(input_ids)

        engine = e.SigLIP.__new__(e.SigLIP)
        engine.torch, engine.device = torch, "cpu"
        engine.processor = SimpleNamespace(tokenizer=Tokenizer())
        engine.model = Model()
        expected, expected_chunks = [], []
        for text in counts:
            chunks, lengths = e.text_chunks(engine.processor.tokenizer, text)
            expected_chunks.extend(chunks)
            previous_vectors = [e._unit_vectors(features(torch.tensor(chunks[start:start + 64])), 768)
                                for start in range(0, len(chunks), 64)]
            mean = torch.cat(previous_vectors).mean(dim=0, keepdim=True)
            expected.append({"vector": e._unit_vectors(mean, 768)[0].tolist(), "chunks": len(chunks),
                             "token_count": sum(lengths), "chunk_token_counts": lengths,
                             "encoded_chunks_sha256": c._digest(chunks)})
        actual = engine.text_vectors(list(counts))
        self.assertEqual(actual, expected)
        self.assertEqual([len(batch) for batch in batches], [64, 21])
        self.assertEqual([chunk for batch in batches for chunk in batch], expected_chunks)
        self.assertEqual(engine.text_vectors([]), [])

    def test_text_manifest_normalization_and_deduplicated_cache_binding(self):
        text = "Question: Amber  triangle <image 1>"
        normalized = c.normalize_text(text)
        sha = hashlib.sha256(normalized.encode()).hexdigest()
        manifest = self.root / "texts.json"
        manifest.write_text(json.dumps({sha: text}))
        self.assertEqual(e.load_manifest(manifest, texts=True), {sha: normalized})
        manifest.write_text(json.dumps({"a"*64: text}))
        with self.assertRaisesRegex(ValueError, "normalized source text SHA256 differs"):
            e.load_manifest(manifest, texts=True)

    def test_ocr_invocation_pins_language_psm_omp_and_propagates_nonzero_exit(self):
        engine = e.OCR.__new__(e.OCR)
        engine.binary, engine.tessdata = Path("/authored/tesseract"), Path("/authored/tessdata")
        engine.timeout, engine.env = 12, {"OMP_NUM_THREADS": "1"}
        with patch.object(e.subprocess, "run", return_value=SimpleNamespace(stdout=b"  \n")) as run:
            payload = engine.recognize(Image.new("RGB", (12, 15)))
        self.assertEqual(payload["status"], "NO_TEXT")
        args, kwargs = run.call_args
        self.assertEqual(args[0][-4:], ["-l", "eng", "--psm", "3"])
        self.assertEqual(kwargs["timeout"], 12)
        self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
        self.assertTrue(kwargs["check"])
        with patch.object(e.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "authored")):
            with self.assertRaises(subprocess.CalledProcessError):
                engine.recognize(Image.new("RGB", (12, 15)))


if __name__ == "__main__":
    unittest.main()
