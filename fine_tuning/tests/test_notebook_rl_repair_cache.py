"""CPU cache-contract wiring, not model loading or independent reload evidence."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fine_tuning.notebook_rl import quantized_cache
from fine_tuning.notebook_rl.checkpoint import file_hash
from fine_tuning.notebook_rl.local_profile import REPAIR_PROTOCOL_ID, build_protocol
from fine_tuning.notebook_rl.local_run import cache_readiness


class RepairCacheTests(unittest.TestCase):
    def test_explicit_source_extension_preserves_original_twelve(self) -> None:
        legacy, repair = quantized_cache._sources(), quantized_cache._sources(repair=True)
        self.assertEqual(len(legacy), 12)
        self.assertEqual({key: repair[key] for key in legacy}, legacy)
        extra = {f"notebook_rl/{name}.py" for name in (
            "compact_note", "repair_contract", "repair_credit", "repair_tokens", "repair_notebook", "repair_training",
            "repair_evidence", "repair_replay",
            "reasoning_trace", "verifier", "verifier_service", "schema", "rollout", "prompts", "losses", "updates")}
        self.assertEqual(set(repair) - set(legacy), extra)
        invalid_types: tuple[Any, ...] = (1, None, "true")
        for value in invalid_types:
            with self.subTest(value=value), self.assertRaises(ValueError):
                quantized_cache._sources(repair=value)

    def test_saved_readiness_requires_exact_repair_sources(self) -> None:
        protocol = build_protocol(repair_notebook=True, context_limit=24576)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nf4_base.json").write_text("{}")
            (root / "protocol.json").write_text(json.dumps(protocol))
            sources = quantized_cache._sources(repair=True)
            (root / "sources.json").write_text(json.dumps(sources))
            result = {"status": "NF4_CACHE_FRESH_PROCESS_RELOAD_GENERATION_VERIFIED",
                "denied_original_access_attempts": 0,
                "cache_metadata_sha256": file_hash(root / "nf4_base.json")}
            (root / "result.json").write_text(json.dumps(result))
            self.assertEqual(cache_readiness(root, root, protocol), result)
            for invalid in (quantized_cache._sources(), sources | {"notebook_rl/repair_credit.py": "0" * 64}):
                (root / "sources.json").write_text(json.dumps(invalid))
                with self.assertRaisesRegex(ValueError, "independent NF4 cache"):
                    cache_readiness(root, root, protocol)

    def test_cli_exclusivity_before_files_or_gpu_and_explicit_source_dispatch(self) -> None:
        import torch

        base = ["check", "--cache", "/private/base", "--parent", "/private/parent",
                "--output", "/private/unused", "--repair-notebook"]
        invalid: tuple[list[str], ...] = ([], ["--context-limit", "16384"], ["--context-limit", "24576", "--pair-draft-synthesis"],
                   ["--context-limit", "24576", "--whole-reference-notebook"],
                   ["--context-limit", "24576", "--verified-note-synthesis-credit"])
        for extra in invalid:
            with self.subTest(extra=extra), patch("sys.stderr"), patch.object(Path, "mkdir") as mkdir, \
                    patch.object(torch.cuda, "reset_peak_memory_stats") as gpu, self.assertRaises(SystemExit):
                quantized_cache.main(base + extra)
            mkdir.assert_not_called()
            gpu.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cache-check"
            with patch.object(quantized_cache, "_sources", side_effect=ValueError("CPU source boundary")) as sources, \
                    patch.object(quantized_cache, "load_quantized_policy") as loader, \
                    patch("fine_tuning.notebook_rl.policy.enable_deterministic_execution"), \
                    patch.multiple(torch.cuda, reset_peak_memory_stats=lambda: None,
                                   max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0), \
                    patch("traceback.print_exc"), patch("builtins.print"):
                self.assertEqual(quantized_cache.main(["check", "--cache", "/private/base",
                    "--parent", "/private/parent", "--output", str(output), "--repair-notebook",
                    "--context-limit", "24576"]), 1)
            sources.assert_called_once_with(repair=True)
            loader.assert_not_called()
            self.assertEqual(json.loads((output / "result.json").read_text())["error"], "CPU source boundary")
            self.assertEqual(build_protocol(repair_notebook=True, context_limit=24576)["protocol_id"], REPAIR_PROTOCOL_ID)


if __name__ == "__main__":
    unittest.main()
