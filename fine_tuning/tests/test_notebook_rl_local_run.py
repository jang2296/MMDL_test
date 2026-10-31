"""Controller safety/function checks; no model capability or RL success claims."""
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.notebook_rl.local_run import (
    _install_actor_guards, cache_readiness, main, minimal_actors, read_actor_descriptor,
    reload_capture, verify_changes,
)
from fine_tuning.notebook_rl.smoke_data_evidence import _descriptor
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.tests.test_notebook_rl_schema import records


class LocalRunTests(unittest.TestCase):
    def test_context_cli_reaches_prepare_minimum_and_capture_protocol(self):
        from fine_tuning.notebook_rl.local_profile import FACT_CREDIT_24K_PROTOCOL_ID, FACT_CREDIT_PROTOCOL_ID
        from fine_tuning.notebook_rl.local_run import capture

        for extra, context in (([], 16384), (["--context-limit", "16384"], 16384),
                               (["--context-limit", "24576"], 24576)):
            with self.subTest(context=context, extra=extra):
                with patch("fine_tuning.notebook_rl.local_data.prepare_local_data",
                           return_value={"status": "LOCAL_DATA_REVALIDATED"}) as prepare, patch("builtins.print"):
                    self.assertEqual(main(["prepare", "--previous-request", "/private/request.json",
                        "--selection", "/private/selection.json", "--output", "/private/data",
                        "--verified-note-fact-credit", *extra]), 0)
                    self.assertEqual(prepare.call_args.args[2]["context_limit"], context)
                    self.assertEqual(prepare.call_args.args[2]["protocol_id"],
                                     FACT_CREDIT_24K_PROTOCOL_ID if context == 24576 else FACT_CREDIT_PROTOCOL_ID)
                with patch("fine_tuning.notebook_rl.local_run.capture") as called:
                    self.assertEqual(main(["minimal", "--preparation", "/private/preparation.json",
                        "--model-path", "/private/base", "--output", "/private/run",
                        "--verified-note-fact-credit", *extra]), 0)
                    self.assertEqual(called.call_args.kwargs["context_limit"], context)
                with patch("fine_tuning.notebook_rl.local_run.read_preparation",
                           side_effect=ValueError("CPU preparation boundary")) as prepared, \
                        patch("fine_tuning.notebook_rl.quantized_policy.build_quantized_policy") as loader:
                    with self.assertRaisesRegex(ValueError, "CPU preparation boundary"):
                        capture(Path("/private/preparation.json"), Path("/private/base"), Path("/private/run"),
                                standalone=False, verified_note_fact_credit=True, context_limit=context)
                    self.assertEqual(prepared.call_args.args[1]["context_limit"], context)
                    loader.assert_not_called()

    def test_cache_context_cli_propagates_without_model_or_gpu_work(self):
        import torch
        from fine_tuning.notebook_rl import quantized_cache

        with tempfile.TemporaryDirectory() as directory:
            for stage in ("save", "check"):
                for index, (extra, context) in enumerate((([], 16384), (["--context-limit", "16384"], 16384),
                                                          (["--context-limit", "24576"], 24576))):
                    with self.subTest(stage=stage, context=context), \
                            patch("fine_tuning.notebook_rl.quantized_cache._sources", return_value={}), \
                            patch("fine_tuning.notebook_rl.quantized_cache.build_protocol",
                                  side_effect=ValueError("CPU protocol boundary")) as build, \
                            patch("fine_tuning.notebook_rl.policy.enable_deterministic_execution"), \
                            patch.multiple(torch.cuda, reset_peak_memory_stats=lambda: None,
                                           max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0), \
                            patch("fine_tuning.notebook_rl.quantized_cache.build_quantized_policy") as save, \
                            patch("fine_tuning.notebook_rl.quantized_cache.load_quantized_policy") as load, \
                            patch("traceback.print_exc"), patch("builtins.print"):
                        self.assertEqual(quantized_cache.main([stage, "--model-path", "/private/original",
                            "--cache", "/private/cache", "--parent", "/private/parent", "--output",
                            str(Path(directory) / f"{stage}-{index}"), "--verified-note-fact-credit", *extra]), 1)
                        self.assertEqual(build.call_args.kwargs["context_limit"], context)
                        self.assertIs(build.call_args.kwargs["verified_note_fact_credit"], True)
                        save.assert_not_called()
                        load.assert_not_called()

    def test_context_cli_rejects_unsupported_or_non_fact_credit_context_before_work(self):
        from fine_tuning.notebook_rl import quantized_cache

        commands = ((main, ["prepare", "--previous-request", "/private/request.json",
                           "--selection", "/private/selection.json", "--output", "/private/unused"]),
                    (main, ["minimal", "--preparation", "/private/preparation.json",
                            "--model-path", "/private/base", "--output", "/private/unused"]),
                    (quantized_cache.main, ["save", "--model-path", "/private/original",
                                           "--cache", "/private/cache", "--output", "/private/unused"]),
                    (quantized_cache.main, ["check", "--parent", "/private/parent",
                                           "--cache", "/private/cache", "--output", "/private/unused"]))
        for command, args in commands:
            for extra in (["--context-limit", "32768"], ["--context-limit", "24576"],
                          ["--context-limit", "24576", "--verified-note-atoms"],
                          ["--context-limit", "24576", "--verified-note-components"]):
                with self.subTest(stage=args[0], extra=extra), patch("sys.stderr"), \
                        patch.object(Path, "mkdir") as mkdir, self.assertRaises(SystemExit) as stopped:
                    command(args + extra)
                self.assertEqual(stopped.exception.code, 2)
                mkdir.assert_not_called()

    def test_fact_credit_cli_is_explicit_and_reaches_prepare_and_capture(self):
        from fine_tuning.notebook_rl.local_profile import FACT_CREDIT_PROTOCOL_ID

        args = ["prepare", "--previous-request", "/private/request.json", "--selection", "/private/selection.json",
                "--output", "/private/facts", "--verified-note-fact-credit"]
        with patch("fine_tuning.notebook_rl.local_data.prepare_local_data",
                   return_value={"status": "LOCAL_DATA_REVALIDATED"}) as prepare, patch("builtins.print"):
            self.assertEqual(main(args), 0)
            self.assertEqual(prepare.call_args.args[2]["protocol_id"], FACT_CREDIT_PROTOCOL_ID)
        with patch("fine_tuning.notebook_rl.local_run.capture") as capture:
            self.assertEqual(main(["minimal", "--preparation", "/private/preparation.json", "--model-path", "/private/base",
                "--output", "/private/minimum", "--verified-note-fact-credit"]), 0)
            self.assertIs(capture.call_args.kwargs["verified_note_fact_credit"], True)
        for flag in ("--verified-note-atoms", "--verified-note-components"):
            with self.assertRaises(SystemExit):
                main(args + [flag])

    def test_component_cli_reaches_prepare_and_minimum_and_refuses_mixed_selection(self):
        from fine_tuning.notebook_rl.local_profile import COMPONENT_PROTOCOL_ID

        args = ["prepare", "--previous-request", "/private/request.json", "--selection", "/private/selection.json",
                "--output", "/private/components", "--verified-note-components"]
        with patch("fine_tuning.notebook_rl.local_data.prepare_local_data",
                   return_value={"status": "LOCAL_DATA_REVALIDATED"}) as prepare, patch("builtins.print"):
            self.assertEqual(main(args), 0)
            self.assertEqual(prepare.call_args.args[2]["protocol_id"], COMPONENT_PROTOCOL_ID)
        with patch("fine_tuning.notebook_rl.local_run.capture") as capture:
            self.assertEqual(main(["minimal", "--preparation", "/private/preparation.json", "--model-path", "/private/base",
                "--output", "/private/minimum", "--verified-note-components"]), 0)
            self.assertIs(capture.call_args.kwargs["verified_note_components"], True)
            self.assertIs(capture.call_args.kwargs["verified_note_atoms"], False)
        with self.assertRaises(SystemExit):
            main(args + ["--verified-note-atoms"])

    def test_atomic_objective_cli_selection_reaches_prepare_and_minimum(self):
        from fine_tuning.notebook_rl.local_profile import ATOMIC_PROTOCOL_ID

        with patch("fine_tuning.notebook_rl.local_data.prepare_local_data",
                   return_value={"status": "LOCAL_DATA_REVALIDATED"}) as prepare, patch("builtins.print"):
            self.assertEqual(main(["prepare", "--previous-request", "/private/request.json",
                "--selection", "/private/selection.json", "--output", "/private/atoms",
                "--verified-note-atoms"]), 0)
            self.assertEqual(prepare.call_args.args[2]["protocol_id"], ATOMIC_PROTOCOL_ID)
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch("fine_tuning.notebook_rl.local_run.capture") as capture:
                args = ["minimal", "--preparation", "/private/preparation.json", "--model-path", "/private/base",
                        "--standalone-base", "--cache-check", "/private/cache", "--output", "/private/minimum"]
                self.assertEqual(main(args + (["--verified-note-atoms"] if enabled else [])), 0)
                self.assertIs(capture.call_args.kwargs["verified_note_atoms"], enabled)

    def test_prepare_cli_preserves_status_output_and_returns_nonzero_when_blocked(self):
        for status, expected in (("LOCAL_DATA_REVALIDATED", 0), ("BLOCKED", 2)):
            with self.subTest(status=status), \
                    patch("fine_tuning.notebook_rl.local_data.prepare_local_data", return_value={"status": status}), \
                    patch("fine_tuning.notebook_rl.local_run.build_protocol", return_value={}), \
                    patch("builtins.print") as printed:
                code = main(["prepare", "--previous-request", "/private/request.json",
                             "--selection", "/private/selection.json", "--output", "/private/new-output"])
                self.assertEqual(code, expected)
                self.assertEqual(json.loads(printed.call_args.args[0]),
                                 {"status": status, "output": "/private/new-output"})

    def test_nf4_dependencies_are_pinned_and_prepared_before_actor_guard(self):
        events = []
        prepared = {"build_root": "/tmp/local-graph"}
        request = {key: {"path": "/tmp/private/" + key}
                   for key in ("index", "source_registry", "evidence", "data_report")}
        with patch("fine_tuning.notebook_rl.shared_smoke._prepare_hf_dependencies",
                   side_effect=lambda: events.append("hf")), \
                patch("fine_tuning.notebook_rl.quantized_policy._versions",
                      side_effect=lambda: events.append("versions")), \
                patch("fine_tuning.notebook_rl.local_run.import_module",
                      side_effect=lambda name: events.append(name)), \
                patch("fine_tuning.notebook_rl.guards.install_guards",
                      side_effect=lambda **kw: events.append(kw)):
            _install_actor_guards(prepared, request, extra_vault_files=[Path("/tmp/external/gold.jsonl")])
        self.assertEqual(events[:3], ["versions", "hf", "bitsandbytes"])
        self.assertFalse(events[3]["allow_vault"])
        self.assertFalse(events[3]["allow_benchmark"])
        self.assertIn(Path("/tmp/private/index"), events[3]["vault_files"])
        self.assertIn(Path("/tmp/external/gold.jsonl"), events[3]["vault_files"])
        hub = Path.home() / "mmdl-cache/huggingface/hub"
        for name in ("datasets--MMMU--MMMU", "datasets--MMMU--MMMU_Pro"):
            self.assertIn(hub / name, events[3]["forbidden_roots"])

    def test_cache_check_binds_protocol_sources_and_actual_cache(self):
        from fine_tuning.notebook_rl.checkpoint import file_hash
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nf4_base.json").write_text("{}")
            (root / "protocol.json").write_text('{"n":{"1":4}}')
            (root / "sources.json").write_text('{"policy.py":"frozen"}')
            result = {"status": "NF4_CACHE_FRESH_PROCESS_RELOAD_GENERATION_VERIFIED",
                      "denied_original_access_attempts": 0,
                      "cache_metadata_sha256": file_hash(root / "nf4_base.json")}
            (root / "result.json").write_text(json.dumps(result))
            with patch("fine_tuning.notebook_rl.quantized_cache._sources",
                       return_value={"policy.py": "frozen"}):
                self.assertEqual(cache_readiness(root, root, {"n": {1: 4}}), result)
                (root / "nf4_base.json").write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, "independent NF4 cache"):
                    cache_readiness(root, root, {"n": {1: 4}})

    def test_hash_bound_actor_list_and_tamper_rejection(self):
        actor = records()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actors.json"
            path.write_text("[" + to_json(actor) + "]")
            descriptor = _descriptor(path)
            self.assertEqual(read_actor_descriptor(descriptor), [actor])
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                read_actor_descriptor(descriptor)
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "list required"):
                read_actor_descriptor(_descriptor(path))

    def test_minimal_selection_is_fixed_by_input_not_rewards(self):
        actor = records()[0]
        first = replace(actor, problem_id="first", image_paths=("one.png",), image_sha256=("a"*64,))
        pair = replace(actor, problem_id="pair", image_paths=("one.png", "two.png"), image_sha256=("a"*64, "b"*64))
        self.assertEqual(minimal_actors([pair, first, replace(first, problem_id="later")]), [first, pair])
        with self.assertRaisesRegex(ValueError, "coverage"):
            minimal_actors([first])

    def test_parameter_change_requires_llm_merger_and_verified_vision_route(self):
        before = {group: {"tensor": "before"} for group in ("L", "M", "V")}
        after = {group: {"tensor": "after" if group != "V" else "before"} for group in before}
        update = {"status": "UPDATED_UNSYNCHRONIZED", "grad_norm": {"L": 1., "M": 2., "V": None},
                  "routes": {"problem": False}}
        verified = verify_changes(before, after, update)
        self.assertTrue(verified["vision_off_only_observed"])
        self.assertFalse(verified["both_routes_verified"])
        after["V"] = {"tensor": "after"}
        with self.assertRaisesRegex(ValueError, "Vision"):
            verify_changes(before, after, update)
        update["routes"]["problem"] = True
        update["grad_norm"]["V"] = 3.
        self.assertTrue(verify_changes(before, after, update)["vision_on_observed"])
        after["M"] = before["M"]
        with self.assertRaisesRegex(ValueError, "Merger"):
            verify_changes(before, after, update)

    def test_reload_rejects_live_capture_before_model_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "launch.json").write_text(json.dumps({"pid": os.getpid() + 100000}))
            (root / "result.json").write_text("{}")
            (root / "reload_expected.json").write_text("{}")
            with patch("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy") as loader, \
                    patch("psutil.pid_exists", return_value=True) as alive:
                with self.assertRaisesRegex(ValueError, "closed successful update"):
                    reload_capture(root, root / "new")
                loader.assert_not_called()
                alive.assert_called_once_with(os.getpid() + 100000)
            self.assertFalse((root / "new").exists())


class HostScopedLivenessTests(unittest.TestCase):
    """A recorded PID is meaningful only on the host/boot/PID namespace that issued it."""

    def test_owner_alive_is_false_for_another_host_even_if_the_pid_exists_here(self):
        from fine_tuning.notebook_rl.local_run import host_identity, owner_alive

        here, me = host_identity(), os.getpid()
        self.assertEqual(set(here), {"hostname", "boot_id", "pid_ns"})
        self.assertTrue(owner_alive({"pid": me, "host": here}))
        for key in here:
            with self.subTest(different=key):
                self.assertFalse(owner_alive({"pid": me, "host": {**here, key: "elsewhere"}}))
        with patch("psutil.pid_exists", return_value=True) as exists:
            self.assertFalse(owner_alive({"pid": me, "host": {**here, "boot_id": "previous-boot"}}))
            exists.assert_not_called()

    def test_legacy_records_without_host_keep_the_plain_local_pid_check(self):
        from fine_tuning.notebook_rl.local_run import owner_alive

        for exists in (True, False):
            with patch("psutil.pid_exists", return_value=exists) as called:
                self.assertIs(owner_alive({"pid": 4242}), exists)
                called.assert_called_once_with(4242)

    def test_same_process_needs_the_same_host_when_both_records_name_one(self):
        from fine_tuning.notebook_rl.local_run import host_identity, same_process

        here = host_identity()
        other = {**here, "boot_id": "previous-boot"}
        self.assertTrue(same_process({"pid": 7, "host": here}, {"pid": 7, "host": here}))
        self.assertFalse(same_process({"pid": 7, "host": other}, {"pid": 7, "host": here}))
        self.assertFalse(same_process({"pid": 7, "host": here}, {"pid": 8, "host": here}))
        self.assertTrue(same_process({"pid": 7}, {"pid": 7, "host": here}))  # legacy: PID alone, as before

    def test_reload_accepts_a_capture_from_another_boot_whose_pid_is_live_here(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from fine_tuning.notebook_rl.local_run import host_identity

            other = {**host_identity(), "boot_id": "previous-boot"}
            (root / "launch.json").write_text(json.dumps({"pid": os.getpid(), "host": other}))
            (root / "result.json").write_text("{}")
            (root / "reload_expected.json").write_text("{}")
            with patch("psutil.pid_exists", return_value=True):
                with self.assertRaises(Exception) as caught:  # later checks fail on the empty fixture
                    reload_capture(root, root / "new")
            self.assertNotIn("closed successful update in a fresh process", str(caught.exception))

    def test_disk_low_checks_the_output_filesystem_only_when_a_minimum_is_bound(self):
        from fine_tuning.notebook_rl.local_run import disk_low

        gib = 2**30
        with patch("shutil.disk_usage", return_value=SimpleNamespace(total=9 * gib, used=8 * gib, free=gib)) as usage:
            self.assertIsNone(disk_low(Path("/out"), None))
            usage.assert_not_called()
            self.assertIsNone(disk_low(Path("/out"), 1.0))
            self.assertEqual(disk_low(Path("/out"), 1.5), {"path": "/out", "free_bytes": gib, "min_free_bytes": int(1.5 * gib)})


VLLM_SCHEMA_DRIVER = """
import json, sys
from importlib import import_module
from pathlib import Path
from fine_tuning.notebook_rl.local_run import _install_actor_guards
root = Path(sys.argv[1])
prepared = {"build_root": str(root / "build")}
request = {key: {"path": str(root / "private" / key)} for key in ("index", "source_registry", "evidence", "data_report")}
schema = {} if sys.argv[2] == "vllm" else None
_install_actor_guards(prepared, request, vllm_schema=schema)
out = {"schema": schema, "openai_loaded": "openai" in sys.modules}
try:
    out["schema_types"] = import_module("openai.types.chat").ChatCompletionMessageParam is not None
except PermissionError as exc:
    out["schema_error"] = str(exc)
if sys.argv[2] == "vllm":
    import openai
    try:
        openai.OpenAI(api_key="fixture")
    except PermissionError as exc:
        out["client_error"] = str(exc)
(root / "out.json").write_text(json.dumps(out))
"""


class VllmSchemaGuardTests(unittest.TestCase):
    """REAL actor guards in a fresh process (guards are process-wide audit hooks); CPU import only, no model.

    `import vllm` itself cannot be exercised here: under the guards this venv's vLLM files fail the guard's
    hard-link check ("hard-linked open path cannot be checked"), a local install layout. The OpenAI schema
    types vLLM imports (openai.types.chat) stand in; the Pod run showed vLLM loading under the guards."""

    def run_driver(self, engine: str) -> dict:
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as directory:
            done = subprocess.run([sys.executable, "-B", "-c", VLLM_SCHEMA_DRIVER, directory, engine],
                                  capture_output=True, text=True, timeout=600, check=False,
                                  env=os.environ | {"CUDA_VISIBLE_DEVICES": ""})
            self.assertEqual(done.returncode, 0, done.stdout[-2000:] + done.stderr[-4000:])
            return json.loads((Path(directory) / "out.json").read_text())

    def test_schema_types_load_under_guards_only_after_the_dependency_and_api_clients_stay_blocked(self):
        out = self.run_driver("vllm")
        self.assertEqual(out["schema"], {"schema_dependency_guard": {
            "dependency": "openai", "version": "3.16.2", "purpose": "VLLM_SCHEMA_TYPES_ONLY",
            "client_construction_and_requests_blocked": True, "os_network_sandbox": False}})
        self.assertIs(out.get("schema_types"), True, out)
        self.assertIn("API client use blocked", out["client_error"])

    def test_transformers_guards_never_import_openai_and_still_block_the_schema_import(self):
        out = self.run_driver("hf")
        self.assertIsNone(out["schema"])
        self.assertFalse(out["openai_loaded"])  # nothing before the vLLM attempt imported the SDK
        self.assertEqual(out.get("schema_error"), "teacher import blocked", out)


if __name__ == "__main__":
    unittest.main()
