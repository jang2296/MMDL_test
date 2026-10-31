"""CPU-only tests for the Pod chain runner (pod/chain.py) and its plan files.

No GPU, network or real training command: plans are checked against the real argparse parsers,
runs use tiny shell/python steps in temporary run directories."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import shlex  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest import mock  # noqa: E402

from fine_tuning.notebook_rl.pod import chain, cost  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PLANS = REPO / "fine_tuning/notebook_rl/pod/plans"
SHA = "b" * 64
CHAIN_MODULE = "fine_tuning.notebook_rl.pod.chain"
STAGE = {"stage1": "stage1", "pilot-300": "pilot-300", "main-selection": "main", "main-resume": "main"}
JOB_STAGE = {"stage1": "bf16-smoke-throughput", "pilot-300": "pilot-300", "main": "main"}
PREFILTER_STEPS = ["prefilter-propose", "prefilter-prepare", "passrate-measure", "passrate-select"]
PROOF_STEPS = ["propose", "prepare", "minimal", "minimal-reload"]
TAIL_STEPS = ["train", "train-reload", "eval-f0", "eval-ft", "replay-f0", "replay-ft", "compare", "stage-receipt"]
EXPECTED_NAMES = {"stage1": PREFILTER_STEPS + PROOF_STEPS + TAIL_STEPS,
                  "pilot-300": PREFILTER_STEPS + PROOF_STEPS + TAIL_STEPS,
                  "main-selection": ["selection-pin"] + PROOF_STEPS + TAIL_STEPS,
                  "main-resume": TAIL_STEPS}
TIMEOUT_KEY = {"prefilter-propose": "cpu", "prefilter-prepare": "cpu", "passrate-measure": "passrate",
               "passrate-select": "cpu", "selection-pin": "cpu", "propose": "cpu", "prepare": "cpu",
               "minimal": "minimal", "minimal-reload": "reload", "train-reload": "reload",
               "eval-f0": "evaluate", "eval-ft": "evaluate", "replay-f0": "cpu", "replay-ft": "cpu",
               "compare": "cpu", "stage-receipt": "cpu"}
VLLM_TAIL = ["--policy-precision", "bf16", "--rollout-engine", "vllm", "--vllm-gpu-memory-utilization", "0.45",
             "--vllm-max-num-seqs", "8", "--vllm-kv-cache-memory-bytes", "8589934592"]
RULE_25 = '{"every":25,"first":true,"last":true}'
RULE_50 = '{"every":50,"first":true,"last":true}'
RECEIPT = {"version": "throughput-receipt-v1", "items": 10, "completion_tokens": 8000, "wall_seconds": 1000,
           "updates": 5, "update_seconds": 200, "peak_memory_gib": 20.5, "n_samples": 4, "retries_per_item": 0}


def expected_scale(items, per_update, checkpoint=()):
    return ["--problems-per-update", str(per_update), "--max-updates", str(items // per_update),
            "--train-problems", str(items), *checkpoint, *VLLM_TAIL]


# stage1 sizes N/M at run time from its own selection (chain with-train-size fills the tokens).
STAGE1_SCALE = ["--problems-per-update", "4", "--max-updates", chain.SIZE_TOKENS[1], "--train-problems",
                chain.SIZE_TOKENS[0], "--checkpoint-every-updates", "1", *VLLM_TAIL]
TRAIN_SCALE = {"stage1": STAGE1_SCALE,
               "pilot-300": expected_scale(300, 4, ["--checkpoint-every-updates", "25",
                                                    "--evidence-audit-rule", RULE_25]),
               "main-selection": expected_scale(12, 4, ["--checkpoint-every-updates", "50",
                                                        "--evidence-audit-rule", RULE_50])}
PREFILTER_SCALE = {"stage1": expected_scale(16, 1), "pilot-300": expected_scale(3420, 1)}
PREFILTER_VLLM = {"gpu_memory_utilization": 0.85, "max_num_seqs": 64, "kv_cache_memory_bytes": 21474836480,
                  "enable_prefix_caching": True}  # a valid prefilter_vllm budget for the tests below


def make_params(case):
    """Valid synthetic params for one case (stage1, pilot-300, main-selection, main-resume)."""
    params = {
        "version": chain.PARAMS_VERSION, "python": "/workspace/venv/bin/python",
        "run_dir": "/workspace/runs/r1", "model_path": "/workspace/mmdl-volume/model",
        "vault": "/workspace/vault", "approval_request": "/workspace/data/data_request.json",
        "frozen_approval": {"path": "/workspace/data/approval_export.json", "sha256": "a" * 64},
        "seed": 3407, "sentinels_per_subject": 2, "min_free_gib": 10,
        "vllm": {"gpu_memory_utilization": 0.45, "max_num_seqs": 8, "kv_cache_memory_bytes": 8589934592,
                 "enable_prefix_caching": False},
        "timeouts": {"cpu": 1800, "passrate": 2400, "minimal": 2400, "reload": 1200, "train_close": 1200,
                     "evaluate": 2400}}
    if case == "stage1":
        params["sentinels_per_subject"] = 1
        params["train"] = {"problems": 8, "per_update": 4, "checkpoint_every": 1, "evidence_audit_rule": None,
                           "max_seconds": 1800}
        params["prefilter"] = {"items": 16, "k": 4, "seed_namespace": "proof-stage1-f0", "min_pass": 1,
                               "max_pass": 3, "max_blocking": 0}
    elif case == "pilot-300":
        params["train"] = {"problems": 300, "per_update": 4, "checkpoint_every": 25,
                           "evidence_audit_rule": {"first": True, "last": True, "every": 25},
                           "max_seconds": 43200}
        params["prefilter"] = {"items": 3420, "k": 4, "seed_namespace": "proof-pilot-f0", "min_pass": 1,
                               "max_pass": 3, "max_blocking": 0}
    else:
        params["train"] = {"problems": 12, "per_update": 4, "checkpoint_every": 50,
                           "evidence_audit_rule": {"first": True, "last": True, "every": 50},
                           "max_seconds": 86400}
        if case == "main-selection":
            params["selection"] = {"path": "/workspace/runs/pilot/out/passrate-selection.json",
                                   "sha256": "b" * 64, "report": "/workspace/runs/pilot/out/passrate"}
        else:
            params["resume"] = {"from": "/workspace/runs/seg1/out/train",
                                "preparation": "/workspace/runs/seg1/out/prep/preparation.json",
                                "minimal": "/workspace/runs/seg1/out/minimal",
                                "minimal_reload": "/workspace/runs/seg1/out/minimal-reload"}
    return params


def planned(case):
    return chain.plan(STAGE[case], make_params(case), SHA)


def argvs(spec):
    return {item["name"]: item["argv"] for item in spec["steps"]}


def value(argv, flag):
    hits = [index for index, token in enumerate(argv) if token == flag]
    assert len(hits) == 1, (flag, argv)
    return argv[hits[0] + 1]


def contains(argv, sequence):
    return any(argv[i:i + len(sequence)] == sequence for i in range(len(argv) - len(sequence) + 1))


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(*args, **kwargs)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child_env(run_dir):
    env = {**os.environ, "MMDL_RUN_DIR": str(run_dir), "PYTHONPATH": f"{REPO}/src:{REPO}",
           "CUDA_VISIBLE_DEVICES": ""}
    env.pop("MMDL_JOB_STAGE", None)
    return env


# ---------------------------------------------------------------- real-parser helper

class _Parsed(Exception):
    def __init__(self, parser, namespace):
        super().__init__("parse_args reached")
        self.parser, self.namespace = parser, namespace


_REAL_PARSE_ARGS = argparse.ArgumentParser.parse_args


def _stop_after_parse(self, args=None, namespace=None):
    raise _Parsed(self, _REAL_PARSE_ARGS(self, args, namespace))


def parse_with_real_parser(test, argv):
    """Run `<python> -B -m fine_tuning.notebook_rl.<x> <subcommand> ...` through the module's own argparse
    parser (nothing after parse_args runs) and require every `--flag` to be an exact option (no abbreviation)."""
    test.assertEqual(["-B", "-m"], argv[1:3], argv)
    name = argv[3]
    test.assertIn(name, {f"fine_tuning.notebook_rl.{x}" for x in
                         ("local_run", "local_train", "local_eval", "passrate", "pod.chain")}, argv)
    module = importlib.import_module(name)
    stderr = io.StringIO()
    with mock.patch.object(argparse.ArgumentParser, "parse_args", autospec=True,
                           side_effect=_stop_after_parse), contextlib.redirect_stderr(stderr):
        try:
            module.main(argv[4:])
        except _Parsed as stop:
            parsed = stop
        except SystemExit:
            test.fail(f"argparse rejected {argv[3:]}: {stderr.getvalue().strip()}")
        else:
            test.fail(f"parse_args was never reached for {argv[3:]}")
    action = next(a for a in parsed.parser._actions if isinstance(a, argparse._SubParsersAction))
    sub = action.choices[argv[4]]
    tokens, inner = argv[5:], None
    if "--" in tokens:
        cut = tokens.index("--")
        tokens, inner = tokens[:cut], tokens[cut + 1:]
    for token in tokens:
        if token.startswith("--"):
            test.assertIn(token, sub._option_string_actions,
                          f"{argv[3]} {argv[4]}: {token} is not an exact flag of the real parser")
    if (name, argv[4]) == (CHAIN_MODULE, "with-sha256"):
        test.assertIsNotNone(inner, argv)
        command = parsed.namespace.cmd
        test.assertEqual(inner, command[1:] if command[:1] == ["--"] else command)
        test.assertIn(chain.PREP_TOKEN, inner)
        parse_with_real_parser(test, [token.replace(chain.PREP_TOKEN, "0" * 64) for token in inner])
    if (name, argv[4]) == (CHAIN_MODULE, "with-train-size"):
        test.assertIsNotNone(inner, argv)
        command = parsed.namespace.cmd
        test.assertEqual(inner, command[1:] if command[:1] == ["--"] else command)
        parse_with_real_parser(test, chain.sized_argv({"selected_ids": ["a"] * 8}, 8, 4, inner))


# ---------------------------------------------------------------- A. plan

class PlanTests(unittest.TestCase):
    def test_plan_is_deterministic_and_the_cli_writes_identical_files(self):
        for case, stage in STAGE.items():
            with self.subTest(case), tempfile.TemporaryDirectory() as tmp:
                params = make_params(case)
                self.assertEqual(chain.plan(stage, params, SHA), chain.plan(stage, params, SHA))
                source, outputs = Path(tmp) / "params.json", [Path(tmp) / "a.json", Path(tmp) / "b.json"]
                source.write_text(json.dumps(params))
                for output in outputs:
                    self.assertEqual(0, quiet(chain.main, ["plan", "--stage", stage, "--params", str(source),
                                                           "--output", str(output)]))
                self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
                self.assertEqual(chain.plan(stage, params, sha256_of(source)), json.loads(outputs[0].read_text()))
                before = outputs[0].read_bytes()
                self.assertEqual(2, quiet(chain.main, ["plan", "--stage", stage, "--params", str(source),
                                                       "--output", str(outputs[0])]))
                self.assertEqual(before, outputs[0].read_bytes())

    def test_every_planned_flag_is_an_exact_flag_of_the_real_parser(self):
        for case in STAGE:
            spec = planned(case)
            self.assertIs(spec, chain.check_spec(spec))
            for item in spec["steps"]:
                with self.subTest(case=case, step=item["name"]):
                    parse_with_real_parser(self, item["argv"])

    def test_commands_of_one_preparation_carry_the_same_protocol_flags(self):
        for case in ("stage1", "pilot-300", "main-selection"):
            steps = argvs(planned(case))
            for name in ("propose", "prepare", "minimal"):
                with self.subTest(case=case, step=name):
                    self.assertTrue(contains(steps[name], chain.PROOF_FLAGS + TRAIN_SCALE[case]), steps[name])
            if case in PREFILTER_SCALE:  # K=1, M=prefilter.items, no checkpoint or audit flags
                for name in ("prefilter-propose", "prefilter-prepare", "passrate-measure"):
                    with self.subTest(case=case, step=name):
                        self.assertTrue(contains(steps[name], chain.PROOF_FLAGS + PREFILTER_SCALE[case]),
                                        steps[name])
                params = make_params(case)["prefilter"]
                self.assertEqual(str(params["k"]), value(steps["passrate-measure"], "--k"))
                self.assertEqual(params["seed_namespace"], value(steps["passrate-measure"], "--seed-namespace"))

    def test_proof_t1_cap_is_8192_in_every_protocol_command(self):
        # user decision 2026-10-31: T1 cap 8192 (= the fast8k eval cap); retries/notes keep their own caps
        self.assertEqual("8192", value(chain.PROOF_FLAGS, "--turn1-max-new-tokens"))
        for case in ("stage1", "pilot-300", "main-selection"):
            for name, argv in argvs(planned(case)).items():
                if "--turn1-contract" in argv:
                    self.assertEqual("8192", value(argv, "--turn1-max-new-tokens"), (case, name))

    def test_prefix_caching_flag_follows_the_vllm_block_in_every_protocol_command(self):
        params = make_params("stage1")
        params["vllm"]["enable_prefix_caching"] = True
        steps = argvs(chain.plan("stage1", params, SHA))
        for name in ("prefilter-propose", "prefilter-prepare", "passrate-measure", "propose", "prepare", "minimal"):
            self.assertIn("--vllm-enable-prefix-caching", steps[name], name)
        for name, argv in argvs(planned("stage1")).items():
            self.assertNotIn("--vllm-enable-prefix-caching", argv, name)

    def test_train_takes_only_the_train_flags_and_the_admission(self):
        for case in STAGE:
            with self.subTest(case):
                train = argvs(planned(case))["train"]
                self.assertTrue(contains(train, chain.TRAIN_FLAGS), train)
                self.assertEqual([], [token for token in train if token.startswith("--turn1-")])
                for flag in ("--retry-out-of-choice", "--problems-per-update"):
                    self.assertNotIn(flag, train)
                self.assertEqual("verified-functional-repair-exploration-v1", value(train, "--admission"))
                self.assertEqual("notebook", value(train, "--arm"))
                self.assertEqual(str(make_params(case)["train"]["max_seconds"]), value(train, "--max-seconds"))

    def test_frozen_approval_is_bound_to_every_step_that_builds_or_trains_and_to_no_reload_or_eval(self):
        bound = {"stage1": ["prefilter-propose", "prefilter-prepare", "passrate-measure", "propose", "prepare",
                            "minimal", "train"],
                 "main-selection": ["propose", "prepare", "minimal", "train"], "main-resume": ["train"]}
        bound["pilot-300"] = bound["stage1"]
        for case in STAGE:
            steps, params = argvs(planned(case)), make_params(case)["frozen_approval"]
            for name in bound[case]:
                with self.subTest(case=case, step=name):
                    self.assertEqual(params["path"], value(steps[name], "--frozen-approval"))
                    self.assertEqual(params["sha256"], value(steps[name], "--frozen-approval-sha256"))
            for name in ("minimal-reload", "train-reload", "eval-f0", "eval-ft"):
                with self.subTest(case=case, step=name):
                    if name in steps:
                        self.assertNotIn("--frozen-approval", steps[name])
                        self.assertNotIn("--frozen-approval-sha256", steps[name])

    def test_stage_shape(self):
        for case, stage in STAGE.items():
            with self.subTest(case):
                params, spec = make_params(case), planned(case)
                steps, names = argvs(spec), [item["name"] for item in spec["steps"]]
                run_dir, py = params["run_dir"], params["python"]
                self.assertEqual(EXPECTED_NAMES[case], names)
                self.assertEqual("stage-receipt", names[-1])
                for item in spec["steps"]:
                    key = TIMEOUT_KEY.get(item["name"])
                    if key:
                        self.assertEqual(params["timeouts"][key], item["timeout_seconds"], item["name"])
                    for path in item["expect_files"]:
                        self.assertTrue(path.startswith(run_dir + "/"), (item["name"], path))
                train = next(item for item in spec["steps"] if item["name"] == "train")
                self.assertEqual(params["train"]["max_seconds"] + params["timeouts"]["train_close"],
                                 train["timeout_seconds"])
                meta = spec["meta"]
                self.assertEqual(JOB_STAGE[stage], meta["job_stage"])
                self.assertEqual(stage, meta["stage"])
                self.assertEqual(run_dir, meta["run_dir"])
                self.assertEqual(SHA, meta["params_sha256"])
                self.assertEqual(case == "main-resume", meta["resume"])
                self.assertEqual(sum(item["timeout_seconds"] for item in spec["steps"]),
                                 meta["total_timeout_seconds"])
                if case == "stage1":  # its own selection, sized at run time for propose/prepare/minimal only
                    self.assertEqual(f"{run_dir}/out/passrate-selection.json",
                                     value(steps["propose"], "--passrate-selection"))
                    self.assertEqual(f"{run_dir}/out/passrate", value(steps["propose"], "--passrate-report"))
                    size = [py, "-B", "-m", CHAIN_MODULE, "with-train-size", "--selection",
                            f"{run_dir}/out/passrate-selection.json", "--max-items", "8", "--per-update", "4", "--"]
                    for name in names:
                        self.assertIs(steps[name][:len(size)] == size, name in ("propose", "prepare", "minimal"), name)
                if case == "pilot-300":
                    self.assertEqual(f"{run_dir}/out/passrate-selection.json",
                                     value(steps["propose"], "--passrate-selection"))
                    self.assertEqual(f"{run_dir}/out/passrate", value(steps["propose"], "--passrate-report"))
                    self.assertEqual(f"{run_dir}/out/passrate-selection.json",
                                     value(steps["passrate-select"], "--output"))
                elif case == "main-selection":
                    chosen = params["selection"]
                    self.assertEqual(names[0], "selection-pin")
                    self.assertEqual([py, "-B", "-m", CHAIN_MODULE, "pin", "--file", chosen["path"],
                                      "--sha256", chosen["sha256"]], steps["selection-pin"])
                    self.assertEqual(chosen["path"], value(steps["propose"], "--passrate-selection"))
                    self.assertEqual(chosen["report"], value(steps["propose"], "--passrate-report"))
                if case == "main-resume":
                    resume = params["resume"]
                    self.assertEqual(resume["from"], value(steps["train"], "--resume-from"))
                    for name in ("train", "eval-f0", "eval-ft"):
                        self.assertEqual(resume["preparation"], value(steps[name], "--preparation"), name)
                    self.assertEqual(resume["minimal"], value(steps["train"], "--minimal"))
                    self.assertEqual(resume["minimal_reload"], value(steps["train"], "--minimal-reload"))
                    self.assertEqual(resume["minimal"], value(steps["eval-ft"], "--minimal-capture"))
                    self.assertEqual(resume["minimal_reload"], value(steps["eval-ft"], "--minimal-reload"))
                else:
                    self.assertNotIn("--resume-from", steps["train"])
                    self.assertEqual(f"{run_dir}/out/prep/preparation.json", value(steps["train"], "--preparation"))
                    self.assertEqual(f"{run_dir}/out/minimal", value(steps["train"], "--minimal"))
                    self.assertEqual(f"{run_dir}/out/minimal-reload", value(steps["train"], "--minimal-reload"))

    def test_params_are_refused(self):
        delete = object()
        multiple = make_params("main-resume")["resume"]
        cases = [
            ("N not a multiple of K", "pilot-300", ("train", "problems"), 302, "multiple"),
            ("main over the 3000 cap", "main-selection", ("train", "problems"), 3004, "3000"),
            ("prefilter k above range", "pilot-300", ("prefilter", "k"), 9, None),
            ("prefilter k below range", "pilot-300", ("prefilter", "k"), 3, None),
            ("prefilter pass band above k", "stage1", ("prefilter", "max_pass"), 9, None),
            ("missing common key", "stage1", ("seed",), delete, None),
            ("unknown key", "stage1", ("bogus",), 1, None),
            ("selection key on stage1", "stage1", ("selection",), make_params("main-selection")["selection"], None),
            ("missing prefilter", "pilot-300", ("prefilter",), delete, None),
            ("wrong version", "stage1", ("version",), "pod-chain-params-v0", None),
            ("main with both selection and resume", "main-selection", ("resume",), multiple, None),
            ("main with neither", "main-selection", ("selection",), delete, None),
            ("relative path", "stage1", ("run_dir",), "workspace/runs/r1", None),
            ("path with ..", "stage1", ("vault",), "/workspace/runs/../vault", None),
            ("path with trailing slash", "stage1", ("model_path",), "/workspace/model/", None),
            ("relative resume path", "main-resume", ("resume", "from"), "out/train", None),
            ("uppercase sha256", "stage1", ("frozen_approval", "sha256"), "A" * 64, None),
            ("short selection sha256", "main-selection", ("selection", "sha256"), "b" * 63, None),
            ("max_seconds above the cap", "main-selection", ("train", "max_seconds"), 86401, None),
            ("vllm utilization not a float", "stage1", ("vllm", "gpu_memory_utilization"), 1, None),
            ("extra train key", "stage1", ("train", "extra"), 1, None),
            ("zero timeout", "stage1", ("timeouts", "cpu"), 0, None),
            ("prefilter item_batch zero", "pilot-300", ("prefilter", "item_batch"), 0, None),
            ("prefilter item_batch not an int", "stage1", ("prefilter", "item_batch"), True, None),
            ("prefilter item_batch string", "stage1", ("prefilter", "item_batch"), "8", None),
            ("unknown prefilter key", "stage1", ("prefilter", "bogus"), 1, None),
        ]
        for label, case, path, new, pattern in cases:
            params = make_params(case)
            target = params
            for key in path[:-1]:
                target = target[key]
            if new is delete:
                del target[path[-1]]
            else:
                target[path[-1]] = new
            with self.subTest(label), self.assertRaisesRegex(chain.ChainError, pattern or ""):
                chain.plan(STAGE[case], params, SHA)

    def test_the_main_cap_itself_is_accepted(self):
        params = make_params("main-selection")
        params["train"]["problems"] = 3000
        self.assertEqual(750, int(value(argvs(chain.plan("main", params, SHA))["propose"], "--max-updates")))

    def test_prefilter_item_batch_is_optional_and_forwarded_only_to_passrate_measure(self):
        for case in ("stage1", "pilot-300"):
            with self.subTest(case):
                base = argvs(planned(case))
                self.assertFalse(any("--item-batch" in argv for argv in base.values()))
                params = make_params(case)
                params["prefilter"]["item_batch"] = 8
                steps = argvs(chain.plan(STAGE[case], params, SHA))
                self.assertEqual("8", value(steps["passrate-measure"], "--item-batch"))
                for name, argv in steps.items():
                    if name != "passrate-measure":
                        self.assertNotIn("--item-batch", argv, name)
                        self.assertEqual(base[name], argv, name)
                stripped = list(steps["passrate-measure"])
                cut = stripped.index("--item-batch")
                self.assertEqual(["--item-batch", "8"], stripped[cut:cut + 2])
                del stripped[cut:cut + 2]
                self.assertEqual(base["passrate-measure"], stripped)
                parse_with_real_parser(self, steps["passrate-measure"])

    def test_prefilter_vllm_budget_applies_only_to_the_prefilter_steps(self):
        prefilter_names = ("prefilter-propose", "prefilter-prepare", "passrate-measure")
        for case in ("stage1", "pilot-300"):
            with self.subTest(case):
                params = make_params(case)
                params["prefilter_vllm"] = {**PREFILTER_VLLM}
                steps = argvs(chain.plan(STAGE[case], params, SHA))
                base = argvs(planned(case))
                for name in prefilter_names:
                    argv = steps[name]
                    self.assertEqual("0.85", value(argv, "--vllm-gpu-memory-utilization"), name)
                    self.assertEqual("64", value(argv, "--vllm-max-num-seqs"), name)
                    self.assertEqual("21474836480", value(argv, "--vllm-kv-cache-memory-bytes"), name)
                    self.assertIn("--vllm-enable-prefix-caching", argv, name)
                for name, argv in steps.items():
                    if name not in prefilter_names:
                        self.assertEqual(base[name], argv, name)
                parse_with_real_parser(self, steps["passrate-measure"])
                parse_with_real_parser(self, steps["prefilter-propose"])

    def test_absent_prefilter_vllm_leaves_the_planned_argv_unchanged(self):
        for case in STAGE:
            with self.subTest(case):
                spec = chain.plan(STAGE[case], make_params(case), SHA)
                self.assertEqual(planned(case), spec)
                for item in spec["steps"]:
                    self.assertNotIn("0.85", item["argv"], item["name"])
        steps = argvs(planned("stage1"))
        for name in ("prefilter-propose", "prefilter-prepare", "passrate-measure"):
            self.assertTrue(contains(steps[name], VLLM_TAIL), name)

    def test_prefilter_vllm_is_refused_when_misplaced_or_malformed(self):
        cases = [
            ("no prefilter in the stage", "main-selection", {**PREFILTER_VLLM}),
            ("missing key", "stage1", {key: val for key, val in PREFILTER_VLLM.items() if key != "max_num_seqs"}),
            ("extra key", "stage1", {**PREFILTER_VLLM, "bogus": 1}),
            ("utilization is an int", "stage1", {**PREFILTER_VLLM, "gpu_memory_utilization": 1}),
            ("max_num_seqs zero", "stage1", {**PREFILTER_VLLM, "max_num_seqs": 0}),
            ("kv bytes is a bool", "stage1", {**PREFILTER_VLLM, "kv_cache_memory_bytes": True}),
            ("prefix caching is a string", "stage1", {**PREFILTER_VLLM, "enable_prefix_caching": "yes"}),
        ]
        for label, case, budget in cases:
            with self.subTest(label):
                params = make_params(case)
                params["prefilter_vllm"] = budget
                with self.assertRaises(chain.ChainError):
                    chain.plan(STAGE[case], params, SHA)


def fill(item, key=""):
    if isinstance(item, dict):
        return {name: fill(child, name) for name, child in item.items()}
    if isinstance(item, list):
        return [fill(child, key) for child in item]
    if isinstance(item, str) and item.startswith("<"):
        return "a" * 64 if key == "sha256" else 16 if key in ("items", "problems") else "/x/" + key
    return item


class TemplateTests(unittest.TestCase):
    stages = {"params-stage1": "stage1", "params-pilot-300": "pilot-300", "params-main": "main",
              "params-main-resume": "main"}

    def test_templates_need_filling_and_then_plan(self):
        files = sorted(PLANS.glob("params-*.json"))
        self.assertEqual(sorted(self.stages), sorted(path.stem for path in files))
        for path in files:
            stage, template = self.stages[path.stem], json.loads(path.read_text())
            with self.subTest(path.name):
                with self.assertRaises(chain.ChainError):
                    chain.check_params(stage, template)
                filled = fill(template)
                self.assertNotEqual(template, filled)
                chain.check_params(stage, filled)
                spec = chain.check_spec(chain.plan(stage, filled, SHA))
                if "prefilter" in template:  # user decision 2026-10-31: k=4, keep 1..3 passes
                    pre = template["prefilter"]
                    self.assertEqual((4, 1, 3, 0), (pre["k"], pre["min_pass"], pre["max_pass"], pre["max_blocking"]))
                    steps = argvs(spec)
                    self.assertEqual("4", value(steps["passrate-measure"], "--k"))
                    self.assertEqual(("1", "3"), (value(steps["passrate-select"], "--min-pass"),
                                                  value(steps["passrate-select"], "--max-pass")))
                else:
                    self.assertIn("k=4", " ".join(template["notes"]))

    def test_cost_plans_project_a_synthetic_receipt(self):
        expected = {"cost-plan-v1.json": ["stage1", "pilot-300", "main"],
                    "cost-plan-prefilter-v1.json": ["stage1-prefilter", "pilot-300-prefilter"]}
        for name, stages in expected.items():
            plan = json.loads((PLANS / name).read_text())
            with self.subTest(name):
                self.assertEqual(0.59, plan["hourly_usd"])
                self.assertEqual(0.5, plan["range"])
                result = cost.project(RECEIPT, plan, receipt_sha256="r" * 64, plan_sha256="p" * 64)
                self.assertEqual(stages, [row["name"] for row in result["stages"]])
                self.assertEqual("projection", result["label"])
                self.assertTrue(plan["notes"][0].startswith("ESTIMATES:"))
                if name == "cost-plan-prefilter-v1.json":  # k=4 first attempts per item
                    self.assertEqual({4}, {row["n_samples"] for row in plan["stages"]})


# ---------------------------------------------------------------- B. run

def live_in_group(pgid):
    """Pids of non-zombie processes whose process group is pgid."""
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                tail = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            except (OSError, IndexError):
                continue
            if tail[0] != "Z" and int(tail[2]) == pgid:
                found.append(int(entry.name))
    return found


def alive(pid):
    try:
        return (Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]) != "Z"
    except OSError:
        return False


def wait_until(predicate, seconds=30.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if result := predicate():
            return result
        time.sleep(0.05)
    return predicate()


class RunCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.run_dir = self.tmp / "run"
        self.run_dir.mkdir()
        self.enterContext(mock.patch.dict(os.environ, {"MMDL_RUN_DIR": str(self.run_dir)}))
        os.environ.pop("MMDL_JOB_STAGE", None)
        self.log = self.run_dir / chain.LOG_NAME

    @staticmethod
    def step(name, argv, expect=(), timeout=None):
        item = {"name": name, "argv": list(argv), "expect_files": [str(path) for path in expect]}
        return item if timeout is None else {**item, "timeout_seconds": timeout}

    @staticmethod
    def py(code):
        return [sys.executable, "-c", code]

    def touch(self, path):
        return self.py(f"open({str(path)!r}, 'w').close()")

    def write_spec(self, steps, meta=None):
        spec = {"version": chain.SPEC_VERSION, "steps": steps, **({"meta": meta} if meta else {})}
        path = self.tmp / "steps.json"
        path.write_text(json.dumps(spec))
        self.spec_path, self.sha = path, sha256_of(path)
        return path, self.sha

    def execute(self, steps, meta=None, **kwargs):
        path, sha = self.write_spec(steps, meta)
        code = quiet(chain.run, path, sha, **kwargs)
        return code, chain.verify_log(self.log, sha)

    def test_spec_sha_mismatch_is_refused_before_anything_runs(self):
        path, _ = self.write_spec([self.step("one", self.touch(self.tmp / "m"))])
        with self.assertRaises(chain.ChainError):
            quiet(chain.run, path, "0" * 64)
        self.assertEqual(2, quiet(chain.main, ["run", "--spec", str(path), "--spec-sha256", "0" * 64]))
        self.assertFalse(self.log.exists())
        self.assertFalse((self.tmp / "m").exists())

    def test_the_first_failure_stops_the_chain(self):
        first, last = self.tmp / "first", self.tmp / "last"
        code, records = self.execute([self.step("one", self.touch(first), [first]),
                                      self.step("two", self.py("raise SystemExit(3)")),
                                      self.step("three", self.touch(last))])
        self.assertEqual(3, code)
        self.assertTrue(first.exists())
        self.assertFalse(last.exists())
        self.assertEqual(["CHAIN_START", "STEP_START", "STEP_END", "STEP_START", "STEP_END", "CHAIN_END"],
                         [record["event"] for record in records])
        self.assertEqual(["OK", "FAILED"], [r["status"] for r in records if r["event"] == "STEP_END"])
        failed = records[4]
        self.assertEqual((3, "two"), (failed["exit_code"], failed["name"]))
        self.assertEqual(("STOPPED", 3, "two"),
                         tuple(records[-1][key] for key in ("status", "exit_code", "failed_step")))

    def test_a_missing_expected_file_stops_the_chain_with_exit_1(self):
        missing, later = self.tmp / "missing", self.tmp / "later"
        code, records = self.execute([self.step("one", self.py("pass"), [missing]),
                                      self.step("two", self.touch(later))])
        self.assertEqual(1, code)
        self.assertFalse(later.exists())
        end = next(record for record in records if record["event"] == "STEP_END")
        self.assertEqual(("EXPECTED_FILES_MISSING", 1, {str(missing): False}),
                         (end["status"], end["exit_code"], end["expect_files"]))
        self.assertEqual(("STOPPED", 1, "one"),
                         tuple(records[-1][key] for key in ("status", "exit_code", "failed_step")))

    def test_a_timeout_kills_the_step_even_when_it_ignores_sigterm(self):
        pidfile = self.tmp / "pid"
        script = f"echo $$ > {shlex.quote(str(pidfile))}; trap '' TERM; sleep 30"
        began = time.monotonic()
        code, records = self.execute([self.step("slow", ["bash", "-c", script], timeout=1)], grace=0.5)
        self.assertEqual(124, code)
        self.assertLess(time.monotonic() - began, 10)
        end = next(record for record in records if record["event"] == "STEP_END")
        self.assertEqual(("TIMEOUT", 124), (end["status"], end["exit_code"]))
        self.assertEqual(("STOPPED", 124, "slow"),
                         tuple(records[-1][key] for key in ("status", "exit_code", "failed_step")))
        pid = int(pidfile.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)  # killed and reaped by the chain
        self.assertTrue(wait_until(lambda: not live_in_group(pid), 5))

    def test_sigterm_reaches_the_step_group_and_the_chain_exits_143(self):
        pidfile, sleepfile, mark = self.tmp / "pid", self.tmp / "sleep-pid", self.tmp / "mark"
        script = (f'trap "echo term > {mark}; exit 7" TERM; sleep 60 & echo $! > {sleepfile}; '
                  f"echo $$ > {pidfile}; wait")
        path, sha = self.write_spec([self.step("long", ["bash", "-c", script])])
        proc = subprocess.Popen([sys.executable, "-B", "-m", CHAIN_MODULE, "run", "--spec", str(path),
                                 "--spec-sha256", sha], cwd=REPO, env=child_env(self.run_dir),
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        bash_pid = None

        def cleanup():
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if bash_pid and live_in_group(bash_pid):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(bash_pid, signal.SIGKILL)

        self.addCleanup(cleanup)
        wait_until(lambda: pidfile.exists() and pidfile.read_text().strip().isdigit() or proc.poll() is not None)
        if proc.poll() is not None:
            self.fail(f"chain exited early with {proc.returncode}: {proc.stderr.read()}")
        bash_pid, sleep_pid = int(pidfile.read_text()), int(sleepfile.read_text())
        proc.send_signal(signal.SIGTERM)
        _, stderr = proc.communicate(timeout=30)
        self.assertEqual(143, proc.returncode, stderr)
        self.assertTrue(mark.exists())
        self.assertTrue(wait_until(lambda: not alive(bash_pid) and not alive(sleep_pid), 10))
        self.assertEqual([], live_in_group(bash_pid))
        records = chain.verify_log(self.log, sha)
        end = next(record for record in records if record["event"] == "STEP_END")
        self.assertEqual(("INTERRUPTED", 143), (end["status"], end["exit_code"]))
        self.assertEqual(("INTERRUPTED", 143, "long"),
                         tuple(records[-1][key] for key in ("status", "exit_code", "failed_step")))

    def test_the_log_is_a_hash_chain_anchored_at_the_spec(self):
        mark = self.tmp / "mark"
        code, records = self.execute([self.step("one", self.touch(mark), [mark]), self.step("two", self.py("pass"))])
        self.assertEqual(0, code)
        self.assertEqual(["CHAIN_START", "STEP_START", "STEP_END", "STEP_START", "STEP_END", "CHAIN_END"],
                         [record["event"] for record in records])
        self.assertEqual(self.sha, records[0]["prev_sha256"])
        self.assertEqual(["one", "two"], records[0]["steps"])
        self.assertEqual({str(mark): True}, records[2]["expect_files"])
        self.assertEqual(("COMPLETE", 0, None),
                         tuple(records[-1][key] for key in ("status", "exit_code", "failed_step")))
        log = self.log.read_text().splitlines()
        self.assertEqual(0, quiet(chain.main, ["verify", "--log", str(self.log), "--spec-sha256", self.sha]))
        with self.assertRaises(chain.ChainError):
            chain.verify_log(self.log, "0" * 64)
        flipped = log[:2] + [log[2].replace('"status":"OK"', '"status":"OX"', 1)] + log[3:]
        self.assertNotEqual(log, flipped)
        for label, lines in (("flipped character", flipped), ("dropped record", log[:1] + log[2:])):
            with self.subTest(label):
                tampered = self.tmp / "tampered.jsonl"
                tampered.write_text("\n".join(lines) + "\n")
                with self.assertRaises(chain.ChainError):
                    chain.verify_log(tampered, self.sha)
                self.assertEqual(2, quiet(chain.main, ["verify", "--log", str(tampered), "--spec-sha256", self.sha]))

    def test_unsafe_runs_are_refused_without_writing_a_log(self):
        steps = [self.step("one", self.py("pass"))]
        path, _ = self.write_spec(steps)
        other = self.tmp / "other"
        other.mkdir()
        for label, meta, env in (("run dir differs from MMDL_RUN_DIR", {"run_dir": str(other)}, {}),
                                 ("job stage differs", {"job_stage": "main"}, {"MMDL_JOB_STAGE": "pilot-300"})):
            path, sha = self.write_spec(steps, meta)
            with self.subTest(label), mock.patch.dict(os.environ, env), self.assertRaises(chain.ChainError):
                quiet(chain.run, path, sha)
            self.assertFalse(self.log.exists())
        path, sha = self.write_spec(steps)
        for label, change in (("MMDL_RUN_DIR unset", None), ("MMDL_RUN_DIR missing dir", str(self.tmp / "nope")),
                              ("MMDL_RUN_DIR relative", "run")):
            with self.subTest(label), mock.patch.dict(os.environ), self.assertRaises(chain.ChainError):
                if change is None:
                    del os.environ["MMDL_RUN_DIR"]
                else:
                    os.environ["MMDL_RUN_DIR"] = change
                quiet(chain.run, path, sha)
        self.log.write_text("keep")
        with self.subTest("existing chain.jsonl"), self.assertRaises(chain.ChainError):
            quiet(chain.run, path, sha)
        self.assertEqual("keep", self.log.read_text())

    def test_invalid_specs_are_refused(self):
        ok = self.step("one", self.py("pass"))
        specs = {
            "duplicate step names": [ok, self.step("one", self.py("pass"))],
            "unknown step key": [{**ok, "retries": 2}],
            "relative expect file": [self.step("one", self.py("pass"), ["out/result.json"])],
            "empty argv": [self.step("one", [])],
            "zero timeout": [self.step("one", self.py("pass"), timeout=0)],
            "no steps": [],
        }
        for label, steps in specs.items():
            with self.subTest(label), self.assertRaises(chain.ChainError):
                chain.check_spec({"version": chain.SPEC_VERSION, "steps": steps})
        with self.assertRaises(chain.ChainError):
            chain.check_spec({"version": "pod-chain-spec-v0", "steps": [ok]})
        path, sha = self.write_spec(specs["duplicate step names"])
        with self.assertRaises(chain.ChainError):
            quiet(chain.run, path, sha)
        self.assertFalse(self.log.exists())

    def test_matching_meta_and_job_stage_run(self):
        with mock.patch.dict(os.environ, {"MMDL_JOB_STAGE": "main"}):
            code, _ = self.execute([self.step("one", self.py("pass"))],
                                   {"run_dir": str(self.run_dir), "job_stage": "main"})
        self.assertEqual(0, code)

    def test_steps_inherit_environment_and_cwd(self):
        out = self.tmp / "seen.json"
        code = ("import json, os; json.dump([os.environ['MMDL_RUN_DIR'], os.getcwd(), os.environ['FIXTURE_X']], "
                f"open({str(out)!r}, 'w'))")
        with mock.patch.dict(os.environ, {"FIXTURE_X": "x-value"}):
            exit_code, _ = self.execute([self.step("seen", self.py(code), [out])])
        self.assertEqual(0, exit_code)
        self.assertEqual([str(self.run_dir), os.getcwd(), "x-value"], json.loads(out.read_text()))


# ---------------------------------------------------------------- C. helpers

class HelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_pin(self):
        target = self.tmp / "pinned.json"
        target.write_text('{"a": 1}')
        self.assertEqual(0, quiet(chain.main, ["pin", "--file", str(target), "--sha256", sha256_of(target)]))
        self.assertEqual(1, quiet(chain.main, ["pin", "--file", str(target), "--sha256", "0" * 64]))

    def test_with_train_size_takes_min_selected_and_cap_rounded_to_k_and_refuses_below_k(self):
        cmd = ["x", "--max-updates", chain.SIZE_TOKENS[1], "--train-problems", chain.SIZE_TOKENS[0]]
        for selected, items in ((9, 8), (8, 8), (7, 4), (4, 4)):
            with self.subTest(selected=selected):
                self.assertEqual(["x", "--max-updates", str(items // 4), "--train-problems", str(items)],
                                 chain.sized_argv({"selected_ids": ["p"] * selected}, 8, 4, cmd))
        for selection in ({"selected_ids": ["p"] * 3}, {"selected_ids": []}, {}, []):
            with self.subTest(selection=selection), self.assertRaisesRegex(chain.ChainError, "selection"):
                chain.sized_argv(selection, 8, 4, cmd)
        with self.assertRaisesRegex(chain.ChainError, "tokens"):
            chain.sized_argv({"selected_ids": ["p"] * 8}, 8, 4, cmd[:3])
        target, out = self.tmp / "selection.json", self.tmp / "out.json"
        target.write_text(json.dumps({"selected_ids": ["p"] * 5}))
        code = f"import sys, json; json.dump(sys.argv[1:], open({str(out)!r}, 'w'))"
        base = [sys.executable, "-B", "-m", CHAIN_MODULE, "with-train-size", "--selection", str(target),
                "--max-items", "8", "--per-update", "4", "--"]
        env = child_env(self.tmp)
        done = subprocess.run([*base, sys.executable, "-c", code, *chain.SIZE_TOKENS], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(0, done.returncode, done.stderr)
        self.assertEqual(["4", "1"], json.loads(out.read_text()))
        out.unlink()
        target.write_text(json.dumps({"selected_ids": ["p"] * 3}))
        refused = subprocess.run([*base, sys.executable, "-c", code, *chain.SIZE_TOKENS], cwd=REPO, env=env,
                                 capture_output=True, text=True, timeout=120)
        self.assertEqual(2, refused.returncode)
        self.assertIn("fewer than one update of K=4", refused.stderr)
        self.assertFalse(out.exists())

    def test_with_sha256_replaces_the_token_by_the_file_hash_and_execs(self):
        target, out = self.tmp / "prep.json", self.tmp / "out.json"
        target.write_text('{"prep": true}')
        code = f"import sys, json; json.dump(sys.argv[1:], open({str(out)!r}, 'w'))"
        base = [sys.executable, "-B", "-m", CHAIN_MODULE, "with-sha256", "--file", str(target), "--token", "TOK", "--"]
        env = child_env(self.tmp)
        done = subprocess.run([*base, sys.executable, "-c", code, "TOK", "x"], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(0, done.returncode, done.stderr)
        self.assertEqual([sha256_of(target), "x"], json.loads(out.read_text()))
        out.unlink()
        refused = subprocess.run([*base, sys.executable, "-c", code, "x"], cwd=REPO, env=env,
                                 capture_output=True, text=True, timeout=120)
        self.assertEqual(2, refused.returncode)
        self.assertFalse(out.exists())

    def test_receipt_copies_a_throughput_receipt_without_mirrored_from(self):
        source, output = self.tmp / "source.json", self.tmp / "stage" / "receipt.json"
        source.write_text(json.dumps({**RECEIPT, "mirrored_from": "/workspace/runs/r1/out/train/receipt.json"}))
        self.assertEqual(0, quiet(chain.main, ["receipt", "--source", str(source), "--output", str(output),
                                               "--step", "train"]))
        written = json.loads(output.read_text())
        self.assertEqual({**RECEIPT, "stage_receipt_step": "train", "stage_receipt_source": str(source.absolute()),
                          "stage_receipt_source_sha256": sha256_of(source)}, written)
        self.assertNotIn("mirrored_from", written)
        self.assertEqual(written, chain.stage_receipt(source, self.tmp / "again.json", "train"))

    def test_receipt_refuses_another_version(self):
        source, output = self.tmp / "source.json", self.tmp / "receipt.json"
        source.write_text(json.dumps({**RECEIPT, "version": "throughput-receipt-v0"}))
        with self.assertRaises(chain.ChainError):
            chain.stage_receipt(source, output, "train")
        self.assertEqual(2, quiet(chain.main, ["receipt", "--source", str(source), "--output", str(output),
                                               "--step", "train"]))
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
