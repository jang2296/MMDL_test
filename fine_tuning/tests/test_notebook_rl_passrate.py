"""CPU checks for the pass-rate pre-filter; scripted engines, no GPU and no model-quality claim."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.notebook_rl import passrate
from fine_tuning.notebook_rl.dataset import build_splits, generate_pool
from fine_tuning.notebook_rl.engines import Completion, RolloutResult
from fine_tuning.notebook_rl.local_data import POOL_FORMAT_V2, PROBE_COVERAGE_RULE, project_observation_actors
from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash, build_tree, seeds
from fine_tuning.notebook_rl.vault import read_actors
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.tests.test_notebook_rl_repair_rollout import Engine as CollectorEngine
from fine_tuning.tests.test_notebook_rl_engines import Processor
from fine_tuning.tests.test_training import tiny_model

WRONG = "Final answer: 999"


def message_ids(request) -> list[int]:
    """Deterministic 'tokenization' of the exact messages, so equal ids mean equal prompts."""
    if request.prompt_token_ids is not None:
        return list(request.prompt_token_ids)
    return list(hashlib.sha256(json.dumps(request.messages, sort_keys=True).encode()).digest())


class ScriptedEngine:
    """Returns script(request) -> list[str]; prompts are the hash of the real messages."""
    policy_version = 0

    def __init__(self, script, *, blocked=(), fail_on=None):
        self.script, self.blocked, self.fail_on = script, set(blocked), fail_on
        self.requests: list = []

    def prepare(self, request):
        if request.request_id.split(":v0")[0] in self.blocked:
            raise ValueError("CONTEXT_BLOCKED: scripted prompt plus completion reservation")
        return SimpleNamespace(token_ids=message_ids(request), pixels={}, images=[], image_key="x", image_spans=())

    def sample(self, requests, *, on_completion=None):
        request = requests[0]
        if self.fail_on is not None and len(self.requests) == self.fail_on:
            raise RuntimeError("scripted crash")
        self.requests.append(request)
        texts = self.script(request)
        assert len(texts) == request.n
        completions = [Completion([1000 + i, 2], [-.1, -.2], "stop", text) for i, text in enumerate(texts)]
        return [RolloutResult(request.request_id, self.prepare(request).token_ids, completions)]


class PassrateTests(unittest.TestCase):
    temp: tempfile.TemporaryDirectory
    root: Path
    build: Path
    protocol: dict
    ids: list[str]
    gold: dict[str, str]

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.build = cls.root / "build"
        pools = {s: generate_pool("graph", 3 if s == "train" else 1, 3407, s, 1.0) for s in
                 ("train", "dev", "sealed", "style_sealed")}
        build_splits(cls.build, pools)
        cls.protocol = build_protocol(repair_notebook=True, context_limit=24576)
        cls.ids = [a.problem_id for a in read_actors(cls.build / "train")]
        cls.gold = {pid: item.gold for pid, item in Verifier(cls.build / "train").items.items()}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.count = 0

    def out(self) -> Path:
        self.count += 1
        return self.root / f"{self._testMethodName}-{self.count}"

    def right(self, pid: str) -> str:
        return f"Final answer: {self.gold[pid]}"

    def script(self, rights: dict[str, int]):
        """rights[problem id] correct answers out of n, rest wrong."""
        def make(request):
            pid = request.request_id.split(":v0")[0]
            return [self.right(pid)] * rights[pid] + [WRONG] * (request.n - rights[pid])
        return make

    def measure(self, rights, *, ids=None, k=4, namespace="pre", output=None, **kwargs):
        engine = kwargs.pop("engine", None) or ScriptedEngine(self.script(rights))
        return passrate.measure(self.build, "train", ids, protocol=self.protocol, engine=engine, k=k,
                                seed_namespace=namespace, output=output or self.out(), **kwargs)

    def rules(self, **override):
        return {"version": "passrate-rules-v1", "min_pass": 1, "max_pass": 3, "max_blocking": 0} | override

    # ---------------------------------------------------------------- measure

    def test_measure_writes_records_and_summary_and_select_freezes_mixed_items(self):
        a, b, c = self.ids
        seen: list[dict] = []
        out = self.out()
        summary = self.measure({a: 4, b: 0, c: 2}, output=out, on_record=seen.append)
        self.assertEqual([row["problem_id"] for row in seen], self.ids)
        lines = (out / "records.jsonl").read_text().splitlines()
        rows = [json.loads(line) for line in lines]
        self.assertEqual([r["problem_id"] for r in rows], self.ids)
        self.assertEqual([r["passes"] for r in rows], [4, 0, 2])
        row = rows[2]
        self.assertEqual(row["statuses"], {"CORRECT": 2, "INCORRECT": 2})
        self.assertEqual(row["finish_reasons"], {"stop": 4})
        self.assertEqual((row["k"], row["blocking"], len(row["response_sha256"]), len(row["completion_tokens"])),
                         (4, 0, 4, 4))
        self.assertEqual(row["response_sha256"][0], hashlib.sha256(self.right(c).encode()).hexdigest())
        self.assertGreater(row["prompt_tokens"], 0)
        self.assertEqual(oct((out / "records.jsonl").stat().st_mode & 0o777), "0o600")
        manifest = (self.build / "manifest.yaml").read_bytes()
        self.assertEqual(summary, json.loads((out / "summary.json").read_text()))
        self.assertEqual(summary["version"], passrate.VERSION)
        self.assertEqual(summary["weights"], {"checkpoint": None, "trainable_sha256": None})
        self.assertEqual((summary["protocol_sha256"], summary["k"], summary["items"], summary["samples_total"]),
                         (self.protocol["sha256"], 4, 3, 12))
        self.assertEqual(summary["build_manifest_sha256"], hashlib.sha256(manifest).hexdigest())
        self.assertEqual(summary["status_counts"], {"CORRECT": 6, "INCORRECT": 6})
        self.assertEqual(summary["pass_histogram"], {"0": 1, "2": 1, "4": 1})
        self.assertEqual(summary["records_sha256"], hashlib.sha256((out / "records.jsonl").read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            self.measure({a: 4, b: 0, c: 2}, output=out)

        target = self.root / f"sel-{out.name}.json"
        frozen = passrate.select(out, self.rules(), target)
        self.assertEqual(frozen, json.loads(target.read_text()))
        self.assertEqual(frozen["status"], "SELECTION_FROZEN_NOT_TRAINING_APPROVAL")
        self.assertEqual(frozen["selected_ids"], [c])
        self.assertEqual(frozen["excluded"], [{"id": a, "reason": "ALL_CORRECT"}, {"id": b, "reason": "ALL_WRONG"}])
        self.assertEqual((frozen["rules"], frozen["training_allowed"]), (self.rules(), False))
        self.assertEqual(frozen["report_sha256"], hashlib.sha256((out / "summary.json").read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            passrate.select(out, self.rules(), target)

    def test_ids_order_subset_and_seeds_are_deterministic_namespaced_and_disjoint_from_training(self):
        a, b, c = self.ids
        rights = {a: 1, b: 1, c: 1}
        first = self.measure(rights, ids=[c, a], namespace="pre")
        self.assertEqual(first["ids"], [c, a])
        rows = {}
        for name in ("pre-1", "pre-2", "other"):
            out = self.out()
            self.measure(rights, ids=[c, a], namespace=name.split("-")[0], output=out)
            rows[name] = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
        self.assertEqual(rows["pre-1"], [{**r, "seconds": s["seconds"]} for r, s in zip(rows["pre-2"], rows["pre-1"])])
        for row, other in zip(rows["pre-1"], rows["other"]):
            self.assertEqual(len(set(row["seeds"])), 4)
            self.assertFalse(set(row["seeds"]) & set(other["seeds"]))
            self.assertEqual(row["seeds"], seeds(row["problem_id"], 0, 1, "PASSRATE:pre", 4, master_seed=3407))
            for branch in ("T1:root", "T2:x", "T3:x"):
                for turn in (1, 2, 3):
                    self.assertFalse(set(row["seeds"]) & set(seeds(row["problem_id"], 0, turn, branch, 8)))

    def test_prompt_token_ids_equal_the_collector_t1_prompt(self):
        pid = self.ids[0]
        actor = project_observation_actors(read_actors(self.build / "train"), self.protocol)[0]
        self.assertEqual(actor.problem_id, pid)
        verifier = Verifier(self.build / "train")
        cfg = RolloutConfig.from_protocol(self.protocol, "ALG-NB3", master_seed=3407)
        collector = CollectorEngine([[WRONG] * 4])
        collector.prepare = lambda request: SimpleNamespace(token_ids=message_ids(request), pixels={}, images=[],
                                                            image_key="x", image_spans=())
        tree = build_tree(actor, [object()] * len(actor.image_paths), collector, verifier, local_prompts(self.protocol),
                          cfg, 0, self.protocol["route_rule"], on_raw=lambda event: None,
                          should_stop=lambda: len(collector.requests) >= 1)
        call = tree.notebook["repair_collector"]["calls"][0]
        engine = ScriptedEngine(self.script({pid: 2, self.ids[1]: 2, self.ids[2]: 2}))
        out = self.out()
        self.measure({}, ids=[pid], output=out, engine=engine)
        row = json.loads((out / "records.jsonl").read_text())
        request = engine.requests[0]
        self.assertEqual(request.messages, collector.requests[0].messages)
        self.assertEqual(message_ids(request), call["prompt_token_ids"])
        self.assertEqual((row["prompt_sha256"], row["prompt_tokens"]),
                         (_hash(call["prompt_token_ids"]), len(call["prompt_token_ids"])))
        self.assertEqual((request.max_new_tokens, request.n), (call["max_new_tokens"], 4))
        self.assertNotEqual(request.seeds, call["seeds"])

    def test_crash_keeps_completed_items_and_leaves_no_summary(self):
        out = self.out()
        engine = ScriptedEngine(self.script({pid: 1 for pid in self.ids}), fail_on=2)
        with self.assertRaisesRegex(RuntimeError, "scripted crash"):
            self.measure({}, output=out, engine=engine)
        rows = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
        self.assertEqual([r["problem_id"] for r in rows], self.ids[:2])
        self.assertFalse((out / "summary.json").exists())
        with self.assertRaises(ValueError):
            passrate.select(out, self.rules(), self.root / "crash.json")

    def test_blocking_statuses_are_counted_not_converted_to_failures(self):
        a, b, c = self.ids
        real = Verifier(self.build / "train")

        class Blocking:
            def grade(self, pid, text, finish, mode, **options):
                result = real.grade(pid, text, finish, mode, **options)
                return replace(result, status="UNVERIFIABLE") if text == "unverifiable" else result

        def script(request):
            pid = request.request_id.split(":v0")[0]
            return [self.right(pid), WRONG, "unverifiable", "unverifiable"] if pid == a else [self.right(pid)] * 2 + [WRONG] * 2
        out = self.out()
        engine = ScriptedEngine(script, blocked={c})
        self.measure({}, output=out, engine=engine, verifier=Blocking())
        rows = {r["problem_id"]: r for r in map(json.loads, (out / "records.jsonl").read_text().splitlines())}
        self.assertEqual(rows[a]["statuses"], {"CORRECT": 1, "INCORRECT": 1, "UNVERIFIABLE": 2})
        self.assertEqual((rows[a]["passes"], rows[a]["blocking"]), (1, 2))
        self.assertEqual(rows[c]["statuses"], {"CONTEXT_BLOCKED": 4})
        self.assertEqual((rows[c]["passes"], rows[c]["blocking"], rows[c]["prompt_tokens"]), (0, 4, None))
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual((summary["blocking_total"], summary["status_counts"]["UNVERIFIABLE"]), (6, 2))
        strict = passrate.select(out, self.rules(), self.root / f"strict-{out.name}.json")
        self.assertEqual((strict["selected_ids"], {e["id"]: e["reason"] for e in strict["excluded"]}),
                         ([b], {a: "BLOCKING", c: "BLOCKING"}))
        tolerant = passrate.select(out, self.rules(max_blocking=2), self.root / f"tolerant-{out.name}.json")
        self.assertEqual((tolerant["selected_ids"], [e["id"] for e in tolerant["excluded"]]), ([a, b], [c]))

    def test_measure_rejects_bad_arguments_before_sampling(self):
        engine = ScriptedEngine(self.script({pid: 1 for pid in self.ids}))
        for kwargs, error in (({"k": 3}, ValueError), ({"k": 9}, ValueError), ({"k": True}, ValueError),
                              ({"namespace": ""}, ValueError), ({"namespace": "a/b"}, ValueError),
                              ({"ids": []}, ValueError), ({"ids": [self.ids[0]] * 2}, ValueError),
                              ({"ids": ["missing"]}, ValueError), ({"output": self.build / "inside"}, ValueError)):
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                self.measure({}, engine=engine, **kwargs)
        with self.assertRaises(ValueError):
            passrate.measure(self.build, "bogus", None, protocol=self.protocol, engine=engine, k=4,
                             seed_namespace="x", output=self.out())
        self.assertEqual(engine.requests, [])

    # ---------------------------------------------------------------- select / hard negatives

    def report(self, rights, **kwargs) -> Path:
        out = self.out()
        self.measure(rights, output=out, **kwargs)
        return out

    def test_select_validates_rules_and_reports_each_exclusion_reason(self):
        a, b, c = self.ids
        report = self.report({a: 4, b: 0, c: 3})
        for bad in ({"version": "other"}, {"min_pass": True}, {"min_pass": 2, "max_pass": 1}, {"max_pass": 5},
                    {"min_pass": -1}, {"max_blocking": -1}, {"max_blocking": 1.0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                passrate.select(report, self.rules(**bad), self.root / "never.json")
        for rules in ({k: v for k, v in self.rules().items() if k != "max_blocking"}, self.rules() | {"extra": 1}):
            with self.assertRaises(ValueError):
                passrate.select(report, rules, self.root / "never.json")
        self.assertFalse((self.root / "never.json").exists())
        narrow = passrate.select(report, self.rules(max_pass=2), self.root / f"narrow-{report.name}.json")
        self.assertEqual({e["id"]: e["reason"] for e in narrow["excluded"]},
                         {a: "ALL_CORRECT", b: "ALL_WRONG", c: "OUT_OF_RANGE"})
        self.assertEqual(narrow["selected_ids"], [])

    def test_select_refuses_tampered_or_incomplete_reports(self):
        report = self.report({pid: 2 for pid in self.ids})
        records = report / "records.jsonl"
        records.chmod(0o600)
        lines = records.read_text().splitlines()
        records.write_text("\n".join(lines[:2]) + "\n")
        with self.assertRaisesRegex(ValueError, "matching records"):
            passrate.select(report, self.rules(), self.root / "t1.json")
        summary = json.loads((report / "summary.json").read_text())
        summary["records_sha256"] = hashlib.sha256(records.read_bytes()).hexdigest()
        summary["items"] = 3
        (report / "summary.json").write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "malformed"):
            passrate.select(report, self.rules(), self.root / "t2.json")
        (report / "summary.json").write_text(json.dumps(summary | {"version": "x"}))
        with self.assertRaisesRegex(ValueError, "complete measurement"):
            passrate.select(report, self.rules(), self.root / "t3.json")

    def test_weights_are_bound_into_the_summary_and_so_into_the_selection_hash(self):
        a, b, c = self.ids
        rights = {a: 4, b: 0, c: 2}
        ft = {"checkpoint": "/ckpt/step-2", "trainable_sha256": "a" * 64}
        f0_report, ft_report = self.report(rights), self.report(rights, weights=ft)
        self.assertEqual(json.loads((ft_report / "summary.json").read_text())["weights"], ft)
        f0_sel = passrate.select(f0_report, self.rules(), self.root / f"f0-{f0_report.name}.json")
        ft_sel = passrate.select(ft_report, self.rules(), self.root / f"ft-{ft_report.name}.json")
        self.assertEqual((f0_sel["selected_ids"], ft_sel["selected_ids"]), ([c], [c]))
        self.assertNotEqual(f0_sel["report_sha256"], ft_sel["report_sha256"])
        _, summary, _ = passrate.verify_selection(self.root / f"ft-{ft_report.name}.json", ft_report)
        self.assertEqual(summary["weights"], ft)
        # Relabeling the weights after the fact changes the summary hash: the saved selection no longer verifies.
        path = ft_report / "summary.json"
        path.chmod(0o600)
        edited = json.loads(path.read_text())
        path.write_text(json.dumps(edited | {"weights": dict(passrate.F0_WEIGHTS)}))
        with self.assertRaisesRegex(ValueError, "differs from the one its measurement report derives"):
            passrate.verify_selection(self.root / f"ft-{ft_report.name}.json", ft_report)

    def test_measure_refuses_malformed_weights_before_sampling_and_reports_without_weights_are_refused(self):
        engine = ScriptedEngine(lambda request: [WRONG] * request.n)
        for bad in ({"checkpoint": "/c", "trainable_sha256": None}, {"checkpoint": None, "trainable_sha256": "a" * 64},
                    {"checkpoint": "relative/c", "trainable_sha256": "a" * 64},
                    {"checkpoint": "/c", "trainable_sha256": "A" * 64}, {"checkpoint": "/c", "trainable_sha256": "a" * 63},
                    {"checkpoint": "/c"}, {"checkpoint": "/c", "trainable_sha256": "a" * 64, "extra": 1}, "F0", []):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "weights"):
                self.measure({}, engine=engine, weights=bad)
        self.assertEqual(engine.requests, [])
        report = self.report({pid: 2 for pid in self.ids})
        summary = json.loads((report / "summary.json").read_text())
        (report / "summary.json").chmod(0o600)
        (report / "summary.json").write_text(json.dumps({k: v for k, v in summary.items() if k != "weights"}))
        with self.assertRaisesRegex(ValueError, "bind its weights"):
            passrate.select(report, self.rules(), self.root / "no-weights.json")

    def test_trainable_sha256_of_a_loaded_checkpoint_is_the_hash_of_its_recorded_tensor_hashes(self):
        from fine_tuning.notebook_rl.checkpoint import capture_rng, load_eval, make_optimizer, read_metadata, save
        from fine_tuning.tests.test_notebook_rl_checkpoint import action_step, fixture, fresh_tiny

        with tempfile.TemporaryDirectory(prefix="passrate-weights-") as tmp:
            root = Path(tmp)
            model_path, _, protocol, binding = fixture(root)
            f0 = fresh_tiny(model_path, "MOD-RV", protocol)
            f0_sha = passrate._trainable_sha256(f0)
            optimizer, scheduler = make_optimizer(f0, protocol)
            action_step(f0, optimizer, scheduler)
            f0.route.apply(False)
            cursor = {"epoch": 0, "position": 1, "discarded_problem_ids": []}
            save(f0, optimizer, scheduler, 1, cursor, capture_rng(), root / "ckpt", protocol=protocol, binding=binding)
            loaded = load_eval(model_path, root / "ckpt", binding=binding, _test_policy_factory=fresh_tiny)
            state = read_metadata(root / "ckpt", binding=binding)["state_sha256"]
            self.assertEqual(passrate._trainable_sha256(loaded), hashlib.sha256(passrate._bytes(state)).hexdigest())
            self.assertNotEqual(passrate._trainable_sha256(loaded), f0_sha)

    def test_hard_negatives_select_still_failing_trained_items(self):
        a, b, c = self.ids
        after = self.report({a: 4, b: 1, c: 0}, namespace="post")
        frozen = passrate.hard_negatives(after, output=self.root / f"hard-{after.name}.json")
        self.assertEqual(frozen["kind"], "HARD_NEGATIVES")
        self.assertEqual(frozen["status"], "SELECTION_FROZEN_NOT_TRAINING_APPROVAL")
        self.assertEqual(frozen["selected_ids"], [b, c])
        self.assertEqual(frozen["excluded"], [{"id": a, "reason": "ALL_CORRECT"}])
        self.assertEqual(frozen["rules"], {"version": "passrate-rules-v1", "min_pass": 0, "max_pass": 1, "max_blocking": 0})
        self.assertEqual(frozen["report_sha256"], hashlib.sha256((after / "summary.json").read_bytes()).hexdigest())
        wider = passrate.hard_negatives(after, max_pass=3, output=self.root / f"wider-{after.name}.json")
        self.assertEqual(wider["selected_ids"], [b, c])
        only_wrong = passrate.hard_negatives(after, max_pass=0, output=self.root / f"zero-{after.name}.json")
        self.assertEqual(only_wrong["selected_ids"], [c])
        self.assertEqual({e["id"]: e["reason"] for e in only_wrong["excluded"]}, {a: "ALL_CORRECT", b: "OUT_OF_RANGE"})
        with self.assertRaises(ValueError):
            passrate.hard_negatives(after, max_pass=5, output=self.root / "x.json")

    # ---------------------------------------------------------------- real tiny HF engine and CLI

    def test_tiny_transformers_engine_runs_the_real_sampling_path(self):
        from fine_tuning.notebook_rl.engines import HFRolloutEngine

        engine = HFRolloutEngine(tiny_model(), Processor(), {"preprocessing": "hf_processor"}, 24576)
        out = self.out()
        with patch("fine_tuning.notebook_rl.passrate.RolloutConfig.from_protocol",
                   return_value=SimpleNamespace(max_new_tokens={1: 6}, context_limit=24576)):
            self.measure({}, ids=[self.ids[0]], output=out, engine=engine)
        row = json.loads((out / "records.jsonl").read_text())
        self.assertEqual((row["k"], len(row["completion_tokens"]), len(set(row["seeds"]))), (4, 4, 4))
        self.assertEqual(sum(row["statuses"].values()), 4)
        self.assertTrue(set(row["statuses"]) <= {"INVALID_NO_FINAL", "TRUNCATED", "INCORRECT", "CORRECT", "INVALID_MULTIPLE"})

    def test_cli_select_and_hard_negatives_and_argument_errors(self):
        a, b, c = self.ids
        report = self.report({a: 4, b: 1, c: 2})
        with patch("builtins.print"):
            self.assertEqual(passrate.main(["select", "--report", str(report), "--min-pass", "1", "--max-pass", "3",
                                            "--output", str(self.root / f"cli-sel-{report.name}.json")]), 0)
            self.assertEqual(passrate.main(["hard-negatives", "--report", str(report), "--max-pass", "1",
                                            "--output", str(self.root / f"cli-hard-{report.name}.json")]), 0)
            with patch("traceback.print_exc"):
                self.assertEqual(passrate.main(["select", "--report", str(report), "--min-pass", "3", "--max-pass", "1",
                                                "--output", str(self.root / "bad.json")]), 1)
        self.assertEqual(json.loads((self.root / f"cli-sel-{report.name}.json").read_text())["selected_ids"], [b, c])
        self.assertEqual(json.loads((self.root / f"cli-hard-{report.name}.json").read_text())["selected_ids"], [b])
        base = ["measure", "--build-root", str(self.build), "--model-path", "/m", "--k", "4",
                "--seed-namespace", "x", "--output", str(self.root / "cli-run")]
        for extra in (["--standalone-base"], ["--checkpoint", "/c"], ["--answer-verified-route"],
                      ["--finalization-reward"], ["--slotless-source", "s"]):
            with self.subTest(extra=extra), patch("sys.stderr"), self.assertRaises(SystemExit):
                passrate.main([*base, *extra])

    def test_cli_measure_refuses_when_another_process_uses_the_gpu(self):
        completed = SimpleNamespace(stdout="4242, python3\n")
        with patch("fine_tuning.notebook_rl.hf_pool_exclusion.subprocess.run", return_value=completed), \
                patch("fine_tuning.notebook_rl.quantized_policy.build_quantized_policy") as loader, \
                patch("traceback.print_exc"), patch("builtins.print"):
            code = passrate.main(["measure", "--build-root", str(self.build), "--model-path", "/m", "--k", "4",
                                  "--seed-namespace", "x", "--output", str(self.root / "cli-busy")])
        self.assertEqual(code, 1)
        loader.assert_not_called()
        self.assertFalse((self.root / "cli-busy").exists())


class FakeVLLM(ScriptedEngine):
    """Records the order of weight initialization and sampling; gated by the (patched) synchronization."""

    def __init__(self, events: list):
        super().__init__(lambda request: [WRONG] * request.n)
        self.events = events

    def initialize_weights(self, named, version):
        self.events.append(("initialize_weights", version))

    def sample(self, requests, *, on_completion=None):
        self.events.append(("sample", requests[0].request_id.split(":v0")[0]))
        return super().sample(requests, on_completion=on_completion)


class VllmSamplerSynchronizationTests(unittest.TestCase):
    """_cli_measure with mocked model/engines: the vLLM sampler is probe-checked against HF before it samples."""
    temp: tempfile.TemporaryDirectory
    root: Path
    build: Path
    protocol: dict

    @classmethod
    def setUpClass(cls):
        # Import first: a module first imported while policy.merged_named_tensors is patched would keep the mock.
        for name in ("checkpoint", "hf_pool_exclusion", "local_run", "quantized_policy", "scale", "trainer",
                     "verifier_service"):
            import_module(f"fine_tuning.notebook_rl.{name}")
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.build = cls.root / "build"
        build_splits(cls.build, {s: generate_pool("graph", 2 if s == "train" else 1, 3407, s, 1.0)
                                 + generate_pool("graph_pair", 1, 3407, s, 1.0)
                                 for s in ("train", "dev", "sealed", "style_sealed")})
        cls.protocol = build_protocol(repair_notebook=True, context_limit=24576)
        actors = project_observation_actors(read_actors(cls.build / "train"), cls.protocol)
        cls.singles = [a.problem_id for a in actors if len(a.image_paths) == 1]
        cls.pair = next(a.problem_id for a in actors if len(a.image_paths) == 2)
        cls.manifest = json.loads((cls.build / "manifest.yaml").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.events: list = []
        self.sampler = FakeVLLM(self.events)
        self.hf = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text, add_special_tokens=False: [1, 2, 3]))
        self.bundle = object()
        stack = self.enterContext(ExitStack())
        outer = self

        class Service:
            def __init__(self, root, **kwargs):
                self.verifier = Verifier(root)
                self.metadata = {"vault_manifest_sha256": outer.manifest["split_manifests"]["train"]}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def grade(self, *args, **kwargs):
                return self.verifier.grade(*args, **kwargs)

        def patched(name, *new, **kwargs):
            return stack.enter_context(patch(name, *new, **kwargs))

        patched("fine_tuning.notebook_rl.hf_pool_exclusion.gpu_guard")
        patched("fine_tuning.notebook_rl.passrate._install_guards", return_value=[])
        patched("fine_tuning.notebook_rl.verifier_service.VerifierService", Service)
        patched("fine_tuning.notebook_rl.policy.merged_named_tensors", return_value=[])
        self.loader = patched("fine_tuning.notebook_rl.scale.build_bf16_policy", return_value=self.bundle)
        self.load_eval = patched("fine_tuning.notebook_rl.checkpoint.load_eval", return_value=self.bundle)
        self.make_engines = patched("fine_tuning.notebook_rl.scale.make_engines", return_value=(self.sampler, self.hf))
        self.sync = patched("fine_tuning.notebook_rl.scale.synchronize", side_effect=self.synchronize)
        patched("builtins.print")
        patched("traceback.print_exc")
        self.count = 0

    def synchronize(self, bundle, sampler, preparation, probes, protocol, step, *, push):
        self.assertIs(bundle, self.bundle)
        self.assertIs(sampler, self.sampler)
        self.assertIs(preparation, self.hf)
        self.events.append(("synchronize", step, push, [p.request.request_id for p in probes],
                            [len(p.request.images) for p in probes]))
        report = [{"request_id": p.request.request_id, "policy_version": step, "mean_abs_delta": 0., "max_abs_delta": 0.}
                  for p in probes]
        return [], report

    def run_cli(self, *extra: str) -> tuple[int, Path]:
        self.count += 1
        out = self.root / f"{self._testMethodName}-{self.count}"
        argv = ["measure", "--build-root", str(self.build), "--model-path", "/m", "--k", "4", "--seed-namespace", "x",
                "--output", str(out), *extra]
        return passrate.main(argv), out

    def vllm_args(self, *extra: str) -> list[str]:
        return ["--policy-precision", "bf16", "--rollout-engine", "vllm", "--vllm-gpu-memory-utilization", ".55",
                "--vllm-max-num-seqs", "16", "--vllm-kv-cache-memory-bytes", "10737418240", *extra]

    def test_checkpoint_weights_are_initialized_then_probe_synchronized_then_sampled(self):
        binding = self.root / "binding.json"
        binding.write_text(json.dumps({"binding": {}}))
        # Subset selection {paired, second single}: the probes come from the selection, not the whole split.
        with patch("fine_tuning.notebook_rl.passrate._trainable_sha256", return_value="b" * 64) as trainable:
            code, out = self.run_cli(*self.vllm_args("--checkpoint", "/ckpt", "--binding", str(binding),
                                                     "--ids", self.pair, self.singles[1]))
        self.assertEqual(code, 0)
        trainable.assert_called_once_with(self.bundle)
        self.assertEqual(json.loads((out / "summary.json").read_text())["weights"],
                         {"checkpoint": "/ckpt", "trainable_sha256": "b" * 64})
        probes = [f"local-sync:{self.singles[1]}", f"local-sync:{self.pair}"]
        self.assertEqual(self.events, [("initialize_weights", 0), ("synchronize", 0, False, probes, [1, 2]),
                                       ("sample", self.pair), ("sample", self.singles[1])])
        self.load_eval.assert_called_once()
        report = json.loads((out / "sampler-synchronization.json").read_text())
        self.assertEqual([row["request_id"] for row in report], probes)
        self.assertEqual({row["policy_version"] for row in report}, {0})
        self.assertEqual(oct((out / "sampler-synchronization.json").stat().st_mode & 0o777), "0o600")

    def test_f0_sampler_is_probe_synchronized_without_weight_initialization(self):
        code, out = self.run_cli(*self.vllm_args())
        self.assertEqual(code, 0)
        self.assertEqual([event[0] for event in self.events], ["synchronize", "sample", "sample", "sample"])
        self.assertEqual(self.events[0][3:], ([f"local-sync:{self.singles[0]}", f"local-sync:{self.pair}"], [1, 2]))
        self.assertTrue((out / "sampler-synchronization.json").is_file())
        self.load_eval.assert_not_called()
        self.assertEqual(json.loads((out / "summary.json").read_text())["weights"],
                         {"checkpoint": None, "trainable_sha256": None})

    def test_vllm_without_both_probe_kinds_refuses_before_loading_or_sampling(self):
        protocol = build_protocol(repair_notebook=True, context_limit=24576)
        for ids in ([self.singles[0], self.singles[1]], [self.pair]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "coverage"):
                passrate._probe_actors(self.build, "train", ids, protocol)
        self.assertEqual([a.problem_id for a in passrate._probe_actors(self.build, "train", None, protocol)],
                         [self.singles[0], self.pair])
        code, out = self.run_cli(*self.vllm_args("--ids", *self.singles))
        self.assertEqual(code, 1)
        self.loader.assert_not_called()
        self.make_engines.assert_not_called()
        self.sync.assert_not_called()
        self.assertEqual(self.events, [])
        self.assertFalse(out.exists())

    def test_transformers_path_builds_no_probes_and_writes_no_synchronization_file(self):
        engine = ScriptedEngine(lambda request: [WRONG] * request.n)
        self.make_engines.return_value = (engine, engine)
        with patch("fine_tuning.notebook_rl.quantized_policy.build_quantized_policy", return_value=self.bundle) as nf4, \
                patch("fine_tuning.notebook_rl.local_run.make_probes") as make_probes:
            code, out = self.run_cli("--ids", *self.singles)
        self.assertEqual(code, 0)
        nf4.assert_called_once()
        make_probes.assert_not_called()
        self.sync.assert_not_called()
        self.assertEqual(sorted(path.name for path in out.iterdir()),
                         ["records.jsonl", "summary.json", "throughput_receipt.json"])


class PreparationProbeTests(unittest.TestCase):
    """A v2 pool preparation: trained items are single-image; the two-image probe is probe-only."""
    temp: tempfile.TemporaryDirectory
    build: Path
    protocol: dict
    singles: list

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.build = Path(cls.temp.name) / "build"
        build_splits(cls.build, {s: generate_pool("graph", 2 if s == "train" else 1, 3407, s, 1.0)
                                 + generate_pool("graph_pair", 1, 3407, s, 1.0)
                                 for s in ("train", "dev", "sealed", "style_sealed")})
        cls.protocol = build_protocol(repair_notebook=True, context_limit=24576)
        actors = project_observation_actors(read_actors(cls.build / "train"), cls.protocol)
        cls.singles = [a for a in actors if len(a.image_paths) == 1]
        cls.pair = next(a for a in actors if len(a.image_paths) == 2)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_v2_preparation_probes_come_from_probe_set_and_probe_only_actor_is_never_measured(self):
        # v2 binds the probe actors separately: the probe set, not the trained selection, supplies the probes.
        # The patched reader stands in for the descriptor bytes, which this test does not write.
        prepared = {"format": POOL_FORMAT_V2, "probe_actors": {"file_sha256": "bound-by-patched-reader"},
                    "probe_coverage": {"rule": PROBE_COVERAGE_RULE, "image_counts": [1, 2]}}
        selected = [self.singles[0].problem_id]
        with patch("fine_tuning.notebook_rl.local_run.read_actor_descriptor", return_value=[self.singles[1], self.pair]):
            probes = passrate._probe_actors(self.build, "train", selected, self.protocol, prepared)
        self.assertEqual([a.problem_id for a in probes], [self.singles[1].problem_id, self.pair.problem_id])
        out = Path(self.temp.name) / "v2-measure"
        passrate.measure(self.build, "train", selected, protocol=self.protocol,
                         engine=ScriptedEngine(lambda request: [WRONG] * request.n), k=4,
                         seed_namespace="pre", output=out)
        measured = [json.loads(line)["problem_id"] for line in (out / "records.jsonl").read_text().splitlines()]
        self.assertEqual(measured, selected)
        self.assertNotIn(self.pair.problem_id, measured)

    def test_single_image_coverage_probes_and_synchronizes_exactly_one_actor(self):
        # probe-coverage-trained-image-counts-v1 bound to single-image counts: one probe, forwarded counts (1,).
        single = {"rule": PROBE_COVERAGE_RULE, "image_counts": [1]}
        prepared = {"format": POOL_FORMAT_V2, "probe_actors": {"file_sha256": "bound-by-patched-reader"},
                    "probe_coverage": single}
        selected = [self.singles[0].problem_id]
        reader = "fine_tuning.notebook_rl.local_run.read_actor_descriptor"
        with patch(reader, return_value=[self.singles[1]]):
            probes = passrate._probe_actors(self.build, "train", selected, self.protocol, prepared)
        self.assertEqual(probes, [self.singles[1]])
        # Never fewer or more than the bound coverage: a probe file and its coverage must agree exactly.
        for actors, coverage in (([self.singles[1], self.pair], single), ([self.singles[1]], {**single, "image_counts": [1, 2]}),
                                 ([], single), ([self.singles[1]], {**single, "rule": "any-probes"})):
            with self.subTest(actors=len(actors), coverage=coverage), patch(reader, return_value=actors), \
                    self.assertRaises(ValueError):
                passrate._probe_actors(self.build, "train", selected, self.protocol, prepared | {"probe_coverage": coverage})
        events = []
        out = Path(self.temp.name) / "single-probe-sync"
        with patch("fine_tuning.notebook_rl.local_run.make_probes",
                   side_effect=lambda actors, images, engine, prompts, *, image_counts: events.append(
                       ("probes", [a.problem_id for a in actors], image_counts)) or ["probe"]), \
                patch("fine_tuning.notebook_rl.scale.synchronize",
                      side_effect=lambda *args, **kwargs: events.append(("sync", args[3], kwargs)) or ([], [{"row": 1}])):
            passrate._synchronize_sampler(None, "sampler", "hf", probes, self.protocol, self.build, out, prepared)
        self.assertEqual(events, [("probes", [self.singles[1].problem_id], (1,)),
                                  ("sync", ["probe"], {"push": False, "image_counts": (1,)})])
        self.assertEqual(json.loads((out / "sampler-synchronization.json").read_text()), [{"row": 1}])


if __name__ == "__main__":
    unittest.main()
