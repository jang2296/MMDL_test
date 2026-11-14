"""SINGLE v15 version-plumbing CPU checks (wave 1): the protocol delta over v14, the v11-v14 hash pins, the runtime
protocol check, the CLI version and the train-block contract. Tiny random models only; not pretrained Qwen evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.error_note import single
from fine_tuning.error_note import train_v4 as train
from fine_tuning.error_note.audit import text_sha256
from fine_tuning.error_note.contract import ROUTE_FULL
from fine_tuning.error_note.protocol import (ROLLOUT_ADAPTER_V15, SINGLE_REWARD_GATE_V9, SINGLE_RESOLVE_STAGES_V15,
                                             SINGLE_RESOLVE_V15, SINGLE_ROUTES_V15, VERSION_SINGLE,
                                             VERSION_SINGLE_V12, VERSION_SINGLE_V13, VERSION_SINGLE_V14,
                                             VERSION_SINGLE_V15, _digest, plain_protocol_single)
from fine_tuning.tests import test_error_note_single_v12 as v12
from fine_tuning.tests.test_error_note_single_loss import block
from fine_tuning.tests.test_error_note_single_v14 import as_v14
from fine_tuning.tests.test_error_note_train import tiny_factory
from fine_tuning.tests.test_error_note_v3_audit import PROVENANCE

PINS = v12.PINS
# Captured before v15 was added, built from PINS with 2 slots (v14: schedule_steps=2).
HASHES = {VERSION_SINGLE: "0efc1079ba0ccd077b70293c0b2bb860ecd8cca8a39be971eb4170b25ee9ffbd",
          VERSION_SINGLE_V12: "54ad4af98ec168fe85bcc669340a23c8ecc7ddc7e912d9423d4dc80e2923876c",
          VERSION_SINGLE_V13: "5238019cd23b510bb3e9b1e88027e2a054756c0b5add3fd21102d812096e2b8d",
          VERSION_SINGLE_V14: "1d5eb67c84b93b5463e89add215c57d45266d6bc2c483be8bfb5edb600529152"}
V15_DELTA = ("version", "training_contract.version", "training_contract.hashes.source",
             "training_contract.generation_counts", "training_contract.routes", "training_contract.resolve_attempts",
             "rollout_adapter")


def protocol(version, steps=2):
    extra = {"schedule_steps": steps} if version in (VERSION_SINGLE_V14, VERSION_SINGLE_V15) else {}
    return plain_protocol_single("bf16", 2, contracts=PINS, version=version, **extra)


def strip(value, paths):
    """The protocol without its sha256 and without each dotted path that is present."""
    value = copy.deepcopy(value)
    del value["sha256"]
    for path in paths:
        *parents, leaf = path.split(".")
        node = value
        for key in parents:
            node = node.get(key, {})
        node.pop(leaf, None)
    return value


def reseal(value):
    body = {k: v for k, v in value.items() if k != "sha256"}
    return body | {"sha256": _digest(body)}


class ProtocolV15(unittest.TestCase):
    def test_v11_to_v14_protocol_hashes_are_unchanged(self):
        for version, expected in HASHES.items():
            with self.subTest(version=version):
                self.assertEqual(protocol(version)["sha256"], expected)

    def test_v15_protocol_delta_over_v14_is_exact(self):
        v14, v15 = protocol(VERSION_SINGLE_V14), protocol(VERSION_SINGLE_V15)
        self.assertNotEqual(v14["sha256"], v15["sha256"])
        self.assertEqual(strip(v14, V15_DELTA), strip(v15, V15_DELTA))
        contract = v15["training_contract"]
        self.assertEqual((v15["version"], contract["version"]), (VERSION_SINGLE_V15, VERSION_SINGLE_V15))
        self.assertEqual(contract["generation_counts"], {"t1": 1, "note": 1, "resolve": 3})
        self.assertEqual(contract["routes"], SINGLE_ROUTES_V15)
        self.assertEqual(contract["resolve_attempts"], SINGLE_RESOLVE_V15)
        self.assertEqual(SINGLE_RESOLVE_V15["stages"], list(SINGLE_RESOLVE_STAGES_V15))
        self.assertEqual(v15["rollout_adapter"], ROLLOUT_ADAPTER_V15)
        self.assertEqual(contract["schedule_steps"], 2)
        self.assertEqual(v15["optimizer"]["total_steps"], 2)
        for key in ("resolve_attempts",):
            self.assertNotIn(key, v14["training_contract"])
        self.assertNotIn("rollout_adapter", v14)
        self.assertEqual(v15["training_contract"]["rl"]["reward"], SINGLE_REWARD_GATE_V9)

    def test_v15_needs_schedule_steps_and_an_unknown_version_still_raises(self):
        with self.assertRaisesRegex(ValueError, "schedule_steps belongs to v14/v15"):
            plain_protocol_single("bf16", 2, contracts=PINS, version=VERSION_SINGLE_V15)
        with self.assertRaisesRegex(ValueError, "schedule_steps belongs to v14/v15"):
            plain_protocol_single("bf16", 2, contracts=PINS, version=VERSION_SINGLE_V13, schedule_steps=2)
        with self.assertRaisesRegex(ValueError, "version must be v11, v12, v13, v14 or v15"):
            plain_protocol_single("bf16", 2, contracts=PINS, version="error-note-single-response-v16")


class ProtocolCheck(unittest.TestCase):
    def test_check_accepts_v15_and_its_gate(self):
        single._protocol_check(protocol(VERSION_SINGLE_V15))
        self.assertEqual(single.PROTOCOL_GATES[VERSION_SINGLE_V15], SINGLE_REWARD_GATE_V9)

    def test_check_refuses_a_v15_protocol_with_a_changed_adapter_route_or_attempts(self):
        base = protocol(VERSION_SINGLE_V15)
        edits = {"adapter": lambda p: p.update(rollout_adapter="merged"),
                 "missing adapter": lambda p: p.pop("rollout_adapter"),
                 "routes": lambda p: p["training_contract"].update(routes={"RL": "LM", "A1": "LM", "A2_B": "x"}),
                 "missing routes": lambda p: p["training_contract"].pop("routes"),
                 "attempts": lambda p: p["training_contract"]["resolve_attempts"].update(max_attempts=2),
                 "missing attempts": lambda p: p["training_contract"].pop("resolve_attempts"),
                 "one resolve": lambda p: p["training_contract"]["generation_counts"].update(resolve=1)}
        for name, edit in edits.items():
            changed = copy.deepcopy(base)
            edit(changed)
            with self.subTest(name), self.assertRaisesRegex(ValueError, "invalid single-response protocol"):
                single._protocol_check(reseal(changed))

    def test_check_refuses_an_older_protocol_that_carries_the_v15_adapter_or_three_resolves(self):
        for version in (VERSION_SINGLE, VERSION_SINGLE_V12, VERSION_SINGLE_V13, VERSION_SINGLE_V14):
            single._protocol_check(protocol(version))
            for edit in (lambda p: p.update(rollout_adapter=ROLLOUT_ADAPTER_V15),
                         lambda p: p["training_contract"]["generation_counts"].update(resolve=3)):
                changed = protocol(version)
                edit(changed)
                with self.subTest(version=version), self.assertRaisesRegex(ValueError, "invalid single-response protocol"):
                    single._protocol_check(reseal(changed))


class CliV15(unittest.TestCase):
    REQUIRED = ["prepare", "--order-sha256", "a" * 64] + [
        item for name in ("build-root", "order", "frozen-approval", "source-provenance", "training-rules",
                          "model-path", "output") for item in ("--" + name, "/x")]

    def test_cli_resolves_v15_and_keeps_v12_as_the_default(self):
        parser = single.build_parser()
        self.assertEqual(parser.parse_args(self.REQUIRED).single_version, "v12")
        self.assertEqual(parser.parse_args(self.REQUIRED + ["--single-version", "v15"]).single_version, "v15")
        self.assertEqual(single.SINGLE_CLI_VERSIONS["v15"], VERSION_SINGLE_V15)
        self.assertEqual(single.SINGLE_CLI_VERSIONS["v14"], VERSION_SINGLE_V14)

    def test_prepare_seals_a_v15_protocol_with_schedule_steps_and_v14_seals_none_of_the_v15_keys(self):
        from fine_tuning.error_note.answer_contract import VERSION as RULES_VERSION
        from fine_tuning.error_note.source_evidence import VERSION as SOURCE_VERSION
        from fine_tuning.notebook_rl.schema import ActorItem, to_json

        out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        actors = [ActorItem(f"p{i}", "authored-test", "fixture", "plain", f"Question {i}", ("red", "blue"), "mcq",
                            ("fixture.png",), ("a" * 64,)) for i in range(20)]
        items = [{"problem_id": a.problem_id, "subject": "Biology", "source": a.source} for a in actors]
        root = out / "vault"
        (root / "train").mkdir(parents=True)
        (root / "train" / "manifest.yaml").write_text("authored fixture train")
        (root / "manifest.yaml").write_text("authored fixture union")
        train_sha = single.file_sha(root / "train" / "manifest.yaml")
        sources = {a.problem_id: PROVENANCE | {"actor_sha256": text_sha256(to_json(a)), "image_sha256": list(a.image_sha256),
                   "vault_manifest_sha256": train_sha} for a in actors}
        (out / "source.json").write_text(json.dumps({"version": SOURCE_VERSION, "items": sources}))
        (out / "rules.json").write_text(json.dumps({"version": RULES_VERSION, "vault_manifest_sha256": train_sha, "items": {}}))
        (out / "model").mkdir()
        (out / "model" / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "torch_dtype": "bfloat16"}))

        def args(name):
            return SimpleNamespace(build_root=root, source_provenance=out / "source.json", training_rules=out / "rules.json",
                                   order_sha256="a" * 64, model_path=out / "model", output=out / name, mode="local-smoke",
                                   token_scale=1024., max_num_seqs=1, max_num_batched_tokens=24576, token_budget=0,
                                   stop_after_seconds=None, limit=10, single_version=name)

        with patch.object(single, "_validate_data", return_value=({"items": items}, {"authored_fixture": True})), \
                patch("fine_tuning.notebook_rl.vault.read_actors", return_value=actors), \
                patch.object(single, "source_manifest", return_value={"base_commit": "a" * 40, "actual_files": {}}), \
                patch.object(single, "generation_identity", return_value={"model_files_sha256": "b" * 64}):
            new, old = single.prepare(args("v15")), single.prepare(args("v14"))
        for plan, version in ((new, VERSION_SINGLE_V15), (old, VERSION_SINGLE_V14)):
            with self.subTest(version=version):
                self.assertEqual((plan["protocol"]["version"], len(plan["blocks"])), (version, 2))
                self.assertEqual(plan["protocol"]["training_contract"]["schedule_steps"], 2)
                self.assertEqual(plan["protocol"]["optimizer"]["total_steps"], 2)
                single._protocol_check(plan["protocol"])
        self.assertEqual(new["protocol"]["rollout_adapter"], ROLLOUT_ADAPTER_V15)
        self.assertNotIn("rollout_adapter", old["protocol"])


class TrainBlockV15(unittest.TestCase):
    """`_validate_single_block` / `backward_single_block` take a v15 block like a v14 one, except that the row of an
    image item is FULL (wave 2; the route rule itself is covered by test_error_note_single_v15_route)."""

    def setUp(self):
        self.bundle = tiny_factory()
        self.rows, base = block(self.bundle, count=1, reward=0)
        self.contract = as_v14(self.rows, base)
        v15 = protocol(VERSION_SINGLE_V15)["training_contract"]
        self.contract.update({key: v15[key] for key in ("version", "routes", "resolve_attempts", "generation_counts")})
        for row in self.rows:
            row["route"] = ROUTE_FULL  # the fixture rows carry image hashes

    def test_a_valid_v15_block_runs(self):
        stats = train.backward_single_block(self.bundle, self.rows, self.contract, components=("PG", "KL"))
        self.assertTrue(stats["replay_verified"])

    def test_a_v15_contract_with_changed_routes_attempts_or_counts_is_refused(self):
        edits = (("routes", {"RL": "LM", "A1": "LM", "A2_B": "LM"}),
                 ("resolve_attempts", {**SINGLE_RESOLVE_V15, "max_attempts": 1}),
                 ("generation_counts", {"t1": 1, "note": 1, "resolve": 1}))
        for key, value in edits:
            with self.subTest(key), self.assertRaisesRegex(ValueError, "single train: v15 routes"):
                train.backward_single_block(self.bundle, self.rows, self.contract | {key: value}, components=("PG",))
        with self.assertRaisesRegex(ValueError, "single train: v15 routes"):
            missing = dict(self.contract)
            del missing["resolve_attempts"]
            train.backward_single_block(self.bundle, self.rows, missing, components=("PG",))

    def test_v15_keeps_the_v14_rules(self):
        bad = copy.deepcopy(self.contract)
        del bad["rl"]["tail_praise"]
        with self.assertRaisesRegex(ValueError, "single train: v14"):
            train.backward_single_block(self.bundle, self.rows, bad, components=("PG",))


if __name__ == "__main__":
    unittest.main()
