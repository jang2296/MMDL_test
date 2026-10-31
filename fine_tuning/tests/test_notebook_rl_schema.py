"""CPU wire-contract checks only; all examples are authored synthetic fixtures."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from fine_tuning.notebook_rl.schema import (
    ActorItem, DiagTarget, Group, Lineage, OracleProbe, ProblemTree, Sample,
    SlotReport, VerifierItem, VerifierResult, from_json, load_protocol, to_json,
    validate_actor,
)

ROOT = Path(__file__).resolve().parents[1] / "notebook_rl"


def records():
    actor = ActorItem("fake:family:train:1", "fake", "table", "obs", "표를 읽으세요 {x}",
                      (), "int", ("images/a.png",), ("a" * 64,))
    verifier = VerifierItem(actor.problem_id, "int", "7", {}, (), {"cell[a][b]": "7"},
                            {"cell[a][b]": "number"}, ("cell[a][b]",), ("cell a,b = 7",), ("7",))
    lineage = Lineage(actor.problem_id, "fake", "fixture-v1", "INTERNAL_FIXTURE", "PROGRAM",
                      "rendered:fixture", "procedural", "fixture-v1", 3407, "read-cell",
                      "c" * 16, "train", ())
    slots = SlotReport(1, 1, 0, ("cell[a][b]",), 0, 1, ())
    result = VerifierResult("INCORRECT", "8", 0.0, slots, "strict-final-v1")
    sample = Sample("g#0", actor.problem_id, "g", "T1", 1, None, [1, 2], [8, 2],
                    [-0.1, -0.2], "stop", "Final answer: 8", 3407, 0, result)
    other = replace(sample, sample_id="g#1", seed=3408)
    group = Group("g", "T1", actor.problem_id, "b" * 64, [sample, other], True, None)
    tree = ProblemTree(actor.problem_id, 0, [group], "VISION_ON",
                       {"g#0": "VISUAL_MISREAD"}, {"verified": True, "expected": "7"})
    target = DiagTarget("VISUAL", ("cell[a][b]",))
    probe = OracleProbe("p0", actor.problem_id, "oracle", [1], [2], "CORRECT", 0)
    return actor, verifier, lineage, slots, result, sample, group, tree, target, probe


class SchemaTests(unittest.TestCase):
    def test_all_records_round_trip(self):
        for record in records():
            with self.subTest(record=type(record).__name__):
                wire = to_json(record)
                self.assertEqual(from_json(type(record), wire), record)
                self.assertEqual(to_json(from_json(type(record), wire)), wire)
        self.assertIn("표를", to_json(records()[0]))

    def test_actor_forbidden_and_unknown_keys(self):
        for key in ("gold", "solution", "answer", "slots", "target_text", "required_slots", "other"):
            data = asdict(records()[0]) | {key: "leak"}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unknown keys"):
                from_json(ActorItem, json.dumps(data))

    def test_scalar_types_nested_keys_and_nan(self):
        sample = asdict(records()[5])
        for key, value in (("seed", True), ("turn", "1"), ("policy_version", 1.0),
                           ("sampler_logprobs", [True, False]), ("reward", float("nan")),
                           ("completion_token_ids", [True, 2]), ("kind", "oracle")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                from_json(Sample, json.dumps(sample | {key: value}))
        sample["result"]["gold"] = "secret"
        with self.assertRaises(ValueError):
            from_json(Sample, json.dumps(sample))

    def test_duplicates_missing_unknown_record(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            from_json(ActorItem, '{"question":"a","question":"b"}')
        with self.assertRaises(ValueError):
            from_json(ActorItem, "{}")
        with self.assertRaises(ValueError):
            from_json(dict, "{}")
        with self.assertRaises(ValueError):
            to_json({"question": "not a record"})

    def test_actor_paths_image_hash_and_choices(self):
        actor = records()[0]
        for path in ("../a.png", "/a.png", "a/../b.png", "a//b.png", "C:/a.png", "a\\b.png", ""):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "relative image"):
                validate_actor(replace(actor, image_paths=(path,)))
        for update in ({"question": " "}, {"image_sha256": ()}, {"image_sha256": ("bad",)},
                       {"choices": ("one",)}, {"answer_format": "mcq", "choices": ("one",)},
                       {"answer_format": "mcq", "choices": ("one", "one")}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                to_json(replace(actor, **update))
        to_json(replace(actor, answer_format="mcq", choices=("one", "two")))

    def test_group_state_policy_and_identity(self):
        group = records()[6]
        for change in ({"prompt_token_ids": [3]}, {"policy_version": 1},
                       {"parent_sample_id": "other"}, {"group_id": "other"}):
            mixed = replace(group, samples=[group.samples[0], replace(group.samples[1], **change)])
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "group:"):
                to_json(mixed)
        with self.assertRaises(ValueError):
            to_json(replace(group, samples=[group.samples[0]]))
        with self.assertRaises(ValueError):
            to_json(replace(group, samples=[group.samples[0]] * 2))

    def test_oracle_probe_cannot_be_a_sample(self):
        with self.assertRaises(ValueError):
            from_json(Sample, to_json(records()[-1]))
        group = replace(records()[6], samples=[records()[-1]] * 2)
        with self.assertRaises(ValueError):
            to_json(group)

    def test_tree_sample_ids_are_unique_across_groups(self):
        tree = records()[7]
        first = tree.groups[0]
        second = replace(first, group_id="g2", samples=[
            replace(s, group_id="g2") for s in first.samples])
        with self.assertRaisesRegex(ValueError, "duplicate sample ID across groups"):
            to_json(replace(tree, groups=[first, second]))
        second.samples = [replace(s, sample_id=f"g2#{i}") for i, s in enumerate(second.samples)]
        to_json(replace(tree, groups=[first, second]))

    def test_float_fields_have_canonical_wire_roundtrip(self):
        sample = replace(records()[5], reward=0, diag_reward=1, sampler_logprobs=[-1, -2])
        wire = to_json(sample)
        self.assertEqual(to_json(from_json(Sample, wire)), wire)
        self.assertEqual(json.loads(wire)["reward"], 0.0)

    def test_slots_rewards_and_invalid_logprobs(self):
        with self.assertRaisesRegex(ValueError, "counts"):
            to_json(replace(records()[3], required_correct=2))
        with self.assertRaisesRegex(ValueError, "reward mismatch"):
            to_json(replace(records()[4], reward_answer=1.0))
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            to_json(replace(records()[5], sampler_logprobs=[]))
        with self.assertRaises(ValueError):
            to_json(replace(records()[8], cause3="NON_VISUAL"))

    def test_slot_report_legacy_wire_allows_only_the_two_new_fields_to_be_absent(self):
        report = records()[3]
        legacy = asdict(report)
        del legacy["false_relations"], legacy["unverified_relations"]
        self.assertEqual(from_json(SlotReport, json.dumps(legacy)), report)
        nested = asdict(records()[4]) | {"slots": legacy}
        self.assertEqual(from_json(VerifierResult, json.dumps(nested)), records()[4])
        for missing in ("false_relations", "unverified_relations", "conflicting", "parsed"):
            wire = asdict(report)
            del wire[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                from_json(SlotReport, json.dumps(wire))
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            from_json(SlotReport, json.dumps(legacy | {"extra": []}))

    def test_relation_fields_are_unique_normalized_disjoint_and_separate_from_counts(self):
        report = replace(records()[3], false_relations=("edge[h,d]",),
                         unverified_relations=("edge[z,d]",), conflicting=("edge[h,d]",))
        self.assertEqual(from_json(SlotReport, to_json(report)), report)
        for change in ({"false_relations": ("edge[h,d]",) * 2},
                       {"unverified_relations": ("edge[z,d]",) * 2},
                       {"false_relations": ("edge[H,D]",)},
                       {"unverified_relations": (" edge[z,d] ",)},
                       {"false_relations": ("",)},
                       {"unverified_relations": report.false_relations},
                       {"false_relations": report.wrong_required},
                       {"unverified_relations": report.wrong_required},
                       {"parsed": 2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                to_json(replace(report, **change))

    def test_protocol_and_prompt_hashes(self):
        path = ROOT / "protocol_v1.yaml"
        p = load_protocol(path)
        self.assertEqual(p["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(p["prompts_sha256"], hashlib.sha256((ROOT / "prompts_v1.yaml").read_bytes()).hexdigest())
        self.assertEqual(p["rollout"]["n"], {1: 8, 2: 4, 3: 4})
        self.assertEqual(p["loss"]["kl"]["beta"], 0.01)
        self.assertEqual(p["context_limit"], 16384)
        self.assertEqual(p["expected_trainable"], {"L": 33030144, "M": 109105152, "V": 2359296})
        self.assertEqual(p["loss"]["updates_per_batch"], 1)
        self.assertEqual(p["data_selection"], {
            "min_open_answer_fraction": 0.60, "mcq_choices": 10,
            "phase_a": {"n": 8, "max_pass_rate": 0.75, "max_zero_pass_fraction": 0.50},
            "blind": {"n": 4, "accuracy_floor": 0.25, "threshold": "max_chance_and_floor"}})
        self.assertEqual(load_protocol(path), p)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.yaml"
            p["forbidden"]["additional_sft"] = False
            bad.write_text(yaml.safe_dump(p))
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_protocol(bad)

    def test_prompts_are_complete_and_sources_not_approved(self):
        p = yaml.safe_load((ROOT / "prompts_v1.yaml").read_text())
        self.assertEqual(set(p), {"turn1_plain", "turn1_obs", "retry_incorrect", "retry_invalid",
                                  "diagnose", "oracle_facts", "self_perception_probe", "observation_instruction"})
        for key, value in p.items():
            for text in (value.values() if isinstance(value, dict) else [value]):
                with self.subTest(key=key):
                    self.assertTrue(text.endswith("\n"))
                    self.assertFalse(text.endswith("\n\n"))
        registry = yaml.safe_load((ROOT / "source_registry_v1.yaml").read_text())
        self.assertTrue(all(x["training_allowed"] is False for x in registry["sources"].values()))

    def test_schema_import_does_not_load_torch_or_training(self):
        code = ("import sys; import fine_tuning.notebook_rl.schema; "
                "assert 'torch' not in sys.modules; assert 'fine_tuning.train' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
