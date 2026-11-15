"""v16b one-run preregistration boundaries, synthetic IDs only."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fine_tuning.notebook_rl import proof_eval as pe
from fine_tuning.tests.test_notebook_rl_proof_eval import DECLARED, IDS, REST, write_run, variant_cfg
from mmdl.runtime.artifacts import sha256_file


class EvalV16BTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.b2 = self.root / "b2.jsonl"
        self.b2.write_text("".join(json.dumps({"id": sid, "correct": sid in REST}) + "\n" for sid in IDS))
        self.paths = {}
        for kind in ("wrong", "rest", "all"):
            self.paths[kind] = self.root / f"{kind}.json"
            pe.make_subset(kind, self.paths[kind], self.b2)
        self.ref = {"export": {"training_protocol": {"version": "error-note-v16b",
                    "training_contract": {"prep_updates": 12, "rl_blocks": 188, "total_slots": 203}},
                    "campaign": {"stage": "final", "slot": 203, "updates": 201}}}

    def preregister(self):
        path = self.root / "prereg.json"
        with mock.patch.object(pe, "_lora_model_ref", return_value=(self.ref, DECLARED)):
            pe.preregister(path, self.paths["wrong"], self.paths["rest"], self.b2, "v11",
                           ft_single_model_ref=self.root / "model_ref.json", all_subset=self.paths["all"])
        return path

    def test_one_unmerged_full900_single_seed_and_separate_supplement(self):
        path = self.preregister()
        value, _ = pe.load_preregistration(path)
        self.assertEqual(value["format"], pe.FORMAT_V11)
        self.assertNotIn(pe.FORMAT_V11, pe.TWO_STAGE)
        self.assertEqual(value["runs"], [{"arm": "FT-SINGLE", "variant": pe.FINAL_VARIANT_V8,
                                         "subset": "all", "role": "statistical"}])
        self.assertEqual(value["evaluation_subsets"]["all"]["count"], 900)
        self.assertEqual(value["mmmu_val_budget"]["runs"], 1)
        self.assertEqual(value["model_application"]["application"], pe.APPLICATION)
        self.assertEqual(value["scorer"]["parser"], pe.PARSER)
        self.assertTrue(value["supplementary_scoring"]["separate_from_v8"])
        self.assertFalse(value["supplementary_scoring"]["used_for_training"])
        self.assertEqual(value["display_thresholds"], {"minimum_correct": 559, "target_correct": 585})
        self.assertNotIn("screen_gate", value)
        self.assertIn("12 preparation updates", value["arms"]["FT-SINGLE"]["definition"])
        ledger = self.root / "ledger.jsonl"
        pe.ledger_start(path, "FT-SINGLE", pe.FINAL_VARIANT_V8, "only-run", ledger, "all")
        with self.assertRaises(ValueError):
            pe.ledger_start(path, "FT-SINGLE", pe.FINAL_VARIANT_V8, "second", ledger, "all")
        with self.assertRaises(ValueError):
            pe.ledger_start(path, "F0", pe.FINAL_VARIANT_V8, "f0", ledger, "all")

    def test_analyze_full_ft_only_keeps_v8_count(self):
        path = self.preregister()
        ledger = self.root / "ledger.jsonl"
        with mock.patch.object(pe, "gpu_info", return_value=("NVIDIA A100-SXM4-80GB", 81920)):
            run = write_run(self.root, "only-run", variant_cfg(pe.FINAL_VARIANT_V8),
                            DECLARED["model_sha256"], lambda i: i < 585, ids=IDS)
            pe.ledger_start(path, "FT-SINGLE", pe.FINAL_VARIANT_V8, "only-run", ledger, "all")
            result = pe.analyze(path, [run], None, ledger)
        self.assertEqual(result["full_900"]["FT-SINGLE"]["mean_correct"], 585)
        self.assertEqual(result["scorer"], pe.PARSER)
        self.assertNotIn("screen", result)

    def test_actual_tiny_v16b_export_is_unmerged(self):
        import warnings
        import torch
        from fine_tuning.error_note import campaign_v16b, protocol, rl, single
        from fine_tuning.error_note.prep_v16b import seal
        from fine_tuning.notebook_rl import checkpoint
        from fine_tuning.tests.test_notebook_rl_checkpoint import fixture, fresh_tiny
        from mmdl.runtime.artifacts import digest

        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
        base, manifest, _, _ = fixture(self.root)
        pins = {name: "a" * 64 for name in ("source", "dataset", "order", "prep_rows", "monitor")}
        proto = protocol.plain_protocol_v16b("bf16", rl_blocks=188, prep_updates=1, contracts=pins)
        ids = [f"synthetic-{i}" for i in range(3000)]
        body = {"version": campaign_v16b.CAMPAIGN_V16B, "protocol": proto,
                "blocks": [ids[i:i + 16] for i in range(0, 3000, 16)],
                "items": [{"problem_id": sid} for sid in ids], "prep_batches": [[]],
                "total_slots": 192, "time_cut": None, "local_sequential": False,
                "capacity": {"engine": "vllm", "max_num_seqs": 32},
                "source": {"actual_files": {"synthetic.py": "b" * 64}}}
        plan = body | {"sha256": digest(body)}
        campaign_v16b._plan_check(plan)
        out = self.root / "campaign"
        out.mkdir()
        seal(out / "campaign-plan.json", body)
        bundle = fresh_tiny(base, protocol.ARM, proto)
        optimizer, scheduler = checkpoint.make_optimizer(bundle, proto)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(192):
                scheduler.step()
        cursor = {"slot": 192, "updates": 0, "phase": "final", "consumed_problem_ids": ids,
                  "policy_sha256": rl.policy_sha256(bundle)}
        saved = out / "checkpoint-192"
        single.save_single_continuation(bundle, optimizer, scheduler, cursor=cursor, path=saved,
            protocol=proto, binding=campaign_v16b._binding(plan), allow_save=True)
        seal(out / "campaign-state.json", {"status": "COMPLETE", "plan_sha256": plan["sha256"],
             "checkpoint": campaign_v16b.descriptor(saved / "metadata.json"), "cursor": cursor, "pending": None})
        ref = pe.export_for_eval(saved, out / "campaign-plan.json", base, self.root / "export", "f" * 40,
            _test_policy_factory=fresh_tiny, _test_base_manifest=manifest)
        self.assertEqual(ref["kind"], "unmerged-lora")
        self.assertEqual(ref["llm_lora"], pe.lora_section())
        self.assertEqual(ref["export"]["campaign"]["stage"], "final")
        pe._single_model_ref(self.root / "export/model_ref.json", lora=True)

    def test_resigned_screen_or_extra_run_is_rejected(self):
        path = self.preregister()
        original = json.loads(path.read_text())
        for change in ({"screen_gate": {}}, {"runs": original["runs"] * 2}):
            path.write_text(json.dumps(original | change))
            path.with_name(path.name + ".sha256").write_text(sha256_file(path))
            with self.assertRaises(ValueError):
                pe.load_preregistration(path)

    def test_v11_rejects_legacy_and_merged_versions(self):
        ref = copy.deepcopy(self.ref)
        ref["export"]["training_protocol"]["version"] = "error-note-single-response-v15"
        with mock.patch.object(pe, "_lora_model_ref", return_value=(ref, DECLARED)), self.assertRaises(ValueError):
            pe.build_preregistration_v11(self.paths["wrong"], self.paths["rest"], self.paths["all"],
                                        self.b2, self.root / "model_ref.json")
        self.assertEqual(pe._single_stage(1, 16, 203, v16b=True), "stopped")
        self.assertEqual(pe._single_stage(203, 3000, 203, v16b=True), "final")
        with self.assertRaises(ValueError):
            pe._single_stage(0, 0, 203, v16b=True)


if __name__ == "__main__":
    unittest.main()
