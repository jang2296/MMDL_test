"""Public CLI plans, portable ZIP acquisition, and sealed protocol isolation."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from fine_tuning.error_note import plan, protocol, v18_data_range as ranges
from fine_tuning.notebook_rl.pod.chain import check_spec
from fine_tuning.tests.test_error_note_v17_split_plan import split_params


class V18PlanDataTests(unittest.TestCase):
    def test_training_plan_rebuilds_source_before_reading_data(self):
        p = split_params(Path("/private/train"), "v17-train")
        p["version"] = plan.PARAMS_VERSION_V18_SPLIT
        del p["bundle_archive_path"]
        for key in plan.V17_TRAIN_HASHED:
            del p[key + "_path"], p[key + "_sha256"]
        p.update(cache_dir="/private/cache")
        result = check_spec(plan.plan("v18-train", p))
        steps = {s["name"]: s["argv"] for s in result["steps"]}
        self.assertEqual(result["meta"]["protocol_version"], protocol.VERSION_V18)
        self.assertEqual(result["meta"]["protocol_contract"]["adapter_strength"], 1)
        self.assertLess(list(steps).index("prepare-data"), list(steps).index("guard-falsealarm-gate"))
        self.assertIn(p["model_path"], steps["prepare-data"])
        self.assertIn(p["data_root"], steps["guard-falsealarm-gate"])
        self.assertNotIn("data_bundle", json.dumps(steps))
        self.assertNotIn("evaluation-decision", json.dumps(steps))

    def test_nine_slot_eval_uses_cross_version_seal(self):
        p = split_params(Path("/private/eval"), "v17-eval")
        p["version"] = plan.PARAMS_VERSION_V18_SPLIT
        with self.assertRaisesRegex(ValueError, "evaluation decision"):
            plan.plan("v18-eval", p)
        p["evaluation_authorization"] = "fixture decision"
        result = check_spec(plan.plan("v18-eval", p))
        self.assertEqual(result["meta"]["protocol_contract"]["eval_concurrency"], 9)
        self.assertIn("--expected-protocol", result["steps"][-3]["argv"])
        with self.assertRaises(ValueError):
            plan.plan("v18-eval", p | {"pf5_vision_exception": "old waiver"})

    def test_named_resume_inherits_seal_900_ids_and_only_executes_remaining_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine-validation.json"
            path.write_text("{}")
            p = split_params(Path(tmp), "v17-eval")
            p.update(version=plan.PARAMS_VERSION_V18_SPLIT, evaluation_authorization="fixture consent")
            checked = {"slots": 9, "evaluation_watchdog_seconds": 200, "estimated_evaluation_seconds": 100,
                       "selected_mode": "eager", "training_artifact_path": p["training_artifacts_path"],
                       "evaluation_authorization": {"evaluation_decision": {"authorization": p["evaluation_authorization"]}},
                       "artifact_verification": {"path": p["out_root"] + "/training-artifacts-verified.json"}}
            with patch("fine_tuning.error_note.v17_eval_engine.load_engine_validation", return_value=checked):
                original = plan._evaluation_continuation(p, path)
                resumed = plan._resume_steps(original)
                with self.assertRaisesRegex(ValueError, "another evaluation"):
                    plan._evaluation_continuation(p | {"evaluation_authorization": "unrelated job"}, path)
            self.assertEqual([s["name"] for s in resumed["steps"]], ["evaluate-900", "v8-analyze", "compass-separate"])
            self.assertEqual(resumed["steps"][0]["argv"].count("--sample-id"), 900)
            expected = list(next(s for s in original["steps"] if s["name"] == "evaluate-900")["argv"])
            expected[expected.index("run")] = "resume"
            self.assertEqual(resumed["steps"][0]["argv"], expected)

    def test_different_update_receipts_cannot_be_exchanged(self):
        kwargs = dict(rl_blocks=4, prep_updates=1, contracts={k: "a"*64 for k in protocol.V17_CONTRACT_HASHES},
                      local_smoke=True)
        old = protocol.plain_protocol_v17("nf4", **kwargs)
        new = protocol.plain_protocol_v18("nf4", **kwargs)
        protocol.validate_protocol_v18(new)
        self.assertNotEqual(old["sha256"], new["sha256"])
        with self.assertRaises(ValueError):
            protocol.validate_protocol_v18(old)
        with self.assertRaises(ValueError):
            protocol.validate_protocol_v17(new)
        altered = copy.deepcopy(new)
        altered["training_contract"]["sft_weights"]["A1"] = .18
        with self.assertRaises(ValueError):
            protocol.validate_protocol_v18(altered)

    def test_member_directory_ranges_are_checked_and_cached(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("selected.csv", b"23,45")
            archive.writestr("other.txt", b"unused")
        blob = buf.getvalue()
        meter_events = []
        class Response(io.BytesIO):
            pass

        def serve(req, meter):
            meter["requests"] += 1
            meter_events.append(req.get_method())
            r = Response()
            if req.get_method() == "HEAD":
                r.headers = {"Content-Length": str(len(blob)), "ETag": '"fixed"'}
            else:
                i, j = map(int, req.headers["Range"][6:].split("-"))
                r = Response(blob[i:j+1])
                r.status = 206
                r.headers = {"ETag": '"fixed"', "Content-Range": f"bytes {i}-{j}/{len(blob)}"}
            return r

        spec = {"member":"selected.csv", "url":"https://example.invalid/fixed.zip",
                "sha256":hashlib.sha256(b"23,45").hexdigest(), "bytes":5}
        with tempfile.TemporaryDirectory() as tmp, patch.object(ranges, "_request", serve), patch.dict(ranges._directories, clear=True):
            path = ranges.fetch_member(spec, Path(tmp))
            self.assertEqual(path.read_bytes(), b"23,45")
            requests = len(meter_events)
            self.assertGreater(requests, 1)
            self.assertEqual(ranges.fetch_member(spec, Path(tmp)), path)
            self.assertEqual(len(meter_events), requests)
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                ranges.fetch_member(spec | {"sha256":"a"*64}, Path(tmp))
