"""CPU checks for v17 quarantine, fixed scopes, and private/public separation."""
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from fine_tuning.error_note import v17_data as data
from fine_tuning.notebook_rl.pod import data_bundle, verify


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()


def candidate(pid, source="source", **kwargs):
    return dict(problem_id=pid, source=source, native_id=pid, family_id=pid,
                image_sha256=["image:" + pid], pixel_sha256=["pixel:" + pid],
                eligible=True, exclusion_reasons=[]) | kwargs


class DataTests(unittest.TestCase):
    def test_transitive_native_family_byte_pixel_closure(self):
        a, bridge, c = candidate("a"), candidate("bridge"), candidate("c")
        bridge["image_sha256"] = a["image_sha256"]
        bridge["family_id"] = c["family_id"]
        components = data.identity_components([a, bridge, c])
        self.assertEqual(len(set(components.values())), 1)
        alias = candidate("alias")
        alias["native_id"] = a["native_id"]
        self.assertEqual(len(set(data.identity_components([a, alias]).values())), 1)
        alias = candidate("alias")
        alias["pixel_sha256"] = a["pixel_sha256"]
        self.assertEqual(len(set(data.identity_components([a, alias]).values())), 1)

    def test_historical_union_only_explicit_complete_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(2):
                path = root / f"v{index}.json"
                path.write_text(json.dumps({"items": [candidate("old", split="sealed", training_allowed=False)]}))
                paths.append(path)
            rows, report = data.historical_reservations(paths)
            self.assertEqual(report["union_ids"], 1)
            self.assertEqual(len(rows), 2)
            authorization = root / "user.txt"
            authorization.write_text("Explicitly release old from both historical reservations.")
            release = root / "release.json"
            receipt = {"schema": "v17-reservation-release-v1", "authority": "USER_EXPLICIT",
                "authorization_ref": data.curation._ref(authorization),
                "reservation_refs": [data.curation._ref(x) for x in paths], "problem_ids": ["old"]}
            release.write_text(json.dumps(receipt))
            self.assertEqual(data.historical_reservations(paths, [release])[0], [])
            receipt["reservation_refs"].pop()
            release.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "RELEASE_INCOMPLETE"):
                data.historical_reservations(paths, [release])

    def test_deterministic_replacement_and_disjoint_effect_probe(self):
        rows = [candidate(str(x), source="a" if x % 2 else "b") for x in range(30)]
        current = {"train": rows[:3], "prep": rows[3:4], "monitor": rows[4:5]}
        caps = {x["problem_id"]: {"repair_catalog_eligible": False} for x in rows}
        kwargs = dict(capabilities=caps, counts=(3, 1, 1), effect_min=4, effect_max=8, performance_count=2)
        roles, receipt = data.select_roles(rows, current, [rows[0]], **kwargs)
        reverse, _ = data.select_roles(list(reversed(rows)), current, [rows[0]], **kwargs)
        self.assertEqual({s: [x["problem_id"] for x in v] for s, v in roles.items()},
                         {s: [x["problem_id"] for x in v] for s, v in reverse.items()})
        self.assertEqual(receipt["quarantine"][0]["reason"], "HISTORICAL_RESERVATION_CLOSURE")
        items = [x for group in roles.values() for x in group]
        self.assertEqual(len(items), len(set(data.identity_components(items).values())))
        self.assertEqual(len(roles["effect"]), 8)
        self.assertEqual(len(roles["performance"]), 2)
        with self.assertRaisesRegex(ValueError, "EFFECT_DATA_INSUFFICIENT"):
            data.select_roles(rows[:8], current, [], **kwargs)

    def test_training_order_separates_actual_obs_and_repair_strata(self):
        rows = [candidate(str(x)) for x in range(60)]
        capabilities = {x["problem_id"]: {"obs_eligible": index % 3 == 0,
            "repair_catalog_eligible": index % 3 != 2} for index, x in enumerate(rows)}
        order = data.training_order(rows, capabilities)
        reverse = data.training_order(list(reversed(rows)), capabilities)
        self.assertEqual(order["items"], reverse["items"])
        self.assertEqual(set(order["v17_capability_prefix_balance"]["population_counts"]),
                         {"obs=1,repair=1", "obs=0,repair=1", "obs=0,repair=0"})
        self.assertLess(order["v17_capability_prefix_balance"]["max_abs_deviation"], 2)

    def test_load_items_consumes_exact_frozen_order_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for pid in ("a", "b", "c"):
                item = candidate(pid, split="train", training_allowed=True,
                                 verifier={"gold": "A", "answer_type": "mcq", "accept": {}})
                item["v17_private_verifier"] = data.private_verifier(item, tokenizer=Tokenizer())
                item["v17_capability"] = data.capability_manifest(item, tokenizer=Tokenizer())
                rows.append(item)
            (root / "dataset-manifest.json").write_text(json.dumps({"version": data.VERSION, "items": rows}))
            ids = ["c", "a", "b"]
            order = {"items": [{"problem_id": pid} for pid in ids], "batches": [{"problem_ids": ids}]}
            (root / "order.json").write_text(json.dumps(order))
            self.assertEqual([x["problem_id"] for x in data.load_items(root, "train")], ids)
            order["items"][1] = {"problem_id": "c"}
            (root / "order.json").write_text(json.dumps(order))
            with self.assertRaisesRegex(ValueError, "V17_ORDER_IDS"):
                data.load_items(root, "train")

    def test_copy_image_is_unaliased_and_actual_bundle_spec_accepts_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.png", root / "output.png"
            source.write_bytes(b"source image fixture")
            data._copy_image(source, target, data.curation._sha(source))
            self.assertEqual(target.stat().st_nlink, 1)
            self.assertNotEqual(source.stat().st_ino, target.stat().st_ino)
            spec = {"version": data.legacy.SPEC_VERSION, "paths": [str(target)]}
            self.assertEqual(data.legacy.expand_spec(spec), [str(target)])
            os.link(target, root / "alias.png")
            with self.assertRaisesRegex(ValueError, "ALIASED"):
                data._copy_image(source, target, data.curation._sha(source))

    def test_performance_request_manifest_is_p0_and_identity_disjoint(self):
        rows = {role: [candidate(role, split=role, training_allowed=role in {"train", "prep"},
                    question="Question text", choices=["Choice one", "Choice two"], image_paths=["/private/image.png"])]
                for role in ("train", "prep", "monitor", "effect", "performance")}
        result = data.performance_requests(rows, benchmark_exclusion_sha256="a" * 64)
        self.assertEqual(result["sha256"], data.digest({k: v for k, v in result.items() if k != "sha256"}))
        request = result["items"][0]
        self.assertEqual(request["messages"][0]["content"][0], {"type": "image"})
        self.assertIn("Please select the correct answer", request["messages"][0]["content"][-1]["text"])
        self.assertFalse({"gold", "annotation", "facts"} & request.keys())
        rows["performance"][0]["pixel_sha256"] = rows["effect"][0]["pixel_sha256"]
        with self.assertRaisesRegex(ValueError, "PERFORMANCE_IDENTITY_OVERLAP"):
            data.performance_requests(rows, benchmark_exclusion_sha256="a" * 64)

    def test_clean_pack_stage_validate_and_v17_pod_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "assembled"
            (root / "vault").mkdir(parents=True)
            items, roles = [], {}
            for role, count in (("train", 3000), ("prep", 512), ("monitor", 200), ("effect", 64), ("performance", 16)):
                roles[role] = []
                for index in range(count):
                    item = candidate(f"{role}-{index}", split=role, training_allowed=role in {"train", "prep"},
                        question="Fixture question", choices=[], image_paths=[], image_sha256=[],
                        verifier={"gold": "A", "answer_type": "exact", "accept": {}})
                    item["v17_private_verifier"] = data.private_verifier(item, tokenizer=Tokenizer())
                    item["v17_capability"] = data.capability_manifest(item, tokenizer=Tokenizer())
                    items.append(item)
                    roles[role].append(item)
            manifest = root / "dataset-manifest.json"
            data.legacy._write(manifest, {"version": data.VERSION, "items": items})
            ids = [item["problem_id"] for item in reversed(roles["train"])]
            data.legacy._write(root / "order.json", {"items": [{"problem_id": pid} for pid in ids],
                "batches": [{"problem_ids": ids[i:i + 16]} for i in range(0, len(ids), 16)]})
            data.legacy._write(root / "performance-requests.json", data.performance_requests(roles,
                benchmark_exclusion_sha256="b" * 64))
            files = sorted(path for path in root.rglob("*") if path.is_file())
            receipt = root / "payload-receipt.json"
            data.legacy._write(receipt, {"source_sha256": data.curation._sha(Path(data.__file__)),
                "files": {str(path.relative_to(root)): data.curation._sha(path) for path in files},
                "legacy_files": {}, "original_bindings": {}, "selection": {
                    "effect_status": "EXPOSURE_HISTORY_INCOMPLETE", "effect_eligible": False}})
            spec = root / "bundle-spec.json"
            data.legacy._write(spec, {"version": data_bundle.SPEC_VERSION,
                "paths": [str(path) for path in files] + [str(receipt), str(spec)]})
            data.validate(root)
            packed = data_bundle.pack(spec, temporary / "transport", "v17")
            self.assertIn(str(spec), {record["path"] for record in packed["files"]})
            shutil.move(root, temporary / "retained-original")
            bundle = temporary / "transport/v17.bundle.json"
            data_bundle.stage(temporary / "transport/v17.tar", bundle, data.curation._sha(bundle))
            self.assertTrue(spec.is_file())
            self.assertEqual(data.validate(root)["counts"]["train"], 3000)
            pins = {"dataset_manifest": str(manifest), "dataset_manifest_sha256": data.curation._sha(manifest),
                    "vault_root": str(root / "vault")}
            detail = verify.check_dataset(sys.executable, Path(__file__).resolve().parents[2], pins)
            self.assertIn("v17_data.validate", detail)
            with self.assertRaisesRegex(ValueError, "pinned"):
                verify.check_dataset(sys.executable, Path(__file__).resolve().parents[2],
                                     pins | {"dataset_manifest_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "manifest/vault path"):
                verify.check_dataset(sys.executable, Path(__file__).resolve().parents[2],
                                     pins | {"vault_root": str(temporary)})

    def test_fixed_development_permission_ignores_original_train_split(self):
        item = candidate("x", lineage={"split": "train"}, split="monitor", training_allowed=False)
        cap = data.capability_manifest(item)
        self.assertEqual(cap["source_split"], "train")
        self.assertFalse(cap["training_allowed"])

    def test_generic_schema_does_not_change_with_annotation(self):
        item = candidate("x", source="hf-gvlqa", raw_annotation={"n": 3, "m": 2, "edges": [[0, 1], [1, 2]]},
                         native_metadata={"directed": True})
        before = data.public_schema(item)
        item["raw_annotation"]["edges"] = [[0, 2], [2, 1]]
        self.assertEqual(before, data.public_schema(item))
        self.assertTrue({"entity", "value", "facts", "target", "ordered_entities"}.isdisjoint(before))

    def test_graph_fixed_scope_induced_edges_and_token_limits(self):
        item = candidate("x", source="hf-gvlqa", raw_annotation={"n": 6, "m": 5,
                         "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]}, native_metadata={"directed": True})
        verifier = data.private_verifier(item, tokenizer=Tokenizer())
        self.assertTrue(verifier["obs_eligible"])
        self.assertEqual(verifier["closed_world_scopes"]["visible_nodes"], [0, 1, 2, 3])
        self.assertEqual(len(verifier["facts"]), 7)
        self.assertEqual(set(verifier["supported_claim_kinds"]), set(data.CLAIM_KINDS["graph"]))
        self.assertFalse(any(f["entity"] == [3, 4] for f in verifier["facts"]))
        class TooLarge:
            def encode(self, text, **kwargs):
                return [0] * 769
        self.assertEqual(data.capability_manifest(item, tokenizer=TooLarge())["obs_reason"], "SCOPE_TOKEN_LIMIT")
        item["v17_private_verifier"] = verifier
        self.assertEqual(verifier, data.private_verifier(item))

    def test_unsupported_parent_replays_frozen_token_measurement(self):
        item = candidate("unsupported")
        verifier = data.private_verifier(item, tokenizer=Tokenizer())
        item["v17_private_verifier"] = verifier
        self.assertEqual(data.private_verifier(item), verifier)
        self.assertFalse(verifier["obs_eligible"])

    def test_partial_geometry_repair_is_separate_from_obs(self):
        item = candidate("x", source="hf-geometry3k", raw_annotation={"diagram_logic_forms": [
            "Equals(LengthOf(Line(A, B)), 5)"]})
        cap = data.capability_manifest(item, tokenizer=Tokenizer())
        self.assertTrue(cap["repair_catalog_eligible"])
        self.assertFalse(cap["obs_eligible"])
        self.assertEqual(cap["obs_reason"], "SCOPE_COMPLETENESS_UNVERIFIED")

    def test_chart_ocr_is_not_positional_or_order_certification(self):
        chart = {"type": "v_bar", "models": [{"name": "Revenue", "x": ["Jan", "Feb"], "y": ["5", "7"]}],
                 "general_figure_info": {"y_axis": {"label": {"text": "USD"}}}}
        table = "Month,Revenue\nFeb,7\nJan,5\n"
        raw = {"chart": chart, "table_csv": table, "visibility_receipt": {
            "engine_type": "NON_LLM_OCR", "status": "COMPLETE", "text": "Jan 5 Feb 7 USD",
            "annotation_sha256": data.digest(chart), "image_sha256": "image:x"}}
        item = candidate("x", source="hf-chartqa", raw_annotation=raw,
                         native_metadata={"image_filename": "two_col_1.png"})
        self.assertEqual(data.chart_facts(item)[1], "POSITION_VISIBILITY_ORDER_UNVERIFIED")
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "positions.json"
            evidence.write_text("source-bound fixture position review")
            raw["position_visibility_order_receipt"] = {"schema": "v17-chart-position-visibility-order-v1",
                "status": "VERIFIED", "image_sha256": "image:x", "annotation_sha256": data.digest(chart),
                "table_sha256": hashlib.sha256(table.encode()).hexdigest(), "position_bound": True,
                "units_verified": True, "evidence_refs": [data.curation._ref(evidence)],
                "series_order": ["Revenue"], "category_order": ["Jan", "Feb"], "units": {"Revenue": "USD"},
                "visible_values": [{"series": "Revenue", "category": "Jan", "value": "5"},
                                   {"series": "Revenue", "category": "Feb", "value": "7"}]}
            facts, reason, _ = data.chart_facts(item)
            self.assertIsNone(reason)
            self.assertEqual([x["entity"] for x in facts], [["Revenue", "Jan"], ["Revenue", "Feb"]])
            changed = copy.deepcopy(item)
            changed["raw_annotation"]["position_visibility_order_receipt"]["category_order"].reverse()
            self.assertEqual(data.chart_facts(changed)[1], "CHART_POSITION_VALUES_MISMATCH")
            percent = copy.deepcopy(item)
            percent_raw = percent["raw_annotation"]
            percent_raw["chart"]["models"][0]["y"] = ["5%", "7%"]
            percent_raw["table_csv"] = "Month,Revenue\nFeb,7%\nJan,5%\n"
            percent_raw["visibility_receipt"].update(text="Jan 5% Feb 7% USD", annotation_sha256=data.digest(percent_raw["chart"]))
            cert = percent_raw["position_visibility_order_receipt"]
            cert.update(annotation_sha256=data.digest(percent_raw["chart"]),
                        table_sha256=hashlib.sha256(percent_raw["table_csv"].encode()).hexdigest(), units={"Revenue": "%"})
            cert["visible_values"][0]["value"], cert["visible_values"][1]["value"] = "5%", "7%"
            percent_facts, percent_reason, _ = data.chart_facts(percent)
            self.assertIsNone(percent_reason)
            self.assertEqual(percent_facts[0]["value"], "5")
            self.assertEqual(percent_facts[0]["unit"], "%")
            cert["units"] = {"Revenue": "USD"}
            self.assertEqual(data.chart_facts(percent)[1], "CHART_UNITS_UNVERIFIED")
            changed = copy.deepcopy(item)
            changed["image_sha256"] = ["different"]
            self.assertEqual(data.chart_facts(changed)[1], "CHART_IMAGE_ANNOTATION_BINDING_UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
