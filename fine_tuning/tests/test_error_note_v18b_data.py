"""Portable panel selection and identity gates stay separate from training roles."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fine_tuning.error_note import v18b_data as data
from fine_tuning.error_note.v18b_monitor import REASONING_SOURCES


class PanelTests(unittest.TestCase):
    def test_public_metadata_selects_64_preexisting_dev_reasoners(self):
        recipe = data.panel_recipe()
        self.assertEqual(Counter(x["source"] for x in recipe["items"]), dict.fromkeys(REASONING_SOURCES, 16))
        self.assertTrue(recipe["selection"]["before_any_new_generation"])
        self.assertTrue(all(x["training_allowed"] is False for x in recipe["items"]))
        self.assertEqual(data.ROLES, ("train", "prep", "monitor", "effect", "performance", "reasoning_monitor"))

    def test_public_panel_is_disjoint_from_all_base_roles_and_live_monitor_has_213_ids(self):
        from fine_tuning.error_note.v18_recipe import load_recipe
        from fine_tuning.error_note.v18b_monitor import monitor_comparison
        from fine_tuning.tests.test_error_note_v18b_monitor import record
        base = load_recipe()
        panel = data.panel_recipe()
        def identity(row):
            return {key: row[key] for key in ("problem_id", "source", "native_id", "family_id")} | {
                "image_sha256": [x["sha256"] for x in row["images"]],
                "pixel_sha256": [x["pixel_sha256"] for x in row["images"]]}
        data._isolated([identity(x) for x in panel["items"]], [identity(x) for x in base["items"]])
        live = [x for x in base["items"] if x["role"] == "monitor" and x["source"] != "hf-scienceqa"]
        self.assertEqual(len(live), 149)
        self.assertEqual(len([x for x in base["items"] if x["role"] == "monitor"]), 200)
        items = [dict(identity(x), training_allowed=False, benchmark_derived=False) for x in live + panel["items"]]
        records = [record(x["problem_id"]) for x in items]
        comparison = monitor_comparison(records, records, records, items)
        self.assertEqual((comparison["generation_n"], comparison["decision_n"]), (213, 213))
        self.assertNotIn("hf-scienceqa", comparison["sources"])

    def test_id_family_native_byte_pixel_and_transitive_bridges_are_excluded(self):
        panel = {"problem_id": "panel", "source": "s", "native_id": "one", "family_id": "family",
                 "image_sha256": ["bytes"], "pixel_sha256": ["pixels"]}
        for field, value in (("problem_id", "panel"), ("family_id", "family"),
                             ("image_sha256", ["bytes"]), ("pixel_sha256", ["pixels"]),
                             ("native_id", "one")):
            existing = {"problem_id": "existing", "source": "s", field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "DUPLICATE|OVERLAP"):
                data._isolated([panel], [existing])
        data._isolated([panel], [{"problem_id": "different", "source": "other", "native_id": "one"}])
        bridge = {"problem_id": "bridge", "family_ids": ["family", "other"]}
        with self.assertRaisesRegex(ValueError, "OVERLAP"):
            data._isolated([panel], [{"problem_id": "other", "family_id": "other"}, bridge])

    def test_prepare_path_materializes_before_returning_full_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "data"
            with patch.object(data.base, "prepare") as build, patch.object(data, "materialize_reasoning") as panel, \
                    patch.object(data, "validate", return_value={"reasoning_monitor_count": 64}) as validate:
                result = data.prepare(Path("recipe"), Path("cache"), output, offline=True, tokenizer="local")
                build.assert_called_once_with(Path("recipe"), Path("cache"), output,
                    offline=True, workers=4, tokenizer="local")
                panel.assert_called_once_with(output, Path("cache"), offline=True, workers=4)
                validate.assert_called_once_with(output, tokenizer="local")
                self.assertEqual(result["reasoning_monitor_count"], 64)

    def test_changed_source_or_training_role_is_rejected_even_with_valid_document_digest(self):
        recipe = data.panel_recipe()
        for change in ({"training_allowed": True}, {"source_revision": "main"}, {"role": "train"}):
            bad = deepcopy(recipe)
            bad["items"][0].update(change)
            with patch.object(data, "_document", return_value=bad), self.assertRaisesRegex(ValueError, "PERMISSION_OR_SOURCE"):
                data.panel_recipe()


if __name__ == "__main__":
    unittest.main()
