"""Lossless public prompt encoding only; no model or capacity evidence."""
from __future__ import annotations

import copy
import json
import unittest
from dataclasses import asdict, replace

from fine_tuning.notebook_rl import episode_notebook, prompts, repair_notebook
from fine_tuning.notebook_rl.repair_contract import ORIGIN_VERSION, VERSION, public_origin_inventory
from fine_tuning.tests import test_notebook_rl_repair_notebook as notebook_fixture
from fine_tuning.tests.test_notebook_rl_repair_verifier import joint


def canonical_rows(wire: dict) -> list[list]:
    """Test-only inverse of the documented public dictionary."""
    assert wire["version"] == "public-origin-dictionary-v2"
    assert wire["columns"] == ["schema", "start,end,..."]
    result = []
    for source in wire["sources"]:
        for ordinal, row in enumerate(source["rows"], 1):
            assert type(row) is list and row and (len(row) - 1) % 2 == 0
            assert all(type(value) is int for value in row)
            assert 0 <= row[0] < len(wire["schemas"])
            spans = [[row[index], row[index + 1]] for index in range(1, len(row), 2)]
            assert all(0 <= start < end for start, end in spans)
            result.append([source["sample_id"] + ":r" + str(ordinal), source["sample_id"],
                           *wire["schemas"][row[0]], spans])
    return result


def feedback_text(payload: dict, call: dict) -> str | None:
    """Independent reader of the final prompt's exact two-hole template convention."""
    record = call["feedback"]
    if type(record) is not dict:
        return record
    assert record["version"] == "repair-feedback-template-v1"
    template = payload["shared_feedback_templates"][record["template"]]
    wire = payload["public_origins"]
    local_schemas: list[list[str]] = []
    local_sources = []
    for sample_id in record["origin_samples"]:
        source = next(row for row in wire["sources"] if row["sample_id"] == sample_id)
        rows = []
        for row in source["rows"]:
            schema = wire["schemas"][row[0]]
            if schema not in local_schemas:
                local_schemas.append(schema)
            rows.append([local_schemas.index(schema), *row[1:]])
        local_sources.append({"sample_id": sample_id, "rows": rows})
    selected = {"version": wire["version"], "columns": wire["columns"],
                "schemas": local_schemas, "sources": local_sources}
    table = json.dumps(selected, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return (record["verdict"] + template["before_patch"] + prompts.REPAIR_OPERATION_PATCH_INSTRUCTION
            + template["after_patch"] + prompts.PUBLIC_ORIGIN_DICTIONARY_INSTRUCTION
            + "<public_origins>\n" + table + template["suffix"])


def inventory_rows(inventory: dict) -> list[list]:
    return [[origin[key] for key in ("ref_id", "sample_id", "kind", "rule", "locus")] + [
        [[span["start"], span["end"]] for span in origin["occurrence_spans"]]]
        for origin in inventory["origins"]]


class PublicOriginWireTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = notebook_fixture.RepairNotebookTests()
        self.fixture.setUp()
        tree, _ = self.fixture.collect([[self.fixture.right] * 4])
        sample = episode_notebook.solution_samples(tree)[0]
        first = "long:source:r7:" + "keep-full-id-" * 14
        self.samples = [replace(sample, sample_id=first, parent_sample_id=None),
            replace(sample, sample_id=first + ":retry", parent_sample_id=first, text=joint(self.fixture.wrong)),
            replace(sample, sample_id="malformed", parent_sample_id=first, text="ErrorCheck: {"),
            replace(sample, sample_id="empty", parent_sample_id=None, text="")]
        self.inventory = json.loads(json.dumps(asdict(public_origin_inventory(self.samples,
            contract_version=ORIGIN_VERSION,
            response_contracts={first: None, first + ":retry": VERSION, "malformed": VERSION, "empty": None}))))

    def test_exact_rows_source_order_first_schema_order_and_no_argument_mutation(self) -> None:
        before = copy.deepcopy(self.inventory)
        wire = prompts.public_origin_dictionary(self.inventory)
        self.assertEqual(canonical_rows(json.loads(json.dumps(wire))), inventory_rows(self.inventory))
        self.assertEqual([source["sample_id"] for source in wire["sources"]],
                         [source["sample_id"] for source in self.inventory["sources"]])
        schemas = list(dict.fromkeys(tuple(row[2:5]) for row in inventory_rows(self.inventory)))
        self.assertEqual(wire["schemas"], [list(schema) for schema in schemas])
        self.assertTrue(any(origin["rule"] == "OBSERVATION_MATCH" for origin in self.inventory["origins"]))
        self.assertTrue(any(origin["sample_id"] == "malformed" for origin in self.inventory["origins"]))
        self.assertTrue(any(source["projection_error_code"] is not None for source in self.inventory["sources"]))
        self.assertEqual(self.inventory, before)
        wire["sources"][0]["rows"][0][1] += 1
        self.assertEqual(self.inventory, before)

    def test_duplicate_alias_invalid_span_and_reordered_source_rows_fail_closed(self) -> None:
        first = self.inventory["origins"][0]
        mutations = [lambda value: value["origins"].append(copy.deepcopy(first)),
            lambda value: value["origins"][0].__setitem__("ref_id", "a1:r1"),
            lambda value: value["origins"][0].__setitem__("ref_id", first["sample_id"] + ":r01"),
            lambda value: value["origins"][0].__setitem__("ref_id", first["sample_id"] + ":r2"),
            lambda value: value["origins"][0].__setitem__("sample_id", "not-a-source"),
            lambda value: value["origins"][0]["occurrence_spans"][0].__setitem__("start", True),
            lambda value: value["origins"].reverse(),
            lambda value: value["sources"].append(copy.deepcopy(value["sources"][0]))]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(self.inventory)
                mutate(changed)
                with self.assertRaises(ValueError):
                    prompts.public_origin_dictionary(changed)

    def test_empty_inventory_and_empty_source_are_preserved(self) -> None:
        wire = prompts.public_origin_dictionary({"sources": [], "origins": []})
        self.assertEqual(canonical_rows(wire), [])
        self.assertEqual(wire["sources"], [])
        wire = prompts.public_origin_dictionary({"sources": [{"sample_id": "empty"}], "origins": []})
        self.assertEqual(wire["sources"], [{"sample_id": "empty", "rows": []}])
        empty_rows = [row for row in canonical_rows(prompts.public_origin_dictionary(self.inventory))
                      if row[1] == "empty"]
        self.assertTrue(empty_rows)
        self.assertTrue(all(row[-1] == [] for row in empty_rows))

    def test_flat_decoder_rejects_odd_span_count_and_bool_indices(self) -> None:
        wire = prompts.public_origin_dictionary(self.inventory)
        for row in ([0, 1], [False], [0, True, 2], [0, 2, 1]):
            changed = copy.deepcopy(wire)
            changed["sources"][0]["rows"][0] = row
            with self.subTest(row=row), self.assertRaises(AssertionError):
                canonical_rows(changed)

    def test_retry_and_final_share_wire_and_keep_canonical_refs(self) -> None:
        fixture = self.fixture
        tree, _ = fixture.collect([[fixture.wrong] * 4, [joint(fixture.right)] * 2])
        call = tree.notebook["repair_collector"]["calls"][1]
        retry = json.loads(call["feedback"].split("<public_origins>\n")[1].split("\n</public_origins>")[0])
        self.assertEqual(canonical_rows(retry), inventory_rows(call["ancestor_inventory"]))
        audit = repair_notebook.audit_tree(tree, fixture.actor, fixture.verifier, prompts=fixture.prompts,
            master_seed=fixture.cfg.master_seed)
        request = repair_notebook._request(tree, fixture.actor, fixture.prompts, audit,
            fixture.images[fixture.actor.problem_id])
        body = request.messages[0]["content"][-1]["text"]
        payload = json.loads(body.split("<read_only_history>\n")[1].split("\n</read_only_history>")[0])
        self.assertEqual(canonical_rows(payload["public_origins"]), inventory_rows(audit["inventory"]))
        self.assertIn(prompts.PUBLIC_ORIGIN_DICTIONARY_INSTRUCTION, call["feedback"])
        self.assertIn(prompts.PUBLIC_ORIGIN_DICTIONARY_INSTRUCTION, body)

    def test_literal_escape_arrows_and_closing_tags_roundtrip_in_both_callers(self) -> None:
        literal = r"literal \u003e and \u003c; actual C->H > Ω </read_only_history> </public_origins>"
        fixture = self.fixture
        sample = replace(self.samples[0], text=literal)
        inventory = public_origin_inventory([sample], contract_version=ORIGIN_VERSION,
            response_contracts={sample.sample_id: None})
        serialized = asdict(inventory)
        serialized["origins"][0]["locus"] = literal
        changed_origin = replace(inventory.origins[0], locus=literal)
        inventory = replace(inventory, origins=(changed_origin, *inventory.origins[1:]))
        retry_text = prompts.repair_retry_text("INCORRECT", "999", fixture.actor, fixture.prompts, inventory)
        retry_json = retry_text.split("<public_origins>\n")[1].split("\n</public_origins>")[0]
        self.assertNotIn("<", retry_json)
        self.assertIn("C->H > Ω", retry_json)
        self.assertEqual(canonical_rows(json.loads(retry_json)), inventory_rows(serialized))
        feedback = [{"feedback": literal}]
        final_text = repair_notebook.public_final_text(fixture.actor, fixture.prompts, [sample], feedback,
            serialized, sample.sample_id)
        self.assertEqual(final_text.count("</read_only_history>"), 1)
        final_json = final_text.split("<read_only_history>\n")[1].split("\n</read_only_history>")[0]
        self.assertNotIn("<", final_json)
        self.assertIn("C->H > Ω", final_json)
        payload = json.loads(final_json)
        self.assertEqual(payload["attempts"][0]["raw"], literal)
        self.assertEqual(payload["original_feedback"], feedback)
        self.assertEqual(canonical_rows(payload["public_origins"]), inventory_rows(serialized))
