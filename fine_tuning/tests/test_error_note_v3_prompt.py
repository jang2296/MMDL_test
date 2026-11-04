"""Authored CPU prompt cases; no model generation, runtime artifacts or teacher explanations."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fine_tuning.error_note import collect, collect_v3, contract, protocol
from fine_tuning.notebook_rl.prompts import p0_eval_text
from fine_tuning.tests.test_error_note_collect import default_output, make_actors, make_items
from fine_tuning.tests.test_error_note_v2_collect import FakeStream, KWARGS, load, t1_rows, v2_note_text
from fine_tuning.tests.test_error_note_v3_collect import CollectionFixture


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class NotePromptV3Tests(unittest.TestCase):
    def test_legacy_instructions_prompts_and_protocol_hashes_are_unchanged(self):
        self.assertEqual(sha(contract.NOTE_INSTRUCTION), "08804791323968b0d3ade3d79f5ad729036336c8e6716c9e30a53a6301fdc742")
        self.assertEqual(sha(contract.NOTE_INSTRUCTION_V2), "c76ab86bf018baa275deaadc3eb9b083c3242f73299674371e83ebf21228229a")
        self.assertEqual(sha(contract.note_prompt("Q?", "earlier", "INCORRECT", "B")),
                         "28f8b3ca9fbbaf193495593a08bf701579b49bbe16d92ad7d827b11c2767ec74")
        self.assertEqual(sha(contract.note_prompt_v2("Q?", "earlier", "INCORRECT", "B")),
                         "142e94e9dc98164461412578a68a4d1f9de76a408f94536bb936337d09b4e507")
        self.assertEqual(protocol.plain_protocol("bf16")["sha256"], "8c4548feeff1ff3acff79c7319751dc99a7a34f25d09effb3311c626b1645d77")
        self.assertEqual(protocol.plain_protocol_v2("bf16", 214)["sha256"], "fce086a2268d1c61d40d86b9ec3c6ab34d3923e5ca0c12a02546e44fbe5a0de9")

    def generate(self, out, *, answer_contract="training-answer-v2"):
        stream = FakeStream(lambda request: default_output(request) | {"text": v2_note_text()})
        collect.run_note(make_items(1), make_actors(1), t1_rows(["INCORRECT"]), {"p0": "B"}, load,
            stream, collect.StreamStats(), lambda *args: self.fail("note must not contain a corrected solution"),
            out, **KWARGS, version="v2", answer_contract=answer_contract)
        return stream

    def test_only_opt_in_generation_uses_v3_instruction_and_keeps_v2_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stream = self.generate(out)
            expected = contract.note_prompt_v3(p0_eval_text("Question 0?", ("alpha", "beta")),
                                                "my first solution 0", "INCORRECT", "B")
            self.assertEqual(stream.requests[0]["prompt"], expected)
            self.assertEqual(stream.requests[0]["messages"][0]["content"][-1]["text"], expected)
            self.assertEqual(stream.requests[0]["max_tokens"], 8192)
            row = collect.read_jsonl_done(out / "note.jsonl")["p0"]
            self.assertEqual(row["note_version"], "v2")
            self.assertEqual(row["note_instruction_sha256"], sha(contract.NOTE_INSTRUCTION_V3))
            self.assertIsNotNone(contract.parse_note_v2(row["text"]))
            self.assertNotIn("Corrected solution", expected)
            self.assertEqual(self.generate(out).requests, [])
        with tempfile.TemporaryDirectory() as tmp:
            stream = self.generate(Path(tmp), answer_contract=contract.TURN1_ANSWER_CONTRACT)
            self.assertTrue(stream.requests[0]["prompt"].endswith(contract.NOTE_INSTRUCTION_V2))
            self.assertNotIn("note_instruction_sha256", collect.read_jsonl_done(Path(tmp) / "note.jsonl")["p0"])

    def test_resume_rejects_legacy_mix_instruction_or_exact_input_mismatch(self):
        for changed in ("legacy", "instruction", "prompt", "missing_prompt"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                self.generate(out, answer_contract=contract.TURN1_ANSWER_CONTRACT if changed == "legacy" else "training-answer-v2")
                row = collect.read_jsonl_done(out / "note.jsonl")["p0"]
                if changed == "instruction":
                    row["note_instruction_sha256"] = "0" * 64
                elif changed == "prompt":
                    row["prompt"] = row["prompt"].replace("my first solution 0", "changed earlier text")
                elif changed == "missing_prompt":
                    row.pop("prompt")
                (out / "note.jsonl").write_text(json.dumps(row) + "\n")
                with self.assertRaises((ValueError, KeyError)):
                    self.generate(out)
        with tempfile.TemporaryDirectory() as tmp:
            self.generate(Path(tmp))
            with self.assertRaisesRegex(ValueError, "instruction differs"):
                self.generate(Path(tmp), answer_contract=contract.TURN1_ANSWER_CONTRACT)

    def test_collection_protocol_and_stored_notes_bind_the_exact_v3_prompt(self):
        hashes = dict.fromkeys(("rows", "order", "audit", "sidecar", "admission"), "a" * 64)
        expected = protocol.plain_protocol_v3("bf16", 1, arm="note", contracts=hashes)
        self.assertEqual(expected["prompts_sha256"], protocol._digest({"note": contract.NOTE_INSTRUCTION_V3,
                                                                    "resolve_header": contract.RESOLVE_HEADER}))
        with patch.object(protocol, "NOTE_INSTRUCTION_V3", contract.NOTE_INSTRUCTION_V3 + "\n"):
            self.assertNotEqual(expected["sha256"], protocol.plain_protocol_v3("bf16", 1, arm="note", contracts=hashes)["sha256"])
        with tempfile.TemporaryDirectory() as tmp:
            fixture = CollectionFixture(tmp)
            stored = json.loads((fixture.out / "collection.json").read_text())
            self.assertEqual(stored["note_prompt_sha256"], sha(contract.NOTE_INSTRUCTION_V3))
            stored["note_prompt_sha256"] = sha(contract.NOTE_INSTRUCTION_V2)
            collect.write_json(fixture.out / "collection.json", stored)
            with self.assertRaisesRegex(ValueError, "immutable evidence changed"):
                collect_v3.bind_collection(fixture.args, fixture.items)
        for changed in ("instruction", "prompt", "missing_prompt"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                fixture = CollectionFixture(tmp)
                row = collect.read_jsonl_done(fixture.out / "note.jsonl")[fixture.pid]
                if changed == "instruction":
                    row.pop("note_instruction_sha256")
                elif changed == "missing_prompt":
                    row.pop("prompt")
                else:
                    row["prompt"] = row["prompt"].replace("Correct answer: B", "Correct answer: A")
                (fixture.out / "note.jsonl").write_text(json.dumps(row) + "\n")
                with fixture.vault(), self.assertRaisesRegex(ValueError, "v3 note .* binding mismatch"):
                    collect_v3.records_and_sources(fixture.args)


if __name__ == "__main__":
    unittest.main()
