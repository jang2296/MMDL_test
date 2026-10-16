"""Synthetic CPU split-vault checks, including corruption and actor isolation."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from fine_tuning.notebook_rl.schema import ActorItem, Lineage, VerifierItem, to_json
from fine_tuning.notebook_rl.vault import (
    ACTOR_KEYS, read_actors, read_lineage, read_verifier, write_split,
)


def records():
    actor = ActorItem("fixture:one", "fixture", "family", "plain", "Read the number.", (),
                      "int", ("fixture/image.png",), ("a" * 64,))
    verifier = VerifierItem(actor.problem_id, "int", "53", {}, (), {}, {}, (), (), ())
    lineage = Lineage(actor.problem_id, actor.source, "v1", "INTERNAL_FIXTURE", "PROGRAM",
                      "rendered", "procedural", "v1", 7, "read", "b" * 16, "train", ())
    return [actor], [verifier], [lineage]


class VaultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "split"

    def write(self, values=None):
        return write_split(self.root, *(values or records()))

    def rehash(self, name, raw, *, rows=None):
        (self.root / name).write_bytes(raw)
        path = self.root / "manifest.yaml"
        data = json.loads(path.read_bytes())
        data["files"][name]["sha256"] = hashlib.sha256(raw).hexdigest()
        if rows is not None:
            data["files"][name]["rows"] = rows
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_roundtrip_manifest_hashes_and_private_permissions(self):
        actors, verifiers, lineages = records()
        manifest = self.write()
        self.assertEqual(read_actors(self.root), actors)
        self.assertEqual(read_verifier(self.root), {verifiers[0].problem_id: verifiers[0]})
        self.assertEqual(read_lineage(self.root), {lineages[0].problem_id: lineages[0]})
        self.assertEqual(yaml.safe_load((self.root / "manifest.yaml").read_bytes()), manifest)
        self.assertEqual(ACTOR_KEYS, frozenset(actors[0].__dataclass_fields__))
        for name, meta in manifest["files"].items():
            self.assertEqual(meta["sha256"], hashlib.sha256((self.root / name).read_bytes()).hexdigest())
            self.assertEqual(meta["rows"], 1)
            self.assertEqual((self.root / name).stat().st_mode & 0o777, 0o600)
        here = Path(__file__).resolve().parents[1] / "notebook_rl"
        self.assertEqual(manifest["protocol_sha256"],
                         hashlib.sha256((here / "protocol_v1.yaml").read_bytes()).hexdigest())
        for name, digest in manifest["code_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((here / name).read_bytes()).hexdigest())

    def test_sets_and_duplicate_ids_rejected_before_writes(self):
        actors, verifiers, lineages = records()
        for values in ((actors, [], lineages), (actors, verifiers, []),
                       (actors * 2, verifiers, lineages), (actors, verifiers * 2, lineages),
                       (actors, verifiers, lineages * 2)):
            with self.subTest(values=tuple(map(len, values))), self.assertRaises(ValueError):
                self.write(values)
            self.assertFalse(self.root.exists())

    def test_forbidden_or_invalid_lineage(self):
        actors, verifiers, lineages = records()
        for change in ({"benchmark_derived": True}, {"model_touched": True},
                       {"label_provenance": "MODEL"}, {"benchmark_derived": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.write((actors, verifiers, [replace(lineages[0], **change)]))
            self.assertFalse(self.root.exists())

    def test_cross_record_metadata_and_types_rejected(self):
        actors, verifiers, lineages = records()
        for values in ((actors, [replace(verifiers[0], answer_type="decimal")], lineages),
                       (actors, verifiers, [replace(lineages[0], source="other")]),
                       (verifiers, verifiers, lineages)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.write(values)

    def test_write_once_keeps_existing_bytes_and_incomplete_payloads(self):
        self.write()
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        with self.assertRaises(FileExistsError):
            self.write()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})
        (self.root / "manifest.yaml").unlink()
        with self.assertRaises(FileExistsError):
            self.write()
        with self.assertRaises(ValueError):
            read_actors(self.root)

    def test_leak_warning_contains_only_identity_and_field(self):
        actors, verifiers, lineages = records()
        actors = [replace(actors[0], question="Use SENSITIVEGOLD here.")]
        verifiers = [replace(verifiers[0], answer_type="short_text", gold="SENSITIVEGOLD")]
        actors = [replace(actors[0], answer_format="short_text")]
        manifest = self.write((actors, verifiers, lineages))
        self.assertEqual(manifest["leak_review"], [{"problem_id": actors[0].problem_id,
                                                   "field": "question"}])
        self.assertNotIn("SENSITIVEGOLD", (self.root / "manifest.yaml").read_text())

    def test_word_boundaries_and_mcq_do_not_warn(self):
        actors, verifiers, lineages = records()
        manifest = self.write(([replace(actors[0], question="The value is 1530.")],
                               verifiers, lineages))
        self.assertEqual(manifest["leak_review"], [])
        other = Path(self.temp.name) / "mcq"
        actor = replace(actors[0], question="A or B?", answer_format="mcq", choices=("A", "B"))
        verifier = replace(verifiers[0], answer_type="mcq", gold="A", choices=actor.choices)
        self.assertEqual(write_split(other, [actor], [verifier], lineages)["leak_review"], [])

    def test_actor_loader_does_not_open_private_files_under_audit_guards(self):
        self.write()
        code = """
import os, sys
from pathlib import Path
from fine_tuning.notebook_rl.guards import install_guards
from fine_tuning.notebook_rl.vault import read_actors
root = Path(sys.argv[1])
private = [root/'verifier_private.jsonl', root/'lineage.jsonl']
install_guards(forbidden_roots=[], vault_files=private, allow_vault=False, allow_benchmark=False)
opened = []
def track(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        opened.append(os.fsdecode(args[0]))
sys.addaudithook(track)
assert len(read_actors(root)) == 1
assert not any(str(p) in opened for p in private)
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.root)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        # The actor process intentionally cannot certify private payload bytes.
        (self.root / "verifier_private.jsonl").unlink()
        (self.root / "lineage.jsonl").unlink()
        self.assertEqual(len(read_actors(self.root)), 1)

    def test_each_loader_rejects_its_own_tampered_payload(self):
        self.write()
        for name, loader in (("actor_inputs.jsonl", read_actors),
                             ("verifier_private.jsonl", read_verifier),
                             ("lineage.jsonl", read_lineage)):
            path = self.root / name
            raw = path.read_bytes()
            path.write_bytes(raw + b" ")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "hash mismatch"):
                loader(self.root)
            path.write_bytes(raw)

    def test_actor_forbidden_key_even_with_updated_file_hash(self):
        self.write()
        row = json.loads(to_json(records()[0][0]))
        row["gold"] = "private"
        self.rehash("actor_inputs.jsonl", json.dumps(row).encode() + b"\n")
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            read_actors(self.root)

    def test_duplicate_ids_blank_lines_and_missing_newline(self):
        self.write()
        name = "actor_inputs.jsonl"
        raw = (self.root / name).read_bytes()
        for bad in (raw * 2, raw + b"\n", raw.rstrip(b"\n"), b"\xff\n",
                    b'{"question":"a","question":"b"}\n'):
            self.rehash(name, bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                read_actors(self.root)

    def test_wrong_id_and_manifest_shape_fail_closed(self):
        self.write()
        row = replace(records()[0][0], problem_id="fixture:other")
        self.rehash("actor_inputs.jsonl", to_json(row).encode() + b"\n")
        with self.assertRaisesRegex(ValueError, "identity/count"):
            read_actors(self.root)
        path = self.root / "manifest.yaml"
        good = path.read_bytes()
        for change in ({"code_sha256": {}}, {"protocol_sha256": "0" * 64},
                       {"leak_review": [{"problem_id": "x", "gold": "private"}]},
                       {"files": {}}, {"extra": True}):
            path.write_text(json.dumps(json.loads(good) | change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                read_actors(self.root)
        path.write_bytes(b'{"format":1,"format":2}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_actors(self.root)

    def test_symlink_hardlink_and_parent_aliases_rejected(self):
        self.write()
        alias = Path(self.temp.name) / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            read_actors(alias)
        with self.assertRaisesRegex(ValueError, "symlink"):
            write_split(alias / "child", *records())
        with self.assertRaisesRegex(ValueError, "parent path"):
            read_actors(self.root / ".." / "split")
        target = self.root / "actor_inputs.jsonl"
        raw = target.read_bytes()
        target.unlink()
        source = Path(self.temp.name) / "source"
        source.write_bytes(raw)
        target.symlink_to(source)
        with self.assertRaisesRegex(ValueError, "unaliased"):
            read_actors(self.root)
        target.unlink()
        os.link(source, target)
        with self.assertRaisesRegex(ValueError, "unaliased"):
            read_actors(self.root)

    def test_unicode_line_separator_and_empty_split(self):
        actors, verifiers, lineages = records()
        actor = replace(actors[0], question="Read\u2028this number.")
        self.write(([actor], verifiers, lineages))
        self.assertEqual(read_actors(self.root), [actor])
        empty = Path(self.temp.name) / "empty"
        write_split(empty, [], [], [])
        self.assertEqual(read_actors(empty), [])
        self.assertEqual(read_verifier(empty), {})
        self.assertEqual(read_lineage(empty), {})


if __name__ == "__main__":
    unittest.main()
