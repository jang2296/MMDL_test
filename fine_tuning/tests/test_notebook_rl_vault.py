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
from unittest.mock import patch

import yaml

from fine_tuning.notebook_rl import vault
from fine_tuning.notebook_rl.schema import ActorItem, Lineage, VerifierItem, to_json
from fine_tuning.notebook_rl.vault import (
    ACTOR_KEYS, read_actors, read_lineage, read_verifier, write_split,
)
from fine_tuning.notebook_rl.verifier_service import VerifierService

LEGACY_CODE = {
    "vault.py": "45b2e61e8eb98fe0e392a201dc9fad8382c61546b4312d40db1c81cbc243d1ce",
    "schema.py": "0605c5f1a1c0f00b08e2b0e5f7dee83e240573e7bdd1ad3b071964c173b44996",
    "guards.py": "e37f158845012e11d22e46d8965d09e422b0e59103dbb68345a857426fe5638a",
}


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

    def legacy_manifest(self):
        data = self.write()
        data["code_sha256"] = dict(LEGACY_CODE)
        (self.root / "manifest.yaml").write_text(json.dumps(data), encoding="utf-8")
        return data

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

    def test_pinned_legacy_vault_loaders_and_fresh_service_preserve_payloads(self):
        self.legacy_manifest()
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        actors, verifiers, lineages = records()
        self.assertEqual(read_actors(self.root), actors)
        self.assertEqual(read_verifier(self.root), {verifiers[0].problem_id: verifiers[0]})
        self.assertEqual(read_lineage(self.root), {lineages[0].problem_id: lineages[0]})
        with VerifierService(self.root, forbidden_roots=[]) as service:
            self.assertNotEqual(service.metadata["pid"], os.getpid())
            self.assertTrue(service.metadata["python_guards_installed"])
            self.assertEqual(service.metadata["vault_manifest_sha256"],
                             hashlib.sha256(before["manifest.yaml"]).hexdigest())
            result = service.grade(actors[0].problem_id, "Final answer: 53", "stop", "plain")
            self.assertEqual(result.status, "CORRECT")
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_legacy_source_exception_requires_exact_complete_old_triple(self):
        data = self.legacy_manifest()
        current = vault._code_hashes()
        candidates = [{}, LEGACY_CODE | {"unknown.py": "0" * 64}]
        for name in LEGACY_CODE:
            candidates.extend((LEGACY_CODE | {name: "0" * 64},
                               {key: value for key, value in LEGACY_CODE.items() if key != name}))
            if current[name] != LEGACY_CODE[name]:
                candidates.append(LEGACY_CODE | {name: current[name]})
        for code in candidates:
            (self.root / "manifest.yaml").write_text(json.dumps(data | {"code_sha256": code}))
            for loader in (read_actors, read_verifier, read_lineage):
                with self.subTest(code=code, loader=loader.__name__), self.assertRaisesRegex(
                        ValueError, "code or protocol hash mismatch"):
                    loader(self.root)

    def test_legacy_source_exception_rejects_changed_current_schema_or_guards(self):
        self.legacy_manifest()
        current = vault._code_hashes()
        for name in ("schema.py", "guards.py"):
            for digest in ("0" * 64, LEGACY_CODE[name]):
                if digest == current[name]:
                    continue
                with self.subTest(name=name, digest=digest), patch.object(
                        vault, "_code_hashes", return_value=current | {name: digest}):
                    for loader in (read_actors, read_verifier, read_lineage):
                        with self.assertRaisesRegex(ValueError, "code or protocol hash mismatch"):
                            loader(self.root)

    def test_legacy_reader_keeps_protocol_payload_counts_and_lineage_checks(self):
        data = self.legacy_manifest()
        path = self.root / "manifest.yaml"
        path.write_text(json.dumps(data | {"protocol_sha256": "0" * 64}))
        for loader in (read_actors, read_verifier, read_lineage):
            with self.assertRaisesRegex(ValueError, "code or protocol hash mismatch"):
                loader(self.root)
        path.write_text(json.dumps(data))
        for name, loader in (("actor_inputs.jsonl", read_actors),
                             ("verifier_private.jsonl", read_verifier), ("lineage.jsonl", read_lineage)):
            payload = self.root / name
            raw = payload.read_bytes()
            payload.write_bytes(raw + b" ")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "payload hash mismatch"):
                loader(self.root)
            payload.write_bytes(raw)
        bad_counts = dict(data["files"])
        bad_counts["actor_inputs.jsonl"] = bad_counts["actor_inputs.jsonl"] | {"rows": 0}
        path.write_text(json.dumps(data | {"files": bad_counts}))
        with self.assertRaisesRegex(ValueError, "row count mismatch"):
            read_actors(self.root)
        path.write_text(json.dumps(data))
        self.rehash("lineage.jsonl", to_json(replace(records()[2][0], model_touched=True)).encode() + b"\n")
        with self.assertRaisesRegex(ValueError, "forbidden benchmark or model lineage"):
            read_lineage(self.root)


if __name__ == "__main__":
    unittest.main()
