"""Small synthetic byte-recovery boundary checks; no Pod or actual benchmark access."""

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.bottleneck_diagnostic import recovery as r


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "analysis_exports"
        self.root.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(r, "PRIVATE_ROOT", self.root).start()
        patch.object(r, "check_storage", return_value={"sufficient": True}).start()
        self.source = self.root / "closed.jsonl"
        self.source.write_text('{"test":1}\n')
        self.files = self.root / "allowlist.json"
        self.files.write_text(json.dumps({"results/closed.jsonl": str(self.source)}))
        self.archive = self.root / "bytes.tar.gz"

    def packed(self):
        return r.pack(self.files, self.archive, r.POD_ID, assert_closed=True)

    def test_roundtrip_and_no_overwrite(self):
        packed = self.packed()
        output = self.root / "received"
        received = r.receive(self.archive, packed["archive_sha256"], output)
        self.assertEqual((output / "results/closed.jsonl").read_bytes(), self.source.read_bytes())
        self.assertEqual(received["status"], "PRIVATE_BYTES_VERIFIED_NOT_EXPERIMENT_VERIFIED")
        self.assertFalse(received["cleanup_authorized"])
        self.assertEqual(len(received["files"]), 2)
        with self.assertRaises(FileExistsError):
            self.packed()
        with self.assertRaises(FileExistsError):
            r.receive(self.archive, packed["archive_sha256"], output)
        with self.assertRaises(ValueError):
            r.receive(self.archive, "0" * 64, self.root / "wronghash")
        self.assertFalse((self.root / "wronghash").exists())
        with self.assertRaises(ValueError):
            r.receive(self.archive, packed["archive_sha256"], self.root.parent / "outside-private")

    def test_allowlist_symlinks_paths_and_closed_assertion(self):
        with self.assertRaises(ValueError):
            r.pack(self.files, self.archive, r.POD_ID, assert_closed=False)
        link = self.root / "link.jsonl"
        link.symlink_to(self.source)
        for label, path in [("results/link.jsonl", str(link)), ("../escape.json", str(self.source)),
                            ("C:\\escape.json", str(self.source)), ("weights.safetensors", str(self.source)),
                            ("secret.json", str(self.source))]:
            self.files.write_text(json.dumps({label: path}))
            with self.assertRaises(ValueError):
                self.packed()
        self.files.write_text('{"a.json":"a","a.json":"b"}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.packed()
        self.assertFalse(self.archive.exists())
        outside = self.root.parent / "outside.json"
        outside.write_text('{"must_not_be_read":true}')
        with (patch.object(r, "sha", side_effect=AssertionError("Unexpected contents read")),
              patch.object(Path, "open", side_effect=AssertionError("Unexpected file open"))):
            with self.assertRaisesRegex(ValueError, "Source must stay"):
                r.pack(outside, self.archive, r.POD_ID, assert_closed=True)
        self.files.write_text(json.dumps({"results/outside.json": str(outside)}))
        real_sha, real_open, read_paths = r.sha, Path.open, []

        def scoped_sha(path):
            read_paths.append(Path(path))
            self.assertNotEqual(Path(path), outside)
            return real_sha(path)

        def scoped_open(path, *args, **kwargs):
            self.assertNotEqual(Path(path), outside)
            return real_open(path, *args, **kwargs)

        with patch.object(r, "sha", scoped_sha), patch.object(Path, "open", scoped_open):
            with self.assertRaisesRegex(ValueError, "Source must stay"):
                self.packed()
        self.assertEqual(read_paths, [self.files])
        self.assertFalse(self.archive.exists())

    def test_unsafe_duplicate_or_nonregular_archive_before_extract(self):
        packed = self.packed()
        with tarfile.open(self.archive) as archive:
            members = [(member, archive.extractfile(member).read()) for member in archive.getmembers()]
        for kind in ("escape", "windows", "duplicate", "symlink", "badbytes"):
            malicious = self.root / f"{kind}.tar"
            with tarfile.open(malicious, "w") as archive:
                for index, (member, data) in enumerate(members):
                    info = tarfile.TarInfo(member.name)
                    info.size = len(data)
                    if index == 1 and kind == "badbytes":
                        data = b"x" * len(data)
                    archive.addfile(info, io.BytesIO(data))
                extra = tarfile.TarInfo({"escape": "../escape.json", "windows": "C:\\escape.json",
                                        "duplicate": members[1][0].name}.get(kind, "extra.json"))
                if kind == "symlink":
                    extra.type, extra.linkname = tarfile.SYMTYPE, "../outside"
                if kind != "badbytes":
                    archive.addfile(extra, io.BytesIO(b""))
            output = self.root / f"received-{kind}"
            with self.assertRaises(ValueError):
                r.receive(malicious, r.sha(malicious), output)
            self.assertFalse(output.exists())
        self.assertEqual(packed["file_count"], 1)

    def test_source_mutation_preserves_failed_output_and_space_gate(self):
        real_addfile = tarfile.TarFile.addfile

        def mutate(archive, info, fileobj=None):
            result = real_addfile(archive, info, fileobj)
            if info.name == "results/closed.jsonl":
                self.source.write_text('{"test":2}\n')
            return result

        with patch.object(tarfile.TarFile, "addfile", mutate):
            with self.assertRaisesRegex(ValueError, "mutated"):
                self.packed()
        self.assertTrue(self.archive.exists())
        self.assertFalse(Path(str(self.archive) + ".receipt.json").exists())
        with patch.object(r, "check_storage", return_value={"sufficient": False}):
            with self.assertRaises(RuntimeError):
                r.pack(self.files, self.root / "nospace.tar", r.POD_ID, assert_closed=True)
        self.assertFalse((self.root / "nospace.tar").exists())


if __name__ == "__main__":
    unittest.main()
