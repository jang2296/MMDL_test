"""Unit tests for the v3 quota order: order.build_v3, read_order's v3 checks and the `quota` CLI."""
from collections import Counter
from fractions import Fraction
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from fine_tuning.error_note import order
from fine_tuning.error_note.protocol import single_blocks

PILOT = order.PILOT_COUNT
QUOTAS = {"a": 160, "b": 100, "c": 60, "d": 32}
HEAD = 352
QUOTAS_ALIGNED = {"a": 150, "b": 100, "c": 50, "d": 32}
HEAD_ALIGNED = 332
REAL = Path("/home/jang2/mmdl-artifacts/self-note-single-20261105/v10-weak-v1/lot-v10/order.json")
REAL_QUOTAS = {"hf-geometry3k": 300, "hf-scienceqa": 400, "hf-ssmr": 200, "hf-chartqa": 150,
               "hf-mapqa": 100, "hf-slake": 50, "hf-gvlqa": 100, "hf-ptbxl": 100, "hf-nctcrc": 80,
               "hf-ai2d": 50, "hf-clevr": 20, "hf-plantvillage": 30, "hf-chebi-role": 20}


def make_base():
    available = {"a": 240, "b": 180, "c": 120, "d": 60}
    ids = [f"{source}-{n:03d}" for source, count in available.items() for n in range(count)]
    random.Random(1).shuffle(ids)
    items = [{"problem_id": pid, "source": pid[0],
              "cluster": "C1" if pid[0] in ("a", "b") else "C2"} for pid in ids]
    return {
        "version": "error-note-order-v2",
        "master_seed": 3407,
        "count": len(items),
        "items": items,
        "ids_sha256": order._ids_sha256(ids),
        "selection": {
            "pilot_ids_sha256": order._ids_sha256(ids[:PILOT]),
            "cluster_pilot_counts": dict(Counter(item["cluster"] for item in items[:PILOT])),
            "source_caps": dict(available),
        },
    }


def ids_of(result):
    return [item["problem_id"] for item in result["items"]]


def assert_block_mix(test, ids, source_of, quotas, head):
    """Full blocks inside the head: each source within +-1 of its proportional share."""
    checked = 0
    start = 0
    for block in single_blocks(ids, pilot_count=PILOT):
        if start + len(block) <= head:
            counts = Counter(source_of[pid] for pid in block)
            for source in set(quotas) | set(counts):
                target = Fraction(len(block) * quotas.get(source, 0), head)
                got = counts.get(source, 0)
                test.assertLessEqual(abs(got - target), 1, msg=f"{start} {source} {got} {target}")
            checked += 1
        start += len(block)
    test.assertGreaterEqual(checked, 20)
    return checked


class OrderV3Tests(unittest.TestCase):
    def setUp(self):
        self.base = make_base()
        self.base_ids = [item["problem_id"] for item in self.base["items"]]
        self.source_of = {item["problem_id"]: item["source"] for item in self.base["items"]}

    def build(self, quotas=QUOTAS, head=HEAD):
        return order.build_v3(self.base, "0" * 64, quotas, head)

    def test_deterministic(self):
        first = json.dumps(self.build(), indent=1)
        second = json.dumps(self.build(), indent=1)
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base_path, quotas_path = tmp / "base.json", tmp / "quotas.json"
            base_path.write_text(json.dumps(self.base, indent=1))
            quotas_path.write_text(json.dumps(QUOTAS))
            argv = ["quota", "--base", str(base_path), "--quotas", str(quotas_path),
                    "--head-count", str(HEAD), "--output"]
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(order.main(argv + [str(tmp / "out1.json")]), 0)
            self.assertIn("count=600 ids_sha256=", stdout.getvalue())
            self.assertIn("pilot_ids_sha256=", stdout.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(order.main(argv + [str(tmp / "out2.json")]), 0)
            self.assertEqual((tmp / "out1.json").read_bytes(), (tmp / "out2.json").read_bytes())
            with self.assertRaises(SystemExit):
                order.main(argv + [str(tmp / "out1.json")])

    def test_same_ids(self):
        result = self.build()
        new_ids = ids_of(result)
        self.assertEqual(sorted(new_ids), sorted(self.base_ids))
        self.assertEqual(result["count"], 600)
        self.assertEqual(result["ids_sha256"], order._ids_sha256(new_ids))
        self.assertEqual(result["version"], "error-note-order-v3")

    def test_head_quotas(self):
        head_ids = ids_of(self.build())[:HEAD]
        self.assertEqual(Counter(self.source_of[pid] for pid in head_ids), Counter(QUOTAS))
        for source, quota in QUOTAS.items():
            with self.subTest(source=source):
                got = [pid for pid in head_ids if self.source_of[pid] == source]
                expected = [pid for pid in self.base_ids if self.source_of[pid] == source][:quota]
                self.assertEqual(got, expected)

    def test_block_mix(self):
        cases = (("unaligned", QUOTAS, HEAD), ("aligned", QUOTAS_ALIGNED, HEAD_ALIGNED))
        for label, quotas, head in cases:
            with self.subTest(label=label):
                new_ids = ids_of(self.build(quotas, head))
                assert_block_mix(self, new_ids, self.source_of, quotas, head)

    def test_tail_keeps_base_order(self):
        new_ids = ids_of(self.build())
        head_set = set(new_ids[:HEAD])
        self.assertEqual(new_ids[HEAD:], [pid for pid in self.base_ids if pid not in head_set])

    def test_metadata_and_read_order(self):
        result = self.build()
        new_ids = ids_of(result)
        selection = result["selection"]
        self.assertEqual(selection["pilot_ids_sha256"], order._ids_sha256(new_ids[:PILOT]))
        self.assertEqual(selection["cluster_pilot_counts"],
                         dict(Counter(item["cluster"] for item in result["items"][:PILOT])))
        self.assertEqual(selection["source_caps"], self.base["selection"]["source_caps"])
        expected_base = {"version": "error-note-order-v2", "sha256": "0" * 64,
                         "ids_sha256": self.base["ids_sha256"]}
        self.assertEqual(result["base_order"], expected_base)
        self.assertEqual(result["quotas"], QUOTAS)
        self.assertEqual(result["head_count"], HEAD)
        self.assertIsInstance(result["rule"], str)
        self.assertTrue(result["rule"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "order.json"
            path.write_text(json.dumps(result, indent=1))
            self.assertEqual(order.read_order(path), result)

            def tampered(mutate):
                bad = json.loads(json.dumps(result))
                mutate(bad)
                path.write_text(json.dumps(bad, indent=1))
                return path

            def shift_quotas(d):
                d["quotas"]["a"] += 1
                d["quotas"]["b"] -= 1

            def wrong_pilot_digest(d):
                d["selection"]["pilot_ids_sha256"] = "0" * 64

            def swap_items(d):
                items = d["items"]
                items[0], items[400] = items[400], items[0]
                d["ids_sha256"] = order._ids_sha256([item["problem_id"] for item in items])

            for label, mutate in (("quotas", shift_quotas), ("pilot digest", wrong_pilot_digest),
                                  ("swapped items", swap_items)):
                with self.subTest(label=label):
                    with self.assertRaises(ValueError):
                        order.read_order(tampered(mutate))

    def test_refusals(self):
        cases = [
            ("sum != head", dict(QUOTAS, a=161), HEAD),
            ("quota > available", {"a": 131, "b": 100, "c": 60, "d": 61}, HEAD),
            ("unknown source", {"a": 160, "b": 100, "c": 60, "d": 31, "z": 1}, HEAD),
            ("zero quota", {"a": 160, "b": 100, "c": 92, "d": 0}, HEAD),
            ("negative quota", {"a": 200, "b": 100, "c": 60, "d": -8}, HEAD),
            ("float quota", {"a": 160.5, "b": 100, "c": 60, "d": 31.5}, HEAD),
            ("bool quota", {"a": 160, "b": 100, "c": 91, "d": True}, HEAD),
            ("head_count 0", QUOTAS, 0),
            ("head_count 601", QUOTAS, 601),
        ]
        for label, quotas, head in cases:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self.build(quotas, head)

    def test_v1_v2_still_read(self):
        ids = ["p0", "p1"]
        items = [{"problem_id": pid, "subject": "S", "source": "x"} for pid in ids]
        with tempfile.TemporaryDirectory() as tmp:
            for version in ("error-note-order-v1", "error-note-order-v2"):
                with self.subTest(version=version):
                    data = {"version": version, "master_seed": 3407, "count": 2, "items": items,
                            "ids_sha256": order._ids_sha256(ids)}
                    path = Path(tmp) / f"{version}.json"
                    path.write_text(json.dumps(data))
                    self.assertEqual(order.read_order(path), data)
        self.assertEqual(order.PILOT_COUNT, 300)


@unittest.skipUnless(REAL.exists(), f"real v10 order not found: {REAL}")
class RealOrderTests(unittest.TestCase):
    def test_real_v2_order_builds_v3_quota_head(self):
        base = order.read_order(REAL)
        self.assertEqual(base["version"], "error-note-order-v2")
        source_of = {item["problem_id"]: item["source"] for item in base["items"]}
        base_ids = [item["problem_id"] for item in base["items"]]
        result = order.build_v3(base, hashlib.sha256(REAL.read_bytes()).hexdigest(),
                                REAL_QUOTAS, 1600)
        new_ids = ids_of(result)
        self.assertEqual(sorted(new_ids), sorted(base_ids))
        self.assertEqual(Counter(source_of[pid] for pid in new_ids[:1600]), Counter(REAL_QUOTAS))
        assert_block_mix(self, new_ids, source_of, REAL_QUOTAS, 1600)


if __name__ == "__main__":
    unittest.main()
