"""One version-4 round end to end on CPU: fake generation and grading -> rows -> tiny-model training -> fresh reload.

It checks that each stage accepts the previous stage's files (the contract between collect_v4 and train_v4).
Fake texts and a tiny random model: orchestration evidence only, never evidence of learning.
"""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from fine_tuning.error_note import collect_v4 as c4
from fine_tuning.error_note import train_v4 as t4
from fine_tuning.error_note import v4
from fine_tuning.error_note.collect import StreamStats, run_note
from fine_tuning.error_note.order import VERSION, _ids_sha256
from fine_tuning.tests.test_error_note_train import COMMIT, encoded, tiny_factory
from fine_tuning.tests.test_error_note_v4_collect import GOLD, FakeStream, actors, grader, items, load, words

KW = dict(S=2, B=16384, engine="vllm", gpu_probe=lambda: (None, None, None))
NOTE = ("What the image shows: The chart has 3 bars on the left side. The axis starts at zero.\n"
        "Image reading: MISREAD\nCause: IMAGE\nWhat went wrong: I read the second bar as the tallest one.\n"
        "Lesson: Compare every bar against the axis ticks before ranking them. " + words(30))


def word_ids(text, cap=12):
    """Tiny-vocabulary ids for words: stable, inside the tiny model's 8..27 range, at most `cap` of them."""
    return [8 + sum(word.encode()) % 20 for word in text.split()][:cap] or [8]


def word_ids_all(text):
    return [8 + sum(word.encode()) % 20 for word in text.split()]


def word_encode(row):
    item = encoded(row["row_id"], word_ids(row["prompt"]), word_ids(row["target"]), row["route"], row["kind"],
                   seed=int(row["problem_id"][1:]) % 5)
    if not row.get("append_end", True):
        item["input_ids"], item["target_len"] = item["input_ids"][:-1], item["target_len"] - 1
    return t4._with_row_fields(item, row)


def answers(request):
    pid, j = request["item"], request["j"]
    if request.get("lesson"):
        return words(40) + (" RIGHT" if j == 0 else " wrong"), "stop"
    if pid == "p0" or (pid == "p1" and j in (0, 3)):
        return words(40) + " RIGHT", "stop"
    return words(40) + " wrong", "stop"


class PipelineV4(unittest.TestCase):
    def test_one_round(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp), Path(tmp) / "collect-r1"
            out.mkdir()
            ids = [item["problem_id"] for item in items()]
            order = root / "order.json"
            order.write_text(json.dumps({"version": VERSION, "items": items(), "count": 4, "ids_sha256": _ids_sha256(ids)}))
            c4.run_rollout(items(), actors(), load, FakeStream(answers), StreamStats(), grader([]), out, round_index=1, **KW)
            chosen = c4.note_inputs(c4.rollouts_of(out), ids)
            wrong = [item for item in items() if item["problem_id"] in chosen]
            self.assertEqual([item["problem_id"] for item in wrong], ["p2", "p3"])
            run_note(wrong, actors(), chosen, GOLD, load, FakeStream(lambda r: (NOTE, "stop")), StreamStats(),
                     grader([]), out, **KW, version="v2")
            c4.write_gates(out, [item["problem_id"] for item in wrong], GOLD, actors(), 1)
            c4.run_selfcheck(items(), actors(), c4.read_gates(out), load, FakeStream(lambda r: ("TRUE.", "stop")),
                             StreamStats(), out, round_index=1, **KW)
            c4.run_resolve_v4(items(), actors(), c4.read_gates(out), c4.read_selfcheck(out), load, FakeStream(answers),
                              StreamStats(), grader([]), out, round_index=1, control=True, **KW)
            rows, drops, summary = c4.collect_rows_v4(c4.make_records_v4(items(), actors(), out), word_ids_all)
            self.assertEqual(summary["rows_by_kind"], {"A1": 1, "RL": 4, "A2": 2, "B": 2})
            self.assertEqual(summary["rows_by_route"], {"LM": 5, "FULL": 4})  # MISREAD + self-consistent notes
            rows_path = out / "rows.jsonl"
            rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

            args = t4.build_parser().parse_args([
                "train", "--rows", str(rows_path), "--build-root", tmp, "--order", str(order), "--round", "1",
                "--start", "0", "--limit", "4", "--arm", "hybrid", "--precision", "bf16", "--model-path", tmp,
                "--output", str(root / "train-r1"), "--code-commit", COMMIT])
            code = t4.run_train_v4(args, policy_factory=tiny_factory, device="cpu", make_encoder=lambda b, r: word_encode)
            result = json.loads((root / "train-r1/result.json").read_text())
            self.assertEqual((code, result["status"], result["updates"]), (0, "COMPLETE", 1))
            self.assertEqual(result["rows_by_kind"], summary["rows_by_kind"])
            update = json.loads((root / "train-r1/updates.jsonl").read_text().splitlines()[0])
            self.assertEqual(update["rl"]["groups"], 1)
            self.assertTrue(all(update["grad_norms"][g] for g in ("L", "M", "V")))
            reload = t4.build_parser().parse_args(["reload", "--run", str(root / "train-r1"), "--model-path", tmp,
                                                   "--precision", "bf16"])
            self.assertEqual(t4.run_reload_v4(reload, policy_factory=tiny_factory, device="cpu"), 0)
            self.assertNotIn(GOLD["p2"] + ". ", (out / c4.GATES_FILE).read_text())
            self.assertEqual(v4.VERSION, json.loads((out / c4.SELFCHECK_FILE).read_text())["version"])


if __name__ == "__main__":
    unittest.main()
