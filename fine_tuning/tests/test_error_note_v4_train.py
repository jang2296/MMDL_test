"""Version-4 training on a tiny random Qwen3-VL (CPU): RL + SFT updates, save, fresh reload, round chaining, the
rl-only arm and the no-update dry run. Orchestration checks only; a tiny model is no evidence of 4B learning."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from fine_tuning.error_note import train_v4 as t4
from fine_tuning.error_note.contract import KIND_NOTE, ROUTE_FULL, ROUTE_LM
from fine_tuning.error_note.order import VERSION, _ids_sha256
from fine_tuning.error_note.protocol import PER_UPDATE_ITEMS_V2, plain_protocol_v4
from fine_tuning.notebook_rl import checkpoint as ckpt
from fine_tuning.tests.test_error_note_train import COMMIT, encoded, tiny_factory

N = 2 * PER_UPDATE_ITEMS_V2


def tiny_encode(row):
    item = encoded(row["row_id"], [int(t) for t in row["prompt"].split()], [int(t) for t in row["target"].split()],
                   row["route"], row["kind"], seed=int(row["problem_id"].split("-")[1]) % 5)
    if not row.get("append_end", True):  # a truncated rollout: no end token
        item["input_ids"], item["target_len"] = item["input_ids"][:-1], item["target_len"] - 1
    return t4._with_row_fields(item, row)


def row(pid, kind, suffix, target, route=ROUTE_LM, prompt="7 8", **extra):
    return {"row_id": f"{pid}:{kind}:{suffix}", "problem_id": pid, "kind": kind, "prompt": prompt, "target": target,
            "route": route, "image_paths": ["x.png"], "target_tokens": len(target.split()), "subject": "s",
            "source": "t", "append_end": True, **extra}


def rl_row(pid, j, target, adv, append_end=True):
    return row(pid, "RL", str(j), target, advantage=adv, group_size=2, append_end=append_end, reward=int(adv > 0))


ROWS = [rl_row("item-0", 0, "9 10", .5), rl_row("item-0", 1, "11 12 13", -.5, append_end=False),
        row("item-1", "A1", "0", "9 10"),
        row("item-2", "A2", "0", "13 14", ROUTE_FULL), row("item-2", KIND_NOTE, "note", "12 13 14", ROUTE_FULL,
                                                           prompt="7 8 8 8 11"),
        rl_row("item-16", 0, "9 10 11", .5), rl_row("item-16", 1, "12", -.5),
        row("item-17", KIND_NOTE, "note", "12 13", prompt="7 8 11")]


class TrainV4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        ids = [f"item-{i}" for i in range(N)]
        self.order = self.root / "order.json"
        self.order.write_text(json.dumps({"version": VERSION, "items": [{"problem_id": i} for i in ids], "count": N,
                                          "ids_sha256": _ids_sha256(ids)}))
        self.rows = self.root / "rows.jsonl"
        self.rows.write_text("".join(json.dumps(r) + "\n" for r in ROWS))

    def slice_rows(self, start, limit):
        """A collect_v4 rows file covers exactly its round's slice of the order."""
        path = self.root / f"rows-{start}-{limit}.jsonl"
        inside = {f"item-{i}" for i in range(start, start + limit)}
        path.write_text("".join(json.dumps(r) + "\n" for r in ROWS if r["problem_id"] in inside))
        return path

    def args(self, command="train", output="run", round_index=1, start=0, limit=N, arm="hybrid", init=None):
        extra = ["--init-run", str(self.root / init)] if init else []
        rows = self.rows if (start, limit) == (0, N) else self.slice_rows(start, limit)
        return t4.build_parser().parse_args([
            command, "--rows", str(rows), "--build-root", str(self.root), "--order", str(self.order),
            "--round", str(round_index), "--start", str(start), "--limit", str(limit), "--arm", arm,
            "--precision", "bf16", "--model-path", str(self.root), "--output", str(self.root / output),
            "--code-commit", COMMIT, *extra])

    def train(self, args):
        return t4.run_train_v4(args, policy_factory=tiny_factory, device="cpu", make_encoder=lambda b, r: tiny_encode)

    def reload(self, run):
        args = t4.build_parser().parse_args(["reload", "--run", str(self.root / run), "--model-path", str(self.root),
                                             "--precision", "bf16"])
        return t4.run_reload_v4(args, policy_factory=tiny_factory, device="cpu")

    def test_round_one_trains_saves_and_reloads_then_round_two_continues(self):
        self.assertEqual(self.train(self.args()), 0)
        result = json.loads((self.root / "run/result.json").read_text())
        self.assertEqual((result["status"], result["updates"], result["total_steps"]), ("COMPLETE", 2, 2))
        self.assertEqual(result["rows_by_kind"], {"RL": 4, "A1": 1, "A2": 1, "B": 2})
        self.assertTrue(all(result["changed_tensors"][g] > 0 for g in ("L", "M", "V")))
        updates = [json.loads(line) for line in (self.root / "run/updates.jsonl").read_text().splitlines()]
        self.assertEqual([u["rl"]["groups"] for u in updates], [1, 1])
        self.assertEqual(updates[0]["rl"]["action_tokens"], 3 + 3)  # "9 10"+end and "11 12 13" without end
        self.assertEqual(set(updates[0]["sft"]), {"A1", "A2", "B"})
        self.assertEqual(updates[0]["sft"]["B"]["weight"], .05)
        self.assertEqual(self.reload("run"), 0)
        self.assertEqual(json.loads((self.root / "run/reload.json").read_text())["status"], "PASS")

        with self.assertRaises(ValueError):
            self.train(self.args(output="r2-missing", round_index=2, start=PER_UPDATE_ITEMS_V2, limit=PER_UPDATE_ITEMS_V2))
        with self.assertRaises(ValueError):
            self.train(self.args(output="r1-init", init="run"))
        second = self.args(output="r2", round_index=2, start=PER_UPDATE_ITEMS_V2, limit=PER_UPDATE_ITEMS_V2, init="run")
        self.assertEqual(self.train(second), 0)
        final = json.loads((self.root / "run/checkpoint-final/metadata.json").read_text())["state_sha256"]
        initial = json.loads((self.root / "r2/initial_state.json").read_text())["sha256"]
        self.assertEqual(initial, {name: final[name] for name in initial})  # round 2 starts from round 1's weights
        recorded = json.loads((self.root / "r2/checkpoint-final/metadata.json").read_text())["protocol"]
        self.assertEqual(recorded["training_contract"]["hashes"]["init"],
                         ckpt.file_hash(self.root / "run/checkpoint-final/metadata.json"))
        self.assertEqual(self.reload("r2"), 0)

    def test_rl_only_arm_keeps_rl_and_a1(self):
        groups = t4.read_groups_v4(self.rows, self.order, 0, N, "rl-only")
        self.assertEqual({r["kind"] for g in groups for r in g}, {"RL", "A1"})
        self.assertEqual(self.train(self.args(arm="rl-only", output="control")), 0)
        protocol = json.loads((self.root / "control/checkpoint-final/metadata.json").read_text())["protocol"]
        self.assertEqual(protocol["training_contract"]["sft_weights"]["B"], 0.0)

    def test_rl_only_keeps_the_hybrid_schedule_where_only_note_rows_were(self):
        rows = [rl_row("item-0", 0, "9 10", .5), rl_row("item-0", 1, "11 12", -.5),
                row("item-16", "A2", "0", "13 14", ROUTE_FULL), row("item-16", KIND_NOTE, "note", "12 13 14", ROUTE_FULL,
                                                                      prompt="7 8 8 8 11")]
        self.rows.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.assertEqual(t4.schedule_v4(self.rows, self.order, 0, N), (2, [True, True]))
        self.assertEqual(self.train(self.args(output="hyb")), 0)
        self.assertEqual(self.train(self.args(output="ctl", arm="rl-only")), 0)
        hybrid = [json.loads(line) for line in (self.root / "hyb/updates.jsonl").read_text().splitlines()]
        control = [json.loads(line) for line in (self.root / "ctl/updates.jsonl").read_text().splitlines()]
        self.assertEqual([u["skipped"] for u in hybrid], [False, False])
        self.assertEqual([(u["skipped"], u.get("scheduler_only")) for u in control], [(False, None), (True, True)])
        self.assertEqual([u["step"] for u in control], [1, 2])
        for name in ("hyb", "ctl"):
            result = json.loads((self.root / name / "result.json").read_text())
            self.assertEqual(result["total_steps"], 2)
            self.assertEqual(json.loads((self.root / name / "checkpoint-final/metadata.json").read_text())["step"], 2)
        self.assertEqual(json.loads((self.root / "ctl/result.json").read_text())["updates"], 1)
        protocols = [json.loads((self.root / n / "checkpoint-final/metadata.json").read_text())["protocol"]["optimizer"]
                     for n in ("hyb", "ctl")]
        self.assertEqual(protocols[0], protocols[1])  # one LR schedule for both arms
        self.assertEqual(self.reload("ctl"), 0)

    def test_kl_reference_is_f0_after_lora_and_merger_change(self):
        """PEFT disable_adapter must switch off both LoRA and the Merger copies, or the KL anchor would drift."""
        protocol = plain_protocol_v4("bf16", 1, arm="hybrid", round_index=1, start=0, limit=N,
                                     contracts={"rows": "a" * 64, "order": "b" * 64, "init": None})
        bundle = tiny_factory(protocol=protocol)
        bundle.model.eval()
        rows = [tiny_encode(row("item-2", "A2", "0", "13 14 15", ROUTE_FULL))]
        with torch.no_grad():
            before_ref = t4._reference(bundle, rows)[0]
            before_cur = t4.batch_logprobs(bundle, rows)[0]
            for group in ("L", "M", "V"):
                for parameter in bundle.params[group]:
                    parameter.add_(torch.randn_like(parameter) * 0.5)
            after_ref = t4._reference(bundle, rows)[0]
            after_cur = t4.batch_logprobs(bundle, rows)[0]
        self.assertTrue(torch.allclose(before_ref, after_ref, atol=1e-6))
        self.assertFalse(torch.allclose(before_cur, after_cur, atol=1e-4))
        self.assertTrue(torch.allclose(before_ref, before_cur, atol=1e-6))  # F0: LoRA B starts at zero, Mergers copied
        self.assertTrue(bundle.route.vision_on)

    def test_zero_advantage_without_kl_has_no_rl_gradient(self):
        protocol = plain_protocol_v4("bf16", 1, arm="hybrid", round_index=1, start=0, limit=N,
                                     contracts={"rows": "a" * 64, "order": "b" * 64, "init": None})
        bundle = tiny_factory(protocol=protocol)
        rows = [tiny_encode(rl_row("item-0", j, "9 10", 0.0)) for j in range(2)]
        contract = protocol["training_contract"] | {"rl": protocol["training_contract"]["rl"] | {"beta": 0.0}}
        t4.backward_v4(bundle, rows, contract, kinds=["RL"])
        norms = [float(p.grad.abs().sum()) for p in bundle.model.parameters() if p.grad is not None]
        self.assertTrue(norms and max(norms) == 0.0)

    def test_dry_run_checks_every_kind_without_changing_weights(self):
        args = self.args(command="dry-run", output="dry", limit=3)
        code = t4.run_dry_run_v4(args, policy_factory=tiny_factory, device="cpu", make_encoder=lambda b, r: tiny_encode)
        report = json.loads((self.root / "dry/dry_run.json").read_text())
        self.assertEqual((code, report["status"]), (0, "PASS"))
        self.assertEqual(set(report["components"]), {"RL", "A1", "A2", "B"})
        self.assertEqual(report["components"]["A2"]["expected"]["V"], "finite_positive")
        self.assertEqual(report["components"]["A1"]["expected"]["V"], "none")
        self.assertEqual(report["paired_a2_b_problem_ids"], ["item-2"])
        self.assertTrue(report["weights_unchanged"] and report["gradients_cleared"])
        self.assertEqual((report["optimizer_steps"], report["weights_saved"]), (0, False))
        self.assertFalse((self.root / "dry/checkpoint-final").exists())

    def test_invalid_rows_are_refused(self):
        for bad in ({k: v for k, v in ROWS[0].items() if k != "advantage"}, ROWS[2] | {"problem_id": "nowhere"},
                    ROWS[2] | {"kind": "C"}):
            self.rows.write_text(json.dumps(bad) + "\n")
            with self.assertRaises(ValueError):
                t4.read_groups_v4(self.rows, self.order, 0, N, "hybrid")
        self.rows.write_text(json.dumps(ROWS[1]) + "\n" + json.dumps(ROWS[1]) + "\n")
        with self.assertRaises(ValueError):
            t4.read_groups_v4(self.rows, self.order, 0, N, "hybrid")


if __name__ == "__main__":
    unittest.main()
