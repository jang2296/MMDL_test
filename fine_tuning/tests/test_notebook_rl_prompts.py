"""Prompt contract checks with authored inputs and the unchanged existing P0 files."""
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fine_tuning.notebook_rl import prompts
from fine_tuning.notebook_rl.resources import resource_path
from fine_tuning.notebook_rl.schema import ActorItem

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "fine_tuning/notebook_rl/prompts_v1.yaml"


def actor():
    return ActorItem("fixture", "authored", "chart_bar", "obs", "What is {x}?", (), "int",
                     ("image.png",), ("a" * 64,))


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.p = prompts.load_prompts(PROMPTS)

    def test_load_hash(self):
        self.assertEqual(self.p["sha256"], hashlib.sha256(PROMPTS.read_bytes()).hexdigest())

    def test_exact_p0_bytes(self):
        question = "What is {x} if {P0_BODY} stays literal?"
        for choices, file in (((), "mmmu_open_v1.txt"), (("A {x}", "B"), "mmmu_mcq_v1.txt")):
            template = resource_path("prompts/" + file).read_text()
            expected = template.format(hint_prefix="", question=question,
                                       options="".join(f"{chr(65 + i)}. {v}\n" for i, v in enumerate(choices)))
            self.assertEqual(prompts.p0_body(question, choices).encode(), expected.encode())

    def test_placeholder_payload_is_not_recursively_expanded(self):
        a = replace(actor(), question="Literal {OBSERVATION_INSTRUCTION} and {P0_BODY}")
        text = prompts.turn1_text(a, self.p)
        self.assertIn(a.question, text)
        self.assertIn("bar[<series>][<label>]", text)

    def test_modes_and_family(self):
        for mode in ("obs", "plain"):
            text = prompts.turn1_text(replace(actor(), mode=mode), self.p)
            self.assertEqual("Observed facts:" in text, mode == "obs")
        with self.assertRaises(ValueError):
            prompts.turn1_text(replace(actor(), family="unknown"), self.p)

    def test_retry_incorrect_only_own_answer(self):
        text = prompts.retry_text("INCORRECT", "7", "obs", "chart_bar", self.p)
        self.assertIn('previous answer "7": INCORRECT', text)
        self.assertIn("correct answer is not provided", text)
        self.assertIn("Error check:", text)
        self.assertIn("Observed facts:", text)
        self.assertNotIn("Question:", text)
        with self.assertRaises(ValueError):
            prompts.retry_text("INCORRECT", None, "plain", "chart_bar", self.p)

    def test_invalid_is_not_incorrect(self):
        for status in ("INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"):
            text = prompts.retry_text(status, None, "plain", "chart_bar", self.p)
            self.assertIn("correctness is unknown", text)
            self.assertNotIn("INCORRECT", text)
            self.assertNotIn("Observed facts:", text)
        for status in ("CORRECT", "UNVERIFIABLE", "CONTEXT_BLOCKED", "SYSTEM_ERROR"):
            with self.assertRaises(ValueError):
                prompts.retry_text(status, None, "plain", "chart_bar", self.p)

    def test_diagnosis(self):
        self.assertIn("attempt 2 above was verified CORRECT", prompts.diagnose_text(2, self.p))
        for turn in (1, 4, True):
            with self.assertRaises(ValueError):
                prompts.diagnose_text(turn, self.p)

    def test_diagnostic_inputs_preserved(self):
        self.assertIn("bar[X] = 7", prompts.oracle_text(actor(), ("bar[X] = 7",), self.p))
        self.assertIn("Notes:\nmy {P0_BODY} note", prompts.probe_text("my {P0_BODY} note", actor(), self.p))
        with self.assertRaises(ValueError):
            prompts.probe_text("notes", replace(actor(), choices=("a", "b"), answer_format="mcq"), self.p)

    def test_image_order_and_count(self):
        self.assertEqual(prompts.user_message("text", 2), {"role": "user", "content": [
            {"type": "image"}, {"type": "image"}, {"type": "text", "text": "text"}]})
        self.assertEqual(prompts.user_message("probe", 0)["content"], [{"type": "text", "text": "probe"}])
        with self.assertRaises(ValueError):
            prompts.user_message("text", 3)

    def test_duplicate_or_malformed_template_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.yaml"
            path.write_text(PROMPTS.read_text() + "\nturn1_plain: replaced\n")
            with self.assertRaises(ValueError):
                prompts.load_prompts(path)
            path.write_text(PROMPTS.read_text().replace("{PREV_ANSWER}", "{GOLD}"))
            with self.assertRaises(ValueError):
                prompts.load_prompts(path)


if __name__ == "__main__":
    unittest.main()
