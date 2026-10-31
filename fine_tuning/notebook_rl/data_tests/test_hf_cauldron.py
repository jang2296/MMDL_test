"""Authored synthetic parquet fixtures only; no real Cauldron rows or upstream images."""
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from fine_tuning.notebook_rl.adapters import hf_cauldron as hc
from fine_tuning.notebook_rl.adapters import hf_common as hf

SCHEMA = pa.schema([
    ("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
    ("texts", pa.list_(pa.struct([("user", pa.string()), ("assistant", pa.string()), ("source", pa.string())]))),
])
_counter = [0]


def png() -> bytes:
    _counter[0] += 1
    n = _counter[0]
    out = io.BytesIO()
    Image.new("RGB", (6, 5), (n % 256, (n // 256) % 256, 7)).save(out, format="PNG")
    return out.getvalue()


def row(turns, n_images=1):
    return {"images": [{"bytes": png(), "path": None} for _ in range(n_images)],
            "texts": [{"user": u, "assistant": a, "source": "s"} for u, a in turns]}


def write(repo: Path, config: str, rows, name="train-00000-of-00001-x.parquet", group=2):
    (repo / config).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), repo / config / name, row_group_size=group)


def ai2d_user(question, *choices):
    lines = "\n".join(f"{'ABCDEFGHIJ'[i]}. {c}" for i, c in enumerate(choices))
    return f"Question: {question}\nChoices:\n{lines}\nAnswer with the letter."


class CauldronCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)


class SpecTests(unittest.TestCase):
    def test_specs_and_iterators(self):
        self.assertEqual(hc.REVISION, "847a98a779b1652d65111daf20c972dfcd333605")
        self.assertEqual(set(hc.SPECS), set(hc.ITERATORS))
        want = {"hf-ai2d": ("CC-BY-SA-4.0", "HUMAN", "diagram"), "hf-mapqa": ("CC-BY-SA-4.0", "PROGRAM", "map"),
                "hf-finqa": ("MIT", "HUMAN", "table"), "hf-tatqa": ("CC-BY-4.0", "HUMAN", "table"),
                "hf-plotqa": ("CC-BY-4.0", "PROGRAM", "chart_line")}
        self.assertEqual({k: (s.license, s.label_provenance, s.family) for k, s in hc.SPECS.items()}, want)
        for source, spec in hc.SPECS.items():
            self.assertEqual((spec.source, spec.repo, spec.revision), (source, "HuggingFaceM4/the_cauldron", hc.REVISION))
            self.assertTrue(spec.upstream_origin.endswith("via the_cauldron"))


class Ai2dTests(CauldronCase):
    def test_conversion_subjects_and_one_per_row(self):
        write(self.repo, "ai2d", [
            row([(ai2d_user("Bad letter", "a", "b"), "Answer: D"),                           # beyond choices: skipped
                 (ai2d_user("Which season is cold?", "winter", "summer"), "Answer: A"),
                 (ai2d_user("Second good turn", "x", "y"), "Answer: B")]),                   # same row: not emitted
            row([(ai2d_user("Zero images", "a", "b"), "Answer: A")], n_images=0),
            row([(ai2d_user("Three images", "a", "b"), "Answer: A")], n_images=3),
            row([(ai2d_user("Which circuit is closed?", "one", "two", "three"), "Answer: C")], n_images=2),
            row([(ai2d_user("What eats grass?", "rabbit", "rock"), "Answer: A"),             # 'rock' -> Geography
                 ]),
            row([(ai2d_user("What eats grass?", "rabbit", "wolf"), "Answer: B")]),
            row([(ai2d_user("Dup", "same", "same"), "Answer: A")]),                          # duplicate options: skipped
            row([(ai2d_user("Fmt", "a", "b"), "Answer: A, because")]),                       # not an exact answer line
        ])
        got = list(hc.iter_ai2d(self.repo))
        stem = "train-00000-of-00001-x"
        self.assertEqual([c.native_id for c in got],
                         [f"ai2d/{stem}:0:1", f"ai2d/{stem}:3:0", f"ai2d/{stem}:4:0", f"ai2d/{stem}:5:0"])
        first, circuit, rock, wolf = got
        self.assertEqual((first.subject, first.question, first.choices, first.answer_type, first.gold),
                         ("Geography", "Which season is cold?", ("winter", "summer"), "mcq", "A"))
        self.assertEqual((circuit.subject, circuit.gold, len(circuit.images)), ("Physics", "C", 2))
        self.assertEqual((rock.subject, rock.gold), ("Geography", "A"))
        self.assertEqual((wolf.subject, wolf.gold, wolf.accept), ("Biology", "B", {}))
        for c in got:
            self.assertNotIn("Choices:", c.question)
            self.assertNotIn("Answer", c.question)
            self.assertEqual(c.family_key, hashlib.sha256(c.images[0]).hexdigest())
            self.assertEqual(c.source, "hf-ai2d")


    def test_subject_keywords_match_at_word_starts(self):
        write(self.repo, "ai2d", [row([(ai2d_user("Which player is shown?", "left", "right"), "Answer: A")]),     # 'layer' mid-word
                                  row([(ai2d_user("Which layer is shown?", "top", "bottom"), "Answer: A")]),
                                  row([(ai2d_user("What do we reinforce?", "left", "right"), "Answer: A")]),      # 'force' mid-word
                                  row([(ai2d_user("What is a force?", "left", "right"), "Answer: A")])])
        self.assertEqual([c.subject for c in hc.iter_ai2d(self.repo)], ["Biology", "Geography", "Biology", "Physics"])


class MapqaTests(CauldronCase):
    def test_yes_no_short_text_and_skips(self):
        rows = [
            row([("Does A have the highest value?\nShort answer required.", "Yes.")]),
            row([("Does B have the highest value?\nQuick response, please.", "No.")]),
            row([("What is the value of Florida?\nConcise answer only.", "15-17.")]),
            row([("Name the states?\nBe succinct.", "Maine, Utah.")]),                         # list: skipped
            row([("Name it?\nBe succinct.", "An extremely long answer string.")]),             # > 20 chars
            row([("Bad then good?\nBe succinct.", "A, B."), ("Does C beat D?\nBe succinct.", "No.")]),
        ] + [row([(f"Does S{i} have more?\nBe succinct.", "Yes.")]) for i in range(14)]
        write(self.repo, "mapqa", rows)
        got = list(hc.iter_mapqa(self.repo))
        self.assertEqual(len(got), 3 + 1 + 14)
        yes, no, value, late = got[:4]
        self.assertEqual((yes.answer_type, yes.choices, yes.gold, yes.question),
                         ("mcq", ("Yes", "No"), "A", "Does A have the highest value?"))
        self.assertEqual((no.answer_type, no.gold), ("mcq", "B"))
        self.assertEqual((value.answer_type, value.choices, value.gold, value.question),
                         ("short_text", (), "15-17", "What is the value of Florida?"))
        self.assertTrue(late.native_id.endswith(":5:1"))
        for c in got:
            self.assertEqual(c.subject, "Sociology" if int(c.family_key[:8], 16) % 2 == 0 else "Geography")
        self.assertEqual({c.subject for c in got}, {"Sociology", "Geography"})


    def test_not_available_golds_skipped(self):
        write(self.repo, "mapqa", [row([("Value of A?\nBe succinct.", "N/A.")]),
                                   row([("Value of B?\nBe succinct.", "n/a")]),
                                   row([("Value of C?\nBe succinct.", "N/A."), ("Value of D?\nBe succinct.", "Maine.")])])
        got = list(hc.iter_mapqa(self.repo))
        self.assertEqual([(c.question, c.gold) for c in got], [("Value of D?", "Maine")])
        self.assertTrue(got[0].native_id.endswith(":2:1"))


class PlotqaTests(CauldronCase):
    def test_numbers_yes_no_and_subjects(self):
        write(self.repo, "plotqa", [
            row([("What is the percentage of female population in 1993?\nOffer a very short reply.", "87.42.")]),
            row([("How many dotlines are there?\nMake the answer very short.", "2.")]),
            row([("Maximum amount earned by services?\nGive a very brief answer.", "2.83e+09.")]),   # exponent: skipped
            row([("Is the number of dotlines equal to the number of labels?\nYour answer should be compact.", "Yes.")]),
            row([("What is the value in US$ for 2001?\nBe succinct.", "-5.5.")]),
            row([("A weird one?\nBe succinct.", "1,000.")]),                                         # comma: skipped
            row([("What is the cost in 2010?\nBe succinct.", "12%.")]),                              # percent: skipped
        ])
        got = list(hc.iter_plotqa(self.repo))
        self.assertEqual([(c.answer_type, c.gold, c.accept, c.subject) for c in got], [
            ("decimal", "87.42", {"round_to": 2}, "Economics"),
            ("decimal", "2", {"abs_tol": "0.000001"}, "Economics"),
            ("mcq", "A", {}, "Economics"),
            ("decimal", "-5.5", {"round_to": 1}, "Finance"),
        ])
        self.assertEqual(got[2].choices, ("Yes", "No"))
        self.assertEqual(got[0].question, "What is the percentage of female population in 1993?")

    def test_finance_keyword(self):
        write(self.repo, "plotqa", [row([("What is the cost of x?\nBe succinct.", "3.")]),
                                    row([("What are the exports in 2000?\nBe succinct.", "4.")]),
                                    row([("Which is largest, costly?\nBe succinct.", "5.")])])
        self.assertEqual([c.subject for c in hc.iter_plotqa(self.repo)], ["Finance", "Finance", "Economics"])


class FinqaTests(CauldronCase):
    def test_final_answer_only_never_computations(self):
        ctx = "Some filing text .\nMore text on taxes ."
        write(self.repo, "finqa", [
            row([(f"{ctx}\nWhat is the change?", "\nComputations: (3.8 / (100 / 100))\nAnswer: 3.8.")]),
            row([("Plain.\nWhat is the rate?", "0.0029.")]),                                       # bare number form
            row([("Ctx .\nIs it larger?", "\nComputations: (1 > 0)\nAnswer: yes.")]),              # not a number
            row([("Ctx .\nWhat percent?", "\nComputations: (1 / 7)\nAnswer: 14.1%.")]),            # percent
            row([("Ctx .\nWhat is the amount?", "\nComputations: (a - b)\nAnswer: -12.")]),
            row([("Notes on taxes .\nWhat was the depreciation expense?", "\nComputations: (a)\nAnswer: 7.5.")]),
        ])
        got = list(hc.iter_finqa(self.repo))
        self.assertEqual([(c.answer_type, c.gold, c.accept) for c in got], [
            ("decimal", "3.8", {"rel_tol": "0.01"}), ("decimal", "0.0029", {"rel_tol": "0.01"}),
            ("decimal", "-12", {"abs_tol": "0.000001"}), ("decimal", "7.5", {"rel_tol": "0.01"})])
        # the subject keywords are read from the question line only; 'taxes' in the context does not count
        self.assertEqual([c.subject for c in got], ["Finance", "Finance", "Finance", "Accounting"])
        self.assertEqual(got[0].question, f"{ctx}\nWhat is the change?")
        for c in got:
            self.assertNotIn("Computations", repr(c))
            self.assertEqual(c.choices, ())


class TatqaTests(CauldronCase):
    def test_numbers_with_scale_line_kept(self):
        write(self.repo, "tat_qa", [
            row([("What does it consist of?\nBe succinct.", "Rate of inflation, discount rate."),     # text span: skipped
                 ("Ctx.\nHow much is the 2019 rate?\nAnswer scale should be: percent.", "2.9.")]),
            row([("How much?\nAnswer scale should be: thousand.", "16,284.")]),                    # comma: skipped
            row([("What was the total?\nAnswer scale should be: million.", "29259.")]),
        ])
        got = list(hc.iter_tatqa(self.repo))
        self.assertEqual([(c.gold, c.accept, c.subject, c.answer_type) for c in got],
                         [("2.9", {"rel_tol": "0.01"}, "Accounting", "decimal"),
                          ("29259", {"rel_tol": "0.01"}, "Accounting", "decimal")])
        self.assertEqual(got[0].question, "Ctx.\nHow much is the 2019 rate?\nAnswer scale should be: percent.")
        self.assertTrue(got[0].native_id.endswith(":0:1"))


class StreamingTests(CauldronCase):
    def test_sorted_files_row_index_and_many_row_groups(self):
        make = lambda i: row([(f"Does S{i} win?\nBe succinct.", "Yes.")])        # noqa: E731
        write(self.repo, "mapqa", [make(i) for i in range(70)], name="train-00001-of-00002-b.parquet", group=8)
        write(self.repo, "mapqa", [make(i) for i in range(3)], name="train-00000-of-00002-a.parquet")
        (self.repo / "mapqa" / "README.md").write_text("ignored")
        ids = [c.native_id for c in hc.iter_mapqa(self.repo)]
        self.assertEqual(len(ids), 73)
        self.assertEqual(ids[:3], [f"mapqa/train-00000-of-00002-a:{i}:0" for i in range(3)])
        self.assertEqual(ids[3:], [f"mapqa/train-00001-of-00002-b:{i}:0" for i in range(70)])

    def test_turn_choice_is_label_blind_smallest_digest(self):
        turns = [(f"Q{i}?\nAnswer scale should be: percent.", f"{i}.") for i in range(4)]
        write(self.repo, "tat_qa", [row(turns)])
        (got,) = hc.iter_tatqa(self.repo)
        stem = "train-00000-of-00001-x"
        digest = lambda i: hashlib.sha256(f"tat_qa/{stem}:0:{i}".encode()).hexdigest()   # noqa: E731
        want = min(range(4), key=digest)
        self.assertNotEqual(want, 0)                                  # fixture must not coincide with 'first turn'
        self.assertEqual((got.native_id, got.gold), (f"tat_qa/{stem}:0:{want}", str(want)))

    def test_every_source_converts_to_a_valid_problem(self):
        write(self.repo, "ai2d", [row([(ai2d_user("Which season?", "a", "b"), "Answer: B")])])
        write(self.repo, "mapqa", [row([("Does A win?\nBe succinct.", "No.")]), row([("Value?\nBe succinct.", "9-10.")])])
        write(self.repo, "finqa", [row([("Ctx .\nWhat is it?", "\nComputations: (1)\nAnswer: 1.5.")])])
        write(self.repo, "tat_qa", [row([("How much?\nAnswer scale should be: percent.", "2.9.")])])
        write(self.repo, "plotqa", [row([("How many?\nBe succinct.", "2.")])])
        for source, iterate in hc.ITERATORS.items():
            for i, c in enumerate(iterate(self.repo)):
                with self.subTest(source=source, i=i):
                    problem = hf.to_problem(c, hc.SPECS[source], index=i)
                    self.assertEqual(problem.verifier.gold, c.gold)
                    self.assertEqual(problem.lineage.source_revision, hc.REVISION)


if __name__ == "__main__":
    unittest.main()
