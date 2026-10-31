"""Synthetic-fixture tests for the hf-gvlqa converter (no network, no GPU); one real-data check when downloaded."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

from fine_tuning.notebook_rl.adapters import hf_graph_vlqa as g
from fine_tuning.notebook_rl.adapters.hf_common import to_problem

CONN_Q = "Is there a path between node {} and node {} in this undirected graph?"
FLOW_Q = ("This graphic illustrates a directed graph, with each edge's capacity indicated by a numerical label in "
          "close proximity.\nQ: What is the maximum flow from node 8 to node 6:")
MATCH_Q = "There are 4 hosts.\nQ:  What is the maximum number of hosts that can be assigned a task they are interested in?"


def png(color):
    out = io.BytesIO()
    Image.new("RGB", (6, 6), color).save(out, format="PNG")
    return out.getvalue()


def row(task, level, graph, question, answer, suffix=""):
    return {"id": f"{task}-{level}-graph{graph}{suffix}", "image": f"data/{task}/graph_visual/{level}/graph{graph}.png",
            "conversations": [{"from": "human", "value": f"<image>\n{question}"}, {"from": "gpt", "value": answer}]}


ROWS = {
    "connectivity": [
        row("connectivity", "easy", 1, CONN_Q.format(1, 6), "Yes.", "-1_6"),
        row("connectivity", "easy", 1, CONN_Q.format(2, 3), "No.", "-2_3"),     # same graph, second question
        row("connectivity", "hard", 7, CONN_Q.format(0, 4), "Maybe."),           # not Yes/No: skipped
        row("connectivity", "hard", 8, CONN_Q.format(0, 4), "No.", "-0_4"),      # image not downloaded: skipped
    ],
    "cycle": [
        row("cycle", "medium", 3, "Is there a cycle in this undirected graph?", "No."),
        row("cycle", "medium", 4, "Is there a cycle in this undirected graph?", "Yes."),
    ],
    "flow": [
        row("flow", "easy", 2, FLOW_Q, "4."),
        row("flow", "easy", 5, FLOW_Q, "1/2."),                                   # not an integer: skipped
        row("flow", "hard", 9, "   ", "3."),                                      # empty question: skipped
    ],
    "matching": [
        row("matching", "hard", 6, MATCH_Q, "11."),
        row("matching", "hard", 12, MATCH_Q, "3->4."),                            # a path: skipped
        {**row("matching", "hard", 13, MATCH_Q, "5."), "image": "data/flow/graph_visual/hard/graph13.png"},  # wrong task
    ],
}
MISSING_IMAGES = {"data/connectivity/graph_visual/hard/graph8.png"}


def write_repo(root):
    for task, rows in ROWS.items():
        (root / "data" / task / "QA").mkdir(parents=True)
        (root / "data" / task / "QA" / "Vision_Only_train.json").write_text(json.dumps(rows), encoding="utf-8")
        # The creators' test split and the text variants must never be opened.
        (root / "data" / task / "QA" / "Vision_Only_test.json").write_text("not json", encoding="utf-8")
        (root / "data" / task / "QA" / "Text_Only_train.json").write_text("not json", encoding="utf-8")
        for i, r in enumerate(rows):
            if r["image"] not in MISSING_IMAGES:
                path = root / r["image"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(png((i * 30, 40, 40)))


class AnswerTests(unittest.TestCase):
    def test_yes_no_is_a_two_option_mcq(self):
        self.assertEqual(g._answer("connectivity", "Yes."), (("Yes", "No"), "mcq", "A"))
        self.assertEqual(g._answer("cycle", "No."), (("Yes", "No"), "mcq", "B"))

    def test_integer_tasks_give_int_gold(self):
        self.assertEqual(g._answer("flow", "10."), ((), "int", "10"))
        self.assertEqual(g._answer("matching", "0"), ((), "int", "0"))

    def test_anything_else_is_not_a_single_value(self):
        for task, value in (("cycle", "Maybe."), ("cycle", "yes"), ("flow", "5->6."), ("flow", "-3."), ("flow", "1.5."),
                            ("matching", ""), ("flow", None), ("matching", "1234567.")):
            with self.subTest(task=task, value=value):
                self.assertIsNone(g._answer(task, value))


class IterGvlqaTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        write_repo(self.root)
        self.candidates = list(g.iter_gvlqa(self.root))
        self.by_id = {c.native_id: c for c in self.candidates}

    def test_kept_rows_in_task_then_file_order(self):
        self.assertEqual([c.native_id for c in self.candidates], [
            "connectivity-easy-graph1-1_6", "connectivity-easy-graph1-2_3", "cycle-medium-graph3",
            "cycle-medium-graph4", "flow-easy-graph2", "matching-hard-graph6"])

    def test_question_is_the_creators_text_without_the_image_token(self):
        self.assertEqual(self.by_id["connectivity-easy-graph1-1_6"].question, CONN_Q.format(1, 6))
        self.assertEqual(self.by_id["flow-easy-graph2"].question, FLOW_Q)
        self.assertEqual(self.by_id["matching-hard-graph6"].question, MATCH_Q)      # inner double space kept

    def test_gold_and_types(self):
        got = {c.native_id: (c.answer_type, c.choices, c.gold) for c in self.candidates}
        self.assertEqual(got["connectivity-easy-graph1-1_6"], ("mcq", ("Yes", "No"), "A"))
        self.assertEqual(got["connectivity-easy-graph1-2_3"], ("mcq", ("Yes", "No"), "B"))
        self.assertEqual(got["cycle-medium-graph4"], ("mcq", ("Yes", "No"), "A"))
        self.assertEqual(got["flow-easy-graph2"], ("int", (), "4"))
        self.assertEqual(got["matching-hard-graph6"], ("int", (), "11"))

    def test_family_is_the_graph_image(self):
        a, b = self.by_id["connectivity-easy-graph1-1_6"], self.by_id["connectivity-easy-graph1-2_3"]
        self.assertEqual(a.family_key, b.family_key)
        self.assertEqual(a.family_key, "connectivity-easy-graph1")
        self.assertEqual(len({c.family_key for c in self.candidates}), 5)

    def test_subject_source_and_image_bytes(self):
        for c in self.candidates:
            self.assertEqual((c.source, c.subject, len(c.images)), ("hf-gvlqa", "Computer_Science", 1))
        self.assertEqual(self.by_id["flow-easy-graph2"].images,
                         ((self.root / "data/flow/graph_visual/easy/graph2.png").read_bytes(),))

    def test_deterministic(self):
        self.assertEqual(self.candidates, list(g.iter_gvlqa(self.root)))

    def test_candidates_convert_to_program_problems(self):
        for c in self.candidates:
            problem = to_problem(c, g.SPECS[c.source], index=0)
            self.assertEqual((problem.lineage.label_provenance, problem.verifier.gold), ("PROGRAM", c.gold))
            self.assertEqual(problem.actor.answer_format, c.answer_type)


class SpecTests(unittest.TestCase):
    def test_pinned_source_and_iterator(self):
        self.assertEqual(set(g.SPECS), {"hf-gvlqa"})
        self.assertEqual(set(g.ITERATORS), set(g.SPECS))
        spec = g.SPECS["hf-gvlqa"]
        self.assertEqual((spec.repo, spec.revision, spec.license, spec.family), (
            "Yanbin99/GVLQA-BASE", "7aeff19faa34ddb19f79b8a5a80d31c1fe9db18b", "MIT", "graph"))
        self.assertEqual(len(spec.revision), 40)

    def test_registry_lists_both_new_sources(self):
        from fine_tuning.notebook_rl.adapters.hf_registry import sources
        registry = sources()
        self.assertIn("hf-gvlqa", registry)
        self.assertIn("hf-chebi-role", registry)
        self.assertIn("hf-chebi", registry)


REAL = Path("/home/jang2/mmdl-data/training/raw/hf-pool-v1/Yanbin99__GVLQA-BASE")


@unittest.skipUnless(all((REAL / f"data/{t}/QA/Vision_Only_train.json").is_file() for t in (*g.YES_NO_TASKS, *g.INT_TASKS)),
                     "GVLQA-BASE not downloaded")
class RealGvlqaTests(unittest.TestCase):
    def test_pinned_counts(self):
        candidates = list(g.iter_gvlqa(REAL))
        by_task = Counter(c.native_id.split("-")[0] for c in candidates)
        self.assertEqual(dict(by_task), REAL_QUESTIONS)
        self.assertEqual(len({c.family_key for c in candidates}), REAL_GRAPHS)
        self.assertEqual(len({c.native_id for c in candidates}), len(candidates))


REAL_QUESTIONS = {"connectivity": 11486, "cycle": 2870, "flow": 1050, "matching": 1302}
REAL_GRAPHS = 6865

if __name__ == "__main__":
    unittest.main()
