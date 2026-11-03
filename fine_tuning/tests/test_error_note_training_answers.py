"""Training-only answer contracts, authored CPU fixtures and private-worker boundaries."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fine_tuning.notebook_rl.answers import (
    TRAINING_ANSWER_CONTRACT, TURN1_EVAL_CONTRACT, compare_training, extract_training,
)
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.notebook_rl.verifier_service import (
    VerifierService, VerifierServiceError, _dispatch, _Issued, _Rejected,
)
from fine_tuning.tests.test_notebook_rl_schema import records


class TrainingAnswersTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base, self.index = Path(temp.name), 0

    def vault(self, kind="decimal", gold="60", accept=None, choices=()):
        self.index += 1
        root = self.base / str(self.index)
        actor, item, lineage, *_ = records()
        actor = replace(actor, answer_format=kind, choices=choices)
        item = replace(item, answer_type=kind, gold=gold, accept=accept or {}, choices=choices)
        write_split(root, [actor], [item], [lineage])
        return root, actor, item, lineage

    def rules(self, root, actor, lineage, kind, accept):
        rule = {"actor_sha256": hashlib.sha256(to_json(actor).encode()).hexdigest(),
                "image_sha256": list(actor.image_sha256), "source": actor.source,
                "source_revision": lineage.source_revision, "split": "train", "native_id": "fixture-row-1",
                "answer_type": kind, "accept": accept,
                "provenance": {"label_provenance": lineage.label_provenance, "field": "answer.unit",
                    "location": "authored-fixture:1", "sha256": hashlib.sha256(b"authored unit record").hexdigest(),
                    "human_verified": False, "review_scope": "authored CPU contract fixture"}}
        data = {"version": "training-answer-rules-v1",
                "vault_manifest_sha256": hashlib.sha256((root / "manifest.yaml").read_bytes()).hexdigest(),
                "items": {actor.problem_id: rule}}
        path = root / "training_rules.private.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest(), data

    def grade(self, verifier, pid, text, contract=TRAINING_ANSWER_CONTRACT, finish="stop"):
        return verifier.grade(pid, text, finish, "plain", answer_contract=contract)

    def test_bare_range_is_a_single_whole_answer_and_legacy_is_unchanged(self):
        root, actor, *_ = self.vault("short_text", "78-80")
        verifier = Verifier(root)
        for text, status in (("78-80", "CORRECT"), ("Final answer: 78-80", "CORRECT"),
                             ("78-81", "INCORRECT"), ("-78-80", "INCORRECT"),
                             ("78 or 80", "INVALID_MULTIPLE"), ("78, 80", "INVALID_MULTIPLE")):
            self.assertEqual(self.grade(verifier, actor.problem_id, text).status, status)
        self.assertEqual(self.grade(verifier, actor.problem_id, "78-80", None).status, "INVALID_NO_FINAL")
        self.assertEqual(self.grade(verifier, actor.problem_id, "78-80", TURN1_EVAL_CONTRACT).status,
                         "INVALID_NO_FINAL")

    def test_numeric_and_fraction_wrappers_preserve_sign_and_math(self):
        root, actor, item, lineage = self.vault("fraction", "-3/4", {"accept_decimal": True})
        path, sha, _ = self.rules(root, actor, lineage, "fraction", {"accept_decimal": True})
        verifier = Verifier(root, training_rules_path=path, training_rules_sha256=sha)
        for text in (r"Final answer: $\boxed{\frac{-3}{4}}$", r"\boxed{-3/4}", "-0.75"):
            self.assertEqual(self.grade(verifier, actor.problem_id, text).status, "CORRECT")
        self.assertEqual(self.grade(verifier, actor.problem_id, r"\boxed{3/4}").status, "INCORRECT")
        for text in ("Final answer: 2**3", "Final answer: sqrt(60)", "Final answer: 60 rad"):
            self.assertEqual(self.grade(verifier, actor.problem_id, text).status, "UNVERIFIABLE")
        self.assertFalse(compare_training("3/4", item))
        for value in (r"2\frac{1}{3}", r"\frac{1}{3}2"):
            self.assertIsNone(compare_training(value, replace(item, gold="7")))

    def test_balanced_markdown_targets_and_headers_are_gold_blind(self):
        forms = ("Final answer: **{}**", "**Final answer: {}**", "**Final answer:** {}",
                 "**Final answer**: **{}**", "Reasoning.\n**Final answer:** **{}**")
        for kind, gold, wrong in (("decimal", "60", "61"), ("short_text", "liver", "kidney")):
            root, actor, item, _ = self.vault(kind, gold)
            verifier = Verifier(root)
            for form in forms:
                text = form.format(gold)
                with self.subTest(kind=kind, text=text):
                    result = self.grade(verifier, actor.problem_id, text)
                    self.assertEqual((result.status, result.extracted), ("CORRECT", gold))
                    verifier.items[actor.problem_id] = replace(item, gold=wrong)
                    changed_gold = self.grade(verifier, actor.problem_id, text)
                    self.assertEqual((changed_gold.status, changed_gold.extracted), ("INCORRECT", gold))
                    verifier.items[actor.problem_id] = item
                    self.assertEqual(self.grade(verifier, actor.problem_id, form.format(wrong)).status, "INCORRECT")
            for text in (f"Final answer: **{gold} or {wrong}**",
                         f"**Final answer:** **{gold}**\n**Final answer:** **{wrong}**"):
                self.assertEqual(self.grade(verifier, actor.problem_id, text).status, "INVALID_MULTIPLE")
        root, actor, *_ = self.vault()
        verifier = Verifier(root)
        for text in ("Final answer: 60**", "**Final answer: 60", "Final answer: **60** rad", "Final answer: 2**3"):
            self.assertNotEqual(self.grade(verifier, actor.problem_id, text).status, "CORRECT")

    def test_units_need_source_rules_and_wrong_units_are_not_erased(self):
        root, actor, _, lineage = self.vault()
        path, sha, _ = self.rules(root, actor, lineage, "unit_number",
                                 {"unit_aliases": ["°"], "require_unit": False, "variable": "x"})
        verifier = Verifier(root, training_rules_path=path, training_rules_sha256=sha)
        for text in (r"\boxed{60^\circ}", r"Final answer: $x = \boxed{60^{\circ}}$", "60°", "60"):
            self.assertEqual(self.grade(verifier, actor.problem_id, text).status, "CORRECT", text)
        for text in ("60 rad", "60 cm", "60°^2", "-60°", "61°"):
            self.assertEqual(self.grade(verifier, actor.problem_id, text).status, "INCORRECT", text)
        self.assertEqual(self.grade(verifier, actor.problem_id, "y = 60°").status, "UNVERIFIABLE")
        self.assertEqual(self.grade(Verifier(root), actor.problem_id, "60°").status, "UNVERIFIABLE")
        self.assertEqual(self.grade(Verifier(root), actor.problem_id, r"\boxed{60}").status, "CORRECT")

    def test_ranges_units_and_aliases_are_explicit_typed_policies(self):
        root, actor, _, lineage = self.vault("short_text", "-3--1")
        path, sha, _ = self.rules(root, actor, lineage, "range", {})
        verifier = Verifier(root, training_rules_path=path, training_rules_sha256=sha)
        self.assertEqual(self.grade(verifier, actor.problem_id, "-3 to -1").status, "CORRECT")
        self.assertEqual(self.grade(verifier, actor.problem_id, "-3–-2").status, "INCORRECT")
        self.assertEqual(self.grade(verifier, actor.problem_id, "-3–-1 cm").status, "UNVERIFIABLE")
        root, actor, _, lineage = self.vault("short_text", "myocardial infarction")
        path, sha, _ = self.rules(root, actor, lineage, "short_text", {"aliases": ["MI"]})
        self.assertEqual(self.grade(Verifier(root), actor.problem_id, "MI").status, "INCORRECT")
        verifier = Verifier(root, training_rules_path=path, training_rules_sha256=sha)
        self.assertEqual(self.grade(verifier, actor.problem_id, "MI").status, "CORRECT")
        self.assertEqual(self.grade(verifier, actor.problem_id, "infarction").status, "INCORRECT")

    def test_multiple_candidates_and_mcq_content_conflicts_never_select_gold(self):
        for text in ("Final answer: 60\nFinal answer: 61", "The answer is 60.\nActually, the answer is 61.",
                     r"\boxed{60} or \boxed{61}", "60 and 61", "60; 61"):
            self.assertEqual(extract_training(text, "stop", "Find x.", (), "decimal").status, "INVALID_MULTIPLE")
        root, actor, _, _ = self.vault("mcq", "B", choices=("42", "24"))
        verifier = Verifier(root)
        for text in ("Final answer: B. 42", "Final answer: B or A"):
            self.assertNotEqual(self.grade(verifier, actor.problem_id, text).status, "CORRECT")
        self.assertEqual(self.grade(verifier, actor.problem_id, "Final answer: B. 24").status, "CORRECT")

    def test_gold_changes_never_change_extraction_and_length_stays_truncated(self):
        root, actor, item, _ = self.vault()
        verifier = Verifier(root)
        for text in ("60", "61", "Final answer: 60 rad", "78-80", "Final answer: sqrt(60)", "60 or 61"):
            first = self.grade(verifier, actor.problem_id, text)
            verifier.items[actor.problem_id] = replace(item, gold="61")
            second = self.grade(verifier, actor.problem_id, text)
            self.assertEqual(first.extracted, second.extracted)
            verifier.items[actor.problem_id] = item
        self.assertEqual(self.grade(verifier, actor.problem_id, "60", finish="length").status, "TRUNCATED")

    def test_rule_hash_actor_binding_and_provenance_tampering_fail_closed(self):
        root, actor, _, lineage = self.vault()
        path, sha, data = self.rules(root, actor, lineage, "unit_number",
                                     {"unit_aliases": ["°"], "require_unit": False})
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            Verifier(root, training_rules_path=path, training_rules_sha256=sha)
        for field, value in (("actor_sha256", "0" * 64), ("source_revision", "invented"),
                             ("image_sha256", []), ("split", "dev")):
            changed = json.loads(json.dumps(data))
            changed["items"][actor.problem_id][field] = value
            path.write_text(json.dumps(changed))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "binding mismatch"):
                Verifier(root, training_rules_path=path, training_rules_sha256=digest)
        del data["items"][actor.problem_id]["provenance"]["field"]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            Verifier(root, training_rules_path=path, training_rules_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def test_worker_reextracts_and_rejects_forged_answers(self):
        root, actor, *_ = self.vault()
        verifier = Verifier(root)
        result = self.grade(verifier, actor.problem_id, "60")
        request = {"id": 1, "method": "grade", "problem_id": actor.problem_id,
                   "params": {"text": "60", "finish_reason": "stop", "mode": "plain",
                              "answer_contract": TRAINING_ANSWER_CONTRACT}}
        with patch.object(verifier, "grade", return_value=replace(result, extracted="61")):
            with self.assertRaisesRegex(_Rejected, "VERIFIER_ERROR"):
                _dispatch(verifier, _Issued(), request)

    def test_fresh_guarded_actor_never_opens_rules_and_subprocess_checks_hash(self):
        root, actor, _, lineage = self.vault()
        path, sha, _ = self.rules(root, actor, lineage, "unit_number",
                                 {"unit_aliases": ["°"], "require_unit": False})
        script = """
import json, pathlib, sys
from fine_tuning.notebook_rl.guards import install_guards
from fine_tuning.notebook_rl.verifier_service import VerifierService
root, rules = map(pathlib.Path, sys.argv[1:3])
private=[root/'verifier_private.jsonl',root/'lineage.jsonl',rules]
install_guards(forbidden_roots=[],vault_files=private,allow_vault=False,allow_benchmark=False)
opened=[]
def audit(event,args):
    if event=='open' and isinstance(args[0],str): opened.append(args[0])
sys.addaudithook(audit)
with VerifierService(root,forbidden_roots=[],training_rules_path=rules,training_rules_sha256=sys.argv[3]) as service:
    result=service.grade(sys.argv[4],'60°','stop','plain',answer_contract='training-answer-v1')
    assert result.status=='CORRECT'
    assert service.metadata['training_rules_sha256']==sys.argv[3]
    assert not any(str(p) in opened for p in private)
    for p in private:
        try: p.read_bytes()
        except PermissionError: pass
        else: raise AssertionError('private open allowed')
    wire=json.dumps([service.metadata,result.__dict__])
    assert 'unit_aliases' not in wire and 'provenance' not in wire
print('guarded training worker PASS')
"""
        result = subprocess.run([sys.executable, "-B", "-c", script, str(root), str(path), sha, actor.problem_id],
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS", result.stdout)
        with self.assertRaisesRegex(VerifierServiceError, "VAULT_UNAVAILABLE"):
            VerifierService(root, forbidden_roots=[], training_rules_path=path, training_rules_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
