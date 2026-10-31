"""Authored fixtures only: a tiny synthetic features JSONL and image zip in a temp directory (no RDKit needed)."""
import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from fine_tuning.notebook_rl.adapters import hf_chem as chem
from fine_tuning.notebook_rl.adapters import hf_common as hf


def template_of(cid):
    return int(hashlib.sha256(f"chebi:{cid}".encode()).hexdigest(), 16) % 3


def cids_for(template, count):
    found = []
    cid = 1000
    while len(found) < count:
        if template_of(cid) == template:
            found.append(cid)
        cid += 1
    return found


def png(color):
    out = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(out, format="PNG")
    return out.getvalue()


ALLOW_LIST_SHA256 = "52c5992d736cb6d12e4a84b7e7e5f40e403a9ac3d254788668762e17ec9269c9"
T0 = cids_for(0, 3)
T1 = cids_for(1, 2)
T2 = cids_for(2, 4)


def record(cid, formula, heavy_atoms, rings=0, n_o=0):
    return {"cid": cid, "formula": formula, "heavy_atoms": heavy_atoms, "rings": rings, "n_O": n_o, "n_N": 0,
            "aromatic_rings": 0, "smiles_sha256": "0" * 64}


RECORDS = (
    record(T0[0], "C5H10O", 10, rings=2, n_o=1),
    record(T1[0], "C4H8O3", 11, rings=0, n_o=3),
    record(T2[0], "C6H12", 12, rings=1),
    record(T2[1], "C7H14", 13),
    record(T0[1], "C9H18", 15),                    # 3 heavy atoms from C6H12: outside its pool window
    record(T2[2], "C20H40", 40),                   # alone in its window: fewer than 2 formulas, skipped
    record(T0[2], "C60H120", 61),                  # heavy_atoms > 60: skipped
    record(T0[0] + 3000, "C25H50", 50, rings=13),  # rings > 12: skipped (template irrelevant)
    record(T1[1], "C15H30", 30, n_o=2),            # image missing from the zip: skipped
)
IMAGE_MISSING = {T1[1]}
KEPT = (T0[0], T1[0], T2[0], T2[1], T0[1])


def write_chebi(root, records=RECORDS):
    (root / "derived").mkdir()
    (root / "derived" / "chebi_features_v1.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    with zipfile.ZipFile(root / "image.zip", "w") as archive:
        archive.writestr("image/", b"")
        for i, r in enumerate(records):
            if r["cid"] not in IMAGE_MISSING:
                archive.writestr(f"image/CID_{r['cid']}.png", png((i * 20, 20, 20)))


class ChebiTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        write_chebi(self.root)
        self.candidates = list(chem.iter_chebi(self.root))
        self.by_cid = {int(c.native_id.split(":")[1]): c for c in self.candidates}

    def test_kept_molecules_in_file_order(self):
        self.assertEqual([c.native_id for c in self.candidates], [f"cid:{cid}" for cid in KEPT])
        self.assertTrue(all(c.family_key == c.native_id and c.source == "hf-chebi" for c in self.candidates))

    def test_ring_template_is_an_int_from_rings(self):
        c = self.by_cid[T0[0]]
        self.assertEqual((c.question, c.subject, c.answer_type, c.gold, c.choices), (
            "How many rings does the molecule in the image contain?", "Chemistry", "int", "2", ()))

    def test_oxygen_template_is_an_int_from_n_o(self):
        c = self.by_cid[T1[0]]
        self.assertEqual((c.question, c.subject, c.answer_type, c.gold), (
            "How many oxygen atoms does the molecule in the image contain?", "Pharmacy", "int", "3"))

    def test_formula_template_pool_is_heavy_atom_window(self):
        a = self.by_cid[T2[0]]
        self.assertEqual((a.question, a.subject, a.answer_type), (
            "Which molecular formula matches the molecule in the image?", "Pharmacy", "mcq"))
        self.assertEqual(a.choices["ABCD".index(a.gold)], "C6H12")
        self.assertEqual(set(a.choices), {"C5H10O", "C4H8O3", "C6H12", "C7H14"})
        b = self.by_cid[T2[1]]
        self.assertEqual(b.choices["ABCD".index(b.gold)], "C7H14")
        self.assertEqual(set(b.choices), {"C4H8O3", "C6H12", "C7H14", "C9H18"})

    def test_skips(self):
        for cid in (T2[2], T0[2], T0[0] + 3000, T1[1]):
            with self.subTest(cid=cid):
                self.assertNotIn(cid, self.by_cid)

    def test_image_bytes_come_from_the_zip_member(self):
        with zipfile.ZipFile(self.root / "image.zip") as archive:
            for cid, c in self.by_cid.items():
                self.assertEqual(c.images, (archive.read(f"image/CID_{cid}.png"),))

    def test_iteration_is_deterministic(self):
        self.assertEqual(self.candidates, list(chem.iter_chebi(self.root)))

    def test_candidates_convert_to_program_problems(self):
        for c in self.candidates:
            with self.subTest(native_id=c.native_id):
                problem = hf.to_problem(c, chem.SPECS[c.source], index=0)
                self.assertEqual((problem.lineage.label_provenance, problem.verifier.gold), ("PROGRAM", c.gold))

    def test_missing_features_file_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(FileNotFoundError, "chebi_features"):
                list(chem.iter_chebi(Path(empty)))


def description(*roles, tail="It is a carboxylic acid."):
    if not roles:
        return "The molecule is a thing. " + tail
    listed = ", ".join(roles[:-1]) + (" and " if len(roles) > 1 else "") + roles[-1]
    return f"The molecule is a thing. It has a role as {listed}. {tail}"


def write_roles(root, rows, missing_image=()):
    """rows: (cid, smiles, description). Writes train.csv plus a zip holding one tiny PNG per cid."""
    with open(root / "train.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["CID", "SMILES", "description"])
        writer.writerows(rows)
    with zipfile.ZipFile(root / "image.zip", "w") as archive:
        for i, (cid, _, _) in enumerate(rows):
            if cid not in missing_image:
                archive.writestr(f"image/CID_{cid}.png", png((i * 10, 20, 20)))


class ParseRolesTests(unittest.TestCase):
    def test_roles_as_written_without_article(self):
        self.assertEqual(chem.parse_roles(description("an antineoplastic agent", "an antiplasmodial drug", "a plant metabolite")),
                         ("antineoplastic agent", "antiplasmodial drug", "plant metabolite"))
        self.assertEqual(chem.parse_roles(description("a human metabolite", "a Saccharomyces cerevisiae metabolite")),
                         ("human metabolite", "Saccharomyces cerevisiae metabolite"))

    def test_and_inside_one_role_is_not_a_split_point(self):
        self.assertEqual(chem.parse_roles(description("a human and mouse metabolite")), ("human and mouse metabolite",))

    def test_dots_inside_a_role_do_not_end_the_sentence(self):
        text = description("an EC 2.7.10.1 (receptor protein-tyrosine kinase) inhibitor", "an E. coli metabolite")
        self.assertEqual(chem.parse_roles(text), ("EC 2.7.10.1 (receptor protein-tyrosine kinase) inhibitor", "E. coli metabolite"))

    def test_missing_full_stop_before_the_sentence_is_still_read(self):
        self.assertEqual(chem.parse_roles("The molecule is a dye It has a role as a dye."), ("dye",))

    def test_unsplittable_sentence_contributes_nothing(self):
        self.assertEqual(chem.parse_roles("It has a role as antibiotics and others. It is a thing."), ())

    def test_no_role_sentence(self):
        self.assertEqual(chem.parse_roles(description()), ())


class RoleAllowListTests(unittest.TestCase):
    def test_allow_list_is_pinned(self):
        self.assertEqual({f: len(p) for f, p in chem.ROLE_FAMILIES.items()}, {
            "anti_infective": 29, "antineoplastic": 25, "inflammation_immune_pain": 26, "cardiovascular_blood": 31,
            "cns_neuro_psychiatric": 46, "endocrine_metabolic": 18, "gastrointestinal_hepatic": 10,
            "respiratory": 5, "other_therapeutic": 15})
        lines = sorted(f"{f}\t{p}" for f, phrases in chem.ROLE_FAMILIES.items() for p in phrases)
        # Changing the allow-list is a deliberate act: update this digest together with the build manifest.
        self.assertEqual(hashlib.sha256("\n".join(lines).encode()).hexdigest(), ALLOW_LIST_SHA256)

    def test_non_pharmacological_roles_stay_out(self):
        for phrase in ("metabolite", "plant metabolite", "human xenobiotic metabolite", "drug metabolite", "xenobiotic",
                       "agrochemical", "carcinogenic agent", "mutagen", "epitope", "fluorochrome", "histological dye",
                       "antioxidant", "prodrug", "orphan drug", "herbicide", "solvent", "toxin"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, chem.ROLE_FAMILY)

    def test_every_phrase_is_a_whole_a_an_item_of_a_role_sentence(self):
        for phrase in chem.ROLE_FAMILY:
            with self.subTest(phrase=phrase):
                self.assertEqual(chem.parse_roles(description(f"a {phrase}")), (phrase,))


class RoleOptionsTests(unittest.TestCase):
    def test_one_gold_role_and_three_distractors_from_other_families(self):
        text = description("an antineoplastic agent", "a plant metabolite", "an apoptosis inducer")
        choices, gold = chem.role_options(7, text)
        self.assertEqual(len(set(choices)), 4)
        self.assertIn(choices["ABCD".index(gold)], ("antineoplastic agent", "apoptosis inducer"))
        families = [chem.ROLE_FAMILY[c] for c in choices]
        self.assertEqual(len(set(families)), 4)
        self.assertEqual(families.count("antineoplastic"), 1)

    def test_distractors_avoid_the_family_of_every_allow_listed_role(self):
        text = description("an antidepressant", "an antibacterial agent", "a sedative")
        for cid in range(40):
            choices, gold = chem.role_options(cid, text)
            others = [c for c in choices if c != choices["ABCD".index(gold)]]
            self.assertFalse({chem.ROLE_FAMILY[c] for c in others} & {"anti_infective", "cns_neuro_psychiatric"})

    def test_related_only_role_blocks_its_family(self):
        text = description("an antineoplastic agent", "an antifungal agrochemical")
        for cid in range(40):
            choices, _ = chem.role_options(cid, text)
            self.assertNotIn("anti_infective", {chem.ROLE_FAMILY[c] for c in choices})

    def test_ambiguous_related_role_blocks_every_family_it_could_belong_to(self):
        text = description("an analgesic", "a protein synthesis inhibitor")
        for cid in range(40):
            families = {chem.ROLE_FAMILY[c] for c in chem.role_options(cid, text)[0]}
            self.assertFalse(families & {"anti_infective", "antineoplastic"})

    def test_distractor_named_in_the_description_text_is_dropped(self):
        tail = "It exhibits antibacterial agent, antifungal agent and antiviral agent properties."
        text = description("an antineoplastic agent", tail=tail)
        for cid in range(40):
            choices, _ = chem.role_options(cid, text)
            self.assertFalse({"antibacterial agent", "antifungal agent", "antiviral agent"} & set(choices))

    def test_deterministic_and_gold_letters_vary(self):
        text = description("an analgesic")
        self.assertEqual(chem.role_options(5, text), chem.role_options(5, text))
        self.assertGreater(len({chem.role_options(cid, text)[1] for cid in range(40)}), 1)

    def test_none_without_an_allow_listed_role(self):
        self.assertIsNone(chem.role_options(1, description("a plant metabolite", "an agrochemical")))
        self.assertIsNone(chem.role_options(1, description()))

    def test_none_when_fewer_than_three_families_remain(self):
        roles = ("an antineoplastic agent", "an antibacterial agent", "an analgesic", "a diuretic", "an antidepressant",
                 "a hypoglycemic agent", "an antiemetic")
        self.assertIsNone(chem.role_options(1, description(*roles)))   # 7 of 9 families used, 2 left
        self.assertIsNotNone(chem.role_options(1, description(*roles[:6])))


class RoleIteratorTests(unittest.TestCase):
    ROWS = (
        (101, "CCO", description("an antineoplastic agent", "a plant metabolite")),
        (102, "CCN", description("a plant metabolite")),                          # no allow-listed role
        (103, "CCC", description("an analgesic")),                                # image missing from the zip
        (104, "C" * 61, description("an anticonvulsant")),                        # 61 heavy atoms: too large
        (105, "[H]C([Cl])=O" + "C" * 57, description("a diuretic")),              # exactly 60 heavy atoms: kept
        (106, "CCS", description("an antimalarial", "an antiviral agent")),
    )

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        write_roles(self.root, self.ROWS, missing_image={103})
        self.candidates = list(chem.iter_chebi_role(self.root))

    def test_kept_molecules_in_file_order(self):
        self.assertEqual([c.native_id for c in self.candidates], ["cid:101", "cid:105", "cid:106"])
        for c in self.candidates:
            self.assertEqual((c.source, c.subject, c.answer_type, c.question, c.family_key), (
                "hf-chebi-role", "Pharmacy", "mcq", chem.ROLE_QUESTION, c.native_id))
            self.assertEqual(len(c.choices), 4)
            self.assertIn(c.gold, "ABCD")

    def test_gold_is_a_role_listed_for_that_molecule(self):
        by_id = {c.native_id: c for c in self.candidates}
        gold = lambda c: c.choices["ABCD".index(c.gold)]   # noqa: E731
        self.assertEqual(gold(by_id["cid:101"]), "antineoplastic agent")
        self.assertEqual(gold(by_id["cid:105"]), "diuretic")
        self.assertIn(gold(by_id["cid:106"]), ("antimalarial", "antiviral agent"))

    def test_image_bytes_come_from_the_zip_member(self):
        with zipfile.ZipFile(self.root / "image.zip") as archive:
            for c in self.candidates:
                self.assertEqual(c.images, (archive.read(f"image/CID_{c.native_id.split(':')[1]}.png"),))

    def test_question_never_contains_the_answer(self):
        for c in self.candidates:
            self.assertNotIn(c.choices["ABCD".index(c.gold)].casefold(), c.question.casefold())

    def test_iteration_is_deterministic(self):
        self.assertEqual(self.candidates, list(chem.iter_chebi_role(self.root)))

    def test_candidates_convert_to_human_problems(self):
        for c in self.candidates:
            problem = hf.to_problem(c, chem.SPECS[c.source], index=0)
            self.assertEqual((problem.lineage.label_provenance, problem.verifier.gold), ("HUMAN", c.gold))
            self.assertNotIn(c.gold + ".", problem.actor.question)


REAL = Path("/home/jang2/mmdl-data/training/raw/hf-pool-v1/liupf__ChEBI-20-MM")


@unittest.skipUnless((REAL / "train.csv").is_file() and (REAL / "image.zip").is_file(), "ChEBI-20-MM not downloaded")
class RealChebiRoleTests(unittest.TestCase):
    def test_pinned_candidate_count_and_vocabulary(self):
        candidates = list(chem.iter_chebi_role(REAL))
        self.assertEqual(len(candidates), 4078)
        with open(REAL / "train.csv", encoding="utf-8", newline="") as handle:
            seen = {role for row in csv.DictReader(handle) for role in chem.parse_roles(row["description"])}
        self.assertEqual(set(chem.ROLE_FAMILY) - seen, set())
        self.assertEqual(len({c.native_id for c in candidates}), 4078)


class SpecTests(unittest.TestCase):
    def test_role_source_is_human_labelled_on_the_same_pinned_repo(self):
        spec = chem.SPECS["hf-chebi-role"]
        self.assertEqual((spec.repo, spec.revision, spec.label_provenance, spec.family), (
            "liupf/ChEBI-20-MM", "7d44959bad65cd31e481a7561dac25a513e8957b", "HUMAN", "molecule_role"))
        self.assertEqual(spec.license, chem.SPECS["hf-chebi"].license)

    def test_pinned_source_and_iterator(self):
        self.assertEqual(set(chem.SPECS), {"hf-chebi", "hf-chebi-role"})
        self.assertEqual(set(chem.ITERATORS), set(chem.SPECS))
        spec = chem.SPECS["hf-chebi"]
        self.assertEqual((spec.repo, spec.revision, spec.label_provenance, spec.family), (
            "liupf/ChEBI-20-MM", "7d44959bad65cd31e481a7561dac25a513e8957b", "PROGRAM", "molecule"))
        self.assertEqual(spec.license, "MIT (ChEBI CC-BY-4.0, PubChem public domain)")


if __name__ == "__main__":
    unittest.main()
