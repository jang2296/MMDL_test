"""Synthetic creator annotations only; no benchmark examples."""
from copy import deepcopy
import hashlib
import json
import zipfile
import unittest
import tempfile
from pathlib import Path
from fine_tuning.error_note.audit import digest, text_sha256
from fine_tuning.error_note.content_v16 import MANIFEST_SCHEMA, PROMPTS_SHA256, chart_source, chart_target, content_prompt, geometry_target, graph_target, link_geometry3k, load_content_manifest, observation_row, parse_gvlqa_structure, scene_target, validate_content_manifest, validate_content_row, validate_human_j_capability

def graph_item():
    return {'problem_id': 'synthetic-graph:1', 'parent_id': 'synthetic-graph:1', 'group_id': 'connectivity-easy-graph1', 'source_repo': 'Yanbin99/GVLQA-BASE', 'source_revision': 'a' * 40, 'split': 'train', 'original_split': 'train', 'training_allowed': True, 'label_source': 'PROGRAM_EXACT', 'image_sha256': 'b' * 64, 'image_path': 'figure.png', 'visible_nodes': [0, 1, 3], 'annotation': {'task': 'connectivity', 'level': 'easy', 'graph_id': 'graph1', 'directed': False, 'weighted': False, 'structure_text': '5 2 7\n0 1\n1 3\n3 4\n'}, 'provenance': {'parent_manifest_sha256': 'c' * 64, 'parent_record_sha256': 'd' * 64, 'parent_status': 'INCLUDED', 'rendering_verified': True, 'annotation_sources': [{'path': 'source.json', 'sha256': 'e' * 64}]}}

def bind_parent_qa(item, parent, root):
    records = {}
    for role in ("actor", "verifier"):
        file = root / f"{role}.jsonl"
        if file.exists():
            record = json.loads(file.read_text().splitlines()[0])
        else:
            record = {"problem_id": item["parent_id"]}
        if role == "actor":
            record.setdefault("question", "Synthetic parent question")
            record.setdefault("choices", [])
        else:
            record.setdefault("gold", "Yes")
            record.setdefault("answer_type", "string")
            record.setdefault("accept", {})
        line = json.dumps(record)
        file.write_text(line + "\n")
        item["provenance"][f"{role}_path"] = str(file)
        parent.setdefault("record_sha256", {})[role] = text_sha256(line)
        records[role] = record
    evaluation = {"question": records["actor"]["question"], "choices": records["actor"]["choices"],
                  "gold": records["verifier"]["gold"], "answer_type": records["verifier"]["answer_type"],
                  "accept": records["verifier"]["accept"]}
    item.update(evaluation)
    item["evaluation"] = dict(evaluation)


class TestContentV16(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)

    def test_graph_edge_block_ignores_queries_and_isolates(self):
        struct = parse_gvlqa_structure('5 2 7\n0 1\n1 3\n3 4\n')
        self.assertTrue(struct['edges'] == [(0, 1), (1, 3)])
        self.assertTrue(struct['visible_nodes'] == [0, 1, 3])
        target, reason = graph_target(struct, directed=False)
        self.assertTrue(reason is None and 'Drawn nodes: 0, 1, 3.' in target and ('3-4' not in target))

    def test_malformed_graph_fails_closed_0(self):
        text = ''
        with self.assertRaisesRegex(ValueError, 'MALFORMED_GRAPH'):
            parse_gvlqa_structure(text)

    def test_malformed_graph_fails_closed_1(self):
        text = '3'
        with self.assertRaisesRegex(ValueError, 'MALFORMED_GRAPH'):
            parse_gvlqa_structure(text)

    def test_malformed_graph_fails_closed_2(self):
        text = '3 2\n0 1'
        with self.assertRaisesRegex(ValueError, 'MALFORMED_GRAPH'):
            parse_gvlqa_structure(text)

    def test_malformed_graph_fails_closed_3(self):
        text = '3 1\n0 4'
        with self.assertRaisesRegex(ValueError, 'MALFORMED_GRAPH'):
            parse_gvlqa_structure(text)

    def test_malformed_graph_fails_closed_4(self):
        text = '3 1\n0 0'
        with self.assertRaisesRegex(ValueError, 'MALFORMED_GRAPH'):
            parse_gvlqa_structure(text)

    def test_malformed_graph_fails_closed_5(self):
        text = '3 2\n0 1\n0 1'
        with self.assertRaisesRegex(ValueError, 'MALFORMED_GRAPH'):
            parse_gvlqa_structure(text)

    def test_malformed_graph_fails_closed_6(self):
        text = '3 2\n0 1\n1 2 3'
        with self.assertRaisesRegex(ValueError, 'MALFORMED_GRAPH'):
            parse_gvlqa_structure(text)

    def test_medium_graph_uses_deterministic_neighbor_prompt_and_large_rejects(self):
        text = '40 31\n' + '\n'.join((f'0 {i}' for i in range(1, 32)))
        struct = parse_gvlqa_structure(text)
        target, reason = graph_target(struct, directed=False, parent_id='fixed')
        self.assertTrue(reason is None and 'Requested node:' in target and ('Drawn nodes:' not in target))
        self.assertTrue(graph_target(struct, directed=False, parent_id='fixed')[0] == target)
        item = graph_item()
        item['annotation']['structure_text'] = text
        item['parent_id'] = 'fixed'
        self.assertTrue('Requested node:' in content_prompt(item, 'graph'))
        struct['m'] = 61
        self.assertTrue(graph_target(struct, directed=False, parent_id='fixed') == (None, 'TOO_MANY_EDGES'))

    def test_geometry_join_uses_image_answer_text_not_row_index(self):
        source = {99: {'image_sha256': 'image', 'answer_value': '7', 'texts': ['Find side PQ']}, 5: {'image_sha256': 'wrong', 'answer_value': '7', 'texts': ['Find side PQ']}}
        row = {'index': 5, 'image_sha256': 'image', 'answer': '7', 'problem': 'Find side PQ'}
        self.assertTrue(link_geometry3k([row], source) == {5: 99})
        source[100] = deepcopy(source[99])
        self.assertTrue(link_geometry3k([row], source) == {5: 'LINK_AMBIGUOUS'})
        del source[100]
        self.assertTrue(set(link_geometry3k([row, {**row, 'index': 6}], source).values()) == {'LINK_AMBIGUOUS'})

    def test_geometry_lists_diagram_labels_and_marks_all_or_reject(self):
        forms = ['PointLiesOnLine(Q, Line(R, N))', 'Equals(LengthOf(Line(A, B)), 2x+1)', 'Equals(MeasureOf(Angle(A, B, C)), 45)', 'Parallel(Line(A, B), Line(C, D))']
        target, reason = geometry_target(forms)
        self.assertTrue(reason is None and 'segment AB = 2x+1' in target and ('ABC = 45°' in target))
        self.assertTrue('AB ∥ CD' in target and 'PointLies' not in target)
        self.assertTrue(geometry_target(forms + ['Tangent(Line(A,B), Circle(G))']) == (None, 'UNSUPPORTED_FORM'))
        self.assertTrue(geometry_target([forms[0]]) == (None, 'NO_FACTS'))
        self.assertTrue(geometry_target(['PointLiesOnLine(broken)']) == (None, 'UNSUPPORTED_FORM'))

    def test_scene_orders_every_object_by_image_x_not_world_x(self):
        objects = [{'size': 'large', 'color': 'red', 'material': 'metal', 'shape': 'cube', 'pixel_coords': [300, 30, 4], '3d_coords': [-10, 0, 0]}, {'size': 'small', 'color': 'blue', 'material': 'rubber', 'shape': 'sphere', 'pixel_coords': [10, 40, 5], '3d_coords': [10, 0, 0]}]
        target, reason = scene_target({'objects': objects})
        self.assertTrue(reason is None and target.index('blue') < target.index('red'))
        objects[0]['pixel_coords'][0] = float('nan')
        self.assertTrue(scene_target({'objects': objects}) == (None, 'UNSUPPORTED_SCENE'))

    def test_chart_source_value_agreement_and_visibility_are_independent(self):
        annotation = {'models': [{'x': ['Alpha', 'Beta'], 'y': ['12.4', '9.1']}], 'type': 'v_bar', 'general_figure_info': {'y_axis': {'label': {'text': 'Percentage'}}}}
        table = 'category,value\nAlpha,12.4\nBeta,9.1\n'
        receipt = {'engine_type': 'NON_LLM_OCR', 'status': 'COMPLETE', 'text': '12.4 9.1', 'tokens': ['12.4', '9.1']}
        self.assertTrue(chart_source('123.png') == 'PEW')
        self.assertTrue(chart_target(table, annotation, imgname='123.png')[1] == 'PEW_NO_SOURCE_TABLE')
        self.assertTrue(chart_target(table, annotation, imgname='two_col_a.png')[1] == 'VISIBILITY_UNVERIFIED')
        target, reason = chart_target(table, annotation, imgname='two_col_a.png', visibility_receipt=receipt)
        self.assertTrue(reason is None and 'Alpha: 12.4' in target)
        self.assertTrue(chart_target(table.replace('12.4', '12.5'), annotation, imgname='two_col_a.png', visibility_receipt=receipt)[1] == 'TABLE_ANNOTATION_MISMATCH')
        self.assertTrue(chart_target(table, annotation, imgname='two_col_a.png', visibility_receipt={**receipt, 'text': '12.4', 'tokens': ['12.4']})[1] == 'VISIBILITY_UNVERIFIED')

    def test_row_rejects_mutations_0(self):
        field = 'target'
        value = 'invented visual fact'
        item = graph_item()
        row = observation_row(item, layer='graph')
        validate_content_row(row, item)
        row[field] = value
        with self.assertRaisesRegex(ValueError, 'ROW_BINDING_MISMATCH'):
            validate_content_row(row, item)

    def test_row_rejects_mutations_1(self):
        field = 'prompt'
        value = 'Question: solve'
        item = graph_item()
        row = observation_row(item, layer='graph')
        validate_content_row(row, item)
        row[field] = value
        with self.assertRaisesRegex(ValueError, 'ROW_BINDING_MISMATCH'):
            validate_content_row(row, item)

    def test_row_rejects_mutations_2(self):
        field = 'source'
        value = 'other'
        item = graph_item()
        row = observation_row(item, layer='graph')
        validate_content_row(row, item)
        row[field] = value
        with self.assertRaisesRegex(ValueError, 'ROW_BINDING_MISMATCH'):
            validate_content_row(row, item)

    def test_row_rejects_mutations_3(self):
        field = 'route'
        value = 'LM'
        item = graph_item()
        row = observation_row(item, layer='graph')
        validate_content_row(row, item)
        row[field] = value
        with self.assertRaisesRegex(ValueError, 'ROW_BINDING_MISMATCH'):
            validate_content_row(row, item)

    def test_row_rejects_mutations_4(self):
        field = 'image_sha256'
        value = 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'
        item = graph_item()
        row = observation_row(item, layer='graph')
        validate_content_row(row, item)
        row[field] = value
        with self.assertRaisesRegex(ValueError, 'ROW_BINDING_MISMATCH'):
            validate_content_row(row, item)

    def test_row_rejects_mutations_5(self):
        field = 'target_origin'
        value = 'self'
        item = graph_item()
        row = observation_row(item, layer='graph')
        validate_content_row(row, item)
        row[field] = value
        with self.assertRaisesRegex(ValueError, 'ROW_BINDING_MISMATCH'):
            validate_content_row(row, item)

    def test_verified_receipt_does_not_admit_forged_target_or_changed_annotation(self):
        item = graph_item()
        row = observation_row(item, layer='graph')
        row['target'] = 'invented'
        row['verification_receipt']['target_sha256'] = text_sha256('invented')
        with self.assertRaises(ValueError):
            validate_content_row(row, item)
        item['annotation']['structure_text'] = '5 2\n0 1\n1 4'
        item['visible_nodes'] = [0, 1, 4]
        with self.assertRaises(ValueError):
            validate_content_row(row, item)

    def test_heldout_rows_diagnostic_only_and_split_overlap_rejected(self):
        item = graph_item()
        row = observation_row(item, layer='graph')
        heldout = {**deepcopy(item), 'problem_id': 'heldout', 'split': 'dev'}
        with self.assertRaisesRegex(ValueError, 'NON_TRAIN_ROW'):
            observation_row(heldout, layer='graph')
        diagnostic = observation_row(heldout, layer='graph', purpose='diagnostic')
        validate_content_row(diagnostic, heldout, purpose='diagnostic')
        with self.assertRaises(ValueError):
            validate_content_row(diagnostic, heldout)
        manifest = {'schema': MANIFEST_SCHEMA, 'prompts_sha256': PROMPTS_SHA256, 'items': [item, heldout], 'rows': [row]}
        with self.assertRaisesRegex(ValueError, 'SPLIT_OVERLAP'):
            validate_content_manifest(manifest)

    def test_manifest_pins_file_bytes_and_rejects_changed_image(self):
        tmp_path = self.tmp_path
        item = graph_item()
        image = tmp_path / 'figure.png'
        image.write_bytes(b'synthetic image bytes')
        item['image_sha256'] = hashlib.sha256(image.read_bytes()).hexdigest()
        source = tmp_path / 'connectivity-easy-graph1.txt'
        source.write_text(item['annotation']['structure_text'])
        item['provenance']['annotation_sources'] = [{'path': 'connectivity-easy-graph1.txt', 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'role': 'graph_structure'}]
        review = tmp_path / 'review.json'
        review.write_text(json.dumps({'version': 'v16-graph-rendering-task-spotcheck-v1', 'human_verified': False, 'samples': [{'task': 'connectivity', 'directed': False, 'weighted': False}]}))
        item['provenance']['annotation_sources'].append({'path': 'review.json', 'sha256': hashlib.sha256(review.read_bytes()).hexdigest(), 'role': 'graph_rendering_review'})
        parent = {'problem_id': item['parent_id'], 'status': 'INCLUDED', 'split': 'train', 'source': 'hf-gvlqa', 'native_id': item['group_id'], 'source_revision': item['source_revision'], 'image_sha256': [item['image_sha256']]}
        parent_path = tmp_path / 'parents.json'
        bind_parent_qa(item, parent, tmp_path)
        parent_path.write_text(json.dumps({'items': [parent]}))
        item['provenance'].update(parent_manifest_path='parents.json', parent_manifest_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(), parent_record_sha256=digest(parent))
        manifest = {'schema': MANIFEST_SCHEMA, 'prompts_sha256': PROMPTS_SHA256, 'items': [item], 'rows': [observation_row(item, layer='graph')]}
        path = tmp_path / 'manifest.json'
        path.write_text(json.dumps(manifest))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertTrue(load_content_manifest(path, sha) == manifest)
        for key, changed in [("question", "stale question"), ("choices", ["other"]),
                             ("gold", "No"), ("answer_type", "decimal"), ("accept", {"eps": 1})]:
            for nested in (False, True):
                with self.subTest(key=key, nested=nested):
                    mutated = deepcopy(manifest)
                    view = mutated["items"][0]["evaluation"] if nested else mutated["items"][0]
                    view[key] = changed
                    mutated["rows"] = [observation_row(mutated["items"][0], layer="graph")]
                    path.write_text(json.dumps(mutated))
                    with self.assertRaisesRegex(ValueError, "EVALUATION_BINDING_MISMATCH"):
                        load_content_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())
        path.write_text(json.dumps(manifest))
        image.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'SOURCE_FILE_HASH_MISMATCH'):
            load_content_manifest(path, sha)
        with self.assertRaisesRegex(ValueError, 'MANIFEST_HASH_MISMATCH'):
            load_content_manifest(path, '0' * 64)

    def test_geometry_loader_rebuilds_candidates_and_checks_parent_records(self):
        tmp_path = self.tmp_path
        image_bytes = b'synthetic original PNG bytes'
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        forms = ['Equals(LengthOf(Line(A, B)), 7)']
        raw_problem = {'choices': ['7', '8'], 'answer': 'A', 'annotat_text': 'Find $AB$.', 'problem_text': 'Find AB.'}
        problems, logic = (tmp_path / 'train.zip', tmp_path / 'logic_forms.zip')
        with zipfile.ZipFile(problems, 'w') as z:
            z.writestr('train/99/data.json', json.dumps(raw_problem))
            z.writestr('train/99/img_diagram.png', image_bytes)
        with zipfile.ZipFile(logic, 'w') as z:
            z.writestr('logic_forms/diagram_logic_forms_annot.json', json.dumps({'99': {'diagram_logic_forms': forms}}))
        item = graph_item()
        item.update(source_repo='hiyouga/geometry3k', group_id=image_sha, image_sha256=image_sha, source_image_sha256=image_sha, annotation_process_documented=False, join_method='image-answer-text-v1')
        annotation = {'local': {'index': 5, 'image_sha256': image_sha, 'answer': '7', 'problem': 'Find AB.'}, 'candidates': {'99': {'image_sha256': image_sha, 'answer_value': '7', 'texts': ['Find $AB$.', 'Find AB.'], 'diagram_logic_forms': forms}}, 'matched_id': '99'}
        item.update(annotation=annotation, join_receipt_sha256=digest(annotation))
        (tmp_path / 'figure.png').write_bytes(image_bytes)
        raw_records = {'actor': {'problem_id': item['parent_id'], 'question': '<image> Find AB.'}, 'verifier': {'problem_id': item['parent_id'], 'gold': '7'}}
        record_sha = {}
        for role, record in raw_records.items():
            line = json.dumps(record)
            (tmp_path / f'{role}.jsonl').write_text(line + '\n')
            item['provenance'][f'{role}_path'] = f'{role}.jsonl'
            record_sha[role] = text_sha256(line)
        parent = {'problem_id': item['parent_id'], 'status': 'INCLUDED', 'split': 'train', 'source': 'hf-geometry3k', 'source_revision': item['source_revision'], 'image_sha256': [image_sha], 'record_sha256': record_sha}
        parent_path = tmp_path / 'parents.json'
        bind_parent_qa(item, parent, tmp_path)
        parent_path.write_text(json.dumps({'items': [parent]}))
        item['provenance'].update(parent_manifest_path='parents.json', parent_manifest_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(), parent_record_sha256=digest(parent), annotation_sources=[{'path': str(file), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest(), 'role': role} for file, role in [(problems, 'geometry_problem_zip'), (logic, 'geometry_logic_zip')]])
        manifest = {'schema': MANIFEST_SCHEMA, 'prompts_sha256': PROMPTS_SHA256, 'items': [item], 'rows': [observation_row(item, layer='geometry')]}
        path = tmp_path / 'manifest.json'
        path.write_text(json.dumps(manifest))
        self.assertTrue(load_content_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest()) == manifest)
        item['annotation']['candidates']['99']['diagram_logic_forms'] = ['Equals(LengthOf(Line(A, B)), 8)']
        item['join_receipt_sha256'] = digest(item['annotation'])
        manifest['rows'] = [observation_row(item, layer='geometry')]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'ANNOTATION_SOURCE_MISMATCH'):
            load_content_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_scene_loader_uses_question_index_not_question_position(self):
        tmp_path = self.tmp_path
        item = graph_item()
        image_bytes = b'synthetic scene image'
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        image_name = 'CLEVR_train_000003.png'
        scene = {'split': 'train', 'image_index': 3, 'image_filename': image_name, 'objects': [{'size': 'small', 'color': 'blue', 'material': 'rubber', 'shape': 'sphere', 'pixel_coords': [10, 30, 4]}]}
        item.update(source_repo='HuggingFaceM4/clevr', image_sha256=image_sha, image_filename=image_name, group_id=image_name, annotation=scene, split='sentinel')
        (tmp_path / 'figure.png').write_bytes(image_bytes)
        questions = {'questions': [{'question_index': 99, 'image_filename': image_name, 'image_index': 3, 'question': 'What shape is the object?', 'answer': 'sphere'}]}
        sources = [('scene.json', 'scene_json', {'scenes': [scene]}), ('questions.json', 'scene_questions_json', questions)]
        item['provenance']['annotation_sources'] = []
        for name, role, document in sources:
            file = tmp_path / name
            file.write_text(json.dumps(document))
            item['provenance']['annotation_sources'].append({'path': name, 'role': role, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
        hashes = {}
        for role, extra in [('actor', {'question': 'What shape is the object?'}), ('verifier', {'gold': 'sphere'})]:
            line = json.dumps({'problem_id': item['parent_id'], **extra})
            (tmp_path / f'{role}.jsonl').write_text(line + '\n')
            item['provenance'][f'{role}_path'] = f'{role}.jsonl'
            hashes[role] = text_sha256(line)
        parent = {'problem_id': item['parent_id'], 'status': 'INCLUDED', 'split': 'train', 'source': 'hf-clevr', 'native_id': 'train:99', 'record_sha256': hashes, 'source_revision': item['source_revision'], 'image_sha256': [image_sha]}
        parent_path = tmp_path / 'parents.json'
        bind_parent_qa(item, parent, tmp_path)
        parent_path.write_text(json.dumps({'items': [parent]}))
        item['provenance'].update(parent_manifest_path='parents.json', parent_record_sha256=digest(parent), parent_manifest_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest())
        train_item = graph_item()
        train_item.update(problem_id='train-graph', parent_id='train-graph')
        manifest = {'schema': MANIFEST_SCHEMA, 'prompts_sha256': PROMPTS_SHA256, 'items': [item], 'rows': []}
        item['split'] = 'train'
        manifest['rows'] = [observation_row(item, layer='scene')]
        path = tmp_path / 'manifest.json'
        path.write_text(json.dumps(manifest))
        self.assertTrue(load_content_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest()) == manifest)
        item['image_filename'] = 'CLEVR_train_000004.png'
        item['group_id'] = item['image_filename']
        item['annotation']['image_filename'] = item['image_filename']
        manifest['rows'] = [observation_row(item, layer='scene')]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'ANNOTATION_SOURCE_MISMATCH'):
            load_content_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())
    def test_multiseries_chart_joins_reordered_rows_and_columns_without_unit_loss(self):
        annotation = {"type": "v_bar", "models": [
            {"name": "North", "x": ["Alpha", "Beta"], "y": ["1.0%", "2%"]},
            {"name": "South", "x": ["Beta", "Alpha"], "y": ["4%", "3%"]}],
            "general_figure_info": {"y_axis": {"label": {"text": "Share in percent"}}}}
        table = "category,South,North\nBeta,4%,2.00%\nAlpha,3%,1%\n"
        receipt = {"engine_type": "NON_LLM_OCR", "status": "COMPLETE", "text": "1% 2% 3% 4%"}
        target, reason = chart_target(table, annotation, imgname="multi_col_fixture.png",
                                     visibility_receipt=receipt)
        self.assertIsNone(reason)
        self.assertIn("Series: North.", target)
        self.assertIn("Series: South.", target)
        self.assertIn("Alpha: 1.0%", target)
        self.assertEqual(chart_target(table.replace("2.00%", "2.00"), annotation,
                                      imgname="multi_col_fixture.png",
                                      visibility_receipt=receipt)[1], "TABLE_ANNOTATION_MISMATCH")
        wrong_names = deepcopy(annotation)
        wrong_names["models"][0]["name"] = "South"
        self.assertEqual(chart_target(table, wrong_names, imgname="multi_col_fixture.png",
                                      visibility_receipt=receipt)[1], "TABLE_ANNOTATION_MISMATCH")
        self.assertEqual(chart_target(table, annotation, imgname="multi_col_fixture.png",
                                      visibility_receipt={**receipt, "text": "1% 2% 3%"})[1],
                         "VISIBILITY_UNVERIFIED")

    def test_horizontal_chart_retains_creator_axis_label(self):
        annotation = {"type": "h_bar", "models": [{"name": {}, "x": ["Alpha"], "y": ["15%"]}],
                      "general_figure_info": {"y_axis": {"label": {"text": "Share of imports"}}}}
        target, reason = chart_target("category,Share\nAlpha,15%\n", annotation,
                                     imgname="two_col_fixture.png",
                                     visibility_receipt={"engine_type": "NON_LLM_OCR",
                                                         "status": "COMPLETE", "text": "15%"})
        self.assertIsNone(reason)
        self.assertIn("Axis labels: Share of imports.", target)


    def test_human_j_requires_an_authenticated_source_decoder(self):
        manifest = {"items": [graph_item()]}
        capability = validate_human_j_capability(manifest)
        self.assertEqual(capability["status"], "HOLD")
        self.assertEqual(capability["eligible_parent_ids"], [])
        with self.assertRaisesRegex(ValueError, "J_CAPABILITY_HOLD"):
            validate_human_j_capability(manifest, requested_weight=0.1)
        manifest["items"][0]["human_solution"] = {
            "dataset_authored": True, "ai_generated": False, "solution": "injected explanation",
            "solution_sha256": text_sha256("injected explanation")}
        with self.assertRaisesRegex(ValueError, "J_CAPABILITY_HOLD"):
            validate_human_j_capability(manifest, requested_weight=1.0)
        for invalid in (-1, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "INVALID_J_WEIGHT"):
                validate_human_j_capability(manifest, requested_weight=invalid)


if __name__ == '__main__':
    unittest.main()
