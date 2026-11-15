"""Synthetic v16 contracts only: no pretrained Qwen, benchmark data, or effect claims."""
from copy import deepcopy
import json
import unittest
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from fine_tuning.error_note.audit import digest, text_sha256
from fine_tuning.error_note.self_note_v16 import FORMULAS, build_note_prompt, build_problem_contract, build_resolve_prompt, build_self_note_rows, create_note_yield_gate, earlier_assertion_status, first_solution_row, human_j_row, measure_note_yield, public_note_catalog, validate_self_note_row, verify_note, verify_retry

def graph_item():
    return {'problem_id': 'synthetic-graph:1', 'parent_id': 'synthetic-graph:1', 'group_id': 'connectivity-easy-graph1', 'source_repo': 'Yanbin99/GVLQA-BASE', 'source_revision': 'a' * 40, 'split': 'train', 'original_split': 'train', 'training_allowed': True, 'label_source': 'PROGRAM_EXACT', 'image_sha256': 'b' * 64, 'image_path': 'figure.png', 'visible_nodes': [0, 1, 3], 'annotation': {'task': 'connectivity', 'level': 'easy', 'graph_id': 'graph1', 'directed': False, 'weighted': False, 'structure_text': '5 2 7\n0 1\n1 3\n3 4\n'}, 'provenance': {'parent_manifest_sha256': 'c' * 64, 'parent_record_sha256': 'd' * 64, 'parent_status': 'INCLUDED', 'rendering_verified': True, 'annotation_sources': [{'path': 'source.json', 'sha256': 'e' * 64}]}}

class NativeBytes:
    all_special_tokens = ['<|im_end|>']
    unk_token_id = None
    byte_encoder = bytes_to_unicode()

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode('utf-8'))

    def convert_tokens_to_ids(self, token):
        assert token == '<|im_end|>'
        return 256

    def convert_ids_to_tokens(self, ids):
        return [self.all_special_tokens[0] if value == 256 else self.byte_encoder[value] for value in ids]
TOKENIZER = NativeBytes()

def output(text, *, status='CORRECT', extracted='4'):
    """Key set of a real stream record (collect.py:1245-1253): completion_tokens only on the outer dict."""
    actions = TOKENIZER.encode(text) + [256]
    generation = {'prompt_token_ids': [1, 2, 3], 'completion_token_ids': actions, 'sampling_logprobs': [-0.5] * len(actions),
                  'support_sha256': [digest([a]) for a in actions], 'support_token_ids': [[a] for a in actions],
                  'sampling': {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 1.5},
                  'policy_sha256': 'b' * 64, 'protocol_sha256': 'c' * 64, 'sampling_sha256': '1' * 64,
                  'input_sha256': '2' * 64, 'finish_reason': 'stop', 'request_sha256': '3' * 64, 'text': text,
                  'eos_token_ids': [256], 'completion_count': 1}
    return {'text': text, 'finish_reason': 'stop', 'prompt_tokens': 3, 'completion_tokens': len(actions), 'seconds': 0.0,
            'completion_count': 1, 'generation': generation,
            'status': status, 'extracted': extracted, 'prompt': 'actual generation prompt'}

def fixture():
    facts = [{'fact_id': ref, 'kind': 'OBSERVATION', 'entity': ref, 'evidence_origin': 'synthetic-source:' + ref, 'value': value, 'unit': 'count'} for ref, value in (('f0', '12'), ('f1', '3'))]
    rules = {'DIVIDE_EXACT': {'operation': 'DIV', 'formula': FORMULAS['DIV'], 'operand_units': ['count', 'count'], 'result_unit': 'dimensionless'}}
    note = {'observed_facts': [{'fact_ref': 'f0', 'value': '12'}, {'fact_ref': 'f1', 'value': '3'}], 'error': {'earlier_span': [0, 10], 'earlier_quote': '12 / 3 = 5', 'kind': 'ARITHMETIC', 'operand_fact_refs': ['f0', 'f1'], 'operation': 'DIV', 'old_value': '5', 'corrected_value': '4', 'rounding': 'none', 'unit': 'dimensionless', 'rule_id': 'DIVIDE_EXACT'}, 'lesson': {'rule_id': 'DIVIDE_EXACT', 'formula': FORMULAS['DIV']}}
    record = {'problem_id': 'synthetic:1', 'parent_id': 'parent:1', 'source': 'creator', 'subject': 'counting', 'layer': 'scene', 'image_paths': ['image.png'], 'image_sha256': ['a' * 64], 'p0_text': 'Question: Divide the displayed counts.', 't1': output('12 / 3 = 5\nFinal answer: 5', status='INCORRECT', extracted='5'), 'note': output(json.dumps(note)), 'resolve': output('12 / 3 = 4\nConclusion: 4\nFinal answer: 4')}
    record.update(experiment_sha256='d' * 64, content_manifest_sha256='e' * 64, protocol_sha256='c' * 64, annotation_sha256='f' * 64, actor_sha256='a' * 64)
    return (record, facts, rules, note)

def verified(record, facts, rules, gold='4', choices=()):
    return verify_note(record, fact_catalog=facts, rules=rules, choices=choices, gold=gold)

def yield_evidence(record, facts, rules, gold='4', choices=()):
    return record | {'note_receipt': verified(record, facts, rules, gold, choices), 'note_validation': {'fact_catalog': facts, 'rules': rules, 'gold': gold, 'choices': choices}}

def with_note(record, note):
    return record | {'note': output(json.dumps(note))}

class SelfNoteV16Contracts(unittest.TestCase):

    def test_note_is_independent_of_retry_and_intermediate_values_need_not_equal_gold(self):
        record, facts, rules, _ = fixture()
        receipt = verified(record, facts, rules)
        assert receipt['accepted'] and receipt['earlier_assertion_status'] == 'ASSERTED'
        # The quoted step produced T1's 'Final answer: 5', so its correction is an answer claim.
        assert receipt['quote_is_final_computation'] and receipt['answer_specific_values'] == ['4']
        record.pop('resolve')
        assert verified(record, facts, rules) == receipt
        record['t1']['status'] = 'CORRECT'
        assert verified(record, facts, rules)['reason'] == 'SKIP_NOTE_NO_ERROR'
        with self.assertRaisesRegex(ValueError, 'SKIP_NOTE_NO_ERROR'):
            build_note_prompt(record, fact_catalog=facts, rules=rules, gold='4')

    def test_note_rejects_schema_binding_and_value_errors(self):
        for edit, reason in [(lambda n: n.update(extra='unbounded prose'), 'NOTE_SCHEMA'), (lambda n: n['error'].update(extra='explanation'), 'NOTE_SCHEMA'), (lambda n: n['observed_facts'][0].update(value='11'), 'OBSERVED_FACT_MISMATCH'), (lambda n: n['error'].update(operand_fact_refs=['missing', 'f1']), 'OPERAND_BINDING_OR_UNIT'), (lambda n: n['error'].update(unit='metre'), 'OPERAND_BINDING_OR_UNIT'), (lambda n: n['error'].update(corrected_value='4.01'), 'CORRECTION_MISMATCH'), (lambda n: n['error'].update(earlier_span=[1, 11]), 'EARLIER_ASSERTION_UNKNOWN'), (lambda n: n['lesson'].update(formula='result=4'), 'LESSON_RULE_MISMATCH'), (lambda n: n['error'].update(old_value='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabc'), 'ATOMIC_VALUE_REQUIRED')]:
            with self.subTest(edit=edit, reason=reason):
                record, facts, rules, note = fixture()
                edit(note)
                assert verified(with_note(record, note), facts, rules)['reason'] == reason

    def test_negated_hypothetical_quoted_retracted_and_unknown_assertions(self):
        # Neutral earlier/later prose does not block an in-place assertion (v16 D3); a hypothetical lead-in does.
        for prefix, suffix, status in [('If ', ', suppose this were true', 'UNKNOWN'), ('"', '"', 'UNKNOWN'), ('', ' is incorrect; 12 / 3 = 4', 'RETRACTED'), ('', '\nCorrection: 12 / 3 = 4', 'RETRACTED'), ('Example: ', '', 'UNKNOWN'), ('For a deliberately wrong example, consider:\n', '', 'UNKNOWN'), ('Step 1: compute the ratio:\n', '', 'ASSERTED'), ('', '\nPerhaps it follows', 'ASSERTED'), ('', ', actually 12 / 3 = 4', 'RETRACTED'), ('', 'x', 'UNKNOWN')]:
            with self.subTest(prefix=prefix, suffix=suffix, status=status):
                quote = '12 / 3 = 5'
                text = prefix + quote + suffix
                assert earlier_assertion_status(text, [len(prefix), len(prefix) + len(quote)], quote) == status

    def test_native_spans_mask_wrong_quote_and_old_value_and_keep_original_tokens(self):
        record, facts, rules, _ = fixture()
        receipts = []
        for index in range(32):
            other = deepcopy(record)
            other['problem_id'] = f'synthetic:{index}'
            receipts.append(yield_evidence(other, facts, rules))
        gate = measure_note_yield(receipts)
        original = deepcopy(record)
        rows, decisions = build_self_note_rows(record, tokenizer=TOKENIZER, fact_catalog=facts, rules=rules, choices=(), gold='4', note_yield_gate=gate)
        assert [row['kind'] for row in rows] == ['C', 'N']
        assert record == original
        for row in rows:
            assert row['generation'] is record[{'C': 'resolve', 'N': 'note'}[row['kind']]]['generation']
            assert not row['loss_mask'][-1] and row['route'] == 'FULL'
            validate_self_note_row(row, record, tokenizer=TOKENIZER, fact_catalog=facts, rules=rules, choices=(), gold='4', note_yield_gate=gate)
        note_row = rows[1]
        selected = bytes((token for token, active in zip(note_row['generation']['completion_token_ids'], note_row['loss_mask']) if active)).decode()
        assert '12 / 3 = 5' not in selected and '"5"' not in selected
        assert '"corrected_value": "4"' in selected
        assert decisions['C']['verified_evidence_count'] == 1
        forged = note_row | {'target': note_row['target'] + ' changed'}
        with self.assertRaisesRegex(ValueError, 'SELF_NOTE_ROW_BINDING_MISMATCH'):
            validate_self_note_row(forged, record, tokenizer=TOKENIZER, fact_catalog=facts, rules=rules, choices=(), gold='4', note_yield_gate=gate)

    def test_yield_gate_is_closed_before_32_distinct_wrong_t1_and_rejects_forgery(self):
        record, facts, rules, _ = fixture()
        rows, decisions = build_self_note_rows(record, tokenizer=TOKENIZER, fact_catalog=facts, rules=rules, choices=(), gold='4')
        assert [row['kind'] for row in rows] == ['C']
        assert decisions['N']['reason'] == 'NOTE_YIELD_NOT_PASSED'
        records = []
        for index in range(32):
            other = deepcopy(record)
            other['problem_id'] = f'synthetic:{index}'
            if index >= 4:
                other.pop('note')
            records.append(yield_evidence(other, facts, rules))
        assert create_note_yield_gate(records[:31])['status'] == 'HOLD'
        assert create_note_yield_gate(records)['yield'] == 0.125
        assert create_note_yield_gate(records)['status'] == 'PASS'
        assert create_note_yield_gate(records[1:])['status'] == 'HOLD'
        with self.assertRaisesRegex(ValueError, 'INVALID_NOTE_YIELD_RECEIPT'):
            create_note_yield_gate(records + records[:1])
        records[0]['note_receipt']['accepted'] = False
        with self.assertRaisesRegex(ValueError, 'INVALID_NOTE_YIELD_RECEIPT'):
            create_note_yield_gate(records)

    def test_note_yield_gate_cannot_mix_experiments_or_replace_failed_initial_sample(self):
        record, facts, rules, _ = fixture()
        records = []
        for index in range(40):
            other = deepcopy(record)
            other['problem_id'] = f'synthetic:{index}'
            if index < 30:
                other.pop('note')
            records.append(yield_evidence(other, facts, rules))
        assert create_note_yield_gate(records)['status'] == 'FAIL'
        changed = deepcopy(record)
        changed['problem_id'] = 'other-experiment'
        changed['experiment_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'MIXED_NOTE_YIELD_EXPERIMENTS'):
            measure_note_yield([records[-1], yield_evidence(changed, facts, rules)])
        unbound = deepcopy(record)
        unbound.pop('actor_sha256')
        assert verified(unbound, facts, rules)['reason'] == 'NOTE_IDENTITY_NOT_BOUND'
        with self.assertRaisesRegex(ValueError, 'INVALID_NOTE_YIELD_RECEIPT'):
            measure_note_yield([yield_evidence(unbound, facts, rules)])

    def test_changed_generation_or_forged_note_receipt_never_admits_retry(self):
        record, facts, rules, _ = fixture()
        receipt = verified(record, facts, rules)
        receipt['lesson']['formula'] = 'result=4'
        assert verify_retry(record, note_receipt=receipt, fact_catalog=facts, rules=rules, choices=(), gold='4')['reason'] == 'NOTE_NOT_VERIFIED'
        record['note']['generation']['text'] += ' changed'
        assert verified(record, facts, rules)['reason'] == 'CHANGED_ORIGINAL_GENERATION'

    def test_retry_rejects_wrong_evidence_unknown_or_mismatched_conclusions_and_multiple_answers(self):
        for text, reason in [('12 / 3 = 5\nConclusion: 4\nFinal answer: 4', 'CONTENT_CONTRADICTION'), ('12 / 3 = 4\nConclusion: 5\nFinal answer: 4', 'REASONING_ANSWER_MISMATCH'), ('12 / 3 = 4\nConclusion: 5\nConclusion: 4\nFinal answer: 4', 'REASONING_ANSWER_MISMATCH'), ('12 / 3 = 4\nConclusion: perhaps four\nFinal answer: 4', 'REASONING_ANSWER_UNKNOWN'), ('I think that it is correct.\nConclusion: 4\nFinal answer: 4', 'ANSWER_ONLY_C'), ('12 / 3 = 4\nConclusion: 4\nFinal answer: 4\nFinal answer: 4', 'MULTIPLE_FINAL_ANSWERS'), ('Since 12 / 3 = 5, continue.\nConclusion: 4\nFinal answer: 4', 'UNSUPPORTED_CONTENT_CLAIM')]:
            with self.subTest(text=text, reason=reason):
                record, facts, rules, _ = fixture()
                note = verified(record, facts, rules)
                record['resolve'] = output(text)
                receipt = verify_retry(record, note_receipt=note, fact_catalog=facts, rules=rules, choices=(), gold='4')
                assert receipt['reason'] == reason

    def test_retry_has_no_256_token_floor_and_no_best_of_n(self):
        record, facts, rules, _ = fixture()
        assert len(record['resolve']['generation']['completion_token_ids']) < 256
        note = verified(record, facts, rules)
        assert verify_retry(record, note_receipt=note, fact_catalog=facts, rules=rules, choices=(), gold='4')['accepted']
        record['resolve2'] = record['resolve']
        assert verify_retry(record, note_receipt=note, fact_catalog=facts, rules=rules, choices=(), gold='4')['reason'] == 'MULTIPLE_RETRIES'

    def test_decimal_correction_is_recomputed_and_only_answer_specific_values_compare_to_gold(self):
        record, facts, rules, note = fixture()
        rules['DIVIDE_EXACT']['answer_specific'] = True
        assert verified(record, facts, rules)['accepted']
        assert verified(record, facts, rules, gold='4.01')['reason'] == 'NOTE_GOLD_CONTRADICTION'
        record['resolve']['extracted'] = '5'
        _, decisions = build_self_note_rows(record, tokenizer=TOKENIZER, fact_catalog=facts, rules=rules, choices=(), gold='4')
        assert decisions['N']['reason'] == 'NOTE_RESOLVE_CONTRADICTION'
        note['error']['corrected_value'] = '4.00000000001'
        assert verified(with_note(record, note), facts, rules)['reason'] == 'CORRECTION_MISMATCH'

    def test_decimal_rounding_regression_rejects_a_wrong_note_even_when_retry_is_right(self):
        record, facts, rules, note = fixture()
        facts[0]['value'], facts[1]['value'] = ('2858.4', '70')
        quote = '2858.4 / 70 = 40.84'
        record['t1'] = output(quote + '\nFinal answer: 40.84', status='INCORRECT', extracted='40.84')
        record['resolve'] = output('2858.4 / 70 = 40.83\nConclusion: 40.83\nFinal answer: 40.83', extracted='40.83')
        note['observed_facts'][0]['value'], note['observed_facts'][1]['value'] = ('2858.4', '70')
        note['error'].update(earlier_span=[0, len(quote)], earlier_quote=quote, old_value='40.84', corrected_value='40.84', rounding='decimal:2')
        rules['DIVIDE_EXACT'].update(rounding='decimal:2', answer_specific=True)
        assert verified(with_note(record, note), facts, rules, gold='40.83')['reason'] == 'CORRECTION_MISMATCH'
        note['error']['corrected_value'] = '40.83'
        assert verified(with_note(record, note), facts, rules, gold='40.83')['accepted']

    def test_zero_division_unit_mismatch_and_unestablished_geometry_premise_fail_closed(self):
        record, facts, rules, _ = fixture()
        facts[1]['value'] = '0'
        assert not verified(record, facts, rules)['accepted']
        rules['TRIANGLE'] = {'operation': 'TRIANGLE_REMAINDER', 'formula': FORMULAS['TRIANGLE_REMAINDER'], 'operand_units': ['degree', 'degree'], 'result_unit': 'degree'}
        assert verified(record, facts, rules)['reason'] == 'RULE_PRECONDITION_MISSING'

    def test_note_catalog_and_resolve_projection_exclude_private_source_values(self):
        record, facts, rules, _ = fixture()
        catalog = public_note_catalog(facts, rules)
        assert all((set(fact) == {'fact_id', 'kind', 'entity', 'evidence_origin'} for fact in catalog['facts']))
        prompt = build_note_prompt(record, fact_catalog=facts, rules=rules, gold='4')
        assert '"given_answer": "4"' in prompt
        receipt = verified(record, facts, rules)
        retry_prompt = build_resolve_prompt(record['p0_text'], receipt['lesson'])
        assert '12' not in retry_prompt and 'corrected_value' not in retry_prompt and ('given_answer' not in retry_prompt)

    def test_problem_contract_uses_verified_source_annotation_not_all_values_as_gold(self):
        item = graph_item() | {'layer': 'graph', 'gold': 'yes'}
        contract = build_problem_contract(item)
        assert contract['supported'] and contract['expected_final'] == 'yes'
        facts = {fact['fact_id']: fact for fact in contract['fact_catalog']}
        assert facts['visible_node_count']['value'] == '3' and facts['edge_count']['value'] == '2'
        assert not any((fact['answer_specific'] for fact in facts.values()))
        item['annotation']['structure_text'] = 'bad'
        assert not build_problem_contract(item)['supported']

    def test_symbolic_geometry_label_is_never_truncated_into_a_numeric_fact(self):
        local = {'index': 7, 'image_sha256': 'b' * 64, 'answer': '4', 'problem': 'Find side CD'}
        candidates = {'native': {'image_sha256': 'b' * 64, 'answer_value': '4', 'texts': ['Find side CD'], 'diagram_logic_forms': ['Equals(LengthOf(Line(A, B)), 2x+3)', 'Equals(LengthOf(Line(C, D)), 4)']}}
        annotation = {'local': local, 'candidates': candidates, 'matched_id': 'native'}
        item = graph_item() | {'layer': 'geometry', 'gold': '4', 'annotation': annotation, 'group_id': 'b' * 64, 'source_image_sha256': 'b' * 64, 'annotation_process_documented': False, 'join_method': 'image-answer-text-v1', 'join_receipt_sha256': digest(annotation)}
        contract = build_problem_contract(item)
        assert contract['supported']
        assert [(fact['entity'], fact['value']) for fact in contract['fact_catalog']] == [('segment CD', '4')]

    def test_cloned_rawless_receipts_cannot_certify_note_yield(self):
        record, facts, rules, _ = fixture()
        receipt = verified(record, facts, rules)
        clones = []
        for index in range(32):
            clone = receipt | {'problem_id': str(index)}
            clone['sha256'] = digest({key: value for key, value in clone.items() if key != 'sha256'})
            clones.append(clone)
        with self.assertRaisesRegex(ValueError, 'RAW_NOTE_YIELD_EVIDENCE_REQUIRED'):
            measure_note_yield(clones)

    def test_human_j_is_dataset_authored_has_no_eos_and_checks_baseline_style(self):
        record, _, _, _ = fixture()
        target = 'The two counts give a quotient of four.'
        fields = {'solution': target, 'dataset_authored': True, 'ai_generated': False, 'solution_sha256': text_sha256(target)}
        row, reason = human_j_row(record, fields, tokenizer=TOKENIZER, baseline_correct_length_median=60)
        assert reason is None and row['target'] == target and (row['target_origin'] == 'human')
        assert row['supervise_end'] is False and 256 not in row['target_token_ids']
        assert human_j_row(record, fields, tokenizer=TOKENIZER, baseline_correct_length_median=100)[1] == 'TARGET_STYLE_SHIFT'
        assert human_j_row(record, fields, tokenizer=TOKENIZER, baseline_correct_length_median=0)[1] == 'BASELINE_STYLE_NOT_MEASURED'
        fields['ai_generated'] = True
        assert human_j_row(record, fields, tokenizer=TOKENIZER, baseline_correct_length_median=60)[1] == 'J_NOT_DATASET_AUTHORED'

    def test_correct_t1_retains_v15_admission_without_calling_it_verified_reasoning(self):
        record, _, _, _ = fixture()
        prefix = ' '.join((text_sha256(str(index)) for index in range(5)))
        record['t1'] = output(prefix + '\nFinal answer: 4')
        record['t1']['prompt'] = record['p0_text']
        row, reason = first_solution_row(record, tokenizer=TOKENIZER)
        assert reason is None and row['kind'] == 'A1' and (row['target_source'] == 'T1')
        assert row['verification_receipt']['content_status'] == 'UNKNOWN'
        assert row['generation'] is record['t1']['generation'] and (not row['loss_mask'][-1])
        record['t1'] = output('Final answer: 4')
        record['t1']['prompt'] = record['p0_text']
        assert first_solution_row(record, tokenizer=TOKENIZER)[1] == 'TOO_SHORT'
        record['t1'] = output(prefix + '\nFinal answer: 4', status='INCORRECT')
        record['t1']['prompt'] = record['p0_text']
        assert first_solution_row(record, tokenizer=TOKENIZER)[1] == 'NOT_CORRECT'

    def test_real_stream_schema_end_to_end(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from fine_tuning.error_note import train_v16
        record, facts, rules, _ = fixture()
        for stage in ('t1', 'note', 'resolve'):
            assert 'completion_tokens' not in record[stage]['generation']
            assert record[stage]['completion_tokens'] == len(record[stage]['generation']['completion_token_ids'])
        assert verified(record, facts, rules)['accepted']
        for holder in ('outer', 'generation'):
            bad = deepcopy(record)
            target = bad['note'] if holder == 'outer' else bad['note']['generation']
            target['completion_tokens'] = 1
            assert verified(bad, facts, rules)['reason'] == 'CHANGED_ORIGINAL_GENERATION'
        prefix = ' '.join((text_sha256(str(index)) for index in range(5)))
        record['t1'] = output(prefix + '\nFinal answer: 4')
        record['t1']['prompt'] = record['p0_text']
        row, reason = first_solution_row(record, tokenizer=TOKENIZER)
        assert reason is None and row['generation'] is record['t1']['generation']
        generation = row['generation']
        prepared = {'inputs': {'input_ids': torch.tensor([generation['prompt_token_ids']])},
                    'prompt_token_ids': generation['prompt_token_ids'], 'input_sha256': generation['input_sha256']}
        processor, config = SimpleNamespace(tokenizer=TOKENIZER), SimpleNamespace(image_token_id=1000, video_token_id=1001)
        with patch('fine_tuning.error_note.train_v16.prepare_protocol_inputs', return_value=(prepared, None)):
            encoded = train_v16.encode_content_row(processor, config, row, ['image'], image={})
            assert encoded['target_len'] == len(generation['completion_token_ids'])
            assert encoded['input_ids'][-encoded['target_len']:].tolist() == generation['completion_token_ids']
            forged = deepcopy(row)
            forged['generation']['completion_tokens'] = 1
            with self.assertRaisesRegex(ValueError, 'complete original generation'):
                train_v16.encode_content_row(processor, config, forged, ['image'], image={})

    def test_note_correcting_final_to_non_gold_rejected(self):
        local = {'index': 7, 'image_sha256': 'b' * 64, 'answer': '7', 'problem': 'Find x'}
        candidates = {'native': {'image_sha256': 'b' * 64, 'answer_value': '7', 'texts': ['Find x'], 'diagram_logic_forms': ['Equals(LengthOf(Line(A, B)), 12)', 'Equals(LengthOf(Line(C, D)), 3)']}}
        annotation = {'local': local, 'candidates': candidates, 'matched_id': 'native'}
        item = graph_item() | {'layer': 'geometry', 'gold': '7', 'annotation': annotation, 'group_id': 'b' * 64, 'source_image_sha256': 'b' * 64, 'annotation_process_documented': False, 'join_method': 'image-answer-text-v1', 'join_receipt_sha256': digest(annotation)}
        contract = build_problem_contract(item)
        assert contract['supported']
        facts, rules = contract['fact_catalog'], contract['rules']
        assert [(fact['fact_id'], fact['value']) for fact in facts] == [('f0', '12'), ('f1', '3')]
        assert not any(fact['answer_specific'] for fact in facts) and not any(rule.get('answer_specific') for rule in rules.values())
        rule_id = 'DIV_' + text_sha256('length')[:8]
        record, _, _, note = fixture()
        prose, quote = 'So the ratio of the two sides is ', '12 / 3 = 5'
        record['t1'] = output(prose + quote + '\nFinal answer: 5', status='INCORRECT', extracted='5')
        note['error'].update(earlier_span=[len(prose), len(prose) + len(quote)], rule_id=rule_id)
        note['lesson'] = {'rule_id': rule_id, 'formula': FORMULAS['DIV']}
        record = with_note(record, note)
        assert verified(record, facts, rules, gold=contract['expected_final'])['reason'] == 'NOTE_GOLD_CONTRADICTION'
        control = verified(record, facts, rules, gold='4')
        assert control['accepted'] and control['quote_is_final_computation']

    def test_intermediate_correction_not_compared_with_gold(self):
        record, facts, rules, note = fixture()
        record['t1'] = output('12 / 3 = 5\nThe total is 5 + 2 = 7\nFinal answer: 7', status='INCORRECT', extracted='7')
        receipt = verified(with_note(record, note), facts, rules, gold='6')
        assert receipt['accepted'] and receipt['quote_is_final_computation'] is False
        assert receipt['answer_specific_values'] == []

    def test_inline_prose_equation_asserted(self):
        text, quote = 'So the area is 12 * 3 = 35 square units.', '12 * 3 = 35'
        start = text.index(quote)
        assert earlier_assertion_status(text, [start, start + len(quote)], quote) == 'ASSERTED'

    def test_fragment_of_longer_expression_unknown(self):
        quote = '12 / 3 = 5'
        for text in ('2 + 12 / 3 = 5', '12 / 3 = 5.5', '12 / 3 = 5 + 1', '112 / 3 = 5', '12 / 3 = 5x', '12 / 3 = 5%'):
            with self.subTest(text=text):
                start = text.index(quote)
                assert earlier_assertion_status(text, [start, start + len(quote)], quote) == 'UNKNOWN'

    def test_latex_operator_prefix_unknown(self):
        quote = '12 / 3 = 5'
        for text in ('2 \\cdot 12 / 3 = 5', '\\times 12 / 3 = 5', '\\(12 / 3 = 5\\)'):
            with self.subTest(text=text):
                start = text.index(quote)
                assert earlier_assertion_status(text, [start, start + len(quote)], quote) == 'UNKNOWN'

    def test_hypothetical_line_unknown(self):
        text, quote = 'If 12 / 3 = 5 then the total is 10.', '12 / 3 = 5'
        assert earlier_assertion_status(text, [3, 3 + len(quote)], quote) == 'UNKNOWN'

    def test_later_same_lhs_different_result_retracted(self):
        quote = '12 / 3 = 5'
        for text in (quote + '\nSo 12 / 3 = 4 after all.\nFinal answer: 4', quote + '\nThen 12 ÷ 3 = 4.'):
            with self.subTest(text=text):
                assert earlier_assertion_status(text, [0, len(quote)], quote) == 'RETRACTED'
        assert earlier_assertion_status(quote + '\nAgain 12 / 3 = 5.', [0, len(quote)], quote) == 'ASSERTED'

    def test_later_wait_unknown(self):
        quote = '12 / 3 = 5'
        for later in ('Wait, that needs another look.', 'Let me re-evaluate the division.', 'Actually the total is larger.'):
            with self.subTest(later=later):
                text = quote + '\n' + later + '\nFinal answer: 5'
                assert earlier_assertion_status(text, [0, len(quote)], quote) == 'UNKNOWN'
        record, facts, rules, note = fixture()
        record['t1'] = output(quote + '\nWait, that needs another look.\nFinal answer: 5', status='INCORRECT', extracted='5')
        assert verified(with_note(record, note), facts, rules)['reason'] == 'EARLIER_ASSERTION_UNKNOWN'

    def test_separator_question_relation_and_soft_doubt_unknown(self):
        for text, quote in (('The total is 1,200 - 300 = 900', '200 - 300 = 900'), ('12 / 3 = 5,000 units', '12 / 3 = 5'),
                            ('Is 12 / 3 = 5?', '12 / 3 = 5'), ('Perhaps 12 / 3 = 5 here', '12 / 3 = 5'),
                            ('√12 / 3 = 5', '12 / 3 = 5'), ('x ≠ 12 / 3 = 5', '12 / 3 = 5'),
                            ('225 - 986 = -950\\cos(x)', '225 - 986 = -950'), ('12 / 3 = 5(2)', '12 / 3 = 5'),
                            ('Suppose we wrongly compute:\n12 / 3 = 5\nFinal answer: 5', '12 / 3 = 5')):
            with self.subTest(text=text):
                start = text.index(quote)
                assert earlier_assertion_status(text, [start, start + len(quote)], quote) == 'UNKNOWN'

    def test_markdown_final_line_uses_graded_extraction(self):
        # '**Final Answer:** 5' is unreadable for the plain final pattern; the stored graded extraction binds it.
        record, facts, rules, note = fixture()
        record['t1'] = output('12 / 3 = 5\n**Final Answer:** 5', status='INCORRECT', extracted='5')
        record = with_note(record, note)
        assert verified(record, facts, rules, gold='7')['reason'] == 'NOTE_GOLD_CONTRADICTION'
        control = verified(record, facts, rules, gold='4')
        assert control['accepted'] and control['quote_is_final_computation']

    def test_thus_wrong_equation_rejected(self):
        choices = ('3', '4', '5', '7')
        record, facts, rules, _ = fixture()
        record['t1'] = output('12 / 3 = 5\nFinal answer: C', status='INCORRECT', extracted='C')
        note = verified(record, facts, rules, gold='B', choices=choices)
        assert note['accepted'] and note['quote_is_final_computation']
        # The conclusion text may also read as option B (last number 4); its equation is still checked.
        for text, reason in [('12 / 3 = 4\nThus 12 / 3 = 5, so the answer is 4.\nConclusion: B\nFinal answer: B', 'CONTENT_CONTRADICTION'),
                             ('12 / 3 = 4\nHence 2 / 1 = 3, giving 4.\nConclusion: B\nFinal answer: B', 'CONTENT_CONTRADICTION'),
                             ('12 / 3 = 4\nTherefore y = 4.\nConclusion: B\nFinal answer: B', 'UNSUPPORTED_CONTENT_CLAIM'),
                             ('12 / 3 = 4\nThus 12 / 3 = 4, so 1 + 1 = 4.\nConclusion: B\nFinal answer: B', 'CONTENT_CONTRADICTION'),
                             ('12 / 3 = 4\nThus 12 / 3 = 4 and x = 4.\nConclusion: B\nFinal answer: B', 'UNSUPPORTED_CONTENT_CLAIM'),
                             ('12 / 3 = 4\nThus 12 / 3 = 7.\nConclusion: B\nFinal answer: B', 'REASONING_ANSWER_MISMATCH'),
                             ('12 / 3 = 4\nThus 12 / 3 = 4.\nConclusion: B\nFinal answer: B', None)]:
            with self.subTest(text=text):
                record['resolve'] = output(text, extracted='B')
                receipt = verify_retry(record, note_receipt=note, fact_catalog=facts, rules=rules, choices=choices, gold='B')
                assert receipt['reason'] == reason

if __name__ == '__main__':
    unittest.main()
