"""CPU fixtures for `collect pack-eval`: the recovery of a Pod's MMMU evaluation stage (run dirs, ledger,
prereg, model_ref.json) and its local verification. Fixture files only: no RunPod, network, GPU, weights
or MMMU data; the ledger is the real proof_eval one, the chain log the real pod/chain.py one."""
import contextlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fine_tuning.notebook_rl import proof_eval
from fine_tuning.notebook_rl.pod import chain, collect, common, eval_stage
from fine_tuning.tests import test_notebook_rl_pod as pod_tests
from fine_tuning.tests import test_notebook_rl_pod_eval as eval_tests  # module import: its cases are not re-collected here

POD = 'pod-eval-1'
PREREG = b'{"format": "fixture-prereg"}\n'
PREREG_SHA = common.sha256_bytes(PREREG)
WEIGHTS = b'WEIGHTS' * 1000


class EvalPackFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.volume = self.root/'vol'
        self.private = self.root/'private'  # stands for /home/jang2/mmdl-artifacts
        self.private.mkdir()
        patch = mock.patch.dict(os.environ, {'MMDL_ARTIFACT_ROOT': str(self.private), 'RUNPOD_POD_ID': POD})
        patch.start()
        self.addCleanup(patch.stop)
        self.artifacts = self.volume/'eval-data/artifacts'
        self.ledger = self.volume/'eval-data/proof/ledger.jsonl'
        self.prereg = self.volume/'eval-data/proof/prereg.json'
        self.prereg.parent.mkdir(parents=True)
        self.prereg.write_bytes(PREREG)
        self.prereg.with_name('prereg.json.sha256').write_text(PREREG_SHA+'\n')
        self.export = self.volume/'ft-export'
        (self.export/'model').mkdir(parents=True)
        (self.export/'model/model.safetensors').write_bytes(WEIGHTS)
        (self.export/'model_ref.json').write_text(json.dumps({'kind': 'merged', 'files': []}))

    def params(self, stage, **changes):
        value = eval_tests.EvalStagePlanTests.params(self, stage, volume=str(self.volume), prereg_sha256=PREREG_SHA)
        if stage == 'eval-course':
            value['ft_model_ref_sha256'] = common.sha256_file(self.export/'model_ref.json')
        value.update(changes)
        return value

    def planned(self, params, stage):
        return [(arm, variant, eval_stage.run_id(params['run_prefix'], arm, variant))
                for arm, variant in eval_stage._runs(stage)]

    def start_all(self, params, stage, skip=()):
        for arm, variant, rid in self.planned(params, stage):
            if rid not in skip:
                proof_eval._append(self.ledger, 'START', arm=arm, variant=variant, run_id=rid,
                                   preregistration_sha256=PREREG_SHA)

    def write_run(self, rid, variant, status='COMPLETE'):
        base = self.artifacts/'runs'/rid
        (base/'samples').mkdir(parents=True)
        files = {'run_manifest.json': '{"identity": {}}', 'summary.json': json.dumps({'status': status}),
                 'predictions.jsonl': '{"id": "a"}\n', 'resolved_eval_config.yaml': 'k: v\n',
                 'environment.json': '{}', 'failures.jsonl': '', 'source.patch': 'diff', '.lock': '',
                 'samples/a.json': '{"raw_response": "private"}'}  # never needed by analyze
        if variant != proof_eval.COURSE:
            files[proof_eval.REGISTRATION_FILE] = '{"protocol_id": "x"}'
        for name, text in files.items():
            (base/name).write_text(text)
        return base

    def stage(self, stage, state='MAIN_EXIT 0', complete=True, spec_params=None, **changes):
        """A closed stage: ledger STARTs, run dirs, the start.sh run dir with a chain log of the spec."""
        params = self.params(stage, **changes)
        self.start_all(params, stage)
        if complete:
            for _, variant, rid in self.planned(params, stage):
                self.write_run(rid, variant)
        run_dir = pod_tests.make_run_dir(self.root/'runs', f'run-{stage}', state=state, result=False)
        spec = json.dumps(eval_stage.plan(stage, spec_params or params), indent=2, sort_keys=True)+'\n'
        log = chain.ChainLog(run_dir/'chain.jsonl', common.sha256_bytes(spec.encode()))
        log.append('CHAIN_START', steps=[])
        log.append('CHAIN_END', status='COMPLETE' if state == 'MAIN_EXIT 0' else 'STOPPED', exit_code=0,
                   failed_step=None)
        return params, run_dir

    def pack(self, stage, params, run_dir, name='pack'):
        return collect.pack_eval(params, stage, run_dir, self.root/'outbox', name)

    def verify(self, name='pack', target='recovered'):
        out = self.root/'outbox'
        # a downloaded pack belongs under the private root, like its extraction
        for suffix in ('.tar', '.manifest.json'):
            (self.private/'inbox').mkdir(exist_ok=True)
            (self.private/'inbox'/(name+suffix)).write_bytes((out/(name+suffix)).read_bytes())
        return collect.verify_archive(self.private/'inbox'/f'{name}.tar', self.private/'inbox'/f'{name}.manifest.json',
                                      self.private/target, self.root/f'{target}.receipt.json')


class PackEvalTests(EvalPackFixture):
    def test_eval_ft_pack_contains_exactly_the_analyze_inputs_and_no_weights(self):
        params, run_dir = self.stage('eval-ft')
        manifest = self.pack('eval-ft', params, run_dir)
        rids = [rid for _, _, rid in self.planned(params, 'eval-ft')]
        self.assertEqual(('eval', 'eval-ft', POD, rids), (manifest['kind'], manifest['stage'], manifest['pod_id'],
                                                         manifest['eval_run_ids']))
        self.assertEqual(['COMPLETE']*3, [r['status'] for r in manifest['eval_runs']])
        paths = {f['path'] for f in manifest['files']}
        analyze_inputs = {f'eval/runs/{rid}/{name}' for rid in rids for name in (
            'run_manifest.json', 'summary.json', 'predictions.jsonl', 'resolved_eval_config.yaml',
            'variant_registration.json')}
        outside_run_dir = {p for p in paths if p.startswith('eval/')}
        self.assertEqual(analyze_inputs | {'eval/ledger.jsonl', 'eval/prereg.json', 'eval/prereg.json.sha256',
                                           'eval/model_ref.json'}, outside_run_dir)
        self.assertIn('run-eval-ft/chain.jsonl', paths)
        self.assertFalse([p for p in paths if 'safetensors' in p or '/model/' in p or 'samples' in p])
        for record in manifest['files']:  # absolute source, size and hash of what was read
            source = Path(record['source'])
            self.assertTrue(source.is_absolute())
            self.assertEqual((record['bytes'], record['sha256']),
                             (source.stat().st_size, common.sha256_file(source)))
        with tarfile.open(self.root/'outbox/pack.tar') as tar:
            self.assertFalse([m.name for m in tar.getmembers() if 'safetensors' in m.name])
            self.assertNotIn(WEIGHTS, (self.root/'outbox/pack.tar').read_bytes())

    def test_verify_extracts_privately_rehashes_and_the_ledger_gate_accepts_the_copy(self):
        params, run_dir = self.stage('eval-ft')
        self.start_all(self.params('eval-f0'), 'eval-f0')  # the other stage's STARTs share the ledger
        manifest = self.pack('eval-ft', params, run_dir)
        receipt = self.verify()
        self.assertEqual(('LOCAL_VERIFIED', 'eval', 'eval-ft', POD),
                         (receipt['status'], receipt['kind'], receipt['stage'], receipt['pod_id']))
        self.assertEqual(manifest['archive_sha256'], receipt['archive_sha256'])
        recovered = self.private/'recovered'
        self.assertEqual(0o700, recovered.stat().st_mode & 0o777)
        runs = {(r['arm'], r['variant']): recovered/r['dir'] for r in manifest['eval_runs']}
        events, _ = proof_eval.read_ledger(recovered/'eval/ledger.jsonl')
        self.assertEqual(6, len(events))
        # `analyze --ledger` on the F0 STARTs also needs those runs; the FT part of the gate passes
        # against the extracted copy: the same run ids, the same preregistration.
        self.assertEqual(PREREG_SHA, events[0]['preregistration_sha256'])
        self.assertEqual({r['run_id'] for r in manifest['eval_runs']}, {p.name for p in runs.values()})
        self.assertEqual(PREREG, (recovered/'eval/prereg.json').read_bytes())
        self.assertEqual(PREREG_SHA+'\n', (recovered/'eval/prereg.json.sha256').read_text())
        self.assertEqual(common.sha256_file(self.export/'model_ref.json'),
                         common.sha256_file(recovered/'eval/model_ref.json'))
        # verify re-hashes: a changed record, a changed archive and a repeated extraction are refused
        bad = json.loads((self.private/'inbox/pack.manifest.json').read_text())
        bad['files'][0]['sha256'] = '0'*64
        (self.private/'inbox/bad.manifest.json').write_text(json.dumps(bad))
        with self.assertRaisesRegex(collect.CollectError, 'differs from manifest'):
            collect.verify_archive(self.private/'inbox/pack.tar', self.private/'inbox/bad.manifest.json',
                                   self.private/'recovered2', self.root/'r2.json')
        with self.assertRaisesRegex(collect.CollectError, 'already exists'):
            self.verify()

    def test_verify_refuses_an_eval_pack_or_extraction_outside_the_private_root(self):
        params, run_dir = self.stage('eval-f0')
        self.pack('eval-f0', params, run_dir)
        outbox = self.root/'outbox'  # not under the private root
        with self.assertRaisesRegex(collect.CollectError, 'must lie under'):
            collect.verify_archive(outbox/'pack.tar', outbox/'pack.manifest.json', self.private/'x', self.root/'r.json')
        self.verify(target='ok')
        (self.private/'inbox').mkdir(exist_ok=True)
        for place in (pod_tests.REPO/'recovered-eval', self.root/'elsewhere', self.private/'..'/'escape'):
            with self.assertRaisesRegex(collect.CollectError, 'must lie under'):
                collect.verify_archive(self.private/'inbox/pack.tar', self.private/'inbox/pack.manifest.json',
                                       place, self.root/'r3.json')
            self.assertFalse(place.exists())

    def test_eval_f0_has_no_model_ref_and_eval_course_checks_the_pinned_one(self):
        params, run_dir = self.stage('eval-f0')
        manifest = self.pack('eval-f0', params, run_dir)
        self.assertEqual(3, len(manifest['eval_runs']))
        self.assertIsNone(manifest['model_ref_sha256'])
        self.assertNotIn('eval/model_ref.json', {f['path'] for f in manifest['files']})
        params, run_dir = self.stage('eval-course')
        manifest = self.pack('eval-course', params, run_dir, name='course')
        self.assertEqual([proof_eval.COURSE], [r['variant'] for r in manifest['eval_runs']])
        self.assertIn('eval/model_ref.json', {f['path'] for f in manifest['files']})
        self.assertNotIn(f"eval/runs/{manifest['eval_run_ids'][0]}/variant_registration.json",
                         {f['path'] for f in manifest['files']})  # the course run has none
        (self.export/'model_ref.json').write_text('{"kind": "merged", "files": ["changed"]}')
        with self.assertRaisesRegex(collect.CollectError, 'ft_model_ref_sha256'):
            self.pack('eval-course', params, run_dir, name='course2')

    def test_failed_runs_are_packed_marked_failed_when_the_ledger_says_so(self):
        params, run_dir = self.stage('eval-ft', state='MAIN_EXIT 1', complete=False)
        first, second, third = (rid for _, _, rid in self.planned(params, 'eval-ft'))
        self.write_run(first, 'mmmu-val-v8-c16-seed3407-v1')
        self.write_run(second, 'mmmu-val-v8-c16-seed3408-v1', status='RUNNING')  # killed mid-run
        with self.assertRaisesRegex(collect.CollectError, f'{second}: summary is not COMPLETE'):
            self.pack('eval-ft', params, run_dir)
        for rid in (second, third):  # the third never started its directory
            proof_eval.ledger_fail(self.ledger, rid, 'chain stopped', self.artifacts)
        manifest = self.pack('eval-ft', params, run_dir)
        self.assertEqual(['COMPLETE', 'FAILED', 'FAILED'], [r['status'] for r in manifest['eval_runs']])
        self.assertEqual([], manifest['eval_runs'][2]['files'])
        self.assertIn(f'eval/runs/{second}/summary.json', {f['path'] for f in manifest['files']})
        self.assertEqual(['FAILED', 'FAILED'], [r['status'] for r in self.verify()['eval_runs'][1:]])
        # a ledger FAIL cannot turn a COMPLETE run into a failed one
        proof_eval._append(self.ledger, 'FAIL', run_id=first, reason='forged')
        with self.assertRaisesRegex(collect.CollectError, 'FAILED but summary.json is COMPLETE'):
            self.pack('eval-ft', params, run_dir, name='forged')

    def test_refusals_ledger_start_summary_symlinks_prereg_and_owner(self):
        params, run_dir = self.stage('eval-ft', complete=False)
        rids = [rid for _, _, rid in self.planned(params, 'eval-ft')]
        for rid, (_, variant, _) in zip(rids, self.planned(params, 'eval-ft')):
            self.write_run(rid, variant)
        # a planned run without a ledger START
        self.ledger.unlink()
        self.start_all(params, 'eval-ft', skip=rids[1:2])
        with self.assertRaisesRegex(collect.CollectError, f'{rids[1]}: no ledger START'):
            self.pack('eval-ft', params, run_dir)
        proof_eval._append(self.ledger, 'START', arm='FT', variant='mmmu-val-v8-c16-seed3408-v1',
                           run_id=rids[1], preregistration_sha256=PREREG_SHA)
        # STARTs recorded under another preregistration
        good_ledger = self.ledger.read_bytes()
        self.ledger.unlink()
        for arm, variant, rid in self.planned(params, 'eval-ft'):
            proof_eval._append(self.ledger, 'START', arm=arm, variant=variant, run_id=rid,
                               preregistration_sha256='9'*64)
        with self.assertRaisesRegex(collect.CollectError, 'under this preregistration'):
            self.pack('eval-ft', params, run_dir)
        self.ledger.write_bytes(good_ledger)
        # a symlinked packed file
        target = self.root/'elsewhere.jsonl'
        target.write_text('{"id": "outside"}\n')
        victim = self.artifacts/'runs'/rids[0]/'predictions.jsonl'
        victim.unlink()
        victim.symlink_to(target)
        with self.assertRaisesRegex(collect.CollectError, 'symlink'):
            self.pack('eval-ft', params, run_dir)
        victim.unlink()
        victim.write_text('{"id": "a"}\n')
        # a symlinked run dir, ledger, prereg, model_ref.json
        moved = self.root/'moved-run'
        (self.artifacts/'runs'/rids[2]).rename(moved)
        (self.artifacts/'runs'/rids[2]).symlink_to(moved)
        with self.assertRaisesRegex(collect.CollectError, 'symlink'):
            self.pack('eval-ft', params, run_dir)
        (self.artifacts/'runs'/rids[2]).unlink()
        moved.rename(self.artifacts/'runs'/rids[2])
        for path in (self.ledger, self.prereg, self.export/'model_ref.json'):
            real = self.root/(path.name+'.real')
            path.rename(real)
            path.symlink_to(real)
            with self.assertRaisesRegex(collect.CollectError, 'symlink'):
                self.pack('eval-ft', params, run_dir)
            path.unlink()
            real.rename(path)
        # prereg bytes differ from params.prereg_sha256
        self.prereg.write_bytes(PREREG+b' ')
        with self.assertRaisesRegex(collect.CollectError, 'prereg'):
            self.pack('eval-ft', params, run_dir)
        self.prereg.write_bytes(PREREG)
        # nothing was written by the refusals, and the clean state packs
        self.assertFalse((self.root/'outbox').exists() and list((self.root/'outbox').glob('*.tar')))
        self.assertEqual(3, len(self.pack('eval-ft', params, run_dir)['eval_runs']))

    def test_run_dir_must_be_closed_and_must_have_run_this_spec(self):
        params, run_dir = self.stage('eval-f0')
        live = pod_tests.make_run_dir(self.root/'runs', 'live', state='MAIN_RUNNING', owner_pid=os.getpid(),
                                      result=False)
        with self.assertRaisesRegex(collect.CollectError, 'still alive'):
            self.pack('eval-f0', params, live)
        running = pod_tests.make_run_dir(self.root/'runs', 'running', state='MAIN_RUNNING', result=False)
        with self.assertRaisesRegex(collect.CollectError, 'not closed'):
            self.pack('eval-f0', params, running)
        nolog = pod_tests.make_run_dir(self.root/'runs', 'nolog', state='MAIN_EXIT 0', result=False)
        with self.assertRaisesRegex(collect.CollectError, 'lacks result/receipt'):
            self.pack('eval-f0', params, nolog)
        failed_early = pod_tests.make_run_dir(self.root/'runs', 'early', state='PREFLIGHT_FAILED', result=False)
        with self.assertRaisesRegex(collect.CollectError, 'chain never started'):
            self.pack('eval-f0', params, failed_early)
        # a chain that ran another spec (other params) or whose log was cut
        _, other_run = self.stage('eval-course', spec_params=self.params('eval-course', run_prefix='other'))
        with self.assertRaisesRegex(collect.CollectError, 'not the log of the spec'):
            self.pack('eval-course', self.params('eval-course'), other_run)
        # MAIN_EXIT 0 whose chain did not end COMPLETE
        stopped = pod_tests.make_run_dir(self.root/'runs', 'stopped', state='MAIN_EXIT 0', result=False)
        spec = json.dumps(eval_stage.plan('eval-f0', params), indent=2, sort_keys=True)+'\n'
        log = chain.ChainLog(stopped/'chain.jsonl', common.sha256_bytes(spec.encode()))
        log.append('CHAIN_START', steps=[])
        with self.assertRaisesRegex(collect.CollectError, 'COMPLETE chain end'):
            self.pack('eval-f0', params, stopped)
        with self.assertRaisesRegex(collect.CollectError, 'archive name'):
            collect.pack_eval(params, 'eval-f0', run_dir, self.root/'outbox', 'bad name')
        with mock.patch.dict(os.environ):
            os.environ.pop('RUNPOD_POD_ID')
            with self.assertRaisesRegex(collect.CollectError, 'RUNPOD_POD_ID'):
                self.pack('eval-f0', params, run_dir)

    def test_params_are_validated_like_the_planner(self):
        params, run_dir = self.stage('eval-f0')
        outside = self.params('eval-f0', artifact_root=str(self.volume/'elsewhere'))
        with self.assertRaisesRegex(ValueError, 'must be under'):
            collect.pack_eval(outside, 'eval-f0', run_dir, self.root/'outbox', 'x')
        with self.assertRaisesRegex(ValueError, 'exactly the keys'):
            collect.pack_eval(self.params('eval-ft'), 'eval-f0', run_dir, self.root/'outbox', 'x')

    def test_spec_hash_equals_what_eval_stage_plan_prints(self):
        params, run_dir = self.stage('eval-ft')
        file = self.root/'params.json'
        file.write_text(json.dumps(params))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(0, eval_stage.main(['plan', '--stage', 'eval-ft', '--params', str(file),
                                                 '--output', str(self.root/'steps.json')]))
        self.assertEqual(out.getvalue().strip(), self.pack('eval-ft', params, run_dir)['spec_sha256'])

    def test_cli_pack_eval_and_refusal_exit_codes(self):
        params, run_dir = self.stage('eval-f0')
        file = self.root/'params.json'
        file.write_text(json.dumps(params))
        arguments = ['pack-eval', '--params', str(file), '--stage', 'eval-f0', '--run-dir', str(run_dir),
                     '--out-dir', str(self.root/'outbox')]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(0, collect.main(arguments))
        printed = json.loads(out.getvalue())
        self.assertEqual(('eval-f0', 'CLOSED'), (printed['stage'], printed['closure']))
        self.assertTrue((self.root/'outbox'/f'{run_dir.name}.tar').is_file())
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(2, collect.main(arguments))  # never overwritten
        self.assertIn('REFUSED', err.getvalue())


class PackEvalLedgerTests(EvalPackFixture):
    def remote(self, ledger, name, pod_id=POD):
        out = self.root/'outbox'
        manifest = json.loads((out/f'{name}.manifest.json').read_text())
        collect.append_state(ledger, 'REMOTE_VERIFIED', {
            'archive_sha256': manifest['archive_sha256'],
            'manifest_sha256': common.sha256_file(out/f'{name}.manifest.json'), 'pod_id': pod_id})
        return manifest

    def local(self, ledger, name, manifest):
        target = f'rec-{name}-{ledger.stem}'
        self.verify(name, target=target)  # the receipt `collect verify` writes
        return collect.append_state(ledger, 'LOCAL_VERIFIED', {'archive_sha256': manifest['archive_sha256']},
                                    receipt_file=self.root/f'{target}.receipt.json')

    def test_cleanup_ledger_accepts_an_eval_pack_and_checks_the_typed_pod_id(self):
        params, run_dir = self.stage('eval-f0')
        self.pack('eval-f0', params, run_dir)
        wrong = self.root/'wrong.jsonl'
        manifest = self.remote(wrong, 'pack', pod_id='some-other-pod')
        with self.assertRaisesRegex(collect.CollectError, 'pod_id typed at REMOTE_VERIFIED'):
            self.local(wrong, 'pack', manifest)
        good = self.root/'good.jsonl'
        self.remote(good, 'pack')
        self.assertEqual('LOCAL_VERIFIED', self.local(good, 'pack', manifest)['state'])

    def test_owned_storage_deletion_needs_every_run_training_and_eval_locally_verified(self):
        collect.pack([pod_tests.make_run_dir(self.root/'runs', 'train-run')], self.root/'outbox', 'train')
        params, run_dir = self.stage('eval-f0')
        self.pack('eval-f0', params, run_dir, name='ev')
        training, evaluation = self.root/'train.ledger.jsonl', self.root/'eval.ledger.jsonl'
        self.local(training, 'train', self.remote(training, 'train'))
        collect.append_state(training, 'POD_TERMINATED', {
            'pod_id': POD, 'terminated_utc': pod_tests.utc(), 'confirmed_by': 'user'})
        eval_manifest = self.remote(evaluation, 'ev')  # not yet LOCAL_VERIFIED
        deleted = {'deleted_resources': 'volume:v', 'deleted_utc': pod_tests.utc(), 'confirmed_by': 'user'}
        with self.assertRaisesRegex(collect.CollectError, 'lacks REMOTE_VERIFIED \\+ LOCAL_VERIFIED'):
            collect.append_state(training, 'OWNED_STORAGE_DELETED', deleted, run_ledgers=[training, evaluation])
        self.local(evaluation, 'ev', eval_manifest)
        record = collect.append_state(training, 'OWNED_STORAGE_DELETED', deleted,
                                      run_ledgers=[training, evaluation])
        bound = json.loads(record['evidence']['run_ledgers'])
        self.assertEqual([POD, POD], [entry['pod_id'] for entry in bound])
        self.assertEqual(collect.read_ledger(evaluation)[-1]['record_sha256'], bound[1]['last_record_sha256'])


if __name__ == '__main__':
    unittest.main()
