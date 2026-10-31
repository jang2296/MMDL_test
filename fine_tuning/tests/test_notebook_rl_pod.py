"""CPU-only fixtures for the RunPod preparation scripts: fake mount/git remote/python, tiny
synthetic volumes; no RunPod, network, GPU, weights or dependency installation."""
import copy
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fine_tuning.notebook_rl.pod import collect, common, cost, data_bundle, job, runner, verify

POD = Path(__file__).resolve().parents[1]/'notebook_rl/pod'
BUDGET = Path(__file__).resolve().parents[1]/'budget.py'
COMMIT_A = 'a'*40
REPO = Path(__file__).resolve().parents[2]
REAL_POOL = Path('/home/jang2/mmdl-artifacts/notebook-rl-local-nf4-20261021/hf-pool-v3')
ORIGIN = 'https://github.com/jang2296/MMDL_test.git'
# Stand-ins for what the fixture cannot install: `<venv>/bin/python -m pip freeze --all` prints
# <venv>/fixture-freeze.txt; `python -m venv X` (shadowing the stdlib module through the
# PYTHONPATH volume.sh exports) writes a recording shim as X/bin/python.
FAKE_PIP = ('import pathlib, sys\n'
            'if sys.argv[1:] != ["freeze", "--all"]:\n    raise SystemExit(2)\n'
            'sys.stdout.write(pathlib.Path(sys.prefix, "fixture-freeze.txt").read_text())\n')
FAKE_VENV = '''import os, pathlib, sys
target = pathlib.Path(sys.argv[1])
with open(os.environ["FIXTURE_CALLS"], "a") as stream:
    stream.write("base -m venv " + str(target) + "\\n")
(target/"bin").mkdir(parents=True)
shim = target/"bin/python"
shim.write_text("#!/usr/bin/env bash\\n"
                'printf "%s\\\\n" "venv $*" >> "$FIXTURE_CALLS"\\n'
                'if [[ $* == "-m pip freeze --all" ]]; then printf "%s\\\\n" "fixture==1"; fi\\n'
                "[[ $1 == - ]] && cat > /dev/null\\nexit 0\\n")
shim.chmod(0o755)
'''
# Stdlib stand-in for `python -m fine_tuning.notebook_rl.pool_freeze verify` in the fixture clone (the
# real module needs numpy/PIL/yaml); reports a mismatch while <vault>/MISMATCH exists.
FAKE_POOL_FREEZE = '''import hashlib, json, pathlib, sys
args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
errors = ["vault: fixture mismatch"] if pathlib.Path(args["--vault"], "MISMATCH").exists() else []
digest = hashlib.sha256(pathlib.Path(args["--manifest"]).read_bytes()).hexdigest()
print(json.dumps({"status": "MISMATCH" if errors else "VERIFIED", "errors": len(errors),
                  "first_errors": errors, "manifest_sha256": digest}))
sys.exit(1 if errors else 0)
'''


def run(*command, cwd=None, env=None, timeout=60):
    return subprocess.run([str(c) for c in command], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=timeout)


def utc(**delta):
    return (dt.datetime.now(dt.timezone.utc)+dt.timedelta(**delta)).isoformat(timespec='seconds')


DATA = {'bundle_manifest': '/workspace/data/proof.bundle.json', 'bundle_manifest_sha256': 'a'*64,
        'dataset_manifest': '/home/u/freeze/dataset_manifest.json',
        'dataset_manifest_sha256': 'b'*64, 'vault_root': '/home/u/vault'}


def financial(**changes):
    return {'owned_pod_id': 'fixture-pod', 'gpu_cutoff_utc': utc(hours=2),
            'billing_start_utc': utc(), 'balance_usd': 10, 'reserve_usd': 1, 'hourly_usd': 1,
            **changes}


def approval(**changes):
    return {'approved_by_user': True, 'approved_utc': utc(minutes=-5),
            'approval_note': 'user approved stage 1 in chat', 'approved_sha256': 'SEAL', **changes}


def seal(value):
    """Stands in for the user's approval step: approved_sha256 = the job's content sha256."""
    if isinstance(value.get('approval'), dict) and value['approval'].get('approved_sha256') == 'SEAL':
        try:
            value['approval']['approved_sha256'] = job.content_sha256(value)
        except ValueError:  # NaN/inf: no canonical JSON, the job is refused for that anyway
            pass
    return value


def approved_job(root, **changes):
    value = {'version': 'pod-job-v1', 'commit': COMMIT_A, 'volume_root': '/workspace/vol',
             'run_dir': '/workspace/runs/r1', 'stage': 'bf16-smoke-throughput',
             'smoke': ['python', '-c', 'pass'], 'main': ['python', '-c', 'pass'],
             'max_main_seconds': 600, 'min_free_gib': 0.001, 'financial': financial(),
             'data': dict(DATA), 'approval': approval()}
    value.update(changes)
    return seal(value)


def prep_job(**changes):
    value = {'version': 'pod-prep-v1', 'commit': COMMIT_A, 'volume_root': '/workspace/vol',
             'environment': 'experiments', 'financial': financial(), 'approval': approval()}
    value.update(changes)
    return seal(value)


class CommonTests(unittest.TestCase):
    def test_quick_digest_matches_single_pass_and_full_for_small_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'f.bin'
            data = bytes(range(256))*3
            path.write_bytes(data)
            for span in (1, 10, 100, 400, 4096):
                size, full, quick = common.file_digests(path, span, chunk=7)
                self.assertEqual(len(data), size)
                self.assertEqual(hashlib.sha256(data).hexdigest(), full)
                self.assertEqual((size, quick), common.quick_digest(path, span, chunk=5))
                expected = hashlib.sha256(data[:span]+data[max(span, len(data)-span):]).hexdigest()
                self.assertEqual(expected, quick)
            self.assertEqual(full, quick)  # span >= size: quick covers the whole file
            path.write_bytes(data[:100]+b'X'+data[101:])
            self.assertNotEqual(full, common.file_digests(path, 10)[1])
            self.assertEqual(common.file_digests(path, 10)[2],
                             hashlib.sha256(data[:10]+data[-10:]).hexdigest())


class JobTests(unittest.TestCase):
    def check_refused(self, **changes):
        with self.assertRaises(job.JobError, msg=str(changes)):
            job.validate_job(approved_job(None, **changes))

    def test_valid_job_and_each_refusal(self):
        self.assertEqual('main', job.validate_job(approved_job(None, stage='main'))['stage'])
        self.check_refused(commit='abc')
        self.check_refused(commit='A'*40)
        self.check_refused(stage='full')
        self.check_refused(smoke='python -c pass')
        self.check_refused(smoke=[])
        self.check_refused(main=['ok', 3])
        self.check_refused(main=['ok', ''])
        self.check_refused(run_dir='/workspace/../etc')
        self.check_refused(run_dir='/elsewhere/r1')
        self.check_refused(volume_root='/workspace')
        self.check_refused(volume_root='/workspace/a/../b')
        for value in (0, -1, 1.5, True, '10', 10**9):
            self.check_refused(max_main_seconds=value)
        with self.assertRaises(job.JobError):
            job.validate_job({**approved_job(None), 'extra': 1})
        broken = approved_job(None)
        del broken['smoke']
        with self.assertRaises(job.JobError):
            job.validate_job(broken)

    def test_approval_must_be_recorded_by_the_user(self):
        for value in (None, {}, {'approved_by_user': True},
                      approval(approved_by_user=False), approval(approved_by_user='true'),
                      approval(approved_utc='2026-10-30T00:00:00'), approval(approved_utc=utc(days=1)),
                      approval(approval_note=' '), approval(approved_sha256='0'*64),
                      {k: v for k, v in approval().items() if k != 'approved_sha256'}):
            self.check_refused(approval=value)
        broken = approved_job(None)
        del broken['approval']
        with self.assertRaises(job.JobError):
            job.validate_job(broken)

    def test_old_approvals_are_refused(self):
        def aged(hours, **changes):
            value = approved_job(None, **changes)
            value['approval']['approved_utc'] = utc(hours=-hours)
            return value
        job.validate_job(aged(71))  # default limit is 72 h
        job.validate_job(aged(100, max_approval_age_hours=168))
        job.validate_job(aged(2, max_approval_age_hours=2.5))
        for value in (aged(73), aged(2, max_approval_age_hours=1), aged(200, max_approval_age_hours=168)):
            with self.assertRaisesRegex(job.JobError, 'older than'):
                job.validate_job(value)
        for limit in (0, -1, 169, True, '72', None, float('nan'), float('inf')):
            with self.assertRaisesRegex(job.JobError, 'max_approval_age_hours', msg=str(limit)):
                job.validate_job(aged(0, max_approval_age_hours=limit))

    def test_interpreter_must_be_the_manifest_venv(self):
        python = '/workspace/v/venv-x/bin/python'
        good = approved_job(None, smoke=[python, '-c', 'pass'], main=[python, '-m', 'x'])
        with tempfile.TemporaryDirectory() as temporary:
            manifest, path = Path(temporary)/'m.json', Path(temporary)/'job.json'
            manifest.write_text(json.dumps({'venv': '/workspace/v/venv-x'}))
            self.assertEqual(python, job.check_interpreter(good, manifest))
            for changes in ({'smoke': ['python', '-c', 'pass']}, {'main': ['/usr/bin/python3']},
                            {'main': [python+'3']}, {'smoke': ['/workspace/v/venv-y/bin/python']}):
                with self.assertRaisesRegex(job.JobError, 'volume venv interpreter'):
                    job.check_interpreter({**good, **changes}, manifest)
            for content in ('not json', '{}', '[]', '{"venv": ""}', '{"venv": 3}'):
                manifest.write_text(content)
                with self.assertRaises(job.JobError, msg=content):
                    job.check_interpreter(good, manifest)
            with self.assertRaises(job.JobError):
                job.check_interpreter(good, Path(temporary)/'missing.json')
            manifest.write_text(json.dumps({'venv': '/workspace/v/venv-x'}))
            path.write_text(json.dumps(good))
            self.assertEqual(0, job.main([str(path), '--require-interpreter', str(manifest)]))
            path.write_text(json.dumps({**good, 'main': ['python']}))
            with mock.patch('sys.stderr', new=io.StringIO()) as err:
                self.assertEqual(2, job.main([str(path), '--require-interpreter', str(manifest)]))
            self.assertIn('REFUSED', err.getvalue())

    def test_budget_fit_uses_watchdog_deadline_and_cli_reports_refusals(self):
        value = approved_job(None, max_main_seconds=1000)
        fit = job.budget_fit(value)
        self.assertTrue(fit['fits'])
        self.assertEqual(1000+300+600, fit['required_seconds'])
        tight = approved_job(None, max_main_seconds=1000,
                             financial=financial(gpu_cutoff_utc=utc(seconds=1500)))
        self.assertFalse(job.budget_fit(tight)['fits'])
        self.assertEqual(job.PREP_MAX_SECONDS, job.budget_fit(prep_job())['required_seconds'])
        self.assertIn(f'max_seconds={job.PREP_MAX_SECONDS} ', (POD/'volume.sh').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'job.json'
            path.write_text(json.dumps(tight))
            with mock.patch('sys.stderr', new=io.StringIO()) as err:
                self.assertEqual(2, job.main([str(path)]))
            self.assertIn('does not fit', err.getvalue())
            path.write_text(json.dumps(value))
            with mock.patch('sys.stdout', new=io.StringIO()) as out:
                self.assertEqual(0, job.main([str(path)]))
            self.assertTrue(json.loads(out.getvalue())['budget']['fits'])
            with mock.patch('sys.stdout', new=io.StringIO()) as out:
                self.assertEqual(0, job.main([str(path), '--get', 'commit']))
            self.assertEqual(COMMIT_A, out.getvalue().strip())
            path.write_text(json.dumps({**value, 'approval': None}))
            with mock.patch('sys.stderr', new=io.StringIO()):
                self.assertEqual(2, job.main([str(path)]))
            with mock.patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                job.main([str(path), '--hourly-usd', '1'])  # no financial CLI flags exist

    def test_approval_is_bound_to_the_job_content(self):
        value = approved_job(None)
        job.validate_job(value)
        for change in (lambda v: v.update(commit='b'*40), lambda v: v['main'].append('--x'),
                       lambda v: v['financial'].update(hourly_usd=0.5),
                       lambda v: v['financial'].update(owned_pod_id='other-pod'),
                       lambda v: v['data'].update(vault_root='/home/u/other'),
                       lambda v: v.update(min_free_gib=5), lambda v: v.update(max_approval_age_hours=2)):
            changed = copy.deepcopy(value)
            change(changed)
            with self.assertRaisesRegex(job.JobError, 'approved_sha256'):
                job.validate_job(changed)
        renoted = copy.deepcopy(value)
        renoted['approval']['approval_note'] = 'other words'  # approval block is not content
        job.validate_job(renoted)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'job.json'
            path.write_text(json.dumps({**value, 'approval': None}))  # unapproved: sha still printed
            with mock.patch('sys.stdout', new=io.StringIO()) as out:
                self.assertEqual(0, job.main([str(path), '--content-sha256']))
            self.assertEqual(value['approval']['approved_sha256'], out.getvalue().strip())
        text = json.dumps({k: v for k, v in value.items() if k != 'approval'}, sort_keys=True,
                          separators=(',', ':'))
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), job.content_sha256(value))
        with self.assertRaises(job.JobError):  # NaN is not canonical JSON
            job.validate_job({**value, 'max_approval_age_hours': float('nan')})

    def test_financial_data_and_disk_bounds_are_validated(self):
        refusals = [{'financial': financial(**{'gpu_cutoff_utc': utc(hours=2), 'extra': 1})},
                    {'financial': {k: v for k, v in financial().items() if k != 'reserve_usd'}},
                    {'financial': financial(balance_usd=True)},
                    {'financial': financial(balance_usd=float('inf'))},
                    {'financial': financial(hourly_usd=0)},
                    {'financial': financial(balance_usd=1, reserve_usd=1)},
                    {'financial': financial(gpu_cutoff_utc='2026-10-30T00:00:00')},
                    {'financial': financial(billing_start_utc=utc(minutes=10))},  # future
                    {'financial': financial(owned_pod_id='')},
                    {'financial': financial(owned_pod_id='a b')},
                    {'data': {k: v for k, v in DATA.items() if k != 'vault_root'}},
                    {'data': {**DATA, 'bundle_manifest': '/elsewhere/b.json'}},
                    {'data': {**DATA, 'dataset_manifest': 'relative/m.json'}},
                    {'data': {**DATA, 'vault_root': '/home/u/../vault'}},
                    {'data': {**DATA, 'dataset_manifest_sha256': 'B'*64}},
                    {'min_free_gib': 0}, {'min_free_gib': -1}, {'min_free_gib': True},
                    {'min_free_gib': '10'}]
        for changes in refusals:
            self.check_refused(**changes)
        job.validate_job(approved_job(None, financial=financial(billing_start_utc=utc(minutes=2))))
        broken = approved_job(None)
        del broken['data']
        with self.assertRaises(job.JobError):
            job.validate_job(seal({**broken, 'approval': approval()}))

    def test_financial_argv_and_fields_come_from_the_job(self):
        value = approved_job(None, financial=financial(hourly_usd=0.59))
        bounds = value['financial']
        self.assertEqual(['--owned-pod-id', 'fixture-pod', '--gpu-cutoff-utc', bounds['gpu_cutoff_utc'],
                          '--billing-start-utc', bounds['billing_start_utc'], '--balance-usd', '10',
                          '--reserve-usd', '1', '--hourly-usd', '0.59'], job.financial_argv(value))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'job.json'
            path.write_text(json.dumps(value))
            for arguments, expected in ((['--financial-argv'], '\0'.join(job.financial_argv(value))+'\0'),
                                        (['--get', 'owned_pod_id'], 'fixture-pod\n'),
                                        (['--get', 'min_free_gib'], '0.001\n')):
                with mock.patch('sys.stdout', new=io.StringIO()) as out:
                    self.assertEqual(0, job.main([str(path), *arguments]))
                self.assertEqual(expected, out.getvalue())
            with mock.patch('sys.stderr', new=io.StringIO()) as err:
                self.assertEqual(2, job.main([str(path), '--get', 'environment']))
            self.assertIn('no field environment', err.getvalue())

    def test_prep_job_for_volume_sh(self):
        value = prep_job()
        self.assertEqual('experiments', job.validate_job(value)['environment'])
        for changes in ({'environment': 'cuda'}, {'run_dir': '/workspace/runs/r1'},
                        {'volume_root': '/elsewhere/v'}, {'commit': 'short'},
                        {'financial': financial(hourly_usd=-1)}):
            with self.assertRaises(job.JobError, msg=str(changes)):
                job.validate_job(prep_job(**changes))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'prep.json'
            path.write_text(json.dumps(value))
            with mock.patch('sys.stderr', new=io.StringIO()) as err:
                self.assertEqual(2, job.main([str(path), '--argv', 'smoke']))
            self.assertIn('pod-job-v1', err.getvalue())


def make_git_repo(root):
    """Real local git repo (no network) with the files the scripts need, then commit."""
    repo = root/'repo'
    for sub in ('fine_tuning/notebook_rl/pod', 'env', 'manifests/models'):
        (repo/sub).mkdir(parents=True)
    (repo/'fine_tuning/__init__.py').write_text('')
    (repo/'fine_tuning/notebook_rl/__init__.py').write_text('')
    shutil.copy(BUDGET, repo/'fine_tuning/budget.py')
    for source in POD.iterdir():
        if source.suffix in ('.py', '.sh'):
            shutil.copy(source, repo/'fine_tuning/notebook_rl/pod'/source.name)
    (repo/'env/requirements-experiments.lock').write_text('fixture==1\n')
    (repo/'fine_tuning/notebook_rl/pool_freeze.py').write_text(FAKE_POOL_FREEZE)
    env = dict(os.environ, GIT_CONFIG_GLOBAL='/dev/null', GIT_CONFIG_SYSTEM='/dev/null')
    run('git', 'init', '-q', repo, env=env)
    run('git', '-C', repo, 'remote', 'add', 'origin', ORIGIN, env=env)
    return repo, env


def commit_repo(repo, env):
    run('git', '-C', repo, 'add', '-A', env=env)
    result = run('git', '-C', repo, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit',
                 '-q', '-m', 'fixture', env=env)
    assert result.returncode == 0, result.stderr
    return run('git', '-C', repo, 'rev-parse', 'HEAD', env=env).stdout.strip()


def make_volume(root, repo, span=16):
    """Synthetic volume: real tiny venv, fake model files, manifest written by the real writer."""
    volume = root/'ws/vol'
    snapshot = volume/'hf-cache/hub/models--x/snapshots/rev'
    blobs = volume/'hf-cache/hub/models--x/blobs'
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    files = []
    for name, payload in (('config.json', b'{"a":1}'), ('w.safetensors', bytes(range(256))*4)):
        (blobs/name).write_bytes(payload)
        (snapshot/name).symlink_to(os.path.relpath(blobs/name, snapshot))
        files.append({'path': name, 'bytes': len(payload),
                      'sha256': hashlib.sha256(payload).hexdigest()})
    (repo/'manifests/models/baseline.json').write_text(json.dumps({'files': files}))
    lock = repo/'env/requirements-experiments.lock'
    lock_sha = common.sha256_file(lock)
    venv = volume/verify.venv_name(lock_sha)
    run(sys.executable, '-m', 'venv', '--without-pip', venv, timeout=60)
    site = next((venv/'lib').glob('python*/site-packages'))
    (site/'pip').mkdir()
    (site/'pip/__init__.py').write_text('')
    (site/'pip/__main__.py').write_text(FAKE_PIP)
    (venv/'fixture-freeze.txt').write_text('fixture==1\n')
    (volume/'pip-freeze.txt').write_text('fixture==1\n')
    manifest = verify.write_volume_manifest(volume, commit=COMMIT_A, environment='experiments',
        lock=lock, venv=venv, freeze=volume/'pip-freeze.txt', snapshot=snapshot,
        model_files=files, span=span)
    return volume, manifest


def make_data(root):
    """Private data at absolute temp paths, packed by the real data_bundle; left in place = staged."""
    private = (root/'private').resolve()
    vault, manifest = private/'vault', private/'freeze/dataset_manifest.json'
    (vault/'train').mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    (vault/'train/actor_inputs.jsonl').write_text('{"problem_id": "p1"}\n')
    (vault/'freeze_binding.json').write_text('{}\n')
    manifest.write_text('{"version": "pool-freeze-manifest-v1", "items": []}\n')
    spec = root/'spec.json'
    spec.write_text(json.dumps({'version': data_bundle.SPEC_VERSION,
                                'paths': [str(vault), str(manifest)]}))
    data_bundle.pack(spec, root/'ws/data-bundles', 'proof')
    bundle = root/'ws/data-bundles/proof.bundle.json'
    return {'bundle_manifest': str(bundle), 'bundle_manifest_sha256': common.sha256_file(bundle),
            'dataset_manifest': str(manifest),
            'dataset_manifest_sha256': common.sha256_file(manifest), 'vault_root': str(vault)}


def fake_commands(root, mountpoint_fails_for='notmount'):
    commands = root/'commands'
    commands.mkdir(exist_ok=True)
    mount = commands/'mountpoint'
    mount.write_text(f'#!/usr/bin/env bash\n[[ $2 != *{mountpoint_fails_for}* ]]\n')
    mount.chmod(0o755)
    return commands


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo, self.env = make_git_repo(self.root)
        (self.root/'ws').mkdir()
        # Volume files are written first because the baseline manifest lives in the repo.
        self.volume, self.manifest = make_volume(self.root, self.repo)
        self.commit = commit_repo(self.repo, self.env)
        self.data = make_data(self.root)
        home = self.root/'home'  # training preflight: $HOME/mmdl-data/evaluation -> <volume>/eval-data
        (home/'mmdl-data').mkdir(parents=True)
        (home/'mmdl-data/evaluation').symlink_to(self.volume.resolve()/'eval-data')
        patcher = mock.patch.dict(os.environ, {
            'PATH': f"{fake_commands(self.root)}{os.pathsep}{os.environ['PATH']}",
            'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_CONFIG_SYSTEM': '/dev/null', 'HOME': str(home)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def checks(self, **changes):
        arguments = dict(repo_root=self.repo, commit=self.commit, volume_root=self.volume,
                         run_dir=self.root/'ws/runs/r1', mount_path=str(self.root/'ws'),
                         min_free_gib=0.001, data=self.data)
        arguments.update(changes)
        result = verify.run_checks(**arguments)
        self.failed = [c['name'] for c in result['checks'] if not c['ok']]
        return result

    def rewrite_manifest(self, **changes):
        path = self.volume/verify.MANIFEST_NAME
        value = json.loads(path.read_text())
        value.update(changes)
        path.write_text(json.dumps(value))

    def test_pass_full_hash_and_each_check_present(self):
        result = self.checks()
        self.assertTrue(result['ok'], result['checks'])
        self.assertEqual(
            ['clone', 'volume_manifest', 'lock_sha256', 'venv', 'pinned_baseline', 'model_files',
             'data_bundle', 'dataset_vault', 'disk_free', 'workspace_mount'],
            [c['name'] for c in result['checks']])
        self.assertEqual('full', result['hash_mode'])
        self.assertTrue(result['full_sha256_verified'])

    def test_quick_is_labelled_and_never_reported_as_a_full_hash_check(self):
        payload = bytes(range(256))*4
        blob = self.volume/'hf-cache/hub/models--x/blobs/w.safetensors'
        blob.write_bytes(payload[:500]+b'\xff'+payload[501:])  # middle byte: outside head/tail 16
        quick = self.checks(quick=True)
        self.assertTrue(quick['ok'])
        self.assertEqual('quick', quick['hash_mode'])
        self.assertFalse(quick['full_sha256_verified'])
        detail = next(c['detail'] for c in quick['checks'] if c['name'] == 'model_files')
        self.assertIn('quick', detail)
        self.assertIn('NOT a full', detail)
        full = self.checks()
        self.assertFalse(full['ok'])
        self.assertEqual(['model_files'], self.failed)
        blob.write_bytes(b'\xff'+payload[1:])  # head byte changed: quick catches it too
        self.assertFalse(self.checks(quick=True)['ok'])

    def test_each_mismatch_fails_with_a_named_check(self):
        self.assertEqual(['clone'], self.checks(commit='b'*40) and self.failed)
        self.assertEqual(['clone'], self.checks(commit='not-a-sha') and self.failed)
        (self.repo/'untracked.txt').write_text('dirty')
        self.assertFalse(self.checks()['ok'])
        self.assertEqual(['clone'], self.failed)
        (self.repo/'untracked.txt').unlink()
        run('git', '-C', self.repo, 'remote', 'set-url', 'origin',
            'https://github.com/other/repo.git', env=self.env)
        self.assertEqual(['clone'], self.checks() and self.failed)
        run('git', '-C', self.repo, 'remote', 'set-url', 'origin', ORIGIN, env=self.env)
        self.assertTrue(self.checks()['ok'])

        original = copy.deepcopy(self.manifest)
        self.rewrite_manifest(status='VOLUME_PREPARING')  # no verified venv: dataset check not run
        self.assertEqual(['volume_manifest', 'dataset_vault'], self.checks() and self.failed)
        self.rewrite_manifest(**original)

        self.rewrite_manifest(lock_sha256='0'*64)
        self.assertIn('lock_sha256', self.checks() and self.failed)
        self.rewrite_manifest(**original)

        self.rewrite_manifest(python_version='3.11.0')
        self.assertIn('venv', self.checks() and self.failed)
        self.rewrite_manifest(**original)
        (self.volume/'pip-freeze.txt').write_text('changed==2\n')
        self.assertIn('venv', self.checks() and self.failed)
        (self.volume/'pip-freeze.txt').write_text('fixture==1\n')

        bad = copy.deepcopy(original['model_files'])
        bad[0]['sha256'] = '0'*64
        self.rewrite_manifest(model_files=bad)
        self.assertIn('pinned_baseline', self.checks() and self.failed)
        self.rewrite_manifest(**original)
        self.assertTrue(self.checks()['ok'])

        blob = self.volume/'hf-cache/hub/models--x/blobs/config.json'
        blob.write_bytes(b'{"a":2}')  # same size, different bytes
        self.assertEqual(['model_files'], self.checks() and self.failed)
        blob.write_bytes(b'{"a":1}')
        blob.write_bytes(b'{"a":1}x')  # size mismatch
        self.assertEqual(['model_files'], self.checks() and self.failed)
        blob.write_bytes(b'{"a":1}')

        self.assertEqual(['disk_free'], self.checks(min_free_gib=1e9) and self.failed)
        self.assertEqual(['workspace_mount'],
                         self.checks(mount_path=str(self.root/'ws/notmount')) and self.failed)
        self.assertTrue(self.checks()['ok'])

    def test_training_data_bundle_and_frozen_vault_are_checked(self):
        detail = {c['name']: c['detail'] for c in self.checks()['checks']}
        self.assertIn('pool_freeze.verify(dataset_manifest, vault_root) == []', detail['dataset_vault'])
        staged = Path(self.data['vault_root'])/'train/actor_inputs.jsonl'
        staged.write_text('{"problem_id": "p2"}\n')  # same size, other bytes
        self.assertEqual(['data_bundle'], self.checks() and self.failed)
        staged.write_text('{"problem_id": "p1"}\n')
        (Path(self.data['vault_root'])/'MISMATCH').write_text('')  # pool_freeze reports errors
        self.assertEqual(['dataset_vault'], self.checks() and self.failed)
        (Path(self.data['vault_root'])/'MISMATCH').unlink()
        for key, failed in (('bundle_manifest_sha256', 'data_bundle'),
                            ('dataset_manifest_sha256', 'dataset_vault')):
            self.assertEqual([failed], self.checks(data={**self.data, key: '0'*64}) and self.failed)
        outside = {**self.data, 'vault_root': str(self.root/'elsewhere')}
        self.assertIn('data_bundle', self.checks(data=outside) and self.failed)  # not in the bundle
        self.assertEqual(['data_bundle', 'dataset_vault'], self.checks(data=None) and self.failed)
        self.assertTrue(self.checks()['ok'])

    @unittest.skipUnless((REAL_POOL/'vault-v1/freeze_binding.json').is_file(),
                         'local frozen proof data not present')
    def test_real_frozen_vault_passes_the_real_pool_freeze_verify(self):
        manifest = REAL_POOL/'freeze-v2/dataset_manifest.json'
        data = {'dataset_manifest': str(manifest), 'vault_root': str(REAL_POOL/'vault-v1'),
                'dataset_manifest_sha256': common.sha256_file(manifest)}
        self.assertIn('== []', verify.check_dataset(sys.executable, REPO, data))
        with self.assertRaisesRegex(ValueError, 'pinned'):
            verify.check_dataset(sys.executable, REPO, {**data, 'dataset_manifest_sha256': '0'*64})
        with self.assertRaisesRegex(ValueError, 'is not \\[\\]'):  # an empty vault is not the manifest's
            verify.check_dataset(sys.executable, REPO, {**data, 'vault_root': str(self.root)})

    def venv_freeze(self):
        return Path(self.manifest['venv'])/'fixture-freeze.txt'

    def test_live_pip_freeze_must_match_the_recorded_freeze(self):
        result = self.checks()
        self.assertTrue(result['ok'], result['checks'])
        self.assertIn('live pip freeze matches',
                      next(c['detail'] for c in result['checks'] if c['name'] == 'venv'))
        # Installed after the volume was prepared; the record file and manifest are untouched.
        self.venv_freeze().write_text('fixture==1\nevil==9\n')
        result = self.checks()
        self.assertEqual(['venv', 'dataset_vault'], self.failed)  # no pool_freeze in a bad venv
        self.assertIn('installed packages differ',
                      next(c['detail'] for c in result['checks'] if c['name'] == 'venv'))
        self.venv_freeze().write_text('fixture==1\n')
        self.assertTrue(self.checks()['ok'])
        pip = next((Path(self.manifest['venv'])/'lib').glob('python*/site-packages/pip/__main__.py'))
        pip.rename(pip.with_name('moved.py'))  # pip unusable in the venv: a failed freeze is a failure
        self.assertEqual(['venv', 'dataset_vault'], self.checks() and self.failed)
        pip.with_name('moved.py').rename(pip)
        self.assertTrue(self.checks()['ok'])

    def test_live_pip_freeze_runs_like_volume_sh_with_a_60_second_limit(self):
        seen = []
        real = subprocess.run

        def spy(command, *args, **kwargs):
            if 'freeze' in command:
                seen.append((command[0], kwargs.get('timeout'), kwargs['env']['PYTHONPATH'],
                             kwargs.get('cwd')))
            return real(command, *args, **kwargs)
        with mock.patch.object(verify.subprocess, 'run', spy):
            self.assertTrue(self.checks()['ok'])
        repo = self.repo.resolve()
        self.assertEqual([(self.manifest['venv']+'/bin/python', 60, f'{repo}/src:{repo}', repo)],
                         seen)

    def test_lock_includes_are_hashed(self):
        env = self.root/'locks'
        env.mkdir()
        (env/'top.lock').write_text('-r inc.lock\n--constraint=c.lock\nx==1\n')
        (env/'inc.lock').write_text('--extra-index-url https://example.invalid\ny==2\n')
        (env/'c.lock').write_text('z==3\n')
        first = verify.lock_closure(env/'top.lock')
        self.assertEqual({'top.lock', 'inc.lock', 'c.lock'}, set(first))
        (env/'inc.lock').write_text('--extra-index-url https://example.invalid\ny==3\n')
        self.assertNotEqual(first, verify.lock_closure(env/'top.lock'))  # top sha unchanged
        (env/'top.lock').write_text('-r ../escape.lock\n')
        (self.root/'escape.lock').write_text('')
        with self.assertRaises(ValueError):
            verify.lock_closure(env/'top.lock')
        self.assertIn('requirements-experiments.lock', self.manifest['lock_closure_sha256'])
        self.rewrite_manifest(lock_closure_sha256={'requirements-experiments.lock': '0'*64})
        self.assertEqual(['lock_sha256'], self.checks() and self.failed)

    def test_manifest_cannot_point_outside_the_volume(self):
        outside = self.root/'outside.json'
        outside.write_text('{}')
        link = self.volume/'hf-cache/hub/models--x/snapshots/rev/config.json'
        link.unlink()
        link.symlink_to(outside)
        self.assertEqual(['model_files'], self.checks() and self.failed)
        self.rewrite_manifest(model_snapshot='../ws')
        self.assertIn('model_files', self.checks() and self.failed)

    def test_cli_writes_preflight_json_and_exit_code(self):
        runs = self.root/'ws/runs/r2'
        job_path = self.root/'frozen-job.json'
        job_path.write_text(json.dumps({'data': self.data, 'min_free_gib': 0.001,
                                        'stage': 'bf16-smoke-throughput'}))
        arguments = ['--job', str(job_path), '--commit', self.commit, '--volume-root',
                     str(self.volume), '--run-dir', str(runs), '--repo-root', str(self.repo),
                     '--mount-path', str(self.root/'ws')]
        with mock.patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(0, verify.main(arguments))
            written = json.loads((runs/'preflight.json').read_text())
            self.assertTrue(written['ok'])
            self.assertEqual(0, verify.main([*arguments, '--quick']))
            self.assertEqual('quick', json.loads((runs/'preflight.json').read_text())['hash_mode'])
            with mock.patch('sys.stderr', new=io.StringIO()):
                arguments[arguments.index('--commit')+1] = 'b'*40
                self.assertEqual(1, verify.main(arguments))
        self.assertFalse(json.loads((runs/'preflight.json').read_text())['ok'])


class RunnerTests(unittest.TestCase):
    def test_state_is_append_only_and_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'run_state.json'
            runner.record_state(path, 'PREFLIGHT_OK')
            runner.record_state(path, 'MAIN_EXIT 7', exit_code=7)
            value = json.loads(path.read_text())
            self.assertEqual('MAIN_EXIT 7', value['state'])
            self.assertEqual(['PREFLIGHT_OK', 'MAIN_EXIT 7'], [h['state'] for h in value['history']])
            with self.assertRaises(ValueError):
                runner.record_state(path, 'DONE')

    def test_run_logged_exit_code_timeout_and_group_kill(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary)/'log.txt'
            self.assertEqual(3, runner.run_logged(
                [sys.executable, '-c', 'import sys;print("out");print("err",file=sys.stderr);'
                 'sys.exit(3)'], log, 10))
            self.assertIn('out', log.read_text())
            self.assertIn('err', log.read_text())
            pidfile = Path(temporary)/'pid'
            started = time.monotonic()
            code = runner.run_logged([sys.executable, '-c',
                f'import os,time,pathlib;pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()));'
                'time.sleep(60)'], log, 1, grace=5)
            self.assertEqual(124, code)
            self.assertLess(time.monotonic()-started, 20)
            pid = int(pidfile.read_text())
            with self.assertRaises(ProcessLookupError):
                for _ in range(50):  # allow the reaped group a moment to vanish
                    os.kill(pid, 0)
                    time.sleep(.1)
                os.kill(pid, 0)

    def test_disk_low_stops_the_group_records_the_state_and_samples_are_logged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pidfile, state = root/'pid', root/'run_state.json'
            runner.record_state(state, 'MAIN_RUNNING')
            usage = shutil.disk_usage(root)
            samples = iter([usage._replace(free=10*2**30)]*3 + [usage._replace(free=2**29)]*99)
            started = time.monotonic()
            with mock.patch.object(runner.shutil, 'disk_usage', lambda _path: next(samples)):
                code = runner.run_logged([sys.executable, '-c',
                    f'import os,time,pathlib;pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()));'
                    'time.sleep(60)'], root/'log', 60, grace=5, disk_path=root, min_free_gib=1,
                    disk_log=root/'disk.jsonl', state_file=state, disk_interval=0.2)
            self.assertEqual(runner.DISK_LOW_CODE, code)
            self.assertLess(time.monotonic()-started, 20)
            record = json.loads(state.read_text())
            self.assertEqual('DISK_LOW', record['state'])
            self.assertEqual({'free_gib': 0.5, 'min_free_gib': 1}, {
                k: record['history'][-1][k] for k in ('free_gib', 'min_free_gib')})
            samples = [json.loads(line) for line in (root/'disk.jsonl').read_text().splitlines()]
            self.assertEqual([10.0, 10.0, 10.0, 0.5], [x['free_gib'] for x in samples])
            if pidfile.exists():
                with self.assertRaises(ProcessLookupError):
                    for _ in range(50):
                        os.kill(int(pidfile.read_text()), 0)
                        time.sleep(.1)
                    os.kill(int(pidfile.read_text()), 0)
            with mock.patch('sys.stdout', new=io.StringIO()) as out:
                self.assertEqual(0, runner.main(['last', str(state)]))
            self.assertEqual('DISK_LOW\n', out.getvalue())
            # Enough space: the command runs to completion and the samples keep coming.
            self.assertEqual(0, runner.run_logged([sys.executable, '-c', 'import time;time.sleep(.5)'],
                root/'log', 30, disk_path=root, min_free_gib=0.001, disk_log=root/'disk2.jsonl',
                state_file=state, disk_interval=0.1))
            self.assertGreaterEqual(len((root/'disk2.jsonl').read_text().splitlines()), 2)
            for bad in ({'state_file': None}, {'disk_log': None}, {'min_free_gib': 0},
                        {'disk_interval': 0}):
                arguments = dict(disk_path=root, min_free_gib=1, disk_log=root/'d.jsonl',
                                 state_file=state, disk_interval=1)
                arguments.update(bad)
                with self.assertRaises(ValueError, msg=str(bad)):
                    runner.run_logged([sys.executable, '-c', 'pass'], root/'log', 5, **arguments)
            for terminal in ('BUDGET_STOP', 'DISK_LOW'):
                runner.record_state(root/'s2.json', terminal)

    def test_repeated_sigterm_during_grace_still_kills_the_child_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            pidfile, log = Path(temporary)/'pid', Path(temporary)/'log'
            child = ('import os,signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                     f'pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(120)')
            code = ('import sys;from fine_tuning.notebook_rl.pod.runner import run_logged;'
                    f'sys.exit(run_logged([sys.executable,"-c",{child!r}],{str(log)!r},60,grace=2))')
            process = subprocess.Popen([sys.executable, '-B', '-c', code],
                                       cwd=str(Path(__file__).resolve().parents[2]))
            try:
                deadline = time.monotonic()+30
                while not pidfile.exists() or not pidfile.read_text():
                    self.assertLess(time.monotonic(), deadline, 'child never started')
                    time.sleep(.05)
                process.send_signal(15)
                time.sleep(.5)
                process.send_signal(15)  # with SIG_DFL restored this killed the runner
                process.wait(timeout=30)
            finally:
                if process.poll() is None:
                    process.kill()
            pid = int(pidfile.read_text())
            try:
                with self.assertRaises(ProcessLookupError):
                    for _ in range(50):
                        os.kill(pid, 0)
                        time.sleep(.1)
                    os.kill(pid, 0)
            finally:
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass

    def test_owner_record_has_startticks_and_collect_sees_it_alive(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'owner.json'
            owner = runner.write_owner(path, os.getpid())
            self.assertEqual(common.process_startticks(os.getpid()), owner['startticks'])
            self.assertTrue(collect.owner_alive(Path(temporary)))
            with mock.patch.object(collect, 'process_startticks', return_value=None):
                self.assertTrue(collect.owner_alive(Path(temporary)))  # unreadable -> alive
                finished = subprocess.Popen([sys.executable, '-c', 'pass'])
                finished.wait()
                common.write_json(path, {'pid': finished.pid, 'startticks': 1})
                self.assertFalse(collect.owner_alive(Path(temporary)))


class DataBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root/'artifacts/pool'
        (self.source/'vault/train').mkdir(parents=True)
        (self.source/'vault/train/a.jsonl').write_bytes(b'{"problem_id": "p1"}\n')
        (self.source/'vault/images').mkdir()
        (self.source/'vault/images/x.png').write_bytes(bytes(range(256))*50)
        (self.source/'manifest.json').write_text('{"items": []}\n')
        self.out = self.root/'out'

    def spec(self, *paths, name='spec.json'):
        path = self.root/name
        path.write_text(json.dumps({'version': data_bundle.SPEC_VERSION,
                                    'paths': [str(p) for p in paths]}))
        return path

    def pack(self):
        manifest = data_bundle.pack(self.spec(self.source/'vault', self.source/'manifest.json'),
                                    self.out, 'proof')
        return manifest, self.out/'proof.tar', self.out/'proof.bundle.json'

    def snapshot(self):
        return {str(p): p.read_bytes() for p in sorted(self.source.rglob('*')) if p.is_file()}

    def test_pack_then_stage_restores_identical_absolute_paths(self):
        before = self.snapshot()
        manifest, archive, bundle = self.pack()
        self.assertEqual(sorted(before), [f['path'] for f in manifest['files']])
        self.assertFalse(manifest['public_upload_allowed'])
        self.assertEqual([], data_bundle.verify(bundle, common.sha256_file(bundle), archive,
                                                covers=(str(self.source/'vault'),)))
        with self.assertRaisesRegex(data_bundle.BundleError, 'already exist'):
            data_bundle.stage(archive, bundle)  # the files are still there: never overwritten
        self.assertEqual(before, self.snapshot())
        shutil.move(self.source, self.root/'moved')  # the Pod: nothing at those paths yet
        self.assertTrue(data_bundle.verify(bundle))  # missing files are reported
        result = data_bundle.stage(archive, bundle, common.sha256_file(bundle))
        self.assertEqual(('STAGED', 3), (result['status'], result['files']))
        self.assertEqual(before, self.snapshot())
        self.assertEqual([], data_bundle.verify(bundle, common.sha256_file(bundle)))
        self.assertEqual(0o600, (self.source/'manifest.json').stat().st_mode & 0o777)
        (self.source/'vault/train/a.jsonl').write_bytes(b'{"problem_id": "p2"}\n')
        self.assertEqual(1, len(data_bundle.verify(bundle)))
        self.assertTrue(data_bundle.verify(bundle, '0'*64))  # pinned manifest sha
        self.assertTrue(any('not covered' in error for error in
                            data_bundle.verify(bundle, covers=(str(self.root/'x'),))))
        with self.assertRaisesRegex(data_bundle.BundleError, 'already exists'):
            data_bundle.pack(self.spec(self.source/'manifest.json'), self.out, 'proof')

    def test_stage_refuses_tampering_and_rolls_back_what_it_wrote(self):
        _, archive, bundle = self.pack()
        shutil.move(self.source, self.root/'moved')
        with self.assertRaisesRegex(data_bundle.BundleError, 'pinned'):
            data_bundle.stage(archive, bundle, '0'*64)
        with tarfile.open(archive) as tar:
            offset = tar.getmembers()[0].offset_data  # first byte of the first file's content
        tampered = bytearray(archive.read_bytes())
        tampered[offset] ^= 1
        archive.write_bytes(bytes(tampered))
        with self.assertRaisesRegex(data_bundle.BundleError, 'archive sha256'):
            data_bundle.stage(archive, bundle)
        value = json.loads(bundle.read_text())  # consistent archive sha, wrong file content
        value['archive_sha256'] = common.sha256_file(archive)
        bundle.write_text(json.dumps(value))
        with self.assertRaises(data_bundle.BundleError):
            data_bundle.stage(archive, bundle)
        self.assertEqual([], [p for p in self.root.rglob('*') if p.is_file()
                              and str(p).startswith(str(self.source))])  # rolled back

    def crafted(self, members):
        archive = self.root/'evil.tar'
        write_tar(archive, members)
        records = [{'path': '/'+info.name if not info.name.startswith('/') else info.name,
                    'bytes': info.size, 'sha256': hashlib.sha256(data or b'').hexdigest()}
                   for info, data in members]
        bundle = self.root/'evil.bundle.json'
        bundle.write_text(json.dumps({'version': data_bundle.VERSION, 'files': records,
                                      'archive_sha256': common.sha256_file(archive), 'bytes': 0}))
        return archive, bundle

    def test_stage_refuses_unsafe_members_and_symlinked_parents(self):
        base = str(self.root/'target').lstrip('/')
        for members in ([(tar_info(base+'/link', tarfile.SYMTYPE, link='/etc/passwd'), None)],
                        [(tar_info(base+'/hard', tarfile.LNKTYPE, link=base+'/f'), None)],
                        [(tar_info(base+'/dev', tarfile.CHRTYPE), None)],
                        [(tar_info(base+'/../escape', size=1), b'x')],
                        [(tar_info('/'+base+'/abs', size=1), b'x')]):
            archive, bundle = self.crafted(members)
            with self.assertRaises(data_bundle.BundleError, msg=members[0][0].name):
                data_bundle.stage(archive, bundle)
            self.assertFalse((self.root/'target').exists())
            self.assertFalse((self.root/'escape').exists())
        (self.root/'real').mkdir()
        (self.root/'target').symlink_to(self.root/'real')
        archive, bundle = self.crafted([(tar_info(base+'/sub/f', size=1), b'x')])
        with self.assertRaisesRegex(data_bundle.BundleError, 'symlink'):
            data_bundle.stage(archive, bundle)
        self.assertEqual([], list((self.root/'real').iterdir()))

    def test_pack_refuses_benchmark_cache_links_and_aliases(self):
        refused = []
        for name in ('MMMU_val', 'mmmu-pro', 'huggingface', 'hf-cache', 'models--Qwen'):
            path = self.root/name/'f.json'
            path.parent.mkdir()
            path.write_text('{}')
            refused.append(path)
        evaluation = self.root/'data/evaluation/set/f.json'
        evaluation.parent.mkdir(parents=True)
        evaluation.write_text('{}')
        hf_home = self.root/'cache-home/f.json'
        hf_home.parent.mkdir()
        hf_home.write_text('{}')
        with mock.patch.dict(os.environ, {'MMDL_DATA_ROOT': str(self.root/'data'),
                                          'HF_HOME': str(self.root/'cache-home')}):
            for index, path in enumerate([*refused, evaluation, hf_home]):
                with self.assertRaisesRegex(data_bundle.BundleError, 'refused', msg=str(path)):
                    data_bundle.pack(self.spec(path), self.out, f'b{index}')
            with self.assertRaisesRegex(data_bundle.BundleError, 'refused'):
                data_bundle.pack(self.spec(self.root/'data'), self.out, 'tree')  # inside a tree
        (self.source/'vault/link.json').symlink_to(self.source/'manifest.json')
        with self.assertRaisesRegex(data_bundle.BundleError, 'symlink'):
            data_bundle.pack(self.spec(self.source/'vault'), self.out, 'linked')
        (self.source/'vault/link.json').unlink()
        os.link(self.source/'manifest.json', self.source/'vault/alias.json')
        with self.assertRaisesRegex(data_bundle.BundleError, 'hardlink'):
            data_bundle.pack(self.spec(self.source/'vault'), self.out, 'aliased')
        for bad in ('relative/path', str(self.source)+'/../x', str(self.source)+'/'):
            with self.assertRaises(data_bundle.BundleError, msg=bad):
                data_bundle.pack(self.spec(bad), self.out, 'bad')
        self.assertFalse(any(self.out.glob('*.tar')) if self.out.exists() else False)

    def test_cli_pack_stage_verify(self):
        spec = self.spec(self.source/'vault', self.source/'manifest.json')
        with mock.patch('sys.stdout', new=io.StringIO()) as out:
            self.assertEqual(0, data_bundle.main(['pack', '--spec', str(spec), '--out-dir',
                                                  str(self.out), '--name', 'proof']))
        packed = json.loads(out.getvalue())
        bundle, archive = self.out/'proof.bundle.json', self.out/'proof.tar'
        self.assertEqual(common.sha256_file(bundle), packed['manifest_sha256'])
        shutil.move(self.source, self.root/'moved')
        with mock.patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(0, data_bundle.main(['stage', '--archive', str(archive), '--manifest',
                                                  str(bundle), '--manifest-sha256',
                                                  packed['manifest_sha256']]))
            self.assertEqual(0, data_bundle.main(['verify', '--manifest', str(bundle),
                                                  '--archive', str(archive)]))
            with mock.patch('sys.stderr', new=io.StringIO()) as err:
                self.assertEqual(2, data_bundle.main(['stage', '--archive', str(archive),
                                                      '--manifest', str(bundle)]))
            self.assertIn('already exist', err.getvalue())
            (self.source/'manifest.json').unlink()
            self.assertEqual(1, data_bundle.main(['verify', '--manifest', str(bundle)]))


class CostTests(unittest.TestCase):
    receipt = {'version': 'throughput-receipt-v1', 'items': 10, 'completion_tokens': 8000,
               'wall_seconds': 1000, 'updates': 5, 'update_seconds': 200, 'peak_memory_gib': 20.5,
               'n_samples': 4, 'retries_per_item': 0}
    plan = {'version': 'cost-plan-v1', 'hourly_usd': 0.6, 'setup_minutes': 12, 'reserve_usd': 1.0,
            'range': 0.5, 'gpu_memory_gib': 48,
            'stages': [{'name': 'pilot', 'items': 30, 'n_samples': 4, 'retries_per_item': 0},
                       {'name': 'main', 'items': 100, 'n_samples': 8, 'retries_per_item': 0}]}

    def project(self, cap=None, receipt=None, plan=None):
        return cost.project(receipt or self.receipt, plan or self.plan, cap,
                            receipt_sha256='r'*64, plan_sha256='p'*64)

    def test_arithmetic_on_synthetic_receipt(self):
        result = self.project()
        # generation 800 s / 10 items = 80 s per item; update 200 s / 10 items = 20 s per item
        pilot, main = result['stages']
        self.assertAlmostEqual(30*(80+20)/3600, pilot['hours']['mid'], places=4)  # 0.8333 h
        self.assertAlmostEqual(0.5, pilot['usd']['mid'], places=4)
        self.assertAlmostEqual(100*(160+20)/3600, main['hours']['mid'], places=4)  # 5.0 h, n x2
        self.assertAlmostEqual(3.0, main['usd']['mid'], places=4)
        self.assertEqual({'low': 1.5, 'mid': 3.0, 'high': 4.5}, main['usd'])
        self.assertAlmostEqual(0.2, result['setup']['hours']['mid'], places=4)
        self.assertAlmostEqual(0.2+30*100/3600+5.0, result['total']['hours']['mid'], places=3)
        self.assertAlmostEqual(3.62, result['total']['usd']['mid'], places=3)
        self.assertAlmostEqual(5.43, result['total']['usd']['high'], places=3)
        self.assertAlmostEqual(6.43, result['total']['usd_high_plus_reserve'], places=3)
        self.assertEqual(10.0, result['derived']['completion_tokens_per_generation_second'])
        self.assertEqual([], result['warnings'])
        self.assertEqual(result, self.project())  # deterministic

    def test_everything_is_labelled_projection_and_receipt_sha_recorded(self):
        result = self.project()
        self.assertEqual('projection', result['label'])
        for row in (*result['stages'], result['setup'], result['total'], result['derived']):
            self.assertEqual('projection', row['label'])
        self.assertEqual('r'*64, result['receipt_sha256'])
        self.assertEqual('measured', result['measured_receipt']['label'])
        self.assertIn('PROJECTIONS', cost.table(result).splitlines()[0])
        self.assertIn('r'*64, cost.table(result).splitlines()[0])

    def test_cap_refusal_uses_high_end_plus_reserve(self):
        self.assertFalse(self.project(7.0)['refuse'])
        self.assertTrue(self.project(6.0)['refuse'])
        self.assertIsNone(self.project()['cap_usd'])
        self.assertTrue(self.project(plan={**self.plan, 'cap_usd': 5.0})['refuse'])
        for bad in (float('nan'), 0.0, -1.0):  # NaN compared False and never refused
            with self.assertRaises(cost.CostError, msg=str(bad)):
                self.project(bad)

    def test_retries_scale_generation_and_memory_warning(self):
        retry_plan = {**self.plan, 'gpu_memory_gib': 20,
                      'stages': [{'name': 's', 'items': 10, 'n_samples': 4, 'retries_per_item': 1}]}
        result = self.project(plan=retry_plan)
        self.assertAlmostEqual(10*(80*2+20)/3600, result['stages'][0]['hours']['mid'], places=4)
        self.assertEqual(1, len(result['warnings']))

    def test_startup_seconds_leave_the_rates_and_count_once_per_stage(self):
        result = self.project(receipt={**self.receipt, 'startup_seconds': 100})
        # generation (1000-200-100)/10 = 70 s per item, update 20 s per item, startup 100 s per stage
        pilot, main = result['stages']
        self.assertAlmostEqual((30*(70+20)+100)/3600, pilot['hours']['mid'], places=4)
        self.assertAlmostEqual((100*(140+20)+100)/3600, main['hours']['mid'], places=4)
        self.assertEqual(100, result['measured_receipt']['startup_seconds'])
        self.assertAlmostEqual(8000/700, result['derived']['completion_tokens_per_generation_second'],
                               places=3)
        self.assertEqual(0, self.project()['measured_receipt']['startup_seconds'])  # optional
        for bad in (800, 1000, -1, float('nan'), True, '5'):  # update 200 + startup must be < wall
            with self.assertRaises(cost.CostError, msg=str(bad)):
                self.project(receipt={**self.receipt, 'startup_seconds': bad})

    def test_invalid_inputs_are_refused(self):
        for bad in ({**self.receipt, 'update_seconds': 1000}, {**self.receipt, 'items': 0},
                    {**self.receipt, 'wall_seconds': float('nan')},
                    {**self.receipt, 'updates': 0}, {**self.receipt, 'version': 'x'},
                    {**self.receipt, 'peak_memory_gib': True}):
            with self.assertRaises(cost.CostError, msg=str(bad)):
                self.project(receipt=bad)
        with self.assertRaises(cost.CostError):
            self.project(plan={**self.plan, 'stages': []})
        with self.assertRaises(cost.CostError):
            self.project(plan={**self.plan, 'range': 1})

    def test_cli_writes_json_with_receipt_sha_and_signals_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'r.json').write_text(json.dumps(self.receipt))
            (root/'p.json').write_text(json.dumps(self.plan))
            with mock.patch('sys.stdout', new=io.StringIO()) as out:
                self.assertEqual(0, cost.main(['--receipt', str(root/'r.json'), '--plan',
                    str(root/'p.json'), '--cap-usd', '7', '--output', str(root/'o.json')]))
                self.assertEqual(3, cost.main(['--receipt', str(root/'r.json'), '--plan',
                    str(root/'p.json'), '--cap-usd', '6', '--output', str(root/'o2.json')]))
            written = json.loads((root/'o2.json').read_text())
            self.assertEqual(common.sha256_file(root/'r.json'), written['receipt_sha256'])
            self.assertTrue(written['refuse'])
            self.assertIn('REFUSE = True', out.getvalue())
            with mock.patch('sys.stderr', new=io.StringIO()):
                self.assertEqual(2, cost.main(['--receipt', str(root/'r.json'), '--plan',
                    str(root/'p.json'), '--output', str(root/'o.json')]))  # no overwrite


def make_run_dir(root, name, state='MAIN_EXIT 0', result=True, owner_pid=None, ticks=None):
    run_dir = root/name
    (run_dir/'logs').mkdir(parents=True)
    (run_dir/'logs/main.log').write_text('log\n')
    if state:
        runner.record_state(run_dir/'run_state.json', state)
    if result:
        (run_dir/'result.json').write_text('{"ok":true}')
    if owner_pid is None:
        finished = subprocess.Popen([sys.executable, '-c', 'pass'])
        finished.wait()
        owner_pid, ticks = finished.pid, 12345
    ticks = common.process_startticks(owner_pid) if ticks is None else ticks
    common.write_json(run_dir/'owner.json', {'pid': owner_pid, 'startticks': ticks})
    return run_dir


def write_tar(path, members):
    with tarfile.open(path, 'w') as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)


def tar_info(name, kind=tarfile.REGTYPE, size=0, link=''):
    info = tarfile.TarInfo(name)
    info.type, info.size, info.linkname = kind, size, link
    return info


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_pack_refuses_live_owner_unclosed_and_resultless_runs(self):
        live = make_run_dir(self.root, 'live', state='MAIN_RUNNING', owner_pid=os.getpid())
        with self.assertRaisesRegex(collect.CollectError, 'still alive'):
            collect.pack([live], self.root/'out', 'a')
        done_but_alive = make_run_dir(self.root, 'alive2', owner_pid=os.getpid())
        with self.assertRaisesRegex(collect.CollectError, 'still alive'):
            collect.pack([done_but_alive], self.root/'out', 'a')
        running = make_run_dir(self.root, 'running', state='MAIN_RUNNING')
        with self.assertRaisesRegex(collect.CollectError, 'not closed'):
            collect.pack([running], self.root/'out', 'a')
        nostate = make_run_dir(self.root, 'nostate', state=None)
        with self.assertRaisesRegex(collect.CollectError, 'run_state.json'):
            collect.pack([nostate], self.root/'out', 'a')
        resultless = make_run_dir(self.root, 'resultless', result=False)
        with self.assertRaisesRegex(collect.CollectError, 'lacks result/receipt'):
            collect.pack([resultless], self.root/'out', 'a')
        self.assertFalse((self.root/'out/a.tar').exists())
        self.assertEqual([], [p for p in (self.root/'out').iterdir()] if (self.root/'out').exists()
                         else [])

    def test_pack_failed_runs_are_recoverable_and_owner_pid_reuse_is_not_alive(self):
        for state in ('BUDGET_STOP', 'DISK_LOW', 'INTERRUPTED', 'MAIN_EXIT 3'):
            self.assertEqual('CLOSED', collect.closed_run(make_run_dir(
                self.root, state.replace(' ', '_'), state=state, result=False)))
        failed = make_run_dir(self.root, 'failed', state='SMOKE_FAILED', result=False)
        reused = make_run_dir(self.root, 'reused', owner_pid=os.getpid(), ticks=1)
        manifest = collect.pack([failed, reused], self.root/'out', 'both')
        self.assertEqual(['failed', 'reused'], manifest['runs'])
        self.assertEqual(common.sha256_file(self.root/'out/both.tar'), manifest['archive_sha256'])
        self.assertIn('failed/run_state.json', [f['path'] for f in manifest['files']])
        with self.assertRaisesRegex(collect.CollectError, 'already exists'):
            collect.pack([failed], self.root/'out', 'both')

    def fake_proc(self, entries):
        """A /proc look-alike. entries: pid -> environ bytes, None (unreadable) or 'gone'."""
        proc = Path(tempfile.mkdtemp(dir=self.root))
        (proc/'cpuinfo').write_text('not a process')
        for pid, value in entries.items():
            (proc/pid).mkdir()
            if value is None:
                (proc/pid/'environ').mkdir()  # reading a directory raises OSError, as EACCES would
            elif value != 'gone':
                (proc/pid/'environ').write_bytes(value)
        return proc

    @staticmethod
    def names(run_dir, suffix=b''):
        return b'A=1\0MMDL_RUN_DIR=' + os.fsencode(run_dir) + suffix + b'\0B=2\0'

    def test_unclosed_dead_owner_pack_needs_a_dead_owner_and_no_process_naming_the_run_dir(self):
        out = self.root/'out'
        killed = make_run_dir(self.root, 'killed', state='MAIN_RUNNING', result=False)
        with self.assertRaisesRegex(collect.CollectError, 'not closed'):  # flag is opt-in
            collect.pack([killed], out, 'k0')
        live = make_run_dir(self.root, 'livekill', state='MAIN_RUNNING', owner_pid=os.getpid())
        with mock.patch.object(collect, 'PROC', self.fake_proc({})):
            with self.assertRaisesRegex(collect.CollectError, 'still alive'):
                collect.pack([live], out, 'k0', unclosed_dead_owner=True)
        # an orphaned main command that still carries MMDL_RUN_DIR=<run dir>
        proc = self.fake_proc({'11': b'A=1\0', '4242': self.names(killed)})
        with mock.patch.object(collect, 'PROC', proc):
            with self.assertRaisesRegex(collect.CollectError, r"still name this run dir: \['4242'\]"):
                collect.pack([killed], out, 'k1', unclosed_dead_owner=True)
        self.assertFalse(out.exists())
        # unreadable environ: fail closed. A process that vanished mid-scan or another run dir
        # whose name merely starts the same does not block.
        with mock.patch.object(collect, 'PROC', self.fake_proc({'7': None})):
            with self.assertRaisesRegex(collect.CollectError, r"7 \(environ unreadable\)"):
                collect.pack([killed], out, 'k1', unclosed_dead_owner=True)
        with mock.patch.object(collect, 'PROC', self.fake_proc({})):
            self.assertEqual([], collect.run_dir_holders(killed))
        with mock.patch.object(collect, 'PROC', self.root/'no-proc'):  # no /proc: cannot prove it
            with self.assertRaisesRegex(collect.CollectError, 'unavailable'):
                collect.pack([killed], out, 'k1', unclosed_dead_owner=True)
        proc = self.fake_proc({'12': 'gone', '13': self.names(killed, b'0'),
                               '14': self.names(self.root/'other')})
        with mock.patch.object(collect, 'PROC', proc):
            manifest = collect.pack([killed], out, 'k2', unclosed_dead_owner=True)
        self.assertEqual(('UNCLOSED', ['killed']), (manifest['closure'], manifest['unclosed_runs']))
        saved = json.loads((out/'k2.manifest.json').read_text())
        self.assertEqual('UNCLOSED', saved['closure'])
        receipt = collect.verify_archive(out/'k2.tar', out/'k2.manifest.json', self.root/'local',
                                         self.root/'r.json')
        self.assertEqual('UNCLOSED', receipt['closure'])
        self.assertIn('killed/run_state.json', [f['path'] for f in saved['files']])

    def test_closed_runs_stay_closed_with_the_flag_and_need_no_proc_scan(self):
        done = make_run_dir(self.root, 'done')
        failed = make_run_dir(self.root, 'failed', state='SMOKE_FAILED', result=False)
        with mock.patch.object(collect, 'PROC', self.root/'no-proc'):
            manifest = collect.pack([done, failed], self.root/'out', 'c', unclosed_dead_owner=True)
        self.assertEqual(('CLOSED', []), (manifest['closure'], manifest['unclosed_runs']))
        self.assertEqual('CLOSED', collect.closed_run(done))
        resultless = make_run_dir(self.root, 'resultless', result=False)  # MAIN_EXIT 0, no result
        with self.assertRaisesRegex(collect.CollectError, 'lacks result/receipt'):
            collect.closed_run(resultless, allow_unclosed=True)

    def test_run_dir_holders_reads_the_real_proc_and_the_cli_flag_labels_the_archive(self):
        killed = make_run_dir(self.root, 'real', state='MAIN_RUNNING', result=False)
        child = subprocess.Popen(['sleep', '30'], env={**os.environ, 'MMDL_RUN_DIR': str(killed)})
        try:
            self.assertIn(str(child.pid), collect.run_dir_holders(killed))
        finally:
            child.kill()
            child.wait()
        arguments = ['pack', '--out-dir', str(self.root/'out'), '--name', 'cli', str(killed)]
        with mock.patch.object(collect, 'PROC', self.fake_proc({})):
            with mock.patch('sys.stderr', new=io.StringIO()) as err:
                self.assertEqual(2, collect.main(arguments))
            self.assertIn('REFUSED', err.getvalue())
            with mock.patch('sys.stdout', new=io.StringIO()) as out:
                self.assertEqual(0, collect.main([*arguments[:-1], '--unclosed-dead-owner',
                                                  arguments[-1]]))
        self.assertEqual('UNCLOSED', json.loads(out.getvalue())['closure'])

    def test_pack_refuses_symlinks_and_leaves_no_partial_archive(self):
        run_dir = make_run_dir(self.root, 'linked')
        (run_dir/'escape').symlink_to('/etc/passwd')
        with self.assertRaisesRegex(collect.CollectError, 'only regular files'):
            collect.pack([run_dir], self.root/'out', 'x')
        self.assertEqual([], list((self.root/'out').iterdir()))

    def pack_good(self):
        good = make_run_dir(self.root, 'good')
        (good/'sub').mkdir()
        (good/'sub/data.bin').write_bytes(b'0123456789'*100)
        with mock.patch.dict(os.environ, {'RUNPOD_POD_ID': 'pod-from-env'}):
            manifest = collect.pack([good], self.root/'out', 'good')
        return manifest, self.root/'out/good.tar', self.root/'out/good.manifest.json'

    def test_verify_round_trip_and_tamper_detection(self):
        manifest, archive, manifest_path = self.pack_good()
        receipt = collect.verify_archive(archive, manifest_path, self.root/'local', self.root/'r.json')
        self.assertEqual('LOCAL_VERIFIED', receipt['status'])
        self.assertEqual(manifest['archive_sha256'], receipt['archive_sha256'])
        self.assertEqual('pod-from-env', receipt['pod_id'])  # what the Pod recorded, not typed
        self.assertEqual(b'0123456789'*100, (self.root/'local/good/sub/data.bin').read_bytes())
        with self.assertRaisesRegex(collect.CollectError, 'already exists'):
            collect.verify_archive(archive, manifest_path, self.root/'local', self.root/'r2.json')
        tampered = bytearray(archive.read_bytes())
        tampered[2048] ^= 1
        archive.write_bytes(bytes(tampered))
        with self.assertRaisesRegex(collect.CollectError, 'archive sha256 differs'):
            collect.verify_archive(archive, manifest_path, self.root/'local2', self.root/'r3.json')
        self.assertFalse((self.root/'local2').exists())

    def rebuilt(self, members, files=None):
        """Archive whose manifest is self-consistent, so only member validation can reject it."""
        archive = self.root/'evil.tar'
        write_tar(archive, members)
        names = files if files is not None else [i.name for i, d in members if d is not None]
        manifest = {'version': collect.PACK_VERSION, 'archive_sha256': common.sha256_file(archive),
                    'files': [{'path': n, 'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()}
                              for n in names]}
        (self.root/'evil.json').write_text(json.dumps(manifest))
        return archive, self.root/'evil.json'

    def test_verify_rejects_traversal_absolute_and_link_members(self):
        for index, members in enumerate((
                [(tar_info('../evil'), b'')],
                [(tar_info('good/../../evil'), b'')],
                [(tar_info('/abs/evil'), b'')],
                [(tar_info('good/link', tarfile.SYMTYPE, link='/etc/passwd'), None)],
                [(tar_info('good/link', tarfile.SYMTYPE, link='file'), None)],
                [(tar_info('good/hard', tarfile.LNKTYPE, link='good/file'), None)],
                [(tar_info('good/dev', tarfile.CHRTYPE), None)])):
            archive, manifest = self.rebuilt(members, files=[])
            target = self.root/f'target{index}'
            with self.assertRaises(collect.CollectError, msg=str(members[0][0].name)):
                collect.verify_archive(archive, manifest, target, self.root/f'rr{index}.json')
            self.assertFalse(target.exists())
            self.assertFalse((self.root.parent/'evil').exists())

    def test_verify_rejects_unlisted_and_changed_files(self):
        archive, manifest = self.rebuilt([(tar_info('good/a', size=1), b'x')], files=['good/b'])
        with self.assertRaisesRegex(collect.CollectError, 'file set differs'):
            collect.verify_archive(archive, manifest, self.root/'t1', self.root/'r1.json')
        archive, manifest = self.rebuilt([(tar_info('good/a', size=1), b'x')], files=['good/a'])
        with self.assertRaisesRegex(collect.CollectError, 'differs from manifest'):
            collect.verify_archive(archive, manifest, self.root/'t2', self.root/'r2.json')

    def test_ledger_enforces_order_evidence_and_integrity(self):
        ledger = self.root/'ledger.jsonl'
        archive_sha, receipt_sha = 'a'*64, 'b'*64
        for state in ('LOCAL_VERIFIED', 'POD_TERMINATED', 'CLEANUP_VERIFIED', 'NOPE'):
            with self.assertRaises(collect.CollectError, msg=state):
                collect.append_state(ledger, state, {})
        with self.assertRaisesRegex(collect.CollectError, 'evidence missing'):
            collect.append_state(ledger, 'REMOTE_VERIFIED', {'archive_sha256': archive_sha})
        remote = {'archive_sha256': archive_sha, 'manifest_sha256': 'c'*64, 'pod_id': 'pod1'}
        collect.append_state(ledger, 'REMOTE_VERIFIED', remote)
        with self.assertRaisesRegex(collect.CollectError, 'out-of-order'):
            collect.append_state(ledger, 'REMOTE_VERIFIED', remote)
        with self.assertRaisesRegex(collect.CollectError, 'out-of-order'):
            collect.append_state(ledger, 'POD_TERMINATED', {})
        with self.assertRaisesRegex(collect.CollectError, 'evidence missing'):
            collect.append_state(ledger, 'LOCAL_VERIFIED', {'archive_sha256': archive_sha})
        with self.assertRaisesRegex(collect.CollectError, 'differs from REMOTE_VERIFIED'):
            collect.append_state(ledger, 'LOCAL_VERIFIED',
                                 {'archive_sha256': 'd'*64, 'local_receipt_sha256': receipt_sha})
        # A bare hash is not evidence: the receipt written by `collect verify` is required.
        with self.assertRaisesRegex(collect.CollectError, 'receipt written by'):
            collect.append_state(ledger, 'LOCAL_VERIFIED',
                                 {'archive_sha256': archive_sha, 'local_receipt_sha256': receipt_sha})
        receipt = self.root/'receipt.json'
        good = {'version': 'pod-local-receipt-v1', 'status': 'LOCAL_VERIFIED',
                'archive_sha256': archive_sha, 'manifest_sha256': 'c'*64, 'pod_id': 'pod1'}
        for bad in ({**good, 'archive_sha256': 'd'*64}, {**good, 'manifest_sha256': 'd'*64},
                    {**good, 'status': 'FAILED'}):
            receipt.write_text(json.dumps(bad))
            with self.assertRaisesRegex(collect.CollectError, 'receipt written by', msg=str(bad)):
                collect.append_state(ledger, 'LOCAL_VERIFIED', {'archive_sha256': archive_sha},
                                     receipt_file=receipt)
        for bad in ({**good, 'pod_id': 'pod2'}, {**good, 'pod_id': None}):  # typed vs packed pod
            receipt.write_text(json.dumps(bad))
            with self.assertRaisesRegex(collect.CollectError, 'pod_id', msg=str(bad)):
                collect.append_state(ledger, 'LOCAL_VERIFIED', {'archive_sha256': archive_sha},
                                     receipt_file=receipt)
        receipt.write_text(json.dumps(good))
        collect.append_state(ledger, 'LOCAL_VERIFIED', {'archive_sha256': archive_sha},
                             receipt_file=receipt)
        self.assertEqual(common.sha256_file(receipt),
                         collect.read_ledger(ledger)[1]['evidence']['local_receipt_sha256'])
        terminated = {'pod_id': 'pod1', 'terminated_utc': utc(), 'confirmed_by': 'user'}
        for bad in ({**terminated, 'confirmed_by': 'script'}, {**terminated, 'pod_id': 'pod2'},
                    {**terminated, 'terminated_utc': utc(hours=-3)},
                    {**terminated, 'terminated_utc': utc(hours=1)},
                    {**terminated, 'terminated_utc': '2026-10-30T10:00:00'}):
            with self.assertRaises(collect.CollectError, msg=str(bad)):
                collect.append_state(ledger, 'POD_TERMINATED', bad)
        collect.append_state(ledger, 'POD_TERMINATED', terminated)
        deleted = {'deleted_resources': 'volume:abc', 'deleted_utc': utc(), 'confirmed_by': 'user'}
        other = self.root/'other.jsonl'  # an earlier stage that used the same volume
        collect.append_state(other, 'REMOTE_VERIFIED', {**remote, 'pod_id': 'pod0'})
        for bad in ({**deleted, 'confirmed_by': ''}, {**deleted, 'deleted_utc': utc(hours=-1)}):
            with self.assertRaises(collect.CollectError, msg=str(bad)):  # before termination
                collect.append_state(ledger, 'OWNED_STORAGE_DELETED', bad,
                                     run_ledgers=[ledger])
        for runs in (None, [], [other], [ledger, ledger], [ledger, other],
                     [ledger, self.root/'missing.jsonl']):
            with self.assertRaises((collect.CollectError, OSError), msg=str(runs)):
                collect.append_state(ledger, 'OWNED_STORAGE_DELETED', deleted, run_ledgers=runs)
        receipt.write_text(json.dumps({**good, 'pod_id': 'pod0'}))
        collect.append_state(other, 'LOCAL_VERIFIED', {'archive_sha256': archive_sha},
                             receipt_file=receipt)
        record = collect.append_state(ledger, 'OWNED_STORAGE_DELETED',
                                      {**deleted, 'run_ledgers': 'forged'},
                                      run_ledgers=[ledger, other])
        bound = json.loads(record['evidence']['run_ledgers'])
        self.assertEqual(['pod1', 'pod0'], [entry['pod_id'] for entry in bound])
        self.assertEqual(collect.read_ledger(other)[-1]['record_sha256'],
                         bound[1]['last_record_sha256'])
        final = {'residual_query_utc': utc(), 'residual_count': '1', 'confirmed_by': 'user'}
        with self.assertRaisesRegex(collect.CollectError, 'must be 0'):
            collect.append_state(ledger, 'CLEANUP_VERIFIED', final)
        with self.assertRaisesRegex(collect.CollectError, 'between deleted_utc'):
            collect.append_state(ledger, 'CLEANUP_VERIFIED', {
                **final, 'residual_count': '0', 'residual_query_utc': utc(hours=-1)})
        records = collect.append_state(ledger, 'CLEANUP_VERIFIED', {**final, 'residual_count': '0'})
        self.assertEqual(4, records['seq'])
        with self.assertRaisesRegex(collect.CollectError, 'out-of-order'):
            collect.append_state(ledger, 'CLEANUP_VERIFIED', final)
        self.assertEqual(list(collect.ORDER), [r['state'] for r in collect.read_ledger(ledger)])
        lines = ledger.read_text().splitlines()
        ledger.write_text('\n'.join([lines[0].replace('pod1', 'podX'), *lines[1:]])+'\n')
        with self.assertRaisesRegex(collect.CollectError, 'corrupt'):
            collect.read_ledger(ledger)

    def test_ledger_cli_uses_receipt_file_hash(self):
        ledger, receipt = self.root/'l.jsonl', self.root/'receipt.json'
        receipt.write_text(json.dumps({'version': 'pod-local-receipt-v1', 'status': 'LOCAL_VERIFIED',
                                       'archive_sha256': 'a'*64, 'manifest_sha256': 'b'*64,
                                       'pod_id': 'p'}))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        base = [sys.executable, '-m', 'fine_tuning.notebook_rl.pod.collect', 'state',
                '--ledger', str(ledger), 'append']
        cwd = str(Path(__file__).resolve().parents[2])
        first = run(*base, 'REMOTE_VERIFIED', '--evidence', f'archive_sha256={"a"*64}',
                    '--evidence', f'manifest_sha256={"b"*64}', '--evidence', 'pod_id=p', cwd=cwd,
                    env=env)
        self.assertEqual(0, first.returncode, first.stderr)
        second = run(*base, 'LOCAL_VERIFIED', '--evidence', f'archive_sha256={"a"*64}',
                     '--receipt-file', str(receipt), cwd=cwd, env=env)
        self.assertEqual(0, second.returncode, second.stderr)
        self.assertEqual(common.sha256_file(receipt),
                         collect.read_ledger(ledger)[1]['evidence']['local_receipt_sha256'])
        refused = run(*base, 'CLEANUP_VERIFIED', cwd=cwd, env=env)
        self.assertEqual(2, refused.returncode)
        self.assertIn('REFUSED', refused.stderr)
        third = run(*base, 'POD_TERMINATED', '--evidence', 'pod_id=p', '--evidence',
                    f'terminated_utc={utc()}', '--evidence', 'confirmed_by=user', cwd=cwd, env=env)
        self.assertEqual(0, third.returncode, third.stderr)
        deleted = ['--evidence', 'deleted_resources=volume:v', '--evidence', f'deleted_utc={utc()}',
                   '--evidence', 'confirmed_by=user']
        refused = run(*base, 'OWNED_STORAGE_DELETED', *deleted, cwd=cwd, env=env)
        self.assertEqual(2, refused.returncode)
        self.assertIn('--run-ledger', refused.stderr)
        fourth = run(*base, 'OWNED_STORAGE_DELETED', *deleted, '--run-ledger', str(ledger),
                     cwd=cwd, env=env)
        self.assertEqual(0, fourth.returncode, fourth.stderr)


class ScriptFixture(unittest.TestCase):
    """Sandboxed copy of start.sh/volume.sh: /workspace -> temp dir, fake mountpoint, real git."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo, self.git_env = make_git_repo(self.root)
        (self.root/'ws').mkdir()
        self.volume, self.manifest = make_volume(self.root, self.repo)
        self.workspace = self.root/'ws'
        self.data = make_data(self.root)

    def install(self, smoke_timeout=None):
        pod = self.repo/'fine_tuning/notebook_rl/pod'
        for name in ('start.sh', 'volume.sh'):
            text = (POD/name).read_text().replace('/workspace', str(self.workspace))
            if smoke_timeout:
                text = text.replace('smoke_timeout=300', f'smoke_timeout={smoke_timeout}')
            (pod/name).write_text(text)
            (pod/name).chmod(0o755)
        self.commit = commit_repo(self.repo, self.git_env)
        # Training preflight (eval_isolation): $HOME/mmdl-data/evaluation -> <volume>/eval-data, as on a Pod.
        home = self.root/'home'
        (home/'mmdl-data').mkdir(parents=True, exist_ok=True)
        if not (home/'mmdl-data/evaluation').is_symlink():
            (home/'mmdl-data/evaluation').symlink_to(self.volume.resolve()/'eval-data')
        self.environment = {**self.git_env,
            'PATH': f"{fake_commands(self.root)}{os.pathsep}{os.environ['PATH']}",
            'RUNPOD_POD_ID': 'fixture-pod', 'MMDL_POD_PYTHON': sys.executable,
            'PYTHONDONTWRITEBYTECODE': '1', 'HOME': str(home)}
        return pod

    @property
    def venv_python(self):
        return self.manifest['venv']+'/bin/python'

    def job_file(self, name='job.json', **changes):
        value = approved_job(None, **{
            'commit': self.commit, 'volume_root': str(self.volume),
            'run_dir': str(self.workspace/'runs/r1'), 'max_main_seconds': 60,
            'smoke': [self.venv_python, '-c', 'print("smoke ok")'],
            'main': [self.venv_python, '-c', 'print("main ok")'], 'data': self.data, **changes})
        path = self.root/name
        path.write_text(json.dumps(value))
        return path

    def start(self, pod, job_path, *extra, execute=True, environment=None):
        arguments = ['--job', str(job_path), *extra]
        if execute:
            arguments.append('--execute')
        return run('bash', pod/'start.sh', *arguments, env=environment or self.environment,
                   timeout=120)

    def states(self, run_dir=None):
        path = (run_dir or self.workspace/'runs/r1')/'run_state.json'
        return [h['state'] for h in json.loads(path.read_text())['history']]


class StartScriptTests(ScriptFixture):
    def test_shell_syntax(self):
        for name in ('start.sh', 'volume.sh'):
            result = run('bash', '-n', POD/name)
            self.assertEqual(0, result.returncode, result.stderr)

    def test_dry_run_prints_plan_without_executing(self):
        pod = self.install()
        marker = self.root/'ran'
        job_path = self.job_file(main=[self.venv_python, '-c',
                                       f'import pathlib;pathlib.Path({str(marker)!r}).touch()'])
        result = self.start(pod, job_path, execute=False)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('DRY_RUN: nothing executed', result.stdout)
        self.assertIn('bf16-smoke-throughput', result.stdout)
        self.assertIn('"fits": true', result.stdout)
        self.assertFalse(marker.exists())
        self.assertFalse((self.workspace/'runs').exists())

    def test_refuses_missing_approval_bad_job_wrong_pod_and_existing_run_dir(self):
        pod = self.install()
        for changes in ({'approval': None}, {'stage': 'everything'}, {'commit': 'abc'},
                        {'main': 'python x'}):
            result = self.start(pod, self.job_file(**changes))
            self.assertEqual(2, result.returncode, changes)
            self.assertFalse((self.workspace/'runs/r1').exists())
        result = self.start(pod, self.job_file(), environment={**self.environment,
                                                                'RUNPOD_POD_ID': 'someone-else'})
        self.assertEqual(2, result.returncode)
        self.assertIn('not job.financial.owned_pod_id', result.stdout+result.stderr)
        result = self.start(pod, self.job_file(), '--hourly-usd', '0.01')  # bounds only from the job
        self.assertEqual(2, result.returncode)
        self.assertIn('Unknown argument', result.stderr)
        edited = json.loads(self.job_file().read_text())
        edited['financial']['balance_usd'] = 100  # changed after the user approved it
        (self.root/'edited.json').write_text(json.dumps(edited))
        result = self.start(pod, self.root/'edited.json')
        self.assertEqual(2, result.returncode)
        self.assertIn('approved_sha256', result.stderr)
        result = self.start(pod, self.job_file(financial=financial(billing_start_utc=utc(hours=1))))
        self.assertEqual(2, result.returncode)
        self.assertIn('future', result.stderr)
        self.assertFalse((self.workspace/'runs/r1').exists())
        result = self.start(pod, Path(os.path.relpath(self.job_file(), Path.cwd())))
        self.assertEqual(2, result.returncode)
        self.assertIn('must be absolute', result.stderr)
        too_long = self.job_file(max_main_seconds=7200)  # does not fit before the 2 h cutoff
        result = self.start(pod, too_long)
        self.assertEqual(2, result.returncode)
        self.assertIn('does not fit', result.stderr)
        (self.workspace/'runs/r1').mkdir(parents=True)
        result = self.start(pod, self.job_file())
        self.assertEqual(2, result.returncode)
        self.assertIn('Existing run dir', result.stdout+result.stderr)

    def test_reexecs_under_the_budget_watchdog_with_exact_bounds(self):
        pod = self.install()
        calls = self.root/'calls.txt'
        fake = self.root/'fakepython'
        fake.write_text('#!/usr/bin/env bash\nif [[ $* == *fine_tuning.budget* ]]; then\n'
                        f'printf "%s\\n" "$*" >> {calls}; exit 99; fi\n'
                        f'exec {sys.executable} "$@"\n')
        fake.chmod(0o755)
        result = self.start(pod, self.job_file(), environment={
            **self.environment, 'MMDL_POD_PYTHON': str(fake), 'MMDL_BUDGET_WATCHDOG_PID': 'wrong'})
        self.assertEqual(99, result.returncode, result.stderr)
        text = calls.read_text()
        self.assertIn('-m fine_tuning.budget --owned-pod-id fixture-pod', text)
        run_dir = self.workspace/'runs/r1'  # 60 main + 300 smoke + 600 preflight:
        self.assertIn(f'--max-seconds 960 --result-file {run_dir}/budget_result.json -- bash', text)
        self.assertIn('--reserve-usd 1 --hourly-usd 1 --max-seconds', text)
        self.assertIn('start.sh --job', text)
        self.assertFalse((self.workspace/'runs/r1').exists())

    def test_runs_under_the_real_watchdog_and_exports_the_run_dir(self):
        pod = self.install()
        run_dir = self.workspace/'runs/r1'
        result = self.start(pod, self.job_file(main=[self.venv_python, '-c',
            'import os;print("env", os.environ["MMDL_BUDGET_WATCHDOG_PID"], os.environ["MMDL_RUN_DIR"])']))
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('"status": "EXITED"', result.stdout)  # printed by the real fine_tuning.budget
        _, watchdog, exported = (run_dir/'logs/main.log').read_text().split()
        self.assertNotEqual(str(os.getpid()), watchdog)
        self.assertEqual(str(run_dir), exported)  # collect's hard-kill scan looks for this


    def test_spoofed_watchdog_refused(self):
        pod = self.install()
        result = self.start(pod, self.job_file(), environment={
            **self.environment, 'MMDL_BUDGET_WATCHDOG_PID': str(os.getpid())})  # our parent is us
        self.assertEqual(2, result.returncode, result.stderr)
        self.assertIn('watchdog argv mismatch', result.stderr)
        self.assertFalse((self.workspace/'runs').exists())

    def test_watchdog_with_any_other_argv_is_refused_and_the_exact_one_is_accepted(self):
        pod = self.install()
        job_path = self.job_file()
        bounds = job.financial_argv(json.loads(job_path.read_text()))
        script = os.path.realpath(pod/'start.sh')
        result_file = f"{self.workspace/'runs/r1'}/budget_result.json"

        def under_budget(max_seconds):
            return run(sys.executable, '-m', 'fine_tuning.budget', *bounds,
                       '--max-seconds', max_seconds, '--result-file', result_file, '--', 'bash',
                       script, '--job', str(job_path), '--execute',
                       cwd=self.repo, env={**self.environment,
                       'PYTHONPATH': f'{self.repo}/src:{self.repo}'}, timeout=120)
        # A genuine fine_tuning.budget parent, one second off: only the exact argv passes.
        result = under_budget('961')
        self.assertEqual(2, result.returncode, result.stdout+result.stderr)
        self.assertIn('watchdog argv mismatch', result.stderr)
        self.assertFalse((self.workspace/'runs').exists())
        result = under_budget('960')  # 60 main + 300 smoke + 600 preflight
        self.assertEqual(0, result.returncode, result.stdout+result.stderr)
        self.assertEqual('MAIN_EXIT 0', self.states()[-1])

    def test_foreign_interpreter_is_refused_after_preflight(self):
        pod = self.install()
        marker = self.root/'ran'
        touch = f'import pathlib;pathlib.Path({str(marker)!r}).touch()'
        good = [self.venv_python, '-c', 'pass']
        for index, (smoke, main) in enumerate((
                ([sys.executable, '-c', touch], good), (good, [sys.executable, '-c', touch]),
                ([self.venv_python+'3', '-c', touch], good), (['python', '-c', touch], good))):
            run_dir = self.workspace/f'runs/f{index}'
            result = self.start(pod, self.job_file(name=f'f{index}.json', run_dir=str(run_dir),
                                                   smoke=smoke, main=main))
            self.assertEqual(2, result.returncode, result.stderr)
            self.assertIn('volume venv interpreter', result.stderr)
            self.assertEqual(['PREFLIGHT_OK', 'PREFLIGHT_FAILED'], self.states(run_dir))
            self.assertEqual('CLOSED', collect.closed_run(run_dir))  # recoverable as a failed run
        self.assertFalse(marker.exists())

    def test_success_records_all_transitions_logs_and_copies_the_job(self):
        pod = self.install()
        result = self.start(pod, self.job_file())
        self.assertEqual(0, result.returncode, result.stderr)
        run_dir = self.workspace/'runs/r1'
        self.assertEqual(['PREFLIGHT_OK', 'SMOKE_OK', 'MAIN_RUNNING', 'MAIN_EXIT 0'], self.states())
        self.assertIn('smoke ok', (run_dir/'logs/smoke.log').read_text())
        self.assertIn('main ok', (run_dir/'logs/main.log').read_text())
        preflight = json.loads((run_dir/'preflight.json').read_text())
        self.assertTrue(preflight['ok'])
        self.assertIn('dataset_vault', [c['name'] for c in preflight['checks']])
        plan = json.loads((run_dir/'plan.json').read_text())
        self.assertTrue(plan['budget']['fits'])
        self.assertEqual(common.sha256_file(self.root/'job.json'), plan['job_sha256'])
        budget = json.loads((run_dir/'budget_result.json').read_text())
        self.assertEqual(('EXITED', 0), (budget['status'], budget['returncode']))
        samples = (run_dir/'logs/disk.jsonl').read_text().splitlines()
        self.assertGreaterEqual(len(samples), 2)  # at least one each for smoke and main
        self.assertIn('free_gib', json.loads(samples[0]))
        self.assertEqual((self.root/'job.json').read_bytes(), (run_dir/'job.json').read_bytes())
        owner = json.loads((run_dir/'owner.json').read_text())
        self.assertEqual({'pid', 'startticks', 'utc'}, set(owner))
        self.assertIsNone(common.process_startticks(owner['pid']))  # the script has exited

    def test_closed_run_needs_result_after_success(self):
        pod = self.install()
        self.assertEqual(0, self.start(pod, self.job_file()).returncode)
        run_dir = self.workspace/'runs/r1'
        with self.assertRaisesRegex(collect.CollectError, 'lacks result/receipt'):
            collect.closed_run(run_dir)
        (run_dir/'result.json').write_text('{}')
        collect.closed_run(run_dir)

    def test_nonzero_main_exit_is_recorded_and_returned(self):
        pod = self.install()
        result = self.start(pod, self.job_file(main=[self.venv_python, '-c',
            'import sys;print("failing");sys.exit(7)']))
        self.assertEqual(7, result.returncode, result.stderr)
        self.assertEqual(['PREFLIGHT_OK', 'SMOKE_OK', 'MAIN_RUNNING', 'MAIN_EXIT 7'], self.states())
        state = json.loads((self.workspace/'runs/r1/run_state.json').read_text())
        self.assertEqual(7, state['history'][-1]['exit_code'])
        collect.closed_run(self.workspace/'runs/r1')  # failed runs stay recoverable

    def test_smoke_failure_and_timeout_never_start_main(self):
        marker = self.root/'main-ran'
        main = [self.venv_python, '-c', f'import pathlib;pathlib.Path({str(marker)!r}).touch()']
        pod = self.install()
        result = self.start(pod, self.job_file(smoke=[self.venv_python, '-c', 'import sys;sys.exit(5)'],
                                                main=main))
        self.assertEqual(4, result.returncode, result.stderr)
        self.assertEqual(['PREFLIGHT_OK', 'SMOKE_FAILED'], self.states())
        self.assertFalse(marker.exists())
        pod = self.install(smoke_timeout=2)
        self.assertIn('smoke_timeout=2', (pod/'start.sh').read_text())
        started = time.monotonic()
        result = self.start(pod, self.job_file(
            name='slow.json', run_dir=str(self.workspace/'runs/r2'),
            smoke=[self.venv_python, '-c', 'import time;time.sleep(60)'], main=main))
        self.assertEqual(4, result.returncode, result.stderr)
        self.assertLess(time.monotonic()-started, 30)
        self.assertEqual(['PREFLIGHT_OK', 'SMOKE_FAILED'], self.states(self.workspace/'runs/r2'))
        history = json.loads((self.workspace/'runs/r2/run_state.json').read_text())['history']
        self.assertEqual(124, history[-1]['exit_code'])
        self.assertFalse(marker.exists())

    def test_main_timeout_is_recorded_as_exit_124(self):
        pod = self.install()
        result = self.start(pod, self.job_file(max_main_seconds=1, main=[
            self.venv_python, '-c', 'import time;time.sleep(60)']))
        self.assertEqual(124, result.returncode, result.stderr)
        self.assertEqual('MAIN_EXIT 124', self.states()[-1])

    def test_preflight_failure_stops_before_smoke(self):
        pod = self.install()
        (self.volume/'hf-cache/hub/models--x/blobs/config.json').write_bytes(b'{"a":2}')
        marker = self.root/'smoke-ran'
        result = self.start(pod, self.job_file(smoke=[
            self.venv_python, '-c', f'import pathlib;pathlib.Path({str(marker)!r}).touch()']))
        self.assertEqual(3, result.returncode, result.stderr)
        self.assertEqual(['PREFLIGHT_FAILED'], self.states())
        self.assertFalse(marker.exists())
        self.assertFalse(json.loads((self.workspace/'runs/r1/preflight.json').read_text())['ok'])

    def test_unverified_training_data_stops_before_smoke(self):
        pod = self.install()
        marker = self.root/'smoke-ran'
        touch = [self.venv_python, '-c', f'import pathlib;pathlib.Path({str(marker)!r}).touch()']
        (Path(self.data['vault_root'])/'MISMATCH').write_text('')  # pool_freeze.verify != []
        self.assertEqual(3, self.start(pod, self.job_file(smoke=touch)).returncode)
        (Path(self.data['vault_root'])/'MISMATCH').unlink()
        staged = Path(self.data['dataset_manifest'])
        staged.write_text(staged.read_text()+' ')  # staged file differs from the bundle and pin
        run_dir = self.workspace/'runs/r2'
        self.assertEqual(3, self.start(pod, self.job_file('j2.json', run_dir=str(run_dir),
                                                          smoke=touch)).returncode)
        for directory, failed in ((self.workspace/'runs/r1', ['dataset_vault']),
                                  (run_dir, ['data_bundle', 'dataset_vault'])):
            preflight = json.loads((directory/'preflight.json').read_text())
            self.assertEqual(failed, [c['name'] for c in preflight['checks'] if not c['ok']])
            self.assertEqual(['PREFLIGHT_FAILED'], self.states(directory))
        self.assertFalse(marker.exists())

    def test_disk_low_after_smoke_stops_before_main(self):
        # The runner's own DISK_LOW path is unit-tested; here the smoke command records it, to show
        # that start.sh then stops with exit 5 and never starts main.
        pod = self.install()
        marker = self.root/'main-ran'
        smoke = [self.venv_python, '-c', 'import os;from fine_tuning.notebook_rl.pod.runner import '
                 'record_state;record_state(os.environ["MMDL_RUN_DIR"]+"/run_state.json","DISK_LOW")']
        result = self.start(pod, self.job_file(smoke=smoke, main=[
            self.venv_python, '-c', f'import pathlib;pathlib.Path({str(marker)!r}).touch()']))
        self.assertEqual(5, result.returncode, result.stderr)
        self.assertIn('DISK_LOW', result.stderr)
        self.assertEqual(['PREFLIGHT_OK', 'DISK_LOW'], self.states())
        self.assertFalse(marker.exists())

    def test_quick_verify_is_recorded(self):
        pod = self.install()
        result = self.start(pod, self.job_file(), '--quick-verify')
        self.assertEqual(0, result.returncode, result.stderr)
        preflight = json.loads((self.workspace/'runs/r1/preflight.json').read_text())
        self.assertEqual('quick', preflight['hash_mode'])
        self.assertFalse(preflight['full_sha256_verified'])

    def terminate_main(self, before=''):
        """Start a run, SIGTERM the watchdog while main sleeps; return the main pid."""
        pod = self.install()
        pidfile = self.root/'main.pid'
        job_path = self.job_file(main=[self.venv_python, '-c',
            f'import os,time,pathlib;{before}'
            f'pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(120)'])
        arguments = ['bash', str(pod/'start.sh'), '--job', str(job_path), '--execute']
        process = subprocess.Popen(arguments, env=self.environment, start_new_session=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic()+60
            while not pidfile.exists() or not pidfile.read_text():
                self.assertLess(time.monotonic(), deadline, 'main never started')
                self.assertIsNone(process.poll())
                time.sleep(.1)
            os.killpg(process.pid, 15)  # start.sh exec'd the real watchdog (same pid): it forwards
            process.communicate(timeout=60)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
        self.assertEqual(143, process.returncode)
        with self.assertRaises(ProcessLookupError):
            for _ in range(50):
                os.kill(int(pidfile.read_text()), 0)
                time.sleep(.1)
            os.kill(int(pidfile.read_text()), 0)
        return json.loads((self.workspace/'runs/r1/budget_result.json').read_text())

    def test_sigterm_from_the_watchdog_records_interrupted_and_stops_the_command(self):
        budget = self.terminate_main()
        self.assertEqual('INTERRUPTED', self.states()[-1])
        self.assertEqual('INTERRUPTED', budget['status'])  # a manual kill, not a budget stop

    def test_budget_stop_is_recorded_apart_from_a_manual_kill(self):
        # The watchdog writes BUDGET_STOP to budget_result.json before stopping the group (tested
        # in test_budget); main forges that file here so the deadline need not really pass.
        forged = ('pathlib.Path(os.environ["MMDL_RUN_DIR"],"budget_result.json").write_text('
                  '"{\\"status\\": \\"BUDGET_STOP\\"}");')
        self.terminate_main(before=forged)
        self.assertEqual('BUDGET_STOP', self.states()[-1])
        self.assertEqual('CLOSED', collect.closed_run(self.workspace/'runs/r1'))


class VolumeScriptTests(ScriptFixture):
    def setUp(self):
        super().setUp()
        shutil.rmtree(self.volume)  # volume.sh must start from a nonexistent target
        self.target = self.workspace/'new-volume'
        (self.repo/'src/venv').mkdir(parents=True)  # shadows `python -m venv` via volume.sh's PYTHONPATH
        (self.repo/'src/venv/__init__.py').write_text('')
        (self.repo/'src/venv/__main__.py').write_text(FAKE_VENV)

    def prep(self, name='prep.json', **changes):
        path = self.root/name
        path.write_text(json.dumps(prep_job(**{'commit': self.commit,
                                               'volume_root': str(self.target), **changes})))
        return path

    def arguments(self, *extra, execute=True, **changes):
        return ['--job', str(self.prep(**changes)), *extra, *(['--execute'] if execute else [])]

    def test_dry_run_default_prints_plan_and_creates_nothing(self):
        pod = self.install()
        result = run('bash', pod/'volume.sh', *self.arguments(execute=False), env=self.environment)
        self.assertEqual(0, result.returncode, result.stderr)
        lock_sha = common.sha256_file(self.repo/'env/requirements-experiments.lock')
        self.assertIn('DRY_RUN: nothing executed', result.stdout)
        self.assertIn('"fits": true', result.stdout)
        self.assertIn(f'venv-{lock_sha[:16]}', result.stdout)
        self.assertIn('VOLUME_READY', result.stdout)
        self.assertFalse(self.target.exists())

    def test_refusals_before_any_change(self):
        pod = self.install()
        for changes in ({'commit': 'b'*40}, {'commit': 'short'}, {'volume_root': '/elsewhere/vol'},
                        {'volume_root': str(self.workspace)}, {'volume_root': f'{self.target}/../x'},
                        {'environment': 'cuda'}, {'approval': None},
                        {'financial': financial(owned_pod_id='someone-else')}):
            result = run('bash', pod/'volume.sh', *self.arguments(**changes), env=self.environment)
            self.assertEqual(2, result.returncode, changes)
        training = self.root/'training.json'
        training.write_text(json.dumps(approved_job(None, commit=self.commit)))
        result = run('bash', pod/'volume.sh', '--job', str(training), env=self.environment)
        self.assertEqual(2, result.returncode)
        result = run('bash', pod/'volume.sh', '--commit', self.commit, '--volume-root',
                     str(self.target), env=self.environment)  # no CLI path around the job
        self.assertEqual(2, result.returncode)
        edited = json.loads(self.prep().read_text())
        edited['environment'] = 'vllm'  # changed after approval
        (self.root/'edited.json').write_text(json.dumps(edited))
        result = run('bash', pod/'volume.sh', '--job', str(self.root/'edited.json'), '--execute',
                     env=self.environment)
        self.assertEqual(2, result.returncode)
        self.assertIn('approved_sha256', result.stderr)
        (self.repo/'dirty.txt').write_text('x')
        result = run('bash', pod/'volume.sh', *self.arguments(execute=False), env=self.environment)
        self.assertEqual(2, result.returncode)
        (self.repo/'dirty.txt').unlink()
        self.target.mkdir()
        (self.target/'keep.txt').write_text('keep')
        result = run('bash', pod/'volume.sh', *self.arguments(), env=self.environment)
        self.assertEqual(2, result.returncode)
        self.assertEqual('keep', (self.target/'keep.txt').read_text())
        self.assertEqual(['keep.txt'], [p.name for p in self.target.iterdir()])

    def test_execute_runs_the_exact_lock_steps_under_the_watchdog(self):
        pod = self.install()
        calls = self.root/'calls.txt'
        fake = self.root/'fakepython'
        fake.write_text('#!/usr/bin/env bash\nif [[ $* == *fine_tuning.budget* ]]; then\n'
                        f'printf "%s\\n" "budget $*" >> {calls}; exit 99; fi\n'
                        f'exec {sys.executable} "$@"\n')
        fake.chmod(0o755)
        result = run('bash', pod/'volume.sh', *self.arguments(), env={
            **self.environment, 'MMDL_POD_PYTHON': str(fake), 'MMDL_BUDGET_WATCHDOG_PID': 'wrong'})
        self.assertEqual(99, result.returncode, result.stderr)
        self.assertIn(f'--hourly-usd 1 --max-seconds 5400 --result-file {self.target}/budget_result.json'
                      ' -- bash', calls.read_text())
        self.assertFalse(self.target.exists())
        calls.unlink()
        # The real fine_tuning.budget is the parent: volume.sh re-execs itself under it.
        watched = {**self.environment, 'FIXTURE_CALLS': str(calls)}
        result = run('bash', pod/'volume.sh', *self.arguments(), env=watched)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('"status": "EXITED"', result.stdout)
        self.assertEqual('EXITED', json.loads((self.target/'budget_result.json').read_text())['status'])
        lock = (self.repo/'env/requirements-experiments.lock').resolve()
        venv = self.target/f'venv-{common.sha256_file(lock)[:16]}'
        lines = calls.read_text().splitlines()
        self.assertEqual([f'base -m venv {venv}', f'venv -m pip install -r {lock}',
                          'venv -m pip check', 'venv -m pip freeze --all'], lines[:4])
        self.assertTrue(lines[4].startswith(f'venv - {self.target} {self.commit} experiments'))
        self.assertEqual('fixture==1\n', (self.target/'pip-freeze.txt').read_text())
        self.assertIn('VOLUME_READY; no training', result.stdout)
        again = run('bash', pod/'volume.sh', *self.arguments(), env=watched)
        self.assertEqual(2, again.returncode)  # existing target refused, nothing deleted
        self.assertTrue((self.target/'pip-freeze.txt').exists())

    def test_volume_spoofed_watchdog_refused(self):
        pod = self.install()
        result = run('bash', pod/'volume.sh', *self.arguments(), env={
            **self.environment, 'MMDL_BUDGET_WATCHDOG_PID': str(os.getpid())})
        self.assertEqual(2, result.returncode, result.stderr)
        self.assertIn('watchdog argv mismatch', result.stderr)
        self.assertFalse(self.target.exists())

    def test_volume_manifest_schema_written_by_the_real_writer(self):
        manifest = self.manifest  # written by the real writer in setUp
        for key in ('commit', 'lock_sha256', 'python_version', 'pip_freeze_sha256', 'model_files',
                    'venv', 'created_utc', 'status'):
            self.assertIn(key, manifest)
        self.assertEqual('VOLUME_READY', manifest['status'])
        self.assertEqual(f"{sys.version_info.major}.{sys.version_info.minor}",
                         '.'.join(manifest['python_version'].split('.')[:2]))
        self.assertTrue(all({'path', 'bytes', 'sha256', 'quick_sha256'} <= set(f)
                            for f in manifest['model_files']))
        with self.assertRaises(ValueError):
            verify.write_volume_manifest(self.root/'x', commit=COMMIT_A, environment='experiments',
                lock=self.repo/'env/requirements-experiments.lock', venv=self.root/'x/wrong-name',
                freeze=self.repo/'env/requirements-experiments.lock', snapshot=self.root/'x/s',
                model_files=[])


if __name__ == '__main__':
    unittest.main()
