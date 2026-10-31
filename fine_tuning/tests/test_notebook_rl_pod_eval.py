"""CPU fixtures for the MMMU-val proof evaluation on Pods: the optional eval-data step of volume.sh (fake
acquisition), the training preflight's eval-root isolation (and the real actor guards it models), the
eval-stage preflight, and eval_stage plans checked against the real argparse/validation of proof_eval,
scripts/eval.sh (mmdl.evaluation.cli) and pod/chain.py. No RunPod, network, GPU, weights or MMMU data."""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mmdl.evaluation import cli

from fine_tuning.notebook_rl import proof_eval
from fine_tuning.notebook_rl.pod import chain, common, eval_stage, job, verify
from fine_tuning.tests import test_notebook_rl_pod as pod_tests

REPO = pod_tests.REPO
DATA_REVISION = '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68'
REAL_PREREG = Path('/home/jang2/mmdl-artifacts/notebook-rl-local-nf4-20261021/hf-pool-v3/proof-prereg/prereg.json')
EVAL_PAYLOADS = {'README.md': b'readme', 'Art/validation-00000-of-00001.parquet': b'art-parquet',
                 'Math/validation-00000-of-00001.parquet': b'math-parquet'}


def records(payloads):
    return [{'path': n, 'bytes': len(d), 'sha256': hashlib.sha256(d).hexdigest()} for n, d in payloads.items()]


def add_eval_data(volume, payloads=EVAL_PAYLOADS):
    """The layout volume.sh's acquisition leaves: own HF cache, data root of symlinks, manifest.json."""
    base = volume/'eval-data'
    folder = base/'hf-cache/hub/datasets--MMMU--MMMU'
    for name, data in payloads.items():
        blob = folder/'blobs'/hashlib.sha256(data).hexdigest()
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
        snapshot = folder/'snapshots'/DATA_REVISION/name
        for link, target in ((snapshot, blob), (base/'mmmu'/name, snapshot)):
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(target, link.parent))
    value = {'repo_id': 'MMMU/MMMU', 'revision': DATA_REVISION, 'files': records(payloads)}
    (base/'mmmu/manifest.json').write_text(json.dumps(value))
    return value


def rewrite_manifest(volume, repo, manifest, eval_data):
    """write_volume_manifest with the arguments make_volume used, plus eval_data."""
    return verify.write_volume_manifest(
        volume, commit=pod_tests.COMMIT_A, environment='experiments',
        lock=repo/'env/requirements-experiments.lock', venv=manifest['venv'],
        freeze=volume/'pip-freeze.txt', snapshot=volume.resolve()/manifest['model_snapshot'],
        model_files=[{k: f[k] for k in ('path', 'bytes', 'sha256')} for f in manifest['model_files']],
        span=manifest['quick_span_bytes'], eval_data=eval_data)


def reseal(value):
    value['approval']['approved_sha256'] = job.content_sha256(value)
    return value


class JobEvalTests(unittest.TestCase):
    def test_eval_stages_carry_no_training_data(self):
        for stage in job.EVAL_STAGES:
            value = pod_tests.approved_job(None, stage=stage)
            del value['data']
            self.assertEqual(stage, job.validate_job(reseal(value))['stage'])
            with self.assertRaisesRegex(job.JobError, 'unknown'):  # training data on an eval Pod job
                job.validate_job(pod_tests.approved_job(None, stage=stage))
        training = pod_tests.approved_job(None, stage='main')
        del training['data']
        with self.assertRaisesRegex(job.JobError, 'missing'):
            job.validate_job(reseal(training))

    def test_prep_eval_data_flag_is_optional_boolean_and_bound_by_the_approval(self):
        self.assertTrue(job.validate_job(pod_tests.prep_job(eval_data=True))['eval_data'])
        self.assertFalse(job.validate_job(pod_tests.prep_job(eval_data=False))['eval_data'])
        for bad in ('yes', 1, None):
            with self.assertRaisesRegex(job.JobError, 'eval_data'):
                job.validate_job(pod_tests.prep_job(eval_data=bad))
        flipped = pod_tests.prep_job(eval_data=False)
        flipped['eval_data'] = True  # changed after the approval
        with self.assertRaisesRegex(job.JobError, 'approved_sha256'):
            job.validate_job(flipped)
        with self.assertRaisesRegex(job.JobError, 'unknown'):  # not a training/eval job field
            job.validate_job(pod_tests.approved_job(None, eval_data=True))
        with tempfile.TemporaryDirectory() as temporary:
            for value, expected in ((pod_tests.prep_job(eval_data=True), 'true\n'),
                                    (pod_tests.prep_job(), 'false\n')):
                path = Path(temporary)/'prep.json'
                path.write_text(json.dumps(value))
                with mock.patch('sys.stdout', new=io.StringIO()) as out:
                    self.assertEqual(0, job.main([str(path), '--get', 'eval_data']))
                self.assertEqual(expected, out.getvalue())
            path.write_text(json.dumps(pod_tests.approved_job(None)))
            with mock.patch('sys.stderr', new=io.StringIO()) as err:
                self.assertEqual(2, job.main([str(path), '--get', 'eval_data']))
            self.assertIn('no field eval_data', err.getvalue())


class VerifyEvalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo, git_env = pod_tests.make_git_repo(self.root)
        (self.root/'ws').mkdir()
        self.volume, self.manifest = pod_tests.make_volume(self.root, self.repo)
        self.commit = pod_tests.commit_repo(self.repo, git_env)
        self.data = pod_tests.make_data(self.root)
        self.home = self.root/'home'
        (self.home/'mmdl-data').mkdir(parents=True)
        patcher = mock.patch.dict(os.environ, {
            'PATH': f"{pod_tests.fake_commands(self.root)}{os.pathsep}{os.environ['PATH']}",
            'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_CONFIG_SYSTEM': '/dev/null', 'HOME': str(self.home)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def checks(self, stage, **changes):
        arguments = dict(repo_root=self.repo, commit=self.commit, volume_root=self.volume,
                         run_dir=self.root/'ws/runs/r1', mount_path=str(self.root/'ws'),
                         min_free_gib=0.001, data=self.data, stage=stage)
        arguments.update(changes)
        result = verify.run_checks(**arguments)
        self.failed = [c['name'] for c in result['checks'] if not c['ok']]
        self.details = {c['name']: c['detail'] for c in result['checks']}
        return result

    def test_training_preflight_requires_the_guard_root_to_be_the_eval_root(self):
        guard = self.home/'mmdl-data/evaluation'
        self.assertFalse(self.checks('main')['ok'])  # no link: the guards would not cover the volume
        self.assertEqual(['eval_isolation'], self.failed)
        self.assertIn('ln -s', self.details['eval_isolation'])
        guard.symlink_to(self.volume.resolve()/'eval-data')  # eval-data need not exist yet
        result = self.checks('pilot-300')
        self.assertTrue(result['ok'], result['checks'])
        self.assertEqual(['clone', 'volume_manifest', 'lock_sha256', 'venv', 'pinned_baseline', 'model_files',
                          'data_bundle', 'dataset_vault', 'eval_isolation', 'disk_free', 'workspace_mount'],
                         [c['name'] for c in result['checks']])
        other = self.root/'ws/other-volume/eval-data'
        other.mkdir(parents=True)
        self.assertFalse(self.checks('main')['ok'])  # MMMU outside the guarded root
        self.assertIn('other-volume', self.details['eval_isolation'])
        shutil.rmtree(other.parent)
        guard.unlink()
        guard.symlink_to(self.root/'elsewhere')
        self.assertEqual(['eval_isolation'], self.checks('bf16-smoke-throughput')['ok'] or self.failed)
        guard.unlink()
        guard.mkdir()  # a real directory on the container disk does not cover the volume
        self.assertEqual(['eval_isolation'], self.checks('main')['ok'] or self.failed)
        self.assertNotIn('eval_isolation', [c['name'] for c in self.checks(None)['checks']])

    def test_eval_stage_checks_the_recorded_eval_data_and_no_training_data(self):
        self.assertFalse(self.checks('eval-f0', data=None)['ok'])  # prep had eval_data: false
        self.assertEqual(['eval_data'], self.failed)
        self.assertIn('no eval data', self.details['eval_data'])
        record = add_eval_data(self.volume)
        manifest = rewrite_manifest(self.volume, self.repo, self.manifest, record)
        self.assertEqual({'repo_id': 'MMMU/MMMU', 'revision': DATA_REVISION, 'data_root': 'eval-data/mmmu',
                          'cache': 'eval-data/hf-cache', 'usage': 'evaluation_only', 'training_allowed': False,
                          'data_manifest_sha256': common.sha256_file(self.volume/'eval-data/mmmu/manifest.json'),
                          'files': records(EVAL_PAYLOADS)}, manifest['eval_data'])
        for stage in ('eval-f0', 'eval-ft', 'eval-course'):
            result = self.checks(stage, data=None)
            self.assertTrue(result['ok'], result['checks'])
            names = [c['name'] for c in result['checks']]
            self.assertIn('eval_data', names)
            self.assertFalse({'data_bundle', 'dataset_vault', 'eval_isolation'} & set(names))
        self.assertIn(DATA_REVISION, self.details['eval_data'])
        blob = self.volume/'eval-data/hf-cache/hub/datasets--MMMU--MMMU/blobs'/records(EVAL_PAYLOADS)[2]['sha256']
        blob.write_bytes(b'math-parquEt')
        self.assertEqual(['eval_data'], self.checks('eval-ft', data=None)['ok'] or self.failed)
        blob.write_bytes(EVAL_PAYLOADS['Math/validation-00000-of-00001.parquet'])
        (self.volume/'eval-data/mmmu/manifest.json').write_text('{}')
        self.assertEqual(['eval_data'], self.checks('eval-course', data=None)['ok'] or self.failed)

    def test_eval_files_must_resolve_inside_the_eval_root_and_match_the_acquisition(self):
        record = add_eval_data(self.volume)
        with self.assertRaisesRegex(ValueError, 'differs'):
            rewrite_manifest(self.volume, self.repo, self.manifest,
                             {**record, 'files': [{**record['files'][0], 'sha256': '0'*64}]})
        outside = self.root/'outside.parquet'
        outside.write_bytes(EVAL_PAYLOADS['Art/validation-00000-of-00001.parquet'])
        art = self.volume/'eval-data/mmmu/Art/validation-00000-of-00001.parquet'
        art.unlink()
        art.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'outside'):
            rewrite_manifest(self.volume, self.repo, self.manifest, record)


# The real training guards with CPU stand-ins only for the version/HF-import preparation and bitsandbytes.
GUARD_SCRIPT = '''import sys, types
from pathlib import Path
from fine_tuning.notebook_rl import local_run, passrate, quantized_policy, shared_smoke
quantized_policy._versions = lambda: None
shared_smoke._prepare_hf_dependencies = lambda: None
sys.modules.setdefault("bitsandbytes", types.ModuleType("bitsandbytes"))
mode, build, target = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
if mode == "local_run":
    request = {k: {"path": str(build/(k + ".json"))} for k in ("index", "data_report", "source_registry", "evidence")}
    local_run._install_actor_guards({"build_root": str(build)}, request)
else:
    passrate._install_guards(build, None, None)
try:
    open(target).close()
except PermissionError:
    print("BLOCKED")
else:
    print("OPENED")
'''


class RealGuardTests(unittest.TestCase):
    def test_actor_guards_forbid_the_eval_root_exactly_when_the_preflight_check_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            volume = root/'ws/vol'
            target = volume/'eval-data/mmmu/Art/validation-00000-of-00001.parquet'
            target.parent.mkdir(parents=True)
            target.write_bytes(b'art-parquet')
            (root/'build').mkdir()
            for linked in (True, False):
                home = root/f'home-{linked}'
                (home/'mmdl-data').mkdir(parents=True)
                if linked:
                    (home/'mmdl-data/evaluation').symlink_to(volume/'eval-data')
                env = {k: v for k, v in os.environ.items() if k not in ('HF_HUB_CACHE', 'HF_HOME')}
                env.update(HOME=str(home), HF_HOME=str(volume/'hf-cache'),
                           PYTHONPATH=f'{REPO}/src{os.pathsep}{REPO}')
                for mode in ('local_run', 'passrate'):
                    done = subprocess.run([sys.executable, '-B', '-c', GUARD_SCRIPT, mode, str(root/'build'),
                                           str(target)], capture_output=True, text=True, timeout=300,
                                          env=env, cwd=REPO)
                    self.assertEqual(0, done.returncode, done.stderr[-2000:])
                    self.assertEqual('BLOCKED' if linked else 'OPENED', done.stdout.strip(), (mode, linked))
                with mock.patch.dict(os.environ, {'HOME': str(home)}):
                    if linked:
                        self.assertIn('forbids', verify.check_eval_isolation(volume, root/'ws'))
                    else:
                        with self.assertRaises(ValueError):
                            verify.check_eval_isolation(volume, root/'ws')


FAKE_DOWNLOAD = '''import hashlib, os, pathlib
MODEL_FILES = ["config.json", "w.safetensors"]


def acquire(repo, revision, kind, filenames, cache, destination=None):
    with open(os.environ["FIXTURE_CALLS"], "a") as stream:
        stream.write(f"acquire {repo} {revision} {kind} {cache} {destination}\\n")
    source = pathlib.Path(os.environ["FIXTURE_HUB"], repo)
    folder = pathlib.Path(cache, "hub", kind + "s--" + repo.replace("/", "--"))
    snapshot = folder/"snapshots"/revision
    records = []
    for name in filenames:
        data = (source/name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        blob = folder/"blobs"/digest
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
        link = snapshot/name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
        records.append({"path": name, "bytes": len(data), "sha256": digest})
        if destination is not None:
            view = pathlib.Path(destination)/name
            view.parent.mkdir(parents=True, exist_ok=True)
            view.symlink_to(os.path.relpath(link, view.parent))
    return {"repo_id": repo, "revision": revision, "kind": kind, "files": records}, snapshot
'''
FAKE_CONTRACTS = f'''MODEL_ID, MODEL_REVISION = "m/x", "rev"
DATA_ID, DATA_REVISION = "MMMU/MMMU", "{DATA_REVISION}"
SUBJECTS = ["Art", "Math"]
'''
FAKE_ARTIFACTS = '''import json, pathlib


def write_json(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2) + "\\n")
'''
# `python -m venv X` (shadowed through volume.sh's PYTHONPATH) installs the shim as X/bin/python; the shim
# runs volume.sh's heredoc with the real interpreter and answers pip freeze.
FAKE_VENV = '''import os, pathlib, shutil, sys
target = pathlib.Path(sys.argv[1])
(target/"bin").mkdir(parents=True)
shutil.copy(os.environ["FIXTURE_SHIM"], target/"bin/python")
'''
SHIM = '''#!/usr/bin/env bash
if [[ $* == "-m pip freeze --all" ]]; then printf '%s\\n' 'fixture==1'; exit 0; fi
if [[ ${1:-} == - ]]; then exec "$FIXTURE_REAL_PYTHON" "$@"; fi
exit 0
'''


class VolumeEvalDataTests(pod_tests.ScriptFixture):
    def setUp(self):
        super().setUp()
        shutil.rmtree(self.volume)  # volume.sh must start from a nonexistent target
        self.target = self.workspace/'new-volume'
        for name, text in (('venv/__init__.py', ''), ('venv/__main__.py', FAKE_VENV), ('mmdl/__init__.py', ''),
                           ('mmdl/data/__init__.py', ''), ('mmdl/data/download.py', FAKE_DOWNLOAD),
                           ('mmdl/runtime/__init__.py', ''), ('mmdl/runtime/contracts.py', FAKE_CONTRACTS),
                           ('mmdl/runtime/artifacts.py', FAKE_ARTIFACTS)):
            (self.repo/'src'/name).parent.mkdir(parents=True, exist_ok=True)
            (self.repo/'src'/name).write_text(text)
        self.hub = self.root/'hub'
        for name, data in (('config.json', b'{"a":1}'), ('w.safetensors', bytes(range(256))*4)):
            (self.hub/'m/x'/name).parent.mkdir(parents=True, exist_ok=True)
            (self.hub/'m/x'/name).write_bytes(data)  # = make_volume's baseline.json files
        for name, data in EVAL_PAYLOADS.items():
            (self.hub/'MMMU/MMMU'/name).parent.mkdir(parents=True, exist_ok=True)
            (self.hub/'MMMU/MMMU'/name).write_bytes(data)
        (self.repo/'manifests/datasets').mkdir()
        (self.repo/'manifests/datasets/mmmu.json').write_text(json.dumps({
            'repo_id': 'MMMU/MMMU', 'revision': DATA_REVISION, 'files': records(EVAL_PAYLOADS),
            'source': f'https://huggingface.co/datasets/MMMU/MMMU/tree/{DATA_REVISION}'}))
        shim = self.root/'shim.sh'
        shim.write_text(SHIM)
        shim.chmod(0o755)
        self.calls = self.root/'calls.txt'
        self.pod = self.install()
        self.environment.update(FIXTURE_CALLS=str(self.calls), FIXTURE_HUB=str(self.hub),
                                FIXTURE_SHIM=str(shim), FIXTURE_REAL_PYTHON=sys.executable)

    def volume_sh(self, execute=True, **changes):
        path = self.root/'prep.json'
        path.write_text(json.dumps(pod_tests.prep_job(**{'commit': self.commit, 'volume_root': str(self.target),
                                                         **changes})))
        return pod_tests.run('bash', self.pod/'volume.sh', '--job', str(path), *(['--execute'] if execute else []),
                             env=self.environment, timeout=120)

    def test_eval_data_true_acquires_pinned_mmmu_under_its_own_root(self):
        dry = self.volume_sh(execute=False, eval_data=True)
        self.assertEqual(0, dry.returncode, dry.stderr)
        self.assertIn('eval data (MMMU validation): true', dry.stdout)
        self.assertFalse(self.target.exists())
        result = self.volume_sh(eval_data=True)
        self.assertEqual(0, result.returncode, result.stdout+result.stderr)
        self.assertIn('VOLUME_READY; no training', result.stdout)
        manifest = json.loads((self.target/'volume_manifest.json').read_text())
        eval_data = manifest['eval_data']
        self.assertEqual((DATA_REVISION, records(EVAL_PAYLOADS), 'eval-data/mmmu', False),
                         (eval_data['revision'], eval_data['files'], eval_data['data_root'],
                          eval_data['training_allowed']))
        data_root = self.target/'eval-data/mmmu'
        self.assertEqual(common.sha256_file(data_root/'manifest.json'), eval_data['data_manifest_sha256'])
        written = json.loads((data_root/'manifest.json').read_text())  # what the evaluator's loader reads
        self.assertEqual(('MMMU/MMMU', DATA_REVISION, records(EVAL_PAYLOADS), 'evaluation_only'),
                         (written['repo_id'], written['revision'], written['files'], written['usage']))
        cache = (self.target/'eval-data/hf-cache').resolve()
        for item in eval_data['files']:
            self.assertIn(cache, (data_root/item['path']).resolve().parents)
        self.assertEqual([], list((self.target/'hf-cache').rglob('datasets--*')))  # training HF_HOME has no MMMU
        self.assertIn(f'acquire MMMU/MMMU {DATA_REVISION} dataset {self.target}/eval-data/hf-cache '
                      f'{self.target}/eval-data/mmmu', self.calls.read_text())
        self.assertIn(DATA_REVISION, verify.check_eval_data(self.target, eval_data))

    def test_eval_data_defaults_to_false_and_acquires_nothing(self):
        result = self.volume_sh()
        self.assertEqual(0, result.returncode, result.stdout+result.stderr)
        self.assertIsNone(json.loads((self.target/'volume_manifest.json').read_text())['eval_data'])
        self.assertFalse((self.target/'eval-data').exists())
        self.assertNotIn('MMMU', self.calls.read_text())

    def test_pinned_mismatch_fails_closed_without_a_ready_manifest(self):
        (self.hub/'MMMU/MMMU/Math/validation-00000-of-00001.parquet').write_bytes(b'other bytes')
        result = self.volume_sh(eval_data=True)
        self.assertNotEqual(0, result.returncode)
        self.assertIn('Pinned MMMU validation hash mismatch', result.stdout+result.stderr)
        self.assertFalse((self.target/'volume_manifest.json').exists())


class Stop(Exception):
    pass


class EvalStagePlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def params(self, stage, volume='/w/vol', **changes):
        value = {'version': 'eval-stage-params-v1', 'commit': 'a'*40, 'repo_root': str(REPO),
                 'python': f'{volume}/venv-0123456789abcdef/bin/python', 'volume_root': volume,
                 'artifact_root': f'{volume}/eval-data/artifacts', 'prereg': f'{volume}/eval-data/proof/prereg.json',
                 'prereg_sha256': 'b'*64, 'ledger': f'{volume}/eval-data/proof/ledger.jsonl',
                 'run_prefix': 'nbrl-proof-v1'}
        if stage == 'eval-ft':
            value.update(checkpoint=f'{volume}/runs/r1/ckpt', checkpoint_metadata_sha256='c'*64,
                         binding=f'{volume}/runs/r1/launch.json', binding_sha256='d'*64,
                         training_commit='e'*40, ft_export=f'{volume}/ft-export')
        if stage == 'eval-course':
            value.update(ft_export=f'{volume}/ft-export', ft_model_ref_sha256='f'*64)
        value.update(changes)
        return value

    def test_plan_is_deterministic_written_once_and_prints_its_sha256(self):
        params = self.root/'params.json'
        params.write_text(json.dumps(self.params('eval-ft')))
        output = self.root/'steps.json'
        arguments = ['plan', '--stage', 'eval-ft', '--params', str(params), '--output', str(output)]
        with mock.patch('sys.stdout', new=io.StringIO()) as out:
            self.assertEqual(0, eval_stage.main(arguments))
        self.assertEqual(common.sha256_file(output)+'\n', out.getvalue())
        first = output.read_bytes()
        with mock.patch('sys.stderr', new=io.StringIO()) as err:
            self.assertEqual(2, eval_stage.main(arguments))  # an existing spec is never overwritten
        self.assertIn('REFUSED', err.getvalue())
        self.assertEqual(first, output.read_bytes())
        again = json.dumps(eval_stage.plan('eval-ft', self.params('eval-ft')), indent=2, sort_keys=True)+'\n'
        self.assertEqual(first, again.encode())
        other = eval_stage.plan('eval-ft', self.params('eval-ft', run_prefix='nbrl-proof-v2'))
        self.assertNotEqual(json.loads(first)['meta']['params_sha256'], other['meta']['params_sha256'])

    def test_specs_pass_the_real_chain_check_and_keep_pin_export_ledger_run_order(self):
        for stage in job.EVAL_STAGES:
            spec = chain.check_spec(eval_stage.plan(stage, self.params(stage)))
            names = [s['name'] for s in spec['steps']]
            kinds = [n.split('-')[0] for n in names]
            pins = {'eval-f0': 1, 'eval-ft': 3, 'eval-course': 2}[stage]
            self.assertEqual(['pin']*pins, kinds[:pins], names)
            runs = 1 if stage == 'eval-course' else 3
            body = (['export'] if stage == 'eval-ft' else []) + ['ledger']*runs + ['eval']*runs
            self.assertEqual(body, kinds[pins:], names)
            for step in spec['steps']:
                if step['name'].startswith('eval-'):
                    rid = step['name'].removeprefix('eval-')
                    self.assertIn(f'ledger-{rid}', names[:names.index(step['name'])])
                    self.assertTrue(all(p.startswith('/w/vol/eval-data/artifacts/runs/'+rid) for p in step['expect_files']))
            self.assertEqual(stage, spec['meta']['stage'])
            self.assertEqual(stage, spec['meta']['job_stage'])  # chain.run binds the spec to the job's stage

    def test_chain_run_refuses_an_eval_spec_under_a_job_of_another_stage(self):
        # A training-stage job (preflight without eval_data, bound to the training-data pins) must never carry an
        # MMMU evaluation spec; the eval spec runs only under a job whose stage it was planned for.
        spec = self.root/'steps.json'
        spec.write_text(json.dumps(eval_stage.plan('eval-f0', self.params('eval-f0')), indent=2, sort_keys=True)+'\n')
        sha = common.sha256_file(spec)
        for job_stage in ('main', 'bf16-smoke-throughput', 'eval-ft'):
            run_dir = self.root/f'run-{job_stage}'
            run_dir.mkdir()
            with mock.patch.dict(os.environ, {'MMDL_RUN_DIR': str(run_dir), 'MMDL_JOB_STAGE': job_stage}):
                with self.assertRaisesRegex(chain.ChainError, 'planned for job stage eval-f0'):
                    chain.run(spec, sha)
            self.assertFalse((run_dir/chain.LOG_NAME).exists())  # refused before any step or log

    def planned_runs(self):
        found = set()
        for stage in job.EVAL_STAGES:
            for step in eval_stage.plan(stage, self.params(stage))['steps']:
                if step['name'].startswith('ledger-'):
                    argv = step['argv']
                    found.add((argv[argv.index('--arm')+1], argv[argv.index('--variant')+1]))
        return found

    def test_runs_and_arguments_match_the_preregistered_design(self):
        statistical = [v for v, row in proof_eval.VARIANTS.items() if row[3] == 'statistical']
        self.assertEqual({(a, v) for a in ('F0', 'FT') for v in statistical} | {('FT', proof_eval.COURSE)},
                         self.planned_runs())  # = proof_eval.build_preregistration's runs
        steps = {stage: eval_stage.plan(stage, self.params(stage))['steps'] for stage in job.EVAL_STAGES}
        snapshot = '/w/vol/hf-cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17'
        for stage, (model_ref, model_path) in {
                'eval-f0': (f'{REPO}/manifests/models/baseline.json', snapshot),
                'eval-ft': ('/w/vol/ft-export/model_ref.json', '/w/vol/ft-export/model'),
                'eval-course': ('/w/vol/ft-export/model_ref.json', '/w/vol/ft-export/model')}.items():
            for step in [s for s in steps[stage] if s['name'].startswith('eval-')]:
                argv = step['argv']
                value = {k: argv[argv.index(k)+1] for k in ('--protocol', '--hardware', '--model-ref', '--model-path',
                                                            '--data-root', '--run-id', '--job-id', '--run-role')}
                self.assertEqual((model_ref, model_path, '/w/vol/eval-data/mmmu', 'evaluation'),
                                 (value['--model-ref'], value['--model-path'], value['--data-root'], value['--run-role']))
                self.assertEqual(f'{REPO}/fine_tuning/notebook_rl/eval_configs/hardware/a40_48gb.yaml', value['--hardware'])
                self.assertEqual(value['--job-id']+'-evaluation', value['--run-id'])
                if stage == 'eval-course':
                    self.assertEqual(f'{REPO}/configs/eval/mmmu_val_v8.yaml', value['--protocol'])
                    self.assertEqual('nbrl-proof-v1-ft-course-s3407-evaluation', value['--run-id'])
                    self.assertEqual(['bash', f'{REPO}/scripts/eval.sh'], argv[2:4])
                else:
                    self.assertEqual(['-m', 'fine_tuning.notebook_rl.proof_eval', 'run-eval'], argv[1:4])
                    self.assertIn('c16_seed', value['--protocol'])
        export = next(s for s in steps['eval-ft'] if s['name'] == 'export-ft')['argv']
        self.assertEqual(snapshot, export[export.index('--base')+1])
        self.assertEqual('e'*40, export[export.index('--code-commit')+1])

    @unittest.skipUnless(REAL_PREREG.is_file(), 'local preregistration not present')
    def test_planned_runs_equal_the_real_preregistration(self):
        prereg = json.loads(REAL_PREREG.read_text())
        self.assertEqual({(r['arm'], r['variant']) for r in prereg['runs']}, self.planned_runs())

    def test_params_are_refused(self):
        for stage, changes in (
                ('eval-x', {}), ('eval-f0', {'checkpoint': '/w/vol/ckpt'}), ('eval-course', {'ft_export': None}),
                ('eval-f0', {'repo_root': 'relative/repo'}), ('eval-f0', {'ledger': '/w/vol/eval-data/../l.jsonl'}),
                ('eval-f0', {'prereg_sha256': 'B'*64}), ('eval-f0', {'commit': 'abc'}),
                ('eval-f0', {'python': '/usr/bin/python3'}), ('eval-f0', {'python': '/w/vol/venv-x/bin/python'}),
                ('eval-f0', {'python': '/w/vol/sub/venv-0123456789abcdef/bin/python'}),
                ('eval-f0', {'artifact_root': '/w/eval-artifacts'}),
                ('eval-f0', {'artifact_root': '/w/vol/eval-data/mmmu/out'}),
                ('eval-f0', {'prereg': '/w/vol/eval-data/hf-cache/p.json'}),
                ('eval-f0', {'run_prefix': '-bad'}), ('eval-f0', {'run_prefix': 'x'*42}),
                ('eval-f0', {'version': 'v0'})):
            params = self.params(stage if stage != 'eval-x' else 'eval-f0', **changes)
            if changes.get('ft_export', 1) is None:
                del params['ft_export']
            with self.assertRaises(ValueError, msg=(stage, changes)):
                eval_stage.plan(stage, params)

    def test_planned_argv_is_accepted_by_the_real_entry_points(self):
        volume = str(self.root/'vol')
        prereg = Path(volume)/'eval-data/proof/prereg.json'
        prereg.parent.mkdir(parents=True)
        prereg.write_text('{}')
        specs = {stage: eval_stage.plan(stage, self.params(stage, volume=volume,
                                                           prereg_sha256=common.sha256_file(prereg)))
                 for stage in job.EVAL_STAGES}
        pin = specs['eval-f0']['steps'][0]['argv']
        with mock.patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(0, chain.main(pin[3:]))
            prereg.write_text('{"changed": true}')
            self.assertEqual(1, chain.main(pin[3:]))
        seen = []
        real_validate = cli.validate_execution

        def stop(args, hw):  # the evaluator's own checks ran (argparse, run id, protocol+hardware, role rules)
            real_validate(args, hw)
            seen.append((vars(args), hw))
            raise Stop

        for stage, spec in specs.items():
            for step in spec['steps']:
                argv, name = step['argv'], step['name']
                if name.startswith('ledger-'):
                    with mock.patch.object(proof_eval, 'ledger_start', return_value={
                            'arm': 'A', 'variant': 'V', 'run_id': 'R'}) as called, \
                            mock.patch('sys.stdout', new=io.StringIO()):
                        self.assertEqual(0, proof_eval.main(argv[3:]))
                    prereg_path, arm, variant, rid, ledger = called.call_args.args
                    self.assertEqual((Path(volume)/'eval-data/proof/prereg.json', Path(volume)/'eval-data/proof/ledger.jsonl'),
                                     (prereg_path, ledger))
                    self.assertEqual(name.removeprefix('ledger-'), rid)
                elif name == 'export-ft':
                    with mock.patch.object(proof_eval, 'export_for_eval', return_value={
                            'export': {'unmerged_inference_bit_equivalent': True}}) as called, \
                            mock.patch('sys.stdout', new=io.StringIO()):
                        self.assertEqual(0, proof_eval.main(argv[3:]))
                    self.assertEqual((Path(f'{volume}/runs/r1/ckpt'), Path(f'{volume}/runs/r1/launch.json'),
                                      Path(f'{volume}/ft-export'), 'e'*40),
                                     tuple(called.call_args.args[i] for i in (0, 1, 3, 4)))
                elif name.startswith('eval-'):
                    seen.clear()
                    with mock.patch.object(cli, 'validate_execution', stop), self.assertRaises(Stop):
                        if argv[2:4] == ['bash', f'{REPO}/scripts/eval.sh']:
                            self.assertIn('-m mmdl.evaluation.cli "$@"', (REPO/'scripts/eval.sh').read_text())
                            with mock.patch.object(sys, 'argv', ['mmdl.evaluation.cli', *argv[4:]]):
                                cli.main()
                        else:
                            proof_eval.main(argv[3:])
                    args, hw = seen[0]
                    self.assertEqual(('full', 'evaluation', name.removeprefix('eval-'), True, 'a40_48gb'),
                                     (args['mode'], args['run_role'], args['run_id'], args['no_download'], hw['name']))


if __name__ == '__main__':
    unittest.main()
