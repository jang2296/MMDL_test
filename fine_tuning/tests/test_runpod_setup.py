"""CPU shell fixtures only; fake installation, doctor and model acquisition."""
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RunpodSetupTests(unittest.TestCase):
    def fixture(self, root):
        workspace, repo, commands = root/'workspace', root/'repo', root/'commands'
        workspace.mkdir()
        (repo/'fine_tuning').mkdir(parents=True)
        commands.mkdir()
        script = repo/'fine_tuning/runpod_setup.sh'
        source = Path(__file__).resolve().parents[1]/'runpod_setup.sh'
        # Use an isolated stand-in for the Pod mount, never the real /workspace.
        script.write_text(source.read_text().replace('/workspace', str(workspace)))
        git = commands/'git'
        git.write_text("#!/usr/bin/env bash\ncase \"$3\" in\n"
                       "rev-parse) printf '%s\\n' '"+'a'*40+"';;\n"
                       "status) exit 0;;\n"
                       "remote) printf '%s\\n' 'https://github.com/jang2296/MMDL_test.git';;\n"
                       "*) exit 2;;\nesac\n")
        git.chmod(0o755)
        mount = commands/'mountpoint'
        mount.write_text('#!/usr/bin/env bash\nexit 0\n')
        mount.chmod(0o755)
        base_python = commands/'python3.12'
        base_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$SETUP_CALLS"\nexit 99\n')
        base_python.chmod(0o755)
        venv = workspace/'old-runtime/venv'
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(venv)], check=True, timeout=15)
        executable = venv/'bin/python'
        (venv/'bin/python-real').symlink_to(executable.resolve())
        executable.unlink()
        executable.write_text('#!/usr/bin/env bash\n'
            'if [[ $1 == -I ]]; then exec "$(dirname "$0")/python-real" "$@"; fi\n'
            'printf "%s\\n" "$*" >> "$SETUP_CALLS"\n'
            'if [[ $* == "-m pip freeze --all" ]]; then printf "%s\\n" "fixture==1"; fi\n')
        executable.chmod(0o755)
        (venv.parent/'setup').mkdir()
        (venv.parent/'setup/pip-freeze.txt').write_text('original setup receipt\n')
        environment = dict(os.environ, PATH=str(commands)+os.pathsep+os.environ['PATH'],
            RUNPOD_POD_ID='fixture-owned', MMDL_BUDGET_WATCHDOG_PID=str(os.getpid()),
            SETUP_CALLS=str(root/'calls.txt'))
        args = ['--commit', 'a'*40, '--runtime-root', str(workspace/'new-runtime'),
                '--owned-pod-id', 'fixture-owned', '--gpu-cutoff-utc', '2026-10-05T00:00:00Z',
                '--billing-start-utc', '2026-10-04T00:00:00Z', '--balance-usd', '10',
                '--reserve-usd', '1', '--hourly-usd', '.54', '--execute']
        return script, workspace, venv, environment, args

    def test_reuse_checks_real_prefix_then_keeps_all_normal_setup_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script, workspace, venv, environment, args = self.fixture(root)
            old = {str(path): path.read_bytes() for path in
                   (venv/'pyvenv.cfg', venv.parent/'setup/pip-freeze.txt')}
            result = subprocess.run(['bash', str(script), *args, '--reuse-venv', str(venv)],
                env=environment, capture_output=True, text=True, timeout=15)
            self.assertEqual(0, result.returncode, result.stderr)
            calls = (root/'calls.txt').read_text().splitlines()
            self.assertEqual(calls[:3], [f'-m pip install -r {root}/repo/env/requirements-experiments.lock',
                                        '-m pip check', '-m pip freeze --all'])
            self.assertEqual(calls[3], '-m unittest discover -s fine_tuning/tests -v')
            self.assertIn(f'--output {workspace}/new-runtime/setup/doctor.json', calls[4])
            self.assertEqual(calls[5], f'- {workspace}/new-runtime '+ 'a'*40)
            self.assertFalse((workspace/'new-runtime/venv').exists())
            self.assertEqual('fixture==1\n', (workspace/'new-runtime/setup/pip-freeze.txt').read_text())
            self.assertEqual(old, {name: Path(name).read_bytes() for name in old})

    def test_reuse_and_runtime_guards_fail_before_setup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script, workspace, venv, environment, args = self.fixture(root)
            link = workspace/'linked-venv'
            link.symlink_to(venv, target_is_directory=True)
            for supplied in (workspace, root, link, venv/'..', venv.parent/'missing', venv.parent/'./venv'):
                with self.subTest(supplied=supplied):
                    # pathlib normalizes '.', so explicitly retain that input spelling.
                    value = str(venv.parent)+'/./venv' if supplied == venv else str(supplied)
                    result = subprocess.run(['bash', str(script), *args, '--reuse-venv', value],
                        env=environment, capture_output=True, text=True, timeout=15)
                    self.assertEqual(2, result.returncode, result.stderr)
            configuration = venv/'pyvenv.cfg'
            original = configuration.read_text()
            configuration.write_text(original.replace('version = 3.12', 'version = 3.11'))
            result = subprocess.run(['bash', str(script), *args, '--reuse-venv', str(venv)],
                env=environment, capture_output=True, text=True, timeout=15)
            self.assertEqual(2, result.returncode)
            configuration.write_text(original)
            alternate = venv/'configuration.txt'
            configuration.rename(alternate)
            configuration.symlink_to(alternate)
            result = subprocess.run(['bash', str(script), *args, '--reuse-venv', str(venv)],
                env=environment, capture_output=True, text=True, timeout=15)
            self.assertEqual(2, result.returncode)
            configuration.unlink()
            alternate.rename(configuration)
            executable = venv/'bin/python'
            original_python = executable.read_text()
            executable.write_text('#!/usr/bin/env bash\nexec '+shlex.quote(sys.executable)+' "$@"\n')
            result = subprocess.run(['bash', str(script), *args, '--reuse-venv', str(venv)],
                env=environment, capture_output=True, text=True, timeout=15)
            self.assertEqual(2, result.returncode)
            executable.write_text(original_python)
            for changed in (dict(environment, RUNPOD_POD_ID='unowned'), environment):
                if changed is environment:
                    (workspace/'new-runtime').mkdir()
                result = subprocess.run(['bash', str(script), *args, '--reuse-venv', str(venv)],
                    env=changed, capture_output=True, text=True, timeout=15)
                self.assertEqual(2, result.returncode)
            self.assertFalse((root/'calls.txt').exists())
            self.assertFalse((workspace/'new-runtime/setup').exists())

    def test_vllm_lock_and_new_tests_are_selected_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script, workspace, venv, environment, args = self.fixture(root)
            result = subprocess.run(['bash', str(script), *args, '--reuse-venv', str(venv),
                                     '--environment', 'vllm'], env=environment,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(0, result.returncode, result.stderr)
            calls = (root/'calls.txt').read_text().splitlines()
            self.assertEqual(calls[0], f'-m pip install -r {root}/repo/env/requirements-vllm.lock')
            self.assertIn('-m unittest discover -s experiments/input_recovery/tests -v', calls)
            self.assertFalse((workspace/'new-runtime/venv').exists())

    def test_default_dry_run_and_watchdog_still_precede_installation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script, workspace, venv, environment, args = self.fixture(root)
            result = subprocess.run(['bash', str(script), '--commit', 'a'*40],
                env=environment, capture_output=True, text=True, timeout=15)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn('DRY_RUN:', result.stdout)
            self.assertFalse((root/'calls.txt').exists())
            result = subprocess.run(['bash', str(script), *args, '--reuse-venv', str(venv)],
                env=dict(environment, MMDL_BUDGET_WATCHDOG_PID='wrong'),
                capture_output=True, text=True, timeout=15)
            self.assertEqual(99, result.returncode, result.stderr)
            invocation = (root/'calls.txt').read_text()
            self.assertIn('-m fine_tuning.budget --owned-pod-id fixture-owned', invocation)
            self.assertIn('--max-seconds 3600 -- bash', invocation)
            self.assertIn('--reuse-venv '+str(venv), invocation)
            self.assertFalse((workspace/'new-runtime').exists())


if __name__ == '__main__':
    unittest.main()
