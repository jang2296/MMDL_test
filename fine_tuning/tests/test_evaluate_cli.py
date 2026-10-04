"""Evaluator CLI help must work before any model/runtime dependency imports."""
import subprocess
import sys
import unittest
from pathlib import Path


class EvaluationCliTests(unittest.TestCase):
    def test_help_exits_without_model_or_runtime_dependencies(self):
        bootstrap = """
import runpy
import sys

class UnavailableDependencies:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'PIL', 'mmdl'} or fullname in {
            'fine_tuning.audit', 'fine_tuning.modules', 'fine_tuning.train'
        }:
            raise AssertionError('Help imported runtime dependency: ' + fullname)

sys.meta_path.insert(0, UnavailableDependencies())
sys.argv = ['fine_tuning.evaluate', *sys.argv[1:]]
runpy.run_module('fine_tuning.evaluate', run_name='__main__')
"""
        cases = {
            (): ('prepare-mmmu', 'prepare-external', 'run-external'),
            ('prepare-mmmu',): ('--data-root', '--per-subject'),
            ('prepare-external',): ('--nvw-root', '--trace-root', '--trace-count'),
            ('run',): ('--model-path', '--checkpoint', '--resume-from'),
            ('run-external',): ('--model-path', '--protocol', '--budget-seconds'),
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, '-S', '-c', bootstrap, *command, '--help'],
                    cwd=Path(__file__).resolve().parents[2],
                    capture_output=True, text=True, timeout=5,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertEqual('', result.stderr)
                for flag in expected:
                    self.assertIn(flag, result.stdout)


if __name__ == '__main__':
    unittest.main()
