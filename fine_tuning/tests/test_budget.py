"""Real owned CPU process groups; no cloud, GPU, weights or dependency installation."""
import ctypes
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fine_tuning.budget import check_billing_start, effective_deadline, terminate_group, utc


WORKER = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print(os.getpid(),flush=True); time.sleep(60)"


def leader(normal=False):
    return ("import subprocess,sys,time; "
            f"worker=subprocess.Popen([sys.executable,'-c',{WORKER!r}],stdout=subprocess.PIPE,text=True); "
            "print(worker.stdout.readline().strip(),flush=True); "
            + ("" if normal else "time.sleep(60)"))


class BudgetTests(unittest.TestCase):
    def setUp(self):
        # Adopt and reap only our killed orphan worker; leave no test zombies.
        self.libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        self.assertEqual(0, self.libc.prctl(37, ctypes.byref(previous), 0, 0, 0))
        self.assertEqual(0, self.libc.prctl(36, 1, 0, 0, 0))
        self.addCleanup(self.libc.prctl, 36, previous.value, 0, 0, 0)

    def reap_worker(self, pid):
        until = time.monotonic()+3
        while time.monotonic() < until:
            result, status = os.waitpid(pid, os.WNOHANG)
            if result:
                self.assertEqual(signal.SIGKILL, os.WTERMSIG(status))
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
                return
            time.sleep(.01)
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        self.fail("Owned worker survived process-group cleanup")

    def command(self, command, maximum="3600", pod="fixture-owned-pod", seconds=600):
        now = dt.datetime.now(dt.timezone.utc)
        return [sys.executable, '-S', '-m', 'fine_tuning.budget',
                '--owned-pod-id', pod, '--gpu-cutoff-utc', (now+dt.timedelta(seconds=seconds)).isoformat(),
                '--billing-start-utc', now.isoformat(), '--balance-usd', '1', '--reserve-usd', '0',
                '--hourly-usd', '1', '--max-seconds', maximum, '--', *command]

    def test_term_kills_worker_after_leader_exits_and_reaps_leader(self):
        process = subprocess.Popen([sys.executable, '-c', leader()], start_new_session=True,
                                   stdout=subprocess.PIPE, text=True)
        try:
            worker = int(process.stdout.readline())
            terminate_group(process)
            self.assertEqual(-signal.SIGTERM, process.returncode)
            self.reap_worker(worker)
        finally:
            terminate_group(process)
            process.stdout.close()

    def test_watchdog_stdlib_identity_reserve_timeout_and_child_marker(self):
        environment = dict(os.environ, RUNPOD_POD_ID='fixture-owned-pod')
        marker = ("import os; assert os.environ['MMDL_BUDGET_WATCHDOG_PID']==str(os.getppid()); "
                  "raise SystemExit(7)")
        for maximum, pod, seconds, child, expected, status in (
            ('3600', 'fixture-owned-pod', 600, marker, 7, 'EXITED'),
            ('.05', 'fixture-owned-pod', 600, 'import time; time.sleep(60)', 124, 'BUDGET_STOP'),
            ('3600', 'wrong-pod', 600, marker, 2, 'BLOCKED'),
            ('nan', 'fixture-owned-pod', 600, marker, 2, 'BLOCKED'),
            ('3600', 'fixture-owned-pod', 119, marker, 124, 'BUDGET_STOP'),
        ):
            with self.subTest(maximum=maximum, pod=pod, seconds=seconds):
                result = subprocess.run(self.command([sys.executable, '-c', child], maximum, pod, seconds),
                    env=environment, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=20)
                self.assertEqual(expected, result.returncode, result.stderr)
                receipt = json.loads(result.stdout)
                self.assertEqual(status, receipt['status'])
                if status == 'EXITED':
                    self.assertLess(receipt['timeout_seconds'], 480)

    def test_watchdog_signal_and_timeout_kill_residual_worker(self):
        environment = dict(os.environ, RUNPOD_POD_ID='fixture-owned-pod')
        for interrupted in (signal.SIGTERM, signal.SIGINT, None):
            with self.subTest(interrupted=interrupted):
                # Cold shared-volume Python startup can exceed the old 0.2s fixture budget.
                # Keep the real timeout test, including deliberately delayed readiness.
                child = 'import time; time.sleep(.35); ' + leader()
                watchdog = subprocess.Popen(self.command([sys.executable, '-S', '-c', child], '10' if interrupted else '5'),
                    env=environment, cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True)
                worker = None
                try:
                    worker = int(watchdog.stdout.readline())
                    if interrupted:
                        watchdog.send_signal(interrupted)
                    stdout, stderr = watchdog.communicate(timeout=20)
                    self.assertEqual(128+interrupted if interrupted else 124, watchdog.returncode, stderr)
                    self.assertEqual('INTERRUPTED' if interrupted else 'BUDGET_STOP', json.loads(stdout)['status'])
                finally:
                    try:
                        if watchdog.poll() is None:
                            watchdog.send_signal(signal.SIGTERM)
                            watchdog.wait(timeout=20)
                    finally:
                        watchdog.stdout.close()
                        watchdog.stderr.close()
                        if worker is not None:
                            self.reap_worker(worker)

    def test_controller_cleans_success_timeout_journal_error_and_sigterm(self):
        from fine_tuning.run_experiment import Controller
        previous = signal.getsignal(signal.SIGTERM)
        for mode in ('success', 'timeout', 'journal', 'sigterm'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root/'source'
                source.mkdir()
                controller = Controller(root/'output', dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=10), source)
                owned = {}
                def event(**value):
                    if value['status'] == 'RUNNING':
                        owned['group'] = value['pid']
                        until = time.monotonic()+3
                        while time.monotonic() < until:
                            text = (controller.output/'cpu.log').read_text().strip()
                            if text:
                                owned['worker'] = int(text)
                                break
                            time.sleep(.01)
                        self.assertIn('worker', owned)
                        if mode == 'journal':
                            raise OSError('RUNNING journal failed')
                        if mode == 'sigterm':
                            os.kill(os.getpid(), signal.SIGTERM)
                try:
                    with patch.object(controller, 'event', side_effect=event):
                        if mode == 'success':
                            controller.stage('cpu', [sys.executable, '-c', leader(True)])
                        else:
                            with self.assertRaises({'timeout': TimeoutError, 'journal': OSError, 'sigterm': SystemExit}[mode]):
                                controller.stage('cpu', [sys.executable, '-c', leader()], .2 if mode == 'timeout' else 10)
                    self.assertIs(previous, signal.getsignal(signal.SIGTERM))
                    self.reap_worker(owned['worker'])
                finally:
                    if 'group' in owned:
                        try:
                            os.killpg(owned['group'], signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    def test_nested_graces_let_inner_controller_kill_its_ignoring_term_worker(self):
        from fine_tuning.run_experiment import Controller
        previous = signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source'
            source.mkdir()
            outer = Controller(root/'outer', dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=10), source)
            inner = f"""
import datetime as dt
import sys
from pathlib import Path
from fine_tuning.run_experiment import Controller

controller = Controller(Path({str(root/'inner')!r}), dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=10), Path({str(source)!r}))
try:
    controller.stage('worker', [sys.executable, '-c', {WORKER!r}], termination_grace_seconds=.2)
finally:
    print('INNER_CLEANUP_FINISHED', flush=True)
"""
            owned = {}
            def event(**value):
                if value['status'] == 'RUNNING':
                    owned['inner'] = value['pid']
                    until = time.monotonic()+5
                    worker_log = root/'inner/worker.log'
                    while time.monotonic() < until:
                        text = worker_log.read_text().strip() if worker_log.exists() else ''
                        if text:
                            owned['worker'] = int(text)
                            break
                        time.sleep(.01)
                    self.assertIn('worker', owned)
                    os.kill(os.getpid(), signal.SIGTERM)
            try:
                with patch.object(outer, 'event', side_effect=event), self.assertRaises(SystemExit) as interrupted:
                    outer.stage('nested', [sys.executable, '-c', inner], termination_grace_seconds=.8)
                self.assertEqual(143, interrupted.exception.code)
                self.assertIs(previous, signal.getsignal(signal.SIGTERM))
                self.assertIn('INNER_CLEANUP_FINISHED', (root/'outer/nested.log').read_text())
                with self.assertRaises(ProcessLookupError):
                    os.kill(owned['worker'], 0)
            finally:
                for pid in owned.values():
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if 'worker' in owned:
                    try:
                        os.waitpid(owned['worker'], 0)
                    except ChildProcessError:
                        pass

    def budget(self, command, *extra, billing=None, maximum="30"):
        now = dt.datetime.now(dt.timezone.utc)
        return [sys.executable, '-S', '-m', 'fine_tuning.budget', '--owned-pod-id', 'fixture-owned-pod',
                '--gpu-cutoff-utc', (now+dt.timedelta(seconds=600)).isoformat(),
                '--billing-start-utc', (billing or now).isoformat(), '--balance-usd', '1',
                '--reserve-usd', '0', '--hourly-usd', '1', '--max-seconds', maximum, *extra,
                '--', *command]

    def test_future_billing_start_is_refused_beyond_a_small_skew(self):
        environment = {**os.environ, "RUNPOD_POD_ID": "fixture-owned-pod"}
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory)/'ran'
            touch = [sys.executable, '-c', f'import pathlib;pathlib.Path({str(marker)!r}).touch()']
            now = dt.datetime.now(dt.timezone.utc)
            refused = subprocess.run(self.budget(touch, billing=now+dt.timedelta(hours=1)),
                                     env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(2, refused.returncode, refused.stderr)
            result = json.loads(refused.stdout.splitlines()[-1])
            self.assertEqual('BLOCKED', result['status'])
            self.assertIn('future', result['reason'])
            self.assertFalse(marker.exists())
            skewed = subprocess.run(self.budget(touch, billing=now+dt.timedelta(seconds=60)),
                                    env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(0, skewed.returncode, skewed.stderr)
            self.assertTrue(marker.exists())
        check_billing_start(utc('2026-10-04T00:00:00Z'))  # past starts stay valid
        with self.assertRaises(ValueError):
            check_billing_start(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=301))

    def test_sighup_is_ignored_the_child_keeps_default_hup_and_the_result_is_persisted(self):
        environment = {**os.environ, "RUNPOD_POD_ID": "fixture-owned-pod"}
        child = ("import signal,sys,time;print(signal.getsignal(signal.SIGHUP) is signal.SIG_DFL,"
                 "flush=True);time.sleep(1.5)")
        with tempfile.TemporaryDirectory() as directory:
            result_file = Path(directory)/'budget_result.json'
            process = subprocess.Popen(self.budget([sys.executable, '-c', child], '--result-file',
                                                   str(result_file)),
                                       env=environment, stdout=subprocess.PIPE, text=True,
                                       start_new_session=True)
            try:
                self.assertEqual('True', process.stdout.readline().strip())
                os.kill(process.pid, signal.SIGHUP)  # what a dropped SSH session delivers
                output, _ = process.communicate(timeout=30)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
            self.assertEqual(0, process.returncode)
            printed = json.loads(output.splitlines()[-1])
            self.assertEqual(('EXITED', 0, 1), (printed['status'], printed['returncode'],
                                                printed['sighup_ignored']))
            self.assertEqual(printed, json.loads(result_file.read_text()))
            missing = subprocess.run(self.budget([sys.executable, '-c', 'pass'], '--result-file',
                                                 str(Path(directory)/'no-dir/result.json')),
                                     env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(0, missing.returncode)  # an unwritable result file is reported only
            self.assertIn('result_file_error', json.loads(missing.stdout.splitlines()[-1]))

    def test_budget_stop_is_persisted_before_the_group_is_stopped(self):
        environment = {**os.environ, "RUNPOD_POD_ID": "fixture-owned-pod"}
        with tempfile.TemporaryDirectory() as directory:
            result_file, seen = Path(directory)/'budget_result.json', Path(directory)/'seen.json'
            child = ['bash', '-c', f'trap "cp {result_file} {seen}; exit 143" TERM; sleep 30 & wait']
            done = subprocess.run(self.budget(child, '--result-file', str(result_file), maximum='1'),
                                  env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(124, done.returncode, done.stderr)
            self.assertEqual('BUDGET_STOP', json.loads(seen.read_text())['status'])  # child saw it
            final = json.loads(result_file.read_text())
            self.assertEqual('BUDGET_STOP', final['status'])
            self.assertIn('seconds', final)

    def test_deadline_helpers_retain_financial_and_timezone_behavior(self):
        start = utc('2026-10-04T00:00:00Z')
        cap = start+dt.timedelta(hours=2)
        self.assertEqual(start+dt.timedelta(hours=1), effective_deadline(cap, start, 1.5, .5, 1))
        with self.assertRaises(ValueError):
            utc('2026-10-04T00:00:00')
        for balance, reserve, hourly in ((float('nan'), 0, 1), (1, 1, 1), (1, 0, 0), (1, -1, 1)):
            with self.assertRaises(ValueError):
                effective_deadline(cap, start, balance, reserve, hourly)
        for grace in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                terminate_group(None, grace_seconds=grace)


if __name__ == '__main__':
    unittest.main()
