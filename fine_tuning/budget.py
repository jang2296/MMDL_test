"""Standard-library owned-process watchdog, usable before dependency installation."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import signal
import subprocess
import tempfile
import time

BILLING_SKEW_SECONDS = 300  # clock skew tolerated for a billing start in the future


def utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("UTC deadlines require explicit timezone")
    return parsed.astimezone(dt.timezone.utc)


def effective_deadline(gpu_cutoff: dt.datetime, billing_start: dt.datetime,
                       balance_usd: float, reserve_usd: float, hourly_usd: float) -> dt.datetime:
    if not all(math.isfinite(v) for v in (balance_usd, reserve_usd, hourly_usd)):
        raise ValueError("Nonfinite financial budget")
    if hourly_usd <= 0 or reserve_usd < 0 or balance_usd <= reserve_usd:
        raise ValueError("Positive rate and balance above recovery reserve required")
    financial = billing_start + dt.timedelta(seconds=(balance_usd-reserve_usd)/hourly_usd*3600)
    return min(gpu_cutoff, financial)


def check_billing_start(billing_start: dt.datetime, now: dt.datetime | None = None,
                        skew_seconds: float = BILLING_SKEW_SECONDS) -> None:
    """A billing start in the future moves the financial deadline later than the money allows."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if billing_start > now + dt.timedelta(seconds=skew_seconds):
        raise ValueError("Billing start is in the future; give the Pod's actual first billing time")


def write_result(path: str, result: dict) -> None:
    """Atomic replace in an existing directory (never creates one)."""
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary = tempfile.mkstemp(prefix=".budget-result-", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def terminate_group(process: subprocess.Popen, grace_seconds: float = 15) -> None:
    """Terminate a start_new_session child and kill residual workers after leader exit."""
    if not math.isfinite(grace_seconds) or grace_seconds <= 0:
        raise ValueError("Positive finite termination grace required")
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=15)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("owned-pod-id", "gpu-cutoff-utc", "billing-start-utc"):
        parser.add_argument("--"+name, required=True)
    for name in ("balance-usd", "reserve-usd", "hourly-usd"):
        parser.add_argument("--"+name, type=float, required=True)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--result-file", help="also write the result JSON here: when a budget stop is "
                        "decided (before the group is stopped) and at exit; the directory must exist")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    process, previous = None, {}
    started = time.monotonic()
    result: dict[str, object] = {}

    def persist() -> None:
        if args.result_file:
            try:
                write_result(args.result_file, result)
            except OSError as exc:
                result["result_file_error"] = f"{type(exc).__name__}: {exc}"
    try:
        if not args.owned_pod_id or os.environ.get("RUNPOD_POD_ID") != args.owned_pod_id:
            raise ValueError("Exact owned RunPod identity required")
        if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
            raise ValueError("Positive finite maximum seconds required")
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            raise ValueError("A command after -- is required")
        check_billing_start(utc(args.billing_start_utc))
        deadline = effective_deadline(utc(args.gpu_cutoff_utc), utc(args.billing_start_utc),
                                      args.balance_usd, args.reserve_usd, args.hourly_usd)
        allowed = min(args.max_seconds, (deadline-dt.datetime.now(dt.timezone.utc)).total_seconds()-120)
        result.update(deadline_utc=deadline.isoformat(), timeout_seconds=max(0, allowed))
        if allowed <= 0:
            raise TimeoutError("Budget cannot retain the 120-second recovery reserve")
        def interrupted(signum, _frame):
            raise SystemExit(128+signum)

        def hangup(_signum, _frame):
            # A dropped SSH session must not kill the watchdog and leave its own-session child
            # running unbounded; a Python handler (not SIG_IGN) is reset to default in the child.
            result["sighup_ignored"] = int(result.get("sighup_ignored", 0)) + 1
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        previous[signal.SIGHUP] = signal.signal(signal.SIGHUP, hangup)
        environment = dict(os.environ, MMDL_BUDGET_WATCHDOG_PID=str(os.getpid()))
        process = subprocess.Popen(command, start_new_session=True, stdin=subprocess.DEVNULL,
                                   env=environment)
        result["pid"] = process.pid
        code = process.wait(timeout=allowed)
        result.update(status="EXITED", returncode=code)
        code = code if code >= 0 else 128-code
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        result.update(status="BUDGET_STOP", reason=str(exc))
        persist()  # before the group is stopped, so the child can record BUDGET_STOP, not a kill
        code = 124
    except (KeyboardInterrupt, SystemExit) as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else (exc.code if isinstance(exc.code, int) else 1)
        result.update(status="INTERRUPTED", signal=code-128)
    except Exception as exc:
        result.update(status="BLOCKED", reason=f"{type(exc).__name__}: {exc}")
        code = 2
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if process is not None:
                terminate_group(process, grace_seconds=45)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    result["seconds"] = time.monotonic()-started
    persist()
    try:
        print(json.dumps(result), flush=True)
    except OSError:  # stdout was a terminal that hung up; the result file holds the record
        pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
