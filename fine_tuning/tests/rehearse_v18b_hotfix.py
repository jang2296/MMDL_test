"""Run local C1 through two fixture commits in an isolated clone of a clean candidate.

python -m fine_tuning.tests.rehearse_v18b_hotfix --repo /clean/candidate \
    --data-root /private/v18b-data --output /private/new-hotfix-rehearsal
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from fine_tuning.error_note.prep_v16b import read_sealed, seal
from fine_tuning.error_note.v18b_rehearsal import run_child


MODULE = "fine_tuning.error_note.v18b_rehearsal"
COMMENT_FILE = "fine_tuning/error_note/single_vllm.py"
COMMENT = "\n# Rehearsal fixture for comment-only operational hotfix lineage.\n"


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def isolated_clone(repo: Path, output: Path) -> tuple[Path, str]:
    head = git(repo, "rev-parse", "HEAD")
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("C1 requires a clean committed candidate")
    git(repo, "cat-file", "-e", head + ":fine_tuning/error_note/v18b_rehearsal.py")
    clone = output / "checkout"
    subprocess.run(["git", "clone", "--local", "--shared", "--no-checkout", "--no-tags", "--quiet",
                    str(repo), str(clone)], check=True)
    git(clone, "checkout", "--quiet", "-b", "rehearsal/v18b-hotfix", head)
    return clone, head


def comment_commit(clone: Path) -> str:
    path = clone / COMMENT_FILE
    with path.open("a", encoding="utf-8") as stream:
        stream.write(COMMENT)
    git(clone, "add", "--", COMMENT_FILE)
    git(clone, "-c", "user.name=Codex rehearsal fixture", "-c", "user.email=rehearsal@example.invalid",
        "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "commit", "--quiet",
        "--no-gpg-sign", "-m", "test: comment-only v18b hotfix rehearsal fixture")
    return git(clone, "rev-parse", "HEAD")


def run(args: argparse.Namespace) -> dict:
    start = time.monotonic()
    if not 0 < args.deadline_seconds <= 1800:
        raise ValueError("the local C1 deadline must be within 30 minutes")
    repo, output, data_root = args.repo.resolve(), args.output.resolve(), args.data_root.resolve()
    supplied_model = args.model_path.resolve() if args.model_path else None
    if output == repo or output.is_relative_to(repo):
        raise ValueError("C1 output must be outside the clean source checkout")
    output.mkdir(parents=True, exist_ok=False)
    clone, old = isolated_clone(repo, output)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(clone), str(clone / "src"))),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               VLLM_ENABLE_V1_MULTIPROCESSING="0")
    deadline = start + args.deadline_seconds
    original_cwd = Path.cwd()
    campaign = output / "campaign"
    stages = []
    def execute(label: str, command: list[str], allowed: tuple[int, ...] = (0,)) -> None:
        before = time.monotonic()
        with (output / (label + ".log")).open("x") as stream:
            result = run_child(command, env=env, stream=stream, deadline=deadline)
        stages.append({"stage": label, "returncode": result.returncode, "seconds": time.monotonic() - before})
        seal(output / (label + ".json"), {"reportable": False, "old_commit": old, "stages": stages})
        if result.returncode not in allowed:
            raise subprocess.CalledProcessError(result.returncode, command)
    try:
        os.chdir(clone)
        model = supplied_model or output / "model" / "ebb281ec70b05090aa6165b016eac8ec08e71b17"
        if args.model_path is None:
            execute("tiny-model", [sys.executable, "-c",
                "from pathlib import Path; import sys; "
                "from fine_tuning.tests.test_error_note_single_vllm_lora_gpu import build_tiny; "
                "build_tiny(Path(sys.argv[1]), printable_output=True)", str(model)])
        execute("fixture-byte-check", [sys.executable, "-c",
            "from pathlib import Path; import sys; "
            "from fine_tuning.tests.test_error_note_single_vllm_lora_gpu import printable_fixture_cpu_check; "
            "printable_fixture_cpu_check(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))",
            str(model), str(data_root), str(output / "fixture-byte-proof.json")])
        base = [sys.executable, "-m", MODULE, "prepare", "--profile", "tiny-bf16",
                "--data-root", str(data_root), "--model-path", str(model),
                "--output", str(campaign), "--deadline-seconds", str(args.deadline_seconds),
                "--repo", str(clone), "--code-commit", old]
        def phase(name: str) -> list[str]:
            command = list(base)
            command[3] = name
            return command
        execute("prepare", phase("prepare"))
        execute("mechanical-note", phase("mechanical-note"))
        advance = phase("advance") + ["--exercise-recovery"]
        execute("boundary-kill", advance, (-9,))
        killed = read_sealed(campaign / "rehearsal-killed-checkpoint.json")
        state = read_sealed(campaign / "campaign-state.json")
        if killed["checkpoint"] != state["checkpoint"] or state["cursor"].get("policy_version") != 2:
            raise ValueError("C1 did not stop at its requested committed training boundary")
        new = comment_commit(clone)
        authorization = output / "authorization.json"
        execute("authorize-hotfix", [sys.executable, "-c",
            "from pathlib import Path; import sys; "
            "from fine_tuning.error_note.v18b_hotfix import authorize; "
            "from fine_tuning.error_note.prep_v16b import seal; "
            "seal(Path(sys.argv[5]),authorize(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],sys.argv[4],"
            "'C1 comment-only recovery fixture'))", str(campaign), str(clone), old, new, str(authorization)])
        base[base.index("--code-commit") + 1] = new
        base += ["--hotfix-resume", str(authorization)]
        advance = phase("advance") + ["--exercise-recovery"]
        execute("hotfix-resume-final-interruption", advance, (1,))
        final_marker = campaign / "rehearsal-final-interruption.json"
        if not final_marker.exists():
            raise RuntimeError("C1 resume failed before the deliberate final-monitor interruption; see hotfix-resume-final-interruption.log")
        interrupted = read_sealed(final_marker)
        state = read_sealed(campaign / "campaign-state.json")
        if interrupted["checkpoint"] != state["checkpoint"] or state["status"] != "TRAINED":
            raise ValueError("C1 final-monitor interruption lost the trained checkpoint")
        execute("final-monitor-resume", advance)
        execute("fresh-reload", phase("reload"))
        state = read_sealed(campaign / "campaign-state.json")
        ids = state["cursor"]["consumed_row_ids"]
        if (state["status"] != "COMPLETE" or len(ids) != len(set(ids))
                or not set(killed["cursor"]["consumed_row_ids"]) <= set(ids)
                or state["cursor"]["producer_pid"] == killed["producer_pid"]):
            raise ValueError("C1 recovery duplicated or lost durable learning")
        evaluation = output / "evaluation"
        execute("export-external-evaluation", [sys.executable, "-c",
            "from pathlib import Path; import sys; "
            "from fine_tuning.error_note.v18b_rehearsal_eval import run_rehearsal_evaluation; "
            "run_rehearsal_evaluation(campaign_output=Path(sys.argv[1]),model_path=Path(sys.argv[2]),"
            "data_root=Path(sys.argv[3]),output=Path(sys.argv[4]),mode='tiny-bf16',deadline_monotonic=float(sys.argv[5]))",
            str(campaign), str(model), str(data_root), str(evaluation), str(deadline)])
        if git(repo, "rev-parse", "HEAD") != old or git(repo, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("the caller source checkout changed during C1")
        return seal(output / "receipt.json", {"status": "PASS", "reportable": False,
            "source_repo": str(repo), "source_unchanged": True, "old_commit": old, "new_commit": new,
            "fixture_checkout": str(clone), "authorization_sha256": read_sealed(authorization)["sha256"],
            "state_sha256": state["sha256"], "unique_consumed_rows": len(ids), "stages": stages,
            "evaluation": read_sealed(evaluation / "evaluation.json"), "seconds": time.monotonic() - start})
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-path", type=Path, help="existing printable-output tiny fixture; otherwise build it locally")
    parser.add_argument("--deadline-seconds", default=1800, type=int)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("status", "reportable", "old_commit", "new_commit", "seconds")}))
