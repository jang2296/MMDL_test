#!/usr/bin/env bash
# The only entry for a notebook-RL Pod run: pre-flight, bounded GPU smoke, then the approved main
# command, all under the financial watchdog. Never restarts and never changes hardware,
# precision or limits after a failure. Run only from a clean, pinned MMDL_test clone.
# Financial/time bounds, min_free_gib and the data pins come only from the approved job file
# (job.financial etc.); there are no CLI flags for them, so they cannot differ from what was approved.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$here/../../.." && pwd -P)
workspace=/workspace
smoke_timeout=300
preflight_timeout=600  # = job.PREFLIGHT_ALLOWANCE_SECONDS in the deadline fit
python=${MMDL_POD_PYTHON:-python3.12}
job_file=
execute=false
quick=
original_arguments=("$@")
usage='Usage: bash fine_tuning/notebook_rl/pod/start.sh --job /abs/JOB.json [--quick-verify] [--execute]'
while (($#)); do
  case "$1" in
    --job) job_file=${2:?Missing job file}; shift 2 ;;
    --quick-verify) quick=--quick; shift ;;
    --execute) execute=true; shift ;;
    --help) printf '%s\n' "$usage"; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n $job_file && -f $job_file ]] || { printf '%s\n' "$usage" >&2; exit 2; }
# Absolute: the watchdog re-exec runs from $repo_root, where a relative path names another file.
[[ $job_file == /* ]] || { printf '%s\n' 'Job file path must be absolute' >&2; exit 2; }
read -r job_sha _ < <(sha256sum -- "$job_file")
export PYTHONPATH="$repo_root/src:$repo_root"
pod() { local module=$1; shift; "$python" -B -m "fine_tuning.notebook_rl.pod.$module" "$@"; }
job() { pod job "$job_file" --workspace "$workspace" "$@"; }
# Validation refuses missing/false/stale approval, an approval whose approved_sha256 is not this
# job's content, bad commit/paths/stage/argv/data pins, and a job that cannot finish before the
# financial/GPU deadline in job.financial.
plan=$(job) || exit 2
mapfile -d '' -t financial < <(job --financial-argv)
((${#financial[@]} == 12)) || { printf '%s\n' 'Cannot derive the financial bounds from the job' >&2; exit 2; }
owned_pod_id=$(job --get owned_pod_id)
commit=$(job --get commit)
volume_root=$(job --get volume_root)
run_dir=$(job --get run_dir)
stage=$(job --get stage)
max_main=$(job --get max_main_seconds)
min_free_gib=$(job --get min_free_gib)
if [[ $execute != true ]]; then
  printf 'DRY_RUN: nothing executed. Plan:\n%s\n' "$plan"
  printf 'Would run: preflight (verify.py %s), smoke (hard %ss), then main (max %ss) under budget.py.\n' "${quick:---full}" "$smoke_timeout" "$max_main"
  exit 0
fi
[[ -n ${RUNPOD_POD_ID:-} ]] || { printf '%s\n' 'RunPod only'; exit 2; }
[[ $owned_pod_id == "$RUNPOD_POD_ID" ]] || { printf '%s\n' 'This Pod is not job.financial.owned_pod_id'; exit 2; }
[[ ! -e $run_dir && ! -L $run_dir ]] || { printf '%s\n' 'Existing run dir refused; a new run needs a new run_dir'; exit 2; }
budget_command=("$python" -m fine_tuning.budget "${financial[@]}"
  --max-seconds "$((max_main + smoke_timeout + preflight_timeout))"
  --result-file "$run_dir/budget_result.json" -- bash "$here/start.sh" "${original_arguments[@]}")
if [[ ${MMDL_BUDGET_WATCHDOG_PID:-} != "$PPID" ]]; then
  cd "$repo_root"
  exec "${budget_command[@]}"
fi
# The variable is only a claim: pass through solely when the parent runs exactly the argv we would exec.
mapfile -d '' -t parent_command < "/proc/$PPID/cmdline" || parent_command=()
same=true
((${#parent_command[@]} == ${#budget_command[@]})) || same=false
for i in "${!budget_command[@]}"; do [[ ${parent_command[i]:-} == "${budget_command[i]}" ]] || same=false; done
[[ $same == true ]] || { printf '%s\n' 'watchdog argv mismatch' >&2; exit 2; }
mountpoint -q "$workspace" || { printf '%s\n' 'Require persistent Pod volume'; exit 2; }
mkdir -p "$(dirname -- "$run_dir")"
mkdir -m 700 "$run_dir"
mkdir "$run_dir/logs"
cp -- "$job_file" "$run_dir/job.json"
# From here every read uses the frozen copy; an edit to the original during preflight (e.g. the
# next stage's job) must not change the argv or limits of this run.
read -r copy_sha _ < <(sha256sum -- "$run_dir/job.json")
[[ $copy_sha == "$job_sha" ]] || { printf '%s\n' 'Job file changed after validation; refused' >&2; exit 2; }
job_file=$run_dir/job.json
printf '%s\n' "$plan" > "$run_dir/plan.json"
state_file=$run_dir/run_state.json
pod runner owner "$run_dir/owner.json" --pid "$$"
state() { pod runner state "$state_file" "$@"; }
# The watchdog writes budget_result.json with status BUDGET_STOP before it stops this group, so a
# deadline stop is recorded apart from a manual SIGTERM/SIGINT (INTERRUPTED).
on_signal() {
  local stop=INTERRUPTED
  if grep -qF '"status": "BUDGET_STOP"' "$run_dir/budget_result.json" 2>/dev/null; then stop=BUDGET_STOP; fi
  state "$stop" || true
  exit 143
}
trap on_signal TERM INT
disk_guard=(--disk-path "$run_dir" --min-free-gib "$min_free_gib" --disk-log "$run_dir/logs/disk.jsonl" --state-file "$state_file")
disk_low() {
  [[ $(pod runner last "$state_file") == DISK_LOW ]] || return 1
  printf 'DISK_LOW: free space under %s GiB in %s; command group stopped, see logs/disk.jsonl\n' "$min_free_gib" "$run_dir" >&2
}
export HF_HOME="$volume_root/hf-cache" PYTHONUNBUFFERED=1 MMDL_RUN_DIR="$run_dir" \
  MMDL_VOLUME_ROOT="$volume_root" MMDL_JOB_STAGE="$stage"

code=0
pod runner run --log "$run_dir/logs/preflight.log" --timeout "$preflight_timeout" -- \
  "$python" -B -m fine_tuning.notebook_rl.pod.verify --commit "$commit" --volume-root "$volume_root" \
  --run-dir "$run_dir" --repo-root "$repo_root" --mount-path "$workspace" \
  --job "$job_file" $quick || code=$?
if ((code != 0)); then
  state PREFLIGHT_FAILED --exit-code "$code"
  printf 'PREFLIGHT_FAILED; see %s/preflight.json and logs/preflight.log\n' "$run_dir" >&2
  exit 3
fi
state PREFLIGHT_OK
# Preflight just proved the manifest; both commands must run on its exact venv, not any Pod interpreter.
if ! job --require-interpreter "$volume_root/volume_manifest.json"; then
  state PREFLIGHT_FAILED --exit-code 2
  printf 'PREFLIGHT_FAILED; smoke/main must use the volume venv interpreter\n' >&2
  exit 2
fi
export HF_HUB_OFFLINE=1

mapfile -d '' -t smoke < <(job --argv smoke)
mapfile -d '' -t main_command < <(job --argv main)
((${#smoke[@]} && ${#main_command[@]})) || { printf '%s\n' 'Empty argv'; exit 2; }

code=0
pod runner run --log "$run_dir/logs/smoke.log" --timeout "$smoke_timeout" "${disk_guard[@]}" -- "${smoke[@]}" || code=$?
if disk_low; then exit 5; fi
if ((code != 0)); then
  state SMOKE_FAILED --exit-code "$code"
  printf 'SMOKE_FAILED (exit %s); main not started\n' "$code" >&2
  exit 4
fi
state SMOKE_OK
state MAIN_RUNNING
code=0
pod runner run --log "$run_dir/logs/main.log" --timeout "$max_main" "${disk_guard[@]}" -- "${main_command[@]}" || code=$?
if disk_low; then exit 5; fi
state "MAIN_EXIT $code" --exit-code "$code"
exit "$code"
