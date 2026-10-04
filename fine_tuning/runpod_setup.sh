#!/usr/bin/env bash
# Run only from a clean, pinned MMDL_test clone; no uploaded code overlay.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
task_root=/workspace/nvw-runtime
code_commit=
execute=false
original_arguments=("$@")
owned_pod_id= gpu_cutoff_utc= billing_start_utc= balance_usd= reserve_usd= hourly_usd=
while (($#)); do
  case "$1" in
    --commit) code_commit=${2:?Missing commit}; shift 2 ;;
    --runtime-root) task_root=${2:?Missing runtime root}; shift 2 ;;
    --owned-pod-id) owned_pod_id=${2:?Missing Pod ID}; shift 2 ;;
    --gpu-cutoff-utc) gpu_cutoff_utc=${2:?Missing cutoff}; shift 2 ;;
    --billing-start-utc) billing_start_utc=${2:?Missing billing start}; shift 2 ;;
    --balance-usd) balance_usd=${2:?Missing balance}; shift 2 ;;
    --reserve-usd) reserve_usd=${2:?Missing reserve}; shift 2 ;;
    --hourly-usd) hourly_usd=${2:?Missing rate}; shift 2 ;;
    --execute) execute=true; shift ;;
    --help) printf '%s\n' 'Usage: bash fine_tuning/runpod_setup.sh --commit <40-char SHA> [--runtime-root /workspace/nvw-runtime] [--owned-pod-id ID --gpu-cutoff-utc UTC --billing-start-utc UTC --balance-usd USD --reserve-usd USD --hourly-usd USD --execute]'; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ $code_commit =~ ^[0-9a-f]{40}$ ]] || { printf '%s\n' 'Exact commit required' >&2; exit 2; }
[[ $(git -C "$repo_root" rev-parse HEAD) == "$code_commit" ]] || { printf '%s\n' 'Commit mismatch' >&2; exit 2; }
[[ -z $(git -C "$repo_root" status --porcelain) ]] || { printf '%s\n' 'Clean checkout required' >&2; exit 2; }
origin=$(git -C "$repo_root" remote get-url origin)
[[ $origin == https://github.com/jang2296/MMDL_test.git || $origin == git@github.com:jang2296/MMDL_test.git ]] || { printf '%s\n' 'MMDL_test origin required' >&2; exit 2; }
[[ $task_root == /workspace/* && $task_root != *'/../'* && $task_root != */.. && $task_root != /workspace/ ]] || { printf '%s\n' 'Runtime root must be a child of /workspace' >&2; exit 2; }
if [[ $execute != true ]]; then
  printf 'DRY_RUN: commit=%s runtime=%s; install, CPU checks, GPU doctor, pinned model acquisition only. No training.\n' "$code_commit" "$task_root"
  exit 0
fi
[[ -n ${RUNPOD_POD_ID:-} ]] || { printf '%s\n' 'RunPod only'; exit 2; }
[[ $owned_pod_id == "$RUNPOD_POD_ID" && -n $gpu_cutoff_utc && -n $billing_start_utc && -n $balance_usd && -n $reserve_usd && -n $hourly_usd ]] || { printf '%s\n' 'Exact owned Pod and explicit financial/time bounds required'; exit 2; }
if [[ ${MMDL_BUDGET_WATCHDOG_PID:-} != "$PPID" ]]; then
  export PYTHONPATH="$repo_root/src:$repo_root"
  cd "$repo_root"
  exec python3.12 -m fine_tuning.budget --owned-pod-id "$owned_pod_id" \
    --gpu-cutoff-utc "$gpu_cutoff_utc" --billing-start-utc "$billing_start_utc" \
    --balance-usd "$balance_usd" --reserve-usd "$reserve_usd" --hourly-usd "$hourly_usd" \
    --max-seconds 3600 -- bash "$repo_root/fine_tuning/runpod_setup.sh" "${original_arguments[@]}"
fi
mountpoint -q /workspace || { printf '%s\n' 'Require persistent Pod volume'; exit 2; }
[[ ! -e $task_root && ! -L $task_root ]] || { printf '%s\n' 'Existing runtime refused'; exit 2; }
[[ $(realpath -m -- "$task_root") == "$task_root" ]] || { printf '%s\n' 'Non-canonical runtime path refused'; exit 2; }
mkdir -p "$task_root/setup"
python3.12 -m venv "$task_root/venv"
python=$task_root/venv/bin/python
"$python" -m pip install -r "$repo_root/env/requirements-experiments.lock"
"$python" -m pip check
"$python" -m pip freeze --all > "$task_root/setup/pip-freeze.txt"
export PYTHONPATH="$repo_root/src:$repo_root"
export HF_HOME="$task_root/hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=0
export MMDL_STORAGE_RESERVE_GIB=25
export PYTHONUNBUFFERED=1
cd "$repo_root"
"$python" -m unittest discover -s fine_tuning/tests -v
"$python" -m mmdl.runtime.environment --output "$task_root/setup/doctor.json" --storage-path /workspace --required-gib 30
"$python" - "$task_root" "$code_commit" <<'PY'
import json, pathlib, sys
from mmdl.data.download import MODEL_FILES, acquire
from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION
from fine_tuning.audit import dump
root=pathlib.Path(sys.argv[1])
manifest,snapshot=acquire(MODEL_ID,MODEL_REVISION,'model',MODEL_FILES,root/'hf-cache')
expected=json.loads(pathlib.Path('manifests/models/baseline.json').read_text())
assert manifest['files']==expected['files'], 'Pinned model hash mismatch'
dump(root/'setup/model.json',{'model_path':str(snapshot),'manifest':manifest,'status':'MODEL_HASH_VERIFIED',
    'code_repository':'jang2296/MMDL_test','code_commit':sys.argv[2]})
print('MODEL_READY',snapshot,flush=True)
PY
printf '%s\n' 'SETUP_COMPLETE_MODEL_READY; training/evaluation not automatically started'
