#!/usr/bin/env bash
# One-time preparation of the Secure network volume: exact-lock venv, pinned model, manifest.
# Run only from a clean, pinned MMDL_test clone on a Pod that mounts the volume at /workspace.
# Creates one new directory; never deletes, moves or overwrites anything.
# Commit, volume root, environment and the financial/time bounds come only from the approved
# `pod-prep-v1` job file (job.py validates it and its approval); there are no CLI flags for them.
# With `eval_data: true` in that job it also acquires pinned MMMU validation (its own approval) into
# <volume_root>/eval-data: the dataset's own HF cache (eval-data/hf-cache, never the training HF_HOME)
# and the evaluator's data root eval-data/mmmu (symlinks + manifest.json), hash-checked against
# manifests/datasets/mmmu.json. No token is passed; an HF auth/gating error fails the whole prep closed.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$here/../../.." && pwd -P)
workspace=/workspace
python=${MMDL_POD_PYTHON:-python3.12}
job_file=
max_seconds=5400  # = job.PREP_MAX_SECONDS in the deadline fit
execute=false
original_arguments=("$@")
usage='Usage: bash fine_tuning/notebook_rl/pod/volume.sh --job /abs/PREP.json [--execute]'
while (($#)); do
  case "$1" in
    --job) job_file=${2:?Missing job file}; shift 2 ;;
    --execute) execute=true; shift ;;
    --help) printf '%s\n' "$usage"; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n $job_file && -f $job_file ]] || { printf '%s\n' "$usage" >&2; exit 2; }
[[ $job_file == /* ]] || { printf '%s\n' 'Job file path must be absolute' >&2; exit 2; }
export PYTHONPATH="$repo_root/src:$repo_root"
job() { "$python" -B -m fine_tuning.notebook_rl.pod.job "$job_file" --workspace "$workspace" "$@"; }
plan=$(job) || exit 2
[[ $(job --get version) == pod-prep-v1 ]] || { printf '%s\n' 'volume.sh needs a pod-prep-v1 job' >&2; exit 2; }
mapfile -d '' -t financial < <(job --financial-argv)
((${#financial[@]} == 12)) || { printf '%s\n' 'Cannot derive the financial bounds from the job' >&2; exit 2; }
owned_pod_id=$(job --get owned_pod_id)
code_commit=$(job --get commit)
volume_root=$(job --get volume_root)
environment=$(job --get environment)
eval_data=$(job --get eval_data)
[[ $environment == experiments || $environment == vllm || $environment == proof ]] || { printf '%s\n' 'Environment must be experiments, vllm or proof'; exit 2; }
[[ $code_commit =~ ^[0-9a-f]{40}$ ]] || { printf '%s\n' 'Exact commit required' >&2; exit 2; }
[[ $(git -C "$repo_root" rev-parse HEAD) == "$code_commit" ]] || { printf '%s\n' 'Commit mismatch' >&2; exit 2; }
[[ -z $(git -C "$repo_root" status --porcelain --untracked-files=all) ]] || { printf '%s\n' 'Clean checkout required' >&2; exit 2; }
origin=$(git -C "$repo_root" remote get-url origin)
[[ $origin == https://github.com/jang2296/MMDL_test.git || $origin == git@github.com:jang2296/MMDL_test.git ]] || { printf '%s\n' 'MMDL_test origin required' >&2; exit 2; }
[[ -n $volume_root && $volume_root == /workspace/* && $volume_root != *'/../'* && $volume_root != */.. && $volume_root != /workspace/ && $(realpath -m -- "$volume_root") == "$volume_root" ]] || { printf '%s\n' 'Volume root must be a canonical child of /workspace'; exit 2; }
[[ ! -e $volume_root && ! -L $volume_root ]] || { printf '%s\n' 'Existing volume root refused'; exit 2; }
lock=$repo_root/env/requirements-$environment.lock
[[ -f $lock ]] || { printf 'Missing lock %s\n' "$lock" >&2; exit 2; }
read -r lock_sha _ < <(sha256sum -- "$lock")
venv=$volume_root/venv-${lock_sha:0:16}
if [[ $execute != true ]]; then
  printf 'DRY_RUN: nothing executed. Job plan:\n%s\n' "$plan"
  printf 'Plan:\n  clean MMDL_test clone %s\n  new dir %s\n  venv %s from %s (sha256 %s)\n  pip check; pip freeze -> %s/pip-freeze.txt\n  pinned model -> %s/hf-cache (hash-verified against manifests/models/baseline.json)\n  eval data (MMMU validation): %s -> %s/eval-data/mmmu, cache %s/eval-data/hf-cache (hash-verified against manifests/datasets/mmmu.json)\n  manifest -> %s/volume_manifest.json status VOLUME_READY\n' \
    "$code_commit" "$volume_root" "$venv" "env/requirements-$environment.lock" "$lock_sha" "$volume_root" "$volume_root" "$eval_data" "$volume_root" "$volume_root" "$volume_root"
  exit 0
fi
[[ -n ${RUNPOD_POD_ID:-} ]] || { printf '%s\n' 'RunPod only'; exit 2; }
[[ $owned_pod_id == "$RUNPOD_POD_ID" ]] || { printf '%s\n' 'This Pod is not job.financial.owned_pod_id'; exit 2; }
budget_command=("$python" -m fine_tuning.budget "${financial[@]}"
  --max-seconds "$max_seconds" --result-file "$volume_root/budget_result.json"
  -- bash "$here/volume.sh" "${original_arguments[@]}")
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
mountpoint -q "$workspace" || { printf '%s\n' 'Require persistent network volume'; exit 2; }
mkdir "$volume_root"
"$python" -m venv "$venv"
"$venv/bin/python" -m pip install -r "$lock"
"$venv/bin/python" -m pip check
"$venv/bin/python" -m pip freeze --all > "$volume_root/pip-freeze.txt"
export HF_HOME="$volume_root/hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=0
export MMDL_STORAGE_RESERVE_GIB=25
export PYTHONUNBUFFERED=1
cd "$repo_root"
"$venv/bin/python" - "$volume_root" "$code_commit" "$environment" "$lock" "$venv" "$eval_data" <<'PY'
import json, pathlib, sys
from mmdl.data.download import MODEL_FILES, acquire
from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION
from fine_tuning.notebook_rl.pod.verify import EVAL_CACHE, EVAL_DATA_ROOT, write_volume_manifest
root, commit, environment, lock, venv, eval_flag = sys.argv[1:7]
root = pathlib.Path(root)
manifest, snapshot = acquire(MODEL_ID, MODEL_REVISION, 'model', MODEL_FILES, root/'hf-cache')
expected = json.loads(pathlib.Path('manifests/models/baseline.json').read_text())
if manifest['files'] != expected['files']:  # not assert: PYTHONOPTIMIZE would drop it
    raise SystemExit('Pinned model hash mismatch')
eval_data = None
if eval_flag == 'true':
    from mmdl.runtime.artifacts import write_json
    from mmdl.runtime.contracts import DATA_ID, DATA_REVISION, SUBJECTS
    names = ['README.md'] + [f'{s}/validation-00000-of-00001.parquet' for s in SUBJECTS]
    (root/EVAL_CACHE).mkdir(parents=True)
    data, _ = acquire(DATA_ID, DATA_REVISION, 'dataset', names, root/EVAL_CACHE, root/EVAL_DATA_ROOT)
    pinned = json.loads(pathlib.Path('manifests/datasets/mmmu.json').read_text())
    if (pinned['repo_id'], pinned['revision'], pinned['files']) != (DATA_ID, DATA_REVISION, data['files']):
        raise SystemExit('Pinned MMMU validation hash mismatch')
    # The evaluator's data-root manifest, as mmdl.data.download.prepare writes it.
    write_json(root/EVAL_DATA_ROOT/'manifest.json', dict(data, usage='evaluation_only',
               storage='MMDL_VOLUME/' + EVAL_DATA_ROOT, source=pinned['source']))
    eval_data = {'repo_id': DATA_ID, 'revision': DATA_REVISION, 'files': data['files']}
write_volume_manifest(root, commit=commit, environment=environment, lock=lock, venv=venv,
    freeze=root/'pip-freeze.txt', snapshot=snapshot, model_files=manifest['files'], eval_data=eval_data)
print('VOLUME_READY', root, flush=True)
PY
printf '%s\n' 'VOLUME_READY; no training or evaluation was started'
