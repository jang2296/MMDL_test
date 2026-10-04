#!/usr/bin/env bash
# Private transferred bundle only. No Pod creation, Git operation, or benchmark download.
# Check without writes: bash -n "$0"; bash "$0" --workspace /workspace/bottleneck-runtime
set -euo pipefail
bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
workspace=/workspace/bottleneck-runtime
volume=/workspace
bootstrap=python3.12
execute=false
resume_download=false
deadline=2026-10-03T04:22:38Z
while (($#)); do
  case "$1" in
    --workspace) workspace=${2:?}; shift 2 ;;
    --volume-root) volume=${2:?}; shift 2 ;;
    --python) bootstrap=${2:?}; shift 2 ;;
    --execute) execute=true; shift ;;
    --resume-download) resume_download=true; shift ;;
    --help) printf '%s\n' 'Usage: runpod_setup.sh [--workspace /workspace/task] [--volume-root /workspace] [--python python3.12] [--resume-download] [--execute]'; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ $volume == /workspace || $volume == /workspace/* ]] || { printf '%s\n' 'Volume must be /workspace or a named descendant mount.' >&2; exit 2; }
[[ $workspace == "$volume/"* && $workspace != *'/../'* && $workspace != */.. && $workspace != *'/./'* ]] || { printf '%s\n' 'Use an absolute named workspace below the volume.' >&2; exit 2; }
[[ $(realpath -m "$workspace") == "$workspace" && $(realpath -m "$volume") == "$volume" ]] || { printf '%s\n' 'Symlink or noncanonical workspace refused.' >&2; exit 2; }
[[ $workspace != "$bundle" && $workspace != "$bundle/"* && $bundle != "$workspace/"* ]] || { printf '%s\n' 'Runtime workspace must be separate from the bundle.' >&2; exit 2; }
lock=$bundle/env/requirements-vllm.lock
[[ -f $lock && -f $bundle/manifests/models/baseline.json ]] || { printf '%s\n' 'Private bundle lacks exact lock/model manifest.' >&2; exit 2; }
printf 'Mode=%s bundle=%s workspace=%s deadline=%s\n' "$([[ $execute == true ]] && printf EXECUTE || printf DRY_RUN)" "$bundle" "$workspace" "$deadline"
printf '%s\n' 'Plan: Python 3.12 separate venv; exact lock; pip check/freeze; GPU/BF16 doctor and mount receipt; model-only acquire + baseline hashes.' 'No dataset download, model inference, quantization, GitHub, or cloud API calls.' 'Require verified persistent volume and >=100 GiB free before install; >=160 GB volume recommended. Existing acquisition retains 50 GiB reserve.'
if [[ $execute != true ]]; then exit 0; fi
[[ -n ${RUNPOD_POD_ID:-} && $bundle == "$volume/"* ]] || { printf '%s\n' 'Execution requires a RunPod ID and transferred bundle on the persistent volume.' >&2; exit 2; }
mountpoint -q "$volume" || { printf '%s\n' 'Persistent volume must be an actual mount, not container root.' >&2; exit 2; }
"$bootstrap" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Exact lock requires Python 3.12"'
remaining() {
  "$bootstrap" - "$deadline" <<'PY'
import datetime, sys, time
seconds = int(datetime.datetime.fromisoformat(sys.argv[1].replace('Z', '+00:00')).timestamp() - time.time()) - 10
if seconds <= 0:
    raise SystemExit('Absolute operational deadline expired')
print(seconds)
PY
}
bounded() { timeout --foreground --signal=TERM --kill-after=10s "$(remaining)s" "$@"; }
remaining >/dev/null
"$bootstrap" - "$volume" <<'PY'
import shutil, sys
if shutil.disk_usage(sys.argv[1]).free < 100 * 1024**3:
    raise SystemExit('Insufficient volume free space: retain 50 GiB reserve plus installation/acquisition peak')
PY
export PYTHONPATH="$bundle/src:$bundle${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="$workspace/hf-cache"
# The base image enables an optional package absent from the exact lock.
export HF_HUB_ENABLE_HF_TRANSFER=0
export MMDL_STORAGE_RESERVE_GIB=50
cd "$bundle"
python=$workspace/venv/bin/python
if [[ $resume_download == true ]]; then
  [[ ! -e $workspace/setup/model.json && -x $python ]] || { printf '%s\n' 'Resume requires an incomplete, existing private setup.' >&2; exit 2; }
  bounded "$python" - "$workspace" <<'PY'
import json, os, pathlib, subprocess, sys
work = pathlib.Path(sys.argv[1])
mount = json.loads((work/'setup/mount.json').read_text())
if os.environ['RUNPOD_POD_ID'] not in json.dumps(mount):
    raise SystemExit('Setup mount receipt does not belong to this Pod')
if pathlib.Path(sys.prefix).resolve() != (work/'venv').resolve():
    raise SystemExit('Refuse an unrelated environment')
freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True)
if freeze != (work/'setup/pip-freeze.txt').read_text():
    raise SystemExit('Installed environment changed after initial setup')
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
PY
else
  [[ ! -e $workspace/setup && ! -e $workspace/venv ]] || { printf '%s\n' 'Existing setup refused; use --resume-download only for a verified incomplete download.' >&2; exit 2; }
  mkdir -p "$workspace"
  mkdir "$workspace/setup"
  mkdir -p "$workspace/hf-cache" "$workspace/results"
  findmnt --json --target "$volume" > "$workspace/setup/mount.json"
  bounded "$bootstrap" -m venv "$workspace/venv"
  bounded "$python" -m pip install -r "$lock" > "$workspace/setup/install.log" 2>&1
  bounded "$python" -m pip check > "$workspace/setup/pip-check.txt" 2>&1
  "$python" -m pip freeze --all > "$workspace/setup/pip-freeze.txt"
  bounded "$python" -m mmdl.runtime.environment --output "$workspace/setup/doctor.json" --storage-path "$volume" --required-gib 25
fi
bounded "$python" - "$workspace" "$bundle" <<'PY'
import hashlib, importlib.metadata, json, os, pathlib
from mmdl.data.download import acquire, MODEL_FILES
from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION
work, bundle = map(pathlib.Path, __import__('sys').argv[1:])
for name, expected in {'torch':'2.8.0+cu128', 'transformers':'4.57.1', 'vllm':'0.11.0'}.items():
    if importlib.metadata.version(name) != expected:
        raise SystemExit(f'Wrong locked package version: {name}')
doctor = json.loads((work/'setup/doctor.json').read_text())['environment']
gpus = doctor['torch'].get('gpus', [])
if (len(gpus) != 1 or gpus[0]['memory_total_bytes'] < 23 * 1024**3
        or doctor['torch']['bf16_probe']['passed'] is not True):
    raise SystemExit('Require one visible >=24 GB class GPU and actual BF16 probe')
baseline = json.loads((bundle/'manifests/models/baseline.json').read_text())
if (baseline['repo_id'], baseline['revision'], baseline.get('dtype')) != (MODEL_ID, MODEL_REVISION, 'bfloat16'):
    raise SystemExit('Pinned baseline identity mismatch')
manifest, snapshot = acquire(MODEL_ID, MODEL_REVISION, 'model', MODEL_FILES, work/'hf-cache')
if any(manifest[field] != baseline[field] for field in ('repo_id', 'revision', 'files')):
    raise SystemExit('Acquired model differs from immutable baseline manifest')
manifest.update(model_path=str(snapshot), pod_id=os.environ['RUNPOD_POD_ID'],
                lock_sha256=hashlib.sha256((bundle/'env/requirements-vllm.lock').read_bytes()).hexdigest(),
                status='MODEL_ACQUIRED_HASH_VERIFIED_NOT_INFERENCE', benchmark_downloaded=False)
with (work/'setup/model.json').open('x') as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write('\n')
PY
printf 'Setup completed without inference. Receipts: %s/setup\n' "$workspace"
