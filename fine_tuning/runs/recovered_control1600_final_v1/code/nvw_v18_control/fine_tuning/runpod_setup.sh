#!/usr/bin/env bash
# Private overlay on a fresh, immutable public checkout. HF bytes stay on RunPod.
set -euo pipefail
task_root=/workspace/nvw-runtime
code_commit=342ceff1d0baaaa57aa63c644b9f64b007caa174
[[ -n ${RUNPOD_POD_ID:-} ]] || { printf '%s\n' 'RunPod only'; exit 2; }
mountpoint -q /workspace || { printf '%s\n' 'Require persistent Pod volume'; exit 2; }
[[ ! -e $task_root ]] || { printf '%s\n' 'Existing runtime refused'; exit 2; }
mkdir -p "$task_root/setup"
git clone https://github.com/jang2296/MMDL "$task_root/repo"
git -C "$task_root/repo" checkout --detach "$code_commit"
[[ $(git -C "$task_root/repo" rev-parse HEAD) == "$code_commit" ]]
[[ -z $(git -C "$task_root/repo" status --porcelain) ]]
cp -a /workspace/private-overlay/fine_tuning "$task_root/repo/fine_tuning"
python3.12 -m venv "$task_root/venv"
python=$task_root/venv/bin/python
"$python" -m pip install -r "$task_root/repo/env/requirements-eval.lock"
"$python" -m pip install qwen-vl-utils==0.0.14
"$python" -m pip check
"$python" -m pip freeze --all > "$task_root/setup/pip-freeze.txt"
export PYTHONPATH="$task_root/repo/src:$task_root/repo"
export HF_HOME="$task_root/hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=0
export MMDL_STORAGE_RESERVE_GIB=25
export PYTHONUNBUFFERED=1
cd "$task_root/repo"
"$python" -m unittest discover -s fine_tuning/tests -v
"$python" -m mmdl.runtime.environment --output "$task_root/setup/doctor.json" --storage-path /workspace --required-gib 30
"$python" - "$task_root" <<'PY'
import json, pathlib, sys
from mmdl.data.download import MODEL_FILES, acquire
from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION
from fine_tuning.audit import dump
root=pathlib.Path(sys.argv[1])
manifest,snapshot=acquire(MODEL_ID,MODEL_REVISION,'model',MODEL_FILES,root/'hf-cache')
expected=json.loads(pathlib.Path('manifests/models/baseline.json').read_text())
assert manifest['files']==expected['files'], 'Pinned model hash mismatch'
dump(root/'setup/model.json',{'model_path':str(snapshot),'manifest':manifest,'status':'MODEL_HASH_VERIFIED'})
print('MODEL_READY',snapshot,flush=True)
PY
printf '%s\n' 'SETUP_COMPLETE_MODEL_READY'
