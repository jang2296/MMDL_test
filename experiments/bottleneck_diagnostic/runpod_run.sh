#!/usr/bin/env bash
# Check without inference: bash -n "$0"; bash "$0" --workspace /workspace/bottleneck-runtime --data /workspace/bundle/prepared
set -euo pipefail
bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
workspace=/workspace/bottleneck-runtime
volume=/workspace
data=$bundle/analysis_exports/bottleneck_20261003/prepared
oracles=
protocol=$bundle/experiments/bottleneck_diagnostic/protocol.json
hardware=$bundle/configs/hardware/rtx3090_24gb.yaml
stress_budget=3600
coverage=false
execute=false
while (($#)); do
  case "$1" in
    --workspace) workspace=${2:?}; shift 2 ;;
    --volume-root) volume=${2:?}; shift 2 ;;
    --data) data=${2:?}; shift 2 ;;
    --oracles) oracles=${2:?}; shift 2 ;;
    --protocol) protocol=${2:?}; shift 2 ;;
    --hardware) hardware=${2:?}; shift 2 ;;
    --stress-budget-seconds) stress_budget=${2:?}; shift 2 ;;
    --coverage) coverage=true; shift ;;
    --execute) execute=true; shift ;;
    --help) printf '%s\n' 'Usage: runpod_run.sh --oracles /workspace/reviewed.jsonl [--workspace PATH] [--volume-root MOUNT] [--data PATH] [--protocol PATH] [--hardware PATH] [--stress-budget-seconds 3600] [--coverage] [--execute]'; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ $volume == /workspace || $volume == /workspace/* ]] || { printf '%s\n' 'Use a persistent /workspace mount.' >&2; exit 2; }
[[ $workspace == "$volume/"* && $(realpath -m "$workspace") == "$workspace" && $(realpath -m "$volume") == "$volume" ]] || { printf '%s\n' 'Named canonical workspace required.' >&2; exit 2; }
[[ $workspace != "$bundle" && $workspace != "$bundle/"* && $bundle != "$workspace/"* ]] || { printf '%s\n' 'Runtime workspace must be separate from the bundle.' >&2; exit 2; }
[[ $stress_budget =~ ^[1-9][0-9]*$ ]] && ((stress_budget <= 14400)) || { printf '%s\n' 'Stress budget must be 1..14400 seconds.' >&2; exit 2; }
printf 'Mode=%s workspace=%s data=%s oracles=%s protocol=%s hardware=%s\n' "$([[ $execute == true ]] && printf EXECUTE || printf DRY_RUN)" "$workspace" "$data" "${oracles:-MISSING (all oracle cells blocked)}" "$protocol" "$hardware"
printf 'Plan: 12-item stress (<=%ss) -> actual completion gate -> 60-item pilot (<=14400s) -> optional 42-item coverage=%s. Deadline=2026-10-03T04:22:38Z. First seed=3407 only.\n' "$stress_budget" "$coverage"
if [[ $execute != true ]]; then exit 0; fi
[[ -n ${RUNPOD_POD_ID:-} && $bundle == "$volume/"* ]] || { printf '%s\n' 'Execution requires RunPod and a transferred private bundle on volume.' >&2; exit 2; }
mountpoint -q "$volume" || { printf '%s\n' 'Persistent volume is not mounted.' >&2; exit 2; }
for path in "$data" "$protocol" "$hardware"; do
  [[ $(realpath -e "$path") == "$volume/"* ]] || { printf 'Path outside persistent volume: %s\n' "$path" >&2; exit 2; }
done
if [[ -n $oracles ]]; then
  [[ $(realpath -e "$oracles") == "$volume/"* ]] || { printf '%s\n' 'Oracle path outside persistent volume.' >&2; exit 2; }
fi
python=$workspace/venv/bin/python
[[ -x $python && -f $workspace/setup/model.json ]] || { printf '%s\n' 'Verified setup is missing.' >&2; exit 2; }
export PYTHONPATH="$bundle/src:$bundle${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="$workspace/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$bundle"
model=$("$python" - "$workspace/setup/model.json" "$bundle/env/requirements-vllm.lock" "$bundle/manifests/models/baseline.json" "$workspace/hf-cache" <<'PY'
import hashlib, importlib.metadata, json, os, pathlib, sys
receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
if receipt['lock_sha256'] != hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest():
    raise SystemExit('Lock changed after setup')
if receipt['pod_id'] != os.environ['RUNPOD_POD_ID']:
    raise SystemExit('Setup belongs to a different Pod')
baseline = json.loads(pathlib.Path(sys.argv[3]).read_text())
if any(receipt[field] != baseline[field] for field in ('repo_id','revision','files')):
    raise SystemExit('Baseline manifest changed after setup')
for name, expected in {'torch':'2.8.0+cu128','transformers':'4.57.1','vllm':'0.11.0'}.items():
    if importlib.metadata.version(name) != expected:
        raise SystemExit(f'Locked environment changed: {name}')
model = pathlib.Path(receipt['model_path']).resolve(strict=True)
if not model.is_relative_to(pathlib.Path(sys.argv[4]).resolve(strict=True)):
    raise SystemExit('Model snapshot escaped the persistent cache')
for item in receipt['files']:
    path = model/item['path']
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if path.stat().st_size != item['bytes'] or digest != item['sha256']:
        raise SystemExit('Model changed after setup')
print(model)
PY
)
# Runtime validates ordered image hashes and writes frozen protocol/code/data/oracle identities.
"$python" - "$protocol" "$data" <<'PY'
import json, pathlib, sys
from experiments.bottleneck_diagnostic.cli import selected
protocol = json.loads(pathlib.Path(sys.argv[1]).read_text())
if protocol.get('overall_deadline_utc') != '2026-10-03T04:22:38Z':
    raise SystemExit('Refuse changed absolute operational deadline')
if protocol.get('modes') != ['I','T0','O0','O1','IO0','P','M','MI']:
    raise SystemExit('Refuse changed diagnostic modes')
if protocol.get('oracle_review_policy') != 'ALLOW_AI_REVIEWED_DIAGNOSTIC':
    raise SystemExit('Explicit independently AI-reviewed diagnostic policy required')
data = pathlib.Path(sys.argv[2])
for suite, expected in [('stress',12),('pilot',60),('coverage',42)]:
    if len(selected(data, suite)) != expected:
        raise SystemExit(f'Unexpected declared sample count: {suite}')
PY
remaining() {
  "$python" - "$protocol" <<'PY'
import json, pathlib, sys
from experiments.bottleneck_diagnostic.runner import remaining_window
seconds = int(remaining_window(json.loads(pathlib.Path(sys.argv[1]).read_text()))) - 10
if seconds <= 0:
    raise SystemExit('Absolute deadline expired; preserve existing artifacts')
print(seconds)
PY
}
remaining >/dev/null
[[ ! -e $workspace/results/stress && ! -e $workspace/results/pilot && ! -e $workspace/results/coverage ]] || { printf '%s\n' 'Existing stage output refused; no overwrite or implicit restart.' >&2; exit 2; }
mkdir -p "$workspace/results"
oracle_args=()
if [[ -n $oracles ]]; then oracle_args=(--oracles "$oracles"); fi
stage() {
  local suite=$1 budget=$2 stage_exit=0
  local output=$workspace/results/$suite
  local hard_seconds
  hard_seconds=$(remaining)
  if ((hard_seconds > budget - 10)); then hard_seconds=$((budget - 10)); fi
  if ((hard_seconds <= 0)); then printf '%s\n' 'Insufficient stage time including termination margin.' >&2; return 2; fi
  mkdir "$output"
  timeout --foreground --signal=TERM --kill-after=10s "${hard_seconds}s" "$python" -m experiments.bottleneck_diagnostic.cli run --data "$data" --suite "$suite" --output "$output" --protocol "$protocol" --hardware "$hardware" --model-path "$model" --budget-seconds "$budget" "${oracle_args[@]}" > "$output/run.log" 2>&1 || stage_exit=$?
  if [[ -f $output/responses.jsonl ]]; then
    "$python" -m experiments.bottleneck_diagnostic.cli score --data "$data" --suite "$suite" --output "$output" "${oracle_args[@]}" > "$output/score.log" 2>&1
  fi
  if ((stage_exit != 0)); then
    printf 'Stage %s exited %s; artifacts retained; no next stage.\n' "$suite" "$stage_exit" >&2
    return "$stage_exit"
  fi
  "$python" - "$output/responses.jsonl" <<'PY'
import pathlib, sys
from experiments.bottleneck_diagnostic.ingest import read_rows
if any(row['runtime_status'] == 'SYSTEM_ERROR' for row in read_rows(pathlib.Path(sys.argv[1]))):
    raise SystemExit('Stage has system errors; preserve artifacts and do not advance')
PY
}
stage stress "$stress_budget"
"$python" - "$data" "$workspace/results/stress" "$oracles" <<'PY'
import collections, json, pathlib, sys
from experiments.bottleneck_diagnostic.cli import selected
from experiments.bottleneck_diagnostic.ingest import read_rows
from experiments.bottleneck_diagnostic.oracle_validation import validate_oracle
data, output = map(pathlib.Path, sys.argv[1:3])
identity = json.loads((output/'protocol.json').read_text())
if identity['execution_kind'] != 'ACTUAL_INFERENCE' or identity['repetitions'] != [3407]:
    raise SystemExit('Smoke is not actual pinned first-pass inference')
history = read_rows(output/'responses.jsonl')
if any(row['runtime_status'] == 'SYSTEM_ERROR' for row in history):
    raise SystemExit('Smoke has system errors; pilot blocked')
latest = {(r['sample_id'],r['mode'],r['repetition']):r for r in history}
oracles = {(o['sample_id'],o['level']):o for o in read_rows(pathlib.Path(sys.argv[3]))} if sys.argv[3] else {}
counts = collections.Counter()
for sample in selected(data,'stress'):
    for mode in identity['protocol']['modes']:
        row = latest.get((sample['sample_id'],mode,3407))
        expected = 'COMPLETED'
        if mode in ('O0','O1','IO0'):
            level = 'O1' if mode == 'O1' else 'O0'
            verdict = validate_oracle(oracles.get((sample['sample_id'],level)), sample, level, identity['protocol']['oracle_review_policy'])
            expected = 'COMPLETED' if verdict['status'] == 'VALID' else verdict['status']
        elif not sample['images'] and mode in ('P','M','MI'):
            expected = 'EXCLUDED' if mode == 'P' else 'BLOCKED'
        if row is None or row['runtime_status'] != expected or row.get('run_kind') != 'ACTUAL_INFERENCE':
            raise SystemExit(f'Smoke required cell not completed or correctly excluded: {sample["sample_id"]}/{mode}')
        counts[expected] += 1
if not counts['COMPLETED']:
    raise SystemExit('No actual completed smoke cells')
receipt = dict(status='ACTUAL_STRESS_SMOKE_REQUIRED_AVAILABLE_CELLS_COMPLETED', system_errors=0,
               cell_status_counts=dict(counts), questions=12, provenance='BENCHMARK_DERIVED', training_allowed=False)
with (output/'smoke_gate.json').open('x') as stream:
    json.dump(receipt,stream,indent=2,sort_keys=True)
    stream.write('\n')
print(json.dumps(receipt,sort_keys=True))
PY
stage pilot 14400
if [[ $coverage == true ]]; then
  coverage_budget=$(remaining)
  stage coverage "$coverage_budget"
fi
printf 'Bounded stages ended; partial/BLOCKED/EXCLUDED states are not accuracy failures. Results: %s/results\n' "$workspace"
