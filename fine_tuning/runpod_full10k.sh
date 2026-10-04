#!/usr/bin/env bash
# Private G6 restoration; original trainer/approvals remain unchanged.
set -euo pipefail
[[ ${RUNPOD_POD_ID:-} == muuxuzfdujzgcu ]] || { printf '%s\n' 'Exact owned Pod required'; exit 2; }
task_root=/workspace/nvw-runtime
source_root=$task_root/code/nvw_full10k_resume_v1
python=$task_root/venv/bin/python
model=$task_root/hf-cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
[[ -f $task_root/setup/model.json && ! -e $source_root ]]
"$python" - <<'PY'
import hashlib, pathlib, shutil, tarfile
root=pathlib.Path('/workspace/nvw-runtime')
controller=pathlib.Path('/workspace/run_experiment_full10k_resume_v1.py')
with controller.open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()=='b7d709742492fdb18fcf835e372b44c4e69a1560163926b9e1177c853f2c40d0'
assert not (root/'repo/fine_tuning/generated/pilot_pool_v1').exists()
assert not (root/'private-generated').exists() and not (root/'g4-recovery').exists()
archives={
    'pilot_pool_v1.tar.gz': ('2073404656848e4de45924f1a8aa544c3ab1d34b26506fd6a631658ffe38b353',root/'repo'),
    'replacement_pool_v1.tar.gz': ('de9709da8c8e1da9cd357ebccc7ac1452c59ab38ecedff4591c9671931a2c84e',root/'private-generated'),
    'full10k_g4_reuse_20261004_v1.tar.gz': ('528ba3dedb8b80787809a7a8cf593bce5aad9df4c4a318b19bdbd8ba1ef9d90c',root/'g4-recovery'),
}
for name,(digest,target) in archives.items():
    archive=pathlib.Path('/workspace')/name
    with archive.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,name
    target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as source:
        assert all((m.isfile() or m.isdir()) and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts for m in source.getmembers())
        source.extractall(target,filter='data')
old=root/'repo/fine_tuning/runs/recovered_control1600_final_v1/code/nvw_v18_control'
new=root/'code/nvw_full10k_resume_v1'
shutil.copytree(old,new)
shutil.copy2(controller,new/'fine_tuning/run_experiment.py')
shutil.copytree(root/'g4-recovery/closed_controller',root/'runs/experiment_v2')
smoke=root/'training/smoke_v3'
(smoke/'approved').mkdir(parents=True)
for name,target in [('records.jsonl',smoke/'approved/records.jsonl'),('smoke_manifest.json',smoke/'smoke_manifest.json')]:
    shutil.copy2(root/'g4-recovery/smoke_approval_v3'/name,target)
print('OWN_GENERATED_POOLS_AND_G4_EVIDENCE_RESTORED',flush=True)
PY
export PYTHONPATH="$source_root:$task_root/repo/src"
export HF_HOME="$task_root/hf-cache" HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1
export MMDL_STORAGE_RESERVE_GIB=25
cd "$source_root"
"$python" -m fine_tuning.prepare_sources --task-root "$task_root" --stage benchmarks
"$python" -m fine_tuning.prepare_training index --coverage "$task_root/benchmark-exclusion/coverage.json" --output "$task_root/benchmark-exclusion/fingerprint_index_v1.json"
"$python" -m fine_tuning.prepare_sources --task-root "$task_root" --stage trace --inventory "$source_root/fine_tuning/source_inventory_20261004.json" --trace-shards 4
approval=(--source-registry "$source_root/fine_tuning/source_registry_v1.json" --benchmark-index "$task_root/benchmark-exclusion/fingerprint_index_v1.json" --model-path "$model")
trace=(--trace-downloads "$task_root/setup/trace-downloads.json" --trace-cache-root "$task_root/hf-cache/hub")
raw=$task_root/training/pilot_restore_raw_v1
if "$python" -m fine_tuning.prepare_training pilot --nvw-root "$task_root/repo/fine_tuning/generated/pilot_pool_v1" "${trace[@]}" "${approval[@]}" --output "$raw"; then
    printf '%s\n' 'Unexpected raw READY: inspect approval chain; no automatic continuation'; exit 2
fi
"$python" - "$raw" <<'PY'
import json,pathlib,sys
receipt=json.loads((pathlib.Path(sys.argv[1])/'preparation_receipt.json').read_text())
assert receipt['status']=='INCOMPLETE_QUOTA' and receipt['stage']=='pilot' and receipt['quarantined_records']>0
print('EXPECTED_QUARANTINE_SHORTFALL_VERIFIED',flush=True)
PY
"$python" -m fine_tuning.prepare_training complete-pilot --parent "$raw" --parent-source-root "$source_root/fine_tuning" "${trace[@]}" "${approval[@]}" --output "$task_root/training/pilot_restore_complete_v1"
"$python" -m fine_tuning.prepare_training disjoint-full-pilot --parent "$task_root/training/pilot_restore_complete_v1" --nvw-root "$task_root/private-generated/replacement_pool_v1" --smoke-records "$task_root/training/smoke_v3/approved/records.jsonl" --smoke-manifest "$task_root/training/smoke_v3/smoke_manifest.json" "${approval[@]}" --output "$task_root/training/common10k_restore_v1"
"$python" -m fine_tuning.evaluate prepare-mmmu --data-root "$task_root/data/evaluation/mmmu" --per-subject 2 --output "$task_root/evaluation/mmmu60_restore_v1"
"$python" -m fine_tuning.evaluate prepare-external --nvw-root "$task_root/repo/fine_tuning/generated/prototype_reviewed" --trace-root "$raw" --roots-per-family 4 --trace-count 24 --output "$task_root/evaluation/external60_restore_v1"
"$python" -m fine_tuning.run_experiment --model-path "$model" --smoke-records "$task_root/training/smoke_v3/approved/records.jsonl" --smoke-manifest "$task_root/training/smoke_v3/smoke_manifest.json" --pilot-records "$task_root/training/common10k_restore_v1/approved/records.jsonl" --pilot-manifest "$task_root/training/common10k_restore_v1/common_manifest.json" --image-root "$task_root/repo/fine_tuning/generated/prototype_reviewed" --pilot-image-root "$task_root/training/common10k_restore_v1" --mmmu-cohort "$task_root/evaluation/mmmu60_restore_v1" --external-cohort "$task_root/evaluation/external60_restore_v1" --protocol "$source_root/fine_tuning/protocol.json" --output "$task_root/runs/experiment_full10k_resume_v1" --owned-pod-id muuxuzfdujzgcu --gpu-cutoff-utc 2026-10-05T05:05:10Z --billing-start-utc 2026-10-04T08:05:10.374Z --balance-usd 12.43 --reserve-usd 1.0 --hourly-usd 0.54 --reuse-smoke-controller "$task_root/runs/experiment_v2" --training-forecast-margin 1.5
printf '%s\n' 'FULL10K_CONTROLLER_CLOSED_CHECK_STATUS_AND_RECOVER_BEFORE_DELETE'
