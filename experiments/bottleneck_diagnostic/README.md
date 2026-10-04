# Private Qwen functional bottleneck diagnostics

Internal only: do not synchronize `MMDL/` or push these files. Inputs/oracles/results
live outside this code folder in `analysis_exports/bottleneck_20261003/`.
Every derived benchmark artifact is `BENCHMARK_DERIVED`, `training_allowed=false`.
Original V8 responses, parser and558/900 are immutable.

Actual private execution status on2026-10-03: the12-item stress set,60-item pilot,
42-item all-subject coverage extension and15-domain60-call Knowledge probe have
closed and been locally verified. Pilot+coverage includes102 items/30 subjects;
it is not the entire262 expansion or a whole-MMMU accuracy estimate. Remaining160,
additional3408/3409 repeats and actual LEGACY re-calls were not executed.
See `claudedocs/Qwen3_VL_4B_Bottleneck_Diagnostic_Report_20261003.md` and
`claudedocs/Qwen3_VL_4B_Knowledge_Oracle_20261003.md` for exact evidence and limits.
The separate historical-V8 Gemma audit attempted900extractions and2equivalence
calls:607RESOLVED/117REVIEW/176SYSTEM_ERROR. Adjudication is INCOMPLETE, the new
score is null, and558/900 remains unchanged. Successful and failed artifacts were
locally verified before owned-Pod deletion; no weights were recovered. Reusing
the expired deployment identities/deadline requires a new authorized protocol.

From the project root with the existing evaluation environment:

```bash
PYTHONPATH=src python -m unittest experiments.bottleneck_diagnostic.tests.test_runtime experiments.bottleneck_diagnostic.tests.test_metrics
PYTHONPATH=src python -m experiments.bottleneck_diagnostic.cli dry-run --suite stress --output <new-directory> --oracles <reviewed-oracles.jsonl>
PYTHONPATH=src python -m experiments.bottleneck_diagnostic.cli processor-audit --suite stress --model-path <pinned-snapshot> --output <new-audit.jsonl>
PYTHONPATH=src python -m experiments.bottleneck_diagnostic.cli inventory --model-path <pinned-snapshot> --output <new-modules.json>
PYTHONPATH=src python -m experiments.bottleneck_diagnostic.cli run --suite pilot --protocol experiments/bottleneck_diagnostic/protocol.json --model-path <pinned-snapshot> --oracles <reviewed-oracles.jsonl> --budget-seconds 14400 --output <run-directory>
PYTHONPATH=src python -m experiments.bottleneck_diagnostic.cli score --suite pilot --oracles <same-reviewed-oracles.jsonl> --output <run-directory> --score-output <new-score-directory>
PYTHONPATH=src python -m experiments.bottleneck_diagnostic.cli verify
```

`run` requires the existing exact vLLM0.11.0 environment and native BF16 GPU.
`local-smoke --suite local --hardware configs/hardware/rtx5060_8gb.yaml` uses a
distinct Transformers CPU-offload protocol; do not combine it with RunPod scores.
Gold is opened only by prepare/audit and the post-run score command, not runtime.
CPU dry-run deliberately blocks M/MI because no actual P response exists.

Conditions I/T0/O0/O1/IO0/P/M/MI use independent conversations. M/MI reuse the same
unedited P response. P is whole-VLM structure extraction, not pure Vision Encoder
evaluation; its exact JSON comparison is not semantic structural certification.
Approved AI-reviewed oracle execution requires the explicit diagnostic policy and
is never described as human-reviewed. Missing/incomplete/target-revealing oracles
stay BLOCKED/EXCLUDED. No LLM is trained and no benchmark-derived data may be trained.

Selection is frozen:12stress cases,60outcome-stratified pilot, nested262expansion.
After the pilot, `--suite coverage` chooses the first frozen correct/wrong pair
for each of the remaining21subjects (42items). This is an expansion subset, not
an outcome-dependent replacement. After both suites complete, `--suite expansion-remainder`
selects the other160 frozen expansion IDs in their original order, without regenerating
the pilot/coverage answers. Launch it from a separately frozen source bundle after the
pilot/time gate; do not replace code in an active run. A/B/C/D are predeclared subject-group hypotheses;
image types and paired recovery/regression are reported separately.
Repeated seeds aggregate within item; uncertainty resamples shared image/text families.
N/A/blocked/system errors are not model wrong. Balanced scores are not population MMMU accuracy.

Each run is append-only with request/code/protocol/input/oracle hashes. No output
cap, image resolution, checkpoint, quantization or sampling fallback is allowed.
Time/token stops preserve completion records and mark unprocessed cells BUDGET_STOP.
Scoring verifies record/raw/protocol/sample hashes and records absent declared cells
as PENDING. Post-run P review has its own oracle hash and cannot rewrite inference.
Model bytes are verified against the pinned manifest before loading. Absolute UTC
deadline bounds every request across runs/restarts, in addition to the4h pilot cap.
LEGACY requires the original image-role audit; the4 IDs with removed explanation
images are BLOCKED rather than called historical-input-equivalent.
The original local2-case smoke retained invalid P marker outputs as schema failures;
the pending cloud protocol now explicitly copies image-ID markers and freezes a new
prompt/code hash. These saved local responses are not repaired or overwritten.
The default serial backend favors simple, auditable P dependencies over throughput;
measure the12-case smoke before estimating the60pilot. Do not label empty files,
CPU fixtures, or a partial run as SMOKE_COMPLETED/PILOT_COMPLETED.

The separately versioned CPU scoring overlay `diagnostic-final-interface-v2-exact-option`
keeps the single terminal `FINAL_ANSWER` declaration and accepts only a valid bare
letter, `LETTER. exact corresponding option text`, or unique exact option text.
Only outer whitespace is trimmed; case, units, signs and modifiers are retained.
No earlier answer or intermediate value rescues a missing/ambiguous final declaration.
Existing strict-v1 scores remain intact. Apply the overlay using a fresh `--score-output`.
`v8_score` here means unchanged V8 applied to the NEW diagnostic raw interface; it
does not represent the preserved historical558/900 result. The running Pod is not patched.

Private continuation is separate from the frozen inference bundle. Check its CPU
tests and default dry plan before privately transferring it:

```bash
python -m unittest experiments.bottleneck_diagnostic.tests.test_finish_runpod
python experiments/bottleneck_diagnostic/finish_runpod.py --pod-id h447a4u6c0vq3k --coverage
```

Actual `--execute` additionally requires the recorded parent process birth and
raw-command hash. It never restarts that parent, verifies the completed480-cell
pilot, and budgets optional42-item coverage from measured request times with
three hours reserved for Gemma and30minutes for recovery. Only validated actual
900-item Gemma completion releases its reserve for a deferred coverage check.
Its new GPU work stops before the fixed recovery window; owned descendants use pidfd
handles for cleanup. This controller neither deletes a Pod nor certifies local
recovery. Original v1 inference and deployed-v5 Gemma gate bytes stay unchanged.

That controller description preserves the original job design. The current Gemma
continuation uses the separately snapshotted v8 code bundle and29-case gate;
failed earlier versions remain separate artifacts. Do not reuse archived parent
PIDs, code hashes or the expired absolute deadline for a future deployment.

After recovering closed stages, export their verified union without rewriting any
raw run or averaging per-run summaries:

```bash
PYTHONPATH=src python -m experiments.bottleneck_diagnostic.export_results \
  --data analysis_exports/bottleneck_20261003/prepared \
  --oracles <exact-oracles-used-by-these-runs.jsonl> \
  --run pilot=<closed-pilot-directory> --run coverage=<closed-coverage-directory> \
  --output analysis_exports/bottleneck_20261003/<fresh-export-directory>
```

Supply only stages that actually exist. Stress/local runs must be exported alone;
compatible, nonoverlapping suites share item/family-aware paired statistics. Missing,
blocked and excluded conditions retain their own status and denominator. CSV tables,
source hashes, separate V8/diagnostic columns, exact-JSON P metrics and per-mode costs
remain private. Failed execution-path attempts do not prove a backend call; budget
reservations are not observed generated tokens. Device snapshots are not true peaks.
Optional `--gemma-run <closed-audit>` and `--gemma-source <original900.jsonl>` validate
and CPU-replay actual saved decisions, retaining REVIEW queues and conditional decision
envelopes rather than declaring a new certified score. Retry/duplicate Gemma ledger
support is not certified. This exporter neither performs inference nor deletes a Pod.
The current post-run exporter also binds and replays saved semantic mapping requests
and outputs, includes mapping in stage costs, and reuses the already frozen source
digest in queue rows. Both final source-mutation checks remain active; no inference
code, raw response or scoring policy changed for these post-run fixes.

Recover only an explicit list of closed regular files; do not archive a live ledger,
weights, environments or caches. `--files` is a JSON object mapping safe archive
labels to exact source paths. This tool reuses the existing archive and disk checks.

```bash
PYTHONPATH=src python -m unittest experiments.bottleneck_diagnostic.tests.test_recovery
PYTHONPATH=src python -m experiments.bottleneck_diagnostic.recovery pack \
  --files <explicit-closed-files.json> --output <fresh-private.tar.gz> \
  --pod-id h447a4u6c0vq3k --assert-closed
PYTHONPATH=src python -m experiments.bottleneck_diagnostic.recovery receive \
  --archive <received-private.tar.gz> --sha256 <recorded-remote-archive-sha256> \
  --output analysis_exports/bottleneck_20261003/<fresh-received-directory>
```

Every member is verified before extraction and every extracted byte is rechecked.
Failed partial outputs remain recoverable; existing destinations are never replaced.
`--assert-closed` is a caller assertion, not process-completion proof. Its receipt
certifies bytes only: `experiment_verified=false`, `cleanup_authorized=false`.
Full experiment/score/ID/source validation and independent local review are still
required before the main agent can delete the owned Pod. Deploy recovery separately
from the running inference source; the five-file tool bundle includes its existing
ingest helper and package initializers, not another model environment.
