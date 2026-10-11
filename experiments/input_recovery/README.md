# Input-recovery experiment (separate 8K diagnostic protocol)

This experiment does not replace the historical V8 558/900 baseline. Its 48
questions are balanced diagnostic strata, not a random MMMU accuracy sample.
The professor submission `MMDL/` is never updated by these commands.

F0 receives I (image), T (O0 transcription), IT (both), IR (image plus subject
rules), and TR (transcription plus the same rules). Four existing selective
checkpoints receive I/T on a preselected 12-question subset. Maximum: 336 calls.
The 12-question smoke is retained when F0 continues to the complete 48-question
set; no completed question-condition pair is regenerated.

`protocol.yaml` fixes the pinned model/data revisions, P0, BF16, original high
resolution preprocessing, sampling, per-question seed, V8 and 8192 output tokens.
`runner.py` uses the existing vLLM continuous scheduler with an explicit 2/4/8
active-request bound; default deployment uses four. It never reduces image
resolution, precision, input context or output budget after a memory error.
Fresh F0 and derived checkpoint exports use the same inference conditions.

Transcriptions remain `AI_REVIEWED_NOT_HUMAN_VERIFIED`. They are not certified
ground truth. `rules.yaml` contains team-authored, source-linked, subject-wide
reminders fixed before new inference; these are non-exhaustive and not a
complete knowledge oracle. Failure in TR cannot by itself establish missing
knowledge, faulty reasoning or the unique responsibility of an internal module.
Old G6 outcome coverage is limited: the approved subset includes retained/lost
examples in group D, but no such claim is made for A/B/C.

## Prepare and execute

Prepare private inputs on the local workstation using
`inputs.prepare(source_root, new_output_root, historical_v8_predictions,
g6_predictions_paths={...})`. This preserves source/oracle hashes, image roles,
gold and the frozen selection. Publish only source code and configurations to
`jang2296/MMDL_test`. Transcriptions, original images, gold, checkpoint weights
and full responses must never be committed.

Clone the published exact SHA into an empty directory on an approved Pod. Use
the existing bootstrap with `--environment vllm`; it checks the exact clean
MMDL_test origin, pinned lock, dependencies, CPU tests, actual CUDA/BF16, storage
and pinned model hashes. Explicit financial/time watchdog arguments are required.

```bash
bash fine_tuning/runpod_setup.sh --environment vllm --commit "$COMMIT" \
  --runtime-root "$RUNTIME" --owned-pod-id "$RUNPOD_POD_ID" \
  --gpu-cutoff-utc "$GPU_CUTOFF" --billing-start-utc "$BILLING_START" \
  --balance-usd "$BALANCE" --reserve-usd 1 --hourly-usd "$ALL_IN_RATE" --execute
```

Transfer the prepared samples/oracles/selection/assets as private data, **without
`gold.jsonl`**. Transfer verified checkpoint directories as private data under
`$CHECKPOINTS/{FL,FM,FV,FML}`. No code overlay or manual Pod code patch is allowed.
Then execute using the bootstrap's Python and source path:

```bash
export PYTHONPATH="$PWD/src:$PWD"
"$RUNTIME/venv/bin/python" -m experiments.input_recovery.runpod \
  --runtime-root "$RUNTIME" --data-root "$PRIVATE_INPUTS" \
  --checkpoint-root "$CHECKPOINTS" --output "$CONTROL_OUTPUT" \
  --commit "$COMMIT" --owned-pod-id "$RUNPOD_POD_ID" --slots 4 \
  --gpu-cutoff-utc "$GPU_CUTOFF" --billing-start-utc "$BILLING_START" \
  --balance-usd "$BALANCE" --reserve-usd 1 --hourly-usd "$ALL_IN_RATE"
```

The checkpoint exporter restores every actual LoRA target and all four merger
groups, fuses in FP32 and creates a new BF16 Hugging Face model with hash-bound
source/output/tensor manifests. Original checkpoints remain untouched. Derived
BF16 inference is not claimed bit-identical to unmerged FP32-master inference.
The controller checks progress every 600 seconds and enforces absolute deadlines;
an expired budget stops requests, not Pod billing. Recover and independently
verify all closed evidence before deleting the exact owned Pod.
The complete bootstrap is one-shot: it will not overwrite an existing derived
export. An interrupted inference can resume directly through `runner infer`
with its original, unchanged manifest/deadline and verified exported model.

## Local scoring and interpretation

After verified recovery, score with the locally retained gold. The inference
program never reads gold; no external LLM/API judge is used.

```bash
python -m experiments.input_recovery.runner score --data-root "$LOCAL_INPUTS" \
  --runs "$RESULTS/F0" "$RESULTS/FL" "$RESULTS/FM" "$RESULTS/FV" "$RESULTS/FML" \
  --output "$NEW_SCORING_EXPORT"
```

Report paired recovery **and** regression (T-I, IT-I, IR-I, TR-T, IT-T), finish
reasons, NO_PARSE, tokens/time, checkpoint I/T transitions, family-grouped
uncertainty, oracle/rule limitations and missing cells. A missing/failed cell is
not an incorrect model answer; incomplete condition accuracy stays null.
Functional recovery is not unique module attribution or fine-tuning improvement.

Teacher-free supervision comparison is the next gated stage, not executed by
this controller: fresh F0, a fixed training scope, equal examples/order/updates,
answer-only versus source/program-verified visual facts plus answer, optionally
verified intermediate states. Benchmark-derived data is never allowed in training.
External development/retention evidence must justify expansion before final
fresh-F0/final-checkpoint MMMU900 evaluation under the same 8K protocol.
