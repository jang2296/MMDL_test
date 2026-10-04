# Selective fine-tuning experiment — MMDL_test

## Current deployment policy — 2026-10-04

The user authorized workspace experimental code, configuration, documentation and small
own-generated CPU fixtures in `jang2296/MMDL_test`. Professor-facing `MMDL/` and
`jang2296/MMDL` remain separate and unchanged. The private-overlay instructions below
are historical, not the current bootstrap. Use the root README's exact-commit clone
command and current `runpod_setup.sh`; no code overlay upload is needed. Weights,
checkpoints, HF/benchmark payloads and large responses stay outside GitHub. Frozen
receipts are not rewritten to manufacture approval. Only required own-data fixtures
and historical v18 source are bundled, not all saved results.

The latest A40 Pod was deleted by the user. Its full10K controller stopped before GPU
stages because the evaluator `--help` preflight exceeded 30 seconds. Prepared remote
data is not a completed training run; the entire final approval/data artifact was not
recovered locally. The help preflight now exits before cold runtime imports.
The new `python -m fine_tuning.runpod_g6 --help` entrypoint rebuilds the own-generated
pools, normally approves data for the current source, excludes smoke roots, and runs
the original full10K four-arm controller from F0. See the root README for exact
commit, owned Pod, financial deadline and recovery-reserve arguments. Installation
uses the same stdlib financial/time watchdog, capped at one hour; preparation
also has a watchdog. The inner GPU controller ends two minutes before that outer
deadline so its separately grouped training subprocess can close safely.
CPU verification is not cloud execution or completion of G6/G7/transfer.

### Historical private-overlay deployment (superseded)

This folder is a private overlay for the existing MMDL evaluation repository.
Do not copy it into the local public `MMDL/` checkout or publish it to GitHub
during this experiment. All generated data and receipts stay under this folder.
HF weights and dataset payloads are acquired only on the approved RunPod.

## Frozen experiment

- Base: `Qwen/Qwen3-VL-4B-Instruct` at
  `ebb281ec70b05090aa6165b016eac8ec08e71b17`, BF16, no quantization.
- Independent arms: FL (decoder attention/MLP LoRA), FM (all four mergers),
  FV (vision attention LoRA excluding mergers), FML (FM + FL).
  Every arm starts from fresh F0, not another arm's checkpoint.
- Trainable master parameters are FP32; forward computation uses BF16 autocast.
- Gates: independent 300-root prototype, separate 120-root/360-view training
  smoke, actual gradients/update/save/fresh-process reload, then common pilot.
- Pilot: 9,000 TRACE + 1,000 independent NVW records. This substitutes unavailable
  professional sources under plan §14.4; it is **not** the original multi-source
  professional mixture. NVW K/I/S counts are 500/250/250, independent of view type.
- Evaluation: newly executed F0 versus FT on the same frozen MMMU validation
  subset (two questions per subject, 60 total) and independent external development
  cohort. The new 4,096-output-token protocol cannot be compared directly with
  the historical 32,768-token 558/900 baseline.

`protocol.json` freezes execution settings. `mixture_policy_v1.json` freezes
the actual substituted pilot. `clean_spec.json` defines the independent generator.
The design document is the existing parent file
`Qwen3_VL_4B_Novel_Visual_World_Expanded_Plan_20261003.md.md`; its double suffix
is intentional in this workspace.

## Historical overlay entrypoints (superseded)

The historical private `runpod_setup.sh` pinned public commit
`342ceff1d0baaaa57aa63c644b9f64b007caa174`, the actual successful cloud checkout.
Its direct parent is the unchanged local public checkout `197838c…`; the only
commit difference is one report sentence, not executable code. The private
setup also pinned `qwen-vl-utils==0.0.14` and its `av==19.0.1` dependency to the
recovered successful environment. Historical setup receipts are not rewritten;
their original script remains in `runs/bootstrap/private_overlay_v1.tar.gz`.

The bootstrap's full CPU suite requires the own generated
`generated/graph_cs_bridge_split_pool32_v2/`,
`generated/graph_cs_bridge_development_cohort_v1/`, and recovered own-data proof
`runs/recovered_graph_pool32_child_approval_v2/proof/` in the private overlay.
A source-only archive is insufficient for those actual-artifact tests; include
these private fixtures, not HF payloads or the whole historical `runs/` tree.

After the fixed public checkout and private overlay are combined on RunPod:

```bash
python -m fine_tuning.prepare_sources --help
python -m fine_tuning.prepare_training --help
python -m fine_tuning.evaluate --help
python -m fine_tuning.run_experiment --help
python -m fine_tuning.run_control --help
python -m fine_tuning.run_bridge_smoke --help
python -m fine_tuning.run_bridge_experiment --help
python -m fine_tuning.ablation --help
python -m fine_tuning.graph_bridge --help
python -m fine_tuning.circuit_bridge --help
python -m fine_tuning.bridge_evaluation --help
python -m fine_tuning.professional_transfer --help
python -m fine_tuning.report --help
python -m unittest discover -s fine_tuning/tests
```

The controller requires explicit paths for pinned model, approved smoke and pilot
records/manifests, image roots, frozen MMMU/external cohorts, protocol and a new
output directory. It also requires the exact owned Pod ID, UTC GPU cutoff,
billing start, available financial budget, recovery reserve and all-in hourly
rate. `--baseline-mmmu` / `--baseline-external` reuse only closed, hash-verified
F0 results. Every FT result must match F0's actual input hashes, prompt token IDs,
image grids, processed sizes and per-question seed before comparison is accepted.
`--reuse-smoke-controller` reuses closed four-arm update/save/reload evidence;
pilot training still starts independently from F0. The default remains the full
10,000-record pilot. If the approved operating budget cannot cover it,
`prepare_training subset-pilot --parent <approved-10K> --output <new-directory>
--count <explicit-N> --reason <reason>` creates a separately identified common
subset without changing approved inputs or the parent. It checks the original
approval-component chain and retains exact source/track/view ratios. Pass its
paths and `--pilot-record-count <N>` explicitly to the controller. Such a run is
an additional smaller pilot, **not completion of the planned 10K comparison**.
Use `--exclude-smoke-records <approved-smoke-ledger>` when creating that subset
to exclude every G4 smoke root before selection; the controller independently
rejects smoke/pilot root overlap. Exclusion does not relax approval or quotas.
For the full 10K parent, `prepare_training disjoint-full-pilot --parent <closed-10K>
--nvw-root <independent-generated-pool> --smoke-records <approved-smoke-ledger>
--smoke-manifest <approved-smoke-manifest> --source-registry <rights-registry>
--benchmark-index <exclusion-index> --model-path <pinned-snapshot>
--output <new-directory>` replaces only overlapping own-NVW records. It preserves
all 9,000 approved TRACE records and the exact track/view quotas. New replacements
must pass the actual pinned CPU processor and normal contamination/rights approval;
quarantine is never waived. Generate new independent candidates locally if a
track/view has too few unused train roots; do not rename roots or borrow holdout data.

The recovered full10K approval binds the historical `train.py` hash
`68d5c7da4fd2c65a71bd5963beb8f93ece34b5251cb732c9c370fb22a8f9368c`.
The current child-contract trainer has a different hash and correctly refuses
that approval. For full10K, restore the complete frozen v13/v18 source tree in
a separate immutable RunPod source root, or obtain a fresh normal approval for
the current trainer. The matching old trainer is retained under
`runs/recovered_control1600_final_v1/code/nvw_v18_control/fine_tuning/`.
Do not edit approval hashes or overwrite current bridge code to bypass the guard.
The local full10K recovery contains metadata only; actual approved records/images
still require hash-checked reconstruction on RunPod, without local HF downloads.

`--reuse-pilot-controller <closed-controller>` reuses only a complete, hash-verified
pilot checkpoint with its original actual exposure and fresh-process reload.
`--reuse-pilot-evaluation-controller <closed-controller>` additionally reuses its
exact-checkpoint, closed paired MMMU/external results. Both references must match
the selected common pilot and inference protocol. Missing arms still train from
fresh F0. Reuse is recorded explicitly, not presented as new training or generation;
the budget forecast subtracts only stages whose completion was verified.
Run the saved-result report on RunPod while the original referenced evidence paths
are available, and recover that report together with the referenced closed artifacts.

If evaluation hits its operating-time budget, `evaluate run --resume-from <closed-result>`
can reuse only its verified contiguous completed prefix in a **new** output directory.
The original result is never overwritten, and parent hashes and reused/fresh counts
are retained. This does not increase the per-answer output token budget.
An interrupted F0 summary-save failure may be recovered with
`python -m fine_tuning.recover_evaluation --help`: original rows and damaged
summary remain unchanged, process closure and launch/model/input identities are
verified, and unknown memory measurements remain unknown. A recovered partial
result is not a completed baseline.
The controller records a live-process/GPU health observation every 600 seconds
and terminates its owned subprocess at the explicit financial/GPU deadline.

The report command reads saved, hash-verified results only. It separates MMMU
paired gains/losses from external QA, structure and state metrics; missing
training, reload or evaluation evidence is reported as incomplete, not a score.

## Data safety and evidence

MMMU/MMMU-Pro and derivatives are never training or generator sources. Their
complete pinned payloads are used only for exclusion fingerprints on RunPod.
Rights, provenance, family split, identifiability, actual processor input and
contamination checks precede approval; rejected records cannot fill quotas.
Fingerprint screening does not certify absence of every semantic/crop overlap
or original model pretraining contamination.

Solver and geometry checks plus limited AI image inspection are recorded as
such, not human or exhaustive visual certification. External development tests
measure registered-family new instances, not unseen DSL/renderer or a sealed
holdout. No source-effect or professional transfer success is claimed without
an appropriate observed comparison.

Use immutable versioned private source directories on RunPod. Never overwrite
Python files used by a running evaluation or training process. Results, input
audits, manifests, source hashes, module inventory, update/exposure journals,
checkpoint hashes, paired-input receipts and costs must be recovered and locally
verified before deleting the exact owned Pod. Dataset caches and base weights
are not recovered to the local workspace.

For the next NVW 0%/10% mixture control, run on Pod:
`python -m fine_tuning.prepare_controls --parent <reference-original-closed-10K>
--reference <frozen-10percent-subset> --output <new-control-directory>`.
It preserves the reference's common TRACE records exactly and replaces only its
NVW quota with distinct already approved TRACE records. Its exact original parent
and copied smoke-exclusion ledger are verified; a subsequently repaired parent
cannot replace that approval lineage. No new labels, downloads,
processor execution or GPU calls occur. The new manifest works with the existing
standalone training command, but is not the controller's four-arm 90/10 pilot.
Replacements require unused roots, not previously unseen training families;
all records remain in the already approved train split. The family separation
from development and holdout is unchanged, and new-family transfer is not claimed.
Fix one module setting before running both conditions; this preparation alone
does not measure a data effect. Record counts are equal, not necessarily token
exposure. It is not the fixed/randomized mapping or professional bridge A0–A3.

`python -m fine_tuning.run_control --help` runs the additional fixed-FL 0% control
against the already closed 10% FL reference. Supply the original parent,
reference/control/smoke directories, both closed reference controllers, frozen
cohorts/F0 results, and explicit ownership/financial arguments. Pass
`--smoke-image-root` separately: the smoke record/manifest directory is not
necessarily the generated-image directory. Approved image hashes are checked
at the supplied image root; files are not copied to disguise a path mismatch.
The previous
four-arm controller must be terminal and each GPU stage requires a successful
empty compute-process query; never launch while that controller is still live.
The control starts from fresh F0 and must complete 1,600 records/100 updates,
save/reload and both paired 60-item evaluations. Its budget forecast is checked
before training, and incomplete results are retained rather than called complete.
FL is fixed for this additional comparison because its closed reference is
available within the operating budget, not because it won the MMMU comparison.
Both conditions were not preregistered together. The report labels them trained
NVW10 reference and trained NVW0 control, reports token exposures separately,
and does not claim original 10K, G7/A0–A3 or professional-transfer completion.

Implementation and CPU checks are not evidence that pretrained GPU training or
the four-arm pilot has completed. `runs/` contains the actual stage receipts.

## Saved common-1600 pilot evidence

The [final four-arm report](runs/pilot1600_report_final_v1/report.md) is now
recovered and locally verified, with all four training/reload stages and eight
paired evaluations closed. MMMU60 correct counts are F0 32, FL 26 (gain 5/loss
11), FM 32 (5/5), FV 31 (4/5), and FML 26 (6/12). The common exposure was 1,600
records/100 updates per arm. No arm increased MMMU60 accuracy in this pilot.
FML's external structure F1 is 0.503441 on 11 observed outputs (one missing),
versus F0's 0.188985 on 12; grounded answer accuracy is 3/12 versus 0/12. These
partial structural changes do not establish real-source transfer or justify
automatic 50K expansion. The report keeps all external metrics/coverage separate.

The additional fresh-F0 FL NVW0/10 mixture control was started only after the
original controller closed, an empty GPU query and the remaining-budget check
passed. Its separate `experiment_v7_FL_NVW0_control1600` outputs do not replace
the four-arm report. Its training and fresh-process reload are now closed; the
[recovered checkpoint receipt](runs/recovered_control1600_training_v1/local_receipt.json)
verifies 1,600 records/100 updates and archived file hashes. All 504 adapter
tensor hashes also match the saved trainable-state metadata; this did not load
the base model locally. Both paired60 evaluations and the control report are now
closed and locally verified. The [final control report](runs/control1600_report_final_v1/report.md)
records MMMU60 F0 32, trained NVW10 FL26, trained NVW0 FL28. NVW10→NVW0 gained5
and lost3 answers, while F0→NVW0 gained6 and lost10. NVW0's external structure
and grounded results were lower than NVW10's; target-token exposures were not
matched. This additional, non-preregistered comparison does not establish a
pure NVW effect or support automatic50K expansion.

The immutable [partial report](runs/pilot1600_report_partial_v1/report.md) has
been generated on RunPod and checked against locally recovered closed results.
F0 scored 32/60; FL 26/60 (gain 5, loss 11), FM 32/60 (gain 5, loss 5), and
FV 31/60 (gain 4, loss 5). All four arms finished 1,600 records/100 updates and
fresh-process checkpoint reload. FML evaluation was still pending when this
snapshot was made; it is not the final four-arm report. External QA, structure
and state metrics remain separate in the report. The original common-10K,
professional bridge ablations, and real-source transfer are not completed by
this smaller pilot. Its final model choice and any expansion remain conditional
on measured transfer and retention, not merely a closed training process.

## Quarantined mapping/bridge prototype

`python -m fine_tuning.ablation --output <new-directory> --count 4 --seed 73041
--start-index 1000000` generates independent Boolean roots under fixed/randomized
rule-to-glyph mapping and absent/present canonical Boolean companion conditions
(A0–A3). All conditions retain the same semantic roots, splits and supervision;
bridge pairs also retain the same first image and user text. Record and image
counts are matched, not measured processor tokens or GPU time. `--verify` rebuilds
every artifact byte and rejects changes; it does not approve training.

Version 2 separates canonical operator names from connecting arrows without
changing version 1's glyph permutations. The rejected/unapproved v1 output stays
unchanged under `generated/boolean_ablation_prototype_v1`; v2 is separate.
The prototype is **not** the full G7 professional canonical bridge, non-synthetic
transfer, a model-input audit or a completed training experiment. Its review queue
and `training_allowed=false` remain until the normal independent-data approval
and actual processor checks pass. No benchmark-derived inputs are accepted.

## Quarantined Graph/CS representation bridge

`python -m fine_tuning.graph_bridge --output <new-directory> --count 4 --seed
73041 --start-index 1000000` generates an independent, bounded directed-graph
reachability bridge. The same nodes, initial state and rules have three forms:
novel SOURCE/TAIL versus TARGET/HEAD endpoint glyphs, standard directed arrows,
and the standard graph in a different layout. Fixed/randomized endpoint roles
and absent/present bridge form A0–A3. Bridge pairs preserve user text, targets,
first image and record/image-reference counts; measured token exposure is not
assumed equal. Labels and positions do not follow topological ordering.

The first prototype is preserved under `generated/graph_cs_bridge_prototype_v1`
but rejected for experimental use: all four QA answers could be obtained by
counting the initial node's undirected one-hop neighbors. Version 2 keeps the
randomization namespace and filters independent roots so that growth after the
first step is required and the directed answer differs from three explicit
shortcuts (one-hop undirected, bounded undirected and reversed propagation).
Its separate `generated/graph_cs_bridge_prototype_v2` is still quarantined.
These guards do not rule out every shortcut or selection bias.

`--verify` reconstructs all v2 artifact bytes. Historical v1 must be checked
against its stored source snapshot, not the current v2 generator. Both remain
`training_allowed=false` and `public_upload_allowed=false`. Four independent
7–8-node roots with at most two-edge paths are not full G7, broad graph reasoning,
non-synthetic professional transfer, or a completed model experiment. Actual
pinned processor and normal source/rights/contamination checks are still needed;
do not automatically change the existing training budgets to fit extra images.

The first actual pinned CPU processor audit passed all 64 single-image QA and
structure records across the Boolean/Graph prototypes but rejected each
multi-image group at the existing 1,048,576 episode-pixel cap. The original
training protocol and outputs are unchanged. `bridge_input_protocol_v1.json`
declares a separate prospective **CPU input check only** with a 3× episode cap;
per-image budgets, original image bytes and the 4,096 sequence cap stay fixed.
It is not approval for training or evidence of GPU memory fit or transfer.

That separate prospective processor check has now passed all 96 records across
the two prototypes and four arms, including every 2/3-image record. Sequence
lengths were 1,044–3,213 tokens without truncation. The recovered
[local receipt](runs/recovered_bridge_processor_prospective_v2/local_receipt.json)
binds image order/input fingerprints, source hashes and actual remote processor
receipts. The original cap failures remain preserved. Source/rights/contamination
approval and GPU fit/training are still pending; this CPU check does not grant them.

`generated/graph_cs_bridge_split_pool_v2` adds a separate, deterministic 12-root
candidate pool (8 train, 4 development; no sealed holdout). Its 144 matched-arm
records/144 PNGs passed byte reproduction and independent breadth-first state
checks. It is still unapproved, has not passed actual processor/GPU checks, and
does not establish generalization. Existing four-root prototypes remain unchanged.

`generated/graph_cs_bridge_split_pool32_v2` is a new root-disjoint candidate
with 32 roots: 28 train, 2 development, and 2 sealed-holdout roots (21/2/2
canonical families). Its 384 rows/PNGs passed byte reproduction, independent
breadth-first checks, paired-condition checks and a bounded independent review
with three native-image spot checks. Family ownership is not an access-control
seal or evidence of unseen-model generalization.

All 384 actual remote processor receipts were recovered and their source,
image-order, input and grid-hash bindings locally verified in
[the separate input receipt](runs/recovered_graph_pool32_processor_v1/local_receipt.json).
Sequences were 1,044–3,216 tokens under the same CPU-only prospective input
protocol. The producing SSH command returned 143 after saving all four PASS
receipts and the summary; a separate successful SSH verified and archived those
closed files. Its termination reason is unknown, so a successful producer exit
is not claimed. That historical CPU-only receipt cannot authorize training.

## Explicit canonical-bridge training input contract

`bridge_training_protocol_v1.json` is a separately frozen child of the unchanged
parent protocol. It keeps BF16, the pinned processor, 4,096 sequence tokens and
262,144–1,048,576 pixels per image; only the episode sum is 3×1,048,576 pixels.
Its immutable byte hash, a fresh actual processor receipt, source rights,
benchmark exclusion and train-only approval must agree before model/GPU loading.
Neither an old CPU-only receipt nor the child JSON alone approves a dataset.

The direct path is `audit.preflight_records(..., training_input_protocol=child)`
then `audit.approve(..., training_input_protocol=child)` and training with
`--input-protocol <child> --min-pixels 262144 --max-pixels 1048576
--episode-pixels 3145728 --sequence-budget 4096`. The original composite/control
preparation paths remain default1M only; they do not propagate child approval.
No automatic resolution, precision or context reduction is permitted.

`source_registry_bridge_v2.json` records the independent Graph/CS generator,
its parent and bundled font evidence without replacing the original registry.
The v20 actual check failed before its first record because the checker used
configured `size` instead of the effective min/max override pair. The v21
checker now follows the installed processor's precedence; limits were not changed.
Old sources, failed checks, candidate manifests and live control sources remain
immutable. The v21 fresh processor run passed all 384 rows; complete benchmark
exclusion and source checks approved only 84 train rows per arm (336 total),
with no quarantine and the same 28 train roots. Development/holdout rows are
excluded from training. The [local receipt](runs/recovered_graph_pool32_child_approval_v2/local_receipt.json)
verifies all 43 archived members, source/input hashes, image order, approved
row contents and the runtime contract. The only allowed row changes are the
training permission and added processor-proof hash. The current source passed
112 CPU tests, Ruff and a bounded four-file mypy check.

Data approval and GPU fit/update/reload are distinct gates. The A3-only bounded
GPU execution check below has completed; no bridge performance comparison has
run. Synthetic graph approval and this smoke are not fullG7 or non-synthetic
professional transfer. The old32 candidate manifest
remains unapproved; approval exists only in the separate train-only outputs.

`python -m fine_tuning.run_bridge_smoke --help` is a bounded A3-only FL
**GPU input/update/reload check**, not a performance experiment or fullG7.
It uses all84 approved train rows (28 roots; all three views), requires the
fresh v21 processor proof and explicit child contract, and starts from freshF0.
The closed FL checkpoint is used only as a module-inventory reference, never
as initialization. FL was chosen for its available closed runtime evidence,
not because it improved MMMU. No development or sealed roots enter training.
The original mixture control's1600-record gate remains unchanged.

Supply the previous controller directory and exact PID/birth. Both its matching
terminal journal and the process's disappearance are required; every stage also
requires an empty GPU compute-process query. The existing controller provides
10-minute health checks and ownership/UTC/financial limits. At least1050seconds
must remain for a maximum600-second train stage,300-second fresh-process reload
plus30seconds for process termination and120seconds for recovery. Each stage
rechecks that margin after hashing and the idle GPU probe. These are operating caps, not a measured bridge
runtime prediction. A partial epoch/OOM is retained, never rescued by lowering
resolution/precision/context or called complete. Complete84-row/6-update journals,
input/target/visual token exposure against actual CPU proofs, checkpoint bytes
and fresh-process state-hash reload are checked. The [closed A3 smoke receipt](runs/recovered_bridge_A3_FL_smoke_v1/local_receipt.json)
now verifies all84 train records/6 updates,140 images,188,779 input/142,100 visual/
44,271 target tokens, all504 saved adapter tensor hashes and fresh-process reload
on A40. Peak reserved memory was27,338,473,472 bytes; the same path's3090 fit is
not proven. This is an execution check, not A0–A3 performance or professional
transfer evidence.

After all closed results/checkpoints were recovered and locally verified, the
owned Pod was deleted on2026-10-04 at03:47:56 UTC. [Cleanup evidence](runs/bootstrap/owned_pod_cleanup_20261004_v1.json)
records API204→get404 and an empty Pod list. Its remote container/workspace/cache
are discarded; own checkpoints/results remain local. The read-only billing
snapshot is not a current balance or finalized invoice. Full common10K G6,
professional A0–A3 G7 and independent non-synthetic transfer remain unfinished;
no new Pod,50K training or900-question run is started by this cleanup.

## Published-chart source-transfer evaluation

`professional_transfer.py` adds a **private evaluation-only** path for actual
ChartQA human questions, separate from the synthetic external cohort. The
[source lock](chartqa_transfer_source_v1.json) pins HF revision, test Parquet
size/hash, provenance schema, attribution and rights limitations. Only metadata
has been checked locally; no ChartQA images, questions, labels or Parquet have
been downloaded here. Actual-source preparation and GPU evaluation have **not
run**. The implementation passed offline, independently authored fixtures;
that is not evidence of transfer or completion of G7.

On an approved RunPod, first provide a JSON file with a `datasets` list. Include
every completed checkpoint to be compared, even when several share the same
training records. Each entry needs `records`, `manifest`, `checkpoint_metadata`,
its receipt-verified `checkpoint_metadata_sha256`, `source_root` containing the
original frozen `train.py` and `modules.py`, and `image_root`. Paths refer to
actual restored/downloaded remote artifacts, not locally downloaded HF data.
The existing metadata and approved manifest must match the exact records,
original sources, image hashes, train-only lineage and completed exposure.
Reading this historical evidence does not approve new training with changed
code. Any ChartQA training ancestor blocks this source-holdout evaluation.

The builder downloads only the pinned released test shard on RunPod, filters
to provenance class0 (human), and excludes training question/pixel matches and
image dHash distances of at most3. It selects a seed3407 hash-ordered sample,
one question per exact RGB chart family, **before reading labels**. Invalid
selected gold or insufficient eligible families aborts rather than replacing
questions. This checks exact chart families and near-image fingerprints, not
publisher/document/semantic/crop duplicates or prior pretraining exposure.

```bash
# All paths below are supplied by the operator on RunPod; outputs must be new.
python -m fine_tuning.professional_transfer \
  --task-root "$FT_TASK_ROOT" --training-evidence "$FT_TRAINING_EVIDENCE" \
  --output "$FT_CHARTQA_COHORT" --count 60 --seed 3407
python -m fine_tuning.evaluate run-external \
  --model-path "$FT_MODEL_PATH" --cohort "$FT_CHARTQA_COHORT" \
  --protocol fine_tuning/protocol.json --output "$FT_CHARTQA_F0_OUTPUT" \
  --budget-seconds "$FT_EVAL_BUDGET_SECONDS"
python -m fine_tuning.evaluate run-external \
  --model-path "$FT_MODEL_PATH" --cohort "$FT_CHARTQA_COHORT" \
  --protocol fine_tuning/protocol.json --output "$FT_CHARTQA_FT_OUTPUT" \
  --checkpoint "$FT_CHECKPOINT" --budget-seconds "$FT_EVAL_BUDGET_SECONDS"
```

The same frozen images/questions/prompt and protocol serve fresh F0 and FT.
Before GPU/model loading, FT's actual checkpoint metadata hash must occur in
the cohort's training-exclusion evidence. Gold remains in a separate ledger,
never in the model message. Both runs retain raw responses, input/grid/token
hashes, seeds, finish reasons, output counts, elapsed time and memory. Existing
`report.load_evaluation`, `compare_inputs` and `external_groups` can validate
closed60-row results and calculate paired gains/losses; the four-arm pilot
report is not automatically relabeled as a ChartQA report. Run comparison on
RunPod before referenced data are discarded and recover the closed results.

The team metric is `chartqa_human_exact`: one complete submitted JSON object,
only `answer`, containing one string or number. Text uses trimmed casefolded
equality; numbers use finite **exact Decimal** comparison, preserving JSON
decimal precision. There is no5% tolerance, unit stripping, substring matching,
intermediate-candidate rescue or LLM judge. Invalid/ambiguous JSON is NO_PARSE;
missing budget-stopped results are not silently included as completed scores.
This is not the official ChartQA leaderboard metric and does not modify V8,
P0, historical558/900 or the MMMU evaluation protocol. Images/records and their
derivatives have `training_allowed=false` and `public_upload_allowed=false`.

Independent review found and checked fixes for unbound evaluation checkpoints
and JSON-float rounding false positives. At that checkpoint the private suite
passed129 CPU tests, Ruff and a two-file mypy check. Full common10K G6 training, the full
professional fixed/randomized A0–A3 bridge comparison, and **actual** real-source
transfer evaluation remain pending. The existing1600 results do not justify
50K expansion. Additional GPU execution requires a verified available balance
and bounded operating plan; a billing-history total is not a wallet balance.

## Common Graph/CS development probes

`bridge_evaluation.py` freezes a **single common evaluation cohort**, rather than
giving each trained condition its own probe images. It reads the unchanged
32-root own synthetic pool and all four recovered train-only approvals, checks
their row/image/processor-receipt identities, and hash-selects development roots
before reading selected targets. Every selected root retains all A0–A3 conditions
and QA/structure/grounded views. With the existing two development roots this is
24 records, **not 24 independent roots**. Train and sealed roots never fill a
missing development quota; this is not a sealed or real-source transfer test.

```bash
# Own generated images and approval records only; preparation needs no HF/GPU.
python -m fine_tuning.bridge_evaluation \
  --pool "$FT_GRAPH_POOL" --approvals "$FT_GRAPH_APPROVALS" \
  --output "$FT_GRAPH_COHORT" --count 2 --seed 3407
# On RunPod after budget/ownership checks, use the same cohort for F0 and FT.
python -m fine_tuning.evaluate run-external \
  --model-path "$FT_MODEL_PATH" --cohort "$FT_GRAPH_COHORT" \
  --protocol fine_tuning/protocol.json --output "$FT_GRAPH_F0_OUTPUT" \
  --budget-seconds "$FT_EVAL_BUDGET_SECONDS"
python -m fine_tuning.evaluate run-external \
  --model-path "$FT_MODEL_PATH" --cohort "$FT_GRAPH_COHORT" \
  --protocol fine_tuning/protocol.json --output "$FT_GRAPH_FT_OUTPUT" \
  --checkpoint "$FT_GRAPH_CHECKPOINT" --budget-seconds "$FT_EVAL_BUDGET_SECONDS"
```

Gold and root/condition/view membership are separate from generation records.
Before model loading, an FT checkpoint must bind the exact approved train
manifest/record hashes and complete exposure. The existing `paired_input_gate`
checks actual F0/FT inputs on this same cohort; `report.load_evaluation` remains
a 60-row gate and must not be used to certify this 24-row cohort.

The separate `graph_structure` policy accepts visible A–H node IDs in the actual
1200×1000 frame. It measures exact typed IDs/initial activity/directed topology
and leaf F1, ignoring array ordering. Valid boxes have their separate IoU≥0.5
metric, not a topology requirement. Renderer glyph indices, duplicate line
coordinates, printed-text transcription and private solver states are not part
of this structure target. The old integer-ID `structure` metric is unchanged;
its previous scores must not be compared directly with this new policy. QA uses
V8; grounded answer and supported states remain separate. Do not average these
three tasks into a MMMU accuracy or treat four views of a shared root as four
independent observations.

Preparation/CPU checks are not GPU evidence. Full common10K training, actual
fixed-module A0–A3 training/performance comparison and professional
non-synthetic transfer remain unfinished. This path does not choose a winning
module, launch a Pod, change V8/558-of-900, or justify 50K expansion.

That cohort/scorer stage passed138 CPU tests plus Ruff and a two-file mypy check.
The actual own-data cohort is `generated/graph_cs_bridge_development_cohort_v1`:
24 records/two roots/two families, eight rows per task and16 stored PNGs. Its
input/gold/condition-ledger hashes, all image bytes, reference-target scorer
identity checks and the existing closed A3 checkpoint's exact training-exclusion
binding are verified. Reference-target checks are not model-generated answers
or measured accuracy. No GPU evaluation of this cohort has run.

## Bounded fixed-module graph comparison controller

`python -m fine_tuning.run_bridge_experiment --help` connects all four approved
84-record/28-root conditions to the existing trainer, fresh-process reload and
paired evaluator. Explicitly fix one `--arm FL|FM|FV|FML`; every fresh condition
starts from F0. This is an **additional bounded Graph/CS comparison**, not the
full professional G7 or a proven winning module after the full10K G6.

Supply the actual approval/image roots, child training input protocol, unchanged
parent inference protocol, common graph24 and pinned MMMU validation60 cohorts,
closed inventory reference and previous controller/process identity. The owned
Pod ID, current balance, recovery reserve, hourly rate, UTC cutoff and stage caps
are mandatory operational checks. The existing600-second health polling,
hard subprocess deadline and idle-GPU guard are reused; no fallback quantization,
lower resolution or extra output tokens are added. Local `--help` and CPU tests
do not rent a Pod, acquire HF data or run model inference.

`--reuse-a3-controller` accepts only the original completed A3-FL checkpoint with
matching actual inputs/processor token exposure, full6-update epoch, exact module
inventory/payloads and closed train/fresh-reload proof. It does not initialize
A0/A1/A2 and is rejected for another module. New/extra load-trigger files are
rejected before stages. Historical sources, approvals and checkpoints are not
rewritten to satisfy reuse. Existing graph inputs and saved A3 weights are not
re-created under a new name.

The MMMU F0 baseline is fresh by default. `--baseline-mmmu` reuse additionally
requires `--baseline-evaluator-source` with the exact historical hash, identical
generation-function ASTs and current CPU score replay. A resumed baseline also
needs `--baseline-parent-evaluator-source` for its completed prefix. Missing
historical code or a changed prompt helper is not certified as equivalent; leave
reuse options unset and execute a separately identified F0 baseline on RunPod.
The existing32/60 result remains an unchanged historical artifact.

All checkpoints receive **the same entire24-record graph cohort and60-record
MMMU cohort**, never only their condition's probes. Required observed input
hashes, token IDs, image grids/sizes and seeds are validated before comparing
results. Reports retain F0/FT gains and losses, A1−A0/A2−A0/A3−A2/A3−A1 contrasts,
task/condition/root/family groups and actual training exposures. Shared roots are
not independent row replications. `f0`/`ft` fields in pairwise contrasts denote
the earlier/later side named by the contrast, not a claim that A0 is frozen F0.
Partial checkpoints/responses and completed-condition receipts remain on a
budget stop; no incomplete score is presented as the full comparison.

This path has no new GPU results yet. Full common10K G6, professional G7 and
independent real-source transfer remain required before completion/expansion.

The private suite passed146 CPU tests (273.989 seconds), Ruff, a three-file
mypy check and CLI help with the current controller sources. The eight new
tests exercise approval pairing, all module-group checkpoint layouts, score
replay and malformed evidence, historical-source/prefix rejection, shared-root
grouping, overlapping-process rejection, pre-GPU budget stop and a mocked
18-stage workflow. Mocked stages are not measured training or model accuracy.
The original closed A3 checkpoint also passed a separate actual-artifact CPU
reuse check against its input, update, processor, saved payload and reload proof.
See [the controller validation receipt](runs/bootstrap/private_bridge_controller_validation_v1.json).
These checks do not certify the unavailable historical MMMU baseline's source
equivalence or a complete actual MMMU-cohort gate; use a separately identified
fresh F0 evaluation if the required original-source proof is unavailable.

## Quarantined professional Circuit bridge

`circuit_bridge.py` and `circuit_clean_spec_v1.json` add independent ideal DC
circuit semantics required by plan §11.1–11.2: nodes, resistors, component
values/units, source polarity and an open/closed switch. The circuit is a series
resistor feeding two return branches, with the second branch switched. Exact
rational nodal KCL agrees with a separately implemented series/parallel
reduction; no tolerance, unit guessing or rounding is used.

```bash
python -m fine_tuning.circuit_bridge \
  --output fine_tuning/generated/circuit_bridge_prototype_v2 \
  --count 4 --seed 73041 --start-index 1000000
python -m fine_tuning.circuit_bridge \
  --output fine_tuning/generated/circuit_bridge_prototype_v2 --verify
```

Generation refuses existing outputs. The actual four-root private prototype
contains48 rows/48 PNGs: each A0–A3 condition has12 rows and20 image references.
QA and canonical-structure views use the first novel image; the grounded view
uses all three slots. A0/A1 repeat the novel image, whereas A2/A3 provide novel,
standard and relaid-out standard representations of the **same** circuit.
Bridge pairs preserve user text, targets, first-image bytes and image counts.
All conditions retain the same roots, switch state, values, question/options and
family split; voltage/render variants remain in their canonical family.

The first `generated/circuit_bridge_prototype_v1` is preserved but **rejected**:
its distractors made the correct answer the second-smallest option, which an
options-only heuristic recovered on1000/1000 independent procedural roots.
The [rejection receipt](runs/bootstrap/circuit_bridge_v1_rejection.json) keeps
this failure even though its original unit checks passed. Version2 preserves
world/glyph randomization but repairs choice construction with varied sorted
answer ranks and alternative circuit reductions. The regression checks this
known shortcut, not every possible shortcut or selection bias.

Every artifact remains `training_allowed=false`, `public_upload_allowed=false`
and unapproved. Solver and program geometry/reproduction checks plus limited AI
native-image spot checks do **not** certify independent render review, source
rights, contamination screening or actual pinned processor inputs. Values and
units can also identify component roles; a pure mapping-only effect is not
claimed. This one DC topology is not broad Electronics, full professional G7,
non-synthetic transfer, or a completed training experiment. It is not silently
inserted into the existing approved10K mixture or Graph/CS-only controller.
HF payloads, model weights, GPU execution and publication are not needed for
this preparation. Full10K training and real-source evaluation still require an
approved, financially bounded RunPod execution.
