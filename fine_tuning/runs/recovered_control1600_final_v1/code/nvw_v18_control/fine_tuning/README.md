# Private selective fine-tuning experiment

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

## Entrypoints

After the fixed public checkout and private overlay are combined on RunPod:

```bash
python -m fine_tuning.prepare_sources --help
python -m fine_tuning.prepare_training --help
python -m fine_tuning.evaluate --help
python -m fine_tuning.run_experiment --help
python -m fine_tuning.run_control --help
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
