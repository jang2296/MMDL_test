# Error-notebook self-correction RL

Implements the execution core of the **v2** contract in the private design document
`claudedocs/research_error_notebook_self_correction_rl_plan_20261016.md`.
This is not the earlier episode-reward/G4/TRL proposal. The implementation keeps
same-state turn groups (T1:8, each T2:4), program-verifiable notebook rewards,
and independently verified visual-error routing. No additional SFT or teacher labels.
The complete data approval / GPU preflight / Phase-A workflow is **not yet complete**.

## Current state

Implemented and CPU-tested: strict wire records; actor/verifier separation through
a fresh JSON-only subprocess; deterministic generators and source adapters; candidate
build/selection; exclusion scanner; literal-token retry trees; class-balanced diagnosis
groups; actual tiny-Qwen LoRA/Merger gradients; fixed-reference RL loss; per-problem
Vision gradient routing; checkpoint/reload/export; and a bounded training loop.

`trainer.run_loop` saves raw trees (including rejected problems), performs one update,
pushes weights and checks two synchronization probes before the next batch. The run
seed affects both problem order and actual completion seeds. It preserves health
history and the consumed-ID cursor on resume. Failed synchronization is not a completed
step and its diagnostic checkpoint is blocked from automatic continuation. A grounding
decrease on two matched probes disables Vision updates, not the other partitions.

The tiny-model loop test performs three real optimizer updates and checks resumed
next-step weights against uninterrupted execution. Its controlled action/reward and
sampler-sync fixtures are **not** pretrained 4B inference, learned self-correction, a
vLLM GPU certificate, or a model-improvement result. Separate actual vLLM GPU gates
remain necessary.

No pretrained 4B RL run, approved training pool, or performance improvement has been
completed. Existing V8 results and all files under `MMDL/` remain untouched. Only
this package and `fine_tuning/tests/test_notebook_rl_*.py` are implementation targets.

`monitor.measure_dev` measures 96 dev items with separate plain-answer and observation
requests (192 independent n=1 calls). It uses the fixed official sampler, compares
identical inputs/seeds against F0, and stores raw responses before scoring. A stopped
or failed measurement has null health metrics. Its dev length measurement is **not**
the Phase-A training-length reference. `validate_f0_reference` checks the closed
raw/input/score ledger before the launcher allocates a model, including short runs
that finish before the first scheduled dev check.

`approval.validate_for_measurement` checks the entire candidate pool before any
Phase-A result or selection exists. It retains full exclusion/index/source checks,
split separation, D1–D8 and real human-review requirements. One quarantined train
candidate blocks the pool; it is not silently omitted. Success has a distinct
`VALIDATED_FOR_MEASUREMENT` status and **`training_allowed=false`**. It only verifies
data eligibility, not G0–G13, actual F0 weights, spending authority or completed
measurements. It cannot be passed to the training launcher as approval.

```bash
python -m fine_tuning.notebook_rl.cli check-measurement-data \
  --job /absolute/private/measurement-data-job.json \
  --out /absolute/private/measurement-data-check.json
```

This CPU command reads a `notebook-rl-measurement-data-job-v1` job with `build_root`,
path/SHA descriptors for `index`, `source_registry`, `evidence`, `data_report`, and
an `expected_measurement_scope`. Private payloads are opened only in the trusted
worker, not the actor. The measurement data scope binds protocol, current source,
all input files and the data-proof producer; it contains no future selection or GPU
claim. These data proofs can be reused by final training only with **unchanged**
bindings. Changing protocol bytes—including DESIGNED→FROZEN or token caps—requires
fresh evidence; this implementation does not silently re-label old proofs.

`phase_a.measure_candidates` is a measurement core, not a GPU launcher. It records
actual T1 n=8 and gray-image n=4 responses for all 576 train candidates, then measures
retries for a deterministic 128-item subset. It preserves raw returns before scoring,
uses the isolated verifier and literal previous-response tokens, and records budget
stops/errors without fabricated metrics. The 8,192-token measurement reservation is
separate from the final training caps. It measures one deterministic failed T3
parent per measured T2 group (measurement only, not a change to the training
branch rule). Each turn has its own observed complete-final p95/cap; no turn
borrows another turn's cap. N/A, missing finals and context-blocked branches remain
explicit, with conditional rates separate from whole-measurement rates. Attribution
is derived from actual observation/verifier records, not asserted module causality.
A completed core run is explicitly `MEASUREMENTS_COMPLETED_NOT_PHASE_A_APPROVAL`:
oracle/held-out measurements and the final Phase-A assessment are not supplied by
this core. Controlled-engine tests are not evidence that pretrained Qwen or a GPU
produced these responses.

`measure` connects this fixed-F0 core to data eligibility and the existing owned
runtime/watchdog checks. It loads only the original vLLM model, never an optimizer
or adapter. Its memory floor accounts for one weight copy plus the unchanged
10-GiB KV cache; the training launch retains two copies. This floor is not a peak
memory/throughput certificate. No precision, resolution, context, cohort size or
measurement cap is automatically reduced.

```bash
python -m fine_tuning.notebook_rl.measure --job /absolute/private/measurement-job.json --check
# Only under the live owned financial watchdog in the exact-SHA RunPod clone:
python -m fine_tuning.notebook_rl.measure --job /absolute/private/measurement-job.json --execute
```

The `notebook-rl-candidate-measurement-job-v1` JSON contains `build_root`, descriptors
for `index/source_registry/evidence/data_report`, `expected_measurement_scope`,
`runtime_scope`, `model_path`, `seed`, `output`, `deadline_utc`, `commit` and
`forbidden_roots`. Runtime scope contains purpose `FROZEN_WEIGHT_MEASUREMENT_ONLY`,
owned `runtime_id`, pinned `environment_lock_sha256`, and exact GPU identity
(`type/uuid/name/index/total_memory_bytes/driver_version`). It is not a GPU-gate
report. Data eligibility is rechecked before allocation. Launch, candidate raw
records and the final receipt are private/write-once. Failures retain their state;
the owned engine is shut down in `finally`. Engine shutdown is not Pod deletion.

`launch` connects the approval worker, actual runtime checks, fresh/resumed policy,
isolated verifier, synchronization probes, monitor and training loop. These commands
require a real private job file; a test fixture is not a training approval:

```bash
python -m fine_tuning.notebook_rl.launch --job /absolute/private/job.json --check
# Only in the approved exact-SHA RunPod clone, below the existing financial watchdog:
python -m fine_tuning.notebook_rl.launch --job /absolute/private/job.json --execute
```

The job binds the build, selection, Phase-A report, model snapshot, protocol arm,
algorithm, seed, commit, output, resume state and deadline. Index/source/evidence,
data/GPU gate reports, F0 report and sync probes use explicit path+SHA descriptors.
`--check` does not rent a Pod or certify GPU execution. `--execute` requires the
owned live watchdog, pinned installed lock/model files and the actually measured
GPU scope. It neither manufactures missing evidence nor silently reduces settings.

Before allocating the rollout engine, `gpu_gates.measure_inventory` checks the
instantiated policy: all four original/master Merger pairs, actual Vision/text
target modules, adapter settings, tensor identities/shapes/dtypes, route-specific
gradients and frozen-base hashes. It writes `policy_inventory.json`; declared
counts alone cannot pass. Once allocated, the owned rollout engine is shut down
on normal completion and on preparation, synchronization or training failure.
This is not Pod deletion or a complete G0–G13 certificate.

`gate_session` connects the existing G0–G13 measurements without granting training
approval. Its CPU parent owns five sequential children (capture, replay, CPU export,
default-cache control, fresh-engine control) and retains actual process handles.
Capture checks inventory, independent F0 reference and gradients before allocating
the rollout engine. It then checks input parity, 32 short logprob trajectories,
token transport and the pre-update cache state. It performs update0→1 once for the shared gates,
saves a diagnostic step1 checkpoint, then collects new version1 actions with fresh
images after measuring weight parity, cache state, another 32 short trajectories and
the post-sync independent F0 reference. G12 saves step1 with the next batch/RNG before executing update1→2. The
second child reloads that saved step1 and replays the identical update1→2. Policy
and reference logits, next loss and optimizer state must match in the existing
native comparator. A CPU-only child exports the saved first-update checkpoint to
BF16; the remaining two children compare a deliberately enabled default-cache
control with an untouched fresh engine from that export. The mutated live step2
model is never exported as step1. A Merger update rounded away in BF16 fails the
cache measurement; the runner does not increase learning rates to force a pass.

```bash
python -m fine_tuning.notebook_rl.gate_session --job /absolute/private/gate-job.json --check
# Separately authorized bounded gate updates, under the existing owned watchdog:
python -m fine_tuning.notebook_rl.gate_session --job /absolute/private/gate-job.json --execute
```

The job is `notebook-rl-gate-session-job-v2`: the candidate measurement job fields
plus `arm`, path/SHA `sync_probes` and `probe_scope` (`probes`: ordered id/image_count/
input_sha256 records). Runtime purpose must be `BOUNDED_RL_GATE_MEASUREMENT_ONLY`,
not frozen-weight-only authority. Both probes must be exact eligible actors, one
with one image and one with two, each with 2–32 literal completion tokens. Algorithm is fixed to `ALG-NB3`; there is no cap,
subset or additional-step override. Existing D1–D8/human/exclusion requirements,
clean checkout/lock/owned watchdog and two-weight-copy+10GiB KV floor are retained.
The frozen-weight data validator supplies eligibility only, not update authority.

This command combines observed components with the existing scalar validation rules;
it does not create GPU approval receipts, complete Phase A or certify pretrained-model
improvement. G13 observes only shared0→1; G12 capture/replay use the production RL
update outside that patch context. Gate checkpoints are not full training-resume
cursors. Missing actual two-image/T3 coverage, update/synchronization failures or
timeouts remain failures with private records. CPU mock/tiny-model tests cannot
be promoted to native CUDA success. Engine/process cleanup does not delete a Pod;
results and checkpoints must still be recovered before owned-Pod deletion.
The post-sync F0 check temporarily needs three model copies plus the fixed KV cache
and training state; the two-copy preflight floor is not proof that this peak fits.

The separate G10 numerical component loads no model or training data:

```bash
python -m fine_tuning.notebook_rl.gpu_gates --numeric-check --device cuda:0 \
  --out /absolute/private/g10-numerical-probe.json
```

It runs 60 FP32/BF16 value and gradient checks on the requested device, including
independent scalar calculations, masked/weighted microbatch equivalence, and a
narrow comparison with pinned TRL DR-GRPO formulas. The TRL trainer itself is not
executed; project-specific MIS and group weights are checked separately. CPU is
available explicitly as `--device cpu`; unavailable CUDA never falls back to CPU.
The write-once result records actual device, tolerances and source hashes, keeps
`training_allowed=false` and `gpu_gates_certified=false`, and retains failure or
interruption status. Passing this synthetic check is not pretrained-4B execution,
memory fitness, learned self-correction, or permission to train.

Two additional measurement components operate on an **already-owned** native
vLLM 0.11.0 engine; they do not allocate a model, rent a GPU or issue approvals:

- `gate_images.measure_image_parity(engine, probes)` temporarily observes actual
  worker visual forwards for the fixed single-/two-image probes. It checks BF16
  pixel identities, grids and final/DeepStack feature counts against HF preparation.
  It checks exact image identity and order across encoder calls and removes its hooks even on
  failure. Matching image identities/counts does not certify equal HF/native
  feature values or downstream image-to-token placement.
- `gate_weights.measure_weight_parity(bundle, engine)` streams merged HF hashes
  and reads the complete worker tensor inventory back, accounting for packed text
  QKV/gate-up tensors and the tied embedding/head alias. All 24 tensors in the four
  Merger modules and the frozen HF base are checked. A successful load marker is
  required; a newly constructed F0 engine is not reported as a completed push.

These paths have authored CPU boundary tests, not completed native 4B GPU runs.
They are components for the remaining G0–G13 runner, not a complete gate report.
Full readback is intended for gates, not an added full-model CPU copy every step.

Additional components reuse the production policy/update paths:

- `gate_reference.measure_reference` compares all vocabulary logits at completion
  prediction positions with a separate fresh F0, separately for INITIAL and
  POST_SYNC. It restores original RoPE tensor identities/values, gradient flags,
  modes and RNG even on failure. Unexpected forward-created gradients fail the
  measurement; pre-existing gradients are rejected without erasing them.
- `gate_update.measure_update` performs **one real RL optimizer step**, records all
  24 FP32 Merger masters and AdamW counters before/after, and checks frozen-base
  hashes and parameter probes. It deliberately leaves the sampler unsynchronized.
  A failed post-update check does not roll back weights or optimizer state.
- `gate_gradients.measure_gradient_routes` measures OFF, direct hard-freeze, ON
  and mixed gradients. FML has no installed Vision adapter: ON/mixed results are
  explicit N/A, not fabricated successes.
- `gate_logprobs.measure_logprobs` journals engine-validated T=1 sampler returns and compares
  generated-action probabilities with the HF policy for at least 32 trajectories,
  including single- and two-image requests. INITIAL and POST_SYNC are separate
  measurements. It reports token and sequence mismatch distributions and a proposed
  initial MIS threshold; it never silently edits the protocol. Synthetic arithmetic
  checks and real tiny-HF CPU generations do not certify native vLLM behavior.
  Failures inside the engine's return validation retain an error journal, not a
  claim that the unavailable malformed backend payload was preserved.
- `gate_tokens.measure_tokens` issues two unscored turns for each fixed
  single-/two-image probe. It checks native-returned prompt IDs, literal prior
  actions, the conversation-end insertion rule, unchanged image tensors and the
  production completion-only action selector. Prior replies and feedback remain
  outside the next action mask; a generated token is not dropped just because its
  ID equals the pad token. The actual EOS branch is recorded. Tiny-HF CPU tests
  and authored opposite-branch tests do not certify native 4B execution.
- `gate_cache` compares the post-update normal engine with a fresh engine loaded
  from the same merged tensors. A separate disposable engine deliberately retains
  vLLM's default caches across a weight load: actual prefix-hit counts and encoder
  cache hits/absence of visual recomputation are recorded independently. Fixed-action
  prompt logprobs bypass prefix caching in vLLM 0.11, so a separate one-token
  generated-action probe exercises prefix reuse. Fresh-engine agreement uses the
  maximum token logprob difference, not a mean that can hide individual errors.
  A default-cache mismatch is measured, not manufactured or required to occur.
  The coordinator must retain three actual owned process handles and sequence their
  exits. Disposable engines check predecessor exit before allocation and reject a
  CUDA-owning coordinator; constructor failures still require the outer process
  watchdog. Complete pre/post tensor readback binds the comparison, while snapshot
  provenance and the real G4 optimizer-update evidence remain separate requirements.
- `gate_checkpoint` saves the actual policy/optimizer/scheduler/RNG/cursor, captures
  policy and reference full-vocabulary logits at two bounded completion probes,
  then performs the next production RL update. A separate process reloads the saved
  state and repeats that same update. The model-free coordinator compares raw
  logits, next loss, all parameter/optimizer states, inputs and cursor, retaining
  failed comparisons and the original checkpoint. The tiny-Qwen subprocess test
  passes with SFT/CE entry points forbidden; it is not a pretrained-4B CUDA result.
  Each probe covers 1–32 completion positions, not the entire prompt vocabulary.
- `gate_inputs` connects G12 capture/replay to one private write-once input packet.
  It preserves actual version-1 trees, rewards, sampled-action token IDs, seeds,
  ordered source-image hashes and decoded pixel fingerprints. Message-based sync
  probes are prepared once and saved as literal prompt IDs, without changing the
  caller's request. Both capture and replay consume this same packet; a capture
  receipt binds it to the completed result. Replay reconstructs images before
  loading checkpoint RNG and sets the new preparer's policy version to the saved
  step. Changed bytes, pixels, order, action versions or capture bindings fail.
  The tiny-Qwen two-process test uses this handoff and compares all logits, next
  RL loss and updated state. These are authored CPU actions, not real pretrained
  4B responses. The packet is neither data approval nor a `trainer.run_loop`
  resume cursor; the complete owned-process gate runner is still required.
- `gate_throughput.measure_step` chooses the longest initial input at maximum image
  count from a frozen gate cohort before observing rewards. It requires an actual
  third-attempt action, collects a complete batch, reuses the single G4 RL update,
  then pushes weights and runs synchronization probes. No successful answer is
  forced to retry just to pass a gate. Missing coverage or an MIS-masked stress
  action remains INCOMPLETE. Actual native KV/concurrency/context/cache settings,
  original images and processed tensors are checked, including retries and both
  sampler/trainer preparation immediately before update. CUDA-finished stage
  times and sampled device memory are recorded separately from allocator peaks.
  This measures the selected cohort's observed workload, not a theoretical full-cap
  worst case; CPU fixtures produce no GPU memory certificate.
- `gate_guards.measure_guards` wraps that same G4/G11 update in a fresh owned
  process. The isolated worker rechecks data eligibility before and after, while
  actual batch IDs/images and initial prompts are bound to the exclusion allowlist.
  CE/SFT entry points and teacher/API calls are blocked, including caught attempts;
  parameter changes and one actual optimizer step must be observed. It does not
  generate human-review declarations or loosen partial-index rejection. Guards
  remain installed after exit. This component observes the RL update, not sampler
  synchronization, and does not authorize spending or subsequent training.

The caller must bind these components to the complete gate sequence, private
write-once journals and the actually measured runtime. Their existence does not
complete G0–G13, Phase A, data clearance, or a pretrained-4B training run.

Both GPU entry points use a narrow trusted dependency exception: pinned vLLM 0.11.0
imports OpenAI SDK 3.16.2 schema types internally. Before vLLM construction, startup
rejects any preloaded teacher SDK and disables all concrete OpenAI/Azure/Bedrock
client constructors plus the common sync/async request methods. Schema imports
are then allowed; external teacher/API use is not. The default verifier/data
workers still reject SDK imports entirely. This is Python defense in depth, not
an OS/network sandbox. Offline checks with the actual cached SDK verified type
imports and rejected client construction; they do not certify vLLM/GPU execution.

Remaining prerequisites are concrete, not completed by passing CPU fixtures:

| Prerequisite | Current limitation |
|---|---|
| Full benchmark exclusion | The scanner is implemented; full pinned input/index, OCR, embeddings and calibration evidence are not available. |
| Human render/leak review | D3/D7 require real reviewers. No generated or AI-reviewed sample is labelled human-approved. |
| Actual GPU gates | G0–G13 evidence is validated by the approval worker; the complete GPU evidence-producing runner and its execution are not complete. |
| Actual Phase A | The candidate/retry measurement core is implemented; complete provenance-bound assessment, held-out baselines, cap freezing and actual F0 GPU execution remain unfinished. Its partial measurement format is deliberately rejected by selection. |
| Paid deployment | No completed clean-clone GPU run, 4B RL checkpoint, or 150-question improvement result exists for this implementation. |

Thus the current launcher is an integrated **guarded entry point**, not a claim that
the whole research pipeline can already be reproduced end-to-end with one command.

```bash
PYTHONPATH="$PWD/src:$PWD" /home/jang2/mmdl-env/bin/python -m unittest discover \
  -s fine_tuning/tests -p 'test_notebook_rl_*.py' -v
```

The source-generating environment is separate and uses `requirements-data.lock`.
For a small **unapproved** candidate build (all output stays outside the checkout):

```bash
export NOTEBOOK_BUILD_ROOT=/absolute/private/notebook-candidates
for split in train dev sealed style_sealed; do
  python -m fine_tuning.notebook_rl.cli generate --generator graph \
    --split "$split" --count 2 --seed 3407 --out "$NOTEBOOK_BUILD_ROOT/$split"
done
python -m fine_tuning.notebook_rl.cli build \
  --inputs "$NOTEBOOK_BUILD_ROOT/train" "$NOTEBOOK_BUILD_ROOT/dev" \
    "$NOTEBOOK_BUILD_ROOT/sealed" "$NOTEBOOK_BUILD_ROOT/style_sealed" \
  --out "$NOTEBOOK_BUILD_ROOT/build"
```

That command was exercised using the actual graph generator: eight images/records,
four disjoint splits, still `PREPARED_PENDING_GATES`. `audit` and `select` expose the
existing scanner and Phase-A selection functions; neither changes human-review or
training permission flags. Outputs are write-once. `--help` needs no Torch import
in the data environment. Chemistry requires a separately pinned real ChEBI source;
its authored CPU fixtures do not constitute source acquisition.

Data/rollouts/weights go outside the checkout. Source entries are candidates, not
approvals. Human render/leak review and the complete benchmark exclusion gates
must be recorded before training; AI review is not human certification.

The previous 100-file/16,740-row exclusion index is incomplete for the v2 protocol.
Full pinned split coverage, per-image provenance, OCR, SSCD/SigLIP and calibration
evidence must be supplied. Partial scans return `BLOCKED_NOT_CHECKED`, never a clean
training allowlist. Python file/import guards add defense in depth; they are not an
OS sandbox against native code or another process with the same user permissions.

The user's $2.30 budget prioritizes actual bounded RL updates and fresh checkpoint
reload. A full pilot or 150-question evaluation is not promised within that budget.
The design's 48GB two-engine configuration must not silently be shrunk to fit 24GB.
GPU choice and human review are pending; CPU implementation continues independently.
The outer watchdog must interrupt in-flight GPU work; loop-level time checks alone
cannot enforce a wall-clock limit. Saving or stopping a process does not delete a Pod.
