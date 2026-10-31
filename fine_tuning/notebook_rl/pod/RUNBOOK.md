# RunPod runbook for notebook-RL (A40 Secure + network volume)

**Nothing in this runbook or in `pod/*` has been executed on RunPod.** The scripts were written and
tested only on CPU fixtures (fake mount, local git repo, fake python, a stdlib stand-in for
`pool_freeze` inside the fixture clone). One test also runs the real `pool_freeze.verify` on the local
frozen vault. No RunPod API, MCP, network download, upload or GPU was used. Every `[USER]` step is done
by the user in the RunPod console/CLI; the scripts never call RunPod. Cost figures below come from
`analysis_exports/runpod_data_speed_plan_20261030/engineering.md` and are **JUDGMENT** estimates
(about +-50%) until `pod/cost.py` is re-run on a measured receipt.

Rules this follows (CLAUDE.md): §2.6 (explicit approval per stage, watchdog, no auto-restart, no
fallback, cleanup order, only owned resources), §2.2 (no benchmark data in training inputs), §2.4 (code
only under `fine_tuning/`), §5.2 (Pod runs a clean pinned MMDL_test clone, never an uploaded overlay).

Layout (`/workspace` is the network volume; `/home/...` is the Pod's container disk):

```
/workspace/mmdl-volume/            created once by volume.sh (volume.sh never overwrites or deletes it;
                                   the user deletes the volume in step 4 at the very end)
    volume_manifest.json           status VOLUME_READY, lock sha256, pip freeze sha256, model sha256 list
    venv-<lock sha256[:16]>/       exact-lock venv (environment experiments | vllm | proof)
    hf-cache/hub/.../snapshots/<rev>/   pinned Qwen3-VL-4B-Instruct files
    budget_result.json             watchdog result of the volume.sh run
    eval-data/                     only with `eval_data: true` (GE); every MMMU byte on the volume, so the
                                   training guards forbid the whole tree (symlink in 2b)
        mmmu/, hf-cache/           pinned MMMU validation and its own HF cache
        artifacts/runs/<run id>/   the evaluator's run dirs of the eval stages (raw responses: private)
        proof/                     prereg.json + .sha256, ledger.jsonl (one ledger for all eval stages)
    ft-export/                     eval-ft only: model/ (merged BF16 weights, never collected) + model_ref.json
/workspace/data-bundles/<B>.tar, <B>.bundle.json   private training-data bundle (uploaded by the user)
/workspace/jobs/<stage>.json       job file (approval written by the user); eval stages also
                                   <stage>.params.json and <stage>.steps.json (2c)
/workspace/runs/<run-id>/          one new dir per start.sh run (never reused)
    job.json plan.json owner.json run_state.json preflight.json budget_result.json chain.jsonl
    logs/{preflight,smoke,main}.log  logs/disk.jsonl
/workspace/outbox/                 step 4: <RUN>.tar + <RUN>.manifest.json written by `collect pack|pack-eval`
/home/jang2/mmdl-artifacts/...     the bundle's files, staged at their ORIGINAL absolute paths (container
                                   disk: staged again on every Pod, see step 2b)
```

Placeholders: `<COMMIT40>` full MMDL_test SHA, `<POD_ID>` the RunPod Pod id, `<RUN>` run id,
`<B>` bundle name.

## Approval gates (the user answers each one; nothing proceeds on silence)

| Gate | Before | User approves |
|---|---|---|
| G0 | step 0.5 | publish this commit to MMDL_test |
| G1 | step 1 | creating the network volume (size from the formula in step 1, data center, monthly cost) |
| GD | step 2b | **uploading the private training-data bundle to the RunPod volume** (an external upload of non-benchmark training data; its file list and hashes are in `<B>.bundle.json`) |
| G2 | step 2 | one short prep Pod: type, hourly price, max seconds, GPU/financial cutoff (the `pod-prep-v1` job) |
| G3 | step 3, stage 1 | A40 Pod: price, balance snapshot, cutoff; job `bf16-smoke-throughput` |
| G4 | step 3, stage 2 | job `pilot-300`, after step 5 re-ran `cost.py` on the stage-1 receipt |
| G5 | step 3, stage 3 | job `main`, after step 5 re-ran `cost.py` on the pilot receipt |
| GE | step 2 | MMMU validation on the volume (`eval_data: true` in the `pod-prep-v1` job; benchmark data next to training data, so every training Pod must prove its guards forbid it) |
| GF0 | stage `eval-f0` (2c) | uploading `prereg.json` + `.sha256` and the 342-ID subset file to `<volume>/eval-data/proof/`; an A40 Pod and the job (price, balance, cutoff, spec sha256); 3 F0 runs on the 342 items. May run while training runs on another Pod |
| GFT | stage `eval-ft` (2c) | after the training stage is collected (step 4) and the one FT checkpoint is named: export + 3 FT runs |
| GC | (dropped) | no course run: the user dropped it from the proof plan on 2026-10-31 (342-item subset only) |

**How an approval is recorded and bound.** The user writes the `approval` block of the job file:
`approved_by_user: true`, `approved_utc`, `approval_note`, and `approved_sha256`, the sha256 of the
canonical JSON (sorted keys, no spaces, ASCII) of the job *without* its `approval` block. Print it with
`python -m fine_tuning.notebook_rl.pod.job <job.json> --content-sha256` after reading the job; no script
writes the approval. `pod/job.py` refuses a job whose approval is missing, false, older than
`max_approval_age_hours` (optional job field, default 72, at most 168), or whose `approved_sha256` does
not match the job's current content: any later edit of the commit, argv, limits, `financial`, `data` or
`min_free_gib` voids the approval. The approval covers `financial.owned_pod_id`, so it is given after the
Pod exists (the Pod is billing while it waits; keep that step short).

**Financial bounds come only from the job.** `financial` holds `owned_pod_id, gpu_cutoff_utc,
billing_start_utc, balance_usd, reserve_usd, hourly_usd`; `start.sh` and `volume.sh` take no financial
flags (an unknown flag exits 2) and derive the watchdog arguments from the approved job
(`job.py --financial-argv`). `billing_start_utc` must be the Pod's actual first billing time: a start
more than 300 s in the future is refused by `job.py` and by `fine_tuning.budget` (it would move the
financial deadline later than the money allows).

**What the reserve has to cover.** A `BUDGET_STOP` stops the commands, not the Pod: billing continues
until the user terminates the Pod. `reserve_usd` must cover that time (deadline -> user terminates the
Pod) plus recovery (`collect pack`, download).

## 0. Local (before any Pod exists)

1. CPU suites (the full one is also what `runpod_setup.sh` runs on a Pod):
   `cd /mnt/c/projects/multimodel && PYTHONPATH=src:. /home/jang2/mmdl-envs/notebook-nf4-portability-v1/bin/python -B -m unittest discover -s fine_tuning/tests`
   (Pod scripts only: `... -m unittest fine_tuning.tests.test_notebook_rl_pod fine_tuning.tests.test_budget`.)
2. Run (or re-use) the data contamination receipts, input manifests and approvals the run needs.
   Record their paths/hashes in `docs/PROJECT_STATUS.md`; this runbook does not replace them.
3. **Pack the training-data bundle.** The frozen proof data (vault with
   `contamination/union_receipt.json`, its dataset manifest, the exported frozen approval,
   preparation/rule files) name each other by absolute `/home/jang2/mmdl-artifacts/...` paths and the code
   compares them as strings, so the Pod must see every file at exactly the same path. Write a spec
   `{"version": "pod-data-bundle-spec-v1", "paths": [<absolute file or directory>, ...]}` (a directory adds
   every regular file below it) and pack it:

   ```bash
   cd /mnt/c/projects/multimodel
   PYTHONPATH=src:. python -m fine_tuning.notebook_rl.pod.data_bundle pack --spec <spec.json> \
     --out-dir /home/jang2/mmdl-artifacts/pod-outbox --name <B>
   # prints archive sha256, file count, bytes and manifest_sha256 (= sha256 of <B>.bundle.json)
   ```

   `pack` refuses symlinks (also symlinked parents), hardlinked files, special files, paths with an
   `mmmu` component, Hugging Face cache paths (`huggingface`, `hf-cache`, `models--*`, `datasets--*`,
   `$HF_HOME`, `$HF_HUB_CACHE`, `$HF_DATASETS_CACHE`, `~/mmdl-cache`) and `$MMDL_DATA_ROOT/evaluation`.
   It checks paths only; it does not read the files' contents. Which files go into the spec (for example
   whether shard scan evidence, about 2 GB per shard, is needed on the Pod) is a data-governance decision
   to make and record before GD.
4. Dry-run the Pod scripts locally to see the plans (they execute nothing):
   `bash fine_tuning/notebook_rl/pod/start.sh --job </abs/job.json>` (absolute job path; job paths must be
   under `/workspace`, so this validates a job written for the Pod).
5. **[G0]** Sync to MMDL_test: `python -m scripts.sync_test_repo`, then the user pushes. Note the full
   SHA: `git -C /mnt/c/projects/multimodel/MMDL_test rev-parse HEAD` -> `<COMMIT40>`.
   (The preflight later refuses any dirty or different checkout.)

## 1. [USER, G1] Create the Secure network volume

- Secure Cloud only (network volumes do not exist on Community Cloud and are pinned to one data center).
- Data center: one with A40 stock. On 2026-10-30 `engineering.md` read A40 Secure at HIGH stock in
  CA-MTL-1 and EU-SE-1 and $0.59/h; re-check live before creating.
- **Size** (JUDGMENT; the 1.7 GB per checkpoint comes from the plan, not from a measured A40 run):

  `size = 20 GB + 2 x 1.7 GB x ceil(max_updates / checkpoint_every) + bundle size + logs/journal margin`

  20 GB is the venv (about 10-12 GB) plus the model (about 9 GB). Each kept checkpoint is about 1.7 GB
  and counts twice: once in the run dir and once inside the `collect pack` tar that step 4 writes to
  `/workspace/outbox` on the same volume. Checkpoints are not pruned.
  Worked example for the proof run: about 3,000 items with K = 4 problems per update gives
  `max_updates = 750`; with `checkpoint_every = 10` that is `ceil(750 / 10) = 75` checkpoints, so
  `20 + 2 x 1.7 x 75 = 275 GB`, plus about 0.5 GB for a vault-only bundle and 5 GB for logs/journals:
  about 280 GB; ask for 300 GB. At the plan's $0.07/GB-month that is about $21/month (confirm on the
  pricing page).
- **If the MMMU-val proof (section 2c) also runs on this volume**, add about 20 GB (JUDGMENT): 0.7 GB eval
  data (325 MiB of parquet per `manifests/datasets/mmmu.json` plus about 333 MiB of `.arrow` the evaluator
  writes next to it, both measured locally), about 1 GB per evaluation run x 6 runs (342 items each, so less) (the local B2 run dir
  is 1009 MB) and about 9 GB for the merged BF16 FT export.
- The job's `min_free_gib` (step 3) is the stop line for free space during the run; set it to at least
  one checkpoint plus margin (for example 10).
- Record the volume id and data center. The volume is an owned resource: it is deleted only in step 4
  after recovery is verified.

## 2. One-time volume preparation (cheapest Pod that mounts the volume at `/workspace`)

**[G2]** Start the cheapest Pod in the same data center with the volume attached at `/workspace`
(python3.12 and git required; a CPU Pod is enough because nothing uses the GPU). Record its id, price,
first billing time. Then write `/workspace/jobs/prep.json` and have the user approve it:

```json
{
  "version": "pod-prep-v1",
  "commit": "<COMMIT40>",
  "volume_root": "/workspace/mmdl-volume",
  "environment": "proof",
  "eval_data": false,
  "financial": {"owned_pod_id": "<POD_ID>", "gpu_cutoff_utc": "<UTC>", "billing_start_utc": "<UTC>",
                "balance_usd": 0.0, "reserve_usd": 0.0, "hourly_usd": 0.0},
  "approval": {"approved_by_user": false, "approved_utc": "<UTC with Z>", "approval_note": "<what>",
               "approved_sha256": "<job.py --content-sha256>"}
}
```

```bash
git clone https://github.com/jang2296/MMDL_test.git /root/MMDL_test
cd /root/MMDL_test && git checkout <COMMIT40> && git status --short   # must print nothing
bash fine_tuning/notebook_rl/pod/volume.sh --job /workspace/jobs/prep.json             # dry-run plan + budget fit
bash fine_tuning/notebook_rl/pod/volume.sh --job /workspace/jobs/prep.json --execute
```

`environment` is `experiments`, `vllm` or `proof` (`env/requirements-<environment>.lock`); the proof uses `proof` (= the vllm lock + `bitsandbytes==0.48.2`: training imports bitsandbytes even under the BF16 policy, and the MMMU eval stages need vllm). It runs under
`fine_tuning.budget` for at most 5400 s (the job must fit that before its deadline), refuses an existing
target, creates `venv-<lock sha256[:16]>`, runs `pip install -r <lock>`, `pip check`, `pip freeze --all`,
downloads the pinned model with hash verification (`mmdl.data.download.acquire`, compared with
`manifests/models/baseline.json`), writes `volume_manifest.json` with `status: VOLUME_READY` and the
watchdog result to `budget_result.json`. It does not run the CPU suite or GPU doctor (do that in step 0
and in the stage-1 smoke).

Check: `python3.12 -m json.tool /workspace/mmdl-volume/volume_manifest.json | head -30`. Failures are
kept as they are; a new attempt needs a new, non-existing `volume_root` (and so a new approval).

**[GE] Evaluation data: a separate user decision.** `eval_data` is optional (default `false`). Set it to
`true` only after the user approved putting MMMU validation on this volume (benchmark data on the same
volume as the training data; every training Pod then has to prove its guards forbid it, see 2b). The flag
is part of the approved content, so changing it voids the approval. Eval data cannot be added to an
existing volume root later (volume.sh never writes into one); decide before G2. With `true`, after the
model, volume.sh runs the repo's acquisition code with the pinned revision:

```text
acquire("MMMU/MMMU", "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68", "dataset",
        ["README.md"] + ["<subject>/validation-00000-of-00001.parquet" for the 30 subjects],
        cache=<volume_root>/eval-data/hf-cache, destination=<volume_root>/eval-data/mmmu)
```

(`mmdl.data.download.acquire`: the Hub revision must equal the pin; each file's size and LFS sha256 must
equal the Hub metadata), then compares the 31 records with `manifests/datasets/mmmu.json` (repo, revision,
files; else `Pinned MMMU validation hash mismatch`), writes the evaluator's `eval-data/mmmu/manifest.json`
the way `mmdl.data.download.prepare` does, and records `eval_data` in `volume_manifest.json` (revision,
`files` with bytes and sha256, `data_manifest_sha256`, `training_allowed: false`; `null` when the flag is
false). The dataset has its own HF cache: the model cache that training Pods use as `HF_HOME` never holds
MMMU, and every MMMU byte on the volume (cache, data root, the `.arrow` cache the evaluator adds, the
evaluation outputs of 2c) is under `<volume_root>/eval-data`. Network: volume.sh passes no token, and
whether the Hub serves MMMU/MMMU without one was not checked in this work (no network). A 401/403/gated
answer fails the prep closed (no `volume_manifest.json`, no `VOLUME_READY`); a retry needs a new
`volume_root` and a user decision about a token (`export HF_TOKEN=...` before volume.sh, which
huggingface_hub reads; there is no script flag for it).

The evaluation stages need vLLM 0.11.0: the volume they use must be prepared with `environment: vllm`
(`env/requirements-vllm.lock`); the `experiments` lock has no vllm. Check after an `eval_data: true` prep:
`python3.12 -c "import json;m=json.load(open('/workspace/mmdl-volume/volume_manifest.json'));e=m['eval_data'];print(e['revision'],len(e['files']))"`
(expect the pinned revision and 31).

## 2b. Data bundle: upload once, stage on every Pod before GPU work

**[USER, GD]** Upload `<B>.tar` and `<B>.bundle.json` to `/workspace/data-bundles/` (console /
`runpodctl` / `scp`; outside these scripts). This is an external upload of the private (non-benchmark)
training data and needs the user's explicit approval; nothing here uploads anything.

The staged files live on the container disk at their original absolute paths, which a new or restarted
Pod does not keep. Stage and verify on the prep Pod right after the upload (proves the bundle before any
GPU time), then again on every training Pod before `start.sh`:

```bash
cd /root/MMDL_test
B=/workspace/data-bundles/<B>
python3.12 -B -m fine_tuning.notebook_rl.pod.data_bundle stage --archive $B.tar --manifest $B.bundle.json \
  --manifest-sha256 <manifest_sha256 printed by pack>
python3.12 -B -m fine_tuning.notebook_rl.pod.data_bundle verify --manifest $B.bundle.json \
  --manifest-sha256 <manifest_sha256> --archive $B.tar      # exit 0 and "VERIFIED"
```

`stage` checks the manifest sha256 and the archive sha256 first, refuses if any target already exists
(nothing is overwritten), refuses symlink/hardlink/device members, absolute or `..` member names and a
symlinked existing parent, writes each file with `O_EXCL` (mode 0600) while hashing it, and on any error
removes the files it created in that call. The container disk must hold the bundle (size in
`<B>.bundle.json`, field `bytes`). The training preflight re-verifies all of it (step 3.4).

**Training Pods: make the guards forbid the eval data (every training Pod, before `start.sh`).** Every
training-side guard (`local_run._install_actor_guards` used by local_run/local_train/local_eval/passrate,
passrate's build-root mode, the verifier services, the local_eval scorer) forbids
`Path.home()/"mmdl-data/evaluation"`; the actor guards also forbid `datasets--MMMU--MMMU(_Pro)` in the HF
hub of `HF_HUB_CACHE`/`HF_HOME`. None of them reads `MMDL_DATA_ROOT`. On a Pod `$HOME` is on the container
disk, so point that path at the volume's eval root (also before `eval-data` exists, because an F0 Pod may
create it while training runs):

```bash
mkdir -p "$HOME/mmdl-data" && ln -s /workspace/mmdl-volume/eval-data "$HOME/mmdl-data/evaluation"
```

(`ln -s` fails if the path exists; inspect an existing one, never replace it.) The training preflight check
`eval_isolation` (stages `bf16-smoke-throughput`, `pilot-300`, `main`) refuses the run unless that path
resolves exactly to `<volume_root>/eval-data` in the environment start.sh hands to smoke and main, and
unless no other `/workspace/*/eval-data` exists. The guard is a Python audit hook (`guards.py`), not an OS
sandbox. `fine_tuning.notebook_rl.launch` and `gate_session` take explicit `forbidden_roots` from their own
job files; if either runs on a Pod, its job must list `/workspace/mmdl-volume/eval-data` itself.
Evaluation Pods need neither the training bundle nor this link.

## 2c. MMMU-val proof evaluation (stages `eval-f0`, `eval-ft`; `eval-course` is not planned)

Preregistered design (`proof_eval preregister --subset ...`; local file
`/home/jang2/mmdl-artifacts/notebook-rl-local-nf4-20261021/hf-pool-v3/proof-prereg-v2/prereg.json`, sha256
`f392cb3abc174dde5d56f07a188bad619ede69ea833ff3eb44a18528bfeb0848`, with its `.sha256`; its `notes` and
`limitations` record the decisions below; the 900-item `proof-prereg/` is superseded and kept unchanged). User decisions 2026-10-31:

- **Items:** only the 342 MMMU-val items the frozen B2 + `team-final-answer-v8` baseline answered wrongly,
  listed in `hf-pool-v3/proof-eval-subset-v1/b2_v8_wrong_ids.json` (sha256
  `edc16a9d89b25a24f9125d699217461d2728e2aefba4385d85be2d896353b55c`; evaluation only, never training).
  The prereg embeds the IDs and binds that sha256. Every run is `--mode partial` with one `--sample-id` per ID.
- **Arms:** F0 (untrained BF16 base, re-run on the same 342; the stored B2 answers are not the comparator) and
  FT (merged export of the single training checkpoint), paired per item, seeds 3407/3408/3409.
- **Protocol:** the fast8k variants `mmmu-val-v8-fast8k-seed{3407,3408,3409}-v1` = `mmmu-val-v8` except seed,
  32 concurrent requests, `max_new_tokens` 8192, pixels 262144..1003520 (= the training image budget),
  `max_model_len` 24576. The c16 files stay in `eval_configs/` but `run-eval` refuses them.
- **Primary outcome:** per-seed strict V8 accuracy on the 342 (= recovered count / 342), reported as the seed
  mean, FT - F0 with a paired item bootstrap (CLAUDE.md §2.8: seeds are combined only as the mean).
  `proof_eval` has only the V8 scorer; Qwen's own MMMU scoring uses an LLM-assisted judge, not used here.
- **Course run dropped:** no FT x `mmmu-val-v8` (c2, 32K, official pixels) run is preregistered, so this
  proof yields **no score comparable to the 62.00% B2 baseline**, and regressions on the 558 items B2 got
  right cannot be detected. The course code path (`scripts/eval.sh`, `COURSE` in `proof_eval`) is kept.

No other MMMU-val run, item, seed or checkpoint; MMMU-val never selects a checkpoint. `proof_eval analyze`
runs locally after collection. Nothing here was run on a Pod; the specs were checked on CPU only
(`fine_tuning/tests/test_notebook_rl_pod_eval.py`).

| Gate | Before | User approves |
|---|---|---|
| GE | step 2 | MMMU validation on the volume (`eval_data: true`), see section 2 |
| GF0 | stage `eval-f0` | uploading `prereg.json` + `.sha256` and the subset file (MMMU item ids and B2 correctness, no item content) to `<volume>/eval-data/proof/`; an A40 Pod (price, balance, cutoff) and the F0 job with its spec sha256. May run while training runs on another Pod |
| GFT | stage `eval-ft` | after the training stage is collected (section 4) and the one FT checkpoint is named (path, `metadata.json` sha256, binding file sha256): export + 3 FT runs |

1. Params (`eval-stage-params-v1`, absolute paths). `artifact_root`, `ledger`, `prereg` and `subset` must be
   under `<volume_root>/eval-data/` (raw responses hold MMMU questions, gold and images; the training guards
   forbid that root). `python` is the manifest venv (vllm lock). Example for `eval-ft`:

   ```json
   {"version": "eval-stage-params-v1", "commit": "<COMMIT40>", "repo_root": "/root/MMDL_test",
    "python": "/workspace/mmdl-volume/venv-<LOCK16>/bin/python", "volume_root": "/workspace/mmdl-volume",
    "artifact_root": "/workspace/mmdl-volume/eval-data/artifacts",
    "prereg": "/workspace/mmdl-volume/eval-data/proof/prereg.json", "prereg_sha256": "<sha256>",
    "ledger": "/workspace/mmdl-volume/eval-data/proof/ledger.jsonl", "run_prefix": "nbrl-proof-v1",
    "subset": "/workspace/mmdl-volume/eval-data/proof/b2_v8_wrong_ids.json",
    "subset_sha256": "edc16a9d89b25a24f9125d699217461d2728e2aefba4385d85be2d896353b55c",
    "checkpoint": "<abs checkpoint dir>", "checkpoint_metadata_sha256": "<sha256 of its metadata.json>",
    "binding": "<abs launch.json or binding JSON>", "binding_sha256": "<sha256>",
    "training_commit": "<40-hex training commit>", "ft_export": "/workspace/mmdl-volume/ft-export"}
   ```

   `eval-f0` has only the common keys (no `checkpoint`..`ft_export`). `plan --stage eval-course` is refused.
2. Plan (writes once, prints the spec sha256; same params and subset file give the same bytes; `plan` reads
   the subset only after its sha256 equals `subset_sha256`):
   `python -m fine_tuning.notebook_rl.pod.eval_stage plan --stage eval-f0 --params <P> --output /workspace/jobs/eval-f0.steps.json`.
   Steps, all argv lists run by `pod/chain.py` from the clone root:
   - every stage: `chain pin` of the subset file, then of the prereg (and of the checkpoint `metadata.json` +
     binding for `eval-ft`) before anything else;
   - `eval-ft`: `proof_eval export --checkpoint --binding --base <pinned snapshot> --output <ft_export>
     --code-commit <training_commit>` (one merge of the one checkpoint);
   - then all of the stage's `proof_eval ledger-start` lines (arm/variant exactly as preregistered), then the runs:
     `proof_eval run-eval` x seeds 3407/3408/3409 with the fast8k variant files. All runs: hardware
     `fine_tuning/notebook_rl/eval_configs/hardware/a40_48gb.yaml`, `--data-root <volume>/eval-data/mmmu
     --no-download`, `--require-commit <commit> --job-id <id> --run-id <id>-evaluation --run-role standalone
     --mode partial` and one `--sample-id` per subset ID (the evaluator refuses run role `evaluation` for a
     partial run; a finished partial run's `summary.json` status is `PARTIAL`, an unfinished one
     `INCOMPLETE`), run ids `<run_prefix>-{f0,ft}-fast8k-s<seed>-evaluation`.
3. Job: `pod-job-v1` with `stage` `eval-f0` | `eval-ft`, **no `data` key** (refused for eval
   stages), `main` = `["<venv python>", "-m", "fine_tuning.notebook_rl.pod.chain", "run", "--spec",
   "<steps>", "--spec-sha256", "<printed sha>"]` (the approval binds the argv, so it binds the spec), `smoke`
   = a GPU check that does **not** touch MMMU (an MMMU smoke run would be an MMMU-val run outside the
   preregistered budget). `max_main_seconds`: no A40 throughput has been measured, so every number below is
   an estimate; set the value with the user and report a `BUDGET_STOP` or timeout as such, never retried
   silently.
   - `eval-f0` (3 fast8k runs of 342 items, the first measurement): output is capped at 8192 tokens per item
     (at most 2.80M generated tokens per run by construction; for scale, the 32K R2 run generated 3.72M for
     all 900) and 32 requests share each decode step. The B2-wrong items are likely longer than average
     (R2's 75 length stops are failures by definition), so do not scale R2's mean down by 342/900. Read each
     finished run's `summary.json` `evaluation_seconds_this_invocation`.
   - `eval-ft`: from eval-f0's measured seconds per run x 3, plus the export, with margin.
   KV cache (estimate, from the model config: 36 layers x 8 KV heads x head_dim 128 x K,V x BF16 = 147,456 B
   = 144 KiB per token): about 45 GiB visible x `gpu_memory_utilization` 0.90 (fixed in src) = about
   40.5 GiB, minus about 8.3 GiB BF16 weights and an unmeasured 2-6 GiB activation/encoder reserve, leaves
   about 26-30 GiB = about 190K-220K tokens. Per request: input at most 2,655 tokens (the measured R2 maximum
   at the larger 1310720-pixel budget; mean 593) + at most 8,192 output. Worst case 32 x 10,847 = 347K
   tokens does not fit, so vLLM preempts and recomputes (slower, no truncation, no error); with the R2 means
   (593 input, 4,132 output at the 32K cap) the average resident load is about 32 x 2.7K = 85K tokens, which
   fits. `max_model_len` 24576 only bounds one request (input <= 16,384 tokens) and vLLM's start check needs
   one 24,576-token sequence (about 3.4 GiB of KV) to fit.
4. Run as in section 3 (one Pod per stage; dry run first, then `--execute`). The preflight for an eval
   stage checks clone, lock, venv, model, disk and mount as for training, skips the training data, and
   adds `eval_data`: each recorded MMMU file under `eval-data/mmmu` resolves inside `eval-data` with its
   recorded size and full sha256, and `manifest.json` has the recorded sha256.

Rules for these stages:

- `eval-f0` and `eval-ft` use the same `commit`: `analyze` requires one `code_sha256` and one registration
  (src and proof_eval bytes) across the six statistical runs, and `--require-commit` makes the evaluator
  refuse any other or dirty checkout.
- One ledger for all stages. `ledger-start` locks it with `flock`, which is not verified across Pods on a
  network volume, and concurrent evaluators would share `eval-data/mmmu/.arrow`: run the evaluation stages
  one at a time (F0 next to a training Pod is fine; training never touches either).
- On failure the chain stops at the failed step. A started run is never replaced because of its score:
  record it with `proof_eval ledger-fail --ledger <L> --run-id <R> --reason <text> --artifact-root
  <artifact_root>` and report it, or resume it in place under the same run id (`--resume`). Neither is
  planned by `eval_stage.py`; a resume or partial re-run spec needs its own review and approval.
- `analyze` needs, collected from the volume: `<artifact_root>/runs/<run id>/` of all 6 runs, the ledger,
  the prereg and `<ft_export>/model_ref.json`. It refuses a run whose `run_manifest.json` `expected_ids`,
  `identity.ids_sha256` or `predictions.jsonl` is not exactly the preregistered 342, a run that is not
  `mode: partial` with summary `PARTIAL`, and keeps the seed-mean-only and ledger rules. `collect pack-eval`
  packs exactly these per stage (step 4), each finished run marked `PARTIAL` (or `FAILED` per the ledger):
  from each planned run dir only `run_manifest.json`, `summary.json`, `predictions.jsonl`,
  `resolved_eval_config.yaml` and `variant_registration.json`, plus the ledger, the prereg with its
  `.sha256`, `model_ref.json` (`eval-ft`) and the stage's start.sh run dir (`chain.jsonl`, logs).
  `proof_eval ledger-fail` refuses a run whose summary is `PARTIAL` (finished) or `COMPLETE`.
  Not collected, so lost with the volume: `samples/`, `prompts/`, the evaluator's other run files, and the
  merged model weights (`<ft_export>/model/`). Raw MMMU responses are never published (§2.2, §5): eval packs
  are downloaded and extracted only under `/home/jang2/mmdl-artifacts/` and `collect verify` refuses
  anything else for them.

## 3. Training Pod (A40 Secure, same data center, volume at `/workspace`)

Every stage is one new Pod session or one new `<RUN>` directory; never reuse a run directory.

1. **[G3/G4/G5]** Start the A40 Secure Pod with the volume. Record `<POD_ID>`, hourly price, first
   billing time, actual balance, cutoff. Clone and check out as in step 2, then stage the data (2b).
1b. **Plan the step chain** (CPU, seconds; writes only the spec file). A job has one `main` argv, so the
   stage's commands are a fixed list run by `pod/chain.py`. Copy the stage's template from
   `fine_tuning/notebook_rl/pod/plans/` and fill every `<...>` (each template's `notes` say how), then:

   ```bash
   cd /root/MMDL_test
   V=/workspace/mmdl-volume/venv-<LOCK16>/bin/python
   cp fine_tuning/notebook_rl/pod/plans/params-stage1.json /workspace/jobs/stage1.params.json   # then edit
   PYTHONPATH=$PWD/src:$PWD $V -B -m fine_tuning.notebook_rl.pod.chain plan --stage stage1 \
     --params /workspace/jobs/stage1.params.json --output /workspace/jobs/stage1.steps.json
   # prints {"spec_sha256": ..., "steps": N, "total_timeout_seconds": T}; refuses placeholders and an existing output
   ```

   `--stage` is `stage1` (job stage `bf16-smoke-throughput`), `pilot-300` or `main`
   (`params-main.json` for the first main segment, `params-main-resume.json` for a later one). The same
   params give the same bytes. The proof protocol is fixed in `chain.py`, not in the params:
   `--note-contract repair-note-compact-v1 --answer-verified-route --finalization-reward`, the ten
   `--slotless-source`s of the local `chain-ten-v15.sh`, `--turn1-contract p0-eval-aligned-v1
   --turn1-max-new-tokens 8192 --retry-out-of-choice incorrect-v1` (`local_train train` gets only the
   first four; it restores the rest from the minimum). Every run also gets `--policy-precision bf16
   --rollout-engine vllm` with the params' vLLM budget, admission
   `verified-functional-repair-exploration-v1`, and `--frozen-approval`/`--frozen-approval-sha256` on
   every command that has them. Each step's output goes under `<run_dir>/out/`:

   | stage | steps (in order) |
   |---|---|
   | `stage1` | `prefilter-propose` (all-eligible, N0=`prefilter.items`=16, K=1) -> `prefilter-prepare` -> `passrate-measure` (k=4, F0) -> `passrate-select` (1..3, rehearsal only) -> `propose` (all-eligible, N=8, K=4) -> `prepare` -> `minimal` (2 problems, 1 update, save) -> `minimal-reload` (new process) -> `train` (2 updates, checkpoint every 1, `--max-seconds 1800`) -> `train-reload` -> `eval-f0`, `eval-ft` (`evaluate-pool` on the sentinels) -> `replay-f0`, `replay-ft`, `compare` (CPU) -> `stage-receipt` |
   | `pilot-300` | the same, with `prefilter.items` = every eligible distinct-structure train item, and `propose` takes `--passrate-selection`/`--passrate-report` from this run (N=300, K=4) |
   | `main` (first segment) | `selection-pin` (the pilot's `out/passrate-selection.json` sha256) -> `propose` from that selection (N = min(selected, 3000) rounded down to K) -> `prepare` -> `minimal` -> `minimal-reload` -> `train` (checkpoint every 50) -> `train-reload` -> `eval-f0`, `eval-ft` -> `replay-*`, `compare` -> `stage-receipt` |
   | `main` (later segment) | `train --resume-from <stopped segment>` with the first segment's preparation and minimum -> `train-reload` -> `eval-f0`, `eval-ft` -> `replay-*`, `compare` -> `stage-receipt` |

   main starts at fresh F0 and does **not** continue the pilot. `local_train --resume-from` accepts only
   a closed, unfinished segment (no `reload_expected.json`) with the identical protocol, execution
   contract (minus `resume`) and preparation. The pilot is complete and binds N=300 in its protocol.
   `local_train` caps `--max-seconds` at 86400, so a main segment that ends `BUDGET_STOP` (train exit
   2, chain exit 2, `MAIN_EXIT 2`) is collected (step 4) and continued by a new approved job planned
   from `params-main-resume.json` in a new `<RUN>`. Updates after the last committed checkpoint (up to
   49) are redone.
2. Write `/workspace/jobs/<stage>.json`; the user approves it for the stage that was just approved
   (`approved_by_user: true`, `approved_sha256` from `--content-sha256`; the template ships with `false`):

   ```json
   {
     "version": "pod-job-v1",
     "commit": "<COMMIT40>",
     "volume_root": "/workspace/mmdl-volume",
     "run_dir": "/workspace/runs/<RUN>",
     "stage": "bf16-smoke-throughput",
     "smoke": ["/workspace/mmdl-volume/venv-<LOCK16>/bin/python", "-m", "fine_tuning.notebook_rl.pod.gpu_smoke"],
     "main": ["/workspace/mmdl-volume/venv-<LOCK16>/bin/python", "-m", "fine_tuning.notebook_rl.pod.chain", "run",
              "--spec", "/workspace/jobs/stage1.steps.json", "--spec-sha256", "<spec_sha256 printed by chain plan>"],
     "max_main_seconds": "<>= total_timeout_seconds printed by chain plan>",
     "min_free_gib": 10,
     "financial": {"owned_pod_id": "<POD_ID>", "gpu_cutoff_utc": "<UTC>", "billing_start_utc": "<UTC>",
                   "balance_usd": 0.0, "reserve_usd": 0.0, "hourly_usd": 0.0},
     "data": {"bundle_manifest": "/workspace/data-bundles/<B>.bundle.json",
              "bundle_manifest_sha256": "<manifest_sha256>",
              "dataset_manifest": "/home/jang2/mmdl-artifacts/notebook-rl-local-nf4-20261021/hf-pool-v3/freeze-v2/dataset_manifest.json",
              "dataset_manifest_sha256": "<sha256 of that file>",
              "vault_root": "/home/jang2/mmdl-artifacts/notebook-rl-local-nf4-20261021/hf-pool-v3/vault-v1"},
     "approval": {"approved_by_user": false, "approved_utc": "<UTC with Z>", "approval_note": "<what was approved>",
                  "approved_sha256": "<job.py --content-sha256>"}
   }
   ```

   `smoke[0]` and `main[0]` must be exactly `<manifest venv>/bin/python` (the `venv` path in
   `volume_manifest.json`); `start.sh` refuses any other interpreter after the preflight passed.
   `stage` is one of `bf16-smoke-throughput`, `pilot-300`, `main`. `max_main_seconds` is an integer
   (the string above is a placeholder) and at least the plan's `total_timeout_seconds` (the sum of the
   per-step timeouts). The approval covers the argv, so it binds the spec's sha256 and the whole step
   list; a re-planned spec needs a new approval. The job decides the argv; `start.sh` never alters
   precision, hardware or limits.

   **smoke = `pod/gpu_smoke.py`** (hard 300 s). It records `nvidia-smi`, torch/CUDA/transformers
   versions, free/total GPU memory, one real BF16 2048x2048 matmul against FP32 (relative error < 0.05),
   loads the processor and model in BF16 from the offline cache (`HF_HUB_OFFLINE=1`,
   `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17`, `local_files_only`), and does one
   sampled generation on a synthetic image (eval sampler: temperature 0.7, top_p 0.8, top_k 20,
   generated-only presence penalty 1.5, 32 new tokens; the text is recorded, not judged). It writes
   `<run_dir>/gpu_smoke.json` after every phase, so a 300 s kill still leaves the finished phases. Any
   failure exits 1 (`SMOKE_FAILED`, main not started). There is no other precision, device or model. The
   model load from the network volume has not been timed and must fit in 300 s.

   **main = `chain run`**. It refuses a spec whose sha256 differs, `MMDL_RUN_DIR` other than the planned
   `run_dir`, `MMDL_JOB_STAGE` other than the planned job stage, and an existing `chain.jsonl`. It runs
   the steps in order from the clone root with the job's environment unchanged, each in its own session
   with its own timeout. The train step's timeout is `train.max_seconds + timeouts.train_close`, so
   training stops itself first. Records are hash-chained JSON lines in `<run_dir>/chain.jsonl` (start,
   end, exit code, elapsed time, which `expect_files` exist). The first record links to the spec sha256.
   Check with `chain verify --log <run_dir>/chain.jsonl --spec-sha256 <sha>`. The first nonzero exit, a
   missing expected file (exit 1) or a step timeout (exit 124) stops the chain with that code: no retry,
   no fallback, no skipped step. A SIGTERM from the runner or watchdog goes to the running step's group
   (SIGKILL after 20 s, inside the runner's 30 s), and the chain exits 143.

   **Stage receipt.** `passrate measure`, `local_train train` and `local_eval evaluate-pool` each write
   `throughput_receipt.json` (`throughput-receipt-v1`, measured values) in their output directory and
   mirror it into `<run_dir>/throughput_receipt.json`, where the last writer wins. The chain's final
   step `stage-receipt` overwrites that root file with the **train step's** receipt, adding
   `stage_receipt_step: train`, the source path and its sha256. If the chain stopped earlier, the root
   file is whichever step mirrored last (`mirrored_from` names it). Then pass
   `<run_dir>/out/train/throughput_receipt.json` or `out/passrate/throughput_receipt.json` to `cost.py`
   explicitly. `collect pack` accepts any of `result.json`, `receipt.json`, `throughput_receipt.json` for
   a successful run. Both commands run with `HF_HOME=<volume>/hf-cache`, `HF_HUB_OFFLINE=1`,
   `MMDL_RUN_DIR`, `MMDL_VOLUME_ROOT`, `MMDL_JOB_STAGE` and `PYTHONPATH=<clone>/src:<clone>`.
3. Dry-run (prints the plan and the budget fit from `job.financial`, executes nothing), then run:

   ```bash
   cd /root/MMDL_test
   bash fine_tuning/notebook_rl/pod/start.sh --job /workspace/jobs/<stage>.json
   bash fine_tuning/notebook_rl/pod/start.sh --job /workspace/jobs/<stage>.json --execute
   ```

   `--quick-verify` (model files by size plus first/last 64 MiB hash, recorded as `quick`, never reported
   as a full hash check; the data bundle is always fully hashed) is optional. `--execute` requires
   `RUNPOD_POD_ID == financial.owned_pod_id`. `start.sh` re-execs itself under `fine_tuning.budget`, which
   ignores SIGHUP (a dropped SSH session no longer kills the watchdog; the child keeps the default SIGHUP
   action but runs in its own session, which a terminal hangup does not reach) and writes its result to
   `<run_dir>/budget_result.json`. Still start it inside `tmux`/`nohup`: messages that `start.sh` prints to
   a hung-up terminal are lost (the state files keep the record). `start.sh` (and `volume.sh`) pass
   through to the watchdog branch only when `MMDL_BUDGET_WATCHDOG_PID` is their parent's PID **and**
   `/proc/$PPID/cmdline` is exactly the `fine_tuning.budget ... -- bash start.sh ...` argv they would have
   exec'd themselves; otherwise exit 2 `watchdog argv mismatch`. Setting the variable by hand does not get
   around the watchdog.
4. What `start.sh` does, in order: validate the job (approval incl. `approved_sha256`, commit *format*,
   paths, stage, argv, data pins, budget fit) -> refuse an existing run dir -> watchdog re-exec -> freeze
   the job to `<run_dir>/job.json` (later reads use the copy) and write `plan.json` -> `verify.py` under a
   hard 600 s limit: clean clone whose HEAD equals the job commit and MMDL_test origin (checked here, not
   in the validation step), lock sha256 and every `-r`/`-c` include equal the manifest, venv python and
   `pyvenv.cfg`, **a live `<venv>/bin/python -m pip freeze --all` (60 s limit) whose sha256 equals the
   manifest's `pip_freeze_sha256`**, model sizes and sha256 in parallel, pinned baseline match,
   **`data_bundle`** (the bundle manifest's sha256 equals `data.bundle_manifest_sha256`, every bundle
   file is a regular unaliased file at its absolute path with the manifest's size and sha256, and the
   bundle holds the dataset manifest and the vault), **`dataset_vault`** (`dataset_manifest` sha256 equals
   the pin and `pool_freeze.verify(dataset_manifest, vault_root)` returns `[]`, run in the volume venv
   because pool_freeze needs numpy/PIL/yaml), free disk >= `min_free_gib`, `/workspace` is a mount;
   writes `preflight.json` -> after `PREFLIGHT_OK`, `smoke[0]` and `main[0]` must equal the manifest venv
   interpreter (else `PREFLIGHT_FAILED`, exit 2, nothing started) -> smoke (hard 300 s) -> main. During
   smoke and main the runner samples free disk on the run-dir filesystem every 60 s into
   `logs/disk.jsonl`; below `min_free_gib` it stops the command group and records `DISK_LOW`.
   `run_state.json` states: `PREFLIGHT_OK` | `PREFLIGHT_FAILED`, `SMOKE_OK` | `SMOKE_FAILED`,
   `MAIN_RUNNING`, `MAIN_EXIT <code>` (124 = the per-step timeout), `DISK_LOW`, `BUDGET_STOP` (the
   watchdog's financial/GPU deadline; `budget_result.json` says `BUDGET_STOP`), `INTERRUPTED` (a manual
   SIGTERM/SIGINT). Exit codes: 2 refused, 3 preflight failed, 4 smoke failed, 5 disk low, 143 signal,
   otherwise the main command's code. `MMDL_RUN_DIR=<run_dir>` is exported to every command (step 4 uses
   it to find survivors).
5. Every 10 minutes (CLAUDE.md §2.6): `cat /workspace/runs/<RUN>/run_state.json`,
   `tail -n 2 /workspace/runs/<RUN>/chain.jsonl` (current step),
   `tail -n 20 /workspace/runs/<RUN>/logs/main.log`, `tail -n 1 /workspace/runs/<RUN>/logs/disk.jsonl`,
   `nvidia-smi`, `ps -o pid,etime,cmd -C python`, and the account balance against the cutoff. Never
   restart a failed run automatically; the next attempt needs a new approval, a new `<RUN>`, and a stated
   reason that differs from the failure.
6. Stage order is fixed: stage 1 `bf16-smoke-throughput` -> step 5 -> **G4** `pilot-300` -> step 5 ->
   **G5** `main` -> its collection (step 4) -> **GFT** `eval-ft` (no course run: dropped 2026-10-31); **GF0** `eval-f0`
   is independent of training and may run any time after **GE** (section 2c; one evaluation Pod at a
   time). A stage-1 or pilot failure stops the plan; report it as it is. A main segment that ends
   `BUDGET_STOP` or `DISK_LOW` in its train step is not a completed main: it continues only as a new
   approved main segment (step 1b, `params-main-resume.json`).

## 4. Collect, verify, clean up (never before the previous step has evidence)

Ledger order is fixed: `REMOTE_VERIFIED -> LOCAL_VERIFIED -> POD_TERMINATED -> OWNED_STORAGE_DELETED
-> CLEANUP_VERIFIED`. `collect.py` records evidence only; it never calls RunPod. The user performs and
confirms each remote action.

**Which states each stage reaches.** The network volume holds the venv, the model and the data bundle
for every later stage, so it is kept until the last one. The ledger of an intermediate stage
(`bf16-smoke-throughput`, `pilot-300`, `main`, `eval-f0`, `eval-ft`) therefore **stops at `POD_TERMINATED`**;
that is the correct end state for it, not an unfinished cleanup. `OWNED_STORAGE_DELETED` and
`CLEANUP_VERIFIED` are appended only in the ledger of the final stage (the last one that will run, normally
`eval-ft`), after the user deleted the volume (and anything else this job created) because no later
stage will use it. `collect.py` enforces part of this: `OWNED_STORAGE_DELETED` needs
`--run-ledger` for every run that used the storage (this ledger included) and refuses unless each of
those ledgers already holds `REMOTE_VERIFIED` and `LOCAL_VERIFIED`; the list of runs itself is the
user's to make complete. `CLEANUP_VERIFIED` can only follow it.

```bash
# on the Pod: closed run dirs only (owner pid dead, terminal state, result/receipt when MAIN_EXIT 0)
python3.12 -m fine_tuning.notebook_rl.pod.collect pack --out-dir /workspace/outbox --name <RUN> /workspace/runs/<RUN>
# a hard-killed run (SIGKILL / Pod restart: no INTERRUPTED was written) is refused as "not closed"; add
# --unclosed-dead-owner to pack it anyway. Allowed only if the owner (pid + start ticks) is dead AND no
# /proc/*/environ contains MMDL_RUN_DIR=<run dir> (an unreadable entry or a missing /proc counts as
# alive); the manifest then says "closure": "UNCLOSED" and `collect verify` copies it into the receipt.
python3.12 -m fine_tuning.notebook_rl.pod.collect pack --unclosed-dead-owner --out-dir /workspace/outbox --name <RUN> /workspace/runs/<RUN>
# still on the Pod, BEFORE downloading: the hashes REMOTE_VERIFIED records (a hash taken from the
# downloaded copy would make that step circular)
sha256sum /workspace/outbox/<RUN>.tar /workspace/outbox/<RUN>.manifest.json
# [USER] download <RUN>.tar and <RUN>.manifest.json (console / runpodctl / scp; outside these scripts)
# local:
python -m fine_tuning.notebook_rl.pod.collect verify --archive <RUN>.tar --manifest <RUN>.manifest.json \
  --extract-to recovered/<RUN> --receipt recovered/<RUN>.local_receipt.json
L=cleanup/<RUN>.ledger.jsonl
python -m fine_tuning.notebook_rl.pod.collect state --ledger $L append REMOTE_VERIFIED \
  --evidence archive_sha256=<Pod sha of .tar> --evidence manifest_sha256=<Pod sha of .manifest.json> --evidence pod_id=<POD_ID>
python -m fine_tuning.notebook_rl.pod.collect state --ledger $L append LOCAL_VERIFIED \
  --evidence archive_sha256=<same sha> --receipt-file recovered/<RUN>.local_receipt.json
# [USER terminates the Pod]
python -m fine_tuning.notebook_rl.pod.collect state --ledger $L append POD_TERMINATED \
  --evidence pod_id=<POD_ID> --evidence terminated_utc=<UTC> --evidence confirmed_by=user
# final stage only. [USER deletes only storage this job created; never shared/pre-existing volumes]
python -m fine_tuning.notebook_rl.pod.collect state --ledger $L append OWNED_STORAGE_DELETED \
  --evidence deleted_resources=<ids> --evidence deleted_utc=<UTC> --evidence confirmed_by=user \
  --run-ledger cleanup/<RUN1>.ledger.jsonl --run-ledger cleanup/<RUN2>.ledger.jsonl --run-ledger $L
# [USER lists remaining owned resources]
python -m fine_tuning.notebook_rl.pod.collect state --ledger $L append CLEANUP_VERIFIED \
  --evidence residual_query_utc=<UTC> --evidence residual_count=0 --evidence confirmed_by=user
```

`LOCAL_VERIFIED` also refuses unless the receipt's `pod_id` (copied by `collect verify` from the pack
manifest, which recorded `RUNPOD_POD_ID` on the Pod) equals the `pod_id` typed at `REMOTE_VERIFIED`.
`verify` rejects a changed archive, absolute or `..` member names, any symlink/hardlink/device member,
files not in the manifest, and an existing extraction directory. A failed run (`PREFLIGHT_FAILED`,
`SMOKE_FAILED`, `INTERRUPTED`, `BUDGET_STOP`, `DISK_LOW`, non-zero `MAIN_EXIT`) is packed as is so its
logs are recovered. The pack tar is written on the volume next to the runs and needs free space of about
the run dir's size (counted in the step 1 formula). A Pod "stop" is not cleanup.

### Evaluation stages: `pack-eval`, local `verify`, `analyze`

An eval stage's start.sh run dir holds only logs and `chain.jsonl`; its results are under
`<volume>/eval-data` and `<ft_export>`. Plain `pack` refuses a successful eval run dir (no result/receipt
file) and would not carry those results anyway. `pack-eval` writes the same tar + manifest
(`pod-pack-v1`, plus `"kind": "eval"`, `stage`, `eval_runs`), so `verify` and the `state` commands above
are unchanged: **one pack and one cleanup ledger per eval stage**, with the same typed-`pod_id` check.
Before packing, on the Pod: the stage's start.sh run dir is closed (owner dead, terminal state), and every
started run of the stage that is not COMPLETE has a `proof_eval ledger-fail` record (2c). It needs the
volume venv (`numpy`/`yaml`/`mmdl` for the ledger check), not the bare system python:

```bash
cd /root/MMDL_test
PYTHONPATH=$PWD/src:$PWD /workspace/mmdl-volume/venv-<LOCK16>/bin/python -B -m fine_tuning.notebook_rl.pod.collect pack-eval \
  --params /workspace/jobs/<STAGE>.params.json --stage <STAGE> --run-dir /workspace/runs/<RUN> --out-dir /workspace/outbox
sha256sum /workspace/outbox/<RUN>.tar /workspace/outbox/<RUN>.manifest.json    # BEFORE downloading, as above
# [USER] download both files into the private inbox below
```

`--params` is the file `eval_stage plan` read, `--stage` one of `eval-f0|eval-ft` (`eval-course` is refused: not planned); `--name`
defaults to the run dir name. The pack refuses unless: `RUNPOD_POD_ID` is set (recorded as `pod_id`);
`chain.jsonl` is the log of exactly the spec these params plan (its hash chain starts at that spec's
sha256) and, for `MAIN_EXIT 0`, ends `CHAIN_END COMPLETE`; the ledger is a valid hash chain with a START
(this prereg, this arm/variant) for every planned run id; each planned run is finished (`summary.json`
status `PARTIAL` for the subset runs) or recorded FAILED in the ledger (a FAIL for a finished run is
refused); the prereg and its `.sha256` equal `params.prereg_sha256`; no packed file, and no directory between it and its root, is a symlink
and every path lies under `artifact_root`, `<volume>/eval-data` or `ft_export`. A FAILED run is packed with
whatever of its five files exist, marked `FAILED` in `eval_runs` (its directory may be absent). If the
chain stopped before the ledger STARTs (pin failure, preflight/smoke failure) there is nothing to
`pack-eval`; recover that run dir's logs with plain `collect pack`.

Archive layout: `<RUN>/**` (the start.sh run dir) and `eval/{<ledger>,<prereg>,<prereg>.sha256,model_ref.json}`,
`eval/runs/<run id>/{run_manifest.json,summary.json,predictions.jsonl,resolved_eval_config.yaml[,variant_registration.json]}`.
Each manifest `files` record has the archive `path`, absolute `source` path, `bytes` and `sha256`.

Local. Eval packs hold raw MMMU responses (§2.2, §5): `verify` refuses an eval archive or extraction
directory outside `$MMDL_ARTIFACT_ROOT` (default `/home/jang2/mmdl-artifacts`) and extracts with mode 0700;
never put them in the repo:

```bash
cd /mnt/c/projects/multimodel; export PYTHONPATH=src:.
A=/home/jang2/mmdl-artifacts/pod-recovery        # <RUN>.tar and <RUN>.manifest.json downloaded here
python -m fine_tuning.notebook_rl.pod.collect verify --archive $A/<RUN>.tar --manifest $A/<RUN>.manifest.json \
  --extract-to $A/recovered/<RUN> --receipt cleanup/<RUN>.local_receipt.json
# receipt: status LOCAL_VERIFIED, kind eval, stage, eval_runs [{run_id, status}], pod_id, extracted_to
# then REMOTE_VERIFIED / LOCAL_VERIFIED / POD_TERMINATED with L=cleanup/<RUN>.ledger.jsonl exactly as above
```

`OWNED_STORAGE_DELETED` (final ledger only) lists **every** run's cleanup ledger with `--run-ledger`: the
training stages and each eval stage pack (`eval-f0`, `eval-ft`), the final one included;
each must already hold `REMOTE_VERIFIED` and `LOCAL_VERIFIED`, so no stage's results can be left
unrecovered when the volume goes. Analysis is local and only after all three eval packs are verified
(`E=$A/recovered`; the last pack's ledger extends the earlier snapshots, so use it):

```bash
python -m fine_tuning.notebook_rl.proof_eval analyze \
  --prereg $E/<F0 RUN>/eval/prereg.json --ledger $E/<COURSE RUN>/eval/ledger.jsonl \
  --runs $E/<F0 RUN>/eval/runs/* $E/<FT RUN>/eval/runs/* $E/<COURSE RUN>/eval/runs/* \
  --ft-model-ref $E/<FT RUN>/eval/model_ref.json --output $A/report/proof-report
```

Pass only runs whose `eval_runs` status is `COMPLETE`. A run marked `FAILED` stays failed: `analyze`
refuses it and a missing preregistered statistical run, and the result is reported as it is (§2.8); it is
never replaced by another run. `analyze` re-checks the run dirs against the prereg (the run dir name is the
ledger run id), writes `<output>.json` and `<output>.md`; keep them under the private root until a
decision to publish aggregated numbers is made.

## 5. Cost and time from measured receipts (before asking G4 and G5)

After stage 1 and again after the pilot, with the receipts recovered in step 4 (or read on the Pod).
Two plans ship in `fine_tuning/notebook_rl/pod/plans/`, one per receipt kind:

```bash
P=fine_tuning/notebook_rl/pod/plans
# training stages (stage1, pilot-300, main) from the TRAIN receipt (the chain's stage receipt)
python -m fine_tuning.notebook_rl.pod.cost --receipt <run>/throughput_receipt.json --plan $P/cost-plan-v1.json \
  --cap-usd <USD> --output projections/<name>-train.json
# F0 pass-rate prefilter (k=4 first attempts, no update) from the PASSRATE receipt
python -m fine_tuning.notebook_rl.pod.cost --receipt <run>/out/passrate/throughput_receipt.json \
  --plan $P/cost-plan-prefilter-v1.json --cap-usd <USD> --output projections/<name>-prefilter.json
```

`cost-plan-v1.json`: A40 `hourly_usd` 0.59 (2026-10-30 read), `range` 0.5, `reserve_usd` 2.0 and
`setup_minutes` 30 (judgment), stages `stage1` 8 / `pilot-300` 300 / `main` 3000 items (3000 is the cap),
`n_samples` 4. `retries_per_item` 1.4 is the only measured value: the local NF4/HF fixed-ten receipt
`hf-pool-v3/local-test-v3/ten/throughput_receipt.json`, sha256 `3278b5c9...230812`, with a T1 cap of
1024 tokens, not the proof's 8192. Treat it as an estimate until stage 1 measures it.
`cost-plan-prefilter-v1.json`: `stage1-prefilter` 16 and `pilot-300-prefilter` 3420 items (the freeze-v2
INCLUDED train count, an upper bound), `n_samples` 4 (k=4), no retries, setup and reserve 0 (they are counted
in the training plan). Add the two totals. Before G4/G5, copy a plan and replace the item counts with the
planned ones (`prefilter.items`, the main `train.problems`). The sentinel evaluations are in neither
plan; their receipts are `out/eval-*/throughput_receipt.json`.

Plan format (`cost-plan-v1`): `hourly_usd, setup_minutes, reserve_usd, range, gpu_memory_gib?,
cap_usd?, stages: [{name, items, n_samples, retries_per_item}]`, other keys (`notes`) ignored. If the receipt has `startup_seconds`
it is taken out of `wall_seconds` before the per-item rates and added once per stage. Every output number is labelled
`projection`, carries the receipt sha256, and uses a +-`range` band (default 0.5). Exit code 3 and
`refuse: true` mean the high-end projection plus reserve exceeds the cap: do not request the next
approval. The projection is not a measurement of the next stage.

## Not covered here (state it when reporting)

- Resume is manual and needs a new approval. `local_train train --resume-from <segment>` continues a
  closed, unfinished segment (`BUDGET_STOP`, `DISK_LOW`, killed; no `reload_expected.json`) from its last
  committed cadence checkpoint, under the identical protocol, execution contract and preparation (its
  prefix is re-verified on CPU). `chain run` never resumes, retries or re-plans anything by itself. A
  completed run cannot be resumed, so main does not continue the pilot. Within one chain, a failed step
  is not re-entered either: a new job re-runs the whole stage in a new `<RUN>`.
- Free space: `local_run minimal` and `local_train train` get `--min-free-gib` (params `min_free_gib`)
  and stop with `DISK_LOW` instead of writing a checkpoint below it. Earlier committed checkpoints stay
  resumable. Set params `min_free_gib` >= job `min_free_gib` plus one checkpoint (about 1.7 GB, plan
  figure), so training stops itself before the runner's guard fires. That guard stops the chain and the
  chain stops the step group (SIGTERM, SIGKILL after 20 s). `passrate measure` and `evaluate-pool` have
  no free-space flag and rely on the runner's guard alone.
- `chain.py` and `gpu_smoke.py` were tested on CPU only (fake steps; `gpu_smoke` only up to its CUDA
  check). Their GPU phases, the 300 s model load and every planned training command are unverified on
  a Pod.
- The vLLM budget in the params templates (`gpu_memory_utilization` 0.45, 8 sequences, 8 GiB KV) is an
  unmeasured proposal for one A40. It is bound into the protocol hash, so it cannot change between
  the commands of one preparation. The pass-rate prefilter always has its own protocol (K=1,
  N=`prefilter.items`). `propose-proof` checks a selection's vault, split and F0 weights, not its
  protocol, so keep the vLLM budget the same from the pilot prefilter to main, or the selection was
  measured under another sampler budget. Exception, opt-in: params `prefilter_vllm` (same schema as
  `vllm`, prefilter stages only) sets the budget of `prefilter-propose`/`prefilter-prepare`/
  `passrate-measure` alone (no training in that process); absent, the planned argv is unchanged. The
  selection is then recorded as measured under that larger budget, not the training one.
- vLLM sync threshold (user decision 2026-10-31, "vLLM 전용 기준 0.04로 보정"): the HF<->HF probe
  threshold stays 0.02; only the separate vLLM-sampler<->HF-policy report uses the calibrated 0.04
  (`scale.vllm_sync_threshold` = `vllm-hf-bf16-calibrated-0.04-v1`, bound into vLLM protocols only).
- The chain's `eval-f0`/`eval-ft` steps are the held-out sentinel pool evaluation
  (`first_attempt_accuracy`; `recovery_rate_diagnostic` is not an accuracy, §2.8). MMMU evaluation is
  section 2c.
- `data_bundle` refuses benchmark/cache roots by path only; it does not scan file contents.
- The training and generation code path (vLLM sampler fix, BF16 policy, batching) is separate work and is
  unverified on a GPU; the volume only provides the pinned environment and model.
- The pinned venv lives on the network volume; import speed from a network filesystem was not measured.
  The stage-1 smoke receipt is where that shows up.
