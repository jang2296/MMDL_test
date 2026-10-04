# Internal frozen Gemma answer verifier

This private audit never changes Qwen responses, V8 code or the published558/900.
`google/gemma-4-E4B-it` is pinned to `ee0ef6023621cff504d758262d4e04895a5af4a2`.
No training, commercial API, local weight download, MMDL sync or GitHub publication.

The first literal-evidence protocol attempted900 on3090 and ended with607RESOLVED,
117REVIEW and176SYSTEM_ERROR. Its actual outputs and cleanup remain historical
evidence, not successful900 adjudication. A separately authorized source-segment
protocol passed CPU900 and actual GPU29 output-contract checks. It sends the entire
unchanged response as numbered lines; the host materializes selected source spans.
Valid line IDs prove location, not semantic correctness. Retraction, ambiguity,
units, literal-answer and incomplete-output guards remain enabled.

The second protocol stopped at its authorized GPU cutoff on2026-10-03. It has749
actual extraction responses:558RESOLVED/110REVIEW/81SYSTEM_ERROR, plus151BUDGET_STOP
records without completed extraction. RESOLVED is a pipeline status, NOT558 correct
answers. New assisted accuracy is null; the original558/900 (62.00%) is unchanged.
Parent729 actual responses were reused byte-for-byte; continuation added20 and
preserved the interrupted request. Full input grew to7,820,217 tokens, with maximum
122,159+1,024=123,183 within configured131,072. No source was truncated. The complete
long-context GPU limit and universal judge semantic correctness are not certified.

Evidence under `analysis_exports/bottleneck_20261003/`:
`cloud_gemma_segments_parent_final_v2/`, `cloud_gemma_segments_continuation_final_v1/`,
`gemma_segments_final_export_v1/`, `gemma_segments_final_main_verification_v1.json`.
Final independent checks are in `gemma_segments_final_independent_v1.json`.
Both owned Pods were deleted after verified recovery; deletion204, lookup404 and
absence from Pod lists are recorded in `owned_gemma_segments_cleanup_final_v1.json`
and the earlier `owned_pod_cleanup_final_v1.json`. No separate network volume survives.
Earlier literal-protocol evidence remains in `cloud_gemma_final_v8/` and
`gemma_final_export_v1/`. Failure and review records are never silently repaired.
The one-shot `rerun.py` and `continue_runpod.py` preserve the expired execution
deadlines and owned Pod IDs; do not reuse them as generic future launchers.

Prepare the complete stored 900 without installing dependencies, loading a GPU,
or calling any endpoint (from the project root, using the existing Python environment):

```bash
PYTHONPATH=src python -m experiments.gemma_verifier.cli \
  --source analysis_exports/mmmu_20260924/team_final_v8/official_b2/predictions.jsonl \
  --output analysis_exports/gemma_verifier/prepare
```

Outputs are append-only `prepared.jsonl`, `summaries.jsonl`, and an immutable
`run_manifest.json`. Prepared requests contain the full gold-blind extraction
payloads, so they are private benchmark artifacts and must not be published.
Dry-run records have PENDING status and make no scoring or inference claims.

For a separately authorized, already running local vLLM server (or SSH tunnel
terminating at loopback), provide an actual model manifest and pinned revision:

```bash
PYTHONPATH=src python -m experiments.gemma_verifier.cli \
  --source analysis_exports/mmmu_20260924/team_final_v8/official_b2/predictions.jsonl \
  --output analysis_exports/gemma_verifier/live \
  --endpoint http://127.0.0.1:8000/v1 \
  --model google/gemma-4-E4B-it \
  --revision "$GEMMA_REVISION" \
  --model-manifest "$GEMMA_MODEL_MANIFEST" \
  --semantic-open --execute
```

The supplied JSON model manifest must contain `model_id`, `revision` and
`tokenizer_revision` (actual 40-character commit SHAs), `weights_manifest_sha256`,
`chat_template_sha256`, `environment_lock_sha256` (actual 64-character hashes),
`frozen_weights: true`, `dtype`, `max_model_len`, and exact `vllm_version`.
The weights hash should identify a retained file/hash inventory of the checkpoint.
These are operator attestations: `/version` and `/v1/models` establish server and
served-name agreement but cannot prove the loaded revision or weight bytes.
The attestation and its hash are retained, never misreported as endpoint proof.

The direct CLI above is a low-level runtime interface. Use the CPU/smoke-gated
entrypoint below for the900 audit, with fresh private output directories and an
explicit newly authorized UTC deadline. Do not reuse an expired job deadline,
archived Pod process identities, or a smoke gate produced by different code.

```bash
PYTHONPATH=src python -m experiments.gemma_verifier.preflight cpu \
  --source <original900.jsonl> --snapshot <pinned-gemma-snapshot> \
  --output <fresh-cpu-output> --deadline-utc <authorized-UTC-deadline>
PYTHONPATH=src python -m experiments.gemma_verifier.preflight smoke \
  --source <same-original900.jsonl> --cpu-preflight <cpu-output> \
  --model-manifest <actual-model-identity.json> --endpoint http://127.0.0.1:8000/v1 \
  --output <fresh-smoke-output> --deadline-utc <same-UTC-deadline> --execute
PYTHONPATH=src python -m experiments.gemma_verifier.preflight audit \
  --source <same-original900.jsonl> --cpu-preflight <cpu-output> \
  --smoke-gate <passed-smoke-output> --model-manifest <same-model-identity.json> \
  --endpoint http://127.0.0.1:8000/v1 --output <fresh-audit-output> \
  --deadline-utc <same-UTC-deadline> --semantic-open --execute
```

The legacy literal protocol is `gemma-frozen-final-audit-v6-exhaustive-choice-assessment`.
For the separate `gemma-frozen-final-audit-v7-source-segment-citations`, add
`--extraction-mode segments` consistently to CLI and CPU/smoke/audit commands.
Changing extraction mode requires fresh output directories and matching gates.
These Gemma protocol numbers are unrelated to historical Qwen answer-parser versions.
Submission extraction sees only question type, full raw response and finish reason;
not question, choices, gold or V8. Optional MCQ semantic mapping sees the frozen
submitted answer and original choices only. It returns every choice's assessment;
exactly one SUPPORTED with every other choice EXCLUDED is required for a mapped
label. Mapping judgements remain REVIEW. Preserve full actual requests/responses
and usage for extraction, mapping and equivalence in post-run replay and costs.

Only explicit HTTP loopback endpoints with a port are accepted. Environment
proxies and redirects are disabled; no commercial API or credentials are used.
`/tokenize` must report the entire templated chat count; a context overflow is a
system error. No raw-response truncation or context-budget fallback is performed.
Requests use temperature zero, thinking disabled, and strict JSON-schema output.
The server's supported tokenizer/template behavior must be checked during the
separately authorized GPU smoke; this CPU suite does not establish it.

Live output adds `server_identity.jsonl` and `ledger.jsonl`: actual raw HTTP
responses/usage, validated judge JSON, all exact evidence positions, comparison,
timing, input/protocol/model hashes, and attempts. Resume requires matching
source, code, comparison policy, model manifest, and input keys. Successful and
REVIEW records are reused. `--retry-failures` explicitly appends attempts for
SYSTEM_ERROR records; it does not rewrite earlier attempts. A Linux/WSL `fcntl`
single-writer lock is automatically released after a crash; its empty lock file
is retained safely for reuse. An interrupted partial ledger line stops resume;
inspect and preserve that evidence before manual recovery. There is no silent repair.
`--deadline-utc` accepts an absolute UTC timestamp. Each request timeout is capped
by the remaining budget; unprocessed rows are retained as BUDGET_STOP and the
audit remains INCOMPLETE with denominator 900. No score is invented at cutoff.

Gold and V8 answers/correctness never reach extraction. Optional open equivalence
sees only the question, already frozen answer, and decoded reference aliases.
The existing V8 comparator runs first; definite numeric/sign/unit conflicts
cannot be reversed by the equivalence judge. Open answers remain verbatim, and
semantic comparison must retain essential direction/modifiers. All equivalence
judgments and disagreements require REVIEW, including false-negative and
false-positive candidates. Ambiguous/no-final answers remain REVIEW; invalid
JSON/evidence, timeout, output truncation and context errors are SYSTEM_ERROR,
never Qwen wrong. Review flags are safeguards, not a complete semantic parser.
MCQ evidence must ground the returned label through a supported gold-blind
label/content check. Open evidence is checked against complete source lines:
a shortened quote cannot silently drop a sign, unit, or directional modifier.
Unsupported grounding remains REVIEW and cannot be accepted by agreement alone.

Summary denominators remain 900. Partial audits, missing records, SYSTEM_ERROR,
and pending REVIEW cannot produce a final score; even complete agreement is not
proof of judge accuracy. This initial tool has no human-adjudication importer
and never adopts an improved score automatically. Any human-reviewed adoption
requires a separately retained review ledger and policy.

CPU tests use explicitly synthetic judge strings and a fake in-process transport:

```bash
PYTHONPATH=src python -m unittest experiments.gemma_verifier.tests.test_runtime
```

Pure API: `audit_rows(rows, judge_outputs, equivalence_outputs=None, mapping_outputs=None)` returns audit
rows without network/filesystem mutation. `execute(source, output, ..., run=False)`
prepares by default; actual calls require explicit execution and a model manifest.

The vLLM request shape follows its [Gemma 4 JSON-schema and thinking guide](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html).
Local policy follows `claudedocs/gemma4_mmmu_judge_plan_20261002.md` and the Art_22 /
Manage_20 sections of `Qwen3_VL_4B_MMMU_V8_Deep_Audit_20261002.md`.

## Private deployment candidate, 2026-10-03

[deployment.json](deployment.json) pins the official vLLM `0.24.0+cu129` release
wheel by URL/SHA-256 and source commit. Its exact source registers Gemma4 and
supports our chat JSON schema, thinking kwargs, and full-chat `/tokenize` count.
The matched stack is Python3.12, Torch/TorchAudio `2.11.0+cu129`, TorchVision
`0.26.0+cu129`, Transformers `5.5.3`, and huggingface-hub `1.5.0`.
The authorized Pod installation passed native imports and pip check. Installation
alone is not a successful GPU run; the later29-case GPU proof is separate. Retain
both the actual installation receipt/transitive lock and the runtime results.

The newer `0.30.0` candidate pins Torch2.13 with TorchAudio2.11. TorchAudio's
official wheel metadata declares no Requires-Dist, so pip conflict cannot be
asserted from that metadata; native compatibility across those versions is
unproven. The selected older stack keeps Torch and TorchAudio at2.11. All required
packages remain installed; no `--no-deps` or omitted-audio workaround is used.

Install only on the separately authorized Pod, in a new isolated environment,
using the exact direct wheel URL and digest from deployment.json and normal
dependency resolution. Explicitly request the matched CUDA12.9 Torch family,
Transformers5.5.3 and huggingface-hub1.5.0; the official PyTorch cu129 index is
`https://download.pytorch.org/whl/cu129`. A full transitive lock is created from
the actual successful installation; none is fabricated in this source tree.
The launcher requires pip's direct-wheel SHA attestation and checks pinned source
bytes, normally installed dependency versions, pip check and native imports.

After the specified HF revision has been downloaded on the Pod and Qwen has
released the GPU, prepare the command without starting anything:

```bash
bash experiments/gemma_verifier/start_server.sh \
  --model-path /workspace/gemma_snapshot \
  --output /workspace/artifacts/gemma_server_identity
```

Add `--execute` only within the already authorized Pod workflow. That hashes the
actual downloaded checkpoint/tokenizer/config/template bytes, checks one3090 and
a tiny BF16 operation, writes observed `model_identity.json`, file inventory,
resolved environment lock, and environment evidence, then starts a loopback
text-only server. Context remains131072, concurrency1, eager, BF16, image/audio/video0,
thinking false and greedy defaults. It performs no download/install/cloud action.
There is no quantization or shorter-context fallback. Actual smoke demonstrated
the real longest50,454-token input, not an arbitrary131,072-token workload.

`experiments/gemma_verifier/` is the reusable private code unit; keep model weights,
private original responses, generated evidence, credentials and environments outside
it. Future publication of code requires separate approval and public allowlist work;
neither this directory nor its artifacts are currently synchronized to MMDL/GitHub.
`finish_runpod.py` is an archived job-specific controller, not a portable Pod ID or
deadline to reuse blindly for a future experiment.

Before judging, `/tokenize` must measure all900 complete gold-blind payloads,
including template overhead, with this actual tokenizer. Check the longest input
plus1024 output budget, JSON-schema behavior, extraction/equivalence fixtures and
usage/count agreement. Qwen32K tokens and128K architectural support do not prove
the workload fits3090 memory. OOM or incompatible kernels are blockers, not
permission to change precision or truncate responses.

Evidence: [v0.24.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.24.0),
[CUDA dependencies](https://raw.githubusercontent.com/vllm-project/vllm/ee0da84ab9e04ac7610e28580af62c365e898389/requirements/cuda.txt),
[Gemma registry](https://raw.githubusercontent.com/vllm-project/vllm/ee0da84ab9e04ac7610e28580af62c365e898389/vllm/model_executor/models/registry.py),
[pinned model config](https://huggingface.co/google/gemma-4-E4B-it/raw/ee0ef6023621cff504d758262d4e04895a5af4a2/config.json).
