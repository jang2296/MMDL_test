# Optional local data audit

This is an explicit entrypoint to the existing `notebook_rl` audit tools. Importing
this package or starting training does not run an audit. No detector, threshold,
review requirement or approval policy is replaced here.

Run the audit locally before freezing a training bundle. The benchmark index,
native benchmark images, OCR/embedding caches and review packets are private
exclusion evidence; they must not be copied into the training bundle or published.
Benchmark material is an exclusion signal only, never training input.

## Prerequisites

Use the project's existing pinned Python environment (`env/requirements-proof.lock`
for the established proof environment); this wrapper installs nothing. Full local
auditing additionally needs the existing scanner's pinned Tesseract 4.1.1 tree,
SSCD TorchScript artifact, SigLIP snapshot, calibration and complete benchmark
inventory/cache. Their paths are explicit arguments to `exclusion engines`.
The pinned inventory covers 100 files and 16,740 records; missing coverage blocks
approval. GPU use by the audit engines requires their explicit `--allow-gpu` flag.
No GPU, download, paid service or cloud run is started by this wrapper itself.

The professor may rerun these tools with their own private benchmark cache and
external candidate data. Use existing valid caches instead of downloading again.
The index builder defaults to a read-only plan; its execution is a separate choice.

## Existing CLI sequence

The arguments after the first command are forwarded unchanged. Inspect each
tool's help for all required paths and the current fixed options:

```bash
python -m fine_tuning.data_audit index --help
python -m fine_tuning.data_audit exclusion --help
python -m fine_tuning.data_audit review --help
python -m fine_tuning.data_audit pool --help
python -m fine_tuning.data_audit export-approval --help
```

1. `index`: build/verify the private native benchmark index using the pinned
   inventory. This does not create OCR, embeddings, calibration or approval.
2. `exclusion registry`, then `exclusion engines`, then `exclusion assemble`:
   use the existing full-coverage checks and fixed detector identities.
3. `review sample`, `review sheets`, `review compile`, `review check`: compare
   actual questions/images and record DISTINCT/DUPLICATE/UNRESOLVED decisions.
   Never automatically turn unreviewed candidates into DISTINCT.
4. `exclusion apply --assemble-dir ... --decisions ... --output ...`: apply the
   review; DUPLICATE/UNRESOLVED families remain excluded.
5. `pool freeze`, `pool materialize`, `pool verify`: preserve actor, verifier,
   lineage and image bytes with their manifest and union-receipt bindings.
6. `export-approval --build ... --approval ... --output ...`: perform the existing
   fresh local approval export. Record its printed SHA-256 outside the bundle.

Each existing tool preserves its own failure status and refusal to overwrite
outputs. `FROZEN` alone is insufficient: the existing freeze can retain shortages,
and order construction can drop repeated structures. Actual 3,500-item selection,
all warnings and any supply shortage still require local verification.

## Runtime integrity check

`fine_tuning.error_note.frozen_data.verify_frozen_data` reuses
`local_run.frozen_approval_output` (`frozen-approval-verify-v1`) and
`pool_freeze.verify`. The caller supplies the pre-registered approval and order
SHA-256 values from its clean, source-bound job:

```python
from pathlib import Path
from fine_tuning.error_note.frozen_data import verify_frozen_data

verification = verify_frozen_data(
    Path(build_root),
    {"path": str(Path(approval_path).absolute()), "file_sha256": expected_approval_sha256},
    Path(order_path),
    order_sha256=expected_order_sha256,
    expected_pilot_sha256=expected_pilot_ids_sha256,
)
```

It checks approval/code/union/manifest/record/image hashes, exactly 3,500 unique
INCLUDED train IDs, identical approved and ordered sets, no held-out or excluded
order IDs, and a 300-item pilot prefix. Held-out records may remain materialized;
the total materialized count is not the train count. The canonical ID digest is
`sha256("\n".join(ids).encode()).hexdigest()`.

The externally SHA-pinned order uses the existing order v1/v2 format. Its optional
`selection` object supports:

- `pilot_ids_sha256`: the pilot prefix digest; required if the caller does not
  supply `expected_pilot_sha256`. If both exist, both must match.
- `cluster_train_counts`, `cluster_pilot_counts`: exact counts checked against
  each item's manifest category, also matching order `cluster` when present.
  The legacy mode derives categories from `category_map[subject]`; explicit
  selection instead binds each ID's category in its pinned selection receipt.
- `source_train_counts`: exact counts derived from manifest sources.
- `source_caps`: declared per-source maximum counts, checked against those sources.

Selection quotas must be fixed locally before model generation. Reuse/new quotas
refer to original item lineage, not source names; verify them in the local
selection receipt. A source can supply both reused and new items.

For explicit selection, `pool freeze --selection <receipt.json>
--selection-sha256 <expected-sha256>` requires the `pool-freeze-selection-v1`
receipt documented in `notebook_rl/pool_freeze.py`. Its bytes are canonical
`dataset._bytes(body) + b"\n"`. It binds the reviewed shard identities and each
selected ID/category; it cannot admit excluded or held-out items. Unselected
eligible items remain `EXCLUDED_QUOTA`. Runtime validation recomputes the pinned
receipt hash and rejects missing or inconsistent selection markers. Creating
this receipt does not establish that the actual 3,500-item quotas were met.

Runtime needs the approved external data and minimal manifests/receipts only;
benchmark index, original shards, OCR and embedding artifacts may be unavailable.
Keep all `notebook_rl` Python source: `approval.code_hashes` still checks it.
`error_note` and this entrypoint lie outside that source-hash scope and must be
bound separately by the job's clean commit. A self-signed hash recomputed inside
an untrusted bundle is not an external expected value. Changed data or audit
policy requires a new local audit/freeze, never a generic skip switch.

A passing runtime integrity check does not grant paid execution or parameter
updates, certify human review, or prove absolute absence of contamination.
CPU fixtures are guard tests, not actual-data selection or model evidence.
