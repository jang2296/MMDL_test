# V8 evaluation contract

V8 is the team's deterministic, gold-blind **final-answer extractor**, not an LLM judge
and not a reproduction of Qwen's complete official evaluation. Its purpose is to fix
known false acceptance paths in V6 while retaining supported, unambiguous answer forms
that V7 lost. Score increases are not the acceptance criterion.

## Frozen scope

- Parser: `team-final-answer-v8`; comparison: `single-final-mmmu-round2-v1`.
- Keep the raw response, question, options, reference answer, termination reason and
  generation metadata unchanged. Extraction receives the question/options but no gold.
- Inspect answer declarations in response order. Check the selected answer's own clause
  and later withdrawal/correction. A later different answer needs an explicit correction;
  unresolved conflicts/alternatives fail closed. An invalid latest declaration cannot be
  rescued by searching earlier reasoning.
- Normalize supported Markdown, checkmarks, mathematical delimiters and boxed notation
  on a copy. A self-answer blockquote may be an answer; an attributed quotation is not.
- If no answer declaration exists, allow a terminal standalone answer, boxed answer,
  uniquely matching option content or an explicitly supported conclusion grammar.
  Do not interpret arbitrary capital letters, negation or `or` inside an option as
  additional/withdrawn labels. Exact option text does not override a later withdrawal.
- Open answers are a **single value**. Compare a whole normalized string, not a substring;
  do not test every number in the reasoning. Gold lists retain dataset alias semantics,
  not ordered-pair semantics. General tuples, symbolic equivalence and unit conversion
  are unsupported. Limited unit/step/point interpretation must derive from the question,
  not the gold. An explicit incompatible unit must not match a unitless numeric gold.
- Numeric comparison retains pinned MMMU's float/two-decimal normalization. Exact-decimal
  comparison is a different policy and is not silently introduced here.
- A named final statement such as `The value is 65` or `Ratio = 30` may identify
  one supported right-hand value. Nested statement parsing is capped at eight;
  longer or non-shrinking forms abstain rather than recurse indefinitely. Qualified
  strings such as `2960 unfavorable` are not silently reduced to a magnitude.
- `finish_reason=length` is not by itself a rejection. A completed answer may survive a
  truncated explanation, but an observed withdrawal/unresolved alternative may not.
- No V2/V6/V7 fallback, random selection, LLM call, model regeneration or per-item exceptions.
  `EMPTY` and `NO_PARSE` score zero within the full 900 denominator, separately from wrong
  extracted answers. Correct choices do not certify the accompanying reasoning.

## Source and validation boundaries

The pinned MMMU source is
[`eval_utils.py` at 268471d0d488258990025331c7528359c324aa25](https://github.com/MMMU-Benchmark/MMMU/blob/268471d0d488258990025331c7528359c324aa25/mmmu/utils/eval_utils.py)
(Apache-2.0). Only its numeric/string normalization primitives are appropriate here;
its open-response multi-candidate extraction and substring acceptance are not V8's
comparison policy. `scorer_source.json` and the adjacent source snapshot identify the
actual executable dependencies; a Git HEAD preceding these changes is not a V8 commit.

The 33 external proposed tests in `docs/P1` require adaptation: their `question_schema`
does not exist in this pipeline; units/step numbers use ordinary question text; ordered
pairs are unsupported; P17 uses round-to-two-decimals, not exact-decimal equality.
These adaptations must be visible in the runnable tests.

The 64 previously inspected V6/V7 disagreements are **development regression data**,
not an independent holdout. Check the 46 clear V7 misses, two step-policy differences,
12 ambiguous/truncated cases and four valid V7 gains separately. Additional synthetic
positive/negative pairs, gold-permutation checks, full saved-response replay, source
hashes and score arithmetic are complementary checks; none proves perfect semantic
judgment for arbitrary language.

## Reproduction

```bash
# CPU only; original inference artifacts remain immutable. Use a new output directory.
bash scripts/rescore.sh --artifact-root "$MMDL_ARTIFACT_ROOT" \
  --run-id "$MMDL_RUN_ID" --parser team-final-answer-v8 --output-dir "$NEW_RESCORE_DIR"
python -m scripts.validate_rescore \
  --source-run "$MMDL_ARTIFACT_ROOT/runs/$MMDL_RUN_ID" --scored-dir "$NEW_RESCORE_DIR"
python -m scripts.report_baseline \
  --run-dir "$MMDL_ARTIFACT_ROOT/runs/$MMDL_RUN_ID" \
  --scored-dir "$NEW_RESCORE_DIR" --output Assignment_1.md
```

The Assignment renderer accepts a complete continuous-32K or high-resolution B2 run,
independently verifies the CPU scoring overlay, and keeps time/memory measurements
from the original GPU inference. The rescore's text-only `review.html` contains raw
answers and scoring metadata; images remain in the original recovered bundle.

To repeat the complete saved-run development audit (no new GPU calls):

```bash
python -m scripts.audit_v8 \
  --v6-root analysis_exports/mmmu_20260924/v6_v7_audit/v6_replayed \
  --audit-root analysis_exports/mmmu_20260924/v6_v7_audit \
  --output "$NEW_V8_SUITE_DIR"
```

The supplied V6 provenance files locate the three original recovered inference runs.
The command refuses an existing destination and verifies source/scorer stability,
2700 gold-permutation checks, per-run replay and historical-regression records.
`v6_comparison.json` compares V8 with V6; `comparison.json` instead compares V8 with
the **original inference scorer**, which was not V6. Do not interchange those totals.

For **future, separately authorized inference**, `configs/eval/mmmu_val_v8.yaml` uses
the same model/data/P0/sampling/32768/high-resolution/continuous-two-request settings
as B2, with V8 in the online scorer. Older frozen configs retain their original parsers.
CPU tests of this integration are not a new GPU or RTX4090 validation.
