# Private I/O0 transition evidence audit — 2026-10-03

The 14 supplied pairs separate into five capped I responses that gain a final marker under O0, two capped O0 responses that lose one, six pairs with different explicit final letters, and one unchanged answer commitment with a final-marker change. These are descriptive observations from a selected subset, not correctness or improvement counts.

`BENCHMARK_DERIVED`; `training_allowed=false`; `AI_REVIEWED_NOT_HUMAN_VERIFIED`; `human_verified=false`; `runtime_metadata_verified=false`. The tools do not expose serving-model verification for this reviewer.

Source: [gold-blind packet](../analysis_exports/bottleneck_20261003/pilot60_O0_I_transition_gold_blind_packet_v1.jsonl), 195,974 bytes, SHA256 `a785638b8226c9a3d88c0ea1360c0efb22bbe8b0bcf7d0d720c11e7e1e862696`. All 14 upstream-selected CPU-v2 score-change pairs appear exactly once. Gold, score direction, V8 verdicts, scoring artifacts and other conditions were not inspected. Questions were not solved or re-answered.

The existing `input_builder.build_request` path passes images for I and passes serialized AI-reviewed facts without images for O0. This is image-to-text replacement; IO0 is the separate images-plus-facts condition. No inference was run and no oracle or scorer was edited. Images retain their original descriptors/roles/order in the JSON; none were viewed because all visual observations below concern what the responses state, not pixel truth.

| Descriptive primary category | Cases |
|---|---:|
| I capped without final marker → O0 final marker | 5 |
| I final marker → O0 capped without final marker | 2 |
| Distinct finals with changed stated visual interpretation | 4 |
| Distinct finals with changed rule/calculation statements | 2 |
| Same B commitment, final-marker format changes | 1 |

The [machine-readable review](../analysis_exports/bottleneck_20261003/pilot60_O0_I_transition_evidence_review_v1.json) contains all 70 exact quotes, per-condition finish/token/source hashes, original image descriptors and per-case limits. `NO_FINAL_MARKER` describes literal marker absence, not a replayed scorer verdict. Offsets below are 0-based Python characters, end-exclusive, in `responses.<condition>.raw_response`.

## validation_Accounting_6

Observed commitments: I `NO_FINAL_MARKER; repeated alternatives` → O0 `A`. I repeatedly assigns 3,409 to AA3 in January and cycles over unmatched balances. O0 assigns zero January cost to AA3 and eventually commits A, after continued accounting-rule reversals. This is changed stated cell binding plus completion, not a clean rule-only recovery.

I E01, `responses.I.raw_response[4813:4876]`:

> AA3: incurred Jan cost — not completed until April — so in WIP.

O0 E02, `responses.O0.raw_response[8731:8792]`:

> But what about AA3: January: $0 — so no cost — so not in WIP.

Limit: Neither stated table is certified from pixels here. O0's earlier WIP/finished-goods reversals remain evidence of unresolved reasoning instability; an eventual marker does not certify the explanation.

## validation_Biology_21

Observed commitments: I `C. Nucleus` → O0 `B. Plasma membrane`. The declared identity of the B-labeled structure changes from nucleus to membrane. O0 explicitly uses the supplied surface/endpoint wording to justify its different final commitment.

I E01, `responses.I.raw_response[0:124]`:

> The part labeled 'B' is the nucleus, which is the large, purple, spherical structure containing the cell's genetic material.

O0 E02, `responses.O0.raw_response[0:188]`:

> The label 'B' points to the "outer purple surface at upper-right of purple body," which is consistent with the plasma membrane, as it is the outermost boundary of the cell structure shown.

Limit: The label endpoint and cell-part classification were not independently checked against pixels; the quoted description is evidence of the model's interpretation, not anatomical truth.

## validation_Biology_24

Observed commitments: I `B` → O0 `C`. I binds a dark-background fluorescent-looking description to the second option image; O0 binds the supplied black-background/bright-spots wording to Image 3. This is an option-image identity/classification change in the response.

I E01, `responses.I.raw_response[146:335]`:

> The second image displays brightly colored, labeled structures against a dark background, which is characteristic of fluorescent microscopy used to visualize specific molecules or proteins.

O0 E02, `responses.O0.raw_response[138:301]`:

> Image 3 shows “multicolored bright small spots on black background; broad diagonal clusters,” which is consistent with fluorescent labeling of cellular components.

Limit: All three descriptors have option role and original order 0/1/2. No pixels were inspected; no microscopy classification is adjudicated. The change cannot establish which image the vision path actually represented internally.

## validation_Computer_Science_7

Observed commitments: I `NO_FINAL_MARKER; A/C alternatives` → O0 `C`. Both responses list the same occupied table indices. I repeats an A/C ambiguity while its A trace skips a slot it calls free. O0's A trace places 52 at the free slot and rejects A, then emits C. The observable distinction is procedural trace/termination, not a changed stated table.

I E03, `responses.I.raw_response[8985:9089]`:

> 52 tries 2 (occupied), 3 (free), 4 (occupied), 5 — 52 at 5. But 3 is free at that time, so 52 goes to 5.

O0 E04, `responses.O0.raw_response[734:854]`:

> - 52 → h=2 → collision at 2 → probe 3 (free), put 52 at 3? But in final state, 52 is at 5 — contradiction. So A invalid.

Limit: No insertion order is independently solved or graded here. I's prose mentions possible answers but has no final marker; interpreting that as a definite prior answer would overstate the evidence.

## validation_Materials_17

Observed commitments: I `C. 121 psi` → O0 `NO_FINAL_MARKER; repeated force hypotheses`. I uses a moment equation and emits C. O0 starts with a resultant/load-balance strategy, repeatedly returns to 5/0.30 and later considers moments, but hits the cap without a final marker. I calls 75 degrees relative to vertical; O0 calls it relative to the forearm/horizontal.

I E03, `responses.I.raw_response[673:715]`:

> F_B * sin(75°) * 2 in. - 5 lb * 14 in. = 0

O0 E04, `responses.O0.raw_response[995:1091]`:

> Tensile stress = Force / Area = 5 lb / 0.30 in² ≈ 16.67 psi — but this is not among the options.

Limit: No force solution or angle truth is certified. The O0 cap confounds any simple answer-change claim; it includes both changed geometry wording and changed mechanics strategy.

## validation_Materials_19

Observed commitments: I `NO_FINAL_MARKER; unmatched strain alternatives` → O0 `A`. Both use change in diagonal length divided by original length. I initially uses only horizontal A and vertical C displacement, then cycles over vertical sign choices. O0 states both horizontal and vertical components at A and C and uses those altered endpoints before emitting A.

I E01, `responses.I.raw_response[298:403]`:

> - The horizontal displacement at A is 3 mm to the right.
> - The vertical displacement at C is 2 mm upward.

O0 E02, `responses.O0.raw_response[204:363]`:

Exact quote, with JSON-escaped newlines:

`"Displacement of C: 2 mm up and 2 mm right → displacement vector ΔC = (2, 2) mm  \nDisplacement of A: 3 mm right and 2 mm up → displacement vector ΔA = (3, 2) mm"`

Limit: The supplied displacement facts are AI-reviewed, not certified geometry. This distinguishes changed calculation inputs from a pure arithmetic or output-format effect; no strain answer is independently computed.

## validation_Materials_2

Observed commitments: I `C` → O0 `B`. Both repeatedly question the angle convention and compare formulas to options. I finishes with a sin(theta)cos(theta) expression; O0 finishes with sin-squared(theta). The raw evidence is a changed derivation/final choice, without a demonstrated correction of an image reading.

I E01, `responses.I.raw_response[6721:6811]`:

> So normal stress = [P sin(θ)] / [A / cos(θ)] = P sin(θ) cos(θ) / A = (P/A) * (1/2) sin(2θ)

O0 E02, `responses.O0.raw_response[3139:3188]`:

> σ_θ = (P / A) * cos²(90° - θ) = (P / A) * sin²(θ)

Limit: No stress formula is endorsed. Both contain self-revisions; the transcript cannot isolate whether the convention change came from visual interpretation, text priming, or reasoning.

## validation_Materials_23

Observed commitments: I `D. 11.9 mm` → O0 `NO_FINAL_MARKER; repeated diameter alternatives`. Both start with the same sine-based horizontal equilibrium and force-ratio relation. I proceeds through area ratio to a diameter and emits D. O0 repeatedly recomputes tensions/units, states d-squared=100sqrt(2) followed by d=10sqrt(2), and reaches the cap. A different initial visual interpretation is unnecessary to describe this divergence.

I E03, `responses.I.raw_response[1297:1365]`:

> Then diameter d_AB = √(4*A_AB/π) = √(4*110.9/π) ≈ √(141.5) ≈ 11.9 mm

O0 E04, `responses.O0.raw_response[2236:2318]`:

> So √2 = d_AB² / 100 → d_AB² = 100√2 → d_AB = 10·√2 ≈ 14.14 mm — not among options.

Limit: Only the statements are compared; no cable diameter is solved. O0 later tries alternate angle conventions, so its whole trace is not geometry-invariant. Capped output cannot be treated as a distinct committed final answer.

## validation_Materials_30

Observed commitments: I `NO_FINAL_MARKER; zero-strain/geometry alternatives` → O0 `A`. I repeatedly identifies x with a fixed bottom/base edge and cycles over zero strain and inconsistent triangle lengths. O0 identifies x with the lower-side extension and projects the horizontal displacement along that direction before emitting A.

I E01, `responses.I.raw_response[2091:2138]`:

> The x-axis is the original axis along the base.

O0 E02, `responses.O0.raw_response[81:162]`:

> The x-axis is defined as the lower-left direction along the lower-side extension.

Limit: No axis geometry or strain result is certified. O0 also calls A a top vertex although the supplied oracle says right apex, so a final marker does not imply complete adoption of all supplied geometry.

## validation_Music_10

Observed commitments: I `B. False; no FINAL_ANSWER marker` → O0 `B`. I is the complete short response 'B. False'; O0 adds an explanation and a FINAL_ANSWER: B marker. The answer commitment is the same. This is a format-contract candidate rather than evidence of answer recovery or a changed prior visual reading.

I E01, `responses.I.raw_response[0:8]`:

> B. False

O0 E03, `responses.O0.raw_response[251:266]`:

> FINAL_ANSWER: B

Limit: I supplies no reasoning, so its interpretation cannot be compared. O0's musical assertions are not adjudicated. No scorer path was inspected to assign the exact reason this pair was selected.

## validation_Music_15

Observed commitments: I `D` → O0 `A`. I states bass clef and second staff line; O0 states C-clef and a note below a ledger line. The final letters change with those stated symbol/position bindings.

I E01, `responses.I.raw_response[0:119]`:

> The note shown is a whole note (or semibreve) on the bass clef, which is positioned on the second line from the bottom.

O0 E02, `responses.O0.raw_response[0:87]`:

> The note is a hollow oval without a stem, placed below a ledger line beneath the staff.

Limit: No clef/note truth or musical reasoning is certified. O0's note naming remains its own inference, not an oracle-provided answer.

## validation_Physics_13

Observed commitments: I `C. 2 s` → O0 `B. Between 1 and 2 s`. Both use zero net displacement/area under the velocity-time graph. I asserts cancellation by t=2 without numeric areas; O0 states different interval area magnitudes and nonzero net area at t=2, then commits a between-times answer. This is changed quantified graph/area interpretation under the same criterion.

I E01, `responses.I.raw_response[327:447]`:

> by t=2, the total area from 0 to 2 is zero (the negative area from 0 to 1 is canceled by the positive area from 1 to 2).

O0 E02, `responses.O0.raw_response[513:578]`:

> Net displacement from t=0 to t=2 s: -0.5 + 1 = +0.5 m → not zero.

Limit: The graph shape/areas are not independently certified. The text does not show whether I had the same internal numerical graph representation, so calculation versus visual-reading responsibility remains unresolved.

## validation_Physics_7

Observed commitments: I `D` → O0 `E`. Both invoke foci on the major axis. I calls B the center, then entertains D under a vertical/diagonal-axis possibility. O0 calls B left of center on a horizontal midline, C near center, and rejects D as off the major axis before committing E.

I E01, `responses.I.raw_response[309:346]`:

> the Sun cannot be at point B (center)

O0 E03, `responses.O0.raw_response[515:648]`:

> Point B is on the left side of the ellipse on the horizontal midline — this could be one of the foci if the major axis is horizontal.

Limit: No focus location is independently solved. O0 still uses plausibility language; changed point/axis interpretation is not a certification of geometric inference.

## validation_Public_Health_9

Observed commitments: I `NO_FINAL_MARKER; multiple specificity candidates` → O0 `A`. I already states all four numeric rows, but repeatedly switches diagnostic-category labels and sometimes uses only one row. O0 also initially uses one row and cycles, then explicitly pools the two electronic-negative/non-lesion rows and the two positive rows before emitting A. The evidence supports category binding/pooling and completion, without requiring newly recovered numbers.

I E02, `responses.I.raw_response[13702:13832]`:

> - True Negative: cervical lesions and colposcopy - = 5 + 10 = 15
> - False Negative: non-cervical lesions and colposcopy - = 220 + 2

O0 E03, `responses.O0.raw_response[3633:3774]`:

Exact quote, with JSON-escaped newlines:

`"True Negative = non-lesions with electronic colposcopy negative = 465  \nFalse Positive = non-lesions with electronic colposcopy positive = 35"`

Limit: No diagnostic answer is independently computed or judged. Some row assignments vary within I, and O0 also self-corrects. This selected case cannot prove that text transcription universally fixes table understanding.

## Validation and interpretation limits

The read-only check below verifies packet SHA/size, exact ordered coverage of 14 unique cases, all 28 canonical response hashes, all 70 quote slices/bounds/source references, stored finish/token/marker metadata, and the descriptive counts. It changes no source artifact. No lint/typecheck is applicable because the assigned outputs are JSON and Markdown, with no implementation change.

```bash
PYTHONPATH=src python3 - <<'PY'
import hashlib, json
from pathlib import Path
from experiments.bottleneck_diagnostic.input_builder import canonical_hash
root = Path('analysis_exports/bottleneck_20261003')
packet = (root / 'pilot60_O0_I_transition_gold_blind_packet_v1.jsonl').read_bytes()
a = json.loads((root / 'pilot60_O0_I_transition_evidence_review_v1.json').read_text())
assert len(packet) == a['source_packet']['bytes']
assert hashlib.sha256(packet).hexdigest() == a['source_packet']['sha256']
rows = [json.loads(line) for line in packet.splitlines()]
assert len(rows) == len(a['cases']) == 14
assert len({c['sample_id'] for c in a['cases']}) == 14
assert [r['sample_id'] for r in rows] == [c['sample_id'] for c in a['cases']]
quotes = 0
for row, case in zip(rows, a['cases']):
    assert case['training_allowed'] is False
    assert case['provenance'] == 'BENCHMARK_DERIVED'
    assert case['runtime_metadata_verified'] is False
    assert case['verification_kind'] == 'AI_REVIEWED_NOT_HUMAN_VERIFIED'
    assert case['original_image_descriptors'] == row['images']
    assert case['source_oracle_hash'] == row['oracle_O0']['oracle_hash']
    for condition, response in row['responses'].items():
        raw = response['raw_response']
        assert canonical_hash(raw) == response['response_hash']
        meta = case['source_responses'][condition]
        assert meta['raw_char_count'] == len(raw)
        assert meta['final_marker_present'] == ('FINAL_ANSWER:' in raw)
        assert all(meta[k] == v for k, v in response.items() if k != 'raw_response')
    for evidence in case['evidence']:
        raw = row['responses'][evidence['condition']]['raw_response']
        assert evidence['field_path'] == 'responses.' + evidence['condition'] + '.raw_response'
        assert 0 <= evidence['start'] < evidence['end'] <= len(raw)
        assert raw[evidence['start']:evidence['end']] == evidence['quote']
        assert evidence['source_response_hash'] == canonical_hash(raw)
        if evidence['evidence_kind'] == 'CAPPED_RESPONSE_END':
            assert evidence['end'] == len(raw)
        quotes += 1
assert quotes == a['descriptive_tally']['evidence_quotes']
assert sum(a['descriptive_tally']['primary_categories_mutually_exclusive'].values()) == 14
print('PASS: 14 cases, 28 response hashes,', quotes, 'verbatim quote spans')
PY
```

AI-reviewed O0 is not human-certified truth. Some cases change stated geometry; others already contain the same table/initial equations and change rule use, pooling, algebra, or completion. The Music_10 pair has too little I reasoning to compare strategies. These distinctions are candidates for diagnosis, not module assignments. Sampling, modality replacement and termination remain entangled; text recovery cannot establish Vision/Merger causality. This one-repetition, score-selected packet cannot provide an error prevalence or whole-pilot semantic accuracy.
