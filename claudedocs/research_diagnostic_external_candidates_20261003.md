# External candidates for conditional diagnostic follow-up

Private research, not a training authorization or final ranking. Official creator sources accessed **UTC 2026-10-02 20:22:45–20:23:53** (KST October 3); exact URLs appear inline. No datasets downloaded, parameters updated, or cloud actions performed. Source/task descriptions have high confidence; transfer hypotheses and contamination cleanliness remain unverified.

Main-agent cross-check: creator repositories, AI2 creator page/original paper, and FinQA LICENSE were independently opened and key annotation/license/leakage statements checked at UTC20:26 on October2. This is source verification, not a dataset-rights clearance or independent contamination audit. This note and its diagnostic evidence references are not training inputs.

Evidence boundary: [diagnostic report §3/4/6](Qwen3_VL_4B_Bottleneck_Diagnostic_Report_20261003.md) records normal, schema-valid P outputs but observed structural omissions/errors; Music_3 omitted a ledger-line note position and invented a natural. O1 improved Music_3/25 and Computer_Science_19, but worsened Music_26. Accounting_19 recovered under O0/M and regressed under MI. These are fixed-stress observations, not population estimates, certified semantic metrics, or identification of a faulty Vision/Merger/Decoder module. The four sources below are external in provenance; **none is certified independent-clean**.

## MUSCIMA++: notation primitives and attachments

The [creator repository](https://github.com/OMR-Research/muscima-pp) provides handwritten score annotations: symbol classes, bounding boxes, pixel masks, and directed attachment edges among noteheads, stems, beams, accidentals, and composite notation. XML notation graphs are not ordinary question–answer labels; some staff annotations are automatically inferred. [Creator project page](https://ufal.mff.cuni.cz/muscima).

Conditional fit: independent symbol-position/attachment supervision could test the Music_3 omission/invention hypothesis. It does not establish that printed-score errors transfer from handwriting, fix musical reasoning, resolve Music_15 answer-format conflict, or show which module requires tuning. Annotation license is **CC BY-NC-SA 4.0**, with MUSCIMA++ and CVC-MUSCIMA attribution. Underlying CVC images are distributed separately; recheck image permissions and obligations for the chosen release rather than treating repository availability as blanket permission.

Duplicate gate: compare original/cropped/deskewed score images and musical-content fingerprints against all forbidden MMMU/MMMU-Pro material; different handwriting of the same score is not independent content. Preserve writer-independent external splits and composition families.

## AI2D: diagram constituents and relations

[Ai2’s creator page](https://prior.allenai.org/projects/diagram-understanding) describes raster science diagrams, constituent segmentations, text/diagram elements, and QA. The [original paper §3/6](https://arxiv.org/html/1603.07396) defines blobs, text boxes, arrows, arrowheads, and directed diagram-parse relationships; it distinguishes syntactic linkage from its domain meaning. Prefer original relation annotations over a QA-only mirror.

Conditional fit: relation-supervised extraction could probe the reported P structural gaps and Computer_Science_19’s O1 recovery. That recovery alone does not establish its cause or AI2D’s suitability; grade-school science diagrams are not universal graph/algorithm supervision. Neither annotations nor QA uniquely diagnose a merger versus decoder defect.

License: **not verified from the creator page/paper**. Do not transfer AI2D-RST’s license or a third-party mirror’s tag to original AI2D. The paper reports Google-image collection; distribution terms and underlying image rights need explicit review before acquisition/training. Duplicate gate: source URLs, exact/perceptual image matches, crops, OCR text, and diagram-relation fingerprints; web textbook diagrams may recur across benchmarks. Reserve creator test IDs.

## MedMNIST: specialized visual classification, not VQA

The [creator README](https://github.com/MedMNIST/MedMNIST) describes standardized 2D/3D biomedical images with binary, multiclass, multilabel, or ordinal labels; sizes include 28 and larger variants. These are **classification labels**, not questions, clinical explanations, OCR, or object-relation annotations. See [official subset metadata](https://raw.githubusercontent.com/MedMNIST/MedMNIST/main/medmnist/info.py).

Conditional fit: a modality-matched external subset could test specialized feature discrimination **only if later diagnostic evidence establishes that need**. The reviewed stress-report sections establish no MedMNIST-like classification failure; general P omissions are insufficient. It cannot directly repair music, numerical reasoning, clinical VQA, or certify clinical use.

README data terms: **CC BY 4.0 except DermaMNIST CC BY-NC 4.0**; code is Apache-2.0. Verify exact subset/release terms and original-source attribution separately. Duplicate gate: underlying source datasets, patient/slide/volume identifiers where available, near-image matching, and modality-specific families; resized images must not evade exclusion. Missing provenance blocks cleanliness certification.

## FinQA: quantitative reasoning over text and tables

The [creator repository](https://github.com/czyssrs/FinQA) supplies financial-report text, structured tables, questions, reasoning programs, supporting-fact indices, and execution answers in JSON. It is **not native raster-table/VLM image-input supervision**. Conditional fit: evidence selection and executable arithmetic supervision could test a reasoning hypothesis suggested by Accounting_19’s O0/M recovery. It cannot diagnose the observed MI regression, establish OCR improvement, or distinguish accounting knowledge from computation without a controlled pilot.

The [creator LICENSE](https://raw.githubusercontent.com/czyssrs/FinQA/main/LICENSE) is MIT and refers to software/documentation; it is not proof of blanket rights over underlying corporate reports. Confirm dataset/report reuse scope. The README documents a historical table-format leakage bug; freeze corrected preprocessing, not gold-retrieval inputs disguised as inference evidence.

Duplicate gate: report/company/year IDs, normalized question/table/text matches, program templates, and any overlap with benchmark questions. Group by original report; preserve external held-out splits.

## Decision gate

After the actual pilot/coverage and semantic-review evidence closes: choose a specific hypothesis, verify release/license and exact/near-duplicate exclusions, then seek approval for a small independently sourced pilot. Benchmark data/oracles/responses remain **training_allowed=false** and may supply exclusion signals only. No selective-module effect, clean-data claim, or fine-tuning benefit is established here.
