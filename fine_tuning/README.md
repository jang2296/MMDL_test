# Selective fine-tuning experiment — MMDL_test

## 현재 구현과 실제 검증 범위 (2026-10-28)

최신 로컬 경로는 실제 Qwen3-VL-4B의 동결 NF4 기저와 학습 가능한 어댑터·4개 Merger를
사용한다. `ErrorCheck` → `RevisedSolution` → 전체 이력 최종노트 → 독립 검증 → 원래 생성
토큰에 대한 RL로 연결한다. 정답은 검증기에만 있고, SFT·교사 풀이 모방은 사용하지 않는다.
재시도 직전의 미검증 재관찰은 한 번만 생성해 재시도와 최종노트에 공유하며 학습 대상에서 제외한다.

동결 소스10의 실제 최소2문항은 업데이트1회·624개 학습 텐서 저장과 새 프로세스 재로딩을
통과했다. optimizer/step 및 단일·복수 이미지의 실제 forward·생성 일치, 동결 기저 보존과
원본 가중치 접근0을 확인했다. 이는 기능 검증(F)이다. 해당 시험의 검증된 오류·수정 노트
토큰 학습 신호는0이므로 노트 수정학습(S)이나 자기개선 효과(E)의 증거는 아니다.
근거는 비공개 `verified-repair-notebook-v59/training-reload-closure-main-v10.json`
(SHA256 `c30bfa2deda64327822af8c012c3aeb72c73bd47ae1b3fecafbc6387956e2464`)이다.

같은 프로토콜의 고정10문항 탐색은 진행 중이며 전체 종료·새 재로딩은 아직 미완료다.
정확한 최종노트 학습, 현재 프로토콜의 Vision 포함/동결 양 경로, 별도 문항 효과도 남아 있다.
노트의 오류·수정 내용을 먼저 작성하는 비교 후보는 CPU 준비만 완료했으며 생산 정책에 적용하지
않았다. 작은 기능시험에서 성능 상승을 필수로 요구하지 않지만 기능 통과를 효과로 해석하지 않는다.
현재 명령·계약은 [notebook_rl/README.md](notebook_rl/README.md)를 따른다.
아래는 과거 단계의 기록이며 옛 미완료 표기를 현재 상태로 읽지 않는다.

## Historical redesign status (2026-10-27)

The corrected actual fixed-two Qwen trial completed one answer-reward update,
saved its checkpoint and passed a fresh-process reload of624 trainable tensors,
optimizer/step/cursor and real single/paired-image forward/generation probes.
The frozen NF4 base stayed unchanged and original-weight access was blocked.
This establishes functional learning/storage/reload (F); verified native repair
learning (S) and benefit (E) remain false. All four final notes omitted required
success metadata and submitted no repair claims. Neither saved history contained
a native violated-to-compliant descendant; answer completion is a separate event.

The subsequent no-update original/clarified retry comparison reproduced the
original outputs exactly. Asking for fresh complete observations recovered18
submitted observations but only17 were correct; the wrong relation persisted.
Another response exposed an ASCII blank-line parser defect. The narrow boundary
fix preserves raw spans and semantic checks under a new explicit protocol pin;
382 connected CPU tests, Ruff and scoped strong mypy on52 files passed. It has
not yet had a new model run. The next two cases are fixed before generation for
source-only diagnosis, without final-note generation or an optimizer. Fixed-ten
training, current-route verification and independent benefit evaluation remain
pending; these diagnostics do not establish notebook learning or MMMU improvement.

An explicit `local_train train --admission verified-functional-repair-exploration-v1`
now separates permission for the fixed-ten local learning experiment from proof
that native repair learning already happened. It requires a current-source actual
minimum update/save/fresh reload (F), carries the observed S value unchanged, and
keeps strict admission/defaults, all optimizer signal checks and fixed-ten budgets.
The four-file change passed focused CPU checks and independent review. Combined
regression and a new current-source minimum are still required before execution;
the older v6 F receipt cannot certify the new source/profile. S=false remains
unverified repair learning, and this entry policy permits no large-scale run.

The verified-repair redesign is being implemented in this standalone package.
It separates a model-authored retry `ErrorCheck` from its actual `RevisedSolution`,
then checks whether the proposed local correction was really performed. A final
notebook will cover the complete recorded history while submitting at most three
explicit, independently checked relationships. This is bounded claim verification,
not a claim to understand every sentence or diagnose an internal module.

The codec, explicit solution-only verifier projection, public source inventory,
fresh native audits, complete-history retry/final collection and original-token
group credit are implemented, CPU-tested and independently reviewed. Fresh physical
input and semantic preparation now reaches the shared optimizer. Answer, online
repair and final-note losses are clipped separately with one KL term; malformed
siblings stay in the original group denominator with zero repair credit.
The integrated 71-test CPU check passed, alongside an independent numerical
loss/gradient oracle and an ON-to-OFF update that preserved Vision weights and
existing Adam state. These are authored tiny-policy tests, not Qwen improvement.
The canonical profile, cache binding, minimum/fixed-ten callers, answer-only
control and reload gates are CPU-tested. Fresh replay connects original raw
responses, physical inputs, native checks, saved gradient evidence and every
checkpoint payload; a stale reload proof cannot certify replaced weights or
optimizer state. The later combined caller suite passed85 CPU tests, followed by
an independently checked runtime-file propagation change. The separate n1 native
evaluator now shares the typed retry and final-note contracts, preserves missing
notes in coverage denominators, and binds evaluation to current training/reload
evidence. Its focused CPU tests and independent review passed. The connected
825-test frozen-source run ended with20 failures,1 error and1 skip. All21 failure
records were reproduced in the pre-redesign archive; the new repair modules had
no failures. This is a no-new-failure comparison, not an all-green suite.

Fresh CPU data preparation then preserved the170 bound inputs, fixed ten items
and fifteen images. Actual processor measurements found that repeated public
reference metadata made final inputs exceed the24K context. The repair-only
input now uses shared source/schema dictionaries and a versioned, exactly
reconstructible feedback representation. Raw attempts and literal collector
feedback stay intact; output references remain full canonical IDs. The frozen input wire passed123 connected CPU tests, independent review and actual
processor capacity checks; preserved legacy histories required8749/14971/15882
input tokens with8192 output reserved within24576. A fresh source-bound preparation
and actual base-cache generation/separate-process check also passed.

The subsequent actual fixed-two Qwen trial was operator-stopped before optimizer
commit. A complete malformed T2 sibling incorrectly suppressed a valid sibling's
T3 and its final-note path. The other item's four actual final notebooks contained
empty attempts/claims and failed validation. This is neither a successful update
nor a notebook-learning result. The new repair-note-v2/full-history-v2 contract
retains a complete syntax-invalid member with result=None and the separate public
outcome INVALID_RESPONSE_CONTRACT/answer=null. Valid siblings retain existing
retry eligibility; original n, zero malformed PG and full-action KL remain.
Missing raw data, log probabilities or integrity evidence still block admission.
The corrected collection, final audit, training and n1 evaluation require their
own combined CPU checks, source preparation and actual model validation.

The earlier V58 assessment-credit minimum did complete an actual pretrained NF4
update and a separate-process reload. It established learning from some verified
error/unknown classifications; all four complete notebooks still failed and their
repair relationships were null. It does not establish the new repair objective.
The base stays frozen NF4, with adapter/Merger training rather than full-parameter
fine-tuning. `MMDL/`, V8 results, historical responses and shared model weights are
preserved. See [`notebook_rl/README.md`](notebook_rl/README.md) for the interfaces,
checks and remaining dependencies. The sections below retain historical protocol
states and commands; an old “pending” statement is not the current status above.

## Historical local error-notebook NF4 workflows

The latest opt-in whole-note assessment objective keeps the complete model-authored
notebook and adds original-token credit for independently checked error/unknown
classifications. A commit now requires a finite, nonzero signed sum of all retained
assessment-only parameter gradients; answer reward or KL cannot count as notebook
learning. The explicit v3 normalization experiment multiplies only mapped assessment
coefficients by8192 to cancel that channel's existing cap8192 loss division. Other
reward/loss terms, n4, masks, group weights and routing stay unchanged; identically
wrong classifications still have zero relative signal. A historical v1 trial completed one actual Qwen update and an independent
reload, with a nonzero assessment-only contribution measured for one candidate.
Its closed same-history comparison remained0/4→0/4 correct, with unchanged generated
text/tokens and no recoveries or regressions despite changed log probabilities.
V2 null-safe mapping and the aggregate-gradient check still require actual
Qwen validation. V3 retains them and requires its own fresh source preparation,
cache checks, minimum training and reload; historical evidence cannot certify it.
Coefficient ratios do not establish total parameter-gradient ratios or improvement. Whole-note improvement is
unverified. This is a step toward
the complete solve/retry/notebook/RL pipeline, not its completion. The flags and
limits are documented in [`notebook_rl/README.md`](notebook_rl/README.md).

As of 2026-10-25, actual pretrained NF4 RL updates and separate-process reloads
have been verified, but accurate complete notebooks and self-improvement remain
unproven. The complete-reference trial produced692 null relationship decisions.
The subsequent paged trial was stopped after all four headers failed and a page
linked different items within the same attempt as corrections; it made zero updates.
The historical trials below are preserved as separate contracts; they do not
establish the latest full-history requirement. The `--pair-draft-synthesis` path
uses its intermediate judgments only as unverified drafts; only the final model
notebook can receive notebook credit. A separate explicit circuit evaluation path
now checks full-history reference notebooks on CPU, including units, physical
rules, unknown dependencies and singleton training rejection. Its actual model
evaluation remains unverified. See
[`notebook_rl/README.md`](notebook_rl/README.md) for the new protocol, commands and
distinction from the historical contracts below.

A separate paged full-history implementation now connects header/per-attempt
model responses to the same independent verifier and original-token RL. It keeps
all original history/images and the8192 total output budget, treats all pages as
one logical notebook, and never fills missing model judgments. CPU input capacity
has been measured; actual paged Qwen notebook correctness, update/reload and
self-improvement are not yet established. This is local pipeline validation,
not approval for large-scale training.

The new explicit pair-action V2 stage asks the model to judge every public ancestor
comparison, including unchanged observations, both retry branches and rule-application
traces. These are **partial notebook judgments**, not a complete notebook. Collection
may continue through the original fixed ten items without a successful first-two
native update; every optimizer commit still requires positive original-token native
relationship credit after MIS and a nonzero isolated native gradient. No signal means
no forced update. The actual V53 minimum ended with zero updates: both verified
native relationships were correct in all four candidates, giving zero centered
native advantage. Only final-answer-output rewards varied. False relationship
claims outside an actual earlier issue receive no local partial credit or penalty;
this limitation must not be mistaken for comprehensive notebook learning.

A separate `fact_context_probe --pair-to-whole baseline|trained` path now connects
a verified native update/reload to newly generated full-reference notes with the
same complete history, images, prompt and seeds. Its CPU checks pass; V53 has no
eligible learned checkpoint, so this diagnostic has not run on Qwen. See the local
guide for its commands. Accurate full-note learning and self-improvement remain
unverified.

The current **full-history notebook correctness and self-improvement remain unverified**;
actual partial-fact RL updates and fresh-process reload have been verified separately.
`local-nf4-full-history-notebook-v13` covers all attempted branches and the final
successful retry, independently checks each note claim, then finalizes evidence-based
adapter routing. Failed notebooks may receive verifier checks and submitted-fact
error-location IDs, with up to two
additional four-sample rounds, with exact earlier notebook tokens retained. No
correct values or replacement claims are supplied. Binary rewards are unchanged;
this additional verifier feedback and sampling budget are a distinct protocol.
Context16384/note2048 are explicit new settings; the base remains
frozen NF4, not full-parameter fine-tuning. Existing results below belong to the
older first-attempt notebook contract and do not prove this new requirement.

The latest source uses whole-history reward `local-all-attempt-notebook-v5`:
equivalent error-location ID spellings are normalized before strict/partial
scoring, without changing raw responses, fact values or evidence requirements.
Canonical order and uniqueness remain mandatory after normalization. This changes
the protocol hash; historical V4 runs keep their frozen source and original scores.
It is a representation fix, not evidence of accurate notebooks or self-improvement.

The current initial/repair instructions also clarify the native-tree fact contract:
record the root and drawn child links, omit missing child positions, and keep JSON
`null` for an attempt with no extracted answer. No new absence-verification rule is
added. The prompt hashes change, while strict reward V5 remains unchanged; this
clarification is for new source-bound runs, not the live fixed-ten trial.

The closed V13 minimum generated14 solutions, one unverified image reread and12
whole-history notebooks. All12 notes failed the existing verifier (11 content
failures, one invalid JSON), so optimizer updates remained0. The successful
retry's correct answer did not establish correct facts or diagnoses. A separate
`fact_context_probe --notebook-layout` diagnostic compares lossless history layouts
without changing the production prompt, rewards, routing or weights; see the local guide.
The closed layout comparison had zero valid notes in both four-sample groups.
The closed `--notebook-drafts` diagnostic generated six independent drafts and
four final notes in817.318seconds. All four final notes failed strict verification
(three content failures and one invalid attempt field). It supplied all original
attempts and the successful retry; history was not omitted. It made no updates.

The explicit opt-in `local-nf4-verified-note-atoms-v14` adds a separate host-verified
training objective while preserving every strict reward and the whole-note success
gate. It is not the default and is not a demonstrated repair: a CPU posthoc replay
of the twelve V13 notes still gave twelve zero scores and zero group variance.
No new GPU training was started for this candidate. See the guide for its contract,
invocation and limits; old outputs and the V13 protocol remain unchanged.

The separate opt-in V15 component objective preserves those strict gates but gives
independent partial credit to verified facts and attempt-level history, cause and
error-location checks. It does not certify a whole note or healthy Vision from an
empty error list. See `notebook_rl/README.md` for the explicit four-fraction reward.
CPU tests are not actual pretrained-model updates or evidence of self-improvement.

The actual V15 minimum subsequently completed one partial-reward update and a
fresh-process reload, but all12 whole-history notes failed strict verification.
A paired diagnostic on the same history gave0/4 valid notes before and0/4 after
that update. Both observed routes were Vision-enabled; neither Vision-frozen
learning nor unseen-question improvement is established for this workflow.

The separate opt-in V16 fact-token-credit objective keeps full-history notes and
strict whole-note rewards/gates. Only the partial fact reward is assigned to its
own original JSON value tokens; unrelated claims receive no such partial credit.
This is a new objective, not a retrospective fix of V15 results. All-wrong facts
still have zero relative reward signal. See the guide before using it; CPU checks
do not establish that it fixes notebook quality or authorizes fixed-ten training.

The current fact-token objective is `verified-note-fact-token-credit-v2`. It accepts
supported equivalent fact-ID spellings and binds each canonical ID to the model's
original JSON value tokens, rejecting alias collisions even when values agree.
Values, strict notebook checks and unknown/null-relation guards are unchanged.
Historical v1 evidence must use its frozen source; the new full contract/hash needs
fresh preparation and actual validation. This representation fix is not notebook
correctness or learning evidence.

The actual V16 minimum completed one fact-token-reward update and fresh-process
reload. Five notes had verified partial fact-token/isolated-gradient contributions,
but all12 notes failed strict whole-note verification. The subsequent same-history
comparison remained0/4→0/4; it is not a held-out evaluation. On2026-10-22 the separate
explicit exploratory policy admitted a fresh-initialization fixed-ten trial after
the actual parent-evidence gate passed. That trial was deliberately stopped after
an explicit nonexistent tree-child claim exposed a verifier omission; its partial
responses and committed checkpoint are preserved, not a completed ten-item result.
The revised verifier requires fresh source-bound input checks and rejects historical
reuse if any regraded outcome changes. Both learning routes, accurate whole notes,
disjoint performance and the answer-only control remain to be checked. No
full-parameter training or large-scale run is implied.

After a real update and trained fresh reload, the existing probe's explicit
`--trained-notebook-reload` option checks full notebooks on the exact original
history/images/seeds. It separately records historical and trained policy versions
and strict-note recoveries/regressions. This is a same-history diagnostic, not
unseen-question evaluation or permission to bypass the fixed-ten learning gate.
See `notebook_rl/README.md`; the completed same-history diagnostic did not show
strict-note improvement. New-question performance remains unverified.

V11 records explicit nonexistent edges separately from registered-slot coverage.
Only validated complete native BFS/shortest worlds can prove such an edge false;
unknown endpoints or unsupported values remain unverified. Wrong edges participate
in full-history citations, notebook verification and Vision-enabled routing, even
when the final answer is correct or truncated. Unverified edge claims prevent a
Vision-frozen admission; missing evidence is not a clean bill of health. This
changes the verification protocol, not the frozen V10 worker or its saved results.
CPU counterexamples are not proof of actual notebook learning or module causality.
The unchanged private data vault is read through an explicitly pinned historical
schema-compatibility case, not by disabling manifest checks. Only runtime response
fields changed; actor/verifier/lineage formats did not. Unknown source hashes,
changed payloads and protocol mismatches remain blocked. The initial V11 data
preparation failure and all original data are preserved.
Scanner provenance has a separate exact six-file compatibility check: unchanged
detectors/configuration, reviewed schema/reader bytes and unchanged guards only.
Historical receipts stay intact and current hashes remain separately recorded;
this does not claim a fresh contamination scan or change its thresholds.

V5 additionally defers learning on unresolved visual evidence: Vision-frozen
updates require complete correct required observations in every actual attempt;
an independently observed wrong required slot enables the Vision-adapter route.
Missing observations are not a healthy-Vision certificate. Unresolved fixed IDs
are preserved and reported separately, never replaced. Forced/random routing and
Vision-disable overrides are not part of this strict local protocol. The frozen
V4 experiment is unchanged; the new admission path still needs actual-model validation.

V6 location feedback contains no reference values, answers, expected causes or
replacement prose. It is additional verifier supervision: rejecting a binary
direction can imply its alternative. The model must still reinspect original images
and independently assess all historical attempts. The minimal controller stops
before any optimizer update when there is no usable notebook reward variance;
all-invalid and all-valid constant groups do not establish notebook learning.
The fixed-ten answer-only control is unchanged. This new path remains unverified
on the actual pretrained model.

V7 explicitly requests observations for all ten fixed training items. Original
actors/vault/IDs/questions/images/answers remain intact; a separately bound
execution copy changes only the actor mode to `obs`. Frozen V6 plain actors were
never inspected for visual slots, so they could not enter either strict route.
Capture, training/control, reload and synchronization probes now validate and use
the same mode-only projection. This is an input-protocol correction, not evidence
of accurate observations, valid notes or successful learning; all actual V7 gates
remain to be run. Previous sources/results are not overwritten.

The frozen full-history v2 minimum did complete one answer-reward update and fresh
reload; its four notes contributed no reward gradient. All four actual v3 notes also
failed validation: malformed range arrays and independently observed
incorrect arrow facts. All twelve V4 notes also failed; the last repair repeated a
wrong image fact and incorrect assessments despite verifier feedback. No valid-note learning, both-route completion, fixed-ten
extension or improvement is established for this full-history workflow.

The folder-only local entrypoint is the [notebook RL guide](notebook_rl/README.md#local-pretrained-nf4-workflow-2026-10-21).
It documents the bundled two-stage pinned installation, fresh destination-bound
data preparation, standalone NF4 cache, minimal update/reload, fixed-ten notebook
and matched answer-only arms, and separate full-history circuit-evaluation v3 commands. Run
from the copied folder's parent; model/data/provenance/artifact paths are explicit
external private inputs, not code dependencies on this repository's siblings.

Historical training rewardv3 used model-selected first-attempt line IDs and exact host-resolved
raw spans, with strict independent answer/facts/cause checks; UNKNOWN is never
healthy Vision. Actual frozen copied-source minimal-v6 update1, physical624 payload
verification and fresh-process reload passed, with L252/M24/V48 changed. All four
notes earned0 and had zero isolated reward-gradient contribution, so notebook
benefit remains unproven. Reserved8,608,808,960bytes exceeds reported dedicated
capacity8,546,942,976; fully VRAM-resident execution or paging causality is not proven.

The fresh-F0 ten-notebook-v4 arm completed all ten problems, 90 sampled responses
and two real updates. All624 saved tensors and a fresh-process optimizer/forward/
generation reload were verified. Two of20 notes earned reward1; one note group
contributed to the combined RL+KL backward pass, without isolating its reward-only
gradient. Both independently routed training paths were exercised. The initialization-
matched answer-only control completed ten problems, three real updates, physical
checkpoint verification and fresh-process reload. Both arms were evaluated against
the same untrained NF4 baseline on ten separate synthetic circuit development items:
all three scored5/10 first attempt and8/10 after at most three attempts; each had
zero of three generated notes pass the complete verifier. Paired first-attempt
gains/losses were0/0 for both trained arms. No learning updates occurred in evaluation.
Those ten circuit items are source/image-disjoint, not unconditionally structure-
disjoint: a later audit found the same four-node/five-edge simple undirected topology
in one training DFA when labels, directions and parallel edges are discarded.
That is not an identical-question finding. A strict structural-generalization claim
requires a separately frozen eligible cohort; do not replace the old results silently.
The CPU-only `notebook_rl.local_ladder` preparer now provides a separate candidate
with ten native circuit instances in three five/six-node topology families; it
checks the original training bindings and complete-graph separation. It grants no
training/evaluation permission and does not produce a model score. See the local
guide for the command and remaining data/native-evaluation checks.
The separate CPU `notebook_rl.local_ladder_score` can now replay ten saved native
numeric/unit responses through the isolated verifier. Its output does not certify
that a model generated those responses; native GPU inference remains pending.
`notebook_rl.local_ladder_generation` now supplies the matching first-attempt
generation core: ten native inputs, literal token/pixel records, and fixed F0/FT
seeds. Its CPU tests and actual processor check are not model evaluations; the
gated `local_eval evaluate-ladder` caller is implemented and CPU-tested. It requires
actual full-history notebook learning/reload, a closed fixed-ten arm, matched
initialization and a separately bound native data receipt. Its actual F0/FT GPU
evaluation remains pending. Neither frozen runtime is changed.
The closed full-history V8 trial produced12 solutions and24 notes; every note
earned zero, so no optimizer update or trained reload occurred. V9 removes the
conflicting solve-format instruction from the notebook request while preserving
full solution history, images, rewards and strict routing. Its standalone cache
save/reload checks passed, but its closed minimum also produced12 solutions and24
zero-reward notes, with0 updates and no trained reload. Wrong/missing visual facts
remain substantive failures, not just output-format errors.
V10 gives an explicit JSON shape and public observation-slot syntax. It removes
the redundant model-generated next-action field; the host derives that action only
after the attempt diagnosis is verified. All historical attempts, corrected facts,
answers, causes, evidence and strict routing remain required. Binary validity reward
is retained with a new contract version; this is not retrospective rescue of old
notes. Actual V10 notebook learning and generalization remain unverified.
The older contract's local functional pipeline is verified; the current full-history
pipeline and self-improvement are not yet demonstrated.
Notebook/control update counts and token exposure differ, so this is not an isolated
compute-matched estimate of note reward effects. Do not automatically scale training.
Frozen execution copies retain their original documentation and hashes.
Original BF16 cache must remain available for six dependent legacy checkpoints.
No cloud/API/upload, weight/data bundling, publication or professor-facing MMDL/V8
change is implied. Older BF16/cloud sections below are historical protocols.

## Historical 8K input-recovery and teacher-free A/B trial

`experiments.input_recovery.runpod` runs the frozen 48-item I/T/IT/IR/TR diagnosis
and the preselected 12-item I/T checkpoint comparison: 336 calls in total, including
the initial 60-call smoke. Inference uses the pinned Qwen BF16 model, 8,192 output
tokens and four continuous requests. This is not a new 900-item MMMU score.
Rules are incomplete hints, not certified knowledge oracles; recovery does not
uniquely identify a Vision, Merger or Decoder fault.

The separate TabMWP trial compares native answer-only A with ordered native table
facts plus the same answer B. No upstream solutions or teacher-model outputs are
used. Each planned variant starts from fresh F0 and trains all four Merger modules
on 1,600 native examples plus the same 400 historical retention examples, for
125 updates. A/B share images, IDs and order, but not assistant-token exposure.
Near-benchmark candidates must be excluded before a new selection is frozen;
an existing incomplete approval is never rescued or relabeled as approved.

```bash
python -m fine_tuning.tabmwp --help
python -m fine_tuning.tabmwp_images --help
python -m fine_tuning.tabmwp_trial prepare --help
python -m fine_tuning.tabmwp_trial approve-cpu --help
python -m fine_tuning.tabmwp_evaluate prepare --help
python -m fine_tuning.tabmwp_runpod --help
python -m fine_tuning.tabmwp_evaluate score --help
```

Preparation and CPU approval are not GPU execution authority. The controller
requires the actual closed 336-call diagnosis, complete source/rights/benchmark
and processor checks, the exact new execution commit and owned Pod, and a bound
private authorization file. Every inference gold file stays off the Pod. A fresh
F0 and both candidates use the same frozen external 132 calls (48 QA, 24 table
transcriptions and 60 retention tasks), with 8K/four continuous requests. Selection
uses QA transitions, exact/cell structure recovery and equal-weight QA/TRACE
retention; it does not use MMMU outcomes or mere JSON validity. It never starts
50K training or a final 900-item evaluation automatically.

Data, checkpoints, gold, private receipts and full responses are not published.
TabMWP data is CC-BY-NC-SA-4.0, not the repository code's MIT license; this trial
is private noncommercial research with attribution. Source-native labels and
sample AI inspection are not independent human pixel-semantic certification.
Use a new clean exact-SHA MMDL_test clone and the normally verified setup for a
new stage. Do not hot-patch a running clone, extend its original financial deadline,
or modify the professor-facing MMDL checkout and historical V8 result.

## Historical deployment policy — 2026-10-04

Final original10K state, 2026-10-05 14:57 UTC (supersedes the running snapshots
below): all four arms completed10,000 records/625 updates, fresh-process reload,
MMMU60 and external60. All checkpoints/raw results were recovered and CPU-verified;
the integrated local gate binds the original controller, source freeze, outer
watchdog, closed-process probes and all four per-arm receipts. Owned Pod
`22u0uqi171v0cn` was deleted with204, then get404/empty Pod and network-volume lists
were verified. No new GPU work started. Full checkpoints stay outside GitHub.

Same-protocol4K MMMU60 scores are F036/FL22/FM27/FV21/FML23. Every trained arm
regressed against its paired F0. Preregistered external-development primary ranks
FM25/48 > FML21/48 > FL18/48 = FV18/48 (F015/48); MMMU is excluded from selection
and retention risk is reported separately. Whole-structure exact is0/12 and actual
rule-present exact is0/2 in every arm. Absent/null rule matches are not rule mastery.
This is not independent professional transfer, G7,50K training or final900 completion;
the historical V8 558/900/32768 baseline remains unchanged.

Private evidence: `analysis_exports/g6_runtime_20261005/final_closed_controller/`
(`report.json`, `local_receipt.json`), `pod_cleanup_receipt.json` in its parent,
and `claudedocs/g6_full10k_final_analysis_20261005.md`. These files are not included
in the test repository. The final presentation update passed native, byte-preservation
and independent visual checks at 2026-10-05 16:16 UTC. It updates seven of42 slides,
preserving the original18, the user's saved slide38 edit, all other35 slide/notes
parts and10 embedded P1 captures. Evidence is private:
`analysis_exports/ppt_updates_20261005/verification_closed10k.json` and
`final_closed10k/{native_verification,visual_review}.json`. The earlier unsaved-edit
blocker was resolved by reviewing the user's saved source before replacement.
Local documentation updates do not publish a new execution commit or start G7.

Latest observation, 2026-10-05 05:06 UTC: original full10K FL training had
consumed6,733 records (420 closed updates covering6,720 records), with actual
RTX3090 use99%/20,800MiB. Training is live, not a completed arm or FT score.
The fixed447c0fb checkout, original four arms, financial deadline and recovery
reserve remain unchanged; new Circuit code is for a later fixed checkout only.

Latest milestone, 2026-10-05 03:53 UTC: all four RTX3090 calibration arms
completed 360 records, 23 real updates, checkpoint export and fresh-process reload.
Their files, all 1,152 trainable tensors and input/update journals were recovered
and CPU-verified locally, with the closed journal prefix bound to its remote SHA.
The unchanged 2x financial gate passed (79,942.903 seconds forecast versus
82,780.886 seconds available). Original full10K FL training began from fresh F0
at 03:37:27 UTC; at 03:50:05 it had consumed 940 records with 58 closed updates,
using 98% GPU. Main training for the other arms and FT evaluations are pending.
The live checkout remains `447c0fb6056d2e18618dd22f21ef80197cb57103`; later
published code is not applied to it. This G6 forecast covers paired60 diagnosis,
not G7, independent transfer or a final900/32768 run. Those remain unfinished;
the professor-facing 558/900 V8 baseline is unchanged.

The user authorized workspace experimental code, configuration, documentation and small
own-generated CPU fixtures in `jang2296/MMDL_test`. Professor-facing `MMDL/` and
`jang2296/MMDL` remain separate and unchanged. The private-overlay instructions below
are historical, not the current bootstrap. Use the root README's exact-commit clone
command and current `runpod_setup.sh`; no code overlay upload is needed. Weights,
checkpoints, HF/benchmark payloads and large responses stay outside GitHub. Frozen
receipts are not rewritten to manufacture approval. Only required own-data fixtures
and historical v18 source are bundled, not all saved results.

The previous A40 Pod was deleted by the user. Its full10K controller stopped before GPU
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

Latest GPU milestone, 2026-10-04 15:52 UTC: the frozen f3 commit has completed
full10K data preparation, smoke-root exclusion and both 60-record cohorts.
Twenty-five near-image review candidates remain excluded; normally approved
TRACE replacements restored the exact 10K quota without changing the 9,975
parent records. F0 MMMU inference started at 15:47:43 UTC; at 15:52:30 it had
saved 4/60 responses, using 83% GPU and 10,039 MiB. This is inference, not
training completion. Four-arm smoke and the measured 2x feasibility gate still
precede full10K training. G6/G7/transfer, final artifact recovery and cleanup
remain incomplete; financial deadlines and the professor baseline are unchanged.
The future CPU optimizations below are not applied to this running commit.

The replacement owned A40 passed setup at exact commit
`f3f37ab624cfa6d35cc8a7cc85ba95cbac553ce1`: 180 CPU tests, actual CUDA/BF16,
and the 13-file pinned model manifest. At 2026-10-04 14:31 UTC, its controller
was converting TRACE images during pilot preparation, with GPU use still zero.
All data approval and evaluation cohorts precede the GPU controller; its F0
evaluation then precedes training smoke. This is preparation, not a completed
GPU training run. Future benchmark indexing fingerprints identical raw image
bytes once; counts, raw hashes, schemas and coverage checks are unchanged.
That CPU optimization is tested on authored Parquets, not applied to the running
fixed commit, and has no measured cloud speedup yet.

At 15:00 UTC the same pilot process was still live with zero GPU use. TRACE
conversion and the 10,000-row proposed ledger were complete, but processor
preflight and final approval were not yet observed complete. Proposed rows are
not approved or trained rows. Future approval code avoids eager fallback sets
and binds every row to the digest of the same parsed receipt bytes. A final
receipt rehash rejects mutation before output creation; source, processor,
rights, contamination and coverage gates stay intact. Related 40 CPU tests,
Ruff, bounded mypy and independent delta review passed. The running commit and
historical receipts are unchanged; new code requires fresh normal approval.

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

## Bounded independent DC-circuit bridge

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

Every artifact in the original four-root pool remains `training_allowed=false`, `public_upload_allowed=false`
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

The existing `bridge_evaluation` CPU builder also supports the v2 Circuit
contract. It requires complete train-only A0–A3 approvals and current pinned
processor proofs, replays the deterministic solver/render pool, selects only
family-disjoint development roots, and freezes all12 condition/view inputs per
root before emitting separate typed gold. The scorer accepts bare A–D QA and
strict exact-rational structure/state JSON; there is no V8 fallback or gold-based
candidate rescue. A trained checkpoint must match the frozen approved training
manifest, records and complete exposure before model loading.

The current four-root candidate has train3/sealed1/**development0**, so it
cannot produce this cohort or borrow a sealed root. Its48 actual local CPU
processor checks passed on2026-10-05, but source/render/contamination approval
and a larger predeclared split pool are still required. Constructed CPU test
receipts are not actual approvals. Neither the builder nor the tests execute
GPU training or complete full professional G7/non-synthetic transfer. New code
does not hotpatch a running frozen G6 checkout.

### Separate100-root preparation, 2026-10-05

A new private pool (seed73041, start-index1000000) has100 roots/97 canonical
families: train77/75 families, development10/10 families, sealed13/12 families.
Deterministic generation and both exact solvers replayed all100 roots. A Sol
helper independently inspected six actual rendered images; this is an AI spot
review, not human or exhaustive semantic certification. The font rights and
actual file hash are recorded. OPEN/CLOSED text and component values/units may
provide shortcuts; mapping-only causality is not claimed.

The actual pinned CPU processor checked all1,200 original rows (300 per A0–A3):
maximum sequence3194, assistant231, child sequence budget4096 and episode pixels
3145728. Normal provenance/processor/contamination approval produced231 train
rows per condition, using the same77 roots/75 families, with zero quarantined
rows. Original pool permissions and old receipts were not rewritten. Exclusion
fingerprints reuse the verified historical index with a checked path projection,
not a new read of the original Parquets. A separate gold-blind120-row development
cohort uses the same10 development roots across all four conditions/three views.
The pool, approvals, images and detailed receipts remain outside GitHub.
The actual current-source CPU consumer also passed:1,089 frozen input files,
four231-row training sets, development120 and MMMU60. Its separate immutable
receipt binds exact paths, metadata/file hashes, source freeze and the prior
receipt. A Sol review checked the digest and metadata/source binding; this was
not another image-payload replay, model load or GPU fit test.

The existing `run_bridge_experiment` supports this domain through `--bridge-cohort`
(`--graph-cohort` remains an alias). It verifies all train-only approvals, current
processor proofs, source/family separation and the typed Circuit gold contract
before GPU work. The same typed contract is enforced during result verification;
generic QA/V8 routing cannot silently substitute for strict Circuit metrics.
Fresh F0 per condition, fixed module, original budget/source-freeze safeguards
and actual exposure reporting remain required. A historical Graph A3 checkpoint
cannot supply a Circuit condition, and the Graph-only smoke entrypoint rejects it.
The final191 CPU tests passed, along with Ruff, three-source-file mypy and the
publication-safety checks. These include authored routing and adversarial
typed-gold fixtures; they are not GPU training or independent model performance.

The full10K G6 external results must precede evidence-based module selection.
Before those results, a private follow-up rule was fixed: equal-weight QA-type
accuracy, existing task-success flags as a secondary rank, then actual trainable
capacity and fixed arm order. Grounded answer success and state/rule metrics
remain distinct. Selection requires all four full10K arms and recovered paired
proofs, excludes MMMU/sealed/transfer outcomes, and cannot manufacture a positive
signal from tie-breaks. With no positive QA signal it is only a bounded probe;
there is no automatic50K expansion or claim of internal-module causality.
CPU preparation is not GPU memory certification, Circuit training, full
professional G7 or independent non-synthetic transfer. No quantization, hidden
resolution reduction, live G6 hotpatch or professor-facing baseline change occurs.
