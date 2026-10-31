# Error-notebook self-correction RL

기존 BF16 v2 경로와 현재 로컬 NF4 자기수정 RL 경로를 함께 보존하는 패키지다.
이 문서의 과거 BF16 설정(T1:8/T2:4)을 현재 로컬 설정과 혼용하지 않는다.
현재 경로는 T1:n4/T2:n2/T3:n2·최대3단계, 풀이1024토큰·최종노트8192토큰·문맥24576이다.
정답은 독립 검증기에만 전달하며 SFT·교사 풀이 모방은 사용하지 않는다.

## 현재 실제 검증 상태 (2026-10-28)

`repair-retry-collector-v3` 계열의 재관찰·전체 이력 연결을 사용한 동결 소스10 최소시험은
실제 업데이트1회·624개 학습 텐서 저장·새 프로세스 재로딩을 통과했다. optimizer/step과
단일·복수 이미지의 실제 forward/생성을 대조했으며 NF4 기저는 불변이고 원본 접근은0이다.
이는 답 보상에 의한 기능 검증(F)이며, online/final 오류·수정 내용의 signed·positive VJP는0이다.
따라서 정확한 전체노트 학습(S)이나 새 문항 성능 개선(E)으로 보고하지 않는다.
정본은 비공개 `verified-repair-notebook-v59/training-reload-closure-main-v10.json`
(SHA256 `c30bfa2deda64327822af8c012c3aeb72c73bd47ae1b3fecafbc6387956e2464`)이다.

같은 프로토콜의 고정10문항은 진행 중이며 전체 종료/새 재로딩, 정확한 노트 내용 학습과
Vision 포함/동결 양 경로의 실제 통합 검증은 남아 있다. 현재는 NF4 기저를 동결한 어댑터와
Merger 학습이며 전체4B 가중치를 바꾸는 full fine-tuning이 아니다. 아래 과거 기록은 보존한다.

## 현재 실행과 다음 소스의 메모리 배치 (2026-10-27)

동결 소스8의 실제 최소시험은 업데이트·저장·새 프로세스 재로딩을 통과했고,
고정10은 70풀이·24최종노트를 수집한 뒤 외부14700초 한도에서 종료됐다(exit124).
업데이트3회와 중간저장은 보존됐지만 마지막 업데이트·정상 종료·전체10 재로딩은 미완료다.
모든 문항의 원래/적용 online·final token 계수는0이며 노트 학습이나 새 문항 개선 증거가 아니다.

현재 작업 소스는 `checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, step)`를
actor별 `build_tree`와 `append_rewarded_notebooks` 구간에 적용한다. 초기화된 CUDA AdamW의
`exp_avg`·`exp_avg_sq`만 생성 동안 CPU에 보관하고, 학습·분기·저장 전에 원래 장치로 복원한다.
CPU scalar `step`, 상태값·dtype, 파라미터, 학습 수식과 생성 조건을 유지한다. 저장소를 공유하는
상태나 지원하지 않는 상태는 전송 전에 거부하고 복원 실패는 전파한다.

실행 원장의 `optimizer_state_collection_policy`는
`adamw-moments-cpu-during-collection-v1`이며 재로딩에서 정확히 대조한다.
실제 Qwen 중간저장(step3/cursor6)의624개 optimizer 상태로 왕복을 검증했다. GPU allocated는
4,769,837,056→3,613,880,320→4,769,837,056bytes로 변했고 두 moment의
1,155,956,736bytes(약1.077GiB)를 생성 구간에서 비울 수 있었다. 값·dtype·원래 장치·가중치·
스케줄러·난수·동결 NF4는 보존됐다. 첫 진단의 device 비교 오류와 수정본 결과를 별도 보존한다.
79개 CPU 검사와 작은 AdamW의 실제 CUDA 왕복·예외4검사도 통과했다. 중간저장 하드웨어
검사는 전체10 정상 재로딩이 아니며 실제 생성 속도·새 코드의 전체 학습/재로딩 검증은 남아 있다.
CUDA 단위검사는 `MMDL_TEST_OPTIMIZER_PARKING_CUDA=1`일 때만 실행한다. 이 검사의 통과도
실제 Qwen의 오답노트 학습이나 자기개선 증거로 사용하지 않는다.

### 첫 재시도 전 같은 정책의 재관찰 — v3 구현

현재 `--repair-notebook`은 collector/feedback/full-history/evaluator v3를 명시적으로 결합한다.
T1과 native 우선 부모 선택 뒤 첫 eligible 재시도 직전에만 원본 이미지·질문/보기·공개 관측
문법으로 n1/1024 재관찰을 한 번 생성한다. 이전 풀이·정오·gold는 입력하지 않는다.
T2/T3는 동일한 미검증 원문을 인용하고 NOTE는 공유 feedback suffix를 통해 한 번 받는다.
잘못되거나 잘린 재관찰도 그대로 보존하며, 보상·시도·그룹·관측 라우팅·학습 loss에서 제외한다.
기존 n4/n2/n2·1024/NOTE8192·seed·부모 전체 token prefix·의미 판정과 보상은 유지한다.

후보 없음/시작 전 예산 차단은0회다. 재관찰 완료 후 재시도 차단은1회+명시 skip이며 불완전
수집을 학습에 승인하지 않는다. STARTED/COMPLETED 원장·원문 receipt·이미지와 입력 token을
학습 CPU 재생에서 대조하고 옛 protocol proof는 새 계약에 자동 승계하지 않는다.
n1 평가의 순수 의미 replay는 기존대로 tokenizer/pixel 재처리나 모델 provenance를 인증하지 않는다.
v3의 실제 Qwen 생성·답 보상 업데이트·저장/새 프로세스 재로딩은 위 최소시험에서 확인했다.
검증된 NOTE 수정 내용에 따른 실제 학습과 고정10 전체 재로딩은 아직 미검증이다.

### 형식 오류 응답의 남은 재시도 기회

새 `repair-retry-collector-v2`는 정상적으로 판정된 재시도 후보를 먼저 선택한다. 그런 후보가
없을 때만 실제 저장 원문에서 문법 오류와 SHA를 다시 확인한 응답 하나를 다음 부모로 쓴다.
`stop`/`length` 모두 원래 종료 이유를 유지하며 최대3회·T2/T3의 n2·1024토큰·seed 식은 같다.
정오 미판정(`result=None`)과 보상0은 유지하고 피드백에는 문법 코드와 `NOT JUDGED`만 준다.
잘못된 원문에서 관측값을 구제하거나 정답·교사 풀이를 만들지 않는다.

- `rollout.malformed_retry_receipt`: 원문·완료 토큰·실제 문법 실패를 확인한 코드/SHA 반환.
- `rollout.select_repair_parents`: 기존 native 순서를 우선하고, 없을 때 문법 오류 부모 최대1개 선택.
- `rollout.repair_parent_feedback`: 원래 판정 또는 문법 오류에 맞는 공개 피드백 생성.
- 생성·`repair_notebook._inspect_tree`·평가는 위 함수를 함께 사용하고 원래 이력과 seed를 재생한다.

피드백은 `repair-retry-feedback-v2`, 평가기는 별도 v2와 prompt/protocol hash로 고정한다.
과거 소스8의 결과는 소스8로 검증하며 새 프로토콜로 소급 승인하지 않는다. 문법을 고쳐도
과거 미확정 관측은 남고, Vision 동결 인증이나 검증된 native 오류 수정 학습이 되지는 않는다.
CPU 연결 검사는 통과했으며 새 정책으로 실제 Qwen이 T3를 생성하는 검증은 남아 있다.

## Verified-repair redesign — implementation checkpoint (2026-10-27)

This section supersedes historical “current/pending” wording below. The new
workflow has CPU-tested collection and optimizer connections. Its first actual
fixed-two Qwen trial was stopped before optimizer commit after a collector defect
suppressed valid-sibling retries. The corrected trial completed one actual
answer-reward update/save and a fresh-process reload:624 trainable tensors,
optimizer/step/cursor and single/paired-image forward/generation matched. F is
true, while native repair learning S and benefit E remain false. Its four final
notes omitted success_sample_id and submitted no claims; source histories had
zero native correction opportunities. Source8 fixed-ten collected all10 items but hit
the external timeout during its last update; full training, fresh reload and evaluation
remain unverified. Historical
assessment credit is a separate objective: V58 completed one actual NF4 update and
fresh reload, but strict notebook correctness stayed0/4 and repair relationships
were null. A successful code test is not a model-authored repair or learning result.

The explicit `--admission verified-functional-repair-exploration-v1` shares the
existing fixed-ten train/control/reload/eval path, with a verified current minimum
F as its entry prerequisite. It preserves the fresh replay's actual S value and
all existing strict admission requirements. Its focused CPU tests and independent
review passed. Source8 combined397 CPU regression and its actual minimum/fresh reload
passed; the optimizer-placement change above has CPU evidence only. No historical-protocol bridge, generation/reward/MIS change or
large-scale authorization is added. Final notes and native S must still be verified.

| Boundary | Implemented interface | Current evidence |
|---|---|---|
| Retry envelope | `repair_contract.parse_retry_sections`, `parse_error_check`, `project_solution` | Explicit ASCII blank-line boundary policy, original raw SHA/character spans, exact native suffix; quoted old answers cannot be scored as the revised answer |
| Explicit grading | `Verifier.grade/trace` and `VerifierService.grade/trace(response_contract=...)` | Legacy `None` preserved; only explicit `repair-check-v1` projects a suffix; native evidence is lifted to original coordinates |
| Public work items | `public_origin_inventory(samples, contract_version, response_contracts, parent_sample_id)` | Ordered source bindings include raw/projection hashes and malformed-source errors; every normal/unknown origin stays in the denominator; no gold/reward filtering |
| Public input representation | `prompts.public_origin_dictionary`, `render_public_origin_dictionary` | Shared source/schema dictionary; canonical `:rN` equals the one-based row position; flat character-span pairs retain empty spans and all origins. Output parsers accept canonical refs only |
| Original feedback in final input | `repair_notebook.expand_public_feedback(call, templates, originals, inventory)` | A versioned final-only representation reuses exact instruction text and original ancestor tables; it must reconstruct the original feedback/hash. Collector journals and runtime parent prefixes stay literal |
| One actual response | `notebook_synthesis.audit_native_sample(sample, actor, verifier, response_contract=...)` | Fresh grading/native checks, detached input snapshot, exact extracted-answer binding; cached caller results/rewards are ignored |
| Final JSON syntax | `repair_contract.parse_final_note(raw, contract_version="repair-note-v2")` | Full authored attempt list, at most3 claims, typed dependency-blocked UNKNOWN and original spans; syntax alone certifies no relation |
| Online local repair | `repair_credit.audit_online_repair(history, child, actor, verifier, inventory, response_contracts, sections)` | Fresh same-identity violated-to-compliant descendant and exact emitted patch required; intent alone earns0; all frozen origins remain in the denominator |
| Final bounded semantics | `repair_credit.audit_final_note(note, history, actor, verifier, inventory, response_contracts)` | Fresh relation truth, explicit refutation, dependency-blocked UNKNOWN and separate precision/coverage; full tree attachment and learning remain separate |
| Original-token ownership | `repair_tokens.emitted_patch_spans`, `span_token_ownership` | Original claim/patch spans only; unstable decode abstains, mutated decoder inputs are rejected |
| Separated loss kernel | `losses.repair_component_losses` | Answer/online/final clipped separately, KL once, original group size and cap retained |
| Complete collection | Explicit repair `build_tree` branch and `repair_notebook.append` | Request/config/image mutation is rejected; complete raw history and malformed siblings remain; final payload survives sorted JSON persistence |
| Fresh training preparation | `repair_training.prepare_tree_credit` | Physical inputs and native semantics checked again; original n2/n4, cap and public-origin denominators; unresolved routing excluded |
| Group credit | `repair_credit.repair_group_credit` | Original token ownership precedes group centering; malformed members remain zero-valued original members |
| Optimizer and S evidence | Repair branch of `updates.optimizer_step` | CPU numerical/gradient checks passed; actual Qwen answer-reward update/reload passed; native CORRECTED-only Qwen S remains unverified |
| Profile and cache | `build_protocol(repair_notebook=True, context_limit=24576)` and explicit repair cache binding | Legacy protocols unchanged; explicit CLI and cache-source CPU tests passed |
| Saved update evidence | `repair_evidence.repair_update_evidence` | Recomputes original masks, coefficients, retained signal and recorded signed/positive VJP consistency; does not remeasure a GPU gradient |
| Original capture replay | `repair_replay.replay_capture/replay_trees/checkpoint_snapshot` | Fresh CPU physical/native replay, same-byte raw evidence and all five checkpoint files bound before proof hashing |
| Fixed-ten/control/reload | Repair branches of `local_train` and `local_run.reload` | Actual fixed-two F/update/reload passed; fixed-ten/control and current S gates remain unverified |
| Independent n1 evaluation | Repair branches of `local_ladder_recovery`, `local_ladder_generation`, `local_ladder_gate`, `local_eval` | Typed retries and full-history final notes; all n1 groups remain untrainable; fresh native replay and distinct repair precision/coverage |

`PublicOriginInventory` is the implemented refinement of the design's shorter
`origins` argument: it binds every source's explicit response contract and parse
state as well as its refs. A ref string alone is not evidence. Pass a contract-map
key for every supplied response; never infer a contract from an answer/header.
Malformed typed responses are retained as opaque sources, not parsed as legacy
solutions. `public_attempt_outcome` independently verifies syntax failure and
projects it to repair-local INVALID_RESPONSE_CONTRACT/answer=null while preserving
Sample.result=None. These complete members remain in their original usable group
with zero PG; only independently graded valid members can become retry parents.
Genuinely missing grades or corrupted captures remain blocking. Wrong authored
versions are field failures; wrong supplied contracts are fatal configuration
errors. Parent scope validates real ancestry before selecting ancestors.

The frozen dictionary/feedback wire passed123 connected CPU tests, independent
28-test/33-negative-case review, actual processor capacity checks and source-bound
preparation. Preserved legacy histories with explicitly hypothetical current typed
feedback required8749/14971/15882 input tokens plus8192 output reserved within24576;
future histories are not guaranteed to fit. The subsequent actual base readiness
save/check passed two generations and68 identical log probabilities in distinct
processes. Save quantized the local original in memory for cache comparison; check
loaded the unchanged NF4 cache with original access blocked. No trained checkpoint
reload was established.

That actual trial exposed the malformed-sibling continuation defect above.
`repair-note-v2` / `repair-notebook-full-history-v2` now includes syntax-invalid
attempt metadata while preserving raw text, token probabilities and full original
history. Complete solution groups must be usable and their episode complete before
training preparation; no quarantine exception converts missing integrity to success.
The corrected frozen profile passed380 connected CPU tests and actual Qwen F
validation. A subsequent no-update retry probe reproduced original outputs but
retained the wrong visual relation after explicit observation re-emission. Its
other response exposed a harmless blank line between JSON and RevisedSolution.
The current boundary-only fix passed382 connected tests, Ruff and scoped strong
mypy on52 files plus independent review. It changes the explicit protocol pin,
keeps original text/character spans, and has not yet had a new model run. The next
two prescribed cohort cases are reserved for source-only diagnosis; their source
opportunities will not certify model-authored patches, S or benefit. Recognized
old typed-wire records require their preserved frozen source; plain legacy feedback
and null root feedback remain literal.

The earlier frozen825-test run had20 failures,1 error and1 skip; all21 failure
records were reproduced in the pre-redesign archive. This is a no-new-failure
comparison rather than an all-green suite. Existing170 inputs, the fixed ten and
fifteen-image cohort remain unchanged.

All `RawSpan` values are half-open Python character positions, not UTF-8 byte
positions or token indices. `claim_token_ownership` can map original token indices
wholly inside a retry claim; unstable decoding abstains and boundary-crossing
tokens are excluded. Training ownership includes original emitted patches and is
freshly checked before optimizer commit. Final `lesson_text` is retained
as untrusted prose and will receive no direct policy-gradient reward.

A native `COMPLIANT` row certifies only its named bounded rule. For example, a
circuit arithmetic check does not establish that every authored operand matched
the image. A completed local unit and whole-response truncation are recorded
separately. Neither a correct final answer nor a model's diagnosis establishes
normal Vision or identifies the responsible internal module.

Validation checkpoints are kept privately under `verified-repair-notebook-v59`:
foundation195CPU tests, public inventory56CPU tests, and the later combined
native/final-syntax103CPU tests (33.564seconds). These overlapping suites are not
354 distinct tests. The last four changed files also pass Ruff and scoped mypy
with `--check-untyped-defs`; dependencies were not all typechecked. Independent
review reproduced484 legacy helper cases and closed a mutable-input audit race.
These early checks are authored CPU fixtures, without a new GPU run. Later tiny
CPU optimizer checks are described below and must not be read as Qwen training.

The online repair audit subsequently passed91 connected CPU tests (5.748seconds),
Ruff, scoped strong mypy and independent review. This earlier checkpoint returned
original character spans; later credit code maps them to generated tokens. Complete local corrections can be checked
even when the final answer is wrong or the response later reaches its token cap;
those facts do not certify the complete solution. The closure is
`online-repair-closure-main-v1.json` under the same private evidence directory.

The final relation audit passed the later162-test connected CPU suite and an
independent review, including malformed-source, truncation, ambiguous observation,
metadata and transitive-coverage cases. This tests bounded supplied-history
semantics, not model-authored notebook quality. Its receipt is
`final-semantics-closure-main-v1.json`. The same suite's retry collector was **not**
approved at that checkpoint: review found callback mutation of budgets/seeds and an
underspecified native patch format. The fixes now freeze detached configuration and
reject changes to actual requests before another draw. The later collector/final
review closed those findings, including image mutation and tree identity.

Original-token mapping and the separated numerical loss kernel subsequently passed
27 connected CPU tests (2.703seconds) and independent review. The mapper rejects
mutated decode prefixes or inconsistent repeated full decodes. The loss preserves
the original n/cap denominator for microbatches and MIS exclusions. The receipt is
`token-loss-closure-main-v1.json`; it certifies neither fresh group semantics nor
optimizer integration. No actual model call or weight update was made for these checks.

The later preparation/credit/final-attachment check passed44 tests in32.845seconds,
and the optimizer/profile/loss integration passed71 tests in64.103seconds.
The latter18 source files were unchanged during execution. Independent review
matched direct loss and L/M/V derivative calculations, original denominators,
single KL and signed/positive CORRECTED-only sums; an ON-to-OFF update preserved
Vision weights and nonzero Adam state. Late semantic drift aborted before commit.
Ruff and scoped strong mypy passed; full dependency typechecking was not performed.
These suites overlap and must not be summed as unique tests. The closure is
`p10-profile-closure-main-v1.json`, SHA
`3b2341e020db49c63f0565a47a5d7a10c76e537e4a68bf131746ebd60fda5020`.

Functional update evidence (F), native repair-gradient evidence (S) and measured
benefit (E) remain separate. F permits honest answer-driven updates; S requires
retained verified CORRECTED tokens and nonzero combined signed and positive
repair-only gradient sums. Neither fixture result establishes E. The original
full-action KL and generation remain in the answer-only control; only applied
online/final repair coefficients become zero. No-reward-variance is recorded as
no learning signal rather than manufactured by replacing responses.

The later saved-evidence check passed32 CPU tests in102.042seconds. After a found
checkpoint-chain defect was fixed, the combined minimum/fixed-ten/legacy caller
suite passed85 tests in57.360seconds. All five files of every committed and final
checkpoint now enter the replay proof. State hashes also match recorded parameter
changes; substituting earlier weights or optimizer payloads cannot reuse an old
reload receipt. The subsequent additive matched-arm runtime-file propagation was
verified separately, including the full minimum parent gate with genuine tiny CPU
optimizer payloads. Native semantics and reload outcomes in those adversarial
fixtures were scripted. This is not a real Qwen reload or self-improvement result.
The private closure is `p13-callers-closure-main-v1.json`, SHA
`a1649b024b10ee53de3e7ecf8f6f3c25e6f892082f0fb4df74caa4bafa42b914`.

The n1 circuit evaluator is now connected under the explicit policy
`circuit-ladder-verified-repair-notebook-v1`, with `--with-recovery` and context24576.
It requires the current minimum F+S evidence and a closed fixed-ten run/reload;
a fixed-ten run with F but S=false can still be evaluated honestly. The R0 arm also
binds its original R1 capture/reload. All source and checkpoint bindings are checked
before and after inference. The evaluator never calls the training collector.
Its malformed or missing notes remain in the independent coverage denominator,
and historical strict-note metrics are null. `new_error_count` covers only
identity-bound native child issues absent from their parent; syntax/unknown states
remain separate. CPU replay does not certify fresh feedback retokenization.
Focused authored CPU tests and independent boundary reviews passed; no real
Qwen evaluation or GPU reload has run under this new protocol.

Next dependencies: freeze and run combined CPU/portability checks → fresh
source-bound data/capacity/cache validation
→ actual fixed1–2item NF4 test and fresh-process reload.
The old fixed10, answer-only control and independent evaluation cannot be relabeled
as evidence for this new contract. Preserve the pinned model, data exclusions,
old protocols, complete raw history and existing evidence throughout.

## Historical local pretrained NF4 workflow (2026-10-21)

### Whole-note assessment credit (2026-10-26)

The opt-in `local-nf4-whole-reference-assessment-credit-24k-v3` keeps the complete
history, images, monolithic notebook prompt, strict grader and evidence-based
adapter routing. It adds verified assessment credit to the original generated
flags, dependencies and nonnull decision tokens. An authored null decision receives
no positive or negative assessment token credit. Correct flag/dependency judgments
can still earn partial credit without claiming a complete error relationship.
Every public native reference remains in
the denominator, including entries without a verified issue. No target text is
filled in, repaired or imitated through SFT.

Version 3 is an explicit assessment-only normalization experiment, not a bug repair.
Its `verified-whole-reference-assessment-credit-v3` contract multiplies each mapped
assessment coefficient by the fixed normalization cap8192:
`8192 * 0.0625 * (reward - original_n4_mean) / all_public_native_refs / mapped_union_tokens`.
The shared loss still divides by `n4 * cap8192`, so only this channel cancels the
cap division. Strict, fact, relationship, answer and KL terms, original n4 means,
fixed denominators, action/group weights, masks, clipping and routing stay unchanged.
Finite raw coefficients can exceed1; they are not clipped to1. The optimizer still
requires original n4 and cap8192. Scaling does not create variance when all four
classifications are identically wrong, nor any missing N or relationship signal.

The saved v1 coefficient diagnosis measured effective assessment coefficients of
about5.71e-10–1.713e-9, versus one answer-token example7.3242e-5. This fixed scale
would yield about4.678e-6–1.4034e-5 for those same arithmetic inputs; coefficient
ratios do not establish parameter-gradient ratios or predict notebook improvement.
Provenance: private `whole-assessment-learning-v54/assessment-scale-diagnosis-main-v1.json`.

Version 2 fixes a formatting-dependent v1 path that could reinforce a missing
decision encoded as `null`. Existing v1 captures and checkpoints remain bound to
their frozen v1 code; they are not v2 or v3 minimum-training or reload evidence.
Version 3 retains the null-safe mapping. V2 and the aggregate-gradient check remain
unverified on actual Qwen training. V3 requires fresh source-bound preparation,
cache save/check, actual minimum training and separate-process reload; historical
proofs cannot certify the new policy.

Use all five flags for source-bound `local_run prepare`, `quantized_cache save`
and `check`, and `local_run minimal`:

```bash
--verified-note-assessment-credit --verified-note-synthesis-credit \
  --pair-draft-synthesis --whole-reference-notebook --context-limit 24576
```

The minimum requires positive assessment credit after MIS and a finite, nonzero
signed sum of all retained assessment-only parameter VJPs before committing an
update (`verified-native-assessment-aggregate-gradient-v2`). Negative-only vectors
participate; retained zero vectors and MIS exclusions stay in the membership record.
The sum uses stable L/M/V parameter identities, so opposing candidates can cancel
and block the update. Per-candidate norms cannot establish aggregate contribution.
The CPU float64 sums are released before the optimizer; actual model memory usage
under this check is not yet measured. Facts, relationships, answer
reward and KL cannot substitute for this evidence. Save/reload records bind the
same evidence; `assessment_learning_verified` does not certify complete notebook
correctness, native relationship learning or self-improvement.

After an actual minimum update and a closed fresh reload, the explicit
`verified-assessment-credit-exploration-v1` admission connects the original fixed
ten items to training, reload and the matched answer-only control. It requires
the exact assessment protocol and independently replays every committed pair's
original responses, rewards, MIS decisions and gradient evidence. Both arms start
from F0; they do not continue training the minimum checkpoint. No-signal or
unresolved pairs remain recorded without replacement or a forced update.

```bash
python -m fine_tuning.notebook_rl.local_train train \
  --preparation "$CURRENT_PREPARATION" --model-path "$NF4_BASE" \
  --cache-check "$MINIMUM_CACHE_CHECK" \
  --minimal "$MINIMUM" --minimal-reload "$MINIMUM_RELOAD" \
  --admission verified-assessment-credit-exploration-v1 \
  --arm notebook --max-seconds 14400 --output "$TEN_NOTEBOOK"

python -m fine_tuning.notebook_rl.local_train reload \
  --capture "$TEN_NOTEBOOK" --output "$TEN_NOTEBOOK_RELOAD"

python -m fine_tuning.notebook_rl.local_train train \
  --preparation "$CURRENT_PREPARATION" --model-path "$NF4_BASE" \
  --cache-check "$MINIMUM_CACHE_CHECK" \
  --minimal "$MINIMUM" --minimal-reload "$MINIMUM_RELOAD" \
  --admission verified-assessment-credit-exploration-v1 \
  --arm answer-only --match-initialization "$TEN_NOTEBOOK/initialization.json" \
  --max-seconds 14400 --output "$TEN_CONTROL"

python -m fine_tuning.notebook_rl.local_train reload \
  --capture "$TEN_CONTROL" --output "$TEN_CONTROL_RELOAD"
```

The control removes DIAG reward credit while retaining the original DIAG actions
for KL. It does not certify assessment learning. The arms can produce different
trajectories, eligible pairs and update counts, so this is a training-procedure
comparison, not an isolated or compute-matched notebook effect. Use one common
source-bound preparation and circuit evaluation gate for F0, notebook-trained and
control-trained evaluation. Full-notebook evaluation requires the explicit
`circuit-ladder-reference-notebook-v1` policy; the legacy short-note evaluator is
insufficient. These paths have CPU tests; actual Qwen fixed-ten training and
new-question improvement under this protocol remain unverified.

Historical v1 CPU integration checks passed44 tests. A separate diagnostic regraded the four
original V49 notes without training on them: strict0/4, assessment correctness
3/10/7/7 out of167, with27 positive and13 negative mapped original tokens. All157
N references were wrong in all four candidates and therefore still have zero
relative signal. All relationship decisions remain null. This partial signal is
not a completed notebook or evidence of improvement; actual Qwen update/reload
under v2 or v3 remains unverified. The frozen v1 minimum generated the same four notes,
completed one actual update and a separate-process reload, and measured a nonzero
assessment-only VJP for its first candidate. That is not evidence of the signed
aggregate or accurate complete notebooks. Its closed same-history post-training
comparison remained strict0/4→0/4, with zero recoveries or regressions. All four
texts and token sequences were unchanged, although their log probabilities differed;
verified facts, native relationships and OUTPUT units remained0 and the summed
atomic reward stayed0.010104790419161677. The fresh CPU replay is private
`whole-assessment-followup-v55/same-history-audit/terminal-v2.json`
(SHA256 `27d1a1d0b5eb61bb145a410ec4a721c58a428112c16ecccf907e0af3d68404bb`).
This seen-history comparison does not establish new-question improvement or an
isolated notebook-reward effect. A tokenizers-only check of those actual notes
found no mapped null-decision tokens with nonzero assessment credit; this does not
make the general v1 mapping safe for new response formats.

After a verified minimum update and independent fresh reload, the same-history
comparison also supports this assessment objective. It reuses the original full
history, successful retry, drafts, images and four seeds; both saved and newly
generated notes are checked with the same objective and unchanged strict grader:

```bash
python -m fine_tuning.notebook_rl.fact_context_probe \
  --preparation "$CURRENT_PREPARATION" --model-path "$NF4_BASE" \
  --cache-check "$MINIMUM_CACHE_CHECK" --parent "$MINIMUM" \
  --trained-notebook-reload "$MINIMUM_RELOAD" --output "$NOTE_COMPARISON"
```

`CURRENT_PREPARATION` must bind the executing source. The original cache check
can be reused only when its twelve readiness sources, model metadata and protocol
still match. The minimum's original preparation and checkpoint evidence remain
unchanged. This comparison performs no update and is not held-out evaluation or
proof of the notebook reward's isolated effect. Its three focused CPU checks,
Ruff and bounded mypy passed; actual trained-Qwen comparison remains unverified.

### All ancestor comparisons: bounded native-learning stage (2026-10-25)

`--pair-action-notebook --verified-note-synthesis-credit --context-limit 24576`
now selects `local-nf4-pair-action-notebook-synthesis-credit-24k-v2`. The same
model receives all original attempts and images, and independently labels every
public ancestor comparison with four sampled bare labels, up to128 output tokens
each. Selection includes stable and changed observations, TRACE operations and
both retry branches; it never consults private truth. The original sampled label
tokens receive independently verified credit. No corrected notebook is supplied
as a supervised target. Ancestor comparisons omit isolated errors, sibling
comparisons and full facts/unknown inventories, so `full_notebook_verified=false`.

The optional `fixed-ten-native-exploration-v1` admission separates collection from
proof of learning. It uses fresh F0, the original ordered ten items, five two-item
batches and an explicit time limit. Failed/no-signal pairs remain recorded and are
not replaced. Every committed update must retain positive native error-relationship
credit after MIS and a nonzero isolated native VJP. Answer-only, output-only or
KL-only evidence cannot satisfy that requirement. The default minimum-first
admission remains unchanged; answer-only controls require a separate compatible
protocol and cannot bypass a native-only requirement.

After preparing current source-bound inputs and matching cache save/check records
with the flags above, the explicit collection command is:

```bash
python -m fine_tuning.notebook_rl.local_train train \
  --preparation "$PREPARATION" --model-path "$NF4_BASE" \
  --cache-check "$CACHE_CHECK" --output "$TRAIN_OUTPUT" \
  --arm notebook --admission fixed-ten-native-exploration-v1 --max-seconds 14400
```

Reload rechecks the source/data/cache/ordered-cohort contract and every counted
native commit. A collection record is not a successful-minimum receipt. Actual
Qwen V2 update/reload, complete-notebook correctness and disjoint first-attempt
improvement remain unverified. A later full-notebook evaluation must use the
explicit `circuit-ladder-reference-notebook-v1` evaluation protocol with newly
bound inputs; the default legacy2048-token note does not establish that outcome.

The actual V53 minimum completed54 comparison groups/216 labels with zero updates.
Two native issue targets (`dir[e4]`, attempt3 to5/6) each earned `[1,1,1,1]`, hence
zero centered advantage. Only the two OUTPUT targets had reward variance. The
other50 native targets were outside an earlier verified issue; their false
relationship claims received no unit credit or penalty. This is the declared
partial-objective scope, not proof of correct complete assessments.

### New full-note comparison after a verified pair update

The following CPU-tested path requires an actual native minimum update and a
closed fresh reload. V53's zero-update result is ineligible. Both arms use a new
whole-reference prompt, four seeds, an8192 output budget and24576 context; old
pair labels are not baseline notes. Use a current source-bound preparation with
the same pair-action protocol. Historical source/preparation evidence stays bound
to the original training run, and the target model must match its recorded weights.

```bash
python -m fine_tuning.notebook_rl.fact_context_probe \
  --preparation "$PREPARATION" --model-path "$NF4_BASE" \
  --cache-check "$CACHE_CHECK" --parent "$MINIMUM" \
  --trained-notebook-reload "$MINIMUM_RELOAD" \
  --pair-to-whole baseline --output "$WHOLE_BASELINE"

python -m fine_tuning.notebook_rl.fact_context_probe \
  --preparation "$PREPARATION" --model-path "$NF4_BASE" \
  --cache-check "$CACHE_CHECK" --parent "$MINIMUM" \
  --trained-notebook-reload "$MINIMUM_RELOAD" \
  --pair-to-whole trained --pair-to-whole-baseline "$WHOLE_BASELINE" \
  --output "$WHOLE_TRAINED"
```

The supervisor limits each arm to one hour. The first recovered fixed-two history
is selected without replacement; all original attempts, siblings, successful
solutions, images and the unverified reread remain present. The F0 and trained
arms must have identical actual prompt tokens, image tensors and seeds. Complete
raw outputs are independently graded under the unchanged whole-note rules.
No optimizer runs, and these diagnostic outputs are not training actions or
held-out evidence. Actual Qwen execution of this new comparison is unverified.

### Paged full-history notebooks (implementation under validation)

The current page-specific prompt is protocol
`local-nf4-whole-reference-pages-synthesis-credit-24k-v2`. Header requests contain
only header grammar/examples; assessment requests contain only assessment
grammar/examples and a read-only successful-retry identifier. The quoted draft
wrapper requests the current page. Original histories, images, drafts, verification,
numeric types, rewards and learning gates are unchanged. V1 captures remain bound
to their frozen V1 source; they cannot be restored as V2 or used as new actions.
The actual V1 minimum was interrupted after all four page1 responses returned
the wrong fields, making every notebook ordinal structurally invalid. It made
zero updates. The actual V2 trial was also stopped with zero updates after all four
headers failed and an assessment page proposed same-attempt corrections. Raw
responses are preserved. Correct complete notebooks remain unverified.

`--whole-reference-pages --pair-draft-synthesis --verified-note-synthesis-credit
--context-limit 24576` selects a separate training protocol. Use these flags with
the existing prepare, cache save/check and minimal commands. Fresh source-bound
input and cache records are required; the old captures are not reused as training
actions. Reload restores the protocol recorded in the actual checkpoint.

The model writes a header and one assessment page for each original attempt.
Every request contains the full original history, images, unverified reread and
drafts. No earlier generated page is supplied to another page. Four actual samples
are generated per page; ordinal k across every page forms logical notebook k.
Assembly copies those samples' literal fields without selecting, repairing or
inventing judgments. A missing page prevents training; a malformed or truncated
page gives the whole ordinal zero credit. Completed raw PAGE events are saved
immediately, while trainable GROUP events wait for complete verification.

Each logical notebook retains the total8192 output budget: header2048 and6144
distributed by public assessment-reference counts, with deterministic remainder
allocation. All pages together contribute one DIAG group weight and use the
4×8192 action normalizer. MIS uses the token-weighted mean across an ordinal's
actual pages and retains or rejects them together. Four-candidate centering is
computed before MIS, without recentering survivors. Verified partial credit maps
only to its original generated page tokens; assembled text is never a sampled
training action. Positive retained native-relationship credit and a finite,
nonzero native-only VJP remain mandatory before committing notebook training.

The exact trained-history replay keeps policy0 inputs and generates genuine
policy1 page responses for diagnosis only. This is not new-item evaluation.
The separate n1 circuit evaluation policy below does not yet generate pages.
For frozen V1 on the preserved actual six-attempt history, CPU processor checks found seven
page requests with12844–13017 input tokens and at most14892 tokens including
the reserved page output, within24576. The original image tensors were preserved.
These are input-capacity measurements, not model generation or learning evidence.
Paging increases requests and repeated input processing; the same output budget
does not imply the same elapsed time. Accurate real-model notes, actual paged RL
and fresh reload remain unverified until the new run completes.

### Complete reference notebooks in the existing RL trainer (2026-10-25)

The explicit `local-nf4-whole-reference-synthesis-credit-24k-v1` profile connects
the complete reference representation to the existing episode collector, strict
verifier, token-credit loss, optimizer and checkpoint workflow. Select it for
`local_run prepare`, `quantized_cache save/check` and `local_run minimal` with:

```bash
--whole-reference-notebook --pair-draft-synthesis \
--verified-note-synthesis-credit --context-limit 24576
```

Use fresh source-bound preparation and cache-readiness records. The existing
`local_run reload --capture <completed-minimal> --output <new-reload>` command
restores the saved protocol. The old diagnostic captures are never training input.

Every original solution, successful retry, image, public termination reason and
unverified draft remains available. The model supplies all image facts, per-attempt
outcomes and issue/unknown/relationship decisions. Expansion only resolves its
references into bookkeeping; the loss always uses its original generated tokens.
Complete-note strict correctness remains separate from individually verified claims.
The fixed n4 group, 8192-token note budget, 24K context and existing reward weights
are retained; missing judgments are not filled by the host.

This profile requires positive native error-relationship credit after token masks,
dtype conversion and MIS, then a finite nonzero native-only VJP before committing.
Fact, final-answer-completion or KL signals alone cannot satisfy that requirement.
Answer-only controls explicitly omit notebook reward and cannot certify notebook
learning. Independent observations still determine Vision routing; the notebook's
own module diagnosis never does. No-recovery and blocked inputs remain recorded.

The post-update same-history probe preserves the original policy0 history and
scores new policy1 replies as diagnostic-only; these scores cannot enter RL.
CPU-authored checks exercise real differentiable updates and rejection paths.
The actual V49 minimum finished in2967.898seconds with14 solution responses and
four final notes. All four stopped naturally and covered173 references each, but
all692 relationship decisions were null. Blanket unknown/error flags and invalid
fact types also remained. Native and OUTPUT relative credit were both zero;
`NO_ERROR_RELATIONSHIP_LEARNING_SIGNAL` stopped execution with zero updates and
no new checkpoint. A23.276-second CPU replay reproduced that failure with all337
frozen source files unchanged. This is failure evidence, not notebook learning.
This profile's accurate notes, actual update/reload and new-item improvement
remain unverified; earlier failed captures remain unchanged.

### Explicit circuit reference-notebook evaluation (CPU validated)

The existing `local_eval evaluate-ladder` command accepts
`--with-recovery --evaluation-prompt-policy circuit-ladder-reference-notebook-v1`.
It still requires the existing source-bound cohort/data gate, completed training
capture and separate-process reload evidence. This new policy is for all compared
F0/notebook/answer-only conditions, never for only the favorable arm. It uses the
same pretrained VLM with n1, solution1024, note8192 and context24576; at most three
solution attempts are recorded. Every original attempt, successful retry and image
remains in the note input. There are no fresh rereads, draft calls or note repairs
in this evaluation protocol, so its assistance is explicitly different from training.

Circuit answers, per-attempt outcomes, structural operations, unknown regions and
complete-note correctness are separate measurements. The bounded native ladder
trace checks node potentials, resistor endpoints/units, Ohm's law, KCL, equivalent
resistance and the requested quantity with exact rational arithmetic. Native nodal
and series-parallel solvers must agree; the trace checker never reads the answer
key. Bridges and unsupported circuits remain outside this contract. Explicit wrong
component units fail; missing units or unsupported prose remain unverified. The
new evaluation-only observation routing uses these checks instead of legacy
unit-stripping scores. It certifies neither module causality nor training admission.

The existing raw-request reconstruction and fresh CPU replay bind the full prompt,
images, seeds, native contract and strict grading. Singleton groups are always
unusable for RL, even if evaluation markers are removed. Authored test fixtures
validate these contracts; no new circuit model evaluation or improvement has yet
been observed. The historical generic policy, graph training and saved baseline
remain separate and unchanged.

### Complete reference-assessment diagnosis (2026-10-25)

The last full-history coverage diagnostic produced four naturally stopped notes,
but all failed strict verification and supplied zero native relationship signal.
`fact_context_probe --notebook-whole-reference` now requests one model-authored
judgment for every public observation, trace operation and uncertainty scope,
including successful attempts. Original solutions, repeated occurrences, images,
unverified reread/drafts and four seeds remain intact. Compact line references
reduce repeated input metadata without truncating the source text.

The host expands only the model's explicit choices into the existing strict
grading format; it never fills missing decisions or supplies expected facts.
Diagnostic credit maps to original generated JSON spans and reports native
relationships separately from final-answer completion. This command is a zero-update
diagnostic; the separately versioned training profile above uses fresh rollouts.
The same command below can use `--notebook-whole-reference` instead of
`--notebook-whole-synthesis`, with fresh current-source preparation. CPU checks
have passed. The actual V48 diagnostic subsequently finished in 2365.584 seconds:
all four notes stopped naturally, but strict verification passed 0/4 and no raw
relationship pairs were submitted. Uniform assessment flags, incorrect image
facts and malformed fields remain. Its unspecified `outcome_string` vocabulary
also confused outcome statuses with assessment flags. The current public grammar
defines both vocabularies, literal JSON types and observed termination metadata.
Existing strict checks and saved outputs remain unchanged.
The 16.227-second CPU replay reproduced this failure, not successful learning.
Subsequent RL integration and accurate whole-notebook learning remain unverified.

New reference captures also write `completion-NNNN.json` immediately after each
finished seed, so a later timeout preserves those raw tokens, log probabilities
and text. These sidecars are diagnostic evidence only; partial groups are never
scored, resumed or used for RL. The original n4 request and its full-group credit
calculation are unchanged. CPU generation parity and interruption tests pass;
the completed frozen V48 capture was not modified and did not have this hook.

The optimizer now records a separate `native_relationship_gradient` for the first
retained action with positive verified native credit after token masking. Its VJP
excludes answer-completion, fact, strict-note and KL terms. An earlier OUTPUT-only
measurement cannot consume this measurement; identical native-only vectors reuse
the existing same-forward VJP. CPU tests verify isolation, masks, zero derivatives,
failure cleanup and frozen Vision. Historical admission rules remain unchanged;
the new explicit reference profile additionally requires native-only commit evidence.
This instrumentation does not certify a correct complete notebook, model-module
causality or improvement, and has not yet been measured on the new real-model run.

### Saved-history whole-notebook diagnosis (2026-10-24)

The V45 minimum completed two fixed items but made zero optimizer updates. Its
four final notebooks all stopped naturally and failed strict verification: each
left the error/relationship inventories empty despite revising an observed fact.
This is not an output-budget or model-loading failure. The original capture,
strict scores and zero-update result remain unchanged.

`fact_context_probe --notebook-whole-synthesis` tests a separately hashed prompt
on that closed history. It requests image facts, per-attempt issues, then lessons,
and explains the public trace/unknown-region grammar. This combined prompt change
does not isolate output order alone. The same complete solution history, successful
retry, images, unverified reread, all unverified drafts and four original seeds
are retained. Only the final n4/8192 notebooks are regenerated, within 24K context
and the existing one-hour diagnostic limit. No optimizer or solution retry runs.

```bash
python -m fine_tuning.notebook_rl.fact_context_probe \
  --notebook-whole-synthesis \
  --preparation <current-source-pair-draft-preparation.json> \
  --model-path <standalone-NF4-base> --cache-check <matching-cache-check> \
  --parent <closed-V45-minimal-directory> --output <new-diagnostic-directory>
```

The existing strict notebook and per-claim graders are unchanged. Complete-note
success, partial verified claims, potential relative-reward signals and actual
learning are separate results. CPU tests do not establish better model-generated
notes; this diagnostic also cannot establish learned improvement on new items.

### Observation-verifier correction (2026-10-24)

The fixed-ten pair-action trial was stopped after eight items and two completed
updates. Its verifier rejected quoted but correct relation values such as
`dir["e0"] = "A->D"` and mistakenly rewarded some labels as persistent visual
errors. Among seven variable-reward groups in the eighth item, five reflected
this representation error and two contained actual reversed arrows. The third
update was interrupted; original responses, scores and intermediate saves remain
unchanged. Mechanical replay of the old scores does not certify their semantics.

The corrected verifier has identity `native-graph-relations-v3`. Every new local
protocol binds that version, and the whole-notebook draft protocol is
`local-nf4-pair-draft-synthesis-credit-24k-v2`. Old preparation files cannot be
rehashed into the current contract. Explicit historical v1 replay still requires
identical freshly recomputed semantic fields; v2 is not automatically accepted.
The new contract needs CPU checks and fresh source-bound preparation before actual
whole-notebook generation, learning and reload can be certified.

### Mandatory whole-history notebook after unverified drafts (2026-10-24)

The new opt-in is `--pair-draft-synthesis --verified-note-synthesis-credit
--context-limit 24576`, consistently for `local_run prepare`, `quantized_cache
save/check` and `local_run minimal`. Reload restores the recorded protocol. It is
exclusive with the partial/action generation flags and requires fresh source-bound
preparation. It has not yet passed actual pretrained-model generation or learning.

After a successful retry, the model rereads the images and generates public pair
proposals, then writes one final n4/8192 whole-history notebook. Every original
solution, sibling branch, successful retry, image and raw proposal is preserved.
Malformed, truncated, contradictory and UNRESOLVED proposals remain explicitly
unverified context. Missing or ambiguous public comparisons are recorded and do
not prevent the final notebook; budget/context exhaustion preserves the available
records and blocks further generation without trimming history.

Only final model-authored notebook samples enter the existing verifier, reward
and RL loss. Draft tokens are neither DIAG actions nor rewarded targets, so the
same error receives no additional draft-stage policy credit. Routing still uses
independent checks of the original solutions. The final checker retains complete
history coverage, unsupported-claim rejection, unknown regions and fixed reward
denominators. A host-rendered summary or successful draft label is not a final
notebook, and free-form explanatory prose is not semantically certified.

The interrupted fixed-ten trial used the earlier pair-action protocol described below;
it was not an execution of this whole-notebook path. Actual notebook quality,
final-token learning, both routes and new-item improvement require separate evidence.

### One-pair diagnosis: observed result and implementation boundary (2026-10-24)

The actual frozen Qwen4B NF4 diagnostic preserved the original six attempts,
successful retry, both images and unverified reread. Three independent requests
produced twelve bare relationship labels in 187.511 seconds. All twelve satisfied
the output format. The independently checked counts were: persistent observation
error 4/4, a second observation comparison 0/4, final-answer completion 3/4.
This demonstrates a usable simpler output interface on one history, with remaining
semantic mistakes. It does not establish complete notebook accuracy or learning.

The three n4 reward vectors were `[1,1,1,1]`, `[0,0,0,0]` and `[1,0,1,1]`.
Consequently native relationship advantages were zero; only finalization had
relative reward variation. The diagnostic used no optimizer, and its saved
responses are not training inputs. The subsequent explicit pair-action policy
must generate new actions, bind each actual label to its public target and retain
independent regrading, routing, token credit, update and fresh-reload checks.
It is a partial notebook policy: the first ancestry edge and varying authored
slots do not cover every error in the episode. Host summaries are display
artifacts, never model-generated completions or evidence of comprehensive insight.

Independent target calls still include the entire input history. More calls mean
more prompt prefills and backward passes. Using output cap 128 instead of 8192
changes the existing `n_group * cap` loss normalization; additional DIAG groups
also change group mixture weights. This is a separately recorded training
protocol, not a matched estimate of the old notebook policy's effect. The older
layout and long-history diagnostic commands do not measure this new action policy.

The implemented opt-in path is
`--pair-action-notebook --verified-note-synthesis-credit --context-limit 24576`.
Use these flags consistently with `local_run prepare`, `quantized_cache save/check`
and `local_run minimal`, each with fresh output paths. `local_run reload` restores
the saved contract without changing the policy. This path is exclusive with the
older partial, referenced and paired generation flags. The actual fixed-two run
completed one RL update in 921.842 seconds. CPU replay verified 14 solutions and
12 note labels; the saved 624 tensors and a separate-process optimizer/weight/
forward/generation reload passed. L252/M24/V48 tensors changed with the NF4 base
frozen. Isolated note reward gradient was OUTPUT_FINALIZATION-only; native
relationship learning and whole-notebook correctness remain unverified.

A subsequent same-history replay used unchanged prompts, images, targets and
seeds with the verified trained checkpoint. Native judgments remained 4/8 and
OUTPUT judgments remained 3/4 (no recovered or regressed judgments). This is not
held-out evaluation or evidence of improvement. The existing
`verified-synthesis-credit-exploration-v2` admission passed on the actual closed
update/reload; the fixed-ten local notebook trial is running, not completed.
Its control arm, current-protocol dual-route validation and new-item evaluation
remain pending. No large training run is authorized by these results.

Each independent target keeps its original prompt, image binding, seeds and raw
sampled label tokens. Legacy whole-note assessment is retained for diagnosis, but
its applied reward is explicitly zero in label mode. Only independently verified
relationship-unit rewards contribute; unsupported labels or legacy JSON do not
gain strict reward. `UNRESOLVED` receives zero reward; on a known original issue
its actual label can receive a negative centered coefficient relative to correct
siblings. Equal rewards produce no policy-gradient signal. Update checks reject
the new policy when no relationship policy-gradient signal survives masking.
Output-finalization-only learning remains a separately reported valid scope; it
does not certify native reasoning correction or a complete error notebook.


### Public-pair notebook generation (2026-10-24; no actual learning signal)

`--verified-note-synthesis-credit --paired-notebook --context-limit 24576`
selects a separate protocol. It is mutually exclusive with the partial and
reference generation policies. Preparation, cache verification and the minimum
run use the same flags; trained reload restores the exact saved contract.

The model receives every original solution, branch, successful retry, image and
unverified reread. A public list identifies candidate comparisons between actual
ancestor and descendant attempts. Observation pairs refer to the same authored
slot; native trace pairs use authored operation identities rather than array
positions alone. Ambiguous identities remain candidates for independent rejection.
The list is not filtered by private correctness, gold answers or expected lessons.
Candidate membership does not certify an error, correction or repeated error.

The generated notebook contains only `coverage: "PARTIAL"` and
`lessons: [{"pair_id": ..., "relation": ...}]`. The model chooses both the pair
and `CORRECTED`, `PERSISTED` or `FINALIZED`. The host expands only the referenced
bookkeeping, then applies the existing semantic verifier. Unsupported claims earn
no credit. The first identifiable claim for each original issue is selected before
grading, so repeating different guesses cannot rescue a wrong first claim. Only
the originally sampled lesson-object tokens can receive relationship credit.
Omitted fact credit remains zero; weights and denominators are not redistributed.

The minimum retains its default relationship-signal requirement. The weaker
answer-update exploration flag is unavailable for this protocol. After a real
relationship-gradient update and fresh reload, the existing explicit
`verified-synthesis-credit-exploration-v2` admission can check a fixed-ten run.
The default strict whole-notebook admission is unchanged. Native relationship
learning and final-answer-presence (`FINALIZED`) learning must be reported
separately. Neither partial notes nor finalization-only credit certify complete
diagnosis, healthy Vision, module causality or improvement on new problems.

The actual V41 two-item run ended after 818.034 seconds with zero optimizer
updates. All four notes stopped naturally: three submitted empty lessons, and
one copied inventory-style rows without relationship labels. CPU replay preserved
all 18 responses and reproduced the failed result. Complete input delivery and
working weight persistence in earlier runs do not establish notebook learning.

### Isolate notebook relationship judgment before another RL run

`fact_context_probe --notebook-pair-classification` reuses a closed, zero-update
V41 history. It independently asks about one public comparison at a time, keeping
all original attempts, successful retries, images and the unverified reread.
Targets are the first direct retry edge, every slot with differing authored values
across the histories/reread, and final-answer presence. Selection does not use
private correctness or expected relationships.

Each of the three targets in the current history receives the original four
notebook seeds. The diagnostic permits 128 output tokens, uses the existing 24K
context and stops within one hour. It requires one label: `CORRECTED`, `PERSISTED`,
`FINALIZED` or `UNRESOLVED`. The host copies comparison bookkeeping only and
checks the chosen label with the unchanged independent semantic verifier.
Abstention and finalization-only success do not certify visual diagnosis.

```bash
python -m fine_tuning.notebook_rl.fact_context_probe \
  --notebook-pair-classification \
  --preparation <current-paired-source-preparation.json> \
  --model-path <standalone-NF4-base> --cache-check <matching-cache-check> \
  --parent <closed-V41-minimal-directory> --output <new-diagnostic-directory>
```

This mode performs no optimizer update, retry, feedback, routing change or RL
credit calculation. Host-bound fields are not model-generated notebook tokens.
It tests semantic comparison separately from notebook assembly; its success alone
does not satisfy the learning or generalization gates. Historical parent evidence
and current source-bound input approval remain separate.

### Public-reference notebook generation (2026-10-24; actual signal not obtained)

`--verified-note-synthesis-credit --referenced-notebook --context-limit 24576`
selects a separate protocol, mutually exclusive with `--partial-notebook`. Pass
the same flags to preparation, cache save/check and the minimal run; training and
reload restore the exact saved protocol. The existing strict relationship-signal
gate remains the default.

This policy keeps every original attempt, branch, successful retry and image. A
public index enumerates observation slots, syntactic native Trace operations and
an output endpoint for every attempt. The index uses only authored response text
and attempt positions; it contains no private truth labels or expected lessons.
The model selects `earlier_ref`, `later_ref` and a `relation`. The host expands
those addresses into the existing verifier format and cites the entire original
response. This removes citation bookkeeping; it does not measure citation
localization or independently certify the selected claim.

The existing semantic, same-operation and ancestry checks decide whether the
claim is supported. Only the model's original compact claim tokens receive unit
credit; host-expanded fields are never training targets. Unknown references,
unsupported claims and wrong ancestry cannot be rescued using the answer key.
The original strict whole-notebook score, unit denominators, first-claim policy,
all-four centering and independently verified Vision routing remain separate.
CPU fixture success and actual model learning must be reported separately.

The actual V40 two-item run ended after 736.401 seconds with
`NO_ERROR_RELATIONSHIP_LEARNING_SIGNAL` and zero optimizer updates. Its four
notes included two empty outputs and two outputs that confused endpoint addresses
with image slot names. The nonempty lessons selected same-attempt or unrelated
endpoints, or used an unsupported relation. An independent CPU replay reproduced
all 18 responses and the failed outcome. This is a demonstrated interface failure;
it is not evidence that model loading, history preservation or the optimizer failed.
The original capture and zero-update result remain unchanged.

### Partial notebook generation (2026-10-24; real-model signal not obtained)

`--verified-note-synthesis-credit --partial-notebook --context-limit 24576`
selects `local-nf4-partial-notebook-synthesis-credit-24k-v1`. Use the same flags
for preparation, cache save/check and the minimal run. Training and reload restore
the exact saved protocol. This is a separate opt-in generation policy; historical
captures keep their original protocol and scores.

The model sees every original attempt, branch, successful retry and image. It
generates one group of four partial notebooks containing `coverage: "PARTIAL"`,
`corrected_facts` and typed `lessons`. It need not duplicate the final answer or
inventory every error. Notebook repair calls are disabled; problem-solving retries
are unchanged. Empty, malformed, wrong and incomplete outputs are retained.

`PARTIAL_NOTEBOOK_CAPTURED` means only that a nonempty artifact has the expected
shape. It is not a correct notebook or successful learning. The unchanged strict
whole-notebook score and the independent per-fact/per-relationship checks remain
separate. Only original tokens of verified claims receive unit credit, centered
over all four sampled notes; equal scores produce no policy-gradient signal.
No reference fact, answer, teacher lesson or corrected model text is supplied.
The independent observation audit still determines the operational Vision route.

The actual two-item V39 trial ended with four identical, naturally terminated
notes: 17/18 image facts and empty `lessons`. Equal unit rewards supplied no
notebook policy-gradient signal. The default gate stopped before any optimizer
update. This failed generation trial is preserved; it is not notebook learning.

This change addresses a generation/objective mismatch: exhaustive notebook
completion and repeated strict repairs were required even though the objective
already supports individual verified claims. CPU fixtures check this wiring,
including actual optimizer mutation, but do not prove Qwen can write useful
notebooks. Actual generated claims, nonzero reward gradients, checkpoint reload
and separate evaluation are still required before claiming notebook learning or
model improvement.

Local minimal/fixed-ten runners print `UPDATE_PROGRESS` for each action's old-policy
log probabilities, reference log probabilities and forward/backward call, followed
by optimizer START. Events contain IDs/token counts, not response text, and do not
enter the raw-response journal. RETURN means a host call returned, not independently
timed CUDA completion; no extra GPU synchronization is added. An observer failure
aborts before optimizer mutation and clears accumulated backward gradients. There
is no post-commit observer: update/checkpoint/reload evidence still determines
success. This logging applies only to newly frozen runs, never a live snapshot.

The shared HF sampler retains one vocabulary score vector and the selected-token
log probabilities, rather than one full vocabulary vector per generated token.
The observer leaves HF's normalized sampling scores unchanged, reads each actual
chosen token from the next prefix, and flushes the final token after releasing the
returned KV cache. Tiny Qwen CPU tests match the previous generation tokens,
log probabilities and RNG exactly, including single/multiple images and early EOS.
Actual NF4 GPU memory savings, speed and numerical parity remain unmeasured.
Existing frozen runs do not receive this change.

### Opt-in structured whole-history synthesis (v27; actual model validation pending)

The public native-trace instructions now use field descriptions instead of invalid
pseudo-JSON examples. Within the Trace JSON, node/state/edge labels must be quoted
strings, numeric fields must be integers, and a null edge-ID list must be actual
JSON null. Observation assignment syntax and the native verifier are unchanged.
This changes the solution-prompt and protocol hashes; historical captures retain
their frozen instructions and source. CPU regression checks cover the contract,
but improved real-model trace or notebook quality has not yet been measured.

`--verified-note-synthesis` selects a separate local protocol. Pass the same flag
and explicit `--context-limit 16384` or `24576` to `local_run prepare`,
`quantized_cache save/check`, and `local_run minimal`. It cannot be combined with
the older partial-reward flags. Saved protocols are restored only when their
complete prompt, reward, feedback and trace contracts match. The old default and
historical frozen runs retain their original behavior. The trace contract
accepts horizontal whitespace after section headers while preserving original
text and evidence offsets. Frozen v25 responses and their original grades remain
separate; this format fix does not certify their answers or reasoning.
The synthesis consumer uses the same parser and validates the original evidence,
so accepted whitespace cannot fail later under a different header grammar.

This native graph experiment asks the model to submit image observations and one
structured operation trace before its final answer. Supported tasks are directed
shortest paths, BFS, binary-tree traversals and DFA execution. It does not provide
a reference trace or extend certification to arbitrary natural-language reasoning.
Native numeric observations are dimensionless integer strings: `0` or nonzero-led
ASCII digits, at most32 digits, with optional outer spaces/tabs. Units, prose,
signs, leading zeros, decimals and exponents are outside this explicit grammar.
Fact keys/values and string answers must remain single lossless scalar lines;
hidden line separators or controls cannot disappear through projection.

The notebook still receives all original attempts, sibling branches, retries,
successful responses and images. Its typed claims separate observed errors from
incorrect rule use. A repair or repeated error requires matching operations in a
descendant retry; another branch is only a comparison. Every detected issue needs
evidence. Where a failed prerequisite prevents checking a later operation, the
bounded contract requires that exact unverified region and its prerequisite to be
reported. A malformed historical trace, extra prose, or unverified image observation
may now be correctly reported as unknown instead of making every later notebook
automatically fail. The complete unknown inventory must match the independent
audit: whole-response unknowns require all original lines, and unknown observations
require their actual occurrence spans. They do not certify the original operation.
Unsupported native contracts still block learning. Missing known errors, fabricated
unknowns, wrong corrected facts, or a wrong final answer still fail.

If a later response cannot be interpreted as a whole, the notebook cannot claim an
earlier error was absent, fixed or repeated there. Its explicit
`SCOPE_UNVERIFIABLE_LATER` relation only records that evidence limitation. DFA
acceptance unknowns similarly require the actual failed word-completion or final
transition prerequisite; a wrong earlier transition does not invalidate an otherwise
verifiable terminal-state acceptance check.

This protocol explicitly raises only the notebook output cap from 2,048 to 8,192
tokens. CPU capacity examples using the real Qwen tokenizer required 2,203 tokens
for a compact six-attempt BFS note and up to 8,051 for the measured eight-attempt
high-error DFA note, including the stop token. These are authored capacity tests,
not generated Qwen successes or universal length bounds. The chosen 16K/24K
context limit, solution budgets, whole history, binary reward and three notebook
rounds stay explicit. Requests exceeding context are recorded as `CONTEXT_BLOCKED`;
history is never silently shortened. Actual long-note GPU fit and quality still
require measurement.

Successful bounded claims are **not** a complete diagnosis: `all_reasoning_verified`,
`full_diagnosis`, and `lesson_semantics_verified` remain false. Any rendered summary
is a display of checked typed claims, never a teacher target. The existing
independent observation audit decides the operational Vision route; the notebook's
self-attribution does not. The quantized base stays frozen; Vision/LLM adapters and
the four Mergers are the trainable groups, not all model parameters.

Before learning, original text/token/image/history bindings and every note grade
are independently checked again, including immediately before optimizer mutation.
The answer-only control removes note policy-gradient reward while retaining its
specified KL behavior. Expansion still requires positive note advantage and actual
gradient/change/reload evidence. CPU fixtures verify these checks, not accurate
Qwen notebooks or self-improvement. Source-bound input approval, real-model
generation/update/reload, both operational routes and new-item evaluation remain
unverified for v27. The older circuit evaluator does not certify this contract.

### Current full-history extension — implementation; actual notebook learning pending

The current source protocol is `local-nf4-full-history-notebook-v13`, with reward
`local-all-attempt-notebook-v5`. It quotes every actual T1/T2/T3 output, including
sibling branches and the final successful retry, in original order with parent
IDs and lossless line offsets. No earlier attempt is silently dropped. A new
note prompt contains the original image/question and the complete quoted history;
it is not claimed to preserve the retry's literal token sequence. Retries themselves
still preserve their actual conversation. Role delimiter spellings are reversibly
escaped in quoted text. Gold and reference observation values stay in the verifier.
The model receives compact [line ID, original string] pairs and an exact line count.
Full character-offset maps stay in the audit. No attempt, blank line, line ending
or output text is removed to reduce input size.

Reward V5 normalizes only each attempt's `wrong_slots` IDs with the existing
observation parser's case/NFKC/whitespace/bracket-quote rules before comparison.
Normalized order and multiplicity are preserved: out-of-order IDs, duplicates,
wrong or missing IDs still fail. Raw generated text and corrected-fact keys/values
are unchanged. The separate fact-token objective v2 below maps supported fact-ID
spellings to canonical IDs without rewriting their original value tokens.
The reward version and objective contract change the protocol hash. Historical
V4 runs, including the running frozen fixed-ten trial, retain their original
source and scores. Replay/reload those runs with that frozen source; V5 requires
fresh source-bound preparation and validation, never a mid-run switch. This
representation fix does not certify missing facts, diagnoses or prose lessons.

The current initial and repair prompts make the existing native-tree contract
explicit: include the root and every drawn child link using node labels, omit
missing child positions rather than submitting null/none/unknown facts, and do not
invent links. An attempt with no extracted answer still uses JSON `null` in its
`answer` field. This tree-specific instruction does not ban non-tree zero/false
observations or add a verifier for absent children. The prompt/retry hashes and
full protocol hash change, but reward V5, the grader, all-attempt history, routing
and budgets do not. Authored CPU cases check this contract; they do not establish
that the pretrained model follows it or writes a correct whole notebook. Existing
frozen runs retain their original instructions and cannot switch mid-run.

V10's closed minimum produced 12 complete notes on one recovered problem, each
reviewing all six attempts. Every note retained a wrong directed-edge observation;
the answer was correct but both the facts and all-attempt checks failed. Its other
problem had no verified recovery. The exact 26-response CPU replay confirms zero
notebook reward variance and zero updates; no trained reload or improvement exists.
V12 tests history/success anchoring as a hypothesis: write re-inspected image facts
first, compare every attempt against them, then diagnose and summarize. It separates
an incorrect observed arrow from incorrect use of a correctly observed arrow.
Only the note instructions/output-key order and protocol identity change from V11.
The grader still accepts either key order; reward, full history, sample selection,
images, seeds, budgets and route checks do not change. This is a new experiment,
not a fix proven to produce valid notes. The closed V12 minimum subsequently had
12 notes with zero reward, zero optimizer updates, and no trained reload. All notes
retained the same wrong directed edge; one of the two fixed problems recovered its
answer, but this did not establish a valid notebook or learning signal.

A separate, no-training paired context probe used four fixed seeds per problem.
On the second fixed problem, all four fresh-image-only readings had all18 facts
correct, while all four readings with the complete history repeated the reversed
edge (17/18 facts). This is evidence of context interference on that problem, not
proof of a Vision-module cause or general performance improvement. The first
problem still had formatting and false-edge errors; it was not rescued for scoring.

V13 therefore adds exactly one same-policy image reread before the full-history
notebook. It sees the original question/images and public schema, but no previous
solutions or verifier answers. Its unmodified output, even when wrong or truncated,
is quoted as UNVERIFIED context alongside every original attempt. It is not an
oracle, an extra solution attempt, a rewarded action, or routing evidence. There is
no candidate selection or reread retry. The seed is stage-specific, n=1 and the
output cap is1024; actual calls, tokens and elapsed time are recorded separately.
Both notebook-RL and answer-only control use this procedure, although their actual
costs can differ if recovery differs. Existing binary rewards, solution/notebook
budgets and independent routing remain unchanged. Approved actor/schema/seed,
original images, actual prompt/output tokens and notebook insertion are validated
again before updating. CPU contract tests are not actual V13 learning evidence.

The closed V13 minimum produced14 solutions and12 whole-history notes, plus the
unverified reread. All notes stopped naturally but none passed:11 had incorrect
facts/attempt assessments/evidence, and one was invalid JSON. There were zero
optimizer updates and no trained reload. A correct final answer is not sufficient.

The separate layout-only diagnostic reuses the closed actual history, including
the successful retry and the unmodified reread. It compares the original compact
JSON transcript against public metadata headers and individually numbered,
JSON-quoted source lines. Both use the original four first-note seeds, images,
2048-token output cap and unchanged checker. It does not retry, train, generate
replacement solutions or reveal reference facts/expected diagnoses. Episodes
without a successful retry are recorded as ineligible, not replaced. A successful
diagnostic note would not establish notebook learning or transfer.

That closed comparison produced eight notes in1474.200seconds: both layouts had
zero valid notes. CURRENT_JSON reproduced the original four first-DIAG outputs
(actual prompt/completion tokens, text and logprobs); all retained the reversed
edge and incorrect attempt assessments. Numbered lines produced two content
failures, one invalid range and one truncated JSON. This single eligible episode
does not support a layout-only repair. No weights were updated.

```bash
python -m fine_tuning.notebook_rl.fact_context_probe --notebook-layout \
  --preparation "$CURRENT_PREPARATION" --model-path "$NF4_BASE" \
  --cache-check "$CACHE_CHECK" --parent "$CLOSED_MINIMAL" \
  --output "$NEW_DIAGNOSTIC_OUTPUT"
```

The CLI supervises at most one hour, including cleanup. Current source-bound data
validation is required; historical model-cache readiness may be reused only when
its protocol and exact model/runtime source hashes still match. Output must be new.

The separate diagnostic, `--notebook-drafts`, separates per-attempt review from final
synthesis. Each independent draft sees only its original attempt, public outcome,
original question/images/schema and the same unverified reread. Each is sampled
once with a distinct frozen stage seed and384-token cap. No draft gets grader
feedback, ranking, repair or reference facts. The final notebook receives every
original attempt again, including the successful retry, plus every raw unverified
draft—even invalid or truncated ones. Drafts never replace the original history.
The final four original seeds,2048 output cap,16384 context and strict checker
remain unchanged; overflow blocks rather than dropping text. This is a separate
no-update diagnostic, not validated notebook learning or an extra training reward.

```bash
python -m fine_tuning.notebook_rl.fact_context_probe --notebook-drafts \
  --preparation "$CURRENT_PREPARATION" --model-path "$NF4_BASE" \
  --cache-check "$CACHE_CHECK" --parent "$CLOSED_MINIMAL" \
  --output "$NEW_DIAGNOSTIC_OUTPUT"
```
The original notebook prompt/reward/protocol remain unchanged by this diagnostic.

Its closed actual run took817.318seconds, seven calls/ten samples, zero updates.
All six drafts and four final notes stopped naturally. The final notes all retained
the reversed `e4` direction: three failed content checks and one used the wrong
attempt-field name. Supplying the successful retry and all historical solutions
did not produce a valid notebook in this trial.

### Post-update full-notebook check — diagnostic, not a training gate

After an actual minimal update and its closed fresh-process reload, the existing
probe can generate whole notebooks with the restored trained weights. This fills
a gap left by the short reload generation probes: restored tensors alone do not
show whether the model now reviews its previous solutions more accurately.

```bash
python -m fine_tuning.notebook_rl.fact_context_probe \
  --trained-notebook-reload "$CLOSED_TRAINED_RELOAD" \
  --preparation "$CURRENT_PREPARATION" --model-path "$NF4_BASE" \
  --cache-check "$CACHE_CHECK" --parent "$CLOSED_MINIMAL" \
  --output "$NEW_DIAGNOSTIC_OUTPUT"
```

This opt-in requires a verified real update/checkpoint/reload, not a base-only
cache check. It preserves the parent's protocol and historical policy-zero
solutions, final successful retry, unverified reread, original first-note prompt,
images, four seeds and2048-token cap. Only the restored weights change; new
generations are explicitly policy one. It does not relabel or regenerate old
solutions. Old and new notes are checked independently with the same strict
verifier. The result records recovery and regression, and episodes without a
successful retry remain excluded with their reasons. No output is used for training.

Current source-bound input validation is still required; the original parent and
its source binding stay intact. This one-hour diagnostic needs no valid-note
training gate, so a partial-reward update can be inspected without authorizing
fixed-ten training. It cannot certify a positive strict-note reward gradient,
healthy Vision, a unique module cause, unseen-question improvement or the isolated
benefit of notebook rewards. Those require the separate existing gates and
answer-only control. CPU fixtures test this connection, not actual trained quality.

### Explicit verified-atom objective — CPU-tested candidate, no actual learning signal yet

V13 remains the default. `build_protocol(verified_note_atoms=True)` selects
`local-nf4-verified-note-atoms-v14` / `verified-note-atoms-v1`. No old sampled
`reward`, `diag_reward`, strict report, selection or response is overwritten.
The separate training reward is `strict_reward + 0.25 * verified_atoms / denominator`,
where the fixed denominator is all actual attempts times the private required-slot
count. Missing or incorrect atoms do not shrink the denominator. Strict successes
remain above all partial-only notes; the coefficient is a design choice, not a
measured optimum or a guarantee of learning.

An atom requires an independently correct required final fact, a citation to the
actual original observation, and correct wrong-slot membership. That attempt must
also pass history, cause and conclusion-evidence checks. Final-answer verification,
complete typed note structure, canonical slot IDs and conflict/unknown-relation
guards remain required. Only submitted values are tested against the private
verifier. No reference values, replacement prose or expected diagnoses enter the
actor; scalar rewards are not target-text imitation. Free-form lesson semantics
and causal responsibility of internal modules are not certified.

The host binds these scores to the original actor/history/audit/note groups, then
independently recomputes them with the live verifier before optimization. Sampled
DIAG tokens must decode exactly to the scored text. The answer-only control uses
zero DIAG rewards even when objective receipts exist. Constant groups still yield
no relative reward signal; no candidate selection or forced variance is added.
Recorded task gradients combine strict and atomic terms, not separately measured
per-term or per-claim gradients. A good atom can still share its completion with
incorrect prose, so this objective alone cannot certify a correct notebook.

The fixed-ten gate still requires a **strictly valid whole note** with positive
effective advantage and isolated task-gradient contribution, plus actual saved
update/fresh reload. Partial-only progress cannot pass that gate. Both module
routes remain adapter-based; this is not full-parameter4B fine-tuning.

To explicitly select V14, add `--verified-note-atoms` to **both** `quantized_cache`
save/check commands and to `local_run prepare`/`minimal` in the workflow below.
Use new source-bound evidence/output directories and the same immutable NF4 base;
do not relabel V13 cache/preparation receipts. Reload, fixed-ten/control and
evaluation restore and check the full recorded V13 or V14 contract automatically.
No protocol switch is permitted within a saved run.

CPU authored tests cover scoring, tamper rejection, actual verifier-service calls,
token/text binding, reward control and the unchanged strict expansion gate. These
are not pretrained-model training evidence. A separate posthoc replay of the
actual twelve V13 notes gave three `[0,0,0,0]` groups (each denominator108), with
all original strict scores unchanged. Nine structurally eligible notes failed the
per-attempt prerequisite combination everywhere; two had invalid slot IDs and one
invalid JSON. Thus this candidate has **not** solved the actual no-signal barrier;
repeating the unchanged GPU run is not justified by these CPU tests.

### Explicit component objective — new local trial, not validated self-improvement

`--verified-note-components` selects `local-nf4-verified-note-components-v15`.
Use it on cache save/check and local prepare/minimal, in new evidence directories.
It is mutually exclusive with V14; default V13 and both historical rewards remain
unchanged. The training reward is strict reward plus0.25 times the equal-weight
mean of four fractions: verified required final facts, history with conclusion
evidence, cause with matching history, and exact error locations with evidence.
Denominators include every required fact or actual attempt. An unrelated wrong
cause no longer removes credit for a separately verified fact. Matching empty
error sets only certifies that empty set, not visual evidence or healthy Vision.
Cause matches certify the registered taxonomy, not internal module causality.

The typed-envelope, independently verified final answer, canonical-ID and
false/unknown-relation guards remain. Every reward is rechecked before optimization;
gold never enters the actor and no reference text is imitated. This is a new
training objective, not a relaxed whole-note pass criterion. Partial-only notes
still cannot pass the fixed-ten expansion gate. Full completion gradients can
reinforce incorrect prose alongside correct fields; component-specific learning
and generalization require actual experiments, not these CPU tests.

### Explicit fact-token credit — V16 candidate, not whole-notebook success

`--verified-note-fact-credit` selects `local-nf4-verified-note-fact-credit-v16`
on `quantized_cache save/check` and `local_run prepare/minimal`. The three optional
objectives are mutually exclusive; default V13 and old results remain unchanged.
Use fresh source-bound evidence directories with the same immutable NF4 base.
Reload/control/evaluation restore the recorded contract; never relabel an old run.

The current objective version is `verified-note-fact-token-credit-v2`. Only this
objective normalizes submitted `corrected_facts` IDs with the existing `slot_id`
rules before partial scoring. The atom/component objectives remain canonical-only.
Canonical IDs map to the exact original JSON value literal offsets; values and
response/token bytes are not rewritten. Two submitted keys mapping to one ID are
rejected even if their values agree. Unknown/false/conflicting/unverified relations,
including unsupported null claims, retain the existing checks. This does not make
an incomplete or inaccurate whole notebook correct.

The protocol family name remains V16, but the objective version, mapping contract
and full protocol hash change. Current-source restore rejects v1 contracts even
after rehashing; historical runs and checkpoints require their frozen source.
New v2 preparation/minimum/reload evidence is required before a new execution.
CPU mapping tests do not establish model-generated notebook quality or improvement.

The full solution history, successful retry, original images, model-written
whole notebook, strict verification and independently determined module routing
are unchanged. The change is where **partial** reward enters the policy gradient:

- Whole-note strict rewards are centered over the original full sample group and
  applied to the whole completion, with their own clipped surrogate.
- For each submitted required fact, the verifier scores its actual value0/1.
  Missing/invalid claims count as0 in the full group mean. Its centered score is
  applied only to tokens wholly inside that original JSON value literal, scaled
  by `0.0625 / required_slot_count`, with a separately clipped surrogate.
- Optional registered facts receive no required-fact partial credit. Duplicate
  or unknown/conflicting claims keep the existing fail-closed checks. No correct
  reference value or replacement answer is generated or supplied to the model.
- The0.0625 coefficient preserves V15's fact fraction (`0.25 × 0.25`), without
  transferring the other three partial weights. Those attempt-level partial
  rewards are0 here; all their whole-note correctness requirements remain.
- MIS and KL still cover the original full completion; group weights, original
  sample denominators and generation caps remain unchanged. Separate clipping
  matters when strict and partial advantages have opposing signs.

`effective_reward = strict_reward + 0.0625 * verified_required_fact_fraction`
is an audit scalar, **not** an advantage broadcast over every token. The recorded
`fact_token_credit` includes per-slot reward/mean/advantage, literal offsets,
selected token indices and source hashes. Stable prefix decoding must reproduce
the exact sampled text; unstable byte-prefix decoding abstains from all partial
credit for that sample, without removing it from group means, strict loss or KL.
Full-decode mismatch stops the update. Boundary-crossing and zero-width tokens
get no partial fact credit. Tests cover escaped/repeated JSON values and malformed
receipts as well as the optimizer connection and answer-only control.

If every candidate says the same wrong `H→C`, its fact advantage is0: this change
does not invent a correct `C→H` action or teach a missing answer. Shared parameters
and KL can still indirectly change uncredited text. Scope-limited partial credit
is therefore a team-designed shaping objective, not proof of accurate causes,
valid free-form lessons or generalization. A strict-valid **whole** note with
positive strict advantage and isolated gradient contribution remains necessary
for default strict fixed-ten admission; partial-only updates and same-history
probes do not pass that policy. The explicit exploratory policy below is separate.
The actual V16 minimum completed one update and fresh-process reload. Five notes
contributed verified partial fact-token/isolated-gradient signals, but strict notes
remained0/12. A subsequent same-history diagnostic remained0/4→0/4. Actual learning
and reload are therefore verified; accurate whole notes and self-improvement are not.

### Explicit exploratory fixed-ten admission

The historical/default admission is `strict-notebook-learning-v1`: a fully
verified note, positive strict advantage and isolated note-gradient evidence are
required. This remains the meaning of strict notebook success. It is not the
only way to test whether learning can improve initially imperfect notes.

For V16 only, `local_train train --admission verified-fact-credit-exploration-v1`
selects a separate, bounded **exploratory** execution contract. It requires the
actual minimal update, changed weights, saved checkpoint and fresh-process
reload. It also regrades the original complete solution/note groups, reconstructs
fact-value token credit before MIS, and requires a retained sample with positive
correct-fact credit, zero strict advantage and an isolated nonzero note-task
gradient. Answer-only, KL-only, forged or negative-only evidence cannot admit it.
The gradient combines the sample's fact terms; it is not measured separately for
each individual fact. No failed note is relabeled correct.

The admission policy and deterministic parent proof are hashed into the execution
contract. Both notebook and answer-only arms must use the same admission, parent,
fresh initialization, fixed ten/order and existing five-update/time budget.
Reload and native evaluation recheck the recorded policy and parent proof.
Historical contracts without this field retain strict admission. The model,
loss, rewards, prompts, source separation and strict note grader are unchanged.

This option removes a pre-learning behavioral prerequisite, not a scientific
success criterion. Full-note correctness, both actual learning routes, disjoint
first-attempt performance and the matched control still require separate actual
evidence. It does not authorize large training or cloud execution. The fixed-ten
exploratory trial started on2026-10-22 after the actual parent gate passed; completion
and improvement are not established. See the private runtime records and project status.

### Fixed-ten wall-time budget

`local_train train --max-seconds N` fixes a training-only wall-time limit before
launch (integer 1–86400; default 14400). The versioned `training_budget` is hashed
into the execution contract and checked again by reload and evaluation. The
answer-only control must use the same limit as its notebook arm. Historical
contracts without this field keep the original 14400-second limit. This does not
change the minimum trial, evaluation deadline, generation settings, fixed ten
items or maximum five updates, and does not extend an already running process.

Local preparation and execution revalidation explicitly allow up to 600 seconds
for the trusted CPU data preflight. This operational limit is separate from the
training budget; completed input validation is still required. Only the explicit
`limited_smoke` stage accepts this limit. The gate dispatcher's default remains
300 seconds, and other validation stages remain capped at 300 seconds.

The earlier two-item minimum took 9630 seconds; multiplying by five gives roughly
13.4 hours, not a measured fixed-ten forecast. For a new local ten-item experiment,
an explicit 72000-second (20-hour) training cap leaves operational headroom. Use
the same training value for both arms and an external watchdog of at least
72900 seconds (72000 training + 600 CPU preflight + 300 shutdown), for example
`timeout --signal=TERM --kill-after=30s 72900s ... local_train train ... --max-seconds 72000`.
Model setup or other work may need additional headroom. This future-launch
example does not extend existing watchdogs, including the live 72300-second run,
or add training budget. Internal deadline checks occur between operations, so the
external watchdog is necessary to stop a stalled or long optimizer operation. Completion
and learning are still unverified, and a killed in-flight update may be lost.

### Preserving completed updates on interruption

`local_train` writes a separate `checkpoint-pair-NN/` and matching JSON receipt
after each actual update passes parameter-change, frozen-base and synchronization
checks. It reuses the complete checkpoint format: adapters, all trainable Mergers,
optimizer, scheduler, RNG and the consumed/discarded problem cursor. Earlier
checkpoints and cursors are never overwritten. `result.json` points to the last
committed checkpoint if a later pair stops or fails; each receipt also stands on
its own if the process is forcibly killed before that final result is written.

These are partial preservation artifacts, not a resume command or fixed-ten
completion. Existing full-ten/reload/evaluation gates still reject partial runs.
A no-signal pair or failed synchronization creates no new committed checkpoint.
An interrupted checkpoint write may leave an incomplete directory, which the
existing hash/file validator rejects. Earlier committed snapshots remain usable
for inspection. Retaining snapshots adds disk usage; there is no automatic deletion.

CPU tests exercise the real tiny-model optimizer and checkpoint payload with
scripted rollouts. They do not establish actual Qwen notebook learning or GPU
resumption. The unchanged per-run time budget is not a guarantee that all ten
problems and five possible updates fit; budget stops remain incomplete.

V11 closes a measured native-graph verification gap: an explicitly asserted
`edge[u,v]` absent from a complete native BFS/shortest graph used to be ignored
because only registered positive edges were counted. `false_relations` now records
those claims after validating the native world, endpoint labels, positive numeric
assertion and exact reference-slot/type parity. `unverified_relations` keeps unknown
endpoints, unsupported values and unavailable world proofs separate. Required-slot
coverage and existing metrics keep their original denominators; they do not count
extra false edges as missing or correct registered slots.

The subsequent `native-graph-relations-v2` verifier also checks explicitly asserted
`child[parent,L/R]=node` links absent from a complete native tree. The same native
world and exact fact/type parity proof is required; unknown nodes, malformed values,
and `null`/`none` absence claims remain unverified. Extra links do not change required
slot denominators. Actual saved TEN2 attempts 1/4/6 exposed this omission: a partial
CPU replay changes conditional admission from UNRESOLVED to VISION_ON, but all four
checked notes still fail strict validation. This replay is not a closed-episode,
GPU-learning or internal Vision-causality result. Historical frozen sources, rewards
and receipts remain unchanged; new executions bind the revised verifier source hash.

Exploratory parent admission can recheck a closed relation-v1 minimum with the
relation-v2 verifier only through an explicit, narrow compatibility proof. Every
original solution, including trees without notes, and every submitted notebook
fact projection must retain all result fields except that exact version transition.
The original audits, strict grades, full-group objective receipts, token credit,
MIS decisions and isolated-gradient evidence must still match. `cause` receives
the freshly issued verifier result; stored responses, results, versions and hashes
are never rewritten. Unknown/mixed versions or any semantic change are rejected.
The deterministic proof is carried into train/control, reload and evaluation.
Ordinary live verification remains exact. This reuses mechanical learning evidence,
not successful whole notebooks, and does not certify new GPU execution or improvement.

Full-history routing, diagnosis targets, cited observation evidence and notebook
checks consume verified false edges. Required correct observations plus any false
or unverified edge cannot certify Vision-frozen admission. Answer CORRECT and
TRUNCATED retain their cause precedence, while their incorrect observations still
participate in routing and must be acknowledged in the note. The strict rule is
`all-attempt-observations-and-false-relations-admission-v3`; the fresh verifier
service also binds the native graph validator source. Note feedback can name only
submitted incorrect facts, never replacement edges, private values or missing facts.
The actor's prose lesson remains unverified. Legacy seven-field SlotReport inputs
can be read, but reserialization adds two arrays and changes serialized hashes;
use frozen historical source to replay old receipts, never relabel them as V11.
The first V11 data preparation correctly stopped because the vault manifest binds
the entire schema source, including runtime-only SlotReport changes. The reader now
supports only the reviewed exact historical vault/schema/guards hash triple, and
only with the pinned compatible new schema and unchanged guards. ActorItem,
VerifierItem and Lineage semantics are unchanged. Protocol, payload, identity,
count and forbidden-lineage checks still apply; unknown or future schema hashes
are rejected. Original vaults and the failed preparation are not rewritten. This
compatibility does not make historical runtime result hashes equivalent to V11.
The scanner receipt independently binds six source files. Its reviewed historical
six-file set is accepted only against the exact compatible current set and unchanged
guards; the four detector/config files must remain byte-identical. Current file
hashes are recorded separately, never substituted into the old receipt. Unknown,
partial or mixed transitions remain blocked. This reuses the original exclusion
scan; it is not a new scan or a relaxation of duplicate thresholds.
This is implementation and CPU-test evidence only until actual V11 generation,
notebook-reward updates and fresh reload succeed. V10's live source is unchanged.

V9 removes the initial solve-format instruction from the new notebook request:
its prefix is now only the original question/options, followed by the notebook
contract, public observation schema and complete transcript. V8 also prepended
`Observed facts / Solution / Final answer` output instructions, competing with
the JSON-only notebook contract. The wrapper policy is explicitly bound as
`question-choices-images-and-full-transcript-v1`; old or missing policy is rejected
before generation. Grading, rewards, routing, original solve/retry prompts, images,
seeds and token budgets are unchanged. This is a prompt-contract correction, not
proof that it fixes the observed empty facts or incorrect check actions. The live
frozen V8 run was not patched. V9 subsequently ended without notebook learning,
as recorded below; its source and failed responses remain unchanged.

The first actual v1 compact run produced four truncated 2048-token notes with
zero reward: they copied input-only fields and source line strings instead of
completing the required diagnosis. v2 separates input `source_lines` from output
integer `lines`, brackets the full read-only transcript, and repeats the output
contract after it. The grader, rewards, full history and token caps are unchanged.
The prompt-only correction passed 70 focused CPU tests, scoped Ruff/mypy and an
independent static review. The frozen v2 input/context and actual NF4 readiness
checks passed. Its actual four notes earned zero reward: three were truncated and
one complete JSON chose success_attempt3 instead of the designated successful retry6.
Some enumerated IDs beyond400 although the first attempt had38 lines. These are
preserved failures, not successful notebook learning. The frozen v2 minimum completed
one answer-reward update and fresh-process reload, but its four notes had zero reward
and zero isolated reward-gradient contribution; only the Vision-enabled route updated.
The v3 contract prints line IDs/counts and requests inclusive, disjoint
`line_ranges`, expanding them to the same original evidence lines before applying
the existing semantic checks. It rejects invalid/out-of-bounds ranges and repeats
the designated retry ID after the history. Binary reward, facts/cause/evidence
requirements, routing and token caps are unchanged; no failed old output is repaired.
The frozen v3 CPU/readiness checks passed. Its actual four notes finished naturally
but used flat ranges rather than pairs and retained a wrong arrow direction, so all
earned zero reward. Fixing formatting alone would not certify their semantic claims.
The frozen v1 run completed one answer-reward update and fresh-process reload,
but all four notes had zero reward and zero gradient contribution. It is preserved.

V4 adds bounded self-correction of the notebook itself, not a repaired target or SFT.
After an all-zero four-note group, the next round always continues sample0 using
its exact prompt/completion tokens, the original images, and allowlisted failure
codes/check booleans. The host never supplies private expected facts, causes, answers,
replacement claims or evidence text. Partial checks remain explicitly incomplete.
There are at most three notebook rounds (initial plus two repairs). The first repair
can fix an early format rejection; the second can then act on semantic check results.
Any valid note stops retries, even if all four rewards are1 and provide no relative
learning signal. Each round is a separate same-state group with distinct deterministic
seed branches; advantages are not pooled across rounds. Only new completion tokens
receive loss. Previous failed groups and all solution attempts remain in the ledger.
The actual tokenizer rechecks feedback-to-token binding before generation finalization
and optimizer entry; tokenizer-free metadata checks alone are not that attestation.

This is additional verifier supervision, although it contains no target values and
uses RL rather than target-text imitation. Extra DIAG groups increase compute and
alter their share of the existing loss/KL normalization. The answer-only control
uses the same generation/retry path and disables notebook rewards only afterward;
later trajectories may differ after updates. Record that difference, not a claim of
identical realized exposure. V4 passed71 focused CPU tests including folder-only
and optimizer integration checks, plus an independent static review. The actual
frozen V4 produced twelve notes, all reward0. Later repairs fixed range formatting
but retained a wrong directed-image fact and misclassified earlier attempts; the
last four notes repeated the preceding selected note exactly despite check feedback.
This is not valid notebook learning; historical outputs remain unchanged.

V6 adds `submitted-fact-reinspection-slots-v1` feedback. After semantic grading,
the next notebook round may receive sorted IDs of its own submitted corrected facts
that the independent verifier found wrong or conflicting. No reference values,
expected causes, next-check labels, replacement prose or answer keys are supplied.
Hints concern the submitted corrected facts, not the historical attempts; the model
must reinspect the original images and assess every attempt separately. The hint
list is incomplete: missing facts and wrong optional nonconflicting facts are not
enumerated, and an empty list does not certify correctness. Feedback is bound to
the actual DIAG submitted-facts verification result and original note text.
Generic cause/coverage precedence is clarified, but binary scoring is unchanged.

This is extra verifier supervision, not unaided error discovery. For binary facts
such as arrow direction, rejecting a submitted value may imply its alternative even
without printing the correct value. It is still not target-text imitation/SFT.
The minimal controller now preserves raw attempts and stops before optimization
with `NO_NOTEBOOK_LEARNING_SIGNAL` when admitted trees contain no usable notebook
reward variance, including all-invalid or all-valid constant groups. Repeating an
answer-only update cannot verify notebook learning. The fixed-ten answer-only
control remains separate and unchanged. The actual frozen V6 minimum ended with
`NO_NOTEBOOK_LEARNING_SIGNAL`: twelve naturally terminated notes, seven range-format
failures and five semantic failures, zero optimizer updates. The last four notes
retained the prior wrong direction/diagnoses. Closed whole-tree CPU replay reproduced
all26 responses; this is verification of failure, not successful notebook learning.

Each generated note must assess every attempt, cite supporting source lines,
report extracted answers, wrong observation slots and a next-check action, and
give the final answer and corrected registered facts. Types and numeric finiteness
are checked before normalization. Cited observations must come from the original
observation block, not an identical assignment elsewhere. Nonvisual claims require
complete required observations and the actual answer; missing/ambiguous-answer
claims require whole-response evidence. Free-form lessons and internal module
causality remain uncertified, even when structured checks pass.

After the notebook stage, independent verification of ALL attempts finalizes the
route. Any observed wrong required slot enables Vision adapters, including an
error in a truncated or coincidentally correct answer. V5 admits Vision-frozen
learning only when every actual attempt has nonempty, complete, correct and
unconflicted required observations. Otherwise admission is `UNRESOLVED`, not a
healthy-Vision finding: preserve the raw tree, report its ID separately from
constant-reward trees, and do not train on it or replace it with another problem.
Admission is recomputed from bound verifier results; altered audit booleans alone
cannot enable training. This strict contract permits `MOD-RV` only and rejects
forced/random routes and the Vision-disable override. Legacy protocols retain
their original policies. The binary serialized route remains for compatibility;
inspect admission before treating an OFF record as a learning decision.
The solution, audit and generated note groups are hash-bound before updating.
The frozen NF4 base is not full-parameter fine-tuned. The two routes remain Vision
adapter+four Mergers+LLM adapter versus Vision frozen+four Mergers+LLM adapter.

The original fixed ten contain six plain and four observation-eliciting actors.
In the frozen V5/V6 code the verifier does not inspect observation slots at all for
plain actors, even when their responses contain facts. Those six cannot establish
either strict learning route. V7 therefore adds the explicit
`all-required-observations-v1` elicitation policy. Preparation preserves original
`actors.json`, vault, IDs, order, questions, images and answers, and writes a separately
hash-bound `execution_actors.json` with only `mode="obs"` changed. Every original
actor must still pass the existing input/vault membership checks. The execution
loader and evaluation synchronization probes reject any other projection change.
Minimal capture, fixed-ten/control training and both reload paths use those same
projected actors. External evaluation actors are not automatically changed.
This changes the prompt/input protocol, not the verifier, binary reward, strict
admission or source data. Missing observations remain unknown; requesting them
does not guarantee that the model supplies correct facts or valid notebooks.
Fresh V7 input/context and actual NF4 readiness checks are required; V6 evidence
is not reused as V7 execution success. V7 actual learning remains unverified.
This admission correction does not validate generated notebooks: all twelve
closed V4 notes earned zero. Five failed range syntax; seven reached semantic
checks and failed. Format-only diagnostic corrections still failed because the
notes retained the wrong direction and misreported prior attempts. Such CPU
counterfactuals are tests, never model-generated success or training targets.
The frozen V4/V6 sources and their records are unchanged; current actual-model
notebook learning, both-route execution and transfer evaluation remain unverified.

V8 adds a public observation-difference index to the notebook input. It lists all
occurrences of differing model-written values for a slot with their actual attempt
and source-line IDs, derived only from accepted observation blocks. It never queries
the verifier, supplies a reference value, labels a value correct, or selects a majority.
Formatting differences may also appear; absent observations are not certified by an
empty index. The full original history remains lossless, and role delimiters remain
escaped in both views. Output instructions also distinguish status TRUNCATED from
cause FINALIZATION and give a nested-range syntax example. This is explicit input
organization, not an independently discovered model capability or teacher target.
The binary validity reward, fact/cause/evidence requirements, routing, seeds and
token caps are unchanged. V7 CPU evidence is preserved; V8 requires fresh input/context
and actual-model validation. No V8 notebook-learning success is claimed.

V9 changes only the notebook task wrapper: use the original question/choices and
the full-history notebook instruction, without the competing turn-1 solve-format
instruction. All solution text (including the successful retry), images, note
checks, binary rewards and routing remain unchanged. The closed V8 trial had12
solutions and24 zero-reward notes, with zero optimizer updates; its source and
failed responses are preserved. V9 standalone base save/reload passed, but its
minimum ended after3980.607seconds with12 solutions and24 zero-reward notebooks.
One item had insufficient observation evidence (UNRESOLVED), the other had verified
visual errors (VISION_ON). Neither led to an update: notebook reward variance was0,
optimizer updates0, trained reload and improvement unverified. Cache reload is not
trained reload. Missing corrected facts and retained wrong arrow direction are real
content failures; invalid JSON and redundant next-action checks are separate failures.

V10 retains binary validity rewards and all answer/fact/cause/evidence/history checks,
but removes generated `next_check` from the output schema. The host records
`verified_next_check` only after all five attempt checks pass, labels it host-derived,
and never sends it as retry feedback. This action is not a model achievement.
An explicit placeholder-only JSON shape quotes enum strings and demonstrates nested
line ranges. Public source-specific observation instructions cover directed graphs,
DFAs and binary trees without supplying actual labels, values or answers. Fixed IDs,
images, full history, routing, sampling, budgets and optimization are unchanged.
This is a separately versioned output/input contract, not a repaired old response
or permission to accept false facts. Actual V10 learning remains to be verified.

This explicit new protocol reserves context16384 and note2048; solution caps1024,
solution groups4/2/2, four notes per round, seed3407 and at most three solution
attempts stay fixed. V4 added bounded notebook repair rounds; V5 adds strict route
admission, V6 adds submitted-fact location feedback and the minimal notebook-signal
gate, and V7 elicits observations in all fixed training actors without changing
those generation or reward settings.
Exhaustion, missing recovery and context/budget stops are preserved, never repaired
by fabricated success or silent truncation. Input validation, fresh actual NF4
readiness, minimum update/reload and matched evaluations must be rerun with this
protocol. **Valid full-history notebook learning and improvement are not yet verified.**
CPU fixtures below prove contracts only. Historical results cannot certify this extension.

The existing ten external circuit items do not satisfy an unqualified structural-
disjointness claim. Source/image identity differs, but all ten share a simple
undirected four-node/five-edge topology with one training DFA after removing labels,
directions and edge multiplicity (10/100 train/external pairs). Multiplicity-preserving
comparison found0/100. This is a scope limitation, not proof of identical questions
or benchmark contamination. Preserve the cohort and its old scores; the goal's
strict structural transfer evaluation remains pending a separately frozen cohort.

`local_ladder` prepares a separate CPU-only candidate cohort with the existing
native circuit generator and two exact solvers. It binds the original fixed-ten
selection to the previously validated train16 vault, then checks source and input
hashes again after writing. Its ten numeric/query instances span **three** ladder
topology families, with five or six nodes versus four, seven or eight in the selected
training graphs. This proves complete-graph nonisomorphism, not absence of shared
subgraphs, concepts or pretraining exposure. It is independent program-generated
evaluation data, not an externally collected real-world dataset.

```bash
CUDA_VISIBLE_DEVICES='' "$NF4_PY" -m fine_tuning.notebook_rl.local_ladder \
  --preparation "$LOCAL_DATA/preparation.json" --output "$LADDER_COHORT"
```

Use a new private output path outside the input roots; previous cohorts and scores
are never replaced. `load_ladder_actors()` opens only public actor records and
images, not verifier/gold or lineage payloads. Preparation returns
`CPU_COHORT_PREPARED_PENDING_GATES`, with training/evaluation permission false.
It does not run a new benchmark exclusion scan, rights/render review, model,
training or evaluation. The legacy four-node A–D `local_eval` command below cannot
evaluate this native unit/fraction cohort; a bound native evaluation run is still
required before any transfer score can be reported.

`local_ladder_score` supplies the CPU scoring connection for **already saved**
native numeric/unit responses, using a fresh `VerifierService` process rather than
the old A–D comparator. It requires the pinned preparation SHA and exactly ten
ordered objects containing only `problem_id`, `text`, and `finish_reason` (`stop`
or `length`) in a JSON array. Missing/duplicate responses are rejected; invalid
answers retain the ten-item denominator, while unverifiable gold contracts make
the aggregate incomplete. It never searches intermediate reasoning for gold.

```bash
CUDA_VISIBLE_DEVICES='' "$NF4_PY" -m fine_tuning.notebook_rl.local_ladder_score \
  --cohort "$LADDER_COHORT" --cohort-preparation-sha256 "$LADDER_PREPARATION_SHA" \
  --responses "$NATIVE_RESPONSES_JSON" --output "$NEW_PRIVATE_SCORE_JSON"
```

The score receipt binds response bytes, cohort and verifier code, **not** the
provenance of model generation or identical F0/FT inputs. This command grants no
GPU execution/training authority and does not replace the native inference
runner or its data/protocol gates. Its authored CPU fixtures are not model scores.
Keep a live training/reload package frozen: adding this scorer changes the current
package's code hash, not the historical training binding.

`local_ladder_generation.generate_ladder()` is the matching first-attempt core,
not a standalone execution command. It accepts an already verified engine and
pins the cohort preparation SHA, local protocol and prompt templates. It preflights
all ten native inputs, then records one 1,024-token-budget sample per item under
the same policy-independent seed before and after learning. Full text, token IDs,
log probabilities, image grids/pixel hashes, token counts and elapsed time are
retained; a partial failure leaves per-call files but no complete result.
`validate_paired_records()` requires F0 version zero and an updated FT version,
matching actual inputs, image tensors, seeds and source/protocol bindings. It
does not certify that those weights were loaded: provenance stays false until a
gated caller checks model/checkpoint identity, source separation and the deadline.
The native runtime is now `local_eval evaluate-ladder`; its actual F0/FT experiment
remains pending. CPU fake-engine tests and processor-only checks are not evidence
of model learning or accuracy.

Before that command, the trusted CPU API `local_ladder_gate.build_gate(cohort,
training_preparation, review, output, protocol)` independently reconstructs the ten
native circuits, checks both exact solvers and training separation, and binds an
all-image AI review. `protocol` is `local_ladder_generation.protocol(local_prompts(
build_protocol()), build_protocol()["sha256"])`. The review must identify every
original image and whether its reviewer saw reference data; AI review is never
human certification. The original preparation's pending flags are not overwritten.
Use a new private gate file in a dedicated sibling directory; source/preparation
changes require a new receipt, not editing an old one. This is bounded diagnostic
data permission, never training/GPU authority or a new benchmark exclusion scan.
The runtime's `validate_gate()` opens only public inputs/review, not private gold.

After the minimum and requested fixed-ten arm have both completed their
fresh-process reloads under the same recorded strict or explicit exploratory
admission policy, run each evaluation in a new process:

```bash
timeout --signal=TERM --kill-after=30s 14400s "$NF4_PY" \
  -m fine_tuning.notebook_rl.local_eval evaluate-ladder \
  --capture "$TEN_CAPTURE" --reload-result "$TEN_RELOAD/result.json" \
  --minimal-capture "$MINIMAL_CAPTURE" --minimal-reload "$MINIMAL_RELOAD" \
  --cohort "$LADDER_COHORT" --data-gate "$LADDER_GATE" \
  --condition F0 --arm notebook --output "$NEW_F0_OUTPUT"
```

For notebook FT use the same capture/reload with `--condition FT`; for the
answer-only FT arm use its own capture/reload and `--arm answer-only`. Its stored
matched-notebook receipt and initialization must agree exactly. Historical training
code hashes are checked separately from current evaluation code; new training
still requires current-code identity. The external timeout is required: this
caller does not implement an independent process watchdog. Keep the local
14400-second per-run limit and do not run alongside another GPU job.
The actor writes complete `generation/` records and `responses.json`, never opens
native gold, and never updates weights. After it exits, run `local_ladder_score`
above. Bind scored bytes to those generation records and use
`validate_paired_records()` for identical F0/FT inputs before reporting a comparison;
neither a CPU score alone nor a generation summary proves the complete experiment.

The explicit `--with-recovery` option selects
`local-native-ladder-recovery-notebook-v1` for this same native cohort. Build a
**new** `local_ladder_gate` receipt with
`local_ladder_recovery.protocol(local_prompts(build_protocol()), build_protocol()["sha256"])`;
the first-attempt receipt cannot authorize recovery. The gate accepts only the
exact canonical first-attempt or recovery contract and binds recovery/scoring
sources. Keep F0, notebook FT and answer-only FT on this same new receipt and
protocol. The default command above retains first-attempt behavior.

```bash
timeout --signal=TERM --kill-after=30s 14400s "$NF4_PY" \
  -m fine_tuning.notebook_rl.local_eval evaluate-ladder \
  --capture "$TEN_CAPTURE" --reload-result "$TEN_RELOAD/result.json" \
  --minimal-capture "$MINIMAL_CAPTURE" --minimal-reload "$MINIMAL_RELOAD" \
  --cohort "$LADDER_COHORT" --data-gate "$NEW_RECOVERY_GATE" \
  --condition F0 --arm notebook --with-recovery --output "$NEW_F0_RECOVERY_OUTPUT"
```

Recovery keeps the original first-attempt seeds and records one sample per call,
at most three attempts, 1,024 solve tokens, 2,048 notebook tokens and a 16,384-token
context. The original 14,400-second evaluation limit includes setup, model loading,
generation and scoring; the external watchdog remains required. Inputs and output
reservations are not reduced. A fresh trusted verifier grades native numeric/unit
answers, gives bounded retry feedback and checks a notebook only after a genuine
recovery. The inference actor guards still deny private verifier/lineage files.
Evaluation performs zero learning updates. Full recovery records are saved under
`recovery/`, the caller embeds them in `result.json`, and `responses.json` retains
the separate original ten first responses. Failed or incomplete runs retain their
failure records and cannot report completion or recovery accuracy.

After the actor exits, replay the saved core record with the GPU hidden:

```bash
CUDA_VISIBLE_DEVICES='' "$NF4_PY" -B -m fine_tuning.notebook_rl.local_eval replay-ladder \
  --cohort "$LADDER_COHORT" --cohort-preparation-sha256 "$LADDER_PREPARATION_SHA256" \
  --record "$NEW_F0_RECOVERY_OUTPUT/recovery/result.json" --output "$NEW_CPU_REPLAY"
```

`local_eval compare` accepts paired completed recovery `result.json` files only
when fresh CPU regrading of both original records passes and the recorded
inputs/seeds/protocol/source/metrics and the caller's initialization, current-code,
unchanged-parameter/base and runtime gates agree. Comparison invokes those fresh
verifiers itself, so the bound cohort and its private verifier inputs must remain
accessible; stored hashes alone do not suffice.
CPU replay regrades saved evidence; it does not certify tokenizer/model execution,
lesson semantics, module responsibility, learning benefit or general improvement.
This implementation has no new native GPU result or score; the actual three-policy
recovery experiment and its fresh CPU replays remain pending.

Default strict fixed-ten training (both notebook and matched answer-only arms)
requires a positive-reward validated note from the minimal run to have an isolated
reward-gradient contribution. The gate links the saved checked note, DIAG sample,
effective group rewards, centered advantage and actual before/after gradient hashes;
the cited TREE files join the parent evidence hashes. Summary flags alone, zero
reward variance, negative-only notebook updates or KL-only changes do not pass.
This is a consistency check of saved evidence, not a new semantic regrade or an
improvement certificate. Only the legacy A-D external diagnosis accepts a valid
answer-update/reload without this extra learning gate. The native ladder command
revalidates the trained arm's recorded admission: strict evidence by default, or
the separate V16 exploratory evidence described above. Exploratory admission does
not certify a correct whole notebook or improvement.

### Explicit 24K whole-history capacity experiment

`--verified-note-fact-credit --context-limit 24576` selects the separate
`local-nf4-verified-note-fact-credit-24k-v19` contract. The default remains16K;
24K is allowed only with the verified-fact-credit objective. This changes the
context budget, not the model, images, sampling, solution/note output caps,
reward, routing, fixed item selection or training time limit. All original
solutions, successful retries and selected notebook-repair ancestry remain intact.
There is no automatic context expansion, truncation or memory fallback.

Use the flags consistently on `local_run prepare`, `quantized_cache save/check`
and `local_run minimal`, always with new evidence/output paths. For example:

```bash
CUDA_VISIBLE_DEVICES='' "$NF4_PY" -m fine_tuning.notebook_rl.local_run prepare \
  --previous-request "$PREVIOUS_REQUEST" --selection "$SELECTION" --output "$LOCAL_DATA_24K" \
  --verified-note-fact-credit --context-limit 24576
```

The previous request must still bind the current frozen source; an old approval
does not become valid by changing this flag. Cache readiness needs fresh save and
separate-process check receipts for this same protocol, even when the immutable
NF4 cache bytes are reused. Minimal/fixed-ten reload, both training arms and native
ladder evaluation derive the context from their bound training protocol. Native
gate construction also requires that context explicitly:

```python
p = build_protocol(verified_note_fact_credit=True, context_limit=24576)
native = local_ladder_recovery.protocol(local_prompts(p), p["sha256"],
                                       context_limit=p["context_limit"])
```

First-attempt native evaluation uses the same keyword on
`local_ladder_generation.protocol`. F0, notebook FT and answer-only FT must share
the same complete evaluation contract; mixed16K/24K records are not a paired
comparison. The historical A-D circuit evaluator remains its separate16K protocol.

Graph-trace training protocols such as bounded synthesis v27 require an explicit
circuit evaluation prompt policy. Construct the new trusted CPU data gate with
`local_ladder_generation.circuit_evaluation_prompts()` in place of `local_prompts(p)`
in the native protocol call above, then pass
`--evaluation-prompt-policy generic-circuit-observations-v1` to `evaluate-ladder`.
This reuses the public generic circuit component/connection/polarity instructions;
the training protocol, cache readiness, reload and synchronization probes still
use the original training prompts. The evaluation protocol and saved generation
records bind the policy and its complete template hash. Use the same policy for
F0, notebook FT and answer-only FT; old gates and mixed-policy comparisons fail.
Omitting the flag retains the historical evaluation behavior.

With recovery, `note_verified_count` remains the existing
`episode_notebook.grade_note` metric, identified by its reward version in the new
evaluation protocol. It does **not** verify the v27 reasoning trace or bounded
synthesis note (`bounded_synthesis_verified=false`). This prompt split supplies
no new grader, gold payload, learning approval or GPU-success evidence.

The shared audit and grading APIs accept explicit `evaluation_only=True` for
complete, ordered n1 evaluation histories. They freshly verify every outcome and
retry ancestry while retaining `usable=False` and `training_admission=False`;
the training credit path rejects these evaluation audits. Native recovery uses
this shared path for generation and replay, with the same legacy note scores.
This fixes the audit mismatch; it does not implement whole-reference circuit
evaluation. Circuit trace semantics and the corresponding generation/replay
protocol remain unsupported, and `NATIVE_CONTRACT` still blocks strict synthesis.

Future minimum runs can explicitly select
`--execution-policy verified-update-exploration-v1` with the current
`--verified-note-synthesis-credit --context-limit 24576` profile. This versioned
execution contract is bound into the launch and checkpoint alongside the unchanged
protocol and input preparation. It permits a minimum update from retained answer
or partial-credit policy coefficients even when no notebook relationship learns.
Routing and eligibility still apply; after MIS, absent retained policy coefficients
close as `NO_RETAINED_LEARNING_SIGNAL` before backward or optimization. The default
minimum keeps its existing notebook-specific requirements and historical binding.

The same explicit `--admission verified-update-exploration-v1` on fixed-ten
training requires that minimum's real update, fresh reload, matching source/data/
execution binding and exact single-/multiple-image input identities. This is
mechanical pipeline exploration, with no notebook-learning or self-improvement
certification. Fixed-ten initialization, cohort, budget and matched answer-only
control remain unchanged; both arms require retained policy coefficients. Existing
cache evidence can be reused only when its source/protocol checks still match;
the new source still needs its own data preparation. Earlier failures and frozen
runs are not relabelled or modified.

The minimum's source identity comes from its descriptor-verified input validation
and remains bound to its original capture and checkpoint. A later training stage
validates and records its own current source and preparation separately. A passed
historical minimum does not certify a later code change or notebook-learning effect.

The verified-unit experiment, introduced in v29, selects `--verified-note-synthesis-credit
--context-limit 24576` on prepare, cache save/check and minimal capture. It keeps
8,192 note tokens and 1,024 solution tokens. Native observation/trace comparison
and unknown-region checks remain independent of the verified-unit token-credit
objective. The minimum closes with
`NO_ERROR_RELATIONSHIP_LEARNING_SIGNAL` before optimization unless an original
DIAG group has variance on a supported verified relationship with
mapped original tokens. Facts alone do not pass this minimum.

The v30 input policy asks the model to write the evidence-backed relationships
first, then the same complete fact and attempt inventory. A public index lists
every actual retry-parent pair, without selecting pairs by their answers or
verified errors. All original responses remain in the input. A parent without a
final answer still contains observations to inspect, and a correct retry does not
prove that an earlier observation was corrected. Abstention remains permitted.
Repair feedback distinguishes the number of submitted relationships, their
verification status, and coverage of issues; an empty list is not a successful
relationship verification. These are public task/feedback changes, not new
generation calls, supplied error targets, or relaxed grading. Earlier captures
retain their frozen source and protocol; new runs require fresh source-bound
preparation and cache checks. This candidate's actual learning outcome is not yet
verified.

The v31 candidate adds optional `OUTPUT / FINALIZATION / final_answer` endpoints
and a `FINALIZED` relationship. This represents a stopped response with no final
answer record followed by an actual descendant with one nonempty terminal final
answer. The earlier evidence must cover its entire response. Empty answer markers,
truncated responses, multiple answers, retractions, trailing prose and sibling
comparisons do not establish this relationship. It certifies answer completion;
the later answer's correctness is checked and reported separately. It cannot
clear an observation/trace error or an unverified region.

Bounded synthesis feedback v5 additionally checks well-formed scalar facts from
the model's own `corrected_facts`, even when another submitted field is malformed.
Only independently wrong required slot IDs appear in `reinspection_slots`; no
correct values, missing-slot inventory, expected issues or teacher text is supplied.
Malformed, ambiguous, unsupported and absent-relation claims supply no hint. The
private verifier receipt remains outside the model payload, and update-time fresh
regrading checks its authenticity. Strict grading, rewards, routing and original
group/token credit are unchanged. This adds explicit verifier feedback and changes
the protocol hash; historical captures use their original frozen code. CPU checks
do not establish that the model will correct its observations or write valid lessons.

Bounded synthesis feedback v6 also reports submitted top-level attempt entries
with exactly `evidence_ranges: []`, using 1-based JSON array positions rather than
the entries' claimed attempt IDs. This public metadata is available even when
field validation prevents semantic checks. It reports at most 32 positions with
an explicit truncation flag and is bound to the actual submitted note. The retry
instruction requires nonempty valid original-line citations for nonempty originals;
`[]` remains valid for an actually empty original, with no invented line. These
positions are not verified errors or host-selected citations. Grading and rewards
remain unchanged, and actual model improvement is unverified.

Synthesis protocols additionally pin `notebook_repair_parent_policy` to
`first-public-shape-valid-v1`. A repair continues the first stopped candidate whose
submitted JSON containers, original-attempt count/order and literal citation bounds
are valid; if none qualifies, it continues candidate 0. The selector never reads
private grades, rewards, expected facts or verified issues. Public shape validity
does not establish semantic correctness, and an empty lesson list is not promoted
to a verified lesson. Every sampled sibling remains in the original reward group.
The selected candidate's feedback, complete token ancestry and selection receipt
are recomputed during replay; legacy notebook protocols retain fixed-first repair.
This addresses wasted retries on malformed candidate 0, not the separate failure
to generate accurate relationships. Actual generation effects require a new,
source-bound diagnostic; earlier frozen runs keep their original repair policy.

The separate `fact_context_probe --notebook-relationships` diagnosis isolates why
full-history notebooks contain `lessons=[]`. It reuses the first recovered fixed
actor's complete V35 history, original images, unverified reread and four original
DIAG seeds, but requests only cross-attempt relationship proposals. It uses fresh
F0,24K context,8K output and one call/four candidates, with no optimizer or feedback.
The closed historical source/protocol is pinned separately from fresh current-code
input/cache validation. Other probe modes retain their existing contracts.

Every proposal is preserved and independently checked by the existing relationship
verifier; malformed and truncated proposals remain failures. Full-notebook output
violates this diagnostic's output contract without erasing its actual scores.
Output-finalization evidence is distinct from native observation/operation repair.
Four identical valid proposals can show capability but supply no centered RL signal.
This diagnostic neither certifies a complete notebook nor authorizes larger training.
Any later notebook composition must retain all proposals as unverified model text;
verifier-approved proposals must not be selected as teacher targets.

```bash
"$NF4_PY" -B -m fine_tuning.notebook_rl.fact_context_probe \
  --preparation "$CURRENT_DATA/preparation.json" --model-path "$NF4_CACHE" \
  --cache-check "$CURRENT_CACHE_CHECK" --parent "$CLOSED_V35_MINIMUM" \
  --output "$NEW_RELATIONSHIP_OUTPUT" --notebook-relationships
```

`--notebook-relationship-comparisons` is a separate coverage-first diagnostic.
Use it instead of `--notebook-relationships` in the command above. It requires
model-written NATIVE and OUTPUT comparisons for each public direct retry pair,
followed by the unchanged `lessons` list. All original attempts and images stay
in the input. Pair coverage, citations, comparison prose and search statuses earn
no reward; `UNKNOWN` and `NO_SUPPORTED_ERROR` are unverified search outcomes.
Only independently verified relationships in the original output text can receive
the existing token credit. This restricted direct-pair task does not replace the
full-notebook schema or routing policy.

The actual v37 lessons-only diagnostic returned four identical empty lists
(8 tokens each, natural stop), despite complete history and available output
budget. Its saved-output CPU replay reproduced zero relationship credit. The
preceding v35 real update/save/reload succeeded on answer reward, while all twelve
notebooks had zero centered credit and zero additional gradient. These results
identify a generation/learning-signal failure, not a missing optimizer or reload.
The comparison variant is an intervention under test, not a successful repair.

Output checks are optional lesson evidence, not new mandatory attempt issues.
Credit v2 fixes its relationship denominator to all original native issues plus
all eligible missing-final-answer origins, including unrepaired branches. The
total relationship coefficient remains 0.0625. Only the model's actually submitted
and independently verified lesson tokens receive credit. Typed receipts distinguish
output-finalization credit from native error relationships; an aggregate
relationship gradient alone does not prove a visual-error learning contribution.
Historical v29 captures and their no-signal result retain their original scores.

Fixed-ten `--admission verified-synthesis-credit-exploration-v2` requires the
closed real minimum and fresh reload, freshly regraded complete synthesis groups,
unchanged full-group credit and MIS retention, and a retained positive verified
relationship whose separately measured relationship-only gradient is finite and
nonzero. That measurement excludes facts, strict reward and KL; OFF samples must
leave Vision gradients unchanged. Partial admission records strict correctness
as unverified, and does not certify whole-note accuracy or self-improvement.

This is a capacity candidate, not an8GB-fit or learning-success claim. A saved
eight-solution history already consumed14,284 input tokens, leaving only52 after
reserving a2,048-token note under16K. Notebook repair additionally retains the
previous note and feedback, so a successful first request does not prove that all
repair rounds fit. CPU-test the actual complete requests and each output
reservation; then, after the existing GPU run has exited, measure the longest
actual-history generation **and training forward/backward** on the target GPU.
Tiny cache-readiness probes and the model's positional capacity do not establish
this memory bound. Preserve OOM/timeout results without changing modules or budgets.
Actual24K calibration, update/fresh reload, strict note success and independent
learning benefit remain unverified; do not launch fixed-ten from CPU checks alone.

The separate `local_calibrate long-history` stage measures one actual complete
history before another training run. It requires a **closed** fixed-ten parent
and its pinned terminal CPU replay audit; a live prefix or minimal-run result is
rejected. Select the problem ID before this calibration and retain every solution
branch, successful retry, unverified reread and original image. The original
history's policy/protocol and the new fresh-F0 24K generation are recorded separately.

```bash
"$NF4_PY" -B -m fine_tuning.notebook_rl.local_calibrate long-history \
  --preparation "$LOCAL_DATA_24K/preparation.json" --model-path "$NF4_CACHE" \
  --cache-check "$CACHE_CHECK_24K" --parent "$CLOSED_TEN_RUN" \
  --parent-audit "$TERMINAL_AUDIT" --parent-audit-sha256 "$TERMINAL_AUDIT_SHA256" \
  --problem-id "$PRESELECTED_PROBLEM_ID" --output "$NEW_CALIBRATION_OUTPUT"
```

This supervised stage has a one-hour wall-time envelope including preparation
and cleanup. It samples one notebook with a2,048-token cap, preserves its raw
output **before** backward, and measures one signed generated-token log-probability
backward with Vision/LLM adapters and all four Mergers enabled. It creates no
optimizer and uses no answer/notebook reward or target labels. Gradients must be
finite and nonzero in all three groups, then cleared; frozen base and trainable
parameter values must remain unchanged. It reports actual sequence length,
generation/backward time and overall peak allocated/reserved memory.

Early stopping measures only the generated suffix, not a full24K sequence or
every repair round. This is neither RL nor SFT and proves no optimizer-state fit,
notebook correctness, module causality or learning benefit. Rebuild source-bound
preparation/cache readiness after code changes. Run it only after the existing
GPU process has exited; preserve failures without truncation or automatic fallback.

### Historical first-attempt notebook contract

This is a separate RTX5060 8GB experiment, not the BF16 baseline or tiny-model
fixture test. The reviewed training contract is `local-program-verified-notebook-v3`
with hardware profile `local-nf4-notebook-rl-v1`: standalone NF4 base/BF16 compute,
rank16 LLM and Vision adapters, all four FP32 trainable Mergers, 8192 context,
1024-token solutions, 768-token notes, T1/T2/T3 groups4/2/2 plus note group4,
seed3407 and one completion at a time. Nothing silently shrinks on OOM.
The minimal two-problem gate precedes fixed-ten training, capped at five updates.

After an actual incorrect-to-correct recovery the model samples its own JSON note.
It selects nonempty, distinct, increasing positive integer `first_attempt_lines`
from the complete numbered rendering of its actual first output. The host resolves
exact original Unicode/CRLF spans and binds them to the first sample/text hash;
there is no free-text quote fallback or repaired model output. The literal prior
conversation is retained; overflow is blocked, not truncated. Public source-specific
slot instructions contain no target values. Previous/corrected answer, all registered
facts, history and applicable independently verified diagnosis checks must all pass
for reward1; otherwise0. Known diagnoses remain strict. Independently UNKNOWN cause
requires honest UNKNOWN/none, with diagnosis null/UNVERIFIABLE, never healthy Vision.
Verifier evidence controls routing; a model's note never changes that route.
Lessons are not semantically certified. These are sampled RL actions, not SFT,
teacher labels or benchmark-derived training targets.

### Actual local evidence and limits

The frozen copied-source `minimal-v6` completed real update1: L252/M24/V48 tensors
changed, with positive gradients in all three groups. All624 saved tensors/payloads
were CPU-verified; a genuinely new process restored optimizer/state and matched
actual forward/short-generation probes. Private main closure
`minimal-v6-closed-main-v1.json` SHA `3abc7994…` binds reload result `cd9965dd…`
and the independent saved-response audit. All four actual notes earned0 and had
zero isolated policy-gradient contribution: registered facts passed0/4, despite
literal history passing3/4. This is real answer-reward RL/reload, not demonstrated
notebook benefit or self-improvement. Peak allocated7,399,128,576/reserved8,608,808,960
bytes; reserved exceeds reported SMI capacity8,546,942,976. Neither fully dedicated
VRAM residency nor paging causality is proven.

`ten-notebook-v4` completed all ten fixed problems, 90 sampled responses and two
real updates from fresh F0. Its initial624 hashes match the minimum's pre-update
state and differ from its post-update state. All624 saved tensors and a new-process
optimizer/state/forward/short-generation reload were verified. Both routing paths
were exercised: the second update changed L504/M24/V0 tensors and preserved all
Vision tensors. Two of20 actual notes earned reward1, and four note actions in one
group contributed to the combined RL+KL backward pass; reward-only parameter
gradients were not isolated. Six problems supplied no update-eligible variance.
The private closure `ten-notebook-v4-closed-main-v1.json` SHA `f0233b78…` binds these
observations. Training took7,686.107seconds; peak allocated8,331,329,536/reserved
9,579,790,336bytes do not prove fully dedicated-VRAM residency.

The initialization-matched answer-only control completed ten problems,92 responses
and three real updates. All624 saved tensors and a genuinely new-process
optimizer/state/forward/short-generation reload passed with canonical original
paths empty-masked. Private closure `ten-answer-only-v1-closed-main-v1.json`
SHA `c68d9de7…` binds107 files. Its last update changed L504/M24/V0 tensors.
Control disables note reward advantage, not KL; its note gradients were not
separately measured. Same-protocol external F0/FT comparisons are complete below;
no first-attempt or final-recovery improvement was observed.
The procedures share fixed inputs, initialization and budget; realized eligible
trees, token exposure, normalization and update counts can differ and must be
reported before attributing a difference specifically to notebook rewards.
Receipts belong to their exact frozen execution copies, which are not changed by
this documentation update or inherited by a moved checkout.

The fixed external set contains ten separate synthetic circuit **development**
items, not MMMU or a sealed professional-transfer test. The same frozen evaluator,
images, first-request tokens, seeds and NF4 precision were checked for both paired
comparisons. Evaluation performed zero updates; all three runs ended successfully.

| Model | First attempt | After second attempt | After third attempt | Verified recovery notes |
|---|---:|---:|---:|---:|
| F0: pretrained NF4 | 5/10 | 7/10 | 8/10 | 0/3 |
| NFT: notebook RL, 2 updates | 5/10 | 7/10 | 8/10 | 0/3 |
| CFT: answer-reward control, 3 updates | 5/10 | 8/10 | 8/10 | 0/3 |

Both paired first-attempt transitions are gains0/losses0, not merely equal totals.
Each run recovers three of its five initial errors (60%); final accuracy is80%.
All three recovery notes per model fail the corrected-structure requirement despite
passing history/answer checks. Training notes with positive reward do not establish
successful note generalization. Bare-letter first responses do not support visual
error attribution; UNKNOWN/null is not evidence of a healthy Vision pathway.
Actual generation calls are21/21/20 and output tokens1745/1683/1738 for F0/NFT/CFT.
Saved evaluation elapsed seconds are412.938/381.684/398.523. Sleep/pause and observed
host-clock discrepancies prohibit treating this as a reliable throughput comparison.

All raw evaluations passed independent CPU replay and file-identity checks. Private
evidence root: `/home/jang2/mmdl-artifacts/notebook-rl-local-nf4-20261021/`:
`external-eval2-{F0,NFT,CFT}-v1/`, both `external-eval2-F0-*-compare-v1.json`,
`ten-arm-independent-v1/external-eval2-*-audit-v2.json`, and
`local-nf4-final-closure-v1.json`. The earlier CPU auditor timeout is preserved;
v2 only caches the unchanged vocabulary-size bound and retains rejection tests.
No GPU response was regenerated to repair that CPU audit.

The historical first-attempt contract's local functional validation is complete, not a demonstrated
performance gain or completion of the current full-history objective. No automatic large training, BF16 cloud validation or new MMMU
evaluation follows this null result. One seed and ten circuit items cannot establish
generalization, module responsibility or that the method cannot work at another scale.

Historical rewardv2 `ten-notebook-v3` did complete ten trees, update3 and fresh
reload. All20 notes earned0; there was no note reward-gradient signal. Later KL
contribution must not be called notebook-reward learning. Earlier v1/v2 interrupted
runs and minima remain separate immutable history, not v3 evidence. The historical
BF16/shared-policy and authored tiny-fixture sections below are separate protocols.

Original BF16 cache **must be retained** because six legacy checkpoints logically
depend on it for original replay. Historical v2 standalone train/reload was tested
with canonical original files reversibly absent; they were then restored.
Current v3 fresh notebook reload additionally passed with the canonical original
root read-only empty-masked in a child mount namespace: five native original-path
opens returned ENOENT while host files remained unchanged. The control training
used the same mask and completed training and fresh reload. This verifies canonical
path independence, not exclusion of every possible alias. Both original weight
blobs remain required by six legacy checkpoints; eligible nonshared blobs and
deleted bytes are both zero. Private data, raw responses, receipts, checkpoints
and model weights stay outside this package; no cloud/API/upload or Git publication
is implied.

### Copy only the package; install the declared environment

Run from the directory containing the copied `fine_tuning/`, not from inside it.
Set `PACKAGE_PARENT` and the external `NF4_ENV` path explicitly. No sibling
`src/`, `mmdl`, `experiments`, `env/` or external manifest code is required for
these local entrypoints. Model/data/provenance/artifact paths remain explicit private
inputs, not bundled payloads. A clean Python3.12/Linux x86_64 environment was installed
and checked on the same OS/interpreter; that is not clean-OS or cross-GPU proof.

```bash
cd "$PACKAGE_PARENT"
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
python3.12 -m venv "$NF4_ENV"
NF4_PY="$NF4_ENV/bin/python"
"$NF4_PY" -m pip install --index-url https://pypi.org/simple --only-binary=:all: \
  -r fine_tuning/notebook_rl/resources/requirements-vllm.lock
"$NF4_PY" -m pip install --index-url https://pypi.org/simple --only-binary=:all: \
  --no-deps --require-hashes -r fine_tuning/notebook_rl/requirements-local-nf4.txt
"$NF4_PY" -m pip check
```

The base lock includes the official cu128 extra index and exact runtime pins,
including Torch2.8.0+cu128, Transformers4.57.1, PEFT0.17.1 and Accelerate1.11.0.
It is **not** a hashlocked file; do not add `--require-hashes` or combine the two
requirements files in one install. The overlay pins bitsandbytes0.48.2,
qwen-vl-utils0.0.14 and av16.0.1 with wheel hashes; it is not a complete environment.
Actual clean-install checks found zero mismatches against all163 base plus3 overlay
pins and `pip check` passed. CPU imports/help resolved only the copied package with
CUDA uninitialized. New-device NF4/model execution still needs its actual gates.

### Fresh destination data and execution gates

Every output variable below must name a new external private location. Inputs are
existing reviewed graph data, rights/exclusion proofs, original fixed selection,
pinned snapshot and an independently frozen circuit cohort. Moving code or changing
its bytes requires fresh destination-bound components, producer descriptor and data
request; never rewrite old receipts or accept basename-only scanner matches.

The existing CPU measurement CLI is:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
"$NF4_PY" -m fine_tuning.notebook_rl.smoke_data_evidence \
  --build "$ORIGINAL_BUILD" --review "$REVIEW" --review-sha256 "$REVIEW_SHA256" \
  --benchmark-probe "$BENCHMARK_DENIAL_TARGET" --out "$COMPONENTS"
```

This measures components, not GPU training or a new contamination scan. Assemble
them with existing `smoke_data_evidence.assemble`, preserving original review,
exclusion, source registry, binding and metadata_rebind; use current
`approval.code_hashes()` and the fresh components' measured producer descriptor
in the measurement scope. `PREVIOUS_REQUEST` below must be the resulting fresh
destination-bound data request, not an old absolute-source request.
`local_run prepare` revalidates it and keeps the original fixed-ten selection;
BLOCKED evidence is preserved and exits2. Hash/path failures stop execution.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$NF4_PY" -m fine_tuning.notebook_rl.quantized_cache save \
  --model-path "$PINNED_ORIGINAL_SNAPSHOT" --cache "$NF4_BASE" --output "$CACHE_SAVE"
# Separate process; original root/blob access is blocked in this process only.
"$NF4_PY" -m fine_tuning.notebook_rl.quantized_cache check \
  --cache "$NF4_BASE" --parent "$CACHE_SAVE" --output "$CACHE_CHECK"
"$NF4_PY" -m fine_tuning.notebook_rl.local_run prepare \
  --previous-request "$PREVIOUS_REQUEST" --selection "$SELECTION" --output "$LOCAL_DATA"
timeout --signal=TERM --kill-after=30s 14400s \
  "$NF4_PY" -m fine_tuning.notebook_rl.local_run minimal \
  --preparation "$LOCAL_DATA/preparation.json" --model-path "$NF4_BASE" \
  --standalone-base --cache-check "$CACHE_CHECK" --output "$MINIMAL_RUN"
# Only after the actual capture closes with a saved update.
"$NF4_PY" -m fine_tuning.notebook_rl.local_run reload \
  --capture "$MINIMAL_RUN" --output "$RELOAD_CHECK"
"$NF4_PY" -m fine_tuning.notebook_rl.local_train train \
  --preparation "$LOCAL_DATA/preparation.json" --model-path "$NF4_BASE" \
  --cache-check "$CACHE_CHECK" --minimal "$MINIMAL_RUN" --minimal-reload "$RELOAD_CHECK" \
  --arm notebook --output "$TEN_NOTEBOOK"
"$NF4_PY" -m fine_tuning.notebook_rl.local_train reload \
  --capture "$TEN_NOTEBOOK" --output "$TEN_NOTEBOOK_RELOAD"
"$NF4_PY" -m fine_tuning.notebook_rl.local_train train \
  --preparation "$LOCAL_DATA/preparation.json" --model-path "$NF4_BASE" \
  --cache-check "$CACHE_CHECK" --minimal "$MINIMAL_RUN" --minimal-reload "$RELOAD_CHECK" \
  --arm answer-only --match-initialization "$TEN_NOTEBOOK/initialization.json" --output "$TEN_CONTROL"
"$NF4_PY" -m fine_tuning.notebook_rl.local_train reload \
  --capture "$TEN_CONTROL" --output "$TEN_CONTROL_RELOAD"
```

Both arms start fresh NF4 F0, not the minimum's trained weights. Both sample/check
notes; answer-only disables their reward advantage, retaining answer actions,
group normalization, caps and KL. KL may still contribute note-action gradients.
No-signal pairs are recorded, never replaced; later rollouts/compute and actual
update counts can diverge after learning. A terminal NO_OPTIMIZER_UPDATES is not
a successful trained/reloaded parent and must not be forced through evaluation.

Use an owned timeout/PID, detached stdin and direct regular-file stdout/stderr for
long jobs; a transient terminal or `tee` exit is not model completion. Preserve
all failed output directories and frozen sources through closure/reload.

### Separate circuit evaluation v3

`local-nf4-circuit-recovery-notebook-v3` checks every actual attempt, including the
successful retry, against independently regraded full-frontier journals. Each note
cites original line IDs and reports per-attempt answers/statuses; corrected_structure
remains a strict circuit schema, not graph facts. The actor never receives structure
gold. Bare-letter attempts cannot verify visual cause: `visual_cause=null` is
unsupported attribution, not healthy Vision. Literal retry histories, three-attempt
cap, 1024-token solution cap and images remain fixed; notes use a separate lossless
quoted history with original images/question, context16384 and output2048.
The bounded worker transport allows8MiB for complete token/response journals and
fails rather than truncating excess; actual request/response byte counts are saved.
Current v3 CPU checks are not an actual-model execution result.

Freeze one evaluator source/protocol for F0, notebook FT and control FT. Versionv3
changes stage seeds, so an old v2 F0 cannot be reused. Current evaluator identity
is recorded separately from historical closed-v3 training identity; an evaluation-only
revision need not retrain those weights. New training after evaluator source changes
still requires newly bound training preparation. All captures need actual closed
updates and fresh reload; the unchanged external preparation binds the frozen
ten-case circuit development cohort. It is not sealed professional transfer,
MMMU, or universal contamination certification. Whole train/circuit multigraphs
differ in all100 cross-pairs, but subgraph/semantic overlap is not ruled out.

```bash
"$NF4_PY" -m fine_tuning.notebook_rl.local_eval evaluate \
  --capture "$TEN_NOTEBOOK" --reload-result "$TEN_NOTEBOOK_RELOAD/result.json" \
  --minimal-capture "$MINIMAL_RUN" --minimal-reload "$RELOAD_CHECK" \
  --external-preparation "$EXTERNAL_DATA/preparation.json" --condition F0 --output "$EXTERNAL_F0"
"$NF4_PY" -m fine_tuning.notebook_rl.local_eval evaluate \
  --capture "$TEN_NOTEBOOK" --reload-result "$TEN_NOTEBOOK_RELOAD/result.json" \
  --minimal-capture "$MINIMAL_RUN" --minimal-reload "$RELOAD_CHECK" \
  --external-preparation "$EXTERNAL_DATA/preparation.json" --condition FT --output "$EXTERNAL_FT"
"$NF4_PY" -m fine_tuning.notebook_rl.local_eval evaluate \
  --capture "$TEN_CONTROL" --reload-result "$TEN_CONTROL_RELOAD/result.json" \
  --minimal-capture "$MINIMAL_RUN" --minimal-reload "$RELOAD_CHECK" \
  --external-preparation "$EXTERNAL_DATA/preparation.json" --condition FT --output "$EXTERNAL_CONTROL_FT"
"$NF4_PY" -m fine_tuning.notebook_rl.local_eval compare \
  --before "$EXTERNAL_F0/result.json" --after "$EXTERNAL_FT/result.json" --output "$PAIRED_RESULT"
"$NF4_PY" -m fine_tuning.notebook_rl.local_eval compare \
  --before "$EXTERNAL_F0/result.json" --after "$EXTERNAL_CONTROL_FT/result.json" --output "$CONTROL_PAIRED_RESULT"
```

A single F0 is reusable only if actual initial hashes and all evaluator/input/seed/
image/pixel conditions match each corresponding FT capture. These commands document
the gates; they do not assert that control, evaluation, promotion or their new
integrated CPU/source validation has already completed.

### Opt-in AI-reviewed mixed pool (2026-10-30; CPU fixtures only)

Approval stage `ai_reviewed_local` keeps every measurement check (exclusion receipt, full
index, source rights, split leaks, family quarantine over the whole pool, calibration and
engine coverage). It allows HUMAN labels and replaces the human D3/D7 review with an AI
review record per source (`ai-reviewed-local-review-v1`, `human_verified: false`, the
first min(20, n) IDs in seed-keyed hash order, `render_ok`/`leak_free`/`gold_supported`;
one failing verdict blocks). D1/D2/D4 rerun project generators only; HUMAN and external
PROGRAM labels are covered by the review. `pool_data_evidence produce` writes D1–D8 and
the data request (D4: interventions for graph sources, `same-template-natural-variation-v1`
for other generators). A pool preparation (`notebook-rl-local-pool-data-v1`) keeps each
built mode and binds a label-blind rule (`pool-local-selection-v1`): minimal = the first
single- and two-image fixed actors, fixed = ten train IDs (five pairs in order),
evaluation = held-out IDs whose sources, structure families and images are disjoint
from training. `local_run minimal`, `reload` and `local_train` read it through the same
`read_preparation`. The pool evaluator (`local-pool-recovery-evaluation-v1`) reuses the
ladder repair helpers under the training protocol's route policy; it is not MMMU.

```bash
"$NF4_PY" -m fine_tuning.notebook_rl.pool_data_evidence review-sample --build "$POOL"
"$NF4_PY" -m fine_tuning.notebook_rl.pool_data_evidence produce --build "$POOL" --review "$AI_REVIEW" \
  --benchmark-probe "$BENCH_FILE" --index "$INDEX" --source-registry "$REGISTRY" --evidence "$EVIDENCE" --out "$POOL_D"
FLAGS=(--note-contract repair-note-compact-v1 --answer-verified-route --finalization-reward --slotless-source hf-...)
"$NF4_PY" -m fine_tuning.notebook_rl.local_run propose-pool --build "$POOL" "${FLAGS[@]}" --output "$RULE"
"$NF4_PY" -m fine_tuning.notebook_rl.local_run prepare-pool --build "$POOL" --approval "$POOL_D/data_request.json" \
  --rule "$RULE" "${FLAGS[@]}" --output "$POOL_PREP"
"$NF4_PY" -m fine_tuning.notebook_rl.local_eval evaluate-pool --capture "$TEN" --preparation "$POOL_PREP/preparation.json" \
  --reload-result "$TEN_RELOAD/result.json" --minimal-capture "$MIN" --minimal-reload "$MIN_RELOAD" \
  --condition FT --arm notebook --output "$POOL_FT"
"$NF4_PY" -m fine_tuning.notebook_rl.local_eval replay-pool --record "$POOL_FT/recovery/result.json" \
  --preparation-sha256 "$PREP_SHA" --output "$POOL_FT_REPLAY"
"$NF4_PY" -m fine_tuning.notebook_rl.local_eval compare-pool --record F0="$POOL_F0/recovery/result.json" \
  --record FT-notebook="$POOL_FT/recovery/result.json" --record FT-control="$POOL_CTL/recovery/result.json" \
  --preparation-sha256 "$PREP_SHA" --output "$POOL_TABLE"
```

### Opt-in scale options (`scale-options-v1`, 2026-10-30; CPU/mock tests only)

`scale.py` adds one protocol group `scale`, bound into the repair protocol (and its sha256) only
when a flag is given; without flags every protocol hash and capture replay is unchanged. Give the
same flags to every protocol-building command of one chain (`local_run prepare|prepare-pool|minimal`,
`quantized_cache save|check` for NF4, `passrate measure`); `local_train`, reloads and `local_eval`
restore them from the minimum's protocol. These commands share `local_profile.add_repair_cli_arguments`
/ `repair_cli_keywords`, so `quantized_cache` now also takes `--answer-verified-route`,
`--finalization-reward` and `--slotless-source`, and its readiness protocol equals the run's.

| flag | key | effect |
|---|---|---|
| `--rollout-batch-rows N` | `rollout_batch_rows` | row-seeded batched HF sampling; probes/receipts carry `hf-row-seeded-batch-v1` |
| `--problems-per-update K --max-updates M --train-problems N` | `problems_per_update`, `max_updates`, `train_problems` | N = K*M problems in K-problem updates (default 2/5/10) |
| `--checkpoint-every-updates k` | `checkpoint_every_updates` | intermediate checkpoint every k updates (sync after every update; final always saved) |
| `--evidence-audit-rule '{"first":true,"last":true,"every":m}'` | `evidence_audit_rule` | evidence VJPs only on those updates; others `measured: false`, S three-state (S only from audits) |
| `--policy-precision bf16` | `policy_precision` | `policy.build_policy` on CUDA, no NF4 cache/guard, cheap base checksum per update + full SHA at saves |
| `--rollout-engine vllm --vllm-gpu-memory-utilization F --vllm-max-num-seqs N --vllm-kv-cache-memory-bytes B [--vllm-enable-prefix-caching]` | `rollout_engine`, `vllm` | vLLM sampler (bf16 only) with a merged-weight push after every update; HF engine keeps inputs, updates and saved probes |
| `--no-park-adam-moments` | `park_adam_moments` | keep AdamW moments resident (always so under bf16) |

`local_train train` adds `--resume-from <closed segment>` (new segment directory; parent prefix is
replayed on CPU, batches after its last cadence checkpoint are re-collected with the same seeds) and
`--problem-range a:b` (a fixed chunk of the preparation's selection as its own fresh-F0 capture);
both need a bound scale group. `--cache-check` is required only for NF4. Every `local_train`,
`local_eval` and `passrate measure` run writes `<output>/throughput_receipt.json`
(`throughput-receipt-v1`, measured values; mirrored to `$MMDL_RUN_DIR` when the output lies inside
it). Only the v1 local preparations (`notebook-rl-local-pool-data-v1` and the earlier fixed-ten ones,
`local_data.py`) hold exactly ten fixed actors; N > 10 uses the v2 / proof preparation formats
(`notebook-rl-local-pool-data-v2`, `notebook-rl-local-pool-data-proof-v1`; see "RunPod 증명 경로" below).
No GPU, BF16-GPU or vLLM run has been made with these options.

## RunPod 증명 경로 (2026-10-31)

CPU/목(mock) 시험으로만 확인한 opt-in 경로다. 실제 GPU·RunPod 실행은 없었고 어떤 단계도 자동 시작하지 않는다
(CLAUDE.md §2.6). 아래 플래그는 각 명령의 argparse에서 확인한 것만 적었다. Pod 절차의 정본은 `pod/RUNBOOK.md`이며
여기서는 요약만 한다. 기본값은 모두 불변이고, 옵션을 주지 않으면 기존 protocol hash도 그대로다.

### 1. 데이터: 동결 union vault + 동결 approval

```bash
M=fine_tuning.notebook_rl
"$PY" -m $M.pool_freeze freeze --shard ROOT [--shard ...] --quality-review REVIEW.json|none [...shard마다] \
  [--approval OUT.json|none ...] --category-map MAP.json --quota QUOTA.json --seed N --output FREEZE \
  [--quality-exclusions EXCL.json]                          # -> FREEZE/dataset_manifest.json
"$PY" -m $M.pool_freeze materialize --manifest FREEZE/dataset_manifest.json --shard ROOT [...] \
  --source-registry REGISTRY.json [...shard마다] --evidence EVIDENCE.json [...shard마다] --output VAULT
"$PY" -m $M.pool_freeze verify --manifest FREEZE/dataset_manifest.json --vault VAULT   # VERIFIED, exit 0 (불일치 exit 1)
"$PY" -m $M.local_run export-approval --build VAULT --approval DATA_REQUEST.json --output APPROVAL_EXPORT.json
```

- Approval stage는 `ai_reviewed_local`(`human_verified=false`). v2 review(`ai-reviewed-local-review-v2`)는
  `acceptance-sampling-v1`: 실패 0이면 통과, 실패 1 이상은 `min(60, n)`개 검토·실패 F<=2일 때만 실패 항목만 제외하고
  아니면 source 전체 차단. 표본은 `pool_data_evidence review-sample --build VAULT --count 60`, 이후 위 절의
  `pool_data_evidence produce`로 D1–D8과 `data_request.json`을 만든다.
- `export-approval`은 fresh approval 출력을 저장하고, 아래 frozen round trip이 통과한 경우에만 파일을 남기며
  `{"status":"APPROVAL_OUTPUT_EXPORTED","path":...,"file_sha256":...}`를 출력한다. 그 SHA-256을 pin한다.
- **frozen-approval mode**(`frozen-approval-verify-v1`, 명시적 opt-in, 기본 아님): `--frozen-approval APPROVAL_EXPORT.json
  --frozen-approval-sha256 HEX64` (둘 다 함께). approval을 다시 실행하지 않고 저장 출력의 byte/hash와 union vault 전체
  재해시(`pool_freeze.verify`, union receipt·build manifest 결속)만 검증한다. 받는 명령: `local_run minimal`, `propose-proof`,
  `prepare-pool`, `local_train train`, `passrate measure`(`--preparation`일 때만; `--build-root`와는 불가).
  `local_run reload`, `local_train reload`, `local_eval`에는 이 플래그가 없고 capture에 기록된 `data_verification`을 읽는다.
  `prepare-pool`은 frozen일 때도 `--approval DATA_REQUEST.json`(결속되는 요청)을 요구한다.

### 2. 증명 선택과 학습·평가 체인

공통 플래그(아래 3절)를 `FLAGS`, scale 플래그를 `SCALE`로 둔다. `SCALE`은 protocol hash에 결속되므로 protocol을 만드는
모든 명령(`passrate measure`, `local_run propose-proof|prepare-pool|minimal`, NF4면 `quantized_cache`)에 같게 준다.
`local_train train`은 scale/turn1/retry 플래그를 받지 않고 minimum의 protocol에서 복원한다(note-contract·route·
finalization·slotless는 minimum과 같게 다시 줘야 하며 다르면 거부).

```bash
SCALE=(--problems-per-update K --max-updates M --train-problems N   # N == K*M
       --checkpoint-every-updates k --evidence-audit-rule '{"first":true,"last":true,"every":m}'
       --policy-precision bf16 --rollout-engine vllm --vllm-gpu-memory-utilization F
       --vllm-max-num-seqs S --vllm-kv-cache-memory-bytes B)
FROZEN=(--frozen-approval APPROVAL_EXPORT.json --frozen-approval-sha256 HEX64)   # Pod; 로컬은 --approval DATA_REQUEST.json
# 1) F0 통과율 -> 학습 선택 (weights는 report에 결속: 학습 선택은 F0, 즉 --checkpoint/--binding 없이 측정한 것만)
"$PY" -m $M.passrate measure --build-root VAULT --split train --model-path MODEL --k 4 --seed-namespace NS \
  "${FLAGS[@]}" "${SCALE[@]}" --output PASS        # --preparation PREP.json "${FROZEN[@]}" 도 가능; NF4면 --standalone-base --cache-check
"$PY" -m $M.passrate select --report PASS --min-pass 1 --max-pass 3 [--max-blocking 0] --output SEL.json
# 2) label-blind 규칙 pool-proof-selection-v1 -> 준비 notebook-rl-local-pool-data-proof-v1
"$PY" -m $M.local_run propose-proof --build VAULT "${FLAGS[@]}" "${SCALE[@]}" --sentinels-per-subject S \
  [--exclude-source GLOB ...] [--seed 3407] --passrate-selection SEL.json --passrate-report PASS "${FROZEN[@]}" --output RULE.json
"$PY" -m $M.local_run prepare-pool --build VAULT --approval DATA_REQUEST.json --rule RULE.json \
  "${FLAGS[@]}" "${SCALE[@]}" "${FROZEN[@]}" --output PREP
# 3) minimal -> reload -> train -> reload
"$PY" -m $M.local_run minimal --preparation PREP/preparation.json --model-path MODEL --repair-notebook \
  "${FLAGS[@]}" "${SCALE[@]}" "${FROZEN[@]}" [--min-free-gib G] --output MIN       # NF4면 --standalone-base --cache-check
"$PY" -m $M.local_run reload --capture MIN --output MIN_RELOAD
"$PY" -m $M.local_train train --preparation PREP/preparation.json --model-path MODEL --minimal MIN --minimal-reload MIN_RELOAD \
  --arm notebook --admission verified-functional-repair-exploration-v1 [--max-seconds T] [--min-free-gib G] \
  --note-contract repair-note-compact-v1 --answer-verified-route --finalization-reward --slotless-source SRC \
  "${FROZEN[@]}" --output TRAIN                                                   # NF4면 --cache-check
"$PY" -m $M.local_train reload --capture TRAIN --output TRAIN_RELOAD
# 4) 새 프로세스 재로딩 뒤 held-out sentinel 평가 (F0/FT/control 각각; replay-pool/compare-pool은 위 절과 같다)
"$PY" -m $M.local_eval evaluate-pool --capture TRAIN --preparation PREP/preparation.json \
  --reload-result TRAIN_RELOAD/result.json --minimal-capture MIN --minimal-reload MIN_RELOAD \
  --condition FT --arm notebook --output POOL_FT
```

- `propose-proof`은 `--approval`(로컬, fresh)과 `--frozen-approval*` 중 정확히 하나, `--train-problems`와
  `--problems-per-update`, 그리고 `--passrate-selection`/`--passrate-report` 동시 지정(생략하면 eligible 전체)을 요구한다.
  체크포인트로 측정한 report는 `hard-negatives`에만 쓸 수 있고 학습 선택으로는 거부된다.
- 규칙은 문항당 구조 family 하나, 상한 N, 남은 항목은 `dropped_ids`, subject별 sentinel(dev/sealed, 학습과 구조·이미지 분리,
  같은 source 허용 = 분포 내 held-out이지 source 전이가 아님). 프로그램 생성 source(`gen-*`, `hf-chebi`, `hf-ptbxl`)는 기본 제외.
- `evaluate-pool` 결과 `recovery/result.json`의 `metrics.by_subject`/`by_source`가 sentinel별 분해다. 정확도로 보고하는
  것은 `first_attempt_accuracy`뿐이며 `recovery_rate_diagnostic`(검증기 피드백 뒤 CORRECT 비율)은 진단 지표이고
  정확도로 보고하지 않는다(§2.8).
- 모든 repair admission은 실행 허용이지 S/E 인증이 아니다. 선택은 job 승인 내용이다(`verified-native-repair-learning-v1` /
  `verified-functional-repair-exploration-v1`).

### 3. 증명이 쓰는 protocol 옵션 (`FLAGS`)

```bash
FLAGS=(--note-contract repair-note-compact-v1 --answer-verified-route --finalization-reward --slotless-source SRC [--slotless-source ...]
       --turn1-contract p0-eval-aligned-v1 --turn1-max-new-tokens 1024 --retry-out-of-choice incorrect-v1)
```

| 플래그 | 의미 / 제약 (argparse + `local_profile` 검증) |
|---|---|
| `--note-contract repair-note-compact-v1` | compact 오답노트 계약(§3.10); 기본 `repair-note-v2` |
| `--answer-verified-route` + `--slotless-source SRC` | slot 없는 트리는 답 검증만으로 `ANSWER_VERIFIED` admission, Vision 동결(Merger+LLM 어댑터만 학습), VISION_OFF 증거 아님; 두 플래그는 정확히 함께, compact 계약 필요, source 반복 가능·중복 불가 |
| `--finalization-reward` | 한 번의 파싱 가능한 최종 답 + 상한 내 stop 보상(gold 미사용); compact 계약 필요 |
| `--turn1-contract p0-eval-aligned-v1` | slotless actor의 T1을 MMMU P0 메시지·V8 파서/채점에 정렬; route 필요. 재시도·ErrorCheck·노트는 불변 |
| `--turn1-max-new-tokens N` | T1 상한(생략 시 1024, 24576 미만); `--turn1-contract` 필요 |
| `--retry-out-of-choice incorrect-v1` | 재시도가 보기에 없는 객관식 답을 확정하면 INCORRECT; `--turn1-contract` 필요 |

`FLAGS`는 `local_run prepare|propose-pool|propose-proof|prepare-pool|minimal`, `passrate measure`, `quantized_cache`가 공유한다.
`local_train train`은 앞의 네 개(note-contract·route·finalization·slotless)만 받는다.

### 4. 평가: `proof_eval` (사전 등록, seed 평균만)

```bash
"$PY" -m $M.proof_eval export --checkpoint CKPT --binding BINDING_OR_LAUNCH.json --base BF16_SNAPSHOT --output EXPORT --code-commit SHA40
"$PY" -m $M.proof_eval preregister --output PREREG.json [--b2-predictions PATH]            # PREREG.json.sha256도 기록
"$PY" -m $M.proof_eval ledger-start --prereg PREREG.json --arm F0|FT --variant VARIANT --run-id RID --ledger LEDGER.jsonl   # 매 run 전
"$PY" -m $M.proof_eval run-eval <scripts/eval.sh와 같은 인자>          # c16 변형 파일만; --protocol .../eval_configs/mmmu_val_v8_c16_seed{3407,3408,3409}.yaml
"$PY" -m $M.proof_eval ledger-fail --ledger LEDGER.jsonl --run-id RID --reason TEXT [--artifact-root ROOT]   # COMPLETE run은 거부
"$PY" -m $M.proof_eval analyze --prereg PREREG.json --ledger LEDGER.jsonl --runs RUN_DIR [...] --ft-model-ref EXPORT/model_ref.json --output REPORT
```

- `export`는 체크포인트 하나를 BF16 평가 모델 + `model_ref.json`으로 병합한다. 변형 id는 `mmmu-val-v8-c16-seed{3407,3408,3409}-v1`
  (statistical, 동시 16)과 `mmmu-val-v8`(course, 동시 2, seed 3407, `configs/eval/mmmu_val_v8.yaml`).
- 과제용 점수는 사전 등록한 course run 하나(`mmmu-val-v8`, 평소의 `scripts/eval.sh`, 동시 2·seed 3407)이며 이것도
  `ledger-start` 뒤에 시작한다. `run-eval`은 c16 변형 3개만 허용한다.
- `analyze --ledger`는 ledger에 시작 기록이 없는 run, 다른 (arm, variant)로 시작된 run, FAILED run, 등록된 변형의 누락·
  교체된 반복을 거부한다. seed는 문항별 평균으로만 합치며 reducer 인자·반복 run·두 번째 FT 체크포인트는 없다(§2.8).
  `<output>.json`과 `<output>.md`를 쓴다.

### 5. Pod (요약; `pod/RUNBOOK.md`가 정본)

- 로컬: `python -m $M.pod.data_bundle pack --spec SPEC.json --out-dir DIR --name B` (spec `pod-data-bundle-spec-v1`;
  archive sha256·`manifest_sha256` 출력). Pod: `data_bundle stage --archive B.tar --manifest B.bundle.json --manifest-sha256 SHA`
  후 `data_bundle verify --manifest B.bundle.json --manifest-sha256 SHA [--archive B.tar]` (VERIFIED, exit 0). 업로드는 사용자 승인(GD).
- `job.json`(`pod-job-v1`): `commit`(40자), `volume_root`, `run_dir`, `stage`, `smoke`/`main` argv(venv python), `max_main_seconds`,
  `min_free_gib`, `financial{owned_pod_id,gpu_cutoff_utc,billing_start_utc,balance_usd,reserve_usd,hourly_usd}`,
  `data{bundle_manifest,bundle_manifest_sha256,dataset_manifest,dataset_manifest_sha256,vault_root}`,
  `approval{approved_by_user,approved_utc,approval_note,approved_sha256}`. `approved_sha256`은
  `python -m $M.pod.job JOB --content-sha256` 값이며 이후 어떤 수정도 승인을 무효로 한다. 금융 한도는 job에서만 온다.
- 일회성 환경 준비: `bash fine_tuning/notebook_rl/pod/volume.sh --job prep.json [--execute]` (`version: pod-prep-v1`). 학습: `bash fine_tuning/notebook_rl/pod/start.sh
  --job /abs/JOB.json [--quick-verify] [--execute]` (dry-run 기본). preflight는 clean clone·lock·모델 hash·`data_bundle`·
  `pool_freeze.verify(dataset_manifest, vault_root)`를 확인하고 smoke(300 s) -> main 순으로 감시자 아래에서 실행한다.
  `min_free_gib` 아래면 `DISK_LOW`, 재정/GPU 마감이면 `BUDGET_STOP`; 자동 재시작은 없다.
- 회수는 `pod.collect pack|verify|state` 순서 `REMOTE_VERIFIED -> LOCAL_VERIFIED -> POD_TERMINATED -> OWNED_STORAGE_DELETED
  -> CLEANUP_VERIFIED`. 중간 단계 ledger는 `POD_TERMINATED`에서 멈춘다.
- 한 job의 `main`은 argv 하나이므로 단계 목록은 `pod/chain.py`(`plan`으로 `pod-chain-spec-v1`을 쓰고 `run --spec --spec-sha256`으로
  순서대로 실행, 재시도·fallback 없음)가 실행하고, MMMU 평가 단계(`eval-f0|eval-ft|eval-course`)의 spec은 `pod/eval_stage.py plan`이 만든다.
  GPU 확인은 `pod/gpu_smoke.py`(MMMU를 건드리지 않는 smoke), 평가 결과(7개 run 디렉터리·ledger·prereg·`model_ref.json`, 병합 가중치 제외)는
  `collect pack-eval --params P --stage S --run-dir D --out-dir O`로 회수하고 로컬 `verify`는 `/home/jang2/mmdl-artifacts/` 아래에만
  풀며 그 뒤 `proof_eval analyze`를 로컬에서 돌린다. Pod venv는 `env/requirements-proof.lock`(vllm lock + `bitsandbytes==0.48.2`; 학습 코드는
  BF16에서도 bitsandbytes를 import한다)이고, 학습 쪽 가드가 MMMU를 막도록 `$HOME/mmdl-data/evaluation`을 `<volume_root>/eval-data`로
  가리키는 symlink를 학습 Pod마다 만든다(`eval_isolation` preflight가 확인).

### 6. 호스트 범위 생존 판정

`local_run.host_identity()`는 `{hostname, boot_id, pid_ns}`를 기록하고 launch/restored 기록에 `host`로 들어간다.
`owner_alive(record)`는 다른 host/boot/PID namespace에서 쓴 기록(PID가 여기서 의미 없음)은 False로, `host`가 없는 옛 기록은
기존의 단순 PID 검사로 판정한다. `same_process`도 PID와 host를 함께 비교한다(`pod/collect.py`의 `owner_alive`는 pid+startticks 기반의 별개 검사).

## Historical/alternative BF16 shared-policy smoke (3090 / 4090 target)

`shared_smoke` adds a separate, bounded **ten-problem / at most two RL updates**
execution profile, `hf_shared_policy_24gb_v1`. One live unmerged BF16 PEFT policy
generates one completion at a time, then performs the existing RL update. The
adapter-disabled F0 reference shares frozen base storage. It does not construct a
second vLLM model, reserve its fixed 10GiB KV cache, or export merged weights on
each update. Actual single/two-image logprob probes are required initially and
after each update. A separate process checks the saved weights/step/cursor only
after the capture process has exited.

The original `protocol_v1.yaml`, BF16 precision, image budgets, 16K context,
3072-token solution caps, T1/T2/T3 sample counts, rewards, routed parameter groups
and optimizer are unchanged. `ALG-NB3` retains three solution turns;
`ALG-NB+D` retains its existing two-turn, reward-trained DIAG configuration.
The caller selects exactly one; this profile does not silently combine them.
Additional final audit notes are explicitly bound into the execution identity;
they are preserved for rejected trees too, and are **not** SFT targets, loss groups
or routing authority. Existing verifier-based routing and balanced DIAG RL remain
the only corresponding training signals. No real recovery means no claimed
self-correction success; uniform rewards can yield zero updates.

Use an explicit private job with format `notebook-rl-shared-hf-smoke-job-v1`,
purpose `BOUNDED_SHARED_HF_RL_SMOKE_ONLY`, arm `MOD-RV`, one supported algorithm,
and ten eligible, structure-distinct `problem_ids` including the bound one/two-image
probe actors. Other fields follow `gate_session`'s checked descriptor/path/device
schema. All exclusions, current-source data approvals, pinned model checks, clean
MMDL_test commit and the owned financial watchdog still apply. Old approvals are
not reusable after code changes. This is not approval for general training.

```bash
python -m fine_tuning.notebook_rl.shared_smoke --job /private/shared24-job.json --check
python -m fine_tuning.notebook_rl.shared_smoke --job /private/shared24-job.json --execute
# After the preceding process exits, start a new Python process:
python -m fine_tuning.notebook_rl.shared_smoke --job /private/shared24-job.json --reload-check
```

`--check` validates data only, not GPU fit. `--execute` returning
`CAPTURED_PENDING_FRESH_RELOAD` is **not** final verification. A failed synchronization,
incomplete cohort, budget stop or zero-update run cannot pass the reload check.
The launcher estimates a persistent floor of about **10.42GiB** (pinned serialized
base plus conservative trainable/Adam allowance), excluding dynamic KV,
activations, generation score tensors and workspace. HF keeps one completion's
scores until extraction, then releases them before the next seed. **Actual 4B
3090/4090 peak memory and runtime are unmeasured**; allocation failures stop without
quantization, offload or automatic image/token reductions. Sequential generation
trades throughput for memory; no speed or cost guarantee is implied.

The native vLLM 48GB gate path remains unchanged. This path does not certify its
G0–G13 gates, model efficacy or a finished large-scale pipeline. No Pod, model
download or GitHub publication is started by these changes. Future smoke weights
must only be removed after evidence recovery and successful verification; original
caches and old checkpoints are never cleanup targets.

## Local full-chain verification (2026-10-18)

From a directory containing only the copied `fine_tuning/` package, using an
environment with the pinned Torch/Transformers/PEFT dependencies already installed:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m unittest fine_tuning.tests.test_notebook_rl_local_pipeline \
  fine_tuning.tests.test_notebook_rl_rollout \
  fine_tuning.tests.test_notebook_rl_portability -v
```

This is a **local software-contract check**, not a pretrained-4B training result:

- A real random-initialized tiny Qwen generates six responses through three turns;
  missing answers remain failures, not fabricated recovery.
- Separate authored branch fixtures exercise two wrong attempts, a recovered third
  attempt, and a final model-note record. Likelihoods and three optimizer updates
  use the actual tiny Qwen image forward and production RL loss, with SFT calls
  explicitly forbidden. The successful fixture texts are not model reasoning.
- Verified wrong image facts enable Vision+four Mergers+LLM. Correct image facts
  disable Vision while Mergers/LLM still change, including after an earlier Vision
  update (no Adam momentum leak). A deliberately wrong model diagnosis cannot
  override verified routing. Balanced diagnosis groups receive their own reward.
- The fixed F0 reference stays unchanged; a new Python process reloads the saved
  checkpoint and must reproduce every parameter hash and optimizer step.
- An isolated folder-copy check denies `mmdl` and `experiments` imports and has no
  sibling `src/`, `env/`, `manifests/` or `prompts/` directories. Pinned resources
  and the unchanged processor preparation contract are bundled inside this package.

`RolloutConfig.final_notebook` (or the explicit
`rollout.final_notebook: true` protocol setting) adds **one audit-only generation
per recovered problem**, even for `ALG-NB3` or a class excluded from DIAG training.
It preserves the successful path's exact token history. The final audit note is
not a loss group, teacher label, reward or routing authority. Existing class-balanced
DIAG learning remains separate. Budget/context stops, empty notes, truncation and
no recovered path remain explicit. The old protocol file is unchanged and opt-in;
enabling this extra call requires a newly bound protocol and budget, not reuse of
old approvals. Raw trees retain the note even when the collector rejects a
uniform-reward problem for lack of an advantage signal.

For private, write-once detailed evidence, set `MMDL_LOCAL_CONTRACT_EVIDENCE` to a
new absolute directory before the command. Data, responses and checkpoints are
not uploaded. This check does not run cloud GPUs, download models, approve a
training campaign, or demonstrate MMMU improvement. The proposed multi-domain
training pool and native Qwen4B/vLLM verification are still separate next steps.

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

The original full BF16/vLLM campaign and general training-pool approval remain
incomplete. The separate bounded local NF4 run above has completed one real 4B RL
update and fresh reload, but not a demonstrated performance improvement.
Existing V8 results and all files under `MMDL/` remain untouched. Only
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

The separately authorized `AI_PROGRAM_REVIEWED_SMOKE_ONLY` path accepts only a
bounded program-labelled pool (at most 64 total / 24 train items), with every
original input independently AI reviewed and program verified
(`human_verified=false`). It authorizes only the existing two-update gate sequence,
not Phase A or a training campaign. Ordinary measurement/training retain their
human-review requirements. Full pinned MMMU/MMMU-Pro exclusion, actual
OCR/SSCD/SigLIP caches and calibration, exact-SHA checkout, owned financial watchdog
and native GPU gates remain required. The separately authorized direct overlap
review preserves every original warning and binds each verdict to the complete
scan, exact input/index identities and reviewed evidence files. `DISTINCT` may
release a similarity warning; `DUPLICATE` and `UNRESOLVED` exclude the item and its
shared structure/image family. Exact benchmark image copies and source/split
violations remain excluded. The limited smoke returns only eligible train IDs to
the actor and refuses an empty eligible set. Ordinary measurement/training retain
their stricter whole-pool selection checks. Authorized hardware/budget does not establish runtime fitness or
improvement. Generated `graph_pair` plain-mode questions still request observations;
they are not a pure answer-only control.

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
| Full benchmark exclusion | The earlier 40-item graph smoke pool has a complete pinned scan and AI warning review. This is not clearance of the proposed multi-domain pool, human certification, or a reusable approval after code/protocol changes. |
| Human render/leak review | D3/D7 require real reviewers. No generated or AI-reviewed sample is labelled human-approved. |
| Actual GPU gates | G0–G13 evidence is validated by the approval worker; the complete GPU evidence-producing runner and its execution are not complete. |
| Actual Phase A | The candidate/retry measurement core is implemented; complete provenance-bound assessment, held-out baselines, cap freezing and actual F0 GPU execution remain unfinished. Its partial measurement format is deliberately rejected by selection. |
| Paid native-vLLM deployment | No completed native-vLLM RL/reload or 150-question improvement result; the separate local NF4 update1/reload is documented above. |

Thus the current launcher is an integrated **guarded entry point**, not a claim that
the whole research pipeline can already be reproduced end-to-end with one command.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH="$PWD/src:$PWD" python -m unittest discover \
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

The old fingerprint-only 100-file/16,740-row exclusion index alone is incomplete
for the v2 protocol. The later private graph-smoke evidence adds pinned coverage,
per-image provenance, OCR, SSCD/SigLIP and calibration; new inputs must supply their
own bound evidence. Partial scans return `BLOCKED_NOT_CHECKED`, never a clean
training allowlist. Python file/import guards add defense in depth; they are not an
OS sandbox against native code or another process with the same user permissions.

The user's $2.30 budget prioritizes actual bounded RL updates and fresh checkpoint
reload. A full pilot or 150-question evaluation is not promised within that budget.
The design's 48GB two-engine configuration must not silently be shrunk to fit 24GB.
The previous owned A40 was deleted after preserving its pre-update failures;
its native BF16 RL/reload remains unverified. The separate local NF4 update/reload
above does not grant paid-GPU or general-training authority. Human review remains
separate from AI smoke review.
The outer watchdog must interrupt in-flight GPU work; loop-level time checks alone
cannot enforce a wall-clock limit. Saving or stopping a process does not delete a Pod.
