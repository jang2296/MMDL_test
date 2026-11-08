# 오답노트형 자기개선 파인튜닝 (`fine_tuning/error_note/`)

계획 정본: `claudedocs/research_self_note_finetuning_plan_20261101.md`. 규칙: `CLAUDE.md` §2.1, §3.0.

문항을 한 번 풀고(정답 없이), V8로 채점하고, 모델이 정답을 보고 오답노트와 고쳐 쓴 풀이를 쓴 뒤, 풀이와 노트를 다음 토큰 지도학습한다.
노트가 이미지 오독이라 하면 Vision 조정층 + Merger + LLM 조정층, 그 외에는 Merger + LLM 조정층을 갱신한다. 원본 가중치는 고정이다.
상위 모델 풀이, 튜터 모델, MMMU·MMMU-Pro 학습, 여러 답 생성 후 선택은 금지다.

## 파일

| 파일 | 내용 |
|---|---|
| `contract.py` | 고정값, 노트 프롬프트, 노트 파싱, 학습 행 생성과 거르기. 그래픽카드 불필요 |
| `order.py`, `item_order.json` | 결과를 보기 전에 고정한 3,417문항 순서. 앞 300개가 첫 시험 |
| `item_order_v2.json` | 2판 순서: 같은 3,417문항을 출처가 고르게 섞이도록 재배열(문항 ID·출처만 사용) |
| `protocol.py` | 학습 설정 (조정층, 학습률, 정밀도) |
| `collect.py` | 풀이·노트 생성(`solve`, `note`, `sweep`), 2판 다시 풀기(`resolve`), 학습 행 만들기(`rows`) |
| `train.py` | 학습(`train`), 새 프로세스 재로딩 검사(`reload`) |
| `plan.py` | 대여기 단계 계획 작성 |

재사용: `notebook_rl`의 `policy`, `checkpoint`, `export`, `input_preparation`, `prompts`, `answers`, `v8_parser`, `verifier_service`, `vault`, `dataset`, `guards`, `pod/`.
평가는 `notebook_rl/proof_eval.py`와 `pod/eval_stage.py`로 `src/mmdl` 평가 코드를 그대로 호출한다. 그래서 평가만은 저장소 전체가 필요하다.

## 고정값

P0 프롬프트, 이미지 262144–1310720 픽셀(`qwen_vl_utils 0.0.14`), 최대 8192 토큰, temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, seed `SHA256("3407:{문항ID}")`, 채점 `team-final-answer-v8`.

## 실행

```bash
# 로컬 4비트: 작동 확인 전용. 성능 근거로 쓰지 않는다
python -m fine_tuning.error_note.collect solve --build-root V --order fine_tuning/error_note/item_order.json --limit 2 --model-path SNAP --output OUT --engine hf
python -m fine_tuning.error_note.collect note  --build-root V --order fine_tuning/error_note/item_order.json --limit 2 --model-path SNAP --output OUT --engine hf
python -m fine_tuning.error_note.collect rows  --build-root V --order fine_tuning/error_note/item_order.json --limit 2 --tokenizer-path SNAP --output OUT
python -m fine_tuning.error_note.train train --rows OUT/rows.jsonl --build-root V --order fine_tuning/error_note/item_order.json --limit 2 --arm note --precision nf4 --model-path SNAP --output RUN --code-commit HEX
python -m fine_tuning.error_note.train reload --run RUN --model-path SNAP --precision nf4

# 런포드 16비트: 단계 계획을 만들어 pod/start.sh 작업으로 실행한다 (pod/RUNBOOK.md 맨 앞 절)
python -m fine_tuning.error_note.plan plan --stage note-300 --params PARAMS.json --output STEPS.json
```

## 버전 2 (2026-11-02, `claudedocs/plan_self_note_v2_20261102.md`)

1판 명령과 출력은 그대로다. 2판은 아래 인자로만 고른다. 이미지 값은 1판과 같은 262144–1310720, 최대 8192 토큰.

- 노트 v2(`note --note-version v2`): 고쳐 쓴 풀이 없음. 제목은 What the image shows / Image reading / Cause / What went wrong / Lesson.
  교훈 누설 검사(`contract.hint_leak`, 거짓 양성 쪽으로 보수적)는 정답을 아는 노트 단계에서 하고, 통과한 교훈만 `lessons.json`에 쓴다.
- 다시 풀기(`resolve`): T1이 정답이 아닌 문항만 1회(재시도 없음). 입력은 교훈 블록 + P0뿐이다. 정답·T1·채점 결과 없음.
  T1과 같은 훅으로 정답 파일을 막고, 정답이 든 `note.jsonl`은 열지 않는다(`lessons.json`과 그 크기만 확인).
- 행 v2(`rows --rows-version v2`): A = 맞은 T1(노트가 MISREAD면 제외) 또는 맞은 다시 풀기(입력은 힌트 없는 P0), B = 노트.
  정답을 보고 쓴 풀이 행은 없다. 끝 3,000자 압축률 12% 미만은 반복으로 제외. `rows_summary.json`에 다시 풀기 성공률, 누설 제외 수,
  단계별 8192 도달 글의 반복/비반복 수(`length_finishes`)를 쓴다. 한도는 자동으로 늘리지 않는다.
- 순서 `item_order_v2.json`(1판과 같은 3,417문항, 출처 안 순서 유지, 출처를 학습 전체에 고르게 섞음),
  학습 `--protocol-version v2`(갱신당 16문항, warmup 10회 뒤 마지막 갱신 다음에 0이 되도록 선형 감소; LoRA·LR·clip은 1판과 같음).

```bash
O=fine_tuning/error_note/item_order_v2.json   # order.py spread 로 만든 고정 파일
python -m fine_tuning.error_note.collect solve   --build-root V --order $O --limit 2 --model-path SNAP --output OUT --engine hf
python -m fine_tuning.error_note.collect note    --build-root V --order $O --limit 2 --model-path SNAP --output OUT --engine hf --note-version v2
python -m fine_tuning.error_note.collect resolve --build-root V --order $O --limit 2 --model-path SNAP --output OUT --engine hf
python -m fine_tuning.error_note.collect rows    --build-root V --order $O --limit 2 --tokenizer-path SNAP --output OUT --rows-version v2
python -m fine_tuning.error_note.train train --rows OUT/rows.jsonl --build-root V --order $O --limit 2 --arm note --precision nf4 --model-path SNAP --output RUN --code-commit HEX --protocol-version v2
python -m fine_tuning.error_note.train reload --run RUN --model-path SNAP --precision nf4 --protocol-version v2

# 런포드: params 형식 error-note-plan-params-v2. 1판 T1(t1.jsonl)을 OUT/collect에 복사해 쓰면 reuse_t1_sha256에 그 sha256을 넣는다
# (sweep 생략, note·resolve는 --max-num-seqs auto = KV 크기에서 고른 가운데 후보). null이면 sweep으로 T1부터 푼다.
python -m fine_tuning.error_note.plan plan --stage note-v2 --params PARAMS_V2.json --output STEPS.json

# 평가: 기존 변형 mmmu-val-v8-8k-seed3407-v2, 사전 등록 v4(F0, FT-V2), wrong + rest = 900문항 1회
python -m fine_tuning.notebook_rl.proof_eval preregister --format v4 --output PREREG.json --subset WRONG.json --rest-subset REST.json
python -m fine_tuning.notebook_rl.proof_eval gate-62 --prereg PREREG.json --ledger LEDGER.jsonl --runs RUN_WRONG RUN_REST --output GATE.json [--f0-reference N]
```

- `gate-62`: FT-V2의 900문항 V8 정답 수가 558을 넘으면 PASS. 1회 실행이라 표준오차가 약 1.6점이므로 1~2점 차이는 우연과 구분되지 않고, 개선 입증이 아니다.
  같은 원장에 F0가 없으면 "not run in this ledger"로 쓰고, `--f0-reference`(2판 이전 원장의 F0 값)는 표시만 한다.
- 평가 단계 계획(`pod/eval_stage.py`)에서 FT-V2는 seed 3407만, `vllm_inprocess: true`(VLLM_ENABLE_V1_MULTIPROCESSING=0)가 필수다.
  같은 원장의 F0도 같은 값을 써야 한다(등록 파일이 달라 `analyze`가 거부한다).

## 버전 3: 출처 검토·학습 채점·노트 대조

v1/v2의 기본 동작은 유지한다. 현재 v3는 `training-answer-v2`, `rows-version v3`, `protocol-version v3`를 명시해서 실행한다.
학습 채점 v2는 같은 값·같은 단위의 반복 답 선언을 한 답으로 처리한다. 상충·철회·불확실한 답은 거부하며 V8/MMMU 채점은 바꾸지 않는다.
기존 `training-answer-v1` 결과는 당시 동결 소스로만 재현한다. 새 소스에서 v1을 v2로 바꾸어 해석하거나 원래 판정을 덮어쓰지 않는다.
생성은 계속 8192토큰·기존 일반 이미지 범위다. 클라우드는 원본 BF16, 로컬 NF4는 기능 확인용이다.

1. `source_evidence prepare --build-root V --manifest MANIFEST --order ORDER --limit N --raw-root RAW --output EVIDENCE`
   로 기존 승인 자료의 private `rules.json`, `source-provenance.json`을 만든다. 파일·문항·이미지·출처 해시를 검사하며
   출처 설명을 모델의 입력이나 학습 target에 넣지 않는다. 원래 adapter의 수치 허용오차는 출처 저자 주석과 구분한다.
2. `collect solve`/`sweep`, `collect note --note-version v2`에
   `--answer-contract training-answer-v2 --training-rules RULES --training-rules-sha256 SHA`를 붙인다.
   v2 제목 형식은 재사용하되 지시문은 v3다. 실제 틀린 주장의 교정·근거·불확실성을 쓰고, 이미지 관측과 외부 지식을 구분한다.
   원본 모델 파일·실제 코드·엔진·정밀도·환경·순서가 `collection.json`에 고정되므로 다른 조건으로 결과를 이어 쓰지 못한다.
3. 생성된 노트를 원문/이미지/출처와 검토하고 private `note-reviews-v1` 원장을 만든다.
   `audit.review_bindings`가 원문과 근거를 결합한다. `ai_review`는 반드시 `human_verified=false`다.
   전체 노트·전체 교훈을 확인하지 못하면 UNKNOWN/UNCERTAIN으로 남긴다. 형식 통과나 모순 미검출로 PASS를 만들지 않는다.
4. `collect audit`에 공통 build/order/limit/output·채점 옵션과 `--source-provenance SOURCE --reviews REVIEWS`를 준다.
   `audit.json`과 `lessons-v3.json`을 생성한다. 같은 위치에서 감사 결과를 바꿔 덮어쓰지 않는다.
5. `collect resolve`는 같은 새 채점 옵션과 `--source-provenance SOURCE --audit AUDIT`로 실행한다.
   별도 검증 프로세스가 원본 노트·감사·교훈의 해시와 검토 범위를 재검사한다. 모델 프로세스에는 허용 교훈만 전달한다.
   노트 내용 PASS와 교훈 CLEAR를 모두 만족하는 오답만 한 번 다시 푼다.
6. `collect rows --rows-version v3`에 같은 채점·근거·감사 옵션을 준다. A는 정답인 원문 풀이 **후보**, B는 전체 검토된 원문 노트다.
   미검증 노트, 누설 교훈, 실패 재풀이는 사유와 함께 제외한다. 문장을 고쳐 넣지 않는다.
7. 실제로 생성된 각 A 후보의 **전체 풀이**를 이미지/원출처와 별도로 대조하여 private `solution-audit-file-v1`을 만든다.
   `audit.solution_review_bindings(row, note_audit)`로 행·목표·입력·출처 해시를 결합하고, 파일에는 후보 rows와 note audit 파일 해시를 고정한다.
   항목은 `solution-review-v1`, 해당 row_id/problem_id/target_source, reviewer_kind/human_verified/method,
   `content={result: PASS|FAIL|UNKNOWN, scope: entire_solution|partial, evidence: [...]}`를 기록한다. AI 검토는 `human_verified=false`다.
   최종 정답만 맞거나 B가 통과한 것으로 A의 풀이 검토를 대신하지 않는다. 첫 풀이와 재풀이 모두 별도 검토 대상이다.
8. `train admit`로 검토된 A만 encoding 전에 수용하고 공통 A가 있는 16문항 그룹·탈락 사유·총 update 수를 고정한다.
   필수 인자는 기존 rows/build/order/limit/model/precision/code-commit에 `--audit AUDIT --sidecar RULES --solution-audit SOLUTIONS --output ADMISSION`이다.
   FAIL/UNKNOWN/검토 누락/부분 검토 A는 탈락하며, 오래되거나 변조된 결합은 오류다. 후보 원문은 그대로 보존한다.
   로컬 NF4만 `--max-row-tokens`를 미리 지정할 수 있고 그 제외 내역도 고정된다. BF16에서는 이 제한을 허용하지 않는다.
9. `train train --protocol-version v3 --arm note|no-note --admission ADMISSION --audit AUDIT --sidecar RULES --solution-audit SOLUTIONS`로 학습한다.
   두 arm은 같은 A·경로·순서·초기값·step/LR을 쓰고, NOTE는 `0.9 mean(A NLL)+0.1 mean(B NLL)`,
   NO-NOTE는 `0.9 mean(A NLL)`이다. B가 없다고 A 계수를 다시 정규화하지 않는다. 90:10은 손실 계수이지 실제 gradient 비율이 아니다.
   고정 뒤 encoding 결과가 달라지면 중단한다. 각각 별도 프로세스 `reload --protocol-version v3`를 거친다.

실행 계획은 `plan --stage note-v3-collect|note-v3-resolve|note-v3-train --params PARAMS --output STEPS`의 세 단계다.
collect 뒤에는 노트를 검토하고, resolve 뒤에는 완성된 풀이 후보를 검토한다. train은 이미 닫힌 rows/audit/solution-audit 경로와 해시를 받아 재생성 없이 학습한다.
이전 동결 소스의 후보도 해당 원문·감사에 묶인 풀이 검토로 수용할 수 있다. 새 소스의 collector로 기존 수집 결과를 재개하지 않는다.
미확정 내용을 자동 승인하지 않으며 `plan.py`의 v3 params 명세가 정본이다. 클라우드 실행 승인은 별도다.
동시 생성은 기존 KV 메모리 기반 `auto`, 학습은 기존 token-budget microbatch를 사용한다. 공통 admission을 한 번 만들어 두 arm이 공유한다.
300문항 결과를 로컬에 회수·검토한 뒤 본학습으로 진행한다. 코드·CPU 통과, 실제 가중치 갱신, 노트 정확성, 성능 상승은 각각 따로 보고한다.

검사: `python -m unittest discover -s fine_tuning/tests -p 'test_error_note*.py'`와 공유 채점/Pod/체크포인트 호출자의 회귀 검사를 실행한다.
실제 학습/평가 결과는 별도 실행 receipt가 필요하다. 상위 모델 풀이·벤치마크 학습·다중 답 선택은 계속 금지한다.

가중치를 바꾸지 않는 로컬 기능 확인은 같은 `admit` 결과에 `train dry-run`을 사용한다.
필수 인자는 `--rows --build-root --order --limit --model-path --output --precision nf4 --code-commit --audit --sidecar --solution-audit --admission`이다.
`--limit`은 사전 고정한 첫 1~10문항이며, `--max-row-tokens`를 썼다면 admission과 같아야 한다.
실제 A/B 손실과 각각의 기울기를 계산하고 모두 비운 뒤, 전체 파라미터와 동결 NF4 상태가 그대로인지 검사한다.
optimizer/scheduler를 만들거나 step을 실행하지 않고 가중치도 저장하지 않는다. `dry_run.json`의 PASS는
손실·기울기 연결과 가중치 미변경을 뜻한다. 실제 학습·재로딩·정확도 향상 증거가 아니며, A나 검토된 B가 없으면 통과하지 않는다.
각 손실에서 LLM·Merger 기울기가 유한한 양수인지, Vision 기울기가 행의 학습 경로와 일치하는지 따로 검사한다.
전체 오답 교정 사이클은 추가로 `paired_resolve_note_problem_ids`의 같은 문항에 대해 원본 T1 오답 → 검토 통과 노트 → 정답 없는 재풀이 정답을 대조해야 한다. 첫 풀이 정답 행만으로는 이 사이클을 확인했다고 하지 않는다.

외부 dev 비교는 `source_evidence prepare-dev --build-root V --manifest MANIFEST --output EVIDENCE`로
별도 dev 채점 규칙을 준비한 뒤 `evaluate preregister`로 고정한다. 학습용 train 규칙을 dev에 재사용하지 않는다.
등록에는 `--manifest`, `--vault-root`, `--train-order`, `--train-rows`, `--train-limit`, `--models`,
`--dev-rules`, `--code-commit`, `--run-root`, `--output`을 지정한다. 세 모델 F0/NOTE/NO-NOTE는 각각 하나다.
`evaluate run --prereg P --arm ARM --output ROOT/ARM`은 미리 정한 경로에서 정답 없이 한 번 생성한다.
생성 프로세스 종료 후 별도 `evaluate grade --prereg P --run ROOT/ARM`으로 원문을 보존하며 채점한다.
`evaluate analyze --prereg P --f0 ROOT/F0 --note ROOT/NOTE --no-note ROOT/NO-NOTE --output REPORT`는
가족을 묶어 재표집한 문항 정확도 차이 구간을 두 대조에 보고한다. 구간이 0을 포함하면 HOLD이며 자동 확대하지 않는다.

최종 MMMU는 `proof_eval preregister --format v5 --ft-v3-model-ref REF`로 BF16 NOTE export를 먼저 고정한다.
`FT-NOTE-V3`는 seed 3407·8K·P0·V8로 원래 900개 ID를 한 번씩 평가하며 `gate-62`/`gate-67.4`의
최소 정답 수는 각각 559/607이다. 실패 실행을 새 실행으로 대체하지 않는다. 실제 export 파일과 원본 checkpoint의
연결을 확인한 뒤 평가하며, 과거 F0는 참조값이고 같은 등록의 F0만 짝 비교에 사용한다.

v3 Pod의 `out_root`는 해당 job `run_dir` 안에 둔다. `pod.collect pack`은 기록된 job/spec 해시와 실제 완료한
chain·필수 산출물을 확인하고 수집/학습/체크포인트를 함께 회수한다. 밖에 저장된 merged export는 별도로 회수해야 한다.
실패 결과는 실패 상태로 보존하고, 로컬 무결성 확인 뒤에만 이 작업이 만든 Pod와 저장소를 삭제한다.

## 버전 4: RL + 오답노트 SFT (`claudedocs/plan_rl_sft_hybrid_v4_20261104.md`)

v1–v3 명령·출력·protocol hash는 그대로다. v4는 별도 모듈(`v4.py`, `rl.py`, `collect_v4.py`, `train_v4.py`, `plan_v4.py`)로만 실행한다.
한 라운드는 순서 파일의 `--start`/`--limit` 구간이다. 1라운드는 F0에서, 다음 라운드는 이전 라운드의 `checkpoint-final`(`--init-run`)과 그 export 모델로 rollout한다.

1. `collect_v4 rollout`: 정답 없이 P0로 j=0,1을 풀고, 둘 다 맞지 않은 문항만 j=2,3을 더 푼다(엔진 1회, `rollouts.jsonl`).
   `--reuse-t1`(1라운드만)은 F0 첫 풀이를 j=0으로 쓰고 현재 계약으로 다시 채점한다. `--loop-stop`은 같은 64토큰 블록이 8회 반복되는 학습 표본을
   멈추고 보상 0(`LOOP_STOPPED`)으로 둔다. 평가에는 쓰지 않으며 8192 상한은 그대로다. `--prefix-caching`은 vLLM prefix caching이다.
2. `collect_v4 note`: 전부 틀린 그룹만 정답을 본 v2 노트를 쓴다. 같은 프로세스에서 프로그램 게이트(형식, MISREAD+NONE 모순, 보기 문자·정답 문구·
   정답 보기 문장, `--leak-terms`의 별칭·약어, 압축 반복)를 적용해 `gates-v4.json`을 쓴다. 이 파일에는 통과한 교훈과 확인 가능한 이미지 주장만 있고
   정답 문자열이나 실패한 교훈은 없다. **사람·AI 내용 검토를 학습 수용 조건으로 쓰지 않는다.**
3. `collect_v4 selfcheck`: 정답 없이 이미지와 주장 한 문장만 보고 TRUE/FALSE/CANNOT_TELL로 2번 답한다. 하나라도 FALSE면 자기모순으로 노트를 쓰지 않는다.
   같은 모델의 재확인이라 정확성 증명이 아니다(`reviewer_kind=program+self`).
4. `collect_v4 resolve`: 교훈 + P0로 2회 다시 푼다. `--lesson-control`은 같은 seed로 교훈 없이 P0만 2회 더 풀어 교훈 효과를 비교한다(같은 파일, `lesson` 필드).
5. `collect_v4 rows`: `v4.build_rows_v4`. MIXED 그룹 → RL 행(보상 − 그룹 평균, 잘리거나 멈춘 표본은 끝 토큰 없음),
   ALL_CORRECT → 유지용 A1 1행, ALL_WRONG → 정답 재풀이 A2(입력은 교훈 없는 P0) + 노트 B(자기검증 일치 또는 A2가 있을 때).
6. `train_v4 train`: 16문항당 1회 갱신. 손실 = RL 항(그룹 행동 토큰 정규화, old = 현재 정책 detach, F0 기준 k3 KL 0.02)
   + 0.5·A2 + 0.05·B + 0.05·A1(행 평균 NLL). 없는 종류는 0이며 다시 정규화하지 않는다. `--arm rl-only`는 같은 RL·A1 행에서 A2·B만 뺀 대조군이다.
   `train_v4 dry-run`(≤10문항)은 종류별 역전파·기울기 경로·가중치 불변만 확인한다. `train_v4 reload`는 새 프로세스 재로딩 검사다.
7. `plan_v4 plan --params P --output STEPS`: 위 단계와 reload·export를 담은 라운드 하나의 Pod chain spec(`error-note-plan-params-v4`).

```bash
C="--round 1 --build-root V --order O --start 0 --limit 500 --output OUT --training-rules RULES --training-rules-sha256 SHA"
E="--model-path SNAP --engine vllm --kv-cache-gib auto --max-num-seqs auto --prefix-caching --loop-stop"
python -m fine_tuning.error_note.collect_v4 rollout $C $E [--reuse-t1 T1.jsonl --reuse-t1-sha256 H]
python -m fine_tuning.error_note.collect_v4 note $C $E
python -m fine_tuning.error_note.collect_v4 selfcheck $C $E
python -m fine_tuning.error_note.collect_v4 resolve $C $E --lesson-control
python -m fine_tuning.error_note.collect_v4 rows $C --tokenizer-path SNAP
python -m fine_tuning.error_note.train_v4 train --rows OUT/rows.jsonl --build-root V --order O --round 1 --start 0 --limit 500 \
  --arm hybrid --precision bf16 --model-path SNAP --output RUN --code-commit HEX --token-budget 24000 --resume
python -m fine_tuning.error_note.train_v4 reload --run RUN --model-path SNAP --precision bf16
```

라운드와 대조군: 1라운드는 F0로 한 번만 수집(`collect-r1`)하고 두 arm이 같은 행으로 학습한다(rl-only는 `shared_rows_sha256`로 그 행을 고정).
2라운드부터는 arm마다 자기 이전 export로 rollout한다(`collect-r<k>-<arm>`). rl-only는 노트·자기검증·재풀이를 생성하지 않는다.
rl-only에서 노트 행만 있던 그룹은 가중치를 바꾸지 않고 스케줄러만 한 칸 진행한다(`scheduler_only`). 그래서 두 arm의 학습률과 총 step 수가 같다.

평가 연결:
- `proof_eval export`는 v4 최종 체크포인트(step = total_steps)만 받고, model_ref에 `training_protocol`을 기록한다.
- 외부 dev(`evaluate preregister`)는 NOTE = hybrid, NO-NOTE = rl-only 마지막 라운드 export를 받는다. 두 arm은 rows·order·optimizer·라운드가 같아야 하며, init만 다를 수 있다.
- 최종 MMMU는 `proof_eval preregister --format v6 --ft-v4-model-ref REF`로 FT-HYB-V4 하나를 고정하고, `verify-v4-export` 후 seed 3407·8K·V8로 900문항을 한 번 평가한다.
  `gate-62`/`gate-67.4`의 최소 정답 수는 559/607이다. `pod.eval_stage`의 `eval-ft`가 FT-HYB-V4를 v3와 같은 규칙으로 계획한다.

CPU 검사: `python -m unittest discover -s fine_tuning/tests -p 'test_error_note_v4*.py'`(작은 무작위 모델·가짜 엔진. 4B 학습 효과의 증거가 아니다).

## SINGLE: 실제 메모리 측정과 서로 다른 문항 묶음

현재 상위 SINGLE 코드는 **`error-note-single-response-v2`**다. 이전 v1 실행은 당시 동결 코드·원장으로만 해석하며
새 코드에서 조용히 재개하거나 reward를 소급 변경하지 않는다.
T1의 답이 CORRECT이고 **원래 풀이 전체**의 출처 검토가 PASS인 경우에만 PG 보상 1이다.
같은 A1 검토 receipt를 재사용하며 FAIL/UNKNOWN/승인 누락은 0이다. 기존 길이·반복 조건으로 A1이 부적격이면
검토를 우회하지 않고 이유를 남겨 0으로 둔다. 정오·원래 action/EOS·q는 고치지 않는다.
모든 완료 T1은 고정 B×C 분모와 전체 raw KL에 남고, A2 승인으로 틀린 원래 T1의 보상이 바뀌지 않는다.
receipt·재계산 결정·출처/evidence hash는 controller metadata이며 processor 입력이나 target에 넣지 않는다.
인코딩/역전파에서 결정을 다시 검증하므로 reward/accepted 값만 바꾸거나 낡은 receipt를 재사용할 수 없다.
검토 WAIT는 유지하고 malformed/stale binding은 오류로 차단한다.
SINGLE 노트의 parser-invalid 피드백은 “parser가 승인하지 않았다”이며 “최종답이 없었다는 뜻은 아니다”라고 명시한다.
원래 풀이·gold·v3 노트 지시문/스키마는 그대로 주고, 기존 v1–v3 프롬프트 함수/판정은 바꾸지 않는다.
새 protocol의 prompt hash는 실제 SINGLE 피드백 맵과 지시문을 함께 고정한다.

`single`의 기본값은 HF 동시 1문항, 생성 토큰 예산 24576, 학습 singleton이며 그대로 유지된다.
`--precision nf4|bf16`은 명시 선택이다. 아래 기능은 GPU 측정을 자동 시작하거나 OOM 뒤 재생성하지 않는다.
동일 정책 블록은 최대 16개의 **서로 다른** 문항이며 문항별 seed·원래 action·EOS·sampling replay를 유지한다.

승인된 새 동결 소스에서 다음처럼 **원래 수집 작업 자체**를 측정할 수 있다. `RESERVE`는 운영자가 사전에 정한 양의 byte 값이다.
공통 인자·수집 결과 디렉터리는 재개 때 같게 유지하고, partial 파일명만 매번 새것으로 준다.
이 예시는 사전 고정 2문항 경로이며 측정값이나 8192토큰 적합성의 증거가 아니다.

```bash
C=(--build-root "$BUILD" --order "$ORDER" --order-sha256 "$ORDER_SHA"
   --frozen-approval "$APPROVAL" --source-provenance "$SOURCE" --training-rules "$RULES"
   --model-path "$SNAP" --mode local-smoke --precision nf4 --limit 2 --token-scale 256
   --max-num-seqs 1 --max-num-batched-tokens 24576 --token-budget 49152)
python -m fine_tuning.error_note.single collect "${C[@]}" --output "$OUT" \
  --capture-capacity "$PART/collect-1.json" --capacity-reserve-bytes "$RESERVE"
# WAIT_NOTE_REVIEW 뒤 원문/출처 검토를 기록한 다음 재개한다. 완료된 답은 다시 생성하지 않는다.
python -m fine_tuning.error_note.single collect "${C[@]}" --output "$OUT" --note-reviews "$NOTE_REVIEWS" \
  --capture-capacity "$PART/collect-2.json" --capacity-reserve-bytes "$RESERVE"
# WAIT_SOLUTION_REVIEW 뒤 검토를 기록한다. dry-run은 update 없이 원래 행의 backward도 측정한다.
python -m fine_tuning.error_note.single dry-run "${C[@]}" --output "$OUT" \
  --note-reviews "$NOTE_REVIEWS" --solution-reviews "$SOLUTION_REVIEWS" \
  --capture-capacity "$PART/backward.json" --capacity-reserve-bytes "$RESERVE"
# 각 실행이 출력한 파일 SHA를 외부 원장에 고정한다. 다음 명령은 CPU 전용이다.
python -m fine_tuning.error_note.capacity assemble \
  --input "$PART/collect-1.json" --input-sha256 "$PART1_SHA" \
  --input "$PART/collect-2.json" --input-sha256 "$PART2_SHA" \
  --input "$PART/backward.json" --input-sha256 "$BACKWARD_SHA" --output "$PROFILE"
```

수집은 실제 생성 길이만 기록한다. 짧게 끝난 응답으로 쓰지 않은 8192토큰 상한을 인증하지 않는다.
학습 측정은 기존 loss/backward의 실제 microbatch 분할을 따른다. `--token-budget 0`은 singleton이며,
양의 예산은 기존 planner로 분할하고 singleton도 예산을 넘으면 HOLD한다. 각 항목은 실제 실행한 묶음의 크기와 ID를 기록한다.
원래 전체 T1의 B×C 분모와 종류별 전체 SFT 행 수·가중치를 유지하며, microbatch 사이의 기울기를 누적한다.
정책/원본 가중치는 바꾸지 않고 transaction마다 Adam moment와 전체 gradient의 상주 공간을 한 번 확보한다.
각 peak는 앞선 replay·microbatch도 포함할 수 있는 보수적인 transaction 누적 최고값이며, 해당 묶음만의 독립 비용이 아니다.
optimizer step의 임시 공간은 측정 범위 밖이다.

정상 종료·검토 WAIT·backward 증명 HOLD에서도 완료된 측정은 새 partial에 보존한다.
backward 증명 보고서는 선택적 측정 전에 저장한다. 증명 오류·파라미터 변경이면 추가 측정을 하지 않으며,
측정 실패는 완료 항목을 보존하되 `UNCALIBRATED_FAILED`로 남아 합치거나 실행 입장에 사용할 수 없다.
partial의 외부 SHA, 소스/모델/정밀도/실제 GPU UUID/라이브러리/attention backend/전체 tensor dtype·layout/
초기 정책/reserve가 맞아야 합칠 수 있고, 같은 work를 중복 합쳐 증거를 늘릴 수 없다.
실제 generation, full-vocabulary KL, 보상 1 PG, A1/A2/B, LM/FULL 경로가 모두 없으면 `UNCALIBRATED`(종료 코드 2)다.
2문항에서 A2/B가 없으면 최종 profile이 될 수 없다. 추가 증거는 별도로 사전 고정한 **다른** 문항에서만 얻는다.
partial/최종 파일은 덮어쓰지 않으며 동시에 같은 파일을 발행하는 writer도 차단한다.

완성된 profile은 **별도 실행 계획**에 `--capacity-profile "$PROFILE" --capacity-profile-sha256 "$PROFILE_SHA"`로 고정한다.
이 모드에서 생략한 동시성 상한은 16, 생성/학습 토큰 예산 상한은 각각 16×24576이다.
기존 FIFO/token planner가 매 단계 실제 행 길이·전체 image grid와 현재 여유 VRAM으로 입장 전 분할한다.
미측정 shape/긴 singleton/부족한 메모리는 STARTED 전에 HOLD하며, STARTED 뒤 OOM·오류는 보존하고 재시도하지 않는다.
학습 API는 `backward_single_block(..., capacity=session)`을 사용하며 optimizer 실행기는 추가하지 않았다.
BF16 loader 선택은 지원하지만 이 경로의 cloud controller·cloud-ready 인증은 제공하지 않는다.
이 구현의 CPU 회귀는 실행했지만 실제 GPU calibration profile은 아직 만들지 않았다.

## SINGLE campaign: 300문항 파일럿과 나머지 3,200문항 연결

`single_campaign.py`는 위 메모리 측정 경로와 별도인 캠페인 실행기다. `prepare`는 모델을 불러오지 않고
**실제 승인·동결된 정확한 3,500문항**의 순서, 전체 actor/이미지/출처, 원래 정답·허용오차·단위 근거를 검증해
계획을 고정한다. 기존 풀의 2/10문항 확인으로 이 검사를 대신할 수 없다. 아래 변수는 아직 확보되지 않은
실물 입력을 포함한 자리표시자이며, `SNAP`은 NF4 cache가 아닌 원본 BF16 snapshot 경로다.

```bash
C=(--build-root "$BUILD" --order "$ORDER" --order-sha256 "$ORDER_SHA"
   --frozen-approval "$APPROVAL" --frozen-approval-sha256 "$APPROVAL_SHA"
   --source-provenance "$SOURCE" --source-provenance-sha256 "$SOURCE_SHA"
   --training-rules "$RULES" --training-rules-sha256 "$RULES_SHA"
   --note-reference-supplement "$SUPP" --note-reference-supplement-sha256 "$SUPP_SHA"
   --pilot-ids-sha256 "$PILOT_IDS_SHA" --dev-ids-sha256 "$DEV_IDS_SHA"
   --retention-ids-sha256 "$RETENTION_IDS_SHA" --model-path "$SNAP")
P=(--output "$CAMPAIGN_PROOF" --campaign-mode proof --precision nf4 --max-blocks 1)
python -m fine_tuning.error_note.single_campaign prepare "${C[@]}" "${P[@]}"
# 별도로 승인된 실제 모델 실행에서만 사용한다. 기본 proof는 optimizer/가중치 저장이 없다.
python -m fine_tuning.error_note.single_campaign advance "${C[@]}" "${P[@]}" \
  --allow-model-run --stop-after-seconds "$STAGE_SECONDS"
```

명령 생략 시 `prepare`, 모드 기본값은 `proof`, `--max-blocks` 기본값은 1이다. 블록 하나는 서로 다른 최대16문항이다.
실제 실행에는 `--allow-model-run`이 필요하며, 고정한 모드·입력·메모리 설정·실행 블록 상한을 같은 캠페인에서 바꿀 수 없다.
완성된 메모리 profile은 `prepare`부터 `--capacity-profile "$PROFILE" --capacity-profile-sha256 "$PROFILE_SHA"`로 고정한다.
`WAIT_NOTE_REVIEW`/`WAIT_SOLUTION_REVIEW`에서는 실행을 끝내고, 각각 `--note-reviews`/`--solution-reviews`와 대응하는
`--note-reviews-sha256`/`--solution-reviews-sha256`를 붙여 같은 `advance`를 재호출한다. 완료 응답은 재생성하지 않는다.
불확실한 STARTED 요청은 재추첨하지 않으며, 부분 checkpoint는 `RECOVERY_REQUIRED`로 보존한다.

다음은 **향후 별도 갱신·저장 승인을 받은 BF16 실행용**이다. proof 캠페인을 train으로 바꿀 수 없으므로 새 출력 경로가 필요하다.

```bash
T=(--output "$CAMPAIGN_BF16" --campaign-mode train --precision bf16 --max-blocks 1)
python -m fine_tuning.error_note.single_campaign prepare "${C[@]}" "${T[@]}"
python -m fine_tuning.error_note.single_campaign advance "${C[@]}" "${T[@]}" \
  --allow-model-run --allow-update --allow-save --stop-after-seconds "$STAGE_SECONDS"
# 저장된 정책·optimizer·LR·RNG·cursor를 복원/검사하며, 다음 블록을 실행하지 않는다.
python -m fine_tuning.error_note.single_campaign reload "${C[@]}" "${T[@]}" --allow-model-run
# 파일럿/외부 dev/예산 gate와 본학습 승인이 모두 있는 경우에만 다음 구간을 실행한다.
python -m fine_tuning.error_note.single_campaign advance "${C[@]}" "${T[@]}" \
  --allow-model-run --allow-update --allow-save --continue-main \
  --pilot-gate "$PILOT_GATE" --pilot-gate-sha256 "$PILOT_GATE_SHA" --stop-after-seconds "$STAGE_SECONDS"
```

300문항은 18×16+12인19번째 슬롯에서 `PILOT_COMPLETE`로 멈춘다. BF16 본학습은 계획 정본
`claudedocs/review_rl_sft_hybrid_v4_20261105.md` §11.2의 기능/외부 dev/예산 조건, 같은 파일럿 checkpoint에 결합한
gate, `--continue-main`을 요구하며 다음301번째 문항부터3,200개만 이어간다. optimizer·LR·RNG를 초기화하지 않고 총219슬롯이다.
proof의 추가 무갱신 순회에도 `--continue-main`이 필요하며, 순회 완료와 PG/A1/A2/B 필수 기울기 증명은 별도다. 누락 성분은 HOLD로 남는다.

현재 연결 검증은 **작은 무작위 CPU 모델의57검사 통과**(CUDA 초기화 차단)다. 실제3,500문항 승인, 실제 Qwen GPU 실행,
메모리 적합성·학습 효과를 인증한 결과가 아니다. NF4 train/checkpoint continuation은 명시적으로 HOLD이며 로컬 NF4는 무갱신 확인만 지원한다.
이 CLI는 RunPod를 자동 생성·실행하지 않으며 기존 Pod의 SINGLE 미지원 거부를 완화하지 않는다.

## SINGLE v11 (2026-11-08, `claudedocs/plan_v11_20261108.md`)

- 보상 게이트 v6, 기준선 없음. `reward`는 부호(+1/0/−1), `advantage`는 표: 정답 수용 +1, 맨답 오답·답 미확정·반복 −1,
  풀이를 쓴 오답(REASONED_WRONG)·반복 없는 잘림(TRUNCATED_NO_LOOP) −0.5, 그 밖의 정답·검증 불가 0(정답은 음수 금지).
- PG = −A·min(1, 2048/n)·Σℓ/(16·1024). 양의 행 ℓ는 저장 후보 분포 log q, 음의 행 ℓ는 원래 분포 log p(T=1, 전체 어휘).
  11-05 계획의 'q만 사용'·'길이와 무관한 고정 분모' 원칙의 명시적 예외(protocol `plan_exceptions`).
- 노트 입력에만 데이터셋 참고 정보(`references.py`): ScienceQA 강의·풀이, CLEVR 장면·질문 프로그램, SLAKE 영상 종류·부위,
  ChEBI 설명·SMILES(번들 `source-provenance.json`), PTB-XL 판독문·진단 문장(보충 파일, sha 고정). SSMR은 원자료에 ABC가
  없어 참고 정보 0. 4,000자 상한(필드 단위 제외). MMMU/MMMU-Pro 13-gram 중복 검사 0건(제외 신호로만 사용).
- 캠페인 필수 입력 `--note-reference-supplement`(+sha256), plan params `note_reference_supplement_path/_sha256`.
- 그대로: 첫 4토큰 F0 고정, KL, 재생 검사·행 제외, SFT 가중(A1 .05, A2 .35 행당, B .05), T1·노트·재풀이 각 1회.

## 격리 원칙

- 첫 풀이 프로세스는 정답 파일을 열 수 없다(프로세스 안 감사 훅, 운영체제 격리는 아님). 채점은 별도 프로세스의 채점 서비스가 한다.
- 노트 프로세스는 시작할 때 정답을 읽고 같은 훅을 건다. 학습 프로세스는 수집 파일만 읽는다.
- 모든 학습 단계에서 MMMU·MMMU-Pro 경로와 상위 모델 SDK를 막는다.

## 보고할 때

- 노트의 이미지 오독은 **자기 신고**다. 실제 시각 오류로 쓰지 않는다.
- 틀렸던 342문항은 학습 없이 다시 풀어도 일부 맞는다. 미학습 재실행과 같은 문항·seed로만 비교한다.
- 300문항 통과는 돈을 더 쓸지 정하는 문턱이지 개선 입증이 아니다.
