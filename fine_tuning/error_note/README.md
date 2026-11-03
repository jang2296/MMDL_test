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

v1/v2의 기본 동작은 유지한다. v3는 `training-answer-v1`, `rows-version v3`, `protocol-version v3`를 명시해서 실행한다.
생성은 계속 8192토큰·기존 일반 이미지 범위다. 클라우드는 원본 BF16, 로컬 NF4는 기능 확인용이다.

1. `source_evidence prepare --build-root V --manifest MANIFEST --order ORDER --limit N --raw-root RAW --output EVIDENCE`
   로 기존 승인 자료의 private `rules.json`, `source-provenance.json`을 만든다. 파일·문항·이미지·출처 해시를 검사하며
   출처 설명을 모델의 입력이나 학습 target에 넣지 않는다. 원래 adapter의 수치 허용오차는 출처 저자 주석과 구분한다.
2. `collect solve`/`sweep`, `collect note --note-version v2`에
   `--answer-contract training-answer-v1 --training-rules RULES --training-rules-sha256 SHA`를 붙인다.
   원본 모델 파일·실제 코드·엔진·정밀도·환경·순서가 `collection.json`에 고정되므로 다른 조건으로 결과를 이어 쓰지 못한다.
3. 생성된 노트를 원문/이미지/출처와 검토하고 private `note-reviews-v1` 원장을 만든다.
   `audit.review_bindings`가 원문과 근거를 결합한다. `ai_review`는 반드시 `human_verified=false`다.
   전체 노트·전체 교훈을 확인하지 못하면 UNKNOWN/UNCERTAIN으로 남긴다. 형식 통과나 모순 미검출로 PASS를 만들지 않는다.
4. `collect audit`에 공통 build/order/limit/output·채점 옵션과 `--source-provenance SOURCE --reviews REVIEWS`를 준다.
   `audit.json`과 `lessons-v3.json`을 생성한다. 같은 위치에서 감사 결과를 바꿔 덮어쓰지 않는다.
5. `collect resolve`는 같은 새 채점 옵션과 `--source-provenance SOURCE --audit AUDIT`로 실행한다.
   별도 검증 프로세스가 원본 노트·감사·교훈의 해시와 검토 범위를 재검사한다. 모델 프로세스에는 허용 교훈만 전달한다.
   노트 내용 PASS와 교훈 CLEAR를 모두 만족하는 오답만 한 번 다시 푼다.
6. `collect rows --rows-version v3`에 같은 채점·근거·감사 옵션을 준다. A는 정답인 원문 풀이, B는 전체 검토된 원문 노트다.
   미검증 노트, 누설 교훈, 실패 재풀이는 사유와 함께 제외한다. 문장을 고쳐 넣지 않는다.
7. `train admit`로 encoding을 CPU에서 확인하고 공통 A가 있는 16문항 그룹·탈락 사유·총 update 수를 고정한다.
   필수 인자는 기존 rows/build/order/limit/model/precision/code-commit에 `--audit AUDIT --sidecar RULES --output ADMISSION`이다.
   로컬 NF4만 `--max-row-tokens`를 미리 지정할 수 있고 그 제외 내역도 고정된다. BF16에서는 이 제한을 허용하지 않는다.
8. `train train --protocol-version v3 --arm note|no-note --admission ADMISSION --audit AUDIT --sidecar RULES`로 학습한다.
   두 arm은 같은 A·경로·순서·초기값·step/LR을 쓰고, NOTE는 `0.9 mean(A NLL)+0.1 mean(B NLL)`,
   NO-NOTE는 `0.9 mean(A NLL)`이다. B가 없다고 A 계수를 다시 정규화하지 않는다. 90:10은 손실 계수이지 실제 gradient 비율이 아니다.
   고정 뒤 encoding 결과가 달라지면 중단한다. 각각 별도 프로세스 `reload --protocol-version v3`를 거친다.

RunPod 계획은 `plan --stage note-v3-collect|note-v3-train --params PARAMS --output STEPS`로 나눈다.
두 단계 사이에서 실제 노트 검토가 필요하며 미확정 내용을 자동 승인하지 않는다. `plan.py`의 v3 params 명세가 정본이다.
동시 생성은 기존 KV 메모리 기반 `auto`, 학습은 기존 token-budget microbatch를 사용한다. 공통 admission을 한 번 만들어 두 arm이 공유한다.
300문항 결과를 로컬에 회수·검토한 뒤 본학습으로 진행한다. 코드·CPU 통과, 실제 가중치 갱신, 노트 정확성, 성능 상승은 각각 따로 보고한다.

검사: `python -m unittest discover -s fine_tuning/tests -p 'test_error_note*.py'`와 공유 채점/Pod/체크포인트 호출자의 회귀 검사를 실행한다.
실제 학습/평가 결과는 별도 실행 receipt가 필요하다. 상위 모델 풀이·벤치마크 학습·다중 답 선택은 계속 금지한다.

가중치를 바꾸지 않는 로컬 기능 확인은 같은 `admit` 결과에 `train dry-run`을 사용한다.
필수 인자는 `--rows --build-root --order --limit --model-path --output --precision nf4 --code-commit --audit --sidecar --admission`이다.
`--limit`은 사전 고정한 첫 1~10문항이며, `--max-row-tokens`를 썼다면 admission과 같아야 한다.
실제 A/B 손실과 각각의 기울기를 계산하고 모두 비운 뒤, 전체 파라미터와 동결 NF4 상태가 그대로인지 검사한다.
optimizer/scheduler를 만들거나 step을 실행하지 않고 가중치도 저장하지 않는다. `dry_run.json`의 PASS는
손실·기울기 연결과 가중치 미변경을 뜻한다. 실제 학습·재로딩·정확도 향상 증거가 아니며, A나 검토된 B가 없으면 통과하지 않는다.

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

## 격리 원칙

- 첫 풀이 프로세스는 정답 파일을 열 수 없다(프로세스 안 감사 훅, 운영체제 격리는 아님). 채점은 별도 프로세스의 채점 서비스가 한다.
- 노트 프로세스는 시작할 때 정답을 읽고 같은 훅을 건다. 학습 프로세스는 수집 파일만 읽는다.
- 모든 학습 단계에서 MMMU·MMMU-Pro 경로와 상위 모델 SDK를 막는다.

## 보고할 때

- 노트의 이미지 오독은 **자기 신고**다. 실제 시각 오류로 쓰지 않는다.
- 틀렸던 342문항은 학습 없이 다시 풀어도 일부 맞는다. 미학습 재실행과 같은 문항·seed로만 비교한다.
- 300문항 통과는 돈을 더 쓸지 정하는 문턱이지 개선 입증이 아니다.
