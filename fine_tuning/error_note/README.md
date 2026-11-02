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

## 격리

- 첫 풀이 프로세스는 정답 파일을 열 수 없다(프로세스 안 감사 훅, 운영체제 격리는 아님). 채점은 별도 프로세스의 채점 서비스가 한다.
- 노트 프로세스는 시작할 때 정답을 읽고 같은 훅을 건다. 학습 프로세스는 수집 파일만 읽는다.
- 모든 학습 단계에서 MMMU·MMMU-Pro 경로와 상위 모델 SDK를 막는다.

## 보고할 때

- 노트의 이미지 오독은 **자기 신고**다. 실제 시각 오류로 쓰지 않는다.
- 틀렸던 342문항은 학습 없이 다시 풀어도 일부 맞는다. 미학습 재실행과 같은 문항·seed로만 비교한다.
- 300문항 통과는 돈을 더 쓸지 정하는 문턱이지 개선 입증이 아니다.
