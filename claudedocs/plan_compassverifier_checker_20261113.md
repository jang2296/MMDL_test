# V8 보완 판정 도구 설계 — `CompassVerifier_checker`

- 작성: 2026-11-13. 상태: **설계만 완료, 코드 미구현**. 구현은 사용자 명령을 받은 뒤 시작한다.
- 대상 독자: 이 문서만 보고 구현할 작업자(Opus 5.5, GPT-6.1 sol, GPT-6 astra 등)와 검토자.
- 이 문서는 MMDL_test에 게시될 수 있다. 벤치마크 문제 원문·모델 응답 원문은 넣지 않았고, 구현 중에도 넣지 않는다.

## 0. 한눈에 보기

| 항목 | 결정 |
|---|---|
| 목적 | V8 파서는 그대로 두고, V8이 정답으로 인정하지 못한 일부 문항만 오픈소스 판정 모델로 다시 확인해 **점수 두 개**(V8, V8+판정)를 낸다 |
| 판정 모델 | `opencompass/CompassVerifier-7B` (사용자 결정 2026-11-13) |
| 방향 | **V8 오답 → 정답만**. V8이 맞다고 한 문항은 절대 바꾸지 않는다 (사용자 결정) |
| 쓰는 방식 | 글을 생성하지 않는다. 첫 출력 위치에서 `A`/`B`/`C` 세 글자의 확률만 읽는다(판정 전용 모델처럼 사용) |
| 코드 위치 | `src/mmdl/evaluation/CompassVerifier_checker/` 한 폴더 (V8 파서 파일 옆) |
| 폴더 원칙 | 자립형. 폴더째 MMDL로 옮기거나 폴더째 지울 수 있어야 한다 (사용자 결정) |
| MMDL | 지금은 넣지 않는다. `scripts/sync_submission.py`에 제외 규칙을 추가한다 |
| MMDL_test | 게시 대상이다. `scripts/sync_test_repo.py`는 고치지 않아도 이 폴더를 포함한다 |
| 로컬 시험 | CompassVerifier-3B와 4비트 양자화 모델로 **코드가 도는지만** 확인한다. 로컬 결과는 점수로 보고하지 않는다 |
| 공식 판정 | 7B·16비트(BF16)·고정 리비전. 24GB 이상 GPU가 필요해 런포드에서 돌린다. **사용자 명령 후에만** |
| 입력 | 이미 저장된 `predictions.jsonl`. 새 생성 없음 |
| 과제 점수 | V8 558/900 그대로. V8+판정은 참고 점수다 |

## 1. 배경

### 1.1 원본 Qwen 평가의 AI 채점

고정한 Qwen 코드(커밋 `96588727e44c78b25ba03ea03b8e12f7e64fd0da`, `evaluation/mmmu/run_mmmu.py`, `eval_utils.py`)는 세 단계로 채점한다.

1. 규칙: 보기 글자가 하나만 나오거나 보기 문장 하나만 응답에 들어 있으면 그것을 답으로 본다.
2. AI 판정: 규칙이 실패하면 `gpt-3.5-turbo-0125`에게 문제·보기·응답을 주고 보기 글자 하나를 받는다. 최대 25번 다시 묻는다.
3. 무작위: 그래도 안 되면 보기 중 하나를 무작위로 고른다.

주관식은 정답을 보기 A, "Other Answers"를 보기 B로 바꿔 같은 흐름을 탄다.

우리 V8은 1단계를 정교하게 만든 것이고 3단계는 일부러 뺐다. 이 도구는 2단계를 오픈소스 모델로 보완한다. 3단계(무작위)는 만들지 않는다.

### 1.2 판정 대상 규모 (저장 결과에서 실측, 2026-11-13)

대상은 "정상 종료(`finish_reason == "stop"`)했고 V8이 오답 처리한 문항" 중 아래 두 종류뿐이다.

- `NO_PARSE_STOP`: V8이 최종 답을 찾지 못함(`parse_status == "NO_PARSE"`).
- `OPEN_MISMATCH_STOP`: 주관식이고 V8이 답을 찾았지만 정답과 다르다고 판정함.

| 실행 | 문항 | V8 정답 | 길이 한도로 잘림 | NO_PARSE_STOP (객관식/주관식) | OPEN_MISMATCH_STOP | 대상 합계 | 이론 상한 | 대상 응답 최대 길이(토큰) |
|---|---|---|---|---|---|---|---|---|
| B2 원본 (32K) | 900 | 558 | 90 | 7 / 14 | 18 | 39 | 597 | 13,433 |
| F0 8K | 900 | 541 | 111 | 13 / 7 | 24 | 44 | 585 | 7,613 |
| v11b 8K | 900 | 552 | 70 | 7 / 10 | 23 | 40 | 592 | 6,577 |
| v11c 8K | 900 | 552 | 67 | 10 / 14 | 20 | 44 | 596 | 8,155 |
| v12 8K | 900 | 522 | 97 | 11 / 9 | 28 | 48 | 570 | 7,425 |
| v13 B2 | 900 | 555 | 27 | 9 / 16 | 27 | 52 | 607 | 19,151 |

- "이론 상한"은 대상이 전부 정답으로 바뀔 때의 값이다. 실제 증가는 돌려 봐야 알며, 클 것으로 기대할 근거는 없다.
- 길이 한도로 잘린 문항은 대상이 아니다. 끝까지 쓰지 못한 응답이므로 오답으로 둔다.
- 객관식인데 V8이 답을 찾았고 그 답이 틀린 문항도 대상이 아니다. 모델이 다른 보기를 명시했기 때문이다.

## 2. 반드시 지킬 규칙

1. **V8 파일을 고치지 않는다.** 아래 네 파일은 SHA256이 고정돼 있고 다른 코드가 이 해시에 의존한다.

   | 파일 | SHA256 |
   |---|---|
   | `src/mmdl/evaluation/final_answer_parser.py` | `fc2c6a757ce7361efae13be9f22f8bedcc9c35d66373124b8815b8528b2e31a0` |
   | `src/mmdl/evaluation/final_answer_parser_v8.py` | `2beb1230283cb882f47d6e8af72b01180847fd7aa4b7ac640a5173e391f3f891` |
   | `src/mmdl/evaluation/parsers.py` | `1bfdabe251d5cbf14b6554f2c979492ec18207de5fb826a0045dd2d93ca3e221` |
   | `third_party/mmmu/eval_utils.py` | `cc1a89b4a27f697702d340bc6a7578b5237135754ddaa9ef105239a3d9847367` |

   `fine_tuning/notebook_rl/v8_parser.py`의 `SOURCE_SHA256`과 V8 채점 해시(`results/scoring-v8-20260924/summary.json`의 `scoring_sha256`)가 이 파일들에 묶여 있다.
2. **`src/mmdl/` 아래 기존 파일은 하나도 고치지 않는다.** `cli.py`, `writer.py`도 포함이다. 이 파일들은 MMDL 제출본과 같아야 한다.
3. **`MMDL/` 폴더와 `jang2296/MMDL` 저장소를 건드리지 않는다.** `python -m scripts.sync_submission --sync`를 실행하지 않는다(§9.1 참고).
4. **방향은 한쪽뿐이다.** 문항별로 `최종 정답 >= V8 정답`이어야 하고, "정답→오답" 전환 수는 항상 0이어야 한다.
5. **잘린 응답과 빈 응답은 판정 모델에 보내지 않는다.** 무작위 고르기도 없다.
6. **판정 모델은 평가 전용이다.** 학습 보상·학습 행 선별에 쓰지 않는다. `fine_tuning/`의 학습 코드에서 import하지 않는다.
7. **외부 서비스로 보내지 않는다.** 판정은 로컬 장비나 우리가 만든 포드에서만 한다. 판정 중에는 `HF_HUB_OFFLINE=1`로 둔다.
8. **벤치마크 원문이 든 산출물은 저장소에 두지 않는다.** `targets.jsonl`, `verdicts.jsonl`, `merged.jsonl`, `review.md`는 `$MMDL_ARTIFACT_ROOT` 아래에만 쓴다.
9. **설정을 결과를 본 뒤 바꾸지 않는다.** 기준값·질문문·모델 리비전·응답 자르기 길이를 바꾸면 새 버전(`...-v2`)이다.
10. **보고할 때 섞지 않는다.** V8 점수와 V8+판정 점수를 항상 나란히 쓰고, 로컬 시험 결과(3B·양자화)는 점수로 쓰지 않는다. 과제 점수와 v14 사전 등록 판정(앞 절반 281)은 V8 기준 그대로다.
11. commit·push는 사용자가 요청할 때만 한다. 런포드 실행도 사용자 명령 후에만 한다.

## 3. 지금 있는 코드와 자료의 위치

### 3.1 저장소 세 개

| 경로 | 역할 | 이 작업에서 |
|---|---|---|
| `/mnt/c/projects/multimodel` | 원본 작업공간 (브랜치 `feat/mmmu-baseline`) | **여기서 구현한다** |
| `/mnt/c/projects/multimodel/MMDL` | 교수님 제출용 checkout (`jang2296/MMDL`) | 건드리지 않는다 |
| `/mnt/c/projects/multimodel/MMDL_test` | 실험 코드 게시용 checkout (`jang2296/MMDL_test`, 브랜치 `feat/workspace-repro`) | 사용자 요청 시 `scripts/sync_test_repo.py`로만 복사한다 |

### 3.2 V8 채점 코드 (읽기만 한다)

| 파일 | 내용 | 이 도구가 쓰는 것 |
|---|---|---|
| `src/mmdl/evaluation/final_answer_parser_v8.py` | V8 본체. `VERSION = "team-final-answer-v8"`, `score_response(raw, question_type, options, answer, finish_reason, *, question)` | 직접 import하지 않는다 |
| `src/mmdl/evaluation/parsers.py` | `score_with_parser(raw, question_type, options, answer, parser_id, finish_reason, *, question)`, `scoring_source_files(parser_id)`, 상태 상수 `EMPTY`/`NO_PARSE`/`PARSED` | `score_with_parser`, `scoring_source_files` |
| `src/mmdl/evaluation/cli.py` | `rescore_main()` — 저장 응답 재채점 명령. 구조 참고용 | 없음 |
| `src/mmdl/runtime/artifacts.py` | `sha256_file(path)`, `digest(value)`, `write_json(path, value)`, `read_json(path)`, `atomic_text(path, text)`, `default_path(variable, suffix)` | 전부 재사용 |
| `fine_tuning/notebook_rl/proof_eval.py` 1286~1299행 | 저장 행의 무결성 검사와 V8 재채점 대조 방식. 같은 방식을 쓴다 | 참고만. import 금지 |

`score_with_parser(..., "team-final-answer-v8", ...)`의 반환값은 다음 키를 가진 dict다: `parsed_answer`, `parse_status`, `correct`, `extraction_rule`, `extraction_evidence`, `extraction_status`.

### 3.3 입력 파일 형식 — `predictions.jsonl`

한 줄이 한 문항이다. 평가 엔진(`src/mmdl/evaluation/writer.py`)이 만든 형식이고, 원본 B2와 미세조정 모델 평가가 모두 같은 형식이다. 이 도구가 읽는 필드는 다음뿐이다.

| 필드 | 형 | 설명 |
|---|---|---|
| `id` | str | 문항 ID. 실행 안에서 유일 |
| `subject` | str | 과목 (30개) |
| `question_type` | str | `"multiple-choice"` 또는 `"open"` |
| `prompt` | str | 모델이 실제로 본 P0 본문. 900개 모두 `"Question:"`로 시작하고, 객관식은 보기와 선택 지시문이 들어 있다 |
| `question` | str | 문제 문장만. V8 재채점에 넘긴다 |
| `options` | list[str] | 보기 내용만(글자 없음). 글자는 순서대로 A, B, C… |
| `answer` | str | 정답. 객관식은 글자 하나, 주관식은 문자열. list인 경우도 받아들여야 한다(V8과 같은 "하나라도 같으면 정답" 의미) |
| `raw_response` | str | 모델 응답 원문 |
| `finish_reason` | str | `"stop"` 또는 `"length"` |
| `parse_status`, `parsed_answer`, `correct`, `extraction_rule` | | 저장된 V8 판정 |
| `generated_tokens` | int | 응답 토큰 수 |
| `_record_sha256` | str | `digest(이 필드를 뺀 나머지)`와 같아야 한다 |

한 실행이 파일 두 개로 나뉜 경우가 있다(8K 평가는 342문항 + 558문항). 여러 파일을 받아 합치고 ID 중복이 없는지 확인해야 한다.

### 3.4 저장된 결과 위치

`A = /home/jang2/mmdl-artifacts`, `R = final/extract/mmdl-volume/eval-data/proof/artifacts/runs`로 줄여 쓴다.

| 실행 | `predictions.jsonl` 위치 |
|---|---|
| B2 원본 | `/mnt/c/projects/multimodel/analysis_exports/mmmu_20260924/team_final_v8/official_b2/` |
| F0 8K | `A/notebook-rl-local-nf4-20261021/hf-pool-v3/pod-note3417/final/x/mmdl-volume/eval-data/proof/artifacts/runs/note300r-f0-8k-s3407-wrong-evaluation/` 와 `.../note3417-f0-8k-s3407-rest-evaluation/` |
| v11b 8K | `A/self-note-single-20261105/pod-v11b-c6oh363fiko7tc/R/v11-ft-single-8k-s3407-{wrong,rest}-evaluation/` |
| v11c 8K | `A/self-note-single-20261105/pod-v11c-lsthaknal3tff1/R/v11c-ft-single-8k-s3407-{wrong,rest}-evaluation/` |
| v12 8K | `A/self-note-single-20261105/pod-v12-lhi0arls1o0lbl/R/v12-ft-single-8k-s3407-{wrong,rest}-evaluation/` |
| v13 B2 | `A/self-note-single-20261105/pod-v13-v3detjqj9cmiyi/R/v13-ft-single-b2-s3407-all-evaluation/` |
| v14 | 회수되면 `A/self-note-single-20261105/pod-v14-e3odtf0md92kr1/final/` 아래. 앞 절반만 있으면 450문항이다 |

이 파일들은 읽기만 한다. 수정·이동·삭제하지 않는다.

### 3.5 판정 모델 정보 (2026-11-13 조회)

| 항목 | CompassVerifier-7B (공식 판정용) | CompassVerifier-3B (로컬 시험용) |
|---|---|---|
| Hugging Face ID | `opencompass/CompassVerifier-7B` | `opencompass/CompassVerifier-3B` |
| 리비전 | `8fee654b23ec971ad4a5c6d69bd99cf0f0188197` | `27d6d660bdf6c2b8a9f45906d4b76c508af65ab0` |
| 구조 | `Qwen2ForCausalLM` (기반 `Qwen/Qwen2.5-7B-Instruct`) | `Qwen2ForCausalLM` |
| 크기 | 약 15.2GB | 약 6.8GB |
| 라이선스(모델 카드 메타데이터) | apache-2.0 | apache-2.0 |

- 질문문 출처: GitHub `open-compass/CompassVerifier`, 커밋 `2d7cba6df0b21f9c6121786ac1e5770c68473598`, 파일 `src/prompts.py`(파일 SHA256 `6a4bde9a47fc670ab7a5d93b9c7e1d5aa9bf13c00fedf05cc2a87ac03811411a`), 상수 `CV_PROMPT`.
- `CV_PROMPT`의 자리표시자는 `{question}`, `{gold_answer}`, `{llm_response}` 세 개다. 출력 글자의 뜻은 `A` = CORRECT, `B` = INCORRECT, `C` = INVALID(잘림·반복·답 없음 등)다.
- 공식 사용법: 사용자 메시지 하나에 채운 질문문을 넣고 `tokenizer.apply_chat_template(..., add_generation_prompt=True)`를 적용한다. 시스템 메시지를 따로 넣지도 빼지도 않는다(토크나이저 기본 템플릿 그대로).
- 풀이를 쓰게 하는 `CV_COT_PROMPT`는 쓰지 않는다.
- 확인할 것: 모델 카드 메타데이터는 apache-2.0인데 카드 본문 배지는 CC BY-SA 4.0이고, GitHub 저장소에는 LICENSE 파일이 잡히지 않는다. 구현 첫 단계에서 다시 확인해 `README.md`에 적는다. 평가에 쓰는 데는 둘 다 문제가 없다.

## 4. 새 폴더 구조

```text
src/mmdl/evaluation/CompassVerifier_checker/
├── __init__.py          버전 상수만. 무거운 import 금지
├── __main__.py          `python -m mmdl.evaluation.CompassVerifier_checker` 진입점 (cli.main 호출)
├── README.md            사용법, 규칙 요약, 라이선스 확인 결과 (한국어)
├── checker_v1.yaml      고정 설정 (§5)
├── prompt.py            CV_PROMPT 원문 사본과 출처·해시, 질문문 채우기
├── selection.py         입력 검증, 대상 선정, 판정 입력 만들기   (torch 없음)
├── backend.py           판정 모델 호출. torch를 쓰는 유일한 파일. 시험용 FakeJudge 포함
├── scoring.py           판정 규칙, V8 결과와 병합, 요약         (torch 없음)
├── cli.py               select / judge / report / run / checkset 명령
├── checkset_v1.yaml     직접 만든 점검 사례 (벤치마크 문항 금지)
└── tests/
    ├── __init__.py
    ├── test_selection.py
    ├── test_scoring.py
    ├── test_cli.py
    └── test_isolation.py
```

폴더 밖에서 바꾸는 파일은 두 개뿐이다.

| 파일 | 변경 |
|---|---|
| `scripts/sync_submission.py` | 이 폴더를 MMDL 복사 대상에서 빼는 규칙 추가 (§9.1) |
| `tests/unit/test_sync_submission.py` | 위 규칙의 검사 1개 추가 (이 파일은 원래 MMDL에 복사되지 않는다) |

자립형 원칙을 지키기 위한 조건은 다음과 같다.

- 이 폴더의 코드는 `mmdl.evaluation.parsers`, `mmdl.runtime.artifacts`, 표준 라이브러리, PyYAML, (backend.py만) torch·transformers만 import한다. `fine_tuning`, `scripts`를 import하지 않는다.
- 폴더 밖의 어떤 코드도 이 폴더를 import하지 않는다(위 두 파일은 경로 문자열만 쓴다).
- 설정·질문문·점검 사례·시험이 모두 폴더 안에 있다. 설정 파일은 `Path(__file__).with_name("checker_v1.yaml")`로 찾는다.
- 폴더 이름 `CompassVerifier_checker`는 사용자가 정한 이름이다. 파이썬 관례(소문자)와 다르지만 바꾸지 않는다.

## 5. 고정 설정 — `checker_v1.yaml`

```yaml
checker_version: compassverifier-checker-v1
scoring_id: team-final-answer-v8+compassverifier-checker-v1
base_parser: team-final-answer-v8
direction: v8_wrong_to_correct_only

base_parser_files:            # 실행 시작 때 SHA256을 대조한다 (§2 규칙 1의 표와 같은 값)
  src/mmdl/evaluation/final_answer_parser.py: fc2c6a757ce7361efae13be9f22f8bedcc9c35d66373124b8815b8528b2e31a0
  src/mmdl/evaluation/final_answer_parser_v8.py: 2beb1230283cb882f47d6e8af72b01180847fd7aa4b7ac640a5173e391f3f891
  src/mmdl/evaluation/parsers.py: 1bfdabe251d5cbf14b6554f2c979492ec18207de5fb826a0045dd2d93ca3e221
  third_party/mmmu/eval_utils.py: cc1a89b4a27f697702d340bc6a7578b5237135754ddaa9ef105239a3d9847367

judge:
  official:
    model: opencompass/CompassVerifier-7B
    revision: 8fee654b23ec971ad4a5c6d69bd99cf0f0188197
    dtype: bfloat16
    quantization: none
  smoke_allowed:              # 기술적 허용 목록. 실행 승인이 아님; 로컬은 3B만(2026-11-14 정정)
    - {model: opencompass/CompassVerifier-3B, revision: 27d6d660bdf6c2b8a9f45906d4b76c508af65ab0}
    - {model: opencompass/CompassVerifier-7B, revision: 8fee654b23ec971ad4a5c6d69bd99cf0f0188197}
  prompt:
    repo: open-compass/CompassVerifier
    commit: 2d7cba6df0b21f9c6121786ac1e5770c68473598
    path: src/prompts.py
    constant: CV_PROMPT
    text_sha256: <구현 때 CV_PROMPT 문자열의 UTF-8 SHA256을 계산해 적는다>
  labels: {A: CORRECT, B: INCORRECT, C: INVALID}

selection:
  finish_reason: stop
  categories: [NO_PARSE_STOP, OPEN_MISMATCH_STOP]

input:
  question_field: prompt
  mcq_gold_format: "{letter}. {option_text}"
  response_tail_tokens: 3072   # 판정 모델 토크나이저 기준. 넘으면 뒤쪽만 남긴다

decision:
  accept_threshold: 0.90
```

`accept_threshold`와 `response_tail_tokens`는 근거가 약한 초깃값이다(§11). 점검 사례 단계에서 문제가 드러나면 사용자에게 보고하고, 바꿀 때는 v2로 올린다. 실제 MMMU 결과를 본 뒤에는 바꾸지 않는다.

## 6. 처리 흐름

```text
predictions.jsonl (1개 이상)
   │
[select]  CPU ── 무결성 검사 → V8 재채점 대조 → 대상 선정 → 판정 입력 만들기
   │             출력: targets.jsonl, selection.json
[judge]   GPU ── 모델 로드 → 대상마다 순전파 1회 → A/B/C 확률 기록
   │             출력: verdicts.jsonl, judge.json
[report]  CPU ── 판정 규칙 적용 → V8 결과와 병합 → 점수 두 개와 검토 목록
                 출력: merged.jsonl, summary.json, review.md
```

세 단계를 나눈 이유는 GPU가 필요한 단계가 `judge`뿐이기 때문이다. `select`와 `report`는 로컬 CPU에서 돌리고, `judge`만 다른 장비(포드)에서 돌릴 수 있다. `run`은 세 단계를 이어서 실행하는 편의 명령이다.

### 6.1 select — 입력 검증과 대상 선정

1. `checker_v1.yaml`의 `base_parser_files` 해시를 실제 파일과 대조한다. 다르면 중단한다.
2. 모든 행에 대해 `_record_sha256 == digest(그 필드를 뺀 행)`인지 확인한다.
3. 모든 행을 `score_with_parser(raw_response, question_type, options, answer, "team-final-answer-v8", finish_reason, question=question)`로 다시 채점해 저장된 `correct`, `parse_status`, `parsed_answer`와 같은지 확인한다. 하나라도 다르면 중단한다.
4. ID 중복이 없는지 확인한다.
5. 행마다 분류한다. 위에서부터 처음 맞는 것으로 정한다.

   | 순서 | 조건 | 분류 | 판정 모델 |
   |---|---|---|---|
   | 1 | `correct`가 참 | `V8_CORRECT` | 보내지 않음 |
   | 2 | `finish_reason != "stop"` | `LENGTH` | 보내지 않음 |
   | 3 | `parse_status == "EMPTY"` 또는 응답이 공백뿐 | `EMPTY` | 보내지 않음 |
   | 4 | `parse_status == "NO_PARSE"` | `NO_PARSE_STOP` | **보냄** |
   | 5 | 주관식이고 `parse_status == "PARSED"` | `OPEN_MISMATCH_STOP` | **보냄** |
   | 6 | 그 밖(객관식, 답을 찾았고 오답) | `MCQ_EXPLICIT_WRONG` | 보내지 않음 |

6. 대상 행마다 판정 입력을 만든다.
   - `question` 자리: 행의 `prompt` 필드를 그대로 넣는다.
   - `gold_answer` 자리: 객관식은 `"{글자}. {보기 내용}"`(보기 내용은 `options[ord(글자) - 65]`), 주관식은 정답 문자열. `answer`가 list면 원소마다 입력을 하나씩 만든다(`alias_index` 0, 1, …).
   - `llm_response` 자리: `raw_response`. 판정 모델 토크나이저로 `response_tail_tokens`를 넘으면 뒤쪽만 남기고, 잘린 지점 뒤의 첫 줄 시작에서 시작하도록 맞춘다. 잘랐는지 여부를 기록한다. 토크나이저는 torch 없이 `transformers.AutoTokenizer`만 로드한다. 시험에서는 가짜 토크나이저를 주입한다.
7. `targets.jsonl`에 대상 입력을, `selection.json`에 분류별 개수와 식별 정보를 쓴다.

선정에 `correct`(정답 의존)를 쓰는 것은 의도된 설계다. 사용자가 정한 방향이 "V8 오답→정답만"이기 때문이다. 그 대가로 이 점수는 한쪽으로 치우친다(§11).

### 6.2 judge — 확률 읽기

대상 입력마다 다음을 한 번 한다.

1. `prompt.build(question, gold_answer, llm_response)`로 질문문을 채운다.
2. 사용자 메시지 하나로 `apply_chat_template(..., tokenize=False, add_generation_prompt=True)`를 적용하고 토큰화한다.
3. 생성 없이 순전파 1회를 하고, 마지막 위치의 로짓을 float32로 바꿔 전체 어휘에 softmax를 적용한다.
4. `A`, `B`, `C` 토큰의 확률 `p_a`, `p_b`, `p_c`와 1위 토큰 ID·문자열을 기록한다.

로드할 때 `tokenizer.encode("A", add_special_tokens=False)` 등이 각각 토큰 하나인지 확인하고, 아니면 중단한다. 묶음 크기는 1로 둔다(대상이 수십 개뿐이고, 묶음 채움이 확률에 영향을 주지 않게 하기 위해서다).

### 6.3 report — 판정 규칙과 병합

입력 하나의 판정은 다음과 같다(`t`는 `accept_threshold`).

| 조건 | 판정 |
|---|---|
| 1위 토큰이 `A`이고 `p_a >= t` | `ACCEPT` |
| 1위 토큰이 `A`이고 `p_a < t` | `UNSURE` |
| 1위 토큰이 `B` | `REJECT` |
| 1위 토큰이 `C` | `INVALID` |
| 1위 토큰이 그 밖 | `UNSURE` |

문항의 판정은 그 문항의 입력(별칭) 중 하나라도 `ACCEPT`면 `ACCEPT`다. `ACCEPT`가 없으면 `p_a`가 가장 큰 입력의 판정을 따른다.

병합 규칙은 한 줄이다: `final_correct = v8_correct or (판정 == ACCEPT)`. `UNSURE`, `REJECT`, `INVALID`는 모두 오답으로 남는다. `UNSURE`는 사람 검토 목록에만 올린다.

`report`는 `predictions.jsonl`에서 대상을 다시 계산해 `verdicts.jsonl`의 `input_sha256`과 대조한다. 대상이 빠졌거나 남거나 해시가 다르면 중단한다(다른 실행의 판정이 섞이는 것을 막는다).

## 7. 모듈별 명세

아래 시그니처는 고정 인터페이스다. 내부 구현은 작업자가 정한다.

### 7.1 `__init__.py`

```python
CHECKER_VERSION = "compassverifier-checker-v1"
SCORING_ID = "team-final-answer-v8+compassverifier-checker-v1"
```

### 7.2 `prompt.py`

```python
SOURCE = {"repo": "open-compass/CompassVerifier", "commit": "2d7cba6d…", "path": "src/prompts.py", "constant": "CV_PROMPT"}
CV_PROMPT: str            # 원문 그대로 복사. 한 글자도 바꾸지 않는다
CV_PROMPT_SHA256: str     # CV_PROMPT.encode("utf-8")의 SHA256

def build(question: str, gold_answer: str, llm_response: str) -> str: ...
```

- `build`는 `CV_PROMPT.format(question=..., gold_answer=..., llm_response=...)`와 같은 결과를 낸다. 원문에 `{{}}`로 이스케이프된 중괄호가 있으므로 `str.format`을 그대로 쓴다.
- 파일 첫머리에 출처 저장소·커밋·라이선스를 적는다.

### 7.3 `selection.py` (torch import 금지)

```python
CATEGORIES = ("V8_CORRECT", "LENGTH", "EMPTY", "NO_PARSE_STOP", "OPEN_MISMATCH_STOP", "MCQ_EXPLICIT_WRONG")
TARGET_CATEGORIES = ("NO_PARSE_STOP", "OPEN_MISMATCH_STOP")

def load_config(path: Path | None = None) -> dict: ...
def verify_base_parser(config: dict, root: Path) -> None: ...            # 해시 불일치 → ValueError
def load_rows(paths: Sequence[Path]) -> list[dict]: ...                  # 무결성·ID 중복 검사
def replay_v8(rows: Sequence[dict]) -> None: ...                         # 저장 판정과 다르면 ValueError
def categorize(row: Mapping) -> str: ...
def gold_texts(row: Mapping) -> list[str]: ...                           # 객관식 "B. 내용", 주관식 별칭 목록
def tail(text: str, max_tokens: int, encode: Callable[[str], Sequence[int]],
         decode: Callable[[Sequence[int]], str]) -> tuple[str, bool]: ... # (남긴 글, 잘랐는지)
def build_targets(rows: Sequence[dict], config: dict, encode, decode) -> list[dict]: ...
```

`build_targets`가 만드는 대상 한 개의 필드: `id`, `subject`, `question_type`, `category`, `alias_index`, `question`, `gold_answer`, `llm_response`, `response_truncated`, `response_tokens`, `input_sha256`. `input_sha256`은 `digest([CHECKER_VERSION, CV_PROMPT_SHA256, question, gold_answer, llm_response])`다.

### 7.4 `backend.py` (torch·transformers를 함수 안에서 import)

```python
class Judge(Protocol):
    identity: dict   # model, revision, dtype, quantization, label_token_ids, torch/transformers 버전
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...
    def probabilities(self, prompt: str) -> dict: ...   # {"p_a","p_b","p_c","top_token_id","top_token","input_tokens"}

class FakeJudge:                       # 시험용. 미리 정한 확률을 돌려준다. 공백 단위 토크나이저
    def __init__(self, script: Mapping[str, dict] | Callable[[str], dict]): ...

def load_judge(model: str, revision: str, *, quantization: str = "none", device: int = 0) -> Judge: ...
def check_mode(mode: str, model: str, revision: str, quantization: str, config: dict) -> bool: ...
```

- `load_judge`: `quantization`은 `"none"` 또는 `"nf4"`. `"none"`이면 `dtype=torch.bfloat16`, `"nf4"`면 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`. 4비트 설정의 기존 예는 `fine_tuning/notebook_rl/quantized_policy.py` 178행 근처에 있다(참고만, import 금지). `device_map={"": device}`, `local_files_only=True`, `trust_remote_code=False`, `model.eval()`, `torch.inference_mode()`.
- `check_mode`: `mode == "official"`이면 모델·리비전·dtype·양자화가 `judge.official`과 정확히 같아야 하고, 아니면 `ValueError`. `mode == "smoke"`면 `smoke_allowed` 목록에 있어야 한다. 반환값은 `reportable`(official일 때만 참)이다.

### 7.5 `scoring.py` (torch import 금지)

```python
VERDICTS = ("ACCEPT", "UNSURE", "REJECT", "INVALID")

def verdict(probabilities: Mapping, label_token_ids: Mapping[str, int], threshold: float) -> str: ...
def item_verdict(alias_results: Sequence[Mapping]) -> dict: ...
def merge(rows: Sequence[dict], targets: Sequence[dict], verdicts: Sequence[dict], config: dict) -> list[dict]: ...
def summarize(merged: Sequence[dict], identity: Mapping) -> dict: ...
def review_markdown(merged: Sequence[dict], rows: Sequence[dict], verdicts: Sequence[dict]) -> str: ...
```

- `merge`가 만드는 행: `id`, `subject`, `question_type`, `finish_reason`, `category`, `v8_parse_status`, `v8_correct`, `checker_verdict`(대상이 아니면 `null`), `p_a`, `final_correct`.
- `merge`는 다음을 확인하고 어기면 `ValueError`를 낸다: 대상이 아닌 행에 판정이 있음, 대상인데 판정이 없음, `input_sha256` 불일치, `v8_correct`인데 `final_correct`가 거짓.
- `summarize` 결과의 필드는 §8.2와 같다.

### 7.6 `cli.py`

```text
python -m mmdl.evaluation.CompassVerifier_checker select   --predictions P [P ...] --run-label L [--tokenizer PATH] [--output-root DIR]
python -m mmdl.evaluation.CompassVerifier_checker judge    --targets FILE --mode {smoke,official} --model ID --revision SHA [--quantization {none,nf4}] [--limit N]
python -m mmdl.evaluation.CompassVerifier_checker report   --predictions P [P ...] --targets FILE --verdicts FILE
python -m mmdl.evaluation.CompassVerifier_checker run      (select + judge + report 인자를 합친 것)
python -m mmdl.evaluation.CompassVerifier_checker checkset --mode {smoke,official} --model ID --revision SHA [--quantization {none,nf4}]
```

- `--output-root` 기본값: `default_path("MMDL_ARTIFACT_ROOT", "mmdl-artifacts") / "compassverifier_checker"`.
- 출력 폴더: `<output-root>/<run-label>/<mode>-<모델 짧은 이름>-<양자화>-<식별 해시 앞 12자>/`. 이미 있으면 덮어쓰지 않고 중단한다.
- `--limit`은 smoke에서만 허용한다. official에서 쓰면 오류다.
- `--run-label`은 `[A-Za-z0-9][A-Za-z0-9_.-]{0,60}` 형식만 허용한다.
- `report`가 화면에 찍는 마지막 줄 형식(문항이 900개일 때만 백분율을 붙인다. 900이 아니면 개수만 찍는다):

```text
V8           : 558/900 (62.00%)
V8+checker   : 5xx/900 (xx.xx%)   accept=n unsure=n reject=n invalid=n   reportable=true|false
```

## 8. 출력 파일

### 8.1 위치와 내용

모두 `$MMDL_ARTIFACT_ROOT/compassverifier_checker/<run-label>/<...>/` 아래에 쓴다. 저장소 안에는 쓰지 않는다.

| 파일 | 만든 단계 | 내용 | 벤치마크 원문 |
|---|---|---|---|
| `selection.json` | select | 분류별 개수, 입력 파일 경로·SHA256, V8 채점 해시, 설정 해시 | 없음 |
| `targets.jsonl` | select | 판정 입력 | **있음** |
| `judge.json` | judge | 모델·리비전·dtype·양자화·라벨 토큰 ID, 라이브러리 버전, GPU 이름, mode, reportable | 없음 |
| `verdicts.jsonl` | judge | `id`, `alias_index`, `input_sha256`, `p_a`, `p_b`, `p_c`, `top_token_id`, `top_token`, `input_tokens` | 없음 |
| `merged.jsonl` | report | §7.5의 병합 행 | 없음 |
| `summary.json` | report | §8.2 | 없음 |
| `review.md` | report | `ACCEPT`·`UNSURE` 문항 목록: ID, 과목, 분류, 정답, 확률, 응답 끝부분 발췌 | **있음** |

### 8.2 `summary.json` 필드

```text
scoring_id, checker_version, mode, reportable
denominator
v8_correct, final_correct
transitions: {"0->0", "0->1", "1->1", "1->0"}      # "1->0"은 항상 0
categories: 분류별 개수
verdicts: 판정별 개수 (대상 문항 기준)
truncated_inputs: 응답 뒤쪽만 남긴 대상 수
by_subject: 과목별 {n, v8_correct, final_correct}
identity: {source_predictions_sha256 목록, v8_scoring_sha256, config_sha256, prompt_sha256, judge(모델·리비전·dtype·양자화)}
notes: ["one-directional: only V8-wrong items can change", "judge sees the gold answer", ...]
```

## 9. 저장소 간 복사에 미치는 영향

### 9.1 MMDL (지금은 넣지 않는다)

`scripts/sync_submission.py`의 `_DIRS`에는 `("src", lambda p: p.suffix == ".py")`가 있다. 그대로 두면 새 폴더의 `.py` 파일이 전부 MMDL 복사 대상이 된다. 그래서 제외 규칙을 넣는다.

- 상수 `_DEV_ONLY_PREFIXES = ("src/mmdl/evaluation/CompassVerifier_checker/",)`를 추가한다.
- `_selected_files`에서 `relative in _DEV_ONLY` 검사 옆에 `relative.startswith(_DEV_ONLY_PREFIXES)` 검사를 추가한다.
- `tests/unit/test_sync_submission.py`에 검사 1개를 추가한다: 이 폴더 아래 파일을 만들고 `sync(root, write=True)` 후 `MMDL/src/mmdl/evaluation/CompassVerifier_checker`가 없어야 한다.

주의할 현재 상태(2026-11-13 실측): `python -m scripts.sync_submission --check`는 지금도 차이 4건을 보고한다(`.gitignore`, `README.md`, `src/mmdl/evaluation/backends/vllm_backend.py`가 다르고 `tests/unit/test_sync_test_repo.py`가 MMDL에 없다). 이 작업과 무관한 기존 차이다. 이 작업에서 `--sync`를 실행하면 MMDL이 바뀌므로 **실행하지 않는다.** 완료 기준은 "`--check` 출력의 차이 목록이 작업 전과 같다"이다.

### 9.2 MMDL_test (게시 대상)

`scripts/sync_test_repo.py`는 `src/`와 `tests/` 아래의 `.py`, `.md`, `.yaml` 파일을 자동으로 고른다. 스크립트를 고칠 필요가 없다. 지켜야 할 조건은 다음과 같다.

- 폴더 안에 `.json`, `.jsonl`, `.txt` 파일을 두지 않는다. 이 확장자는 게시 대상이 아니라서 MMDL_test에서 시험이 깨진다. 설정과 점검 사례는 `.yaml`, 시험 자료는 시험 `.py` 안에 직접 쓴다.
- 파일 하나가 5MB를 넘지 않게 한다.
- 시험 자료와 점검 사례에 MMMU·MMMU-Pro 문항이나 그 변형을 쓰지 않는다. 직접 만든 사례만 쓴다.
- 게시는 사용자가 요청할 때 `python -m scripts.sync_test_repo --sync` → MMDL_test에서 시험 → commit → push 순서로 한다.

### 9.3 나중에 폴더를 MMDL로 옮길 때 (사용자 명령이 있을 때만)

1. `scripts/sync_submission.py`의 `_DEV_ONLY_PREFIXES`에서 이 폴더를 뺀다.
2. `_DIRS`에 `("src/mmdl/evaluation/CompassVerifier_checker", lambda p: p.suffix in {".yaml", ".md"})`를 추가한다(`.py`는 기존 `src` 규칙이 고른다).
3. §9.1의 기존 차이 4건을 어떻게 할지 사용자에게 먼저 확인한다.
4. 프로젝트 지침 §5.1의 순서(`--sync` → `--check` → MMDL에서 검사 → commit → push)를 따른다.

폴더가 `mmdl.evaluation.parsers`와 `mmdl.runtime.artifacts`만 쓰고 MMDL에도 같은 파일이 있으므로, 폴더를 그대로 복사하면 동작한다.

### 9.4 폴더를 지울 때

폴더를 지우면 끝이다. 다른 코드가 import하지 않으므로 깨지는 것이 없다. `scripts/sync_submission.py`의 제외 규칙과 그 검사는 남겨 둬도 해가 없다. 산출물(`$MMDL_ARTIFACT_ROOT/compassverifier_checker/`)은 따로 지운다.

## 10. 시험

### 10.1 CPU 단위 시험 (모델 없음)

실행: `PYTHONPATH=src python -m unittest discover -s src/mmdl/evaluation/CompassVerifier_checker/tests -t src -v`

| 파일 | 확인할 것 |
|---|---|
| `test_selection.py` | 분류 6종이 표(§6.1)대로 나온다. `_record_sha256` 불일치·ID 중복·V8 재채점 불일치는 `ValueError`. 객관식 정답 글이 `"B. 내용"` 형식이다. `answer`가 list면 입력이 여러 개다. `tail`은 한도를 넘을 때만 자르고 줄 시작에서 시작한다. `base_parser_files` 해시가 실제 파일과 같다 |
| `test_scoring.py` | `p_a`가 0.90이면 `ACCEPT`, 0.8999면 `UNSURE`, 1위가 `B`면 `REJECT`, `C`면 `INVALID`, 그 밖이면 `UNSURE`. 별칭 중 하나만 `ACCEPT`여도 문항은 `ACCEPT`. `merge`의 금지 조건 네 가지가 각각 `ValueError`. 요약의 `"1->0"`이 0이고 합계가 맞다. 900문항이 아니면 백분율이 없다 |
| `test_cli.py` | `FakeJudge`로 `select → judge → report`를 임시 폴더에서 끝까지 돌린다. 출력 폴더가 이미 있으면 거부한다. official에서 `--limit`·다른 모델·다른 리비전·`nf4`를 주면 거부한다. 같은 입력으로 두 번 돌리면 `verdicts.jsonl`과 `summary.json`이 바이트 단위로 같다 |
| `test_isolation.py` | 새 프로세스에서 `selection`, `scoring`, `cli`를 import해도 `torch`가 `sys.modules`에 없다. `CV_PROMPT_SHA256`이 실제 문자열 해시와 같고 자리표시자가 정확히 세 개다. 폴더 밖 `.py` 파일(`src`, `scripts`, `fine_tuning`, `tests` 중 있는 것) 가운데 `CompassVerifier_checker` 문자열을 가진 것은 `scripts/sync_submission.py`와 `tests/unit/test_sync_submission.py`뿐이다 |

시험 자료는 직접 만든 짧은 행(예: 산수 문제)으로 만든다. 행을 만들 때 `_record_sha256`은 `digest`로 계산하고, V8 판정 필드는 `score_with_parser`로 실제 계산해 넣는다.

기존 시험도 그대로 통과해야 한다: `python -m unittest discover -s tests/unit -v`.

### 10.2 점검 사례 — `checkset_v1.yaml` (모델을 실제로 돌린다)

직접 만든 사례 30개 안팎이다. 사례마다 `question`, `gold_answer`, `llm_response`, `expect`(`accept` 또는 `not_accept`)를 적는다. 넣을 유형은 다음과 같다.

| 유형 | 기대 |
|---|---|
| 보기 글자 없이 보기 내용으로 정답을 말함 | accept |
| 정답과 같은 수를 다른 표기로 씀(분수·소수, 단위 표기, 쉼표) | accept |
| 긴 풀이 끝에 정답을 문장으로 말함(자르기 길이를 넘는 긴 응답 포함) | accept |
| 다른 보기를 고름 | not_accept |
| 정답과 가까우나 다른 수 | not_accept |
| 정답을 언급하지만 최종적으로는 버림 | not_accept |
| 답 두 개를 함께 제시함 | not_accept |
| 최종 답 없이 끝남, 같은 말 반복 | not_accept |

`checkset` 명령은 사례별 판정과 두 수치를 낸다: `not_accept` 사례 중 `ACCEPT`가 나온 수(잘못된 인정), `accept` 사례 중 `ACCEPT`가 나온 비율.

**공식 판정의 사전 조건(미리 고정): 7B·BF16에서 잘못된 인정이 0건이어야 한다.** 1건이라도 있으면 실제 결과에 적용하지 않고 사용자에게 보고한다. `accept` 쪽 비율은 보고만 한다.

### 10.3 로컬 작동 확인 (사용자 지시: 3B와 양자화 모델로 코드가 도는지만)

**사용자 정정(2026-11-14): 로컬에서는 3B만으로 코드 작동을 확인한다. 7B는 로컬에 다운로드하거나 실행하지 않으며, 추후 사용자 명령에 따라 런포드에서 진행한다. 아래에서 과거 7B NF4 로컬 확인 단계는 삭제했다.**

| 순서 | 모델 | 양자화 | 목적 |
|---|---|---|---|
| 1 | CompassVerifier-3B | nf4 | 가장 가벼운 경로로 `checkset`과 `run --limit 4`가 끝까지 도는지 |
| 2 | CompassVerifier-3B | none | 8GB에서 BF16이 올라가면 확인. 메모리 부족이면 건너뛰고 그렇게 기록 |

- 환경(2026-11-13 실측): `/home/jang2/mmdl-envs/notebook-nf4/bin/python` — Python 3.12.13, torch 2.8.0+cu128, transformers 4.57.1, bitsandbytes 0.48.2, accelerate 1.11.0, PyYAML 6.0.3. GPU는 RTX 5060 Laptop 8GB.
- 모델 저장 위치: `HF_HOME=/home/jang2/mmdl-cache/huggingface`. 내려받을 때 리비전을 지정한다. 디스크 여유는 473GB였다(50GiB 여유 규칙 충족).
- 로컬 결과는 모두 `mode=smoke`, `reportable=false`다. 숫자를 점수로 보고하지 않는다. 보고할 것은 "끝까지 돌았는가, 출력 형식이 맞는가, 같은 입력에 같은 출력이 나오는가"다.

## 11. 알려진 한계와 미검증 사항

| 항목 | 내용 | 대응 |
|---|---|---|
| 한쪽으로 치우친 점수 | V8 오답만 다시 본다. 판정 모델의 잘못된 인정은 점수를 올리지만, V8의 잘못된 인정은 고쳐지지 않는다 | 요약의 `notes`와 보고문에 항상 적는다. `ACCEPT` 문항은 사람이 전부 열어 본다(실행당 최대 50개 안팎) |
| 정답을 본 판정 | 판정 모델은 정답을 보고 같은지 판단한다. 원본 Qwen의 객관식 판정(정답을 보지 않음)과 다르다 | 기준값 0.90과 점검 사례의 "잘못된 인정 0건" 조건으로 줄인다 |
| 기준값 0.90 | 근거 없이 정한 초깃값이다 | 결과를 보기 전에 고정한다. 바꾸면 v2 |
| 응답 자르기 3,072토큰 | 대상 응답은 최대 19,151토큰이다. 공식 예시는 짧은 입력(vLLM 예시 `--max-model-len 2048`)을 쓰므로 긴 입력에서의 동작은 확인되지 않았다 | 점검 사례에 긴 응답을 넣어 확인한다. `truncated_inputs`를 보고한다 |
| 4비트와 16비트의 차이 | 양자화하면 확률이 달라진다. 로컬 결과로 공식 결과를 예측할 수 없다 | 로컬은 작동 확인만. 공식은 BF16 |
| 이미지를 보지 못함 | 판정 모델은 글만 본다. 그림을 가리키는 답("왼쪽 그림")은 판정이 어렵다 | `INVALID`·`UNSURE`로 남으면 오답 유지 |
| 장비 간 재현 | 같은 설정이라도 GPU가 다르면 확률 끝자리가 다를 수 있다 | `judge.json`에 GPU·라이브러리 버전을 기록한다 |
| 라이선스 표기 불일치 | §3.5 | 구현 첫 단계에서 확인해 `README.md`에 기록 |

## 12. 공식 판정 실행 (사용자 명령 후에만)

전 단계를 런포드에서 실행한다. 사용자 결정(2026-11-14)에 따라 7B는 로컬에 받지 않으므로, 7B 토크나이저가 필요한 `select`·`report`는 로컬에서 돌릴 수 없다. 로컬은 결과 검토만 한다.

1. 런포드에 24GB 이상 GPU 한 대를 만든다. 가격을 먼저 밝힌다(2026-11-13 조회: A40 0.59달러/시).
2. 포드에서 MMDL_test 고정 커밋을 깨끗하게 받아 `env/requirements-eval.lock`으로 설치한다(uv 사용). 로컬 코드·가상환경을 올리지 않는다.
3. 입력 `predictions.jsonl`을 SHA256과 함께 올린다. 벤치마크 원문과 응답이 들어 있으므로 우리 포드로만 보낸다. 올리는 입력은 두 개다: v15 실행 결과, 그리고 같은 문항 ID로 제한한 B2 결과. 제한본은 B2 `predictions.jsonl`에서 해당 ID의 줄만 바꾸지 않고 그대로 복사해 만든다(행마다 자체 해시가 있고 `select`는 행 수 제한이 없다).
4. 7B를 고정 리비전으로 받고 `checkset --mode official`을 먼저 돌린다. 잘못된 인정(오수락)이 0건일 때만 다음으로 간다.
5. 입력마다 `run --mode official --checkset-receipt <영수증>`을 한 번 돌린다(select·judge·report 포함).
6. 출력 폴더 전체(`targets/verdicts/merged/summary/review.md/judge.json/checkset.json`)를 내려받아 SHA256을 포드 쪽과 대조한 뒤 포드를 삭제하고 목록이 비었는지 확인한다.
7. 로컬에서 `review.md`의 `ACCEPT`·`UNSURE` 문항을 사람이 확인한다.

v15 포드 절차(`claudedocs/plan_v15_20261114.md`)는 V8 평가가 끝난 뒤 이 절차를 실행한다.

예상 규모(이전 6개 실행 기준): 판정 약 270회다. 이번 절차의 실행은 2개다. 시간은 대부분 설치와 모델 내려받기(15GB)에 든다. 비용은 측정 전이며, 수십 분 수준이면 1달러 미만으로 추정한다.

## 13. 작업 순서

각 단계는 앞 단계가 끝나고 시험이 통과한 뒤 시작한다. 단계마다 한 사람이 한 파일을 맡는다.

| 단계 | 할 일 | 건드리는 파일 | 끝난 기준 |
|---|---|---|---|
| 0 | `git status`로 작업공간 확인. `python -m scripts.sync_submission --check` 출력을 저장해 둔다. 판정 모델 라이선스 확인 | 없음 | 차이 목록 4건 기록 |
| 1 | 설정·질문문 고정 | `__init__.py`, `checker_v1.yaml`, `prompt.py`, `tests/__init__.py`, `tests/test_isolation.py`(질문문 부분) | 질문문 해시 검사 통과 |
| 2 | 입력 검증·대상 선정 | `selection.py`, `tests/test_selection.py` | 시험 통과. B2 파일에 `select`를 돌리면 대상이 39개(7/14/18) |
| 3 | 판정 규칙·병합·요약 | `scoring.py`, `tests/test_scoring.py` | 시험 통과 |
| 4 | 판정 모델 호출 | `backend.py` | `FakeJudge` 시험 통과. 실제 모델은 6단계에서 |
| 5 | 명령줄 | `cli.py`, `__main__.py`, `tests/test_cli.py`, `tests/test_isolation.py`(나머지) | 시험 통과. `ruff check src/mmdl/evaluation/CompassVerifier_checker` 통과 |
| 6 | MMDL 제외 규칙 | `scripts/sync_submission.py`, `tests/unit/test_sync_submission.py` | `tests/unit` 전체 통과. `--check` 차이 목록이 0단계와 같다 |
| 7 | 점검 사례 작성과 로컬 작동 확인 | `checkset_v1.yaml`, `README.md` | §10.3의 3B NF4 경로가 끝까지 돈다. 3B BF16은 메모리가 허용할 때만 확인 |
| 8 | 문서 갱신 | `docs/PROJECT_STATUS.md`, `CLAUDE.md` = `AGENTS.md`(`cmp`로 동일 확인) | 기록 완료 |
| 9 | (사용자 명령) MMDL_test 게시 | `python -m scripts.sync_test_repo --sync` 후 MMDL_test에서 시험·commit·push | 원격 HEAD 일치 |
| 10 | (사용자 명령) 공식 판정 | §12 | 실행별 `summary.json`과 검토 완료 |

2단계의 대상 수 확인값(§1.2 표): B2 39, F0 8K 44, v11b 40, v11c 44, v12 48, v13 52. 구현이 맞으면 이 수가 그대로 나와야 한다.

위험도가 높은 부분은 2단계의 분류와 3단계의 병합이다. 점수가 여기서 정해지므로, 이 두 파일은 구현한 작업자와 다른 검토자가 다시 본다.

## 14. 하지 않는 것

- V8 파서 수정, V8 점수 변경, V8이 맞다고 한 문항의 재판정.
- 전 문항 점검표(정상 종료한 모든 문항을 판정 모델로 따로 채점). 사용자가 "V8 오답→정답만"으로 정했다.
- 잘린 응답에서 답 꺼내기, 무작위 고르기, 여러 번 물어 고르기.
- 평가 실행 흐름(`fine_tuning/notebook_rl/proof_eval.py`, 포드 스크립트)에 끼워 넣기. v1은 저장 결과를 따로 다시 채점하는 도구다.
- vLLM 서버, 묶음 처리, 다른 판정 모델 지원, 외부 API 지원.
- 결과에 따른 기준값 조정.

## 15. 참고

- 원본 Qwen 평가 코드: `third_party/qwen3_vl/run_mmmu.py`(작업공간 사본), <https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/eval_utils.py>
- CompassVerifier: <https://huggingface.co/opencompass/CompassVerifier-7B>, <https://github.com/open-compass/CompassVerifier>, 논문 <https://arxiv.org/abs/2508.03686>
- 판정 전용 모델 사용 방식의 참고(확신할 때만 받아들이고 애매하면 넘긴다): <https://arxiv.org/pdf/2609.26550>
- 프로젝트 지침: `CLAUDE.md` §2(절대 규칙), §5(저장소·게시), §6~§7(평가 조건)
- V8 설명: `claudedocs/mmmu_v8_pipeline_20260924.md`, 채점 버전 조건 `docs/EVALUATION_V2.md`
