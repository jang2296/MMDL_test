# Qwen3-VL-4B 오답노트형 검증 자기수정 RL — 연구·데이터·구현 설계서 v2

작성 2026-10-16 KST · 내부 연구 문서 · 파이프라인 도식: [error_notebook_rl_pipeline_20261016.html](error_notebook_rl_pipeline_20261016.html)

**문서 상태: RESEARCHED / DESIGNED.** 이 문서는 구현 계약서다. 새 RL 코드 구현, GPU 학습, 성능 개선은 아직 하나도 완료되지 않았다. 이 문서의 수치 중 "실측"은 저장소의 기존 원장에서 온 값이고, "추정"은 아직 측정하지 않은 값이다.

선행 문서 [research_verified_self_correction_rl_20261016.md](research_verified_self_correction_rl_20261016.md)(이하 **v1**)를 대체하지 않고 확장한다. v1의 원칙(추가 SFT 금지, benchmark 격리, 첫 시도 중심 평가)은 그대로 유지하고, 사용자가 제안한 5단계 오답노트 구조를 구현 가능한 수준으로 구체화했다. v1과 다른 결정은 §0.3에 모았다.

---

## 읽는 방법 — 누가 무엇을 읽어야 하나

이 문서는 장기 추론이 약한 하위 모델도 그대로 따라 구현할 수 있도록 썼다. 모든 구현 작업은 §16의 **작업 계약(TASK contract)** 단위로 나뉘며, 각 계약은 "어떤 파일만 수정하고, 어떤 기존 코드를 참고하고, 어떤 명령이 통과하면 끝인지"를 명시한다.

| 역할 | 반드시 읽을 절 | 하는 일 |
|---|---|---|
| Haiku 5.5 (소규모 대량 코딩) | §1, §14, §15의 해당 파일 절, §16의 자기 계약, 부록 A·C | 계약 하나 = 1~2 파일. 계약에 없는 파일 수정 금지 |
| Sonnet 5.5 (중규모 코딩·통합) | 위 + §7~§9, §12~§13 | 2~5 파일 계약, 생성기·검증기·통합 |
| Codex 6.1 Sol (대량 코딩·중소 통합) | 위 + §17 | 생성기 묶음, 테스트 대량 작성, 통합 |
| Opus 5.5 (고난도 분석·핵심 코드 검토) | 전체, 특히 §10·§11·§15.10~15.14·§17 | 손실·정책·vLLM 동기화 구현, 모든 저가 모델 diff 검토 |
| Fable / Astra (최종 감사) | 전체 | wave 종료 시 통합 diff 감사, 실험 판정 |

**모든 구현자 공통 규칙**

1. 작업 위치는 `C:\projects\multimodel\fine_tuning\notebook_rl\`와 `fine_tuning\tests\test_notebook_rl_*.py`뿐이다. `MMDL\` 폴더(교수님 제출본)는 읽기만 한다.
2. §14.3의 **수정 금지 파일**(해시로 고정된 기존 파일)은 절대 고치지 않는다. 필요한 기능은 새 파일에서 import해서 쓴다.
3. 정답을 따라 쓰는 손실(`labels=`, cross-entropy on gold, `model(...).loss`)을 새 코드에 넣지 않는다. 이유는 §1.
4. 계약의 "Done when" 명령이 통과하지 않으면 완료라고 보고하지 않는다. 테스트를 건너뛰거나 약화하지 않는다.
5. 모르는 것은 추측하지 말고 `BLOCKED: <무엇이 없는지>`로 돌려보낸다.

---

## 0. 결론 요약

### 0.1 무엇을 만드는가

**외부 시각 문제 → 같은 문제를 여러 번 풀이 → 프로그램 검증기 채점 → 틀린 풀이를 문맥에 남긴 채 "오답노트 + 새 풀이" 작성 → 다시 채점 → 맞힌 뒤 원인 진단 노트 → 검증된 원인에 따라 Vision 학습 여부 결정 → 정책경사(GRPO 계열)로 가중치 업데이트 → 처음 보는 문제의 첫 시도 정확도로 평가.**

```text
          ┌────────────── 데이터 경계: MMMU/MMMU-Pro 배제 + 상위모델 라벨 배제 ──────────────┐
          │ 프로그램 생성기(악보·회로·역학·분자·차트·그래프…) + 사람 라벨 train split(TabMWP 등) │
          └──────────────────────────────┬───────────────────────────────────────────────┘
               actor 입력(이미지·질문·보기)     │     verifier 금고(정답·슬롯 사실·solver)
                                              ▼
   turn-1 그룹: 같은 문제 8회 풀이 ──► 검증기 ──► 1/0 보상 ──► advantage(그룹 평균 대비)
          │ 실패 풀이 2개 선택
          ▼
   turn-2 그룹: [실패 풀이 + "오답, 정답 비공개"] 아래에 Error check + 새 풀이 4회 ──► 검증기 ──► advantage
          │ 결국 맞힌 경로
          ▼
   진단 그룹: "시도1의 원인은?" 4회 ──► 검증된 원인과 일치하면 보상
          │
   검증된 원인(슬롯 대조): 시각 오독 있음 → Vision LoRA + 4 Merger + LLM LoRA / 없음 → 4 Merger + LLM LoRA
          ▼
   정책경사 업데이트(자기 생성 토큰 × advantage, 고정 F0 KL) → vLLM에 가중치 동기화 → 다음 batch
          ▼
   평가: 외부 sealed 첫 시도 정확도(주 지표) · retention · 마지막에 MMMU 900 (8K, P0, V8) F0 vs FT 1회
```

### 0.2 핵심 결정 10가지

| # | 결정 | 이유 (근거 절) |
|---|---|---|
| D1 | 손실은 **자기 생성 토큰의 정책경사뿐**. SFT·증류·자기 성공 풀이 모방 없음 | 사용자 조건. 과거 SFT 4종 모두 MMMU 하락 (§3) |
| D2 | 시작점은 **fresh F0** (`ebb281ec…`). 과거 FL/FM/FV/FML·A/B 체크포인트 사용 금지 | 과거 체크포인트는 SFT 결과물 |
| D3 | **트리형 롤아웃 + 턴 단위 그룹**: 같은 입력을 공유하는 샘플끼리만 advantage 비교 | episode 보상의 신용할당 오류 제거, 첫 시도 고의 오답 유인 제거 (§7.1, MM-ReCoder·S²R) |
| D4 | 오답노트는 3층: host 검증 기록 / 재풀이 Error check / 사후 진단. **진단 노트는 검증 가능한 원인과 일치할 때만 보상** | 모델 자기평가는 증거가 아님 (§8) |
| D5 | Vision 학습 여부는 **모델 노트가 아니라 슬롯 대조로 검증된 시각 오독**으로 문제 단위 결정. 기본은 Merger+LLM | 사용자 5단계 + "ViT를 RL로 열면 이득 불명확, grounding 손상 보고" (§11, §4) |
| D6 | 대조군 필수: 고정 FML, 고정 FVML, 무작위 라우팅, 단일턴 RLVR, **무작위 보상(spurious) 대조** | Qwen 계열은 무작위 보상으로도 점수가 오른 사례가 있음 (§18) |
| D7 | 학습 데이터 ≈80%는 **프로그램 생성 + 슬롯 단위 정답**, 나머지는 사람 라벨 train split. LLM 생성 라벨·RL 혼합셋(ViRL39K 등) 배제 | 오염·상위모델 라벨 금지, 귀인 가능성 (§12) |
| D8 | 비중복 검증 v2: 기존 fingerprint + 짧은 질문·옵션 n-gram + **OCR** + crop/타일 해시 + 복제탐지 임베딩, 모든 split 검사, family 단위 제외 | 기존 index의 L1~L15 약점 (§13) |
| D9 | 구현 기반은 **현재 고정 스택(torch 2.8 / transformers 4.57.1 / peft 0.17.1 / vLLM 0.11.0) 위의 최소 자체 루프**. TRL·verl 미사용 | TRL 1.2.0의 sleep 재로딩 버그, in-place merge 오염, 트리 그룹 미지원 (§15.12) |
| D10 | 평가 주 지표는 **처음 보는 외부 문제의 첫 시도 정확도**. MMMU는 방법·체크포인트 고정 후 마지막에 F0/FT 동일 8K 조건으로 1회 | 재시도 정확도는 oracle 피드백에 의존 (Huang et al.) |

### 0.3 v1에서 바뀐 점

| 항목 | v1 (10/16 오전) | v2 (이 문서) |
|---|---|---|
| 보상 단위 | episode `1.0/0.5/0` + 단일턴 50% 혼합 | 턴별 그룹 `R_answer`; 트리형 공유 상태 그룹 |
| 그룹 크기 | G=4 episode | turn-1 `n1=8`, 재풀이 `n2=4` × 실패 문맥 2개 |
| 오답노트 | 감사 기록 전용, 학습 미사용 | 3층. 재풀이 노트는 재풀이 성공 시 함께 강화, 진단 노트는 검증 원인 일치 보상 |
| 모듈 라우팅 | 후속 연구로 보류 | 사용자 5단계를 `MOD-RV`로 구현 + 대조군 3종 |
| 코드 위치 | `experiments/verified_rl/` 제안 | **`fine_tuning/notebook_rl/`** (사용자 지시) |
| 학습 프레임워크 | TRL 후보 | 자체 최소 루프 (TRL 비호환 근거 확인) |
| Merger 학습 방식 | 기존 직접 학습 + 별도 F0 | PEFT `modules_to_save` → `disable_adapter()`로 정확한 F0 reference (소스 확인) |
| 데이터 | 4 실패군 각 1,000 제안 | 약점 과목별 프로그램 생성기 11종 + 사람 라벨 5종, pilot 384 / 확장 4,000 |
| 오염 검사 | 기존 index + 계획 | 기존 index의 구체적 약점 15개 확인 + v2 gate 12단계 |

### 0.4 지금 말할 수 있는 것과 없는 것

| 말할 수 있다 | 말할 수 없다 |
|---|---|
| 설계가 문헌상 검증된 부품(GRPO/Dr.GRPO 손실, shared-first-turn 그룹, 검증 가능한 보상)으로 구성됨 | 이 조합이 Qwen3-VL-4B에서 효과가 있다 |
| 기존 TabMWP 1600 선별본은 고정 index 기준 후보 0 | 모든 후보 데이터가 MMMU/Pro와 겹치지 않는다 |
| 오답노트 진단 보상은 검증 가능한 원인에 대해서만 정의됨 | 모델의 "이미지를 잘못 봤다"는 문장이 사실이다 |
| Vision 경로 차단은 4개 Merger 입력에서 정확히 가능 | 시각 오독이 Vision encoder 결함 때문이다 |
| 비용 계산식과 측정 절차 | pilot이 특정 금액·시간 안에 끝난다 (smoke 실측 전) |

## 1. 절대 규칙과 코드로 강제하는 방법

### 1.1 "지도학습 금지"의 정확한 해석

이 프로젝트에서 금지하는 것은 **지정 Instruct 모델 이후의 추가 지도 미세조정과 상위 모델로부터의 증류**다. RLVR도 정답을 이용해 보상을 계산하므로 "정답 정보가 전혀 없는 학습"은 아니다. 보고서에는 이렇게 쓴다.

> 지정 Instruct checkpoint를 초기 정책으로 사용하되 추가 지도 미세조정이나 상위 모델 증류는 수행하지 않는다. 모델이 직접 생성한 풀이에 대해 독립 프로그램 검증기가 산출한 보상으로 정책을 업데이트한다. 데이터의 정답과 구조 주석은 답안 모방 target이 아니라 보상 계산에만 쓰인다.

| 작업 | 허용 여부 |
|---|---|
| 사람/교사 풀이 문장을 `labels`로 넣은 cross-entropy | **금지** |
| GPT·Gemini·Qwen-235B 등 상위 모델의 풀이·라벨·비평·판정 사용 | **금지** (데이터 라벨, 보상 judge, 피드백 모두) |
| 자기 생성 성공 풀이만 골라 CE로 재학습 (STaR, RFT, rejection-sampling SFT) | **금지** |
| 성공한 재풀이를 첫 시도 문맥으로 옮겨 모방 (문맥 증류) | **금지** (v2 범위) |
| 과거 SFT 체크포인트를 시작점으로 사용 | **금지** |
| 정답·슬롯 사실을 verifier 안에서만 써서 0/1 보상 계산 | 허용 (RLVR) |
| 자기 생성 토큰의 log-prob × advantage로 업데이트 | 허용 (이 파이프라인의 유일한 학습 신호) |
| 고정 F0에 대한 KL 정규화 | 허용 (모방이 아니라 이탈 억제) |
| 정답 사실을 넣은 진단 전용 생성(oracle 반사실, §8.3) | 허용, **단 loss에 들어가면 안 됨** |
| Qwen3-VL-4B 자체를 OCR 도구로 써서 오염 검사 | 허용 (라벨 생성이 아니라 탐지 도구) |

미해결 질문: 교수님이 금지한 것이 SFT뿐인지, 정답을 보상에 쓰는 학습까지인지 실행 전 확인해야 한다(§21). 후자라면 이 설계 전체를 쓸 수 없다.

### 1.2 데이터 금지 규칙

1. MMMU(`MMMU/MMMU` @`98e6ac0c…`)의 dev/validation/test, MMMU-Pro(`MMMU/MMMU_Pro` @`563f3e84…`)의 standard10/standard4/vision과 **그 파생물 전부**를 학습·보상·문제 선택·체크포인트 선택에 쓰지 않는다. MMMU-Pro는 MMMU 이미지를 재사용한다(로컬 실측: MMMU validation 고유 이미지 959개 중 632개가 Pro std10과 byte 동일). 두 벤치마크는 하나의 이미지 family로 다룬다.
2. 라벨 출처(`label_provenance`)가 `PROGRAM` 또는 `HUMAN`인 자료만 쓴다. `MODEL`(LLM/VLM이 만든 질문·답·풀이·번역 후 재작성 포함 의심)은 거부한다.
3. 다른 벤치마크의 평가 split(val/test)도 학습하지 않는다(예: ScienceQA-IMG test, Geometry3K test, ChartQA test). 우리 자체 평가셋과 family가 겹치는 학습 문제도 제외한다.
4. MMMU test 정답은 2026-02-12부터 공개됐다. 그 이후 웹에서 수집된 혼합셋은 기본적으로 고위험이다.

### 1.3 fail-closed 장치 (구현 필수, 테스트로 검증)

| 장치 | 위치 | 동작 |
|---|---|---|
| 손실 경로 감시 | `losses.py`, `trainer.py` | `labels=` 인자 사용 금지. 테스트에서 `torch.nn.functional.cross_entropy`와 `fine_tuning.train.AssistantCollator`를 예외를 던지도록 patch한 상태로 1 step 학습이 성공해야 함 |
| actor 입력 allowlist | `vault.py` | actor 레코드는 `{problem_id, source, family, mode, question, choices, answer_format, image_paths, image_sha256}`(§15.1 `ActorItem` 필드와 동일)만 허용. 다른 키가 있으면 `ValueError` |
| verifier 금고 분리 | `vault.py` | 정답·슬롯·solver 파라미터는 `verifier_private.jsonl`에만. actor 로더는 이 파일 경로를 열 수 없음(`sys.addaudithook`으로 `open` 감시, 위반 시 예외) |
| benchmark 경로 차단 | `guards.py` | 데이터 생성·학습 프로세스에서 `MMDL_DATA_ROOT/evaluation` 아래 파일 open 시 즉시 예외. 오염 검사 프로세스만 예외 허용 플래그를 가짐 |
| 출처 검사 | `contamination.py` | manifest의 모든 행에 `label_provenance ∈ {PROGRAM, HUMAN}`, `benchmark_derived=false`, 부모 lineage 확인 (`fine_tuning.audit.check_lineage` 재사용) |
| 상위 모델 호출 차단 | `guards.py` | 학습·생성 코드에서 `openai`, `anthropic`, `google.generativeai` 등 import 시 예외. 테스트가 `sys.modules` 검사 |
| 정책 버전 고정 | `rollout.py` | 한 step의 트리 생성 중 `policy_version` 변경 시 예외 |
| 시작점 검사 | `policy.py` | 로드한 base 파일 해시 == `manifests/models/baseline.json`; PEFT 설정이 이미 있으면 거부 |
| 평가 격리 | `eval_mmmu8k.py` | 학습 종료 후 고정 checkpoint만 입력. 학습 중 MMMU 접근 함수가 없음 |

## 2. 현재 Qwen3-VL-4B의 문제: 점수와 실패 유형

### 2.1 제출 baseline (`MMDL/reports/mmmu_baseline.md`)

`Qwen/Qwen3-VL-4B-Instruct` @`ebb281ec…`, BF16, MMMU validation 900, P0 프롬프트, 32,768 출력, 고해상도 이미지(1,003,520~4,014,080 px), vLLM 0.11.0, V8 파서 → **558/900 = 62.00%**. 공식 비교값 67.4%보다 5.40%p 낮다.

### 2.2 과목별 정확도 (30과목, 각 30문항)

`Hard`는 MMMU 원래 난이도 표기, `length`는 32K 상한 종료 수, `NO_PARSE`는 최종답 추출 실패 수다([V8 과목 분석](MMMU_V8_70pct_subject_analysis_20260930.md)).

| 과목 | 정확도 | Hard | length | NO_PARSE | 과목 | 정확도 | Hard | length | NO_PARSE |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| **Music** | **20.00** | 7 | 15 | 13 | **Pharmacy** | **63.33** | 0 | 3 | 3 |
| **Diagnostics_and_Lab_Med** | **36.67** | 6 | 0 | 0 | **Accounting** | **66.67** | 4 | 5 | 5 |
| **Electronics** | **40.00** | 18 | 7 | 13 | **Sociology** | **66.67** | 1 | 0 | 0 |
| **Materials** | **40.00** | 13 | 6 | 7 | Finance | 70.00 | 7 | 1 | 5 |
| **Architecture_and_Eng** | **43.33** | 18 | 13 | 13 | Design | 73.33 | 4 | 0 | 0 |
| **Mechanical_Eng** | **43.33** | 17 | 11 | 9 | History | 73.33 | 2 | 0 | 0 |
| **Biology** | **46.67** | 2 | 1 | 2 | Basic_Medical_Science | 76.67 | 3 | 0 | 1 |
| **Energy_and_Power** | **46.67** | 15 | 13 | 11 | Clinical_Medicine | 76.67 | 8 | 1 | 0 |
| **Agriculture** | **50.00** | 12 | 0 | 0 | Art_Theory | 80.00 | 1 | 1 | 0 |
| **Chemistry** | **50.00** | 7 | 4 | 5 | Literature | 80.00 | 0 | 0 | 0 |
| **Computer_Science** | **53.33** | 11 | 1 | 1 | Physics | 80.00 | 3 | 1 | 1 |
| **Geography** | **56.67** | 4 | 0 | 1 | Economics | 83.33 | 2 | 0 | 0 |
| **Art** | **60.00** | 0 | 3 | 1 | Psychology | 83.33 | 1 | 0 | 0 |
| **Manage** | **63.33** | 5 | 0 | 0 | Marketing | 86.67 | 5 | 0 | 0 |
| **Math** | **63.33** | 5 | 2 | 4 | Public_Health | 86.67 | 0 | 2 | 2 |

굵은 18과목이 70% 미만(540문항, 정답 273)이다. 70% 이상 12과목(360문항)의 정답은 285다.

### 2.3 실패 유형 (342 오답의 상호배타 분해)

| 분류 | 수 | 의미 |
|---|---:|---|
| 상한 종료 + 답 추출 실패 | 76 | 장문 반복·미완결 |
| 상한 종료 + 추출됐으나 오답 | 9 | |
| 자연 종료 + 추출 실패 | 21 | 형식 문제 |
| 자연 종료 + 추출됐으나 오답 | **236** (객관식 218 + 주관식 18) | **가장 큰 덩어리: 끝까지 풀었는데 틀림** |

- 상한 종료 90건 중 85건 오답, 이 90건이 전체 생성 토큰의 **71.6%**를 차지했다.
- 출력 길이별 정확도: ≤512 토큰 74.1%, 513~2048 62.8%, 2049~8192 62.7%, 8193~32767 30%, 정확히 32768 5.6%.
- 객관식 543/847 = 64.1%, **주관식 15/53 = 28.3%**.
- 70% 미만 과목군: 상한 종료 15.56%, NO_PARSE 16.3%, 자연 종료·추출 성공 문항만의 정확도 61.14% (70% 이상 과목군 81.38%).

### 2.4 약점 과목이 틀리는 이유 (기존 분석 요약)

1. **기호·공간·연결 관계 + 미완결.** Music, Architecture, Energy, Mechanical, Electronics, Materials 6과목은 70/180 정답, 상한 종료 65, NO_PARSE 66, 전체 토큰의 약 64%를 차지한다. 투영·단면도 문항 0/8, Music 15/30 상한 종료(음표·음자리표 오독 + 규칙 오적용), Biology_6은 계통수 간선 오독.
2. **깔끔하게 끝냈지만 틀린 판별·지식 결합.** Diagnostics 11/30, Agriculture 15/30은 상한 종료·추출 실패가 0이고 모든 오답이 확신에 찬 오답이다. 길이 제어로는 해결되지 않는다.
3. **표·단위·방향.** Accounting 표 문항 19/28, `$2,960 unfavorable` 같은 수식어 문제.
4. **해상도 효과 미미.** 저해상도→고해상도 554→558이지만 문항 단위로는 +61/−57이 서로 상쇄됐다.

### 2.5 고정 가중치 진단에서 알게 된 것

| 진단 | 결과 | 설계에 주는 의미 |
|---|---|---|
| 기능 진단 102문항 (30과목) | I(이미지) 40/102, T0(이미지 없이 질문만) 37/102, O0(AI 전사) 42/85, M(자기 구조 설명만) 33/102, MI 39/102 | **텍스트 사전지식만으로도 거의 비슷하게 맞힘** → 학습 문제는 이미지가 꼭 필요해야 함. 자기 설명으로 바꾸면 정보 손실 |
| Knowledge Oracle K1/K2 | 각 30/30 | 기초 지식은 천장. "지식 부족" 유형의 노트가 주요 개선 수단일 가능성 낮음 |
| 8K 입력 회복 48문항 | I 22, T 24, IT 26, IR 20, TR 22 | 전사·규칙 추가의 회복은 제한적. 모든 신뢰구간이 0 포함 |
| 그룹별 F0 I | A 기호 5/12, B 도식·공간 5/12, C 과학 시각 5/12, D 다단계 수량 7/12 | 네 실패군 모두 약함 |

**설계 결론.** 개선 대상은 (1) 이미지 속 구조를 읽어 끝까지 계산해 답을 완성하는 능력, (2) 길게 반복하다 끝나지 않는 습관, (3) 이미지 없이도 그럴듯하게 고르는 텍스트 지름길 억제다. 해상도·지식 주입만으로는 부족하다.

## 3. 지금까지의 실험이 실패한 방식과 이번 설계의 대응

### 3.1 학습 실험 원장 (모두 SFT였다)

| 실험 | 데이터 (MMMU 없음) | 조건 | 결과 | 실패 양상 |
|---|---|---|---|---|
| pilot1600 (A40) | TRACE 1440 + NVW 160 | FL/FM/FV/FML, 100 update | MMMU60: F0 32 → FL 26, FM 32, FV 31, FML 26. External60: F0 12 → FL 17, FM 14, FV 12, FML 16 | LLM을 건드린 조건이 MMMU 하락. FL 출력 중앙값 772→16 토큰 |
| control1600 (A40) | TRACE 1600 | FL, 100 update | MMMU60 28 (F0 32) | NVW를 빼도 하락 |
| **G6 full10K** (3090) | TRACE 9000(답만 `{"answer": v}`) + NVW 1000 | FL/FM/FV/FML, 625 update, 각 fresh F0 | **MMMU60: F0 36 → FL 22, FM 27, FV 21, FML 23.** External48: F0 15 → FM 25, FML 21, FL 18, FV 18 | 외부 과제는 올랐지만 MMMU는 전 조건 하락. 손실 문항 대부분이 자연 종료·추출 성공 → 답 선택 자체가 바뀜 |
| **TabMWP A/B** (A40) | TabMWP 1600 + retention 400 | FM, 125 update. A=답만, B=표 사실+답 | **QA48: F0 35 → A 30, B 28.** 셀 268 → 253 / 299 (of 310). 확대 gate 둘 다 실패 | B는 표를 더 잘 옮겼지만 사실만 반복하고 답을 못 끝냄(NO_FINAL 8, 상한 종료 3) |

### 3.2 형식 표류의 실체 — 길이 붕괴

G6 full10K의 MMMU60(4K) 결과:

| 조건 | 상한 종료 | NO_PARSE | 한 줄 `LETTER. 보기` 응답 | 출력 토큰 중앙값 | 정답 |
|---|---:|---:|---:|---:|---:|
| F0 | 12 | 14 | 2 | 720 | 36 |
| FL | 2 | 4 | **51** | **8** | 22 |
| FM | 14 | 14 | 25 | 178.5 | 27 |
| FV | 15 | 16 | 4 | 525 | 21 |
| FML | 4 | 5 | **44** | **9** | 23 |

학습 target이 중앙값 2단어의 짧은 답이었고, 모델은 JSON을 복사한 것이 아니라 **풀이 없이 한 줄로 답하는 습관**을 배웠다. 추출 실패와 상한 종료는 줄었지만 정답도 줄었다. **"파싱 실패가 줄었다 ≠ 추론이 좋아졌다."**

### 3.3 교훈 → 설계 대응

| # | 교훈 (근거) | 이번 설계의 대응 |
|---|---|---|
| L1 | 모든 SFT가 MMMU를 낮췄고 LLM 조건이 가장 크게 낮췄다 | SFT 없음. 고정 F0 KL(β), 외부 retention 감시, 첫 시도 dev 악화 시 중단 |
| L2 | 짧은 target이 길이 붕괴를 만들었다 | 정답만으로 보상하면 객관식 찍기로 짧아질 위험 → 출력 길이 중앙값·≤32토큰 비율·한 줄 응답 비율을 매 step 건강 지표로 기록, 임계 초과 시 중단 (§18.6) |
| L3 | 상한 종료는 토큰을 낭비하고 거의 오답 | 턴당 출력 상한(Phase A로 결정), 절단=보상 0, overlong 감점은 별도 비교군 |
| L4 | 형식과 검증기를 먼저 고정하라 (`LETTER. 보기` 24건 손실 등) | 학습용 `strict-final-v1`을 구현 전 반례 테스트와 함께 동결 (§9.1) |
| L5 | LLM judge(Gemma)는 시스템 오류 19.56%로 기각됨 | 보상은 전부 프로그램. 어떤 모델도 judge로 쓰지 않음 |
| L6 | 사실 쓰기가 답 완성을 대체했다 (TabMWP B) | 최종 답이 있어야만 시각 보상. 노트 길이 상한. 노트 자체에는 보상 없음 |
| L7 | 자기 전사·노트는 정보 손실, 규칙 힌트는 무효 | 재풀이마다 이미지 유지. 자기 노트를 검증 없이 사실로 넣지 않음 |
| L8 | 출력만으로 Vision/LLM 책임을 식별할 수 없었다 | 라우팅은 **슬롯 정답이 있는 문제에서 검증된 오독**으로만, 대조군으로 효과 검증 |
| L9 | 지식은 천장, 결합이 병목 | 진단 분류를 VISUAL / NON_VISUAL / FINALIZATION 3개로 단순화 |
| L10 | 텍스트 사전지식이 강하다 (T0 37 vs I 40) | Phase A의 **blind 검사**(회색 이미지)로 이미지 없이 풀리는 문제 제외 |
| L11 | 같은 F0도 GPU에 따라 32 vs 36/60 | 비교는 같은 GPU 종류·같은 protocol·paired 통계, seed 3개 |
| L12 | 첫 update grad norm: V 24.73 vs M 1.35 vs L 1.54 | 파라미터 묶음별 gradient clipping |
| L13 | 모듈 간 용량·LR 교란 | 모듈 비교에 update norm 기록, 무작위 라우팅 대조 |
| L14 | 손실 집계 방식이 혼합 비중을 바꾼다 | 정규화 상수를 고정(Dr. GRPO), 그룹 종류별 가중치 명시 |
| L15 | 3090(24GB)은 rollout+학습 동시 적재가 빠듯 | 기본 하드웨어 A40 48GB, 3090은 비교 실험 불가 |
| L16 | HF generate는 36 tok/s로 rollout에 부적합 | vLLM 동일 프로세스 rollout + 가중치 동기화 (§15.12) |
| L17 | 예산 gate가 full10K를 멈췄다 | smoke 실측 → 2배 여유 gate → 본 실행 |
| L18 | 한 과제 형식 학습이 다른 형식 지시를 망쳤다 | plain(P0 호환) 모드를 학습에 섞고 P0 형식 준수율 감시 |
| L19 | 해상도만으로는 거의 변화 없음 | 학습 이미지 예산은 평가와 동일하게 두되 증대 실험은 하지 않음 |
| L20 | 주관식 28.3%로 매우 약함 | 학습 문제의 60% 이상을 열린 수치·짧은 답으로 |

### 3.4 재사용 가능한 평가 자산과 소진 여부

| 자산 | 상태 | 이번 사용 |
|---|---|---|
| MMMU val 900 | 파서·프로토콜 개발에 사용됨. 학습 금지 | 마지막 F0/FT paired 평가 1회만 |
| MMMU60 코호트 | 12회 이상 평가, 소진 | 학습 중 사용 안 함 |
| G6 external60, TabMWP dev QA48/FACTS24 | 소진 | 개발 참고만 |
| **TabMWP sealed 200** (native test, family 분리) | **한 번도 추론하지 않음** | 표 family sealed holdout |
| TabMWP 남은 dev 152 | 미사용 | 표 family dev |
| **Circuit pool100 sealed 13 roots / dev 10** | GPU 평가 없음 | 회로 sealed/dev |
| TRACE holdout (로컬 변환 1,356) | 일부(24)만 사용 | 일반 retention·전이 probe |
| NVW 생성기 | 새 seed로 무제한 | 새 sealed 생성 가능 |

## 4. 관련 연구: 무엇을 빌리고, 무엇을 빼고, 어떻게 인용하나

2026-10-16에 arXiv·학회 proceedings·공식 저장소에서 직접 확인했다. "PR"=주 학회/저널 게재 확인, "WS"=워크숍, "PP"=preprint, "PR?"=arXiv 코멘트에만 게재 표시. ✗는 우리 규칙(SFT 금지·상위 모델 라벨 금지)에 걸리는 부분이다. 논문의 수치는 그 논문의 모델·데이터·평가 조건의 결과이며 우리 모델의 기대 개선폭이 아니다.

### 4.1 텍스트 LLM의 RL 자기수정·재시도

| 연구 | 상태 | 핵심 방법 | ✗ 부분 | 우리가 쓰는 것 |
|---|---|---|---|---|
| **SCoRe** — Kumar et al., *Training Language Models to Self-Correct via RL* ([2409.12917v2](https://arxiv.org/abs/2409.12917v2)) | PR, ICLR 2025 Oral | 2시도. Stage I: 두 번째 시도 보상 최대화 + 첫 시도 KL 고정. Stage II: r1+r2 + α(r2−r1), α=10. 2번째 지시는 정오를 알려주지 않음 | 없음 | 첫 시도 고정 장치의 필요성, Δi→c / Δc→i 지표. STaR/Pair-SFT가 실패한 근거 |
| **Reflect, Retry, Reward** — Bensal et al. ([2505.24726v1](https://arxiv.org/abs/2505.24726v1)) | PP | 실패 → 자기 반성문 → 재시도. 성공하면 **반성 토큰에만** GRPO advantage | 없음 | 재풀이 `Error check` 노트를 재시도 성공으로 강화하는 근거. 노트 토큰만 강화하는 비교군 `ALG-NB-noteonly` |
| **UFO: A Simple "Try Again"** — Liu et al. ([2507.14295v2](https://arxiv.org/abs/2507.14295v2)) | PP (구판은 ICML'25 AI4Math WS) | "Incorrect, try again"만 주는 다회 PPO. 턴 감쇠 보상 γ^t | 없음 | 단일턴 RL 후 실패의 ~70%가 같은 답 반복 → **재시도 반복률 지표**. 단, "새 답" 보상은 객관식 소거를 장려하므로 쓰지 않음 |
| **Learning from Failures in Multi-Attempt RL** — Chung et al. ([2503.04808v1](https://arxiv.org/abs/2503.04808v1)) | PP | 이전 시도를 문맥에 둔 다회 PPO, 할인 없음 | 없음 | 다회 학습이 단일 시도 정확도도 올린 사례. 할인 없는 보상은 소거 위험 |
| **S²R** — Ma et al. ([2502.12853v1](https://arxiv.org/abs/2502.12853v1)) | PR, ACL 2025 | 과정 단위 RL: 자기 검증 판정이 실제 정오와 맞으면 보상, 같은 prefix 평균 baseline | GPT-4 검증문 SFT ✗ | **검증 가능한 판정 보상**과 **같은 prefix baseline** → 진단 보상 `R_diag`와 트리 그룹 설계 |
| **Self-rewarding correction** — Xiong et al. ([2502.19613v1](https://arxiv.org/abs/2502.19613v1)) | PP | 자기 검증·수정 | 자기 생성 SFT ✗ | **"틀렸다가 맞히면 1.5" 보상이 해킹됨: 첫 시도 18.6% vs 최종 77.6%** → 첫 시도 고의 오답 유인을 구조적으로 막는 근거 |
| **LTE** — Tang et al. ([2510.26109v4](https://arxiv.org/abs/2510.26109v4)) | PR? (ACL 2026 main 표기) | 전부 틀린 그룹에서 자기 오답을 힌트로 재생성, 힌트 없는 원 프롬프트에 off-policy 학습 | 회색지대(긍정 샘플만이면 자기모방) | v2에서는 **사용하지 않음**. 후속 비교 후보로만 기록 |
| **Critique-GRPO** — Zhang et al. ([2506.03106v7](https://arxiv.org/abs/2506.03106v7)) | PR? (ICML 2026 Spotlight 표기) | 비평 기반 수정본을 GRPO 그룹에 섞음 | GPT-4o 비평 ✗ | **이진 피드백만 주면 유효 수정이 ~2%뿐**(CoT 비평 ~36%) → 재풀이 신호가 희소할 수 있다는 기대치 |
| **LUFFY** — Yan et al. ([2504.14945v5](https://arxiv.org/abs/2504.14945v5)) | PR, NeurIPS 2025 | off-policy 궤적을 정책 형상 f(r)=r/(r+γ)로 혼합 | R1 교사 궤적 ✗ | 사용 안 함 (교사) |
| **Negative Sample Reinforcement** — Zhu et al. ([2506.01347v2](https://arxiv.org/abs/2506.01347v2)) | PR, NeurIPS 2025 | 오답 확률만 낮춰도 Pass@1이 PPO/GRPO 수준 | 없음 | "틀렸음을 학습" = 오답 토큰의 음의 advantage라는 해석의 근거 |
| **PAG** — Jiang et al. ([2506.10406v1](https://arxiv.org/abs/2506.10406v1)) | WS, NeurIPS'25 MATH-AI | 자기 검증이 오류를 지적할 때만 수정 | 없음 | 후속: 선택적 수정 |
| **RISE** — Qu et al. ([2407.18219v2](https://arxiv.org/abs/2407.18219v2)) | PR, NeurIPS 2024 | 다회 개선을 MDP로 보고 가중 회귀 학습 | 가중 SFT·교사 ✗ | 개념적 선행연구로만. oracle 유무에 따른 지표 차이 |

### 4.2 자기수정의 한계를 보여주는 연구 (평가 설계 근거)

| 연구 | 상태 | 인용 가능한 주장 |
|---|---|---|
| **Large Language Models Cannot Self-Correct Reasoning Yet** — Huang et al. ([2310.01798v2](https://arxiv.org/abs/2310.01798v2)) | PR, ICLR 2024 | oracle 없는 자기수정은 정확도를 낮춘다(GPT-3.5 CommonsenseQA 75.8→38.1). oracle 라벨을 쓴 개선은 oracle의 성능일 뿐 → **MMMU 평가는 첫 시도만** |
| **When Can LLMs Actually Correct Their Own Mistakes?** — Kamoi et al. ([2406.01297v3](https://arxiv.org/abs/2406.01297v3)) | PR, TACL 2024 | 자기수정은 **신뢰할 수 있는 외부 피드백** 또는 대규모 미세조정이 있을 때 성공 → 학습 중 프로그램 검증기 사용의 근거 |
| **Reflexion** — Shinn et al. ([2303.11366v4](https://arxiv.org/abs/2303.11366v4)) | PR, NeurIPS 2023 | 가중치 없이 반성문을 기억에 저장 — "오답노트"의 개념적 조상(학습 아님) |
| **Self-Refine** — Madaan et al. ([2303.17651v2](https://arxiv.org/abs/2303.17651v2)) | PR, NeurIPS 2023 | 프롬프트 반복 수정(배경) |
| **M2Note: Mistake Notebook Learning** ([2607.00685v2](https://arxiv.org/abs/2607.00685v2)) | PP | 이름이 가장 가까운 선행연구지만 **학습 없는**(검색 증강) 오답노트 → 우리와의 차이(가중치 학습)를 설명하는 대조 인용 |

### 4.3 VLM의 반성·자기수정 RL

| 연구 | 상태 | 핵심 | ✗ 부분 | 우리가 쓰는 것 |
|---|---|---|---|---|
| **VL-Rethinker** — Wang et al. ([2504.08837v3](https://arxiv.org/abs/2504.08837v3)) | PR, NeurIPS 2025 | Selective Sample Replay(advantage 0 그룹 제거), Forced Rethinking | Forced Rethinking에 SFT 손실 추가 ✗ | 0-분산 그룹 제거 아이디어 |
| **SRPO** — Wan et al. ([2506.01713v3](https://arxiv.org/abs/2506.01713v3)) | PR? (NeurIPS 2025 표기) | 반성 보상: 오→정 +0.5, 정→정 +0.25, 오→오 0, 정→오 −0.25 | GPT-o4-mini 반성문 SFT ✗ | "처음 맞힘 > 고쳐서 맞힘 > 맞던 것을 망침" 순서의 인용 근거 |
| **MM-ReCoder** — Tang et al. ([2604.01600v1](https://arxiv.org/abs/2604.01600v1)) | PR, CVPR 2026 | **shared-first-turn**: 첫 턴 하나를 고정하고 그 아래 두 번째 턴 G개로 GRPO. 수정 보너스가 첫 턴 품질을 고의로 낮추는 해킹 보고 | Qwen3-VL-235B 데이터·72B judge ✗ | **트리형 턴 그룹의 직접 근거** (§7) |
| **Sherlock** — Ding & Zhang ([2505.22651v2](https://arxiv.org/abs/2505.22651v2)) | PR, NeurIPS 2025 | 시각 교란으로 만든 부정 예시, 궤적 수준 자기수정 | GPT-4o 주석 SFT/DPO ✗ | 시각 교란 아이디어(진단용) |
| **Reflection-V** — Jian et al. ([2509.12132v1](https://arxiv.org/abs/2509.12132v1)) | PR, EMNLP 2025 | 정답일 때만 시각 주의 보상 | 72B/QwQ cold start ✗ | **보조 보상은 정답일 때만** 원칙. 장문에서 시각 주의가 20~30%로 감소한다는 관측 |
| **SVR-R1** — Wu et al. ([2607.10966v1](https://arxiv.org/abs/2607.10966v1)) | PP | VLM이 YES/NO 자기 판정 후 재시도, 최종 응답만 보상 | gpt-oss-120b judge ✗ | 가장 가까운 VLM 재시도 구조. 첫 시도와 나중 정답을 구분하지 않는 약점 |
| **StructReward** — Li et al. ([2608.08326v2](https://arxiv.org/abs/2608.08326v2)) | PP | **Qwen3-VL-4B**에서 정답일 때만 과정 보상 | 5K SFT·강한 LLM 오답 생성 ✗ | 같은 모델의 RL 여지(단, SFT 혼재로 RL 효과 분리 불가) |

### 4.4 RLVR 알고리즘과 반드시 둘 대조군

| 연구 | 상태 | 우리 설계에 주는 것 |
|---|---|---|
| **GRPO / DeepSeekMath** — Shao et al. ([2402.03300v3](https://arxiv.org/abs/2402.03300v3)) | PP | 그룹 상대 advantage, k3 KL 추정량 (식 출처) |
| **DeepSeek-R1** — DeepSeek-AI ([2501.12948v2](https://arxiv.org/abs/2501.12948v2)) | **PR, Nature 645 (2025)** | R1-Zero: SFT 없이 규칙 보상 RL. "신경망 보상모델은 해킹에 취약" → 프로그램 보상 근거. (R1 본편은 SFT 사용이므로 R1-Zero만 인용) |
| **Dr. GRPO** — Liu et al. ([2503.20783v2](https://arxiv.org/abs/2503.20783v2)) | PP (COLM 2025는 저자 BibTeX에만) | 길이·난도 편향 제거: 평균만 빼고 std로 나누지 않음, 상수 정규화. **Qwen 계열 이득은 기존 행동 유도일 수 있다** |
| **DAPO** — Yu et al. ([2503.14476v2](https://arxiv.org/abs/2503.14476v2)) | PR, NeurIPS 2025 | clip-higher(0.2/0.28), dynamic sampling, soft overlong 감점 |
| **RLOO** — Ahmadian et al. ([2402.14740v2](https://arxiv.org/abs/2402.14740v2)) | PR, ACL 2024 | leave-one-out baseline (평균 중심화와 동치) |
| **GSPO** — Zheng et al. ([2507.18071v2](https://arxiv.org/abs/2507.18071v2)) / **SAPO** ([2511.20347v2](https://arxiv.org/abs/2511.20347v2)) | PP | 시퀀스 비율(대안 비교군). SAPO는 Qwen3-VL 학습에 쓰인 알고리즘 |
| **Spurious Rewards** — Shao et al. ([2506.10947v2](https://arxiv.org/abs/2506.10947v2)) | PR, ICML 2026 | Qwen2.5-Math가 **무작위 보상으로도 +21.4점**. → **무작위 보상 대조군 필수**. 메커니즘은 clipping 편향이므로 μ=1(ρ≡1)에서는 약해짐 |
| **Does RL Really Incentivize…** — Yue et al. ([2504.13837v5](https://arxiv.org/abs/2504.13837v5)) | PR, NeurIPS 2025 Oral | RL은 작은 k에서 이기고 큰 k에서는 base가 이김 → **pass@k 곡선 필수**, 객관식 찍기 제거 |
| **Training–inference mismatch** — Yao et al. (NeurIPS'25 OPT WS) / Qi et al. ([2510.26788v1](https://arxiv.org/abs/2510.26788v1)) | WS / PP | vLLM과 학습 엔진의 확률 차이 → 시퀀스 수준 importance mask(임계값은 G7 실측으로 고정, §10.3). BF16 LoRA 장기 학습 붕괴 보고 |

### 4.5 VLM RL·지각 보상·Vision encoder 학습

| 연구 | 상태 | 요점과 우리 사용 |
|---|---|---|
| **VLAA-Thinker** — Chen et al. ([2504.11468v1](https://arxiv.org/abs/2504.11468v1)) | PR? (TMLR 2025, 2차 출처) | **RL 전 SFT가 성능을 8.9~12.7% 낮춤** → SFT 없는 RL 선택의 근거 (데이터·보상모델은 사용 안 함) |
| **VisualThinker-R1-Zero** ([2503.05132v2](https://arxiv.org/abs/2503.05132v2)) | PP | Instruct 모델 RL은 피상적 추론, 길이 보상은 반복 해킹 → 길이·반성 문구 보상 금지 |
| **Visual-RFT** ([2503.01785v1](https://arxiv.org/abs/2503.01785v1)) | PR, ICCV 2025 | 프로그램 지각 보상(IoU)의 선례 |
| **MM-Eureka** ([2503.07365v2](https://arxiv.org/abs/2503.07365v2)) / **LMM-R1** ([2503.07536v2](https://arxiv.org/abs/2503.07536v2)) | PP | 0-분산 그룹 제거 없으면 붕괴. KL 1e-3. 다중모달 RL이 텍스트 추론을 낮출 수 있음 → 텍스트 retention 검사 |
| **ThinkLite-VL** ([2504.07934v3](https://arxiv.org/abs/2504.07934v3)) | PR, NeurIPS 2025 | 난도 선별이 양보다 중요 → Phase A pass rate 선별 |
| **NoisyRollout** ([2504.13055v4](https://arxiv.org/abs/2504.13055v4)) | PR?, NeurIPS 2025 | 노이즈 이미지 rollout(후속 탐색 보조 후보) |
| **DeepEyes** ([2505.14362v3](https://arxiv.org/abs/2505.14362v3)) | PR?, ICLR 2026 | cold start 없는 RL, **보조 보너스는 정답일 때만** |
| **Perception-R1** — Xiao et al. ([2506.07218v3](https://arxiv.org/abs/2506.07218v3)) | PR, ICLR 2026 | 정답 보상 RL만으로는 지각이 유의하게 바뀌지 않음(McNemar) → 지각 probe를 McNemar로 검정. Gemini 주석·LLM judge ✗ |
| **Vision-SR1** — Li et al. ([2508.19652](https://arxiv.org/abs/2508.19652), v2=ICLR 2026) | PR, ICLR 2026 | 자기 지각 문장만으로(이미지 제거) 다시 답하게 해 지각 충분성 보상. v2는 보상별 advantage 분리. v1 cold-start SFT ✗ → **슬롯 정답이 없을 때의 귀인 probe** (§8.3) |
| **ViCrit** ([2506.10128v1](https://arxiv.org/abs/2506.10128v1)) | PR, NeurIPS 2025 | 캡션 속 오류 구절 찾기 RL, Vision 포함 전체 파라미터 학습으로 양의 전이. GPT-4 교란 ✗ → 프로그램 교란 변형만 후보 |
| **VisPlay** ([2511.15661v2](https://arxiv.org/abs/2511.15661v2)) | PR, CVPR 2026 | 다수결 pseudo-label은 정확도가 72→61%로 하락 → 자기 생성 라벨 사용 안 함 |
| **PROPA** ([2511.10279v1](https://arxiv.org/abs/2511.10279v1)) | PP | MCTS 과정 보상 + SFT 교차 ✗ → 제외 |
| **Asymmetric Optimization** — Wu et al. ([2605.29496v2](https://arxiv.org/abs/2605.29496v2)) | PP | 정답 보상과 지각의 상관 0.34~0.43로 약함. 지각 보상 α=0.5로 최대 +6. 자기 보상은 N=4에서 상관 65.8 → **자기 지각 probe는 N≥4**. 지각 토큰 가중은 RL 불안정 |
| **PAPO** ([2507.06448](https://arxiv.org/abs/2507.06448)) | PR?, ICLR 2026 | GRPO 오류 200개 수동 분석에서 **67%가 지각 오류**. 마스크 이미지 KL 보조손실(붕괴 위험) → 후속 비교군 |
| **VPPO** ([2510.09285v2](https://arxiv.org/abs/2510.09285v2)) | PR, ICLR 2026 | 토큰별 시각 의존도로 gradient 집중 → 라우팅 진단 후보 |
| **Chart-RL** — Bai et al. ([2604.03157v1](https://arxiv.org/abs/2604.03157v1)) | PR?, KDD 2026 | **Qwen3-VL-4B, SFT 없이 단일 24GB LoRA RL이 실행됨** (실행 가능성 근거만; LLM judge ✗, G=2) |
| **Qwen3-VL Technical Report** ([2511.21631v2](https://arxiv.org/abs/2511.21631v2)) | PP | 4B는 SigLIP2-Large(300M), 2층 MLP merger, DeepStack이 ViT 3개 층 특징을 LLM 첫 3층에 더함. 이미 SFT·증류·RL(SAPO) 완료된 모델 → **향상 여지가 작을 수 있음** |

**Vision encoder를 RL로 학습하는 것에 대한 근거 요약.** MAYE([2504.02587v1](https://arxiv.org/abs/2504.02587v1), PP)는 ViT·connector 학습이 이득 없이 느리다고 했고, From Seeing to Thinking([2605.20177v1](https://arxiv.org/abs/2605.20177v1), ICML 2026)은 동결 여부의 차이가 작다고 했으며, Same Reward, Different Skills([2610.01908v1](https://arxiv.org/abs/2610.01908v1), PP)는 encoder를 오래 열면 grounding 정확도가 base 아래로 떨어졌다고 보고했다. 반대로 ViCrit은 전체 파라미터 학습에서 양의 전이를 보였다. **결론: 기본 경로는 Merger+LLM, Vision은 검증된 시각 오류가 있는 문제에서만 여는 것이 문헌과 맞고, 그 효과는 대조군으로 측정해야 한다.**

### 4.6 객관식 피드백 누설

| 연구 | 상태 | 주장 |
|---|---|---|
| **ReVeL** ([2511.17405v2](https://arxiv.org/abs/2511.17405v2)) | PP | 객관식 RFT는 답 신호를 흘리고 찍기를 조장, 객관식 점수가 최대 20%p 부풀려짐 |
| **Rethinking MCQ for RLVR** — Guo et al. ([2603.12826](https://arxiv.org/abs/2603.12826)) | PR? (Findings ACL 2026) | 객관식 RLVR은 찍기·소거를 유도, 학습/평가 보기 수 불일치가 해로움 |

4지선다에서 "틀렸다"를 들으면 무작위로도 두 번째 시도 성공률이 1/3, 두 번 안에 1/2이 된다. 그래서 재풀이 학습은 **열린 답 위주**, 객관식은 **보기 8~10개**(MMMU-Pro와 같은 10개 권장)로 하고 소거 기준선을 함께 보고한다.

### 4.7 빌리는 부품 / 제외 부품 한눈에

| 빌린다 (규칙 위반 없음) | 출처 |
|---|---|
| 그룹 평균 baseline, std 미정규화, 상수 길이 정규화 | GRPO, Dr. GRPO, RLOO |
| clip 0.2/0.28, 0-분산 그룹 건너뛰기, soft overlong(비교군) | DAPO, VL-Rethinker |
| 같은 입력을 공유하는 재시도 그룹 | MM-ReCoder, S²R |
| 재시도 성공 시 반성 토큰 강화 | Reflect, Retry, Reward |
| 검증 가능한 판정에만 보상 | S²R |
| 보조 보상은 정답일 때만 | DeepEyes, Reflection-V, StructReward |
| 자기 지각 문장만으로 재답 → 지각 충분성 probe (N≥4) | Vision-SR1, Asymmetric Optimization |
| 시퀀스 수준 importance mask | Yao et al., Qi et al. |
| 무작위 보상·형식 보상·blind 대조, pass@k | Spurious Rewards, Yue et al., Same Reward Different Skills |
| 난도 기반 문제 선별 | ThinkLite-VL, DAPO |

| 제외한다 | 이유 |
|---|---|
| cold-start SFT, 반성문 SFT, rethinking SFT 손실 | SFT |
| GPT/Gemini/Qwen-235B 데이터·비평·judge, 교사 궤적(LUFFY) | 상위 모델 |
| 자기 생성 정답 풀이 SFT(STaR/RFT) | SFT |
| 다수결 pseudo-label | 정답 보장 없음 |
| "새로운 답" 보너스, 할인 없는 "아무 시도나 맞으면 +1" | 객관식 소거 조장 |
| 지각 토큰 가중 | RL 불안정 보고 |

### 4.8 인용 가능성과 신규성 표현

- **안전하게 인용 가능(PR):** SCoRe, RISE, S²R, LUFFY, NSR, Huang et al., Kamoi et al., Reflexion, Self-Refine, VL-Rethinker, MM-ReCoder, Sherlock, Reflection-V, DeepSeek-R1(Nature), DAPO, RLOO, Spurious Rewards, Yue et al., Visual-RFT, ThinkLite-VL, Perception-R1, Vision-SR1(v2), ViCrit, VisPlay, VPPO, From Seeing to Thinking.
- **"preprint"라고 명시하고 인용:** Reflect-Retry-Reward, UFO(워크숍판 별도), Multi-attempt RL, Self-rewarding correction, GRPO/DeepSeekMath, Dr. GRPO, GSPO, SAPO, MM-Eureka, LMM-R1, Asymmetric Optimization, StructReward, SVR-R1, PROPA, ReVeL, Qwen3-VL TR, Same Reward Different Skills, MAYE.
- **arXiv 코멘트에만 게재 표시(최종 보고 전 proceedings 확인):** SRPO, LTE, Critique-GRPO, NoisyRollout, DeepEyes, PAPO, Chart-RL, VLAA-Thinker(TMLR), Rethinking MCQ.
- **인용 시 금지 표현:** "이 논문들이 우리 방법의 효과를 보장한다", "Vision encoder 결함이 증명됐다", "우리 방법이 SCoRe/MM-ReCoder와 동일하다"(우리는 정오 피드백을 쓰는 verifier-assisted 설정), "StructReward로 4B가 70.3%가 된다"(SFT 혼재·프로토콜 상이).
- **신규성:** 이번 조사 범위에서는 **검증된 오류 유형에 따라 Vision/Merger/LLM 업데이트 경로를 문제 단위로 바꾸는 RL**과 **오류 원인 자기 진단을 프로그램 검증 원인과의 일치로 보상하는 VLM RL**을 찾지 못했다. 검색 범위가 제한적이었으므로 "우리가 찾은 범위에서 선례를 발견하지 못했다"로만 쓰고, 최종 보고 전에 추가 문헌 조사를 한다.

## 5. 사용자 5단계 설계 → 구현 결정 대응표

사용자가 제안한 흐름을 그대로 유지하되, 각 단계에서 **무엇이 학습 신호이고 무엇이 기록인지**를 명확히 한 결과다.

| 사용자 단계 | 사용자 문장(요지) | v2 구현 | 바꾼 점과 이유 |
|---|---|---|---|
| 1 | 문제를 풀고 정답과 비교해 맞았는지 확인 | turn-1 그룹: 같은 문제를 `n1=8`번 독립 풀이 → 독립 verifier가 `strict-final-v1`로 채점 (§7, §9.1) | 한 번이 아니라 8번 푸는 이유: GRPO는 같은 문제의 여러 풀이를 서로 비교해 "더 나은 풀이"의 확률을 올린다. 정답 값은 모델에 보여주지 않는다 |
| 2 | 틀리면 풀이를 남겨두고, 풀이·틀린 답·문제를 보고 틀렸음을 학습하고, 아래에 다시 풀어 채점 | 실패 풀이를 문맥에 그대로 둔 상태 `s1`에서 `n2=4`번 재풀이(`Error check:` + 새 풀이). turn-1 실패 샘플은 자기 그룹 평균보다 낮은 advantage(= "틀렸음"의 학습), 재풀이는 별도 그룹에서 채점·비교 (§7.2) | "틀렸음을 학습"을 **음(−)의 advantage**로 구현한다. 틀린 풀이를 정답처럼 따라 쓰게 하는 손실은 없다 |
| 3 | 또 틀리면 1·2번 풀이와 틀린 답으로 다시 학습 | `max_attempts=3` 실험군 `ALG-NB3`: 두 실패 풀이를 모두 문맥에 둔 `s2`에서 재풀이 그룹 (§7.3) | 비용 때문에 pilot 기본은 2회. 3회는 같은 조건의 비교군으로 효과를 측정. 무제한 반복은 V8의 장문 반복 문제 때문에 금지 |
| 4 | 맞혔을 때 앞선 풀이에서 이미지를 잘못 본 것인지 등 원인을 모델이 오답노트로 작성 | (L1) host가 슬롯 검증으로 **검증된 원인** 기록, (L3) 모델이 사후 진단 노트를 쓰고 L1 원인과 일치하면 보상 `R_diag` (§8, §9.4) | 모델 노트를 그대로 믿지 않는다. 검증 가능한 원인일 때만 노트에 보상을 준다 = "정확히 자기 오류를 진단하는 능력"을 RL로 훈련 |
| 5 | 노트대로 Vision 문제였으면 Vision+Merger+LLM, Vision이 항상 맞았으면 Merger+LLM만 학습 | `MOD-RV`: 검증 원인에 따라 문제 단위로 Merger 입력 detach 훅을 켜고 끔 (§8.4, §11.3). 대조군 `MOD-FML`, `MOD-FVML`, `MOD-RR`, 후속 `MOD-RS`(자기 진단 기반) | 라우팅 근거를 모델 노트 대신 검증 원인으로 바꿨다. 사용자 원안(노트 기반)은 `MOD-RS`로 보존해 비교한다 |

### 5.1 "틀렸음을 학습한다"의 정확한 의미

turn-1 그룹이 (설명을 위해 4개만 적으면; 실제 `n1=8`) `[정답, 오답, 오답, 오답]`이면 advantage는 `[+0.75, −0.25, −0.25, −0.25]`다. 손실을 줄이는 방향은 정답 풀이 토큰의 확률을 올리고 **오답 풀이 토큰의 확률을 내리는 것**이다. 이것이 정답을 보여주지 않고 "이 풀이는 틀렸다"를 가중치에 반영하는 방법이다. 음의 보상만으로도 추론 성능이 오른다는 보고(Zhu et al., 2025, Negative Sample Reinforcement)가 이 해석을 뒷받침한다(§4).

재풀이 그룹에서는 같은 오답 문맥 아래 "이미지를 다시 보고 고친 풀이"와 "같은 실수를 반복한 풀이"가 직접 비교된다. 그래서 "자기 오답을 보고 어떻게 고쳐야 하는가"가 학습된다.

### 5.2 왜 "맞힐 때까지 반복"이 아닌가

- 정답 여부만 반복해서 알려주면 객관식은 보기를 하나씩 지워 결국 맞힌다. 이것은 문제 해결 능력이 아니라 소거다.
- 무제한 반복은 V8에서 관측된 장문 반복과 같은 실패 습관을 학습시킬 위험이 있다.
- 평가(MMMU)에서는 채점 피드백이 없다. 우리가 원하는 것은 **첫 시도 정확도**다. 재풀이 학습은 그 수단이지 목표가 아니다.

### 5.3 SFT가 아닌 이유를 한 문장으로

이 파이프라인의 모든 gradient는 **모델이 스스로 생성한 토큰의 log-probability × 검증 보상으로 계산한 advantage**에서만 나온다. 정답 풀이·상위 모델 풀이·자기 성공 풀이를 정답 문장으로 따라 쓰게 하는 cross-entropy 항은 없다. 데이터의 정답은 verifier 안에서 보상 계산에만 쓰인다(= 검증 가능한 보상 기반 RL, RLVR). "정답 정보가 전혀 없는 비지도학습"은 아니라는 점도 함께 밝힌다.

## 6. 전체 아키텍처: 네 개의 경계와 실행 배치

### 6.1 네 개의 경계

| 경계 | 왼쪽 | 오른쪽 | 강제 방법 |
|---|---|---|---|
| **B1 데이터 경계** | MMMU/Pro 원문·이미지 (오염 검사 프로세스만 읽음) | 학습 데이터 | 오염 검사는 별도 명령(`notebook_rl.cli audit`)에서만. 학습·생성 프로세스는 benchmark 경로 open 시 예외 (§1.3) |
| **B2 actor/verifier 경계** | actor 입력: 이미지·질문·보기·답 형식 | verifier 금고: 정답·슬롯 사실·solver·required_slots | 파일 분리 + loader allowlist + audit hook. actor 레코드에 금지 키가 있으면 로드 실패 |
| **B3 rollout/update 경계** | 한 step의 트리 생성 (정책 버전 고정) | optimizer step | `policy_version` assert, 트리 원장은 append-only, update는 닫힌 트리만 사용 |
| **B4 학습/평가 경계** | 학습 (외부 train만) | MMMU/Pro 평가 (동결 checkpoint만) | 평가 스크립트는 export된 고정 checkpoint 경로만 입력. 학습 코드에 MMMU 로더 없음 |

### 6.2 한 GPU 안의 실행 배치 (A40 48GB 기준)

```text
┌──────────────────────── 하나의 Python 프로세스 (VLLM_ENABLE_V1_MULTIPROCESSING=0) ────────────────────────┐
│                                                                                                           │
│  HF 학습 정책 (BF16 base 8.27GiB + LoRA FP32 + Merger FP32 사본)      vLLM 0.11.0 rollout 엔진             │
│   ├─ LLM LoRA r16 (252 Linear)                                        ├─ 같은 base 가중치 사본 (8.27GiB)   │
│   ├─ 4 Merger: PEFT modules_to_save (FP32 학습 사본 + BF16 원본)     ├─ KV cache (명시 크기, 약 10GiB)    │
│   ├─ (MOD-FVML/RV/RR) Vision LoRA r16 (48 Linear: attn.qkv, proj)    ├─ prefix cache OFF, mm cache 0       │
│   └─ reference = disable_adapter() 문맥 → 정확한 F0                   └─ temperature 1, top_p 1, top_k 0   │
│                                                                                                           │
│  step k: [vLLM] 트리 생성 → [CPU] verifier·귀인·라우팅 → [HF] π_old, π_ref, 정책경사 → optimizer          │
│          → [HF→vLLM] LoRA를 out-of-place 병합한 가중치 + Merger 사본 push → probe logprob 일치 검사        │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
       ▲ CPU 프로세스(별도): verifier는 같은 프로세스의 순수 함수로 호출하되, 금고 파일은 verifier 모듈만 open
```

- 48GB에서는 HF 학습 상태(추정 피크 ~16GiB)와 vLLM(추정 ~20GiB)을 동시에 올릴 수 있어 sleep mode가 필요 없다(§19 추정치, smoke에서 실측).
- 24GB(3090)는 학습과 rollout을 번갈아 CPU로 내리는 ping-pong 방식이 필요해 v2 기본 경로에서 제외한다.
- vLLM에는 LoRA 기능을 쓰지 않는다(vLLM 0.11은 언어 모델에만 LoRA 적용). 매 step 병합된 전체 가중치를 push한다.

### 6.3 파일·저장소 흐름

```text
로컬 WSL (CPU)                                    GitHub MMDL_test (feature branch)          RunPod A40
────────────────                                  ─────────────────────────────────          ──────────
fine_tuning/notebook_rl/*.py ── sync_test_repo ──► exact commit SHA ── clean clone ──────────► runpod_setup.sh
data env: 생성기 → actor/verifier/lineage 파일                                                 (vllm lock, 테스트, doctor, 모델 해시)
  └─ 오염 검사 v2 (index는 RunPod에서 재구축)                                                  notebook_rl controller:
  └─ manifest + SHA256 ─────────── 비공개 전송(scp) ─────────────────────────────────────────►  smoke → phase_a → train arms
                                                                                                 → export → eval
로컬 회수 ◄────────────────── rollouts·notebook·checkpoint·metrics (SHA 검증) ◄──────────────── 결과
→ 검증 후 소유 Pod 삭제
```

데이터(이미지·정답)는 GitHub에 올리지 않는다. 코드·설정·테스트·문서만 MMDL_test에 게시한다.

## 7. 풀이 → 채점 → 재풀이: 트리형 롤아웃과 턴 단위 그룹

### 7.1 왜 10/16 v1의 "episode 보상"을 바꾸는가

v1 설계는 한 episode(최대 2회 시도) 전체에 `1.0 / 0.5 / 0.0`을 주고 G=4 episode를 한 그룹으로 정규화했다. 이 방식에는 v1 문서 §8.2가 스스로 지적한 **신용 할당 문제**가 있다. 네 episode가 모두 첫 시도에서 틀리고 일부만 두 번째에 맞으면, 틀린 첫 시도의 토큰까지 양(+)의 advantage를 받는다. 또 두 번째 시도는 서로 다른 첫 풀이를 문맥으로 가지므로 "같은 상태에서 뽑은 여러 행동"이라는 GRPO의 전제가 깨진다.

v2는 이를 **상태를 공유하는 샘플끼리만 그룹을 만드는 트리형 롤아웃**으로 바꾼다. 같은 문제의 첫 시도들은 모두 같은 입력 `s0`에서 뽑았으므로 한 그룹이다. 특정 오답 풀이 하나를 문맥에 고정한 뒤 그 아래에서 여러 번 다시 풀게 하면, 그 재풀이들도 같은 입력 `s1`을 공유하므로 한 그룹이다. 이 구조의 이점은 다음과 같다.

| 성질 | v1 episode 보상 | v2 트리형 턴 그룹 |
|---|---|---|
| GRPO "공유 상태" 전제 | 두 번째 시도에서 깨짐 | 모든 그룹에서 성립 |
| 틀린 첫 시도가 받는 신호 | episode가 나중에 맞으면 +가 될 수 있음 | 항상 자기 그룹 평균보다 낮음(=틀렸다는 학습) |
| 첫 시도를 일부러 틀릴 유인 | 0.5<1.0 설계로 억제 | 구조적으로 없음: 첫 시도 보상은 첫 시도 정오만 반영 |
| 재풀이 학습 신호 | episode 결과에 섞임 | 같은 오답 문맥에서 맞힌 재풀이 vs 못 맞힌 재풀이의 직접 대비 |
| 첫 시도 학습과 단일턴 RLVR의 관계 | 혼합 그룹 50%가 별도 필요 | turn-1 그룹이 그대로 단일턴 RLVR |

같은 오답 문맥을 공유하는 재풀이 그룹은 MM-ReCoder의 shared-first-turn 대조와 같은 발상이다(§4 참고). 우리 구현은 이를 보조 실험이 아니라 **기본 롤아웃 구조**로 둔다.

### 7.2 한 문제에서 만들어지는 트리

```text
문제 x (이미지 I, 질문 q, 보기 C)          정책 θ_old 고정 (이 트리 생성 중 가중치 변경 없음)
│
├─ s0 = user(I, q, C, mode 지시)
│   ├─ a1^(1)  → verifier: CORRECT          ┐
│   ├─ a1^(2)  → verifier: INCORRECT (35)    │  turn-1 그룹 G1(x): n1 = 8개
│   ├─ a1^(3)  → verifier: INCORRECT (42)    │  보상 r = 1,0,0,1,0,0,0,0 → 평균 0.25
│   ├─ a1^(4)  → verifier: CORRECT           │  advantage = +0.75 (정답 2개), −0.25 (나머지 6개)
│   ├─ a1^(5..8) → INCORRECT / INVALID_NO_FINAL ┘
│
├─ 실패한 첫 시도 중 최대 M1 = 2개를 결정적으로 선택(서로 다른 오답 우선): a1^(2), a1^(3)
│   ├─ s1^(2) = s0 + a1^(2) + 피드백("INCORRECT, 정답 비공개")
│   │   ├─ a2^(2,1..4) → verifier            turn-2 그룹 G2(x,2): n2 = 4개
│   └─ s1^(3) = s0 + a1^(3) + 피드백
│       ├─ a2^(3,1..4) → verifier            turn-2 그룹 G2(x,3): n2 = 4개
│
├─ (max_attempts = 3일 때만) turn-2에서 실패한 풀이 중 문제당 최대 M2 = 1개 선택
│   └─ s2 = s1 + a2 + 피드백 → a3^(1..4)     turn-3 그룹 G3(x): n3 = 4개
│
└─ (진단 과제가 켜진 단계에서만) 결국 맞힌 경로마다, 검증된 오류 귀인이 결정 가능한 경우
    └─ s_d = s0 + 모든 시도 + "마지막만 정답" 고지 → d^(1..4)   진단 그룹 D(x): n_d = 4개
```

- 모든 그룹은 **같은 입력 토큰열을 공유하는 샘플**로만 구성된다. 그룹 간 보상을 함께 정규화하지 않는다.
- turn-1이 전부 정답이면 재풀이 그룹은 생기지 않는다. turn-1이 전부 오답이면 turn-1 advantage는 0이지만 재풀이 그룹에서 신호가 생길 수 있다. 이것이 "다 틀린 어려운 문제에서도 배울 수 있게 한다"는 사용자 아이디어의 수학적 실현이다.
- 실패 시도 선택은 결정적이어야 한다. 규칙: (1) 서로 다른 추출 답을 가진 실패를 우선, (2) 같은 순위면 `sha256(f"{x.id}:{policy_step}:{sample_index}")` 오름차순. 선택 결과와 미선택 실패 수를 로그에 남긴다. 결과를 보고 "좋은 오답"을 사람이 고르지 않는다.
- `INVALID`(답 없음·복수 답·출력 상한)도 재풀이 대상이 될 수 있으나 피드백 문장이 다르다(§7.4).
- `SYSTEM_ERROR`, `UNVERIFIABLE`, `CONTEXT_BLOCKED`가 하나라도 나온 그룹은 업데이트에 쓰지 않는다. 0점으로 바꾸지 않는다. 원인과 개수를 로그한다.

### 7.3 기본 수치와 비용 상한 (사전 고정값, 실측 후 protocol revision으로만 변경)

| 항목 | 기본값 | 근거/비고 |
|---|---:|---|
| `n1` (turn-1 그룹 크기) | 8 | 문헌 다수가 G=8 사용(Dr. GRPO, VisualThinker, MM-Eureka). G=2는 신호가 약함(Chart-RL) |
| `M1` (재풀이로 확장할 실패 첫 시도 수/문제) | 2 | 서로 다른 오류 문맥 2개 |
| `n2` (재풀이 그룹 크기) | 4 | |
| `max_attempts` | 2 (pilot), 3 (ablation `ALG-NB3`) | 사용자 3단계(1·2번 풀이를 함께 보고 다시 풀기)는 3에서 구현 |
| `M2` (turn-3 확장 수/문제) | 1 | 비용 상한 |
| `n3` | 4 | |
| 진단 그룹 `n_d` | 4 | 진단 단계에서만 |
| turn당 `max_new_tokens` | 3072 (Phase A 실측 후 확정) | 아래 7.6 |
| 문맥 상한 | 16384 tokens | 이미지 1장 최대 ~3.9K 토큰(4,014,080px ÷ 32², 다중 이미지는 합산) + 3턴 × 3K + 피드백. 최대 해상도 이미지가 2장 이상이면 turn-3 전에 초과할 수 있다. 초과 시 `CONTEXT_BLOCKED`, 자동 절단 없음. **재풀이 학습 문제는 이미지 ≤2장으로 제한**하고 초과 문제는 제외하며 그 수를 보고한다(pilot 출처는 모두 1장) |
| 문제당 최대 생성 수 | 8 + 2×4 (+ turn-3 4 + 진단 4) = 16~24 | 최악의 경우. 실제는 정답률에 따라 감소 |

### 7.4 피드백 문장 (정답 값은 절대 넣지 않는다)

피드백은 **user turn**으로 붙인다. 모델이 생성한 토큰이 아니므로 loss mask는 0이다. 정확한 문자열은 `fine_tuning/notebook_rl/prompts_v1.yaml`에 고정하고 SHA256을 protocol에 기록한다(부록 A에 전문).

| 이전 시도 상태 | 피드백 요지 | 공개되는 정보 |
|---|---|---|
| `INCORRECT` | "제출한 답 `<추출답>`은 검증 결과 오답이다. 정답은 공개하지 않는다. 먼저 Error check를 쓰고, 이미지를 다시 보고 새 풀이를 아래에 써라." | 1 bit(오답) + 모델 자신의 답 |
| `INVALID_NO_FINAL` / `INVALID_MULTIPLE` / `TRUNCATED` | "정확히 하나의 최종 답이 없어 정오를 판정하지 않았다(사유 코드). 풀이를 점검하고 하나의 최종 답으로 끝내라." | 형식 실패 사유만 |
| (진단 과제) | "시도 k가 검증 결과 정답이고 앞선 시도들은 오답이었다. 시도 1의 주된 원인을 고르고 처음 틀린 항목을 적어라." | 각 시도의 정오 |

- 객관식에서 "B는 오답"이라는 정보는 남은 보기 중에서 고를 확률을 높인다(소거 효과). 따라서 재풀이 학습 문제는 **열린 수치·짧은 문자열 답을 기본(목표 60% 이상)**으로 하고, 객관식은 **보기 8~10개**를 사용한다. 객관식 재풀이 정확도는 항상 소거 기준선 `1/(k−1)`과 함께 보고한다.
- "Re-check the image" 문장은 시각 재관찰을 유도하는 지시다. 이미지를 두 번째로 삽입하지 않는다(토큰 낭비·위치 인코딩 변화). 첫 user turn의 이미지가 문맥에 계속 남는다.

### 7.5 모델이 써야 하는 응답 형식

학습용 prompt mode는 두 가지다. 문제마다 `sha256(problem_id)`의 짝/홀로 고정 배정한다(학습 중 바꾸지 않음).

**mode `obs` (관찰→풀이→답).** 시각 사실을 검증하고 오류 원인을 귀인하기 위한 형식이다.

```text
Observed facts:
<family별 고정 관찰 형식의 줄들, 예: bar[Mar] = 35>
Solution:
<풀이>
Final answer: <답 하나>
```

관찰 형식은 **이미지 종류(family)마다 고정**이며 질문과 무관하다. 예: 막대그래프는 "모든 막대를 `bar[<라벨>] = <값>`으로", 표는 "필요하다고 생각하는 셀을 `cell[<행>][<열>] = <값>`으로". 어떤 슬롯이 이 질문에 필요한지(`required_slots`)는 verifier만 안다. 질문별로 "이 칸을 읽어라"라고 알려주면 지각 과제를 쉽게 바꾸는 힌트가 되므로 금지한다.

**mode `plain` (P0 호환).** MMMU 평가 프롬프트 P0와 같은 질문·보기 직렬화에 마지막 한 줄 지시만 더한다.

```text
{P0 텍스트 그대로}
End your response with exactly one line: Final answer: <answer>
```

`plain`을 섞는 이유는 G6 10K 실험에서 짧은 학습 target이 MMMU 응답을 **풀이 없는 한 줄 답(FL 중앙값 8토큰, 한 줄 `LETTER. 보기` 51/60)**으로 바꾼 형식 표류가 관측됐기 때문이다(§3.2). V8 파서는 `Final answer` 선언을 가장 먼저 인식하므로, 이 한 줄 습관은 평가 파서와 충돌하지 않는다. 그래도 학습 전후 P0 형식 준수율(NO_PARSE, length)을 매 checkpoint마다 외부 dev에서 측정한다.

재풀이(turn ≥2) 응답은 맨 앞에 `Error check:` 블록을 추가한다.

```text
Error check:
- <틀렸을 수 있는 관찰/규칙/계산과 다시 확인할 것>
Observed facts:        ← obs mode일 때만
...
Solution:
...
Final answer: <답 하나>
```

이 `Error check` 블록이 **모델이 직접 쓰는 오답노트**이며, 재풀이가 맞으면 그룹 평균보다 높은 advantage를 함께 받는다. 노트 자체에 별도 점수를 주지 않는다(장식적 반성문 생성 방지). Reflect, Retry, Reward(Bensal et al., 2025)가 "재시도 성공 시 반성 토큰에만 보상"하는 구조와 같은 원리이며, 기본안은 노트와 새 풀이를 한 행동으로 함께 강화한다. 노트 토큰만 강화하는 원형은 비교군 `ALG-NB-noteonly`로 둔다(§18).

### 7.6 출력 길이 상한 정하기 (Phase A에서 실측)

V8 32K 평가에서 상한 종료 90건 중 85건이 오답이었다. RL은 상한 종료를 보상0으로 처리하므로 짧고 완결된 풀이를 선호하도록 압력이 생긴다(DAPO의 overlong 처리와 같은 방향). 그러나 상한이 너무 낮으면 어려운 문제의 긴 추론까지 잘라 학습시킨다.

결정 절차(사전 고정):
1. Phase A(가중치 고정 F0 롤아웃)에서 turn-1 `max_new_tokens=8192`로 family별 길이 분포를 측정한다.
2. `max_new_tokens(turn)` = F0 응답 중 **완결(final answer 있음) 응답 길이의 95번째 백분위수**를 256 단위로 올림, 단 하한 2048·상한 4096.
3. 이 값에서 F0의 절단률이 family별로 10%를 넘으면 그 family는 pilot에서 제외하거나 난도를 낮춘 generator 설정으로 교체한다. 결과를 보고 개별 문제를 골라내지 않는다.
4. 평가(MMMU 8K)와 훈련 상한이 다르다는 사실을 보고서에 명시한다.

### 7.7 상태 코드와 처리

| 코드 | 의미 | 보상 | 업데이트 |
|---|---|---:|---|
| `CORRECT` | 단일 최종 답이 typed gold와 일치 | 1 | 사용 |
| `INCORRECT` | 단일 최종 답이 있으나 불일치 | 0 | 사용 |
| `INVALID_NO_FINAL` | `Final answer:` 줄 없음 | 0 | 사용 |
| `INVALID_MULTIPLE` | 최종 답 줄이 2개 이상이거나 한 줄에 복수 후보 | 0 | 사용 |
| `TRUNCATED` | finish_reason=length | 0 | 사용 |
| `CONTEXT_BLOCKED` | 이전 시도 포함 문맥이 상한 초과 | — | 그 그룹 생성 안 함 |
| `UNVERIFIABLE` | gold/스키마로 판정 불가(타입 불일치 등) | — | 그룹 제외, quarantine 큐 |
| `SYSTEM_ERROR` | 디코드/CUDA/엔진/검증기 예외 | — | 그룹 제외, 같은 seed로 최대 1회 재시도 후 기록 |

### 7.8 롤아웃 단계의 의사코드

```python
# fine_tuning/notebook_rl/rollout.py 의 계약을 설명하는 의사코드
def build_tree(problem, engine, verifier, cfg, policy_step) -> ProblemTree:
    s0 = prompts.turn1_state(problem.actor, mode=problem.mode)
    g1 = engine.sample(s0, n=cfg.n[1], max_new_tokens=cfg.max_new_tokens[1],
                       seeds=seeds(problem.id, policy_step, turn=1))
    g1 = [verifier.grade(problem.id, a) for a in g1]          # verifier는 problem.id로 private vault 조회
    tree = ProblemTree(problem.id, groups=[Group(state=s0, turn=1, samples=g1)])
    if any_blocking(g1):                                        # SYSTEM_ERROR/UNVERIFIABLE
        return tree.mark_unusable()
    frontier = select_failures(g1, k=cfg.branch[1], key=(problem.id, policy_step))
    for turn in range(2, cfg.max_attempts + 1):
        next_frontier = []
        for failed in frontier:
            s = prompts.retry_state(failed.state, failed.sample, failed.status)
            if tokens(s) + cfg.max_new_tokens[turn] > cfg.context_limit:
                tree.log("CONTEXT_BLOCKED", failed); continue
            g = engine.sample(s, n=cfg.n[turn], ...)
            g = [verifier.grade(problem.id, a) for a in g]
            tree.groups.append(Group(state=s, turn=turn, parent=failed.ref, samples=g))
            next_frontier += select_failures(g, k=1, ...)
        frontier = next_frontier[: cfg.branch.get(turn, 0)]
    return tree
```

생성 중 정책 버전이 바뀌지 않았음을 `assert engine.policy_version == policy_step`로 확인한다. 트리 하나의 모든 원문·token IDs·sampler logprobs·finish reason·verifier 결과·seed를 `rollouts/step_XXXX.jsonl`에 append-only로 저장한다.

## 8. 오답노트: 세 개의 층과 "검증된 오류 귀인"

사용자 설계 4단계("맞힌 뒤 앞선 풀이에서 이미지를 잘못 본 것인지 무엇 때문인지 모델이 오답노트를 작성")와 5단계("오답노트대로 Vision 문제였으면 Vision+Merger+LLM, Vision이 항상 맞았으면 Merger+LLM만 학습")를 구현하는 장이다. 핵심은 **모델이 쓴 노트(주장)와 프로그램이 확인한 원인(증거)을 분리**하고, 학습 경로 선택에는 증거를 쓰며, 모델의 노트 쓰기 능력은 그 증거와의 일치도로 RL 보상을 주어 키우는 것이다.

### 8.1 세 층의 정의

| 층 | 누가 쓰나 | 언제 | 학습에 쓰는 방식 | 저장 |
|---|---|---|---|---|
| L1 검증 기록 (host notebook) | host 코드 | 모든 트리 | 보상·라우팅·진단 정답을 계산하는 **증거** | `notebook/step_XXXX.jsonl` |
| L2 재풀이 노트 (`Error check:`) | 모델 | turn ≥2 응답 앞부분 | 재풀이가 맞으면 그 행동 전체(노트+풀이)가 + advantage | 롤아웃 원문에 포함 |
| L3 사후 진단 노트 | 모델 | 결국 맞힌 경로, 진단 단계에서만 | 노트의 원인 분류가 L1의 검증 원인과 일치하면 보상 (§9.4) | 진단 그룹 원문 |

어떤 층도 **cross-entropy target으로 쓰지 않는다.** L1은 사람이 읽는 감사 기록이자 보상 계산 입력이고, L2·L3은 모델이 스스로 생성한 행동이라 정책경사로만 학습된다.

### 8.2 L1 검증 기록의 필드

```json
{
  "schema": "notebook-rl/host-notebook/v1",
  "problem_id": "gen-chart-v1:chart_bar:train:000017",
  "family": "chart_bar",
  "policy_step": 12,
  "mode": "obs",
  "attempt_paths": [
    {"path": "t1#2", "status": "INCORRECT", "answer": "35",
     "slot_report": {"parsed": 6, "required": 2, "required_correct": 1,
                     "wrong_required": ["bar[Mar]"], "all_correct": 5, "all_total": 6},
     "verified_cause": "VISUAL_MISREAD"},
    {"path": "t1#2>t2#1", "status": "CORRECT", "answer": "53",
     "slot_report": {"parsed": 6, "required": 2, "required_correct": 2,
                     "wrong_required": [], "all_correct": 6, "all_total": 6},
     "verified_cause": null}
  ],
  "oracle_counterfactual": {"enabled": false},
  "problem_cause_set": ["VISUAL"],
  "route": "VISION_ON",
  "route_rule": "verified-any-visual-v1",
  "model_diagnosis": [{"group": "D#1", "cause_pred": "VISUAL_MISREAD", "slot_pred": "bar[Mar]", "reward": 1.25}],
  "gold_visible_to_actor": false
}
```

위 값은 형식 설명용 가상 예시다. `expected` 값(정답 슬롯 값)은 L1에도 원문으로 적지 않고 슬롯 ID와 정오만 남긴다. 슬롯 정답 값은 verifier private vault에만 있다.

### 8.3 검증된 원인(verified cause)을 정하는 규칙

**전제:** 원인 귀인은 슬롯 정답이 있는 문제(`slot_schema` 보유 family, mode `obs`)에서만 가능하다. 슬롯이 없거나 `plain` 모드이면 원인은 `UNKNOWN`이다.

실패한 시도 a 하나에 대한 규칙 `verified-cause-v1`:

1. `status ∈ {INVALID_NO_FINAL, INVALID_MULTIPLE, TRUNCATED}` → `FINALIZATION`.
2. `Observed facts` 블록이 없거나, `required_slots` 중 파싱된 슬롯이 절반 미만 → `UNKNOWN` (관찰을 안 썼다는 사실은 비전 오류의 증거가 아님).
3. `required_slots` 중 하나라도 값이 틀림 → `VISUAL` (세부: 값 슬롯이면 `VISUAL_MISREAD`, 연결·위치 슬롯이면 `VISUAL_RELATION`).
4. `required_slots`가 모두 맞는데 최종 답이 틀림 → `NON_VISUAL` (규칙·지식·계산 중 무엇인지는 v1에서 구분하지 않음).

정답을 맞힌 시도에서도 슬롯 오류를 기록한다(우연한 정답 감지용). 단 원인 라벨은 실패 시도에만 붙인다.

**선택적 보강 — oracle 사실 반사실 검사 (`oracle_counterfactual: true`일 때).** turn-1에 실패가 있는 문제에 대해, 정답 슬롯 값을 "Verified image facts" 텍스트로 넣은 상태에서 n_o=4번 더 풀게 한다. 이 생성은 **진단 전용이며 loss에 들어가지 않는다.** `acc_oracle − acc_turn1 ≥ 0.5`이면 문제 단위 `VISUAL` 증거, `acc_oracle ≤ acc_turn1`이면 `NON_VISUAL` 증거로 쓴다. 슬롯 규칙 결과가 `UNKNOWN`일 때만 보강으로 사용하고, 둘이 충돌하면 `UNKNOWN`으로 둔다. 비용이 늘기 때문에 pilot 기본값은 `false`이고 `MOD-RV` 실험에서 켠 조건과 끈 조건을 비교할 수 있다. 과거 input-recovery 진단(I/T/IT 조건)과 같은 계열의 측정이며, 그 결과를 모듈 결함 인증으로 해석하지 않는다는 원칙도 같다.

**선택적 보강 2 — 자기 지각 충분성 probe (`self_perception_probe: true`일 때, 슬롯 정답이 없는 사람 라벨 데이터용).** Vision-SR1의 방식을 진단 전용으로 쓴다. 실패한 obs/plain 시도의 `Observed facts`(없으면 `Solution` 앞부분 512토큰)를 **이미지 없이** 텍스트로만 주고 같은 정책으로 N=4번 다시 풀게 한다. 이 생성도 loss에 들어가지 않는다.

- 4번 중 정답이 ≥3이면: 모델이 적은 관찰만으로 풀 수 있었다 → `NON_VISUAL` 증거.
- 0이면서 같은 문제의 oracle 사실 반사실이 ≥3이면: 관찰이 부족했다 → `VISUAL` 증거.
- 그 외는 `UNKNOWN`.
- 객관식은 우연 정답이 많으므로 이 probe를 쓰지 않는다(열린 답 문제만). Wu et al.(2026)은 자기 보상의 정답 상관이 N=1에서 32.0, N=4에서 65.8이라고 보고했다. 그래서 N≥4를 고정한다.

### 8.4 문제 단위 라우팅 규칙 `verified-any-visual-v1`

사용자 문장을 그대로 규칙으로 옮긴다.

| 문제 x의 검증 결과 | route | 이 문제의 모든 그룹이 업데이트하는 모듈 |
|---|---|---|
| 어떤 시도(정답 시도 포함)에서든 required slot 오독이 확인됨 ("비전도 문제였다") | `VISION_ON` | Vision LoRA + 4 Merger + LLM LoRA |
| 귀인 가능한 모든 시도에서 required slot이 전부 맞음 ("비전은 항상 맞았다") | `VISION_OFF` | 4 Merger + LLM LoRA |
| 귀인 가능한 시도가 없음 (`UNKNOWN`만) | `VISION_OFF` (기본, 보수적) | 4 Merger + LLM LoRA |
| 슬롯 규칙은 `UNKNOWN`이지만 보강 probe가 `VISUAL` 증거 | `VISION_ON` (규칙 `verified-any-visual-v1+probe`, 별도 비교 조건에서만) | Vision LoRA + 4 Merger + LLM LoRA |

- 라우팅은 **문제 단위**로 정한다. 같은 문제의 turn-1·turn-2·진단 그룹이 모두 같은 경로를 쓴다. 한 그룹 안에서 샘플마다 경로가 다르면 advantage 비교의 의미가 흐려진다.
- 라우팅은 **모델의 노트(L2/L3)가 아니라 L1 검증 원인**으로 정한다. 모델 자신의 진단으로 라우팅하는 사용자 원안(자기 진단 라우팅)은 별도 비교군 `MOD-RS`로 남겨, "모델의 자기 진단이 검증 원인만큼 쓸모 있는가"를 실험 질문으로 만든다(§18).
- 라우팅 결과가 Vision을 실제로 바꿨는지는 손실이 아니라 **Vision LoRA 텐서 변화량과 grad norm**으로 확인한다.

### 8.5 무엇을 주장할 수 있고 무엇은 못 하는가

- `VISUAL`은 "모델이 쓴 관찰 줄이 이미지 생성 메타데이터와 다르다"는 **기능적 사실**이다. Vision encoder 내부 표현이 원인인지, Merger 전달이 원인인지, LLM이 시각 토큰을 덜 참조한 것인지는 이 정보로 구분되지 않는다. 그래서 라우팅을 "원인 모듈 치료"가 아니라 **"시각 오류가 확인된 문제에서만 Vision 파라미터에도 학습 신호를 허용하는 정책"**으로 정의하고, 효과는 고정 경로 대조군과 무작위 경로 대조군(`MOD-RR`)으로 측정한다.
- 모델이 관찰 줄에 쓰지 않은 내용을 잘못 봤을 수도 있다. `UNKNOWN`을 0이나 `NON_VISUAL`로 바꾸지 않는다.
- 프로그램 생성 이미지의 렌더러 버그가 있으면 "오독"이 실제로는 gold 오류다. generator마다 렌더-메타데이터 정합성 테스트(§15)를 통과해야 하고, 슬롯 오독률이 특정 슬롯에 몰리면 렌더러부터 의심한다.

## 9. 보상: 정답 하나에서 시작해 한 항목씩 추가

보상은 **단계별로 하나씩만 추가**한다. 어느 항목이 효과를 냈는지 분리하기 위해서다.

| 단계 | 그룹 종류 | 그 그룹 샘플의 보상 |
|---|---|---|
| S1 `ALG-T1` | turn-1 | `R_answer` |
| S2 `ALG-NB` | turn-1, turn-2(, turn-3) | 각 턴의 `R_answer` (턴마다 별도 그룹) |
| S3 `ALG-NB+D` | + 진단 | 진단 그룹은 `R_diag` |
| S4 `ALG-NB+D+V` | obs 모드의 모든 풀이 그룹 | `R_answer + λ_v · R_visual` |
| 선택 `+L` | 모든 풀이 그룹 | `− R_overlong` (DAPO식 soft penalty) |

### 9.1 `R_answer` — 학습용 엄격 추출기

학습 보상은 V8 평가 파서를 쓰지 않는다. V8은 자연어 다양성을 넓게 수용하도록 만든 **평가용**이고(예: 로컬 실측에서 `Answer: 0.2778`이 gold `5/18`과 정답 처리될 정도로 수치 비교가 관대하다), 학습 보상은 "정확히 한 줄의 최종 답"이라는 **좁은 계약**을 강제해야 보상 악용이 줄어든다. 학습 후 MMMU 점수는 기존 V8로 낸다.

추출 규칙 `strict-final-v1`:
1. 응답 끝의 공백을 제거한 뒤, 정규식 `(?im)^\s*final answer\s*:\s*(.+?)\s*$`(§15.5와 동일)에 맞는 줄을 모두 찾는다.
2. 0개 → `INVALID_NO_FINAL`. 2개 이상 → `INVALID_MULTIPLE`. 1개인데 그 줄 뒤에 비공백 텍스트가 200자 넘게 이어지면 → `INVALID_MULTIPLE`(답 이후 철회·재계산 방지). finish_reason=length면 줄 존재와 무관하게 `TRUNCATED`.
3. 답 문자열 정리: 양끝의 `**`, `$`, `\boxed{}`, 마침표 1개 제거. 내부 수정은 하지 않는다.
4. 답 안에 `or`, `/`, `,`로 구분된 복수 후보가 있고 answer_type이 `list`가 아니면 → `INVALID_MULTIPLE`.

비교 규칙(문제의 `answer_type`별, verifier private에 저장):

| `answer_type` | 비교 |
|---|---|
| `mcq` | 답이 `A`~`J` 한 글자(괄호·마침표 허용)이고 보기 범위 안이면 그 글자로 비교. 글자 뒤에 보기 내용이 붙어 있으면 그 내용이 해당 보기와 정규화 후 일치해야 함(충돌 시 `INVALID_MULTIPLE`) |
| `int` | `int()` 변환 가능, 천 단위 쉼표 허용, 값 일치 |
| `decimal` | `Fraction(Decimal(s))`로 변환. gold의 `abs_tol`/`rel_tol` 또는 `round_to` 중 하나만 사전 지정(기본: 소수점 `round_to` 자리 반올림 후 일치) |
| `fraction` | `Fraction`으로 일치 (`3/4`, `0.75` 모두 허용 여부는 문제별 `accept_decimal` 플래그) |
| `unit_number` | 숫자+단위. 단위는 문제의 `unit_aliases` 목록에 있는 표기만 허용. 단위 없는 숫자는 `require_unit=false`일 때만 허용 |
| `short_text` | NFKC·소문자·공백 축약 후 `aliases` 중 하나와 완전 일치 |
| `list` | 순서 규칙(`ordered`/`unordered`)에 따라 원소별 위 규칙 적용 |

gold 값이 풀이 중간에 있다는 이유로 정답 처리하지 않는다. 단위가 다르면 숫자가 같아도 오답이다(단, 사전 등록한 alias 변환만 허용).

### 9.2 턴별 보상 (S2)

각 턴 그룹 안에서 `r = R_answer ∈ {0,1}`. 턴마다 다른 그룹이므로 "처음 맞음 1.0 / 수정 후 맞음 0.5" 같은 할인은 필요 없다. turn-1 샘플은 turn-1 정오만으로 평가되므로, 첫 시도를 일부러 틀려 재풀이 보상을 노리는 전략이 첫 시도 보상을 높여주지 않는다.

다만 turn-2 그룹의 가중치 `w_turn[2]`(기본 1.0)를 크게 하면 정책이 재풀이 상황에 과적합해 첫 시도 분포가 표류할 수 있다. 학습 batch의 그룹 비율과 `w_turn`을 매 step 로그에 남기고, 첫 시도 dev 정확도로만 확장 여부를 정한다. "틀렸다가 맞히면 더 큰 보상"은 실제로 해킹됐다: Self-rewarding correction(Xiong et al., 2025)에서 첫 시도 18.6% vs 최종 77.6%, MM-ReCoder(CVPR 2026)에서 "고의로 나쁜 첫 턴". 턴 분리는 첫 시도 보상이 첫 시도 정오에만 의존하게 만들어 이 유인을 구조적으로 없앤다. SRPO의 보상 순서(처음 맞힘 > 고쳐서 맞힘 > 맞던 것을 망침 > 계속 틀림)와도 같은 방향이다.

### 9.3 `R_visual` (S4)

```text
R_visual(a) = (#required_slots correct in a) / (#required_slots)      a가 obs 모드이고 단일 최종 답이 있을 때
            = 0                                                        그 외
r(a) = R_answer(a) + λ_v · R_visual(a),   λ_v = 0.1 (시작값, 미검증)
```

- 분모는 질문 생성 시 고정된 `required_slots`다. 모델이 쓴 쉬운 사실만 세지 않는다. 같은 슬롯을 두 번 다른 값으로 쓰면 그 슬롯은 오답.
- `λ_v=0.1`이면 사실만 완벽하고 답이 틀린 응답(0.1)이 정답 응답(≥1.0)을 넘지 못한다.
- 최종 답이 없는 응답에는 시각 보상을 주지 않는다. 과거 TabMWP B 조건이 "사실만 반복하고 답을 완성하지 못하는" 실패를 보였기 때문이다(§3).
- 같은 데이터·prompt·슬롯 스키마에서 `λ_v=0`과 `0.1`만 바꿔 비교한다.

### 9.4 `R_diag` — 오답노트 진단 보상 (S3)

진단 응답 형식(부록 A):

```text
Cause: <VISUAL_MISREAD | VISUAL_RELATION | NON_VISUAL | FINALIZATION>
Wrong item: <slot id, 예: bar[Mar]  또는  none>
Lesson: <한 문장>
```

```text
c* = 시도 1의 검증 원인(§8.3), 3분류로 축약: VISUAL(=MISREAD 또는 RELATION) / NON_VISUAL / FINALIZATION
R_diag = 1.0   if 3분류(Cause) == c*
       + 0.25  if c* == VISUAL 이고 Wrong item이 검증된 오독 슬롯 중 하나와 일치
       = 0     형식 위반 또는 불일치
```

- 진단 그룹은 `c*`가 `UNKNOWN`이면 만들지 않는다. 즉 **검증할 수 없는 노트에는 보상을 주지 않는다.** S²R(ACL 2025)이 자기 판정이 실제 정오와 맞을 때만 보상한 방식과 같은 원리다.
- 진단 그룹은 결국 맞힌 경로에서만 만들어지므로 "보조 보상은 정답일 때만"(DeepEyes, Reflection-V) 원칙도 자동으로 지켜진다.
- 다수 클래스만 찍는 전략을 막기 위해 한 update batch 안의 진단 그룹을 `c*` 클래스별로 최대 비율 50%가 되도록 샘플링한다(초과분은 그 step에서 생성하지 않고 로그).
- `Lesson` 문장은 채점하지 않는다. 길이 상한 256 tokens.
- 이 보상은 "자기 오류를 정확히 분류하는 능력"을 훈련한다. 그것이 첫 시도 정확도로 전이되는지는 S2 대비 S3의 첫 시도 dev 정확도로 측정한다. 전이가 없으면 진단 과제는 연구 보조 지표로만 남긴다.

### 9.5 선택 항목 — 과도한 길이 (S1~S4 어디에든 별도 ablation)

DAPO의 soft overlong punishment를 그대로 옮긴다. `L_max = max_new_tokens(turn)`, 완충 `L_buf = 512`.

```text
R_overlong(a) = 0                                   if len(a) ≤ L_max − L_buf
              = (len(a) − (L_max − L_buf)) / L_buf  if L_max − L_buf < len(a) ≤ L_max
r(a) ← r(a) − 0.5 · R_overlong(a)
```

V8에서 길이 상한 종료가 오답과 강하게 겹쳤지만, 짧게 끝내는 압력이 어려운 문제의 추론을 줄일 위험도 있다. 기본 보상에는 넣지 않고 `+L` 비교군으로만 둔다.

### 9.6 보상 악용 반례 (모두 단위 테스트로 고정)

| 반례 응답 | 기대 결과 |
|---|---|
| `Final answer: 53 or 35` | `INVALID_MULTIPLE`, 0 |
| 풀이 중간에 `53`이 있고 `Final answer: 35` | `INCORRECT`, 0 |
| `Final answer: 53` 뒤에 "Actually, ... the answer is 35" (200자 초과) | `INVALID_MULTIPLE`, 0 |
| `Final answer: 53` 두 줄 | `INVALID_MULTIPLE`, 0 |
| 단위 질문에 `Final answer: 53 A` (gold `53 V`) | `INCORRECT` |
| 객관식 `Final answer: B. 42` 인데 B 보기는 `24` | `INVALID_MULTIPLE` |
| obs 블록에 같은 슬롯을 두 값으로 기재 | 그 슬롯 오답 |
| obs 블록에 질문 문장 복사, 슬롯 없음 | 슬롯 0개 파싱, `R_visual=0`, 원인 `UNKNOWN` |
| 진단에서 항상 `Cause: VISUAL_MISREAD` | 클래스 균형 샘플링 하에서 기대 보상 ≤ 0.5 |
| `Error check:`만 길게 쓰고 답 없음 | `INVALID_NO_FINAL`, 0 |
| "I re-checked the image carefully" 같은 문구 반복 | 보상 영향 없음 (문구 보상 없음) |
| verifier에게 명령/파일 요청 텍스트 | 문자열로만 처리; 실행·조회 없음 |

## 10. 가중치 업데이트: 손실 함수와 한 step의 정확한 순서

### 10.1 기호

| 기호 | 의미 |
|---|---|
| `g` | 그룹: 같은 입력 상태 `s_g`를 공유하는 샘플 집합. 종류 `kind(g) ∈ {T1, T2, T3, DIAG}` |
| `i`, `n_g` | 그룹 안의 샘플 번호와 샘플 수 |
| `y_i = (y_i1 … y_iT_i)` | 샘플 i에서 **모델이 생성한 토큰**(마지막 `<|im_end|>` 포함, 패딩 제외) |
| `r_i` | §9의 보상 |
| `π_θ` | 학습 중 정책 |
| `π_old` | 이 batch를 생성한 정책. **update 직전 HF 학습 모델로 다시 계산한 logprob** |
| `π_samp` | 실제로 토큰을 뽑은 vLLM이 반환한 logprob (`logprobs_mode` 기본 raw; T=1·top_p=1·top_k=0이면 정책 분포와 같음) |
| `π_ref` | 고정 F0. `peft_model.disable_adapter()` 문맥의 forward (§11.5) |
| `L_cap(kind)` | 그 그룹 종류의 생성 상한 토큰 수 (정규화 상수) |
| `λ_kind` | 그룹 종류 가중치. 기본 `T1: 1.0, T2: 1.0, T3: 1.0, DIAG: 0.5` |

### 10.2 advantage — 그룹 평균만 빼고 표준편차로 나누지 않음

```text
A_i = r_i − mean_{k∈g}(r_k)
```

- Dr. GRPO가 지적한 표준편차 정규화의 난도 편향을 피한다. RLOO의 leave-one-out baseline과 상수배 차이뿐이다.
- `n_g < 2`인 그룹은 만들지 않는다.
- 모든 `r`이 같으면 `A=0`이다. 이 그룹의 정책 항은 0이고 KL 항만 남는다. 비율을 `degenerate_fraction`으로 기록한다.

### 10.3 토큰 손실

```text
ρ_it   = exp( log π_θ(y_it|·) − log π_old(y_it|·) )
d_i    = mean_t [ log π_old(y_it|·) − log π_samp(y_it|·) ]                 # 토큰당 평균 로그 비율 (기하평균 비율 w_i = exp(d_i))
m_i    = 1  if |d_i| ≤ δ_seq  else 0,   δ_seq = 0.05 nats/token            # 벗어나면 그 샘플을 loss에서 제외(mask)
clip_it = min( ρ_it·A_i ,  clip(ρ_it, 1−0.2, 1+0.28)·A_i )
δ_it   = log π_ref(y_it|·) − log π_θ(y_it|·)
kl_it  = exp(δ_it) − δ_it − 1                                              # k3, 항상 ≥ 0

loss_g = − ( 1 / (n_g · L_cap(kind(g))) ) · Σ_i m_i · Σ_t a_it · ( clip_it − β · kl_it )
LOSS   = Σ_g λ_kind(g) · loss_g  /  Σ_g λ_kind(g)
```

- `a_it`는 action mask다. 이미지·질문·이전 턴 사본·피드백·패딩은 0, 이번 턴에 모델이 생성한 토큰만 1이다. mask는 **길이로 만든다**(`pad_token_id` 151643이 EOS id이기도 하므로 pad 비교로 만들면 틀린다).
- 정규화 분모는 실제 길이가 아니라 상수 `n_g · L_cap`이다(Dr. GRPO). micro-batch 분할과 무관하게 같은 gradient가 나와야 하며 테스트로 고정한다(§17).
- 기본 `updates_per_batch = 1`. 이때 update 직전에는 `π_θ = π_old`이므로 `ρ=1`이고 clip은 작동하지 않는다. 식은 `updates_per_batch > 1` 확장용이다. Spurious Rewards가 지적한 clipping 편향 메커니즘도 이 설정에서는 작동하지 않는다는 점을 대조군 해석에 적는다.
- `w_i`·`m_i`는 vLLM과 HF 학습 forward 사이의 수치 차이(BF16 커널·샘플러)를 다룬다. 토큰 단위 보정은 BF16에서 불안정하다는 보고(Qi et al., 2025)에 따라 **시퀀스 단위 mask**를 쓴다. 기하평균 비율에 흔히 쓰는 τ=2(`1/2 ≤ w_i ≤ 2`)는 토큰당 평균 차이가 0.69 nats를 넘어야 걸리므로, G7 probe 임계(0.02 nats)와 비교하면 사실상 작동하지 않는다. 그래서 임계를 로그 공간의 `δ_seq`로 두고, 시작값 0.05에서 G7 실측(step 0의 `|d_i|` 분포 99.5백분위 × 2와 0.05 중 작은 값)으로 확정해 protocol에 동결한다. 매 step `mean|log π_old − log π_samp|`(token 평균), `masked_fraction`을 기록하고, `masked_fraction > 0.05`가 3 step 연속이면 학습을 멈추고 원인(이미지 입력 불일치·가중치 동기화 누락·캐시)을 조사한다. HF로 생성한 경우 `w_i ≡ 1`.
- **KL β = 0.01** (시작값). 문헌 범위는 0~0.04(DAPO·Dr. GRPO 0, R1/LMM-R1/MM-Eureka 1e-3, Asymmetric/PAPO 0.01, GRPO 원논문 0.04). 과거 SFT들이 MMMU를 떨어뜨렸으므로 중간값에서 시작하고, `β ∈ {0.001, 0.01}`를 비교군으로 둔다.
- logprob 계산은 `model(..., logits_to_keep=…)` 또는 마지막 hidden state를 1,024토큰씩 잘라 `lm_head` → FP32 `log_softmax` → gather로 한다(4K 완성 토큰의 전체 logits는 BF16에서도 수 GiB).

### 10.4 batch 구성과 dynamic sampling

- 한 step의 단위는 **문제 수 `B_prob = 8`**(pilot)이다. 문제마다 트리를 만들고 그 안의 사용 가능한 모든 그룹을 그 step의 loss에 넣는다.
- DAPO의 dynamic sampling을 단순화한다. 트리를 만든 뒤 **advantage가 0이 아닌 풀이 그룹(T1/T2/T3)이 하나도 없는 문제**는 그 step의 문제 수에 세지 않고 다음 문제를 더 뽑는다. 한 step에서 최대 `3·B_prob`개까지만 시도하고 버린 수를 로그한다. 버린 문제는 그 epoch에 다시 뽑지 않는다.
- 이것은 학습 데이터 안에서 신호 없는 그룹을 건너뛰는 표준 장치이며, dev/sealed 결과를 보고 학습 문제를 고르는 것과 다르다.
- 문제 순서: seed로 고정한 family 균형 셔플. 같은 `structure_hash`의 문제는 한 step에 하나만.

### 10.5 시작 하이퍼파라미터 (모두 미검증 시작값)

| 항목 | 값 | 근거 |
|---|---|---|
| LLM LoRA LR | 1e-5 | 전체 미세조정 RL LR 1e-6의 약 10배 (LoRA 관행; 블로그 근거라 비교 필요) |
| Merger(FP32 사본) LR | 1e-6 | 전체 파라미터, RL 관행 1e-6 |
| Vision LoRA LR | 1e-6 | 보수적. grounding 손상 보고 |
| optimizer | AdamW, betas (0.9, 0.95), eps 1e-8, weight_decay 0 | Dr. GRPO·DAPO |
| warmup / schedule | 10 step 선형 warmup 후 상수 | |
| gradient clip | **파라미터 묶음별** `clip_grad_norm_(group, 1.0)` | 과거 첫 update grad norm V 24.73 / M 1.35 / L 1.54 (§3, L12) |
| `B_prob` | 8 문제/step | 메모리·시간 실측 후 조정 |
| epoch | pilot 1 | |
| seed | 3407 (pilot), 3407/3408/3409 (확정 비교) | |

### 10.6 한 optimizer step의 정확한 순서

```text
step k:
 1. assert engine.policy_version == k   (직전 step 끝에서 가중치 push·검사 완료)
 2. B_prob개 문제의 트리 생성 (§7.8). 이 동안 HF 파라미터 변경 금지
 3. 모든 시도 채점 → L1 노트 → 문제별 route 결정 (§8)
 4. (진단 단계) 결국 맞힌 경로에서 진단 그룹 생성·채점
 5. 그룹 확정, 샘플별 action mask·정규화 상수·λ 계산 (이후 변경 금지)
 6. torch.no_grad(): 모든 샘플의 log π_old (현재 정책) 계산
 7. torch.no_grad() + disable_adapter(): 모든 샘플의 log π_ref 계산  → 끝나면 route 상태를 다시 적용 (§11.5 주의)
 8. 샘플마다(micro-batch 1): route.apply(문제의 route) → forward → loss 조각 → backward
 9. 묶음별 grad norm 기록 → 묶음별 clip → optimizer.step() → optimizer.zero_grad(set_to_none=True)
10. 저장 주기면 checkpoint (§15.13)
11. 병합 가중치를 vLLM에 push → reset_prefix_cache → probe 입력의 logprob를 HF와 비교 (§15.12)
12. policy_version = k+1, step 로그 기록 (§18.6)
```

**첫 step 자기검사:** `updates_per_batch=1`이면 8의 forward에서 나온 logprob와 6의 `log π_old`가 같아야 한다. 첫 step에서 `max|Δ| < 1e-3`(BF16 autocast, smoke에서 확정)를 assert해 mask·이미지 텐서·문맥 연결 오류를 잡는다.

### 10.7 하지 않는 것

- `labels=`를 넘기는 forward, cross-entropy on gold — 금지.
- 성공한 재풀이를 첫 시도 문맥으로 옮겨 붙여 학습 — 금지.
- 과거 rollout의 무제한 replay, 같은 batch로 여러 번 update — v2 기본에서 금지(확장 비교군에서만).
- 보상이 전부 0인 문제에 정답 풀이를 만들어 넣기 — 금지.

## 11. Vision · Merger · LLM 학습과 문제별 경로 선택

### 11.1 학습 파라미터 묶음 (transformers 4.57.1 실제 경로, 저장소 inventory로 확인)

| 묶음 | 정확한 경로 (`Qwen3VLForConditionalGeneration` 기준, PEFT 안에서는 앞에 `base_model.model.`) | 방식 | 학습 파라미터 수 |
|---|---|---|---:|
| LLM | `model.language_model.layers.{0..35}.self_attn.{q,k,v,o}_proj`, `.mlp.{gate,up,down}_proj` (252 Linear) | LoRA r=16, α=32, **dropout=0**, bias="none" | 33,030,144 |
| Merger 4개 | `model.visual.merger`, `model.visual.deepstack_merger_list.0`, `.1`, `.2` | PEFT `modules_to_save`, 학습 사본만 `.float()` (FP32 master), forward는 BF16 autocast | 109,105,152 |
| Vision (MOD-FVML/RV/RR만) | `model.visual.blocks.{0..23}.attn.qkv`, `.attn.proj` (48 Linear; 기존 `inspect_modules`의 V target과 동일) | LoRA r=16, α=32, dropout=0 | 2,359,296 |
| 고정 | `patch_embed`, `pos_embed`, vision MLP, norm, `embed_tokens`(= tied `lm_head`) | — | — |

- target 이름은 문자열을 손으로 쓰지 말고 `fine_tuning.modules.inspect_modules(model)["targets"]["L"]`, `["V"]`를 재사용한다(수정 금지 파일이지만 import는 허용).
- `linear_fc1`/`linear_fc2`라는 이름이 vision MLP와 4개 Merger 양쪽에 있다. 맨 이름 `"linear_fc1"`을 target으로 쓰면 Merger까지 LoRA가 붙는다. 전체 경로 목록만 쓴다.
- `modules_to_save`에는 `"model.visual.deepstack_merger_list"`(ModuleList)를 넣으면 PEFT가 `TypeError`를 낸다. 네 개의 정확한 경로를 넣는다.
- 모델 구조 사실: vision depth 24, DeepStack이 vision 블록 **5, 11, 17**의 출력을 각자의 Merger로 투영해 **LLM 0·1·2층 출력 뒤에 더한다**. 따라서 Merger 학습은 LLM 입력 임베딩뿐 아니라 앞쪽 3개 층의 잔차에도 영향을 준다.

### 11.2 모듈 실험군

| 실험군 | Vision LoRA | 4 Merger | LLM LoRA | 경로 결정 |
|---|:-:|:-:|:-:|---|
| `MOD-FML` | — | ✓ | ✓ | 항상 Merger+LLM ("비전은 맞았다") |
| `MOD-FVML` | ✓ | ✓ | ✓ | 항상 Vision 포함 |
| `MOD-RV` | ✓ | ✓ | ✓ | **검증된 시각 오독**이 있는 문제만 Vision 포함 (사용자 5단계) |
| `MOD-RR` | ✓ | ✓ | ✓ | 무작위로 Vision 포함, 포함 비율은 같은 step의 `MOD-RV` 비율과 동일 |
| `MOD-RS` (후속) | ✓ | ✓ | ✓ | **모델 자기 진단(L3)**이 VISUAL이면 Vision 포함 |

해석 규칙: `MOD-RV`가 `MOD-FML`·`MOD-FVML`·`MOD-RR` 모두보다 좋아야 "검증된 시각 오류에서만 Vision을 학습한다"는 가설을 지지한다. `MOD-RR`과 비슷하면 효과는 선택이 아니라 Vision update 양 때문이다. 모두 fresh F0에서 같은 데이터 순서·seed·step 수로 시작한다. `MOD-FML`은 Vision LoRA가 없으므로 학습 파라미터 수가 다르다는 점을 보고에 적는다.

### 11.3 문제별 경로 차단 — `requires_grad` 토글

micro-batch가 한 샘플이고 route는 문제 단위이므로, forward 직전에 Vision LoRA 파라미터의 `requires_grad`를 켜고 끄는 방식이 가장 단순하고 메모리도 아낀다. Vision LoRA가 grad를 요구하지 않으면 autograd는 vision 블록 쪽 그래프를 만들지 않고, gradient는 Merger 입력에서 멈춘다. Merger와 LLM은 정상적으로 gradient를 받는다.

```python
# fine_tuning/notebook_rl/policy.py — 계약 설명용
class VisionRoute:
    def __init__(self, vision_lora_params: list[torch.nn.Parameter]):
        self.params = vision_lora_params          # MOD-FML이면 빈 리스트

    def apply(self, vision_on: bool) -> None:
        for p in self.params:
            p.requires_grad_(vision_on)
```

- **PEFT 부작용 주의:** `disable_adapter()` 문맥을 빠져나오면 PEFT가 모든 LoRA와 `modules_to_save` 사본의 `requires_grad`를 True로 되돌린다. 그래서 reference 계산(§10.6 단계 7) 뒤, 그리고 **매 학습 forward 직전에** `route.apply(...)`를 다시 호출한다. 테스트로 고정한다.
- 혼합 batch가 필요해지면 대안은 4개 Merger(정확히는 PEFT `ModulesToSaveWrapper`)에 forward pre-hook을 걸어 해당 샘플 행만 `detach()`하는 것이다. v2에서는 쓰지 않는다.
- 차단된 샘플에서도 vision forward는 계산된다(입력 표현이 필요). vision 블록 전체를 `torch.no_grad()`로 감싸는 최적화는 Merger까지 얼리는 실수를 부르므로 하지 않는다.

### 11.4 optimizer와 "gradient가 없는 step"

- `torch.optim.AdamW`, param group 3개(LLM LoRA / Merger / Vision LoRA), §10.5의 값.
- **반드시 `zero_grad(set_to_none=True)`.** 한 step의 모든 문제가 Vision 차단이면 Vision LoRA의 `.grad`는 `None`으로 남고 AdamW가 건너뛴다. `.grad`가 0 텐서이면 AdamW는 이전 momentum으로 파라미터를 계속 움직인다. 이것은 "Vision을 학습하지 않았다"는 주장을 깨는 버그이므로 테스트로 막는다.
- 일부 문제만 Vision 허용이면 Vision gradient는 그 문제들의 손실 몫에서만 온다. 허용 비율을 step마다 기록하고 `MOD-RR`이 같은 비율을 쓴다.
- 묶음별 grad norm(clip 전)을 기록한다.

### 11.5 정확한 F0 reference

PEFT 0.17.1 소스에서 확인한 사실: `PeftModel.disable_adapter()`는 LoRA 층을 base 출력으로 되돌리고, `ModulesToSaveWrapper`는 학습 사본 대신 **원본 모듈**로 forward한다. 따라서 LoRA + `modules_to_save` 구성에서는 별도 F0 사본 없이 정확한 reference를 얻는다. 조건:

1. **PEFT의 in-place merge(`merge_adapter`, `merge_and_unload`)를 학습 중 절대 호출하지 않는다.** BF16 base에 더했다 빼는 왕복은 원래 값으로 정확히 돌아오지 않아 base와 reference가 오염된다. vLLM에 줄 병합 가중치는 out-of-place로 만든다(§15.12).
2. reference 계산은 `torch.no_grad()` 안에서, 정책 계산과 **같은 autocast 문맥**으로 한다.
3. `bias="none"` 유지.
4. 검증(GPU gate G3): (a) 초기화 직후 정책 logits == 새로 로드한 F0 logits(같은 autocast, 비트 단위), (b) 몇 step 학습·동기화 뒤 `disable_adapter()` logits == 새 F0 logits(비트 단위), (c) base 가중치와 Merger 원본의 SHA256 불변, (d) 문맥 종료 후 `requires_grad` 상태가 route 규칙대로.

기존 `fine_tuning/modules.py`는 Merger를 PEFT 밖에서 직접 FP32로 학습하므로 `disable_adapter()`가 Merger를 되돌리지 않는다. 그래서 새 코드는 `modules.configure_arm`을 쓰지 않고 `policy.py`에서 `modules_to_save` 방식으로 새로 구성한다. 기존 `modules.save_checkpoint/load_checkpoint`도 이름 체계가 달라 그대로 쓰지 않고 `checkpoint.py`에 새로 만든다(§15.13).

### 11.6 Vision을 학습할 때의 추가 의무

- vLLM의 GPU **encoder 출력 캐시**는 이미지 해시로 시각 특징(Merger·DeepStack 출력 포함)을 재사용한다. prefix cache 초기화로는 지워지지 않는다. 그래서 rollout 엔진은 `enable_prefix_caching=False`, `mm_processor_cache_gb=0`으로 만든다(요청마다 고유 UUID가 되어 요청 간 공유가 없어짐). 이것은 Vision LoRA가 없어도 Merger가 바뀌므로 **모든 실험군에 필수**다.
- 매 step 동기화 후 같은 이미지·프롬프트의 vLLM logprob가 HF와 일치하는지 probe로 확인한다.
- Vision을 연 실험군은 매 평가 주기마다 grounding probe(슬롯 정확도)를 기록하고, F0보다 2회 연속 낮아지면 Vision 학습을 멈추는 규칙을 둔다(Same Reward, Different Skills의 grounding 손상 보고 대응).

## 12. 학습 데이터: 무엇을 넣고 무엇을 빼나

### 12.1 선정 원칙 (모두 만족해야 학습 가능)

1. **라벨 출처**가 프로그램 계산(`PROGRAM`) 또는 사람 작성(`HUMAN`). 모델이 만든 질문·답·풀이·번역 재작성은 제외.
2. **원 출처의 train split**만. 다른 벤치마크의 val/test, MathVista 등의 출처로 쓰인 평가 split 금지.
3. **검증 가능한 답**: 정수·소수·분수·단위 수치·짧은 문자열·보기 문자. 서술형·증명은 제외.
4. **이미지가 꼭 필요**: Phase A의 blind 검사(회색 이미지)에서 F0가 우연 수준 이상으로 풀면 제외.
5. **가능하면 슬롯 정답**(이미지에서 읽어야 할 값·관계의 정답)을 함께 가짐 → 오답노트 귀인·라우팅·시각 보상에 필요.
6. MMMU/Pro 비중복 gate v2 통과(§13), family 단위 split.
7. 라이선스가 비공개 연구 사용을 허용(NC 라이선스는 비공개 연구만, 결과 공개 시 데이터는 배포하지 않음).

### 12.2 MMMU·MMMU-Pro는 어떻게 만들어졌나 (겹침 위험의 근원)

- MMMU: 약 50명의 대학생(공저자 포함)이 **주요 교재와 온라인 자료**, 대학 시험·퀴즈에서 수집. 11.5K 문항, dev 150 / val 900 / test 10,500, 30 과목, 차트·도식·지도·표·악보·화학 구조 등 30종 이미지 ([arXiv 2311.16502v4 §3](https://arxiv.org/html/2311.16502v4)).
- MMMU-Pro: MMMU 풀에서 1,730문항. 그중 577개는 MMMU validation 출신, 나머지는 test로 추정. GPT-4o가 보기를 10개로 늘렸고 vision 설정은 화면 촬영본 ([arXiv 2409.02813v3](https://arxiv.org/html/2409.02813v3)).
- **MMMU test 정답 공개(2026-02-12)** 이후 웹 수집 혼합셋은 test 문항·정답을 품을 수 있다.
- 결론: 교재·웹에서 이미지를 모은 데이터(AI2D, ChartQA, PathVQA 등)는 **출처 공유형 겹침** 위험이 있다. 프로그램으로 새로 그린 이미지는 이 위험이 구조적으로 낮지만, 템플릿을 사람이 쓰므로 텍스트 검사는 여전히 한다.

### 12.3 이미 공개된 RL 혼합셋은 쓰지 않는다 (실측 근거)

| 혼합셋 | 이번 조사의 실측·확인 | 판정 |
|---|---|---|
| **ViRL39K** (VL-Rethinker) | 38,870행을 저장소 규칙(`audit.py`)과 같은 방식으로 검사: **MMMU/Pro near-text 12건**(모두 `Processed` 웹 수집분; 10건은 화학·물리·미적분 실질 일치). 재작성 모델 미공개, 오염 제거 언급 없음 | **제외** |
| R1-OneVision | GPT-4o 설명, DeepSeek-R1 풀이, `infographic(gpt4v)`, `geo170k`, MathV360K 하위셋 | **제외** |
| MathV360K | GPT-4V가 32만 쌍 생성 | **제외** |
| Geo170K | gpt-3.5-turbo 생성 QA | **제외** |
| MM-Eureka-Dataset | MMMU 텍스트 0건이지만 Geo170K 포함, 다른 벤치마크 중복(MathVista 616, MMStar 596) 보고 | **제외** |
| ThinkLite-VL-70k | ScienceQA 10,332 = IMG train+val+test 합과 정확히 같음 → test 포함 추정, Geometry3K도 test 포함 추정 | **제외**(원 출처 train split에서 직접 재구성) |
| MMK12 train | MMMU 0건(78% 행 확인), 수학만, 텍스트 LLM 번역 | 조건부(번역 이력 기록) — 기본 미사용 |

### 12.4 실제 데이터셋 판정표

| 데이터 | 라이선스 | 라벨 | 슬롯 정답 | 약점 과목 | 판정 |
|---|---|---|---|---|---|
| **TabMWP** train ([PromptPG](https://github.com/lupantech/PromptPG)) | CC BY-NC-SA 4.0 | 사람(교육 전문가) | 표 셀(좌표 없음) | Accounting, Manage, Math, Sociology | **USE-FIRST**. 저장소에 고정본·필터 이력 있음. 기존 SFT 승인은 RL 승인으로 재사용하지 않고 v2 gate 재실행 |
| **Geometry3K** train ([InterGPS](https://github.com/lupantech/InterGPS)) / **PGPS9K** train | MIT(저장소) / MIT(HF) | 사람(교재) + 논리식/풀이 프로그램 | 논리식 | Math, Architecture | **USE-FIRST**. Geometry3K test family와 겹치는 PGPS9K 문항 제외 |
| **PlotQA** train | CC BY 4.0 | 템플릿(사람 질문 증류) + 실제 통계 | 데이터 표 | Manage, Sociology, Geography | **USE**(소량, 플롯 family당 1문항) |
| **DVQA** train | CC BY-NC 4.0 | 프로그램 | 막대·표 전체 | Accounting, Manage | **USE**(소량; 비현실적 라벨) |
| **MapQA** S/R train | CC BY-SA 4.0 | 템플릿 + 데이터 | 지역 값·범례 | Geography, Sociology | **USE**(S/R만; U는 웹 수집이라 조건부) |
| **FinQA** / **TAT-QA** train (표를 우리가 렌더) | MIT / CC BY 4.0 | 재무 전문가 + 실행 프로그램 | 프로그램 단계 | Accounting, Manage | **USE**. 정답 근거 셀만 골라 렌더하지 않음(위치 누설 방지) |
| **ChartQA** human train | GPL-3.0 | 사람(AMT) | 데이터 표(일부 잡음) | Sociology, Manage | **조건부**: machine 생성 질문(T5) 제외, 저장소의 ChartQA test 전이 평가와 chart family 분리 |
| **ScienceQA** IMG train | CC BY-NC-SA 4.0 | 사람(IXL) | 없음 | Biology, Geography | **조건부**: 2지선다 제외, 쉬움 |
| **AI2D** train | CC BY-SA (AI2 페이지) | 사람 | 도식 파스 그래프 | Biology, Geography | **조건부**: 구글 이미지 수집 → 이미지 gate 필수 |
| **CLEVR** train | CC BY 4.0 | 프로그램 | scene graph | (귀인 보정용) | **소량 USE**: 시각 귀인 보정 probe |
| GeoQA/GeoQA+/UniGeo | 라이선스 미확인 | 사람 | 프로그램 | Math | 보류 |
| FigureQA | 라이선스 미확인, 예/아니오만 | 프로그램 | 데이터 | — | 보류(찍기 50%) |
| PrIMuS, DeepScores V2, MUSCIMA++ | 미상 / CC BY 4.0 / CC BY-NC-SA | RISM / 렌더 / 수작업 | 기호 | Music | 보류 — 자체 악보 생성기가 우선 |
| PathVQA, VQA-RAD, SLAKE | 이미지 저작권 별도 / CC0 / CC BY 4.0 | 규칙·사람 | SLAKE 마스크 | Diagnostics | 확장 단계 소량, 교재 출처라 이미지 gate 필수 |
| GQA, MMK12, TQA, IconQA | — | — | — | — | 기본 미사용(적합도 낮음·번역·너무 쉬움) |

### 12.5 직접 만들 프로그램 생성기 (약점 과목 대응)

모든 생성기는 **구조를 먼저 무작위로 만들고 → 정답을 프로그램으로 계산하고 → 그림을 그린다.** 그림 속 글자는 우리가 넣은 것이므로 OCR 없이도 `printed_text`로 알고 있다.

| ID | 도구 (버전·라이선스) | 문제 예 | 슬롯 정답 | 약점 과목 (V8) | 구현 기반 |
|---|---|---|---|---|---|
| **G-MUS** | music21 10.5 (BSD-3) + Verovio 6.3 (LGPL-3.0, 실행만) | 두 음의 음정, 화음 종류, 조표의 조, 박자표, 마디 수, 음 이름 | `note[m.i]=C#4`, 조표, 박자 | **Music 20%** | 신규 |
| **G-CKT** | 저장소 `circuit_bridge.solve_nodal` + schemdraw 0.23 (MIT) | 노드 전압, 지로 전류, 등가 저항, 스위치 상태별 값 | `comp[R1]=4 ohm`, `wire[...]` | **Electronics 40%, Energy 46.7%** | 기존 + 위상 다양화 |
| **G-STAT** | sympy 1.14 (BSD) `continuum_mechanics.beam`, PyNite 3.2 (MIT) | 단순보 반력, 최대 모멘트 위치, 트러스 부재력 | 하중 크기·위치, 지지 조건 | **Mechanical 43.3%, Architecture 43.3%** | 신규 |
| **G-MAT** | matplotlib + 해석식, pymatgen (MIT, 선택) | 응력-변형 곡선의 항복점·탄성계수, 이원 상태도 지렛대 규칙, 결정 면 지수 | 곡선 표시점 값, 축 범위 | **Materials 40%** | 신규 |
| **G-THERMO** | CoolProp 8.0 (MIT) + matplotlib | P-v/T-s 선도의 상태 읽기, 사이클 효율 | 상태점 값 | Energy, Mechanical | 확장 단계 |
| **G-CHEM** | RDKit 2026.9 (BSD-3) + ChEBI SMILES (CC BY 4.0) | 분자식, 고리 수, 작용기 개수, 수소결합 공여체 수, 분자량 | `atoms[C]=7`, `rings=1` | **Chemistry 50%, Pharmacy 63.3%** | 신규 |
| **G-PK/LAB** | matplotlib + 해석식 | 농도-시간 곡선 반감기, 검사표 이상치 판정 | 곡선 점·표 셀 | Pharmacy, Diagnostics 36.7% | 확장 단계 |
| **G-BIO** | networkx/graphviz + 규칙 | Punnett square 확률, 가계도 유전 양식, 먹이그물 영양단계 | 칸·간선 | **Biology 46.7%, Agriculture 50%** | 확장 단계 |
| **G-GRAPH** | 저장소 `graph_bridge`/`nvw` + networkx/graphviz | 최단 경로, BFS 순서, DFA 수락 여부, 이진트리 순회, CPM 임계 경로 | `edge[u,v]=w` | **Computer_Science 53.3%**, Manage | 기존 + 유형 확장 |
| **G-MAP** | matplotlib/geopandas (BSD-3) | 단계구분도 범례 읽기, 축척 거리, 등고선 고도 | 지역 값·범례 구간 | Geography 56.7%, Sociology | 확장 단계 |
| **G-CHART** | matplotlib | 재무제표 표·막대·꺾은선에서 비율·증감·유리/불리 차이 | `bar[...]`, `cell[...]` | Accounting, Manage, Sociology | 신규(pilot) |

**Art(60%)는 대상에서 제외한다.** 미술사 지식은 프로그램으로 채점할 수 없다. 이 사실을 결과 보고의 범위 제한으로 적는다.

**생성기 공통 계약** (§15.4): 결정적(seed → 같은 PNG 바이트·같은 정답), `structure_hash`로 family 정의, 난도 knob, 렌더-메타데이터 정합성 자체검사, 질문 템플릿은 이미지 없이 답이 정해지지 않도록(같은 템플릿에서 그림을 바꾸면 답이 바뀌는 쌍 테스트).

### 12.6 수량 계획

**Pilot (384 학습 문제).** 기존 코드로 바로 만들 수 있는 것과 약점 과목 우선 신규 4종으로 구성한다.

| 출처 | 라벨 | 학습 | dev | sealed |
|---|---|---:|---:|---:|
| G-MUS | 프로그램 | 64 | 16 | 32 |
| G-CKT | 프로그램 | 48 | 12 | 24 |
| G-STAT | 프로그램 | 48 | 12 | 24 |
| G-CHEM | 프로그램 | 48 | 12 | 24 |
| G-CHART | 프로그램 | 48 | 12 | 24 |
| G-GRAPH | 프로그램 | 48 | 12 | 24 |
| TabMWP train / dev152 / sealed200 | 사람 | 48 | 12 | 24 (sealed200에서) |
| Geometry3K+PGPS9K train | 사람 | 32 | 8 | 16 |
| **합계** | 프로그램 79% | **384** | **96** | **192** |

추가로 **style-shift sealed 96**(프로그램 생성기 6종 × 16, 다른 글꼴·색·배치·해상도)을 만들어 "같은 생성기 스타일만 외웠는지"를 검사한다.

**확장 (4,000).** 이번 조사의 권장 비율을 따른다: G-MUS 480, G-CKT 400, G-STAT 320, G-THERMO 160, G-MAT 280, G-CHEM 320, G-PK/LAB 240, G-BIO 240, G-GRAPH 320, G-MAP 160, G-CHART 160, TabMWP 200, FinQA/TAT-QA 200, Geometry3K/PGPS9K 240, PlotQA/DVQA 80, ChartQA-H 80, SLAKE/PathVQA/VQA-RAD 120 (프로그램 3,080/4,000 = 77%). 확장은 pilot이 §18.7 gate를 통과한 뒤에만 만든다.

### 12.7 답 형식과 난도

- **열린 답 ≥ 60%**(정수·소수·단위 수치·짧은 이름). 객관식은 **보기 10개**(MMMU-Pro와 같음), 오답 보기는 프로그램으로 "흔한 실수 값"(단위 누락, 부호 반대, 인접 칸 값)을 만든다.
- 난도는 Phase A의 F0 결과로 고른다(§18.2): `n=8` 중 정답 수 `c`로 pass rate `p = c/8`. 학습 풀은 `p ≤ 0.75`(너무 쉬운 문제 제외). `p = 0`인 문제도 재풀이에서 신호가 날 수 있어 남기되 비율을 50% 이하로 제한.
- blind 검사: 같은 문제를 회색 이미지로 `n=4` 풀어 정답률이 `max(1/보기 수, 0.25)`를 넘으면 텍스트만으로 풀리는 문제로 보고 제외.
- 선별은 학습 전에 한 번 동결한다. 학습 중 dev/sealed 결과를 보고 학습 문제를 바꾸지 않는다.

## 13. MMMU/MMMU-Pro 비중복 검증 v2와 상위 모델 라벨 차단

### 13.1 지금 있는 것

| 항목 | 내용 |
|---|---|
| index | `analysis_exports/g7_cpu_refresh_20261005/portable_historical_exclusion_index.json` (SHA `a36047da…`), 100 Parquet / **16,740행**: MMMU dev 150, val 900, test 10,500; Pro std10·std4·vision 각 1,730 |
| 수록 fingerprint | 텍스트 해시 24,733, 7-gram shingle 집합 10,414행, 이미지 pixel SHA 12,168, dHash 12,095 |
| 검사 함수 | `fine_tuning/audit.py:contamination_candidates` (정규화 질문 exact 해시, 7-gram containment ≥0.8, raw/pixel SHA, dHash Hamming ≤2 → 검토), `check_lineage`, `check_splits` |
| index 생성 | `fine_tuning/sources.py:benchmark_index`, `prepare_sources.prepare_benchmarks` (RunPod에서만 다운로드 허용) |
| 로컬 원본 | MMMU val 30 파일, Pro std10 2 파일만 있음. dev/test/std4/vision 68 파일은 삭제된 Pod에만 있었음 |

### 13.2 이번 조사에서 확인한 약점

| # | 약점 | 영향 |
|---|---|---|
| L1 | crop·부분·회전·여백·의미 유사도 검사 없음 (`crop_and_semantic_similarity_complete=false`) | 교재 페이지 속 MMMU 그림, Pro-vision 화면 속 그림을 못 잡음 |
| L2 | OCR 없음. Pro-vision은 질문·보기가 **이미지 안**에 있고 question 필드가 비어 있음 | Pro-vision은 전체 화면 dHash로만 보호 |
| L3 | shingle 10개 미만(약 16단어 미만) 질문은 near-text 검사에서 빠짐 | **6,326행(37.8%)**이 exact 해시+이미지로만 보호 |
| L4 | MMMU의 `<image 1>` 자리표시가 정규화되지 않음 | 패러프레이즈 일치율 저하 |
| L5 | 빈 문자열 해시가 index에 있음 | 빈 질문 행은 무조건 EXACT로 걸림(오탐) |
| L6 | 보기·해설은 shingle하지 않음 | 새 줄기에 MMMU 보기를 그대로 쓴 문제를 못 잡음 |
| L7 | `contamination_candidates`를 원본 JSON index로 부르면 near-text 검사가 **조용히 생략**됨 (`_inverted_shingles` 캐시는 `approve()`만 만듦) | 새 RL gate가 이 함수를 그대로 쓰면 구멍 |
| L8 | train split만 검사, dev/holdout 생략 (`audit.py:403`) | 체크포인트 선택용 dev가 MMMU와 겹쳐도 모름 |
| L9 | fingerprint에 원본 레코드 ID가 없음 | 후보 추적에 Parquet 재독 필요 |
| L10 | 단순한 합성 이미지(표·흰 배경)에서 dHash가 퇴화 (`3333333333333333` 등) | 우리 데이터 영역에서 오탐 증가 |
| L11 | 임계값 불일치(학습 ≤2, 평가 ≤3), 보정 근거 없음 | |
| L12 | dHash 선형 탐색 | 타일 해시·대량 데이터에서 느림 |
| L13 | TRACE 어댑터가 `annotation_gt`를 버림 | 시각 보상에 필요한 값 손실 |
| L14 | Qwen 사전학습 데이터와의 겹침은 검사 불가 | 보고서 한계로 명시 |
| L15 | 출처 계열(IXL, 교재, Statista) 수준의 lineage 없음 | 같은 출처 계열 위험 미모델링 |

### 13.3 v2 gate (순서대로, 하나라도 걸리면 그 문제의 family 전체 quarantine)

| 단계 | 검사 | 임계값 (출처) | 결과 |
|---|---|---|---|
| 1 | 출처·권리·원 split·`label_provenance ∈ {PROGRAM, HUMAN}` | 규칙 | 불만족 → 거부 |
| 2 | 등록 lineage + 출처 계열(`upstream_origin`) 차단 목록 | 규칙 (`check_lineage` 확장) | 거부 |
| 3 | 정규화 질문 exact 해시 (`<image\s*\d+>` 제거, 빈 문자열 해시 제외) | 일치 | quarantine |
| 4 | 질문 7-gram containment; **짧은 질문은 5-gram 또는 MinHash**; **보기·해설도 별도 shingle** | ≥ 0.8 → quarantine; 20토큰 이상에서 8-gram 하나라도 일치 → 검토 (LMMs-Eval 방식) | |
| 5 | **OCR n-gram**: 벤치마크 이미지 12,168장(Pro-vision 포함)과 사람 출처 학습 이미지의 OCR 텍스트에 4단계 규칙 적용. 프로그램 생성 이미지는 `printed_text`를 그대로 사용 | 같음 | |
| 6 | raw SHA, pixel SHA | 일치 | quarantine |
| 7 | 전체 이미지 dHash/pHash + **2×2·3×3 타일 + 중앙 crop** 해시 (BK-tree) | Hamming ≤ 3 → quarantine (Bee 논문 기준); 저정보 해시(같은 바이트 반복)는 검토 전용 | |
| 8 | **SSCD 복제탐지 임베딩**(`sscd_disc_mixup`, MIT) 최근접 | cos ≥ 0.95 → 자동 제외(FineVision), ≥ 0.75 → 검토(SSCD 논문 90% 정밀도 지점) | |
| 9 | SigLIP 임베딩 최근접 (이미지·텍스트 교차) | 보정 세트로 정한 τ 미만 거리 → 검토 | |
| 10 | family 단위 제외: 같은 표·차트·지도·도식·템플릿 root를 공유하는 문제 전부 | 규칙 | |
| 11 | **모든 split**(train/dev/sealed)과 verifier 금고 파일 모두 검사 | 규칙 | |
| 12 | 동결 manifest: 단계별 범위·후보 수·`limitations`·index SHA | — | `EXCLUSION_CHECKED_WITH_LIMITS` |

**보정 세트(B6):** 양성은 MMMU val–Pro std10 byte 동일 이미지 632장에 재인코딩·리사이즈·crop·여백을 가한 것과 MMMU 그림–Pro-vision 화면 쌍, 음성은 TabMWP·TRACE·NVW 이미지 대 MMMU. dHash·타일·SSCD·SigLIP 각각의 TPR@고정 FPR을 보고하고, **임계값을 학습 데이터 검사 전에 protocol에 동결**한다.

**OCR 엔진 결정(§21):** (a) 고정 Qwen3-VL-4B 자체를 OCR 도구로 사용(라벨이 아니라 탐지 도구이므로 상위 모델 금지에 해당하지 않음, 새 의존성 없음, RunPod 1회 약 12K 이미지) 또는 (b) 고정 버전 OCR 패키지 추가. 기본안 (a).

**index 재구축:** 68개 파일이 로컬에 없으므로 RunPod에서 `prepare_benchmarks`로 모든 Parquet을 다시 받아 **fingerprint → (dataset, config, split, id, image_field)** 역참조 sidecar를 포함한 v2 index를 만든다(`training_allowed=false`, 결과는 비공개). 기존 `a36047da…`와의 연속성을 기록한다.

### 13.4 상위 모델 라벨 차단

- `fine_tuning/notebook_rl/source_registry_v1.yaml`에 출처마다 `label_provenance`, `image_provenance`, `upstream_origin`, `license`, `revision`, `train_split_only`, `model_touched`(번역·재작성 포함 여부)를 적는다. `model_touched=true`이면 학습 불가.
- 생성기 코드에서 외부 API·모델 호출 import 금지(§1.3 guard).
- 생성기 템플릿 문장은 사람이 직접 쓴 것만. LLM으로 템플릿을 만들거나 다듬지 않는다(코드 작성 보조와 데이터 문장 생성은 구분: 템플릿 문자열은 PR 리뷰 시 사람이 확인).

### 13.5 판정 어휘

- `CANDIDATE_SCAN_COMPLETE_NOT_APPROVAL`: 후보 0이어도 "깨끗하다는 인증"이 아님.
- `EXCLUSION_CHECKED_WITH_LIMITS`: 위 12단계를 통과, `limitations`에 미검사 범위(사전학습 노출, 의미적 패러프레이즈의 완전성 등) 명시.
- 학습기는 `EXCLUSION_CHECKED_WITH_LIMITS` manifest의 allowlist만 읽는다.

## 14. 코드 구조: `fine_tuning/notebook_rl/`

### 14.1 디렉터리

```text
fine_tuning/
├── notebook_rl/                      # 새 패키지 (이번 작업의 모든 코드)
│   ├── __init__.py
│   ├── README.md                     # 실행 명령 요약 (한 페이지)
│   ├── protocol_v1.yaml              # 동결 protocol: 모델·이미지·rollout·loss·arms·gate 임계값 (SHA 기록)
│   ├── prompts_v1.yaml               # 부록 A의 프롬프트 원문 (SHA 기록)
│   ├── source_registry_v1.yaml       # 데이터 출처·라이선스·라벨 출처·model_touched
│   ├── requirements-data.lock        # 데이터 생성 전용 venv (music21, verovio, rdkit, schemdraw, sympy, PyNiteFEA, matplotlib, imagehash)
│   ├── schema.py                     # 모든 레코드 dataclass + 직렬화 + 검증
│   ├── guards.py                     # audit hook: benchmark 경로·금고 파일·외부 모델 import 차단
│   ├── vault.py                      # actor/verifier/lineage 파일 쓰기·읽기 (allowlist)
│   ├── answers.py                    # strict-final-v1 추출 + typed 비교
│   ├── observations.py               # "Observed facts" 파싱 + 슬롯 대조
│   ├── verifier.py                   # 시도 채점, 진단 채점
│   ├── attribution.py                # 검증 원인, 문제 route, 진단 target, 무작위 route
│   ├── prompts.py                    # P0 직렬화, turn-1/재풀이/진단/oracle/probe 메시지
│   ├── generators/
│   │   ├── __init__.py               # REGISTRY: 이름 → 생성기 모듈
│   │   ├── common.py                 # GeneratedProblem, structure_hash, 결정적 렌더, 변환
│   │   ├── chart.py  music.py  chem.py  statics.py  circuit.py  graph.py   (pilot)
│   │   └── materials.py thermo.py pk_lab.py bio.py maps.py                  (확장)
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── tabmwp_rl.py              # TabMWP native → actor/verifier (solution 미사용)
│   │   └── geometry.py               # Geometry3K / PGPS9K train
│   ├── contamination.py              # v2 gate 12단계 (§13.3)
│   ├── dataset.py                    # split·manifest·Phase A 선별 동결
│   ├── engines.py                    # VLLMRolloutEngine(동일 프로세스) / HFRolloutEngine(테스트·smoke)
│   ├── policy.py                     # base 로드, PEFT(LoRA + modules_to_save), VisionRoute, logprob, reference, 병합 가중치
│   ├── losses.py                     # advantage, MIS mask, k3, 토큰 손실 (순수 torch)
│   ├── rollout.py                    # 트리 생성
│   ├── trainer.py                    # step 루프
│   ├── checkpoint.py                 # 저장·재개·평가용 로드
│   ├── export.py                     # 평가용 병합 BF16 HF 디렉터리
│   ├── metrics.py                    # step 지표·건강 검사·중단 규칙
│   ├── evaluate_external.py          # 외부 dev/sealed: 첫 시도, pass@k, 강제 2차, 피드백 2차, blind
│   ├── eval_mmmu8k.py                # 최종 MMMU 900 (input-recovery-v1 8K protocol + V8)
│   ├── gpu_gates.py                  # G0~G13 GPU gate (§17.2)
│   ├── runpod.py                     # RunPod controller 단계
│   ├── cli.py                        # python -m fine_tuning.notebook_rl.cli <subcommand>
│   └── data_tests/                   # 데이터 venv에서만 도는 생성기 테스트 (학습 venv의 discover 대상 아님)
└── tests/
    └── test_notebook_rl_*.py         # 학습 venv CPU 테스트 (unittest, 생성기 의존성 없음)
```

- JSON 설정 대신 **YAML**을 쓴다. `scripts/sync_test_repo.py`는 `fine_tuning` 아래 `.py .sh .yaml .yml .toml .md .lock`을 자동으로 MMDL_test에 게시하지만, 새 `.json`은 `AUTHORED_JSON` 목록에 넣어야 게시된다(`fine_tuning` 밖 파일 수정이 필요해짐). `.txt`도 `fine_tuning` 아래에서는 게시되지 않으므로 프롬프트는 YAML에 둔다.
- 생성된 데이터·이미지·rollout·checkpoint는 저장소 밖(`$MMDL_DATA_ROOT/training/notebook_rl/…`, `$MMDL_ARTIFACT_ROOT/notebook_rl/…`)에 둔다.

### 14.2 재사용할 기존 코드 (import만, 수정 금지)

| 필요 기능 | 기존 위치 | 사용 방법 |
|---|---|---|
| base 모델 로드 | `fine_tuning/train.py:212-223` `load_base(model_path)` | 그대로 호출. 반환 모델은 CPU, 호출자가 `.to("cuda")` |
| 버전 gate | `train.py:249-250` | 같은 검사를 `policy.py`에 복사 |
| 모듈 탐색·LoRA target | `fine_tuning/modules.py:18-50` `inspect_modules`, `:53-58` `parameter_group` | `["targets"]["L"]`(252), `["targets"]["V"]`(48), Merger 4개 확인 |
| 파라미터 probe | `train.py:149-168` `probe_indices`, `parameter_probes` | 고정 파라미터 불변·학습 파라미터 변화 검사 |
| 순서 해시·예산 | `train.py:202-209` `order_hash`, `stop_for_budget` | |
| 이미지 전처리 | `src/mmdl/evaluation/backends/input_preparation.py:23-74` `prepare_inputs`, `prepare_protocol_inputs` | 학습 forward와 rollout이 **같은 픽셀 텐서**를 쓰도록 |
| vLLM 엔진 구성 패턴 | `src/mmdl/evaluation/backends/vllm_backend.py:65-139, 199-255, 280-346` | 새 엔진에 패턴 복사(파일 수정 금지). 엔진 prompt ids == CPU processor ids 검사 포함 |
| 오염 검사 | `fine_tuning/audit.py:46-74, 77-121, 314-342` | `normalized`, `question_text`, `text_hash`, `shingles`, `image_fingerprint`, `check_lineage`, `check_splits`, `contamination_candidates` — **cache는 `approve()` 387~396행과 같은 방식으로 직접 구성**(L7) |
| index 생성 | `fine_tuning/sources.py:80-167` `benchmark_index`, `prepare_sources.py:17-73` | RunPod에서 v2 index |
| 회로 | `fine_tuning/circuit_bridge.py:41` `solve_nodal`, `:86` `verify_series_parallel`, `:100` `canonical_structure`, `:144` `render` | G-CKT의 solver·렌더 |
| 그래프·NVW | `fine_tuning/graph_bridge.py:52` `exclusion_reason`, `:171` `render`; `fine_tuning/nvw.py:42-124` solver/verifier, `:350` `build_episode`, `:37` `family_split` | G-GRAPH |
| TabMWP | `fine_tuning/tabmwp.py:74` `load_source`, `:105` `table_facts`, `:132` `family_id`, `:236` `exclusion_policy` | 어댑터. `solution` 필드는 읽지 않음 |
| 예산·종료 | `fine_tuning/budget.py:21-49` `effective_deadline`, `terminate_group`; CLI | RunPod 외곽 watchdog |
| controller | `fine_tuning/run_experiment.py:262-337` `Controller` | 단계 실행·source freeze·10분 health poll |
| clean checkout | `fine_tuning/runpod_g6.py:17-27` `checkout` | exact SHA·clean·MMDL_test origin |
| 평가 백엔드 | `experiments/input_recovery/backend.py:46-49` `create_backend`; `runner.py:56-77` `model_identity`; `protocol.yaml` (8K) | 최종 MMMU 8K 평가. `runner.py:27` `ARMS`와 `:80` `make_plan`(48문항 subset, 고정 arm 이름)은 **사용 금지** |
| MMMU 900 로드·P0·seed | `src/mmdl/evaluation/datasets/mmmu.py:16` `load_validation(data_root)`, `:53` `separate_sample(row, subject)`; `src/mmdl/evaluation/prompt.py:121` `build_messages(sample, images, root)`; `src/mmdl/runtime/contracts.py:57` `sample_seed(master_seed, sample_id)`; 조립 예 `src/mmdl/evaluation/engine.py:74-101` `prepare_request` | `eval_mmmu8k.py`에서만 import. 학습 코드에서는 import 금지(B4) |
| V8 채점 | `src/mmdl/evaluation/parsers.py:21-35` `score_with_parser` | **MMMU 평가에만**. 학습 보상에 사용 금지 |
| 파일 유틸 | `src/mmdl/runtime/artifacts.py` `sha256_file`, `write_json`, `atomic_text` | |
| 테스트 tiny model | `fine_tuning/tests/test_training.py:20-47` `tiny_model()` | CPU 정책·손실·재로드 테스트 |

### 14.3 절대 수정 금지 파일

| 파일 | 이유 |
|---|---|
| `fine_tuning/train.py`, `fine_tuning/modules.py` | 기존 APPROVED manifest가 이 두 파일의 SHA256을 고정 → 수정 시 TabMWP A/B·G6 원장 전부 로드 불가 |
| `fine_tuning/protocol.json`, `fine_tuning/bridge_training_protocol_v1.json` | `audit.py`가 SHA를 하드코딩 |
| `experiments/input_recovery/checkpoints.py`, `modules.py` | export manifest가 SHA 고정 |
| `src/**`, `configs/**`, `scripts/**`, `env/*.lock`, `MMDL/**` | 평가·제출 고정, 사용자 지시 |
| `fine_tuning/*.json` 레지스트리, `fine_tuning/runs/**`, `fine_tuning/generated/**` | 역사 원장 |

새 코드에서 `fine_tuning.train`을 import하면 `AssistantCollator`도 함께 로드된다(실행되지는 않음). 이것을 막지는 않되 §1.3의 테스트로 SFT 경로가 호출되지 않음을 증명한다.

## 15. 파일별 상세 명세

각 절은 **인터페이스(바꾸면 안 되는 이름·타입) → 동작 → 경계 사례 → 테스트**의 순서다. 타입 표기는 Python 3.12 문법이다. 모든 공개 함수는 실패 시 `ValueError`(입력 오류) 또는 `RuntimeError`(상태 오류)를 **메시지와 함께** 던지며, 조용히 기본값으로 넘어가지 않는다.

### 15.1 `schema.py` — 레코드 정의 (Wave 0, 설계자 작성)

```python
from dataclasses import dataclass, field
from typing import Literal

AnswerType = Literal["mcq", "int", "decimal", "fraction", "unit_number", "short_text", "list"]
Mode = Literal["obs", "plain"]
Split = Literal["train", "dev", "sealed", "style_sealed"]
Status = Literal["CORRECT", "INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED",
                 "UNVERIFIABLE", "CONTEXT_BLOCKED", "SYSTEM_ERROR"]
Cause = Literal["VISUAL_MISREAD", "VISUAL_RELATION", "NON_VISUAL", "FINALIZATION", "UNKNOWN"]
GroupKind = Literal["T1", "T2", "T3", "DIAG"]
Route = Literal["VISION_ON", "VISION_OFF"]

@dataclass(frozen=True)
class ActorItem:                      # actor(모델)가 볼 수 있는 유일한 정보
    problem_id: str                   # "<source>:<family>:<split>:<index>"
    source: str                       # 예 "gen-music-v1", "tabmwp-5a1a5221"
    family: str                       # 관찰 형식이 같은 단위, 예 "music_staff"
    mode: Mode
    question: str
    choices: tuple[str, ...]          # 주관식이면 ()
    answer_format: AnswerType         # 질문 자체에서 정해지는 형식만
    image_paths: tuple[str, ...]      # image_root 기준 상대경로
    image_sha256: tuple[str, ...]

@dataclass(frozen=True)
class VerifierItem:                   # verifier 금고에만 존재
    problem_id: str
    answer_type: AnswerType
    gold: str                         # 정규 문자열 (예 "53", "3/4", "12.5 ohm", "C")
    accept: dict                      # {"round_to": 2} | {"abs_tol": "0.01"} | {"unit_aliases": [...], "require_unit": true}
                                      # | {"aliases": [...]} | {"ordered": true}
    choices: tuple[str, ...]
    slots: dict[str, str]             # slot_id → 정규 값 (이미지에서 읽을 수 있는 모든 슬롯)
    slot_types: dict[str, str]        # slot_id → "number" | "pitch" | "text" | "relation"
    required_slots: tuple[str, ...]   # 이 질문을 푸는 데 필요한 슬롯 (생성 시 고정)
    oracle_fact_lines: tuple[str, ...]
    printed_text: tuple[str, ...]     # 그림 속에 렌더된 문자열 (오염 검사용)

@dataclass(frozen=True)
class Lineage:
    problem_id: str
    source: str
    source_revision: str
    license: str
    label_provenance: Literal["PROGRAM", "HUMAN"]
    image_provenance: str             # "rendered:<tool@version>" | "source:<dataset>"
    upstream_origin: str              # "procedural" | "IXL" | "textbook:<name>" | ...
    generator_version: str | None
    generator_seed: int | None
    template_id: str
    structure_hash: str               # family 분리 단위
    split: Split
    parents: tuple[str, ...]
    benchmark_derived: bool = False   # True면 로드 거부
    model_touched: bool = False       # True면 로드 거부

@dataclass(frozen=True)
class SlotReport:
    parsed: int
    required: int
    required_correct: int
    wrong_required: tuple[str, ...]
    all_correct: int
    all_total: int
    conflicting: tuple[str, ...]      # 같은 슬롯을 다른 값으로 두 번 이상 씀

@dataclass(frozen=True)
class VerifierResult:
    status: Status
    extracted: str | None
    reward_answer: float              # 1.0 | 0.0 (UNVERIFIABLE/SYSTEM_ERROR면 의미 없음)
    slots: SlotReport | None
    verifier_version: str

@dataclass
class Sample:
    sample_id: str                    # f"{group_id}#{k}"
    problem_id: str
    group_id: str
    kind: GroupKind
    turn: int
    parent_sample_id: str | None
    prompt_token_ids: list[int]       # 엔진이 실제로 소비한 입력 토큰 전체
    completion_token_ids: list[int]
    sampler_logprobs: list[float]     # len == len(completion_token_ids)
    finish_reason: str                # "stop" | "length"
    text: str
    seed: int
    policy_version: int
    result: VerifierResult | None = None
    diag_reward: float | None = None
    reward: float = 0.0

@dataclass
class Group:
    group_id: str                     # f"{problem_id}@{policy_version}/{kind}/{path}"
    kind: GroupKind
    problem_id: str
    state_sha256: str                 # sha256(prompt_token_ids + image sha 목록)
    samples: list[Sample]
    usable: bool
    reason: str | None                # usable=False 사유

@dataclass
class ProblemTree:
    problem_id: str
    policy_version: int
    groups: list[Group]
    route: Route
    causes: dict[str, Cause]          # sample_id → 검증 원인 (실패 시도만)
    notebook: dict                    # L1 host notebook (§8.2 형식)

@dataclass(frozen=True)
class DiagTarget:                     # 진단 그룹의 정답 (verifier 쪽에만 존재)
    cause3: Literal["VISUAL", "NON_VISUAL", "FINALIZATION"]
    wrong_slots: tuple[str, ...]      # VISUAL일 때 검증된 오독 슬롯, 그 외 ()

@dataclass(frozen=True)
class OracleProbe:                    # 진단 전용 생성. loss에 절대 들어가지 않음
    probe_id: str
    problem_id: str
    kind: Literal["oracle", "self_perception"]
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    status: Status
    policy_version: int
    # Group/Sample로 변환하는 함수를 만들지 않는다. trainer는 OracleProbe를 import하지 않는다(테스트로 확인)
```

- `to_json(obj) -> str` / `from_json(cls, s) -> obj`: 키 정렬, `ensure_ascii=False`, 알 수 없는 키가 있으면 `ValueError`.
- `validate_actor(a: ActorItem)`: `len(image_paths)==len(image_sha256)≥1`, mcq면 `len(choices)≥2`이고 `answer_format=="mcq"`, 질문 비어있지 않음.
- **테스트:** 왕복 직렬화, 금지 키(`gold`, `solution`, `answer`, `slots`, `target_text`)가 ActorItem JSON에 있으면 `from_json` 실패.

### 15.2 `guards.py` — fail-closed 감시 (Wave 1, Sonnet)

```python
def install_guards(*, forbidden_roots: list[Path], vault_files: list[Path],
                   allow_vault: bool, allow_benchmark: bool) -> None
def assert_no_teacher_imports() -> None
```

- `sys.addaudithook`으로 `open` 이벤트를 감시한다. 경로가 `forbidden_roots` 아래이고 `allow_benchmark=False`이면 `PermissionError("benchmark path blocked: …")`. `vault_files`이고 `allow_vault=False`이면 같은 방식으로 차단.
- audit hook은 제거할 수 없으므로 **프로세스당 한 번**만 설치하며, 두 번째 호출은 `RuntimeError`.
- `assert_no_teacher_imports`: `sys.modules`에 `openai`, `anthropic`, `google.generativeai`, `google.genai`, `cohere`, `mistralai`가 있으면 `RuntimeError`.
- **테스트:** subprocess에서 guard 설치 후 금지 경로 open이 실패, 허용 경로는 성공. 이미 import된 금지 모듈을 흉내 내 `sys.modules`에 넣었을 때 실패.

### 15.3 `vault.py` — 금고와 allowlist (Wave 1, Sonnet)

```python
ACTOR_KEYS: frozenset[str]   # ActorItem 필드 이름과 정확히 같음
def write_split(root: Path, actors: list[ActorItem], verifiers: list[VerifierItem],
                lineages: list[Lineage]) -> dict         # manifest 반환
def read_actors(root: Path) -> list[ActorItem]           # actor_inputs.jsonl만 open
def read_verifier(root: Path) -> dict[str, VerifierItem]  # verifier 프로세스 전용
def read_lineage(root: Path) -> dict[str, Lineage]
```

- 파일: `actor_inputs.jsonl`, `verifier_private.jsonl`, `lineage.jsonl`, `manifest.yaml`(파일별 SHA256, 행 수, 생성 코드 SHA, protocol SHA). 쓰기는 `open(mode="x")`로 덮어쓰기 금지.
- 세 파일의 `problem_id` 집합이 정확히 같아야 한다. 하나라도 다르면 `ValueError`.
- `read_actors`는 JSON 키가 `ACTOR_KEYS`와 다르면 실패. `Lineage.benchmark_derived` 또는 `model_touched`가 True인 행이 있으면 `write_split` 실패.
- 누설 검사: 각 actor의 `question`/`choices` 문자열에 그 문제의 `gold`가 **단어 경계로** 포함되고 answer_type이 `mcq`가 아니면 경고 목록에 넣고 manifest의 `leak_review`에 기록(생성기 버그 탐지용; 자동 거부는 하지 않음 — "x=53일 때"처럼 정당한 경우가 있음).
- **테스트:** 왕복, 집합 불일치 실패, 금지 키 실패, 덮어쓰기 실패.

### 15.4 `generators/common.py`와 생성기 계약 (Wave 0 공통부는 설계자, 각 생성기는 Wave 1)

```python
@dataclass(frozen=True)
class GeneratedProblem:
    actor: ActorItem
    verifier: VerifierItem
    lineage: Lineage
    image_bytes: tuple[bytes, ...]    # PNG

def structure_hash(obj) -> str                     # 정렬 JSON의 sha256 앞 16자
def png_bytes(fig_or_image, *, dpi: int) -> bytes  # 메타데이터(시간·소프트웨어) 제거된 결정적 PNG
def split_of(structure_hash: str, ratios=(0.8, 0.1, 0.1)) -> Split   # 해시 앞 8자리 정수로 결정
def build(source: str, family: str, index: int, split: Split, mode: Mode, question: str,
          choices, answer_type, gold, accept, slots, slot_types, required_slots,
          oracle_fact_lines, printed_text, structure, template_id, image_bytes,
          generator_version: str, seed: int, style: str) -> GeneratedProblem
```

**모든 생성기 모듈이 지켜야 할 인터페이스:**

```python
GENERATOR_VERSION: str                      # 예 "gen-music-v1"
FAMILY: str                                 # 관찰 형식 family
OBSERVATION_INSTRUCTION: str                # prompts_v1.yaml의 family 문장과 동일해야 함 (테스트)
def generate(index: int, seed: int, split: Split, mode: Mode, style: str = "default") -> GeneratedProblem
def self_check(p: GeneratedProblem) -> None # 렌더-메타데이터 정합성, 독립 재계산, 실패 시 ValueError
```

- **결정성:** 같은 `(index, seed, split, mode, style)` → 같은 PNG 바이트·같은 정답. 시간·난수 전역 상태 사용 금지(`random.Random(seed*1_000_003 + index)` 같은 지역 RNG).
- **split:** 생성기는 구조를 먼저 뽑고 `split_of(structure_hash)`가 요청한 split과 다르면 다음 후보를 뽑는다(최대 1,000회, 넘으면 `RuntimeError`). 이렇게 하면 train/dev/sealed의 구조가 절대 겹치지 않는다.
- **style:** `"default"`(학습·dev·sealed)와 `"shift"`(style_sealed: 다른 글꼴·색·선 두께·배치). shift 스타일은 학습에 쓰지 않는다.
- **이미지 의존성:** 질문 템플릿은 그림 없이 답이 정해지면 안 된다. data_tests에서 "같은 템플릿·다른 구조 2개의 정답이 다르다"를 표본 100쌍에 대해 검사한다.
- **독립 재계산:** 정답은 두 가지 독립 방법으로 계산해 일치해야 한다(예: 회로 nodal vs series-parallel, 보의 sympy vs 평형식 손계산, 분자의 RDKit vs SMILES 원자 세기). 불일치면 그 문제를 버리지 않고 **예외로 중단**(생성기 버그).
- **오답 보기(mcq):** 10개. 정답 1 + 프로그램이 만든 "그럴듯한 실수" 값(단위 누락, 부호 반대, 인접 칸, 1/2배, 다른 공식) + 무작위. 보기끼리 중복 금지. 정답 위치는 RNG로.

**생성기별 핵심 명세 (pilot 6종)**

| 생성기 | 구조 무작위화 | 질문 템플릿 예 | answer_type | 슬롯 | 독립 검증 |
|---|---|---|---|---|---|
| `chart.py` (G-CHART) | 막대/꺾은선/표, 계열 2~4, 항목 4~8, 값 정수·소수1 | "2023년 대비 2024년 매출 증가율(%)은? 소수 첫째 자리" / "가장 큰 차이가 나는 분기는?" / 재무표: "매출총이익률" | decimal/short_text/mcq | `bar[계열][항목]`, `cell[행][열]` | 값 배열에서 직접 계산 vs pandas 계산 |
| `music.py` (G-MUS) | 음자리표(높은·낮은), 조표 0~4개, 박자표, 1~4마디, 음 4~12개 | "표시된 두 음의 음정은?(예: M3)" / "이 화음의 종류는?" / "조표가 나타내는 장조는?" / "몇 번째 마디의 첫 음 이름은?" | short_text/mcq | `note[m.i]`(음고+옥타브), `clef`, `key_sig`, `time_sig` | music21 interval vs 반음 수 표 |
| `chem.py` (G-CHEM) | ChEBI SMILES에서 무거운 원자 6~30개 분자 샘플 | "분자식은?" / "고리 수는?" / "수소결합 공여체 수는?" / "카복실기 개수는?" | short_text/int | `atoms[원소]`, `rings` | RDKit Descriptors vs 직접 원자 세기 |
| `statics.py` (G-STAT) | 단순보/외팔보, 집중하중 1~3·등분포하중 0~1, 길이·위치 정수 | "지점 A의 반력(kN)은?" / "최대 굽힘모멘트 위치(m)는?" | unit_number/decimal | `load[i]=크기@위치`, `span`, `support` | sympy beam vs 평형방정식 |
| `circuit.py` (G-CKT) | 기존 `circuit_bridge` 위상 + 새 위상(직병렬 2~3단, 전원 1~2) | "R2에 흐르는 전류(A)는?" / "노드 N1 전압은?" / "등가저항은?" | unit_number/fraction | `comp[R1]=값 단위`, `wire[R1]=N0-N1` | `solve_nodal` vs `verify_series_parallel` |
| `graph.py` (G-GRAPH) | 기존 `graph_bridge`/`nvw` + 가중 그래프·DFA·이진트리 | "A→F 최단거리는?" / "이 DFA는 '0110'을 수락하는가?" / "후위 순회 결과는?" | int/short_text/list | `edge[u,v]=w`, `trans[q,a]=q'` | networkx vs 자체 Dijkstra, `exclusion_reason` 지름길 검사 |

### 15.5 `answers.py` — 학습용 엄격 추출·비교 (Wave 1, Haiku → Opus 리뷰)

```python
VERSION = "strict-final-v1"
@dataclass(frozen=True)
class Extraction:
    status: Literal["OK", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"]
    answer: str | None
def extract_final(text: str, finish_reason: str) -> Extraction
def compare(answer: str, item: VerifierItem) -> bool | None      # None = 비교 불가(UNVERIFIABLE)
```

- 규칙은 §9.1을 그대로 구현한다. 정규식 `(?im)^\s*final answer\s*:\s*(.+?)\s*$`.
- `finish_reason == "length"` → 항상 `TRUNCATED`.
- 마지막 `Final answer:` 줄 뒤에 비공백 200자 초과 → `INVALID_MULTIPLE`.
- 답 정리: 양끝 `**`, `$`, `\boxed{…}`의 내용, 끝의 마침표 1개만 제거.
- mcq 답 형식: `^\(?([A-J])\)?\.?$`(글자만) 또는 `^\(?([A-J])\)?[.:)]\s+(.+)$`(글자 + 보기 내용). `extract_final`은 보기 목록을 모르므로 답 문자열만 돌려준다. **글자와 보기 내용이 서로 다른지(충돌)는 `verifier.grade`가 검사**해 충돌이면 `INVALID_MULTIPLE`로 처리한다(§15.7). `compare`는 충돌이 없는 경우의 글자 일치만 판단한다.
- 숫자 비교는 `Decimal`/`Fraction`만. `float` 금지. 천 단위 쉼표 허용(`1,234`), 그 외 문자가 섞이면 비교 불가.
- `unit_number`: 정규식으로 숫자와 단위 분리, 단위는 `accept["unit_aliases"]` 안에서만 일치, 단위 없음은 `require_unit=False`일 때만 허용.
- `short_text`: NFKC → casefold → 공백 하나로 → 끝 마침표 제거 후 `aliases` 중 하나와 완전 일치.
- `list`: 쉼표·공백 구분, `ordered`면 순서까지.
- **테스트(반례 전부, §9.6 표):** 복수 답, 중간에 정답이 있고 최종은 오답, 철회 문장, 두 줄, 단위 불일치, `B. 42`(B 보기와 내용 불일치), `\boxed{53}`, `**53**`, `53.`, `1,234`, `3/4` vs `0.75`(`accept_decimal` 유무), 길이 종료.

### 15.6 `observations.py` — 관찰 블록 파싱과 슬롯 대조 (Wave 1, Haiku → Opus 리뷰)

```python
def parse_observations(text: str) -> dict[str, list[str]]       # slot_id → 기록된 값들(원문)
def normalize_slot(value: str, slot_type: str) -> str | None     # 정규 문자열 또는 None
def compare_slots(parsed: dict[str, list[str]], item: VerifierItem) -> SlotReport
```

- 파싱 대상은 `Observed facts:` 줄 다음부터 `Solution:` 줄 전까지. 각 줄 `^\s*-?\s*([a-z_]+(?:\[[^\]]+\]){0,2})\s*=\s*(.+?)\s*$`(대괄호 0~2개; `rings`, `clef`, `key_sig`, `time_sig`처럼 대괄호 없는 슬롯도 잡는다). 블록이 없으면 빈 dict.
- slot_id 정규화: 공백 제거, 소문자, 대괄호 내부의 따옴표 제거.
- `normalize_slot`: number → `Fraction`의 정규 문자열(단위 문자는 slot_type이 `number`면 제거), pitch → `C#4` 형식(♯/♭/영문 sharp/flat 허용), text → NFKC·casefold, relation → 양끝 노드 정렬(무방향일 때만; 방향 슬롯은 `slot_id`가 `trans[` 또는 `dir[`로 시작).
- 같은 slot_id에 서로 다른 정규 값이 둘 이상 → `conflicting`에 넣고 오답 처리.
- `SlotReport.parsed`는 item에 존재하는 slot_id 중 기록된 수(없는 slot_id 기록은 무시).
- **테스트:** 정상, 순서 무관, 중복 동일값(정답), 중복 상이값(conflict), 블록 없음, 알 수 없는 슬롯 무시, 음이름 표기 변형.

### 15.7 `verifier.py` (Wave 2, Sonnet)

```python
class Verifier:
    def __init__(self, vault_root: Path)                  # read_verifier로 로드 (verifier 프로세스만)
    def grade(self, problem_id: str, text: str, finish_reason: str, mode: Mode) -> VerifierResult
    def grade_diagnosis(self, problem_id: str, text: str, target: "DiagTarget") -> float
```

- `grade` 순서: `extract_final` → status가 OK가 아니면 그대로(보상 0) → mcq면 보기 충돌 검사(§15.5)를 여기서 → `compare` → None이면 `UNVERIFIABLE` → True/False면 CORRECT/INCORRECT. mode가 `obs`이고 item에 슬롯이 있으면 `compare_slots` 결과를 붙인다(답 정오와 무관하게 기록).
- `grade_diagnosis`: §9.4 형식 파싱(`Cause:`, `Wrong item:` 줄), 3분류 일치 1.0, 슬롯 일치 +0.25, 형식 위반 0.
- verifier는 actor 텍스트를 **문자열로만** 다룬다. eval/exec/파일 접근/네트워크 없음.
- **테스트:** 가짜 금고로 각 status 경로, obs 모드 슬롯 보고, 진단 채점 4경우.

### 15.8 `attribution.py` (Wave 2, Sonnet → Opus 리뷰)

```python
def attempt_cause(result: VerifierResult, item: VerifierItem) -> Cause | None   # 정답이면 None
def problem_route(results: list[VerifierResult], items: VerifierItem, rule: str,
                  probe: dict | None = None) -> tuple[Route, dict]
def diagnosis_target(first_attempt: VerifierResult, item: VerifierItem) -> "DiagTarget | None"
def random_route(rate: float, key: str) -> Route      # sha256(key) 기반 결정적
```

- `attempt_cause`는 §8.3 규칙을 그대로 구현(절반 미만 파싱 → UNKNOWN 등).
- `problem_route(rule="verified-any-visual-v1")`: 어느 시도든 required slot 오독이 있으면 VISION_ON, 아니면 VISION_OFF. 반환 dict에는 판단 근거(어느 sample, 어느 slot)를 기록.
- 반환 타입은 §15.1의 `DiagTarget`. 시도 1 원인이 UNKNOWN이면 None.
- **테스트:** 규칙 표의 각 행, random_route의 결정성과 비율(10,000 키에서 ±2%).

### 15.9 `prompts.py` (Wave 1, Sonnet → Opus 리뷰)

```python
def load_prompts(path: Path) -> dict                  # prompts_v1.yaml, sha256 반환 포함
def p0_body(question: str, choices: tuple[str, ...]) -> str
def turn1_text(actor: ActorItem, prompts: dict) -> str
def retry_text(status: Status, prev_answer: str | None, mode: Mode, family: str, prompts: dict) -> str
def diagnose_text(k: int, prompts: dict) -> str
def oracle_text(actor: ActorItem, fact_lines: tuple[str, ...], prompts: dict) -> str
def probe_text(observed_text: str, actor: ActorItem, prompts: dict) -> str   # 이미지 없는 자기 지각 probe
def user_message(text: str, n_images: int) -> dict   # Qwen chat 형식: 이미지 content 먼저, 텍스트 마지막
```

- `p0_body`는 `prompts/mmmu_mcq_v1.txt`/`mmmu_open_v1.txt`와 같은 직렬화(`Question: …\nOptions:\nA. …\n…Please select the correct answer from the options above.`). 테스트에서 저장소의 P0 템플릿 파일로 만든 문자열과 바이트 단위로 비교한다. hint는 쓰지 않는다.
- 자리표시자는 `str.replace`로 채운다(질문의 `{`가 깨지지 않게).
- 재풀이 텍스트는 **새 user turn**으로 들어간다. 이전 assistant 응답은 문자열로 다시 넣지 않는다 — rollout이 토큰 ID로 이어 붙인다(§15.11).
- **테스트:** P0 바이트 일치, 모든 status의 재풀이 문구, 중괄호 포함 질문.

### 15.10 `losses.py` — 순수 함수 (Wave 1, **Opus**)

```python
def group_advantages(rewards: torch.Tensor) -> torch.Tensor                      # [n] → [n], r - mean
def mis_mask(old_lp: torch.Tensor, samp_lp: torch.Tensor, action_mask: torch.Tensor,
             delta_seq: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]        # (seq_keep[bool, B], d[B] = 토큰당 평균 로그 비율)
def k3(ref_lp: torch.Tensor, cur_lp: torch.Tensor) -> torch.Tensor               # elementwise ≥ 0
def group_loss(cur_lp, old_lp, ref_lp, adv, action_mask, seq_keep, *,
               eps_low: float, eps_high: float, beta: float, norm_const: float) -> torch.Tensor
def combine(group_losses: list[torch.Tensor], lambdas: list[float]) -> torch.Tensor
```

- 텐서 모양: `cur_lp/old_lp/ref_lp/action_mask`는 `[B, T]`(B=그룹 샘플 수, T=패딩된 완성 길이), `adv`와 `seq_keep`은 `[B]`. `old_lp`, `ref_lp`, `adv`, `seq_keep`은 내부에서 `detach()`.
- 식은 §10.3과 정확히 같다. `norm_const = n_g · L_cap(kind)`.
- micro-batch로 나눠 호출해도 합이 같도록, `group_loss`는 **샘플 부분집합**을 받아도 같은 `norm_const`로 나눈 부분합을 반환한다(호출자가 더한다).
- `combine`의 분모 `Σ_g λ_kind(g)`는 **그 step의 usable 그룹 전체**에 대해 한 번만 계산한다. micro-batch마다 `combine`을 부르면 분모가 달라지므로, 호출자는 step 시작 시 `Σλ_step`을 고정하고 각 조각에 `λ_g / Σλ_step`을 곱해 backward한다(테스트: micro-batch 분할 합 == 전체).
- **테스트(수치):** 손으로 계산한 2그룹 예제와 loss·gradient 일치(`torch.autograd.grad`), 모든 보상 동일 시 정책 항 0이고 KL 항만 남음, β=0이면 KL gradient 0, ρ=1에서 clip 무효, `seq_keep=False` 샘플의 gradient 0, micro-batch 분할 합 == 전체, **토큰당 0.1 nats 불일치를 넣은 합성 샘플은 masked, 0.01 nats는 유지**, action_mask가 0인 토큰은 `d_i` 계산에서 제외.

### 15.11 `rollout.py` — 트리 생성 (Wave 3, **Opus**)

```python
@dataclass
class RolloutConfig:  # protocol_v1.yaml에서 로드
    n: dict[int, int]            # {1: 8, 2: 4, 3: 4}
    branch: dict[int, int]       # {1: 2, 2: 1}   (turn t에서 다음 턴으로 확장할 실패 수)
    max_attempts: int            # 2 또는 3
    max_new_tokens: dict[int, int]
    context_limit: int           # 16384
    diag: bool
    n_diag: int
    oracle_counterfactual: bool
    self_perception_probe: bool

def build_tree(actor: ActorItem, images: list, engine, verifier, prompts: dict,
               cfg: RolloutConfig, policy_version: int, route_rule: str) -> ProblemTree
def seeds(problem_id: str, policy_version: int, turn: int, branch: str, n: int) -> list[int]
def select_failures(samples: list[Sample], k: int, key: str) -> list[Sample]
```

- 입력 토큰 연결 규칙(가장 중요): 턴 t+1의 입력 = **엔진이 턴 t에 실제로 소비한 `prompt_token_ids` + 생성 `completion_token_ids` + (완성이 `<|im_end|>`(151645)로 끝나지 않으면 `<|im_end|>`; 이렇게 덧붙인 토큰은 생성 토큰이 아니므로 action mask 0) + 토크나이저로 인코딩한(`add_special_tokens=False`) `"\n<|im_start|>user\n" + 피드백 + "<|im_end|>\n<|im_start|>assistant\n"`**. 모델 출력 텍스트를 다시 토크나이즈하지 않는다. 이미지는 첫 user turn에만 있고 매 요청에 같은 이미지 데이터를 다시 넘긴다.
- 엔진이 반환한 `prompt_token_ids`가 우리가 넘긴 ID와 다르면 `RuntimeError`(G8).
- `seeds`: `int.from_bytes(sha256(f"{problem_id}|{policy_version}|{turn}|{branch}|{k}".encode()).digest()[:8], "big") % 2**31`.
- `select_failures`: §7.2 규칙(서로 다른 추출 답 우선, 그다음 seed 해시 순).
- `CONTEXT_BLOCKED`: 입력 길이 + `max_new_tokens[t+1]` > `context_limit`이면 그 분기를 만들지 않고 tree 로그에 기록.
- 그룹에 `SYSTEM_ERROR`/`UNVERIFIABLE`이 있으면 `usable=False`.
- 진단 그룹: 결국 맞힌 경로(턴 ≥2 정답)마다, `diagnosis_target`이 None이 아니면 생성. 한 문제당 최대 1개.
- oracle/probe 생성은 `OracleProbe` 레코드(별도 타입)로만 저장하고 `Group`으로 만들지 않는다.
- **테스트(CPU, 가짜 엔진):** 정해진 응답 시나리오로 트리 모양·그룹 수·선택 규칙·토큰 연결·컨텍스트 차단·진단 그룹 생성 조건을 검증. 가짜 엔진은 입력 토큰을 그대로 기록해 연결 규칙을 확인.

### 15.12 `engines.py` — rollout 엔진 (Wave 2, **Opus**)

```python
@dataclass
class RolloutRequest:
    request_id: str
    prompt_token_ids: list[int] | None   # 턴 ≥2
    messages: list[dict] | None          # 턴 1 (chat template 적용 전)
    images: list                          # PIL.Image (전처리 완료본)
    n: int
    max_new_tokens: int
    seeds: list[int]

@dataclass
class Completion:
    token_ids: list[int]; logprobs: list[float]; finish_reason: str; text: str

@dataclass
class RolloutResult:
    request_id: str; prompt_token_ids: list[int]; completions: list[Completion]

class VLLMRolloutEngine:
    def __init__(self, model_path: Path, image_cfg: dict, kv_cache_memory_bytes: int,
                 max_num_seqs: int, max_model_len: int)
    policy_version: int
    def sample(self, requests: list[RolloutRequest]) -> list[RolloutResult]
    def push_weights(self, named: Iterable[tuple[str, torch.Tensor]], version: int) -> None
    def probe_logprobs(self, request: RolloutRequest, completion_ids: list[int]) -> list[float]

class HFRolloutEngine:   # CPU 테스트·GPU smoke 비교용, 같은 인터페이스
```

- 생성 전 `os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"`.
- `LLM(model=..., dtype="bfloat16", enforce_eager=False, enable_prefix_caching=False, mm_processor_cache_gb=0, limit_mm_per_prompt={"image": 5, "video": 0}, mm_processor_kwargs={min/max pixels = 평가와 동일}, kv_cache_memory_bytes=..., max_num_seqs=..., max_model_len=...)`. 정확한 인자 이름은 vLLM 0.11.0 `EngineArgs`에서 확인하고 다르면 gate G0에서 실패시킨다.
- `SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=0, presence_penalty=0.0, repetition_penalty=1.0, max_tokens=…, seed=…, logprobs=0)`. 샘플마다 seed가 다르므로 n개 요청으로 펼친다.
- 이미지 입력: 기존 `vllm_backend.py`처럼 CPU에서 `prepare_protocol_inputs`로 전처리하고, 엔진 prompt ids가 CPU processor ids와 같은지 확인한다. 이 전처리 결과(`pixel_values`, `image_grid_thw`)는 학습 forward에도 그대로 쓴다.
- `push_weights`: `llm.apply_model(lambda m: m.load_weights(named))` (동일 프로세스) 후 `llm.reset_prefix_cache()`가 True인지 assert. `policy_version = version`.
- **병합 가중치 생성(`policy.merged_named_tensors`)은 out-of-place:** LoRA 층은 `base_layer.weight.float() + get_delta_weight("default").float()` → BF16, Merger는 `modules_to_save["default"]`의 파라미터 → BF16, 이름에서 `base_model.model.`·`.base_layer`·`.modules_to_save.default` 제거. PEFT `merge_adapter()` 금지.
- 매 push 후 고정 probe 2개(단일 이미지·다중 이미지)의 vLLM logprob와 HF logprob를 비교해 token 평균 |Δ|가 protocol 임계(초기 0.02 nats, G7에서 확정)를 넘으면 `RuntimeError`.

### 15.13 `policy.py`와 `checkpoint.py` (policy.py: Wave 2 **Opus**; checkpoint.py·export.py: Wave 3 Sonnet → Opus 리뷰, §16.3 W3-checkpoint)

```python
@dataclass
class PolicyBundle:
    model: "PeftModel"; processor; inventory: dict
    params: dict[str, list[torch.nn.Parameter]]   # {"L": [...], "M": [...], "V": [...]}
    route: VisionRoute
    base_sha256: dict[str, str]                   # 고정 base·Merger 원본 해시 (불변 검사)

def build_policy(model_path: Path, arm: str, protocol: dict) -> PolicyBundle
def completion_logprobs(bundle, sample_inputs: dict, completion_len: int, *, chunk: int = 1024) -> torch.Tensor
def reference_logprobs(bundle, sample_inputs, completion_len) -> torch.Tensor
def merged_named_tensors(bundle) -> Iterator[tuple[str, torch.Tensor]]
def assert_base_unchanged(bundle) -> None

def save(bundle, optimizer, scheduler, step: int, cursor: dict, rng: dict, out: Path) -> dict
def load_resume(model_path: Path, ckpt: Path, protocol: dict) -> tuple[PolicyBundle, optimizer, scheduler, int, dict]
def load_eval(model_path: Path, ckpt: Path) -> PolicyBundle
```

- `build_policy`: `train.load_base` → `inspect_modules` → `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=targets["L"] (+targets["V"] if arm uses vision), modules_to_save=[4 merger paths])` → `get_peft_model` → 각 `modules_to_save["default"]`만 `.float()` → `requires_grad` 정리 → 학습 파라미터 수 검사(L 33,030,144 / M 109,105,152 / V 2,359,296) → `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`, `enable_input_require_grads()`, `config.use_cache=False`.
- arm 이름: `MOD-FML`(V 없음), `MOD-FVML`, `MOD-RV`, `MOD-RR`, `MOD-RS`(V 있음).
- `completion_logprobs`: `model(input_ids, attention_mask, pixel_values, image_grid_thw, use_cache=False)`의 hidden state 중 완성 위치만 `chunk`씩 `lm_head` → `float()` → `log_softmax` → gather. forward는 `torch.autocast("cuda", dtype=torch.bfloat16)`.
- `reference_logprobs`: `torch.no_grad()` + `model.disable_adapter()` + 같은 autocast, 끝나면 `bundle.route` 상태 재적용.
- checkpoint 내용: LoRA(safetensors), Merger FP32 사본(safetensors), optimizer·scheduler state, step, 데이터 cursor(epoch, 위치, 버린 문제 목록), Python/NumPy/Torch/CUDA RNG, protocol·prompts·data manifest·코드 SHA, 학습 파라미터별 SHA. 디렉터리는 `mkdir(exist_ok=False)`.
- **테스트(CPU tiny model):** 파라미터 수(작은 모델 기준 비율), reference == 새 모델 logits(비트), 몇 step 후에도 reference 불변, `disable_adapter` 후 route 재적용, 저장→새 프로세스 로드→logits 동일, 재개 후 다음 step loss 동일.

### 15.14 `trainer.py` (Wave 4, **Opus**)

```python
def train(protocol_path: Path, data_root: Path, image_root: Path, arm: str, algorithm: str,
          output: Path, deadline_utc: str, resume: Path | None = None) -> dict
```

- `algorithm ∈ {"ALG-T1", "ALG-NB", "ALG-NB3", "ALG-NB+D", "ALG-NB+D+V", "ALG-NB-noteonly", "CTL-RAND", "CTL-FORMAT", "CTL-BLIND"}`.
  - `ALG-T1`: `max_attempts=1`.
  - `ALG-NB-noteonly`: turn ≥2 샘플에서 `Error check:` 블록 토큰만 action mask=1. 블록 범위: 생성 토큰을 앞에서부터 누적 decode해 각 토큰의 문자 끝 위치를 얻고, 텍스트에서 `Error check:` 줄 시작부터 다음 `Observed facts:` 또는 `Solution:` 줄 시작 직전까지의 문자 구간과 겹치는 토큰을 1로 한다. 블록이 없으면 그 샘플의 mask는 전부 0(그 샘플은 학습 신호 없음)이고 개수를 기록한다.
  - `CTL-RAND`: 모든 풀이 샘플의 보상을 `Bernoulli(p)`로 교체, p는 같은 step의 실제 정답률(seed 고정). 
  - `CTL-FORMAT`: 보상 = 1[`extract_final`의 status == OK].
  - `CTL-BLIND`: 학습 rollout의 이미지를 같은 크기 회색으로 교체(평가는 원본).
- §10.6의 순서를 그대로. 매 step `metrics.step_record`를 `metrics.jsonl`에 fsync append.
- `stop_for_budget`로 deadline 120초 전 checkpoint 후 `BUDGET_STOP` 종료.
- 중단 규칙(§18.6)에 걸리면 checkpoint 후 `HEALTH_STOP:<사유>` 종료.

### 15.15 `evaluate_external.py`, `eval_mmmu8k.py` (Wave 4), `export.py` (Wave 3, W3-checkpoint) — Sonnet / Opus 리뷰

- `evaluate_external.evaluate(model_dir, split_root, output, modes, k_list=(1,4,8), seeds=(3407,))`
  - `first`: 평가 sampling(T 0.7, top_p 0.8, top_k 20, presence 1.5)과 temperature 1 두 설정 모두 기록. 주 지표는 **평가 sampling의 첫 시도 정확도**.
  - `pass@k`: T=1, n=8, 비편향 추정식 `1 − C(n−c,k)/C(n,k)`. 객관식은 별도 표기.
  - `forced_second`: 정오를 알려주지 않고 "Review your solution and give a final answer" 2차 → Δi→c, Δc→i.
  - `feedback_second`: 학습과 같은 정오 피드백 2차(외부 진단 전용).
  - `blind`: 회색 이미지 첫 시도.
  - 슬롯 정확도(obs 프롬프트), 길이 분포, 절단률, NO_FINAL률.
- `export.export_merged(ckpt, model_path, out, arm)`: out-of-place 병합 BF16 HF 디렉터리 + `export_manifest.json`(`status: "EXPORTED"`, `format: "derived-hf-bf16-v1"`, `arm`, `base_id: "Qwen/Qwen3-VL-4B-Instruct"`, `base_revision: "ebb281ec…"`, `files: {상대경로: sha256}`, `precision_policy` — `experiments/input_recovery/runner.py:63-77` `model_identity`가 검사하는 키 전부). `arm` 값은 `eval_mmmu8k.run`에 넘기는 arm 문자열과 같아야 한다.
- `eval_mmmu8k.run(model_dir, arm, data_root, output, slots=4, deadline_utc)`: `experiments/input_recovery/protocol.yaml`(protocol_id `input-recovery-v1`, 8,192 출력, V8, 원래 이미지 설정)을 로드해 `create_backend`로 MMMU validation **900문항 전체**를 P0로 평가하고(900문항은 `src/mmdl/evaluation/datasets/mmmu.py` `load_validation`+`separate_sample`, P0 메시지는 `src/mmdl/evaluation/prompt.py` `build_messages`, seed는 `src/mmdl/runtime/contracts.py` `sample_seed`로 만든다 — §14.2. `experiments/input_recovery/runner.make_plan`은 arm 이름을 F0/FL/FM/FV/FML로 제한하고 48문항 subset만 만들므로 쓰지 않는다) `score_with_parser`(V8)로 채점. 문항별 seed는 기존 규칙 `SHA256("3407:{sample_id}")`. 30과목·약점 18과목(540)·나머지 12과목(360)·상한 종료·NO_PARSE를 보고. **F0와 FT를 같은 GPU 종류에서 연속 실행**.

### 15.16 `metrics.py` — step 기록과 중단 판정 (Wave 3, Haiku)

```python
def step_record(**fields) -> dict          # 아래 키가 모두 있어야 함, 없으면 ValueError
def health_check(history: list[dict], protocol: dict, f0_reference: dict) -> list[str]   # 걸린 HEALTH_STOP 사유 목록
```

`step_record` 필수 키 (값이 없으면 `null`):

| 키 | 타입 | 뜻 |
|---|---|---|
| `step`, `policy_version` | int | |
| `n_problems`, `n_discarded` | int | 사용한 문제 수, dynamic sampling으로 버린 문제 수 |
| `n_groups_by_kind` | dict[str,int] | `T1/T2/T3/DIAG` |
| `reward_mean_by_kind` | dict[str,float] | |
| `degenerate_fraction` | float | advantage 전부 0인 그룹 비율 |
| `kl_mean` | float | 토큰당 k3 평균 |
| `masked_fraction`, `mean_abs_lp_diff` | float | MIS 결과, 토큰당 \|log π_old − log π_samp\| |
| `grad_norm` | dict[str,float\|null] | `L`, `M`, `V` (clip 전) |
| `vision_on_fraction` | float | VISION_ON 문제 비율 |
| `len_median`, `truncation_rate`, `short_fraction_le32` | float | 풀이 샘플 기준 |
| `unverifiable_fraction` | float | |
| `dev_first_attempt_acc` | float\|null | 12 step마다만 값 |
| `seconds` | dict[str,float] | `rollout`, `old`, `ref`, `train`, `push` |

`health_check`는 §18.6 표의 조건을 `protocol["health"]` 값으로 판정한다. `f0_reference`는 Phase A에서 잰 F0의 길이 중앙값과 dev 정확도다. 테스트는 각 조건의 경계값 바로 위·아래를 넣는다.

### 15.17 데이터 디렉터리와 이미지 로딩

```text
$MMDL_DATA_ROOT/training/notebook_rl/<build_id>/          # build_id 예: pilot-v1-20261020
├── manifest.yaml               # vault.write_split 결과 + 오염 receipt SHA + 생성 코드 SHA
├── <split>/actor_inputs.jsonl  # split ∈ train, dev, sealed, style_sealed
├── <split>/verifier_private.jsonl
├── <split>/lineage.jsonl
├── images/<source>/<split>/<index:06d>.png        # ActorItem.image_paths는 images/ 기준 상대경로
├── contamination/receipt_v2.json
└── selection_v1.yaml           # Phase A 선별 동결 (아래)
```

- 이미지 로딩은 `dataset.load_images(actor: ActorItem, image_root: Path) -> list[PIL.Image.Image]` 하나만 쓴다. 파일 SHA256이 `image_sha256`과 다르면 `ValueError`. trainer와 evaluate가 이 함수로 읽어 `rollout.build_tree(images=...)`와 엔진에 넘긴다.
- `selection_v1.yaml` 형식:

```yaml
selection_id: pilot-v1
build_id: pilot-v1-20261020
phase_a_report_sha256: "<sha>"
rules: {max_pass_rate: 0.75, max_zero_pass_share: 0.5, blind_threshold: "max(1/k, 0.25)", truncation_family_max: 0.10}
max_new_tokens: {1: 3072, 2: 3072, 3: 3072}       # §7.6로 결정한 값
selected_train: ["<problem_id>", ...]              # 순서 = 학습 셔플 전 정렬 순서
excluded: [{problem_id: "...", reason: "BLIND_SOLVABLE|TOO_EASY|FAMILY_TRUNCATION|CONTEXT|ZERO_PASS_CAP"}]
frozen_at_utc: "2026-..."
```

### 15.18 명령줄 인터페이스

로컬(`python -m fine_tuning.notebook_rl.cli <cmd>`):

| 명령 | 필수 인자 | 하는 일 |
|---|---|---|
| `generate` | `--generator NAME --split S --count N --seed INT --mode-mix 0.5 --style default\|shift --out DIR` | 생성기 실행 → 임시 디렉터리에 problem 묶음 |
| `build` | `--inputs DIR... --build-id ID --out ROOT` | vault 기록, split 확인, manifest |
| `audit` | `--build ROOT --index PATH --stages 1-7` | §13.3 로컬 단계, receipt 작성 |
| `verify-data` | `--build ROOT` | D1·D2·D5·D7 재검사 (재생성 해시 비교 포함) |
| `select` | `--build ROOT --phase-a REPORT --out selection_v1.yaml` | Phase A 결과로 선별 동결 |
| `report` | `--runs DIR... --out DIR` | 실험군 비교표·CI 산출 |

RunPod(`python -m fine_tuning.notebook_rl.runpod --stage <stage>`), 공통 인자 `--commit SHA --runtime-root DIR --build ROOT --deadline-utc ISO --owned-pod-id ID`:

| stage | 추가 인자 | 출력 |
|---|---|---|
| `gates` | `--index-out DIR` | `gates/report.json` (G0~G13, index v2, D6 임베딩·OCR) |
| `phase_a` | `--n1 8 --blind-n 4 --retry-subsample 128` | `phase_a/report.json` |
| `train` | `--arm MOD-… --algorithm ALG-…\|CTL-… --seed INT --selection PATH [--resume CKPT]` | `runs/<arm>-<alg>-<seed>/` (rollouts, metrics.jsonl, checkpoints) |
| `export` | `--checkpoint DIR --arm NAME` | `exports/<name>/` + `export_manifest.json` |
| `eval_external` | `--model DIR --splits dev,sealed,style_sealed --modes first,pass_at_k,forced_second,feedback_second,blind` | `eval/<name>/results.jsonl`, `summary.json` |
| `eval_mmmu8k` | `--model DIR --arm NAME --slots 4` | `mmmu8k/<name>/predictions.jsonl`, `summary.json` |
| `pack` | `--out TAR` | 닫힌 결과 tar + SHA 목록 |

### 15.19 `evaluate_external` 출력 형식

`results.jsonl` 한 줄 = (모델, 문항, mode, sample k): `{model, problem_id, family, split, mode, k, seed, status, extracted, correct, finish_reason, n_tokens, slots|null}`. `summary.json`: split·family·mode별 정확도(문항당 n의 평균), pass@k(비편향 추정), Δi→c·Δc→i(분모 포함), 소거 기준선, 절단률, NO_FINAL률, 길이 중앙값. 모든 수치에 표본 수를 함께 적는다.

## 16. 구현 작업 계약 (wave 단위)

### 16.1 공통 실행 환경과 명령

| 환경 | 경로 | 용도 |
|---|---|---|
| 학습 CPU venv (기존) | `/home/jang2/mmdl-env` (torch 2.8.0, transformers 4.57.1, peft 0.17.1, ruff 0.14.3, mypy 1.18.2; vLLM 없음) | `fine_tuning/tests/test_notebook_rl_*.py`, ruff/mypy |
| vLLM venv (기존) | `/home/jang2/mmdl-vllm-env` (위 + vLLM 0.11.0; ruff·mypy는 설치되어 있지 않음 — 2026-10-16 실측) | 엔진 import 검사 |
| 데이터 venv (신규, W0-2가 생성) | `/home/jang2/mmdl-envs/notebook-data` (`fine_tuning/notebook_rl/requirements-data.lock`) | 생성기와 `data_tests` |

```bash
# 저장소 루트에서
cd /mnt/c/projects/multimodel
export PYTHONPATH="$PWD/src:$PWD"
# 학습 venv 단위 테스트 하나
/home/jang2/mmdl-env/bin/python -m unittest discover -s fine_tuning/tests -p "test_notebook_rl_answers.py" -v
# 데이터 venv 생성기 테스트 하나
/home/jang2/mmdl-envs/notebook-data/bin/python -m unittest discover -s fine_tuning/notebook_rl/data_tests -p "test_gen_music.py" -v
# lint / type
/home/jang2/mmdl-env/bin/python -m ruff check fine_tuning/notebook_rl fine_tuning/tests/test_notebook_rl_*.py
/home/jang2/mmdl-env/bin/python -m mypy --follow-imports=skip --ignore-missing-imports fine_tuning/notebook_rl/<file>.py
```

테스트는 `unittest`로 쓴다(pytest는 lock에 없음). 기존 관례: 모듈 docstring에 범위 명시, `unittest.TestCase`, `tempfile.TemporaryDirectory`, `self.subTest`, `assertRaisesRegex`로 실패 메시지 확인, 네트워크·모델 가중치 없음. CPU 정책 테스트는 `fine_tuning/tests/test_training.py:20-47`의 `tiny_model()`을 import해 쓴다.

**모든 계약의 공통 Done when:** 해당 테스트 통과 + `ruff check <변경 파일>` 통과 + `git diff --stat`에 계약의 `files` 외 변경 없음.

### 16.2 wave 순서

```text
W0 (설계자: Opus/Fable, 순차)  schema · protocol · prompts · registry · generators/common · data lock
  │
W1 (병렬)  guards · vault · answers · observations · prompts · losses · 생성기 6종 · 어댑터 2종
  │
W2 (병렬)  verifier · attribution · contamination · dataset · policy · engines
  │
W3 (병렬)  rollout · checkpoint+export · metrics
  │
W4 (병렬)  trainer · evaluate_external · eval_mmmu8k
  │
W5         runpod controller + cli + gpu_gates  →  Fable 최종 감사  →  MMDL_test 게시
```

한 wave는 이전 wave가 **병합되고 테스트가 통과한 뒤**에만 시작한다. 같은 wave 안에서 파일 소유자는 하나다.

### 16.3 계약 목록 요약

| ID | 담당 | 위험 | 파일 | 의존 |
|---|---|---|---|---|
| W0-1 | opus | high | `schema.py`, `protocol_v1.yaml`, `prompts_v1.yaml`, `source_registry_v1.yaml`, `__init__.py`, `README.md`, `tests/test_notebook_rl_schema.py` | — |
| W0-2 | opus | medium | `generators/__init__.py`, `generators/common.py`, `requirements-data.lock`, `data_tests/test_common.py` | W0-1 |
| W1-guards | sonnet | medium | `guards.py`, `tests/test_notebook_rl_guards.py` | W0 |
| W1-vault | sonnet | medium | `vault.py`, `tests/test_notebook_rl_vault.py` | W0 |
| W1-answers | haiku | medium | `answers.py`, `tests/test_notebook_rl_answers.py` | W0 |
| W1-obs | haiku | medium | `observations.py`, `tests/test_notebook_rl_observations.py` | W0 |
| W1-prompts | sonnet | medium | `prompts.py`, `tests/test_notebook_rl_prompts.py` | W0 |
| W1-losses | opus | high | `losses.py`, `tests/test_notebook_rl_losses.py` | W0 |
| W1-gen-chart | sonnet | low | `generators/chart.py`, `data_tests/test_gen_chart.py` | W0 |
| W1-gen-music | sonnet | medium | `generators/music.py`, `data_tests/test_gen_music.py` | W0 |
| W1-gen-chem | sonnet | medium | `generators/chem.py`, `data_tests/test_gen_chem.py` | W0 |
| W1-gen-statics | sonnet | medium | `generators/statics.py`, `data_tests/test_gen_statics.py` | W0 |
| W1-gen-circuit | sonnet | medium | `generators/circuit.py`, `data_tests/test_gen_circuit.py` | W0 |
| W1-gen-graph | sonnet | low | `generators/graph.py`, `data_tests/test_gen_graph.py` | W0 |
| W1-ad-tabmwp | sonnet | medium | `adapters/__init__.py`, `adapters/tabmwp_rl.py`, `tests/test_notebook_rl_tabmwp.py` | W0 |
| W1-ad-geometry | sonnet | medium | `adapters/geometry.py`, `data_tests/test_ad_geometry.py` | W0 |
| W2-verifier | sonnet | medium | `verifier.py`, `tests/test_notebook_rl_verifier.py` | answers, obs, vault |
| W2-attribution | sonnet | medium | `attribution.py`, `tests/test_notebook_rl_attribution.py` | obs, verifier 인터페이스(schema) |
| W2-contamination | opus | high | `contamination.py`, `tests/test_notebook_rl_contamination.py` | guards, vault |
| W2-dataset | sonnet | medium | `dataset.py`, `tests/test_notebook_rl_dataset.py` | vault, generators registry |
| W2-policy | opus | high | `policy.py`, `tests/test_notebook_rl_policy.py` | losses |
| W2-engines | opus | high | `engines.py`, `tests/test_notebook_rl_engines.py` | prompts |
| W3-rollout | opus | high | `rollout.py`, `tests/test_notebook_rl_rollout.py` | engines, verifier, attribution, prompts |
| W3-checkpoint | sonnet | high | `checkpoint.py`, `export.py`, `tests/test_notebook_rl_checkpoint.py` | policy |
| W3-metrics | haiku | low | `metrics.py`, `tests/test_notebook_rl_metrics.py` | schema |
| W4-trainer | opus | high | `trainer.py`, `tests/test_notebook_rl_trainer.py` | 전부 |
| W4-eval-ext | sonnet | medium | `evaluate_external.py`, `tests/test_notebook_rl_eval_external.py` | engines, verifier, export |
| W4-eval-mmmu | sonnet | medium | `eval_mmmu8k.py`, `tests/test_notebook_rl_eval_mmmu.py` | export |
| W5-runpod | sonnet | medium | `runpod.py`, `cli.py`, `README.md`, `tests/test_notebook_rl_runpod.py` | 전부 |
| W5-gates | opus | high | `gpu_gates.py`, `tests/test_notebook_rl_gates_cpu.py` | 전부 |

### 16.4 tiered-implement 입력용 계약 (JSON)

아래 JSON은 `Workflow({name: "tiered-implement", args: {goal, contracts, finalAudit}})`에 그대로 넣을 수 있다. 경로는 모두 저장소 루트 기준이며 `NB = fine_tuning/notebook_rl`의 약어를 쓰지 않고 전체 경로를 쓴다. `done`의 `PY`는 `/home/jang2/mmdl-env/bin/python`, `PYD`는 데이터 venv, 둘 다 `PYTHONPATH="$PWD/src:$PWD"`를 전제로 한다.

**Wave 0**

```json
{
  "goal": "notebook_rl Wave 0: freeze record schemas, protocol, prompts, source registry and generator common helpers",
  "finalAudit": true,
  "contracts": [
    {
      "id": "W0-1", "coder": "opus", "risk": "high",
      "files": ["fine_tuning/notebook_rl/__init__.py", "fine_tuning/notebook_rl/schema.py",
                "fine_tuning/notebook_rl/protocol_v1.yaml", "fine_tuning/notebook_rl/prompts_v1.yaml",
                "fine_tuning/notebook_rl/source_registry_v1.yaml", "fine_tuning/notebook_rl/README.md",
                "fine_tuning/tests/test_notebook_rl_schema.py"],
      "context": "Design doc claudedocs/research_error_notebook_self_correction_rl_plan_20261016.md sections 7-11, 15.1, Appendix A-C. Existing style: fine_tuning/nvw.py dataclass-free dict style is NOT required; use frozen dataclasses. Test style: fine_tuning/tests/test_training.py.",
      "interface": "Exactly the dataclasses and Literal types of section 15.1, plus to_json(obj)->str, from_json(cls, s)->obj (unknown keys -> ValueError), validate_actor(ActorItem)->None, load_protocol(path)->dict returning dict with key 'sha256'. protocol_v1.yaml holds every numeric value of sections 7.3, 10.3, 10.5, 11.1, 12.7 and the arm/algorithm names of 11.2 and 15.14. prompts_v1.yaml holds Appendix A texts verbatim plus a per-family OBSERVATION_INSTRUCTION map.",
      "behavior": "Pure data module, no torch import. from_json rejects ActorItem JSON containing any of gold/solution/answer/slots/target_text/required_slots. load_protocol computes sha256 of the raw file bytes.",
      "done": "PY -m unittest discover -s fine_tuning/tests -p 'test_notebook_rl_schema.py' -v passes; tests cover round-trip of every dataclass, forbidden-key rejection, protocol sha stability.",
      "outOfScope": "Any file outside the listed ones; fine_tuning/train.py, modules.py, protocol.json; src/; MMDL/.",
      "depends": []
    },
    {
      "id": "W0-2", "coder": "opus", "risk": "medium",
      "files": ["fine_tuning/notebook_rl/generators/__init__.py", "fine_tuning/notebook_rl/generators/common.py",
                "fine_tuning/notebook_rl/requirements-data.lock", "fine_tuning/notebook_rl/data_tests/__init__.py",
                "fine_tuning/notebook_rl/data_tests/test_common.py"],
      "context": "Section 15.4 of the design doc. Determinism pattern: fine_tuning/nvw.py build_episode (line 350) and family_split (line 37).",
      "interface": "GeneratedProblem dataclass; structure_hash(obj)->str; png_bytes(image_or_fig, dpi:int)->bytes; split_of(structure_hash, ratios=(0.8,0.1,0.1))->Split; build(...)->GeneratedProblem as in 15.4; REGISTRY: dict[str, module] in generators/__init__.py populated lazily by name (import inside function so missing optional deps do not break import of the package).",
      "behavior": "png_bytes strips all PNG metadata chunks except IHDR/PLTE/IDAT/IEND so identical pixels give identical bytes. Create venv /home/jang2/mmdl-envs/notebook-data with python3.12 and pip install exact pins: the versions named in design doc section 12.5 (music21 10.5, verovio 6.3, rdkit 2026.9, schemdraw 0.23, sympy 1.14, PyNiteFEA 3.2) plus matplotlib, networkx, imagehash, pillow==11.3.0, numpy; if a named version is not on PyPI use the latest stable and record the deviation in README.md; write resolved 'pip freeze' to requirements-data.lock.",
      "done": "PYD -m unittest discover -s fine_tuning/notebook_rl/data_tests -p 'test_common.py' -v passes (determinism of png_bytes, split_of distribution 80/10/10 +-2% over 10000 hashes, build() output validates with schema.validate_actor).",
      "outOfScope": "Generators themselves; env/*.lock at repo root.",
      "depends": ["W0-1"]
    }
  ]
}
```

**Wave 1**

```json
{
  "goal": "notebook_rl Wave 1: guards, vault, strict answers, observations, prompts, losses, six generators, two adapters",
  "finalAudit": true,
  "contracts": [
    {"id": "W1-guards", "coder": "sonnet", "risk": "medium",
     "files": ["fine_tuning/notebook_rl/guards.py", "fine_tuning/tests/test_notebook_rl_guards.py"],
     "context": "Design doc 1.3 and 15.2. Python sys.addaudithook docs.",
     "interface": "install_guards(*, forbidden_roots: list[Path], vault_files: list[Path], allow_vault: bool, allow_benchmark: bool) -> None; assert_no_teacher_imports() -> None",
     "behavior": "As 15.2. Second install_guards call raises RuntimeError. Tests must run guard installation inside a subprocess (audit hooks cannot be removed).",
     "done": "PY -m unittest discover -s fine_tuning/tests -p 'test_notebook_rl_guards.py' -v",
     "outOfScope": "Everything else.", "depends": []},
    {"id": "W1-vault", "coder": "sonnet", "risk": "medium",
     "files": ["fine_tuning/notebook_rl/vault.py", "fine_tuning/tests/test_notebook_rl_vault.py"],
     "context": "Design doc 15.3; schema.py from W0-1; write-once pattern fine_tuning/audit.py dump/write_rows (lines 25-44, open mode 'x').",
     "interface": "ACTOR_KEYS; write_split(root, actors, verifiers, lineages)->dict; read_actors(root)->list[ActorItem]; read_verifier(root)->dict[str,VerifierItem]; read_lineage(root)->dict[str,Lineage]",
     "behavior": "As 15.3 including leak_review list and problem_id set equality.",
     "done": "PY -m unittest discover -s fine_tuning/tests -p 'test_notebook_rl_vault.py' -v",
     "outOfScope": "Contamination checks (W2).", "depends": []},
    {"id": "W1-answers", "coder": "haiku", "risk": "medium",
     "files": ["fine_tuning/notebook_rl/answers.py", "fine_tuning/tests/test_notebook_rl_answers.py"],
     "context": "Design doc 9.1, 9.6, 15.5. Do NOT import or copy src/mmdl/evaluation/final_answer_parser_v8.py (different purpose).",
     "interface": "VERSION='strict-final-v1'; Extraction(status, answer); extract_final(text, finish_reason)->Extraction; compare(answer, item: VerifierItem)->bool|None",
     "behavior": "Exactly 15.5. Use decimal.Decimal and fractions.Fraction only, never float.",
     "done": "PY -m unittest discover -s fine_tuning/tests -p 'test_notebook_rl_answers.py' -v with one test per row of the 9.6 table plus the 15.5 list.",
     "outOfScope": "MCQ letter/content conflict handling (verifier.py).", "depends": []},
    {"id": "W1-obs", "coder": "haiku", "risk": "medium",
     "files": ["fine_tuning/notebook_rl/observations.py", "fine_tuning/tests/test_notebook_rl_observations.py"],
     "context": "Design doc 8.3, 9.3, 15.6.",
     "interface": "parse_observations(text)->dict[str,list[str]]; normalize_slot(value, slot_type)->str|None; compare_slots(parsed, item)->SlotReport",
     "behavior": "Exactly 15.6.",
     "done": "PY -m unittest discover -s fine_tuning/tests -p 'test_notebook_rl_observations.py' -v",
     "outOfScope": "Cause attribution.", "depends": []},
    {"id": "W1-prompts", "coder": "sonnet", "risk": "medium",
     "files": ["fine_tuning/notebook_rl/prompts.py", "fine_tuning/tests/test_notebook_rl_prompts.py"],
     "context": "Design doc Appendix A and 15.9; P0 templates prompts/mmmu_mcq_v1.txt and prompts/mmmu_open_v1.txt (read-only).",
     "interface": "load_prompts, p0_body, turn1_text, retry_text, diagnose_text, oracle_text, probe_text, user_message as in 15.9",
     "behavior": "p0_body output must be byte-identical to the repo P0 templates filled with the same question/options (no hint).",
     "done": "PY -m unittest discover -s fine_tuning/tests -p 'test_notebook_rl_prompts.py' -v",
     "outOfScope": "Token stitching (rollout.py).", "depends": []},
    {"id": "W1-losses", "coder": "opus", "risk": "high",
     "files": ["fine_tuning/notebook_rl/losses.py", "fine_tuning/tests/test_notebook_rl_losses.py"],
     "context": "Design doc 10.2-10.3 and 15.10. Reference formulas: TRL v1.2.0 grpo_trainer.py loss section (read only, do not add TRL dependency).",
     "interface": "group_advantages, mis_mask, k3, group_loss, combine as in 15.10",
     "behavior": "Exactly 10.3. Detach old_lp, ref_lp, adv, seq_keep internally.",
     "done": "PY -m unittest discover -s fine_tuning/tests -p 'test_notebook_rl_losses.py' -v covering every test listed in 15.10 including autograd gradient equality with a hand-computed example and micro-batch additivity.",
     "outOfScope": "Model forward code.", "depends": []},
    {"id": "W1-gen-chart", "coder": "sonnet", "risk": "low",
     "files": ["fine_tuning/notebook_rl/generators/chart.py", "fine_tuning/notebook_rl/data_tests/test_gen_chart.py"],
     "context": "Design doc 12.5, 15.4 (generator contract and G-CHART row). Helpers: generators/common.py.",
     "interface": "GENERATOR_VERSION, FAMILY, OBSERVATION_INSTRUCTION, generate(index, seed, split, mode, style='default')->GeneratedProblem, self_check(p)->None",
     "behavior": "15.4 contract: determinism, split by structure_hash, two independent answer computations, 10-option MCQ with programmatic distractors, open answers >= 60% over 1000 samples, style 'shift' changes font/colors/layout.",
     "done": "PYD -m unittest discover -s fine_tuning/notebook_rl/data_tests -p 'test_gen_chart.py' -v (determinism, split disjointness over 500 items, self_check on 500 items, image-dependence pair test on 100 pairs, OBSERVATION_INSTRUCTION equals prompts_v1.yaml).",
     "outOfScope": "Other generators.", "depends": []},
    {"id": "W1-gen-music", "coder": "sonnet", "risk": "medium",
     "files": ["fine_tuning/notebook_rl/generators/music.py", "fine_tuning/notebook_rl/data_tests/test_gen_music.py"],
     "context": "Design doc 12.5 G-MUS row and 15.4. music21 + verovio from requirements-data.lock; render MusicXML -> SVG -> PNG (cairosvg or verovio PNG output).",
     "interface": "Same generator interface as W1-gen-chart.",
     "behavior": "Questions: interval between two marked notes, chord quality, major key from key signature, time signature, pitch name of a given note. Slots note[m.i] with octave, clef, key_sig, time_sig. Independent check: music21 interval name vs semitone-table computation.",
     "done": "PYD -m unittest discover -s fine_tuning/notebook_rl/data_tests -p 'test_gen_music.py' -v with the same test list as W1-gen-chart.",
     "outOfScope": "Other generators.", "depends": []},
    {"id": "W1-gen-chem", "coder": "sonnet", "risk": "medium",
     "files": ["fine_tuning/notebook_rl/generators/chem.py", "fine_tuning/notebook_rl/data_tests/test_gen_chem.py"],
     "context": "Design doc 12.5 G-CHEM row and 15.4. Input: a local ChEBI SMILES TSV path passed via env NOTEBOOK_RL_CHEBI_TSV (CC BY 4.0); record its sha256 in lineage.parents.",
     "interface": "Same generator interface.",
     "behavior": "Sample molecules with 6-30 heavy atoms; questions on molecular formula, ring count, H-bond donors, count of a named functional group (SMARTS). Independent check: RDKit descriptor vs atom counting from the molecule graph.",
     "done": "PYD -m unittest discover -s fine_tuning/notebook_rl/data_tests -p 'test_gen_chem.py' -v (tests use a 50-row fixture TSV created inside the test).",
     "outOfScope": "Downloading ChEBI.", "depends": []},
    {"id": "W1-gen-statics", "coder": "sonnet", "risk": "medium",
     "files": ["fine_tuning/notebook_rl/generators/statics.py", "fine_tuning/notebook_rl/data_tests/test_gen_statics.py"],
     "context": "Design doc 12.5 G-STAT row and 15.4. sympy.physics.continuum_mechanics.beam; draw with matplotlib.",
     "interface": "Same generator interface.",
     "behavior": "Simply supported and cantilever beams, 1-3 point loads and 0-1 uniform load, integer lengths/positions/magnitudes; questions: support reaction, max bending moment location/value. Independent check: sympy Beam vs equilibrium equations with Fraction arithmetic.",
     "done": "PYD -m unittest discover -s fine_tuning/notebook_rl/data_tests -p 'test_gen_statics.py' -v",
     "outOfScope": "Trusses (expansion).", "depends": []},
    {"id": "W1-gen-circuit", "coder": "sonnet", "risk": "medium",
     "files": ["fine_tuning/notebook_rl/generators/circuit.py", "fine_tuning/notebook_rl/data_tests/test_gen_circuit.py"],
     "context": "fine_tuning/circuit_bridge.py solve_nodal (41), verify_series_parallel (86), canonical_structure (100), render (144) - import only, do not modify. schemdraw for new topologies.",
     "interface": "Same generator interface.",
     "behavior": "Reuse the existing DC topology plus new series-parallel ladders with 1-2 sources; answers exact Fractions with units; slots comp[...] and wire[...].",
     "done": "PYD -m unittest discover -s fine_tuning/notebook_rl/data_tests -p 'test_gen_circuit.py' -v",
     "outOfScope": "AC/transient circuits.", "depends": []},
    {"id": "W1-gen-graph", "coder": "sonnet", "risk": "low",
     "files": ["fine_tuning/notebook_rl/generators/graph.py", "fine_tuning/notebook_rl/data_tests/test_gen_graph.py"],
     "context": "fine_tuning/graph_bridge.py exclusion_reason (52), render (171); fine_tuning/nvw.py solvers (42-124). networkx for weighted shortest paths, DFA, binary tree traversal.",
     "interface": "Same generator interface.",
     "behavior": "Shortest path length, BFS order, DFA acceptance, tree traversal; reject items solvable by the shortcut audit (exclusion_reason).",
     "done": "PYD -m unittest discover -s fine_tuning/notebook_rl/data_tests -p 'test_gen_graph.py' -v",
     "outOfScope": "CPM networks (expansion).", "depends": []},
    {"id": "W1-ad-tabmwp", "coder": "sonnet", "risk": "medium",
     "files": ["fine_tuning/notebook_rl/adapters/__init__.py", "fine_tuning/notebook_rl/adapters/tabmwp_rl.py",
               "fine_tuning/tests/test_notebook_rl_tabmwp.py"],
     "context": "fine_tuning/tabmwp.py load_source (74), table_facts (105), family_id (132), exclusion_policy (236); tabmwp_evaluate.py lines 181-184 for the ques_type/choices handling pattern (letter = chr(65 + choices.index(answer))). Never read the native 'solution' field.",
     "interface": "convert(native_problem: dict, pid: str, split: Split, image_rel: str, image_sha: str, mode: Mode) -> tuple[ActorItem, VerifierItem, Lineage]",
     "behavior": "Map ans_type to answer_type (integer_number->int, decimal_number->decimal, extractive_text/other_text->short_text, boolean_text->short_text with aliases, multi_choice->mcq). Slots: cell[row][col] from table_facts. label_provenance HUMAN, upstream_origin 'IXL', structure_hash = tabmwp.family_id.",
     "done": "PY -m unittest discover -s fine_tuning/tests -p 'test_notebook_rl_tabmwp.py' -v using an in-test fixture problem dict.",
     "outOfScope": "Selecting which TabMWP rows to use (dataset.py).", "depends": []},
    {"id": "W1-ad-geometry", "coder": "sonnet", "risk": "medium",
     "files": ["fine_tuning/notebook_rl/adapters/geometry.py", "fine_tuning/notebook_rl/data_tests/test_ad_geometry.py"],
     "context": "Design doc 12.4 Geometry3K/PGPS9K rows. InterGPS data format (logic forms, choices, answer). Train split only; exclude PGPS9K items sharing a Geometry3K test diagram.",
     "interface": "convert_geometry3k(record: dict, split, image_rel, image_sha, mode) -> tuple[ActorItem, VerifierItem, Lineage]; convert_pgps9k(...) same",
     "behavior": "Numeric answers become decimal with round_to taken from the choice precision; keep MCQ as mcq only when choices exist; slots from diagram logic forms that state lengths/angles.",
     "done": "PYD -m unittest discover -s fine_tuning/notebook_rl/data_tests -p 'test_ad_geometry.py' -v with in-test fixtures.",
     "outOfScope": "Downloading datasets.", "depends": []}
  ]
}
```

**Wave 2 ~ Wave 5** (같은 형식; 핵심 필드만 적는다. 실행 시 위 형식의 JSON으로 옮긴다.)

| ID | interface (요약) | behavior 핵심 | done |
|---|---|---|---|
| W2-verifier | `Verifier(vault_root).grade(...)`, `.grade_diagnosis(...)` (§15.7) | mcq 글자/내용 충돌 → `INVALID_MULTIPLE`; obs면 SlotReport 첨부; 문자열만 처리 | `test_notebook_rl_verifier.py` |
| W2-attribution | `attempt_cause`, `problem_route`, `diagnosis_target`, `random_route` (§15.8) | §8.3·§8.4 표의 모든 행을 테스트 | `test_notebook_rl_attribution.py` |
| W2-contamination | `load_index(path)` (approve()의 캐시 구성 방식 재현, L7 해결), `scan(split_root, index, stages)` → 단계별 후보, `family_quarantine(...)`, `write_receipt(...)`; 신규 해시: 타일 pHash/dHash(BK-tree), 짧은 질문 5-gram/MinHash, 보기 shingle, `<image\s*\d+>` 정규화, 빈 문자열 해시 제외. SSCD/SigLIP/OCR 단계는 RunPod 전용 함수(`embed_scan`, `ocr_scan`)로 분리하고 CPU 테스트는 가짜 임베딩으로 | §13.3 12단계, 모든 split 검사, family 단위 제외, 판정 어휘 §13.5 | `test_notebook_rl_contamination.py` (가짜 index로 각 단계 탐지·생략 없음) |
| W2-dataset | `generate_pool(generator, n, seed, split, mode_mix)`, `build_splits(...)`, `load_images(actor, image_root)`, `freeze_selection(phase_a_report, rules)` → `selection_v1.yaml` (§12.7, §15.17) | 생성 → vault 기록 → 오염 검사 호출 → manifest; Phase A 선별 규칙 적용·동결 | `test_notebook_rl_dataset.py` |
| W2-policy | §15.13 `build_policy`, `completion_logprobs`, `reference_logprobs`, `merged_named_tensors`, `assert_base_unchanged`, `VisionRoute` | tiny model로: 학습 파라미터 집합 정확, reference == 새 모델 logits(비트), 학습 후에도 reference 불변, route off에서 vision LoRA grad None, `disable_adapter` 후 route 재적용 | `test_notebook_rl_policy.py` |
| W2-engines | §15.12 `RolloutRequest/Completion/RolloutResult`, `HFRolloutEngine`, `VLLMRolloutEngine` | CPU 테스트는 HF 엔진(tiny model)로 n·seed·logprob 길이·finish_reason. vLLM 클래스는 import 가능성과 인자 구성 함수만 CPU 테스트, 실제 동작은 GPU gate | `test_notebook_rl_engines.py` |
| W3-rollout | §15.11 | 가짜 엔진 시나리오로 트리 모양·토큰 연결·선택·차단·진단 생성 | `test_notebook_rl_rollout.py` |
| W3-checkpoint | §15.13 save/load_resume/load_eval, §15.15 export_merged | 저장→새 프로세스 로드 logits 동일, 재개 후 다음 step loss 동일, export manifest 형식 | `test_notebook_rl_checkpoint.py` |
| W3-metrics | §15.16 `step_record(**fields)`, `health_check(history, protocol, f0_reference) -> list[str]` (§18.6 규칙) | 각 중단 규칙의 경계값 테스트 | `test_notebook_rl_metrics.py` |
| W4-trainer | §15.14 | tiny model + HF 엔진 + 생성기 없는 고정 fixture 문제 4개로 3 step 학습: 학습 파라미터 변화, 고정 파라미터 불변, CE patch 하에서 성공, BUDGET_STOP, resume | `test_notebook_rl_trainer.py` |
| W4-eval-ext | §15.15 | 가짜 엔진으로 지표 산술(pass@k 추정식, Δi→c/Δc→i, 소거 기준선) | `test_notebook_rl_eval_external.py` |
| W4-eval-mmmu | §15.15 | 가짜 백엔드로 900 계획·seed 규칙·과목 집계·약점 540/나머지 360 산술 | `test_notebook_rl_eval_mmmu.py` |
| W5-runpod | §15.18의 `runpod.py` stage와 `cli.py` 명령을 `run_experiment.Controller`로 구현 | 기존 `runpod_setup.sh` 흐름 재사용, 외곽 `fine_tuning.budget` watchdog, 단계당 deadline 120초 여유 | `test_notebook_rl_runpod.py` (dry-run 명령 구성) |
| W5-gates | `gpu_gates.py`: §17.2의 G0~G13 각 함수, JSON 보고 | CPU에서 가능한 부분(G1, G8, G10)은 tiny model로 테스트 | `test_notebook_rl_gates_cpu.py` |

## 17. 검사와 gate

### 17.1 데이터 gate (로컬 CPU, 학습 전)

| Gate | 검사 | 통과 기준 |
|---|---|---|
| D1 결정성 | 각 생성기를 같은 인자로 두 번 실행 | PNG 바이트·정답·슬롯 완전 동일 |
| D2 독립 재계산 | `self_check` 전 문항 | 불일치 0 (있으면 생성기 수정, 문항 삭제로 해결 금지) |
| D3 렌더 정합성 | 생성기별 무작위 50문항 사람 육안 확인 표 + 자동 검사(그림 속 `printed_text`가 슬롯 값과 일치) | 사람 확인 불일치 0, 결과를 `analysis_exports/notebook_rl/render_review_v1.json`에 기록 |
| D4 이미지 의존성 | 템플릿 쌍 테스트(그림만 바꾸면 답이 바뀜) | 100쌍 중 ≥95쌍 |
| D5 split 분리 | `structure_hash` 교차 0, `check_splits` 통과 | 0 |
| D6 비중복 v2 | §13.3 1~7단계(로컬) + 8~9단계·OCR(RunPod) | 후보 family 전부 quarantine, receipt 생성 |
| D7 누설 | actor 파일에 금지 키 0, `leak_review` 사람 확인 | 0 / 확인 기록 |
| D8 금고 격리 | 학습 프로세스에서 `verifier_private.jsonl` open 시도 → 실패 | 실패해야 통과 |

### 17.2 GPU gate (RunPod, 학습 전 smoke; `gpu_gates.py`)

| Gate | 내용 | 통과 기준 |
|---|---|---|
| G0 환경 (이 표의 G0~G13은 GPU gate 번호이며 과거 실험 "G6 full10K"와 무관) | lock 버전 정확 일치, CUDA·BF16 matmul, `VLLM_ENABLE_V1_MULTIPROCESSING=0` 적용(`llm.llm_engine.model_executor` 존재) | 전부 |
| G1 모듈 inventory | vision root 1, text root 1, Merger 4, LoRA L 252(+V 48), `ModulesToSaveWrapper` 4, Merger 사본 FP32·원본 BF16 | 학습 파라미터 33,030,144 + 109,105,152 (+2,359,296) |
| G2 이미지 예산 일치 | 모든 smoke 이미지에서 HF processor `image_grid_thw` == vLLM, 이미지 토큰 수 == 특징 수 | 100% |
| G3 F0 정확성 | (a) 초기 정책 logits == 새 F0 logits (같은 autocast, 비트 단위) (b) K step·K 동기화 후 `disable_adapter()` logits == 새 F0 (c) base·Merger 원본 SHA 불변 (d) 문맥 종료 후 `requires_grad`가 route 규칙대로 | 최대 절대차 0 |
| G4 master 갱신 | 1 step 후 FP32 Merger probe 변화, 고정 probe 불변 (`parameter_probes` 재사용) | 변화/불변 |
| G5 동기화 동일성 | push 후 vLLM 텐서 체크섬 == out-of-place 병합 HF 텐서 (LLM LoRA 대상·vision 대상·Merger 4개), 나머지 불변, HF base 불변 | 100% |
| G6 캐시 무효화 | 같은 이미지·프롬프트를 Merger 변경 전후로: vLLM logprob == 병합 가중치로 새로 띄운 엔진. 기본 캐시 설정에서는 불일치가 재현되는지도 1회 확인 | 의도한 설정에서 일치 |
| G7 logprob 일치 | T=1에서 32개 이상 궤적, step 0과 동기화 후: 토큰당 평균·최대 \|Δ\|, 시퀀스 합 분포. 이 분포로 `δ_seq`를 확정(§10.3) | 임계값을 실행 전에 protocol에 적고 그 안; 합성 0.1 nats/token 불일치 샘플이 masked |
| G8 다회 토큰 연결 | prompt + 시도1 + 피드백 + 시도2 토큰 == 의도한 템플릿; EOS 포함 여부 확인; mask=1은 생성 토큰에만; 엔진이 돌려준 `prompt_token_ids` == 우리가 보낸 것 | 완전 일치 |
| G9 gradient 라우팅 | route off: vision LoRA grad `None`, Merger·LLM grad 유한·0 아님, vision을 하드 동결한 실행과 grad 동일(상대 1e-6). route on: vision grad 0 아님. 혼합 누적에서 vision은 on 샘플 몫만 | 전부 |
| G10 손실 수치 | §15.10 테스트를 GPU에서 다시 + TRL v1.2.0 식과 합성 텐서 대조 | 일치 |
| G11 메모리·처리량 | 최장 샘플(최대 이미지 + 최대 시도)로 1 step 완주, 단계별 시간(rollout tok/s, π_old/π_ref/학습 forward-backward, push), 피크 메모리 | OOM 없음; 수치를 비용 재계산(§19)에 사용 |
| G12 checkpoint 왕복 | 저장 → 새 프로세스 로드 → 정책·reference logits 동일, 재개 후 다음 step 손실 동일 | 동일 |
| G13 데이터 guard | 학습 입력 manifest가 `EXCLUSION_CHECKED_WITH_LIMITS`, 손실 경로에 CE 없음(patch 테스트), 상위 모델 import 없음 | 전부 |

G0~G13 중 하나라도 실패하면 학습을 시작하지 않는다. 실패 기록은 지우지 않고 원인 수정 후 새 commit으로 다시 한다.

### 17.3 학습 중 회귀 검사 (단위 테스트 목록 요약)

- actor 로더가 금지 키·금고 파일 접근을 거부하는가.
- gold를 바꿔도 actor 입력과 rollout seed가 같고 verifier 결과만 달라지는가.
- 오염 후보가 있으면 family 전체가 quarantine되는가, dev/sealed도 검사하는가.
- §9.6 반례가 전부 보상 0인가.
- 트리에서 그룹이 같은 입력을 공유하는 샘플로만 구성되는가, 실패 선택이 결정적인가.
- 피드백·패딩 mask 0, 생성 토큰 mask 1, 이전 시도 토큰이 다음 턴 손실에 다시 들어가지 않는가.
- 손실·gradient가 손계산과 같은가, micro-batch 분할과 무관한가.
- route off에서 vision LoRA가 실제로 변하지 않는가(`set_to_none`), reference가 불변인가.
- 재개 후 같은 다음 batch·같은 손실인가.
- `SYSTEM_ERROR`·`UNVERIFIABLE`·`BUDGET_STOP`이 오답·성공으로 섞이지 않는가.
- MMMU 평가에 피드백·노트·gold oracle이 들어가지 않는가.
- MIS mask가 실제로 작동하는가(합성 불일치 샘플이 masked, 정상 샘플은 유지).

## 18. 실험 계획

### 18.1 단계

| 단계 | 내용 | GPU | 다음 단계 조건 |
|---|---|---|---|
| S0 | W0~W5 구현, CPU 테스트, 데이터 pool 생성, D1~D7 | 없음 | 모든 CPU 테스트·데이터 gate 통과 |
| S1 | RunPod setup, 비중복 index v2·OCR·임베딩 검사(D6 완료), G0~G13 | A40 약 2~4h | gate 전부 통과, G11 실측으로 §19 비용 재계산 후 사용자 승인 |
| S2 | **Phase A** (가중치 고정 F0 측정·선별) | A40 | 선별 동결 |
| S3 | pilot 핵심 비교 (아래 P1 실험군) | A40 | §18.7 gate |
| S4 | 모듈 라우팅 비교 (P2) | A40 | §18.7 gate |
| S5 | 오답노트 확장 비교 (P3) | A40 | 선택 |
| S6 | 확장 데이터 4,000 + seed 3개 확정 비교 (P4) | A40 | 방법·체크포인트 동결 |
| S7 | 최종 MMMU 900 8K (F0 vs 선택 FT 1개), 이후 별도 승인 시 MMMU-Pro std10 | A40 | 보고 |

### 18.2 Phase A — 학습 전에 반드시 재는 것 (가중치 고정)

pilot 후보 pool(학습 후보 576 = 384 × 1.5, dev 96, sealed 192, style_sealed 96)에 대해:

| 측정 | 방법 | 쓰임 |
|---|---|---|
| turn-1 pass rate | 학습 후보마다 T=1, n=8 | 난도 선별 `p ≤ 0.75`, `p=0` 비율 ≤ 50% |
| blind pass rate | 회색 이미지, n=4 | 텍스트로 풀리는 문제 제외 |
| 길이·절단 | turn-1 길이 분포(상한 8192로 측정) | 턴별 `max_new_tokens` 결정(§7.6) |
| 재풀이 성공률 | 무작위 128문제 부분표본에서 실패 2개 × n=4, 정오 피드백 | ALG-NB 신호량 예측 (Critique-GRPO의 "이진 피드백 유효 수정 ~2%" 경고 확인) |
| 귀인 가능률 | obs 모드에서 required slot 파싱률, 원인 분포 | `MOD-RV`가 실제로 Vision을 몇 % 문제에서 여는지 |
| oracle 반사실 | 부분표본, n=4 | 지각 vs 비지각 오류 비율 추정 |
| F0 기준 성적 | dev·sealed·style_sealed 첫 시도(평가 sampling, n=4), dev pass@k(n=8) | 비교 기준 |
| 검증기 이상 | `UNVERIFIABLE`, 렌더 이상 슬롯 | > 1%면 생성기·검증기 수정 후 Phase A 재실행 |

**Phase A 판정 규칙(사전 고정):**
- family별 F0 절단률(선택한 상한 기준) > 10% → 그 family는 난도를 낮춘 설정으로 교체하거나 pilot 제외.
- 전체 재풀이 성공률 < 5% → ALG-NB는 신호가 매우 희소할 것으로 보고, 그래도 pilot은 하되 결과 해석에 명시하고 P3 확장은 보류.
- 귀인 가능 문제 < 20% → `MOD-RV`는 Vision을 거의 열지 않으므로 P2를 축소하고 보고.
- 선별 결과는 `selection_v1.yaml`로 동결하고 이후 바꾸지 않는다.

### 18.3 실험군

모든 학습 실험군은 fresh F0, 같은 선별 데이터·순서·seed·step 수로 시작한다.

| 우선 | 이름 | 알고리즘 | 모듈 | 목적 |
|---|---|---|---|---|
| P1 | `CTL-F0` | 학습 없음 | — | 기준 |
| P1 | `T1-FML` | ALG-T1 (단일턴 RLVR) | MOD-FML | 보상 기반 RL 자체의 효과 |
| P1 | `NB-FML` | **ALG-NB** (오답노트 트리) | MOD-FML | 재풀이 학습의 추가 효과 (vs T1-FML) |
| P1 | `RAND-FML` | CTL-RAND | MOD-FML | Qwen 계열 spurious reward 효과 배제 |
| P2 | `NB-FVML` | ALG-NB | MOD-FVML | Vision 항상 학습 |
| P2 | `NB-RV` | ALG-NB | **MOD-RV** | **사용자 5단계: 검증된 시각 오류에서만 Vision** |
| P2 | `NB-RR` | ALG-NB | MOD-RR | 선택이 아닌 Vision update 양의 효과 배제 |
| P3 | `NBD-best` | ALG-NB+D | P2 최고 모듈 | 오답노트 진단 보상의 추가 효과 |
| P3 | `NBnote-best` | ALG-NB-noteonly | 같음 | 노트 토큰만 강화 (Reflect-Retry-Reward 원형) |
| P3 | `NB3-best` | ALG-NB3 (3회) | 같음 | 사용자 3단계(두 오답을 함께 보고 재풀이) |
| P3 | `NBDV-best` | ALG-NB+D+V | 같음 | 시각 슬롯 보상 λ=0.1 |
| P3 | `BLIND-FML` | CTL-BLIND | MOD-FML | 이미지 없이도 오르는 몫 |
| P3 | `FMT-FML` | CTL-FORMAT | MOD-FML | 형식 보상만의 효과 |
| P3 | `RS-best` | ALG-NB+D | MOD-RS | 모델 자기 진단으로 라우팅(사용자 원안) vs 검증 라우팅 |
| P4 | 최종 후보 + T1-FML + RAND-FML | 확장 데이터, seed 3407/3408/3409 | | 확정 비교 |

**계산량 공정성:** ALG-NB는 같은 문제 수에서 생성·학습 토큰이 T1보다 많다. 그래서 두 가지 비교를 모두 보고한다. (1) **같은 문제·같은 step 수** 비교(기본), (2) T1에 문제를 더 줘서 **실제 생성 토큰 수를 맞춘** 비교(`T1-FML-tokmatched`, P3).

### 18.4 지표

**주 지표: 처음 보는 외부 문제의 첫 시도 정확도** — sealed(192)와 style_sealed(96), 평가 sampling(T 0.7, top_p 0.8, top_k 20, presence 1.5), 문항당 n=4의 평균.

보조 지표:

| 영역 | 지표 |
|---|---|
| 능력 범위 | dev pass@1/4/8 (T=1, n=8, 비편향 추정), 객관식은 따로 |
| 자기수정 | 강제 2차(정오 비공개) Δi→c, Δc→i; 피드백 2차 성공률과 객관식 소거 기준선; 재시도 시 같은 답 반복률 |
| 지각 | obs 프롬프트 required slot 정확도, F0 대비 McNemar |
| 오답노트 | 진단 3분류 정확도(검증 원인 대비), Wrong item 일치율 |
| 형식·길이 | P0 형식 프롬프트에서 NO_FINAL률, 절단률, 출력 길이 중앙값, ≤32토큰 비율, 한 줄 응답 비율 |
| retention | TRACE holdout 300문항(로컬 변환본에서 미사용분), 텍스트만 산술 문제 200(생성기) |
| 학습 동역학 | step별 보상(종류별), degenerate 비율, KL, masked_fraction, 묶음별 grad norm, Vision 허용 비율 |
| 비용 | GPU 시간, 생성 토큰, 학습 토큰 |

### 18.5 통계

- 문항이 아니라 **structure family 단위 paired bootstrap**(2,000회)으로 95% CI. 같은 family의 문항을 독립으로 세지 않는다.
- 지각 변화는 고정 probe에서 paired McNemar.
- seed는 독립 문항이 아니다. seed 간 분산은 따로 보고한다.
- 회복률 분모는 F0 오답 수, 악화율 분모는 F0 정답 수, 분모 0이면 N/A.
- 같은 F0도 GPU 종류에 따라 32 vs 36/60이 나온 전례가 있으므로 **비교는 같은 GPU 종류·같은 엔진 설정**에서만.

### 18.6 학습 중 중단 규칙 (`metrics.health_check`)

| 조건 | 판정 |
|---|---|
| 토큰당 KL 평균 > 0.1 nats가 3 step 연속 | `HEALTH_STOP:KL` |
| 출력 길이 중앙값 < F0의 0.3배, 또는 ≤32토큰 응답 > 20% | `HEALTH_STOP:LENGTH_COLLAPSE` (실험 G6 full10K의 길이 붕괴 재발 방지) |
| 절단률 > 30% (3 step 연속) | `HEALTH_STOP:TRUNCATION` |
| `masked_fraction` > 5% (3 step 연속) | `HEALTH_STOP:MISMATCH` |
| NaN/Inf loss 또는 grad | `HEALTH_STOP:NONFINITE` |
| Vision 실험군에서 grounding probe가 F0보다 2회 연속 낮음 | Vision 학습 중단(나머지 계속), 기록 |
| `UNVERIFIABLE` > 1% | `HEALTH_STOP:VERIFIER` |
| 내부 dev 첫 시도(매 12 step, 96문항 n=1)가 F0보다 5%p 이상 낮음 2회 연속 | `HEALTH_STOP:DEV_DROP` |

중단 시 checkpoint와 원장을 남기고, 덜 학습된 결과를 완료로 표기하지 않는다.

### 18.7 다음 단계로 가는 gate (사전 고정)

다음을 **모두** 만족해야 확장(S6)과 MMMU 평가(S7)로 간다.
1. 후보 FT의 sealed 첫 시도 정확도가 `CTL-F0`보다 높고 family bootstrap 95% CI 하한 > 0.
2. 같은 지표에서 `RAND-FML`보다도 높다(CI 하한 > 0).
3. retention 두 종류가 F0 대비 −2%p보다 나쁘지 않다(점추정; CI 함께 보고).
4. 학습 중 health stop이 없었다.
5. style_sealed에서도 F0보다 낮지 않다.

하나라도 실패하면 `INCONCLUSIVE` 또는 `NEGATIVE`로 보고하고 확장하지 않는다. "재시도하면 맞힌다"만 오른 경우(첫 시도는 그대로)도 개선으로 부르지 않는다.

### 18.8 최종 MMMU 평가

- 대상: `CTL-F0`와 동결된 FT 1개. 둘 다 `eval_mmmu8k.py`(input-recovery-v1, 8,192 출력, P0, V8, 원래 이미지 설정, 문항별 seed 규칙)로 **같은 GPU 종류에서 연속 실행**.
- 보고: 900 전체, 30과목, 약점 18과목 540 / 나머지 12과목 360, 상한 종료, NO_PARSE, F0 대비 paired +/−.
- 기존 32K 558/900(62.00%)과 직접 비교하지 않는다(조건이 다름). 새 8K F0가 기준이다.
- 학습 중 MMMU를 보지 않았고, MMMU 결과로 체크포인트를 고르지 않았음을 명시한다.
- MMMU-Pro std10(1,730)은 별도 예산 승인 뒤 같은 방식.

## 19. 하드웨어·비용·RunPod·MMDL_test 재현 절차

### 19.1 하드웨어 선택

| 장비 | 판정 | 근거 |
|---|---|---|
| 로컬 RTX 5060 8GB | 학습·rollout 불가. CPU 테스트·데이터 생성·오염 검사(1~7단계)만 | BF16 base만 8.27GiB |
| RTX 3090 24GB | v2 기본 경로 제외 | 과거 FML SFT만으로 22.61GiB 예약. rollout 엔진 동시 적재 불가 |
| **A40 48GB** | **기본** | HF 학습 피크 추정 ~16GiB + vLLM(가중치 8.3 + KV ~10 + 여유) ~20GiB. 과거 실측 $0.59/h (secure $0.35/h 관측 이력 있음) |

### 19.2 비용 계산식과 현재 추정 (G11 실측 전까지 모두 추정)

```text
문제 1개 시간 ≈ (생성 토큰 수 / R_gen)  +  Σ_학습샘플 (c_train × 시퀀스 길이)  +  π_old·π_ref forward (≈ 학습의 2/3.2)
R_gen   : vLLM 동일 프로세스 rollout 처리량 (tok/s)      — 가정 400 (과거 eager·동시 4 실측 ~124; CUDA graph·동시성 증가 효과 미측정)
c_train : 학습 forward+backward 토큰당 시간               — 가정 0.56 ms/token (8K 시퀀스 4.5 s 추정에서)
평균 완성 길이                                             — 가정 1,500 tokens (턴당 상한 3,072 가정)
```

| 항목 | 가정 | 추정 GPU 시간 | A40 $0.59/h |
|---|---|---:|---:|
| S1 setup + index v2 + OCR(~12K장) + G0~G13 | | 3~5 h | $1.8~3.0 |
| S2 Phase A (576+384 문제, 부분표본 재풀이·oracle) | 문제당 생성 ~12회 | 7~9 h | $4~5 |
| P1 `T1-FML` (384문제, 48 step) | 문제당 8 생성 + 8 학습 샘플 | ~6 h | ~$3.5 |
| P1 `NB-FML` | 문제당 ~16 생성 + 16 학습 샘플 | ~12 h | ~$7 |
| P1 `RAND-FML` | T1과 같음 | ~6 h | ~$3.5 |
| P1 평가 (4 모델 × sealed·style·dev) | 모델당 ~1.5 h | ~6 h | ~$3.5 |
| **P1까지 합계** | | **~40 h** | **~$24** |
| P2 3개 실험군 + 평가 | | ~40 h | ~$24 |
| S7 MMMU 900 8K × 2 모델 | 과거 8K 336호출 943K 토큰 실측 기반 | 5~11 h | $3~6.5 |

**주의:** 위 값은 처리량 가정이 2배 틀리면 그대로 2배 틀린다. 과거 예산 기록(최초 잔액 약 $5~10 수준)과 비교하면 P1만으로도 큰 금액이다. 그래서 다음을 지킨다.
1. **S1에서 G11로 처리량·step 시간을 실측**한 뒤 이 표를 다시 계산해 사용자에게 승인받는다.
2. 기존 저장소 관행대로 **실측 필요 시간의 2배 여유가 남을 때만** 다음 단계를 시작한다.
3. 비용을 줄이는 사전 선택지(사용자 결정, §21): (a) pilot 학습 문제를 192로 축소(시간 절반), (b) P1에서 `RAND-FML`을 P4로 미룸(단, 개선 주장은 P4 전까지 보류), (c) Phase A 재풀이 부분표본 64로 축소.

### 19.3 MMDL_test 게시

```bash
cd /mnt/c/projects/multimodel
# 1) CPU 검사 (학습 venv + 데이터 venv) 전부 통과 확인
# 2) 게시 대상 확인 (fine_tuning/notebook_rl/** 의 .py/.yaml/.md/.lock, fine_tuning/tests/test_notebook_rl_*.py 가 자동 포함)
python -m scripts.sync_test_repo --sync
python -m scripts.sync_test_repo --check
cd MMDL_test
git status && git diff --stat            # 의도한 파일만인지 확인
git switch -c feat/notebook-rl-v1        # 기능 브랜치 (이름은 §21에서 확정)
git add -A && git commit -m "notebook_rl: <wave 설명>"
git push -u origin feat/notebook-rl-v1   # force push 금지
git rev-parse HEAD                        # 이 40자 SHA를 RunPod에서 사용
```

- 데이터·이미지·정답·rollout·checkpoint는 게시하지 않는다(`sync_test_repo`가 `fine_tuning/runs`, `fine_tuning/generated`를 제외하고, 데이터는 저장소 밖에 있음).
- 이 설계서(`claudedocs/*.md`)는 자동 게시되지만 도식 HTML은 게시 대상 확장자가 아니어서 MMDL_test에는 올라가지 않는다(로컬 전용). MMDL_test에서는 설계서의 HTML 링크가 열리지 않는다는 점을 알고 쓴다.
- 게시 전 `git diff`에 `MMDL/`·`src/`·`configs/`·`scripts/` 변경이 없어야 한다.

### 19.4 RunPod 실행 순서

1. **승인 확인:** 실제 잔액, 시간당 가격, 최대 시간, 회수 예약($1), 최초 과금 시각을 기록. 소유 Pod 하나만 만든다.
2. **clean clone:** `git clone git@github.com:jang2296/MMDL_test.git` → `git checkout --detach <SHA>` → `set -a; source /etc/rp_environment`.
3. **setup:** `bash fine_tuning/runpod_setup.sh --commit <SHA> --environment vllm --runtime-root /workspace/nbrl-<날짜> --owned-pod-id $RUNPOD_POD_ID --gpu-cutoff-utc … --billing-start-utc … --balance-usd … --reserve-usd 1 --hourly-usd … --execute`. 이 스크립트가 lock 설치·`pip check`·**`fine_tuning/tests` 전체 unittest(우리 CPU 테스트 포함)**·doctor·모델 해시 확인을 한다. 우리 테스트는 GPU·생성기 의존성 없이 빨라야 한다.
4. **비공개 데이터 전송:** 로컬 `scp`로 데이터 디렉터리 전송 → Pod에서 manifest SHA 전수 대조. 결과를 `transfer_receipt.json`으로 회수.
5. **gate:** `python -m fine_tuning.notebook_rl.runpod --stage gates …` (G0~G13 + index v2 + D6 임베딩·OCR). 보고 회수 → 비용 재계산 → 사용자 승인.
6. **Phase A → 선별 동결 → 실험군 학습 → export → 외부 평가**: `--stage phase_a`, `--stage train --arm … --algorithm …`, `--stage export`, `--stage eval_external`. 각 단계는 `run_experiment.Controller`가 새 세션에서 실행하고, 600초마다 `HEALTH_POLL`(nvidia-smi)을 기록한다. 외곽에는 `fine_tuning.budget` watchdog을 둔다(내부 단계 deadline은 외곽보다 120초 이르게).
7. **10분 점검:** Pod 상태·프로세스·로그 말미·비용을 10분마다 확인한다(사용자 운영 규칙).
8. **회수와 삭제:** 닫힌 결과를 tar로 묶어 SHA 기록 → 로컬로 회수 → SHA·행 수·checkpoint 파일 해시·LoRA/Merger 텐서 로드 검사 → `recovery_receipt.json` → **정확한 소유 Pod만 삭제** → 조회 404·목록 부재 확인. Stop만으로 과금이 끝났다고 하지 않는다.
9. 실패·중단 기록도 회수해 보존한다. 같은 원인의 재시도는 최대 3회, 그 뒤에는 사용자에게 보고한다.

### 19.5 재현성 기록 (실행마다)

`code_commit`, lock SHA, `protocol_v1.yaml`·`prompts_v1.yaml`·`source_registry_v1.yaml` SHA, 데이터 manifest SHA, 선별 동결 SHA, base 모델 파일 해시, GPU 모델·드라이버, seed, 모든 rollout 원장(append-only), step 지표, checkpoint 텐서 SHA, 평가 원문·채점 결과. 같은 SHA·같은 데이터·같은 seed로 재실행했을 때 **절차는 재현**되지만, GPU·커널 차이로 **응답이 비트 단위로 같다고 보장하지는 않는다**(과거 A40/3090 F0 32 vs 36/60).

## 20. 위험과 대응

| 위험 | 신호 | 대응 |
|---|---|---|
| 재풀이 성공이 너무 드묾 (이진 피드백 한계) | Phase A 재풀이 성공률 < 5% | ALG-NB 해석 제한, 열린 답 비중↑, 난도 하향, 그래도 없으면 T1만 진행하고 보고 |
| 첫 시도 고의 오답 (보상 해킹) | turn-1 정답률↓, turn-2↑ | 구조상 유인 없음(턴 분리). 그래도 관측되면 즉시 중단·원인 분석 |
| 재시도에서 같은 답 반복 (UFO 70%) | 반복률 지표 | `Error check` 프롬프트 유지, NB3 비교, 반복 자체엔 보상 없음 |
| 객관식 소거로 재풀이 정확도 부풀림 | 재풀이 정확도 ≈ 소거 기준선 | 열린 답 ≥60%, 보기 10개, 소거 기준선 보고 |
| 길이 붕괴 (실험 G6 full10K 재발) | 길이 중앙값·≤32토큰 비율 | health stop, KL β, plain 모드 혼합 |
| 장문 반복 증가 | 절단률↑ | 절단=0 보상, overlong 비교군 |
| Qwen spurious reward 효과로 착시 | `RAND-FML`도 비슷하게 오름 | 개선 주장은 `RAND` 대비로만 |
| 텍스트 지름길 학습 | blind 정확도도 오름 | Phase A blind 제외, `BLIND-FML` 대조, 평가 시 이미지 제거 검사 |
| 생성기 스타일 암기 | sealed는 오르고 style_sealed는 그대로 | style_sealed를 gate에 포함 |
| 생성기·렌더 버그로 잘못된 정답 | 특정 슬롯 오독률 쏠림, 사람 확인 불일치 | D2·D3 gate, 독립 재계산, 발견 시 그 생성기 전체 재생성 |
| vLLM–HF 불일치 / 가중치 동기화 누락 | masked_fraction↑, probe 불일치 | G5~G7, 매 step probe, 초과 시 중단 |
| encoder 캐시가 옛 Merger 출력 재사용 | probe 불일치(gate G6) | prefix cache off, mm cache 0 |
| PEFT in-place merge로 base 오염 | base SHA 변화 | merge 금지, 매 checkpoint마다 `assert_base_unchanged` |
| Vision 학습이 grounding 손상 | grounding probe < F0 | Vision 학습 자동 중단 규칙 |
| BF16 LoRA 장기 학습 붕괴 보고 (Qi et al.) | 수백 step 이후 보상·KL 급변 | pilot은 48 step으로 짧음. 확장 시 감시 |
| 오염 검사 오탐으로 단순 레이아웃 편향 | dHash 저정보 해시 다수 | 저정보 해시는 검토 전용, 보정 세트로 임계 결정 |
| 사전학습 노출(검사 불가) | — | 한계로 명시, MMMU 결과 해석 제한 |
| 예산 초과 | G11 실측 | 2배 여유 gate, 축소 선택지 |
| 교수님이 RLVR 자체를 불허 | — | 실행 전 확인(§21) |

## 21. 사용자 결정이 필요한 항목

| # | 결정 | 기본안 | 영향 |
|---|---|---|---|
| Q1 | 교수님 규칙이 "SFT 금지"인지 "정답 기반 보상 학습까지 금지"인지 | SFT 금지로 해석 | 후자면 이 설계 전체 불가 |
| Q2 | 예산과 GPU: P1까지 약 40 GPU-h(추정 ~$24, A40 $0.59/h), P2까지 약 80h | S1(gate·실측, ~$2~3)만 먼저 승인 후 재계산 | 축소안 §19.2 |
| Q3 | OCR 엔진 | 고정 Qwen3-VL-4B를 탐지 도구로 사용 | 새 의존성 없음, RunPod 시간 추가 |
| Q4 | MMDL_test 브랜치 이름 | `feat/notebook-rl-v1` | 기존 `feat/workspace-repro`와 분리 |
| Q5 | NC 라이선스 데이터(TabMWP, DVQA, ScienceQA) 사용 | 비공개 연구로만 사용, 데이터 미배포 | 공개 산출물 범위 |
| Q6 | 재풀이 학습에 객관식 포함 여부 | 포함하되 보기 10개, 40% 이하 | 소거 효과 |
| Q7 | 최종 평가 범위 | MMMU val 900 8K (F0 vs FT 1개), MMMU-Pro std10은 이후 별도 승인 | 비용 |
| Q8 | pilot 규모 | 384 학습 문제 | 192로 줄이면 비용 절반·검정력 감소 |

## 22. 참고문헌 (2026-10-16 확인)

상태 표기: **PR** 주 학회/저널 확인, **PR?** arXiv 코멘트에만 게재 표시, **WS** 워크숍, **PP** preprint. 최종 보고서의 BibTeX는 arXiv/학회 메타데이터에서 가져오고, 인용한 절과 버전을 함께 적는다.

**자기수정·재시도 RL**
1. Kumar et al. Training Language Models to Self-Correct via Reinforcement Learning. ICLR 2025 (Oral). [arXiv:2409.12917v2](https://arxiv.org/abs/2409.12917v2) — PR
2. Bensal et al. Reflect, Retry, Reward: Self-Improving LLMs via Reinforcement Learning. 2025. [arXiv:2505.24726v1](https://arxiv.org/abs/2505.24726v1) — PP
3. Liu et al. A Simple "Try Again" Can Elicit Multi-Turn LLM Reasoning. 2025. [arXiv:2507.14295v2](https://arxiv.org/abs/2507.14295v2) — PP (구판 ICML'25 AI4Math WS)
4. Chung, Du, Fu. Learning from Failures in Multi-Attempt Reinforcement Learning. 2025. [arXiv:2503.04808v1](https://arxiv.org/abs/2503.04808v1) — PP
5. Ma et al. S²R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning. ACL 2025. [arXiv:2502.12853v1](https://arxiv.org/abs/2502.12853v1) — PR
6. Xiong et al. Self-rewarding Correction for Mathematical Reasoning. 2025. [arXiv:2502.19613v1](https://arxiv.org/abs/2502.19613v1) — PP
7. Tang et al. Do Not Step Into the Same River Twice: Learning to Reason from Trial and Error (LTE). [arXiv:2510.26109v4](https://arxiv.org/abs/2510.26109v4) — PR? (ACL 2026)
8. Zhang et al. Critique-GRPO. [arXiv:2506.03106v7](https://arxiv.org/abs/2506.03106v7) — PR? (ICML 2026)
9. Yan et al. Learning to Reason under Off-Policy Guidance (LUFFY). NeurIPS 2025. [arXiv:2504.14945v5](https://arxiv.org/abs/2504.14945v5) — PR
10. Zhu et al. The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning. NeurIPS 2025. [arXiv:2506.01347v2](https://arxiv.org/abs/2506.01347v2) — PR
11. Jiang et al. Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier (PAG). [arXiv:2506.10406v1](https://arxiv.org/abs/2506.10406v1) — WS
12. Qu et al. Recursive Introspection (RISE). NeurIPS 2024. [arXiv:2407.18219v2](https://arxiv.org/abs/2407.18219v2) — PR

**자기수정의 한계·개념적 선행**
13. Huang et al. Large Language Models Cannot Self-Correct Reasoning Yet. ICLR 2024. [arXiv:2310.01798v2](https://arxiv.org/abs/2310.01798v2) — PR
14. Kamoi et al. When Can LLMs Actually Correct Their Own Mistakes? TACL 2024. [arXiv:2406.01297v3](https://arxiv.org/abs/2406.01297v3) — PR
15. Shinn et al. Reflexion. NeurIPS 2023. [arXiv:2303.11366v4](https://arxiv.org/abs/2303.11366v4) — PR
16. Madaan et al. Self-Refine. NeurIPS 2023. [arXiv:2303.17651v2](https://arxiv.org/abs/2303.17651v2) — PR
17. M2Note: Mistake Notebook Learning. 2026. [arXiv:2607.00685v2](https://arxiv.org/abs/2607.00685v2) — PP

**VLM 반성·자기수정**
18. Wang et al. VL-Rethinker. NeurIPS 2025. [arXiv:2504.08837v3](https://arxiv.org/abs/2504.08837v3) — PR
19. Wan et al. SRPO: Reflection-Aware RL for MLLMs. [arXiv:2506.01713v3](https://arxiv.org/abs/2506.01713v3) — PR? (NeurIPS 2025)
20. Tang et al. MM-ReCoder. CVPR 2026. [arXiv:2604.01600v1](https://arxiv.org/abs/2604.01600v1) — PR
21. Ding & Zhang. Sherlock: Self-Correcting Reasoning in VLMs. NeurIPS 2025. [arXiv:2505.22651v2](https://arxiv.org/abs/2505.22651v2) — PR
22. Jian et al. Look Again, Think Slowly (Reflection-V). EMNLP 2025. [arXiv:2509.12132v1](https://arxiv.org/abs/2509.12132v1) — PR
23. Wu et al. SVR-R1. 2026. [arXiv:2607.10966v1](https://arxiv.org/abs/2607.10966v1) — PP
24. Li, Sun, Zhao. StructReward. 2026. [arXiv:2608.08326v2](https://arxiv.org/abs/2608.08326v2) — PP

**RLVR 알고리즘·대조군**
25. Shao et al. DeepSeekMath (GRPO). 2024. [arXiv:2402.03300v3](https://arxiv.org/abs/2402.03300v3) — PP
26. DeepSeek-AI. DeepSeek-R1. Nature 645, 633–638 (2025). [arXiv:2501.12948v2](https://arxiv.org/abs/2501.12948v2) — PR
27. Liu et al. Understanding R1-Zero-Like Training (Dr. GRPO). [arXiv:2503.20783v2](https://arxiv.org/abs/2503.20783v2) — PP
28. Yu et al. DAPO. NeurIPS 2025. [arXiv:2503.14476v2](https://arxiv.org/abs/2503.14476v2) — PR
29. Ahmadian et al. Back to Basics (RLOO). ACL 2024. [arXiv:2402.14740v2](https://arxiv.org/abs/2402.14740v2) — PR
30. Zheng et al. GSPO. [arXiv:2507.18071v2](https://arxiv.org/abs/2507.18071v2) — PP
31. Gao et al. SAPO. [arXiv:2511.20347v2](https://arxiv.org/abs/2511.20347v2) — PP
32. Shao et al. Spurious Rewards: Rethinking Training Signals in RLVR. ICML 2026. [arXiv:2506.10947v2](https://arxiv.org/abs/2506.10947v2) — PR
33. Yue et al. Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? NeurIPS 2025 (Oral). [arXiv:2504.13837v5](https://arxiv.org/abs/2504.13837v5) — PR
34. Yao et al. On the Rollout-Training Mismatch in Modern RL Systems. NeurIPS 2025 OPT workshop; [blog](https://fengyao.notion.site/off-policy-rl) — WS
35. Qi et al. Defeating the Training-Inference Mismatch via FP16. [arXiv:2510.26788v1](https://arxiv.org/abs/2510.26788v1) — PP

**VLM RL·지각**
36. Chen et al. SFT or RL? An Early Investigation into Training R1-Like Reasoning LVLMs (VLAA-Thinker). [arXiv:2504.11468v1](https://arxiv.org/abs/2504.11468v1) — PR? (TMLR 2025)
37. Zhou et al. VisualThinker-R1-Zero. [arXiv:2503.05132v2](https://arxiv.org/abs/2503.05132v2) — PP
38. Liu et al. Visual-RFT. ICCV 2025. [arXiv:2503.01785v1](https://arxiv.org/abs/2503.01785v1) — PR
39. Meng et al. MM-Eureka. [arXiv:2503.07365v2](https://arxiv.org/abs/2503.07365v2) — PP
40. Peng et al. LMM-R1. [arXiv:2503.07536v2](https://arxiv.org/abs/2503.07536v2) — PP
41. Wang et al. ThinkLite-VL. NeurIPS 2025. [arXiv:2504.07934v3](https://arxiv.org/abs/2504.07934v3) — PR
42. Liu et al. NoisyRollout. [arXiv:2504.13055v4](https://arxiv.org/abs/2504.13055v4) — PR? (NeurIPS 2025)
43. Zheng et al. DeepEyes. [arXiv:2505.14362v3](https://arxiv.org/abs/2505.14362v3) — PR? (ICLR 2026)
44. Xiao et al. Perception-R1 (visual perception reward). ICLR 2026. [arXiv:2506.07218v3](https://arxiv.org/abs/2506.07218v3) — PR
45. Li et al. Self-Rewarding VLM via Reasoning Decomposition (Vision-SR1). ICLR 2026. [arXiv:2508.19652](https://arxiv.org/abs/2508.19652) — PR (v2)
46. Wang et al. ViCrit. NeurIPS 2025. [arXiv:2506.10128v1](https://arxiv.org/abs/2506.10128v1) — PR
47. He et al. VisPlay. CVPR 2026. [arXiv:2511.15661v2](https://arxiv.org/abs/2511.15661v2) — PR
48. Jiang et al. PROPA. [arXiv:2511.10279v1](https://arxiv.org/abs/2511.10279v1) — PP
49. Wu, Lin, Chang, Peng. On Asymmetric Optimization of Reasoning and Perception in VLM Post-Training. [arXiv:2605.29496v2](https://arxiv.org/abs/2605.29496v2) — PP
50. Wang et al. PAPO. [arXiv:2507.06448](https://arxiv.org/abs/2507.06448) — PR? (ICLR 2026)
51. Huang et al. VPPO: Spotlight on Token Perception for Multimodal RL. ICLR 2026. [arXiv:2510.09285v2](https://arxiv.org/abs/2510.09285v2) — PR
52. Bai, Dhanda, Jain. Chart-RL. [arXiv:2604.03157v1](https://arxiv.org/abs/2604.03157v1) — PR? (KDD 2026)
53. Ma et al. Rethinking RL Scaling for VLMs (MAYE). [arXiv:2504.02587v1](https://arxiv.org/abs/2504.02587v1) — PP
54. Wu et al. From Seeing to Thinking. ICML 2026. [arXiv:2605.20177v1](https://arxiv.org/abs/2605.20177v1) — PR?
55. Ye et al. Same Reward, Different Skills. [arXiv:2610.01908v1](https://arxiv.org/abs/2610.01908v1) — PP
56. Qwen Team. Qwen3-VL Technical Report. [arXiv:2511.21631v2](https://arxiv.org/abs/2511.21631v2) — PP; [고정 모델 카드](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md)

**객관식 누설·오염 검사**
57. Liu et al. ReVeL: Beyond Multiple Choice. [arXiv:2511.17405v2](https://arxiv.org/abs/2511.17405v2) — PP
58. Guo et al. Rethinking Multiple-Choice Questions for RLVR. [arXiv:2603.12826](https://arxiv.org/abs/2603.12826) — PR? (Findings ACL 2026)
59. Song et al. Both Text and Images Leaked! (MM-Detect). [arXiv:2411.03823v3](https://arxiv.org/abs/2411.03823v3) — PR (EMNLP 2025 Findings)
60. Pizzi et al. A Self-Supervised Descriptor for Image Copy Detection (SSCD). [arXiv:2202.10261v2](https://arxiv.org/abs/2202.10261v2); [코드](https://github.com/facebookresearch/sscd-copy-detection) (MIT)
61. LMMs-Eval contamination analysis. [arXiv:2407.12772](https://arxiv.org/html/2407.12772); FineVision decontamination [arXiv:2510.17269v2](https://arxiv.org/html/2510.17269v2); Keye-VL [arXiv:2507.01949](https://arxiv.org/html/2507.01949)

**벤치마크·데이터 출처**
62. Yue et al. MMMU. [arXiv:2311.16502v4](https://arxiv.org/abs/2311.16502v4); MMMU-Pro [arXiv:2409.02813v3](https://arxiv.org/abs/2409.02813v3)
63. TabMWP [arXiv:2209.14610v3](https://arxiv.org/abs/2209.14610v3); Geometry3K/InterGPS [arXiv:2105.04165](https://arxiv.org/abs/2105.04165); PGPS9K [arXiv:2302.11097](https://arxiv.org/abs/2302.11097); PlotQA [arXiv:1909.00997](https://arxiv.org/abs/1909.00997); DVQA [arXiv:1801.08163](https://arxiv.org/abs/1801.08163); MapQA [arXiv:2211.08545](https://arxiv.org/abs/2211.08545); FinQA [arXiv:2109.00122](https://arxiv.org/abs/2109.00122); TAT-QA [arXiv:2105.07624](https://arxiv.org/abs/2105.07624); ChartQA [arXiv:2203.10244](https://arxiv.org/abs/2203.10244); ScienceQA [arXiv:2209.09513](https://arxiv.org/abs/2209.09513); AI2D [arXiv:1603.07396](https://arxiv.org/abs/1603.07396); CLEVR [저자 페이지](https://cs.stanford.edu/people/jcjohns/clevr/); TRACE [arXiv:2607.19790v1](https://arxiv.org/abs/2607.19790v1); ChEBI [라이선스](https://www.ebi.ac.uk/chebi/aboutChebiForward.do)

**구현 소스 (고정 버전)**
64. transformers v4.57.1 `modeling_qwen3_vl.py`; PEFT v0.17.1 (`peft_model.py`, `utils/other.py`, `tuners/lora/*`); vLLM v0.11.0; TRL v1.2.0 / v1.16.0 (`trl/generation/vllm_generation.py`, 참고만); vLLM issue #29341.

## 23. 이번 작업의 완료 범위와 다음 행동

- **완료:** 저장소 코드·과거 실험 원장·데이터 인프라 확인(읽기 전용), 관련 논문 64건 이상의 원문·게재 상태 확인, 후보 데이터셋의 라이선스·라벨 출처 확인, RL 혼합셋 3종(ViRL39K, MM-Eureka, MMK12)의 MMMU 텍스트 중복 실측, PEFT·vLLM·transformers 고정 버전 소스 확인, 이 설계서와 [파이프라인 도식 HTML](error_notebook_rl_pipeline_20261016.html) 작성.
- **수행하지 않음:** 코드 구현, 데이터 생성, GPU 사용, RunPod 대여, GitHub 게시, `MMDL/` 수정, 기존 V8 결과·파서 수정.
- **아직 모름:** 이 방법의 효과, 실제 처리량·비용, 재풀이 성공률, 귀인 가능률, 새 후보 데이터의 비중복 최종 판정.
- **다음 행동:** (1) §21 Q1·Q2 사용자 결정 → (2) Wave 0 구현(§16.4, `finalAudit: true`) → (3) Wave 1~5 → (4) S1 gate와 처리량 실측 → (5) 비용 재계산·승인 → (6) Phase A → (7) P1 비교와 §18.7 판정.

이 문서는 구현 계약서다. 구현과 다른 결정이 필요하면 이 문서를 고치는 대신 `protocol_v2`와 변경 사유를 남긴다.

## 부록 A. 프롬프트 전문 (구현 시 `fine_tuning/notebook_rl/prompts_v1.yaml`에 키별로 그대로 저장)

`.txt`는 `fine_tuning` 아래에서 MMDL_test로 게시되지 않으므로 YAML 한 파일에 모은다(키: `turn1_plain`, `turn1_obs`, `retry_incorrect`, `retry_invalid`, `diagnose`, `oracle_facts`, `self_perception_probe`, `observation_instruction.<family>`). UTF-8, 각 값 끝 줄바꿈 1개, `{…}` 자리표시자는 `str.format`이 아니라 **명시적 `replace`**로 채운다(질문 본문에 `{`가 있어도 안전하도록). 파일 SHA256을 `protocol_v1.yaml`과 매 실행 원장에 기록한다. 아래는 팀 설계 초안이며 Qwen의 공식 프롬프트가 아니다.

### A.1 `turn1_plain` (P0 호환)

```text
{P0_BODY}
End your response with exactly one line: Final answer: <answer>
```

`{P0_BODY}`는 기존 `prompts/mmmu_mcq_v1.txt`/`mmmu_open_v1.txt`와 **같은 직렬화 함수**로 만든다(질문, `Options:` + `A. ...` 줄, 객관식이면 `Please select the correct answer from the options above.`). 이미지는 원래 순서로 user content 앞쪽에 둔다. 직렬화 함수는 `src/mmdl/evaluation/`의 기존 구현을 import하지 말고(평가 코드 불변 원칙) 같은 규칙을 `prompts.py`에 재구현한 뒤, 기존 함수와 출력이 같은지 테스트로 대조한다.

### A.2 `turn1_obs`

```text
{P0_BODY}

Answer in this format:
Observed facts:
{OBSERVATION_INSTRUCTION}
Solution:
<your reasoning>
Final answer: <answer>
```

`{OBSERVATION_INSTRUCTION}`은 family별 고정 문장이다(예시):

| family | `OBSERVATION_INSTRUCTION` |
|---|---|
| chart_bar | `List every bar as one line: bar[<label>] = <value>` |
| chart_line | `List every marked point as one line: point[<series>][<x>] = <y>` |
| table | `List each table cell you use as one line: cell[<row label>][<column label>] = <value>` |
| circuit | `List each component as one line: comp[<name>] = <value with unit>, and each connection as: wire[<name>] = <node>-<node>` |
| graph | `List each edge as one line: edge[<u>,<v>] = <weight or 1>` |
| music | `List each note in order as one line: note[<measure>.<index>] = <pitch with octave, e.g. C#4>` |
| molecule | `List counts as lines: atoms[<element>] = <count>; rings = <count>` |
| geometry | `List each labeled length or angle as one line: len[<segment>] = <value>  or  angle[<name>] = <degrees>` |

질문과 상관없이 family마다 같은 문장이며, 어떤 슬롯이 이 질문에 필요한지는 알려주지 않는다.

### A.3 `retry_incorrect`

```text
Verifier result for your previous answer "{PREV_ANSWER}": INCORRECT.
The correct answer is not provided.
Look at the image again. Do not reuse your earlier observations or steps without checking them.
Write your response in this format:
Error check:
- <what in your previous solution may be wrong, and what you will re-check>
{OBS_SECTION}Solution:
<your new reasoning>
Final answer: <answer>
```

`{OBS_SECTION}`은 obs 모드에서 `Observed facts:\n{OBSERVATION_INSTRUCTION}\n`, plain 모드에서 빈 문자열.

### A.4 `retry_invalid`

```text
Your previous response was not graded because {REASON}.
Its correctness is unknown.
Write your response in this format:
Error check:
- <what went wrong with the previous response>
{OBS_SECTION}Solution:
<your reasoning>
Final answer: <answer>
```

`{REASON}` 매핑: `INVALID_NO_FINAL` → `it did not end with a line "Final answer: <answer>"`, `INVALID_MULTIPLE` → `it gave more than one final answer`, `TRUNCATED` → `it was cut off before a final answer`.

### A.5 `diagnose`

```text
Your attempt {K} above was verified CORRECT. Attempts 1 to {K_MINUS_1} were INCORRECT or invalid.
Write a short error note about attempt 1.
Choose exactly one cause:
VISUAL_MISREAD - a value, label or symbol read from the image was wrong
VISUAL_RELATION - a connection, position or correspondence in the image was wrong
NON_VISUAL - the image was read correctly but a rule, formula, logic or calculation step was wrong
FINALIZATION - no single final answer was given or the response was cut off
Format:
Cause: <one label>
Wrong item: <the first wrong item, e.g. bar[Mar], or none>
Lesson: <one sentence>
```

### A.6 `oracle_facts` (진단 전용, loss 없음)

```text
Verified image facts (from the image generator):
{FACT_LINES}

{P0_BODY}
End your response with exactly one line: Final answer: <answer>
```

이 프롬프트로 만든 생성은 `oracle_counterfactual` 계산에만 쓰고 rollout 학습 버퍼에 들어가지 않는다는 것을 타입으로 강제한다(`OracleProbe` 레코드는 `Group`으로 변환할 수 없음).

### A.7 `self_perception_probe` (진단 전용, 이미지 없이, loss 없음)

```text
You cannot see the image. Below are notes that were written while looking at it.
Notes:
{OBSERVED_TEXT}

Using only these notes, answer the question.
{P0_BODY}
End your response with exactly one line: Final answer: <answer>
```

## 부록 B. `protocol_v1.yaml` 초안 (W0-1이 이 값으로 작성, 변경 시 protocol revision)

```yaml
protocol_id: notebook-rl-v1
status: DESIGNED            # 구현·gate 통과 후 FROZEN
initial_checkpoint:
  model_id: Qwen/Qwen3-VL-4B-Instruct
  revision: ebb281ec70b05090aa6165b016eac8ec08e71b17
  manifest: manifests/models/baseline.json
forbidden:
  additional_sft: true
  teacher_or_upper_model_labels: true
  benchmark_training: true            # MMMU dev/val/test, MMMU-Pro std10/std4/vision, 파생물
environment:
  lock: env/requirements-vllm.lock     # torch 2.8.0+cu128, transformers 4.57.1, peft 0.17.1, vllm 0.11.0
  data_lock: fine_tuning/notebook_rl/requirements-data.lock
image:                                  # 평가와 동일
  preprocessing: qwen_vl_utils_0_0_14
  min_pixels: 1003520
  max_pixels: 4014080
  max_images_per_problem: 2          # 재풀이 학습 문맥 한도 때문 (§7.3). pilot 출처는 모두 1장
context_limit: 16384
rollout:
  sampler: {temperature: 1.0, top_p: 1.0, top_k: 0, presence_penalty: 0.0, repetition_penalty: 1.0}
  n: {1: 8, 2: 4, 3: 4}
  branch: {1: 2, 2: 1}
  max_attempts: 2                       # ALG-NB3에서 3
  max_new_tokens: {1: 3072, 2: 3072, 3: 3072}   # Phase A 후 확정 (§7.6)
  diag: {enabled_in: [ALG-NB+D, ALG-NB+D+V], n: 4, max_new_tokens: 256, class_cap: 0.5}
  oracle_counterfactual: false
  self_perception_probe: false
  mode_assignment: sha256_parity        # obs / plain 50:50
reward:
  answer_extractor: strict-final-v1
  visual_lambda: 0.1                    # ALG-NB+D+V에서만
  diag: {cause_match: 1.0, slot_bonus: 0.25}
  overlong: {enabled: false, buffer: 512, weight: 0.5}
loss:
  advantage: group_mean_no_std
  clip: {eps_low: 0.2, eps_high: 0.28}
  kl: {estimator: k3, beta: 0.01, reference: peft_disable_adapter}
  mismatch: {level: sequence_mean_log_ratio, delta_seq: 0.05, stop_if_masked_fraction_gt: 0.05, consecutive_steps: 3}   # delta_seq는 G7 실측으로 확정
  normalization: constant_n_times_cap   # Dr. GRPO
  group_weights: {T1: 1.0, T2: 1.0, T3: 1.0, DIAG: 0.5}
  updates_per_batch: 1
optimizer:
  type: adamw
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.0
  lr: {L: 1.0e-5, M: 1.0e-6, V: 1.0e-6}
  warmup_steps: 10
  schedule: constant_after_warmup
  grad_clip_per_group: 1.0
model:
  lora: {r: 16, alpha: 32, dropout: 0.0, bias: none}
  llm_targets: inspect_modules.targets.L     # 252 Linear
  vision_targets: inspect_modules.targets.V  # 48 Linear (attn.qkv, attn.proj)
  modules_to_save:
    - model.visual.merger
    - model.visual.deepstack_merger_list.0
    - model.visual.deepstack_merger_list.1
    - model.visual.deepstack_merger_list.2
  merger_master_dtype: float32
  gradient_checkpointing: non_reentrant
  logprob_chunk: 1024
expected_trainable: {L: 33030144, M: 109105152, V: 2359296}
arms: [MOD-FML, MOD-FVML, MOD-RV, MOD-RR, MOD-RS]
algorithms: [ALG-T1, ALG-NB, ALG-NB3, ALG-NB+D, ALG-NB+D+V, ALG-NB-noteonly, CTL-RAND, CTL-FORMAT, CTL-BLIND]
route_rule: verified-any-visual-v1
batch: {problems_per_step: 8, max_attempted_problems_per_step: 24, epochs: 1}
seeds: [3407, 3408, 3409]
engine:
  vllm: {enforce_eager: false, enable_prefix_caching: false, mm_processor_cache_gb: 0,
         kv_cache_memory_bytes: 10737418240, max_num_seqs: 16, multiprocessing: false}
  probe_max_mean_abs_logprob_diff: 0.02      # G7 후 확정
health:                                      # §18.6
  kl_max: 0.1
  length_median_min_ratio: 0.3
  short_response_max_fraction: 0.2
  truncation_max: 0.3
  unverifiable_max: 0.01
  dev_drop_pp: 5
evaluation:
  external_first_attempt: {sampler: qwen_official, n: 4}
  pass_at_k: {n: 8, k: [1, 4, 8], temperature: 1.0}
  mmmu_final: {protocol: experiments/input-recovery protocol input-recovery-v1, max_new_tokens: 8192, parser: team-final-answer-v8}
```

## 부록 C. 레코드 예시 (형식 설명용 가상 값)

**actor_inputs.jsonl 한 줄** — 모델이 보는 전부

```json
{"problem_id": "gen-music-v1:music_staff:train:000017", "source": "gen-music-v1", "family": "music_staff",
 "mode": "obs", "question": "What is the interval between the two notes marked with arrows?",
 "choices": [], "answer_format": "short_text",
 "image_paths": ["gen-music-v1/train/000017.png"], "image_sha256": ["9f1c…"]}
```

**verifier_private.jsonl 한 줄** — verifier만 읽음

```json
{"problem_id": "gen-music-v1:music_staff:train:000017", "answer_type": "short_text", "gold": "M3",
 "accept": {"aliases": ["M3", "major third", "maj3"]}, "choices": [],
 "slots": {"note[1.1]": "E4", "note[1.2]": "G#4", "note[1.3]": "B4", "clef": "treble", "key_sig": "0", "time_sig": "4/4"},
 "slot_types": {"note[1.1]": "pitch", "note[1.2]": "pitch", "note[1.3]": "pitch", "clef": "text", "key_sig": "text", "time_sig": "text"},
 "required_slots": ["note[1.1]", "note[1.2]"],
 "oracle_fact_lines": ["note[1.1] = E4", "note[1.2] = G#4"],
 "printed_text": ["4", "4"]}
```

**lineage.jsonl 한 줄**

```json
{"problem_id": "gen-music-v1:music_staff:train:000017", "source": "gen-music-v1", "source_revision": "<generator file sha256>",
 "license": "project-authored", "label_provenance": "PROGRAM", "image_provenance": "rendered:verovio@6.3.0",
 "upstream_origin": "procedural", "generator_version": "gen-music-v1", "generator_seed": 3407,
 "template_id": "interval_marked_pair", "structure_hash": "a41f09c2d1e77b30", "split": "train",
 "parents": [], "benchmark_derived": false, "model_touched": false}
```

**rollout 원장의 Sample 한 줄** (요약)

```json
{"sample_id": "gen-music-v1:…:000017@12/T2/t1#3#2", "kind": "T2", "turn": 2, "parent_sample_id": "…/T1/root#3",
 "prompt_token_ids": "<ref:tokens/…npy>", "completion_token_ids": "<ref>", "sampler_logprobs": "<ref>",
 "finish_reason": "stop", "seed": 1883205411, "policy_version": 12,
 "result": {"status": "CORRECT", "extracted": "M3", "reward_answer": 1.0,
            "slots": {"parsed": 3, "required": 2, "required_correct": 2, "wrong_required": [], "all_correct": 3, "all_total": 6, "conflicting": []},
            "verifier_version": "strict-final-v1"},
 "reward": 1.0}
```

## 부록 D. 용어

| 용어 | 뜻 |
|---|---|
| F0 | 지정 revision의 원본 Qwen3-VL-4B-Instruct (학습 전) |
| RLVR | 검증 가능한 보상을 쓰는 강화학습. 정답을 따라 쓰는 학습이 아님 |
| actor | 문제를 푸는 모델(학습 중 정책) |
| verifier | 정답·슬롯을 가진 프로그램 채점기 |
| 슬롯(slot) | 이미지에서 읽어야 하는 값·관계 하나(예: `bar[Mar]`) |
| required slot | 그 질문을 풀 때 필요한 슬롯(verifier만 앎) |
| 그룹 | 같은 입력 상태를 공유하는 샘플들. advantage는 그룹 안에서만 비교 |
| 트리형 롤아웃 | turn-1 그룹 → 실패 풀이를 문맥에 고정한 turn-2 그룹 → … |
| advantage | 자기 보상 − 그룹 평균 보상 |
| KL | 고정 F0에서 얼마나 멀어졌는지의 벌점 |
| route | 그 문제에서 Vision LoRA까지 학습할지(VISION_ON) 말지(VISION_OFF) |
| 오답노트 L1/L2/L3 | host 검증 기록 / 재풀이 Error check / 사후 원인 진단 |
| Phase A | 가중치를 고정한 F0로 난도·재풀이 성공률·귀인율을 재는 사전 측정 |
| sealed | 학습·선택에 한 번도 쓰지 않는 최종 외부 평가셋 |
| style_sealed | 그림 스타일을 바꾼 sealed (생성기 스타일 암기 검사) |
| MIS | 시퀀스 수준 importance mask: vLLM과 학습 엔진 확률 차이가 큰 샘플을 손실에서 제외 |

