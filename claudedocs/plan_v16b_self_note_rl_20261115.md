# v16b 상세 설계서 — 오답노트형 자기개선 (RL + 자기생성 SFT), 원본에서 시작, 3,000문항 전부

- 작성: 2026-11-15, Claude(설계). 구현: Codex. 실행: 구현·검증 뒤 사용자 결정 12번 범위 안에서 진행.
- 상태: **설계서. 아직 구현·실행하지 않았다.**
- 관계:
  - 이 문서가 v16의 학습 정본이다.
  - `claudedocs/plan_v16_grounded_self_note_20261115.md`(관측 문장 SFT 중심 v16)는 사용자가 2026-11-15에 중단했다. 거기서 만든 부품(원 주석 연결, 노트 검증, 비병합 서빙, BF16 가시성 측정)은 재사용한다.
  - 학습 자료는 Codex가 `claudedocs/plan_v16b_data_curation_20261115.md`에 따라 따로 선정한다. 이 문서는 그 산출물을 받는 쪽의 계약만 정한다.
- 조사 근거(비공개): `/home/jang2/mmdl-artifacts/self-note-single-20261105/v16b/design-research-20261115/`
  - `A_data.md`: 자료
  - `B_time_cost.md`: 시간·비용
  - `C_literature.md`: 문헌
  - `D_code_plan.md`: 코드 지도
  - `E_async.md`: 연속 생성 가능성
  - `V_final.md`: 검증
  - 이 조사들은 "문항당 4회 풀이·4,000문항"을 전제로 했다. 사용자가 그 뒤 **1회 풀이 + 재풀이 1회, 3,000문항**으로 바꿨으므로, 이 문서의 시간·구성은 그 결정으로 다시 계산했다(§11).
- 표기:
  - **[M]** 실측(파일 근거 있음)
  - **[M-d]** 실측에서 계산
  - **[A]** 가정
  - **[결정]** 사용자 결정
  - **[설계]** 이 문서의 선택

---

## 0. 한 장 요약

```
준비 SFT ─▶ (연속 투입) 문항마다: 정답 없이 첫 풀이 1회 ─▶ V8식 채점
         ─▶ 틀렸거나(끝까지 풀고 틀림) 맞았지만 오류가 확인되면: 자기 오답노트 1회 ─▶ 주장 검증
         ─▶ 교훈만 보고 정답 없이 재풀이 최대 1회
         ─▶ 16문항마다 가중치 갱신 = 첫 풀이 RL(기준값 차감) + 그림 관측 RL + 검증된 노트·재풀이 SFT
         ─▶ 끝까지 반복 ─▶ 체크포인트 하나로 MMMU 900 1회 평가 (V8 + CompassVerifier 보완)
```

- **학습 문항 3,000개 전부를 RL로 돈다**(16문항 × 187묶음 + 마지막 8문항 1묶음 = 188묶음). 시간으로 학습을 자르지 않는다(사용자 2026-11-15: "시간을 하드코딩으로 하지는 말고 3000문제 전부").
- **준비 SFT**: 학습 3,000문항 **밖**의 오염 검사 통과 문항 512개(별도 준비 집합)를 원본 모델이 먼저 한 번 풀고, 그 정답 풀이와 그 문항들의 데이터셋 사람 풀이로 SFT한다. 3,000문항 안에서 고르면 RL 첫 풀이 전에 그 문항의 정답을 학습하게 되어 "첫 풀이는 정답 없이" 규칙(§2.1)을 어기므로 분리한다(감사 결함 5, §8).
- **문항당 첫 풀이 1회.** 같은 문항을 여러 번 풀지 않는다. 그래서 그룹 평균 대신 **풀기 전에 정한 기준값**을 빼는 단일 응답 RL을 쓴다.
- **학습용 생성은 후보를 자르지 않는다**(top-p 1, top-k 끔, 온도 0.7, presence 1.5). 칭찬과 감점 모두 같은 확률(처리된 전체 어휘 분포)로 계산한다. 평가는 B2 그대로다.
- **생성은 평가처럼 연속 투입한다.** 갱신 순간에도 진행 중인 긴 풀이는 버리지 않고 KV를 들고 기다렸다가 이어 간다. 어느 요청을 어느 갱신 뒤에 넣는지는 정해진 일정(§4.3)을 따르므로 결과가 시간에 따라 흔들리지 않는다.
- **오답노트는 검증된 주장 구간만 학습한다.** 교훈 문장은 학습하지 않고 재풀이 입력으로만 쓴다. 재풀이가 맞으면 그 풀이를 평가와 같은 P0 입력으로 SFT한다.
- **그림 주석이 있는 문항**(도형·그래프·장면)은 모델이 따로 쓴 관측을 주석과 대조해 보상한다. 주석으로 만든 문장을 학습 목표로 쓰지 않는다.
- **GPU**: 커뮤니티 A100 SXM 80GB(1.39달러/시, 허용 목록 안).
  - 예상 전체 시간 약 13.6~22.6시간, 비용 약 19~32달러(§11, 추정).
  - 시간 마감으로 학습을 멈추지 않는다. 잔액이 평가 비용 밑으로 내려갈 때만 충전을 요청하는 자금 보호와 평가 단계 자동 정지만 둔다(§9.1). 이것은 프로젝트 규칙 §2.6이 요구하는 장치이고 학습 계획의 일부가 아니다.
- **기대**: 이 설계가 MMMU 585를 낸다는 근거는 없다. 지금까지 다섯 판이 원본 재실행과 구별되지 않았다. 이 문서는 프로젝트가 의도한 자기개선을 올바르게 구현하고 그 결과를 한 번 측정하는 것을 목표로 한다.

---

## 1. 사용자 결정과 규칙

### 1.1 사용자 결정 (2026-11-15)

| # | 결정 | 이 문서에서 |
|---|---|---|
| 1 | 준비 SFT = 원본이 정답 없이 맞힌 자기 풀이 + 데이터셋 사람 풀이 | §8 |
| 2 | 학습 중 첫 풀이 **1회** + 교훈 재풀이 **최대 1회**(4회 고정에서 변경) | §4, §6 |
| 3 | 학습용 생성은 후보 절단 없이(top-p 1, top-k 끔), 칭찬·감점 같은 확률 | §4.1, §7 |
| 4 | 오답노트 대상 = 틀린 문항 + 맞았지만 관측·계산 오류가 확인된 문항 | §6.1 |
| 5 | 오답노트 = 검사 가능한 구조(읽은 값, 틀린 단계, 고친 값, 짧은 교훈) | §6.2 |
| 6 | 원 주석·재계산으로 확인된 주장만 학습, 나머지 제외 | §6.4 |
| 7 | 재풀이 1회 | §6.5 |
| 8 | 그림 주석 문항에 별도 관측 요청, 주석 대조로 보상(평가 입력은 P0) | §5.4 |
| 9 | 원본(F0)에서 새로 시작 | §7.7 |
| 10 | 3,000문항으로 줄여 MMMU·MMMU-Pro에 도움 되는 문항으로 재선정(Codex), AI2D 오염 통과분 전부 포함 | §3 |
| 11 | 평가 = 체크포인트 하나로 MMMU 900 전체 1회, V8 + CompassVerifier-7B 보완 | §10 |
| 12 | 구현 → CPU 검사 → 로컬 4비트 작동 확인(짧게) → MMDL_test 게시 → 런포드 학습 + 900 평가까지 한 번에 진행 | §14 |
| — | 연속 투입: 16문항이 끝날 때마다 갱신하고, 진행 중인 긴 풀이는 계속 유지, A100 1대 | §4 |
| — | (처음) 테스트까지 15시간 이내 → **(변경) 시간을 하드코딩하지 말고 3,000문항 전부 학습** | §9, §11 |

### 1.2 v16b 한정 규칙 변경 (구현 시작 전에 `CLAUDE.md` §2.1에 날짜와 함께 기록)

1. 학습용 생성 표본 설정을 바꾼다(top-p 1, top-k 끔). 평가 설정은 그대로다.
2. 학습 중 vLLM이 KV를 유지한 채 HF 갱신과 같은 GPU에 함께 올라간다. 이는 v1 경로 §3.0-5 "생성을 끝낸 뒤 학습해 그래픽카드를 나눠 쓰지 않는다"의 v16b 예외다. 계산이 동시에 돌지는 않으며, 갱신 중에는 생성 단계를 멈춘다.
3. 맞힌 문항도 확인된 오류가 있으면 오답노트를 쓴다.
4. 그림 주석 문항에 우리가 쓴 고정 관측 요청문을 붙인다. 주석은 채점에만 쓰고 학습 목표 문장으로 쓰지 않는다. 기존 그림·문항을 바꾸거나 새 문항을 만들지 않는다(2026-10-30 규칙 범위 안).
5. 재풀이는 1회다(v15 한정 3회 예외는 상속하지 않음).
6. 평가는 900 전체 1회다(450+450 2단계 규칙은 v16b에 적용하지 않음).

### 1.3 그대로인 규칙

- 상위 모델 글 학습 금지. MMMU·MMMU-Pro 자료·파생물 학습 금지.
- 첫 풀이는 정답 없이 푼다. 정답은 채점 뒤 노트 입력에서만 보여 준다.
- 평가 입력에는 노트·정답·교훈을 넣지 않는다.
- 평가는 문항당 답 하나이고, 여러 답·체크포인트·seed 중 고르지 않는다.
- V8이 주 점수다. 보완 판정은 따로 보고하며 학습에 쓰지 않는다.
- v11~v16 protocol 해시와 동결 파일은 바꾸지 않는다.
- 교수님 MMDL과 V8 파서 파일은 건드리지 않는다.
- 런포드는 MMDL_test 고정 commit의 clean clone으로만 실행한다. 로컬 모델·가중치 업로드는 금지다(자료 묶음은 기존 `data_bundle` 방식 허용).
- 가격을 먼저 밝히고, 쓰지 않는 포드는 즉시 삭제하며, 해시 확인 뒤에 삭제한다.

---

## 2. 단계 흐름과 시간 배치

| 단계 | 내용 | 시간(A100 SXM) |
|---|---|---|
| S0 설치 | uv 설치, Qwen(BF16)·CompassVerifier-7B 내려받기, 자료 묶음 풀기·해시 확인, GPU 점검 | 0.18시간 [M] |
| S1 엔진 시작·사전 점검 | vLLM(비병합 LoRA + 시각 훅) 시작, PF0~PF5 | 0.09시간 [M-d] |
| S1b 갱신 메모리 측정 | 최대 길이 합성 행 1개로 역전파만 해 X를 잼(기울기 버림) → KV 크기 결정(§4.7) | 약 0.05시간 [A] |
| S2 원본 풀이 생성 | 준비 집합 512문항(3,000 밖) + 감시 집합 약 200문항을 F0로 1회 풀기(B2 평가 설정). 준비 집합 주석 문항에는 OBS 요청도 학습 표본 설정으로 1회(관측 기준값 시작값) | 약 0.5시간 [A] |
| S3 준비 SFT | 자기 정답 풀이 + 사람 풀이로 약 11~18회 갱신 | 약 0.3시간 [A] |
| S3b 감시 집합 재측정 | 준비 SFT 직후 감시 집합 한 번 더(준비로 생긴 길이 변화 확인) | 약 0.1시간 [A] |
| S4 RL 연속 학습 | **188묶음 전부(3,000문항) + 마무리 갱신 3회**. 시간으로 자르지 않음 | 약 7.0~15.2시간 [A] |
| S5 감시 집합 재측정 | 최종 체크포인트로 감시 집합 다시 풀기 | 약 0.15시간 [A] |
| S6 내보내기·평가 사전 점검 | 비병합 조정층 내보내기, 재로딩 확인, 평가 준비 | 0.22시간 [M] |
| S7 MMMU 900 평가 | B2·P0·V8·seed 3407, 비병합, 슬롯 auto(32) | 3.76~4.63시간 [M-d] |
| S8 보완 판정·회수·삭제 | CompassVerifier-7B, 결과 회수·해시 확인, 포드 삭제 | 0.15시간 [M] |
| 여유 | | 0.5시간 |

합계 약 13.6~22.6시간 [A](§11). 학습은 3,000문항을 다 돌 때까지 이어 가고, 감시 규칙(§9.3) 또는 자금 보호 규칙(§9.1)에 걸릴 때만 일찍 멈춘다.

---

## 3. 자료 인터페이스 (Codex 자료 선정 산출물을 받는 계약)

### 3.1 입력 파일

자료 선정 지시서(`plan_v16b_data_curation_20261115.md`)의 산출물을 쓴다.

| 파일 | 쓰임 |
|---|---|
| `train-3000-manifest.json` (+SHA256) | 학습 문항 3,000개의 원문·그림·정답 참조, 주석, 사람 풀이, 사전 난이도, 영수증 |
| `order.json` (+SHA256) | 고정 순서. **3,000개 전부 = RL 집합**(187묶음 × 16 + 마지막 1묶음 × 8). 준비 집합 512개는 §8의 규칙으로 이 3,000 안에서 고른다 |
| `monitor-manifest.json` (+SHA256) | 감시 집합 약 200개(학습 금지) |
| `prep-manifest.json` (+SHA256) | **준비 집합 512개**: 학습 3,000·감시 집합과 계열·그림이 겹치지 않는 오염 검사 통과 문항. 출처 비율은 학습 3,000과 비슷하게, 사람 풀이 있는 문항(ScienceQA 등)을 포함 |
| 조립된 vault | 학습 3,000 + 감시 + 준비 문항의 `actor_inputs`·`verifier_private`·`lineage`·그림을 한 루트로 모으고, 전체를 덮는 `source_provenance`를 함께 낸다. v10 `provisional-vault`에는 INCLUDED 3,500개만 있고 EXCLUDED_QUOTA 2,214개는 여러 루트에 흩어져 있기 때문이다(감사 결함 8) |

자료 선정 지시서의 §9·§10에 다음 두 조건을 추가해 Codex에 전달한다.

1. 묶음마다 출처·사전 난이도·주석 유무·긴 풀이 위험 비율이 전체와 비슷해야 한다. 감시 규칙(§9.3)이 묶음 구성 변화 때문에 잘못 걸리지 않게 하고, 감시 정지로 일찍 멈춰도 학습된 부분이 전체의 축소판이 되게 한다.
2. 준비 집합 512개는 자료 선정 쪽이 학습 3,000 **밖**에서 고른다(`prep-manifest.json`). 학습 코드는 이 목록의 해시를 확인하고, 학습 3,000과의 겹침이 0인지 검사한다.

순서 규칙은 검증된 "2단 고른 간격"을 권한다(`V_final.md` §4: 출처 안에서 사전 난이도·길이 층을 고른 간격으로 섞고, 출처 사이를 다시 고른 간격으로 섞음). 쉬운 것→어려운 것 정렬은 하지 않는다(`C_literature.md` §1: 고정 난이도 순서는 무작위보다 나쁜 경우가 있음).

### 3.2 문항별로 필요한 필드

- 식별: `problem_id`, `source`, `family`, `group_id`, `image_paths`, `image_sha256`, `pixel_sha256`
- 기존 vault 기록 참조: `actor_inputs`(질문·보기), `verifier_private`(정답·답 형식·허용 답), `lineage`
- 주석(있으면): `layer ∈ {geometry, graph, scene}`, 원 주석 연결 영수증(`content_v16`의 연결 결과와 같은 형식)
- 사람 풀이(있으면): 출처 필드(ScienceQA `lecture`·`solution` 등)와 **텍스트 오염 검사 영수증**
- `f0_prior`(있으면): 원본 모델의 예전 정오·길이·종료 사유
- `rights`, `contamination_receipt`

### 3.3 포드 전달

- 자료 선정 산출물이 가리키는 모든 파일(그림·vault 기록·주석·영수증)을 `notebook_rl/pod/data_bundle.py`로 **같은 절대경로**에 풀고 해시를 확인한다(`D_code_plan.md` §10.4).
- MMMU 자료·모델 캐시는 넣지 않는다.
- AI2D 포함 자료는 비공개 포드에서만 쓰고 공개 저장소에는 올리지 않는다.

---

## 4. 생성 런타임 — 연속 투입과 정해진 일정

### 4.1 학습 엔진 설정

| 항목 | 값 | 근거 |
|---|---|---|
| 엔진 | vLLM 0.11, 프로세스 안(`VLLM_ENABLE_V1_MULTIPROCESSING=0`), 비병합 LLM LoRA + 시각 훅 + Merger push (v15 경로) | `llm_lora.py`, `single_vllm.py` [M] |
| 표본 설정(학습 RL·노트·재풀이·관측) | `TRAIN_SAMPLING_V16B = {temperature 0.7, top_p 1.0, top_k 0, presence_penalty 1.5}`, repetition 1.0 | [결정] 3. vLLM 0.11에서 `top_k=0`은 끔(`sampling_params.py:130,406-408`) [M] |
| 표본 설정(준비 풀이·감시 집합) | B2 평가 설정(`SINGLE_SAMPLING`: top-p 0.8, top-k 20) | 평가 때와 같은 풀이를 학습·측정 |
| 확률 기록 | `logprobs_mode="processed_logprobs"`, `logprobs=0`(뽑힌 토큰 하나) | `D_code_plan.md` §2.1 [M]. 절단이 없으면 기록값 = `log_softmax((logit − 1.5·생성 중 등장)/0.7)[행동]` |
| 동시 서열 | `max_num_seqs` **32 고정** | 감사 모의: 생성이 꼬리에 묶여 16·32·48의 묶음 시간이 같다(평균 활성 7.7개). 늘려도 이득 없음 |
| KV 크기 | `KV = min(24, 52.8 − X)` GiB, RL 시작 전에만 정함. 16 GiB 미만이면 정지 | §4.7. 52.8 = 79.25 − vLLM 가중치 8.74 − 그래프·인코더 3 − LoRA 슬롯 0.2 − HF 상주 10.5 − 여유 4 (감사 결함 7) |
| prefix cache | **끔** | 켜면 옛 그림 임베딩이 다른 판의 요청에 재사용될 수 있음(`E_async.md` §2) |
| mm 처리 캐시 | `mm_processor_cache_gb=0` | 같음 |
| LoRA 판 수 | `max_loras=3`, **`max_cpu_loras=3`(같게)** | 동시에 살아 있는 판은 최대 2개(§4.4). 지금 v15 값은 1. `max_cpu_loras`가 더 크면 슬롯이 다 찬 상태의 `add_lora`가 진행 중 요청에 다른 판을 오류 없이 적용한다(`E_verify.md` 수정 1, vLLM 관리자 클래스 CPU 실행으로 확인) |
| 절전 | **RL 중 쓰지 않음.** 진행 중 요청이 있으면 vLLM 절전이 KV를 버린다 | `E_async.md` §3 [M] |
| CUDA graph | 학습 엔진은 v15처럼 사용. 평가 엔진은 B2대로 eager | [M] |

### 4.2 요청 종류

| 종류 | 입력 | 상한 | 표본 | seed 단계 | 대상 |
|---|---|---|---|---|---|
| `T1` | P0(평가와 같은 `prepare_protocol_inputs`, B2 이미지 예산) | 8,192 | 학습 | `v16b-t1` | RL 집합 모든 문항 |
| `OBS` | 그림 + 층별 고정 관측 요청문 | 1,024 | 학습 | `v16b-obs` | RL 집합 중 주석 문항 |
| `NOTE` | 노트 프롬프트(§6.2) | 1,024 | 학습 | `v16b-note` | §6.1 대상 |
| `RESOLVE` | 그림 + 질문 + 검증된 교훈(정답 없음) | 8,192 | 학습 | `v16b-resolve` | 노트 통과 문항 |
| `PREP` | P0 | 8,192 | B2 | `v16b-prep` | 준비 집합 512 (`prep-manifest.json`, 학습 3,000 밖) |
| `MONITOR` | P0 | 8,192 | B2 | `v16b-monitor` | 감시 집합 |

- 학습 상한을 8,192로 둔다. 2026-11-02 규칙("8,192 유지")을 지키고, 상한을 낮추면 풀이가 짧아지는 압력이 생기기 때문이다(`C_literature.md` §2.3).
- `contract.sample_seed`에 분기 하나를 추가한다. 단계 이름이 `v16b-`로 시작하면 `key = f"{master_seed}:{problem_id}:{stage}"`로 한다. 기존 단계(`t1`, `note`, `resolve`, `resolve2`, `resolve3`, `control`)의 값은 바뀌지 않아야 한다.
- 한 요청에 한 표본이다(`n=1`).

### 4.3 정해진 일정 (η = 1)

갱신 횟수를 판 번호 v로 쓴다. 시작은 v=0이고, 갱신 v가 끝나면 v+1이 된다. 묶음 c는 순서의 16문항 [16c, 16c+16)이다(마지막 묶음 187은 8문항).

**판 v가 시작되는 순간**(시작 시점 또는 갱신 v−1 직후)에 다음 요청을 넣는다.

1. 묶음 v+1의 `T1`과 `OBS`. 시작 시점에는 묶음 0과 1을 함께 넣는다.
2. 묶음 v−1 문항의 `NOTE`. 그 묶음의 첫 풀이는 갱신 v−1에 이미 쓰였고, 채점도 끝나 있다.
3. 묶음 v−2 문항의 `RESOLVE`. 그 노트들은 갱신 v−1 전에 끝나 검증까지 마쳤다.

**갱신 v는 다음이 모두 끝나야 시작한다.**

- 묶음 v의 `T1`·`OBS` 전부
- 묶음 v−1의 `NOTE` 전부
- 묶음 v−3의 `RESOLVE` 전부
- **엔진 대기열이 비어 있고, 실행 중인 모든 요청이 프리필(그림 처리 포함)을 마쳤을 것**(`num_computed_tokens ≥ 프롬프트 길이`). 그래야 갱신 뒤 Merger·시각 훅을 바꿔도 이미 들어간 요청의 그림 처리가 옛 판에서 끝나 있어, 그림 판이 실행 시간에 좌우되지 않는다(감사 결함 9, 비용 십여 단계).

**요청을 넣지 않는 경우**: 그 결과를 쓸 갱신 번호가 마지막 갱신(§4.3 끝, 마무리 갱신 포함)보다 크면 넣지 않는다.

**갱신 v에 들어가는 행**:
- RL: 묶음 v의 `T1` 행
- 관측 RL: 묶음 v의 `OBS` 행
- 노트 SFT: 묶음 v−1의 검증된 노트 행
- 재풀이 SFT: 묶음 v−3의 수용된 재풀이 행

**이 일정의 성질**
- 어느 요청이 어느 판으로 생성되고 어느 갱신에 들어가는지가 묶음 번호로 정해진다. 실행 시간에 따라 바뀌지 않으므로, 같은 판·같은 seed면 같은 요청이 된다(엔진 수치 비결정성 제외).
- 묶음 v의 긴 첫 풀이를 기다리는 동안 묶음 v+1의 첫 풀이, 노트, 재풀이가 함께 돌아서 GPU가 쉬지 않는다.
- RL 행은 판 v−1에서 생성되어 갱신 v(판 v)에 쓰인다. 묶음 0만 지연 0이고 나머지는 **지연 1**이다. AReaL에서 지연 1은 보정 없이도 동기와 거의 같았다(41.8 대 42.0, `C_literature.md` §2.4). 그래도 §7.3의 중요도 보정을 적용한다.
- 노트·재풀이는 SFT 행이라 지연과 무관하다.
- **마지막 묶음 처리**: 마지막 RL 묶음(187)의 갱신 187 뒤에도 남은 노트(묶음 187)와 재풀이(묶음 185~187)는 끝까지 생성한다. 같은 일정 규칙으로 **마무리 갱신 188·189·190**(정확히 3회, 새 첫 풀이 없음, SFT 행만)을 한다. 행이 없는 마무리 갱신은 optimizer step 없이 넘어간다. 그래서 3,000문항 모두 첫 풀이·노트·재풀이 경로를 다 거치고, 전체 단계 수는 실행 전에 정해진다(준비 갱신 수 + 191).
- **선택(시험 측정 뒤)**: 생성이 갱신을 기다리며 비는 시간이 길면 η=2(묶음 v+2까지 미리 투입)로 바꿀 수 있다. 바꾸는 판단은 첫 4묶음의 GPU 유휴 비율로만 한다. 바꾸면 그 시점부터의 규칙을 `schedule-change.json`에 봉인하고, 정답률을 보고 바꾸지 않는다.

### 4.4 판 관리

- **LLM LoRA**: 갱신 v가 끝나면 `write_llm_adapter`로 `/dev/shm/v16b-lora-{v+1}`에 쓰고, `add_lora`로 새 id를 올린다. 새로 넣는 요청은 새 id를 쓰고, 진행 중 요청은 원래 id로 끝까지 간다.
  - LoRA id는 1 이상이어야 하고(`lora/request.py:34`), PF5 탐침이 id를 먼저 쓴다. 그래서 id는 표본기의 기존 카운터로 새로 받고, 등록부가 "판 → id" 대응을 가진다(판 번호를 id로 직접 쓰지 않는다).
  - **`add_lora` 전에 빈 GPU 슬롯을 반드시 확인한다.** 살아 있는 판이 `max_loras`와 같으면 올리지 않고 오류로 멈춘다.
  - 판 수 상한은 vLLM 스케줄러에 맡기지 않고 우리 코드에서 보장한다(`E_verify.md` 수정 3). §4.3 일정에서는 살아 있는 판이 최대 2개다. 이를 assert로 확인하고, 넘으면 오류다.
  - 판별 참조 수(대기+실행)가 0이 되면 `remove_lora`를 한 뒤 경로를 지운다. 참조가 남은 판을 지우면 엔진이 그 경로를 다시 읽다가 멈춘다(`E_async.md` §1).
  - **푸시마다 판 일치 검사**: 새 판을 올린 직후, 살아 있는 모든 판에 대해 슬롯 내용 비트 일치(`llm_lora.check_identity`, v15 PF2 방식)와 Merger 비트 일치를 확인한다. 지연 1 행에는 엔진·HF 재생 비교가 없으므로, 잘못 적용된 판을 잡는 장치가 이것뿐이다(`E_verify.md` 수정 5).
- **Merger·시각 훅**: 전역 가중치라 요청별 판을 가질 수 없다. 갱신 직후 **단계와 단계 사이**에 제자리 복사로 바꾼다.
  - `push_mergers`에 `inflight=True` 방식을 추가한다. 유휴 검사, `reset_prefix_cache`, 인코더 캐시 지우기를 하지 않는다. 대신 prefix cache 끔과 mm 처리 캐시 0을 assert로 확인한다.
  - 그림 인코더는 요청마다 프리필에서 한 번 돈다. 그래서 한 요청의 그림 특징은 한 판에서 나오고, 그 판은 요청 기록의 `visual_version`(프리필 시점 판)에 남긴다.
  - LoRA 판과 그림 판이 다를 수 있다(프리필 전에 갱신이 끼면). 이 경우에도 중요도 보정은 성립한다. 실제로 뽑을 때의 확률 μ를 기록하기 때문이다. 다만 `mixed_version=true`로 표시하고 개수를 보고한다.
  - §4.3의 갱신 시작 조건(대기열 비움 + 모든 실행 요청 프리필 완료) 때문에, 정상 경로에서는 `visual_version == policy_version`이 항상 성립한다.
  - 그래도 푸시 시점에 출력 토큰이 아직 없는 요청이 있으면(조건 위반 또는 선점 재개) `mixed_version=true`로 표시하고 개수를 보고한다. `num_computed_tokens == 0` 기준은 그림 앞 토큰만 계산된 요청을 놓치므로 "출력 토큰 없음"을 기준으로 한다(`E_verify.md` 수정 2). 혼합은 오류 표시이며, 묶음 RL 행의 5%를 넘으면 그 갱신을 HOLD한다.
- **선점(재계산)**: 다시 넣어지면 그림을 새 판으로 계산할 수 있다. 엔진 상태에서 `PREEMPTED`를 확인해 `preempted=true`로 표시한다. 선점이 많으면 `max_num_seqs`를 줄인다.

### 4.5 생성 기록 필드 (`single_vllm.finish`의 `generation`에 추가)

| 필드 | 뜻 |
|---|---|
| `policy_version`, `lora_int_id` | 입장 시 판과, 등록부가 그 판에 준 LoRA id(판 번호와 다를 수 있음, §4.4) |
| `visual_version` | 프리필 시점 Merger·시각 훅 판 |
| `mixed_version`, `preempted` | §4.4 |
| `policy_sha256` | **입장 시 판의 해시**(지금은 스트림 생성 시 값이라 바꿔야 함) |
| `sampling`, `sampling_sha256` | 해당 요청의 표본 설정 |
| `probability_space` | `"processed-full-vocab-v1"`(학습 표본) 또는 `"processed-truncated-b2"`(준비·감시) |
| `sampling_logprobs` | 뽑힌 토큰의 기록 확률 μ_t |
| `admit_step`, `finish_step` | 엔진 단계 번호 |
| `kind` | `T1 / OBS / NOTE / RESOLVE / PREP / MONITOR` |
| 기존 필드 | `prompt_token_ids`, `completion_token_ids`, `text`, `finish_reason`, `seed`, `request_sha256`, `input_sha256` 유지. 바깥 출력의 `completion_tokens` 위치는 D1 수정대로 |

- 학습용 전체 분포 표본에서는 후보 목록(`support_token_ids`)을 저장하지 않는다.
- 입력 준비(`prepare_protocol_inputs`)는 문항 하나에 한 번만 하고, 같은 문항의 `T1`·`OBS`·`NOTE`·`RESOLVE`가 그림 처리 결과를 메모해 재사용한다. 이때 입력 해시로 같은 입력인지 확인한다(요청당 1.51초 CPU 부담 [M-d] 절감).

### 4.6 루프 안 처리 (CPU)

출력이 끝날 때마다 다음을 처리한다.

- **T1**: V8식 학습 채점(`single.grade_answer`의 v14 계약), 보상(§5), `t1_error_scan`(§6.1)
- **OBS**: 관측 채점(§5.4)
- **NOTE**: 노트 검증(§6.3)
- **RESOLVE**: 재풀이 수용(§6.5)

결과는 문항 상태 파일(`items/<problem_id>.json`, 원자적 저장)에 남긴다.

### 4.7 메모리

- A100 80GB에 다음이 동시에 올라간다.
  - vLLM: 가중치 8.74 GiB [M] + 그래프·인코더 2~3 GiB + LoRA 슬롯 0.2 GiB + KV
  - HF: 상주 약 10.5 GiB + 갱신 최대치 X
- **X 측정(S1b)**: X를 정하는 것은 가장 긴 한 행이다(프롬프트 약 4,000 + 행동 8,191, FULL 경로). 준비 행은 4,096토큰 이하라 준비 단계 측정으로는 상한을 알 수 없다. 그래서 RL 시작 전에 이런 최대 길이 합성 행 1개로 역전파만 하고(기울기 버림, optimizer 없음) `X = max_memory_reserved − 갱신 직전 값`을 잰다(같은 프로세스).
- **KV 크기**(감사 결함 7, `E_verify.md` 수정 6):
  - `KV = min(24, 52.8 − X)` GiB이고, RL 시작 전에만 정한다(바꾸려면 엔진 재시작 약 340초와 진행 중 서열 손실이 생긴다).
  - KV가 16 GiB 미만이면 정지하고 보고한다.
  - 참고: 실행 중 서열 평균 문맥 약 4,000토큰 기준 동시 32에 약 21 GiB이지만, 감사 모의의 KV 최대 수요는 일정상 10~14 GiB다(대부분 시간에 활성 서열이 적음).
- 갱신은 `token_budget=12,000`으로 행을 묶어 처리하고, 확률은 32단계 덩어리로 계산해 로짓 전체를 한 번에 만들지 않는다.
- **결정성 설정**: v15 스트림은 vLLM 단계 동안 PyTorch 결정성 알고리즘을 끄고 스트림이 끝나면 다시 켠다(`single_vllm.py:36-46, 371-373`). 연속 일정에서는 갱신이 스트림 안(단계 사이)에서 일어나므로, **갱신 직전에 결정성을 켜고 갱신이 끝나면 다시 끈다.** 이 전환을 런타임 한 곳에서만 하고, 시험으로 갱신 중 결정성이 켜져 있음을 확인한다(`E_verify.md` 수정 4).
- **동시 서열 상한 코드**: `single_campaign.py`의 `max_num_seqs ≤ 16` 검사(약 222행)를 v16b 분기에서 32까지로 푼다.

### 4.8 로컬 경로 (작동 확인 전용)

- 8GB 로컬 GPU는 vLLM 4B를 올릴 수 없다. 로컬은 HF NF4 생성기(`collect._single_generate_batch`)로 **같은 일정을 순차 실행**하는 모드(`--local-sequential`)만 쓴다.
- 판 관리(§4.4)는 CPU 가짜 엔진 시험(§13)으로 확인하고, 실제 첫 실행은 포드 시험 단계다.

---

## 5. 보상과 기준값

### 5.1 첫 풀이 보상

| 상태 | 보상 r | 정책 기울기 |
|---|---|---|
| CORRECT | 1 | 사용 |
| INCORRECT, 최종답 없음(정상 종료), 복수 답 | 0 | 사용 |
| 학습 상한(8,192) 도달 | 0 | **사용 안 함**(행 제외). 짧게 끝내라는 압력을 피함(DAPO 방식, `C_literature.md` §2.3) |
| 반복 루프로 잘림 | 0 | 사용 안 함 |
| UNVERIFIABLE | 없음 | 사용 안 함 |

### 5.2 기준값 (현재 첫 풀이를 보기 전에 정함)

문항 i(출처 s)의 기준값은 다음과 같다.

```
b_i = (a · m_s + k_i) / (a + 1)    # f0_prior가 있을 때 (k_i = 예전 원본 정오 0/1), a = 2
b_i = m_s                          # 사전값이 없을 때
```

- **m_s의 시작값** `m0_s`: S2 준비 집합의 F0 정답률(출처 s). 준비 집합에 그 출처가 없으면 전체 정답률을 쓴다.
- **m_s의 계산**(표본이 적은 구간 포함, 감사 결함 11):
  ```
  m_s = (n0 · m0_s + Σ 최근 창의 보상) / (n0 + n),   n0 = 8, 창 = 확정된 RL 묶음의 출처 s 첫 풀이 최근 64개
  ```
  상한 도달도 보상 0으로 센다. 작은 출처는 창이 다 차지 않아도 이 식으로 매끄럽게 넘어간다.
- 갱신 v의 기준값은 묶음 v−1까지의 기록으로만 계산한다. 그래서 결정적이다.
- 기준값과 출처별 m_s는 묶음마다 `update-stats.json`에 기록한다.

### 5.3 이득

- 이득은 A_i = r_i − b_i이다. 표준편차로 나누지 않는다.
- 같은 응답의 보상을 자기 평균으로 빼는 일(이득 0)은 생기지 않는다.

### 5.4 관측 보상 (`OBS`)

- **점수**: `obs_score(item, layer, text)`는 모델 출력과 원 주석에서 만든 기준 사실 집합을 비교한 F1이다(`D_code_plan.md` §6).
  - 도형: 표시된 길이·각도 라벨, 직각·평행
  - 그래프: 그려진 노드 집합, 간선 집합
  - 장면: 물체 목록
  - 읽을 수 없거나 정상 종료가 아니면 0이다.
- 이득은 `F1 − b_obs[layer]`이다. b_obs는 `(n0·시작값 + 최근 32개 관측 점수 합)/(n0 + n)`, n0 = 8이다.
  - 시작값: S2에서 준비 집합의 주석 문항에 OBS 요청을 **학습 표본 설정**으로 한 번씩 넣어 잰 층별 평균 F1이다(약 70요청 × 250토큰). 근거 없는 0.5를 쓰지 않는다. 준비 집합에 그 층 문항이 없으면 그 층 첫 묶음의 OBS 점수 평균으로 시작하고, 그 묶음의 OBS 이득은 0으로 둔다.
- 관측 요청문은 층별 고정 문장이다. `content_v16.content_prompt`에 출력 형식 한 줄을 붙인다. 문제·보기·정답은 넣지 않는다.

---

## 6. 오답노트 N과 재풀이 C

### 6.1 대상

- **T1이 정상 종료했고 오답**(INCORRECT, 최종답 없음, 복수 답) → 노트를 쓴다.
- **T1이 정답이지만 `t1_error_scan`이 확인된 오류를 찾음** → 노트를 쓴다(오류 검사 대상: 도형·그래프·장면 주석과 다른 사실 줄, 또는 재계산이 틀린 단언 등식).
- **T1이 상한·반복으로 잘림** → 노트를 쓰지 않는다. 확인할 내용이 없고, 긴 입력으로 비용만 든다.
- `t1_error_scan(text, catalog, rules)`: 프로그램만 쓴다. `self_note_v16`의 등식 정규식, `earlier_assertion_status == "ASSERTED"`, `_calculate`를 재사용한다(T2 수정본: 문장 속 등식, 가정·예시·조각 제외).
- **`bind_statement(text, catalog) -> list[dict]`(새 함수)**: 풀이 문장의 자연스러운 표기를 카탈로그 사실에 잇는다(감사 결함 6).
  - 기존 `_FACT_LINE`은 `<fact_id> = 값` 꼴만 읽는데, fact_id는 `f3` 같은 내부 이름이라 자유 풀이에는 나오지 않는다.
  - 인식 대상: 선분 `AB = 5`, 각 `∠ABC = 30°`·`angle ABC = 30`, 간선 `A–B`·`edge (1, 2)`, 장면 물체의 색·모양.
  - 정확히 일치하는 개체 표기만 인정하고, 둘 이상에 걸리거나 모호하면 UNVERIFIABLE이다.
  - `t1_error_scan`과 `verify_note_v16b`가 같은 함수를 쓴다.
- **실측 수율과 보고 문구(미리 고정)**:
  - 저장된 F0 정상 종료 풀이 3,134개 중 재계산이 틀린 단언 등식은 오답 1,084개 중 1개, 정답 2,050개 중 0개였다(감사 `scan_check.py`).
  - 따라서 "맞았지만 오류가 확인된 문항" 노트는 거의 생기지 않을 것이다. 노트의 검증 주장은 주로 주석 문항의 관측 값이 된다.
  - 전체 실행에서 검증된 `error` 주장이 30개 미만이면 최종 보고에 "오답노트 교정 학습은 사실상 없음(관측 값 위주)"이라고 쓴다.

### 6.2 노트 프롬프트 (`NOTE_INSTRUCTION_V16B`, 새 상수)

`NOTE_INSTRUCTION_V16`을 바탕으로 세 가지를 바꾼다.

1. 맞힌 풀이에도 쓸 수 있게 한다("채점은 정답이지만 잘못 읽은 값이나 틀린 계산이 있는지 점검").
2. `observed_facts`는 0~3개다. 주석이 없는 문항은 0개다.
3. `lesson`은 **모든 문항에서 `{text}`**(40단어 이하의 추상 교훈)다. 주석 문항의 고정 공식 7개(`self_note_v16.FORMULAS`)만으로는 재풀이에 줄 정보가 없기 때문이다(감사 결함 6). 교훈은 학습하지 않고 재풀이 입력으로만 쓰며, 누설 검사를 통과해야 한다.
4. `error`가 없는 노트도 허용한다(`error: null`). 관측 값만 바로잡는 노트가 이 경우다.

카탈로그가 없는 문항의 `error` 필드 규칙(감사 결함 14):
- `kind = ARITHMETIC`만 가능하고, `rule_id ∈ {ADD, SUB, MUL, DIV}`, `operand_fact_refs = []`, `unit = ""`
- 피연산자는 인용문에서 읽는다
- `rounding`은 인용된 결과의 소수 자릿수로 정한다
- 반올림·버림으로 같아지는 값은 오류가 아니다(`ORIGINAL_CLAIM_ALREADY_CORRECT`)

- 입력 JSON 항목: `question`(P0), `first_solution`(T1 원문), `grading_result`, `given_answer`(정답), `note_catalog`.
  - `note_catalog`는 `public_note_catalog`이고 주석이 없으면 `{}`다. 사실의 값은 넣지 않는다.
- 출력은 JSON 하나다.
  - `error`의 필드: `earlier_span`, `earlier_quote`, `kind ∈ {OBSERVATION, ARITHMETIC}`, `operand_fact_refs`, `operation`, `old_value`, `corrected_value`, `rounding`, `unit`, `rule_id`
- 교훈에 문제 숫자·보기·이름·답을 넣지 않는다. 고쳐 쓴 풀이 전체를 쓰지 않는다.

### 6.3 검증 (`verify_note_v16b`, 새 함수)

주장마다 상태를 정한다: **VERIFIED / CONTRADICTED / UNVERIFIABLE**.

- `observed_facts[i]`
  - 카탈로그 사실과 값이 같으면 VERIFIED, 다르면 CONTRADICTED.
  - 카탈로그가 없으면 UNVERIFIABLE.
- `error` (`self_note_v16.verify_note`의 검사 재사용)
  - 인용이 T1에서 단언된 문장인지(`earlier_assertion_status`) 확인한다.
  - 인용 값이 `old_value`인지 확인한다.
  - 고친 값을 검사한다. LOOKUP이면 카탈로그 값, ARITHMETIC이면 `_calculate`·`_equation_operands`로 재계산한 값과 같아야 한다. 또 고친 값이 원래 값과 달라야 한다.
  - 카탈로그가 없으면 피연산자를 인용문에서 읽는 ARITHMETIC만 VERIFIED가 될 수 있다.
  - 재계산이 틀리면 CONTRADICTED, 그 밖은 UNVERIFIABLE.
- **최종 계산 교정 검사**(D2 수정): 인용 값이 T1의 최종답(마크다운·저장된 추출답 포함)과 같으면, 고친 값이 정답과 다를 때 `NOTE_GOLD_CONTRADICTION`이다.
- 교훈 누설 검사: `hint_leak_single_v2`. 교훈은 VERIFIED가 되지 않는다.
- **N 행 수용**: CONTRADICTED 0, 누설 없음, VERIFIED ≥ 1.
- **재풀이 진행**: 노트 형식이 맞고, 누설이 없고, CONTRADICTED가 0이면 재풀이를 한다(VERIFIED가 0이어도). 재풀이의 성공 여부가 따로 검사 역할을 한다.
- 영수증: `{version:"v16b-note-v1", problem_id, claims:[{path, status, char_span}], verified, contradicted, t1_generation_sha256, note_generation_sha256, lesson_sha256, sha256}`.

### 6.4 노트 학습 구간

- VERIFIED 주장 객체의 글자 구간만 학습한다. 그 안에서도 `earlier_quote`·`old_value` 값 문자열은 뺀다(틀린 내용을 따라 쓰게 되므로).
- 교훈과 UNVERIFIABLE 주장은 학습하지 않는다.
- 구간 → 토큰: `train_v16.token_byte_mask`(경계를 걸치는 토큰 제외)를 쓴다. JSON 글자 위치는 위치 인식 스캐너 `json_spans(text)`(새 함수)로 구한다.
- 노트가 학습에 기여한 양(검증 주장 수, 학습 토큰 수, 기울기 노름)은 RL과 따로 기록한다. 적으면 "노트 학습"이라고 보고하지 않는다.

### 6.5 재풀이 C

- **입력**: 그림 + 질문 + 검증을 통과한 노트의 교훈. 정답은 넣지 않는다.
  - 주석 문항: `self_note_v16.build_resolve_prompt`
  - 그 외: 텍스트 교훈용 프롬프트(새 상수 `RESOLVE_INSTRUCTION_V16B`)
- **C 행 수용**:
  - 같은 채점기로 CORRECT
  - 정상 종료, 반복 아님
  - `single._a2_check`의 교훈 복사·얼버무림 검사 통과
  - 추론–답 일치(`REASONING_ANSWER_MISMATCH` 아님)
  - 256~4,096토큰. 짧은 정답 학습이 풀이 단축을 부른 선례가 있어 v15의 256 하한을 유지한다.
- **학습**: 입력을 **교훈 없는 P0**로 바꾸어(v2 사용자 규칙) 재풀이 원문을 SFT한다.

---

## 7. 손실

### 7.1 학습 쪽 확률 (생성과 같은 정의)

새 함수 `rl.processed_full_logprobs(logits, actions, offset, *, temperature, presence_penalty)`를 둔다.

- 계산: `log_softmax((logits − presence·seen)/temperature)[action]`. `seen`은 그 위치 앞에서 생성된 토큰이다.
- 토큰마다 도는 파이썬 반복 없이 벡터 연산으로 한다(`D_code_plan.md` §3.1의 알고리즘).
- 시험: 참조 반복 구현과 오차 1e-5 이내(실측 최대 차 9.5e-7, 감사 `logprob_ntf_check.py`).
- 학습 표본으로 생성된 행(T1·OBS·NOTE·RESOLVE)은 이 확률을 쓴다. 생성 때 기록한 μ와 같은 정의다.
- PREP_SELF 행(B2 절단 표본으로 생성)도 손실은 이 전체 어휘 처리 확률로 계산한다. 생성 분포(절단)와는 다르므로 IS·재생 검사를 하지 않는다(감사 결함 10). 같은 토큰의 절단 기록값과 전체 값은 최대 0.154까지 다르다(감사 `vllm_sampler_check.py`).
- 감시 집합 행은 학습하지 않는다.
- 시험 허용 오차는 1e-5로 한다(실측 최대 차 9.5e-7).

### 7.2 첫 풀이 RL 항 (묶음 v)

```
L_RL = − (1 / (16 · 1024)) · Σ_i  A_i · c_i · Σ_t  w_t · m_t · log q_θ(a_t)
```

| 항 | 뜻 |
|---|---|
| `c_i` | `min(1, 2048 / n_i)`. n_i는 마스크 전 전체 행동 수다. 긴 행이 묶음을 지배하지 않게 한다 |
| `w_t` | 중요도 보정 `min(exp(log q_θ − log μ_t), 2)`, 기울기 차단(detach). 지연 0이면 엔진 수치 차이만 반영 |
| `m_t` (A_i > 0) | 모든 토큰 1 |
| `m_t` (A_i < 0) | 음의 토큰 거르기(NTF). 현재 정책 확률 `q_θ(a_t) > τ_neg`인 토큰을 손실에서 뺀다. `τ_neg = 0.5`. 분모(16·1024)는 그대로 둔다. 확신 있게 맞게 쓴 공통 부분까지 감점하지 않게 하고, 문항당 1개 RL의 붕괴를 막으려는 것이다 |

- **NTF 출처와 성격**(감사 결함 12): 근거 논문(arXiv 2606.17250, "Rethinking Groups in Critic-Free RLVR")은 음의 응답에서 **확률 순위 상위 비율**의 토큰을 가린다. 이 문서의 "절대 확률 0.5 문턱"은 그것을 바꾼 **[설계: 미검증 변형]**이다.
  - v15 오답 행에 적용하면 토큰의 약 88%가 가려진다(행당 중앙 636토큰 중 64개만 남음, 감사 `logprob_ntf_check.py`). 즉 음의 신호는 모델이 망설인 분기 토큰에만 간다.
  - 묶음마다 마스크 비율을 기록하고, 기대 범위(약 80~95%)를 벗어나면 통계에 경고를 남긴다.
- 상한 도달·반복·UNVERIFIABLE 행은 손실에 넣지 않는다(§5.1).
- HOLD가 나면 그 갱신의 **모든** 기울기(RL·OBS·SFT)를 버리고(`zero_grad`) optimizer step을 하지 않는다. 일정과 묶음 번호는 전진한다.
- **행 차단**: 음의 이득 행이면서 행 평균 |log q_θ − log μ| > 0.3이면 PG에서 뺀다(DeepSeek-V3.2식) [설계].
- **묶음 정지**: 묶음 평균 |log ρ| > 0.5이거나, w가 2에서 잘린 토큰 비율 > 10%이면 갱신 없이 HOLD한다.
  - 묶음 0(지연 0)에는 기존 재생 기준(`SINGLE_REPLAY_*`: 묶음 평균 ≤ 0.05)을 엔진 적용 확인으로 함께 쓴다.

### 7.3 관측 RL 항

- 위와 같은 식에 이득 `F1 − b_obs`를 쓰고 가중치 0.25를 곱한다.
- 분모는 `16·1024`로 같게 둔다.

### 7.4 SFT 항

| 종류 | 입력 → 목표 | 확률 | 가중 |
|---|---|---|---|
| PREP_SELF | P0 → F0의 정답 풀이 | 전체 어휘 처리 확률(T 0.7·presence 1.5). 생성은 B2 절단 표본이므로 생성 분포와 다름(§7.1) | 1.0, 묶음 평균 |
| PREP_HUMAN | P0 → 데이터셋 사람 풀이 | 일반 CE(사람 글이라 처리 분포 없음) | 1.0, 묶음 평균 |

- 경로: 그림이 있는 PREP 행(사람 풀이 포함)은 **FULL 경로**다(v15 규칙). 기존 `single.human_row`는 항상 LM 경로이므로 검사 조건만 가져오고, 행은 `v16b_rows.prep_rows`가 만든다(감사 결함 14).
| C | P0(교훈 없음) → 재풀이 원문 | 처리된 확률 | 0.35, 행마다 |
| N | 노트 프롬프트 → 노트의 검증 구간 | 처리된 확률, 마스크 | 0.05, 묶음 합 |

- 자기 생성 행에 일반 CE를 쓰면 정답 칭찬을 생성과 다른 확률로 하는 셈이 되어 분포가 한쪽으로 쏠린다. 그래서 자기 생성 행에는 생성 때와 같은 처리 확률을 쓴다(`L2_signal_fixes.md` D4).
- 각 행의 손실은 학습 토큰 평균이고, 끝 토큰은 학습하지 않는다.

### 7.5 원본 유지

- 전체 KL은 0이다(v15 실측 크기가 SFT 항의 1/1000).
- 첫 4토큰 고정만 둔다. F0 대비 KL × 0.5이고 v9 규칙이다. 단답 쏠림 방지가 목적이며, 참조 계산은 프롬프트 + 4토큰에서만 한다.
- **적용 범위와 분모**(감사 결함 16):
  - RL 갱신: 손실에 들어간 T1 RL 행 전부, 분모 16 고정
  - 준비 갱신: PREP 행 전부, 분모 = 그 갱신의 행 수. 단답 쏠림 위험이 가장 큰 구간이 준비 SFT이므로 여기에도 적용한다.
  - OBS·NOTE·C 행에는 적용하지 않는다.

### 7.6 경로와 묶음 처리

- 그림이 있는 행은 FULL 경로(Vision LoRA + Merger + LLM LoRA)이고, 그림이 없는 행은 LM 경로다(v15 규칙).
- `plan_microbatches(token_budget=12000)`로 같은 경로 행을 묶는다.
- 재생 검사는 따로 하지 않는다. 기울기 계산 때 μ와 비교한다(§7.2 기준, `D_code_plan.md` §3.3).

### 7.7 최적화

| 항목 | 값 |
|---|---|
| 시작 | 원본(F0), 새 LoRA(B=0), Merger 원본, 새 AdamW |
| 학습률 | L 5e-5 / M 1e-5 / V 1e-5 (v15 실측값. 낮추면 Merger 변화가 BF16에서 사라짐, 설계서 §24.6) |
| AdamW | betas (0.9, 0.95), eps 1e-8, weight decay 0, 그룹별 기울기 자르기 1.0 |
| 일정 | `linear_decay_after_warmup`, warmup 4. 전체 단계 = 준비 갱신 수 + 191(RL 188 + 마무리 3). 준비 행 수는 S2 뒤에 정해지므로, protocol·계획은 S2의 `prep-rows.json`이 봉인된 다음에 만든다. 그 뒤에는 바꾸지 않는다 |
| 갱신 조건 | 학습 신호가 있고 기울기가 있을 때만 optimizer step. 일정과 묶음 번호는 항상 전진(D5·A17 규칙) |

---

## 8. 준비 SFT

1. **준비 집합**: 자료 선정 산출물 `prep-manifest.json`의 512문항이다. 학습 3,000·감시 집합과 계열·그림이 겹치지 않는다(학습 코드가 겹침 0을 검사). RL 문항의 정답을 첫 풀이 전에 학습하지 않게 하려고 학습 3,000 밖에서 고른다(감사 결함 5, §2.1 "첫 풀이는 정답 없이").
2. **준비 풀이 생성(S2)**: 준비 집합 512문항을 F0로 1회 푼다. 엔진은 영(0) 조정층(= F0), B2 표본, P0, 8,192 상한을 쓴다. 같은 단계에서 감시 집합도 F0로 푼다(§9.4).
3. **행 만들기**
   - PREP_SELF: CORRECT, 정상 종료, 256~4,096토큰, 최종답 추출 가능, 답 부정 없음(`self_note_v16.first_solution_row`의 검사 조건만 가져옴. 그 함수는 `layer`·`subject`가 없는 문항을 `T1_SCHEMA`로 탈락시키므로 그대로 호출하지 않는다). 해시 순서로 최대 160행.
     - 참고: F0 기준 적격 비율은 약 18.6%라 512문항에서 약 95행이 예상된다(감사).
   - PREP_HUMAN: **준비 집합 512문항 중** 사람 풀이가 있고 **텍스트 오염 검사를 통과한** 문항. `references.human_solution`으로 읽고 `single.human_row`의 검사 조건(답 일치, 제어 토큰 없음)을 통과해야 하며 32토큰 이상이다. 해시 순서로 최대 128행.
   - 두 종류를 출처별로 섞어 16행씩 묶는다. 사람 풀이를 한 덩어리로 몰지 않는다(`C_literature.md` §1.3).
4. **갱신**: 묶음마다 SFT 항만으로 갱신한다(예상 약 11~14회 — 실제 행 수 ÷ 16, 올림). 같은 optimizer·일정을 RL과 이어서 쓴다. 준비가 끝나면 반드시 체크포인트를 저장한다.
5. **확인**: 준비 직후 감시 집합을 한 번 더 풀어(S3b) F0 대비 길이 중앙값·최종답 누락을 본다. 길이 중앙값이 F0의 0.5배 미만이면 RL로 넘어가지 않고 정지·보고한다.

---

## 9. 캠페인 실행

### 9.1 시간 처리 — 학습을 시간으로 자르지 않는다

- **학습 계획**: 188묶음(3,000문항) + 마무리 갱신을 모두 마친다. 학습률 일정도 이 전체 단계 수로 실행 전에 고정한다.
- **예상 완료 시각(ETA)**: RL 4묶음이 끝나면 평균 묶음 시간으로 남은 시간을 계산해 `eta.json`에 기록하고 사용자에게 알린다. **보고만 하며, 묶음 수·학습률·일정을 바꾸지 않는다.**
- **자금 보호(학습 단계에만 적용)**: 평가가 자금 부족으로 끊기는 것을 막는다(v9가 622/900에서 끝난 선례, 감사 결함 4). 시간을 미리 정해 자르는 장치가 아니다.
  - 로컬 감시(10분마다, 포드 밖)가 RunPod 잔액을 조회한다. `잔액 − (평가 4.63 + 내보내기·준비 0.22 + 감시 재측정 0.15 + 판정·회수 0.15 + 여유 0.5시간) × 시간당 가격 − 예약금 1.5달러`가 다음 1시간 학습 비용보다 작아지면 다음을 한다.
    1. 사용자에게 충전을 요청한다.
    2. 30분 안에 충전되지 않으면 `STOP_REQUESTED`를 써서 다음 저장 지점에서 학습을 멈춘다.
    3. 그 체크포인트로 평가할지, 충전 뒤 재개할지 사용자에게 묻는다.
  - 충전하면 이어서 학습한다. 저장·재개 규칙(§9.2) 덕분에 손실이 없다.
- **포드 수명 상한(규칙 §2.6)**: 평가를 끊지 않도록 학습 종료 후 단계는 상한에서 제외한다.
  - 포드 쪽 자동 정지는 다음 두 가지로 나눈다.
    - 학습: 자금 보호가 담당한다.
    - 평가: 시작 시점부터 6.5시간 한도. 유효한 900 평가는 이 안에서 끝나야 하며, 넘으면 회수 후 보고한다.
  - 자동 정지는 결과 회수 영수증 뒤에만 포드를 삭제한다.
- 2026-11-15 충전 뒤 잔액은 31.23달러이고, §11 중앙 추정(약 24달러)은 충분하다. 비관 추정(약 31달러)이면 자금 보호가 충전을 요청할 수 있다.
- **평가에 쓰는 체크포인트는 하나**다: 계획을 다 마친 마지막 체크포인트. 감시 정지가 일어나면 그 시점의 체크포인트(§9.3).

### 9.2 저장·재개

- 저장(체크포인트 디렉터리)은 8갱신마다, 그리고 준비 끝·감시 정지·자금 보호 정지·마지막에 한다. 그 사이 갱신은 거래 파일과 커서만 남기는 가벼운 커밋이다.
- **저장 지점은 비우는 지점이다**(감사 결함 2).
  - 저장할 갱신 v가 끝나면 새 판의 요청을 넣지 않는다. 진행 중 요청(판 v로 생성 중인 T1(v+1)·RESOLVE(v−2) 등)이 모두 끝날 때까지 엔진을 돌린다. 끝난 결과는 채점·검증해 봉인한다.
  - 그 뒤 저장한다. 저장물은 체크포인트 + 봉인된 미소비 행·결과 파일(+해시) + 기준값·감시 상태·판 등록부다.
  - 저장 뒤에 새 판의 요청을 넣는다.
  - 그래서 저장 시점에 진행 중인 요청이 없고(시험: `has_unfinished_requests() == False`), 재개할 때 다시 넣을 요청도 없다.
  - 비용: 저장당 약 40초(갱신당 약 +5초).
- 프로세스가 죽으면 마지막 저장 이후의 갱신 결과를 `rolled-back/`으로 옮기고, 저장된 체크포인트·커서·RNG·봉인 결과에서 다시 시작한다. 그 지점에서 넣어야 할 요청은 §4.3 일정이 정한 그대로 다시 넣는다(같은 판, 같은 seed).
- 요청 원장: v16b 런타임은 문항·요청 단위 원장(`requests.jsonl`, STARTED/COMPLETED)을 쓴다. 기존 `collect_v4.run_single_requests`의 "한 묶음·한 정책 해시" 원장 규칙(약 157~186행)은 연속 일정과 맞지 않으므로 재사용하지 않는다. 대신 다음 규칙을 따른다.
  - 비우는 지점 이전의 STARTED 요청은 반드시 COMPLETED여야 한다.
  - 마지막 저장 이후의 STARTED/COMPLETED는 되돌림 대상이다.
- 되돌림은 장애 때만 허용하고, 결과를 보고 되돌리지 않는다.
- 저장된 체크포인트는 최신 하나와 준비 끝 하나만 남긴다.

### 9.3 감시 (포드 안 감시 스크립트, v15 `guard_pod_v15.py` 기반)

어느 하나라도 걸리면 `STOP_REQUESTED`를 쓴다. 다음 묶음 경계에서 저장하고 멈추며, 그 체크포인트를 평가한다.

처음 판의 문턱은 정책이 그대로여도 잡음만으로 발동할 확률이 54~75%였다(감사 `guard_fa.py`: 저장된 F0 풀이를 16문항 묶음으로 2,000회 재표집). 그래서 감사가 조정한 "규칙 C"를 쓴다(거짓 정지 0.1%, 40묶음부터의 실제 변화 검출 97~99%, `guard_tune.py`).

| 규칙 | 조건 (모두 36묶음부터 검사) |
|---|---|
| ACC_DROP_SELF | 최근 12묶음 첫 풀이 정답률 < 처음 24묶음 정답률 − 0.15 |
| TRUNC_UP | 최근 12묶음 상한 도달 수 ≥ 2·ref 이고 ≥ ref + 4√ref (ref = 처음 24묶음 수 × 12/24) |
| NOFINAL_DRIFT | 최근 12묶음 최종답 없음 수 ≥ 2·ref 이고 ≥ ref + 4√ref (ref 같은 방식) |
| COLLAPSE_LEN | 6묶음 창의 첫 풀이 길이 중앙값 < 같은 문항 F0 사전 길이 중앙값의 0.5배가 **겹치지 않는 창 2개 연속**(사전값 있는 문항 기준, 12묶음부터) |
| COLLAPSE_SHORT | 최근 2묶음에서 32토큰 미만 첫 풀이가 각각 12/16 이상(바로 검사) |
| TAIL_RATE | 학습 표본 중 평가 후보(top-20 ∩ top-p 0.8) 밖 토큰 비율 > 처음 4묶음 평균의 2배가 2묶음 연속 |
| IS_DRIFT | 갱신 HOLD(§7.2)가 연속 2번 |
| ORDER_MISMATCH | 묶음 문항 ≠ 계획(바로 검사) |

- **사전 확인 영수증**: 실행 전에 동결된 3,000문항의 `f0_prior`(사전값이 있는 문항)로 같은 재표집을 돌린다. 각 규칙과 전체의 거짓 정지율이 2% 이하임을 `guard-falsealarm.json`으로 남긴다. 넘으면 문턱을 고치지 말고 보고한다(감사 스크립트 `guard_tune.py`를 기준으로 쓴다).

시간 자체는 감시 규칙이 아니다(§9.1). 준비 집합은 학습 3,000 밖이므로 정답률 계산에 섞이지 않는다(§8).

이 문턱은 실행 전 사전등록하고, 실행 중에 바꾸지 않는다.

### 9.4 감시 집합

- S2에서 F0로 한 번, S5에서 최종 체크포인트로 한 번 푼다. 둘 다 B2 설정, P0, 8,192다.
- 정답률·길이·최종답 누락·상한 도달의 전후를 보고한다. 학습에는 쓰지 않는다.
- 이것은 학습 분포 안의 진단이다. MMMU 효과 증거가 아니다.

---

## 10. 평가

1. 최종 체크포인트를 `proof_eval export`로 **비병합 LoRA 형식**(`FORMAT_LORA`)으로 내보낸다. 병합하면 변화의 24~55%가 사라진다.
2. 사전등록 `notebook-rl-proof-prereg-v11`(새 함수 `build_preregistration_v11`)을 만든다.
   - 기반: `build_preregistration_v8` + v10 `model_application`
   - 실행 하나: FT-SINGLE, 변형 `mmmu-val-v8-b2-seed3407-v1`, subset `all`(900), seed 3407
   - 두 단계 형식(`TWO_STAGE`)에 넣지 않는다.
   - 표시용 문턱: 559(62%+1), 목표 585
3. `run_eval`로 B2·P0·eager·비병합·`MMDL_EVAL_SLOTS=auto`(A100 → 32)를 돌린다. 평가 한도는 6.5시간이며, 유효한 900 평가는 시간 때문에 끊지 않는다.
4. V8로 채점한다. 주 점수다.
5. CompassVerifier-7B(BF16, 공식 모드, V8 오답→정답 방향만)를 돌린다. 보완 점수로 따로 보고한다(v15의 `cv_stage_v15.sh` 방식).
6. 원본 B2 558과 문항별로 비교한다: 회복·손실, 정확 이항 검정, 종료 상태별 분해, 길이·상한 도달. 원본 재실행 잡음(1회 SD 약 7.7문항)을 함께 적는다.
7. 결과 회수·해시 확인 → 포드 삭제 → 대여기 0대 확인.

---

## 11. 시간·비용 (1회 풀이, 3,000문항, A100 SXM 1.39달러/시)

### 11.1 RL 묶음 하나(16문항)의 작업량 [A, 실측 분포 기반]

| 요청 | 개수 | 평균 토큰 | 토큰 |
|---|---:|---:|---:|
| T1 | 16 | 약 900 (F0 3,500 풀 출처 가중 876 [M-d], 전체 분포 표본으로 길어질 수 있음) | 약 14,400 |
| OBS | 약 3 (주석 문항 약 15~20%) | 약 250 | 약 750 |
| NOTE | 약 6.5 (F0 오답률 약 40%) | 약 300 (v15 노트 [M]) | 약 1,950 |
| RESOLVE | 약 5 (노트 통과분) | 약 2,400 (v15 재풀이 평균 [M]) | 약 12,000 |
| 합계 | | | **약 29,000** |

### 11.2 묶음 시간

| 경우 | 생성(꼬리 단계 수 × 단계 시간 + 입력 준비) | 갱신 | 저장·비우기(8갱신마다, 갱신당 환산) | 합 |
|---|---:|---:|---:|---:|
| 낙관: 꼬리 약 4,000단계 + 입력 준비 메모, 갱신 개선 성공 | 약 68초 | 약 55초 | 약 10초 | **약 133초** |
| 중앙: 입력 준비 일부 메모 + 일정 대기, 갱신 일부 개선 | 약 90초 | 약 100초 | 약 10초 | **약 200초** |
| 비관: v15 수준 CPU 부담, 갱신 개선 없음(27.3초 + 6.47초/1,000토큰 × 약 20,000) | 약 120초 | 약 157초 | 약 10초 | **약 287초** |

- 생성은 처리량보다 **꼬리에 묶인다**: 묶음마다 가장 긴 풀이(최대 8,192토큰)를 기다리며, 평균 활성 서열은 7.7개다(감사 `sched_sim.py`). 그래서 동시 서열을 늘려도 빨라지지 않는다. 바닥은 생성 약 54초 + 입력 준비 약 14초다.
- 저장·비우기: 저장 5.5초 + 비우기 약 40초(§9.2)를 8갱신에 나눈 값에 판 교체(조정층 쓰기·적재·판 일치 검사) 약 4초를 더했다.

- 갱신 행: RL 약 15 + OBS 약 3 + SFT 약 4 ≈ 22행, 학습 토큰 약 20,000.
- 갱신 개선: 벡터화된 확률(§7.1), 재생 통합(§7.6), KL 참조 제거(§7.5). 효과는 측정 전 가정이다.

### 11.3 3,000문항 전부의 예상 시간·비용

| | 낙관 | 중앙 | 비관 |
|---|---:|---:|---:|
| 고정 단계(S0~S3b, S5~S8, 여유) | 약 6.6시간 | 약 7.0시간 | 약 7.4시간 |
| RL 188묶음 + 마무리 갱신 3회 | 약 7.0시간 | 약 10.6시간 | 약 15.2시간 |
| **포드 전체** | **약 13.6시간** | **약 17.6시간** | **약 22.6시간** |
| 비용(1.39달러/시) | 약 19달러 | 약 25달러 | 약 32달러 |

- 묶음 시간은 §11.2의 추정이다. 실제는 첫 4묶음 뒤 ETA로 알린다.
- 학습 쪽 속도 개선(입력 준비 메모, 확률 벡터화, 재생 통합)이 중앙과 비관의 차이를 만든다. Codex 구현 단계에서 CPU 측정으로 확인한다.
- **잔액**: 2026-11-15 충전 뒤 조회 잔액 31.23달러. 중앙 추정(약 25달러)은 충분하고, 비관 추정(약 32달러)이면 학습 후반에 자금 보호(§9.1)가 충전을 요청한다. 실행 직전에 실제 잔액을 다시 조회하고, 첫 4묶음 뒤 ETA와 예상 비용을 알린다.
- **더 빠른 GPU**(H100 등)는 허용 목록(§2.6) 밖이라 사용자 결정이 필요하다. 이 코드로 잰 배율도 없다.
- **두 포드로 평가를 나누는 방법**은 학습된 조정층을 다른 포드로 올려야 해서 "로컬 모델 업로드 금지" 규칙과 충돌한다. 쓰지 않는다.

---

## 12. 파일별 구현 명세 (Codex)

기존 함수 이름 기준이다. 줄 번호는 2026-11-15 작업 트리 기준이므로 구현 때는 이름으로 찾는다. 과거 판의 경로·상수·해시는 바꾸지 않고, v16b 분기로만 추가한다.

### 12.1 `fine_tuning/error_note/protocol.py` (뒤에 추가만)

```python
VERSION_V16B = "error-note-v16b"
TRAIN_SAMPLING_V16B = {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "presence_penalty": 1.5}
EVAL_SAMPLING_V16B = dict(SINGLE_SAMPLING)
PROBABILITY_SPACE_V16B = "processed-full-vocab-v1"
MAX_TOKENS_V16B = {"t1": 8192, "obs": 1024, "note": 1024, "resolve": 8192, "prep": 8192, "monitor": 8192}
REWARD_V16B = {"CORRECT": 1.0, "INCORRECT": 0.0, "NO_FINAL": 0.0, "MULTIPLE": 0.0,
               "CAP_OR_LOOP": {"reward": 0.0, "pg": False}, "UNVERIFIABLE": None}
BASELINE_V16B = {"rule": "shrunk-source-mean-v1", "a": 2, "window_rows": 64, "obs_window": 32, "obs_init": 0.5}
PG_V16B = {"token_scale": 1024, "items_per_block": 16, "row_cap": 2048, "obs_weight": 0.25,
           "ntf_tau_neg": 0.5, "is_cap": 2.0, "neg_row_logratio_mask": 0.3,
           "hold_block_logratio": 0.5, "hold_clip_frac": 0.10}
SCHEDULE_V16B = {"eta": 1, "note_carry": 1, "resolve_carry": 3,     # update v: T1/OBS of v, N of v-1, C of v-3
                 "rl_blocks": 188, "last_block_items": 8, "finish_updates_max": 3,
                 "time_cut": None}                                 # no time-based training cut (user 2026-11-15)
PREP_V16B = {"items": 512, "select": "prep-manifest.json (curation, outside the 3,000)", "self_max": 160,
             "human_max": 128, "self_tokens": [256, 4096], "human_min_tokens": 32, "overlap_with_rl": False}
GUARD_V16B = {"start_block": 36, "ref_blocks": 24, "recent_blocks": 12, "acc_drop": 0.15,
              "count_rule": "recent >= 2*ref and recent >= ref + 4*sqrt(ref)",
              "collapse_len": {"window": 6, "ratio": 0.5, "consecutive_windows": 2, "start_block": 12},
              "false_alarm_max": 0.02}
FUNDING_V16B = {"reserve_usd": 1.5, "post_train_hours": 5.65, "recharge_wait_min": 30}
SFT_V16B = {"PREP_SELF": 1.0, "PREP_HUMAN": 1.0, "C": 0.35, "N": 0.05}
ANCHOR_V16B = {"positions": 4, "beta": 0.5}
CHECKPOINT_V16B = {"every_updates": 8, "always_on": ["prep_end", "deadline", "guard_stop", "final"]}

def plain_protocol_v16b(precision: str, *, rl_blocks: int, prep_updates: int,
                        contracts: dict[str, str], image: dict) -> dict: ...
def validate_protocol_v16b(protocol: dict) -> None: ...   # rebuild-and-compare
def v16b_prompts_sha256() -> str: ...                      # NOTE/RESOLVE/OBS prompt constants
```

- 본문은 `plain_protocol_v2` + 위 상수다. 그 밖에 다음을 넣는다.
  - `image = SINGLE_IMAGE_V14`(B2 예산), `rollout_adapter = ROLLOUT_ADAPTER_V15`
  - `optimizer` (§7.7), `initial_policy = "fresh_F0"`
  - `answer_contracts = SINGLE_ANSWER_CONTRACTS_V14`, `routes {"image":"FULL","text":"LM"}`
  - `hashes` (`source`, `dataset`, `order`, `prep_rows`, `monitor`)
- `max_num_seqs`·KV는 protocol 해시가 아니라 계획의 capacity에 둔다.

### 12.2 `fine_tuning/error_note/contract.py`

- `sample_seed`에 `stage.startswith("v16b-")` 분기를 추가한다. 키는 `f"{master_seed}:{problem_id}:{stage}"`다.

### 12.3 `fine_tuning/error_note/rl.py` (뒤에 추가)

- `processed_full_logprobs(...)` (§7.1)
- `negative_token_mask(logq: Tensor, tau: float) -> Tensor`: `logq.exp() <= tau`

### 12.4 `fine_tuning/error_note/single_vllm.py`

- `SingleVllmSampler.__init__`에 키워드 `sampling=None`, `version_registry=False`를 추가한다. 기본값이면 v15 동작이 그대로여야 한다.
- 전체 분포 모드: `logprobs=0`, 새 함수 `sampled_logprob(token, row)`로 μ를 기록한다. 지지 집합 검사를 하지 않는다.
- 판 등록부 `{version: (LoRARequest, refcount)}`:
  - `add_version(bundle, version)`: LoRA 쓰기 + `add_lora` + Merger·훅 in-flight push
  - `release_version(version)`: 참조 0일 때 `remove_lora`·경로 삭제
- `_LoraEngine.add_request`가 요청의 `policy_version`에 맞는 LoRARequest를 붙인다.
- `finish`에 §4.5의 필드를 추가하고, `policy_sha256`은 입장 시 값을 쓴다.
- 문항별 입력 준비 메모(입력 해시 일치 확인).

### 12.5 `fine_tuning/error_note/llm_lora.py`

- `ENGINE_ARGS_V16B = ENGINE_ARGS | {"max_loras": 3, "max_cpu_loras": 3}`. 기존 상수는 그대로 둔다.
- `free_lora_slots(engine) -> int`(새 함수): `add_lora` 전에 확인한다.
- `check_all_live_identity(engine, bundle_versions) -> dict`(새 함수): 푸시마다 살아 있는 모든 판의 슬롯·Merger 비트 일치를 검사한다(§4.4).
- `push_mergers(..., inflight: bool = False)`: §4.4. `VisionHooks.load`도 in-flight 호출을 허용한다(단계 사이에만).

### 12.6 새 파일 `fine_tuning/error_note/v16b_runtime.py` (연속 일정의 단일 소유자)

```python
class V16bScheduler:
    def __init__(self, sampler, *, plan, protocol, items, grade, verify, out: Path, local_sequential: bool = False): ...
    def submit_for_version(self, v: int) -> None: ...        # §4.3 rules 1-3
    def ready_for_update(self, v: int) -> bool: ...          # T1/OBS of v, NOTE of v-1, RESOLVE of v-3 all closed
    def step(self) -> list[dict]: ...                        # one engine.step(); route finished outputs to §4.6
    def rows_for_update(self, v: int) -> list[dict]: ...      # RL(v), OBS(v), N(v-1), C(v-3); sealed rows file
def run_v16b(bundle, optimizer, scheduler_lr, sampler, plan, *, out: Path) -> dict: ...   # no deadline argument
```

- 루프:
  1. `ready_for_update(v)`가 아니면 `step()`을 반복한다.
  2. 준비되면 행을 봉인한다(`rows-{v}.json`, 해시).
  3. `run_joint_single_update`(§7 손실, v16b 분기)로 갱신한다.
  4. `add_version(v+1)` 뒤 `submit_for_version(v+1)`을 한다.
  5. 저장 규칙(비우는 지점, §9.2)을 적용하고, 정지 요청 파일(`STOP_REQUESTED`: 감시 규칙 또는 자금 보호)을 확인한다.
  6. RL 188묶음 뒤에는 새 첫 풀이 없이 남은 노트·재풀이를 끝내고 마무리 갱신(최대 3회)을 한다.
- 문항 상태 파일과 요청 원장은 `collect_v4.run_single_requests`의 STARTED/COMPLETED 원장 규칙을 따른다. 재개 때 같은 요청을 새 seed로 만들지 않는다.
- `local_sequential=True`면 HF 생성기로 같은 일정을 차례로 실행한다(로컬 작동 확인 전용).

### 12.7 새 파일 `fine_tuning/error_note/v16b_rows.py` (순수 함수)

- `reward_v16b(status, finish_reason, text, extracted) -> (float | None, str, bool pg)`
- `baseline_v16b(item, history, *, prior) -> float`
- `obs_score(item, layer, text) -> dict`
- `t1_error_scan(text, catalog, rules) -> list[dict]`
- `NOTE_INSTRUCTION_V16B`, `RESOLVE_INSTRUCTION_V16B`, `OBS_FORMAT_V16B`
- `build_note_prompt_v16b(...)`
- `verify_note_v16b(...)` (§6.3)
- `json_spans(text)`
- `note_row(...)`, `c_row(...)`, `prep_rows(...)`
- 노트·재풀이 검증은 `self_note_v16`의 T2 수정 함수(`earlier_assertion_status`, `_calculate`, `_equation_operands`, 최종답 교정 검사)를 재사용한다. 그 파일은 바꾸지 않는다.

### 12.8 새 파일 `fine_tuning/error_note/train_v16b.py`

- `v16b_row_logprobs(bundle, rows, *, chunk=32) -> list[Tensor]`: `train_v4._single_hidden` + `processed_full_logprobs`
- `backward_v16b_block(bundle, encoded, contract, token_budget=12000) -> dict`: §7 전체. optimizer는 만들지 않는다.
  - 반환: `task_signal`, `has_gradient`, 통계(RL·OBS·SFT 종류별 행·토큰·손실, 이득 분포, NTF로 뺀 비율, IS 통계, 앵커, 재생)
- `_validate_v16b_block(...)`
  - RL: 묶음 v 문항당 정확히 1행
  - OBS: 주석 문항당 1행
  - N: 묶음 v−1 문항, C: 묶음 v−3 문항, 각 최대 1행
  - 행의 `policy_version`이 일정과 일치
  - 이득을 봉인된 보상·기준값으로 다시 계산해 정확히 일치
- `train_v4.backward_single_block` 첫머리에 `if contract.get("version") == VERSION_V16B: return backward_v16b_block(...)` 분기 하나를 둔다.

### 12.9 새 파일 `fine_tuning/error_note/prep_v16b.py`

- `select`: `prep-manifest.json` 해시 확인과 학습 3,000·감시 집합과의 겹침 0 검사(§8, `PREP_V16B`)
- `generate`: 준비 집합 + 감시 집합 F0 풀이(B2 표본, `v16b-prep`/`v16b-monitor` seed)
- `rows`: PREP_SELF/PREP_HUMAN 봉인(`prep-rows.json` + 해시)
- `eta_report(block_seconds, *, done, total) -> dict`: 기록·알림 전용, 계획을 바꾸지 않음(§9.1)

### 12.10 `fine_tuning/error_note/single_campaign.py`

- `CAMPAIGN_V16B`, `_prepare_v16b(args)`, `_plan_check_v16b(plan)`, `run_campaign`의 v16b 분기를 둔다.
- 입력: 자료 선정 산출물 3개 + vault 경로 + 출처 증거·학습 규칙
- 계획 본문:
  - 묶음: 준비 갱신 묶음 + RL 188묶음(마지막 8문항) + 마무리 갱신 최대 3
  - 저장 규칙, capacity(`max_num_seqs`, KV), 감시 문턱
  - 마감 시각 없음. 자금 보호는 로컬 감시, 평가 6.5시간 한도는 포드 쪽 자동 정지로 둔다(계획 해시와 분리)
- v16b에서 우회할 기존 가정: 3,500/219 강제, 19/300 시험 관문, 슬롯 19 보존, `max_num_seqs ≤ 16`, "묶음 행은 현재 정책" 검사(→ 판 기록 검사), A1/A2/B 종류, 노트 32개 수율 관문(v16 `create_note_yield_gate`는 쓰지 않음).
- 4비트 학습 거부는 `--local-sequential` 작동 확인 계획에서만 푼다(포드 job은 이 계획을 거부).

### 12.11 `fine_tuning/notebook_rl/proof_eval.py`

- `_single_versions`에 `VERSION_V16B`를 추가한다.
- `_single_stage`에서 v16b는 `final` 또는 최신 `stopped`다.
- `build_preregistration_v11`, `FORMAT_V11`(FINAL_VARIANT에 추가, TWO_STAGE 제외)을 둔다.
- `_single_definition`에 v16b 문장을 넣는다(준비 갱신 수, RL 묶음 수, 1회 풀이, 연속 일정).

### 12.12 `fine_tuning/error_note/plan.py`와 포드 스크립트

- 단계 `v16b`: pins → 자료 묶음 stage/verify → `prep_v16b select/generate/rows` → `single_campaign prepare` → `advance`(전체) → `reload` → 감시 재측정 → `export` → `prereg v11` → `run_eval` 900 → V8 → CompassVerifier → `pod.collect pack`.
- 감시 스크립트 `guard_pod_v16b.py`(저장소 밖, v15 기반)에는 §9.3 규칙과 자체 시험을 둔다.
- 런포드 생성 스크립트는 v15 `create_retry_v15.py` 방식을 따른다: 커뮤니티 A100 80GB **SXM만**(허용 목록 §2.6. PCIe 등 목록 밖 GPU가 필요하면 멈추고 사용자에게 묻는다), 가격 출력, 평가 단계 자동 정지(§9.1), 로컬 자금 보호 감시(§9.1).

---

## 13. 시험 명세

### 13.1 CPU 시험 (새 파일, 합성 자료만, MMMU 없음)

| 파일 | 시험 |
|---|---|
| `test_error_note_v16b_protocol.py` | 과거 해시 7개 불변(v11 `b98882d2…`, v12 `46269db2…`, v13 `34121b80…`, v14 `4091a611…`, v15 `4b5ed9fb…`, content-v16 `00e3cee4…`, v2 `b5c949ed…`, 전체 값은 `D_code_plan.md` §14.1), v15 포드 protocol 재구성 `542e9806…`, v16b 변조 거부, seed 단계 새 값·옛 값 불변 |
| `test_error_note_v16b_logprob.py` | `processed_full_logprobs` = 참조 반복(오차 1e-5, 덩어리 경계 0·13·32), `negative_token_mask`, `sampled_logprob`가 단일 유한 토큰 요구 |
| `test_error_note_v16b_guard.py` | §9.3 규칙 C의 문턱 계산(ref 환산, 36묶음 시작, COLLAPSE_LEN 겹치지 않는 창 2개 연속), 동결 풀 `f0_prior` 재표집 영수증 형식 |
| `test_error_note_v16b_resume.py` | 저장 지점에서 `has_unfinished_requests() == False`, 봉인된 미소비 행 해시, 되돌림 뒤 같은 판·같은 seed로 다시 넣기, 준비 집합 ∩ 학습 3,000 = ∅ |
| `test_error_note_v16b_rows.py` | 보상 표, 상한 도달 행 PG 제외, 기준값이 현재 묶음 보상을 쓰지 않음, 관측 F1(정답 1·누락·추가·장면 순서·읽기 실패 0), `t1_error_scan`, 노트 검증(맞은 T1의 틀린 산술 → 노트, 모순 사실 → 거부, UNVERIFIABLE 마스크, 인용·old_value 미학습, 최종 계산을 정답 아닌 값으로 고치면 거부), `json_spans` 유니코드, 교훈 누설 → 재풀이 없음, C 수용(정답·256 하한·P0 입력) |
| `test_error_note_v16b_loss.py` | 작은 무작위 Qwen3-VL CPU 정책: RL 항 = 수동 계산, NTF 위치 기울기 0·분모 불변, IS 가중 상한 2, 음의 행 차단, 묶음 HOLD 시 기울기 0, 마이크로배치 분할 불변, 앵커 4토큰, 갱신 조건(신호·기울기) |
| `test_error_note_v16b_runtime.py` | 가짜 엔진: §4.3 일정(판 v에서 넣는 요청 정확), `ready_for_update` 조건, 행 묶음 결정성(완료 순서를 섞어도 같은 행), 판 등록부(두 판 공존, 참조 0일 때만 제거, 살아 있는 판 2개 초과 시 오류), 슬롯이 다 차면 `add_lora`를 하지 않음, Merger in-flight push가 캐시를 지우지 않음, 출력 토큰 없는 요청의 혼합 표시, 선점 표시, 갱신 중 결정성 켜짐·단계 중 꺼짐, 재개 시 STARTED 요청 새 seed 금지 |
| `test_error_note_v16b_campaign.py` | 준비 집합 선택(출처 비율·해시), 준비 → RL 전환, 마지막 8문항 묶음과 마무리 갱신, 학습률 일정이 실행 중 바뀌지 않음, `eta.json`이 계획을 바꾸지 않음, 8갱신 저장·되돌림, `STOP_REQUESTED`, `WATCHDOG_STOP` 시 평가 자동 시작 안 함 |
| `test_error_note_v16b_eval.py` | v16b 내보내기 = 비병합, prereg v11 900 1회, 보완 점수 별도 |

- 기존 시험(`test_error_note_*`, `test_notebook_rl_proof_eval*`, `test_notebook_rl_pod*`, `test_notebook_rl_checkpoint.py`)이 그대로 통과해야 한다.
- 명령 앞에는 `CUDA_VISIBLE_DEVICES=""`를 붙인다.
- Ruff와 범위 mypy는 설치된 실제 경로를 쓴다(notebook-nf4 환경에는 없다).

### 13.2 로컬 4비트 작동 확인 (30분 이내, 장기 실행 금지)

- `--local-sequential`, NF4 캐시, 2문항 × **4** RL 묶음 + 준비 1갱신, 상한 256, `SMOKE_IMAGE`. RESOLVE(b)는 판 b+2에 들어가 갱신 b+3에 쓰이므로, 4묶음이어야 재풀이 생성과 C 행 후보가 실제로 나온다(감사 결함 13). 마지막 갱신에 C 행 후보가 들어왔는지 영수증에 남긴다.
- **실제 vLLM 다중 판 확인(가능하면)**: 로컬 8GB GPU에서 시험용 초소형 무작위 Qwen3-VL을 vLLM 0.11에 올려, 두 판 공존·참조 0 제거·진행 중 Merger push·요청별 LoRA·슬롯이 찬 상태의 `add_lora` 거부를 10분 안에 확인한다. 안 되면 "미실행"으로 적고, 포드 첫 갱신 1·2의 두 판 공존 확인(§17)으로 대신한다(감사 결함 18).
- 확인할 것:
  - 보상, 기준값, 노트 생성·검증, 재풀이
  - 행 봉인, 갱신 2회(L/M/V 변경), 저장
  - 새 프로세스 재로딩 일치(상태·탐침 차 0)
- 이 확인은 기능 확인일 뿐이다. 4비트 결과는 BF16 근거가 아니다.

### 13.3 포드 첫 구간 (본 실행 안에서, 별도 시험 포드 없음)

- PF0~PF5(v15 증폭 탐침 포함) 통과.
- **두 판 공존 확인**: 갱신 1·2 직후, 한 배치에 옛 판·새 판 요청이 섞인 상태에서 각 판의 고정 탐침 출력이 그 판 단독 실행과 같은지 확인한다(`E_verify.md`: GPU에서만 확인 가능한 항목).
- 묶음 0(지연 0)에서 재생 기준 통과.
- 갱신 최대 메모리 X를 측정하고 KV·동시 서열을 확정한다. 이 값은 첫 4묶음 안에서만 조정할 수 있고, 바꾸면 `capacity-change.json`에 기록한다(정답률을 보고 바꾸지 않음).
- 묶음 1~4에서 IS 통계(지연 1의 |log ρ|, 잘린 비율), GPU 유휴 비율, 묶음 시간을 잰다 → `eta.json` 기록과 알림(계획은 바꾸지 않음).
- 어느 하나라도 실패하면 정지 → 회수 → 삭제 → 보고. 자동 재시도는 하지 않는다.

---

## 14. 작업 순서 (Codex, 의존 단계별)

| 단계 | 작업 | 파일 | 완료 조건 |
|---|---|---|---|
| 1 | protocol·seed | protocol.py, contract.py, single.py(`_protocol_check` 분기), 시험 | §13.1 protocol 시험 |
| 1 | 순수 행 함수 | v16b_rows.py, 시험 | rows 시험 |
| 2 | 확률·표본기 | rl.py, single_vllm.py, llm_lora.py, collect.py(HF 표본기 전체 분포 분기), 시험 | logprob 시험, v15 vLLM 시험 그대로 통과 |
| 2 | 손실 | train_v16b.py, train_v4.py(분기), 시험 | loss 시험 |
| 3 | 연속 일정 | v16b_runtime.py, 시험 | runtime 시험 |
| 3 | 준비·ETA | prep_v16b.py, 시험 | prep·ETA 시험 |
| 4 | 캠페인 | single_campaign.py, 시험 | campaign 시험 |
| 4 | 평가·계획 | proof_eval.py, plan.py, 시험 | eval 시험, plan 단계 목록 |
| 5 | 로컬 작동 확인 | — | §13.2 영수증 |
| 6 | 게시·실행 | MMDL_test 동기화, 포드 스크립트 | 사용자 결정 12번 범위 |

- 위험도는 높다(RL 목적함수, 연속 일정, 저장·되돌림). 단계마다 Opus 검토를 받고, 통합 diff는 포드 실행 전에 Fable 감사를 받는다.
- 동결 소스: 로컬 확인과 포드 실행은 같은 소스 해시여야 한다. 확인 뒤 코드를 바꾸면 로컬 확인을 다시 한다.

---

## 15. 기록과 보고

- **`update-stats.json`**(묶음마다)
  - 묶음 번호, 판, 지연
  - RL 행·토큰·학습 토큰, 이득 분포(양·음), NTF로 뺀 비율, IS 통계, 행 차단 수, HOLD 여부
  - OBS 점수 평균·이득
  - 노트: 생성 수 / 검증 통과 수 / VERIFIED 주장 수 / 학습 토큰 / N 기울기 노름
  - 재풀이: 수 / 정답 / C 수용 / 탈락 사유
  - 준비 행, 기준값 m_s, 첫 풀이 정답률·길이·상한 도달·최종답 없음
  - 묶음 시간(생성·갱신·저장·유휴), 메모리
- **`docs/PROJECT_STATUS.md`**: 구현·확인·실행·결과를 날짜별로 기록한다. 측정값과 가정을 구분한다.
- **최종 보고**에 반드시 넣을 것
  - V8 900 점수, 보완 점수, B2 대비 회복·손실·p값
  - 학습한 RL 문항 수(계획 3,000, 감시 정지 시 실제 수), 준비 행 수
  - 노트·재풀이가 실제 학습에 기여한 양
  - 감시 집합 전후(F0 → 준비 SFT 직후 → 최종)
  - 검증된 노트 주장 수와 유형(관측 값 / 산술 오류). 검증된 `error` 주장이 30개 미만이면 "오답노트 교정 학습은 사실상 없음(관측 값 위주)"이라고 쓴다(§6)
  - 8,192 상한에 걸린 풀이 중 반복이 아닌 비율(2026-11-02 규칙)
  - 혼합 판 요청 수, HOLD 수, NTF 마스크 비율 분포
  - 비용·시간
  - 한계: 1회 평가 잡음, 학습 분포 안 진단은 MMMU 증거가 아님, AI 검토는 사람 검증이 아님

---

## 16. 위험과 정직한 기대

1. **묶음 시간이 측정되지 않았다.** 전체 시간이 약 13.6~22.6시간 사이에서 갈린다. 대응: 4묶음 뒤 ETA 보고, 자금 보호(§9.1).
2. **연속 일정과 판 관리는 처음 만드는 고위험 코드다**(vLLM 다중 LoRA·in-flight push). 대응: CPU 가짜 엔진 시험, 포드 첫 구간 확인, 실패 시 정지.
3. **갱신 메모리 X를 모른다.** 크면 KV가 줄어 동시 서열이 줄고 느려진다.
4. **문항당 1회 RL은 신호가 약하다.** 기준값 품질에 좌우된다. 사전값이 있는 문항은 약 절반이다(`A_data.md`: 4,000 중 1,833).
5. **노트 교정 학습은 거의 없을 가능성이 크다.** 저장된 F0 풀이에서 재계산이 틀린 단언 등식은 오답 1,084개 중 1개, 정답 2,050개 중 0개였다(감사). 검증 가능한 주장은 주로 주석 문항의 관측 값이다. 실측 수율을 보고하고, 적으면 "노트 학습"이라고 말하지 않는다.
9. **감시 규칙의 잡음 정지.** 처음 문턱은 잡음만으로 54~75% 확률로 학습을 끊었다(감사). 규칙 C로 바꿔 거짓 정지를 약 0.1%로 낮췄지만, 새 3,000 풀에서는 사전 확인 영수증(§9.3)으로 다시 확인한다.
6. **준비 SFT가 출력을 짧게 만들 수 있다**(v1 968→413토큰, v12 사람 풀이 행 522점의 선례). 대응: 256토큰 하한, 행 수 제한, 길이 감시.
7. **전체 분포 학습과 B2 평가의 차이**에 대한 통제 연구는 없다. TAIL_RATE로 감시한다.
8. **효과.** 문헌에도 프로젝트 기록에도, 이 규모의 학습으로 MMMU가 +27문항(원본 재실행과 구별되는 크기) 오른 근거는 없다. 585를 관측하기 전에는 65점 달성이라고 하지 않는다.

---

## 17. 실행 전 확인 사항

1. **런포드 잔액**: 2026-11-15 충전 뒤 31.23달러. 실행 직전에 잔액과 A100 SXM 가격을 다시 조회하고 사용자에게 알린다.
2. **자료 선정 산출물 동결 확인**: 3,000, 순서, 감시 집합, **준비 집합 512(학습 3,000 밖)**, 조립된 단일 vault, 사람 풀이 텍스트 오염 검사 완료.
2b. **감시 규칙 거짓 정지 영수증**(§9.3, ≤ 2%).
3. **v16b 한정 규칙 변경을 `CLAUDE.md`·`AGENTS.md` §2.1에 기록**(§1.2).
4. 위 조건이 갖춰지면 사용자 결정 12번 범위 안에서 게시·실행을 진행한다. 다음 상황에서만 멈추고 묻는다.
   - 자금 보호가 충전을 요청할 때
   - 감시 규칙 또는 자금 보호로 학습이 멈췄을 때(평가 여부를 묻는다)
   - 포드 첫 구간 실패
   - 허용 목록 밖 GPU가 필요할 때

---

## 18. 실행 전 구현 검토로 확정한 변경 (2026-11-16)

Codex 구현을 독립 검토(Opus 4갈래 + Fable 반박 검증, 확인 13건·반박 0건)하고 고쳤다. 결과물: 검토 `scratchpad/v16b-implreview-result.json`, 수정 5작업(Opus 3·Sonnet 2) 모두 첫 시도 통과, Fable 최종 감사 PASS(CPU 범위). 아래는 이 문서의 글자와 달라진 점이다. 학습되는 내용의 검증을 완화한 것은 없다.

1. **오답노트 형식**: `lesson`은 문자열 또는 `{text}` 둘 다 받는다(40단어·누설 검사 그대로). `observed_facts`·`error`의 형식 오류는 그 주장만 UNVERIFIABLE(학습 안 함)이고 노트 전체를 버리지 않는다. 그래서 교훈 재풀이는 형식이 맞는 교훈 + 누설 없음 + CONTRADICTED 없음이면 허용된다(§6.2–6.3 변경). 근거: 로컬 실제 노트 2개가 교훈 문자열·`rounding` 정수 때문에 통째로 탈락해 재풀이 0.
2. **그래프 노트 카탈로그**: 공개 카탈로그만 보고 알 수 있는 값(간선 존재 `present`, 간선 수)은 노트 카탈로그와 VERIFIED 계산에서 뺀다. 그림을 봐야 아는 값만 검증 대상이다.
3. **기준값**: F0가 상한에 걸린(TRUNCATED) 문항은 k_i = 0으로 센다(§5.2 그대로 적용, 구현 누락 수정).
4. **앵커**: 손실에 실제로 들어간 RL 행(이득 ≠ 0, 행 제외 아님)과 준비 행에만 건다(§7.5 그대로 적용).
5. **학습용 생성의 logprobs**: 0 대신 20개를 받는다(§4.1 변경). 평가 후보 밖 비율(TAIL_RATE) 진단에만 쓰고, μ는 그대로 뽑힌 토큰의 전체 어휘 처리 확률이다. 동점으로 판정이 모호한 토큰은 진단에서 빼고(None), 실행을 멈추지 않는다.
6. **두 판 공존 검사**: 64토큰 탐욕 생성의 토큰 일치 대신, 고정 완성문에 대한 teacher-forced 로그확률 평균 차 ≤ 0.05로 판정한다(BF16·LoRA 커널의 비결정성 때문). 이 허용 오차로는 갱신 1회 차이의 옛 판·새 판을 구별하지 못한다. 잘못 붙은 판은 슬롯 비트 일치 검사가 잡는다. 준비 생성 시작 직후 같은 가중치 두 판으로 이 검사를 한 번 미리 돌린다(조기 실패).
7. **재개**: 중간 충돌 뒤 되돌림 비교는 요청의 정체(문항·단계·seed·판 등)만 고정하고 내용은 기록만 한다. 자금 보호 정지 뒤에는 `advance --resume-funding-stop`으로 이어 간다(정지 표시 파일을 옮긴 뒤에만).
8. **실행 안전**: 긴 노트 입력(16,384토큰 초과)은 그 노트만 건너뛰고(NOTE_PROMPT_OVER_BOUND), `<|endoftext|>`로 끝난 풀이는 종료 토큰으로 처리하며, 행 만들기 실패는 그 행만 사유와 함께 버린다. 정책 해시는 판마다 한 번만 계산한다(요청마다 계산하면 GPU 유휴 약 70~80분).
9. **준비·감시 생성**: 32개를 다 기다리지 않고 끝나는 대로 채운다.
10. **평가·계획**: 900 평가 단계에 `VLLM_ENABLE_V1_MULTIPROCESSING=0`, 준비 행 단계에 모델 경로를 넣었다. GPU는 커뮤니티 A100 80GB SXM·PCIe 중 가장 싼 것(사용자 2026-11-15). NOFINAL_DRIFT는 상한 도달을 빼고 센다(보정 영수증과 같은 정의).
11. **감시 규칙 영수증**: 저장 F0로 잴 수 있는 6규칙은 합산 거짓 정지 1.5%(≤ 2%). 실행 중 확률이 필요한 TAIL_RATE·IS_DRIFT는 미인증으로 영수증 v3(`PASS_SCOPED`)에 적었다. 이 둘로 멈추면 평가를 자동으로 하지 않고 묻는다.
12. **열린 위험**: 다중 LoRA·진행 중 Merger push·공존 검사 허용 오차는 GPU에서 한 번도 돌지 않았다. 첫 구간을 직접 감시한다.
