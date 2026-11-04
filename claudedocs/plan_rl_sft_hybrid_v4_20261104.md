# 오답노트형 자기주도학습 v4: RL+SFT 하이브리드 전환 전체 계획 (1단계 → 최종)

작성일: 2026-11-04 · 기준 코드: `jang2296/MMDL_test` `13f34f84ca8f07e4582e462ec853fa70ad5d5a67` (`feat/workspace-repro`)
상태: **계획 문서. 코드 수정·데이터 다운로드·GPU 실행은 하지 않았다.** 실행은 §13의 결정과 단계별 승인 뒤에만 한다.

표기: **[확인]** 저장소 코드·저장 결과·1차 출처로 확인 · **[추정]** 근거와 양립하는 판단 · **[제안]** 아직 실행하지 않은 설계 ·
**[결정 필요]** 지침 §2의 규칙이나 이전 사용자 결정을 바꾸는 항목 · **[확인 필요]** 공개 정보가 불충분해 취득 전 재확인할 항목.

이 문서 작성 중 확인한 것: `fine_tuning/error_note/`(contract, audit, train, protocol, collect, collect_v3, evaluate, plan, source_evidence)와
`fine_tuning/notebook_rl/`(losses, policy, engines, scale, trainer, passrate, hf_pool, hf_pool_exclusion, contamination, build_exclusion_index,
exclusion_engines, adapters) 정적 검토. 이 클라우드 세션의 CPU 테스트: `test_error_note*.py` 285개, 공유 채점/체크포인트/평가 128개,
Pod 151개(비공개 데이터 부재 skip 2), V8 파서 20개 통과, Ruff·`error_note` 범위 mypy 통과. **이 CPU 결과는 실제 Qwen 학습 효과의 증거가 아니다.**

---

## 0. 요약

1. **방향**: 현재 v3(정답 노트 SFT 중심)를 **RL 주도 + 오답노트 SFT 보조**로 바꾼다. 학습 문제마다 Qwen이 정답 없이 여러 번(n=4) 풀고,
   프로그램이 채점해 맞은 풀이를 강화한다(GRPO 계열). **전부 틀린 문제에서만** Qwen이 정답을 보고 오답노트를 쓴다. 그 교훈만 보고
   다시 푼 정답을 힌트 없는 문제 입력의 학습 목표로 추가한다(교훈 효과는 같은 새 seed의 교훈 없는 대조 샘플과 비교, §3.3). 모든 생성은 Qwen 자신이 하고, 채점은 프로그램이 하며,
   상위 AI 검토는 학습 수용 판단에서 뺀다.
2. **왜 바꾸는가**: v2는 3,417문항 학습으로 900문항 +3(65 획득/62 손실, p=0.86)에 그쳤다 [확인]. 감독의 92.9%가 이미 맞히던 첫 풀이였고,
   오답을 교정한 재풀이 신호는 2.2%였다 [확인]. RL은 **스스로 맞힐 수 있는 문제의 성공률을 직접 올리고**, 출력이 잘리면 보상이 0이라
   **끝맺기도 함께 학습**한다. 오답노트는 RL이 배울 수 없는 **전부 틀린 문제**를 맡는다.
3. **데이터**: 기존 hf-pool-v3에서 약 1,300문항을 남기고, 약 2,200문항을 새로 더해 **총 3,500문항**을 만든다. 신규분은
   **B2 기준 70% 미만 18과목**(그중 ≤63.3%인 15과목이 핵심)을 겨냥한 대학 수준·도식 계산형 자료를 우선한다. 모든 문항은
   MMMU(dev/validation/test)·MMMU-Pro(standard 4/10지, vision) 제외 인덱스로 검사하고, 같은 가족(같은 그림·같은 원 레코드)은 통째로 뺀다.
4. **현실적 전망**: 67.4%는 607/900, 70%는 630/900이다. 이는 B2 558보다 각각 +49, +72다. 예산 약 25달러와 마감 11/17 안에서
   **559~580(62~64%)을 현실적 목표**로 두고, 607 이상은 "가능성은 있으나 근거 없음"으로 둔다 [추정]. 이 문서는 그 가능성을 최대화하는
   순서를 정한다. 점수를 약속하는 문서가 아니다.
5. **비용**: 절약형 기준 GPU 약 14.5–17.5시간, RTX 6000 Ada 약 $14–18(A40 약 $9–11). 추가 절약(§10.3)을 적용하면 약 $10.5–13이다.
6. **필요한 결정 5가지**(§13): ① 학습 쪽 다중 풀이 허용, ② 상위 AI 검토를 학습 수용에서 제외, ③ 최종 평가 조건, ④ 신규 데이터의
   라이선스·형식 범위, ⑤ 예산 상향 여부.

---

## 1. 출발점: 숫자와 약한 과목

### 1.1 기준 점수 [확인]

| 실행 | 조건 | 정답 | 정확도 |
|---|---|---:|---:|
| B2 (제출 baseline) | 32K 출력, 이미지 1,003,520–4,014,080, 동시 2, V8 | 558 | 62.00% |
| F0 8K | 8K 출력, 이미지 262,144–1,310,720, 동시 40 | 541 | 60.11% |
| v2 (3,417문항 노트 SFT) | F0 8K와 같은 조건 | 544 | 60.44% |
| 공식 비교값 | Qwen 측정(조건 다름) | — | 67.4% |

목표 산술: 62% 초과 559, 67.4% 초과 607, 70% 630. 1회 평가의 표본 표준오차는 약 1.6%p(±14문항)이다.

### 1.2 B2 기준 70% 미만 18과목 [확인: `results/scoring-v8-20260924/subject_scores.csv`]

18과목의 합계는 273/540(50.6%)이고, 70% 이상 12과목은 285/360(79.2%)이다. 18과목이 모두 70%(21/30)가 되면 +105다.
목표 +49~+72는 **18과목 평균 +2.7~+4문항, 그리고 12과목 손실 0**에 해당한다.

| 과목 | B2 | Hard | length | NO_PARSE | 저장소 분석에서 확인한 실패 양상 |
|---|---:|---:|---:|---:|---|
| Music | 6 | 7 | 15 | 13 | 30문항 전부 악보. 절반이 출력 상한 종료. 음자리표·음표 오인과 조성 규칙 적용 오류가 함께 나타남 |
| Diagnostics & Lab Med | 11 | 6 | 0 | 0 | Neuropathology 16·Pathology 10. 19개 오답 모두 답을 확정하고 틀림(판독·지식 연결) |
| Electronics | 12 | 18 | 7 | 13 | 회로 연결·위상·시간응답. Hard 18 |
| Materials | 12 | 13 | 6 | 7 | 22/30이 Mechanics of Materials(힘·응력·변형과 도형 조건) |
| Architecture & Eng | 13 | 18 | 13 | 13 | 구조·측량 계열. 장문 미완결 |
| Mechanical Eng | 13 | 17 | 11 | 9 | 투상도·단면 선택 8문항 0/8, Engineering Graphics 1/9 |
| Biology | 14 | 2 | 1 | 2 | 계통도 연결 방향 오해석(Biology_6) |
| Energy & Power | 14 | 15 | 13 | 11 | Hard·미완결 집중(열·유체 계산) |
| Agriculture | 15 | 12 | 0 | 0 | 15개 오답 모두 답을 확정하고 틀림(병해·생리 상태 구별) |
| Chemistry | 15 | 7 | 4 | 5 | [확인 필요] 세부 과제 집계 |
| Computer Science | 16 | 11 | 1 | 1 | [확인 필요] 세부 과제 집계 |
| Geography | 17 | 4 | 0 | 1 | [확인 필요] |
| Art | 18 | 0 | 3 | 1 | [확인 필요] |
| Manage | 19 | 5 | 0 | 0 | 안정적 출력이어도 계산·비교 규칙 오류 |
| Math | 19 | 5 | 2 | 4 | [확인 필요] |
| Pharmacy | 19 | 0 | 3 | 3 | [확인 필요] |
| Accounting | 20 | 4 | 5 | 5 | 표 28문항 중 19 정답. 회계적 의미 선택·계산 |
| Sociology | 20 | 1 | 0 | 0 | 자연 이미지·일반 개념 |

출처: `claudedocs/MMMU_V8_70pct_subject_analysis_20260930.md` §3–5, 부록 A. 사용자가 말한 "약 15가지 분야"는
≤63.3%인 위 15과목(Music~Pharmacy)이다. 66.7%인 Accounting·Sociology는 경계 과목으로 함께 다룬다.

**두 가지 실패 유형을 분리해야 한다** [확인]:
- **유형 1, 미완결·기호 관계형** (Music, Arch, Energy, Mech, Electronics, Materials): 180문항에 70정답, length 65건이고 생성 토큰의 64%를 쓴다.
  → RL의 "잘리면 0점" 보상과 도식 관계 학습이 직접 겨냥하는 영역이다.
- **유형 2, 끝맺고도 틀리는 판독·지식형** (Diagnostics, Agriculture, Manage, Sociology 등): 잘림 0. → 지식 연결 자료와
  전부 틀린 문제의 오답노트 경로가 겨냥하는 영역이다.

### 1.3 검증 원칙: validation을 데이터 선택에 쓰는 것의 한계 [결정 필요 아님, 보고 의무]

약한 과목 선정에는 **과목·세부분야·이미지 유형 집계만** 쓴다. MMMU 문항의 문장·이미지·정답은 데이터 선택이나 생성에 절대 쓰지 않는다
(§2.2 유지). 그래도 validation 과목 점수로 학습 분포를 정했으므로 **validation은 완전한 보류 평가가 아니다.** 최종 보고서에
이를 명시하고, 가능하면 MMMU-Pro Standard 10지(수업 지정 최종 평가)를 깨끗한 확인으로 함께 보고한다.

---

## 2. 규칙: 유지하는 것과 바꾸는 것

| 항목 | v3 현재 | v4 제안 | 상태 |
|---|---|---|---|
| 상위 모델 풀이·튜터·distillation | 금지 | 금지 | 유지 |
| MMMU/MMMU-Pro 모든 split·파생물 학습 | 금지 | 금지, 제외 인덱스로 차단 | 유지 |
| 평가: 문항당 1회 생성·답 하나 | 유지 | 유지 (best-of-N·다수결·재순위 금지) | 유지 |
| 학습 첫 풀이 | 1회 | **n=4회**(같은 문제를 여러 번 풀어 정오 비교) | **[결정 필요] D1** |
| 교훈 재풀이 | 1회 | **m=2회** + 선택적 교훈 없는 대조 m=2회(같은 새 seed) | **[결정 필요] D1** |
| 노트·풀이 내용 수용 | 사람/AI 검토 receipt | **Qwen 자기검증 + 프로그램 검증**, AI 검토는 감사 표본에만 | **[결정 필요] D2** |
| 학습 방식 | SFT (A 0.9 + B 0.1) | RL(GRPO) + SFT 보조(A2·B·소량 A1) | 이 계획의 핵심 |
| 생성 상한 | 8,192 | 학습 8,192 유지 | 유지 |
| 최종 평가 조건 | 8K | 8K 유지(절약형), 예산 여유 시 B2 32K 계열 추가 | **[결정 필요] D3** |
| 학습 데이터 | hf-pool-v3 10출처 | + 신규 공개 train split | **[결정 필요] D4** (라이선스·형식) |
| 기존 데이터 정책 | 기존 데이터셋만, 자체 문제 생성 금지 | 유지. 형식 변환(B등급)은 상한 30% | 유지 |

**D2를 바꾸는 이유**: v3는 노트와 풀이의 내용을 사람이나 AI가 이미지·원출처와 대조한 receipt가 있어야 학습에 넣는다
(`audit.py`의 `reviewer_kind ∈ {human, ai_review}`, `method=source_review`) [확인]. 이 구조에서는 Qwen이 쓴 글 중 무엇을 학습할지를
더 큰 AI가 고른다. 이는 "튜터 모델 금지"의 취지와 충돌하고, 3,500문항 규모에서는 검토량 때문에 실행도 불가능하다.
v4는 수용 판단을 **정답 일치(프로그램) + Qwen 자기검증 + 데이터셋 제작자 주석과의 프로그램 대조**로만 한다.
그 대가로 노트 내용의 정확성은 보증되지 않는다. 그래서 노트 B의 손실 계수를 작게 두고, 주 학습 신호는 결과 검증이 되는 RL과 A2에 둔다(§3.5).

---

## 3. 학습 방법: RL+SFT 하이브리드 v4

### 3.1 전체 구조: 라운드 방식 [제안]

```text
F0(원본 BF16) ─▶ 라운드 1(파일럿 500문항) ─▶ gate ─▶ 라운드 2(나머지 3,000) ─▶ 최종 체크포인트 1개 ─▶ 평가

각 라운드:
  ① 직전 체크포인트를 병합 export → vLLM으로 이번 라운드 문항 rollout 생성(정답 없음, n=4)
  ② 별도 프로세스 채점(training-answer-v2 확장)
  ③ 전부 틀린 문항만: 노트 생성 → 자동 gate → Qwen 자기검증 → 교훈 재풀이(m=2, 선택적 교훈 없는 대조) → 채점
  ④ 학습 행 고정(admission) → PyTorch에서 RL 손실 + SFT 손실로 갱신(문항 16개당 1회)
  ⑤ 체크포인트 저장 → 새 프로세스 재로딩 검사 → 병합 export
```

**라운드 방식을 쓰는 이유** [확인]: `notebook_rl`에는 vLLM에 LoRA 갱신을 즉시 밀어 넣는 on-policy 경로가 있다
(`engines.VLLMRolloutEngine._load_weights`, `scale.synchronize`). 하지만 이 경로는 BF16/vLLM에서 CPU·mock 검사만 통과했고
(`scale.py` 머리말), 10/31 RunPod 시도에서 갱신 전에 종료됐다. 반면 error_note의 **생성 단계 → 학습 단계 분리**는
RTX 6000 Ada에서 실제로 성공했다(갱신 428, 재로딩 차 0). v4는 이 검증된 구조를 유지한다. 라운드 안에서는 정책이 조금씩 낡는데,
이는 PPO식 비율 clip과 기존 `losses.mis_mask`(vLLM–PyTorch 로그확률 차이 sequence mask)로 처리한다.
라운드를 2개로 두면(절약형, §10) export·vLLM 재시작이 2번만 필요하다.

### 3.2 문항 하나의 처리 [제안]

```text
문항 i (P0 입력, 이미지 포함, 정답 없음)
  │ rollout j=0..3, seed_j = SHA256("3407:{pid}:v4r{round}:{j}")  ← 평가 seed와 분리
  │ 채점 r_j ∈ {1 CORRECT, 0 그 외}, UNVERIFIABLE은 그룹에서 제외(0으로 처벌하지 않음)
  ├─ 0 < Σr < 4   → RL 그룹 (advantage = r − 평균, 표준편차 정규화 없음)
  ├─ Σr = 4       → RL 신호 없음. 유지용 A1 후보 1개(첫 정답 rollout)만 보관(소량)
  └─ Σr = 0       → 오답노트 경로 (§3.3)
```

- 그룹 크기 4는 비용과 신호의 절충이다. [추정] 정답률 p인 문항에서 신호가 생길 확률은 1−p⁴−(1−p)⁴이다.
  p=0.5면 0.875, p=0.1이면 0.34다. 사전필터를 따로 돌리지 않는다. 라운드 rollout 자체가 난도를 측정한다.
- **객관식 추측 문제**: 4지선다는 찍어도 25%가 맞아 보상이 시끄럽다. 그래서 (a) 5지 이상·수치형·10지 변환(B등급) 문항의 비중을
  높이고(§5.5), (b) 최종답 형식 위반·복수 답·철회는 0점으로 둔다(현재 학습 채점 계약 유지). 그래도 남는 추측 잡음은 보고서에 한계로 적는다.

### 3.3 전부 틀린 문항의 오답노트 경로 [제안]

1. **노트**: 기존 `note_prompt_v3`(문제 + 첫 rollout j=0 원문 + 채점 결과 + 정답 → What the image shows / Image reading / Cause /
   What went wrong / Lesson)를 그대로 쓴다. 노트 생성은 문항당 1회다.
2. **자동 gate(프로그램)**: 형식, `hint_leak`, 별칭·약어 후보(`audit_note`의 alias/acronym 검사), `MISREAD_WITH_CAUSE_NONE` 모순,
   재풀이 누설 문구(`RESOLVE_LEAK_PHRASES`), 반복(`is_loop`, 연속 동일 구간 검사 보강, §4.2). 하나라도 걸리면 교훈을 쓰지 않는다.
   이 단계에 사람이나 AI 검토는 없다.
3. **Qwen 자기검증(새 기능)**: 노트의 "What the image shows"와 "What went wrong"에서 이미지에 관한 주장 최대 3개를 규칙으로 뽑는다.
   각 주장을 정답·노트 없이 이미지와 함께 짧게 다시 묻는다:
   `Look at the image. Is this statement true? Statement: "…" Answer TRUE, FALSE or CANNOT TELL, then one sentence why.`
   (seed 2개, 최대 256토큰). 두 번 모두 TRUE일 때만 그 주장을 "자기확인"으로 기록한다. FALSE가 하나라도 있으면 노트를 기각한다.
   - **한계**: 같은 모델이 같은 착시를 반복할 수 있으므로 정확성 증명이 아니다. 자기모순 노트(예: 입자 수를 노트에서는 4,
     재확인에서는 3)를 걸러내는 장치다. `human_verified=false`, `reviewer_kind=program+self`로 기록한다.
   - **프로그램 대조 가능 출처**: Geometry3K/PGPS 도형 논리식, FinQA 표 셀, ChartQA 원본 표, MapQA 메타데이터, CircuitVQA 부품 주석,
     TriView2CAD 치수 주석이 있는 문항은 수치 주장을 제작자 주석과 프로그램으로 대조한다(`source_evidence.py` 확장).
4. **교훈 재풀이와 교훈 없는 대조**: `resolve_prompt(P0, lesson)`으로 새 seed `v4_seed(pid, r, "resolve", j)`를 써서 2회 푼다.
   `--lesson-control`을 켜면 **같은 새 seed로 교훈 없이 P0만** 2회 더 푼다. 두 샘플은 교훈 블록만 다르다.
   (2026-11-04 정정: 초판은 "rollout j=0,1과 같은 seed를 쓰면 교훈 효과가 바로 측정된다"고 썼다. 프롬프트가 다르면
   seed가 같아도 생성 궤적이 달라지고, rollout은 "전부 오답"으로 골라진 표본이라 비교 기준이 될 수 없다. 그래서 대조 샘플을 따로 만든다.)
   [확인] v3에서는 재풀이 seed가 달라 교훈 효과와 재추첨 효과를 분리할 수 없었다(분석 §6).
5. **수용 행**:
   - A2 = 정답이 된 재풀이 원문. 입력은 **교훈 없는 P0**이다. 길이·반복·누설 문구 gate를 적용한다. 32토큰 하한은 단답형 과제에서만
     "단답 계약" 유형으로 예외를 둔다(사전 등록, §4.2 C4).
   - B = 해당 노트. A2가 생긴 문항이거나 자기검증을 통과한 노트만 수용한다.
   - 2회 모두 실패하면 행 0이다(원문은 보존). 자동 재시도나 사람 수정은 없다.

### 3.4 Vision 경로

현재 정책(노트의 MISREAD 자기 신고 → FULL, 그 외 LM)을 RL 행에도 적용하면 근거가 없다. v4 기본값은 다음과 같다.
- RL 행·A1: **LM 경로**(LLM 조정층 + Merger). Vision 조정층은 동결한다.
- A2·B: 자기검증에서 시각 주장이 FALSE→수정으로 확인된 노트만 FULL, 나머지는 LM 경로.
- 이는 "Vision 모듈이 원인"이라는 인과 주장이 아니다. 데이터 공급이 적은 Vision 경로의 과적합을 막기 위한 보수적 선택이다.

### 3.5 손실 함수 [제안, 수치는 사전 등록 후보]

한 갱신(문항 16개)에서:

```text
L = L_RL + λ_A2 · mean(NLL_A2) + λ_B · mean(NLL_B) + λ_A1 · mean(NLL_A1)

L_RL = (1/|G|) Σ_g group_loss(cur, old, ref, adv_g, mask, keep;
                              eps_low=0.2, eps_high=0.28, beta=0.02, norm_const=그룹 행동 토큰 수)
```

| 기호 | 값(후보) | 근거 |
|---|---|---|
| n, L_cap | 4, 8192 | 비용·현재 상한 |
| eps_low / eps_high | 0.2 / 0.28 | PPO clip + 상한 확대(DAPO식). `losses.group_loss`가 지원 [확인] |
| beta (k3 KL, F0 기준) | 0.02 | 기존 RL 경로 KL .01. v2의 62문항 손실을 줄이기 위해 소폭 상향 [추정] |
| norm_const | 그룹의 행동 토큰 수 합 | 토큰 평균(DAPO식). 초판의 n·L_cap(8,192)은 RL 항을 SFT 항보다 수십 배 작게 만들어 구현에서 바꿨다. 그룹 안에서 마이크로배치로 나눠도 합이 같다 |
| λ_A2 | 0.5 | A2는 같은 문제에서 RL이 못 준 신호. NLL 0.95로 정보량이 큼(§17.10) [확인] |
| λ_B | 0.05 | 노트 내용 미보증. v3의 0.1에서 하향 |
| λ_A1 | 0.05 | 유지용. 이미 맞히던 풀이(NLL 0.23)는 정보량이 작음 |

- old_lp 기본안은 **라운드 시작 체크포인트를 PyTorch로 다시 계산**한 값이다. vLLM–PyTorch 차이가 큰 sequence는
  `mis_mask(delta_seq=0.05)`로 제외하고 그 수를 보고한다. 절약안(§10.3의 4번)을 채택하면 별도 forward 없이 vLLM 표본 로그확률을
  old_lp로 쓰고, 사용 시점의 현재 정책 로그확률(detach)과 비교해 같은 mask를 적용한다. 둘 중 하나를 사전 등록한다.
- ref_lp는 adapter를 끈 F0다(`policy.reference_logprobs`, 추가 메모리 없음) [확인].
- 값 조정에 MMMU를 쓰지 않는다. 파일럿 라운드의 외부 dev로만 go/no-go를 판단하고, 값은 바꾸지 않는다.
  바꿔야 하면 새 protocol ID로 처음부터 다시 한다.
- 대안 [제안, 2차]: A2를 SFT 항 대신 그룹 안의 "안내 표본"으로 넣고 π(y|P0)/π(y|P0+교훈)으로 중요도 보정하는 방식
  (Guide-GRPO 계열). 구현이 복잡하므로 v4.1 후보로 둔다.

### 3.6 망각 방지

v2의 손실 62문항이 이득 65를 상쇄했다 [확인]. v4는 다음을 함께 쓴다.
1. F0 기준 KL(beta 0.02).
2. λ_A1 유지 행(전 과목, 특히 70% 이상 12과목용 유지 문항 350개, §5.4).
3. Merger 학습률 재검토: Merger는 학습 파라미터의 75.5%다(109M/144M) [확인]. v4 후보는 M 5e−6(현재 1e−5). 사전 등록한다.
4. 라운드마다 외부 유지셋(강한 과목 계열 150문항) 점수가 2SE 이상 떨어지면 중단한다.

### 3.7 관련 연구 근거와 적용 범위

| 연구 | 이 계획에 쓰는 점 | 한계 |
|---|---|---|
| [Guide-GRPO, arXiv 2506.13923](https://arxiv.org/pdf/2506.13923) | 풀지 못한 문제에만 힌트 rollout을 추가하고 중요도 보정으로 원래 정책을 학습 | 텍스트 수학. 힌트 출처가 외부 |
| [LTE, arXiv 2510.26109](https://arxiv.org/html/2510.26109v1) | 전부 실패한 문제에서 **모델 자신의 오답**을 힌트로 재시도. 외부 전문가 데이터 없음 | 가장 가까운 선례. 멀티모달·노트 형식은 아님 |
| [HINT, arXiv 2510.09388](https://arxiv.org/html/2510.09388v1) | 전부 오답일 때 힌트. 정답 직접 노출은 분포 불일치를 키움 | 답 수준 힌트. 우리는 누설 차단 |
| [STaR, arXiv 2203.14465](https://arxiv.org/html/2203.14465v2) | 자기 생성 정답 풀이 SFT 반복 | 텍스트. 정답 조건부 rationale 포함 |
| [MM-Eureka, arXiv 2503.07365](https://arxiv.org/html/2503.07365v2) | Qwen2.5-VL 규칙 보상 멀티모달 RL, MMK12 | 문항당 8 rollout, 7B 이상 |
| [Qwen3-VL 기술 보고서, arXiv 2511.21631](https://arxiv.org/html/2511.21631v2) | SFT → RL 순서, 검증 가능 과제 | 대규모·teacher 포함, 재현 불가 |
| [SFT Memorizes, RL Generalizes, arXiv 2501.17161](https://arxiv.org/html/2501.17161v2) | RL의 분포 밖 일반화 | 좁은 통제 환경 |

---

## 4. 코드 수정 상세

원칙 [확인: CLAUDE.md §2.4]: 새 코드는 `fine_tuning/` 안에만 둔다. 기존 v1/v2/v3 기본 동작·프롬프트 byte·protocol hash는 그대로 두고
`--protocol-version v4`, `--rows-version v4` 같은 opt-in 분기로만 추가한다. 다음 파일은 수정하지 않는다:
`MMDL/` 전체, `src/mmdl` 평가기·V8, `notebook_rl/{schema.py, guards.py, vault.py, protocol_v1.yaml}`,
`fine_tuning/{train.py, modules.py, protocol.json}`, 기존 동결 복사본.

### 4.1 파일별 변경 목록

| 파일 | 종류 | 변경 요약 |
|---|---|---|
| `error_note/protocol.py` | 수정 | `plain_protocol_v4(precision, total_steps, *, contracts, rl, sft, routes, rounds)` 추가 |
| `error_note/contract.py` | 수정 | v4 seed, rollout 그룹 분류, A2/B/A1 행 생성 `build_rows_v4`, 반복 검사 보강, 단답 계약 유형 |
| `error_note/rollout_v4.py` | **신규** | 라운드 rollout 요청 생성, 그룹 결과 저장(token id·logprob·finish), 그룹 분류 |
| `error_note/selfcheck.py` | **신규** | 주장 추출, 자기검증 프롬프트·파싱, 판정 receipt |
| `error_note/rl.py` | **신규(작게)** | 저장 rollout → 텐서, old/ref logprob 계산, `notebook_rl.losses` 호출로 RL 손실 계산 |
| `error_note/collect.py` | 수정 | `rollout`·`selfcheck` 하위명령, `run_resolve`에 seed 지정 인자, `_child_command`에 v4 인자 전달 |
| `error_note/collect_v3.py` → 재사용 | 수정 | `bind_collection`·`generation_identity`에 v4 필드(round, n, m, 체크포인트 SHA) |
| `error_note/audit.py` | 수정 | `audit_note_v4`: 프로그램 gate + 자기검증 receipt 결합. 사람/AI 검토는 선택 필드(감사 표본) |
| `error_note/train.py` | 수정 | `run_update_v4`, `_read_groups_v4`, `run_admit_v4`, `run_dry_run_v4`, 라운드 시작 checkpoint 로드 |
| `error_note/plan.py` | 수정 | 스테이지 `note-v4-r{1,2,3}-{collect,train}`, `note-v4-export`, `note-v4-eval` |
| `error_note/evaluate.py` | 수정 | arm `F0/HYB/RL-ONLY`, 군집별 dev 보고, 유지셋 중단 규칙 |
| `error_note/source_evidence.py` | 수정 | 신규 출처의 제작자 주석 대조 규칙(§3.3) |
| `notebook_rl/answers.py`, `verifier.py`, `verifier_service.py` | 수정 | `training-answer-v3`: 10지 객관식, 단위 수치, 범위, 텍스트 전용 문항 |
| `notebook_rl/adapters/` | 신규 파일 여러 개 | 신규 출처 adapter(§5.3). 기존 `hf_common.SourceSpec/Candidate` 계약 그대로 |
| `notebook_rl/adapters/hf_registry.py` | 수정 | 신규 모듈 등록 |
| `notebook_rl/hf_pool.py` | 수정 | v4 quota plan(군집·과목·답 형식·출처 상한), 라벨 무관 hash 순서 |
| `notebook_rl/hf_pool_exclusion.py`, `contamination.py` | 수정(작게) | 텍스트 전용 문항 경로, 출처 수준 lineage 검사 항목 |
| `notebook_rl/proof_eval.py` | 수정 | 등록 형식 v6: `FT-HYB-V4`, B2 계열 평가 변형 |
| `fine_tuning/tests/test_error_note_v4_*.py` | **신규** | §4.4 목록 |
| `error_note/README.md` | 수정 | v4 명령·의미·한계 |

### 4.2 모듈별 상세

**(1) `error_note/contract.py`**
- `rollout_seed(problem_id, round_index, j)` = `sample_seed(problem_id, f"v4r{round_index}:{j}")`. 기존 `sample_seed`를 재사용하고
  평가 seed(`3407:{pid}`)와 겹치지 않게 한다.
- `classify_group(statuses) -> "MIXED" | "ALL_CORRECT" | "ALL_WRONG" | "UNVERIFIABLE"`: UNVERIFIABLE이 하나라도 있으면 그 표본을 빼고
  남은 n'≥2이면 분류, 아니면 UNVERIFIABLE로 둔다.
- `build_rows_v4(record, tokenize)`: 출력 행 종류는 `RL_SAMPLE`(그룹의 각 rollout, advantage 포함), `A2`, `B`, `A1_RETAIN`.
  모든 행에 prompt/target/원문 token id SHA, rollout seed, 라운드 체크포인트 SHA를 묶는다.
- 반복 검사 보강: 기존 `is_loop`(끝 3,000자 압축률<0.12)는 64토큰 구간 58회 반복을 놓쳤다(§17.6) [확인].
  `has_block_repetition(token_ids, block=64, min_repeats=8)`를 **추가**하고, 학습 행 제외 기준에만 쓴다. 기존 v1–v3 판정은 바꾸지 않는다.
- 단답 계약(C4 해결): `answer_form ∈ {short_label, numeric, mcq, open}`을 출처 adapter가 사전 지정한다. `short_label`(SLAKE·MapQA 등)은
  32토큰 하한 대신 "최종답 선언 포함·반복 없음"만 요구한다. 단답 행 비중 상한은 15%다. 군더더기를 붙여 늘리지 않는다.

**(2) `error_note/rollout_v4.py` (신규, 약 250줄 예상)**
- `rollout_requests(items, actors, round_index, n, checkpoint_sha)`: `collect.run_solve`의 요청 형식을 그대로 쓰되 seed만 v4 규칙.
- 저장: `rollouts.jsonl`(problem_id, j, seed, text, token_ids, vllm_logprobs, finish_reason, tokens). 원문 무수정·append-only.
- 채점: 기존 `verifier_service` 프로세스 분리 구조 그대로. `groups.json`에 분류와 보상.
- 생성 엔진: 기존 `collect._vllm_stream_builder`·`continuous_stream`·`auto_kv_gib` 재사용(동시 수 KV 기반 auto) [확인].

**(3) `error_note/selfcheck.py` (신규, 약 200줄 예상)**
- `extract_claims(note_text) -> list[str]`: "What the image shows"의 문장 단위, 숫자·위치·개수·방향어 포함 문장 우선, 최대 3개.
  규칙 기반이며 모델 호출은 없다.
- `probe_prompt(claim)`: §3.3의 고정 문구. 정답·이전 풀이·노트 전문을 넣지 않는다.
- `parse_probe(text) -> TRUE|FALSE|CANNOT_TELL|INVALID`, `decide(claims, probes) -> {result: SELF_CONSISTENT|SELF_CONTRADICTED|UNCHECKED}`.
- seed: `sample_seed(pid, f"v4r{r}:probe:{k}:{s}")`, 최대 256토큰.
- receipt: 주장 원문 SHA, probe 원문, 판정. `reviewer_kind="program+self"`, `human_verified=false`.

**(4) `error_note/rl.py` (신규, 약 200줄 예상)**
- `encode_rl_rows(rows, processor, config, images)`: `train.encode_row`를 재사용해 prompt+completion을 인코딩하고 completion 마스크를 만든다.
  기존 `batch_logprobs`(목표 직전 hidden state로 다음 토큰 예측)를 재사용한다 [확인].
- `old_logprobs(bundle, encoded)`: 라운드 시작 체크포인트로 no-grad 계산 후 디스크 저장(라운드 학습 시작 전 1회).
- `ref_logprobs(bundle, encoded)`: adapter 비활성 F0, no-grad.
- `rl_group_loss(cur, old, ref, rewards, mask)`: `losses.group_advantages` → `losses.mis_mask`(vLLM 대비) →
  `losses.group_loss(eps_low, eps_high, beta, norm_const=그룹 행동 토큰 수)`. 새 수식을 만들지 않는다.

**(5) `error_note/train.py`**
- `_read_groups_v4`: 순서 파일의 16문항 그룹마다 RL_SAMPLE·A2·B·A1_RETAIN을 모은다. RL 신호도 A2도 없는 그룹은 비활성으로 기록한다.
- `run_update_v4`: (a) RL 그룹별 `rl_group_loss` backward, (b) A2/B/A1 NLL에 λ를 곱해 backward, (c) 기존 `_clip_groups`(그룹별
  clip 1.0) → `optimizer.step` → `scheduler.step`. 경로 전환은 기존 `bundle.route.apply`를 쓴다.
  기록: 종류별 행 수·토큰·raw/weighted loss·advantage 분포·mis 제외 수·KL 평균·기울기 norm(L/M/V).
- `run_dry_run_v4`: 기존 dry-run처럼 optimizer 없이 각 항 기울기 유한성과 파라미터 불변을 확인한다(로컬 NF4 기능시험용).
- 라운드 연속성: `--init-checkpoint` 인자로 이전 라운드 체크포인트를 받는다. optimizer 상태는 라운드마다 새로 시작하고
  scheduler는 전체 라운드 step 수로 사전 고정한다(재개 identity에 포함).
- `run_reload`: 기존 새 프로세스 재로딩 검사를 라운드마다 그대로 쓴다.

**(6) `error_note/audit.py`**
- `audit_note_v4(record, gold, choices, provenance, selfcheck, aliases, acronym_candidates)`: 기존 `audit_note`의 프로그램 검사를 그대로 호출하고,
  `content.result`를 `SELF_CONSISTENT/SELF_CONTRADICTED/UNCHECKED`로, `leak.result`를 프로그램 검사 결과로 둔다.
  사람·AI `review`는 선택 필드다. 있으면 감사 통계에만 쓰고 수용 판단에는 쓰지 않는다(D2).
- 감사 표본 [제안]: 라운드마다 수용된 A2/B 중 해시 순 30개를 **팀원이 직접** 이미지·원출처와 대조해 오류율을 보고한다.
  이는 수용을 바꾸지 않는 품질 측정이다.

**(7) `notebook_rl/answers.py`·`verifier.py`·`verifier_service.py`**
- `training-answer-v3` 신설(기존 v2 불변): 10지 객관식(A–J), 단위 수치(출처 단위 근거가 있을 때만), 범위(`50.2%-51.1%` 형태),
  텍스트 전용 문항, EXAMS-V류 "문제 전체가 이미지" 문항의 선택지 문자 응답.
- gold를 보고 추출값을 고르지 않는다는 기존 변형 검사(같은 응답, gold만 바꿔도 추출값 동일)를 신규 유형에도 적용한다.

**(8) `notebook_rl/adapters/` 신규 출처 (§5.3)**
- 기존 adapter가 있는 출처(hf-ai2d, hf-chartqa, hf-finqa, hf-plotqa, hf-tatqa, hf-tatdqa, hf-tqa, hf-vqarad, hf-met, hf-cooperhewitt,
  hf-crello, hf-bonemarrow)는 **이미 구현돼 있으므로** quota에 넣기만 하면 된다 [확인].
- 신규 작성 후보: `hf_exams_v.py`(EXAMS-V train, 영어와 지정 언어만), `hf_mmk12.py`(train만, 제공 CoT 제외), `hf_geo.py`(UniGeo/GeoQA+ train),
  `hf_pathvqa.py`(train만), `hf_circuit.py`(CircuitVQA), `hf_triview.py`(TriView2CAD), `hf_musixqa.py`(train 파일만),
  `hf_medmcqa.py`(텍스트 전용, 해설 필드 제외), `hf_mmsci.py`(train, 캡션 선택 MCQ).
- 공통 규칙: 제작자의 train split만 쓴다. 제공 해설·CoT는 입력과 학습 목표 모두에서 제외한다(teacher rationale 금지).
  `label_provenance`, `upstream_origin`, `license`, revision 40자 SHA를 `SourceSpec`에 고정한다.

**(9) `notebook_rl/hf_pool.py`** — `load_plan`에 v4 quota 형식(`cluster`, `subject_task`, `answer_form`, `source_cap`, `b_grade_cap`)을 추가한다.
선택은 기존처럼 `SHA256(seed:source:native_id)` 순이고, 정답·모델 출력·벤치마크를 보지 않는다 [확인].
MapQA의 Sociology/Geography hash 라벨은 쓰지 않고, `subject_task`(실제 작업)를 별도로 기록한다(분석 §8.2).

**(10) `error_note/plan.py`** — `_plan_v4(stage, p)`: 라운드별 collect(rollout→note→gate→selfcheck→resolve→rows) / train(admit→train→reload→export).
각 스테이지 입력 receipt가 없으면 다음 단계로 가지 않는다(기존 chain 원칙).

### 4.3 데이터 흐름과 receipt

```text
pool-v4 build ─(exclusion scan: registry→engines→assemble→apply)─▶ pool-v4 freeze(SHA)
  └▶ order-v4.json(군집 순환, 사전 고정) ─▶ round r:
       rollouts.jsonl → groups.json → notes.jsonl → gates.json → selfcheck.jsonl → resolve.jsonl
       → rows-v4.jsonl → admission-v4.json → train-r{r}/updates.jsonl → ckpt-r{r} → reload.json → export-r{r}
```

모든 파일은 append-only·write-once이고, 다음 단계는 앞 파일의 SHA를 binding에 포함한다(v3와 같은 방식).

### 4.4 새 CPU 검사 (구현 시 작성)

| 검사 | 반드시 포함할 사례 |
|---|---|
| seed 분리 | v4 rollout seed가 평가 seed·v3 seed와 겹치지 않음 |
| 그룹 분류 | 0/4·1/4·4/4·UNVERIFIABLE 혼합 |
| RL 손실 | advantage 0 그룹은 기울기 0, KL 0일 때 순수 PG와 일치, 마이크로배치 분할 합 = 그룹 전체 |
| 교훈 대조 | 재풀이와 대조 샘플의 seed가 같고 입력은 교훈 블록만 다름 |
| 자기검증 | 주장 추출 결정성, 정답·노트 비노출, FALSE 하나면 기각, INVALID 처리 |
| 행 수용 | A2 입력이 교훈 없는 P0, 누설 재풀이 거부, 블록 반복 거부, 단답 계약 상한 |
| 손실 결합 | λ=0이면 해당 항 기울기 0, λ_B만 0일 때 RL-ONLY arm과 기울기 일치 |
| 라운드 연속 | 이전 라운드 체크포인트 SHA 불일치 시 거부, scheduler 전체 step 고정 |
| 데이터 | 신규 adapter의 train split만, CoT 필드 미사용, 텍스트 전용 행 이미지 0 처리 |
| 오염 | 텍스트 전용 n-gram 경로, 가족 격리 전파, MMMU-Pro vision 스크린샷 대응 |
| 회귀 | 기존 285+128+151+20 검사 전부 통과(이번 세션 기준선) |

---

## 5. 데이터: 3,500문항 선정

### 5.1 선정 원칙

1. **출처 등급**(기존 정책 유지, `research_data_policy_contamination_scale_20261030.md` §2) [확인]
   - A: 데이터셋 제작자(사람 또는 제작자 프로그램)가 만든 질문·정답 → 우선 사용.
   - B: 사람 분류 라벨 + 같은 라벨 집합 오답 보기로 만든 MCQ → 전체 30%·출처당 15% 상한.
   - C: 우리가 정답을 계산하거나 그림을 렌더링 → 제외. TAT-DQA PDF 래스터화는 기존 결정상 보류 상태다.
2. **제작자 train split만** 사용한다. val/test split은 쓰지 않는다. "벤치마크 이름이지만 train split을 배포한 자료"는 10/30 결정으로 허용됐다 [확인].
3. **MMMU 형태와 맞추기**: 이미지 필수, 대학·고등 수준 우선, 객관식(4–10지) 중심, 수치 주관식 소량(MMMU 주관식 53/900=5.9%).
4. **라벨 무관 선택**: quota 안에서는 hash 순으로 고른다. 첫 rollout 결과를 보고 문항을 바꾸지 않는다.
   난도는 라운드 rollout이 측정하며, 쉬운 문항(4/4)은 학습량 0으로 자연스럽게 빠진다.

### 5.2 기존 hf-pool-v3에서 유지할 부분 (약 1,300) [확인: 분석 §8.1의 출처별 수치]

| 출처 | 현재 | 유지 | 이유 |
|---|---:|---:|---|
| ScienceQA | 936 | 250 | 정답률 93.8%로 너무 쉬움. 물리·화학·생물 중 grade 높은 문항 우선 |
| Geometry3K | 380 | 300 | 관측+계산. 학습 채점 v2로 단위 문제 수정됨 |
| SSMR (음악) | 393 | 150 | Music 대응. 장문 반복 주의 |
| GVLQA | 364 | 150 | Yes/No 100 + max-flow 등 알고리즘 22개 전부 + 나머지 |
| MapQA | 280 | 150 | 범위 답 계약 수정 후. Sociology 라벨은 쓰지 않음 |
| SLAKE | 198 | 120 | 단답 계약 유형 |
| ChEBI role | 400 | 50 | 템플릿 편중 축소 |
| DeepWeeds / PlantVillage / NCT-CRC | 199/167/100 | 60/60/30 | B등급 상한 준수 |
| **합계** | 3,417 | **1,320** | 이미 오염 검사·동결을 통과한 문항이라 재사용 비용이 낮음 |

### 5.3 신규 후보 (약 2,200): 약한 과목 군집별

**이미 adapter가 있는 출처**(구현 비용 낮음, 오염 검사만 필요) [확인]: AI2D, TQA, ChartQA, PlotQA, FinQA, TAT-QA, TAT-DQA, VQA-RAD,
Met, Cooper Hewitt, Crello, BoneMarrow.
**새 adapter가 필요한 출처** [확인 필요 표시는 취득 전 재확인]:

| 후보 | 규모·형식 | 라이선스·상태 | 대응 과목 |
|---|---|---|---|
| [EXAMS-V](https://huggingface.co/datasets/MBZUAI/EXAMS-V) | 20,932문항, 20과목, 11개 언어, train/test 분리. 시험 문제 | 논문 페이지 CC BY-NC-SA 4.0 [확인 필요: 데이터 카드 라이선스, train 수 16,724 vs 16,494 불일치] | 물리·화학·생물·지리 등. 시험 문제라 MMMU와 형식이 가장 비슷하고 **오염 위험도 가장 높음** |
| [MMK12](https://arxiv.org/pdf/2503.07365) | train 15,616 수학 빈칸형(초455·중9,776·고5,385) | [확인 필요: 라이선스]. 제공 CoT 제외 | Math |
| UniGeo / GeoQA+ ([GAPS 논문](https://arxiv.org/pdf/2401.16287)) | UniGeo 계산 4,998·증명 9,543, GeoQA+ train 6,027 | [확인 필요: 원본 라이선스] | Math(기하) |
| [PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) | train 19,654 QA / 2,599 이미지 | 카드상 MIT. 단, 병리 교과서·PEIR 기반이라 **교과서 출처 MMMU 중복 위험 높음** | Diagnostics, Basic Med |
| [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) | train 182,822, 21과목(약리 포함), **텍스트 전용** | 카드 Apache-2.0, 2차 출처 NC 주장 [확인 필요] | Pharmacy, Diagnostics 지식 연결 |
| [CircuitVQA](https://mlanthology.org/ecmlpkdd/2024/mehta2024ecmlpkdd-circuitvqa) | 5,725 회로 이미지, 115K+ 질문 | [확인 필요: split·라이선스] | Electronics(부품·연결 지각) |
| [TriView2CAD](https://arxiv.org/html/2506.00568v2) | 합성 200,000 + 실제 3,000 정투상도, 치수 주석 | [확인 필요: 라이선스·QA 파일] | **Mechanical Eng 투상도(0/8 영역)**, Arch |
| [MusiXQA](https://arxiv.org/html/2506.23009v2) | 9,600 악보, 130K+ QA(OCR·OMR·화음) | [확인 필요: train 파일 목록이 판본마다 다름] | Music |
| [MMSci](https://arxiv.org/html/2407.04903v3) | Nature Communications 그림, CC BY 4.0, train 분리 | 캡션 선택 MCQ라 과제 형태가 다름 | Materials, Energy |
| MechVQA ([arXiv 2605.30794](https://huggingface.co/papers/2605.30794)) | 기계 도면 이해, 반자동 구축 | 10/30 조사에서 "LLM 생성"으로 제외 → **재확인 전 제외 유지** | Mechanical Eng |

**제외 유지**(기존 조사): PMC-VQA·LLaVA-Med·Quilt·Kvasir-VQA(LLM 생성), PathMMU·OmniMedVQA·AgMMU 등(평가 전용),
VisualWebInstruct·ViRL39K 같은 **웹 수집·재가공 혼합물**(출처 계보를 문항 단위로 확인할 수 없어 MMMU 유입을 배제할 수 없음).
[TIGER-Lab ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K)는 재구성·바꿔쓰기를 거친 RL용 모음이라 같은 이유로 제외한다.

### 5.4 과목·군집별 배분안 (합계 3,500) [제안]

| 군집 | 대상 과목(B2) | 배분 | 구성(기존 + 신규) |
|---|---|---:|---|
| C1 공학 도식·정량 | Mech 13, Materials 12, Energy 14, Arch 13, Electronics 12 | 750 | TriView2CAD 150, CircuitVQA 150, EXAMS-V 물리·공학계 200, MMSci(Materials·Energy·Engineering) 100, AI2D·TQA·ScienceQA 물리 150 |
| C2 기호·악보 | Music 6 | 250 | SSMR 150, MusiXQA 100 |
| C3 과학 도식·지식 | Chemistry 15, Biology 14 | 450 | AI2D 150, TQA 100, EXAMS-V 화학·생물 100, ScienceQA(화·생 상위 grade) 50, ChEBI 50 |
| C4 의학 판독·지식 | Diagnostics 11, Pharmacy 19 | 450 | PathVQA 120, VQA-RAD 60, SLAKE 120, NCT-CRC 30·BoneMarrow 30, MedMCQA 텍스트(약리·병리) 90 |
| C5 정량 문서·차트·수학 | Accounting 20, Manage 19, Math 19 | 650 | Geometry3K 300, MMK12 100, UniGeo/GeoQA+ 50, FinQA·TAT-QA 100, ChartQA-H·PlotQA 100 |
| C6 지도·사회·사진·미술 | Geography 17, Sociology 20, Agriculture 15, Art 18 | 450 | MapQA 150, AI2D·TQA 지구과학 60, EXAMS-V 지리·사회 60, PlantVillage·DeepWeeds 120, Met·CooperHewitt·Crello 60 |
| C7 컴퓨터과학 | CS 16 | 150 | GVLQA 150(알고리즘형 전부 포함) |
| C8 유지(≥70% 12과목) | Physics·Economics·Public Health 등 | 350 | ScienceQA 100, ChartQA/PlotQA(경제·보건) 150, SLAKE·VQA-RAD 임상 50, Met 예술사 50 |
| **합계** | | **3,500** | 기존 약 1,320 + 신규 약 2,180 |

- 이 배분은 **공급이 확인되기 전의 목표치**다. 오염 제외·라이선스·품질 필터 후 부족하면 같은 군집의 다른 출처로 채운다.
  군집 안에서 못 채우면 부족한 채로 두고 보고한다(다른 군집에서 숫자만 채우지 않는다).
- C1은 10/30 조사에서도 "정역학·동역학 0, 구조·측량 0"이었다 [확인]. TriView2CAD·CircuitVQA·EXAMS-V의 실제 공급이 확인되지 않으면
  C1은 750에 못 미칠 가능성이 높다. **이 군집의 성능 향상은 RL의 끝맺기 효과에 더 의존한다** [추정].
- 텍스트 전용 MedMCQA는 이미지가 없어 MMMU 형식과 다르다. 지식 연결 보강용으로 전체의 3% 이하로 제한한다.

### 5.5 MMMU·MMMU-Pro 형태 맞추기

[MMMU-Pro](https://arxiv.org/html/2409.02813v3)는 (1) 텍스트만으로 풀리는 문항 제거, (2) 보기 4→10개 확장, (3) 문제·보기를 사진/스크린샷으로
넣는 vision 설정으로 만들어졌다 [확인]. 학습 쪽 대응은 다음과 같다.
- **이미지 의존성**: 신규 문항에도 "이미지 없이 풀리는 문항"을 걸러 내는 검사를 F0로 한 번 돌릴 수 있다(텍스트만 입력 n=2, 둘 다 맞으면 제외).
  비용이 들고 학습 쪽 다중 생성이므로 D1에 포함해 결정한다.
- **10지선다**: 분류형 출처(PlantVillage 38종, NCT-CRC 9종, BoneMarrow 21종, DeepWeeds 9종)는 **같은 라벨 집합에서** 오답 보기를 늘려 10지로 만든다(B등급 규칙).
  QA형 출처(ScienceQA·AI2D·EXAMS-V)는 제작자 보기만 쓰고 보기를 새로 만들지 않는다(C등급 금지).
- **보기 순서 섞기**: 문항당 1회, seed 고정 순열. "All of the above" 같은 위치 참조 보기가 있는 문항은 섞지 않는다.
- **vision 설정(문제가 이미지 안에 있음)**: EXAMS-V는 시험 문제 이미지 형식이라 이 설정과 가장 가깝다 [확인 필요: 문항별 이미지 구성].
  다른 출처를 스크린샷으로 렌더링하는 것은 "우리 렌더링"이므로 하지 않는다(C등급).

### 5.6 선정 절차

1. **S1-a 출처 확정**: 각 후보의 데이터 카드·논문·저장소로 라이선스, train split, 라벨 계보, 제공 해설 필드, MMMU/Pro 포함 여부를 확인하고
   `source_registry_v4`에 revision SHA와 함께 고정한다.
2. **S1-b 후보 추출**: `hf_pool.select`로 군집별 quota의 **2배**를 hash 순으로 뽑는다(오염·품질 탈락 여유).
3. **S1-c 품질 필터(프로그램)**: 기존 잡음 규칙 재사용(ScienceQA hint 누출, AI2D 제목=정답, SLAKE 잘린 정답, ChartQA 표 일치, FinQA 프로그램 재실행) [확인].
4. **S1-d 오염 검사**(§6) → 제외 → 군집별 hash 순으로 최종 3,500 확정.
5. **S1-e 보류 평가셋 분리**: 신규 출처마다 **가족 단위**로 15%를 외부 dev/sealed로 떼어 학습과 겹치지 않게 한다(같은 문서·이미지·차트 가족은 한쪽에만).
6. **S1-f 동결**: `pool-v4` freeze SHA, `order-v4.json`(군집 순환으로 라운드마다 고르게) 고정. 이후 교체하지 않는다.

---

## 6. 오염 검사: MMMU와 MMMU-Pro가 섞이지 않게

### 6.1 무엇을 "같은 문제"로 보는가 [확인: 10/30 규칙 유지]

학습 후보 t와 벤치마크 문항 b(MMMU dev/validation/test, MMMU-Pro standard 4/10지·vision)에 대해 다음은 **DUPLICATE**다:
같은 질문·같은 그림, 바꿔쓰기·번역·전사, 보기 순서 변경·4→10지 확장, 숫자·단위만 바꾼 변형, **같은 그림에 다른 질문**,
MMMU-Pro vision 스크린샷. 같은 대상의 다른 사진(같은 작품·사례·분자)은 정답이 그 대상 식별에 달려 있으면 DUPLICATE다.
DUPLICATE·UNRESOLVED는 **가족 전체**(같은 이미지 해시 집합·같은 원 레코드)를 제외한다.

### 6.2 제외 인덱스 [확인]

`notebook_rl/benchmark_inventory_v1.json`: 100개 파일, 16,740 레코드. MMMU test 32·dev 30·validation 30 파일,
MMMU-Pro test 8 파일(standard 10지·4지, vision). `training_allowed=false`. MMMU test 정답이 2026-02-12에 공개됐으므로 test도 같은 누출 경로로 본다.

### 6.3 단계 [제안, 기존 도구 재사용]

| 단계 | 내용 | 도구 |
|---|---|---|
| 0 출처 수준 | 데이터셋 구성 문서에 MMMU/Pro 또는 그 원천(교과서·시험·웹 수집)을 쓴다고 밝힌 혼합물은 제외. 웹 재가공 모음 제외. 시험·교과서 기반 출처(EXAMS-V, PathVQA, TQA, AI2D)는 "고위험"으로 표시 | `source_registry_v4`, `contamination._source_checks` |
| 1 정확 일치 | 원본 이미지 SHA, 픽셀 SHA, 정규화 텍스트 일치 | `contamination.scan`, `build_exclusion_index` |
| 2 근접 이미지 | pHash BK-tree, SSCD(복사 검출), SigLIP 임베딩 | `contamination.BKTree`, `exclusion_engines.SSCD/SigLIP` |
| 3 이미지 속 글자 | OCR 후 텍스트 n-gram(문제가 이미지 안에 있는 EXAMS-V·Pro vision 대응) | `exclusion_engines.OCR`, `contamination.ocr_scan` |
| 4 텍스트 | 정규화 n-gram 포함률, 보기 집합 일치. **텍스트 전용 출처(MedMCQA)는 이 단계가 유일한 검사이므로 8-gram까지 낮춰 본다** | `contamination._text_matches` |
| 5 등급 분류 | T1(정확·pHash≤3·SSCD≥0.95·n-gram≥0.8) 자동 가족 제외, T2 검토, T3 약한 신호 | `tier-review-v1`(10/30 사용자 결정) [확인] |
| 6 검토 | T2와 부족 군집의 T3를 **실제 그림 대조**로 DISTINCT/DUPLICATE/UNRESOLVED 판정. 판정자만 벤치마크 정답을 봄. 학습기에 전달 안 됨 | `hf_pool_exclusion.review_packet/run_apply` |
| 7 가족 전파 | DUPLICATE·UNRESOLVED의 가족 전체 제외 | `contamination.family_quarantine` |
| 8 감사 | 자동 통과 층별 무작위 300개를 SigLIP 최근접 5개와 사람이 대조. 1건이라도 나오면 그 층 전체를 수동 검토로 되돌림 | 기존 §4 단계 7 |
| 9 사후 민감도 | 최종 평가에서 "학습 풀에 표시된 이웃이 있는 MMMU 문항"과 "없는 문항"의 FT−F0 차이를 따로 보고. 개선이 앞쪽에 몰리면 누출 의심 | `proof_eval` 보고 항목 |

- 검출기와 임계값은 완화하지 않는다. 오염 검토는 **학습에서 빼는 판단**이라 학습 내용을 정하는 판단(D2)과 다르다.
  다만 검토는 팀원이 직접 하는 것을 기본으로 하고, AI 보조 사용 여부는 10/30 결정(`tier-review-v1`)을 따른다.
- 고위험 출처(EXAMS-V·PathVQA·TQA·AI2D·Met)는 T3까지 전부 검토하거나, 검토 예산이 없으면 그 출처의 T3 문항을 전부 제외한다.

---

## 7. 약한 과목별 연구: 무엇을 학습시켜 어떤 능력을 기르는가

MMMU 문항은 쓰지 않는다. 같은 **능력**을 요구하는 다른 문제로 학습한다. 아래 "필요 능력"은 저장소 분석의 실패 양상에서 도출한 가설이다 [추정].

| 과목 | 필요 능력 | 학습 자료 | 학습 신호 | 위험·공백 |
|---|---|---|---|---|
| Mechanical Eng | 정투상도·단면의 선/숨은선 관계 복원, 3면도 대응 | TriView2CAD(치수·개수 질문), EXAMS-V 기술·물리 | RL(정답 보상), 치수 주석 프로그램 대조 | 공급·라이선스 미확인. 동역학 자료 0 |
| Materials | 힘·응력·변형을 도형 조건과 연결(Mechanics of Materials) | EXAMS-V 물리, MMSci Materials, Geometry3K(도형 관계) | RL + 끝맺기 | 대학 재료역학 문제 자료 거의 없음 |
| Energy & Power | 열·유체 계산을 끝까지 수행 | EXAMS-V 물리, PlotQA 에너지, MMSci Energy | RL(잘리면 0점이므로 끝맺기 학습) | 열역학 도식 자료 부족 |
| Arch & Eng | 구조·측량 도식 판독, 장문 계산 완결 | TriView2CAD 실제 도면, Geometry3K, UniGeo | RL | 구조역학 자료 0 |
| Electronics | 회로 연결·부품·값 판독 → 회로 법칙 적용 | CircuitVQA(지각), EXAMS-V 물리 회로, TQA 회로 | RL, CircuitVQA 주석 대조 | 회로 해석(테브난 등) 자료 0. 지각만 보강됨 |
| Music | 음자리표·음 높이·박자 판독, 조성 규칙, **답을 짧게 끝내기** | SSMR, MusiXQA(OMR·화음) | RL(15/30이 잘림 → 끝맺기 보상 효과 기대), 노트 | SSMR 노트 반복 문제(v3에서 8,192 반복) |
| Diagnostics | 병리 조직 특징 판독 → 질환 범주 연결 | PathVQA, NCT-CRC·BoneMarrow(10지), MedMCQA 병리 텍스트 | 오답노트 경로(끝맺고 틀리는 유형) + RL | PathVQA 교과서 출처 오염 위험. 신경병리 자료 부족 |
| Pharmacy | 분자 구조·약리 지식 연결 | ChEBI(축소), MedMCQA 약리 텍스트 | RL + 노트 | 분자 그림 기반 약리 문제 부족 |
| Biology | 계통도·도식의 연결 방향 해석 | AI2D(생물 도해), TQA 생명과학, EXAMS-V 생물 | RL + 노트(관계 오해석은 노트가 교정 대상) | 계통도 전용 학습 자료는 찾지 못함 |
| Chemistry | 구조식·반응 도식 판독, 계산 | EXAMS-V 화학, TQA 물질과학, ScienceQA 화학(상위 grade) | RL | 반응 도식 QA train 자료는 파싱용(RxnCaption)뿐 |
| Agriculture | 병해·생리 상태 구별(병원체 없음 포함) | PlantVillage·DeepWeeds(10지 변환), PlantDoc | 오답노트 경로 + RL | B등급 상한. 분류형이라 대학 농학과 차이 |
| CS | 그래프·알고리즘 도식 추론 | GVLQA(max-flow 포함) | RL | 자료 다양성 낮음 |
| Geography | 지도·자료 판독 | MapQA(범위 답 수정), AI2D·TQA 지구과학, EXAMS-V 지리 | RL | — |
| Art | 작품 시대·양식·기법 식별 | Met·Cooper Hewitt(B) | RL | 같은 작품 사진 중복 위험(가족 단위 오염 검사 필수) |
| Manage / Accounting | 표·문서의 수치를 회계·경영 규칙으로 계산 | FinQA(표 전용), TAT-QA, ChartQA-H, PlotQA | RL(수치 정답) | TAT-DQA 래스터화 보류 시 문서형 감소 |
| Math | 기하 관계·함수 그래프·계산 | Geometry3K, MMK12, UniGeo/GeoQA+ | RL | 대학 수학(선형대수 등) 자료 거의 없음 |
| Sociology | 자료·개념 연결 | ChartQA-H 사회, EXAMS-V 사회 | RL | 자료 적음 |

**가장 기대가 큰 곳** [추정]: 유형 1 과목(Music·Arch·Energy·Mech·Electronics·Materials)의 **잘림 65건**이다. v2 SFT만으로도 F0의 잘림 111문항
군에서 +14가 관측됐다 [확인]. RL은 잘림에 직접 0점을 주므로 더 크게 작용할 수 있다. 다만 잘림을 해결해도 v2에서 남은 오답 356개 중
239개는 끝맺고 틀린 경우였다 [확인]. 그래서 지식·관계 자료(C1·C3·C4)가 함께 필요하다.

---

## 8. 평가 계획

| 평가 | 내용 | 시점 | 판정 |
|---|---|---|---|
| 외부 dev(학습과 가족 분리) | 군집별 약 55문항 × 8 = 약 450(절약형). F0 / HYB(라운드 1 뒤) / HYB 최종 | 라운드 1 뒤, 최종 뒤 | 가족 단위 bootstrap 95% 구간. 0을 포함하면 HOLD |
| 유지셋 | 강한 과목 계열 150문항 | 매 라운드 뒤 | F0 대비 2SE 이상 하락 시 중단 |
| 교훈 효과 | 교훈 재풀이 회복률 vs 같은 seed의 교훈 없는 대조 회복률(§3.3) | 매 라운드 | 차이를 보고. 0에 가까우면 노트 경로 축소 검토 |
| **MMMU validation 900** | 사전 등록 최종 체크포인트 1개, seed 3407, P0·V8, 문항당 1회 | 최종 | 559 / 607 / 630 문턱. **같은 조건 F0와 짝 비교** 함께 보고 |
| MMMU-Pro Standard 10지 | 수업 지정 최종 평가(단계 F) | 예산이 남으면 | validation 선택 편향이 없는 확인 |

**최종 평가 조건(D3)** [결정 필요]:
- (a) **B2 그대로**(32K, 공식 해상도, 동시 2): 900문항 1회 약 25시간 [확인: 10/30 실측 기반]. F0 재실행까지 하면 약 50시간이라 예산 밖이다.
  F0 대신 B2 558을 참조값으로 쓸 수 있지만, vLLM 비결정성 때문에 짝 비교는 아니다.
- (b) **B2 계열, 동시 수만 확대**(32K, 공식 해상도, 동시 16–40): 새 protocol ID. F0와 FT를 같은 조건으로 각각 1회 → 약 2–4시간씩 [추정].
- (c) **8K 유지**: F0 541과 직접 비교 가능하고 가장 싸다(각 약 48분 [확인]). 대신 잘림 손실이 그대로 남는다.
- **권장(절약형, §10): (c) 8K로 FT 1회. F0는 같은 GPU 종류·같은 8K 조건의 기존 541을 참조값으로 쓴다(다른 날 실행이라 엄밀한 짝 비교는 아님).
  예산이 남으면 (b)로 F0·FT를 추가하고, 과제 보고용 (a)는 그다음이다.**

---

## 9. 단계별 실행 계획 (11/4 → 11/17)

| 단계 | 기간 | 내용 | 산출물 | 통과 조건 | GPU 비용 |
|---|---|---|---|---|---|
| **S0 결정** | 11/4–5 | §13의 D1–D5 결정 | 결정 기록(CLAUDE.md §2 갱신 포함) | 사용자 결정 | $0 |
| **S1 데이터** | 11/5–8 | 출처 확정 → 어댑터 → 후보 추출 → 품질 필터 → 오염 검사 → 3,500 동결 → dev/sealed 분리 | `source_registry_v4`, `pool-v4` freeze, `order-v4.json`, 오염 receipt | 군집별 공급표, 제외 내역 전부 기록 | $0 (CPU; SSCD/SigLIP는 로컬 GPU 또는 CPU) |
| **S2 코드** | 11/5–8 (S1과 병렬) | §4 구현, CPU 검사, Ruff·mypy, 독립 코드 검토 | 새 commit(MMDL_test feature branch) | 신규 검사 + 기존 584개 회귀 통과 | $0 |
| **S3 로컬 NF4** | 11/8–9 | 고정 2문항 → 10문항: rollout n=4 → 그룹 분류 → 노트·gate·자기검증·재풀이와 대조 → RL+SFT dry-run → **실제 1회 갱신·저장·새 프로세스 재로딩** | dry_run/updates/reload receipt | MIXED 그룹 RL 기울기 비영, A2 1개 이상, 재로딩 일치 | $0 |
| **S4 RunPod 준비** | 11/9 | doctor, BF16 2문항 갱신·재로딩, 처리량 30분 측정 → 비용 재계산 | 처리량 receipt | 예산 안이면 진행 | 약 $1–2 |
| **S5 라운드 1(파일럿 500)** | 11/9–10 | rollout→노트→학습→재로딩→export→dev 450·유지셋 평가(F0 포함) | ckpt-r1, dev 보고 | dev 점추정 ≥ F0이고 유지셋 하락 없음 → 계속. 하락이면 중단·보고 | 약 $3–4 |
| **S6 라운드 2(3,000)** | 11/10–12 | 같은 절차 | ckpt-r2(최종 지정), dev·유지셋 보고 | 사전 등록 중단 규칙 | 약 $8–11 |
| **S7 최종 평가** | 11/12–14 | FT 900문항 8K 1회(절약형). 예산이 남으면 F0·FT 32K 계열, MMMU-Pro | 평가 원장·V8 채점 | 559/607/630 판정, F0 541 대비 차이 | 약 $1 (32K 추가 시 +$4–8) |
| **S8 회수·정리** | 11/14–15 | REMOTE_VERIFIED → LOCAL_VERIFIED → POD_TERMINATED → STORAGE_DELETED → CLEANUP_VERIFIED | 회수 receipt | 잔존 자원 0 | — |
| **S9 보고** | 11/15–17 | 결과 보고서·발표 자료 | 보고서 | §2.5 정직성 규칙 | $0 |

각 RunPod 단계는 별도 승인으로 시작한다(§2.6). 실패하면 원인을 수정한 새 commit으로 필요한 단계만 다시 한다. 같은 원인의 수정은 최대 3회다.

---

## 10. 비용 추정 (절약형 기준)

### 10.1 실측 기준 [확인]

`result_self_note_finetuning_20261102.md` §5, RTX 6000 Ada $0.99/h: 첫 풀이 3,117문항 1시간 34분(동시 56, 1,092토큰/초),
노트 3,117문항 2시간 30분, 학습 428갱신 1시간 47분, export 7분, 900문항 8K 평가 약 48분. 이 핵심 경로의 합은 약 6.8시간, 약 $7이었다.

### 10.2 절약형 기준선

**절약형 정의**: rollout n=4 유지, 라운드 2개(파일럿 500 + 나머지 3,000), 최종 평가는 8K로 FT만 1회(F0는 같은 GPU 종류·8K·동시 40 조건의
기존 541을 참조값으로 사용), dev 평가 3회(F0·라운드 1 뒤·최종)는 군집별 약 55문항씩 약 450문항, Pod 세션 2회.

| 항목 | 계산 | 시간(추정) |
|---|---|---:|
| rollout 3,500 × 4 = 14,000 | 3,117표본/1.57시간 × 신규 자료 장문 보정 1.0–1.15 | 7–8h |
| 노트(전부 오답 약 25% ≈ 875) | v1 노트 속도 기준 | 0.6–0.8h |
| 자기검증 probe(≤3주장 × 2) | 짧은 출력 | 0.2–0.3h |
| 교훈 재풀이 875 × 2 | rollout 속도 기준 | 0.6–0.8h |
| 학습(RL 표본 + A2/B/A1, old/ref 추가 forward) | v1 행 처리량 × 행 길이 × forward 추가분 | 4–5h |
| export·재로딩 2회 | 7분 × 2 | 0.3h |
| dev·유지셋 평가 450 × 3 | | 0.6–0.9h |
| 최종 900문항 8K, FT 1회 | 실측 약 48분 | 0.8h |
| Pod 준비·doctor·모델 취득 2회 | | 0.5–0.8h |
| **합계** | | **약 14.5–17.5h** |

- **RTX 6000 Ada($0.99/h): 약 $14–18 / A40($0.59/h, 재고 있을 때): 약 $9–11.** 현재 잔액 약 $25 안에 들어간다. 저장소 비용은 별도다.
- 원안(§10 초판, $20–30)에서 줄인 것: 라운드 3→2, 최종 평가 32K F0+FT → 8K FT만(약 3–7h 절약), dev 600→450, 장문 보정 1.4→1.15.
- 손해: (1) 32K 조건의 잘림 회복 효과가 최종 점수에 반영되지 않는다. (2) F0 541은 같은 조건이지만 다른 날 실행한 값이라
  엄밀한 짝 비교가 아니다(vLLM 비결정성). 보고서에 그대로 적는다.
- v1의 $7보다 비싼 이유는 rollout 4배(약 +5h)와 RL 학습의 긴 표본·추가 forward(약 +2–3h)다. 이 둘은 RL의 본질적 비용이다.

### 10.3 추가로 아낄 수 있는 방법 (조사 결과)

| # | 방법 | 예상 절약 | 근거·한계 | 규칙 영향 |
|---|---|---|---|---|
| 1 | **한 요청에서 n=4 생성**(`SamplingParams(n=4)`) 또는 prefix caching 켜기 | 생성 5–15% [추정] | 지금은 같은 문제를 4번 따로 요청하는 구조라 이미지 전처리·비전 인코딩·prompt prefill이 4번 반복된다. 현재 코드는 `n=1`을 강제하고(`collect._vllm_stream_builder`의 `finish`) `enable_prefix_caching=False`다 [확인] | seed가 "요청 seed 1개 + 표본 순번"이 되므로 새 protocol로 고정 |
| 2 | **적응형 rollout**: 먼저 2회. 2/2 정답이면 중단하고, 나머지는 4회까지 | 생성 25–35% [추정, 쉬운 문항 비율에 의존] | ScienceQA(정답률 94%)처럼 거의 다 맞히는 문항에 4회를 쓰는 낭비를 없앤다. 2/2 정답 문항의 약한 신호는 일부 잃는다 | 사전 등록한 생성 중단 규칙. 평가 답을 고르는 것과 무관 |
| 3 | **반복 고리 조기 종료**(학습 rollout 전용): 64토큰 블록이 8회 이상 반복되면 생성을 멈추고 보상 0(잘림과 같은 처리) | 생성 5–15% [추정] | 실측 사례: 8,192토큰 출력의 45%가 같은 64토큰 블록 58회 반복(§17.6). 끝까지 생성해도 0점인 표본이다. 평가에는 적용하지 않는다 | 상한 8,192는 유지. `FINAL_ONLY` 출력을 스트리밍 감시로 바꿔야 함 |
| 4 | **old_lp를 vLLM 표본 로그확률로 대체** | 학습 20–30% [추정] | 각 표본은 1 epoch에서 한 번만 쓰이므로 별도 forward가 필요 없다. vLLM–PyTorch 수치 차이와 라운드 내 정책 변화는 현재 정책 로그확률(detach)과 비교하는 기존 `losses.mis_mask`로 거른다 | 기존 notebook_rl RL과 같은 방식 |
| 5 | **기존 1,320문항의 F0 첫 풀이 재사용** | 약 0.4–0.5h | 라운드 1이 F0에서 시작하므로 v1 `t1.jsonl` 원문을 j=0 표본으로 쓸 수 있다. 학습 채점 v3로 새로 채점하고 원문은 보존한다 | 해당 문항을 라운드 1에 배치해야 함 |
| 6 | **A40** | 시간당 약 40% | 11/01 재고 부족 이력 | — |
| 7 | **중단 가능(spot/interruptible) Pod** | 시간당 추가 절감 [확인 필요: 당일 가격] | 중단될 수 있다. rollout은 append-only로 재개되고, 학습은 checkpoint 재개를 지원한다 [확인] | 생성 단계에만 권장 |
| 8 | KL 제거(beta=0, ref forward 생략) | 학습 약 20–30% | v2에서 맞히던 문제 62개를 잃었다. 망각 위험이 커지므로 **권장하지 않음** | — |
| 9 | 데이터 준비·오염 검사·채점·CPU 검사를 모두 로컬에서 | 이미 반영 | SSCD/SigLIP는 로컬 RTX 5060 또는 CPU | — |

**1–5를 모두 적용하면**(RTX 6000 Ada): 생성 7–8h → 약 4.5–5.5h, 학습 4–5h → 약 3–3.8h이 되어 **합계 약 10.5–13h, 약 $10.5–13**이다
(A40이면 약 $6–8). 모두 추정이며 S4의 30분 실측으로 다시 계산한다. 1·3·4는 코드 변경(§4.2 (2)·(4))에 포함한다.

### 10.4 현재 GPU·CPU·메모리 사용 방식 점검 [확인: 코드 정적 검토]

- **생성은 GPU에서만 한다.** vLLM 0.11.0 한 프로세스(`tensor_parallel_size=1`, BF16)이고, CPU에서 모델을 돌리는 경로는 없다.
- **GPU 메모리**: KV cache = GPU 총량 − 16 GiB(가중치 8.68 + 기타 vLLM 5.0 + 여유 약 2.3)다(`collect.auto_kv_gib`). 48GB GPU에서
  KV 약 31 GiB였다. 동시 수는 KV 크기 ÷ 평균 문맥 4,800토큰으로 후보를 만들고 sweep으로 골랐으며 실제로 **56개**였다(1,092토큰/초) [확인].
  "GPU 약 20개"보다 이미 많이 돌고 있다.
- **CPU**: 이미지 전처리 작업자 8개(`PREFETCH_WORKERS=8`)가 GPU 생성과 동시에 다음 요청을 준비하고, 별도 프로세스가 채점한다.
  CPU DRAM은 이 대기열과 원본 이미지 보관에 쓰인다.
- **켜져 있는 것**: CUDA graph(`enforce_eager=False`), 연속 배치(`continuous_stream`가 끝난 자리를 바로 채움).
- **꺼져 있는 것**: prefix caching, 멀티모달 전처리 cache(`mm_processor_cache_gb=0`). 문항당 1회 생성인 SFT 경로에서는 합리적이었지만,
  n=4 RL에서는 10.3의 1번으로 바꾸는 것이 맞다.
- **로컬 RTX 5060 8GB**: BF16 가중치(8.88GB)가 VRAM(8.55GB)보다 커서 vLLM을 쓸 수 없고, HF NF4로 한 번에 1개씩(약 11–12.6토큰/초)
  생성한다 [확인]. 기능 확인 전용이다.

**CPU와 GPU에서 동시에 vLLM을 돌리는 방식은 권하지 않는다** [추정]:
1. vLLM 엔진 하나는 한 종류의 장치에서만 돈다. CPU 생성에는 별도 CPU 빌드로 띄운 두 번째 엔진이 필요하다.
2. 4B 멀티모달 모델을 CPU에서 생성하면 GPU 처리량의 몇 % 이하에 그칠 것으로 예상된다. Pod의 vCPU 수도 제한적이다.
3. CPU는 이미 이미지 전처리와 채점에 쓰이고 있어, CPU 생성을 추가하면 GPU에 입력을 공급하는 속도가 오히려 늦어질 수 있다.
4. CPU와 GPU는 수치 결과가 달라 학습 표본의 생성 조건이 섞인다.

**S4에서 실제로 확인할 것**: `ResourceLog`(`resource.jsonl`)의 생성 중 GPU 사용률이 80% 미만이면 동시 수나 `max_num_batched_tokens`를 올리고,
vLLM preemption 수가 많으면 내린다. KV 여유분(16 GiB 예약)을 줄여 동시 수를 늘리는 것은 OOM 위험에 비해 이득이 작다
(예약 2 GiB 감소 → 동시 수 약 +6%) [추정].

## 11. 위험과 중단 조건

| 위험 | 신호 | 대응 |
|---|---|---|
| RL이 GPU BF16에서 처음 실행됨 | S3·S4에서 비유한 손실, 기울기 0, 메모리 초과 | 원인 보존 후 중단. 한도·정밀도 자동 축소 금지 |
| 추측 보상 잡음(4지선다) | MIXED 그룹 비율은 높은데 dev 개선이 없음 | 10지·수치형 비중 확대(다음 protocol) |
| 출력 길이 붕괴 또는 폭주 | 라운드별 rollout 중앙 토큰이 크게 변함 | 유지셋 하락 동반 시 중단 |
| 망각 | 유지셋 2SE 하락 | 중단. KL·λ_A1을 높인 새 protocol은 별도 승인 |
| 노트 경로 무효 | 교훈 대조에서 교훈 효과 ≈ 0 | 노트 B 비중 0으로 둔 RL-ONLY를 주 arm으로 바꿀지 사용자에게 보고(사후 교체가 아니라 사전 등록한 분기로) |
| 오염 | 사후 민감도에서 이웃 표시 문항에 이득 집중 | 결과에 경고, 해당 출처 조사 |
| 공급 부족(C1) | 군집 quota 미달 | 부족 그대로 보고. 다른 군집 숫자로 채우지 않음 |
| 예산 | 잔액·시간당 가격 watchdog | 회수 예약금 남기고 중단 |
| 마감 | S5가 11/11까지 끝나지 않음 | 라운드 2를 축소(1,500)하고 최종 평가 우선 |

---

## 12. 기대 효과와 한계 (정직한 전망)

- **지금 가진 근거**: SFT 노트 방식(v2)은 +3으로 개선이 관측되지 않았다. RL+SFT 하이브리드는 이 프로젝트에서 아직 실행된 적이 없다.
- **긍정 근거** [추정]: (1) 약한 18과목의 잘림 84건은 보상 설계로 직접 겨냥할 수 있다. (2) 문헌에서 규칙 보상 RL은 멀티모달 추론
  벤치마크 향상을 반복 보고한다(§3.7). (3) A2 신호의 정보량이 A1보다 크다는 실측이 있다(NLL 0.95 vs 0.23).
- **부정 근거** [확인/추정]: (1) 약한 과목 다수(공학·신경병리·대학 수학)는 학습 자료 공급이 부족하다. (2) 3,500문항 1 epoch는
  문헌의 RL 규모보다 작다. (3) 4B 모델의 자기 노트 내용 정확도는 낮았다(15문항 중 PASS 약 4).
- **전망**: 559~580을 현실적 목표로 두고, 607 이상은 "가능하지만 근거 없음", 630은 이 예산·기간에서 가능성이 낮다고 본다 [추정].
  어떤 결과든 F0 짝 비교 구간, 군집별 손익, 오염 민감도, 교훈 효과를 함께 보고한다.

---

## 13. 사용자 결정이 필요한 항목

| ID | 결정 | 권장 | 영향 |
|---|---|---|---|
| D1 | 학습 쪽 다중 풀이(rollout n=4, 교훈 재풀이 m=2, 자기검증 probe) 허용. 평가는 단일 답 유지 | 허용 | 없으면 RL이 불가능하고 v3 SFT로 돌아감 |
| D2 | 노트·풀이의 학습 수용에서 사람/AI 내용 검토를 빼고 프로그램 + Qwen 자기검증으로 대체. 팀원 감사 표본 30개/라운드 | 대체 | 자기주도 원칙 회복, 노트 정확성 보증은 약해짐(λ_B 0.05로 대응) |
| D3 | 최종 평가 조건 (a)/(b)/(c) | (c) 절약형, 예산 여유 시 (b) 추가 | 비용·비교 가능성 |
| D4 | 신규 데이터 범위: NC 라이선스(EXAMS-V 등), 텍스트 전용(MedMCQA), 제작자 합성 자료(TriView2CAD·MusiXQA), 분류형 10지 변환, TAT-DQA 래스터화 | NC·텍스트 전용·제작자 합성·10지 변환 허용, 래스터화는 보류 유지 | 공급량 |
| D5 | 예산·절약 방법 | 절약형 $14–18(A40 $9–11). §10.3의 1–5 적용 시 약 $10.5–13 | §10 |

결정이 나면 CLAUDE.md/AGENTS.md §2(§2.1·§2.8·§3.0)를 날짜와 지시 내용과 함께 갱신하고 `cmp`로 byte 동일성을 확인한다.

---

## 15. S2 구현 상태 (2026-11-04, 브랜치 `claude/rl-sft-v4-code`)

**[확인] 구현·CPU 검사 완료. 실제 Qwen 실행·GPU·RunPod는 하지 않았다.** 이 클라우드 세션에는 GPU가 없고 `huggingface.co` 접속이 네트워크 정책에 막혀 있어 실제 가중치를 받을 수 없다.

| 파일 | 내용 |
|---|---|
| `error_note/v4.py` (신규) | seed, 그룹 분류·적응형 규칙, 64토큰 블록 반복 감지(증분), 프로그램 게이트, 주장 추출·probe·자기검증 판정, `build_rows_v4`, 요약 |
| `error_note/rl.py` (신규) | `notebook_rl.losses.group_loss` 호출(old = 현재 정책 detach, 그룹 행동 토큰 정규화, F0 기준 k3 KL) |
| `error_note/collect_v4.py` (신규) | rollout(적응형, 엔진 1회, T1 재사용) → note(+게이트, 정답 없는 `gates-v4.json`) → selfcheck → resolve(+교훈 없는 대조, 엔진 1회) → rows |
| `error_note/train_v4.py` (신규) | RL+SFT 갱신, 라운드 연결(`--init-run`), rl-only 대조 arm, 무갱신 dry-run, 새 프로세스 reload |
| `error_note/plan_v4.py` (신규) | 라운드 하나의 Pod chain spec(prefix caching·loop stop 포함) |
| `error_note/protocol.py` | `plain_protocol_v4` 추가(기존 v1–v3 함수 불변) |
| `error_note/collect.py` | 기본값 꺼짐 옵션만 추가: `continuous_stream`의 `WAIT`·`observe/partial`, vLLM `--prefix-caching`·`--loop-stop`, HF loop stop |
| `fine_tuning/tests/test_error_note_v4*.py` (신규 5개) | 계약·손실·스트림 20, 수집 9, 학습 5, 계획 4, 단계 간 통합 1 |

**계획(§4.1)과 달라진 점**: (1) v1–v3 경로를 건드리지 않으려고 `collect.py`·`train.py`·`plan.py`에 하위 명령을 넣는 대신 v4 전용 모듈로 분리했다.
(2) v3식 별도 admission 파일은 만들지 않았다. v4는 arm이 하나(대조군 rl-only는 같은 RL·A1 행 사용)라 rows SHA와 protocol binding으로 대신한다.
(3) 적응형 2차 풀이와 대조군 재풀이를 같은 엔진 실행 안에서 처리한다. 한 단계에서 vLLM을 두 번 띄우면 GPU 메모리가 제때 해제되지 않을 위험이 있기 때문이다.
(4) RL 정규화를 n·L_cap에서 그룹 행동 토큰 수로 바꿨다(§3.5). (5) 교훈 효과 측정을 같은 새 seed의 교훈 없는 대조로 바꿨다(§3.3 정정).
(6) 신규 출처 adapter(§4.2 (8))와 `training-answer-v3`는 실제 데이터 파일로 형식을 확인해야 하므로 S1(데이터) 단계로 넘겼다. 기존 학습 채점 계약이 이미 A–J 보기, 범위, 단위 수치를 지원한다 [확인].

**검사 결과(이 세션, CPU)**: `test_error_note*.py` 324개 통과(기존 285 + 신규 39), 공유 채점·체크포인트·평가 128개 통과,
Pod 151개 중 149개 통과·비공개 데이터 부재 skip 2, V8 파서 20개 통과, `error_note` Ruff·mypy(17개 파일) 통과.
작은 무작위 모델 검사는 4B 학습 효과의 증거가 아니다. 다음 단계는 S3(로컬 RTX 5060 NF4에서 2→10문항 실제 생성·dry-run·1회 갱신·재로딩)이다.

## 14. 출처

내부: `claudedocs/analysis_self_note_mmmu_root_causes_20261103.md`, `claudedocs/MMMU_V8_70pct_subject_analysis_20260930.md`,
`claudedocs/research_data_policy_contamination_scale_20261030.md`, `claudedocs/result_self_note_finetuning_20261102.md`,
`results/scoring-v8-20260924/subject_scores.csv`, `fine_tuning/error_note/*`, `fine_tuning/notebook_rl/*`.

외부(이번 조사에서 확인한 페이지):
- MMMU-Pro: https://arxiv.org/html/2409.02813v3
- Guide-GRPO: https://arxiv.org/pdf/2506.13923 · LTE: https://arxiv.org/html/2510.26109v1 · HINT: https://arxiv.org/html/2510.09388v1 · Scaf-GRPO: https://arxiv.org/html/2510.19807v2
- STaR: https://arxiv.org/html/2203.14465v2 · MM-Eureka/MMK12: https://arxiv.org/pdf/2503.07365 · Qwen3-VL: https://arxiv.org/html/2511.21631v2 · SFT vs RL: https://arxiv.org/html/2501.17161v2
- EXAMS-V: https://huggingface.co/datasets/MBZUAI/EXAMS-V · https://arxiv.org/html/2403.10378v1
- MedMCQA: https://huggingface.co/datasets/openlifescienceai/medmcqa
- PathVQA: https://huggingface.co/datasets/flaviagiammarino/path-vqa
- GeoQA+/UniGeo(GAPS): https://arxiv.org/pdf/2401.16287 · Geo170K: https://huggingface.co/datasets/Luckyjhg/Geo170K
- CircuitVQA: https://mlanthology.org/ecmlpkdd/2024/mehta2024ecmlpkdd-circuitvqa
- TriView2CAD/CReFT-CAD: https://arxiv.org/html/2506.00568v2 · MechVQA: https://huggingface.co/papers/2605.30794
- MusiXQA: https://arxiv.org/html/2506.23009v2 · MMSci: https://arxiv.org/html/2407.04903v3
- ViRL39K(제외 사유 확인): https://huggingface.co/datasets/TIGER-Lab/ViRL39K
