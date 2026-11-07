# SINGLE v9 최종 종합 검증 보고 (2026-11-07 00:51Z 기준)

## 요약 (Executive summary)

- **67.4(607/900)는 현재 규칙·데이터·예산으로는 도달 범위 밖이다.** (확신 높음) 필요한 순증은 +49문항인데, 같은 조건 비교에서 과거 자기학습 변형은 8K F0 541 대비 −19~+3이었고(537/544/522), 교사 없이·단일 답으로 보고된 문헌 최고치도 강한 베이스라인 위에서 +1.4~+2.4pp(8~32배 표본, 수십~수백 배 연산)다. v9는 이 범위를 넘길 구조적 이유가 없다.
- **558을 넘길 가능성은 "불확실, 50% 미만"**(판단). v9의 변화는 붕괴 방지(첫 4토큰 F0 고정, 재생 실패 행 제외)이지 새 학습 신호가 아니다. 900문항 1회의 표준오차는 약 1.6pp(≈14문항)라 ±14 안의 결과는 잡음과 구별되지 않는다.
- **수업 규칙 준수는 3항목 모두 PASS**(코드로 확인). 상위 모델 텍스트 없음, 생성은 단계마다 n=1, 평가는 seed 3407 1회, MMMU/Pro는 학습 경로에서 차단.
- **"자기주도학습"은 설계대로 작동하지만 내용 신호가 거의 없다.** v8 본학습 27회 갱신 실측: 손실 가중치로는 A2(다시 푼 정답 풀이) 24행이 전체를 지배(3.74 vs A1 0.26, B 0.27, PG 순 −0.09)하지만, 그 24행 중 내용을 실제로 고친 것은 2~5개 수준이고 나머지는 형식 교정·단답 재읽기다. 노트가 진짜 오류를 짚은 것은 2/31, 정답을 부정한 것이 7/31.
- **런포드 처리**: 묶음(16문항) 안에서는 동시 16 슬롯·즉시 보충·T1→노트→재풀이 연결이 맞다. 묶음 경계에서 가장 긴 체인(8192토큰)을 기다린 뒤 갱신(123초)·저장(70초)이 직렬이다. 실측 묶음당 421초 중 생성 54%·갱신 29%·저장 17%.
- **학습 자료가 MMMU 약점과 거의 겹치지 않는다**(확인). 30과목 중 13과목이 학습 항목 0이고, 이 13과목에 MMMU 오답 342 중 159개(46%)가 있다. Electronics·Materials·Architecture·Mechanical·Energy·Diagnostics 모두 0.
- **현실적 다음 단계**: (1) 돈 안 드는 CPU 분석(v9 결과 쌍 비교, 공식식 무작위 추측 채점의 몫 측정 ≈ +22문항 기대, 과목별 손실), (2) 사용자가 10~15달러를 추가하면 "신호 수정판" 1회(잘림 −1 보상, 문항별 F0 기준선, A2 행 가중 정규화, CLEVR/MapQA 보상 제외), (3) 20~30달러·2~4일이 더 있을 때만 약점 과목 데이터 추가. 어느 경우도 67은 목표로 약속하지 않는다.

표기: **VERIFIED** = 이 세션에서 파일·코드·수치를 직접 확인. **REPORTED** = 하위 보고서나 프로젝트 문서의 서술을 인용(직접 재확인 안 함). **UNVERIFIED** = 확인 불가.

---

## 1. v9 설계가 과거 실패(537/544/522)를 극복하고 67.4를 넘을 형태인가

**결론: 아니다. 558 초과는 불확실(50% 미만), 607 도달은 매우 낮음(5% 미만).** (판단; 근거는 아래)

근거 (VERIFIED):
- B2 V8 기준 558/900. NO_PARSE 97(객관식 79·주관식 18), 32K 한도 도달 90(그중 오답 85). `results/scoring-v8-20260924/summary.json`, `analysis_exports/mmmu_20260924/team_final_v8/official_b2/predictions.jsonl`에서 재계산.
- 과거 자기학습 결과: v1 537, v2 544, v4 522, 같은 8K 미학습 F0 541 (CLAUDE.md §4). 즉 어떤 변형도 F0를 유의하게 넘지 못했다.
- v9에서 바뀐 것은 (a) 재생 검사 실패 행을 RL·KL에서 제외, (b) 첫 4토큰 F0 KL 고정(β 0.5/B)뿐이다 (`fine_tuning/error_note/protocol.py:170-171`, `train_v4.py:560-561`, `docs/PROJECT_STATUS.md:128-130`). 둘 다 **붕괴 방지 장치**이지 새 학습 신호가 아니다.
- 일반 F0 KL(β 0.02)은 v8 27회 갱신 합계 0.0044로 사실상 꺼져 있었다 (`pod-v8-main-hgqwu5eo8tb0r9/x/single-v5/campaign/block-*/update-stats.json` 합산). v9도 같은 β다. 5번째 토큰부터는 F0에 묶는 힘이 거의 없다.
- 신호량: v8 본학습 444문항에서 RL 보상 +1 131, −1 39, **0 274(62%)**. 정답인데 보상 0인 행이 대부분 32토큰 미만 단답이다. 즉 데이터의 절반 이상이 학습에 기여하지 않는다.

문헌 비교 (REPORTED, methods 보고서; 수치는 논문 표기 그대로이며 우리 모델로의 전이는 중간 확신):
- 강한 베이스라인 위 교사 없는 단일 답 방법: CARE +1.4pp(Qwen3-VL-8B 69.6→71.0, 그룹 8), VL-Rethinker +2.4pp(8 롤아웃). 우리 v9는 문항당 표본 1개, 연산은 이들의 1/10~1/100.
- 우리 재풀이와 가장 비슷한 CARE의 reflection-resampling 제거 실험은 MMMU에서 약 0 차이. 자기수정 문헌(2311.08516, 2310.01798)도 소형 모델의 자기 오류 탐지가 약하다고 보고.

따라서 "558 초과" 가능성은 붕괴 없이 끝났을 때 잡음 범위(±14) 안에서 양쪽으로 갈 수 있는 정도다. 67.4는 +49가 필요하므로 구조적으로 불가능에 가깝다.

**확신도**: 607 불가 — 높음. 558 초과 50% 미만 — 중간(판단).

---

## 2. 수업 핵심 규칙 준수 여부 (항목별)

| 규칙 | 판정 | 근거 | 확신 |
|---|---|---|---|
| 상위 모델·튜터 풀이 학습 금지 | **PASS** | 학습 목표는 모두 모델 자신의 T1·재풀이·노트 텍스트 (`single.py:129-148`, 손실은 목표 토큰만 `train_v4.py:229`). 노트 입력은 P0+자기 풀이+채점 결과+데이터셋 정답+고정 지시문(`contract.py:290-334`). openai/anthropic/google.genai 등 import 차단(`notebook_rl/guards.py:53-64,91-95`). 정답 라벨은 HF 출처의 사람/프로그램 라벨 | 높음 (REPORTED 행 번호, 구조는 VERIFIED) |
| 여러 답 생성 후 선택 금지 | **PASS** | vLLM `SamplingParams(n=1)` 강제, 완료 1개 아니면 예외 (`single_vllm.py:133,141-142` VERIFIED). 단계별 고정 seed. 평가는 seed 3407 1회(`eval_configs/mmmu_val_v8_b2_seed3407.yaml`, REPORTED). v8은 평가하지 않았고 체크포인트 고르기 없음 | 높음 |
| MMMU/MMMU-Pro 오염 금지 | **PASS(작은 틈 1)** | 3,500개 모두 `split=train`·오염 union receipt에 묶임(`frozen_data.py:24-91` REPORTED). `~/mmdl-data/evaluation`과 HF hub `datasets--MMMU--*` 읽기 차단(`collect.py:266-272` REPORTED). **틈**: 포드의 `/workspace/mmdl-volume/eval-data/b2-predictions.jsonl`(MMMU 문항·답 포함)은 차단 목록에 없다. 학습 코드가 읽는 경로는 없으나 다음 커밋에서 차단 목록에 추가 권고 | 중간-높음 |

주의할 해석 2가지:
- **A2 행 필터는 "학습 쪽 거부 표본추출"**이다(재풀이 1회가 정답이면 학습). CLAUDE.md §2.1(사용자 v2 결정)과 §2.8("학습 행은 정답 기준으로 거르지만 평가 답을 고르는 것이 아니다")이 명시 허용. 단 재풀이 정답률(31/93 등)을 모델 정확도로 보고하면 안 된다.
- 학습 중 그룹 표본(GRPO)은 §2.8이 아니라 사용자의 2026-11-05 단일 응답 결정이 막고 있다(memory `single-rlsft-decisions-20261105.md`). 강의 S4의 해석은 UNVERIFIED.

---

## 3. 자기주도학습이 설계대로 가능한가 — 실제 신호 비중

**설계**: 그렇다. 코드는 풀이→채점→정답 보고 노트→교훈만 보고 재풀이→갱신→저장·재로딩을 3주기와 본학습에서 실제로 수행했다 (PROJECT_STATUS.md:113-125).

**실제 신호** (v8 본학습 27회 갱신 합산, VERIFIED):

| 구성 | 행 수 | 가중 손실 합 | 비고 |
|---|---|---|---|
| RL(PG) | 428~444 | 순 −0.09 (+1 131 / −1 39 / 0 274) | 62%가 신호 0 |
| A1 (정답 첫 풀이 SFT) | 125 | 0.26 | w .05 |
| **A2 (재풀이 정답 SFT)** | **24** (16/27 묶음) | **3.74** | w .5, 행 수와 무관하게 종류당 총 가중 고정(`train_v4.py:575-578`) → 1행이 0.5 전부 |
| B (노트 SFT) | 21 | 0.27 | w .05 |
| KL (β .02) | — | 0.0044 | 사실상 꺼짐 |

- **행 수 기준**: 오답노트 경로(A2+B) = 45/598 ≈ 7.5%. RL+A1이 대부분.
- **손실 가중 기준**: A2가 약 65%(NLL 가중 근사; compliance 보고서 계산, 합계는 내가 재확인). 즉 갱신 방향은 A2 24행이 좌우한다.
- **그런데 A2 내용**: v8 31개 재풀이 정답 중 형식/추출 교정 약 13, 단답 추측→풀이 약 9, 한도 회복 2, 내용 교정 최대 5 (REPORTED, data 보고서가 원문 판독). 3주기 64문항에서 깨끗한 내용 교정 1~2 (PROJECT_STATUS.md:116-117 VERIFIED). 노트는 31개 중 진짜 오류 못 찾음 15, 정답 부정 7 (`claudedocs/analysis_single_v7_cycle2_20261106.md:113,120` VERIFIED).
- **결론**: "틀림→원인 파악→고쳐서 배움"에 해당하는 신호는 전체 행의 1% 미만이고, 그 소수 행이 과대 가중(0.5/행)되어 있다. 현재 방법은 "맞힌 자기 풀이 강화 + 운 좋은 재풀이의 맥락 증류"에 가깝다. 사용자 상태 요약의 "학습 신호 대부분은 RL·A1"은 행 수로는 맞고 손실 가중으로는 틀리다.
- 재풀이에서 교훈의 인과 효과(교훈 없이 다시 풀기 대조군)는 측정한 적이 없다 — UNVERIFIED.

**확신도**: 수치 높음, 해석 중간.

---

## 4. 런포드 동시 처리 — 묶음 경계와 속도

**묶음 안 (VERIFIED)**: `continuous_stream`이 활성 요청을 `max_num_seqs`까지 채우고 엔진 step마다 보충한다(`collect.py:303-313`). 오답 T1은 즉시 노트 요청, 수용된 노트는 즉시 재풀이 요청으로 이어진다(`single.py:358-389` REPORTED). 슬롯 수 16은 REPORTED(로컬 artifact에서 값 확인 못 함).

**묶음 경계 (VERIFIED)**: 16문항 체인이 모두 끝나야 행을 봉인 → `sampler.release()`(vLLM sleep) → HF 갱신 → 전체 체크포인트 저장 → 다음 묶음 (`single_campaign.py:498-537`). 

**실측 (v8 A40, 27묶음, `results.tar` 안 `runs/single-v5/logs/main.log` VERIFIED)**: 생성 평균 227초(최소 44·최대 398), 갱신 123초, 저장 70초, 합 421초(≈7분). v9 상태 보고(21:37Z 시작, 00:33Z 22묶음)도 묶음당 약 8분으로 일치. 생성 시간은 총 토큰보다 "가장 긴 단일 체인"과 상관(r .993, REPORTED) — 즉 8192토큰 T1/재풀이 1개가 나머지 15개를 기다리게 한다.

**더 빠르게 (규칙 위반 없음)**:
1. 체크포인트 저장을 백그라운드 스레드로(다음 묶음 커밋 전 join) — 묶음당 약 70초(16%) 절감, 코드만 변경. compliance 보고서의 fix spec이 적절함.
2. 반복 루프 조기 종료 — 잘린 행 중 `is_loop`에 걸리는 비율이 1/3 정도라 효과 제한(REPORTED).
3. 32문항 묶음 — 1.5배 추정이지만 protocol 버전·KV 캐시·사용자 승인 필요, 갱신 수 절반.
4. 다음 묶음 T1을 미리 생성 — on-policy 가정·정책 해시 검사가 깨짐. 권하지 않음.

평가(MMMU 900 B2)는 별도 세션이 32K 토큰으로 1회 실행(최대 14400초). 평가 동시 슬롯 수는 UNVERIFIED.

---

## 5. 67을 넘기 위한 추가 방법 (규칙 안에서)

### 5.1 평가 프로토콜 몫 — 67.4와 62.0 차이 중 (VERIFIED 계산, 적용 여부는 별개)
- 공식 MMMU 파서는 추출 실패 시 무작위 보기를 고른다(`docs/SOURCE_REVIEW.md:35`). 우리 V8은 NO_PARSE=오답. B2의 미추출 오답 97개 중 객관식 79개에 무작위 추측을 적용하면 **기대 +21.9문항(≈+2.4pp)**. Qwen의 67.4가 LLM 보조 추출+이 규칙을 썼는지는 UNVERIFIED(CLAUDE.md §11에 차이로 기록).
- 나머지 약 3pp는 단일 실행 잡음(SE 1.6pp), 데이터 파일(TSV vs parquet)·스택 차이 — UNVERIFIED.
- **권고**: 저장된 B2 원문을 CPU에서 "공식식 규칙 추출 + 무작위 대체"로 재채점해 **별도 scoring version 진단값**으로 보고서에 적는다. V8을 대체하지 않고, 67.4와의 차이 중 프로토콜 몫을 숫자로 설명하는 용도. 비용 0.

### 5.2 손실의 가장 큰 덩어리: 끝내지 못한 답 (VERIFIED)
- 32K 한도 도달 90개 중 오답 85(= 342 오답의 25%, 전체의 9.4pp). 대부분 반복 루프(마지막 8,000자 zlib 압축비 <0.10이 75개, REPORTED). 8K F0가 B2보다 17개 낮은 것도 같은 원인.
- 현재 보상: 잘림은 0(정답 단답만 −1). **잘림·루프에 −1**을 주는 것은 규칙 중립이고, 과거 세 변형이 모두 "짧아져서" 망한 것과 반대 방향의 신호다. 단 출력이 짧아지는 부작용은 첫 4토큰 고정만으로 못 막으므로 소스별 길이 감시 유지.

### 5.3 RL 신호 수정 (규칙 중립, 표본 추가 없음)
- b=0 → **문항별 기준선**: 같은 문항의 기록된 F0 결과(v4·v8·v9에서 앞 1,200여 문항은 이미 있음)를 baseline으로. 쉬운 문항을 틀릴 때 음의 신호가 생겨 단답 쏠림을 누른다.
- F0 정답률 0 또는 1인 문항은 사전 등록 후 제외(학습 데이터 필터이지 답 선택이 아님).
- **A2 가중을 행 수에 비례**하도록 바꿔 운 좋은 1행이 묶음을 지배하지 않게 함.
- GRPO(문항당 4~8표본): 문헌상 가장 검증된 방법이지만 생성 비용 4~8배, 사용자의 11-05 단일 응답 결정을 뒤집어야 함. 지금 예산으로 불가.
- 노트 B행·노트 생성 축소: 정확도 2/31, 시간 절약 외 기대 효과 없음.

### 5.4 데이터: 짧은 답 출처 → MMMU 약점 과목
**현재 풀 (VERIFIED, `final3500-v1/lot-v9/order.json`)**: chartqa 650, scienceqa 602, mapqa 450, geometry3k 386, clevr 302, ssmr 250, nctcrc 234, slake 216, ai2d 150, gvlqa 150, plantvillage 60, chebi-role 50. 객관식은 전부 4지 이하, yes/no 643(18%).
- 과목 라벨 기준 **13/30 과목이 0개**(Electronics, Materials, Architecture, Mechanical, Energy, Diagnostics, Art, Art_Theory, Design, Accounting, Finance, Psychology, Public_Health). MMMU 오답 342 중 159개(46%)가 이 과목들에 있다.
- Tech&Eng이 오답의 34%(115개)인데 학습 항목 6%(Agriculture 60 + CS 150, 그나마 CS는 경로 yes/no).
- 낮은 가치 출처(F0 정답률 ≥.9이거나 단답이 자연스러움): clevr·mapqa·scienceqa·ai2d·gvlqa = 1,654개(47%). CLEVR은 F0에서 93% 단답이라 A1 0행, RL 보상도 대부분 0.

**후보 데이터셋 (REPORTED — datasets 보고서가 HF 카드 태그로 확인, 나 자신은 미검증; 라이선스·분할·오염은 UNVERIFIED)**:

| 우선 | 데이터셋 | 겨냥 과목(B2 정답/30) | 라이선스·라벨 | 삭제할 필드 | 오염 위험 |
|---|---|---|---|---|---|
| 1 | EEE-Bench (`afdsafas/EEE-Bench`, test 2,860) | Electronics 12, Energy 14 | MIT, 사람 | 없음 | **중간** (논문이 MMMU에도 EEE 문제 있다고 언급) |
| 2 | PhysUniBench (`PrismaX/PhysUniBench`, 3,304 EN+ZH) | Materials 12(재료역학), Physics, Energy | CC-BY-4.0, 사람 | 없음 | 중간(교과서 출처) |
| 3 | MedXpertQA-MM (`TsinghuaC3I/MedXpertQA`, 2,000) | Diagnostics 11, Clinical, Basic Med | MIT, 시험 정답키 | 없음 | 낮음-중간 |
| 4 | TAT-DQA (`next-tat/TAT-DQA`, 16,558 QA) | Accounting, Finance | CC-BY-4.0, 사람 | 없음 | 낮음 |
| 5 | FinMME (`luojunyu/FinMME`, 11,099) | Finance, Econ, Manage | MIT, 사람 | `related_sentences`, `verified_caption` | 낮음 |
| 6 | M4U EN (`M4U-Benchmark/M4U`, 2,977) | Mech Eng 13, EE, Health | MIT, 사람 | `image_caption` | **중간-높음** (MMMU와 같은 유형 출처) |
| 7 | MusiXQA (`puar-playground/MusiXQA`) | Music 6 | MIT, 프로그램 | 없음 | ~0 (합성 악보) — 단, MMMU Music은 화성·음정 추론이라 적합성 낮음 |
| 8 | AgroBench (`Project-AgML/AgroBench`, 4,342) | Agriculture 15 | NC 태그 vs "명시 안 함" 충돌 | `reasoning` | 낮음 |

제외 확정(REPORTED): JMMMU·EMMA(MMMU 파생), VisualWebInstruct·MathV360K·ArxivQA·SciGraphQA·PathMMU·MMR1-SFT·R1-OneVision·LLaVA-CoT(상위 모델 생성), CMMMU/OlympicArena test(정답 없음).

**현실 제약 (REPORTED, data 보고서)**:
- 새 출처 1개 추가 = 어댑터 코드 → `approval.code_hashes()` 변경 → 새 학습 커밋·승인 → 다운로드 → 오염 스캔(SSCD/SigLIP, 약 40분/2샤드) → **미판정 후보 판정이 병목**(지난 풀에서 11,751개 중 51% 제외, UNRESOLVED는 family째 제외) → 품질 → 동결 → 선정 → 번들. 어댑터가 이미 있는 출처(tatqa-table, circuitvqa, tqa 등)는 약 9단계·1~3일, 새 어댑터는 2~4일 이상.
- `TRAIN_COUNT=3500`, `PILOT_COUNT=300`이 코드 상수(`frozen_data.py:12-13`)라 크기 변경도 코드 변경.
- Architecture·Mechanical(투영도)·Art·Design·Sociology·History는 깨끗한 공개 데이터가 없다(`hf-pool-v3/plans/README.md:63-92` "zero eligible supply" VERIFIED). 직접 만들면 2026-10-30 "기존 데이터셋만" 결정과 충돌.
- 위 후보 대부분은 **벤치마크 test split**이다. 수업은 MMMU/Pro만 금지하므로 허용되지만, 학습에 쓰면 외부 dev로는 못 쓴다.

### 5.5 MMMU-Pro (Standard 10-way)
- 학습 풀에 5지 이상 객관식이 0개(VERIFIED). 102문항 진단에서 이미지 있음 40 vs 텍스트만 37(REPORTED)이라 현재 정답의 일부는 텍스트 의존. Pro는 텍스트로 풀리는 문항을 제거하고 10지선다라, 현재 파이프라인으로는 개선 기대 근거가 없다. 취득·해시·1회 보고까지만 계획하는 것이 정직하다.

---

## 6. 남은 예산으로의 현실적 계획

예산: 11-06 21:20Z 잔액 10.59달러(VERIFIED), v9 job 최대 38,000초(≈6.2달러) + 평가 최대 14,400초(≈2.4달러) → v9 종료 후 약 1~2달러. 마감 11-17.

### 우선순위 0 — 비용 0, 지금 할 것
- v9 900 결과를 B2 558과 **쌍 비교**(+/−, p, 과목별·출력 길이·한도 도달·NO_PARSE)로 보고. 537/544/522와 같은 표에.
- CPU 재채점 진단(§5.1): 공식식 무작위 대체의 기대 +22문항을 별도 version으로 기록. 67.4 차이 설명용.
- v9 묶음별 `stats.rl.anchor`, 소스별 32토큰 미만 비율, A2 행 수·내용 판독(표본 AI 검토, `human_verified=false`).
- 통과 기준(사전 고정): v9 ≥ 559이면 "문턱 통과·개선 입증 아님", 573(=558+SE 1배 ≈ 15) 미만이면 잡음과 구별 불가로 명시.

### 우선순위 1 — 사용자가 약 10~15달러 추가 시: "신호 수정판 v10" 1회
- 변경(코드만, 데이터 재스캔 없음): 잘림·루프 −1, 문항별 F0 기준선(기록된 결과 사용, 없는 문항은 0), A2 가중 행 수 비례, CLEVR·MapQA 보상 풀 제외(재가중은 이미 스캔된 항목 내에서, 약 0.5~1일 로컬), 백그라운드 저장.
- 비용·시간: 1,200문항 ≈ 75묶음 × 7~8분 ≈ 9~10시간 × 0.59 ≈ 6달러 + 평가 2.4달러 + 설정 ≈ **9~10달러, 약 13시간**.
- 기대: 문헌과 과거 결과로 보면 −2~+2pp, 가장 가능성 높은 결과는 0~+1pp(판단, 중간 확신). 잘림 −1이 85개 중 일부를 회복하면 +1~2pp 가능(낮음-중간).
- 리스크: 음의 신호가 길이를 줄이는 방향으로 작용할 수 있음(소스별 길이 guard 유지), 재생 검사 정지.
- 통과 기준(사전 등록): 900 1회 ≥ 573이고 쌍 비교 p < 0.05일 때만 "개선"으로 서술. 그 미만은 그대로 보고.

### 우선순위 2 — 20~30달러와 2~4일이 더 있을 때만: 약점 과목 데이터
- EEE-Bench + PhysUniBench + MedXpertQA-MM + TAT-DQA(어댑터 4개 신규, 라이선스·분할 확인 선행) → 로컬 오염 스캔·판정 → CLEVR·MapQA·GVLQA 대체.
- 비용: 로컬 작업 2~4일(판정 병목) + 학습 10~15달러.
- 기대: 과목 겹침이 생기지만 교사 없는 단일 표본 학습의 상한은 그대로(+1~3pp 판단, 낮음). 11-17 마감 안에 끝날지 불확실.
- 통과 기준: 우선순위 1과 동일.

### 하지 말 것
- GRPO/다표본(결정 번복 + 4~8배 비용), 자체 생성 데이터, LLM judge, Thinking 체크포인트, 체크포인트·seed 선택.

**정직한 최종 답**: 규칙(교사 금지·단일 답·MMMU 오염 금지)과 남은 예산 아래서 **67.4는 불가능**하고, 그 사실 자체(왜 안 되는가의 실측 근거)가 이 프로젝트의 보고 가치다. 558 초과도 반반 이하다. 보고서에는 537/544/522/v9를 같은 조건 표로 놓고, 손실의 25%가 미완료 답, 46%가 학습 데이터에 없는 과목, 노트 정확도 2/31이라는 수치로 "사람처럼 오답노트를 쓰는 학습이 4B에서 왜 안 일어났는가"를 설명하는 것이 맞다.

---

## 출처

코드·데이터 (VERIFIED):
- `/mnt/c/projects/multimodel/results/scoring-v8-20260924/summary.json`, `subject_scores.csv`
- `/mnt/c/projects/multimodel/analysis_exports/mmmu_20260924/team_final_v8/official_b2/predictions.jsonl`
- `/mnt/c/projects/multimodel/fine_tuning/error_note/protocol.py` (:27,158,170-171,204,221), `train_v4.py` (:557-561,575-578,586), `rl.py` (:92-98), `single_vllm.py` (:133,141), `collect.py` (:303-313), `single_campaign.py` (:498-537)
- `/home/jang2/mmdl-artifacts/self-note-single-20261105/final3500-v1/lot-v9/order.json`
- `/home/jang2/mmdl-artifacts/self-note-single-20261105/pod-v8-main-hgqwu5eo8tb0r9/x/single-v5/campaign/block-*/update-stats.json`, `results.tar:runs/single-v5/logs/main.log`
- `/home/jang2/mmdl-artifacts/notebook-rl-local-nf4-20261021/hf-pool-v3/plans/README.md`
- `/mnt/c/projects/multimodel/docs/PROJECT_STATUS.md` (:100-136), `docs/SOURCE_REVIEW.md` (:27,35,40), `CLAUDE.md` §2, §3.0, §4, §11
- `/mnt/c/projects/multimodel/claudedocs/analysis_single_v7_cycle2_20261106.md` (:9,42,113,120)
- `/home/jang2/.claude/projects/-mnt-c-projects-multimodel/memory/single-rlsft-decisions-20261105.md`

하위 보고서 인용 (REPORTED): weakness / data / compliance / datasets / methods 보고서(이 작업의 입력), 스크래치 `/tmp/claude-1000/-mnt-c-projects-multimodel/ac0a8f9a-74c8-45ee-acb1-756790005007/scratchpad/`.

외부 (REPORTED, UNVERIFIED): Qwen3-VL 기술보고서 arXiv:2511.21631; CARE arXiv:2512.19554; VL-Rethinker arXiv:2504.08837; ThinkLite-VL arXiv:2504.07934; 자기수정 한계 arXiv:2311.08516, 2310.01798, 2409.12917; DAPO arXiv:2503.14476; Dr.GRPO arXiv:2503.20783; STaR arXiv:2203.14465; HF 데이터셋 카드(EEE-Bench, PhysUniBench, MedXpertQA, TAT-DQA, FinMME, M4U, MusiXQA, AgroBench).

## 확신도 요약
- 규칙 준수 3항목 PASS: 높음
- v9가 607 불가: 높음 / 558 초과 50% 미만: 중간(판단)
- 손실 구성(A2 지배, KL 꺼짐)·보상 분포·묶음 시간: 높음(실측)
- 노트·재풀이 내용 정확도(2/31, 1~2/64): 높음(문서) / 그 해석: 중간
- 학습 풀 과목 공백(13/30, 오답 46%): 높음
- 프로토콜 몫 +21.9문항: 계산 높음, 67.4 적용 여부 UNVERIFIED
- 외부 데이터셋 라이선스·오염·방법 효과 추정: 낮음-중간(미검증)
- 비용·시간 추정: 중간(v8 실측 기반 외삽)

저장소 파일 변경 없음. 스크래치 추출물: `/tmp/claude-1000/-mnt-c-projects-multimodel/ac0a8f9a-74c8-45ee-acb1-756790005007/scratchpad/v8log/runs/single-v5/logs/main.log`.