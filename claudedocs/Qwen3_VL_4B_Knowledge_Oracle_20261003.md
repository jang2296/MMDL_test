# 다분야 Knowledge Oracle — 실제 실행 결과

2026-10-03. 내부 진단 전용이며 MMDL/GitHub에 게시하지 않는다. 모든 자료는
`BENCHMARK_DERIVED`, `training_allowed=false`로 관리한다. 가중치 업데이트는 없다.

## 1. 결과 요약

RTX 3090에서 동일한 Qwen3-VL-4B-Instruct checkpoint를 이미지 없이 사용했다.
15분야마다 서로 다른 두 능력을 선정해 K1 기초 지식 2문항, K2 명시된 상태에 대한
규칙 적용 2문항을 실행했다. 전체 60호출을 완료했고 누락·시스템 오류·잘림·진단 파싱 실패는 0이다.

| 새 시험 | 정답/전체 | 정확도 |
|---|---:|---:|
| K1 — 기초 지식 | 30/30 | 100.00% |
| K2 — 상태→규칙 적용 | 30/30 | 100.00% |

**이 결과는 선택한 기본 지식과 단순 규칙 문제의 최종답을 맞혔다는 뜻이다. 전문지식 전체,
복잡한 추론 또는 중간 풀이까지 모두 정확하다는 뜻이 아니다.** K1/K2는 자체 작성한 더 단순한
문제이며 기존 MMMU I/O1과 같은 문항·난도의 통제 조건이 아니다.

## 2. 분야별 결과와 기존 조건의 맥락

I는 기존 MMMU 실제 이미지 조건, O1은 독립 AI 검토된 해석 전사 조건이다.
아래 두 열은 같은 분야의 **다른 문제** 결과를 재사용한 것이며 K1/K2와의 paired 차이가 아니다.
O1은 정답 노출·미승인·적용 불가 조건을 제외하므로 I와 분모도 다르다.

| 분야 | 이번에 점검한 능력 | K1 | K2 | 기존 I | 기존 O1 |
|---|---|---:|---:|---:|---:|
| Music | 음정·음가 | 2/2 | 2/2 | 2/8 | 3/4 |
| Electronics | DC 커패시터·병렬 저항 | 2/2 | 2/2 | 4/8 | 5/6 |
| Materials | 응력·변형률 | 2/2 | 2/2 | 3/8 | 4/8 |
| Computer_Science | DFA 전이·선형 탐사 해시 | 2/2 | 2/2 | 3/8 | 2/6 |
| Biology | DNA 상보·멘델 유전 | 2/2 | 2/2 | 2/8 | 1/3 |
| Accounting | 재료 흐름·현금 잔액 | 2/2 | 2/2 | 3/8 | 3/8 |
| Physics | 속도 그래프 면적·알짜힘 | 2/2 | 2/2 | 0/4 | 0/4 |
| Public_Health | 특이도·발생 비율 | 2/2 | 2/2 | 1/4 | 1/3 |
| Diagnostics_and_Laboratory_Medicine | 과형성·고전적 세포사멸 특징 | 2/2 | 2/2 | 1/4 | 2/2 |
| Architecture_and_Engineering | 도면 축척·영상 중첩 | 2/2 | 2/2 | 1/2 | 1/2 |
| Mechanical_Engineering | 토크·스프링 에너지 | 2/2 | 2/2 | 1/2 | 0/1 |
| Geography | 등고선·적도와 위도 | 2/2 | 2/2 | 1/2 | 1/1 |
| Chemistry | 몰농도·산화수 | 2/2 | 2/2 | 0/2 | 1/2 |
| Math | 볼록성·그래프 동형의 차수 불변량 | 2/2 | 2/2 | 1/2 | 0/1 |
| Energy_and_Power | 열역학 에너지 수지·수력 | 2/2 | 2/2 | 0/2 | 1/2 |
| **이번 15분야 합계** | 목적 표본, 전체 MMMU 정확도가 아님 | **30/30** | **30/30** | **23/72** | **25/53** |

기존 15분야 집계: I 31.94%, O0 47.54%(29/61), O1 47.17%, M 33.33%(24/72),
MI 36.11%(26/72). 기존 전체 30분야 102문항의 I 39.22%(40/102), O1 44.93%(31/69)와
구분한다. 어떤 집계도 원래 제출 baseline 558/900, 62.00%를 대체하지 않는다.

## 3. 최종답이 맞아도 풀이가 틀릴 수 있다

독립 Sol 검토에서 60개 원문 최종답과 동결 정답키를 전부 확인했고, 이번 자료에서 관측된
진단 파서의 false positive/false negative는 0/0이었다. 그러나 다음 풀이 오류도 확인했다.

- `knowledge_Computer_Science_K2_01`: 문제에서는 y 입력이 상태를 유지하지만 모델 풀이에서는
  y가 상태를 바꾸는 것으로 두 번 계산했다. 최종 q1/accept 선택은 우연히 맞았다.
- `knowledge_Math_K2_02`: 경로 그래프의 중간 정점 w를 차수 1로 잘못 계산했다.
  별 모양 그래프와 동형이 아니라는 최종 선택은 맞았다.
- `knowledge_Music_K2_02`: 중간 음가 계산을 틀렸다가 응답 안에서 정정했다.
- `knowledge_Public_Health_K1_02`: 정답 정의는 맞지만 incidence proportion을 rate와 혼용하는
  부정확한 설명이 있었다.

따라서 K2 100%를 완전한 규칙 적용 능력의 인증으로 해석하지 않는다. 분야당 각 단계 2문항,
기초 문제 위주의 목적 선정, 단일 seed 반복과 천장 효과 때문에 분야별 차이나 일반적 오류율을
추정하기에도 부족하다. 독립 검토는 AI 보조 검토이지 사람 인증이 아니다.

## 4. 현재 지지되는 해석

1. **확인된 현상:** 이번 기초 지식·단순 규칙 최종답 60개는 모두 맞혔다. 원래 이미지 문제의
   실패를 기초 사실 지식의 부재 하나로만 설명하기 어렵다.
2. **복잡한 적용 부담도 남는다:** 기존 해석 전사 O1에서도 실패가 많고 새 K2 풀이에도 오류가
   있으므로, 시각 정보만 정확히 주면 모든 실패가 사라진다는 가설은 지지되지 않는다.
3. **구조 추출의 정보 손실 가능성:** 기존 102문항 paired 결과에서 I→M 회복 10/악화 17,
   M→MI 회복 15/악화 9다. P 텍스트만으로는 원본 정보가 충분히 보존되지 않을 가능성과
   일치하지만, 표본의 전이 차이 불확실성 구간은 0을 포함한다.
4. **모듈 책임은 미확정:** K1/K2와 I/O1은 난도와 문항이 다르다. 이 비교로 Vision Encoder,
   최종 Merger/DeepStack Merger 또는 언어 Decoder 중 하나를 유일한 원인으로 확정하지 않는다.
   P도 전체 VLM을 통과하므로 순수 Vision 평가가 아니다.

P의 69.74%는 O0 AI 참조 대비 항목 단위 지원율(1309/1877)이다. 문항 정답률 또는 순수
시각 인식률과 동일한 척도가 아니므로 K1/K2/I/O1의 정확도와 같은 축에 놓지 않는다.

다음 후보는 기본 상식만 더 학습하는 방식보다는 전문 기호·관계의 정확한 상태 표현과 복잡한
규칙 적용을 함께 점검하는 것이다. 다만 실제 selective fine-tuning 대상은 난도와 목표를
맞춘 추가 실험 및 독립 외부 학습 데이터 계획을 검토한 뒤 정한다. 이번 작업에서는 학습하지 않았다.

## 5. 실행 조건·비용·검증

- 모델: `Qwen/Qwen3-VL-4B-Instruct`, revision
  `ebb281ec70b05090aa6165b016eac8ec08e71b17`; BF16·비양자화·가중치 고정.
- 실제 RTX 3090, vLLM 0.11.0. 이미지/전사/정답키/파트너 응답을 추론에 넣지 않았다.
- sampling: temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0,
  presence_penalty 1.5. base seed 3407, 기존 문항 ID별 SHA256 정책. 각 호출은 독립 대화다.
- 별도 protocol: `qwen-knowledge-rule-probe-v1`, 최종 출력 상한 4096,
  P0에 진단용 `FINAL_ANSWER` 형식을 추가했다. 원래 32768 baseline과 다르다.
- 00:47:27–00:50:02 UTC 실행, 모델 준비를 포함한 호출 프로세스 약 2분 35초.
  완료 요청 시간 합 110.966초, 입력 7,660/출력 4,301토큰. 설치·전체 Pod 과금 시간은 아니다.
- 기존 runner를 재사용했고 3개 CPU 회귀, Ruff/mypy 및 독립 구현 검토를 통과했다.
- archive/member SHA256, 전체 60개 원문·입력·레코드 hash·seed·동결 bank·후속 채점을
  로컬에서 검증했다. 정답키는 Pod에 업로드하지 않고 로컬 후속 채점에서만 사용했다.
- 채점은 별도 `diagnostic-final-interface-v2-exact-option`이다. 원래 V8를 새 출력 형식에 적용한
  호환성 열을 이번 진단 정확도로 쓰지 않으며 기존 V8 코드는 변경하지 않았다.

근거:

- [주 검증 receipt](../analysis_exports/bottleneck_20261003/cloud_knowledge_v1/local_receipt.json)
- [독립 실제 원문 검토](../analysis_exports/bottleneck_20261003/knowledge_actual_independent_review_v1.json)
- [K1/K2 집계](../analysis_exports/bottleneck_20261003/knowledge_scores_v1/summary.json)
- [60개 원문](../analysis_exports/bottleneck_20261003/cloud_knowledge_v1/knowledge/run/responses.jsonl)
- [실제 입력 원장](../analysis_exports/bottleneck_20261003/cloud_knowledge_v1/knowledge/run/input_audit.jsonl)
- [기존 분야별 조건 결과](../analysis_exports/bottleneck_20261003/export_all30_102_v2/per_subject.csv)
- [기존 조건별 전이 집계](../analysis_exports/bottleneck_20261003/export_all30_102_v2/summary.json)

### Gemma와의 관계 — 종료 상태

본 문서의 K1/K2와 기존 MMMU 입력 개입은 두 진단 실험이다. Gemma는 이 둘을 채점하거나
Qwen을 다시 풀게 하는 모델이 아니라, **과거 고해상도 V8 baseline900응답의 별도 감사기**다.
RunPod에서900추출+2동치 호출을 완료했으나607RESOLVED/117REVIEW/176SYSTEM_ERROR로
293개가 미확정이다. 유효 전수 판정은 INCOMPLETE, 새 정확도는 null, 기존558/900은 유지한다.
Gemma 전체 가중치는 로컬에 회수하지 않았으며 코드·실제 응답·원장·환경·실패 로그만 보존했다.
결과를 로컬 검증한 뒤 소유Pod를03:03:45 UTC에 삭제하고 부재를 확인했다.
[Gemma 결과·한계](Qwen3_VL_4B_Bottleneck_Diagnostic_Report_20261003.md#5-gemma900시도의-실제-결과--전수-판정은-미완료),
[회수·삭제 증거](../analysis_exports/bottleneck_20261003/owned_pod_cleanup_final_v1.json).

<details>
<summary>이전 Gemma 진행 기록 — 최종 상태로 사용하지 않음</summary>

Gemma는 별도 감사 작업이다. 01:02 UTC의 실제 검증 27개 중26개 통과/1개 실패는
`cloud_gemma_failed_smoke_v7/`에 보존했다. 모호한 넓은 의미를 세부 보기 하나로 고른 실패였다.
이를 모든 보기의 지지/가능/배제/미확정 검사로 보강하고 원래 계약은 그대로 유지했다.
01:21 UTC 확인에서 새 실제29검증을 전부 통과했으며 저장900응답의 감사가8/900 진행 중이다.
이는 검증 gate 통과·실제 감사 시작의 증거이지 전체900 의미 정확도나 새로운 확정 점수가 아니다.
원래62.00%는 그대로 유지하며 본 문서의 K1/K2 결과는 Gemma 감사와 별개다.

01:30 UTC 추가 확인: 통과29 smoke를 로컬 회수·실제 호출과 입력/코드/모델 hash로 재검증했다.
다만 일부 의미 witness와 reason의 지시 준수가 부정확해29/29를 완전한 의미 판정 인증으로
보지 않는다. 실제900감사 중88개는59RESOLVED/18REVIEW/11SYSTEM_ERROR이며, 원문 인용 불일치
10개·미완료 응답 검사1개는 가드를 통과하지 못했다. 검증 실패를 정답으로 구제하지 않는다.
[실제 smoke 검증 근거](../analysis_exports/bottleneck_20261003/cloud_gemma_passed_smoke_v8/local_receipt.json).
[독립 실제 응답 검토](../analysis_exports/bottleneck_20261003/gemma_passed_smoke_v8_independent_review.json)도
같은29 최종 출력 계약의 통과와 의미 witness/reason의 한계를 확인했다.

### 재개 체크포인트 — 01:32:45 UTC /10:32 KST

Knowledge60의 실행·회수·후속 채점·독립 검토는 완료했다. Gemma 전체 감사는105/900이며
71RESOLVED/21REVIEW/13SYSTEM_ERROR로 아직 미완료다. 실행 코드/원문/가드는 동결 상태다.
소유3090 Pod `h447a4u6c0vq3k`의 controller7819/birth7298428이 실행을 관리하며,
03:52:38 UTC GPU cutoff·04:22:38 UTC 최종회수 기한은 그대로다. 다음 작업은 닫힌
전체 감사 원장/환경/비용의 회수·검증이며, 모든 필요한 결과가 LOCAL_VERIFIED가 된 후에만
이 Pod를 삭제한다. 검증 실패·REVIEW가 남은 상태에서62.00%의 대체 점수를 만들지 않는다.

</details>
