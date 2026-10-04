# V8 평가 파이프라인 완료 기록 — 2026-09-24

범위: V8 구현, 실제 온라인 평가/CPU 재채점 연결, 세 실행2700개 재채점,
소스·원문 불변·산술 검증, Assignment 보고서 갱신. GPU/LLM/학습/본격 오답 분석은 하지 않았다.

| 실행 | V6 | V7 | V8 | V6→V8 증가/감소 |
|---|---:|---:|---:|---:|
| 2K | 479 | 463 | 478 (53.11%) | +1/−2 |
| 이전32K | 558 | 539 | 554 (61.56%) | +1/−5 |
| 고해상도32K | 560 | 539 | 558 (62.00%) | +2/−4 |

각 분모900. 고해상도 실행을 미리 지정한 제출 baseline으로 유지한다.
V8이 V6보다 높다는 통과 기준이나 여러 파서의 정답 OR는 사용하지 않았다.

## 동작·검증

- `final_answer_parser_v8.py`: gold-blind 단일 최종답 추출, 선언 순서/충돌/철회/정정,
  후보에 묶인 부정·복수 후보 검사, 정상 Markdown/수식/자기 blockquote/보기 내용 지원.
- 단일 주관식 전체 비교, pinned MMMU float round2. substring/다수 숫자 후보 채점 제거.
  질문으로 유도한 단위·step·point만 지원, 일반 단위 환산·방향 수식어 제거는 하지 않는다.
- `mmmu_val_v8.yaml`: B2와 parser/protocol ID만 다르다. 이전 protocol과 raw는 불변.
- pipeline167 unittest/Ruff/mypy25/Bash 검사 PASS. 신뢰 경계 손상 검사와 mocked
  online V8→저장→CPU 동일 채점 검사를 포함한다. 실GPU 재검사는 아니다.
- 원본/출력 각각900 ID,30×30, 샘플hash·JSONL일치, 비채점 필드 불변, scorer snapshot,
  다시 계산한 성적표/CSV 검증 PASS. gold 교체2700회에서 추출 불변.
- scoring SHA: `b80a28a4b66906d5c4f1ece046bd7c4be2b447445da4b072144e52ce7ee047fa`.
- V8 module SHA: `2beb1230283cb882f47d6e8af72b01180847fd7aa4b7ac640a5173e391f3f891`.

## 기존64건 회귀와 양방향 변화

V7 명확 누락46 중39 보존, 정책 누락2/2 회수, 유효 V7 증가4/4 보존.
나머지7개는 세 실행의 `Manage_20` 방향 수식어 보존3개와 R2의 명시 정정 없는 충돌4개다.
기존 감사는 답 주변을 중심으로 했으므로 전체 선언의 일관성까지 인증한 독립 라벨은 아니다.
이 자료를 개발에 사용했으므로 holdout 성능이라고 부르지 않는다.

V6 대비 정오 변경15개는 다음으로 전수 설명했다. 이는 채점 변화 검토이지 모델 오답 원인 분석이 아니다.

| 실행/ID | 변화 | 이유 |
|---|---|---|
| R1 Manage_17 | 증가 | 동일65 문장/boxed 표시 지원 |
| R2 Psychology_20 | 증가 | 제한된 embryonic-stage 결론 패턴 |
| B2 Basic_Medical_Science_17 | 증가 | 유효 label이 아닌 E를 정확 보기 내용으로 매핑 |
| B2 Clinical_Medicine_3 | 증가 | 말미 label+보기+체크 표기 지원 |
| R1/R2/B2 Manage_20 | 감소3개 | `2960 unfavorable`의 방향 수식어를 임의 삭제하지 않음. 실제로 지원하지 않는 정상 표현이 남는 한계 |
| R2 Energy_and_Power_26 | 감소 | B→C 선언, 명시적 정정 관계 없음 |
| R2 Energy_and_Power_3 | 감소 | B→C 재계산, 명시적 정정 관계 없음 |
| R2 Materials_10 | 감소 | D→A 선언, 이 실행의 원문에는 명시적 정정 없음 |
| R2 Public_Health_7 | 감소 | A/B 및 둘 다 옳다는 문장이 반복, 명시적 정정 없음 |
| R1 Materials_28 | 감소 | D 뒤 B도 검토하는 미완결 대안, length |
| B2 Art_21 | 감소 | 장문 B/C 충돌, length |
| B2 Biology_20 | 감소 | 장문 D/C 충돌, length |
| B2 Pharmacy_12 | 감소 | 장문 B/A 충돌, length |

감소11건을 모두 'V6가 틀린 답을 맞다고 한 사례'라고 주장하지 않는다.
3건은 수식어 지원 한계이고,8건은 모호성/충돌 정책 차이다. 이를 숨기거나 점수 때문에
방향·충돌 규칙을 변경하지 않았다. 남는 미지원 표현의 확대는 별도 고정 정책·검증이 필요하다.

## 개발 중 검출한 회귀

수식 첫 행만 읽기, 같은 답의 문장/boxed 변형을 충돌로 처리하기, contracted/modal 부정,
정정 뒤 철회, named-point 부정, 명시/질문상 암묵 단위의 충돌, `correct myself`,
선택지와 무관한 수사적 `? No` 오인 등을 양성/음성 쌍으로 고쳤다.
중첩 주관식 재귀 오류는8단계 제한과 감소 검사로 차단했고1500단계 합성 반례를 추가했다.
실패 후보는 `v8_development_candidate_1/2`에 남고 최종과 구분한다.
두 번째 후보는 실행 중 소스 hash 변경 감지가 실제로 게시를 거부했다.

## 산출물·재현

- 내부 전체: `analysis_exports/mmmu_20260924/team_final_v8/` 아래 세 run별
  `samples/`, `predictions.jsonl`, `review.html`(text-only), `summary.json`,
  `subject_scores.csv`, `source.json`, `scorer_source.json`, `scorer_source/`,
  `verification.json`, `v6_comparison.json`, `historical_regression.json`.
- 작은 공개 집계: `results/scoring-v8-20260924/{summary.json,subject_scores.csv,provenance.json}`.
- 재실행: `python -m scripts.audit_v8 --v6-root analysis_exports/mmmu_20260924/v6_v7_audit/v6_replayed --audit-root analysis_exports/mmmu_20260924/v6_v7_audit --output "$NEW_V8_SUITE_DIR"`.
- 공개본에는 이 내부 감사문/AGENTS/docs/P1/raw를 넣지 않는다. 보고서와 README에 필요한
  정책·검증 한계만 모으고 `scripts/sync_submission.py`로 MMDL 제출본을 관리한다.
