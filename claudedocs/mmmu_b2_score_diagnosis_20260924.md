# 32K R2와 고해상도 B2 점수 차이 진단 — 2026-09-24

## 후속 승인: V5 수정·CPU 재채점 완료

아래 진단 후 사용자가 파서 수정/재채점을 승인했다. V4 동작은 보존하고
`team-final-answer-v5`를 명시적으로 선택하는 재채점 경로를 추가했다.
V5는 correct/best 문구를 무조건 우선하지 않고 선언의 등장 순서를 사용하며,
가정형 후보는 마지막 선언을 고르기 전에 제외한다. 이 규칙을 Final 구간 내부에도 동일하게 적용한다.
나머지 충돌/length/open 비교 규칙과 모델 응답은 유지했다. 점수를 본 뒤 규칙을 다시 조정하지 않았다.

| 전체 저장 응답 | V4 | V5 | 오답→정답 | 정답→오답 |
|---|---:|---:|---:|---:|
| R2 |556/900 · 61.78%|558/900 · 62.00%|2|0|
| B2 |557/900 · 61.89%|561/900 · 62.33%|5|1|

R2의 추출 변경 5개 중 정답 회수는 Chemistry_24/Materials_11이다.
B2의 추출 변경 8개 중 Chemistry_24/Mechanical_Engineering_29/Physics_25의 명확한 최종답을 회수했다.
Electronics_7/Energy_and_Power_21도 고정 정책상 명시 답을 추출해 정답이 됐지만 두 응답은 length다.
반대로 B2 Electronics_25는 마지막 답 75 V와 gold 50이 달라 오답으로 바로잡았다.
중간 구간의 50을 잡아 정답 처리했던 결과를 유지하지 않았다.
이는 채점 수정 효과이지 모델/풀이 능력 향상이나 잘림 해결이 아니다.

기존 V4 출력은 저장 1,800개 및 독립 검토의 합성 14,760개에서 동일했다.
V5 회귀 검사 15개를 포함한 전체 126 tests, Ruff, mypy 24 source files, Bash 검사 통과.
두 V5 원장의 900개 coverage/record hash/원응답 및 비채점 필드 불변/독립 재채점/과목 산술을 검증했다.
변경별 근거·코드 스냅샷·재현 명령은
`analysis_exports/mmmu_20260924/team_final_v5/README.md`와 `v4_to_v5_comparison.json`에 있다.
V5 scoring hash: `eb1cfa1f3f75e783205e0a842db387dbe9671ccbdb2ba83c79cf1cae58666543`.
RunPod/GPU 재추론/LLM 호출/Git commit·push는 하지 않았다. 아래의 '미수정' 서술은 선행 진단 당시 기록이다.

## 결론

**과거 64.89%와 이번 61.89%는 서로 다른 채점기 결과다.** 같은 V4로 비교하면
R2 556/900(61.78%) → B2 557/900(61.89%)로 사실상 같다. 같은 감사 V2로 비교하면
584/900(64.89%) → 575/900(63.89%)다. 이미지 확대의 뚜렷한 정확도 향상은 관측되지 않았다.
V4가 무조건 더 정확한 파서라는 뜻도 아니다. 실제 최종답을 놓치는 우선순위 결함을 재현했다.

이번 작업은 저장된 전체 응답의 CPU 진단이다. 평가 코드/확정 채점 정책/원본 결과를 수정하지 않았고,
새 GPU 추론, 유료 작업, 학습, LLM judge, Git commit/push를 하지 않았다.

## 1. 같은 응답 × 같은 채점기 대조

| 저장 응답 | 원래 legacy 규칙 | 감사 V2 | 현재 V4 |
|---|---:|---:|---:|
| R2: 이전 32K, 작은 이미지 budget |459/900 · 51.00%|584/900 · 64.89%|556/900 · 61.78%|
| B2: 새 32K, 큰 이미지 budget |443/900 · 49.22%|575/900 · 63.89%|557/900 · 61.89%|

모든 칸은 전체 900개 원응답을 CPU로 대조했다. V4는 양쪽 저장된 parsed/status/correct와
완전히 일치했다. B2의 V2 적용은 먼저 동일 legacy scorer로 parsed/correct를 재구성한 복사본에
수정하지 않은 `docs/P0/MMMU_audit/audit_parser.py:evaluate_row`를 적용했다.
이는 V2가 open의 입력 판정을 그대로 유지하고 MCQ legacy 일치를 검사하기 때문이다.
B2의 V4 정오 판정을 V2의 open 입력에 그대로 넣어 다른 채점 정책을 혼합하지 않았다.

과거 V2 584와 이번 V4 557의 차이 27개는, 과거 응답을 기준으로
**채점 정책 변경 −28개 + 새 응답의 V4 기준 변화 +1개**로 분해된다.
새 응답을 기준으로 분해하면 **V2 기준 응답 변화 −9개 + V2→V4 −18개**다.
이 분해는 점수 회계이며 이미지 확대의 단독 인과효과 추정은 아니다.

### V2와 V4가 갈리는 이유

V2는 상한 종료 MCQ에서 legacy 전체 응답 검색을 유지하고, open도 옛 판정을 유지한다.
V4는 명시적인 최종답을 추출하고 open도 최종 구간만 공식 evaluator에 전달한다.
상한 종료 자체를 무조건 오답으로 만들지는 않는다. B2의 length 90개 중 V4 정답도 7개다.

| 같은 응답 내 비교 | R2 | B2 |
|---|---:|---:|
| V2만 정답 |32|27|
| V4만 정답 |4|9|
| V2만 정답: length 우선 분류 |23|18|
| V2만 정답: 자연 종료 open |6|4|
| V2만 정답: 자연 종료 MCQ |3|5|

위 원인 분류는 length→open→MCQ 순서로 중복 없이 집계했다.
V2-only 32개 전부를 '가짜 정답'으로 단정하면 안 된다. legacy의 중간 후보 일치도 있고,
V4의 실제 최종답 누락도 있다. 더 높은 점수인 파서를 고르는 것으로 정책을 정하지 않는다.

구체적으로 R2 Electronics_25는 gold 50인데 최종 boxed 답은 100 V다. V2는 앞의 50도
후보에 포함해 정답으로 인정했다. Pharmacy_19도 최종 Step D와 gold B가 다른데 정답으로 처리했다.
반대로 R2 Chemistry_24는 마지막 Answer D가 맞는데 V4가 놓쳤다.
R2 length만 보면 V2 24정답 → V4 1정답으로, 두 scorer의 순차이 28개 중 23개가 이 구간에서 생겼다.
이는 기존 64.89%가 모두 '최종답을 제대로 추출해 맞은 점수'라는 전제가 성립하지 않음을 보여준다.

## 2. 이미지 입력 변경은 실제로 적용됐다

- 지정 모델/revision/BF16, 데이터 900 IDs, 질문/보기/정답, P0/chat prompt, seed,
  문항별 요청 sampling 설정은 양쪽 모두 같다. 출력 상한 32,768과 동시 요청 2도 같다.
- 변경: HF 직접 처리 → qwen-vl-utils 0.0.14 후 pinned HF processor;
  pixels 262,144–1,310,720 → 1,003,520–4,014,080;
  전체 문맥 36,864 → 40,960. GPU는 모두 3090이지만 호스트/driver는 다르다.
- 전체 입력 토큰 534,018 → 1,193,656(2.24배).
  저장 grid의 `t*h*w/4`로 계산한 시각 토큰은 433,177 → 1,092,815(2.52배).
- 900개 입력 tensor fingerprint가 모두 바뀌었다. 36개는 grid/토큰 수가 같아도 pixel 값이 다르다.
  원본 이미지 파일/참조 순서는 900개 모두 일치한다.
- B2 900개 모두 사전 CPU C의 토큰 수, grid, processor tensor hashes와 일치했다.
  vLLM 반환 prompt token IDs도 CPU 입력과 일치했다. 최대 입력 5,627이다.
- 이미지 사용 건수 982개, 고유 이미지 959개다. 누락/시스템 오류/OOM/preemption 또는
  입력 잘림의 증거는 평가 로그에서 발견되지 않았다.
- 한계: `actual_engine_pixel_tensors_verified=false`다. CPU 이미지 tensor와 엔진 prompt IDs를
  확인했지만 vLLM 내부 pixel tensor 자체를 직접 계측한 것은 아니다.

## 3. 정확도 증가분과 감소분이 상쇄됐다

V4로 같은 ID를 대응하면 **양쪽 정답 497, 새로 정답 60, 새로 오답 59, 양쪽 오답 284**다.
객관식은 540/847 → 540/847, 주관식은 16/53 → 17/53이다.

과목별로 Chemistry 9→14, Clinical Medicine 19→22, Psychology 22→25는 증가했으나
Computer Science 20→16, Accounting 23→20 등은 감소했다(모두 과목당 30개).
예: Computer_Science_16은 gold D에서 이전 D → 새 명시적 Final Answer B로 변했다.
이 사례는 파싱 실패가 아니라 모델이 선택한 답 자체의 변화다. 시각 오독과 논리 오류 중 어느 것이
최초 원인인지는 이 통계만으로 확정하지 않는다.

입력 전처리/해상도 변경과 단일 sampling 실현, 호스트 차이가 함께 존재한다.
따라서 '해상도를 높여서 성능이 떨어졌다' 또는 'sampling 탓이다' 중 하나로 단정할 수 없다.
같은 seed라도 입력이 달라진 비교이며 동일 응답을 기대하는 통제는 아니다.
기존 A/B/C 각 6문항은 오류/메모리/입력 경로 검증용이고 모두 2정답이었다.
이 결과를 정확도 개선을 입증한 실험이라고 해석하지 않는다.

## 4. 장문 상한 도달은 여전히 큰 실패 지점이다

| 관측 | 이전 R2 | 새 B2 |
|---|---:|---:|
| length |75|90|
| length 중 V4 오답 |74|83|
| 자연 종료 후 NO_PARSE |15|15|
| 전체 NO_PARSE |84|89|
| 전체 생성 토큰 |3,719,200|4,116,922|

새 B2는 41개가 stop→length, 26개가 length→stop, 49개는 양쪽 length였다.
59개 정답 손실 중 17개는 새 응답이 length였고, 그중 16개는 이전에 자연 종료했다.
반대로 이전 length였던 문항 중 이번 정답이 된 것도 16개다.
따라서 length 증가만으로 전체 정답 변화 −27개를 설명할 수는 없다.

B2의 상한 종료 90개가 생성 토큰의 71.63%를 사용했다.
Accounting_9는 이전 C 정답에서 새 응답은 재고/현금 해석을 반복하다 끝나지 못했고,
Architecture_and_Engineering_19는 새 응답 말미가 ASCII 도식 반복 상태로 잘렸다.
이는 '길이가 모자란 정상 풀이'만 있다고 보기 어려운 구체적 사례다.
반복 발생 자체의 원인이나 해상도와의 인과관계는 추가 통제 없이 확정하지 않는다.

현재 343개 오답을 겹치지 않게 분류하면 length 83, 자연 종료 미추출 15,
자연 종료의 추출된 MCQ 불일치 221, 추출된 open 불일치 24다.
추출된 답의 불일치에도 파서 오선택 가능성이 있으므로 전부 모델 추론 오류라고 자동 확정하지 않는다.

## 5. 확인된 V4 최종답 누락 결함

`src/mmdl/evaluation/final_answer_parser.py:94`의 `_asserted_spans`는
`CORRECT markers or ANSWER markers`로 후보를 고른다. 앞의 correct-answer 문구가 존재하면
뒤의 명확한 Answer 문구를 후보에서 제외할 수 있다. 선택한 앞 문구가 가정 표현으로 걸러져도
나중의 Answer로 다시 탐색하지 않는다. 이 우선순위 때문에 다음 자연 종료 정답을 놓쳤다.

| B2 문항 | 응답 끝의 명시 답 = gold | V4 관측 |
|---|---|---|
| Chemistry_24 |D|앞의 가정형 correct-answer 후보를 버리고 뒤 Answer를 놓쳐 NO_PARSE|
| Mechanical_Engineering_29 |B|앞의 correct-answer 다음 구분선 `---`를 답으로 읽고 뒤 Answer를 놓침|
| Physics_25 |D|앞의 `Real / Larger`가 보기 `Real/Larger`와 불일치; 뒤 Answer D를 놓침|

이 세 사례는 최종 label과 gold의 명시적 일치를 직접 확인했다. 풀이 전체의 사실성 검증은 아니다.
아래 최소 재현은 정답을 추출기에 전달하지 않으며 파일 변경 없이 실행했다.

```python
from mmdl.evaluation.final_answer_parser import extract_mcq
options = ["red", "blue", "green", "yellow"]
assert extract_mcq("Answer: D. yellow", options)["answer"] == "D"
raw = "Perhaps the correct answer is yellow.\n\nAnswer: D. yellow"
assert extract_mcq(raw, options)["answer"] is None  # 현재 결함 재현; 원하는 동작 아님
```

기존 `test_final_answer_parser.py`의 30 tests는 통과했지만 위 상황을 잡지 못했다.
V4가 V2보다 모든 형식에서 우월하다고 말할 근거가 없다.

## 6. 다음 조치 — 아직 실행하지 않음

1. 실제 결함을 점수와 독립적인 합성 예제로 회귀 고정하고, 명시적 답 우선순위/충돌/length 정책을
   합의한 새 scorer 버전으로 수정한다. 옛 판정은 보존한다. LLM 판정이나 GPU 재추론은 필요 없다.
2. 같은 새 scorer로 두 전체 900 응답을 모두 CPU 재채점하고 정답→오답/오답→정답을 함께 공개한다.
   64.89%를 맞추기 위해 파서를 조정하거나 유리한 판정만 선택하지 않는다.
3. 그래도 남는 오류는 이미지 인식, 계산/지식, 답 추출, 반복/미완료를 사례별 분리한다.
   이미지 효과를 주장하려면 같은 실행환경의 한 요인 통제와 고정된 표본/반복이 별도로 필요하다.
   이번에는 추가 유료 실험을 시작하지 않았다.

## 증거와 재현 식별자

- R2 원응답: `analysis_exports/mmmu_20260923/originals/continuous32k-complete-20260923/extracted/runs/mmdl-cont32k-analysis-20260922-analysis/predictions.jsonl`
- R2 V4 원장: `analysis_exports/mmmu_20260923/team_final_v4/continuous_32k/`
- B2 원응답/summary/config/backend/environment: `analysis_exports/mmmu_20260924/official_b2_complete/extracted/runs/mmdl-official-b2-20260923-evaluation/`
- CPU C 원장: `analysis_exports/mmmu_20260923/input_audit_v2/records.jsonl`
- R2 원응답 SHA256: `5f0c6915516e9b82f06a5971be23aa72e116726bc613df4da9dcd94520bfa73d`
- B2 원응답 SHA256: `5dc87237edc66efac96bb58c2afbccdb023567c5e62644e388d48488144a2d42`
- V2 파서 SHA256: `2e86359da8c47951e084851b6add0a6c20d9116aab2d986f36a823b93e96903c`
- V4 파서 SHA256: `4ee95ba0d5d9ed27acc01911c27235caf7f192d9187e32e339863f140629dc08`
- R2 추론 commit: `702ed44c4c45f454746ec4b371ad1b7403b67856`
- B2 추론 commit: `6d6d4a1a4bed1785e8770902d65e12b54340ba75`
