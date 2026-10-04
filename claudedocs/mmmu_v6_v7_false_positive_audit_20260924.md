# V6 재현·V7 정답 누락 및 거짓 정답 위험 감사

작성일: 2026-09-24. 대상: R1(2K), R2(이전32K), B2(고해상도32K)의 저장 응답 각900개.

## 결론

**V7의 점수 하락을 “V6의 거짓 정답을 제거한 결과”로 설명할 수 없다.**
V6 정답 → V7 오답60건 중 답 추출 실패46건, 엄격한 문자열 정책으로 인한 누락2건을 확인했다.
나머지12건은 답 번복·조건부 선택·잘림 때문에 확정 최종답을 인증할 수 없다.
이60건에서 명백한 V6 거짓 정답으로 확정한 사례는0건이지만, 이는 V6 전체에 거짓 정답이 없다는 뜻이 아니다.
V6는 합성 반례에서 부정된 선택지, 복수 후보, 문자열 부분 일치를 정답으로 인정했다.

사용자 요청에 따라 실행 경로는 V6로 롤백했다. **점수가 높아서 채택한 완전한 판정기**가 아니라,
확인된 V7 회귀를 제거한 작업 기준이다. 이번 감사에서는 V6 알고리즘을 추가 변경하지 않았다.

## 1. V6 전수 재채점

| 실행 | V6 재현 | 보존 V7 | V6 정답→V7 오답 | V6 오답→V7 정답 |
|---|---:|---:|---:|---:|
| R1: 2K | 479/900 (53.22%) | 463/900 (51.44%) | 17 | 1 |
| R2: 이전32K | 558/900 (62.00%) | 539/900 (59.89%) | 20 | 1 |
| B2: 고해상도32K | 560/900 (62.22%) | 539/900 (59.89%) | 23 | 2 |

원본 추론 결과에서 기존 `scripts/rescore.sh --parser team-final-answer-v6`를 다시 실행했다.
새2700개 레코드와 세 `predictions.jsonl` 파일 바이트가 보존 V6와 완전히 일치했다.
각900고유ID·30과목×30·record hash·정답 집계·원본 hash·비채점 필드 불변을 검증했다.
**동일 출력의 재현은 파서의 의미 정확성을 증명하지 않는다.**

- V6 scoring SHA256: `ab86eeebbf3efe5ed63b6108d097c32d1c38a74e478aef02a7035cf250a4f74f`
- V6 source archive SHA256: `ffd09c4a0315df1c1ee7c96db08958d21290c5bb58615393fd3d4686c62e1193`
- V6 소스는 보존 snapshot과 일치한다. HEAD `0f10c338f366ca2568595a06fe275a0f46dd0cb7`는 이 미커밋 구현의 commit이 아니다.
- 재현 검증 원장: `analysis_exports/mmmu_20260924/v6_v7_audit/replay_verification.json`.
- 전체132 unit tests 통과. 롤백 후 Ruff·mypy24 source files·Bash 검사도 통과했다.
  이러한 검사는 아래 자연어 반례가 없다는 보증이 아니다.

## 2. 판정이 바뀐64건의 원문 대조

| 실행 | V7 명확한 정답 누락 | 엄격 정책 누락 | 모호·잘림 | V7의 유효한 회수 |
|---|---:|---:|---:|---:|
| R1 | 13 | 0 | 4 | 1 |
| R2 | 17 | 1 | 2 | 1 |
| B2 | 16 | 1 | 6 | 2 |
| 합계 | **46** | **2** | **12** | **4** |

64건의 ID·중복 없음·양방향 변화가 실제 V6/V7 결과와 일치함을 별도로 검사했다.
검토는 저장된 질문·보기·gold·종료 사유·답 선언·말미 문맥에 대한 **AI 보조 텍스트 감사**다.
사람의 독립 인증이 아니며, 이미지를 다시 풀거나 모든 장문 중간 계산을 검산한 것이 아니다.
문항별 실제 읽은 범위와 불확실성은 아래 로컬 ledger에 기록했다.

- `baseline_review.json`: 18건
- `continuous_review.json`: 21건
- `official_b2_review.json`: 25건
- 공통 위치: `analysis_exports/mmmu_20260924/v6_v7_audit/`

“명확한 정답”은 **생성 응답이 선택한 답이 제공 gold와 맞는다는 의미**다.
최종 선택지는 맞지만 중간 풀이가 불안정한 경우, 풀이 오류만으로 파서의 거짓 정답이라고 부르지 않았다.
반대로 여러 답을 번복하다 잘린 경우에는 높은 점수를 유지하려고 명확한 정답으로 분류하지 않았다.
감사 분류를 임의로 가감한 새900점수는 만들지 않았다.

### 확인한 V7 실패 원인

| 결함 | 실제 사례·증거 | 원인 |
|---|---|---|
| 수식·단위를 선택지로 오인 | 세 실행 Physics_8의 선택지A 뒤 `L sin θ`; B2 Electronics_20의 C 뒤 전류 단위A | `_mcq_value`가 suffix의 대문자를 후보 언급으로 검사. 수식 표기 차이로 exact-option 경로도 실패 |
| 숫자 답의 단위·설명 거부 | 세 실행 Electronics_12의 `10 V` 대 gold10; R1/B2 Finance_25의 주당 금액 | `_open_value`의 제한된 수치/문자열 문법. 단위 환산 없이 질문과 동일한 단위인 답도 누락 |
| 모델 자신의 답을 인용으로 삭제 | R1 Clinical_Medicine_27, R2 Energy_and_Power_24의 마지막 `> Correct Answer: ...` | `_visible`이 답 표시용 blockquote와 다른 사람의 인용을 구분하지 못함 |
| 표면 어휘로 과도한 모호성 판정 | R1 Agriculture_11 앞 문맥의 not, Sociology_6의 보기 내용/따옴표 | 부정·or가 실제 답 선택을 부정/양분하는지 확인하지 않고 차단 |
| 동등한 답·정정을 충돌로 판단 | R1 Physics_21의 Point A와 A; Physics_12의 C→D 정정 | 제한된 정정 패턴과 문자열 기반 충돌 판정 |
| 잘림을 답 형식과 혼동 | R1 Electronics_10은 이미 A를 완성했지만 뒤 설명 중 잘림 | 답 완결성을 마지막 구두점 휴리스틱으로 판단 |
| 엄격한 비교 정책 | R2/B2 Pharmacy_3의 Step 2 대 gold2 | 질문은 단계 번호지만 문자열 전체 일치라 탈락. 두 건을 순수 구현 버그와 별도로 집계 |

V7 source는 삭제 전 snapshot에 보존했다:
`analysis_exports/mmmu_20260924/rejected_v7/team_final_v7/scorer_source.tar.gz`.
모듈 SHA256은 `2eaf112136ffbeebacfd20846e2eb05c06cf75dc16b11e6f79e594938637df21`이다.
현재 runtime에는 V7 모듈/선택지가 없다.

V7이 회수한4건도 실제 유효했다: R1 Manage_17의 boxed65,
R2 Psychology_20의 embryonic 결론, B2 Basic_Medical_Science_17의 보기 내용→label,
B2 Clinical_Medicine_3의 최종 C 선택이다. 따라서 V7의 모든 규칙이 불필요했던 것은 아니다.

## 3. V6가 틀린 답을 맞다고 할 가능성

### 3.1 변경60건 외 점검 범위

- 두 파서가 함께 정답 처리한 **주관식32개 응답**을 전부 텍스트 대조했다:
  30개 명확한 일치, 2개 반올림 정책 사례. 명백한 거짓 정답은 이 범위에서 확인하지 못했다.
- 반올림 사례는 R1/B2 Manage_25의 gold0.963 대 출력0.9632다.
  공식 numeric normalization의 소수점 두 자리 반올림으로 둘 다0.96이 된다.
  이를 의미 불일치나 V6만의 문제로 단정하지 않는다. 비교 규칙의 정밀도 한계로 기록한다.
- 공동 정답 **객관식1505개**는 규칙/종료 사유 기반 위험 선별만 했다.
  2개를 추가 대조했고, 나머지1503개를 문맥 전체로 검토하지 않았다.
  선별 통과를 “거짓 정답 없음”으로 보고하지 않는다.
- 추가2개: R1 Architecture_and_Engineering_21은120m 계산 뒤 A 선택에서 잘린 답,
  B2 Biology_20은 C 선택 후 계속 해석을 의심하며 잘린 답이다.
  후자는 안정된 최종 결론인지 불확실하다. 둘 다 완전한 풀이 검증은 아니다.
- 따라서 **전체1597개 V6 정답 판정이 모두 올바르다고 인증하지 않는다.**

근거: `shared_open_review.json`, `shared_mcq_screen.json`.
둘 다 같은 답을 잘못 인정하는 경우는 V6/V7 차이만 봐서는 발견할 수 없다.

### 3.2 합성 반례로 확인한 코드 결함

다음은 **MMMU 실제 오류 개수가 아니라** 일반 문법을 이용한 재현용 예제다.
`synthetic_probes.json`에 변경하지 않은 V6와 보존 V7 결과를 기록했다.

| 입력 / 기준 정답 | V6 | V7 | V6 문제 |
|---|---|---|---|
| Final answer: B or C. / B | 정답 | 미확정 | 첫 B만 채택하고 복수 후보를 무시 |
| Final answer: B is not correct. / B | 정답 | 미확정 | 첫 B 뒤 부정문을 검사하지 않음 |
| Final answer: 50 or 100. / 50 | 정답 | 미확정 | 최종 구간에서 여러 수치 중 하나만 맞아도 인정 |
| Final answer: caterpillar / cat | 정답 | 오답 | 문자열 부분 일치 |

원인 위치:

- [`_mcq_candidate`](../src/mmdl/evaluation/final_answer_parser.py): label 인식 직후 반환하므로
  후속 부정/복수 선택을 검사하지 않는다. 앞부분 가정문 필터가 있다고 이 문제가 해결되지는 않는다.
- [`_open_correct`](../src/mmdl/evaluation/final_answer_parser.py): 확정 구간만 전달하지만,
  해당 구간에 대한 공식 `parse_open_response`·`eval_open`의 다중 숫자/부분 일치 정책은 남아 있다.
- [고정 MMMU evaluator](../third_party/mmmu/eval_utils.py):
  commit `268471d0d488258990025331c7528359c324aa25`, Apache-2.0.
  공식에서 차용한 규칙이라는 사실과 원하는 “단일 최종답 판정”에 적합한지는 별개다.

V6에도 정답 누락이 있다. V7이 유효하게 회수한4건이 직접적인 반례다.
따라서 롤백을 완전한 채점기 완성으로 보고하지 않는다.

로컬 합성 원장의 V6 결과는 아래 CPU 검사로 재현할 수 있다.
이 assert는 **결함의 현행 재현**이지, 올바른 채점 동작을 요구하는 회귀 테스트가 아니다.

```bash
PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
from mmdl.evaluation.final_answer_parser import score_response
p = Path("analysis_exports/mmmu_20260924/v6_v7_audit/synthetic_probes.json")
for case in json.loads(p.read_text())["probes"]:
    actual = score_response(case["synthetic_response"], case["question_type"],
                            case["options"], case["gold"], "stop",
                            parser_id="team-final-answer-v6")
    assert actual == case["V6"]
print("7 stored V6 probes reproduced; known unsafe acceptances remain")
PY
```

## 4. 왜 사전에 걸러지지 않았는가

기존 검증은 코드 테스트·해시·재채점 재현성과 합성 부정/모호성 검사에 집중했다.
이 검사는 “정해진 코드가 일관되게 동작하는가”를 확인하지만,
**정상적인 실제 답 표기를 얼마나 보존하는가**를 충분히 검사하지 못했다.
단위·수식·blockquote·동등 표현을 포함한 원문 회귀 집합의 정답 누락 검증 없이
V7을 제출 보고서의 현재 채점기로 올린 것은 검증 부족이었다.
점수 하락을 엄격성의 불가피한 비용으로 설명할 것이 아니라, 누락 사례부터 분리했어야 했다.

## 5. 다음 수정의 승인·검증 기준

이번 작업은 원인 감사이며 새 파서를 구현하지 않았다.
향후에는 V6를 기준으로 확인된 거짓 정답 위험만 좁게 수정하고,
V7에서 효과가 확인된 표기 복원/결론 추출은 필요한 부분만 검토하는 편이 적절하다.

1. 원문에서 의도한 최종답을 gold와 분리해 먼저 주석하고, 모호한 응답은 미확정으로 남긴다.
   가능하면 사람의 독립 대조를 추가한다. 그 다음 gold와 비교한다.
2. 부정/복수 후보는 선택지 위치와 실제 문장 의미 범위를 검사한다.
   수식 변수·단위 대문자, 보기 문구의 or/not까지 일괄 거부하지 않는다.
3. 표기 정규화와 의미 변경을 분리한다. 단위는 질문이 요구하는 단위와 일치하는지 확인하며
   무조건 제거하거나 새 단위 환산/동의어 추정을 추가하지 않는다.
4. 이번64건, 공동 정답 점검, 짝을 이룬 합성 양성/음성 사례를 회귀 집합으로 고정한다.
   false positive와 false negative를 별도로 보고하고, 이후2700개를 동일 규칙으로 재채점한다.
5. 개발에 쓴 validation 사례는 독립 holdout이라고 부르지 않는다.
   총점 상승을 통과 기준으로 삼지 말고, 정상 답 보존과 거짓 정답 차단을 함께 검사한다.

이 작업에는 GPU 재평가·대규모 fine-tuning·LLM judge API가 필요하지 않다.
모델/입력/생성은 그대로 두고 저장 응답과 고정 파서만 비교하면 된다.

## 6. 파일 위치·재현·보존

- 감사 폴더: `C:\projects\multimodel\analysis_exports\mmmu_20260924\v6_v7_audit`
- 이번 V6 신규 출력: 그 아래 `v6_replayed/{baseline_2048,continuous_32k,official_b2}/`.
- 이전 V6 원장: `analysis_exports/mmmu_20260924/team_final_v6/`; 변경하지 않았다.
- V7 원장/후보/소스: `analysis_exports/mmmu_20260924/rejected_v7/`; 복구 가능하게 보존했다.
- 공개용 소형 집계: [summary](../results/scoring-v6-20260924/summary.json),
  [과목별 CSV](../results/scoring-v6-20260924/subject_scores.csv).
- 제출 본문: [Assignment_1.md](../Assignment_1.md), 동일 본문 [수업 경로](../reports/mmmu_baseline.md).

재채점은 다음 기존 명령을 사용한다. `SOURCE_ARTIFACT_ROOT/runs/SOURCE_RUN_ID`는 원본 실행,
`NEW_RESCORE_DIR`는 아직 없는 출력 경로를 지정한다. 저장소 루트에서 실행한다.

```bash
bash scripts/rescore.sh --artifact-root "$SOURCE_ARTIFACT_ROOT" \
  --run-id "$SOURCE_RUN_ID" --parser team-final-answer-v6 \
  --output-dir "$NEW_RESCORE_DIR"
```

세 원본의 실제 경로/run ID는 보존 V6 및 신규 재채점 각 폴더의 `source.json`에 있다.
scoring hash는 위에, 각 predictions hash는 `replay_verification.json`에 있다.
전체 unit tests는 `PYTHONPATH=src python -m unittest discover -s tests/unit`로 재실행한다.
원본 HTML viewer는 이전 채점이며 자동으로 새 점수로 바뀌지 않는다.
현재 소스는 로컬 미커밋 상태이며 이번 감사에서 commit/push·새 GPU 실행·학습은 하지 않았다.
