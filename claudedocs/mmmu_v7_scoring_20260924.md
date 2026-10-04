# REJECTED / HISTORICAL — MMMU V7 채점 정책과 CPU 재채점 — 2026-09-24

> **V7은 현재 사용하지 않는다.** 사용자 승인으로 runtime을 제거하고 V6을 복원했다.
> V7 module/tests/CLI selection은 저장소에 없으며, 이 문서의 코드·명령은 역사적 기록일 뿐 현재 실행할 수 없다.
> 현재 점수는 V6 R1 479/900, R2 558/900, B2 560/900이다.
> V7→V6 롤백 사유와 사례별 감사는 [V6/V7 false-positive audit](mmmu_v6_v7_false_positive_audit_20260924.md),
> 현재 공개 집계는 `results/scoring-v6-20260924/summary.json`을 참고한다.

아래의 V7 설계·재채점 결과는 삭제하지 않은 재현 기록이며 현재 승인된 scorer나 실행 절차를 뜻하지 않는다.

## 범위와 출처

사용자 승인 범위는 `team-final-answer-v7` 구현과 기존 R1(2K), R2(32K), B2(고해상도32K)의
저장된 900응답씩 CPU 재채점이다. 원문·모델·이미지·프롬프트·sampling·종료 사유는 변경하지 않는다.
GPU/RunPod/LLM judge/학습/재생성/자동 Git 게시를 실행하지 않는다.

- 자체 구현: `src/mmdl/evaluation/final_answer_parser_v7.py`. 기존 formatting/type/choice helper를 재사용하되
  V4~V6의 추출·비교 동작은 수정하지 않는다. V7은 명시적으로 선택하는 CPU rescoring 버전이며
  기존 FROZEN GPU protocol을 V7으로 몰래 바꾸지 않는다.
- 참고·부분 차용: [MMMU eval_utils.py](https://github.com/MMMU-Benchmark/MMMU/blob/268471d0d488258990025331c7528359c324aa25/mmmu/utils/eval_utils.py),
  commit `268471d0d488258990025331c7528359c324aa25`, Apache-2.0.
  **런타임에 재사용하는 함수는 숫자의 `normalize_str`뿐**이다. 문자열 `eval_open`의 부분 일치,
  `parse_open_response`의 다중 숫자 후보 수집, 전체 응답 MCQ 검색/random fallback은 사용하지 않는다.
- Qwen 공식 judge 또는 공식 점수 재현을 주장하지 않는다. 팀의 엄격한 추출/비교 정책이다.

## 확정한 동작 순서

1. 원문은 보존한다. 추출 복사본만 Unicode/Markdown/지원 LaTeX를 정규화한다.
   인용문(큰따옴표·작은따옴표·곡선 큰따옴표), 독립적인 `>` blockquote, fenced code는 선언 후보에서 제외한다.
   단, 자신의 빈 Answer heading 바로 아래 `> 답`은 인용이 아닌 답의 Markdown 표시로 처리한다.
   이는 인용된 답을 모델 자신의 선택으로 취급하지 않으려는 보수적 정책이며 일부 정답 형식을 놓칠 수 있다.
2. `Final answer/decision`, `Correct/Best answer/option/choice`, `Answer` 선언을 등장 순서로 검사한다.
   동일 문단·문장 앞부분의 if/whether/suppose/assuming/maybe/perhaps 등 가정·불확실 문맥을 제외한다.
   Final/Correct라는 단어만으로 나중 선언보다 우선하지 않는다.
3. 답 문단 전체를 검사한다. 복수 후보, 부정, 미지원 수식/표현이면 미확정이다.
   서로 다른 유효 선언은 충돌로 남기며, `Correction:`, `Actually,`, `Instead,` 등 명시적 정정 문맥이나
   끝의 `(correction)`에 한해 갱신한다. 문단을 나눈 독립 `Correction:` 제목도 후속 선언에 적용한다.
   `B is not correct. C is correct.` 같은 제한된 정정 구문도 지원한다.
   빈 줄 뒤에 다른 단독 답/철회가 나와 앞 문단만으로 정답 처리하는 것을 막는다.
   설명 문장 중간의 actually 같은 단어만으로 철회로 판정하지 않으며, 설명의 관사 A를 선택지 A와 구분한다.
4. 선언이 존재하는데 마지막 유효 문맥의 선언을 해석하지 못하면 **즉시 NO_PARSE**다.
   과거 선언/과거 파서/풀이 중 정답 후보로 돌아가지 않는다. 가정/인용만 있는 응답도 fallback하지 않는다.
   유효 선언 뒤의 단순 가정/인용은 별도의 확정 정정이 아니므로 그 자체로 이전 답을 교체하지 않는다.
5. 명시적 선언이 없을 때만 말미 단독 label, 정확한 보기 문구, boxed 답, 아래 결론 문법을 검사한다.
   `therefore/thus/hence/so + 답` 및 `... belongs to (the) X stage/period`에서 X 추출을 지원한다.
   부정/가정형은 제외한다. 이 문법은 문항 ID나 gold에 대한 예외가 아니다.
6. 추출한 단일 답을 고정 규칙으로 gold와 비교한다. 추출 함수는 gold를 받지 않는다.
   문자열/MCQ는 정규화 후 전체 일치이고, dataset이 제공한 복수 gold는 비교 단계에서만 인정한다.
   EMPTY/NO_PARSE는 자동 점수 0점이지만 명확한 답 불일치와 구분한다. 분모는 900이다.

### 숫자·문자열·단위 규칙

- 부호, 소수, 지수, 올바른 천 단위 쉼표, 유리수, 단순 LaTeX fraction, 앞의 통화 기호를 지원한다.
  `1,2` 같은 잘못된 묶음은 12로 바꾸지 않는다. 복수 숫자/수식에서 gold와 같은 값만 찾지 않는다.
- 숫자는 먼저 Decimal/Fraction으로 단일 값을 확인한 뒤 유한 float로 변환하고,
  고정 MMMU `normalize_str`의 **소수점 두 자리 반올림** 결과를 비교한다. 따라서 1.234와 1.23은
  같은 것으로 채점된다. 이는 수학적 완전 동치나 임의 오차 추가가 아니라 공개 benchmark의 숫자 정책이다.
  float 변환/반올림의 한계도 남는다. 비유한 값·0분모·변환 overflow는 정답으로 인정하지 않는다.
- 문자열은 대소문자/공백/문장 끝 구두점 등 표기 정규화 후 전체 일치한다. `cat`과 `caterpillar`는 다르다.
  숫자를 포함한 H2O 같은 짧은 literal phrase도 지원하며 임의 동의어·의미 추론을 하지 않는다.
  `Step 2`를 숫자 2와 같게 만들지 않는다.
- 단위 자동환산·단위 제거는 하지 않는다. 숫자+단위가 단일 지원 값으로 해석되지 않으면 NO_PARSE다.
  예: `100 cm`를 `1 m`로 바꾸거나 `100 V`의 100만 몰래 추출하지 않는다. 이 때문에 실제 정답을 놓칠 수 있다.

### 잘림·모호성의 처리

`finish_reason=length`일 때 명시적 선언의 단일 답이 닫는 구두점/괄호 등으로 끝나고
충돌/철회가 없는 경우만 수용한다. 단독 답/boxed/결론 fallback만으로 length 응답을 구제하지 않는다.
끝 구두점은 완결성의 규칙 기반 증거일 뿐 추론 완료의 증명이 아니며, length 표시는 그대로 보존한다.
현재 응답의 답 일치도를 평가하는 것이지 풀이의 논리·사실성 전체를 검증하는 것은 아니다.

## 재현

다음은 당시 V7을 실행했던 명령을 보존한 historical snippet이다. V7 parser가 제거되어 현재는 실패한다.

```bash
bash scripts/rescore.sh --artifact-root "$SOURCE_ARTIFACT_ROOT" --run-id "$SOURCE_RUN_ID" \
  --parser team-final-answer-v7 --output-dir "$NEW_RESCORE_DIR"
```

`SOURCE_ARTIFACT_ROOT/runs/SOURCE_RUN_ID`에는 원본 COMPLETE900 결과가 있어야 한다.
새 output-dir만 허용하며 원본을 덮어쓰지 않는다. 정확한 세 실행 경로와 결과/소스 hash는
로컬 `analysis_exports/mmmu_20260924/team_final_v7/` README 및 검증 JSON에 기록한다.
source.json은 원본 predictions hash를 연결하고 run_manifest는 새 scoring hash를 고정한다.
새 predictions에는 원문+새 채점 필드가 같이 들어가며 이미지와 HTML은 원본에 남는다.
기존 review.html은 V7 점수로 자동 갱신되지 않는다.

## 검증 원칙

전체 집계 전에 합성 회귀 검사와 독립 검토를 수행한다. 실제 문항 ID별 예외나 목표 정확도는 코드에 없다.
이후 세900의 coverage/subject 산술, 원본 및 모든 비채점 필드 불변, 레코드 hash,
scoring source hash와 재채점 일치를 검사한다. V4/V6 과거 출력도 재현해 호환성을 확인한다.
완전한 자연어 판정기나 오판 0건을 주장하지 않는다. V7 규칙 개발에 사용한 기존 validation 사례는
새로운 독립 holdout 검증으로 표시하지 않는다. 알려진 실패 형식 회귀 통과와 전체 의미 정확성은 다르다.

### 개발 후보 보존

첫 구현 후보는 합성146 tests를 통과했으나, 전수 재채점 후 Markdown 답 표시와 관사 A,
설명 중간의 actually를 잘못 처리하는 일반 형식 결함을 확인했다. 후보 점수는 R1 456/R2 530/B2 530이었다.
이후 새 합성 회귀 검사로 경계를 수정했다. 문항별 예외를 넣지 않았고 원문/정답을 수정하지 않았다.
후보의 세900 및 정확한 source archive는 `team_final_v7_candidate_1/`로 이동 보존했다(삭제 아님).
따라서 최종 V7이 validation 결과를 한 번도 보지 않고 개발됐다고 주장하지 않는다.
단위 환산·동의어 추정·정답 기반 후보 선택 제한은 그대로 유지한다.

두 번째 후보는 R1 463/R2 538/B2 537이었다. 독립 검토에서 답 뒤 화학식의 닫는 `$`만 지워
정확한 보기 일치를 깨뜨리는 결함과 문단을 나눈 `Correction:` 제목 누락을 확인했다.
쌍을 이룬 수식 구분자만 제거하고 정정 제목의 범위를 검사하는 일반 규칙 및 합성 회귀를 추가했다.
후보2의 결과·검증 JSON·소스 archive는 `team_final_v7_candidate_2/`에 이동 보존했다.
독립 검토에서 확인한 미지원 형식인 `boxed 값 → Option D`와 독립 blockquote 속 답 선언은
이번 규칙에서 추측으로 구제하지 않는다. 숫자+단위 및 `Step 2` 대 `2`도 엄격 비교의 누락 원인이다.

<!-- V7_RESULTS:BEGIN -->
## 최종 재채점과 검증 결과

| 실행 | V6 | V7 | V6 대비 정답 증가/감소 | 미확정(900개 분모에 포함) |
|---|---:|---:|---:|---:|
| baseline_2048 | 479/900 | 463/900 (51.44%) | +1/−17 | 249 |
| continuous_32k | 558/900 | 539/900 (59.89%) | +1/−20 | 114 |
| official_b2 | 560/900 | 539/900 (59.89%) | +2/−23 | 129 |

각900개, 총2700개의 원문/모든 비채점 필드 불변·고유 ID·과목별30·record hash·samples 일치·
독립 V7 재채점과 산술을 확인했다. 기존 V4/V6 결과도 각각2700개 재현했다.
전체150 unit tests/Ruff/mypy25 source files/Bash 검사 통과.
이는 모든 응답의 의미 정확성에 대한 사람의 전수 인증이 아니다.

- scoring SHA256: `0db83f5dfc8912e4163502d5b9b89e1a80b967f39c1819ead371ee7ca66838cf`
- V7 source SHA256: `2eaf112136ffbeebacfd20846e2eb05c06cf75dc16b11e6f79e594938637df21`
- 최종 source archive SHA256: `037cb960e269b888014eb735ed26f23f430a4b82926d0ff8ae5463e5c569f103`
- 후보1 archive SHA256: `49705da51bbad6ec1c2d79ede87457e665317cfb32adafb607839b91a2bd886a`
- 후보2 archive SHA256: `f3e6bb0fc5a022eecee04c7237cf4a6a4baa1fead51814e144d38d9b0bbdee43`
- 당시 코드 HEAD `0f10c338f366ca2568595a06fe275a0f46dd0cb7` + 로컬 미커밋 V7 작업본.
  HEAD를 V7 구현 commit으로 표기하지 않는다. snapshot은 정확한 코드/테스트 bytes이며 최종 보고서 본문과는 별도다.

최종 결과는 `analysis_exports/mmmu_20260924/team_final_v7/`, 작은 공개 가능 집계는
`results/scoring-v7-20260924/`에 있다. 새 GPU/LLM/추론/학습/Git 게시를 수행하지 않았다.

### 판정 변화 해석

- R2 Psychology20: `embryonic` 결론을 추출해 회수했다. Electronics25/Pharmacy19는 정답으로 구제하지 않는다.
- 독립 검토에서 후보2의 B2 Chemistry27(화학식)·Materials10(명시적 정정) 누락을 찾아
  일반 수식/정정 규칙 및 합성 검사로 수정했고 최종 결과에 반영했다.
- 반대로 R2 Math22의 `boxed 값 → Option label`, Energy_and_Power24의 독립 blockquote 선언,
  숫자+단위, `Step 2` 대 gold `2`는 정책의 미지원/엄격 일치 한계다.
- B2 Art1은 length 종료 중 후보가 반복 변경되고, R2 Biology24는 조건부 선택/철회가 있어
  미확정이 타당한 사례다. 그 외 모든 손실 사례를 거짓 정답이라고 확정하지 않는다.
- 보이는 독립 Correction 제목을 정정 의사로 신뢰한다. 잘못 붙인 제목까지 의미 판정으로 검증하지 못한다.

V7의 성공 증거는 합성 회귀·알려진 오류 수정·결과 추적 가능성이다.
V2보다 높은 점수나 거짓 정답0건, 완전한 자연어 의미 판정까지 입증한 결과는 아니다.
<!-- V7_RESULTS:END -->
