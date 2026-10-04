# Qwen3-VL-4B 기능적 병목 진단 — 비공개 실행 보고서

## 최종 실행·회수·자원 정리 결과 — 2026-10-03 15:18 KST

**승인 시간 안의 실행과 회수는 종료했다. Gemma의 유효한 전수 채점은 미완료이며,
아래 역사적 체크포인트를 최신 상태로 해석하지 않는다.** 소유Pod `re5qljwwyq7yui`는
로컬 main·독립 검증 후 삭제204→조회404→목록부재를 확인했다. Network Volume은0개다.
최초Pod도 이미 삭제했으므로 해당 실험의 추가 GPU/저장공간 과금 자원은 남기지 않았다.

| 실제 실험 | 관측값 | 해석 범위 |
|---|---|---|
| 기능적 병목 — 30과목102문항 | I 40/102(39.22%), O0 42/85(49.41%), O1 31/69(44.93%) | 층화 목적 표본·조건별 분모 상이; 원래62%와 직접 성능 비교 금지 |
| 구조 추출 P | O0 참조 지지1309/1877(69.74%), 참조 회수710/1382(51.37%) | AI 참조 조건부 항목 지표; 순수 Vision 정확도·사람 검증 아님 |
| Knowledge — 15분야60호출 | K1 30/30, K2 30/30 | 선택한 기초 지식·단순 규칙의 최종답, 복잡한 MMMU와 동일 난도 아님 |
| Gemma 최초 literal 프로토콜 | 900실제 응답:607자동/117검토/176오류 | 유효한 전수 채점 실패, 새 점수null |
| Gemma 추가 source-segment 프로토콜 | 749실제 응답:558자동/110검토/81오류;151상한 종료 | 자동은 파이프라인 상태이지558정답이 아님; 새 점수null |

최종 Gemma 원장은900행이지만151행은 GPU 시간 한도의 `BUDGET_STOP`이다(중단 요청1,
미시작150). 전체900개를 검사했다고 주장하지 않는다. 부모729응답은 byte 그대로 재사용했고
별도 continuation에서20응답만 추가했다. 실제 추출749+동치2호출의 입력7,106,180/출력73,698
token을 기록했다. 전체900개 원문 입력은7,820,217token으로, 짧은 파싱 결과를 CPU 비교한
것이 아니라 긴 풀이 전체의 확정답·정정·철회를 감사한 것이 실행 지연의 주된 작업량이다.
비원문 인용 생성은 행ID 방식으로 막았지만 잘못된ID·비원문 답·출력잘림 및 의미 오판은 남았다.

같은 문항 비교에서 O0−I는 회복15/악화10(85문항), M−I는10/17(102문항),
MI−M은15/9(102문항)이다. 전사와 추론 양쪽이 병목 후보이며 내부 Vision/연결부/Decoder의
유일한 책임은 확정하지 않는다. 모든 paired family 불확실성 구간은0을 포함한다.
파인튜닝은 하지 않았다. 후속 독립 외부 데이터 비교 후보는 §8에 기록한다.

Gemma의 정오 변경 후보6개는 확정 수정이 아니다. Art_Theory_8·Manage_10의 충돌 판단은
독립 텍스트 감사에서 지지되지 않았고, Chemistry_30은 gold별칭 컨테이너 해석 차이였다.
Manage_20의 수치/방향 동치는 별도 정책 검토가 필요하다. 기존 V8 **558/900(62.00%)**,
원문·파서·공개 MMDL/GitHub는 불변이다. 규칙을 완화해 자동으로 점수를 올리지 않았다.

결과 위치: `C:\projects\multimodel\analysis_exports\bottleneck_20261003\`.
기능적 진단은 `export_all30_102_v2/`, 지식 실험은 `knowledge_scores_v1/`, 최종 Gemma는
`gemma_segments_final_export_v1/`와 두 `cloud_gemma_segments_*_final_*` 회수 폴더에 있다.
[최종 main 검증](../analysis_exports/bottleneck_20261003/gemma_segments_final_main_verification_v1.json),
[독립 검증](../analysis_exports/bottleneck_20261003/gemma_segments_final_independent_v1.json),
[삭제 증거](../analysis_exports/bottleneck_20261003/owned_gemma_segments_cleanup_final_v1.json).
최종 API 비용 조회0.404973USD는 지연 집계 snapshot이지 최종 청구액이 아니다.
삭제 전 발생한 비용의 사후 정산은 남을 수 있지만 삭제한 자원의 새 사용료는 중단됐다.

## 추가 승인에 따른 Gemma 재검증 — 역사적 진행 이력

사용자가 추가2시간과 새3090 1개를 승인했다. 새Pod `re5qljwwyq7yui`에서03:41:44UTC
환경 설치를 시작했다. GPU 종료05:52:38UTC/14:52KST, 회수·정리 종료06:22:38UTC/15:22KST다.
별도 `gemma-frozen-final-audit-v7-source-segment-citations`는 전체 원문을 손실 없이 행ID와 함께
전달하고, 실제 선택된 위치의 원문만 host에서 복원한다. 연속 선택한 행만 원문 그대로 묶을 수
있으며 비연속 행을 연결하거나 정답을 참조해 구제하지 않는다. 기존 철회·충돌·단위·답 일치
가드는 유지했다.70 Gemma CPU tests(49base+14source+7controller 분할실행)·24 exporter/controller·4 recovery 검사/Ruff/mypy/Bash를
통과했다. 실제CPU900 입력 검사와 GPU29smoke도 통과해04:00UTC경 전수감사를 시작했다.
실제 최대입력122159+출력상한1024=123183/문맥131072, 원문절단0이다. 전체입력7,820,217token은
이전5,748,953보다약36% 많으며, 초반 실측상 승인 기한 내900완료는 보장할 수 없다.
04:22UTC81건은61RESOLVED/11REVIEW/9SYSTEM_ERROR이며, 잘못된 행ID·답/근거 불일치와
출력잘림을 계속 차단한다. GPU29PASS는 출력 계약 통과이며 일반적인 의미 판정 정확도 인증이 아니다.
이전 실패900과 Qwen/Knowledge 결과 및 baseline558/900은 보존하며 아래 종료 내용은 이전Pod 기록이다.
근거: [새Pod·설치 시작](../analysis_exports/bottleneck_20261003/gemma_segments_launch_receipt_local_v1.json),
[CPU 검증](../analysis_exports/bottleneck_20261003/segments_local_cpu_validation_v2.json),
[실제gate 로컬 검증](../analysis_exports/bottleneck_20261003/gemma_segments_gates_main_verification_v1.json),
[입력 비용 대조](../analysis_exports/bottleneck_20261003/gemma_segments_input_cost_comparison_v1.json).

닫힌첫81을 로컬 회수해 입력token-ID와 전체 판정·9실패를 재현했다. 같은81의 이전58자동판정/
15검토/8오류→새61/11/9로 혼합 변화이며 점수 개선이 아니다. 새9오류는 judge출력1024잘림3,
빈행ID2, 중복ID2, 원문과 다른 답 표기2다. 비원문 근거 생성 경로를 제거했어도 정확한 ID가
의미적 판정 정확성을 보장하지 않으며, 유효한 원문 가드를 완화해 구제하지 않았다.
근거: [81 실제 응답 main 재현](../analysis_exports/bottleneck_20261003/gemma_segments_prefix81_main_verification_v1.json),
[독립 실패 분석](../analysis_exports/bottleneck_20261003/gemma_segments_prefix81_failure_independent_v1.json),
[29 실제 smoke 독립 검증](../analysis_exports/bottleneck_20261003/gemma_segments_gates_independent_verification_v1.json).

05:19UTC 보강: 닫힌첫420 실제기록의 입력·실제token-ID·판정과45실패를 로컬 재현했다.
330자동판정/45검토/45오류다. 오류는 답-원문 불일치21, 출력계약12, 빈행8, 중복ID4다.
입력2만 미만371건은325자동판정/21검토/25오류인 반면,5만 이상32건은1/15/16이다.
장문에 미확정·가드거부가 집중되지만 선택된prefix의 상관 관측이며 인과효과·정확도 인증이 아니다.
[420건 실제 재현](../analysis_exports/bottleneck_20261003/gemma_segments_prefix420_main_verification_v1.json).
원래 run의05:52:38UTC deadline은 변경하지 않는다. 닫힌 원장·소유cleanup과 동일성이 확인되면
별도 samePod continuation으로 남은 문항만 처리한다. 새GPU한도06:07:38UTC, 최종회수06:22:38UTC다.
추가13회귀 포함83 Gemma CPU검사·독립 Sol검토는 통과했으나 현재 재개 실행은 아직 없다.

420건 안의 불일치 후보2개를 독립 AI 텍스트 감사했다. Art_Theory_8은 명확한 최종A가 있는데
Gemma가 충돌이라고 판정한 사례로, V8 거짓정답의 증거가 아니다. Chemistry_30의 `MgS`는
저장된 정답 별칭과 일치한다. 여기서 차이는 화학식 대소문자 능력이 아니라 정답의 별칭 목록이
문자열 `['$MgS$', 'MgS']`로 저장됐는데 baseline 비교에 단일 문자열로 전달된 데 있다.
원문/추출/파서는 그대로 두고 gold의 컨테이너만 해석하는 별도 CPU 통제에서900개 baseline을
558로 재현했으며, 바뀐 판정은 Chemistry_30 하나(통제559)뿐이다. 제출 점수나 Gemma의 전수
의미 정확도로 대체하지 않는다. 두 후보는 REVIEW를 유지하며 기존558/900은 불변이다.
[독립2사례 검토](../analysis_exports/bottleneck_20261003/gemma_segments_prefix420_candidates_independent_v1.json),
[900개 gold컨테이너 CPU통제](../analysis_exports/bottleneck_20261003/gold_alias_container_diagnostic_verified_v2.json).

작성 기준: 2026-10-03 KST. 60파일럿과 추가42문항은 닫힌 결과를 로컬 검증했다.
추가42문항은 2026-10-02 22:29:44 UTC에 종료했고, 합계102문항은30과목 모두를 포함한다.
**아래는 최초Pod의 실행 종료 보고다. Qwen102문항·Knowledge60호출과 Gemma900시도를 로컬 보존·검증했다.
다만 Gemma의 유효한 전수 판정은 실패했고, 일부 구조 의미 지표·반복 실험은 미측정이다.**

Gemma는02:39:37 UTC에900시도를 마쳤다:607RESOLVED/117REVIEW/176SYSTEM_ERROR.
293미확정 때문에 새로운 정확도는 null이며 기존558/900(62.00%)은 불변이다.
소유 Pod는03:03:45 UTC /12:03:45 KST에 삭제204→조회404→목록 부재로 확인했다.
이는 안전한 실행·회수 종료이지 계획서의 모든 과학적 질문이나 유효900채점의 완료가 아니다.
아래 §8에는 관측·지지 가설·반례·미확정 사항과 조건부 학습 후보를 구분했다.

<details>
<summary>이전 진행 체크포인트 — 현재 상태·최종 점수로 사용하지 않음</summary>

01:50 UTC 보강: 닫힌128감사 원장의91RESOLVED/22REVIEW/15SYSTEM_ERROR를 로컬 재현했다.
실패15개는13문항의18개 비원문 인용과2문항의Gemma1024출력 JSON잘림이다. 단순 Markdown
손실뿐 아니라 다른 option label을 인용하거나 추출한 답의수식 기호를 바꾼 사례도 있어,
원문 근거 검사를 완화하거나 실패를 자동 구제하지 않는다. 질문/보기/gold/V8 없이 수행한
[독립 원인 분석](../analysis_exports/bottleneck_20261003/gemma_prefix128_failure_analysis_v1.json)과
[main 인용·실패 재현](../analysis_exports/bottleneck_20261003/gemma_prefix128_failure_main_verification_v1.json)을 보존한다.
후속 집계기의 mapping 재현/비용 누락과 반복 원문 hash를 수정했으며9 CPU회귀와 실제128의
구·신 집계 byte동일 검증을 통과했다. 추론·원문·가드는 변경하지 않았다.
실제900감사는312/900 확인 시 진행 중이며 아직 새로운 확정 점수가 없다.

최신 감사 확인(01:30 UTC /10:30 KST): 통과한29 실제 smoke를 로컬 회수하고 원장·실제 호출·
전체 입력과 배포 identity를 재검증했다. 29/29는 최종 출력 계약의 통과이며, 보기별 witness와
reason에 확인된 의미·지시 준수 오류가 있어 일반적인 판정 정확도 인증은 아니다.
실제900감사 중88개는59RESOLVED/18REVIEW/11SYSTEM_ERROR다. 원문 근거 인용 불일치10개와
미완료 응답 검사1개를 가드로 거부했으며, source나 가드를 바꾸어 구제하지 않는다.
이 상태에서 새로운 전체 정확도를 제시하지 않는다. 근거:
[29 smoke 로컬 검증](../analysis_exports/bottleneck_20261003/cloud_gemma_passed_smoke_v8/local_receipt.json).

최신 실행(10-03 01:02 UTC /10:02 KST):15분야60텍스트 호출의 Knowledge Oracle을 실제 완료했다.
K1 30/30,K2 30/30이며 누락/시스템 오류/잘림/진단 파싱 실패0·원문/입력/후속 점수 로컬/독립 검증.
K3는 기존 실제 이미지 I 응답이며 K1/K2보다 복잡하고 다른 문항이므로 직접 인과 비교하지 않는다.
기초 지식 부족 하나로 실패를 설명하기 어렵지만, DFA/그래프의 중간 풀이 오류와 O1의 실패가 남는다.
[별도 Knowledge 결과 보고서](Qwen3_VL_4B_Knowledge_Oracle_20261003.md)에 분야별 수치와 한계를 기록했다.
Gemma의27smoke 26PASS/1모호성 FAIL은 별도로 보존했다. 이후 선택지 전체의 의미적
모호성을 검사하도록 보강한 새 protocol은01:21 UTC 실제29smoke를 전부 통과했고,
저장900응답 감사가8/900 진행 중이다. 완료·새확정 점수는 아직 아니며 기존 V8/102진단은 불변이다.
이전 v4 smoke12/13, v5 smoke11/13의 실패는 보존했다. 추출에 question/choices/gold/V8를 주지 않는
submission-only 단계, 동결된 답만 보는 보기 연결, native status/null/revision 분기를 구현했다.
공통 문맥 검사로 가정·인용·코드 fence와 복수 답을 차단했다. stable58 CPU tests/Ruff/mypy3 및
독립 반례 검사, 실제 Pod XGrammar0.2.3의11native 계약 검사를 통과했다. 새 CPU900v6는
900COUNTED/overflow0/max50454+1024=51478이며 원문 절단0이다. 실제17GPU smoke 통과 전900감사는 금지한다.
P의80비교 가능 문항 AI 검토는 독립 표본 감사 후4문항7판정을 수정해 `p_structure_measurement_corrected_all30_v2/`에
별도 보존했다. O0 조건부/사람 검증 아님·22비교불가·사후 항목 단위·중요관계 미측정을 유지한다.
Energy_and_Power_20 reference uncertainty의 질문 유래 노출도 기록했다. 원래558/900/공개본은 불변이다.

</details>

## 1. 상태와 불변 기준

| 단계 | 상태 | 실제 근거 |
|---|---|---|
| 입력·이미지 역할 감사 / 표본 고정 | TESTED | [prepared/receipt.json](../analysis_exports/bottleneck_20261003/prepared/receipt.json) |
| 코드 CPU 검사 | TESTED | 현재 비공개 Gemma49+진단59=108 CPU검사·Ruff PASS; 로컬 src를 사용한4파일 mypy PASS. GPU 판정 정확도의 인증은 아님 |
| 로컬 BF16 소규모 점검 | SMOKE_COMPLETED | [10호출 receipt](../analysis_exports/bottleneck_20261003/local_smoke_2cases/completion_receipt.json) |
| RunPod 고정12문항 | SMOKE_COMPLETED | [독립 회수·재채점 receipt](../analysis_exports/bottleneck_20261003/cloud_stress12_v1/local_receipt.json) |
| 60문항 파일럿 | PILOT_COMPLETED | 480조건:444실호출/27차단/9제외, 누락·시스템 오류0; 닫힌 원문·444입력·480원래 채점·집계를 로컬 재현하고 독립 검증 |
| 비공개 결과 집계·CSV 내보내기 | TESTED / 실제102문항·Gemma900시도 집계 완료 | CPU-v2의816채점행·paired/과목/유형/전사분모 CSV와 Gemma293검토 큐·실패 원인 생성; 새 정확도는 null |
| 비공개 결과 회수·소유 자원 정리 | LOCAL_VERIFIED / CLEANUP_VERIFIED | 기존 Qwen/Knowledge/환경 증거와 Gemma42 payload·실패 이력23 payload 검증 후 exact owned Pod 삭제; [최종 receipt](../analysis_exports/bottleneck_20261003/owned_pod_cleanup_final_v1.json) |
| 설치 환경 원본 회수 | TESTED | 실제 Pod→로컬15파일(환경10/회수 코드5) archive/member/추출 byte 검증; 아직 실행 중 결과는 미포함 |
| 30과목 추가42문항 / 잔여160문항 | 추가42 완료·LOCAL_VERIFIED / 잔여160 미실행 | 336조건:305실호출/20차단/11제외·누락/오류0; 원래336채점·집계와305입력을 독립 CPU 재현 |
| Gemma 900개 CPU 입력 검사 | TESTED / 최신900입력 LOCAL_VERIFIED | submission-only protocol의900COUNTED/overflow0·최대50454+1024=51478·원문절단0 |
| Gemma GPU smoke / 900개 감사 | SMOKE_COMPLETED 29/29 /900시도 종료·유효 판정 INCOMPLETE |900추출+2동치 실제 응답,607RESOLVED/117REVIEW/176SYSTEM_ERROR; smoke는 출력 계약 통과이며 의미 정확도 인증 아님 |
| 다분야 K1/K2 지식·규칙 시험 | SMOKE_COMPLETED /60실제 호출 LOCAL_VERIFIED | K1 30/30,K2 30/30·별도 정답키·gold-blind 입력/전체 원문 검증; K3와 비동일 문항/난도 |
| P 전체 의미 측정 | TESTED /80 O0조건부 AI검토·수정본 검증 |102큐·80eligible·22비교불가; corrected_v2 별도원장, 사전 중요관계·이미지 진실 인증 없음 |
| fine-tuning | 미실행 | 어느 모듈에도 학습 업데이트 없음 |

기존 V8 원문·파서·평가 설정·558/900(62.00%)과 공개 MMDL은 변경하지 않았다.
새 진단은 별도 최종답 형식, 최종 출력4096/P1536의 프로토콜이다. 과거32K 점수와
직접 비교하여 모델이 좋아지거나 나빠졌다고 주장하지 않는다. 텍스트 조건도 동일
Qwen checkpoint를 이미지 없이 사용한다. 별도 텍스트 모델의 성능 비교가 아니다.

## 2. 실행 및 데이터 경계

- 사용한3090 Pod: `h447a4u6c0vq3k`(삭제 확인); RTX3090 24GB, BF16·비양자화·미학습 가중치.
- Qwen revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
- MMMU validation revision: `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`.
- 최초 seed3407, 원래 sampling recipe 유지. P만 별도 stage seed. 추가 반복은 미실행.
- 12시간 운영 기한: 2026-10-03 04:22:38 UTC /13:22:38 KST. 최초 파일럿 최대4시간.
- 원래982 이미지 등장 중 질문·보기 입력977회와 해설 전용5회를 구분했다.
  새 진단에서만 해설 전용 이미지를 제외했으며 과거 입력을 수정하지 않았다.
- 모든 전사·모델 응답·파생 결과는 `BENCHMARK_DERIVED`, `training_allowed=false`.
  GitHub/MMDL 게시 금지이며 승인된 비공개 RunPod 전송만 사용한다.
- O0/O1은 서로 다른 Sol의 이미지 검토를 거친 AI 전사다. `AI_REVIEWED_NOT_HUMAN_VERIFIED`이며
  사람 검증으로 인증하지 않는다. 정답 노출·불완전·미승인 조건은 EXCLUDED/BLOCKED로 유지한다.

## 3. 고정12문항에서 확인한 사실

96조건 중94실호출 완료,1제외,1차단,시스템 오류0이다. 입력78,488/출력63,464 token,
개별 호출 시간 합1,558.674초이며 설치·모델 로드 시간과 다르다. 고정 stress set이므로
MMMU 전체 오류율이나 특정 과목 정확도를 추정하는 무작위 표본이 아니다.

진단 전용 CPU 채점 v2는 단일 말미 `FINAL_ANSWER`의 문자, 문자+정확한 해당 보기,
고유한 보기 전문만 처리한다. `B. 16` 같은 형식을 기존 진단 strict-v1이 놓쳤다.
별도 overlay에서19조건의 추출을 변경했다(정확한 보기 형식16수용, 문자/내용 충돌3미확정).
정오 변화는 +9/−0 **조건**이며 새 모델 추론이나9개의 새로운 정답 문항이 아니다.
원래 strict-v1 채점과 모든 raw를 보존했다. 새로운 진단 raw에 적용한 `v8_score`는
동일한 V8 코드의 별도 열일 뿐 과거558/900의 대체 점수가 아니다.

근거: [scoring_overlay_v2](../analysis_exports/bottleneck_20261003/cloud_stress12_v1/scoring_overlay_v2/summary.json),
[변경 원장](../analysis_exports/bottleneck_20261003/cloud_stress12_v1/scoring_overlay_v2/comparison.json).

| 비교 | 적용 가능한 paired 문항 | 회복 | 악화 | 주의 |
|---|---:|---:|---:|---|
| O0 − I | 12 | 2 | 0 | Music_15는 원래 답의 문자/내용 충돌이 있어 추출·의미 판정도 관여 |
| O1 − O0 | 10 | 3 | 1 | 적용 불가2개는 분모 제외; 해석 전사가 항상 유익하지 않음 |
| M − I | 12 | 1 | 0 | 실제 P 응답을 그대로 제공한 조건 |
| MI − M | 12 | 0 | 1 | 이미지를 다시 주는 것이 항상 개선으로 이어지지 않음 |

조건·반복3407·실제 원문/채점 열로 추적할 수 있는 사례:

- `validation_Accounting_19`: I에서B, O0/M에서C, MI에서B로 변경되어 회복과 악화를 함께 관측했다.
- `validation_Music_3`, `validation_Music_25`, `validation_Computer_Science_19`: O1−O0 회복.
- `validation_Music_26`: O1−O0 악화. Music_25와 공유 이미지 family도 불확실성 집계에서 묶는다.
- `validation_Music_15`: I의 문자/보기 내용 충돌로 미확정; O0 회복을 순수 지각 개선으로 단정하지 않는다.

원문: [stress/responses.jsonl](../analysis_exports/bottleneck_20261003/cloud_stress12_v1/stress/responses.jsonl).
이는 초기 관측이며 단계별 원인이나 후속 학습 효과의 확정 결론이 아니다.

### 3.1 닫힌60문항 파일럿의 실제 결과

9과목60문항의480조건을 완료했다(444실호출/27미승인 전사 차단/9정답 노출 조건 제외).
반복은3407 한 번이며3회 반복을 완료했다고 표시하지 않는다. 입력448,130/이미지284,576/
출력445,948 token, 개별 요청 시간 합10,919.161초다. 원장 SESSION_END 경과10,945.316초는
약3시간2분25초이며, 설치·stress12·전송을 포함한 총 Pod 시간이 아니다.

원래 strict-v1 소스17개의 byte·code hash를 원래 전송 archive와 대조하고,
그 frozen scoring 파일로480개 채점 및 전체 summary를 정확히 CPU 재현했다.
현재 로컬은 scoring.py와 cli.py만 다르므로 원래 추론 소스로 표시하지 않는다.
모든444개 payload도 재구성하여 prompt/seed/이미지/독립 조건·120개의 동일 P 재사용을 검증했다.
근거: [로컬 실험 검증](../analysis_exports/bottleneck_20261003/pilot60_local_experiment_verification_v1.json),
[독립 검증](../analysis_exports/bottleneck_20261003/pilot_closed_independent_check_v1.json).

별도 진단 CPU-v2 결과는 다음과 같다. 분모는 실행·적용 가능한 조건이며,
정오 균형으로 뽑은 표본이므로 MMMU 전체 정확도나 기존62%의 대체 점수가 아니다.

| 조건 | 진단 정답 / 실행 분모 | NO_PARSE | 출력 상한 종료 |
|---|---:|---:|---:|
| I | 19/60 | 18 | 14 |
| T0 | 18/60 | 7 | 4 |
| O0 | 25/50 | 11 | 10 |
| O1 | 21/44 | 8 | 8 |
| IO0 | 23/50 | 9 | 8 |
| M | 21/60 | 22 | 12 |
| MI | 20/60 | 18 | 14 |

원래 strict-v1 대비123조건의 추출 결과가 바뀌었고 정오는47증가/0감소했다.
정오가 증가한47조건은 문자+정확한 해당 보기 형식의 인정이며, 저장된 원문의 **채점 조건 변화**다.
원문·기존558/900·추론 코드는 불변이다. 최종답 조건384개 중 NO_PARSE93개에는
상한 종료70개가 포함된다. 자연 종료 추출 실패23개의 별도 gold-blind 감사에서는
13개 명시적 답변 포기와10개 선택지 제출 표현 후보를 구분했다. 후자는 필수 marker 없음5,
보기 원문 오탈자와의 불일치2, 앞 문장을 참고해야 구별되는 문자/내용 충돌3이다.
마지막3개 중 하나는 가정에 근거한 표현이므로 확정 정답으로 구제하지 않는다.
10개를 추가 정답으로 계산하거나 strict-v2를 바꾸지 않았으며53개 원문 인용·23개 source hash를 확인했다.
[형식 감사](../analysis_exports/bottleneck_20261003/pilot60_stop_no_parse_answer_contract_review_v1.json),
[원문 연결 검증](../analysis_exports/bottleneck_20261003/pilot60_stop_no_parse_contract_local_verification_v1.json).
NO_PARSE를 전부 모델 추론 오류나 파서 오류라고 합치지 않는다.
[CPU-v2 집계](../analysis_exports/bottleneck_20261003/export_pilot60_v2/summary.json),
[strict-v1 대조](../analysis_exports/bottleneck_20261003/export_pilot60_v2/strict_v1_to_cpu_v2_comparison.json).

| 비교 | 적용 가능한 paired 문항 | 회복 | 악화 |
|---|---:|---:|---:|
| O0 − I | 50 | 11 | 3 |
| O1 − O0 | 44 | 5 | 5 |
| M − I | 60 | 8 | 6 |
| MI − M | 60 | 5 | 6 |

O0 제공 시 순회복은 관측됐지만, 출력 상한과 답 형식 실패도 있으므로 이를 Vision/Merger의
유일한 책임이나 순수 지각 개선으로 환원하지 않는다. O1·M·MI의 악화도 함께 보존한다.
조건별 원문 hash·문항/반복과 family bootstrap을
[paired_cases.csv](../analysis_exports/bottleneck_20261003/export_pilot60_v2/paired_cases.csv)에 연결한다.

### 3.2 판정 변화와 출력·추출 조건을 구분

214개 적용 가능한 조건쌍을 원문 종료 사유와 CPU-v2 추출 상태에 연결했다.
아래 수치는 **정오가 바뀐 조건쌍만** 상호 배타적으로 나눈 회복/악화 건수다.

| 비교 | 모두 자연 종료·추출 성공 | 한쪽 이상 출력 상한 종료 | 나머지 자연 종료·추출 실패 포함 |
|---|---:|---:|---:|
| O0 − I | 4 / 1 | 5 / 2 | 2 / 0 |
| O1 − O0 | 3 / 3 | 2 / 2 | 0 / 0 |
| M − I | 3 / 2 | 4 / 3 | 1 / 1 |
| MI − M | 1 / 1 | 2 / 4 | 2 / 1 |

예를 들어 O0−I의11회복 중5개는 I가4096상한에 도달하고 O0는 자연 종료한 사례다.
4개는 두 조건 모두 자연 종료·답 추출 성공이며,2개는 최종답 표기 또는 문자/보기 내용의 중첩 때문에 추출 상태도 다르다.
따라서 11개 전체를 시각 정보 인식 개선으로 세지 않는다. 이 분류는 생성 후 관측에 따른
서술적 분해이며, 조건을 사후 선별한 새로운 정확도나 출력 길이를 통제한 인과효과가 아니다.
또한 O0는 이미지에 전사를 추가한 조건이 아니라 **이미지를 원시 전사 텍스트로 대체한 조건**이다.
이미지와 O0를 함께 제공하는 별도 조건은 IO0다. 추출 성공 역시 답의 의미·전사의 완전성 인증은 아니다.

근거: [종료·추출 연결 원장](../analysis_exports/bottleneck_20261003/pilot60_paired_finish_parse_controls_v1.json).
조건쌍은 같은 문항을 재사용하므로 표의 행을 합쳐 독립 문항 수나 전체 오류율로 사용하지 않는다.

[14변경 사례의 독립 gold-blind 검토](pilot60_oracle_transition_evidence_20261003.md)는
판정 방향도 숨긴 채 I/O0 원문에서 무엇이 바뀌었는지 확인했다. main은28개 원문 연결·
70개 인용 offset·7개 잘림 말미를 [검증](../analysis_exports/bottleneck_20261003/pilot60_O0_I_transition_review_local_verification_v1.json)했다.
`Music_10`은 두 조건 모두B를 답했지만 I에 필수 최종답 marker가 없어 점수가 달라진 사례다.
`Computer_Science_7`은 두 응답의 표 값은 같지만 빈 슬롯을 다루는 절차와 답 완결 여부가 달랐고,
`Public_Health_9`도 네 수치 행이 이미 I에 있으면서 범주 연결·합산과 반복/완결이 달랐다.
반대로 `Biology_21`은 시각 구조에 대한 진술이 달라졌지만 O0에서 오답으로 바뀌었다.
이 검토는 이미지를 보지 않은 **응답 진술 비교**다. 원시 전사의 정확성이나 어느 내부 모듈의
책임을 인증하지 않으며, 표 값의 존재를 전체 관계 해독 성공으로 확대하지 않는다.

비공개 내보내기 [export_v2](../analysis_exports/bottleneck_20261003/cloud_stress12_v1/export_v2/summary.json)는
위96개 채점행과9개 입력 hash를 그대로 검증하고 paired/과목/이미지유형/전사분모 CSV를 제공한다.
호환되는 서로 다른 실행은 원문을 덮어쓰지 않고 합치며, stress는 대표 표본과 분리한다.
중단·오류의 실행 경로 진입을 실제 backend 호출로 인증하지 않고 불확실한 호출 수는 null로 기록한다.
예약 토큰과 관측 생성 토큰도 분리했다. 독립 검토에서 발견한 집계 문제와 수정 검증은
[검토 원장](../analysis_exports/bottleneck_20261003/export_results_independent_review_v1.json)에 함께 남겼다.

## 4. P 구조 추출의 형식과 의미를 분리

12개 P 응답은 모두 JSON/schema 유효, 정상 stop,97–340출력 token으로1536상한 미도달이다.
독립 이미지 대조는9문항에서 구조 누락/오류를 관측했으며,3문항은 점검 범위에서
명확한 label/value/edge 오류를 확인하지 못했다. 후자도 구조 완전성 인증이 아니다.
Music_3에는 원본 아래 보조선의 음표 위치가 보고되지 않고 존재하지 않는 natural이
출력된 사례가 있다. 관측 원문26개 offset·이미지 hash·24개 M/MI의 동일 P 재사용을 검증했다.

근거: [stress_p_structure_review.json](../analysis_exports/bottleneck_20261003/stress_p_structure_review.json).
정확한 JSON 문자열 비교는 자연어 구조 동치의 완전한 판정기가 아니다. 위 이미지 감사도
AI 검토이며 사람의 독립 구조 정답률 인증은 아니다. P는 전체 VLM을 통과하므로 순수 Vision 평가가 아니다.
원장에 있는 필드 일치·관계 precision/recall·전체 구조 일치는 **정확한 JSON 표현 비교**로만 해석한다.
의미상 필드 정확도·관계 정확도·누락/환각 비율은 아직 NOT_MEASURED이며, 관측 오류9건을
그 지표의 분자나 전수 구조 정답률로 대체하지 않는다.

파일럿 실행을 중단하거나 원장을 바꾸지 않고,20:32:08 UTC까지 완료된 P 응답42개를
별도의 닫힌 사본으로 회수했다.42개 record/raw hash·이미지 순서/실물 byte와 원문 전송을
[로컬 검증](../analysis_exports/bottleneck_20261003/pilot_p_shard_local_verification_v1.json)했다.
정답·질문·보기·다른 조건 응답을 제외한 packet으로 새 Sol이 원본47이미지를 대조했다.
참고 O0는36 VALID/6 BLOCKED이며 차단 전사를 정답 구조로 사용하지 않는다.
이 사본 자체는 파일럿 전체 완료 증거가 아니다. 이후 닫힌480개 원장에서 앞7,809,462bytes/
334행 prefix hash와42개 P record/raw의 정확한 일치를 main 및 독립 검사로 확인했다.
위 로컬 실험 검증·독립 검증에 연결하되 사람 검증이나 전체 Pod 삭제 gate로 사용하지 않는다.

[42개 P의 gold-blind 검토](../analysis_exports/bottleneck_20261003/pilot_p_semantic_review_v1.json)는
31개 오류/누락 관측,7개 미확정,4개 확인 범위 내 오류 미관측으로 구분했다.
검사한 JSON leaf 1,015개(지원834/모순67/확인불가114)는 반복된 label/value도 별개로 센
검토 원장 수이지 독립 의미 사실 수나 의미 정확도·precision/recall의 분모가 아니다.
[main 검증](../analysis_exports/bottleneck_20261003/pilot_p_semantic_review_local_verification_v1.json)은
42개 원문·47이미지 순서/byte·전체 인용1,051개 offset 및 parse 가능한41개 P의 field pointer/value를 확인했다.
Biology_24의 없는 숫자37, Computer_Science_25의 triangle 반복/누락 및 P1536상한 종료,
Electronics_14의 잘못된 diagonal을 실제 이미지로 별도 점검했다.
반면 동일한 무방향 선분의 endpoint 역순은 추가 오류로 세지 않도록 검토 원장을 수정했다.
이는 구조 의미 감사의 점검 범위이며 원래 문제를 재풀이하거나 오류 모듈을 특정한 결과가 아니다.

[경계5사례의 별도 검토](../analysis_exports/bottleneck_20261003/pilot_p_review_boundary_check_v1.json)에서는
Accounting_11의 header/row_labels, Computer_Science_7의 down/행 순서,
Computer_Science_22의 start/pseudo-node가 명확히 정의되지 않은 schema 분류 문제임을 확인했다.
Electronics_23의 C→c는 문자 표기의 차이이고, Electronics_4의 ground를 voltage source로 묶은
표현은 기기 분류 오류이지만 연결관계 변화나 문항 오답의 원인을 증명하지는 않는다.
P 프롬프트의 array/null schema는 이 의미 계약을 정하지 않았으므로 이러한 사례를 확정 지각 오류로
합산하지 않는다. 기존42개 감사와 그 집계는 보존하고 경계 판정을 별도 overlay로 연결했다.
[원문·35인용·5이미지 검증](../analysis_exports/bottleneck_20261003/pilot_p_review_boundary_local_verification_v1.json)도 통과했다.
따라서 위31개는 첫 검토의 상태 집계이지 최종 인증된 시각/구조 실패 건수나 실패율이 아니다.

닫힌 파일럿의 나머지18개 P(Materials2/Music8/Physics4/Public_Health4)도 질문·보기·gold를
제외한 packet으로 실제21이미지를 대조했다. 기존42개와 겹치지 않아60개 P 모두에 별도 AI 검토가 있다.
원문18개·이미지21개·assertion 인용117개·누락의 문맥 인용16개·parse 가능한17개 JSON pointer를
[main 검증](../analysis_exports/bottleneck_20261003/pilot_p_remaining18_local_verification_v1.json)했다.
Music_13의 잘린 JSON은 복구하지 않았고 차단 O0 4개는 null로 유지했다.

[초기18개 감사](../analysis_exports/bottleneck_20261003/pilot_p_remaining18_semantic_review_v1.json)와
[경계2사례 보완](../analysis_exports/bottleneck_20261003/pilot_p_remaining18_boundary_review_v1.json)은 따로 보존했다.
Music_11의 반복된 eighth 목록에는 음표 ID·전수 전사 계약이 없어 오답으로 단정하지 않고
UNRESOLVED로 정정 권고했다. Physics_18은 기기 이름의 해석과 실제 병렬 연결 오류를 구분했다.
main도 Materials_30의 꼭짓점/중심, Physics_7의 점 위치 및 두 경계 이미지를 직접 확인했다.
이 검토는 서로 다른 단위의 assertion/leaf와 범위를 사용하므로 이전42개의 집계와 합쳐
최종 의미 정답률이나 특정 모듈의 실패율로 제시하지 않는다.

### 4.1 30과목102문항으로 확장한 집계

추가42문항의 [원문 회수](../analysis_exports/bottleneck_20261003/cloud_coverage42_v1/recovery_receipt.json)와
[독립 검증](../analysis_exports/bottleneck_20261003/coverage42_closed_independent_check_v1.json)은
305개 실제 입력·seed·이미지 순서·grid·공유P, 원래 strict-v1의336채점행/전체 집계를 확인했다.
60파일럿과42표본은 ID가 겹치지 않는다. 고정 사례12개를 합계102에 다시 더하지 않는다.
별도 [CPU-v2 합집합](../analysis_exports/bottleneck_20261003/export_all30_102_v2/summary.json)의
816조건은749실호출/47차단/20제외, 누락·시스템 오류0이다. 출력624,416 token,
입력762,598(이미지491,158 포함), 개별 요청 시간 합15,369.813초다. 설치/로드를 포함한 wall time이 아니다.

| 조건 | 정답/적용 가능한 완료 조건 | NO_PARSE |
|---|---:|---:|
| I | 40/102 | 24 |
| T0 | 37/102 | 13 |
| O0 | 42/85 | 17 |
| O1 | 31/69 | 13 |
| IO0 | 40/85 | 15 |
| M | 33/102 | 33 |
| MI | 39/102 | 21 |

분모가 다른 조건의 정확도를 단순 차감하지 않는다.
[동일 문항 전이표](../analysis_exports/bottleneck_20261003/export_all30_102_v2/transitions.csv)는
O0−I85개에서 회복15/악화10, O1−O0 69개에서7/7,
M−I102개에서10/17, MI−M102개에서15/9다.
모든 전체 paired 차이의 family-bootstrap95% 구간은0을 포함한다.
따라서 관측된 회복을 전체 성능 개선이나 내부 모듈의 유일한 책임으로 확정하지 않는다.

P102개는 schema96유효/6무효다. JSON 표현 비교가 가능한80개 중 전체 byte-equivalent 구조0개는
의미 구조 정확도0%가 아니다. 이후80개에 실제 AI 의미 검토를 수행했고 아래에 조건부 지표를 기록한다.
[완료 의무 점검](../analysis_exports/bottleneck_20261003/completion_obligations_independent_v1.json)에
점검 당시 측정 공백, 실제LEGACY 재호출 없음,3408/3409와 잔여160미실행을 별도 남겼다.
계획서의 critical 관계 목록을 결과를 본 뒤 사전 정의했다고 소급해서 표시하지 않는다.

### 4.8 P의 실제 의미 검토 — 원래 이미지가 아닌 O0 조건부

두 disjoint40문항 검토를 합친80문항에서 전체 list-item 단위를 비교했다.
독립 감사는12판정과 행/열 역할64항목·곡선 표현4항목의 한정된 후속 확인을 수행했다.
Accounting_15와 Accounting_5의 열 제목을 행 제목으로 인정한 경우, Physics_7의 C가 B보다 아래라는
주장, Materials_2의 참조에 없는 curved 표현 등4문항7판정을 수정했다. 원래 v1은 덮어쓰지 않았다.

| 지표 | 전체 적용 가능한 분모 | 판정 가능한 항목만 사용한 분모 |
|---|---:|---:|
| 생성된 구조 항목의 참조 지지 |1309/1877 ·69.74%|1309/1347 ·97.18%|
| 참조 구조 항목의 회수 |710/1382 ·51.37%|710/930 ·76.34%|
| 관계 필드의 참조 지지 |347/815 ·42.58%|347/375 ·92.53%|
| 참조 관계의 회수 |230/513 ·44.83%|230/336 ·68.45%|

전체 생성 항목에는 미확정530개, 참조 항목에는 미확정452개가 있다. 오른쪽 열은 이를 제외하므로
전체 구조 해독 성공률로 해석하면 안 된다. tuple/object는 분해하지 않았고, 의미가 일부만 겹치는
복합 주장은 미확정으로 유지했다.22비교 불가 문항은 성공·실패 어느 분모에도 몰래 합치지 않았다.
O0 자체가 AI 전사이며 원래 이미지를 모두 사람이 인증한 정답이 아니다. 독립 검토도80문항의 모든
의미 판정을 재인증한 것은 아니다. Energy_and_Power_20 uncertainty의 질문 유래 문구는 독립 판단에서
제외했으며, 모든 메타데이터가 질문과 완전히 독립적이라고 주장하지 않는다.
필요 구조 완전 일치율·보이지 않는 내용의 발명 비율·내부 모듈의 인과 책임은 여전히 NOT_MEASURED다.

근거: [수정본](../analysis_exports/bottleneck_20261003/p_structure_measurement_corrected_all30_v2/summary.json),
[독립 감사](../analysis_exports/bottleneck_20261003/p_structure_measurement_merged_independent_v1.json),
[7판정 수정 원장](../analysis_exports/bottleneck_20261003/p_structure_measurement_correction_receipt_v2.json),
[38hash·102결정·전체 집계 재현](../analysis_exports/bottleneck_20261003/p_structure_measurement_corrected_main_check_v2.json).

## 5. Gemma900시도의 실제 결과 — 전수 판정은 미완료

고정 Gemma revision: `ee0ef6023621cff504d758262d4e04895a5af4a2`.
새 submission-only protocol의 실제 tokenizer로900/900 전체 입력을 검사했다.
최대50,454(+출력1024=51,478), 문맥 초과0, 원문 절단0이며 CPU 단계 judge 호출0이다.
과거 입력 합/백분위와 새 protocol의 수치를 혼합하지 않는다.
다운로드 bytes·정상 설치·native imports와 CPU 입력 검사는 GPU/KV 메모리 적합성을 증명하지 않는다.

실제 Gemma29사례 smoke는 통과·로컬/독립 검증됐고 같은 code/model/template/CPU gate로
저장900응답을 모두 시도했다.29는 최종 출력 계약의 통과이며, 일부 의미 witness와
reason의 오류 때문에 완전한 판정 정확도 인증으로 보지 않는다. 이후 실패는 그대로 보존했다.
추출에는 question·choices·gold·V8 판정·이미지를 주지 않는다. 문자 없는 객관식 답의 보기 연결과
주관식 의미 비교는 이미 추출한 답을 고정한 별도 단계다. 판정 불일치·모호성은 REVIEW,
시스템 실패는 SYSTEM_ERROR로 분리한다.
Gemma 후보 결과를 원래62%에 자동 반영하거나 사람의 검증 점수로 표시하지 않는다.

### 5.1 완료율과 유효 판정률을 구분

01:20:46–02:39:37 UTC의 감사 프로세스 wall time은4730.943초(1시간18분51초),
개별 요청 시간 합은4722.521초다.900개 모두 실제 추출 응답을 받았고 추가 동치 응답2개를 포함해
902호출의 입력5,749,823/출력84,499 token을 보존했다. 보기 의미 연결 호출은0이다.
missing/재시도/BUDGET_STOP은0이지만, 아래293개가 남아 유효 전수 판정 gate는 통과하지 못했다.

| 상태 | 문항 수 | 해석 |
|---|---:|---|
| RESOLVED |607|고정 가드에서 판정 가능; 독립 의미 정확도 인증은 아님|
| REVIEW |117|모호성·불일치·동치 판정 등 정책상 검토 필요|
| SYSTEM_ERROR |176|Gemma 출력·근거 계약 위반; Qwen 오답으로 세지 않음|

176실패의 원인은 서로 겹치지 않게162/8/5/1로 나뉜다.

- 162문항:342개 근거 인용이 Qwen 원문의 정확한 부분 문자열이 아니었다. Markdown 차이뿐 아니라
  선택지 문자·수식 기호를 바꾼 사례도 있어 단순 정규화로 모두 구제할 수 없다.
- 8문항: 추출한 답이 제시한 원문 인용에 그대로 존재하지 않았다.
- 5문항: Gemma 자체 출력 상한1024에서 JSON이 미완료됐다. Qwen의32768 잘림과 다른 실패다.
- 1문항: `validation_Energy_and_Power_2`에서 제어문자가 포함된 잘못된 JSON이 반환됐다.

원문 인용 가드는 이 출력들을 올바르게 거부했다. 원인은 judge의 증거 생성·출력 신뢰성 한계이며,
이 실패를 V8 오류나 Qwen의 문제풀이 오류로 바꾸어 세지 않는다. 가드를 제거하면 오히려
근거 없는 답·변경된 기호를 수용할 위험이 있다. 모든900입력/원문/entry/model/protocol/실제
token-ID/usage를 검증했고,607+117판정과176실패 경로를 frozen CPU 함수로 그대로 재현했다.
실제 데이터에서 검증 실패가19.56%이므로 이번 Gemma 설정을 최종 채점기로 채택하지 않는다.
RESOLVED만의 정확도나293개를0점 처리한 값을 전체 Qwen 정확도로 발표하지 않는다.

근거: [닫힌900 주 검증](../analysis_exports/bottleneck_20261003/cloud_gemma_final_v8/local_receipt.json),
[독립 회수·계약 검증](../analysis_exports/bottleneck_20261003/gemma_final_recovery_independent_v1.json),
[293개 검토 큐](../analysis_exports/bottleneck_20261003/gemma_final_export_v1/review_queue.jsonl),
[176개 실패 원장](../analysis_exports/bottleneck_20261003/gemma_final_export_v1/error_root_causes.jsonl).

### 5.2 발견한 두 정답 회수 후보와 실제 동치 호출

- `validation_Music_5`: 원문 첫 행의 완결된 `B. False`를 추출했다. 뒤의 혼란스러운 음높이 설명은
  해당 True/False 답을 명시적으로 철회하거나 다른 답으로 바꾸지 않았다. 제출답의 회수 후보는
  독립 원문 검토에서도 유효했지만, 실제 이미지·풀이의 사실성을 인증한 것은 아니다.
- `validation_Public_Health_28`: `C. 12.96`은 실제로 반복 제출한 답이다. 그러나 같은 응답의
  “정답은13” 주장과 모순·비결속 근거 flag가 있어 REVIEW를 유지한다. 이번 검토는 candidate를
  사전에 알았으므로 완전한 독립 blinding을 주장하지 않으며 점수에 자동 반영하지 않는다.
- 실제 동치2호출(`validation_Chemistry_4`, `validation_Pharmacy_26`)은 동결한 제출답을
  고치지 않고 모두 NOT_EQUIVALENT를 반환했다. 두 사례로 일반 의미 판정 신뢰성을 인증하지 않는다.

근거: [네 실제 사례의 독립 검토](../analysis_exports/bottleneck_20261003/gemma_final_candidates_independent_v1.json).
새 전체 점수는 **null**, 원래 V8 **558/900·62.00% 유지**다. 유효 후보1개를 발견했다는 것과
나머지899판정이 검증됐다는 것은 다르다. 추출의 원문 증거 계약을 재설계한다면 별도 protocol의
gold-blind 회귀·독립 검증을 먼저 해야 하며, 현재 실패를 소급 수리하거나 새 유료 실행을 시작하지 않는다.

다음 문단은900감사를 시작하기 전의 환경·검증 실패 경과다. 최종 상태는 위 표와
[실제29 smoke receipt](../analysis_exports/bottleneck_20261003/cloud_gemma_passed_smoke_v8/local_receipt.json)가 우선한다.

실제 v1 서버는22:31:04 UTC에 실패·자식 정리됐다. 로그의 원인은 FlashInfer sampling JIT의
`nvcc`/`/usr/local/cuda` 부재다. vLLM0.24의 지원 환경변수
`VLLM_USE_FLASHINFER_SAMPLER=0`을 명시적으로 고정한 별도v2 controller는
22:50:08 UTC에 같은BF16/131072 서버를 READY로 확인했다.
이후 smoke가 호출 전에 runtime버전`0.24.0`과 휠버전`0.24.0+cu129`를 잘못 동등 비교하여 차단됐다.
뒤의 EngineDead 로그는 controller의SIGTERM 정리 이후이며 새 근본 오류로 세지 않는다.
휠 digest 검증은 유지하고 runtime public버전과 휠 CUDA build를 분리하는 새 CPU 회귀를 추가했다.
원래 v5 preflight/CPU900v2/launcher/실패 결과는 변경하지 않고, 새 소스의900 tokenizer gate를
별도 실행했다. 새 CPU900v3의900행 token 원장은 기존v2와 byte가 동일하고,
새코드 identity·900개 entry hash·요약·실패v2의 실제호출 전 차단을 로컬 검증했다.
새controller v3는23:03:59 UTC 시작,23:05:07 UTC 같은서버 READY 후 실제13smoke를 실행했고
10통과/3실패로900감사 전에 중단·소유자식 정리했다. 복수 후보의 revision evidence 누락,
같은 금액·방향에서 통화 기호만 다른 표기 동치 판단, 가장 긴 입력의CPU/server token불일치를
분리했다. 마지막 문제는 server auto OpenAI-list 변환에 의해 system 경계 공백이1token 추가된 것으로
실제 동일 tokenizer 재현에서 전체 token-ID를 대조했다. 길이차이를 허용하는 우회는 하지 않는다.
새v4 prompt는 evidence 의무와 한정된 표기동치를 명시하고 native string content-format을 고정한다.
validator·비교 규칙·13사례 통과 기준 및 모델/BF16/context/sampling은 유지한다.
local52tests/Ruff/mypy3/Bash·독립 원인 검토 후 실제CPU900v4는900COUNTED/overflow0,
최대50061(+1024=51085)·전체응답 절단0이다. 새 prompt protocol이므로 v3 token과 동일하다고 주장하지 않는다.
[실패v3 원문 회수](../analysis_exports/bottleneck_20261003/cloud_gemma_failed_smoke_v3/local_receipt.json),
[원인 독립 검토](../analysis_exports/bottleneck_20261003/gemma_smoke_failure_review_v1.json).
[버전guard 독립 검토](../analysis_exports/bottleneck_20261003/gemma_version_identity_review_v1.json),
[새CPU900/실패기록 회수](../analysis_exports/bottleneck_20261003/cloud_gemma_cpu900_v3/local_receipt.json).
READY만으로 실제50K 입력·13사례·900감사 성공을 주장하지 않는다.

## 6. 결론의 현재 경계

지정 revision을 Transformers4.57.1의 실제 meta-model `named_modules`/parameter shape로
조사했다. Vision306,242,560 / 연결부109,105,152 / Language4,022,468,096 parameters를
중복 없이 분리했다. 연결부에는 `model.visual.merger`와 `deepstack_merger_list.0/1/2`를 모두 포함한다.
Vision attention의 `attn.qkv`/`attn.proj`와 Decoder의 `q_proj/k_proj/v_proj/o_proj`는 다르다.
이는 존재하는 모듈 이름을 확인한 것이며, 아직 LoRA 대상을 선택하거나 학습 효과를 검증한 것은 아니다.
근거: [module_inventory_final.json](../analysis_exports/bottleneck_20261003/module_inventory_final.json).

확인된 현상은 구조 추출의 실제 누락과 입력/문제 분해에 따른 회복·악화가 모두 존재한다는 것이다.
원시 전사와 해석된 상태가 일부 사례에 도움이 된다는 기능적 가설은 지지하지만,
Vision Encoder·마지막 Merger/DeepStack Merger·Decoder 중 유일한 원인 모듈은 아직 식별하지 못했다.
O1 악화, MI 악화, 답 형식 혼동은 단일 원인 설명의 반례다.
완료된30과목102문항과15분야 Knowledge60의 관측을 바탕으로 §8의 조건부 학습 후보를 제안한다.
Gemma900시도는 종료했지만 유효 전수 판정·사람의 정답 판정과 실제 학습 성과는 미완료다.
학습 데이터는 독립 외부 출처여야 하며, 현재 benchmark 전사나 응답을 학습에 사용하지 않는다.

대기 중 [공식 출처 기반 외부 후보 조사](research_diagnostic_external_candidates_20261003.md)를 준비했다.
악보 기호/관계(MUSCIMA++), 도식 구성/관계(AI2D), 의료 이미지 분류(MedMNIST),
텍스트·표 정량 추론(FinQA)을 서로 다른 과제로 구분했다. 특히 의료 분류는 현재stress의
확인된 실패와 직접 연결하지 않았고, AI2D의 원본 라이선스와 FinQA 원천 보고서 권리는
확정하지 않았다. 모두 데이터 중복·권리 검토 전의 조건부 후보이며 다운로드/학습/모듈 선택은 미실행이다.

## 7. 실행 비용·메모리 증거의 범위

각 Qwen 응답의 장치 메모리는 요청 후 snapshot이다. 별도1초 주기 장치 기록은
2026-10-02 19:21:39 UTC부터 시작했으므로 초기 실행을 포함한 true peak나 worker allocator peak가 아니다.
첫 관측은21,976MiB였으며 이 값을 최대 메모리로 확정하지 않는다.
[sampler receipt](../analysis_exports/bottleneck_20261003/device_sampler_v1_launch_receipt_local.json).

삭제 후 [실제 청구 조회](../analysis_exports/bottleneck_20261003/owned_pod_billing_after_cleanup_v3.json)는
조회 시 반환10 bucket의GPU USD1.9209609823301435/disk USD0.2795138885267079,
합USD2.2004748708568513다. 마지막 bucket은 삭제 직전까지의 비용을 모두 반영하지 않을 수 있어
최종 청구서나 프로젝트 전체 비용으로 부르지 않는다. 총액을 요율×시간 추정으로 대체하지 않는다.

닫힌1초 sampler26,370개에서 장치 사용량의 관측 최대는22,768MiB였다.
기록 시작19:21:39.858–종료02:41:16.629 UTC이며 초기 실행은 빠졌으므로 true peak가 아니다.
마지막 샘플은GPU0%/1MiB, controller와 소유 server descendants의 종료 증거를 보존했다.
감사42 payload+manifest의 archive8,668,683bytes, 실패 controller 이력23 payload의 archive128,371bytes를
기존 Qwen·Knowledge·환경 receipt와 묶어18개 로컬 검증 근거로 삭제 gate를 통과했다.
03:03:45 UTC의 삭제204 이후 get404·동일ID 목록0·network volume0을 확인했다.
생성한30GB컨테이너와200GB Pod 작업 디스크는 폐기돼 원격 복구 불가지만 결과는 로컬에 있다.
별도 Network Volume은 생성하지 않았고 다른 자원은 변경하지 않았다.
실제902호출은 최대50,454입력+1024출력 범위였으며 전체131072문맥 GPU 검증으로 부르지 않는다.
[최종 로컬 gate](../analysis_exports/bottleneck_20261003/final_local_cleanup_gate_v1.json),
[삭제·부재 receipt](../analysis_exports/bottleneck_20261003/owned_pod_cleanup_final_v1.json).

다음 조회·설치 자료는 당시의 역사 기록이다. 이전
[읽기 전용 청구 조회](../analysis_exports/bottleneck_20261003/owned_pod_billing_snapshot_v2.json)는
17:00–02:00 UTC의 반환9 bucket 합 USD1.9485288984142244다. 조회 결과의 마지막 bucket은
진행 중 집계이므로 작업 전체의 최종 청구가 아니다. GPU1.7009594542905688/disk0.24756944412365556이며,
반환 청구를 요율×경과시간 추정값으로 대체하지 않는다. 아래 초기 조회는 역사 기록이다.

RunPod 읽기 전용 도구로 이 작업의 소유 Pod만 청구를 확인했다. 반환된17:00–19:00 UTC
두 bucket 합은 GPU USD0.2525156931951642 + disk USD0.03726851847022772 =
USD0.2897842116653919다. 현재19:00 이후를 포함한 최종 프로젝트 비용이나 청구서가 아니다.
실제 조회 hourly cost는USD0.22이며, 이 요율의 시간 곱으로 반환 청구를 대체하지 않는다.
[billing snapshot](../analysis_exports/bottleneck_20261003/owned_pod_billing_snapshot_v1.json).
최종 표에서는 설치·로드·추론·전송 wall time과 개별 요청 시간 합을 분리하고,
공유 P의 실제 호출은 한 번만 합산한다. 비용을 맞춘 단일/두 단계 추가 실험은 아직 수행하지 않았다.

설치 환경 원문에서 조회한 [선택 metadata](../analysis_exports/bottleneck_20261003/environment_metadata_read_v1.json)는
Qwen의Python3.12.3/Torch2.8.0+cu128/Transformers4.57.1/vLLM0.11.0와
별도Gemma의Torch2.11.0+cu129/Transformers5.5.3/vLLM0.24.0+cu129를 구분한다.
Qwen BF16 probe는 통과했지만, 이 설치 probe의allocator peak는 실제 추론의peak가 아니다.
원문doctor·두 환경의freeze/설치 manifest와 회수 코드5파일을 실제 Pod에서 암호화 전송으로 내려받았다.
[cloud_environment_v1](../analysis_exports/bottleneck_20261003/cloud_environment_v1/recovery_receipt.json)의
15파일·archive/member/추출 byte 검증을 통과했고, Qwen13파일 manifest는 지정 baseline과 일치한다.
[독립 환경 검토](../analysis_exports/bottleneck_20261003/cloud_environment_independent_receipt_v1.json)도
두 checkpoint/lock/회수 metadata/도구5파일의 불일치가 없음을 확인했다. 가중치 실물 재검증은 아니다.
archive17,735bytes/SHA05432b4bc4859556074a770a41a66a8848c720b3b6efe852dcd8f960fd5fd827이다.
가중치·venv/cache는 회수하지 않았으며 Gemma 설치 기록 자체도 `NOT_GPU_READY`다.
이 byte receipt는 실험 전체 완료나 Pod 삭제를 승인하지 않는다.

회수 도구는 기존 archive/storage 함수를 재사용했다. 독립 검토에서 발견한 입력 경로 범위 검사를
내용 읽기·해시 계산 전으로 옮기고 로컬/Pod CPU 각4검사 및 Ruff/mypy2파일을 통과했다.
이미 완료된stress12의5파일 실제 로컬 압축·복원은 byte/hash/96원장·94 raw hash 동일성을
[독립 검증](../analysis_exports/bottleneck_20261003/recovery_stress_roundtrip_independent_v1.json)했다.
이는 새 cloud 실험 회수나 정답 의미 인증이 아니다. 도구는 실행 중 추론과 다른
`/workspace/private_recovery_v1`에만 배치했으며 원래 inference/v5 Gemma gate는 변경하지 않았다.

## 8. 계획서의9개 최종 질문 — 현재 확인 가능한 답

아래 문항 사례는 모두 반복번호3407이다. 원문·모드·종료 사유·채점·hash의 연결은
[독립 검토](../analysis_exports/bottleneck_20261003/report_completion_review_v1.json)와
[주 검증:35모드 사례·4전이표](../analysis_exports/bottleneck_20261003/report_completion_main_verification_v1.json)에 기록했다.
주 검증은 기존 채점 원장의 재현·정체성 확인이며 이미지 정답이나 전사의 완전성을 새로 인증한 것이 아니다.

1. **O0로 회복한 사례:** 적용 가능한85쌍에서 회복15/악화10이다.
   `validation_Materials_2`, `validation_Physics_7`은 I오답→O0정답이며 두 조건 모두 stop/PARSED다.
   전체 회복·악화에 출력/추출 상태 차이가 섞일 수 있으므로 모두 순수 지각 효과로 세지 않는다.

2. **O1까지 제공했을 때 회복한 사례:** I오답·O0오답·O1정답은6개다.
   `validation_Music_25`, `validation_Economics_1`은 세 조건 모두 stop/PARSED다.
   나머지는 `validation_Electronics_10`, `validation_Architecture_and_Engineering_7`,
   `validation_Diagnostics_and_Laboratory_Medicine_3` 및 `_11`이다.
   O0→O1 회복7개 중 `validation_Materials_17`은 이미 I정답이므로 이6개와 혼동하지 않는다.

3. **해석 상태를 줘도 틀리는 사례:** O1완료69개 중 오답38개는 PARSED오답25/NO_PARSE13이다.
   `validation_Computer_Science_22`, `validation_Physics_18`의 O1은 stop/PARSED오답이다.
   O1은 AI 검토 전사이지 완벽한 정답 상태의 인증이 아니므로 남은 오답을 전부 Decoder 책임으로 단정하지 않는다.

4. **P가 정확하지만 I가 틀렸는가:** NOT_CERTIFIED다. 원래 이미지의 진실과 사전 중요 관계를
   기준으로 전체P 정확성을 인증한 사례는 없다. O0조건부로 모순·미확정·누락이 없는 좁은 항목
   기준을 통과한 `validation_Accounting_18`도 I는 정답이었다. 항목 지지율69.74%를 전체P 정확도로 바꾸지 않는다.

5. **M/MI 차이가 반복되는가:** NO_REPEAT다. 실제3407 한 번의102쌍에서 회복15/악화9를
   관측했지만3408/3409 반복은 미실행이다. `validation_Literature_14`는 M정답→MI오답으로,
   둘 다 stop/PARSED인 반례다. 단일 seed의 여러 문항 관측과 반복 재현성을 구분한다.

6. **개선과 악화:** 전체 동일 문항 전이는 아래와 같다. 분모는 각 조건의 적용 가능 교집합이다.

   | 전이 | 쌍 수 | 모두 정답 | 모두 오답 | 회복 | 악화 |
   |---|---:|---:|---:|---:|---:|
   | I→O0 |85|27|33|15|10|
   | O0→O1 |69|24|31|7|7|
   | I→M |102|23|52|10|17|
   | M→MI |102|24|54|15|9|

   [전체 문항 전이 원장](../analysis_exports/bottleneck_20261003/export_all30_102_v2/paired_cases.csv)과
   [전이 집계](../analysis_exports/bottleneck_20261003/export_all30_102_v2/transitions.csv)를 제공한다.
   전체 paired 차이의 family-bootstrap95% 구간은 모두0을 포함한다.

7. **남은 파서·자료 문제:** 완료된 최종답 조건647개 중 NO_PARSE136개는 length94/stop42다.
   이는136개의 고유 문항이나136개의 확정 파서 버그가 아니다. 불완전 응답·철회·형식 위반과
   비교 규칙의 한계를 구분해야 한다.47차단/20제외는 oracle 적용 gate이며 데이터셋 오류율이 아니다.
   AI 전사·gold 설명 이미지 역할·정답표 의문은 별도 원장에서 남기고, Gemma의 실제 실패도
   Qwen 오답이나 V8 오류와 합산하지 않는다. [647조건 채점 원장](../analysis_exports/bottleneck_20261003/export_all30_102_v2/scores_v2.jsonl).

8. **다음 selective fine-tuning 후보:** 아래는 후속 비교 우선순위이며 이번에 학습하지 않는다.

   - F0: 가중치 고정 대조군과 답 형식·자료 타당성부터 고정한다.
   - FL: 독립 외부 자료의 구조화된 상태→복잡한 규칙 적용을 사용하는 Decoder LoRA 비교를
     먼저 검토한다. O1의 PARSED오답25개, CS/Physics 사례와 Knowledge의 중간 풀이 오류가 근거다.
     단순 사실 암기를 추가하자는 결론이나 Decoder의 유일한 책임이라는 결론은 아니다.
   - FV 대 FM: 외부에서 검증된 기호·위치·연결 구조 supervision으로 Vision-only와
     연결부-only를 별도 비교한다. FM은 마지막 Merger와3개 DeepStack Merger를 명시한다.
     P의 누락·관계 회수 한계는 기능적 근거이며 어느 모듈이 원인인지 구분한 결과는 아니다.
   - 공동 업데이트는 분리된 후보의 외부 holdout 효과·악화 및 비용을 확인한 뒤 검토한다.
     가능한 동일 자료/업데이트/평가 예산, 실제 모듈 shape와 freeze 범위를 기록해야 한다.
     현재 MMMU 원본·전사·P·모델 출력·Knowledge 자료는 학습에 넣지 않는다.

9. **아직 모르는 것:** 유일한 오류 모듈, 중요 구조 완전성·이미지 진실, 반복 안정성,
   동일 비용/해상도 개입 효과, 전체 MMMU 정확도에 대한 영향, 신뢰할 수 있는 Gemma 보조 정확도다.
   K1/K2는 더 쉬운 다른 문항의 천장 효과가 있고, P는 전체 VLM을 통과한다. 현재 결과로
   Vision Encoder 또는 Merger를 분리 진단했다고 말하지 않는다. 실제 학습 효과는 미측정이다.

최대262문항 완료나 seed 반복을 수행한 것으로 표시하지 않는다. 현재 완료는102문항/749진단 호출과
60 Knowledge 호출이며,262 목표의 잔여160·LEGACY 신규 재호출·추가 seed는 미실행이다.
Gemma900시도·로컬 회수·소유자원 삭제는 종료했다. 유효900판정은 INCOMPLETE이며 전체 과학적
목표의 완료로 보고하지 않는다. 새 유료 실험·학습은 자동 시작하지 않는다.

## 9. 종료 후 CPU 원인 분석 — 긴 입력과 근거 출력 계약

추가 GPU 호출 없이 최종900행·81오류와 원문900개를 대조했다. 소유Pod는 삭제된 상태다.
이 분석은 실패 원인의 확인이지 새 Gemma 결과나 기존62.00%의 수정이 아니다.

**정답 대조 자체가 느린 것이 아니라, 전체 풀이를 읽고 확정답·철회·충돌을 감사하도록
구현한 작업량과 근거 출력 설계의 한계가 겹쳤다.** 기존V8 추출답과 답지만 비교하면 CPU로
가능하지만, 그 방식만으로는 V8이 실제 최종답을 놓치거나 잘못 추출했는지 확인할 수 없다.
그렇다고 현재 방식의 과도한 입력·출력을 필수 비용이라고 간주하지 않는다.

900개 raw response는 UTF-8 기준13,454,040bytes,295,080행이며 빈행130,302개를 포함한다.
매 행에 `{id,text}`를 붙인 user JSON은 원문 문자열 방식의13,859,346bytes에서
20,118,247bytes로6,258,901bytes 증가했다. 이는 **직렬화 byte 비교**이며 token·속도 측정이 아니다.
별도 실제 tokenizer 원장에서는 전체 입력5,748,953→7,820,217token(+36.03%)을 확인했다.
예를 들어 Electronics_17은8,174행의 원문106,538bytes에178,717bytes의 user JSON 차이가 생겼다.
전체 풀이와 후속 철회가 유지돼야 하지만, 빈행까지 JSON 객체로 전달하는 표현은 개선 여지가 있다.

| 최종 SYSTEM_ERROR | 문항 수 | 원문 대조 결과 |
|---|---:|---|
| 추출 답이 선택한 근거에 없음 |35|11개는 다른 원문 위치에만 존재,24개는 원문 어디에도 그대로 존재하지 않음|
| Gemma 출력 상한1024에서 미완료 |19|전부 근거ID 숫자 배열을 나열하다 종료; 설명문 길이 때문이 아님|
| 빈행ID 선택 |18|원문 위치는 존재하지만 비어 있어 host가 거부|
| 중복ID 선택 |8|모두 중복ID이며 범위 밖 ID는 없음|
| 잘못된JSON |1|Math_14의 문자열 안 literal U+000C 제어문자; 생성기/토크나이저·grammar 경계 원인은 미확정|

19개의 상한 종료는12개의 AMBIGUOUS/CONFLICTING_ANSWERS와7개의
CLEAR/COMPLETED_CORRECTION이다. Architecture_and_Engineering_15는218개의 revision ID를
나열했고, Art_21은16개의 고유 answer ID를164회, Music_8은57개의 고유 revision ID를162회
적었다. 실제 native schema에는 citation 배열의 `maxItems`와 uniqueness 제약이 없고,
ID 범위에는 빈행도 포함된다. 프롬프트의 “all material revision traces” 지시와 긴 반복 원문이
방대한 ID 출력을 요구하는 조합이었다. host의 빈행·중복·원문 일치 검사는 이를 올바르게
거부했으며, 이번 분석에서 잘못된 host 거부는 확인하지 못했다. 거부가 정당하다는 것과
Qwen 답이 틀렸다는 것은 별개다.

다음 수정은 별도 프로토콜로만 검증한다. 원문을 자르거나 gold로 후보를 고르지 않는다.

- 존재하는 비어 있지 않은 위치만 native ID 후보로 허용하고, pinned backend가 지원하는
  제약만 사용한다. host의 중복·범위·원문 검사는 그대로 유지한다.
- 최소 결정 근거를 유한하게 출력하도록 계약을 좁힌다. 근거 한도를 넘거나 정정 관계를
  완결하지 못한 경우는 REVIEW로 남기며, 필요한 철회를 묵시적으로 생략하지 않는다.
- 답의 case/Markdown/TeX 표기를 임의로 다시 쓰는 경로는 원문 span 선택 후 host slicing을
  우선 검토한다. 형식 차이를 모두 수용하도록 가드를 완화하거나 잘못된JSON을 관대하게
  읽는 방법은 사용하지 않는다.
- 별도 CPU 반례·native schema 지원 확인·실제 소규모 GPU gate를 통과하기 전900개를
  재실행하지 않는다. 단순히 출력 상한을 올리는 것이 반복ID의 근본 해결은 아니다.

근거: [CPU 원인 원장](../analysis_exports/bottleneck_20261003/gemma_segments_closed_cpu_root_causes_v1.json),
[변경 전 실제 원장](../analysis_exports/bottleneck_20261003/cloud_gemma_segments_continuation_final_v1/continuation/audit/ledger.jsonl),
[기존 입력token 비교](../analysis_exports/bottleneck_20261003/gemma_segments_input_cost_comparison_v1.json).
읽기 전용 Sol 독립 검토와 main CPU 재현을 사용했으며 사람의 의미 판정 인증은 아니다.
runtime·V8·원문·확정 점수·MMDL/GitHub는 변경하지 않았다. 유효900 Gemma 감사와 그에 따른
새 점수는 여전히 미완료다. 사용자가 추가 GPU 실행 없이 원인 분석만 진행하도록 명시했으므로
새 Pod·추론·학습은 시작하지 않는다.

### 9.1 동일749문항의 두 Gemma 프로토콜 대조

최초 literal 실행900개와 최신 segment 실행 중 실제 응답을 받은749개를 대조했다.
동일 ID의 원문 hash와 저장된 V8 답·정오가 모두 같음을 확인했다. 최신 시간 중단151개는
paired 비교에서 제외하지만 전체900 분모나 완료 기록에서 삭제하지 않는다.

| 상태 | 기존 literal / 동일749개 | 최신 source-segment / 동일749개 |
|---|---:|---:|
| 판정 가능 RESOLVED |494|558|
| 의미·정책 검토 REVIEW |101|110|
| 출력·근거 계약 오류 SYSTEM_ERROR |154|81|

기존 오류154개 중83개는 판정 가능,29개는 의미·정책 검토로 이동했고42개는 여전히 오류다.
반대로 기존 판정 가능494개 중12개는 오류,12개는 검토로 이동했다. 기존 검토101개에서도
27개가 새 오류로 바뀌었다. 따라서 오류의 순감소73개가 전체 검증 완료나73개 정답 회복은 아니다.
양쪽 모두 판정 가능한470개에서는 정오 판정 변경이0개였지만, 일치는 의미 정확도의 독립 인증이 아니다.
두 방식이 모두 답을 추출한 사례 중23개의 답 문자열이 달랐고22개는 host의 보기 연결 결과가
같았다. 나머지 Art_6은 기존D 연결에서 새 미연결/REVIEW로 이동했으며 다른 답의 확정이나
유효 정오 전환이 아니다.

동일749개의 실제 추출 입력은5,159,076→7,105,252token(+37.72%), 출력은70,763→73,575token이다.
이는 앞의 전체900 tokenizer 비교(+36.03%)와 표본·측정 범위가 다르다. 어느 비율도 속도 배율이 아니다.

이 대조는 **순수 직렬화 실험이 아니다.** checkpoint·가중치·BF16·tokenizer/template·vLLM wheel·
XGrammar0.2.3·seed3407·temperature0·출력1024·thinking=false는 유지했지만, 프롬프트와 native
schema, 근거 생성 방식, 근거 위치의 의미가 함께 달라졌다. literal은 같은 인용의 모든 등장
위치를 검사하고 segment는 명시적으로 선택한 위치만 검사하므로, 반복·인용 문맥의 review flag도
달라질 수 있다. 공통 안전 조건을 완화한 것이 아니라 검사에 들어가는 근거 위치가 달라진 것이다.
호스트 driver/kernel도580.95.05/6.8.0-87→580.159.03/6.8.0-124로 바뀌었고 lock의 차이는
MarkupSafe3.0.3→3.0.4, mcp/mcp-types2.2.0→2.3.0이었다. 단독 원인의 인과효과로 해석하지 않는다.

결론은 줄 번호 방식이 비원문 근거 생성 경로의 오류를 일부 줄였지만, 긴 입력과 무제한 ID
출력·답 표기 변경·의미 오판까지 해결하지는 못했다는 것이다. 먼저 출력 계약과 근거 선택을
안정화해야 하며, 현재 자료만으로 Gemma를 신뢰 가능한 최종 채점기로 채택하거나 새 점수를
확정할 수 없다. runtime·원문·V8·MMDL/GitHub와 실행 상태는 변경하지 않았다.
근거: [동일749개 CPU 전이 원장](../analysis_exports/bottleneck_20261003/gemma_literal_vs_segments_paired_cpu_v1.json).
