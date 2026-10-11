# G6 original full10K 네 조건 최종 분석 — 2026-10-05

비공개 내부 분석. **FL/FM/FV/FML 모두 실제 original10K 학습·fresh-process reload·두 paired60 평가·전체 체크포인트 로컬 회수/검증을 완료했고, 소유 Pod도 삭제·부재 확인을 완료했다.** 사전 고정 외부 개발 지표는 FM이 가장 높지만, 동일 MMMU60에서는 모든 학습 조건이 F0보다 낮다. 따라서 이번 결과는 외부 개발 과제의 제한된 개선과 MMMU 유지 위험을 함께 보여 주며, 일반 멀티모달 능력의 개선이나 규칙 학습 성공을 입증하지 않는다.

최종 상태의 정본은 [four-arm local gate](../analysis_exports/g6_runtime_20261005/final_closed_controller/local_receipt.json)와 [삭제 receipt](../analysis_exports/g6_runtime_20261005/pod_cleanup_receipt.json)이다. 이전 [독립 지표 감사](../analysis_exports/g6_runtime_20261005/final_metric_independent_review.md)의 “FML tensor/full gate pending” 문구는 감사 당시의 상태 기록이며, 이후 최종 local gate PASS로 해소됐다. 이전 원문과 receipt는 수정하지 않았다.

## 1. 실제 실행 조건과 완료 범위

실행 commit은 `447c0fb6056d2e18618dd22f21ef80197cb57103`, 모델은 지정 `Qwen/Qwen3-VL-4B-Instruct` / revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, BF16·비양자화이며 실제 장비는 단일 RTX3090 24GB다. 네 조건은 각각 fresh F0에서 시작했다. FL은 language LoRA, FM은 네 visual merger의 실제 파라미터 학습, FV는 vision LoRA, FML은 merger+language LoRA다. FM을 merger LoRA라고 부르지 않는다.

네 조건은 같은 승인된 10,000행을 같은 순서로 1 epoch 소비하고 각각 **625 optimizer updates**를 완료했다. micro-batch1 × accumulation16이며, 실제 공통 노출은 images10,000 / input tokens8,374,856 / visual tokens7,398,051 / target tokens210,728이다. 입력 혼합은 TRACE9,000 + 자체 생성 NVW1,000이고 MMMU/benchmark 파생본은 학습에 사용하지 않았다. 학습 행 수와 독립 root 수는 같은 개념이 아니다.

학습 recipe는 AdamW, gradient clipping1.0, seed3407이며 LoRA rank16/alpha32/dropout0.05를 고정했다. 학습률은 L5e−5 / M1e−5 / V1e−5로 group별로 다르다. 같은 노출과 optimizer update 수가 같은 학습 용량이나 모든 하이퍼파라미터의 동등성을 뜻하지 않는다.

MMMU 진단은 validation **30과목×2=60**, P0, 출력 상한 **4096**, 동일 이미지 조건·관측 입력·sampling·seed·V8 CPU scorer를 사용했다. sampling은 temperature0.7 / top_p0.8 / top_k20 / repetition_penalty1.0 / presence_penalty1.5다. 실제 backend는 `transformers-bf16-sdpa`, 이미지 budget은1,003,520–4,014,080 pixels이며 각 학습 조건을 이번에 새로 실행한 F0와 비교했다. 외부60도 각 조건에 같은 고정 cohort와 recipe를 전달했다.

조건·실제 노출 근거: [고정 protocol](../analysis_exports/g6_runtime_20261005/closed_stages/source/protocol.json), [최종 report.json](../analysis_exports/g6_runtime_20261005/final_closed_controller/report.json), [FL 실제 노출/출력 진단](../analysis_exports/g6_runtime_20261005/first_full_FL_score_drop_diagnosis.json). Protocol 파일의 과거 `DATASET_GATE_PENDING` 표식은 조건을 고정할 당시의 문구이며, 실행·승인 완료 판단은 최종 receipt를 따른다.

## 2. MMMU60: 모든 조건에서 유지 성능 하락

| 조건 | 정답 /60 | 정확도 | F0 정답36 중 유지 | 오답→정답 | 정답→오답 | 순변화 | F0 대비 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 | 36 | 60.00% | 36 | — | — | — | — |
| FL | 22 | 36.67% | 17 | 5 | 19 | −14 | −23.33 pp |
| FM | 27 | 45.00% | 25 | 2 | 11 | −9 | −15.00 pp |
| FV | 21 | 35.00% | 21 | 0 | 15 | −15 | −25.00 pp |
| FML | 23 | 38.33% | 17 | 6 | 19 | −13 | −21.67 pp |

MMMU60에서 FM의 감소가 네 조건 중 가장 작지만, 이것은 외부 모듈 선택의 입력이 아니다. FM도 F0보다9개 낮으므로 이번 paired60의 MMMU 개선 조건이라고 부를 수 없다. FL/FML의 회복5/6개는 각각 악화19개를 상쇄하지 못한다. FV의 정답21개는 모두 F0에서도 정답이었던 항목이며 새 회복은0개다.

**기존 V8 558/900·62.00%·32768-token 결과는 불변이다.** 이번60/4096과 문항 범위·출력 조건이 다르므로 두 결과 사이의 개선/악화량을 직접 계산하지 않는다. 이번 학습 조건의 전체900 성적은 측정하지 않았다. 근거: [최종 paired 결과](../analysis_exports/g6_runtime_20261005/final_closed_controller/report.json), [기존 독립 감사의 원문 연결](../analysis_exports/g6_runtime_20261005/final_metric_independent_review.md).

## 3. 짧은 출력 변화와 정확도 하락을 구분

| 조건 | length /60 | NO_PARSE /60 | 단독 LETTER+보기 한 줄 /60 | answer-only JSON /60 | 출력 token 합 | 중앙값 |
|---|---:|---:|---:|---:|---:|---:|
| F0 | 12 | 14 | 2 | 0 | 93,845 | 720 |
| FL | 2 | 4 | 51 | 0 | 12,769 | 8 |
| FM | 14 | 14 | 25 | 0 | 70,160 | 178.5 |
| FV | 15 | 16 | 4 | 0 | 95,611 | 525 |
| FML | 4 | 5 | 44 | 0 | 27,028 | 9 |

단독 한 줄은 stripped raw 전체가 `[A-Z][.)]\s+[^\n]+`와 일치한 경우이며, answer-only JSON은 정확히 `answer` key 하나인 제출 객체다. **짧은 LETTER 응답을 JSON 출력으로 보고하지 않는다.** length와 NO_PARSE는 겹치므로 단순 합산하지 않는다. FML에서는 둘의 교집합3개이고 length 정답1개다.

FL과 FML에서 정답→오답19개는 **모두 F0/FT 양쪽 PARSED이며 자연 stop**이다. 따라서 이들19개 손실을 출력 잘림이나 파서 실패 증가만으로 설명할 수 없다. FM의 손실11개 중8개, FV15개 중11개도 양쪽 PARSED/stop이다. 출력 양식·분포 적응과 정답 선택 변화가 관측되지만, 짧은 출력 자체가 추론 실패나 망각의 인과 증거는 아니다. Negative transfer/forgetting은 후속 분리 실험이 필요한 가설이다.

TRACE 학습 target의 answer-only 성격과 MMMU의 단답 증가 사이에는 가능한 연관성이 있으나, 이 실험은 target 형식만 바꾼 ablation이 아니다. 근거: [raw 기반 독립 감사](../analysis_exports/g6_runtime_20261005/final_metric_independent_review.md), [FL 기존 출력 진단](../analysis_exports/g6_runtime_20261005/first_full_FL_score_drop_diagnosis.json).

## 4. 외부 개발60: QA 향상과 구조/상태/규칙을 분리

외부 cohort는 QA12 / TRACE-QA24 / structure12 / grounded12다. 전체60은 **36 roots /32 families**이며, NVW의 같은 root를 여러 task view로 평가한다. 따라서60개 독립 표본이나 독립 전문 분야 전이 시험으로 해석하지 않는다. Registered-family development이고 unseen DSL/renderer, 봉인 holdout, 비합성 professional transfer는 이후 별도 평가다.

| 조건 | QA /12 | TRACE-QA /24 | Whole structure exact /12 | Grounded 최종 task 성공 /12 | 혼합 성공수 /60 |
|---|---:|---:|---:|---:|---:|
| F0 | 4 | 7 | 0 | 0 | 11 |
| FL | 4 | 10 | 0 | 4 | 18 |
| FM | 7 | 11 | 0 | 5 | 23 |
| FV | 6 | 6 | 0 | 0 | 12 |
| FML | 6 | 9 | 0 | 5 | 20 |

혼합 성공수는 서로 다른 task 성공 flag의 합이며 MMMU 정확도도, 사전 primary score도 아니다. QA/TRACE의 paired 회복/악화는 각각 FL +2/−2 및 +7/−4, FM +3/−0 및 +8/−4, FV +3/−1 및 +3/−4, FML +4/−2 및 +6/−4다. Grounded는 F0 성공0이므로 학습 조건의 성공수와 회복수가 같고 악화0이다.

Grounded `correct`는 **option, typed answer, answer_identifiable**을 확인한다. 상태나 규칙 전체가 맞아야 하는 지표가 아니다.

| 조건 | Structure leaf F1 /관측수 | Structure 누락 /12 | Bbox hits/denominator | Supported states exact | 저장 rules_exact | 실제 rules target exact |
|---|---|---:|---|---|---|---|
| F0 | 0.201519 /12 | 0 | 0/63 | 0/12 | 10/12 | 0/2 |
| FL | 0.357947 /10 | 2 | 49/56 | 2/12 | 6/12 | 0/2 |
| FM | 0.195484 /6 | 6 | 0/24 | 0/12 | 10/12 | 0/2 |
| FV | 0.255590 /11 | 1 | 2/60 | 0/11 관측, 1누락 | 9/11 관측, 1누락 | 0/2 |
| FML | 0.481084 /12 | 0 | 56/63 | 1/12 | 9/12 | 0/2 |

Whole structure exact는 전 조건0/12다. F1/bbox는 일부 관측 metric이며 조건마다 schema/파싱 누락과 분모가 다르다. 특히 FM의 관측6개 평균을 F0의12개 평균과 누락 없는 동등 비교로 취급하지 않는다. FML은 구조 metric12개 모두 관측되고 F1/bbox가 개선됐지만, whole exact0/12와 상태1/12·규칙0/2를 함께 보고해야 한다.

Grounded target12개 중 실제 `rules`가 있는 것은 **2개뿐**이다. 나머지10개에는 rules key가 없고, frozen scorer는 `.get('rules')`의 absent/null 동치를 exact로 센다. **F0와 모든 네 학습 조건의 실제 규칙 target 점수는0/2**다. F0/FM의 저장10/12, FL6/12, FV9/11, FML9/12는 실제 규칙 학습 성공률이 아니다. FML의 저장9개도 모두 규칙 없는 absent/null 일치다.

FL과 FM의 grounded 회복 각각2개는 기존 F0 숫자문자열을 같은 값의 숫자형으로만 바꿔도 frozen scorer가 정답으로 회수한다. 이런 유형 변화는 관계/규칙 학습과 구분해야 한다. FML의5개 회복 중 같은 type-only 기준에 해당하는 것은0개이나, 이것만으로 규칙 학습을 인증하지 않는다.

근거: [최종 외부 group 집계](../analysis_exports/g6_runtime_20261005/final_closed_controller/report.json), [원문 재현/actual-rule 감사](../analysis_exports/g6_runtime_20261005/final_metric_independent_review.md), [기존 FL/FM type-only 감사](../analysis_exports/g6_runtime_20261005/partial_FL_FM_metric_audit.json), [외부 gold 원장](../analysis_exports/g6_prepared_inputs_20261004/runtime/external60/gold.jsonl).

## 5. 사전 고정 순위와 용량 교란

실제 사전등록 경로는 [module_selection_preregistration.json](../analysis_exports/g6_runtime_20261005/module_selection_preregistration.json)이며 등록 시각은 **2026-10-05T05:09:38Z**다. Original full10K arm의 외부 결과를 관측하기 전에 규칙을 고정했다. Primary는 `(QA/12 + TRACE/24)/2`, 즉 **`(2*QA_correct + TRACE_correct)/48`**로 두 답변 task를 동일 가중한다. Secondary는 `(structure_correct + grounded_correct)/24`, 다음은 실제 trainable numel이 작은 순, 마지막은 고정 FL/FM/FV/FML 순이다. MMMU·봉인·비합성 transfer 결과는 선택에서 제외한다.

| 순위 | 조건 | Primary /48 | Primary score | Secondary /24 | 실제 trainable parameters | 저장 tensor수 | 실제 학습초 |
|---|---|---:|---:|---:|---:|---:|---:|
| — | F0 | 15 | 31.25% | 0 | — | — | — |
| 1 | FM | 25 | 52.08% | 5 | 109,105,152 | 24 | 6,197.683 |
| 2 | FML | 21 | 43.75% | 5 | 142,135,296 | 528 | 7,998.105 |
| 3 | FL | 18 | 37.50% | 4 | 33,030,144 | 504 | 7,910.340 |
| 4 | FV | 18 | 37.50% | 0 | 2,359,296 | 96 | 8,005.754 |

현재는 최종 all-four local gate가 PASS이므로, 이전 감사의 부분 완료 차단 없이 사전 규칙을 적용하면 **FM > FML > FL > FV**다. FL/FV의 primary tie는 secondary4>0으로 해결되며 capacity tie-break까지 가지 않는다. FM25는 baseline gate15보다 높아 사전 정의한 외부 QA 신호를 통과한다. 이는 후속 실험을 위한 실용 recipe 순위이며 통계적 유의성·일반 개선·모듈의 본질적 우월성을 뜻하지 않는다.

특히 FM은109M merger 파라미터를 실제 학습하고 FV는2.36M vision LoRA만 학습하므로 용량이 크게 다르다. 데이터·updates가 같아도 trainable capacity·학습 group·group별 learning rate가 같지 않다. 따라서 “merger가 원래 병목”이나 “vision은 중요하지 않음”과 같은 인과 결론은 지원하지 않는다.

사전등록 원문은 `selected_arm:null`인 등록 당시 상태를 그대로 보존한다. 이 보고서는 검증된 네 결과에 대한 순위 계산이며, 그 파일을 소급 수정하거나 G7을 실행한 증거가 아니다. 학습 시간은 각 metadata 내부의 실제 training seconds다. Controller stage duration은 startup/teardown을 포함해 이 값보다 조금 길며 서로 다른 시간 범위다.

## 6. 전수 회수·검증과 정확한 소유 Pod 삭제

[최종 local receipt](../analysis_exports/g6_runtime_20261005/final_closed_controller/local_receipt.json)는 `all_four_full10K_complete:true`, actual10K/625, 같은 노출, terminal controller/outer watchdog/process probes PASS를 기록하고 각 arm의 기존 tensor 검증과 원문 CPU replay를 결합했다. 해당 파일 SHA256은 **`2ec35018f169710eb17ae7a4d093109d1a06ae31aa41cdc395698263b969d51c`**이며 삭제 receipt가 이 정확한 digest를 참조한다.

Checkpoint 전체 및 원문은 다음 Linux 외부 root에 보존한다. `pilot_<arm>/checkpoint/`가 실제 학습 결과이며 각 root의 `local_receipt.json`에 tensor·hash·fresh reload 근거가 있다.

- [FL 회수 root](/home/jang2/mmdl-artifacts/recovered/g6-full-FL-20261005-ikxy27/)
- [FM 회수 root](/home/jang2/mmdl-artifacts/recovered/g6-full-FM-20261005-22u/)
- [FV 회수 root](/home/jang2/mmdl-artifacts/recovered/g6-full-FV-20261005-22u/)
- [FML 회수 root](/home/jang2/mmdl-artifacts/recovered/g6-full-FML-20261005-22u/)

Windows 쪽 근거 사본은 [FL aggregate](../analysis_exports/g6_runtime_20261005/first_full_FL_local_recovery.json), [FM receipt](../analysis_exports/g6_runtime_20261005/closed_full10k_FM/local_receipt.json), [FV receipt](../analysis_exports/g6_runtime_20261005/closed_full10k_FV/local_receipt.json), [FML receipt](../analysis_exports/g6_runtime_20261005/closed_full10k_FML/local_receipt.json)에 있다. 각 arm receipt의 “other arms pending”/“not complete G6”는 해당 arm 회수 당시의 상태명이며, 모든 arm을 결합한 최종 closure receipt가 현재 전체 상태를 확정한다.

소유 Pod **`22u0uqi171v0cn`**는 검증 후 **2026-10-05 14:57:07 UTC** delete204를 받았고, **14:57:20 UTC** get404 및 계정 Pod 목록0 / network volume 목록0을 확인했다. Network Volume은 생성하지 않았다. Pod container30GB와 Pod workspace160GB 원격 데이터는 폐기됐고 원격 복구가 불가능하지만, 필요한 결과와 네 checkpoint는 로컬에 보존됐다. 이 부재는 receipt 시점의 관측이며 미래 계정 상태를 가정하지 않는다.

Closure receipt의 `pod_deleted:false`/`CLEANUP_PENDING`는 삭제 직전 증거의 의미다. 이후 [pod_cleanup_receipt.json](../analysis_exports/g6_runtime_20261005/pod_cleanup_receipt.json)의 `POD_TERMINATED_AND_ABSENCE_VERIFIED`가 삭제 상태를 확정한다. 새 GPU 실행은 없었다.

## 7. 실행시간 기반 비용 추정

Pod 생성 **01:55:08.301 UTC**부터 delete **14:57:07 UTC**까지는 **46,918.699초 = 13.032971944시간**, 약 **13시간1분59초**다. 이 전체 구간에는 설치·준비·학습·평가·회수가 포함된다.

| 계산 범위 | 적용 rate | elapsed 기반 금액 |
|---|---:|---:|
| GPU rate 추정 | USD0.22/h | USD2.867254 ≈ **2.87** |
| 보수적 all-in 운영 ceiling 추정 | USD0.30/h | USD3.909892 ≈ **3.91** |

계산은 `46,918.699/3600 × rate`다. USD0.30/h는 운영시 사용한 보수적 합산 상한 가정이며 실제 청구서가 아니다. 두 값을 더하지 않는다. 결제 bucket·과금 반영 시각·실제 저장비 또는 청구 반올림을 재구성한 금액이 아니며 현재 잔액도 측정하지 않았다. 생성 시각과 실제 GPU rate는 [삭제 전 read-back](../analysis_exports/g6_runtime_20261005/pod_cleanup_receipt.json)의 `predelete.createdAt`/`predelete.cost`를 사용했다.

## 8. 이번 결과로 확정하지 않은 것

Original10K 네 조건의 실행·평가·전체 회수·검증·소유 자원 정리는 완료다. 외부 development primary에서 FM이 사전 순위1위이고, 모든 학습 조건에서 MMMU60 유지 저하가 관측됐다. 실제 rules target0/2와 structure whole exact0/12는 상태/규칙 완전 학습을 주장할 근거가 되지 않는다.

**G7, 50K 확장, 새 full900/32768, 봉인 holdout, 독립 비합성/전문 분야 전이는 실행·완료하지 않았다.** 외부 개발 신호 하나만으로 이를 자동 시작하지 않는다. 특히 50K 확장에는 고정 protocol의 구조/상태·retention·비합성 transfer 조건이 추가로 필요하다. MMMU60 감소와 규칙 부재를 숨기고 전체 목표 완료나 기존558/900 대비 학습 개선으로 보고하지 않는다.

이 보고서 작성은 기존 닫힌 결과/receipt를 읽고 순위·산술·비용을 정리한 CPU 분석이다. 원문·scorer·source·공개 MMDL/MMDL_test·Git·PPT·모델·GPU를 변경하거나 새 유료 실행을 시작하지 않았다.
