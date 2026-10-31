# Qwen3-VL-4B: 지도 미세조정 없이 수행하는 검증 기반 자기수정 RL

상세 연구·데이터 검증·구현 설계 보고서 · 최초 작성 2026-10-16 / 근거·구현 경계 재검토 2026-10-17 KST

**문서 상태: RESEARCHED / DESIGNED. 아래 G4 episode 설계의 구현·GPU 학습·성능 검증 완료를 뜻하지 않는다. 별도 v2 RL 코어의 CPU 구현과 이 설계의 실행 증거는 구분한다(§13).**

내부 작업 위치: `C:\projects\multimodel`. 교수님용 `MMDL/`, GitHub, 기존 V8 파서·응답·558/900 결과는 변경하지 않는다. 이 문서와 [파이프라인 도식 HTML](verified_self_correction_rl_pipeline_20261016.html)만 이번 요청의 산출물이다. HTML은 외부 라이브러리 없이 열리는 벡터 도식이며 사진 촬영본이나 실험 결과 그림이 아니다. 파일명의 날짜는 최초 작성일을 유지한다. 별도로 존재하는 `research_error_notebook_self_correction_rl_plan_20261016.md`와 그 도식은 수정하지 않으며, 서로 다른 설계의 그룹 크기·손실·학습 범위를 섞지 않는다.

## 0. 결론부터: 무엇을 만들 것인가

권장안은 **외부 시각 문제 → Qwen의 자체 풀이 → 독립 프로그램 검증 → 최대 두 번째 풀이 → 검증 보상에 따른 정책 업데이트 → 새 문제의 첫 시도 평가**다. 학습 중에는 정답을 검증기만 보며, 모델에는 정답 그 자체를 알려주지 않는다.

사용자가 제안한 “풀고, 틀리면 기존 풀이 아래에 다시 풀고, 오답노트를 만든다”는 흐름은 유지하되 다음을 수정한다.

1. **풀이를 쓰는 것과 가중치를 학습하는 것을 분리한다.** 한 문제의 시도 도중에는 가중치를 고정한다. 여러 독립 풀이 궤적을 모은 다음 GRPO 계열 정책 손실로 업데이트한다.
2. **추가 SFT는 전혀 사용하지 않는다.** 정답 풀이, 교사 모델 풀이, 자기 생성 정답 풀이를 `labels`로 넣어 cross-entropy로 따라 쓰게 하지 않는다. 과거 SFT 체크포인트가 아니라 지정 원본 F0에서 시작한다.
3. **오답노트는 감사 기록이지 자동으로 신뢰하는 원인 정답표가 아니다.** “이미지를 잘못 봤다”는 출력만으로 Vision Encoder의 책임이나 학습 대상을 결정하지 않는다.
4. **최초 구현은 정답 보상만으로 시작한다.** 다음에 자기수정, 마지막에 프로그램으로 검증 가능한 시각 사실 보상을 한 요소씩 추가한다. 복잡한 자연어 과정 보상부터 만들지 않는다.
5. **학습 모듈은 실험군 단위로 고정한다.** 초기 공동 후보는 Vision 고정 + 네 Merger + LLM LoRA다. 이것이 원인으로 확정되었기 때문이 아니라 연결과 추론을 함께 조정하는 실용적 가설이다. Vision 추가는 별도 비교군으로 검증한다.
6. **MMMU/MMMU-Pro는 검증·보고 전용이다.** 두 벤치마크와 모든 파생물을 학습, 재시도 학습, 오답노트 재학습, 보상모델 학습에 넣지 않는다. benchmark 정오 피드백으로 계속 풀어 얻은 답을 공식 점수로 제출하지 않는다.

권장 이름은 **검증 기반 자기수정 RL, Verified Self-Correction RL**이다. 기존 논문명이나 검증된 새 알고리즘 이름으로 주장하지 않는다. 연구 질문은 다음이다.

> 독립적인 외부 시각 문제에서 검증 가능한 정오·시각 사실 보상으로 자기수정을 학습하면, Qwen3-VL-4B가 보지 않은 전문 시각 문제를 **첫 시도부터** 더 정확하게 풀 수 있는가?

문헌의 방법 확인과 로컬 코드·원장 확인에 대한 확신은 높다. 우리 모델의 효과, 최적 학습 모듈, 모든 외부 데이터의 비중복, 학습 비용·완료시간은 아직 실측하지 않았다.

### 0.1 사용자의 다섯 단계와 구현의 대응

| 요청한 흐름 | 구현할 책임 | 자동으로 믿거나 수행하지 않을 것 |
|---|---|---|
| ① 문제를 풀고 답 확인 | 고정 Qwen 정책의 첫 응답 → gold를 격리한 typed verifier | 모델에게 정답표 제공 |
| ② 틀리면 이전 풀이 아래 재풀이 | 원본 이미지·실제 이전 응답·정오 피드백을 보존한 두 번째 턴 | 사람이 풀이를 수정하거나 정답 풀이를 주입 |
| ③ 또 틀리면 반복 | 기본 최대2시도, 3시도는 별도 후속 조건 | 맞을 때까지 무한 반복·성공한 기록만 저장 |
| ④ 오답노트 작성 | host의 검증 기록과 모델의 원인 가설을 분리 | 마지막 답이 맞다는 이유로 전체 풀이·원인 인증 |
| ⑤ 오류에 따라 학습 | 먼저 고정 FML/FL 대조, 이후 검증 기반 Vision 추가 routing | 자기평가만으로 모듈 책임 확정·즉시 온라인 업데이트 |

코딩할 때 먼저 볼 절은 **§5 데이터 승인 → §7 두 시도 상태 전이 → §8–9 보상·손실 → §10–11 노트·모듈 → §12–15 코드·환경 → §18 테스트**다. 문헌 근거는 §3·20에 모았다. `G=4`, 보상 `1/0.5/0`, 학습률·256개 pilot 등은 **이 보고서의 제안값**이며 기존 실행 실측 또는 다른 설계 파일의 확정값이 아니다.

**표기와 실행 범위:** 본문·도식의 ‘G4 episode’는 같은 문제에서 **독립 에피소드 네 개를 생성하는 그룹 크기 `G=4`**를 뜻한다. §18의 ‘G4 GPU smoke’라는 검증 단계 번호와는 별개다. 기존 turn-group 구현을 이 보고서만으로 자동 교체하거나, 아직 검증되지 않은 보상·라우팅 정책으로 유료 학습을 시작하지 않는다.

### 0.2 설계에서 구현으로 넘어갈 때 먼저 결정할 세 가지

1. **손실 계약은 하나만 선택한다.** 이 보고서의 독립 episode 그룹과 이미 존재하는 v2의 동일 상태 turn-group은 비교 가능한 대안이지, 숫자만 바꿔 합칠 설정이 아니다. 기존 v2를 선택한다면 그 프로토콜을 따르고, 이 보고서의 G4를 시험한다면 별도 protocol과 수치 검증을 추가한다. 이번 문서 보완은 어느 실행 계약도 변경하지 않는다.
2. **데이터 이름이 아니라 승인된 행을 학습한다.** TabMWP1600의 후보0 원장도 상태가 `CANDIDATE_ONLY_SCAN_COMPLETE_NOT_APPROVAL`이고 `training_allowed=false`다. 신규 source의 전체 비중복을 보장하지 않는다. §5의 배제 범위·검토·family 분리·actor/verifier 격리를 통과한 별도 RL allowlist가 실제 학습 입력이다.
3. **추가 SFT 금지는 실행 경로로 입증한다.** 정답 문장의 CE 학습·teacher 호출은 차단하되, 기존 모델 로더나 schema 타입 재사용을 SFT라고 오인하지 않는다. 실제 정책 손실·허용 입력·음성 테스트·checkpoint 변화로 확인한다(§13.4).

이 세 항목과 실제 GPU 적합성 검사가 미완료이면 상태는 **설계/CPU 준비 완료, 학습 시작 불가**다. 효과를 확인하기 전 데이터를 수천 개로 늘리거나 ‘오답노트가 원인을 해결했다’고 쓰지 않는다.

## 1. ‘지도학습 금지’를 구현 규칙으로 바꾸기

### 1.1 이번 문서의 해석

**“지정 Instruct 모델 이후의 추가 지도 미세조정 금지”**로 해석한다. 이미 배포된 Instruct 모델의 학습 이력까지 감독 신호가 없었다는 뜻이 아니다. 또한 RLVR도 정답·실행 검사로 만든 **보상이라는 감독 신호**를 사용한다. 따라서 이를 “정답 정보가 전혀 없는 비지도학습”이라고 설명하면 틀리다.

| 작업 | 이번 설계 |
|---|---|
| 사람이 쓴 풀이/정답 문장을 target으로 SFT | 금지 |
| GPT·상위 모델이 만든 풀이를 SFT 또는 distillation | 금지 |
| 자신이 맞힌 풀이만 골라 다시 CE 학습하는 rejection-sampling fine-tuning | 금지 |
| 기존 FL/FM/FV/FML 또는 A/B SFT 가중치를 RL 시작점으로 사용 | 금지; fresh F0 사용 |
| 시각 사실 JSON을 정답 target으로 auxiliary CE 학습 | 금지 |
| 정답/원래 구조 주석을 검증기 내부에서만 사용하여 0/1 보상 계산 | 설계상 허용 전제인 RLVR |
| 모델이 직접 생성한 토큰의 log-probability를 보상·advantage로 가중한 정책 업데이트 | 이번 학습 방법 |
| 모델 오답노트를 저장만 하고 분석 | 가능; 그 자체는 학습 아님 |
| 교사 없는 프로그램 렌더러·정확한 계산기로 문제와 검증 정보 생성 | 우선 후보; benchmark 독립성·렌더 정합성 검증 필수 |

이번 설계는 사용자가 설명한 정오 검증·강화학습 범위에 따라 **외부 원출처 정답·구조 주석을 verifier reward에만 사용**한다. teacher SFT·정답 문자열 모방·benchmark 학습은 금지한다. 사용자의 금지 조건을 수업 원문이 직접 금지한 것처럼 소급 설명하지 않는다. 확인한 [assignment_guidance.md](../local_sources/assignment_guidance.md)는 고정 모델·데이터와 동일 평가 조건의 근거이며, 본 방법을 교수님이 별도로 승인했다는 증거로 쓰지 않는다. 설계를 위해 같은 선택을 다시 묻지는 않되, 이후 수업의 명시적 추가 제약이 있다면 새 프로토콜에 반영한다.

보고용 문장:

> 지정 Instruct checkpoint를 초기 정책으로 사용하되 추가 지도 미세조정이나 teacher distillation은 수행하지 않는다. 모델이 직접 생성한 풀이에 대해 독립 검증기가 산출한 보상으로 정책을 업데이트한다. 외부 데이터의 정답과 구조 주석은 답안 모방 target이 아니라 보상 검증에만 사용한다.

### 1.2 고정된 출발점

| 항목 | 값·정책 |
|---|---|
| 모델 | `Qwen/Qwen3-VL-4B-Instruct` |
| 모델/processor revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| 정밀도 | BF16 forward; 학습 가능한 master/optimizer의 FP32 사용은 별도 기록. 자동 4-bit 전환 없음 |
| MMMU | `MMMU/MMMU`, `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68` |
| MMMU-Pro 로컬 manifest | `MMMU/MMMU_Pro`, `563f3e84bb3b90893083a1f039cfa13077f2302b` |
| 새 평가 출력 상한 | 사용자 선택 `8192`; F0와 FT 모두 같은 새 프로토콜 |
| 이미지 입력 기준 | `min_pixels=1003520`, `max_pixels=4014080`, `qwen_vl_utils_0_0_14`; 기존 경로·이미지 순서 유지 |
| 문맥 한도 제안 | 기존 `40960` 유지. 두 시도 전체 입력이 초과하면 명시적으로 차단하고 원문·해상도를 자동 축소하지 않음 |
| 평가 파서 | 기존 `team-final-answer-v8` 동결 |
| 기존 32K baseline | 558/900, 62.00%; 변경·덮어쓰기 금지 |

근거: [모델 manifest](../manifests/models/baseline.json), [MMMU manifest](../manifests/datasets/mmmu.json), [Pro manifest](../manifests/datasets/mmmu_pro.json), [V8 설정](../configs/eval/mmmu_val_v8.yaml), [공식 고정 모델 카드](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md).

주의: 두 데이터셋의 revision은 서로 다르다. 기존 모델/MMMU manifest의 `usage=evaluation_only`, Pro의 `usage=archive_only_no_inference`를 몰래 수정해 학습 승인을 만들지 않는다. 새 RL 모델 참조 문서에서 같은 원본 모델 해시와 새 사용 목적을 명시하며, benchmark 데이터는 여전히 학습 금지다.

## 2. 기존 실험이 실제로 말해 주는 것

| 관측 | 실제 결과 | 이번 설계에 주는 제약 |
|---|---|---|
| 기존 MMMU900/V8/고해상도32K | 558/900. 상한 종료90개 중 오답85개 | 장문 반복·답 완성을 별도 측정. 더 긴 풀이가 항상 좋은 것은 아님 |
| 고정 가중치 입력 진단 | I 40/102, O0 42/85, O1 31/69 | 서로 분모가 다르므로 단순 차를 oracle 인과효과로 계산하지 않음 |
| 기초 Knowledge 시험 | K1·K2 각각30/30 | 쉬운 다른 질문에서의 결과. 전문 시각 문제의 지식 충분성 인증 아님 |
| 10K 지도학습, 동일4K MMMU60 | F0 36, FL22, FM27, FV21, FML23 | 현재 데이터·목적함수·작은 표본에서 모두 하락. 모듈 분리 자체가 무의미하거나 Merger가 유일한 문제라는 증거는 아님 |
| 새8K 입력 진단48 | I22 / T24 / IT26 / IR20 / TR22 | 텍스트·이미지 조합의 회복은 제한적. 일반 규칙 추가도 일관된 개선 아님 |
| TabMWP A/B, 외부 QA48 | F0 35 / A30 / B28 | 구조를 더 잘 적는 것과 문제를 끝까지 맞히는 것은 다름 |
| 같은 실험의 표 셀310개 일치 | 268 / 253 / 299 | B의 셀 보존은 개선되었지만 QA 하락. 셀310개는 독립 문제310개가 아님 |

원장: [기능 진단 보고서](Qwen3_VL_4B_Bottleneck_Diagnostic_Report_20261003.md), [10K 분석](g6_full10k_final_analysis_20261005.md), [336 요약](../analysis_exports/input_recovery_20261011/full336_scores_v1/summary.json), [396 요약](../analysis_exports/input_recovery_20261011/F0_A_B_external396_scores_v1/summary.json), [최종 해석 v3](../analysis_exports/input_recovery_20261011/final_scientific_decision_v3.json).

따라서 이번 목표는 **“구조를 길게 출력하도록 학습”이 아니라 “관측한 정보를 이용해 실제 답을 완성하도록 보상”**하는 것이다. 외부 검증이 가능한 구조는 보조 측정·후속 보상에 사용한다. 출력 구조의 정확성만 높고 QA가 낮아지면 확대하지 않는다.

기존 10K와 A/B는 teacher 풀이를 안 썼더라도 **SFT**였다. 현 [train.py](../fine_tuning/train.py)의 `AssistantCollator`는 `target_text`를 assistant 정답열로 만들고, 학습 루프는 `model(...).loss`를 역전파한다. 이를 RL·비지도학습으로 다시 명명하지 않는다.

## 3. 참고 논문: 채택할 것과 제외할 것

논문의 성과는 해당 저자·모델·자료·평가 조건의 성과다. 아래 연구가 지정 Qwen3-VL-4B revision과 우리의 V8·8K 조건에서 효과를 보장하지 않는다. 원문을 길게 옮기지 않고 설계에 필요한 차이를 요약했다.

### 3.1 직접적인 방법 근거

| 연구 | 실제 방법과 조건 | 이번 프로젝트에서 쓰는 범위 |
|---|---|---|
| **SCoRe**, Kumar et al., 2024 | 같은 모델의 두 시도에 online REINFORCE. 첫 시도 분포를 제어하는 초기 단계와 수정 진전을 보상하는 단계. Gemini의 수학·코드 실험 | 재시도 학습, 정답→오답 악화 측정의 직접 근거. VLM/GRPO 재현으로 주장하지 않음 |
| **DeepSeekMath**, Shao et al., 2024 | 여러 응답의 상대적 보상으로 GRPO; 논문 전체에는 SFT 단계도 존재 | GRPO 알고리즘 출처. 그 전체 recipe가 no-SFT라는 근거로 쓰지 않음 |
| **DeepSeek-R1-Zero**, DeepSeek-AI, 2025 | base 모델에 추가 SFT 없이 보상 기반 RL. R1 본편의 cold-start·SFT와 구별 | 추가 SFT 없는 RL의 선행 사례. 시각 구조 학습 효과나 저비용 재현의 증거는 아님 |
| **Understanding R1-Zero-Like Training**, Liu et al., 2025 | GRPO의 길이·난도 관련 정규화 편향을 분석 | loss/advantage 정규화를 이름만 보고 선택하지 말고 명시·검사할 근거 |

원문: [SCoRe §5·6](https://arxiv.org/html/2409.12917v2), [DeepSeekMath §4](https://arxiv.org/html/2402.03300v3), [R1 보고서](https://arxiv.org/abs/2501.12948), [Dr. GRPO 분석](https://arxiv.org/html/2503.20783v2).

SCoRe의 추론 중 두 번째 지시는 정답이나 첫 답의 정오를 알려주지 않는 **intrinsic self-correction**이다. 사용자의 정오 피드백 방식은 **verifier-assisted correction**이므로 동일 방법이라고 부르면 안 된다. 본 설계의 첫 시도 평가에는 어느 피드백도 없다. SCoRe 공식 학습 코드 링크는 이번 조사에서 확인하지 못했으므로 비공식 구현을 공식 코드로 소개하지 않는다.

### 3.2 앞선 대화에 등장한 VLM 논문 검증

| 연구 | 확인한 사실 | 채택 여부 |
|---|---|---|
| **Vision-SR1** | v1은 Qwen2.5-VL-7B가 생성·필터한 9K로 cold-start SFT 후 GRPO. 이미지에서 쓴 설명만으로도 답을 낼 수 있는지 별도 보상 | 지각 설명과 답 완성을 나눠 측정하는 관점만 참고. v1 recipe 그대로 사용 금지 |
| **PROPA** | MCTS의 후속 성공률로 과정 가치를 추정하고 localized GRPO. GPT-4.1 teacher activation SFT 및 교차 SFT 포함 | 탐색을 통한 과정 가치 추정은 후속 아이디어. 원형 학습/teacher/PRM 학습은 이번 범위 제외 |
| **ViCrit** | 이미지 caption에 단일 오류를 넣고 오류 구절을 찾게 GRPO. 추가 SFT는 없으나 인간 caption과 GPT-4 오류 생성 사용 | 프로그램이 만든 확실한 단일 오류를 찾는 **새 변형**만 후보. 저자 생성 데이터 그대로 사용하지 않음 |
| **VisPlay** | 같은 backbone의 질문자·풀이자가 GRPO로 교대 학습. 다수결 답을 pseudo-label로 사용 | 난도·다양성 관리 아이디어는 참고. 다수결을 독립 정답으로 취급하는 핵심 보상은 초기판에서 제외 |

원문·저자 코드:

- Vision-SR1: [v1](https://arxiv.org/html/2508.19652v1), [v2](https://arxiv.org/html/2508.19652v2), [저자 저장소](https://github.com/zli12321/Vision-SR1).
- PROPA: [논문 §3·4](https://arxiv.org/html/2511.10279), [저자 저장소](https://github.com/YanbeiJiang/PROPA).
- ViCrit: [논문 §3·4](https://arxiv.org/html/2506.10128), [저자 저장소](https://github.com/si0wang/ViCrit).
- VisPlay: [논문 §2](https://arxiv.org/html/2511.15661), [저자 저장소](https://github.com/bruno686/VisPlay).

**버전 주의.** Vision-SR1 v2 본문 §3은 GRPO200step을 설명하며 v1의 cold-start를 명시하지 않지만, 부록 B.1·공개 recipe에는 SFT 관련 내용이 남는다. 따라서 모든 버전이 SFT 필수라거나 v2의 no-SFT 재현 경로가 검증됐다고 일반화하지 않는다. 조사 시 README에는 Qwen3-VL format reward 디버깅 및 LoRA 결과 미검증 주의도 있었다. 구현하려면 논문 버전과 코드 SHA를 함께 동결해야 한다.

**ViCrit의 한계.** 오류 구절을 맞힌다는 보상은 원본 caption의 정확성에 의존한다. 본 연구의 일부 하위 과제 악화도 보고되어 있으므로 전문 과제·retention을 함께 봐야 한다. 저자 저장소에서 완전한 학습 진입점은 확인하지 못했다.

**VisPlay의 한계.** 같은 잘못된 답을 여러 번 생성해도 다수결은 높다. 이것은 픽셀 사실 검증이 아니다. 이미지 해상도·평가기·모델도 우리와 다르므로 논문 성적을 예상 개선폭으로 쓰지 않는다.

**PROPA의 대조군도 구분한다.** §5.3 Table 2에는 저자가 `GRPO only`라고 부르고 SFT 없이 추론 트리에 GRPO를 적용했다고 설명한 ablation도 있다. 따라서 “PROPA 논문의 모든 실험이 SFT를 썼다”는 표현은 피한다. 다만 주 방법의 teacher activation·교차 SFT를 없애는 것만으로 그 대조군의 데이터·초기화·탐색·비용까지 재현되는 것은 아니다. 해당 대조군은 추가 SFT 없는 과정 탐색의 참고 후보이며, 이번 첫 구현에 MCTS나 학습 PRM을 넣을 근거로 삼지는 않는다. [PROPA §5.3](https://arxiv.org/html/2511.10279)

### 3.3 인용할 수 있는 주장 / 인용해서는 안 되는 주장

| 보고서·발표 문장 | 판단 |
|---|---|
| “SCoRe를 참고해 on-policy 두 시도 학습과 정답 유지·수정 지표를 설계했다.” | 적절 |
| “ViCrit을 참고해 프로그램 검증 가능한 시각 오류 탐지 보상을 제안했다.” | 적절; 원논문과 데이터 생성 차이 명시 |
| “Vision-SR1/PROPA가 지각과 과정 보상 분리의 필요성을 검토했다.” | 적절 |
| “이 논문들로 우리 모델의 Merger 결함이 증명됐다.” | 부적절 |
| “PROPA는 SFT를 쓰지 않으므로 그대로 사용한다.” | 사실과 다름 |
| “다수결로 검증된 설명은 정답 오라클이다.” | 부적절 |
| “자기수정 RL을 쓰면 MMMU67.4%에 도달한다.” | 미검증 |

본 조합의 새 요소는 **teacher 없는 프로그램 검증·벤치마크 격리·첫 시도 중심 평가를 결합한 팀 설계**다. 이를 새 알고리즘의 우월성이나 논문 수준의 신규성으로 미리 주장하지 않는다.

### 3.4 더 직접적인 후속 연구 네 편과 중요한 반례

**On Asymmetric Optimization of Reasoning and Perception in VLM Post-Training — Wu et al., 2026.** Qwen3-VL-2B/InternVL3.5-2B의 graph coloring·Sudoku에서 정답 보상과 지각 정확도의 연결이 약한 현상을 분석한다. 지각 보상 추가가 일부 조건에서 도움이 되지만, 한 조건은 악화됐다. **RL도 SFT checkpoint에서 시작**했고 실제 시각 과제 전이는 teacher reward를 썼으므로 no-SFT recipe로 복사할 수 없다. 우리에게는 “QA와 지각을 따로 측정하고 검증 가능한 지각 보상을 대조하라”는 근거다. 4×A10080GB 실험을 단일3090 비용으로 옮겨 계산하지 않는다. 프로젝트 페이지와 최신 v2의 surrogate 개선 수치도 다르므로 v2를 우선한다. [v2 §3.3·부록B](https://arxiv.org/html/2605.29496v2), [저자 코드](https://github.com/PlusLabNLP/asymmetric-VLM)

**MM-ReCoder — Tang et al., 2026.** 이미지 차트→코드→실행/렌더 feedback의 두 시도 GRPO다. backbone은 **Qwen2.5-VL-7B**이며 Qwen3-VL은 비교/teacher로 등장한다. 주 방법은160K SFT와235B teacher의7K 자기수정 SFT를 포함한다. 다만 **부록C.2에는 cold-start 없이, model-based reward 없이 RL하는 별도 ablation**도 있다. 이는 추가 SFT 없는 경로의 참고이나 MMMU QA·정오 피드백과 동일하지 않다. 수정 bonus가 첫 답의 품질을 고의로 낮추는 보상 악용, 첫 답 반복, shared-first-turn 대조가 특히 유용하다. Table A3 캡션의 RL 단계 순서는 본문과 달라 해당 ablation의 세부 순서를 확정해 복제하지 않는다. [§3·4.4·부록C.2](https://arxiv.org/html/2604.01600), [저자 코드](https://github.com/ZitianTang/MMReCoder)

**Chart-RL — Bai et al., 2026.** 직접 **Qwen3-VL-4B-Instruct**에 preliminary SFT 없이 LoRA RL을 적용한 chart QA 사례다. 그러나 이미지 폭300px, completion1024, LLM judge 기반 reward와 single-turn 설정이다. 원문은 judge를 `GPT-4 (gpt-oss-120B)`라고 혼동되게 표기하므로 어느 모델인지 확정해서 인용하지 않는다. 저자 학습 코드 링크도 이번 조사에서 확인하지 못했다. 단일24GB 주장도 이 조건에 한정된다. no-SFT VLM RL의 직접 참고로는 가능하지만 우리 고해상도·프로그램 검증·두 시도·Merger 공동 학습의 성공 근거는 아니다. [원논문 PDF](https://arxiv.org/pdf/2604.03157)

**StructReward — Li et al., 2026.** Qwen3-VL의 구조화된 과정 보상이라는 점은 가깝다. 하지만5K cold-start SFT, VisualPRM의 과정 주석, rollout을 다시 supervised 과제로 사용하는 경로, 강한 LLM의 오류 생성이 포함된다. “온라인 LLM judge 없음”만 보고 no-SFT/teacher-free라고 해석하면 안 된다. 참고 문헌으로는 쓸 수 있으나 **전체 학습 방식은 제외**한다. [v2 §Method](https://arxiv.org/html/2608.08326v2)

따라서 우선 인용 축은 **SCoRe/MM-ReCoder의 bounded correction**, **GRPO/Dr. GRPO의 정책 손실**, **Asymmetric Optimization/ViCrit의 지각 보상 검증**이다. 논문들의 SFT·teacher 단계는 각각 분리하고, 우리가 제안한 조합이 이미 검증됐다는 표현은 사용하지 않는다.

## 4. 데이터 선정: 정답 풀이가 아니라 검증 가능한 환경을 고르기

데이터 이름만 보고 수천 개를 합치지 않는다. 각 자료에서 `문제+이미지`, `검증 전용 정답/사실`, `출처/family`를 분리할 수 있어야 한다. 원래 풀이·해설은 이번 학습에서 사용하지 않는다.

### 4.1 네 실패군과 후보

| 실패군 | 우선 후보 | 모델이 받을 것 | 검증기만 받을 것 | 한계·현재 승인 상태 |
|---|---|---|---|---|
| A 전문 기호 | **새로 생성한 악보 원시기호·음정 문제**, 독립 회로 문제; 후속 MUSCIMA++ | 이미지·질문·보기 | 음표 위치·음높이·관계 또는 회로 topology·정확한 계산답 | 별도 v2 악보·회로 생성기는 CPU 구현됨. 본 설계의 학습 pool 승인은 미완료. MUSCIMA++ 새 overlap/권리 검토 필요 |
| B 도식·공간 | **CLEVR** train 또는 독립 렌더링; 기존 graph/spatial 생성기 | 이미지·관계 질문 | scene graph·functional program·답 | 물체 관계에서 전문 도면으로의 전이는 미검증. 원래 이미지/scene 단위 분리 필수 |
| C 과학 시각 판독 | **AI2D** 원래 train, **ScienceQA** train의 이미지 문항 | 과학 도식/사진·질문·보기 | 원래 QA label, 사용 가능한 관계 주석 | 비중복 미검증. 교과 도식을 의료 전문 판독 대용이라고 부르지 않음 |
| D 다단계 수량 | **TabMWP** 원래 train의 선별본; 후속 **FinQA** | 표 이미지·질문 | typed answer·원래 표, FinQA 실행 프로그램 | TabMWP 기존 선별본에 검사 근거 있음. FinQA는 원래 raster 이미지 자료가 아니어서 별도 렌더 프로토콜 필요 |
| 공통 유지·관계 | 조건부 **GQA** train | 자연 이미지·질문 | scene graph·원래 답 | 전문 기호 대용 아님. Visual Genome 원천/family·권리·중복 검토 필요 |

자료 근거: [CLEVR 저자 페이지](https://cs.stanford.edu/people/jcjohns/clevr/), [생성 코드](https://github.com/facebookresearch/clevr-dataset-gen), [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html), [MUSCIMA++](https://github.com/OMR-Research/muscima-pp), [AI2D 원논문](https://arxiv.org/abs/1603.07396), [ScienceQA](https://github.com/lupantech/ScienceQA), [TabMWP/PromptPG](https://github.com/lupantech/PromptPG), [FinQA](https://github.com/czyssrs/FinQA).

MMFineReason, **Geo170K 전체**, 출처를 풀어보지 않은 visual reasoning 혼합셋은 이번 기본 학습군에서 제외한다. Geo170K는 풀이만 제거해도 충분하지 않다. 저자들은 caption·QA 생성에 ChatGPT3.5를 사용했다고 명시하므로 질문·캡션의 teacher provenance도 남는다. [G-LLaVA §4·6.1](https://arxiv.org/html/2312.11370v2) RAVEN/ARC류도 현재 네 실패군에 모두 필요한 기본 재료로 넣지 않는다. 독립 생성·정확한 검증·전이 목표가 분명할 때 후속 조건으로 검토한다.

### 4.2 데이터별 구현 요점

**TabMWP**

- 이미지와 질문만 actor에 제공한다. 원래 `solution`, 정답을 설명하는 text table, gold supporting cells를 prompt에 넣지 않는다.
- 원래 표는 verifier에서 셀 단위 검사에 사용한다. QA 답은 정수/소수/문자열/단위를 명시적으로 typed 변환한다.
- 기존 A/B의 `target_text`는 삭제·재작성하지 않고 역사 자료로 보존한다. 새 RL adapter가 필요한 prompt 필드만 읽고 target을 actor에서 차단한다.
- 원본 dataset은 CC-BY-NC-SA-4.0이라고 저자 README에 명시되어 있다. 코드 MIT와 데이터 조건을 혼동하지 않는다. [원본 안내](https://github.com/lupantech/PromptPG#license)

**CLEVR / 독립 scene generator**

- scene graph와 functional program을 verifier에서 실행해 답을 계산한다. 모델에게 정답 scene JSON을 주지 않는다.
- 새 seed뿐 아니라 물체 조합·관계 깊이·렌더 스타일의 holdout을 둔다. 새 이미지 seed만 다르고 동일한 장면 구조면 독립 일반화가 아니다.
- 기존 코드가 구형 Blender 환경을 전제로 하므로 무작정 OS 의존성을 바꾸지 않는다. 최초판은 원래 train subset을 검토하거나 기존 단순 렌더러를 활용한다.
- 저자 페이지는 **배포 데이터 CC BY 4.0**, 생성기 저장소는 **코드 BSD**를 명시한다. 별도로 추가하는 asset의 권리는 따로 확인한다. [배포 데이터 안내](https://cs.stanford.edu/people/jcjohns/clevr/), [생성기 LICENSE](https://raw.githubusercontent.com/facebookresearch/clevr-dataset-gen/main/LICENSE)

**기호·회로/악보 합성**

- 원래 기호·관계를 먼저 샘플링하고 이미지를 만든다. 문제는 이미지를 가리고 풀 수 없도록, 동일 질문 템플릿에서 이미지가 바뀌면 답도 달라지는 쌍을 만든다.
- 회로는 node 연결·전압·저항의 정확한 유리수 solver 두 개를 대조한다. 회로 그림의 실제 선 끝점과 node graph의 정합성도 검사한다.
- 악보는 clef·staff position·accidental·음정 이름 변환을 각각 독립 시험한다. 음이름을 묻는 문제의 observation 슬롯에 정답 음이름을 미리 제공하지 않는다.
- synthetic renderer가 틀리면 solver 정답도 그림의 정답이 아니다. 메타데이터 정합성 검사와 이미지 표본 검토를 둘 다 요구한다.

**AI2D / ScienceQA / GQA / MUSCIMA++ / FinQA**

- 원천 주석을 검증에 쓸 수 있다는 것과 신뢰도가 100%라는 것은 다르다. 모호한 주석은 제외·검토 큐로 남긴다.
- ScienceQA의 `lecture`·`solution`뿐 아니라 captioning model이 생성한 `data/captions.json`, GPT 생성 결과도 제외한다. 이미지 없는 문제를 시각 학습으로 집계하지 않는다. 데이터는 CC BY-NC-SA 4.0, 코드는 MIT로 구분한다. [저자 README](https://github.com/lupantech/ScienceQA), [데이터 라이선스](https://github.com/lupantech/ScienceQA/blob/main/LICENSE-DATA)
- ScienceQA가 MMMU-Pro의 원천이라고 단정하지 않는다. Pro 원논문은 MMMU에서의 필터링·보기 확장·vision 변환을 설명한다. ScienceQA와 MMMU/Pro의 원자료 전수 대조는 이번에 하지 않았으므로, 확인된 중복도 확인된 비중복도 아닌 `OVERLAP_UNCHECKED`다. [MMMU-Pro §2](https://arxiv.org/html/2409.02813v2)
- MUSCIMA++는 필기 악보 기호·연결 주석이다. staff 일부는 자동 생성이며 printed-score 전이는 별도다. 주석은 CC BY-NC-SA 4.0이고 CVC 원본 이미지는 별도 취득·조건을 확인한다. 같은 악보를 여러 필기자가 썼으므로 **필기자만 나누지 말고 악곡 family도 분리**한다. 기호·연결 주석을 완전한 음높이·길이 정답이라고 간주하지 않는다. [저자 README·권리](https://github.com/OMR-Research/muscima-pp/blob/master/README.md)
- FinQA 원래 보고서/표를 이미지로 만들 때 gold 근거만 선택해 렌더링하지 않는다. 정답 위치를 표시한 표 형식이 입력에 새어들지 않게 한다. 원래 보고서 단위로 분리한다. 저자는 2022-05-04 업데이트에서 표 직렬화 형식의 label leakage 수정도 명시한다. 저장소 MIT를 모든 원 보고서 이미지의 권리 인증으로 확대하지 않는다. [FinQA 코드·데이터·수정 이력](https://github.com/czyssrs/FinQA#updates)
- [AWS AI2D 등록 정보](https://registry.opendata.aws/allenai-diagrams/)의 라이선스 링크는 CC BY-SA 4.0이다. 2026-10-17 재확인에서 기관 데이터 링크는 `ai2d-all.zip`으로 연결되었다. 실제 배포본·원천 이미지 조건을 확인한 뒤 승인하고, AI2D-RST 라이선스를 AI2D 전체에 옮겨 적용하지 않는다.
- GQA 질문은 scene graph에 기반한 프로그램 엔진 생성이다. 원래 사람 QA라고 일괄 표시하지 않는다. `answer`·`fullAnswer`·`semantic`은 verifier 전용이며, Visual Genome 원천 이미지 family·권리를 별도 확인한다. 저자 페이지의 feature 사전학습에 따른 validation 오염 경고도 검토한다. [GQA 데이터 스키마](https://cs.stanford.edu/people/dorarad/gqa/download.html), [평가 안내](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html)

**Geometry3K / PGPS9K: 코드에 있어도 자동 채택하지 않는 기하 후보**

- 현재 source registry에는 Geometry3K와 PGPS9K의 고정 revision 및 CPU adapter가 있다. 둘 다 `training_allowed=false`이고 권리·출처·benchmark 배제 승인은 미완료다. Geo170K 전체를 제외한다는 방침과 이 원천 후보의 검토는 구분한다.
- PGPS9K 공식 배포 안내에 따르면 **2,891문제·1,738도형이 Geometry3K에서 왔다.** 따라서 두 데이터셋을 서로 독립된 source로 보고 한쪽은 train, 다른 쪽은 test에 넣으면 안 된다. 원문 ID 매핑·도형·문항 family를 먼저 묶고, 공유 family 전체를 같은 split에 둬야 한다. [CASIA 공식 자료의 Collection·Data split](https://nlpr.ia.ac.cn/databases/CASIA-PGPS9K/index.html)
- PGPS9K는 학술 이용 agreement 절차를 확인해야 한다. Hugging Face에서 파일을 받을 수 있다는 사실이 그 절차나 원자료 권리 확인을 대체하지 않는다. 위 공식 페이지의 Use 절을 적용한다.
- 기존 adapter의 비-train 부모·일부 test 이미지 거부 검사는 필요한 방어지만, 전체 원출처 manifest의 진위나 MMMU/Pro 비중복을 입증하지 않는다. 이번 기본 pilot에 추가하지 않고 `SOURCE_IDENTIFIED / OVERLAP_UNCHECKED / RIGHTS_PENDING`로 남긴다.

TabMWP를 사용한다고 PromptPG의 학습 알고리즘을 재현하는 것은 아니다. 원논문의 policy gradient는 prompt 예시 선택기를 학습하는 방법이며, Qwen 가중치 RL 효과의 직접 근거가 아니다. 이번에는 **데이터의 표·정답만 검증용으로 차용**한다. [PromptPG 원논문](https://arxiv.org/abs/2209.14610)

### 4.3 처음부터 데이터셋마다 1,000개씩 학습할 것인가

**기본안은 아니다.** 데이터별 원시 행 1,000개는 서로 같은 정보량·난도·독립 이미지 수를 뜻하지 않는다. 예를 들어 하나의 표에서 나온 질문10개를 서로 독립 문제10개처럼 취급하지 않는다.

제안 순서:

1. **준비 검증용 64개**: 확보·승인된 source/family에서만 선택. CPU 입력·reward·누설 검증용이지 성능 보고 표본이 아니다.
2. **첫 RL pilot 256개**: A64/B64/D128. TabMWP·기존 외부 graph/circuit 기반의 가능한 범위부터 검증한다. C는 별도 자료 승인 전 `BLOCKED`이며 “네 분야 학습 완료”로 쓰지 않는다.
3. **확장 후보 각 실패군 1,000개, 총4,000개**: A 악보500/회로500, B CLEVR500/graph-spatial500, C AI2D500/ScienceQA500, D TabMWP800/FinQA200이라는 **사전 제안**이다. 미승인 source는 다른 source로 조용히 채우지 말고 새 manifest/계획을 만든다.
4. 각 자료를 1,000개씩 쓰려면 그보다 먼저 source별 positive reward 비율, false-positive verifier, 이미지 의존성, family 누수, 비용을 측정한다. 수천 행 확보가 효과의 증거가 아니다.

기존 dev/sealed의 역할을 보존한다. 과거 여러 차례 결과를 본 외부48/60문항은 개발 사례이며 새 독립 holdout으로 재사용하지 않는다.

## 5. MMMU/MMMU-Pro 비중복: 확인한 사실과 남은 검사

### 5.1 이번에 확인한 로컬 증거

기존 fingerprint index의 범위는 다음과 같다. 이것은 인터넷상의 모든 유사 문제를 포괄하는 증명이 아니다.

| 데이터 범위 | 원장 행 수 |
|---|---:|
| MMMU dev / validation / test | 150 / 900 / 10,500 |
| MMMU-Pro standard10 / standard4 / vision | 1,730 / 1,730 / 1,730 |
| 합계 | **16,740행, 100개 파일** |

Pro의 세 형식에는 같은 기저 문항이 반복될 수 있다. 16,740을 모두 독립 질문 수로 해석하지 않는다. 평가 대상은 수업 지정 Standard10-way지만 **배제 검사는 다른 형식까지** 포함한다.

고정 index: [portable_historical_exclusion_index.json](../analysis_exports/g7_cpu_refresh_20261005/portable_historical_exclusion_index.json), SHA256 `a36047da36b311a97321bf6fd07f4704ca0b3a9fa9afd9cf2fc8d74a2a4e7544`.

기존 탐지 구현은 다음을 확인한다.

- 정규화 질문 text exact hash.
- 7-word shingle의 포함 비율 0.8 이상인 near text.
- 이미지 raw bytes 및 크기를 포함한 RGB pixel hash.
- 64-bit dHash Hamming distance ≤2인 near-image **검토 후보**.
- benchmark 파생 lineage 및 family 교차 분리.

근거: [sources.py의 benchmark_index](../fine_tuning/sources.py), [audit.py의 contamination_candidates](../fine_tuning/audit.py). dHash 유사도만으로 두 그림이 동일하다고 확정하지 않는다.

TabMWP의 기존 native1600 선별에서는 near-image 후보 `1148`, `14424`, `29175`의 **표 family 전체를 제외**한 후 같은 seed로 재선정했다. 수정 선별본에서는 위 검사에 걸린 후보가 0개였다. 전체38,431행에서는 split을 가로지른 같은 표60family/813행도 제외한 기록이 있다.

근거: [선별본 scan v2](../analysis_exports/input_recovery_20261011/tabmwp_native1600_candidate_scan_v2.json), SHA256 `4e2adfd591d06521255d09086f00bc09b9f33b467c002d827dd8e0a4f8075f85`; [CPU 준비 확인](../analysis_exports/input_recovery_20261011/tabmwp_cpu_preparation_main_v1.json).

**한계:** 과거 fingerprint를 재사용한 경로 검증에서 원본 Parquet를 새로 전수 읽지는 않았다. 별도의 이미지 대조도 당시 접근 가능한32/100파일·2,630행이었다. source review는 사람의 전수 픽셀·의미 인증이 아니었다. [경로 재사용 근거](../analysis_exports/g7_cpu_refresh_20261005/summary.json)

**2026-10-17 이번 요청의 독립 CPU 재검사:** `fine_tuning.audit.contamination_candidates`로 A2000 중 TabMWP native1600개를 실제로 다시 검사했다. 고유 ID1600개 및 실제 PNG1600개/20,890,889bytes, exact/near text·exact/near image 후보 모두0이었다. scan2.821861초/전체3.929984초, exit0이며 code/index/원장/manifest/전체1600이미지의 실행 전후 hash도 같았다. 모델 로드·다운로드·Parquet 재독·원자료 쓰기는 없었다. 이 수치는 제한된 CPU scanner 재현이며 학습·추론 속도 측정이 아니다. 실행 결과는 도구 stdout과 이 보고서에 남겼으며 별도 새 receipt 파일은 만들지 않았다.

현재 함수는 `_inverted_shingles`가 없으면 near-text 대조를 수행하지 않으므로, 이번 재검사에서는 전체 `long_text_shingles`에서 **10,414개 `_shingle_sets`와 247,768개 key의 역색인을 명시적으로 구성한 뒤** 호출했다. 향후 검사는 캐시 누락을 0건으로 처리하지 말고 오류로 중단해야 한다. 또한 기존 `approve()`는 train 이외 행을 건너뛰므로 새 RL의 dev/sealed 배제 검사를 그 함수 호출만으로 완료했다고 쓰면 안 된다.

**검사 범위의 추가 한계:** legacy near-text 인덱스는 서로 다른 7-word shingle이10개 이상인 benchmark 질문만 보관한다. 따라서 짧은 질문과 이미지 안 OCR 문구까지 near-text 검사했다고 할 수 없다. `question_text()`는 `Options:`/`Choices:` 뒤를 제거하므로 이번 검사는 질문·보기 전체의 의미상 동치 검사도 아니다. 인덱스에는 이미지19,041회/고유 raw-image hash12,168개가 있으나, crop/semantic 완전성은 false다. 평가용 취득 manifest의 validation30파일·Pro Standard10의2파일과, 배제용 역사 인덱스의100파일 범위를 혼동하지 않는다.

이전 초안의 다른 소요시간은 이전 실행 기록이다. 위 시간은 이번 재검사에서 관측한 값으로 교체했으며, 재검증 가능한 보존 근거인 scan v1/v2·입력 원장은 변경하지 않았다. 결과 상태는 여전히 `CANDIDATE_ONLY_SCAN_COMPLETE_NOT_APPROVAL`이며 새 RL 학습 승인으로 승격하지 않는다.

재검사 결합 정보:

- 현재 감사 코드 SHA256: `5fb3765042def4f2dce5c2d31da2ee739f25bb83063df2aee379bee55b9b8c13`.
- 입력: `/home/jang2/mmdl-data/training/raw/tabmwp/5a1a52214521590b075f545b76a4f5ce666345e3/cpu_approved_ab_2000_v3/A/approved/records.jsonl`, SHA256 `1e916ae739681b248c23bdf5b778c28c27bb6df9daa9b18ab40a9093fbe6d809`.
- 같은 폴더 `manifest.json` SHA256: `14d3a20f6902d8a604e02f6342ada780a3d778a48675e2df40ab96385c9428e1`.
- 실제 이미지 root: 같은 source 아래 `trial_ab_2000_benchmark_filtered_v3/`.
- 이미지 `{상대경로: SHA256}` 원장을 키 정렬·공백 없는 JSON으로 직렬화한 SHA256: `056dabaad23dfb2bcd7cd84da4b260f84fad454cd344f3cbdf485169fec0d121`.
- 실행 결과는 당시 도구 stdout과 이 문서에 기록했다. 별도 새 receipt 파일을 생성했다고 주장하지 않는다.

이는 **고정 index에 대한 새 fingerprint 재검사**다. semantic/crop 중복, 모델 사전학습 오염 부재, 모든 픽셀 의미·정답의 인증은 아니다. 현재 A 원장은 SFT 역사 입력이므로 RL에는 아래의 별도 필드 분리·승인 절차를 적용해야 한다.

### 5.2 지금 말할 수 있는 판정

| 후보 | 현재 판정 |
|---|---|
| 기존 filtered TabMWP native1600 | 특정 원본·index·exact/near 검사 범위의 후보0 근거 있음. 새 RL 입력/검증기용 재승인은 필요 |
| 기존 graph/circuit 합성 자료 | 원천·solver·일부 입력/배제 검사 근거 있음. 기존 주석 전체를 actor prompt로 가져가면 안 됨 |
| 새 악보·새 scene 생성 데이터 | 별도 v2 생성기·CPU 시험은 존재. 본 설계의 새 대규모 선별본·권리/렌더/전수 배제 승인 완료는 아님 |
| 새 CLEVR/GQA/AI2D/ScienceQA/MUSCIMA++/FinQA 선별본 | 이번 조사에서 원자료 전수 취득·중복 검사하지 않음. `SOURCE_IDENTIFIED / OVERLAP_UNCHECKED` |
| Geometry3K/PGPS9K | adapter·고정 revision은 있지만 권리/비중복 승인 전. 두 source 사이 공유 문항 family까지 검사해야 함 |
| Qwen 사전학습 자체의 MMMU 노출 | 공개 정보·현재 로컬 검사로 인증 불가 |

따라서 **“이 목록의 데이터는 전부 MMMU/Pro와 겹치지 않는다”는 결론은 내리지 않는다.** 현재 근거가 있는 선별본과 앞으로 검사할 후보를 구분하는 것이 정확한 답이다.

현재 [source_registry_v1.yaml](../fine_tuning/notebook_rl/source_registry_v1.yaml)의 모든 source는 `training_allowed=false`다. 별도 v2의 [데이터 승인 코드](../fine_tuning/notebook_rl/contamination.py)는 필수 증거가 빠지면 `BLOCKED_NOT_CHECKED`와 빈 allowlist로 종료한다. 이는 올바른 차단 동작이지 OCR·embedding·픽셀 의미 검사가 이미 완료됐다는 뜻이 아니다. 외부 검사 결과의 hash·schema 검증과 실제 OCR/SSCD/SigLIP 실행·보정은 별도 증거로 남겨야 한다.

### 5.3 새 RL 데이터 승인 절차

1. 원본 source release/commit·license·원본 ID·전체 parents를 고정한다.
2. benchmark exclusion process가 지정 두 revision의 모든 금지 split/file/count/hash를 확인한다. actor/trainer는 benchmark 정답을 읽을 권한이 없다.
3. 정답지·설명·hint·파일명·URL에 섞인 gold/benchmark ID를 검사한다.
4. 기존 exact/near 검사에 **OCR 문구, crop/resize 유사 후보, 원문 출처/악곡/보고서/표 family** 검토를 추가한다. 문항별 후보/판정/검토자를 기록한다.
5. crop/semantic 검사가 안 된 행은 `not_checked`다. 해당 검사를 필수로 선언한 source는 이 상태로 자동 승인하지 않는다. generic embedding 유사도도 중복 여부의 최종 정답이 아니다.
6. source family 단위로 train/dev/sealed를 고정한다. 같은 음악 악곡의 다른 필기, 같은 보고서의 다른 표, 같은 topology의 사소한 변형도 family 정책에 포함한다.
7. **필터 결과와 새 변환 이미지·새 prompt 모두** 검사한다. 원자료를 검사했다는 이유로 새 변환본을 건너뛰지 않는다.
8. manifest에 `EXCLUSION_CHECKED_WITH_LIMITS`와 지원 범위·미검사 범위를 명시한다. 모든 미해결 후보는 quarantine. 학습기는 approved manifest의 allowlist만 읽는다.

프로그램이 완전히 새 문제를 만들더라도 benchmark 그림/숫자/설명에서 변형한 것이면 `BENCHMARK_DERIVED`다. 단순히 같은 과목·수학 법칙·기호를 쓴다는 이유만으로 모두 중복인 것은 아니다. **문항·원천 내용의 재사용과 일반 개념의 공유를 구분한다.**

## 6. 제안 아키텍처: 네 가지 격리 경계

전체 그림은 [HTML 도식](verified_self_correction_rl_pipeline_20261016.html)에서 확인한다.

### A. 데이터 배제 경계

benchmark 원문/이미지 → fingerprint 배제 전용 프로세스 → 승인 여부만 반환. 승인되지 않은 자료는 training dataset에 들어오지 않는다. benchmark 정답·설명을 생성기에게 제공하지 않는다.

### B. actor와 verifier 경계

actor가 읽는 것은 이미지, 질문, 보기, 사전에 정의한 출력 schema뿐이다. verifier의 private vault에는 정답, 타입, 단위, scene/table ground truth, executable program을 둔다. 별도 프로세스·파일 접근 권한 및 loader allowlist로 경계를 만들고, 타입·입출력 schema와 테스트로 누설을 검사한다. 같은 서버에 있다는 것과 같은 파일 접근 권한을 가진다는 것을 구분한다.

### C. rollout과 업데이트 경계

동일 policy version으로 한 그룹의 모든 episode를 생성한다. episode 도중에는 update하지 않는다. 닫힌 episode의 token IDs·시도별 상태·보상·mask가 검증된 뒤에만 업데이트한다. 업데이트 후 다음 그룹은 새 policy version을 사용한다.

### D. 학습과 최종 평가 경계

학습 verifier는 외부 train 데이터만 조회한다. MMMU/Pro evaluator는 학습 종료·checkpoint 동결 후에만 gold를 읽는다. 평가 때 gold-dependent retry·정답 오답노트 검색·문항별 정답 수선은 없다.

## 7. 실제 두 시도 학습은 어떻게 흘러가는가

### 7.1 episode 계약

기본 `max_attempts=2`. 한 initial state에서 **G=4개의 독립 episode**를 샘플링한다. 네 개를 반드시 VRAM에 동시에 올릴 필요는 없다. 정책을 고정한 채 순차 생성해도 그룹을 구성할 수 있다.

1. 원본 이미지+질문을 받고 풀이1·최종답1을 생성한다.
2. 원문과 token IDs를 저장한다. strict reward extractor가 답 하나를 추출한다.
3. verifier가 `CORRECT / INCORRECT / INVALID / UNVERIFIABLE`을 반환한다.
4. `CORRECT`이면 종료. `INCORRECT`이면 이전 풀이 전문과 원본 이미지가 있는 대화에 짧은 피드백을 추가한다.
5. `INVALID`는 답 형식/미완성 오류이며, 정답이 틀렸다고 거짓 통보하지 않는다. 두 번째 시도가 남았다면 형식·완성만 다시 요청한다.
6. 두 번째 풀이를 생성·채점하고 끝낸다. 또 틀려도 그 episode를 실패로 보존한다. 정답을 알 때까지 무제한 반복하지 않는다.
7. `UNVERIFIABLE`은 지원하는 검증 규칙으로 정오를 결정할 수 없는 상태다. annotation 검토/quarantine 또는 검증기 장애로 분류하고 해당 그룹의 업데이트를 중단한다. 오답 피드백·0점으로 대체하지 않는다.
8. 시스템 오류/OOM/검증기 장애도 오답 보상0으로 바꾸지 않는다. 해당 그룹을 원인 수정 전 업데이트하지 않고 상태를 구분한다.

### 7.2 프롬프트 제안

아래는 **팀 설계 초안**이며 Qwen의 공식 MMMU 프롬프트가 아니다. 실제 구현 전 자체 fixture로 길이·형식·누설을 고정하고 해시를 남긴다. 이 prompt를 사용한 외부 훈련과 P0의 MMMU 평가를 구분한다.

첫 시도:

```text
Solve the question using the provided image(s).
Check the visual information needed for the question and give your reasoning.
Do not assume that a fact is visible when it is not.
Finish with exactly one line: Final answer: <one answer>

Question: {question}
{choices_if_any}
```

오답 뒤 두 번째 시도:

```text
The verifier marked your submitted answer incorrect. The correct answer is not provided.
Re-check the original image(s) and your previous solution.
Write a new solution below; do not treat your previous claims as established facts.
Finish with exactly one line: Final answer: <one answer>
```

형식 실패 뒤 피드백:

```text
Your response did not contain one complete, valid final answer.
Its correctness was not established. Review it and submit one final answer.
```

`INCORRECT` 한 bit도 학습 중 외부 정보다. 이를 사용한 결과를 “아무 피드백 없는 자기수정”이라고 부르지 않는다. 보기에 남은 후보를 하나씩 지워가며 맞힐 수 있으므로 최초 QA는 가능한 수치·상태의 open answer 비중을 높이고, MCQ는 chance baseline·첫 시도 정확도와 함께 평가한다.

### 7.3 이전 풀이를 보존하는 방식

- 이전 assistant 응답은 원문 그대로 다음 대화에 들어간다. 요약·사람 수정·gold를 이용한 문장 수정 없음.
- 이미지는 각 시도에서 계속 문맥에 남긴다. 두 번째 턴에 동일 이미지를 중복 삽입해 token 수를 늘리지 말고 첫 user의 이미지 참조를 보존한다.
- chat template이 앞선 토큰을 다시 쓰거나 삭제하는지 prefix 검사한다. 앞선 reasoning이 사라지면 그 상태로 multi-turn RL을 진행하지 않는다.
- 전체 문맥 한도를 넘으면 `CONTEXT_BLOCKED`. 원문을 몰래 잘라 넣지 않는다.
- 첫째·둘째 답 사이 충돌은 정상적인 수정일 수 있다. verifier는 해당 턴의 **새 응답**을 채점하고, 과거 답은 이력으로만 둔다. 한 턴 안의 미해결 충돌은 별도 `INVALID/AMBIGUOUS`다.

## 8. 보상: 교수님께 설명할 수 있을 만큼 단순하게

### 8.1 R1: 단일 시도 정답 보상 기준선

`R_answer = 1`은 유효한 단일 답이 typed gold와 일치할 때, 그 외 모델 오답/형식 실패는0이다. 정답 문장 길이·그럴듯한 자기반성·JSON 장식에는 보상을 주지 않는다.

숫자는 `Decimal` 또는 `Fraction`으로 비교한다. 요구 단위·방향·반올림 자릿수는 source schema에서 먼저 결정한다. `$2,960 unfavorable`처럼 qualifier가 의미에 관여하는 경우를 무조건 지우지 않는다. exact 비교가 불가능한 질문은 사전 제외하거나 명시적 검증 규칙을 별도 버전으로 만든다.

기존 V8의 폭넓은 자연어 처리·두 자리 반올림을 RL reward에 그대로 적용하지 않는다. **고정 benchmark 파서와 새로운 학습용 strict verifier는 역할이 다르다.** 단, 학습 후 MMMU 점수는 기존 V8로 그대로 낸다.

### 8.2 R2: 두 시도 episode 보상

첫 설계는 다음처럼 명확하게 둔다. 논문에서 그대로 복사한 공식이 아니라 팀 제안값이다.

| episode 결과 | `R_task` |
|---|---:|
| 첫 시도 정답 → 즉시 종료 | 1.0 |
| 첫 시도 오답/형식 실패 → 두 번째 정답 | 0.5 |
| 두 시도 모두 오답/무효 | 0.0 |

이 순서는 **같은 문제에서 최종적으로 정답에 도달한 두 궤적의 보상 순위**를 ‘처음 정답 > 수정 후 정답’으로 정한다. 일부러 틀린 첫 답이 학습 과정에서 강화되지 않는다는 보장은 아니다. 단순한 “수정 성공 bonus”를 크게 주지는 않으며, 첫 버전의 비용 항은0이고 고정 횟수·출력 cap으로 비용을 제한한다.

**주의:** 이 보상 순서만으로 credit assignment가 완벽해지는 것은 아니다. 그룹 안에서 모두 첫 시도를 틀렸고 일부만 회복하면, episode-level advantage는 잘못된 첫 시도 토큰에도 양의 신호를 줄 수 있다. 따라서 R2에는 아래의 **단일 시도 RL 유지 그룹**을 함께 둔다.

- optimizer step의 prompt group 절반: R1 형식의 single-turn, 정답 보상만 적용.
- 나머지 절반: two-turn episode, 위 `R_task` 적용.
- 서로 다른 task mode의 보상은 한 GRPO 그룹에서 정규화하지 않는다.
- 이 혼합비0.5/0.5도 사전 설계값이며 SFT loss를 섞는 것이 아니다. 첫 시도 holdout이 악화되면 stop/review한다.

anchor 혼합은 credit assignment 문제의 **완화 가설이지 해결 보장**이 아니다. 첫 시도·재시도별 정책 항과 gradient 기여를 구분해 기록한다. 오답인 첫 시도에도 양의 episode advantage가 반복 부여되고 첫 시도 성적이 떨어지면, 정답을 맞힌 마지막 풀이만 SFT하는 대신 아래의 별도 shared-state 대조로 설계를 다시 검증한다.

SCoRe 원형의 reward shaping·REINFORCE와 이 GRPO 제안을 동일하다고 쓰지 않는다. 최초 비교에서 R2가 R1보다 못하면 multi-turn을 빼고 원인을 확인한다.

보상 악용 우려는 추측만은 아니다. MM-ReCoder는 correction bonus로 첫 시도 품질이 낮아지는 사례를 보고했다. 후속 대안은 첫 시도를 공유된 고정 문맥으로 두고 여러 두 번째 시도만 RL하는 비교지만, 이것은 별도 protocol이며 현재 기본 episode loss에 몰래 섞지 않는다. [MM-ReCoder §4.4](https://arxiv.org/html/2604.01600)

### 8.3 R3: 검증 가능한 시각 사실 보상 추가

먼저 R1/R2를 통과한 뒤, 구조 정답이 있는 별도 source에서만 추가한다.

`R_total = R_task + 0.1 × R_visual`

`R_visual`은 문제마다 사전에 지정한 raw observation 슬롯의 올바른 비율이다. 예: 지정 표의 행·열 값, 기호 위치, node 연결. **모델이 마음대로 고른 쉬운 사실만 세지 않는다.** 누락·중복·모순은 정답 슬롯으로 세지 않고, 적용 대상 슬롯·분모는 생성 전에 고정한다. 보상은 **episode가 종료된 시도의 슬롯만 한 번** 계산해 `R_visual∈[0,1]`로 제한한다. 두 시도의 값을 합산하거나 첫 시도의 맞은 슬롯만 골라 마지막 결과를 구제하지 않는다. 이전 관측의 변화는 보상이 아닌 별도 감사 지표다.

정답 자체를 observation 정답으로 반복 평가하지 않는다. “음 이름은?”에서 그 음 이름을 observation에 요구하는 식의 target 재포장은 제외한다. 슬롯은 가능한 한 원시 위치·표시·관계이고, 질문의 최종 계산/분류답은 별도로 둔다.

0.1 상한을 둬서, 완벽한 사실만 쓰고 문제를 틀린 응답이 정답 응답보다 높은 보상을 받지 않게 한다. 관측 슬롯이 없는 source는 `visual_reward_applicable=false`; 해당 source의 별도 정답 보상 그룹에서 학습하며 N/A를1점으로 넣지 않는다. R3 비교는 적용 가능한 동일 source·schema·prompt 위에서 λ=0과0.1만 바꿔야 한다.

이 보상은 **확인한 슬롯의 정확성**이지 전체 풀이가 참이라는 인증이 아니다. 모든 자연어 문장의 논리적 진실을 평가하는 LLM judge/학습 PRM은 초기판에 넣지 않는다.

지각 보상과 최종답 보상을 구분할 연구 근거는 있지만, 높은 지각 가중치가 항상 유리한 것은 아니다. Qwen3-VL-2B의 통제 연구도 trade-off를 보고하므로 우리 λ=0.1은 검증해야 할 시작값이지 논문으로 정당화된 최적값이 아니다. [Wu et al. §3.3](https://arxiv.org/html/2605.29496v2)

### 8.4 프로그램 검증형 ViCrit 변형은 별도 과제

보조 후보는 renderer의 참인 원시 사실 목록에서 딱 한 값/관계를 프로그램으로 바꾸고 모델에게 오류 슬롯을 찾게 하는 것이다. `wrong_slot_id` exact match로 보상하고, 일부는 오류가 없는 `NONE` 사례를 넣어 무조건 오류를 지적하는 전략을 막는다.

문장 길이·서식·label 순서가 정답을 노출하지 않게 하고, image-removed 대조에서 너무 잘 풀리면 text shortcut을 점검한다. 이 proxy task는 QA와 분리하여 지표를 낸다. 오류 탐지를 잘한다고 전문 문제를 잘 푼다는 결론은 아직 없다.

### 8.5 보상 악용 방지의 필수 반례

| 반례 | 기대 처리 |
|---|---|
| 모든 선택지/숫자를 나열해 gold를 포함 | 유효 단일 답 아님,0 |
| 중간 계산에 gold가 있지만 final은 다른 값 | final만 비교 |
| 정답 뒤 철회하고 새 답 미완성 | 완료된 단일 제출로 인정하지 않음 |
| 단위 V 질문에 같은 숫자의 A 답 | 단위 정책에 따라 오답 |
| 쉬운 사실1개만 쓰고 나머지 observation 생략 | 고정 전체 슬롯 분모에서 누락 처리 |
| 오류를 먼저 일부러 만든 뒤 수정 | 첫 시도 정답보다 낮은 episode 보상 |
| `I made a visual error` 같은 문구만 생성 | 보상 없음 |
| verifier에게 gold/파일/명령 실행을 요청 | actor 출력은 데이터로만 처리; 실행·파일 조회 금지 |
| 네 episode 전부 보상0 또는 전부 동일 | task advantage0. 성공풀이를 teacher로 만들어 넣지 않음 |

## 9. 실제 가중치 업데이트: multi-turn GRPO를 어떻게 구현할 것인가

### 9.1 그룹의 단위는 ‘시도’가 아니라 ‘독립 episode’

한 이미지·질문·task mode·정책 버전에서 독립 episode **G=4개**를 생성한다. 각 episode 안에 최대 두 번의 시도가 있다. 첫 번째 시도와 그 피드백을 본 두 번째 시도를 서로 같은 GRPO 그룹의 독립 샘플로 취급하면 안 된다. 두 시도는 입력 상태가 다르기 때문이다.

첫 구현의 팀 제안은 다음과 같다. 원래 GRPO/SCoRe를 그대로 재현하는 것이 아니라 명시적인 변형이다.

```text
같은 prompt의 G개 episode 보상: R_1, ..., R_G
advantage_i = R_i - mean(R_1, ..., R_G)

ratio_it = exp(logp_current(action_it | state_it)
               - logp_rollout_policy(action_it | state_it))

policy term = min(ratio_it * advantage_i,
                  clip(ratio_it, 0.8, 1.2) * advantage_i)

log_ref_ratio_it = logp_frozen_F0(action_it | state_it)
                   - logp_current(action_it | state_it)
k3_it = exp(log_ref_ratio_it) - log_ref_ratio_it - 1

loss = sum(actor_mask_it * (-policy_term + beta * ratio_it * k3_it))
       / (G * fixed_episode_token_cap)
```

- 그룹 reward 표준편차로 나누지 않는 `scale_rewards=False`를 최초 후보로 둔다. 보상 차이가 작은/어려운 문제의 정규화 편향을 피하려는 설계이며 실제 효과는 비교한다.
- 길이 정규화는 완료된 응답의 실제 길이 대신 **미리 고정한 cap**을 사용한다. mode별 cap과 전체 loss 가중치를 manifest에 저장한다. 길이 편향이 완전히 사라진다고 보장하지 않는다.
- R1 single-turn과 R2 episode 그룹은 별도 계산하고, 그룹 평균 loss를 사전 고정 비율로 섞는다. gradient accumulation/DDP에서도 그 평균이 같도록 검사한다.
- KL 방향은 `KL(current || frozen_F0)`다. 위 `k3`는 선택 action의 logprob로 계산하는 비음수 추정량이며, old-policy로 뽑은 action에는 `ratio`를 곱한다. 이 KL 항의 ratio는 clip/detach하지 않고, 정책 항과 같은 action mask·고정 분모를 쓴다는 계약이다. 상태는 old rollout에서 온 것이므로 전체 multi-turn 정책의 정확한 trajectory KL로 부르지 않는다. 작은 vocabulary의 exact KL·gradient와 대조하고, 선택 TRL 버전의 실제 loss가 다르면 별도 명시적 loss 연결이 필요하다.
- 최초 clip ε=0.2, KL β=0.02는 **미검증 팀 시작값**이다. 논문에서 우리 모델의 최적값으로 검증한 수치가 아니다.
- `old_logprobs`, frozen-reference logprobs, reward와 advantage는 상수로 detach한다. current logprob와 KL의 current/old ratio는 gradient를 유지한다. 생성 시 분포와 old forward의 분포가 같다는 전제도 검사한다. 작은 vocabulary에서 expectation과 gradient를 직접 대조해 이 계약을 검증한다.
- 동일 보상 그룹은 task advantage0이다. β가0이 아니면 KL 항은 남으므로 “그룹 전체 업데이트가 항상0”이라고 하지 않는다.
- 전체가0인 문제를 결과를 보고 몰래 버리거나 정답 풀이를 만들어 SFT로 구제하지 않는다. task gradient가 거의 없으면 rollout 난도·검증기·초기 정책을 조사하고 중단한다.

**한 번의 업데이트와 clipping의 한계:** 기본안처럼 동일 정책에서 batch 전체를 수집하고 모든 gradient를 누적한 뒤 optimizer를 한 번만 갱신하면, 손실 계산 당시 `current=rollout`이므로 수치 오차를 제외한 `ratio=1`이다. 따라서 이 단계의 PPO clipping은 실제 업데이트 크기를 제한하지 않는다. 고정 F0와의 KL, 학습률, gradient clipping 및 업데이트 후 정책 변화 관측을 구분해 관리한다. clipping을 활성화하려고 같은 rollout에 여러 optimizer step을 적용하는 변경은 별도 protocol이며 기본안에 암묵적으로 추가하지 않는다.

근거: [GRPO 원문](https://arxiv.org/html/2402.03300v3), [정규화 편향 분석](https://arxiv.org/html/2503.20783v2), [TRL GRPO loss 설정](https://huggingface.co/docs/trl/main/en/grpo_trainer). 이 식은 구현 계약의 개요이며 완성된 trainer 코드가 아니다.

### 9.2 어떤 토큰에 loss를 주는가

**actor가 새로 생성한 토큰에만** 정책 손실을 준다. 원래 질문·이미지 token, 이전 대화의 입력 사본, verifier 피드백, padding은 mask0이다. 첫 시도는 episode의 action으로 한 번 계산하고, 두 번째 입력에 재등장했다는 이유로 두 번 학습하지 않는다.

정답은 reward 서비스에서만 읽는다. policy 입력 텐서나 학습용 token label에 gold를 붙이지 않는다. 정책 손실에서 자신이 생성한 action ID를 사용하는 것과, 외부 정답 문자열을 teacher target으로 사용하는 것은 다르다.

rollout의 정책 버전을 고정하고 old log-probability를 저장한다. 한 batch를 모으는 중 optimizer를 갱신하지 않는다. 초기에는 한 rollout batch당 한 policy update만 한다. 저장된 과거 정답 풀이를 무제한 replay하는 off-policy 학습은 넣지 않는다.

**실제 샘플링 분포와 종료 토큰 계약:** T=1/top-p=1만으로 생성 분포와 training forward의 softmax가 같다고 가정하지 않는다. `generation_config`에 상속된 `min_new_tokens`, forced/suppressed token, EOS 억제, bad-word/반복 방지 processor, stop 처리까지 기록한다. 기본안은 분포를 변경하는 미검증 processor를 거부하고 sampler/old-forward logprob를 같은 action ID에서 대조한다. 제약 분포를 쓰려면 current/old/reference에 일관되게 적용하고 action support·KL 의미를 별도 검증해야 한다. 실제 생성된 EOS는 action으로 한 번 포함하며 padding이나 host가 덧붙인 구분자는 제외한다. EOS와 pad ID가 같아도 ID값만으로 mask를 만들지 않고 실제 생성 길이를 사용한다. stop 문자열을 화면에서 제거했어도 학습 원장의 실제 생성 ID를 지우거나 텍스트 재토큰화로 대체하지 않는다. 지원하지 않는 불일치는 자동 무시하지 않고 업데이트를 차단한다.

### 9.3 생성·채점 오류를 reward0으로 뭉개지 않기

| 상태 | 학습 처리 |
|---|---|
| 모델이 틀린 답 또는 지원 형식 밖 답을 완전히 생성 | 정상 task 결과,0점 가능 |
| 출력 상한에서 답 없이 종료 | 모델의 미완료 task 결과,0점. 숨기지 않고 length 지표 기록 |
| 정답을 완성한 뒤 설명만 잘림 | 사전 final-answer·철회 정책으로 검증; 무조건0/1 아님 |
| 이미지 decode·CUDA·network·verifier 예외 | `SYSTEM_ERROR`; 해당 그룹으로 업데이트하지 않음 |
| 필요한 이력이 문맥에 안 들어감 | `CONTEXT_BLOCKED`; 몰래 truncation하지 않음 |
| 프로그램 정답 자체가 모호/틀림 | quarantine; gold를 결과에 맞춰 바꾸지 않음 |

시스템 실패 복구는 동일 입력·같은 정책에서 제한 재시도하고 로그를 남긴다. 성공 응답만 골라 그룹 크기를 맞추지 않는다. 생성 잘림을 자동 제외하는 라이브러리 기본값도 명시적으로 검사한다.

## 10. 오답노트와 오류 원인: 무엇을 기록하고 무엇을 학습하지 않을 것인가

오답노트는 각 episode의 **검증된 관측 + 미확정 가설**을 연결하는 감사 자료다. 정답을 맞힌 마지막 설명이 참이라는 전제로 역사를 재작성하지 않는다.

```json
{
  "episode_id": "external-family123-policy0007-rollout2",
  "source_kind": "EXTERNAL_INDEPENDENT",
  "attempts": [
    {"turn": 1, "raw_response_ref": "...", "answer_status": "INCORRECT"},
    {"turn": 2, "raw_response_ref": "...", "answer_status": "CORRECT"}
  ],
  "verified_observations": [
    {"field": "row_3_column_2", "observed": "35", "expected": "53",
     "verifier_id": "table-cell-exact-v1", "evidence_ref": "..."}
  ],
  "model_error_hypothesis": "The digits may have been read in reverse.",
  "functional_error": "OBSERVATION_MISMATCH",
  "module_cause": "UNKNOWN",
  "sft_target_allowed": false,
  "notebook_used_as_reward": false
}
```

위 값은 **설명용 가상 예시**이며 실제 문항의 측정값이 아니다. `expected`는 verifier/감사 저장소에만 두며 다음 actor 입력으로 전달하지 않는다.

검증 가능한 기능 분류:

- `OBSERVATION_MISMATCH`: 지정 원시 값·위치·기호가 주석과 다름.
- `RELATION_MISMATCH`: 연결·방향·행/열 대응이 다름.
- `PROGRAM_STEP_INVALID`: 지원하는 typed 계산/규칙 실행이 맞지 않음.
- `FINALIZATION_FAILURE`: 답 없음, 철회·충돌, length.
- `ANSWER_MISMATCH_CAUSE_UNKNOWN`: final은 틀렸지만 원인 인증 불가.
- `ANNOTATION_REVIEW`: 정답/구조 주석 자체가 의심됨.

자유로운 `Relevant knowledge` 문장이나 “공식을 잘못 골랐다”는 자기평가에는 독립 판정이 없으면 `UNVERIFIED`를 붙인다. 모델의 오답노트 생성은 optional 추가 호출로 비용을 따로 기록하며 학습 reward에서 제외한다. 가능한 기록은 host가 원문과 verifier 결과에서 직접 만든다.

**기능 오류는 내부 모듈의 결함 인증이 아니다.** 숫자 오독도 Vision 표현, Merger 전달, LLM의 선택적 사용 중 어디서든 발생할 수 있다. 반대로 몇 개 observation이 맞았다고 “Vision은 항상 맞았다”고 결론내릴 수 없다.

### 10.1 ‘오답노트를 쓴다’와 ‘오답노트로 학습한다’의 세 층

| 층 | 실제 동작 | 가중치 업데이트 |
|---|---|---|
| L1 감사 노트 | host가 실제 응답·검증된 슬롯 오류·답 전이를 연결 | 없음. 기록만으로 학습했다고 하지 않음 |
| L2 수정 경험 학습 | 모델이 자기 이전 응답을 문맥으로 보고 새 풀이를 생성; §9의 보상 기반 정책 손실 적용 | 있음. 기본 자기수정 RL |
| L3 진단 능력 학습 | 모델이 처음 틀린 관측 슬롯/지원하는 오류 범주를 직접 선택; 독립 검증 가능한 경우에만 별도 RL reward | 후속 실험. 자유로운 노트 전체를 정답 문장으로 모방하지 않음 |

L3까지 구현한다면 `diagnosis_context = 이미지 + 문제 + 원문 시도들`이고 정답 값이나 host의 정답 진단은 숨긴다. verifier의 목표는 `관측 슬롯 오류/관계 오류/지원 계산 오류/판정불가`처럼 **검증 범위 안의 기능 분류**다. 오류가 확인되지 않은 사례를 모두 ‘추론 오류’로 만들지 않는다. 답을 직접 노출하는 observation이나 모호한 주석은 대상에서 제외한다. 같은 문맥에서 생성한 진단 후보끼리 별도의 그룹을 만들며 풀이 그룹과 reward 평균을 섞지 않는다.

예를 들어 후속 시작값 `R_diag = 1[범주 일치] + 0.25[검증된 오류 슬롯 집합 일치]`를 시험할 수 있다. 이는 **미검증 팀 설계**다. 슬롯 집합은 사전에 정해진 전체 검증 범위에서 비교하고 전부 나열하기·누락·중복을 검사한다. 원인 검증 불가 사례에서는 범주를 강제하지 않고 `NOT_APPLICABLE`로 남긴다. `Lesson` 같은 자유문은 보상0/마스크 제외이며 ‘그럴듯한 반성’에 점수를 주지 않는다. 이 방식도 검증기 label이라는 감독 신호는 있지만 정답 노트 CE를 쓰는 SFT는 아니다.

L3의 성공 판정은 진단 label 정확도뿐 아니라 **독립 문제의 첫 시도 QA와 유지 성능**이다. 노트 작성만 좋아지고 QA가 떨어지면 채택하지 않는다. 학습 후 MMMU 추론에서 훈련 노트를 검색·주입하거나 정답 피드백을 제공하지 않는다.

## 11. Vision·Merger·LLM은 어떻게 학습시킬 것인가

### 11.1 초기부터 문항마다 학습 모듈을 바꾸지 않기

권장 순서는 다음과 같다. 모두 지정 fresh F0에서 독립 시작한다.

| 단계/실험군 | 학습 대상 | 고정 대상 | 목적 |
|---|---|---|---|
| CPU/GPU 정합성 smoke | 먼저 LLM LoRA의 작은 검증 step | 나머지 | policy gradient·재로딩·mask 확인; 개선 성적 아님 |
| R1 주 후보 FML | LLM LoRA + 네 Merger의 FP32 master | Vision encoder 본체 | 연결과 답 완성을 함께 조정하는 가설 |
| R1 대조 FL | 같은 LLM LoRA | Vision 및 네 Merger | Merger 추가의 이득·부작용 분리 |
| R2 | 외부 dev에서 정한 동일 모듈 범위 | 동일 | 자기수정만 추가 |
| R3 | R2와 같은 모듈 범위 | 동일 | 시각 보상만 추가 |
| 후속 FVML | Vision LoRA + 네 Merger + LLM LoRA | 그 밖의 base 파라미터 | 같은 데이터/보상에서 Vision 추가 비교 |

소규모 비교 결과가 불분명하면 모듈 승자를 선언하지 않는다. FML을 시작 후보로 둔 이유는 과거 결과가 FML의 우월성을 증명해서가 아니다. **모든 모듈 full fine-tuning**은 메모리·망각·비용이 커서 기본안에 넣지 않는다.

처음부터 데이터의 추정 오류별로 optimizer를 교체하면 학습량·분포·신뢰도·모듈이 한꺼번에 변한다. adaptive routing은 후속 연구이며, 먼저 고정 모듈 실험군에서 효과를 확인한다. 필요할 때도 self-report가 아니라 독립 검증 특징으로 사전 고정한 정책을 사용하고, 고정 범위 대조군을 유지한다.

### 11.2 실제 Qwen3-VL 구현에서 확인할 것

현 [modules.py](../fine_tuning/modules.py)는 실제 모듈 타입·경로를 조사하고 FL/FM/FV/FML을 설정한다. 마지막 Merger와 세 DeepStack Merger, 총4개 연결부를 별도로 추적한다. 기존 FM 전체는24 tensor/109,105,152 parameter이며 Vision 본체와 중복 계산하지 않는다.

- Vision attention의 `qkv`/`proj`와 언어 Decoder의 `q_proj/k_proj/v_proj/o_proj`를 구분한다. 실제 `named_modules`, shape, requires_grad 목록에서 선택한다.
- 기존 LoRA 후보 r16/α32는 재사용할 수 있지만 RL dropout은0으로 별도 고정한다. 과거 SFT 설정을 변경해 덮어쓰지 않는다.
- 모듈별 `grad_norm`, 실제 tensor 변화, finite 여부를 확인한다. loss 감소만으로 Vision이 학습됐다고 보고하지 않는다.
- Vision container 전체를 `no_grad`로 감싸면서 내부 Merger까지 동결하는 실수를 막는다. DeepStack의 모든 활성 경로에도 gradient가 연결되어야 한다.
- Vision을 학습할 때 과거 vision feature cache를 재사용하면 안 된다. 동일 image hash라도 policy update 후 feature가 달라진다.

### 11.3 KL reference는 정말 고정 F0여야 한다

**LoRA를 disable하는 것만으로 F0 reference가 복구되지 않는다.** 네 Merger 본체를 직접 학습했다면 LoRA를 꺼도 Merger는 이미 바뀌었다. 기본안은 별도의 frozen F0 reference를 사용한다. 메모리를 줄이려면 모든 변경된 연결부까지 정확히 복원하는 reference 경로를 별도 검증해야 하며 자동 fallback으로 넣지 않는다.

checkpoint에는 LoRA뿐 아니라 네 Merger master, 선택한 Vision adapter, optimizer·scheduler·RNG·step·rollout cursor·source/protocol hashes를 포함한다. 기존 adapter+Merger 저장 유틸리티는 재사용하되 RL 재개 상태는 추가해야 한다. BF16 배포본과 FP32 master 실행을 bit-identical하다고 주장하지 않는다.

기존 `fine_tuning.modules.load_checkpoint()`는 `is_trainable=False`로 adapter를 불러오는 **평가용 로더**다. 이것을 RL resume라고 부르지 않는다. 새 resume 경로는 학습 가능한 adapter/Merger, optimizer·scheduler·RNG·next group을 모두 복원하고, 중단 없이 진행한 작은 대조 실행과 다음 loss/update를 비교해야 한다.

### 11.4 사용자가 제안한 오류별 학습 경로를 시험하는 후속 계약

고정 모듈 비교가 통과한 뒤에만 다음 **별도 protocol**을 시험한다. 모델이 쓴 원인 설명 자체를 라우팅 정답으로 쓰지 않는다.

| 독립 검증 결과 | 후속 라우팅 제안 | 해석 금지 |
|---|---|---|
| 필수 원시 관측/관계 슬롯에 확정 오류가 있음 | Vision LoRA + 네 Merger + LLM LoRA | Vision Encoder만의 결함이라는 해석 |
| 필수 슬롯 모두 정확하고 검증 가능한 계산 단계가 틀림 | 네 Merger + LLM LoRA | 모든 시각 정보가 항상 정확했다는 해석 |
| 슬롯 누락·주석 모호·원인 검증 불가 | UNKNOWN; 기본 고정 범위 유지 또는 별도 보류 정책 | 검증 불가를 ‘Vision 정상’으로 변환 |

라우팅은 한 문제 그룹의 수집이 끝난 뒤 **update 전에 한 번 결정**하고, 그룹 내 각 재시도마다 가중치를 바꾸지 않는다. 근거 슬롯·정책 버전·route·실제 trainable parameter 목록을 저장한다. Vision OFF에서 vision adapter의 `grad=None`과 optimizer 상태 불변을 검사하고, Merger gradient는 차단되지 않게 한다. optimizer를 매번 새로 만들지 않는다. reference adapter 비활성화/복원 과정이 `requires_grad`를 다시 켤 수 있으므로 매 forward 전 선언된 route를 재적용·검사한다.

**그룹 내 판정이 엇갈리는 경우의 후속 제안 규칙:** 본 G4 설계에서는 네 episode의 **첫 시도**만 동일한 사전 고정 필수 슬롯으로 검사한다. 네 시도 모두에서 슬롯의 판정이 가능하고 하나라도 확정된 관측·관계 오류가 있으면 그 그룹은 Vision-ON, 모두 정확하면 Vision-OFF로 둔다. 하나라도 필수 슬롯 누락·모호한 주석·검증 불가가 있으면 UNKNOWN으로 남기고 기본 FML 범위만 업데이트한다. verifier 시스템 장애는 UNKNOWN으로 구제하지 않고 그룹 업데이트를 차단한다. 마지막 시도의 정답 여부나 모델의 자기반성 문장으로 이 결정을 뒤집지 않는다. 이 규칙은 모든 첫 시도의 검증 범위를 제한적으로 활용하는 **미실행 제안**이며, 별도 v2 구현의 라우팅 정책을 변경하거나 이미 검증됐다고 뜻하지 않는다. Vision-OFF 역시 해당 슬롯에서 오류가 관측되지 않았다는 뜻이지 Vision 전체가 정상이라는 인증이 아니다.

고정 FML, 고정 FVML, 검증 기반 routing, **같은 Vision-ON 비율의 무작위 routing**을 같은 데이터·노출로 비교해야 오류 정보를 활용한 이득을 구분할 수 있다. 이 대조 전에는 “오답노트가 정확한 학습 모듈을 골랐다”고 주장하지 않는다. 진단 문장을 reward로 학습하는 추가 과제도 검증 가능한 원시 슬롯 ID/명확한 원인 분류가 있을 때만 별도 실험으로 두고, 자유로운 Lesson 문장을 정답 target으로 CE 학습하지 않는다.

## 12. 데이터·rollout·보상 파일 계약

실제 구현 시 아래 세 파일을 **다른 접근 경계**로 둔다. 예시는 schema 설계이지 생성 완료 산출물이 아니다.

| 저장소/파일 | 필수 내용 | 접근자 |
|---|---|---|
| `actor_inputs.jsonl` | source_id/revision/family, 질문·보기, image role/order/hash, answer schema | rollout actor |
| `verifier_private.jsonl` | typed gold, 원시 사실/실행 프로그램, units/rounding, 주석 신뢰 범위 | reward 서비스·별도 evaluator |
| `lineage_manifest.json` | 원본 부모, source/license, exclusion index hash, transform, splits, 승인 | 준비기·guard |

`actor_inputs`의 공개 schema에 특정 답의 단위/값이 새지 않도록 **질문에서 독립적으로 정해지는 형식**만 담는다. 모델이 볼 파일명·caption·EXIF·URL도 검사한다. 실제 접근 격리는 OS 권한/분리 프로세스와 loader allowlist로 시행하며 단지 JSON key를 숨긴 것만으로 충분하다고 하지 않는다.

rollout에는 `policy_hash`, prompt/template/token IDs, image tensor/grid/hash, RNG seed, 각 turn의 raw·finish reason·input/output tokens·latency, feedback, old logprobs, action mask, reward와 verifier version을 저장한다. 과거 생성물은 불변이며 채점 수정은 새 reward overlay로 남긴다. 수정된 reward로 이미 한 업데이트를 소급 정당화하지 않는다.

생성 전 cache key:

```text
SHA256(model/checkpoint + policy_step + protocol + source_revision + sample/family
       + image_bytes/order + processor + template + task_mode + turn_history
       + rollout_seed + generation_config + verifier_version)
```

학습 전후 동일 파일명이라는 이유로 과거 completion·KV·멀티모달 feature를 가져오지 않는다. 데이터 준비 캐시와 정책 의존 추론 캐시를 분리한다.

## 13. 코드 구조: 현재 것을 재사용하고 RL만 별도 구현

### 13.1 실제 존재하는 재사용 후보

| 실제 파일 | 재사용할 책임 | 재사용하지 않을 것/추가할 것 |
|---|---|---|
| `fine_tuning/audit.py`, `sources.py` | lineage/split/오염 검사, 이미지 입력 감사 | 새 source·전체 금지 split coverage·RL 필드 분리 추가 |
| `fine_tuning/modules.py` | 실제 모듈 식별, freeze/LoRA, 전체 checkpoint | RL optimizer/RNG·fixed reference·신규 FVML 검사 추가 |
| `fine_tuning/nvw.py`, `circuit_bridge.py` | 독립 외부 생성·정확한 solver·일부 검증 | 전문 시각 전이·신규 악보 기능이 이미 있다는 주장 금지 |
| `fine_tuning/tabmwp_evaluate.py`, `circuit_scoring.py` | typed QA/facts 검증 논리 | RL reward 계약에 맞는 좁은 wrapper 및 반례 필요 |
| `experiments/input_recovery/backend.py`, `runner.py` | 독립 요청·이미지·원문·hash·비용 기록의 패턴 | 현재 inference 코드를 RL trainer라고 부르지 않음 |
| `fine_tuning/budget.py` | deadline·소유 process group 종료 | 새로운 실행 예산·시간은 별도 승인 |
| `src/mmdl/evaluation/`, `configs/eval/mmmu_val_v8.yaml` | 학습 후 고정 V8 비교 | training reward와 분리; 원본 V8 수정 없음 |

기존 `fine_tuning/train.py`를 이름만 바꿔 RL로 사용하지 않는다. 고정 정답 target CE 경로는 역사 실험용으로 보존하고 새 trainer에서 **호출하지 않도록** 검사한다. 모듈 전체 import를 금지한다는 뜻은 아니다. 실제 `policy.py::build_policy`는 그 파일의 `load_base`를, GPU update 검사기는 parameter probe 유틸리티를 재사용한다. 이 재사용과 `AssistantCollator`·정답 `target_text`·`model(..., labels=...).loss`를 통한 지도학습을 구분한다.

현재 별도 v2 설계의 `fine_tuning/notebook_rl/`에는 **actor/verifier 격리, 생성기·source adapter, rollout, 정책 손실, gradient routing, checkpoint/resume 및 bounded trainer의 CPU 구현과 시험**이 존재한다. [현재 구현 상태](../fine_tuning/notebook_rl/README.md), [rollout.py](../fine_tuning/notebook_rl/rollout.py), [losses.py](../fine_tuning/notebook_rl/losses.py), [trainer.py](../fine_tuning/notebook_rl/trainer.py), [checkpoint.py](../fine_tuning/notebook_rl/checkpoint.py)를 확인했다. README의 tiny-model optimizer 검사는 pretrained4B RL이나 vLLM GPU 인증, 성능 개선 결과가 아니다. 이번 연구 재검토에서는 이 패키지를 수정하거나 그 시험을 재실행하지 않았다.

**설계 차이:** 현재 패키지는 T1:8/T2:4의 동일 상태 turn-group 설계다. 이 보고서는 G4 독립 episode의 보상 설계다. 코드 존재를 이 보고서 구현 완료로 표시하거나 그룹·보상·loss·상한을 서로 섞지 않는다. 승인된 학습 pool·pretrained4B RL·전체 실제 GPU gate·성능 개선은 미완료다.

### 13.2 구현 위치 — 기존 코어를 재사용하고 프로토콜을 분리

```text
fine_tuning/notebook_rl/              # 존재하는 v2 코어; 무단 교체하지 않음
  schema.py / verifier.py            # 기존 입력 격리·typed 검증 재사용 후보
  rollout.py / losses.py             # 현재 turn-group 구현; G4와 다른 계약
  trainer.py / checkpoint.py         # bounded update·reference·resume 재사용 후보
  generators/ / contamination.py     # source 생성·승인 차단 재사용 후보
  protocol_v1.yaml                   # 기존 v2 계약은 그대로 보존
  <별도 episode 프로토콜·전략>       # G4 방식을 선택한 경우에만 새로 구현
fine_tuning/tests/test_notebook_rl_*  # 기존 CPU 검사 + 선택한 전략의 수치 반례
<별도 RL 환경 lock>                  # 실제 선택한 엔진·GPU 검증 후 고정
```

위 꺾쇠 항목은 **제안이지 존재하는 파일명이 아니다**. 새 `experiments/verified_rl/` 전체를 복제하는 대신 기존 격리·저장·checkpoint 코어를 재사용한다. 구현 착수 시 현재 v2를 유지할지 G4 episode 비교군을 추가할지 하나를 명시적으로 동결하고, G4를 추가한다면 reward·loss·mask·resume cache key와 자체 반례까지 독립 protocol ID로 검증한다. 기존 v2 결과를 G4로 재명명하지 않는다. 별도 큐 서버·웹 대시보드·복수 RL 프레임워크·온라인 teacher API는 기본안에 넣지 않는다. multi-turn 구현은 단일 turn의 수치 정합성 시험을 통과한 뒤 확장한다.

### 13.3 핵심 실행 순서의 의사코드

```python
# 설계 의사코드: 실행 가능한 기존 함수/완성 구현을 의미하지 않는다.
assert approved_data.no_benchmark_derivatives
assert frozen_reference.sha256 == pinned_f0.sha256
policy = fresh_pinned_f0_with_declared_trainable_modules()

for frozen_policy_batch in schedule:
    groups = []
    for prompt_group in fixed_family_balanced_sampler():
        episodes = []
        for rollout_id in range(4):
            history = actor_only_payload(prompt_group)
            episode = []
            for turn in range(prompt_group.max_attempts):
                action = policy_generate(history, independent_seed(rollout_id, turn))
                outcome = verifier_service(action, prompt_group.opaque_id)
                episode.append(record(action, outcome))
                if outcome.system_error or outcome.status == "UNVERIFIABLE":
                    abort_group_without_update()
                if outcome.is_correct or turn + 1 == prompt_group.max_attempts:
                    break
                history = append_exact_response_and_safe_feedback(history, action, outcome)
            episodes.append(close_episode(episode))
        groups.append(validate_group_and_compute_rewards(episodes))
    assert policy_version_unchanged_since_rollout_start()
    update_once_using_masked_group_policy_loss(groups, frozen_reference)
    save_recoverable_rl_checkpoint_and_usage()
```

금지사항은 코드에서 fail-closed로 검사한다. `teacher_model`, `sft_loss_weight`, `target_text` imitation, benchmark 입력 lineage가 있으면 학습 시작을 거부한다. 해당 문자열이 소스 파일 어디에 있다는 이유가 아니라 실제 config/dataflow/학습 loss를 검사한다.

### 13.4 ‘SFT/teacher 없음’을 실제 코드에서 입증하는 검사

| 확인할 주장 | 기존 코드와 필요한 검증 | 현재 경계 |
|---|---|---|
| gold를 따라 쓰지 않음 | [policy.py](../fine_tuning/notebook_rl/policy.py)의 학습 입력 allowlist가 `labels`를 거부. `AssistantCollator`·CE·legacy SFT update를 실패하도록 막은 상태에서도 실제 RL update가 가능한지 시험 | 코드·CPU 검사는 존재. native pretrained4B 실행 증거는 별도 필요 |
| 보상이 정책을 업데이트함 | [losses.py](../fine_tuning/notebook_rl/losses.py)·[updates.py](../fine_tuning/notebook_rl/updates.py)의 자기 생성 action logprob, advantage, KL, 실제 optimizer state·파라미터 변화 확인 | 현재 구현의 turn-group loss이며 본문 G4 episode loss의 실행 증거가 아님 |
| 상위 모델 호출이 없음 | [guards.py](../fine_tuning/notebook_rl/guards.py)의 teacher client 생성·요청 차단 및 원문 label provenance 검사 | vLLM의 OpenAI **schema-only 의존성**은 제한적으로 허용. 타입 import를 유료 호출로 오인하지 않음 |
| benchmark가 학습에 안 들어감 | 승인 worker가 전체 입력·이미지 hash·exclusion 원장·allowlist를 결합하고 실제 소비한 ID와 대조 | 승인 파일 존재만으로 부족. 변경된 이미지/행·미검사 source는 차단 |

현재 guard는 **Python 수준 방어이지 OS·네트워크 sandbox가 아니다.** 이미 열린 파일이나 native I/O까지 전부 차단한다고 주장하지 않는다. 실제 배포에서는 actor의 benchmark/vault mount 접근을 제거하고 외부 요청도 차단하는 운영 경계를 별도로 확인한다. 사전 설치된 모델 가중치와 schema 타입을 사용하는 것, 교사에게 풀이·라벨을 요청하는 것, 정답을 CE target으로 쓰는 것은 서로 다른 행위다. 검증 보고에는 이 세 경계를 따로 남긴다.

## 14. 라이브러리 선택과 구현 시 먼저 검증할 지점

### 14.1 TRL은 후보이지 현재 환경에서 검증 완료된 경로가 아니다

우선 현재 v2 코어의 격리·loss·저장 기능을 재사용할 수 있는지 검증한다. Hugging Face **TRL GRPOTrainer**는 수치 대조·대체 backend 후보이며, 이미 존재하는 trainer를 근거 없이 전면 교체하지 않는다. 2026-10-17 재확인한 release는 `v1.16.0`(2026-10-15 게시), 검토 source SHA는 `447be34c59a4ad7fb6212917382c62053bab147b`다. **release tag와 이 main snapshot SHA가 같다는 뜻은 아니다.** TRL을 실제 채택한다면 하나를 정해 lock하고 시험해야 한다. [릴리스](https://github.com/huggingface/trl/releases/tag/v1.16.0), [검토 source](https://github.com/huggingface/trl/blob/447be34c59a4ad7fb6212917382c62053bab147b/trl/trainer/grpo_trainer.py)

확인한 source는 custom `rollout_func(prompts, trainer)`와 `prompt_ids`, `completion_ids`, `logprobs`, 추가 `env_mask` 처리를 제공한다. 이는 연결할 수 있는 API라는 증거이지, 우리의 이미지·두 턴·Merger 업데이트가 자동으로 올바르게 지원된다는 증거는 아니다. 환경 피드백을 action loss에서 빼고 실제 image tensor를 학습 forward에 전달하는 경로를 시험해야 한다. [공식 GRPO 문서](https://huggingface.co/docs/trl/main/en/grpo_trainer)

공식 multi-turn 경로 일부는 Transformers≥5.2와 prefix-preserving chat template 등 새 조건을 요구한다. 기존 평가 환경의 Transformers4.57.1/Torch2.8.0/vLLM0.11.0을 제자리에서 업그레이드하지 않는다. **새 RL 전용 venv와 lock**을 만들고, 지정 모델/processor revision의 입력 일치·Qwen3-VL forward·LoRA 및 네 Merger gradient를 확인한다. generic VLM 지원이나 Qwen3-VL-2B 예제가 우리4B·학습 범위의 성공을 대신하지 않는다.

### 14.2 구현 gate

1. 단일 turn/자체 가상 문제에서 직접 계산한 loss·gradient와 trainer 결과 대조.
2. 두 turn의 원문·이미지·prefix·feedback mask·첫 시도 중복 loss 여부 확인.
3. frozen F0 reference의 해시·logprob가 여러 update 뒤에도 불변인지 확인.
4. 신호가 있는 fixture와 여러 step에서 선언한 모듈의 gradient 연결·파라미터 변화를 확인하고 frozen base는 불변인지 확인. LoRA의 B=0 초기화에서는 첫 step A의 gradient가0인 것이 정상이며, 전부 동일한 reward와 초기 KL0도 정상 zero-update다. 모든 tensor가 매 step 반드시 변해야 한다는 잘못된 gate를 만들지 않는다.
5. 중단/재개 전후 같은 다음 batch·RNG·정책 loss·reward를 재현.
6. fresh process에 전체 checkpoint를 로드해 생성과 scoring 경로 확인.

처음에는 HF generation으로 correctness를 확인한다. vLLM 가속은 **학습한 모든 가중치의 동기화**가 확인된 뒤 도입한다. LLM LoRA만 갱신하고 Vision/Merger는 옛 가중치로 rollout하면 정책이 달라져 잘못된 RL이 된다. 이미지 feature/prefix/KV cache 무효화도 포함한다.

서버형 vLLM과 trainer를 같은 GPU에 올리면 이중 가중치·KV 예산을 실측해야 한다. inference 시 기록한 logprob와 training forward logprob가 다르면 sampler/precision/kernel/padding을 조사한다. 불일치를 무시하고 importance ratio=1로 고정하지 않는다. [TRL vLLM 안내](https://huggingface.co/docs/trl/main/en/grpo_trainer)

메모리 검사에는 vocabulary logits와 backward 저장량을 포함한다. 현재 Qwen의 vocab151,936 × completion4,096 × FP32만 약2.32GiB이며 모델/활성값/gradient/KV는 별도다. LM head를 1,024 token씩 단순 반복해도 autograd가 모든 chunk의 log-softmax를 보존하면 이 비용은 없어지지 않는다. 실제 underlying hidden-state 경로, checkpointed chunk 또는 검증된 fused loss를 선택해 peak를 측정하며 `output_hidden_states=True`만으로 전체 logits 할당이 사라진다고 가정하지 않는다.

대안인 [verl multi-turn](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)은 별도 검토 후보일 뿐, 두 프레임워크를 동시에 구현하지 않는다. 최초판에서 MCTS·PRM·자기 질문 생성까지 추가하지 않는다.

추가로 [RLinf의 Qwen3-VL GRPO 공식 예제](https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/agentic/qwen3_vl_geo3k.html)는 Transformers4.57.1·SGLang0.5.4와2B/Geo3K·0/1 reward 경로를 공개한다. 기존과 같은 Transformers 버전을 쓰는 구현 사례라는 참고이지, 현재 환경 그대로의 성공이나 우리4B/데이터/다중 턴/학습 모듈 검증은 아니다. Geo3K를 승인된 학습 데이터로 자동 추가하지 않는다. 이 예제를 택한다면 TRL 대신 하나만 선택하고 별도 exact lock과 수치 검사를 수행한다.

### 14.3 현재 코어를 연결할 때 남아 있는 구체적인 구현 과제

2026-10-17 로컬 코드를 다시 대조한 결과다. 아래 `G3/G12/G13`은 **별도 v2 코어의 검사 식별자**이며, 본 문서 §18의 단계 번호와 같지 않다. 함수가 존재하는 것과 end-to-end 검사가 실제 GPU에서 통과한 것은 다르다.

| 실제 연결 지점 | 먼저 해결할 것 | 완료 증거 |
|---|---|---|
| reference의 업데이트 후 비교 | 현재 `gate_reference`의 POST_SYNC 경로는 정책·별도 fresh F0·살아 있는 vLLM을 같은 CUDA 장치에서 요구한다. 두 모델 사본만으로 계산한 VRAM 예산을 이 검사에 적용하지 말 것 | 실제 세 사본의 peak 측정 또는 별도 검증된 순차 비교 설계; OOM을 설정 자동 축소로 숨기지 않음 |
| 실제 step·가드·재개 시험 | `gate_throughput`의 한 RL step을 `gate_guards`의 정확히 한 update 검사로 감싼다. `gate_checkpoint`는 다음 step을 다시 실행하므로 같은 context에 중첩하지 않는다 | 첫 step 후 동기화와 새 policy version의 batch, fresh-process 재개 후 동일 next loss·정책/참조 logits |
| 재개 시험의 입력 전달 | `gate_inputs`의 저장·로드 및 capture/replay 연결은 현재 코드에 존재한다. 실제 tree·action token·이미지 순서/해시·literal probe 입력을 하나의 packet에 묶는다. 이를 단순 파일 존재가 아니라 실제 소유 자식 프로세스 실행 순서에 연결해야 한다 | 동일 packet으로 fresh-process 재개 비교. 입력 packet 자체는 데이터 승인·GPU 통과·일반 학습 재개 cursor가 아님 |
| GPU 원장과 최종 승인 | checkpoint 비교의 결과 키와 approval이 요구하는 정책/참조별 결과를 명시적으로 연결한다. CPU fixture의 `passed`를 native GPU 통과로 바꾸지 않는다 | 실제 실행 장비·원문 측정값·소유 child 종료·각 hash에 연결된 proof |
| 사전 측정과 학습 승인 순서 | Phase-A 이전 데이터 검증에는 미래의 selection/Phase-A hash가 없다. 측정 가능 상태와 학습 가능 상태를 분리하고, 상한·protocol이 바뀌면 영향을 받는 증거를 새로 검증한다 | 최종 protocol·selection·Phase-A·모델·코드가 같은 승인 원장; 이전 증거를 소급 재명명하지 않음 |

근거: [reference 검사](../fine_tuning/notebook_rl/gate_reference.py), [한 step 계측](../fine_tuning/notebook_rl/gate_throughput.py), [no-SFT/data guard](../fine_tuning/notebook_rl/gate_guards.py), [checkpoint 비교](../fine_tuning/notebook_rl/gate_checkpoint.py), [승인기](../fine_tuning/notebook_rl/approval.py). 이번에는 이 코드를 수정·실행하지 않았다. 위 연결부를 해결하지 않은 상태에서 ‘설치 후 바로 본학습 가능한 파이프라인’으로 보고하지 않는다.

추가로 [재개 입력 연결](../fine_tuning/notebook_rl/gate_inputs.py)의 `save_inputs`, `capture_from_inputs`, `replay_from_inputs`와 README를 읽기 전용으로 확인했다. 이미지 복원은 checkpoint의 RNG 복원 전에 수행하고, 캡처된 step·batch·입력 hash를 재개 측과 결합하도록 구현되어 있다. 기존 CPU 시험과 native pretrained4B GPU 실행은 구분하며, 이번 문서 검토에서는 해당 시험을 다시 실행하지 않았다. 전체 GPU 검사 조정기와 실제 승인 데이터가 여전히 필요하다.

## 15. 제안 hyperparameter와 비용을 계산하는 방법

다음은 **시작 후보**이며 실행 완료 설정이 아니다. GPU smoke 결과에 따라 바꾸려면 새 protocol revision과 사유를 남긴다.

```yaml
protocol_id: verified-self-correction-rl-draft-1
status: DESIGNED
initial_checkpoint: fresh_pinned_F0
additional_sft: false
teacher_distillation: false
benchmark_training: false
precision: bf16
image:
  min_pixels: 1003520
  max_pixels: 4014080
  preprocessing: qwen_vl_utils_0_0_14
context:
  max_model_len: 40960
  overflow_policy: reject_without_silent_truncation
rollout:
  group_size: 4
  max_attempts: 2
  max_new_tokens_per_attempt: 4096
  max_generated_tokens_per_episode: 8192
  temperature: 1.0
  top_p: 1.0
  top_k: disabled
  repetition_penalty: 1.0
  presence_penalty: 0.0
  generation_micro_batch: 1
optimizer:
  family_groups_per_update: 8
  updates_per_rollout_batch: 1
  llm_lora_lr: 0.000005
  merger_lr: 0.000001
  optional_vision_lora_lr: 0.0000005
  max_grad_norm: 1.0
  warmup_ratio: 0.10
  scheduler: constant_after_warmup
loss:
  objective: clipped_group_policy_loss_with_fixed_length_normalization
  group_reward_std_scaling: false
  clip_epsilon: 0.2
  kl_beta: 0.02
  single_turn_anchor_group_fraction: 0.5
seeds: [3407, 3408, 3409]
evaluation:
  max_new_tokens: 8192
  prompt: frozen_P0
  parser: team-final-answer-v8
  sampling: existing_Qwen_official_recipe
```

**훈련 생성과 제출 평가의 설정은 구분한다.** 첫 RL sampler는 확률 계산을 단순하게 검증하려고 T=1/top-p=1/no penalty를 제안했다. 평가에서는 기존 T0.7/top-p0.8/top-k20/presence1.5 등 고정 recipe를 유지한다. training sampler를 바꾼 사실을 숨기거나 기존 baseline 생성 조건도 바꾸지 않는다.

사용자가 선택한 새 **평가 상한은8K**다. 훈련에서는 첫4K+재시도4K라는 별도 비용 제안을 했으며, 모든 실험이8K 한 번이었다고 쓰지 않는다. 각 첫 시도4K인 R1/R2는 첫 상태·cap을 맞출 수 있지만 R2의 총 계산량은 더 크다. 따라서 **동일 root 노출 비교**와 **동일 실제 생성 token/GPU 시간 비교**를 별도로 보고한다. 둘을 동시에 동일했다고 주장하지 않는다.

최초 pilot은256개/1회 노출/seed3407이다. G4라서 최대1,024episode, 모두 two-turn이면 최대2,048개 응답이다. 제안한 절반 single-turn 혼합에서는 최대1,536응답이다. 실제로 첫 시도 정답이면 재시도가 없어 감소한다. 각 응답이4,096에 도달하면 혼합판 최대 출력은 **6,291,456token**이다. 작은 데이터256개라도 학습은 가벼운 작업이 아니다.

8 prompt groups/update라면 epoch당32update, episode32개/update다. generation micro-batch1은32개를 VRAM에 동시에 올린다는 뜻이 아니다. Vision의 고해상도 활성값·reference forward·backward까지 포함해 계측한다. 추가 seed3408/3409는 첫 pilot 통과와 예산 승인 뒤 수행한다.

4,000개 확장은 `4000/8=500update/epoch`, G4이면16,000episode/epoch로 산출한다. 혼합판 최대24,000응답·98,304,000출력token이다. 이 최대량이 가능하다는 시간/비용 보장은 없으며 실제 평균 길이와 early-stop 비율로 재추정한다.

## 16. 평가: 낮은 과목 전체와 일반 능력을 함께 보기

### 16.1 62%와 새8K 성적의 관계

과거558/900은 원래32K 응답에 대한 고정 V8 결과다. 새 RL 모델8K를 그62%와 바로 비교해 “학습으로 향상/하락”이라고 하지 않는다. **같은 GPU·환경·8K·P0·V8의 새 F0와 FT**를 paired 비교해야 한다. 가능하면 실제 모델 입력 hash·동일 문항 seed까지 맞춘다.

기존 점수에서 `<70%`인18과목은 전체540문항이다. Accounting, Sociology, Manage, Math, Pharmacy, Art, Geography, Computer_Science, Agriculture, Chemistry, Biology, Energy_and_Power, Architecture_and_Engineering, Mechanical_Engineering, Electronics, Materials, Diagnostics_and_Laboratory_Medicine, Music을 사전 고정한다. 정확히70%인 Finance는 이 집합에 넣지 않는다.

하지만 **540개만 보고 일반 능력 보존을 주장하면 안 된다.** 최종 선택한 하나의 FT와 새 F0에 대해 validation900개를 모두 평가하고, 저득점18과목540개와 나머지12과목360개를 나눠 함께 낸다. 학습 중 매 checkpoint를 MMMU에 반복 평가해 승자를 고르는 기본안은 피한다. 외부 dev에서 방법·checkpoint를 고정한 뒤 benchmark를 사용한다.

MMMU-Pro 최종 비교는 수업의 Standard10-way이고, 현재 로컬 고정 자료는1,730행이다.900으로 가정하지 않는다. full test/protocol 및 예산을 별도 승인한 뒤 실행하며 Pro 정답을 학습·문제 난도 선택·보상 조정에 사용하지 않는다.

### 16.2 외부 평가를 먼저 고정

- train/dev/sealed를 **이미지·표·문서·악곡·scene family** 기준으로 분리한다.
- pilot 목표는 별도 dev128개와 sealed256개다. 충분한 독립 family가 없으면 미달 상태로 남기고 기존 평가 사례로 채우지 않는다.
- 확장 시 외부 평가 목표는 네 실패군 각각200개/총800개다. 독립 source 및 renderer/style의 분포 이동도 포함한다. 목표 수량이지 취득 완료 수량이 아니다.
- 회로 한 topology·합성악보·단순 CLEVR로 좋아져도 실제 Materials/Diagnostics 전체가 개선됐다고 하지 않는다. C 분야가 빠져 있으면 결과의 범위도 A/B/D로 제한한다.
- 이미지 없음 T, 이미지 I, 검증된 원시 전사 T-observation 대조는 외부 진단으로만 수행한다. oracle 성공을 Vision/Merger의 유일 책임으로 해석하지 않는다.

### 16.3 필수 비교와 지표

| 비교 | 분리할 효과 |
|---|---|
| fresh F0 vs R1 | 보상 기반 학습 자체 |
| FML-R1 vs FL-R1 | 연결부 추가; 같은 데이터·보상·노출 유지 |
| R1 vs R2 | 두 시도 학습; 동일 노출과 실제 계산량 차이를 함께 보고 |
| R2+동일 observation schema/λ0 vs λ0.1 | 시각 보상만의 추가 효과 |
| 고정 FML vs FVML | Vision LoRA 추가; 다른 변경과 섞지 않음 |

주 지표는 **처음 보는 이미지의 첫 시도 정확도**다. 보조 지표는 외부 데이터에서의 feedback 없는2-turn, 정오 feedback 있는2-turn의 최종 정확도·오답→정답·정답→오답·NO_PARSE·length·정확한 사실 슬롯 비율·평균/상위95% token·실측 시간이다.

정답 즉시 종료하는 학습 rollout에서는 정답→오답 전이를 관측할 수 없다. 따라서 외부 진단의 별도 **forced second attempt**에서 정답 여부를 알려주지 않고 두 번째 시도를 요청해 악화를 측정한다. 이를 학습 보상이나 MMMU 제출 절차에 섞지 않는다.

재시도 효과의 대조군에는 같은 총 출력 예산의 독립 재생성 및 동일 scorer를 둔다. best-of-N은 gold로 고른 oracle upper bound와 실제 무정답 선택기를 구분한다. `pass@k`가 높아졌다는 것만으로 한 번 답하는 능력이 높아졌다고 하지 않는다.

불확실성은 문항뿐 아니라 공유 family를 묶는 paired bootstrap으로 계산한다(초기 제안2,000회). 추가 seed를 독립 새 문제로 세지 않는다. 회복률의 분모는 baseline 오답 수, 악화율은 baseline 정답 수이며0분모는 N/A다. 각 과목30문항의1문제=3.33%p라는 작은 표본 한계를 명시한다.

### 16.4 확대 판단

사전 dev에서 첫 시도 QA가 좋아지고 일반 유지 지표가 악화되지 않으며, verifier 오류·reward hacking·checkpoint 문제가 없어야 확장한다. 예비 운영선으로 QA+3%p, retention−2%p 이내를 검토할 수 있으나 **통계적으로 검증된 상수나 유의성 기준이 아니다.** 미리 동결하고 신뢰구간·다른 seed를 함께 판단한다. 불명확하면 `INCONCLUSIVE`이지 개선 성공이 아니다.

## 17. 하드웨어·시간·비용 운영 설계

현재 로컬 RTX5060 8GB에서 전체 BF16 Qwen4B·frozen reference·GRPO backward가 가능하다고 가정하지 않는다. 로컬은 데이터·검증기·loss 수학·마스킹·파일 검증을 우선한다. 실제 GPU 모델·메모리는 실행 직전에 다시 확인한다.

**초기 원격 후보는48GB급 단일 GPU**다. 이는 지금 대여하겠다는 뜻도,48GB면 무조건 된다는 뜻도 아니다. 기존 모델 파일의 크기를 기준으로 actor와 별도 reference만으로 대략16GiB 이상이 필요하고, FP32 master·gradient·Adam 상태·activation·KV·vision token이 추가된다. 네 Merger의 FP32 weight/gradient/Adam 두 상태만 단순 합산해도 약1.63GiB다. 구현별 중복 저장은 별도다.

3090 24GB는 LLM-only의 작은 rollout/메모리 최적화에서 가능성을 검토할 수 있지만 **Vision+Merger+LLM 공동 RL까지 적합성이 인증된 장비가 아니다.** 사용자 승인 없이 quantization·해상도·문맥·학습 모듈을 줄여 맞추지 않는다. 학습 동시 batch와 rollout의 연속 추론 동시성을 같은 것으로 설명하지 않는다.

운영 순서:

1. CPU에서 취득·이미지 변환·중복 검사·reward 반례·manifest·패키지 준비를 끝낸다.
2. 실제 대여 가격·잔액·시간 상한·회수 공간을 확인하고 유료 작업을 승인받는다.
3. exact commit/lock의 clean clone → CUDA/BF16 검사 → 최대 입력 사례 memory smoke → 실제 policy update·reload.
4. 첫 RL pilot의 GPU 작업 상한은 **4시간 제안**이며 데이터 준비/설치/회수 과금도 전체 예산에 포함한다.
5. 10분 주기로 Pod/학습/오류·비용·진행을 점검하고, 시스템 watchdog은 연결이 끊겨도 cutoff를 지킨다.
6. 시간 부족 시 완료 episode·checkpoint·optimizer를 보존하고 `BUDGET_STOP`. 덜 학습한 조건을 full 완료로 표시하지 않는다.
7. 로컬 회수→SHA/내용/재로딩 검증→정확한 소유 Pod 삭제→목록 부재 확인. Stop만으로 과금0이라 하지 않는다.

예측식은 `설치 + rollout prefill/decode + old/reference/current forward + backward/update + dev + 회수`다. 추론 tokens/s만으로 학습 시간을 계산하지 않는다. pilot에서 실제 episode·optimizer당 시간과 메모리를 측정한 뒤 4,000개 확장 및 새 F0/FT900·Pro 비용을 따로 산출한다. 현재 잔액·가격·12시간 내 완료를 과거 수치로 보장하지 않는다.

## 18. 구현·시험·완료 gate

| Gate | 구현/검증 항목 | 통과 증거 |
|---|---|---|
| G0 조건 | 추가 SFT 금지와 RLVR 보상 허용 범위, fresh F0, 데이터/실행 범위 | 사전 고정 protocol·승인 |
| G1 입력 | 권리·원본/family·전체 금지 split 검사·actor/gold 격리 | source manifest·exclusion receipt·정확한 파일 목록/hash |
| G2 검증기 | 단위·단일답·철회·충돌·gold 중간출현·반올림·image/solver 정합성 | 자체 제작 정상/반례 쌍 및 수작업 독립 표본 검사 |
| G3 CPU RL | G별 advantage, toy loss/gradient, feedback/prompt mask, group 평균·gradient accumulation | 손검산·자동검사 일치; mock과실물 구분 |
| G4 GPU smoke | 실제 pixel tensor·gradient·reference 고정·전후 checkpoint·fresh reload | 실제 소규모 update·tensor 변화·최대입력 memory 원장 |
| G5 외부 pilot |256개·G4·고정 상한·한seed, dev/retention/비용 | 원문·reward·policy version·학습 곡선·오류/미완료 원장 |
| G6 반복/확장 | 추가seed·새 independent holdout·네실패군 source 승인 | paired QA/관측/retention·CI·source coverage |
| G7 benchmark | 방법 고정 후 같은8K F0/FT900, 별도Pro | V8 원장·30과목·저득점540/나머지360·제출 보고 |

구체적 회귀검사 목록:

- actor loader가 `gold`, `solution`, `target_text`, benchmark-derived field를 거부하는지.
- gold를 바꿔도 actor payload·첫 rollout seed가 같고 verifier 결과만 달라지는지.
- benchmark exact/near/lineage가 발견되면 quarantine되는지; 승인 receipt가 이미지 변경을 탐지하는지.
- shared image/table/source family가 train/dev/sealed에 걸치지 않는지.
- 정답을 포함한 여러 후보·unit mismatch·truncated revision·no final이 부당한 reward를 얻지 못하는지.
- 두 턴의 기존 원문·이미지 순서가 유지되고 다른 episode history가 섞이지 않는지.
- feedback·padding mask0, 새 action mask1, 이전 action 중복 loss0인지.
- toy 두 episode의 수치 loss와 gradient가 수작업 계산과 같은지; all-equal rewards 및 KL-only 구분.
- checkpoint마다 Vision/Merger/LLM trainable inventory가 정확하며 frozen F0 reference는 불변인지.
- restart 후 optimizer/RNG·next root·정책 버전·cache key가 맞는지.
- N/A·SYSTEM_ERROR·BUDGET_STOP을 task오답/성공으로 섞지 않고 비용에서 누락하지 않는지.
- benchmark evaluator에 feedback/훈련 노트/gold oracle이 들어가지 않고 V8 원본 결과가 불변인지.

별도 v2 코어의 CPU 구현·시험이 존재하지만 **본 보고서의 G4 계약에는 G3/G4/G5 통과 증거가 없다.** 특히 pretrained4B 정책 업데이트·전체 GPU gate·학습 개선과 tiny CPU 검사를 구분한다. 이번에 완료하는 범위는 **근거 조사·기존 코드 확인·TabMWP1600 CPU 중복 재검사·상세 설계**다.

### 18.1 구현 순서와 산출물

| 순서 | 구현할 것 | 다음 단계 진입 조건 |
|---|---|---|
| 1 | actor/verifier schema·strict verifier·기존 source adapter | 누설·정답 반례·원장 검사 통과 |
| 2 | weights-frozen G4 single/two-turn rollout | 독립 이력·이미지·피드백·mask·비용 검증 |
| 3 | single-turn RL trainer·모듈/고정reference·resume | toy와 실제 GPU 최소 update·reload 통과 |
| 4 | R2 episode reward·single-turn anchor·two-turn trainer | sequence loss·행동 토큰·정책 일치 검증 |
| 5 | 외부 pilot와 fixed-module 비교 | 첫 시도·retention·비용 gate |
| 6 | 검증된 observation reward, 필요시 Vision 추가 | 같은 schema/데이터의 ablation |
| 7 | 범위 확장·새8K 전체 평가 | 외부 dev에서 방법 동결·예산 승인 |

이 순서는 일정 약속이 아니다. CPU 구현에 필요한 시간, GPU 속도, 새 source 검토 인력은 아직 측정하지 않았다. 계획서만 계속 늘리지 않고 각 단계에서 실행 코드·통과 검사·실제 입력·실제 응답·구체적 blocker 중 하나를 남긴다.

## 19. 예상 질문과 답변

**Q. 모델이 답을 맞힐 때까지 반복하면 학습 아닌가?**

A. 가중치가 변하지 않으면 inference-time search다. 이번에는 생성 action의 logprob에 검증 보상으로 계산한 advantage를 적용해 실제 파라미터를 바꾼다. 무제한 반복 대신 두 시도부터 시작한다.

**Q. 정답표가 있는데 지도학습 아닌가?**

A. 감독 신호는 존재한다. 그러나 정답 문자열을 따라 쓰는 SFT와 스스로 생성한 행동에 보상을 주는 RLVR는 손실·데이터흐름이 다르다. 이번에는 외부 원출처 주석을 verifier reward에만 사용한다. 이를 완전한 비지도학습이라고 부르거나 용어만 바꾸어 SFT를 수행하지 않는다.

**Q. 왜 오답노트대로 Vision 또는 LLM을 선택하지 않는가?**

A. 노트는 모델의 사후 설명이지 모듈 인과 분석이 아니다. 원시 값 오류는 검증하되 내부 원인은 UNKNOWN으로 남기고, 고정 모듈 비교에서 학습 효과를 측정한다.

**Q. 최종 정답만 맞으면 충분한가?**

A. 제출 정확도에는 중요하지만 우연한 상쇄·찍기·부정확한 풀이도 가능하다. 정답 보상부터 명확히 하고, 검증 가능한 시각 슬롯·계산 검사를 별도로 측정/추가한다. 자유로운 풀이 전체가 참이라고 인증하지 않는다.

**Q. 이전 학습 실패를 RL로 바꾸면 해결되는가?**

A. 보장할 수 없다. 이전 SFT에서 셀 추출과 QA가 분리된 관측 때문에 정답 완결을 직접 보상하는 목적함수를 시험할 가치가 있다. 하지만 verifier 오류·희소 보상·학습 분포·망각·시각 표현 병목은 RL에도 남는다.

**Q. 지금 추천 데이터가 전부 안전한가?**

A. 아니다. TabMWP 특정1600개의 실제 이미지까지 CPU fingerprint 검사를 다시 실행하여 후보0을 확인했지만, 신규 후보는 승인 전이다. benchmark 원본 Parquet 전수 재독이나 OCR·crop·semantic 검사는 이번에 수행하지 않았다. 원천 overlap·변형·semantic similarity·license·annotation 신뢰도를 함께 관리한다.

## 20. 참고문헌과 인용 메모

논문을 발표·최종 보고서에 인용할 수 있다. 아래 제목/버전 URL을 출발점으로 실제 사용한 절과 코드 SHA를 남기고, 최종 BibTeX는 저자/arXiv 메타데이터에서 가져온다. 논문명 인용만으로 우리 실행·성능이 검증되는 것은 아니다.

arXiv 공개와 학회 채택은 구분한다. SCoRe는 [ICLR 2025 공식 논문집](https://proceedings.iclr.cc/paper_files/paper/2025/file/871ac99fdc5282d0301934d23945ebaa-Paper-Conference.pdf)도 확인할 수 있다. 나머지는 아래 고정 arXiv 버전을 인용하되, 학회·연도·peer review 상태를 확인하지 않고 최상위 학회 논문이라고 쓰지 않는다. 가장 직접적인 근거도 다른 모델·데이터의 결과이며, 우리 모델의 개선폭은 새 실험 전에는 비워 둔다.

1. Kumar et al. **Training Language Models to Self-Correct via Reinforcement Learning**. 2024. [arXiv:2409.12917v2](https://arxiv.org/abs/2409.12917v2). 자기수정 RL의 방법 근거; intrinsic vs verifier-assisted 차이 명시.
2. Shao et al. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**. 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300). GRPO 출처; 논문 전체의 SFT를 배제했다고 명시.
3. DeepSeek-AI. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**. 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948). R1-Zero만 no-additional-SFT 사례로 구분.
4. Liu et al. **Understanding R1-Zero-Like Training: A Critical Perspective**. 2025. [arXiv:2503.20783](https://arxiv.org/abs/2503.20783). 정규화 편향·Dr. GRPO 참고.
5. Li et al. **Self-Rewarding Vision-Language Model via Reasoning Decomposition** (Vision-SR1). 2025 v1. [arXiv:2508.19652v1](https://arxiv.org/abs/2508.19652v1). [v2](https://arxiv.org/abs/2508.19652v2)는 제목·방법 수정이 있어 버전 구분.
6. Jiang et al. **PROPA: Toward Process-level Optimization in Visual Reasoning via Reinforcement Learning**. 2025. [arXiv:2511.10279](https://arxiv.org/abs/2511.10279). 과정 탐색의 참고; teacher·SFT 경로는 사용하지 않음.
7. Wang et al. **ViCrit: A Verifiable Reinforcement Learning Proxy Task for Visual Perception in VLMs**. 2025. [arXiv:2506.10128](https://arxiv.org/abs/2506.10128). 오류 위치 보상의 참고; GPT 기반 생성 대신 독립 프로그램 검증을 제안.
8. He et al. **VisPlay: Self-Evolving Vision-Language Models from Images**. 2025. [arXiv:2511.15661](https://arxiv.org/abs/2511.15661). 자기 생성·난도 관리의 참고; 다수결을 진실로 취급하지 않음.
9. Hugging Face. **TRL GRPOTrainer documentation / source**. [공식 문서](https://huggingface.co/docs/trl/main/en/grpo_trainer), [검토 source SHA](https://github.com/huggingface/trl/blob/447be34c59a4ad7fb6212917382c62053bab147b/trl/trainer/grpo_trainer.py). 프레임워크 구현 참고이지 우리 경로 검증 완료 아님.
10. Qwen Team. **Qwen3-VL Technical Report**. [arXiv:2511.21631](https://arxiv.org/abs/2511.21631), [지정 모델 카드](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md). 모델 구조·공식 조건 출처.
11. Wu et al. **On Asymmetric Optimization of Reasoning and Perception in Vision-Language Model Post-Training**. 2026. [arXiv:2605.29496v2](https://arxiv.org/abs/2605.29496v2). 지각·추론 측정과 보상 결합의 근거; SFT-init/teacher 경로는 제외.
12. Tang et al. **MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction**. 2026. [arXiv:2604.01600](https://arxiv.org/abs/2604.01600). 두 시도 RL·보상 악용·no-cold-start ablation을 구분해 참고.
13. Bai et al. **Chart-RL: Policy Optimization Reinforcement Learning for Enhanced Visual Reasoning in Chart Question Answering with Vision Language Models**. 2026. [arXiv:2604.03157](https://arxiv.org/abs/2604.03157). 직접4B/no-preliminary-SFT 사례; 해상도·judge·구현 한계 명시.
14. Li et al. **StructReward: Efficient Structured Process Rewards for Self-Correcting Multimodal Reasoning**. 2026. [arXiv:2608.08326v2](https://arxiv.org/abs/2608.08326v2). 관련 연구로 인용하되 SFT/teacher/rollout imitation은 채택하지 않음.

데이터의 논문·코드·권리는 §4의 각 원저자 링크와 실제 선택할 release/파일 license에 연결한다. 다른 논문의 혼합 데이터 명칭을 근거 없이 우리 manifest의 출처로 쓰지 않는다.

## 21. 이번 작업의 완료 범위

- **완료:** 관련 원논문/공개 코드 조사와 2026-10-17 재대조, 현재 프로젝트의 SFT·모듈·검증 코드 확인, 특정 TabMWP1600의 실제 이미지·질문 CPU 재검사와 전후 hash 대조, 추가 SFT 없는 RL 설계와 별도 도식 작성. 독립 검토에서 제기된 데이터 생성 provenance, near-text 캐시 누락·짧은 질문 검사 한계, LoRA 초기 zero-gradient, 평가 로더와 학습 재개 구분을 보완했다.
- **산출물 검증:** 2026-10-17 최종 보완본의 서로 다른 로컬 근거 링크35개 존재·코드 블록9개·22개 절·외부 렌더 의존성0개를 확인했다. 기존 Windows Chrome으로 HTML/SVG를 실제 렌더링해 글자·배경 대비·연결선·전체 표시를 확인했다. MCP 브라우저는 해당 Linux Chrome이 없어 사용하지 못했으며 새 브라우저를 설치하지 않고 기존 Windows 브라우저를 사용했다. 독립 Sol 검토에서 구현 경계·데이터 승인·설계 정합성을 다시 확인했다. 이는 문서/도식 검증이며 새 RL 코드의 단위 테스트나 GPU 성공이 아니다. episode 보상의 첫 오답 강화 위험, 단일 optimizer update에서 PPO clipping이 비활성이라는 점, 실제 생성 분포·EOS/action mask의 검증 계약을 명확히 했다.
- **구현 경계:** 별도 v2의 actor/verifier·rollout·정책 손실·routing·checkpoint·bounded trainer와 CPU 시험은 존재한다. 본 보고서 G4 episode 설계의 실행 구현 완료와는 다르다.
- **미완료:** 본 G4 설계의 CPU/GPU 통합 gate, 새 데이터 전체 취득·전수 승인, pretrained4B RL 학습, 보상/모듈 최적값, MMMU/Pro 개선 성적.
- **이번 최종 보완:** SCoRe·Vision-SR1·PROPA·ViCrit·VisPlay·MM-ReCoder·Asymmetric Optimization의 핵심 방법과 원출처 데이터 안내를 다시 대조했다. 별도 Sol 읽기 전용 검토와 실제 CPU 재검사로 scan v2의 미승인 상태·제한된 중복 검사 범위를 확인했다. Geometry3K–PGPS9K 공유 원천·권리 조건은 §4–5에, 실제 GPU 통합의 남은 연결 문제는 §14.3에 명시했다. 새 RL 코드나 학습은 실행하지 않았다.
- **수행하지 않음:** RunPod 대여/기존 Pod 조작, 유료 모델 API, 모델 가중치 변경, GitHub 게시, `MMDL/` 동기화, 기존 V8 파서·응답·점수 변경.

**다음 구현의 시작점:** 외부 원출처 정답은 verifier reward에만 사용하는 범위를 유지하고, §13의 기존 코어와 선택한 학습 계약을 먼저 맞춘다. 새 데이터 승인과 §18의 CPU gate·reward 반례를 통과시키며, 별도 환경의 실제 정책 업데이트 smoke와 예산 승인 전에는 대규모 데이터 취득·GPU 학습을 시작하지 않는다. 본 설계와 실제 구현의 차이는 새 protocol revision으로 남긴다.

이 문서는 다음 구현 작업의 계약 초안이다. 성공 기준은 재시도 후의 높은 숫자만이 아니라 **검증 가능한 reward·누설 없는 데이터·실제 정책 업데이트·독립 문제의 첫 시도 개선·일반 능력 보존**이다.
