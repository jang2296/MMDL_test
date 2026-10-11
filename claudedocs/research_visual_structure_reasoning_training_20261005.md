# 전문 시각 구조와 복합 추론을 함께 학습하는 방법: Qwen3-VL-4B 관련 연구

확인일: 2026-10-05 KST. 내부 연구 기록이며 GitHub 게시본이 아니다. 논문·저자 저장소·저자 데이터 카드만 사용했다. 데이터 취득, 새 학습, 유료 API 호출은 하지 않았다. 실행 중인 네 학습 조건과 V8 baseline은 변경하지 않았다.

## 요약

관련 논문이 있다. 특히 MMFineReason은 Qwen3-VL-4B-Instruct를 출발점으로 사용한 직접적인 학습 근거다. 다른 연구들도 시각적 근거와 관계를 읽고, 그 정보를 규칙·계산·최종답에 연결하는 supervision을 검토할 이유를 제공한다. 다만 논문 성과는 저자들의 데이터·장비·평가 조건에서의 결과다. 우리 3090 학습과 V8 점수 상승을 보장하지 않는다.

확신 수준은 다음처럼 구분한다. 논문에 명시된 모델·방법·표의 수치는 높음, 우리 가설과 방법의 관련성은 중간, 우리 설정에서의 효과·모듈별 인과 책임·데이터 오염 부재는 미검증이다.

## 1. 확인한 학습 연구

| 연구 | 학습 데이터와 supervision | 확인한 결과와 적용 범위 |
|---|---|---|
| [MMFineReason, arXiv v1](https://arxiv.org/html/2601.21821v1) | 공개된 MMFineReason의 시각 근거를 포함하는 단계별 풀이. 정보 추출→문제 설정→풀이→검증 | Table 4: Qwen3-VL-4B 원본 Instruct MMMU-val 67.4, SFT 69.3, 추가 RL 69.6. 우리 평가의 재현값은 아니다. |
| [G-LLaVA, arXiv v2](https://arxiv.org/html/2312.11370v2) | Geo170K: 약 60K 시각 정렬 자료와 110K 이상 문제·풀이. 점·선·각과 관계를 학습한 뒤 문제 풀이 학습 | Table 13: 7B GeoQA 정확도는 정렬 단계 제외 62.8%, 포함 64.2%. 기하 과제의 통제 결과이며 MMMU 전체 개선 수치가 아니다. |
| [LLaVA-CoT, arXiv v6](https://arxiv.org/html/2411.10440v6) | LLaVA-CoT-100k: 과제 요약→시각 설명→추론→결론. Llama-3.2-11B-Vision 학습 | §5.2의 search 미사용 MMStar는 원본 49.8→57.6. 전체 평균 표에는 test-time retracing 표기 충돌이 있어 SFT-only 효과로 인용하지 않는다. |
| [DocVAL, arXiv v2](https://arxiv.org/html/2511.22521v2) | 문서 이미지·질문→근거 위치·풀이·답. 검증된 teacher trace로 SFT 후 오류 피드백을 이용한 학습 | Table 1: Gemma3-4B DocVQA ANLS 78.6→88.7. 문서 답변 지표이며 Qwen의 MMMU 정확도가 아니다. |
| [Revisiting the Necessity of Lengthy CoT, arXiv v1](https://arxiv.org/html/2511.22586v1) | Qwen2.5-VL-7B의 미로 기반 SFT+RL. 언어 풀이·좌표 근거·시각 조작을 비교 | 장황한 풀이보다 필요한 시각 근거를 담은 간결한 경로의 일반화가 좋았다. 최종답만 있는 짧은 label과는 다르다. |

### 가장 직접적인 MMFineReason도 그대로 복사하면 안 되는 이유

§5.1과 Table 8/10은 학습 32K packing·3 epochs·global batch 32, 평가 greedy·32K·LLM judge를 제시한다. 우리 sampling/V8와 다르고 정확한 base revision은 확인되지 않았다. §5.5에서는 caption 추가와 해상도 확대의 이득이 과제별로 달랐다. §7.2는 전문 의료 영상 등을 제외한다. 따라서 모든 저득점 분야를 해결한 근거는 아니다. [실험·제외 조건](https://arxiv.org/html/2601.21821v1)

[저자 데이터](https://huggingface.co/datasets/OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking)는 1.8M 규모로 공개돼 있으나 전체를 다운로드·권리 검사·오염 감사하지 않았다. source/ID와 원본 질문을 따라 학습 적합성을 확인해야 한다. 재집계 데이터 이름에 MMMU가 없다는 이유만으로 독립적인 외부 자료라고 인증하지 않는다.

§3.3의 correctness filtering은 추출된 최종답의 일치를 검사한다. 따라서 모든 중간 시각 관계·상태·규칙이 정확하다고 인증한 데이터는 아니며, 별도의 구조 정답을 갖춘 supervision과 구분해야 한다. 논문의 n-gram 반복 제거도 benchmark 대조 오염 감사와는 다르다.

### 두 연구의 재현 범위 주의

LLaVA-o1은 LLaVA-CoT의 이전 명칭이지 별도 논문이 아니다. 최신 v6 Table 2 caption은 retracing을 포함한다고 쓰지만 Table 4와 §5.2의 설명은 분리가 일관되지 않는다. 이전 v2 평균 수치를 최신 논문의 순수 학습 효과처럼 섞지 않는다. 논문 학습은 8 H100 full-parameter SFT이며 3090 LoRA 재현 증거가 아니다. [저자 코드](https://github.com/PKU-YuanGroup/LLaVA-CoT), [저자 데이터](https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k)

DocVAL은 필터 없이 teacher 답을 복사하기보다 정답·근거 위치·형식을 검증하는 점이 중요하다. 논문의 실제 학습은 2 H100 full fine-tuning이다. 저장소의 3090 LoRA 안내를 논문의 실측 조건으로 바꾸어 쓰면 안 된다. 또한 논문 95K와 README의 생성 예시 2,363개는 같지 않으며, 동일 학습 묶음을 확보했다고 주장하지 않는다. [논문](https://arxiv.org/html/2511.22521v2), [저자 코드](https://github.com/ahmad-shirazi/DocVAL)

## 2. 데이터 후보와 네 실패 군의 연결

아래는 논문 결과로 보장된 개선안이 아니라 다음 후보를 정하기 위한 팀의 해석이다.

| 실패 군 | 우선 검토할 외부 자료 | supervision으로 필요한 정보와 한계 |
|---|---|---|
| A. 기호·전문 시각 표현 | Geo170K, 원천이 확인된 LLaVA-CoT 기하 자료; 음악은 별도의 MUSCIMA++ 검토 | 기호 종류·위치·연결을 전문 상태/규칙으로 연결. 기하 학습이 악보·회로 개선을 보장하지 않음. |
| B. 도식·공간 관계 | Geo170K, 원래 AI2D 관계 주석, 필요하면 CLEVR/GQA | QA만 아니라 객체·화살표·연결·방향을 보존. 일반 도형/자연 영상이 건축·기계 도면을 대체하지 않음. |
| C. 과학적 시각 판독 | 원천이 확인된 AI2D/ScienceQA 및 LLaVA-CoT 과학 자료 | 관측 특징→전문 개념→적용 규칙. 교과서 도식과 병리·검사 영상은 구분해야 하며 전문 의료 자료는 별도 조사 필요. |
| D. 다단계 수량 추론 | TabMWP, MMFineReason의 원천이 확인된 수학·과학 일부, 조건부 FinQA | 행·열/단위→관련 수치 선택→계산/중간 상태→최종답. FinQA는 원래 raster 이미지 학습 자료가 아님. |

데이터 접근 근거: [Geo170K 저자 저장소](https://github.com/pipilurj/G-LLaVA), [AI2D 원저자 페이지](https://prior.allenai.org/projects/diagram-understanding), [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/), [GQA 원논문](https://cs.stanford.edu/people/dorarad/gqa/gqaPaper.pdf), [ScienceQA](https://github.com/lupantech/ScienceQA), [TabMWP](https://github.com/lupantech/PromptPG), [FinQA](https://github.com/czyssrs/FinQA), [MUSCIMA++](https://github.com/OMR-Research/muscima-pp).

TabMWP에는 38,431개 표 문제와 이미지/텍스트 표·다단계 풀이가 있다. 원래 PromptPG는 prompt selection 연구이므로 해당 논문 점수를 Qwen fine-tuning의 성과로 말하지 않는다. [저자 데이터 설명](https://github.com/lupantech/PromptPG)

MUSCIMA++/AI2D/MedMNIST/FinQA의 주석·권리·적용 한계는 이미 확인한 [내부 후보 검토](research_diagnostic_external_candidates_20261003.md)를 재사용한다. 이 자료의 존재만으로 모든 출처가 학습 승인되거나 MMMU와 중복이 없다는 뜻은 아니다.

## 3. 우리 다음 실험으로 번역하면

다음은 팀 제안이며 현재 실행을 바꾸는 지시가 아니다.

1. 현재 네 10K 조건을 먼저 종료·회수·검증한다. 모든 arm이 같은 TRACE+NVW 노출을 사용하므로 부분 점수 하락을 TRACE 단독 원인으로 확정하지 않는다. 형식 적응과 실제 관계·규칙 학습을 분리한다.
2. 이후 원천·권리·오염 검사를 통과한 외부 문제로 학습 표본을 고정한다. 같은 이미지·문제·정답에 대해 최종답-only와 검증된 시각 근거/규칙/중간 상태+최종답을 비교한다. 원본 이미지는 두 조건 모두 유지한다.
3. teacher 풀이의 정답뿐 아니라 근거의 존재·연결·계산을 검사한다. 정답만 맞고 잘못된 관계를 설명한 자료는 학습용 과정 정답으로 인증하지 않는다. JSON 출력 형식 자체를 구조 학습 성공으로 취급하지 않는다.
4. 데이터·모듈·학습률·표본 순서·update 수·평가 조건을 맞춘다. target token 길이 차이와 처리 비용은 별도 보고한다. 간결하고 필요한 과정부터 시험하며 32K 장문을 무조건 요구하지 않는다.
5. 외부 독립 family holdout에서 근거/관계/상태/규칙/최종답을 각각 확인하고 MMMU는 고정 조건의 유지·전이 평가로만 사용한다. 실패 군별 개선·악화를 함께 보고한다. 기존 V8 558/900을 덮어쓰지 않는다.

짧은 합성 relation 학습을 크게 늘리는 것과 전문 이미지에서 시각 근거→규칙→풀이를 학습하는 것은 같은 실험이 아니다. 현재 NVW에도 구조·상태·규칙 target이 있으므로 “우리는 과정 정보가 전혀 없다”는 설명은 잘못이다. 다음에 검증할 것은 전문 표현에 맞는 내용, 정확성, end-to-end 연결과 유지 성능이다.

장비 제약 때문에 공개 대규모 묶음 전체·teacher 모델·RL rollout을 그대로 실행하지 않는다. 승인된 후속 소규모 SFT가 필요하며 LoRA/Merger/Vision 중 대상 선택은 현재 네 조건의 완료 결과와 함께 판단한다. 문헌만으로 Vision-only 또는 Merger-only가 원인이라고 확정할 수 없다.

## 4. 오염 방지와 현재 증거 경계

MMMU/MMMU-Pro의 모든 split, 응답, 전사, teacher 설명, 크롭, 번역, 재렌더링은 BENCHMARK_DERIVED이며 training_allowed=false다. benchmark는 제외 검사 신호로만 사용하고 학습 target으로 만들지 않는다. 외부 자료도 source/split/원본 ID·이미지와 내용의 exact/near overlap, 라이선스, family 분리를 확인해야 한다.

현재 근거 연결: [진단 보고서](Qwen3_VL_4B_Bottleneck_Diagnostic_Report_20261003.md), [FL/FM 지표 감사](../analysis_exports/g6_runtime_20261005/partial_FL_FM_metric_audit.json), [사전 고정 선택 규칙](../analysis_exports/g6_runtime_20261005/module_selection_preregistration.json), [진행 기록](../docs/PROJECT_STATUS.md).

결론: 복합 병목이라는 가설은 검증된 시각 근거와 전문 규칙을 연결하는 학습 후보를 정당화한다. 그 학습의 효과나 내부 모듈 책임을 이미 입증한 것은 아니다. 이번 작업은 문헌 조사·내부 기록이며 기존 RunPod 실행, GitHub, PPT, baseline 응답/파서/점수는 불변이다.
