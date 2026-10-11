### 1. 현재 모델 평가가 제대로 이루어졌는가
- Qwen3-VL-4B-Instruct / MMMU validation 900문항 평가
- 현재 점수: **558/900 = 62.00%**
- 자체 **규칙 기반 파서**로 최종 답을 추출해 평가
- 장점: 재현성이 높고 API 의존성이 없음
- 한계: `$2,960 unfavorable`처럼 의미상 맞는 답도 표현 차이 때문에 오답 처리될 수 있음
- 향후 **LLM API 또는 Gemma 계열 로컬 LLM을 보조 Judge로 활용**하여 파서 결과와 교차검증하는 방안 고려

### 2. 모델의 실패 유형 분석

**70점 기준 과목 분류**\
※ 70점은 공식 기준이 아니라 우리 분석을 위한 기준선

| 구분 | 과목 수 | 주요 과목 |
|---|---:|---|
| **70% 이상** | **12/30** | Art Theory, Medical, Design, Economics, Finance, History, Literature, Marketing, Physics, Psychology, Public Health 등 |
| **70% 미만** | **18/30** | Accounting, Agriculture, Biology, Chemistry, CS, Electronics, Materials, Mechanical Engineering, Music 등 |

특히 **Music 20%, Diagnostics 36.67%, Electronics/Materials 40%, Architecture/Mechanical 43.33%** 등이 낮음.

**저득점 문제를 4개 실패군으로 정리**

| 실패군 | 핵심 문제 | 학습 후보 |
|---|---|---|
| A. 전문 시각·기호 표현 | 악보/수식/전문기호의 위치·연결 해석 | MUSCIMA++, Geo170K |
| B. 도식·공간 관계 | 객체·화살표·연결·방향 관계 보존 실패 | AI2D, CLEVR, GQA |
| C. 과학적 시각 판독 | 관측 특징을 전문 개념·규칙에 연결하는 과정 | AI2D, ScienceQA |
| D. 다단계 수량 추론 | 표/수치 선택→계산→중간상태 유지→답 도출 | TabMWP, MMFineReason, FinQA |

→ 공통적으로 **단순 지식 부족보다는 시각 구조 추출·보존 → 의미 연결 → 여러 규칙 적용 → 중간 상태 유지**가 동시에 필요할 때 성능이 떨어지는 것으로 가설 설정.

### 3. 개선안이 실패 분석에 근거했는가
단순히 이미지 QA를 더 학습시키는 방식보다는

**Image → Visual Evidence/Structure → Domain Rule → Reasoning → Answer**

형태의 supervision을 학습시키는 방향.

예:
- 기호/객체/관계 추출
- 구조를 전문 지식과 연결
- 필요한 규칙 선택
- 중간 계산/상태 유지
- 최종 답 검증

즉 **정답만 학습하는 SFT와 구조·근거·추론까지 포함한 SFT를 비교**하는 식으로 ablation 진행.

### 4. Related Work / Dataset

간단히 3~4개 핵심 관련 연구를 소개.

- **MMFineReason**\
  시각 정보 추출 → 문제 설정 → 풀이 → 검증 형태의 multimodal reasoning 학습. Qwen3-VL-4B에서 SFT 성능 향상을 보고.\
  - Paper: https://arxiv.org/abs/2601.21821\
  - Dataset: https://huggingface.co/datasets/OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking

- **G-LLaVA / Geo170K**\
  기하학 문제의 점·선·각 및 관계 구조를 명시적으로 학습하여 시각적 구조와 추론을 연결.\
  - Paper: https://arxiv.org/abs/2312.11370\
  - GitHub / Geo170K: https://github.com/pipilurj/G-LLaVA

- **LLaVA-CoT**\
  Summary → Visual Interpretation → Reasoning → Conclusion 형태의 단계적 multimodal reasoning 학습.\
  - Paper: https://arxiv.org/abs/2411.10440\
  - GitHub: https://github.com/PKU-YuanGroup/LLaVA-CoT\
  - Dataset: https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k

- **DocVAL**\
  문서 이미지에서 정답뿐 아니라 시각적 근거 위치와 reasoning을 함께 supervision하는 방식.\
  - Paper: https://arxiv.org/abs/2511.22521\
  - GitHub: https://github.com/ahmad-shirazi/DocVAL

**Dataset 후보**

- **Geo170K** — https://github.com/pipilurj/G-LLaVA
- **AI2D (Allen Institute for AI)** — https://prior.allenai.org/projects/diagram-understanding
- **ScienceQA** — https://github.com/lupantech/ScienceQA
- **TabMWP** — https://github.com/lupantech/PromptPG
- **MUSCIMA++** — https://github.com/OMR-Research/muscima-pp
- **CLEVR** — https://cs.stanford.edu/people/jcjohns/clevr/
- **GQA** — https://cs.stanford.edu/people/dorarad/gqa/
- **FinQA** — https://github.com/czyssrs/FinQA

### 5. HW 인프라
- Baseline: **RTX 3090 24GB**
- 추가 실험: **RTX 4090 24GB / A40 48GB** 등 활용 가능
- 4B 전체 Full FT보다는 우선 **LoRA/PEFT 기반 SFT**
- 대규모 dataset 전체보다는 실패군별 소규모 실험 → 효과 확인 후 확대

### 6. 일정 / Milestone
| 단계 | 내용 |
|---|---|
| 1 | Baseline + 실패 유형 분석 **완료** |
| 2 | Dataset 선정·오염/라이선스 확인 |
| 3 | 실패군별 데이터 전처리 및 SFT/LoRA |
| 4 | 중간 MMMU 평가 + ablation |
| 5 | 효과 좋은 데이터 조합 확대 학습 |
| 6 | MMMU/MMMU-Pro 최종 평가 + 실패 사례 분석 |

### 7. Citation
PPT 하단에는 `[1]`, `[2]` 식으로 표시하고 마지막에 **References 슬라이드 1장** 만들면 될 듯.

필수로 넣을 것:
1. Qwen3-VL Technical Report
2. MMMU / MMMU-Pro
3. MMFineReason
4. G-LLaVA
5. LLaVA-CoT
6. DocVAL
7. 실제 사용 Dataset의 원 논문/공식 repository
