# Qwen3-VL-4B 기반 MMMU/MMMU-Pro 성능 향상을 위한 데이터셋 및 복합 데이터 증강 방법론 초안

## 1. 프로젝트 목적

본 팀 프로젝트의 목표는 **Qwen3-VL-4B-Instruct를 base model로 사용하고, 외부 멀티모달 학습 데이터를 fine-tuning하여 MMMU/MMMU-Pro benchmark 성능을 향상시키는 것**이다.

프로젝트의 기본 연구 흐름은 다음과 같다.

**Baseline Evaluation → Failure Analysis → Problem Definition → Training Data Design → Fine-tuning → Evaluation → Error Analysis → 반복 개선**

중요한 제약은 다음과 같다.

- Qwen3-VL-4B-Instruct를 base model로 사용한다.
- MMMU Validation은 baseline 측정, debugging, hyperparameter selection, quick evaluation에 사용한다.
- MMMU Test와 MMMU-Pro는 최종 평가에 사용한다.
- **MMMU/MMMU-Pro 자체의 문제, 이미지, 정답 및 이를 변형한 데이터는 학습에 사용하지 않는다.**
- 학습 데이터는 public/external dataset, self-generated synthetic data, third-party model을 통한 generation/labeling 등을 사용할 수 있다.
- 단순 prompt engineering, few-shot prompting, RAG, inference-time trick만으로는 fine-tuning solution으로 간주하지 않는다.
- 실제 model parameter가 업데이트되는 fine-tuning이 포함되어야 한다.

---

# 2. 핵심 연구 가설

일반적인 VLM fine-tuning은 다음과 같은 단방향 학습에 집중한다.

**Image + Question → Answer**

본 프로젝트에서는 이를 다음과 같이 확장한다.

**Image ↔ Structured Representation ↔ Question ↔ Reasoning ↔ Answer**

즉 하나의 원본 sample을 단순한 QA 1개로 사용하는 대신, 다음과 같은 다양한 task family로 변환한다.

1. Forward Reasoning
2. Inverse Generation
3. Reconstruction
4. Missing-Element Completion
5. Counterfactual Reasoning
6. Error Detection and Correction
7. Cross-Representation Conversion
8. Explanation Generation
9. Explanation Verification
10. Cycle-Consistency

최종 가설은 다음과 같다.

> 정방향 QA 데이터만 추가하는 것보다, 정방향 추론 + 역방향 생성 + 구조 복원 + 반사실 추론 + 오류 검증 + 표현 변환을 함께 사용하는 복합 증강이 Qwen3-VL-4B의 실제 multimodal reasoning 능력을 더 강하게 향상시키며, 그 결과 MMMU뿐 아니라 실제 시각 정보 활용을 더욱 강하게 요구하는 MMMU-Pro에서도 더 높은 일반화 성능을 얻을 수 있다.

---

# 3. 학습 데이터셋 후보

## 3.1 1차 Core Dataset Pool

### 3.1.1 ScienceQA

주요 목적:
- 물리
- 화학
- 생물
- 지구과학
- 일반 과학 reasoning
- 이미지 기반 과학 QA

활용:
- 과학 그림 + 질문 → 답
- 단계별 풀이 생성
- label 일부 삭제 후 복원
- 풀이 + 정답 → 새로운 diagram/problem 생성
- 조건 변경 → counterfactual 문제 생성
- 잘못된 설명 탐지

추천 이유:
- MMMU의 STEM 영역과 직접적으로 관련된 능력을 강화하기 좋다.
- 질문, 정답, 설명/지식 정보를 활용한 증강이 가능하다.

---

### 3.1.2 TQA (Textbook Question Answering)

주요 목적:
- 교과서 기반 science reasoning
- diagram understanding
- 설명문 + 그림 + 질문 통합 이해

활용:
- 교과서 설명 → diagram 생성
- diagram → 설명 복원
- label 제거 → label 복원
- 문제 + 풀이 → 그림 재구성
- diagram + 설명 → 새로운 질문 생성
- diagram → structured relation 추출

추천 이유:
- 단순 VQA보다 교과서형 복합 reasoning을 요구하여 MMMU 스타일과 잘 맞는다.

---

### 3.1.3 ChartQA

주요 목적:
- chart/graph 이해
- 수치 추론
- 시각적 비교
- multi-step numerical reasoning

활용:
- Chart → Answer
- Chart → Table
- Table → Chart
- Chart → 자연어 설명
- 특정 수치 변경 → 새로운 Chart 생성
- 답 + 질문 → 해당 조건을 만족하는 Chart 생성
- 잘못된 bar/axis/legend 탐지
- 여러 series 간 비율/차이/순위 추론

추천 이유:
- MMMU-Pro의 실제 visual information 활용 능력을 강화하는 데 특히 적합하다.

---

### 3.1.4 Geo170K

주요 목적:
- geometry diagram understanding
- geometric relation extraction
- 수학적 시각 추론

활용:
- 도형 → 답
- 답 + 풀이 → 도형 생성
- 도형 → 좌표
- 좌표 → 도형
- 일부 선/각도 삭제 → 복원
- 조건 변경 → 새로운 답 계산
- 그림과 수식 간 상호변환

추천 이유:
- 수학뿐 아니라 spatial/diagram reasoning 능력 강화에 유리하다.

---

### 3.1.5 MAVIS / MMathCoT 계열

주요 목적:
- multimodal mathematics
- 단계별 수학 추론
- geometry/diagram/math CoT

활용:
- 수학 diagram → 풀이
- 풀이 → diagram
- 식 → graph
- graph → 관계식
- 잘못된 풀이 단계 탐지
- 조건 변경 후 답 재계산
- hard sample 선별을 통한 difficulty-aware training

추천 이유:
- 수학 reasoning과 시각 reasoning을 동시에 보강할 수 있다.

주의:
- 대규모 mixture dataset은 provenance 및 MMMU/MMMU-Pro contamination 여부를 별도로 검사한다.
- 전체를 무조건 사용하지 않고 고품질 subset을 선별한다.

---

### 3.1.6 AI2D / AI2D-Caption

주요 목적:
- scientific diagram 이해
- entity/label/relationship understanding
- diagram captioning

활용:
- Diagram ↔ Entity JSON ↔ Caption
- label 제거 → 복원
- 화살표 제거 → 관계 복원
- JSON → diagram 생성
- diagram → structured representation 생성
- 잘못된 label 위치 탐지

추천 이유:
- Image ↔ Structured Representation 방식의 증강에 매우 적합하다.

---

### 3.1.7 DocVQA

주요 목적:
- OCR
- document understanding
- layout reasoning
- table/document QA

활용:
- 문서 이미지 → OCR
- OCR → 질문 답변
- 표 → structured table/JSON
- text 일부 가림 → 복원
- 잘못된 OCR 결과 → 이미지로 수정
- 문서 layout → reading order 복원
- figure-caption matching
- multi-region information reasoning

추천 이유:
- MMMU-Pro에서 중요한 OCR + visual understanding + reasoning 능력을 강화할 수 있다.

---

## 3.2 2차 High-Difficulty Dataset Pool

### 3.2.1 SPIQA

주요 목적:
- scientific paper figure/table 이해
- 고난도 과학 reasoning
- figure-caption-context 통합 이해

활용:
- 논문 figure + caption + context → QA
- figure → structured relation 추출
- caption 일부 제거 → 복원
- 표/그래프 → 설명 생성
- 잘못된 scientific interpretation 탐지

추천 이유:
- 대학 수준에 가까운 고난도 scientific visual reasoning을 제공한다.

---

### 3.2.2 VisualWebInstruct

주요 목적:
- math
- physics
- chemistry
- finance
- engineering
- broad visual reasoning

활용:
- 고난도 visual QA
- reasoning trace augmentation
- domain-balanced fine-tuning
- hard-negative 생성
- cross-domain reasoning

주의:
- 최신 대형 mixture 또는 web-derived dataset은 반드시 contamination audit을 수행한다.
- source URL, exact/near duplicate, image similarity 등을 검사한다.

---

### 3.2.3 MMFineReason 계열

주요 목적:
- STEM visual reasoning
- complex diagram reasoning
- multimodal chain-of-thought
- difficult visual problem solving

활용:
- curated subset 사용
- difficulty-aware sampling
- reasoning verification
- hard example mining

주의:
- 최신 reasoning mixture이므로 provenance 확인과 MMMU/MMMU-Pro contamination 검사가 필수다.
- 전체를 사용하기보다 고품질/고난도 subset을 선별한다.

---

### 3.2.4 MMK12 등 실제 문제 기반 수학 데이터

주요 목적:
- real-world multimodal math
- 실제 교육 문제 distribution 보완

활용:
- synthetic geometry dataset과 실제 문제 distribution을 혼합
- 실제 시험/교재 형태의 시각 문제 대응력 향상

---

# 4. Humanities / Literature / History / Art 보완 데이터

수학·과학·그래프와 달리 humanities 분야는 직접적인 대형 multimodal training dataset이 상대적으로 부족할 수 있다.

따라서 다음 source를 이용하여 자체 synthetic instruction corpus를 구성한다.

- public-domain literature
- public-domain historical documents
- open museum collections
- artwork/image collections with clear metadata
- textbook-style humanities material
- copyright-safe document image
- OCR 가능한 public-domain book/page image

주요 생성 task:
- 문장 순서 복원
- 사건 순서 복원
- 원인/결과 관계 복원
- 인물 관계 추론
- 작품/시대/양식 reasoning
- 이미지 + 설명 대응
- OCR + 문맥 이해
- 잘못된 해석 탐지
- 요약 ↔ 원문 관계 학습

---

# 5. 공통 복합 데이터 증강 프레임워크

각 원본 sample을 다음 task family로 확장한다.

## T1. Forward Reasoning

입력:
- 이미지
- 질문

출력:
- 정답
- 단계별 풀이 또는 시각 정보 기반 근거

목적:
- 기존 VQA 능력 강화
- baseline task 유지

---

## T2. Inverse Generation

입력:
- 정답
- 풀이
- 일부 조건

출력:
- 해당 답과 풀이가 성립하는 문제/이미지/diagram/구조

예:
- 합성저항 6Ω이라는 답과 풀이 → 해당 조건을 만족하는 회로 생성
- 삼각형의 답과 풀이 → 조건을 만족하는 도형 생성
- 표의 통계값과 답 → 해당 통계를 표현하는 chart 생성

목적:
- 결과에서 원인/구조를 역으로 복원하는 능력 강화

---

## T3. Reconstruction

입력:
- 일부가 손상되거나 섞인 데이터

출력:
- 원래 구조 복원

예:
- 회로 선 일부 제거 → 연결 복원
- 도형 label 제거 → label 복원
- chart axis label 제거 → 복원
- 문학 문장 순서 shuffle → 원래 순서 복원
- document 일부 OCR token 제거 → 복원

---

## T4. Missing-Element Completion

입력:
- 일부 parameter/label/부품/문장 제거

출력:
- 누락된 요소 추론

예:
- 전체 합성저항과 나머지 resistor 값을 주고 R3 추론
- 삼각형 두 변/각을 주고 나머지 값 계산
- 과학 diagram에서 특정 organ label 복원

---

## T5. Counterfactual Reasoning

입력:
- 원본 sample
- 조건 하나 변경

출력:
- 변경된 결과

예:
- 저항값 변경 → 합성저항 재계산
- 그래프 특정 값 +20% → 순위/비율 재계산
- 물리 초기속도 변경 → 궤적 변경
- 기하 각도 변경 → 변 길이 재계산

목적:
- 단순 암기보다 관계와 causal dependency 학습

---

## T6. Error Detection and Correction

입력:
- 고의로 일부가 틀린 이미지/풀이/label/계산

출력:
- 잘못된 위치
- 오류 이유
- 수정된 결과

예:
- 회로 직렬/병렬 관계 하나 잘못 설명
- chart maximum을 second maximum으로 잘못 선택
- 기하 각도 하나 잘못 표시
- 생물 label 두 개 교환
- 문학 사건 순서 두 개 뒤집기

---

## T7. Cross-Representation Conversion

하나의 의미를 여러 representation으로 변환한다.

예:
- Circuit Image ↔ Netlist ↔ Graph/JSON
- Geometry Figure ↔ Coordinates ↔ Equation
- Chart ↔ Table ↔ JSON ↔ Text
- Chemical Structure ↔ SMILES ↔ Molecular Graph
- Scientific Diagram ↔ Entity/Relation JSON
- Document Image ↔ OCR Text ↔ Structured Layout

목적:
- modality 사이의 정렬 강화
- underlying structure 학습

---

## T8. Explanation Generation

정답만 존재하는 문제에 대해:
- 단계별 풀이
- 이미지 근거
- 수식 과정
- 관계 설명

등을 생성한다.

주의:
- 가능한 분야에서는 symbolic/numeric verifier로 풀이와 정답의 consistency를 검증한다.

---

## T9. Explanation Verification

입력:
- 문제
- 이미지
- 제시된 풀이

출력:
- 풀이가 맞는지 판정
- 오류 단계 탐지
- 수정된 풀이

목적:
- reasoning verifier 능력 강화
- hallucinated reasoning 억제

---

## T10. Cycle-Consistency

여러 표현을 왕복 변환한 후 의미가 유지되는지 검사한다.

예:
- Circuit Image → Netlist → Image' → Netlist'
- Chart → Table → Chart' → Table'
- Geometry Figure → Coordinates → Figure' → Coordinates'
- Original Paragraph → Shuffle → Reconstruction

통과한 sample을 우선적으로 training pool에 포함한다.

---

# 6. 분야별 구체적 증강

## 6.1 회로 / 전기전자

원본:
회로 이미지 + 질문 + 정답 + 풀이

증강:

1. 회로 이미지 → 합성저항 계산
2. 정답 + 풀이 → 회로 topology 생성
3. 회로 → netlist
4. netlist → 회로 이미지
5. 특정 resistor 제거 → 값 추론
6. 연결선 일부 제거 → 회로 복원
7. resistor 값 변경 → 새 정답 계산
8. 잘못된 series/parallel 판단 → 오류 수정
9. 동일한 답을 가지는 새로운 topology 생성
10. 회로 설명 → graph/JSON → SVG/TikZ rendering

권장 pipeline:

**Answer/Reasoning → Circuit Graph/Netlist → Solver Verification → Renderer → Training Image**

핵심:
이미지 생성 모델이 자유롭게 회로 이미지를 그리게 하지 않고, 먼저 검증 가능한 구조를 생성한다.

---

## 6.2 수학 / 기하

1. 도형 → 정답 계산
2. 답 + 풀이 → 도형 생성
3. 도형 → 좌표
4. 좌표 → 도형
5. 수식 → graph
6. graph → 식 추론
7. 일부 길이/각도 삭제 → 복원
8. 조건 변경 → 정답 재계산
9. 잘못된 도형 조건 탐지
10. 같은 theorem을 다른 형태의 도형으로 재생성

검증:
- SymPy
- 좌표기하
- 수치계산
- geometric constraints

---

## 6.3 그래프 / 차트 / 표

1. Chart → Answer
2. Chart → Table
3. Table → Chart
4. Chart → Text description
5. 특정 값 변경 → 새로운 chart
6. axis/legend 일부 삭제 → 복원
7. 잘못된 bar height/label 탐지
8. 답 + 질문 → 조건을 만족하는 chart 생성
9. 여러 series의 ratio/difference/rank 계산
10. chart type 변환(bar ↔ line ↔ table)

Canonical source:
항상 원본 table/JSON을 유지하여 모든 시각화와 정답을 검증한다.

---

## 6.4 물리학

1. Diagram + 조건 → 물리량 계산
2. 목표 결과 + 일부 조건 → 초기조건 역산
3. trajectory graph → 초기속도/각도 추론
4. 조건 → trajectory graph 생성
5. force vector 일부 삭제 → 복원
6. vector direction 오류 탐지
7. parameter 변경 → 결과 재계산
8. 풀이 → 해당 풀이가 성립하는 diagram 생성
9. diagram → governing equation
10. equation → 물리 상황/그림 생성

가능하면 simulator 또는 수치계산으로 검증한다.

---

## 6.5 화학

1. Chemical structure → 이름/분자식
2. 이름/SMILES → 구조 이미지
3. 구조 이미지 → SMILES
4. reaction diagram → 생성물/reagent
5. 반응물 + 생성물 → 누락 reagent 추론
6. 잘못된 구조 탐지
7. functional group 제거 → 복원
8. 구조 일부 변경 → 반응성 변화 reasoning
9. mechanism → reaction diagram
10. reaction diagram → 단계별 mechanism

검증:
- RDKit
- molecular graph validation

---

## 6.6 생물학 / 과학 Diagram

1. 이미지 → label
2. label → 위치
3. label 제거 → 복원
4. label swap → 오류 수정
5. text description → diagram
6. diagram → structured JSON
7. 구조 일부 가리기 → missing part
8. 기능 설명 → organ/structure 선택
9. 이미지와 설명 간 모순 탐지
10. 동일 구조를 다른 표현으로 재구성

중간 표현:
- object
- parts
- position
- relation
- function

---

## 6.7 문학 / 언어 / 인문학

1. Sentence Order Reconstruction
2. Missing Sentence
3. Intruder Detection
4. Paragraph Reconstruction
5. Event Ordering
6. Dialogue Reconstruction
7. Coreference Resolution
8. Cause-Effect Reconstruction
9. Summary ↔ Passage
10. Question Generation
11. Wrong Interpretation Detection
12. Multi-Passage Reasoning

예:
원문 문장 A-B-C-D를 C-A-D-B 순으로 섞은 뒤 원래 순서 복원.

멀티모달 확장:
책 페이지/고문서 이미지 → OCR → 문장 순서 복원 → 의미 이해 → QA

주의:
copyright-safe 또는 public-domain source를 우선 사용한다.

---

## 6.8 역사 / 사회 / 예술

1. 지도 이미지 → 위치/관계 추론
2. 위치 설명 → map reconstruction
3. 연표 shuffle → 사건 순서 복원
4. 작품 이미지 → 시대/양식/특징
5. 작품 설명 → 적절한 visual feature 선택
6. 역사적 설명과 도표 간 모순 탐지
7. chronological multi-hop reasoning
8. 이미지 label 제거 → 복원
9. 표/통계 → 사회과학적 해석
10. 자료 + 답 → 새로운 문제 생성

---

## 6.9 OCR / 문서 이해

1. Document image → OCR
2. 일부 문자/숫자 제거 → 복원
3. 표 → structured table
4. structured table → document image
5. 잘못된 OCR → 이미지로 수정
6. 위치 정보 + 내용 결합 QA
7. 서로 떨어진 영역의 multi-hop reasoning
8. layout reconstruction
9. caption ↔ figure matching
10. document image → summary/question/answer

---

# 7. Canonical Representation 기반 생성

Synthetic augmentation에서 이미지 생성 모델이 직접 자유롭게 이미지를 생성하게 하면 label 오류와 hallucination이 발생할 수 있다.

따라서 가능한 분야에서는 다음 pipeline을 사용한다.

**Semantic Condition → Canonical Representation → Automatic Verification → Renderer → Image**

예:

- 회로 → Netlist/Graph
- 기하 → Coordinates/Geometry JSON
- 그래프 → Table/JSON
- 화학 → SMILES/Molecular Graph
- 생물 Diagram → Entity/Relation JSON
- 문학 → Sentence/Event Graph

이 방식을 통해 생성 이미지와 정답 사이의 consistency를 유지한다.

---

# 8. 자동 품질 검증

Synthetic data의 핵심은 생성량이 아니라 correctness이다.

분야별 verifier:

- 수학: SymPy / numeric solver
- 회로: SPICE / 자체 circuit solver
- 그래프: source table과 rendering data 비교
- 기하: coordinate geometry
- 화학: RDKit
- 문학: source ordering / event graph comparison
- OCR: normalized string comparison
- 표: JSON/table consistency

pipeline:

**Generated Sample → Verifier → FAIL 제거 / PASS 유지**

---

# 9. Difficulty-Aware Augmentation

난이도는 다음과 같이 구분한다.

### Level 1
이미지에서 직접 읽으면 답할 수 있음

### Level 2
한 번의 계산 또는 관계 이해 필요

### Level 3
2~3단계 visual reasoning 필요

### Level 4
여러 영역/객체/정보를 결합

### Level 5
Inverse problem, reconstruction, counterfactual 등 복합 추론 필요

쉬운 문제만 대량 생성하지 않고 여러 난이도를 균형 있게 구성한다.

---

# 10. Hard-Negative / Verification Dataset

일부러 거의 맞지만 한 부분만 틀린 sample을 생성한다.

예:
- 회로 연결 관계 한 곳만 오류
- 그래프 최고값을 두 번째 값으로 선택
- 기하 angle label 하나 오류
- 생물 label 두 개 swap
- 문학 사건 순서 두 개 swap
- OCR 숫자 한 자리 오류

task:
- 정확한가?
- 어디가 틀렸는가?
- 왜 틀렸는가?
- 수정하라.

목적:
세밀한 visual verification과 reasoning consistency를 강화한다.

---

# 11. MMMU/MMMU-Pro Contamination 방지

최신 multimodal mixture dataset은 benchmark data가 섞였을 가능성이 있으므로 데이터 사용 전에 contamination audit을 수행한다.

검사:

1. dataset card / 논문 / source provenance 확인
2. MMMU/MMMU-Pro 사용 여부 확인
3. question normalized exact match
4. n-gram / MinHash 기반 near-duplicate 검사
5. image hash / perceptual hash 검사
6. 필요 시 text/image embedding similarity 검사
7. source URL 추적
8. 의심 sample 제거
9. 생성 dataset도 원본 lineage 기록
10. 사용한 filtering rule을 README/report에 기록

원칙:
- MMMU/MMMU-Pro 문제/이미지/답의 직접 사용 금지
- MMMU 문제를 paraphrase/수치변경/이미지변형하여 training sample로 만드는 것도 금지
- provenance가 지나치게 불명확한 mixture는 피하거나 강한 filtering 후 제한적으로 사용

---

# 12. 초기 권장 Training Pool

초기 후보 예시:

- ScienceQA: 8K
- TQA: 6K
- AI2D: 4K
- ChartQA: 8K
- Geo170K: 8K
- MAVIS/MMathCoT 계열: 10K
- DocVQA: 5K
- SPIQA: 5K
- Humanities synthetic: 5K

원본 약 59K 수준에서 시작할 수 있다.

이후:

**59K Originals → 150K~250K Augmentation Candidates → Verification → Deduplication → Difficulty Filtering → Contamination Filtering → Final 60K~100K**

정도로 정제하는 것을 목표로 한다.

RTX 5060 8GB 환경에서는 처음부터 전체를 학습하기보다 10K~20K balanced subset으로 pilot experiment를 수행한 뒤 점차 확대한다.

---

# 13. 권장 Task Mix

초기 예시:

- 30% Forward Visual Reasoning
- 20% Inverse Generation / Reconstruction
- 15% Counterfactual Reasoning
- 15% Cross-Representation Conversion
- 10% Error Detection / Verification
- 10% OCR / Document Reconstruction

실제 비율은 MMMU Validation baseline 결과에 따라 조정한다.

예:
- graph/diagram 문제가 약함 → Chart/Diagram 비중 증가
- OCR 문제가 약함 → DocVQA/OCR augmentation 증가
- mathematics가 약함 → Geo170K/MMathCoT 비중 증가
- humanities가 약함 → literature/history synthetic corpus 확대

---

# 14. Ablation Study 설계

### Exp 0
원본 Qwen3-VL-4B baseline

### Exp 1
일반 QA fine-tuning

### Exp 2
QA + 단계별 reasoning

### Exp 3
QA + inverse generation

### Exp 4
QA + inverse + reconstruction

### Exp 5
+ counterfactual reasoning

### Exp 6
+ error detection

### Exp 7
+ cross-representation

### Exp 8
전체 복합 augmentation

각 실험은 동일한 MMMU Validation evaluation pipeline으로 비교한다.

최종적으로:
- MMMU Test
- MMMU-Pro

에서 일반화 성능을 평가한다.

---

# 15. Fine-Tuning 전략

하드웨어 제약을 고려할 때 초기 실험은 QLoRA 기반을 권장한다.

예시:

- Base: Qwen3-VL-4B-Instruct
- Base weights: 4-bit quantization
- Vision encoder: 초기에는 freeze
- LLM attention/MLP 일부: LoRA
- Batch size: 1
- Gradient accumulation 사용
- Gradient checkpointing 사용
- 작은 subset으로 pilot 진행
- 성능 향상 확인 후 학습 범위 확대

비교 가능 실험:
- LLM LoRA only
- LLM + multimodal projector
- 일부 vision block unfreeze
- task mix 변경
- dataset mix 변경
- 난이도 비율 변경

---

# 16. 전체 파이프라인

외부 데이터셋
↓
Source/License/Provenance 확인
↓
MMMU/MMMU-Pro Contamination 검사
↓
Canonical Representation 변환
↓
Forward / Inverse / Reconstruction / Counterfactual / Verification 생성
↓
자동 Verifier
↓
Quality Filtering
↓
Near-Duplicate 제거
↓
Difficulty-aware Sampling
↓
Balanced Training Set 구성
↓
Qwen3-VL-4B QLoRA Fine-tuning
↓
MMMU Validation 평가
↓
분야별/문제유형별 Error Analysis
↓
Dataset/Task Mix 수정
↓
반복 Fine-tuning
↓
MMMU Test + MMMU-Pro Final Evaluation

---

# 17. 프로젝트의 핵심 차별점

본 프로젝트는 단순히 학습 데이터의 양을 증가시키는 것이 목적이 아니다.

핵심 차별점은 다음과 같다.

1. 단방향 QA를 양방향/순환형 multimodal learning으로 확장
2. 풀이와 답에서 원래 diagram을 재구성하는 inverse reasoning
3. structured representation을 통한 정확한 synthetic image 생성
4. counterfactual reasoning을 통한 관계 이해 강화
5. hard-negative와 verification task를 통한 오류 검증 능력 강화
6. cycle-consistency를 이용한 데이터 품질 검사
7. difficulty-aware filtering으로 효율적인 학습 데이터 구성
8. 분야별 automatic verifier로 잘못된 synthetic label 제거
9. MMMU/MMMU-Pro contamination을 강하게 방지
10. ablation study를 통해 어떤 증강 방식이 실제 benchmark 성능 향상에 기여했는지 검증

---

# 18. 제안 가능한 연구 제목

### 영문
**Bidirectional Structure-Preserving Multimodal Data Augmentation for Qwen3-VL-4B**

또는

**Improving MMMU/MMMU-Pro Generalization via Inverse Reconstruction and Verifiable Multimodal Augmentation**

또는

**Cycle-Consistent and Verifiable Multimodal Fine-Tuning for Qwen3-VL-4B**

### 한글
**Qwen3-VL-4B의 MMMU/MMMU-Pro 성능 향상을 위한 양방향 구조 보존형 멀티모달 데이터 증강**

---

# 19. 최종 연구 질문

본 프로젝트에서 최종적으로 검증할 질문은 다음과 같다.

> 일반적인 Image + Question → Answer 형태의 fine-tuning보다, Image ↔ Structured Representation ↔ Question ↔ Reasoning ↔ Answer 형태의 양방향 학습과 inverse reconstruction, counterfactual reasoning, error verification, cross-representation conversion을 함께 적용한 복합적 데이터 증강이 Qwen3-VL-4B의 실제 multimodal reasoning 능력과 MMMU/MMMU-Pro 성능을 더 효과적으로 향상시키는가?

이 질문을 중심으로 baseline, dataset selection, augmentation, fine-tuning, ablation, final evaluation을 구성한다.
