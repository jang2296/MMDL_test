Qwen3-VL-4B Novel Visual World 통합 파인튜닝 연구·실행 계획서
시각 구조·관계·의미·규칙·중간 상태의 공동 학습과 전문 시각 문제로의 전이
문서 버전: 확장안 v2.0
작성 기준: 2026-10-03
대상 모델: Qwen/Qwen3-VL-4B-Instruct
기준 모델 revision: ebb281ec70b05090aa6165b016eac8ec08e71b17
최종 목표: 외부 데이터와 독립 합성 데이터를 이용한 실제 가중치 파인튜닝으로 MMMU 및 MMMU-Pro 성능 개선
문서 상태: 기존 보고서 검토·공개 자료 조사·구현 설계. 데이터 전수 다운로드, 생성기 구현, 학습, 성능 개선 검증은 아직 수행하지 않음
공개·학습 경계: 벤치마크 분석 정보가 포함된 내부 연구 문서. 문서 전체를 학습 데이터나 문제 생성 교사의 예시로 사용하지 않음
0. 핵심 결정과 기존 계획의 변경 사항
0.1 최종 연구 방향
본 프로젝트의 중심 방법론을 다음과 같이 확장한다.
Episode마다 새롭게 구성한 시각 기호 체계와 의미·관계 매핑, 검증 가능한 제한된 규칙 체계를 갖는 Novel Visual World를 생성한다. 모델이 관측 가능한 이미지와 예제에서 구조·관계·의미를 파악하고, 규칙을 선택·발견·조합하며 중간 상태를 추적하도록 학습한다. 이를 실제 전문 분야의 canonical representation 및 이미지 기반 문제와 공동 학습하여, 미학습 표현과 전문 시각 과제로의 전이를 검증한다.

여기서 “domain-independent”는 달성된 성질이 아니라 검증할 연구 목표다. 학습에 포함되지 않은 표현·규칙 조합·원출처·분야에서 개선이 확인되는 범위만 일반화 성과로 주장한다.
0.2 기존 계획에서 유지하는 것
기존 실행 계획의 다음 원칙은 그대로 유지한다.
기존 원칙	확장안의 처리
단일 Vision Encoder 결함보다 다단계 기능적 병목을 검토	유지
FL·FM·FV를 같은 F0에서 독립 시작	유지
주 모듈 비교에 동일한 이미지 기반 학습 데이터 사용	유지
최종 Merger와 세 DeepStack Merger를 함께 관리	유지
구조·규칙·중간 상태 supervision	유지하고 NVW에 적용
반사실·오류 수정·불충분 정보 과제	유지하고 생성 계약을 구체화
모든 MMMU/MMMU-Pro 및 파생 자료 학습 금지	유지
작은 학습 점검 → 10K pilot → 선택 구성 50K 확대	유지


이 원칙들은 직전 계획서에 이미 포함되어 있다. 이번 확장은 이를 다른 계획으로 교체하는 것이 아니라, NVW 데이터 엔진과 공개 데이터 연결부를 추가하는 것이다.    Qwen3_VL_4B_Finetuning_Research…
0.3 이번에 추가·수정하는 것
항목	확장 내용
공개 데이터 수집	실제 Hugging Face 저장소·배포 형식·필수 부속 파일을 구분
NVW 생성	세계·episode·질문·파생 view를 별도 데이터 단위로 정의
규칙 생성	무제한 임의 규칙 대신 타입과 범위가 정해진 DSL 사용
정답 검증	최종답뿐 아니라 의미 매핑·중간 상태의 식별 가능성도 검사
전문 분야 연결	표준 표기·새 표기·새 배치가 같은 의미 상태를 공유하는 bridge 구성
일반화 평가	새 기호·새 조합·새 topology·새 renderer·새 분야를 분리
데이터 ablation	ontology randomization과 canonical bridge의 효과를 구분
평가 우선순위	수업 목표에 맞춰 MMMU-Pro Standard 10-way를 필수 주 평가에 포함


마지막 항목은 이전 실행 계획서의 평가 우선순위를 수정하는 사항이다. 수업 자료는 대상 benchmark를 MMMU와 MMMU-Pro Standard 10-way로 명시한다. 따라서 Pro vision은 중요한 추가 분석이지만 Standard 10-way를 대신하지 않는다.     01_MMDL_Course_Introduction
1. 현재 이론의 근거와 이 연구가 검증할 내용
1.1 현재 관측에서 출발하는 가설
현재 보고서에서는 선택된 기초 지식 K1과 단순 규칙 K2의 최종답이 각각 30/30이었다. 그러나 DFA 상태 전이와 그래프 차수 등 중간 풀이 오류도 확인됐으므로, 이 결과를 전문지식이나 규칙 적용 능력의 완전성으로 해석하지 않는다.    Qwen3_VL_4B_Knowledge_Oracle_20…    Qwen3_VL_4B_Knowledge_Oracle_20…
이미지에서 추출한 구조에도 누락·잘못된 기호·연결 관계 오류가 관측되었다. 다만 이 출력은 전체 VLM을 통과한 결과이므로, 그 오류를 Vision Encoder 단독 오류로 환원할 수 없다.    Qwen3_VL_4B_Bottleneck_Diagnost…
이에 따라 다음의 기능 사슬을 연구 대상으로 삼는다.
이미지 관측
    ↓
객체·기호의 구별
    ↓
위치·연결·포함·순서 관계의 추출
    ↓
질문에 필요한 구조의 보존
    ↓
전문적 의미 또는 episode 내부 의미와 연결
    ↓
적용할 규칙의 선택·발견
    ↓
여러 규칙의 조합과 중간 상태 갱신
    ↓
검증 가능한 최종답

이 사슬은 기능을 구분하기 위한 모델이다. 각 단계를 신경망의 특정 블록 하나와 일대일로 대응시키는 가정은 하지 않는다.
1.2 연구 질문
RQ1 — 구조 보존:
정답만 학습하는 것보다 객체·관계·의미 상태를 함께 학습하는 것이 전문 시각 문제의 정확도를 높이는가?
RQ2 — 새로운 표기 체계:
Episode별 기호·의미 매핑을 바꾸는 학습이, 고정 기호 체계만 학습하는 것보다 새로운 표기와 문맥에 잘 적응하는가?
RQ3 — 전문 지식 연결:
NVW만 학습하는 것보다 NVW와 전문 분야 canonical bridge를 공동 학습할 때 실제 전문 시각 과제로의 전이가 커지는가?
RQ4 — 상태 유지:
검증된 중간 상태 supervision이 긴 규칙 조합, 조건 변경, 순차 연산에서 발생하는 오류를 줄이는가?
RQ5 — 업데이트 위치:
같은 학습 자료를 사용할 때 FL·FM·FV·FML 중 어떤 업데이트 구성이 실제 정확도·구조·상태 지표에서 유리한가?
1.3 범용성의 범위를 미리 제한한다
NVW는 객체·관계·기호·상태가 명확한 과제를 직접 겨냥한다. 따라서 우선 전이 대상은 악보, 회로, 그래프, 표, 차트, 도식, 기하 및 일부 과학 구조 문제로 정한다.
반면 병리 사진, 식물 질환 사진, 작품 감별처럼 미세한 질감·형태·전문 시각 경험이 중요한 과제에서는 별도의 실제 이미지 학습이 필요할 수 있다. 기호 추론 학습이 이런 영역까지 자동으로 해결한다는 가정은 두지 않는다. 기존 계획서 역시 이 한계를 명시한다.    Qwen3_VL_4B_Finetuning_Research…
2. Novel Visual World의 정확한 정의
2.1 “문제마다 모든 것을 무작위로”가 아니라 “episode 안에서는 일관된 세계”
한 세계를 다음 요소로 정의한다.
World W
 ├─ 시각 기호 집합 Σ
 ├─ 기호 → 역할·속성 매핑 μ
 ├─ 선·표식 → 관계 타입 매핑 ρ
 ├─ 허용 규칙 문법 H
 ├─ 선택된 규칙 또는 프로그램 Π
 ├─ 초기 상태 S0
 ├─ 관측 예제 D
 ├─ 질의 Q
 └─ 렌더링 설정 η

핵심 원칙은 다음과 같다.
세계 사이에서는 달라져도, 같은 episode 안에서는 매핑과 규칙이 유지되어야 한다.
한 이미지에서는 같은 기호가 source이고 다음 support 이미지에서는 아무 설명 없이 sink로 바뀐다면, 그것은 어려운 문제라기보다 잘못 구성된 데이터다.
2.2 데이터 단위를 구분한다
단위	정의
World family	같은 규칙 문법·의미 구조·생성 계보를 공유하는 세계군
Episode	하나의 일관된 매핑과 규칙 아래 제공되는 support와 query 묶음
Root problem	특정 초기 상태와 질문으로 정의되는 독립 문제
View	같은 root를 구조 추출·직접 QA·오류 수정 등으로 바꾼 학습 레코드
Render variant	같은 의미 상태를 다른 배치·선·기호 모양으로 그린 이미지
Counterfactual child	의미 있는 조건을 바꾸고 정답을 다시 계산한 파생 문제


학습 레코드 10만 개가 곧 독립 세계 10만 개라는 뜻은 아니다.
2.3 무작위화할 것과 유지할 것
요소	기본 정책
기호의 모양	Episode 또는 alphabet family 단위로 변경
기호와 의미의 연결	Episode마다 새로 배정 가능
선·화살표·표식의 코드	타입을 지키며 변경 가능
객체 배치·색·선 굵기	의미 보존 여부를 확인한 뒤 변경
규칙 조합	허용 DSL 안에서 변경
상태 전이 길이	난도별로 변경
실제 물리·음악·회계 법칙	전문 분야 bridge에서는 유지
이미지의 실제 기하 관계	실제 좌표에 맞게 기록


예를 들어 위에 있는 객체를 ABOVE라고 기록했다면, episode가 바뀌었다고 그 실제 기하 관계를 BELOW로 바꾸지 않는다. 바꿀 수 있는 것은 “새로운 표식 X가 ABOVE를 뜻한다”는 코드 매핑이다.
실제 회로의 열린 스위치와 닫힌 스위치 의미를 임의로 뒤집어 전문 지식 학습에 섞지 않는다. 다른 규칙을 쓰는 세계는 명시적인 가상 신호 처리 세계로 분리한다.
3. NVW의 세 가지 핵심 트랙
3.1 NVW-K: 새 표기와 기존 지식의 연결
목표: 낯선 표기를 익숙한 전문 의미에 연결한다.
새 기호와 legend
    ↓
기호의 전문 역할 식별
    ↓
이미지에서 구조 추출
    ↓
기존 전문 규칙 적용
    ↓
답

예를 들어 저항·전원·스위치를 새 아이콘으로 표시하되, legend에서 의미를 알 수 있게 하고 실제 회로 법칙은 유지한다.
이 트랙은 “새로운 규칙을 발명하는 능력”보다 표현이 달라져도 기존 지식을 올바른 변수와 관계에 연결하는 능력을 겨냥한다.
3.2 NVW-I: 예제에서 규칙 발견
목표: 제한된 가설 집합 안에서 support 예제와 일치하는 규칙을 찾는다.
입력–출력 support 이미지들
    ↓
가능한 규칙 후보 축소
    ↓
새 query에 적용
    ↓
식별 가능한 답

초기에는 boolean 변환, 순열, 제한된 집합 연산, 격자 이동, 유한 상태 전이 등 완전 탐색이 가능한 규칙군을 사용한다.
3.3 NVW-S: 관계 구조 위의 상태 추적
목표: 같은 객체의 역할과 값을 여러 단계 동안 유지한다.
이미지에서 그래프·순서 추출
    ↓
초기 상태 확인
    ↓
규칙 1 적용
    ↓
중간 상태 S1
    ↓
규칙 2 적용
    ↓
중간 상태 S2
    ↓
최종 질문

객체 이름, 기호 종류, 현재 상태를 서로 혼동하지 않도록 분리된 supervision을 제공한다.
3.4 불충분 정보와 오류 수정은 모든 트랙에 적용한다
별도의 네 번째 세계를 만드는 대신 각 트랙에 다음 task flag를 붙인다.
answerable
insufficient_information
counterfactual_sensitive
counterfactual_invariant
corrupted_structure_repair
corrupted_state_repair

정보가 부족한 문제에 생성기가 내부적으로 선택한 답 하나를 무조건 정답으로 붙이지 않는다.
4. 공개 데이터 탐색 결과와 채택 전략
4.1 확인한 후보와 실제 역할
아래 수치는 공개 카드의 배포 설명이다. 이번 작업에서 파일을 전수 내려받아 직접 재계수한 값은 아니다.
데이터	확인한 배포·구성	본 프로젝트의 역할	초기 결정
maveryn/trace	Train 64K, validation 2K. 1,000개 task, 277개 scene grammar, 11개 domain	검증 가능한 다분야 합성과 생성기 참고	우선 조사·선별
HuggingFaceM4/RAVEN	panels, choices, 구조 메타데이터와 정답을 제공하는 HF 배포	추상 관계·패턴 학습	조건부 보조
NVIDIA Nemotron-Image-Training-v3의 raven	41,996개 reasoning 레코드, 이미지 별도 확보 필요	교사 풀이를 포함한 비교 자료	소량·후순위
Fraser/arc-agi-synthetic	4,940개 task. 각 task에 support/query 격자와 프로그램 정보	예제 기반 변환 규칙 발견	권리 확인 전 보류
공식 CLEVR 및 소규모 HF mirror	이미지·scene graph·functional program 연결	객체·관계·조합 추론	우선 보조
zihuwang/ReGuLaR	351K 레코드. Parquet 메타데이터와 별도 이미지 archive	실제 이미지의 관계·근거 연결	원출처별 선별


각 배포의 구성은 데이터 카드와 원저자 자료에서 확인했다. Hugging Face
4.2 공개 자료와 NVW의 관계
공개 데이터는 세 방식으로 사용한다.
공개 데이터 원본
    ├─ Adapter를 거쳐 독립적인 학습 mixture로 사용
    ├─ 허용된 canonical annotation으로 bridge 생성
    └─ 공개 생성기의 검증·렌더링 설계를 참고하거나 재사용

독립 NVW 명세 + 독립 seed
    └─ 새로운 episode 생성

공개 데이터를 모두 NVW 생성기의 입력으로 넣어 재작성하는 방식은 기본 경로가 아니다.
그렇게 하면 원출처·라이선스·분할·벤치마크 파생 여부가 복잡하게 얽힌다. 공개 데이터 학습 경로와 독립 생성 경로는 구분하고, 실제 파생 관계가 있을 때만 부모 ID를 연결한다.
4.3 “완전히 동일한 데이터셋은 없다”는 주장도 제한한다
확인한 데이터만으로 “이 연구와 동일한 접근은 전혀 없다”고 결론 내리지 않는다. 예를 들어 CURI는 적은 예제로 복잡한 개념을 학습하고 불확실성·구성적 일반화를 평가하는 선행 작업이다. NVW의 연구적 가치는 이름의 새로움보다 어떤 무작위화·검증·전문 분야 연결이 실제 성능 개선에 기여했는지에 둔다. GitHub
5. 데이터별 다운로드·변환·검증 설계
5.1 TRACE: 우선 활용할 생성·검증 기반
확인된 내용
TRACE는 같은 의미 상태에서 이미지, typed answer, 이미지 공간 annotation, verifier 정보, execution trace를 구성하는 생성 체계다. 공개 저장소에는 생성·검증·JSONL export 경로가 있다. 코드·템플릿은 Apache-2.0, 데이터는 CC BY 4.0으로 안내되며 제3자 자산은 별도 고지를 확인해야 한다. GitHub
데이터의 annotation_gt는 항상 전체 scene graph가 아니다. 일부 과제에서는 정답 선택지의 bounding box 등 답에 의존하는 근거 annotation이다. 이를 이미지 전체에 대한 정답 독립적인 perception label로 사용하면 안 된다. Hugging Face
수집 대상
이미지 포함 학습 레코드
task / scene / domain 식별자
typed answer
annotation와 그 타입
reward contract
execution trace sidecar와 연결 정보
생성기 버전·라이선스·자산 고지

Adapter 정책
answer_gt, annotation_gt, reward_contract, trace_ref처럼 JSON이 문자열로 저장된 필드는 명시적으로 파싱하고 schema를 검사한다.
Annotation은 다음처럼 분리한다.
OBSERVATION:
이미지에서 직접 읽을 수 있는 위치·객체·연결

QUESTION_EVIDENCE:
질문에 관련된 객체·영역

ANSWER_DEPENDENT:
정답 객체·정답 선택지의 위치

EXECUTION:
프로그램 실행과 중간 상태

ANSWER_DEPENDENT annotation은 풀이의 assistant target에는 쓸 수 있지만, 사용자 입력에 넣거나 전체 구조 추출의 정답으로 취급하지 않는다.
Split 정책
TRACE 공개 validation은 같은 1,000개 task에서 새로운 instance를 제공하는 구성이다. 이를 “미학습 task 일반화” 평가로 부르지 않는다. Task·scene grammar를 통째로 제외하는 추가 split을 별도 구축한다. GitHub
공개 생성기 활용 예
다음은 공식 저장소에 공개된 사용 형태다. 이번 환경에서 설치·실행 검증한 명령은 아니다.
trace-list
trace-list --domain charts

trace-generate \
  --task task_geometry__graph_paper__polygon_area_value \
  --samples-per-task 1 \
  --seed 42 \
  --output trace-output

trace-validate trace-output/datasets/<dataset-id>

NVW 확장에서는 기존 TRACE 전체를 먼저 수정하지 않는다. 승인된 task 몇 개를 골라 입력·출력·검증 계약을 파악하고, 기호 재표기 wrapper와 episode support 생성기를 별도 모듈로 추가한다. TRACE 검증 성공은 원래 계약에 대한 성공이므로, 추가한 ontology randomization과 식별 가능성은 별도 검사해야 한다. Md Tanvirul Alam
5.2 CLEVR: 이미지·scene·program을 함께 가져온다
공식 CLEVR는 train/validation의 이미지, scene graph, 답, functional program을 제공한다. CoGenT는 속성 조합 변화에 대한 별도 평가 자료다. 이 데이터는 관계·조합 학습에 적합하지만, 기본 의미 체계가 episode마다 새로 바뀌는 것은 아니다. Computer Science
권장 수집 경로
초기 loader 점검:
maujim/CLEVR_sample_500

주 실험:
공식 CLEVR train의 승인된 subset
+ 공식 scene graph
+ functional program

HF의 maujim/CLEVR_sample_500은 소규모 mirror이므로 전체 CLEVR의 대표 표본으로 취급하지 않는다. 카드에 안내된 이미지 폴더와 질문·scene JSON을 함께 확보해 연결해야 한다. Hugging Face
변환 절차
image_filename
    ├─ 이미지
    ├─ scene annotation
    └─ 해당 이미지의 질문·program
             ↓
        ID 일치 검증
             ↓
      program 실행·답 대조
             ↓
   이미지 기반 학습 view 생성

초기 task는 속성 비교, 공간 관계, 집합 필터, 개수 비교로 제한한다.
Scene metadata에 존재한다는 이유로 이미지에서 관측하기 어려운 모든 내부 좌표나 가려진 속성을 perception 정답으로 요구하지 않는다. 공개 상태와 시각적으로 관측 가능한 상태를 구분하는 필터를 둔다.
5.3 RAVEN: 원본 구조 자료와 교사 풀이 자료를 구분한다
경로 A — HF 원본 계열
HuggingFaceM4/RAVEN에는 여덟 개 context panel, 여덟 개 choice image, 구조 메타데이터와 정답 필드가 있다. Hugging Face
변환 방식은 다음과 같다.
context panels
    ↓
마지막 칸을 비운 3×3 문제판 구성
    +
choice panels
    ↓
선택지 배치
    ↓
target index → 최종 선택지 label 변환

선택지 순서를 바꾸면 정답 index도 함께 바꾼다. 구조 메타데이터 전체를 사용자 입력에 붙이지 않는다.
경로 B — NVIDIA RAVEN reasoning
NVIDIA 배포는 원본 RAVEN 이미지에 대한 생성 풀이를 제공하며, 카드에서 원본 이미지의 별도 확보를 안내한다. 따라서 JSONL만 내려받고 완전한 이미지 학습 자료가 준비됐다고 표시하지 않는다. Hugging Face
이 경로는 다음 조건을 충족할 때만 사용한다.
원본 이미지와 sample ID 연결 성공
정답 index 일치
선택지 배치 일치
교사 풀이의 핵심 규칙 검산
원래 split 확인
권리·오염 검사 통과

전체 Nemotron-Image-Training-v3를 통째로 내려받지 않는다. raven 관련 파일만 파일 목록을 확인한 뒤 선택적으로 수집한다.
RAVEN shortcut 통제
원래 RAVEN의 선택지 구성 문제를 다룬 I-RAVEN/SRAN 연구를 참고한다. 다만 I-RAVEN이라는 이름만으로 모든 shortcut이 제거됐다고 가정하지 않는다. Option-only 대조와 새 선택지 생성 규칙 검사를 유지한다. arXiv
5.4 ARC-AGI Synthetic: task 내부 support/query 구조를 보존한다
Fraser/arc-agi-synthetic의 한 레코드는 여러 support 입력–출력 쌍과 query 쌍을 가진 task다. hy_code, rule, family 같은 필드도 제공되지만 일부 설명 필드는 자동·휴리스틱 기반이므로 그대로 완전한 의미 정답으로 쓰지 않는다. Hugging Face
올바른 입력 구성
사용자 입력:
support 입력 이미지 + support 출력 이미지
support 입력 이미지 + support 출력 이미지
query 입력 이미지

assistant target:
query 정답 격자
선택적으로 검증된 간결한 풀이

Support의 정답 출력은 규칙을 알아내기 위해 허용되는 예제다. Query의 정답 출력은 입력에 포함하지 않는다.
레코드 내부 train과 test를 프로젝트 전체의 학습·평가 split으로 착각하지 않는다. 먼저 task 전체와 유사 프로그램 family를 기준으로 외부 split을 만든다.
코드·권리 처리
검토한 카드에서 이용 조건을 충분히 확정하지 못했으므로 초기 상태는 다음과 같이 둔다.
rights_status: UNCONFIRMED
training_allowed: false

권리 확인 전에는 학습 quota에 강제로 포함하지 않는다.
또한 공개된 Hy 프로그램을 학습 서버에서 바로 실행하지 않는다. 초기에는 저장된 격자만 사용하고, 코드 실행이 필요하면 네트워크·인증정보가 없는 제한된 환경에서 별도 검토한다.
Self-consistency와 support로부터의 정답 식별 가능성도 별개로 검사한다.
5.5 ReGuLaR: 메타데이터만 로드하면 이미지가 준비되지 않는다
ReGuLaR는 Parquet 메타데이터와 원출처별 .tar.zst 이미지 archive를 별도로 제공한다. image와 images 필드는 추출 후 데이터 루트에 대한 상대경로이며, 관계·scene graph는 JSON 문자열 필드로 들어 있다. 원출처들의 라이선스와 조건을 따르도록 안내한다. Hugging Face
수집·변환 절차
Parquet metadata
    ↓
source·원래 ID·필요 이미지 목록 추출
    ↓
승인한 source의 archive만 확보
    ↓
안전한 추출
    ↓
상대경로·이미지 hash 확인
    ↓
relations_json / scene_graph_json 파싱
    ↓
원출처별 annotation 범위 검사

데이터셋에 관계가 기록되지 않았다는 이유만으로 그 관계가 존재하지 않는다고 학습시키지 않는다. annotation_coverage를 PARTIAL 또는 FULL_FOR_TASK 등으로 표시한다.
Standalone CLEVR와 ReGuLaR 내부 CLEVR가 같은 이미지를 가리킬 수 있으므로 dataset 이름이 아니라 원출처 ID와 이미지 계보로 중복·분할을 통합한다.
6. Hugging Face 수집 파이프라인
6.1 다운로드와 학습 승인은 분리한다
저장소 확인
    ↓
metadata·파일 목록 조사
    ↓
full revision SHA 기록
    ↓
사용할 파일·원래 split·권리 범위 결정
    ↓
선택 다운로드
    ↓
schema·이미지·정답·원출처 검사
    ↓
오염·분할 검사
    ↓
학습 승인 manifest 생성

Hugging Face의 다운로드 함수는 revision과 파일 선택 기능을 제공한다. 재현성을 위해 main이나 태그 이름만 기록하지 않고, 실제 실행에서는 full commit SHA로 고정한다. Hugging Face
6.2 원출처 등록 예시
sources:
  trace:
    repo_id: maveryn/trace
    requested_revision: dataset-v1
    resolved_revision: null
    role: verified_multidomain_synthetic
    rights_status: PENDING_REVIEW
    training_allowed: false

  raven_hf:
    repo_id: HuggingFaceM4/RAVEN
    requested_revision: main
    resolved_revision: null
    role: abstract_relational_reasoning
    rights_status: PENDING_REVIEW
    training_allowed: false

  raven_nvidia:
    repo_id: nvidia/Nemotron-Image-Training-v3
    requested_revision: main
    resolved_revision: null
    allowed_subset: raven
    external_images_required: true
    training_allowed: false

  arc_synthetic:
    repo_id: Fraser/arc-agi-synthetic
    requested_revision: main
    resolved_revision: null
    rights_status: UNCONFIRMED
    training_allowed: false

  clevr_smoke:
    repo_id: maujim/CLEVR_sample_500
    requested_revision: main
    resolved_revision: null
    role: loader_smoke_only
    training_allowed: false

  regular:
    repo_id: zihuwang/ReGuLaR
    requested_revision: main
    resolved_revision: null
    rights_policy: per_upstream_source
    training_allowed: false

resolved_revision: null은 아직 실행용 버전을 확정하지 않았다는 뜻이다. 임의의 SHA를 채워 넣지 않는다.
6.3 Metadata 조사 코드
다음은 수집 도구의 시작 코드다. 이번 작업에서 실행 검증하지 않았으며, 저장소 접근과 설치된 라이브러리 버전 확인이 필요하다. Dataset 파일을 대량 다운로드하거나 학습을 승인하지 않고 metadata와 revision을 기록한다. HfApi의 dataset 정보 조회 기능을 사용한다. Hugging Face
from __future__ import annotationsimport jsonimport refrom datetime import datetime, timezonefrom pathlib import Pathfrom huggingface_hub import HfApiSOURCE_REFS: dict[str, tuple[str, str]] = {    "trace": ("maveryn/trace", "dataset-v1"),    "raven_hf": ("HuggingFaceM4/RAVEN", "main"),    "raven_nvidia": ("nvidia/Nemotron-Image-Training-v3", "main"),    "arc_synthetic": ("Fraser/arc-agi-synthetic", "main"),    "clevr_smoke": ("maujim/CLEVR_sample_500", "main"),    "regular": ("zihuwang/ReGuLaR", "main"),}def inspect_sources(output_path: Path) -> None:    api = HfApi()    sources: dict[str, dict] = {}    failed: list[str] = []    for name, (repo_id, requested_revision) in SOURCE_REFS.items():        try:            info = api.dataset_info(                repo_id=repo_id,                revision=requested_revision,                files_metadata=True,            )            sha = info.sha            if not isinstance(sha, str) or not re.fullmatch(                r"[0-9a-f]{40}", sha


태그 조회에 실패하면 자동으로 main으로 바꿔 받지 않는다. 실패를 기록하고 선택한 revision을 검토한다.
6.4 실제 다운로드 시 추가할 제한
Downloader는 승인된 정확한 파일 목록을 입력으로 받도록 구현한다.
repo_id
full revision SHA
approved file paths
예상 총 용량
수집 목적
권리 검토 상태
원래 split

수집 후에는 파일마다 SHA-256, 크기, 다운로드 revision, 시간, 원래 경로를 기록한다.
Archive는 임의 shell script 실행 대신 제한된 추출기로 처리한다. 경로 이탈, symbolic link, 예상 밖 파일 타입, 과도한 해제 용량을 차단한다. Python의 archive extraction filter도 모든 위험을 제거하는 보장은 아니므로 별도 용량·파일 수·경로 검사가 필요하다. Python documentation
7. Canonical representation의 공통 계약
7.1 모든 분야를 같은 의미 없는 graph로 압축하지 않는다
공통 schema는 교환 형식을 통일하기 위한 것이다. 분야별 의미 차이를 없애는 것이 아니다.
공통:
objects / attributes / relations / observations
semantic_state / rules / execution / answer

분야별:
music notation
circuit netlist
table hierarchy and units
graph transition state
geometry constraints

회로의 connected_to, 계통도의 ancestor_of, 설명 도식의 explains를 단순한 related_to로 합치면 전문 의미 연결을 학습하기 어렵다.
7.2 관측과 해석을 나눈다
{
  "observed_structure": {
    "objects": [],
    "positions": [],
    "visible_connections": []
  },
  "semantic_state": {
    "roles": [],
    "typed_relations": [],
    "units": [],
    "assumptions": []
  },
  "execution": {
    "rule_ids": [],
    "intermediate_states": [],
    "answer": null
  }
}

observed_structure는 픽셀에 대응하는 정보다. semantic_state는 legend·문맥·전문 규약을 적용한 해석이다. execution은 그 상태에서 수행한 계산이다.
이 구분으로 “선은 정확히 읽었는데 그 선의 의미를 잘못 붙인 오류”와 “연결선 자체를 누락한 오류”를 분리한다.
7.3 근거 위치를 보존한다
객체와 관계에는 가능한 경우 다음을 연결한다.
image_id
panel_id
object_instance_id
bounding_box 또는 point/segment
좌표 기준과 원본 이미지 크기
관련 annotation의 출처·검증 상태

같은 glyph가 여러 번 등장하더라도 glyph 종류와 객체 instance ID를 구분한다.
8. 자체 NVW 생성기의 구조
8.1 생성 순서
독립 seed
    ↓
world family와 허용 DSL 선택
    ↓
타입이 맞는 객체·관계·규칙 생성
    ↓
초기 상태와 query 생성
    ↓
독립 solver로 실행
    ↓
support 예제 구성
    ↓
정답·의미·중간 상태의 식별 가능성 검사
    ↓
glyph·relation-code 매핑 결정
    ↓
canonical state에서 이미지 렌더링
    ↓
렌더링·전처리 후 가독성 검사
    ↓
view 생성
    ↓
권리·오염·family split 검사
    ↓
승인 manifest

8.2 첫 구현은 세 world family로 제한한다
Family	초기 과제	검증 방법
Signal graph	연결·차단·조건부 전파·도달성	작은 그래프의 완전 탐색
Spatial composition	포함·방향·순서·속성 필터·집합 조합	좌표·집합 연산
Finite-state world	기호별 상태 전이·연속 변환·조건부 갱신	명시적 전이표 실행


처음부터 모든 전문 분야를 하나의 만능 생성기로 구현하지 않는다. 이 세 family에서 정확한 데이터·학습 연결을 확인한 후 전문 bridge를 확장한다.
8.3 DSL은 작고 명시적으로 시작한다
객체 타입:
Node, Token, Region, Container, IntegerValue, BooleanState

관계:
CONNECTED_TO
DIRECTED_TO
INSIDE
LEFT_OF
ABOVE
PRECEDES

연산:
FILTER
COUNT
COMPARE
UNION
INTERSECTION
FOLLOW_EDGE
LOOKUP_TRANSITION
UPDATE_STATE
IF_THEN

모든 조합을 무조건 허용하지 않는다.
예를 들어 COUNT의 결과를 node처럼 연결하거나, 단위를 가진 값을 단위 없는 상태 flag에 대입하는 프로그램은 타입 검사에서 거부한다.
상태 전이에는 동기식/순차식 업데이트 정책, tie-breaking, 최대 단계, 종료 조건을 명시한다. 초기에는 순환 없는 그래프나 유한 step horizon으로 실행을 제한한다.
8.4 난수 원천을 분리한다
seed_world
seed_structure
seed_rule
seed_support
seed_glyph
seed_layout
seed_style
seed_options

렌더링 seed만 바꿨는데 정답까지 바뀌거나, 선택지 순서를 바꿨는데 latent rule이 달라지는 결함을 막는다.
재현 단위는 seed 하나가 아니라 generator commit + 설정 + 각 seed + 자산 버전이다.
9. 정답 식별 가능성: 생성기가 아는 답과 모델이 알 수 있는 답
9.1 기본 검사
Episode에 허용된 가설 집합을 \(H\), support 예제를 \(D\), query를 \(q\)라고 하자.
\[
C(D)=\{h\in H : h\text{가 모든 support와 일치}\}
\]
\[
A(q,D)=\{h(q):h\in C(D)\}
\]
처리 규칙은 다음과 같다.
조건	처리
\(C(D)=\varnothing\)	모순된 생성 결과로 폐기
(	A(q,D)
(	A(q,D)


이 검사는 명시한 \(H\) 안에서만 유효하다. 허용 규칙의 종류와 값 범위를 모델이 알 수 있도록 문제 계약에 포함한다. 이전 계획서의 식별 가능성 검사를 그대로 계승한다.    Qwen3_VL_4B_Finetuning_Research…
9.2 이번 확장안의 중요한 추가: 중간 상태도 따로 검사한다
최종답이 하나로 정해져도, 숨은 규칙이나 중간 상태는 하나로 정해지지 않을 수 있다.
예를 들어 허용 규칙이 다음 네 가지라고 하자.
IDENTITY: 0→0, 1→1
FLIP:     0→1, 1→0
SET_ZERO: 0→0, 1→0
SET_ONE:  0→1, 1→1

Support가 0→1 하나라면 FLIP과 SET_ONE이 모두 가능하다.
Query가 0일 때 답은 두 규칙 모두 1이다. 그러나 이를 근거로 assistant target에 **“규칙은 FLIP이다”**라고 강제하면 잘못된 supervision이다.
따라서 라벨별 식별 가능성을 따로 기록한다.
answer_identifiable
mapping_identifiable
rule_identifiable
state_prefix_identifiable
evidence_identifiable

식별되지 않는 규칙·중간 상태는 loss에서 제외하거나, 동치 후보 집합으로 표현한다. 생성기가 내부적으로 고른 한 규칙을 모델이 반드시 알아낼 수 있다고 가정하지 않는다.
9.3 Reconstruction에도 같은 원칙을 적용한다
간선 하나를 지운 뒤 복원시키는 과제에서, 가능한 원래 그래프가 여러 개라면 하나를 임의 정답으로 붙이지 않는다.
복원 대상이 충분한 제약으로 정해지는지 검사한다. 정해지지 않으면 INSUFFICIENT_INFORMATION으로 분류하거나 해당 task를 생성하지 않는다.
10. NVW episode 예시와 supervision
다음은 데이터 계약 설명용 예시다. 실제 이미지가 생성·검증된 학습 샘플은 아니다.
10.1 세계 정의
두 종류의 새로운 glyph를 U와 V라고 임시 표기한다. 실제 학습 이미지에서는 독립 생성한 도형을 사용한다.
Support 예제는 다음과 같다.
U:
0 → 1
1 → 0

V:
0 → 0
1 → 0

앞 절의 제한된 규칙 집합 안에서 U는 FLIP, V는 SET_ZERO로 식별된다.
Query 이미지에는 초기값 0과 다음 방향 그래프를 그린다.
입력 0 → U₁ → V₁ → U₂ → 출력

10.2 정답 구조와 상태
객체:
U₁, V₁, U₂

관계:
U₁ PRECEDES V₁
V₁ PRECEDES U₂

의미:
U₁과 U₂는 동일한 glyph 종류이며 FLIP을 수행
V₁은 SET_ZERO를 수행

상태:
S0 = 0
S1 = FLIP(0) = 1
S2 = SET_ZERO(1) = 0
S3 = FLIP(0) = 1

정답:
1

10.3 반사실 쌍
V₁을 우회하도록 연결을 바꾼다.
입력 0 → U₁ → U₂ → 출력

새 상태는 0 → 1 → 0이므로 정답은 0이다.
반면 glyph의 크기·선 굵기만 바꾼 render variant의 정답은 계속 1이어야 한다.
이렇게 의미 변경에 대한 민감도와 의미 보존에 대한 안정성을 함께 학습한다.
10.4 같은 root에서 만들 수 있는 view
직접 QA:
이미지 + 질문 → 1

구조 추출:
이미지 → 객체와 방향 연결

의미 추론:
support + query 이미지 → U와 V의 규칙

상태 추적:
이미지 → 0, 1, 0, 1

오류 수정:
잘못된 중간 상태 + 이미지 → 잘못된 단계와 수정 결과

그러나 모든 root를 모든 view로 복제하지 않는다. 초기에는 root당 최대 3개 정도를 선택하고, epoch 사이에서 view를 순환시키는 방식을 우선 검토한다.
11. 실제 전문 분야와 연결하는 Canonical Bridge
11.1 같은 전문 문제를 세 가지 표현으로 만든다
하나의 전문적 의미 상태
       │
       ├─ 표준 전문 표기
       ├─ 새 기호 + 명시적 legend
       └─ 새 배치·스타일의 표준 표기

세 표현은 같은 법칙·변수·연결을 공유한다. 각각 다른 독립 문제로 세지 않고 하나의 family로 관리한다.
11.2 분야별 bridge
분야	보존할 canonical 정보	NVW와 연결할 공통 기능
Music	오선·음자리표·음표 위치·변화표·음높이 철자·음가·성부	기준틀 결합, 기호 역할, 시간 상태
Circuit	노드·소자·값·단위·전원·접점·스위치 상태	Topology, 조건부 연결, 규칙 적용
Graph/CS	노드·간선·방향·가중치·초기 상태·연산 순서	관계 보존, 상태 전이
Table/Chart	셀·header·단위·기간·series·집계 범위	역할 연결, 수량·기준집단 보존
Biology diagram	명시적 노드·분기·경로·전문 관계 타입	포함·조상·경로·조건
Geometry/Materials	좌표·축·벡터·면·제약·표기 타입	기준 좌표계와 적용 규칙 구분


기존 계획의 분야별 canonical generator를 이 bridge의 기반으로 사용한다.    Qwen3_VL_4B_Finetuning_Research…
11.3 공개 전문 자료의 활용
MUSCIMA++는 기보 기호와 연결 구조를 제공하므로 악보 bridge의 후보가 된다. 다만 원본 이미지·annotation의 권리와 사용 범위를 구분하고, annotation에 없는 화성·성부 의미를 이미 있는 gold처럼 추가하지 않는다. ÚFAL
AI2D-RST는 grouping·connectivity·설명 관계를 제공하므로 도식 bridge에 활용할 수 있다. 모든 연결을 인과관계로 바꾸지는 않는다. GitHub
ChartQA와 FinQA는 각각 차트·표의 시각 해석, 표와 문맥을 이용한 계산 프로그램 연결에 활용한다. FinQA의 정답 근거 index나 program은 assistant target과 검증에만 사용하고 사용자 입력에는 누설하지 않는다. Hugging Face
11.4 Bridge가 곧 전이의 증거는 아니다
같은 canonical state를 새 glyph로 다시 그린 문제를 잘 푸는 것은 표현 변화에 대한 일반화다.
학습에 사용하지 않은 실제 악보 출처나 실제 회로 문제를 잘 푸는 것은 그보다 더 강한 원출처·도메인 전이다.
두 결과를 별도 보고한다.
12. 정답·렌더링·학습 입력의 품질 검증
12.1 네 층위의 검증
검증 층위	확인할 것
Canonical	타입·단위·관계·전제에 모순이 없는가
Solver	답과 중간 상태가 독립 구현에서도 일치하는가
Render	그 구조와 조건이 이미지에 정확히 표현됐는가
Model input	실제 resize·tokenization 뒤에도 필요한 정보가 남는가


생성기가 내부 정답을 갖고 있다는 사실은 두 번째 층위까지만 일부 지원한다. 이미지에 작은 접점이 사라졌거나 support가 불충분하면 유효한 시각 학습 샘플이 아니다.
12.2 독립 solver 정책
초기 작은 문제에서는 효율적인 solver와 단순 완전 탐색 구현을 교차 검증한다.
같은 함수의 결과를 두 번 호출해 같다고 확인하는 것은 독립 검증이 아니다.
수치 과제에는 단위·허용 오차·반올림 시점을 명시하고, 그래프 과제에는 여러 정답이 가능한지와 tie-breaking을 명시한다.
12.3 Render 검사
객체가 화면 밖으로 잘리지 않았는가
support/query panel의 순서가 유지되는가
선의 시작·끝이 올바른 객체와 연결되는가
교차와 접점이 구분되는가
glyph끼리 구별 가능한가
필수 legend가 가려지지 않았는가
문자와 수치가 넘치지 않았는가
이미지 처리 뒤 작은 기호가 사라지지 않는가

SVG/XML/JSON 등 원본 정보가 있으면 이를 우선 사용한다. OCR은 원본 구조를 확보할 다른 방법이 없을 때의 보조 수단으로 제한한다.
12.4 품질 상태
SOURCE_GOLD
DERIVED_VERIFIED
TEACHER_REVIEWED
UNRESOLVED
REJECTED

TEACHER_REVIEWED를 자동으로 DERIVED_VERIFIED로 올리지 않는다. Cycle consistency도 잘못된 두 표현이 서로 일치하는 경우를 배제하지 못하므로 보조 검사로만 사용한다.
13. 학습 목표와 Qwen 입력 포맷
13.1 첫 단계는 다중 task SFT
초기에는 별도 graph head나 새로운 image decoder를 붙이지 않는다. 기존 생성 objective에서 assistant target을 학습한다.
View	입력	목표 출력
Direct QA	이미지·질문	답
Structure	이미지·구조 요청	객체·관계·속성
Grounded solution	이미지·질문	근거·규칙·필요 상태·답
Rule induction	support·query 이미지	식별 가능한 규칙과 답
Repair	이미지·오류 후보	오류 위치·수정 결과
Sufficiency	예제·query	결정 가능 여부와 가능한 결론


여기서 “구조 loss”는 처음에는 구조를 직렬화한 assistant token에 대한 loss다. Graph F1이나 solver 검증 결과가 별도 구현 없이 자동으로 미분 가능한 loss가 되는 것은 아니다. 이 구분은 기존 계획을 유지한다.    Qwen3_VL_4B_Finetuning_Research…
13.2 긴 설명보다 검증 가능한 상태를 우선한다
근거 객체·관계
    ↓
적용 규칙
    ↓
필요한 상태 변화
    ↓
검산
    ↓
최종답

모든 문제에서 장문의 추론을 강제하지 않는다. 단순 구조 문제는 짧게 답하고, 긴 상태 전이가 필요한 문제에만 더 큰 target budget을 준다.
Teacher forcing에서 gold 구조 뒤에 답을 생성하는 능력과, 실제 추론에서 스스로 구조를 찾아 답하는 능력은 다르므로 Direct QA view를 유지한다.
13.3 학습 JSONL 예시
아래는 공식 계열 loader의 image와 conversations 형태를 참고한 schema 예시다. 실제 이미지 경로가 없으므로 그대로 학습하지 않는다. GitHub
{
  "id": "SCHEMA_ONLY_episode_001_query",
  "image": [
    "NOT_RENDERED/support_001.png",
    "NOT_RENDERED/query_001.png"
  ],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n<image>\nUse the examples in the first image to interpret the operators. Follow the directed sequence in the second image. Give the final state with a brief justification."
    },
    {
      "from": "gpt",
      "value": "The first operator flips the bit, and the second sets it to zero. Starting from 0, the states are 1, 0, and 1.\nFinal answer: 1"
    }
  ]
}

이미지 개수와 <image> 개수·순서는 반드시 일치시킨다.
13.4 Private canonical 데이터는 별도 보관한다
{
  "episode_id": "SCHEMA_ONLY_episode_001",
  "family_id": "SCHEMA_ONLY_family_001",
  "source_kind": "INDEPENDENT_SYNTHETIC",
  "generator_commit": null,
  "hypothesis_space_id": "boolean_unary_v1",
  "quality": {
    "solver": "PENDING",
    "render": "PENDING",
    "answer_identifiable": null,
    "rule_identifiable": null,
    "state_prefix_identifiable": null
  },
  "contamination": {
    "benchmark_derived": false,
    "audit_decision": "NOT_AUDITED"
  },
  "training_allowed": false
}

Private canonical 파일에는 답·전체 프로그램이 있어도 되지만, 사용자 입력 serializer는 허용된 필드만 선택한다.
14. 초기 데이터 혼합과 규모
14.1 10K pilot / 50K 확장 제안
다음 수량은 검증을 통과한 학습 레코드의 목표치다. 원본 이미지나 독립 episode 수가 아니다. 최적 비율이라는 뜻도 아니다.
데이터 계열	10K pilot	50K 확장	비율
자체 NVW	1,000	5,000	10%
TRACE 선별	1,200	6,000	12%
CLEVR·I-RAVEN 등 추상 관계	800	4,000	8%
ReGuLaR 원출처 승인 subset	500	2,500	5%
악보 구조·의미	1,500	7,500	15%
그래프·회로·AI2D 계열	1,500	7,500	15%
표·차트·FinQA 계열	1,500	7,500	15%
기하·재료·생물 도식 등	1,200	6,000	12%
일반 시각 능력 보존	800	4,000	8%
합계	10,000	50,000	100%


ARC synthetic은 권리·품질 확인을 통과할 때 추상 관계 quota 일부를 대체한다. 확인 전에는 포함하지 않는다.
14.2 초기 NVW 내부 구성
NVW-K: 새 표기와 기존 규칙 연결       50%
NVW-I: 제한된 규칙 발견              25%
NVW-S: 여러 단계 상태 추적           25%

반사실·오류 수정·불충분 정보 task는 이 세 트랙 안에서 배분한다. 별도 비율로 다시 더해 총합을 100% 이상으로 만들지 않는다.
14.3 목표 형식의 비율은 별도 축이다
Direct QA                         35%
구조 추출                         20%
근거·규칙·중간 상태·답             25%
반사실·오류 수정                   15%
충분성·불확실성 판단                5%

모든 분야가 모든 task를 지원해야 하는 것은 아니다. 존재하지 않는 annotation을 만들면서 비율을 맞추지 않는다.
14.4 부족한 source quota의 처리
권리·품질·중복 검사로 특정 데이터가 부족하면 다음 중 하나를 택한다.
검증된 독립 합성으로 대체
승인된 다른 source로 대체
전체 데이터 규모 축소

어떤 선택이든 모든 비교 arm에 같은 새 manifest를 적용한다. ReGuLaR를 사용하지 못해 자체 합성으로 대체했다면 ReGuLaR의 효과를 실험했다고 쓰지 않는다.
14.5 10%부터 검증한다
NVW는 중요한 연구 축이지만 처음부터 데이터 대부분을 차지하게 하지 않는다.
첫 비교는 NVW 0%와 10%로 하고, 실제 전문 과제 전이가 확인될 때 20% 또는 30%로 확대한다. Synthetic 점수만 오르면 비중을 늘리지 않는다.
15. 분할·중복·미학습 조건 설계
15.1 증강 전에 분할한다
원본 source와 latent root 등록
    ↓
같은 family·근접 중복 묶기
    ↓
train / development / sealed holdout 배정
    ↓
각 split 안에서만 증강

같은 episode의 support/query, 새 glyph 버전, counterfactual, 오류 수정 view를 서로 다른 split에 흩어 놓지 않는다.
15.2 일반화 평가 축
평가	학습과 달라지는 것	주장 가능한 범위
New instance	같은 family의 새 상태	분포 내 instance 일반화
New alphabet	기호 family	표기 변화
New mapping	역할·관계 코드 매핑	문맥 기반 재결속
New composition	본 primitive의 새 조합	구성적 일반화
New topology	그래프 구조	구조 변화
Greater depth	더 긴 상태 전이	길이 일반화
New renderer	배치·스타일·이미지 생성 방식	시각 분포 변화
New source	새로운 작품·문서·도식 원출처	실제 source 전이
Held-out domain	특정 분야 전체 제외	더 강한 분야 전이


단순히 random seed만 다르면 새 ontology라고 부르지 않는다.
15.3 미학습 분야 평가의 조건
예를 들어 악보를 held-out domain으로 정한다면 Music 전용 source만 빼는 것으로 끝내지 않는다. TRACE·기타 mixture·bridge에도 악보 과제가 들어 있는지 확인한다.
이렇게 해도 base model의 사전학습에서 음악을 접하지 않았다는 뜻은 아니다. 정확한 주장은 **“이번 추가 파인튜닝에서 해당 분야를 제외했다”**이다.
16. Curriculum: 쉬운 지각에서 복합 상태로
단계	학습 난도	주 목표
C0	소수 객체·명확한 연결·알려진 표기	Loader·gradient·정답 계약 점검
C1	새 glyph + 완전한 legend	기호와 의미 연결
C2	제한된 support에서 단일 규칙 발견	규칙 식별
C3	2–4단계 연산·상태 변화	중간 상태 유지
C4	다중 관계·방해 객체·조건 변경	선택적 구조 보존
C5	전문 표기와 NVW 공동 학습	전문 지식 연결
C6	미학습 조합·깊이·renderer	외부 일반화 평가


이 단계는 모든 데이터를 순차적으로 완전히 교체한다는 뜻이 아니다. 후반에도 쉬운 관계 추출과 일반 시각 QA를 일부 유지한다.
Curriculum의 효과는 최종적으로 같은 자료를 섞어서 학습한 대조군과 비교할 수 있다. 초기 프로젝트에서는 curriculum·모듈·데이터 종류를 모두 동시에 탐색하지 않는다.
17. Vision·Merger·LLM 파인튜닝 범위
17.1 모듈 정의
구분	범위	초기 방식
V	Merger를 제외한 Vision 계통	Vision attention LoRA
M	마지막 Merger + 세 DeepStack Merger	선택적 full FT
L	언어 Decoder attention·MLP	LoRA
FML	M과 L 공동 업데이트	M full FT + L LoRA
FVML	V·M·L 선택적 공동 업데이트	추가 검증 후 고려


기존 보고서의 지정 revision 조사에서는 V 약 306M, M 약 109M, L 약 4.02B로 분류됐다. 이번 작업에서 실제 가중치를 다시 로드해 재계수한 값은 아니므로 학습 시작 전에 재검증한다.    Qwen3_VL_4B_Finetuning_Research…
17.2 주 실험은 모두 같은 F0에서 시작한다
             ┌─ FL
원본 F0 ─────┼─ FM
             ├─ FV
             └─ FML

FL → FM → FV는 실행 순서다. FL checkpoint를 FM이 이어받는 구조가 아니다.
17.3 같은 이미지 기반 manifest를 사용한다
FL에 텍스트-only 구조 문제를 주고 FM/FV에 이미지 문제를 주면 업데이트 위치와 학습 데이터 효과가 섞인다.
따라서 주 모듈 비교에서는 다음을 동일하게 고정한다.
record IDs
이미지·질문·assistant target
sampling order
학습 노출량
평가 집합·프로토콜

FL-text는 별도의 기능 실험이다. FM/FV에 text-only 레코드를 넣어 놓고 유효한 시각 경로 학습량이 같다고 계산하지 않는다.    Qwen3_VL_4B_Finetuning_Research…
17.4 공식 플래그만 믿지 않는다
확인한 공식 train_qwen.py에서는 LoRA 분기와 일반 set_model 분기가 나뉘며, 일반 설정 함수의 별도 Merger 제어는 마지막 visual.merger를 직접 다룬다. 원하는 FML·DeepStack 구성이 자동으로 적용된다고 가정하지 않는다. GitHub
필수 검사는 다음과 같다.
실제 trainable 파라미터 이름과 shape
V/M/L 그룹의 중복·누락
optimizer parameter 집합
각 그룹의 gradient
실제 update 후 변화량
고정 파라미터의 불변성
저장 후 fresh process 재로드

FM/FV에서는 고정 LLM을 통과하는 입력 gradient가 필요하다. LLM weight를 고정하는 것과 LLM forward를 no_grad로 감싸는 것을 혼동하지 않는다. 기존 계획서의 역전파 경계 검사를 유지한다.    Qwen3_VL_4B_Finetuning_Research…
18. 실험 설계: 모듈과 데이터 효과를 나눠 검증
18.1 1차 — 모듈 비교
실험	업데이트	학습 데이터
F0	없음	없음
FL	L LoRA	공통 10K
FM	M full FT	공통 10K
FV	V LoRA	공통 10K
FML	M full FT + L LoRA	공통 10K


FM full FT와 LoRA arm의 trainable parameter 수가 다르므로, 이 결과는 선택한 실용 레시피의 비교다. 동일 용량에서의 모듈 우열이라고 확대하지 않는다.
18.2 2차 — NVW와 bridge 효과
좋은 모듈 설정 하나를 고정한 뒤 다음 비교를 수행한다.
실험	Episode 매핑	전문 canonical bridge
A0	고정	없음
A1	무작위화	없음
A2	고정	있음
A3	무작위화	있음


가능한 한 같은 latent root·규칙 난도·시각 복잡도를 사용한다.
Bridge가 없는 arm에서도 같은 수의 레코드를 제공하되 일반 이미지 QA나 단일 표현 view로 채운다. Bridge arm에만 데이터를 더 많이 주고 그 차이를 bridge의 효과라고 부르지 않는다.
이 비교가 묻는 것은 다음이다.
A1 − A0:
episode별 의미 매핑의 추가 가치

A2 − A0:
표현 연결 학습의 추가 가치

A3 − A2:
bridge가 있을 때 randomization의 추가 가치

A3 − A1:
randomization이 있을 때 bridge의 추가 가치

이 차이는 선택한 실험 구성에서의 행동적 효과다. 신경망 내부에서 독립적인 추론 알고리즘이 만들어졌다는 증명은 아니다.
18.3 실용 mixture 비교는 별도로 둔다
R: 전문 시각 데이터
P: 공개 추상·관계 데이터
N: 자체 NVW
B: canonical bridge supervision

선택적 확장 비교:
R
R + P
R + N
R + P + N + B

총 학습 예산을 고정하는 주 비교와, 데이터를 실제로 더 추가하는 scaling 비교를 구분한다.
18.4 같은 레코드 수와 같은 token 수는 다를 수 있다
긴 구조 풀이를 제공하는 arm은 더 많은 target token을 학습할 수 있다.
따라서 다음을 모두 기록한다.
레코드 수
고유 root·family 수
이미지 수
실제 visual token 수
loss가 계산된 target token 수
optimizer update 수
GPU 시간

목표 형식이 다른 실험에서는 모든 조건을 동시에 완전히 맞췄다고 주장하지 않는다. 주 통제 기준을 미리 정하고 나머지를 보고한다.
19. 학습 설정과 메모리 운영
아래는 시작 제안값이며 검증된 최적값이나 실제 실행 결과가 아니다.
항목	시작 설정
Base precision	BF16 우선
L LoRA	Rank 16, alpha 32
V LoRA	Rank 16, alpha 32
LoRA dropout	0.05
L learning rate	\(5\times10^{-5}\)
M learning rate	\(1\times10^{-5}\)
V learning rate	\(1\times10^{-5}\)
Optimizer	AdamW
Gradient clipping	1.0
Micro-batch	1
Gradient accumulation	16
초기 epoch	1
초기 sequence budget	4,096 이내에서 실측
이미지 수	1–2장부터 점검
Checkpoint 선택	외부 development set
주 학습 비교	승인된 24GB급 장비에서 memory smoke 후 수행


19.1 NVW 다중 이미지의 특별한 주의점
Support 이미지를 여러 장 추가하면 이미지별 해상도를 그대로 유지할 경우 총 visual token이 급격히 증가할 수 있다.
따라서 이미지 한 장당 budget이 아니라 episode 전체 budget을 관리한다.
Support를 한 panel 이미지로 합칠 때도 작은 glyph와 접점이 사라질 수 있으므로, 합성 전후 해독 가능성을 검사한다. 단순 축소로 OOM을 피하고 정답에 필요한 정보를 잃는 샘플을 승인하지 않는다.
19.2 양자화 실험은 별도 기준점을 둔다
BF16이 맞지 않아 QLoRA를 사용한다면 quantized base F0-Q를 따로 평가한다.
BF16 FT ↔ BF16 F0
QLoRA FT ↔ 같은 quantization의 F0-Q

M full FT가 필요한데 M까지 양자화된 고정 계층으로 남아 있는 상태를 허용하지 않는다.
24GB에서 추론이 됐다는 사실은 이 NVW 다중 이미지 학습이 메모리에 맞는다는 증거가 아니다. 실제 optimizer update 이후의 peak와 재로드를 확인한다.
20. 평가 계획과 일반화 판정
20.1 평가 계층
계층	용도
개발용 외부 dev	LR·epoch·module·mixture 선택
NVW sealed holdout	새 기호·조합·깊이·renderer
실제 전문 source holdout	합성에서 실제 전문 자료로의 전이
MMMU validation	제한된 milestone 비교와 기존 결과 연결
MMMU test	필수 최종 평가
MMMU-Pro Standard 10-way	필수 최종 평가
MMMU-Pro vision	추가 시각 의존성·일반화 분석


최종 benchmark에는 정답 구조, teacher 풀이, 외부 solver, 정답 기반 crop을 제공하지 않는다.
NVW support 예제를 포함한 학습은 모델 parameter를 업데이트하는 데이터 설계다. 최종 MMMU 입력에 별도 support나 추론 도구를 추가하는 것과는 구분한다.
20.2 기준점은 새 프로토콜과 함께 고정한다
F0-legacy:
기존 V8 결과

F0-frozen:
이번에 고정한 평가로 원본 모델을 실행한 결과

FT-frozen:
같은 평가에 학습 checkpoint만 바꾼 결과

가중치 개선량의 주 비교는 FT-frozen − F0-frozen이다. 입력 역할·파서·정답 alias·출력 예산 변경을 학습 효과에 포함하지 않는다.
20.3 MMMU-Pro 원문 중복 관리
기존 계획의 MMMU/Pro 원문 연결·노출 원장을 유지한다. 공식 Pro 전체 점수는 보고하되, 이미 분석한 MMMU 원문이나 공유 이미지 family를 제외한 보조 결과를 별도로 정의할 수 있다. 이것을 공식 leaderboard 점수로 부르지 않는다.    Qwen3_VL_4B_Finetuning_Research…
20.4 단계별 기능 진단
같은 독립 root에 다음 입력 조건을 적용한다.
E-I: 이미지 + 질문
E-R: 정확한 관측 구조 + 질문
E-S: 정확한 의미 상태 + 질문
E-SR: 의미 상태 + 허용 규칙 설명

이 조건들은 기능 진단이며 benchmark score가 아니다.
학습 뒤 E-I는 좋아졌지만 E-S가 그대로인지, E-SR에서도 긴 전이가 여전히 실패하는지 등을 확인해 후속 데이터를 결정한다.
21. 측정 지표와 성공 기준
21.1 최소 지표
영역	지표
최종 성능	QA accuracy, 분모
구조	객체·관계 precision/recall/F1
질문 관련 구조	Critical relation 보존율
의미	역할·단위·legend 연결 정확도
규칙	식별 가능한 규칙의 일치·실행 정확도
상태	단계별 상태·전체 경로·정확한 prefix
반사실	변화 전후 양쪽 정답률
불변성	의미 보존 render pair 양쪽 정답률
충분성	답변 가능/불충분 정보 구별
출력	반복·잘림·NO_PARSE·생성 token
보존	일반 시각 QA와 기존 강점의 변화
비용	학습 시간·peak memory·token 노출량


단순히 답이 바뀐 비율을 반사실 성공률로 사용하지 않는다. 두 답이 모두 틀려도 답 변경률은 높을 수 있기 때문이다.
21.2 일반화 주장의 단계
1단계:
같은 세계 family의 새 instance를 잘 푼다.

2단계:
새 기호·새 매핑에서도 잘 푼다.

3단계:
새 규칙 조합·topology·깊이에서도 개선된다.

4단계:
학습하지 않은 실제 전문 원출처에서 개선된다.

5단계:
일부 미학습 분야에서도 개선된다.

1–2단계만 성공하고 4단계가 실패하면 NVW 과제 적응은 성공했지만 프로젝트의 전문 시각 전이는 확인하지 못한 것으로 보고한다.
21.3 반복과 불확실성
최초 pilot은 한 seed로 수행할 수 있다. 최종 후보와 핵심 대조군은 예산이 허용하면 세 학습 seed로 반복한다.
통계 집계는 같은 이미지·world·문서 family의 여러 view를 독립 표본으로 세지 않는다. 개선·악화 문항을 모두 보고하며, 특정 과목의 한두 문항 변화만으로 범용 능력 향상을 주장하지 않는다.
21.4 확대 여부를 결정하는 기준
50K 확대의 필수 조건은 다음과 같다.
데이터·권리·오염·분할 검사 통과
실제 gradient·update·재로드 통과
외부 end-to-end QA의 개선 신호
관계 또는 상태 지표의 일관된 개선
일반 능력의 큰 붕괴 없음
synthetic-only 개선으로 설명되지 않음

초기 신호가 작은 표본에서 나타났다면 확정 성공이 아니라 후속 검증 가치로만 해석한다.
22. 오염 방지와 학습 경계
22.1 차단 대상
다음은 학습과 학습 문제 생성의 원본으로 사용하지 않는다.
MMMU 모든 split
MMMU-Pro 모든 설정
그 문제의 원본·보기·해설 이미지
정답·해설
현재 진단의 O0/O1/P/M/MI
Knowledge Oracle의 benchmark-derived 자료
벤치마크 응답을 재작성한 교사 풀이
위 자료의 번역·숫자 변경·재렌더링·crop

기존 보고서는 해당 진단 자료를 BENCHMARK_DERIVED, training_allowed=false로 관리한다. 이 경계를 유지한다.    Qwen3_VL_4B_Bottleneck_Diagnost…    Qwen3_VL_4B_Knowledge_Oracle_20…
22.2 계획서와 생성 명세를 분리한다
이 계획서는 분석 근거와 외부 자료 설명을 함께 포함한다.
실제 생성 교사에는 다음만 들어 있는 별도 clean spec을 제공한다.
허용 객체·관계 타입
규칙 DSL
난도 범위
렌더링 계약
검증 계약
출력 schema
독립 예제

벤치마크의 특정 문제·응답·정답을 보고 “비슷하지만 다른 문제를 만들어라”라는 경로를 사용하지 않는다.
22.3 감사 순서
출처·권리·원래 split
    ↓
부모 계보와 benchmark taint
    ↓
정규화 text exact match
    ↓
원본 bytes / decoded pixels hash
    ↓
근접 문장·부분 이미지 후보 검색
    ↓
원문 페이지·작품·논문·보고서 family 검사
    ↓
후보 dataset 사이의 원천 중복
    ↓
증강 후 재검사
    ↓
승인 manifest

비슷한 공식이나 같은 primitive를 공유한다는 이유만으로 모두 같은 문제라고 처리하지 않는다. 반대로 숫자나 색이 바뀌었다는 이유로 독립 문제라고 처리하지도 않는다.
22.4 NVW도 자동으로 오염 없는 데이터가 아니다
Glyph를 바꿔도 benchmark의 topology·수치·질문을 그대로 가져왔다면 파생 오염이다.
NVW의 독립성은 새 모양이 아니라 독립 명세·독립 seed·비벤치마크 원천·추적 가능한 생성 계보로 관리한다.
22.5 승인 규칙
ALLOW :=
  source_known
  AND rights_approved_for_intended_use
  AND all_parents_known
  AND no_benchmark_ancestor
  AND task_specific_quality_pass
  AND identifiable_targets_only
  AND split_family_consistent
  AND contamination_audit_pass
  AND hash_in_frozen_manifest

검증하지 못한 항목은 UNKNOWN으로 유지한다. UNKNOWN을 편의상 PASS로 바꾸지 않는다.
22.6 뒤늦은 오염 발견
문제 family와 그 파생물을 차단하고, 이를 사용한 checkpoint를 추적한다.
깨끗한 추가 학습 효과를 평가하려면 정제된 데이터로 F0에서 다시 학습한다. 데이터 파일만 삭제했다고 이미 학습된 checkpoint의 오염이 사라졌다고 주장하지 않는다.
23. 라이선스·코드·자산 관리
권리 원장은 다음을 구분한다.
생성기 소스 코드
annotation
원본 이미지·문서
렌더링에 사용한 폰트·아이콘·자산
파생 데이터의 재배포
학습 checkpoint의 공개

TRACE의 코드·데이터 조건, ReGuLaR 원출처 조건, MUSCIMA++의 비상업·동일조건변경허락 조건 등을 하나의 “공개 데이터” 범주로 합치지 않는다. GitHub
이는 법률적 적합성 판정이 아니라 연구 운영의 확인 절차다. 불명확한 자료는 초기 주 실험에서 제외하고, 코드·manifest만 공개하는 경우에도 해당 이용 조건을 별도로 확인한다.
24. 구현 폴더와 책임 분해
다음 경로는 신규 구현 구조 제안이며 현재 사용자 로컬에 생성된 폴더가 아니다.
finetune_nvw_v2/
├─ clean_spec/
│  ├─ world_families.yaml
│  ├─ rule_dsl.yaml
│  ├─ rendering_contract.md
│  └─ supervision_contract.md
├─ configs/
│  ├─ sources/
│  ├─ mixtures/
│  ├─ modules/
│  └─ evaluations/
├─ manifests/
│  ├─ source_inventory.pending.json
│  ├─ source_locks/
│  ├─ approved_files/
│  └─ approved_training/
├─ adapters/
│  ├─ trace_adapter.py
│  ├─ clevr_adapter.py
│  ├─ raven_adapter.py
│  ├─ arc_adapter.py
│  └─ regular_adapter.py
├─ generators/
│  ├─ signal_world.py
│  ├─ spatial_world.py
│  ├─ finite_state_world.py
│  └─ professional_bridges/
├─ verifiers/
│  ├─ type_checker.py
│  ├─ independent_solver.py
│  ├─ identifiability.py
│  ├─ render_audit.py
│  └─ input_audit.py
├─ data/
│  ├─ sources/
│  ├─ canonical_private/
│  ├─ rendered/
│  ├─ quarantine/
│  └─ approved/
├─ training/
│  ├─ module_selector.py
│  ├─ view_builder.py
│  ├─ loss_masking.py
│  └─ checkpoint_export.py
├─ evaluation/
│  ├─ external_development/
│  ├─ sealed_contracts/
│  └─ metrics/
├─ audit/
│  ├─ lineage/
│  ├─ rights/
│  ├─ contamination/
│  └─ exposure/
└─ runs/

평가 원문·gold는 학습 데이터 폴더와 별도의 접근 통제 영역에 둔다.
24.1 최소 구현 산출물
모듈	필수 산출물
Source registry	정확한 저장소·revision·파일 목록·권리 상태
Dataset adapter	Source별 field 계약과 변환 테스트
NVW generator	독립 seed 재현·타입이 맞는 세계
Identifiability checker	답·규칙·상태별 식별 가능성
Renderer	Canonical과 연결된 이미지·좌표
Verifier	PASS/FAIL/UNKNOWN과 증거
Split builder	Family 분할·교집합 검사
View builder	Input whitelist·assistant target
Module selector	V/M/L trainable manifest
Trainer	실제 gradient·update·저장/재로드
Evaluator	같은 입력·채점의 F0/FT 대응 결과


25. 단계별 진행과 중단 기준
25.1 실행 순서
단계	작업	완료 기준
G0	기존 F0·평가·모듈 계약 고정	변경 이력과 exact configuration
G1	공개 source metadata 조사	Revision·파일·권리·필요 자산 목록
G2	NVW 세 family의 작은 prototype	Solver·식별 가능성·render 검사
G3	Adapter별 작은 실제 subset 연결	이미지·annotation·split 정합성
G4	별도 120-root 수준의 smoke	Input·loss·gradient·재로드·memory 통과
G5	승인된 공통 10K 구축	Frozen manifest와 source별 사용량
G6	FL·FM·FV·FML 비교	동일 데이터·평가의 실제 결과
G7	NVW·bridge 핵심 ablation	고정 모듈에서 비교
G8	선택 구성 50K 및 반복	품질·전이·보존 신호 유지
G9	최종 checkpoint·분석 동결	평가 전 hash 확정
G10	MMMU·Pro 최종 평가	전체 결과·한계·악화 사례 보고


120-root smoke는 구현 점검용이며 최종 sealed holdout에 포함하지 않는다.
25.2 초기 생성량
첫 NVW prototype은 예를 들어 세 family에서 각각 100개의 episode를 생성하는 규모로 시작한다.
이 300개는 대규모 학습을 위한 수량이 아니라 다음 결함을 찾기 위한 자료다.
예제로 답을 식별할 수 없는 문제
서로 구별되지 않는 glyph
규칙 적용 순서의 모호성
잘못된 중간 상태
렌더링에서 빠진 조건
view 변환 중 정답 누설

결함을 고친 뒤 독립 seed로 학습·development·sealed 후보를 생성한다.
25.3 실패 양상에 따른 다음 행동
관측	다음 행동
NVW만 개선, 실제 source 변화 없음	NVW quota 확대 중단·bridge 강화
구조 F1만 개선, QA 악화	관련 구조 선택·target 길이·상태 적용 점검
규칙 답은 맞지만 상태가 틀림	식별 가능한 중간 상태 supervision 강화
형식·종료만 개선	사용성 개선으로 보고, 추론 효과와 구분
FM/FV gradient가 없음	데이터 확대 중단·학습 경로 수정
ReGuLaR 원출처·권리 확인 실패	해당 source 제외 또는 대체
ARC 규칙이 support로 식별되지 않음	Support 보강·답 가능성 재분류
NVW와 전문 bridge 조합이 반복 개선	제한된 비중 확대·새 source 평가
특정 기존 강점이 크게 하락	Retention·학습률·quota 재검토
오염 발견	Family·checkpoint 추적 후 clean 재학습


25.4 처음부터 하지 않을 것
전체 대형 mixture 다운로드, 모든 모듈의 full FT, 모든 ablation의 50K 학습, 검증되지 않은 장문 교사 풀이 대량 투입, 별도 RL 파이프라인 구축을 동시에 시작하지 않는다.
첫 목표는 정확한 작은 데이터로 실제 가중치 업데이트와 외부 전이 측정을 한 번 끝내는 것이다.
26. 연구 성과를 어떻게 해석할 것인가
26.1 선행 자료가 제공하는 근거
TRACE 연구는 검증 가능한 다분야 합성 자료를 이용한 학습이 Qwen2.5-VL 계열의 외부 benchmark 평균을 개선했다고 보고한다. 이는 합성 시각 학습의 외부 전이를 연구할 실증적 동기를 제공한다. 그러나 대상 모델·학습 방식·평가가 본 프로젝트와 다르므로, 그 개선량을 Qwen3-VL-4B SFT의 예상 성능으로 가져오지 않는다. arXiv
26.2 본 프로젝트가 검증해야 할 차별점
본 연구에서 중요한 것은 “고대문자처럼 생긴 그림을 만들었다”가 아니다.
Episode별 새로운 의미 매핑
        +
관측과 의미 상태의 분리
        +
정답·규칙·중간 상태의 식별 가능성 검사
        +
실제 전문 canonical bridge
        +
동일 데이터의 선택적 모듈 학습
        +
미학습 표현·원출처·분야 전이 평가

이 구성 중 무엇이 필요했는지 ablation으로 확인하는 것이 연구의 핵심이다.
26.3 사용할 수 있는 결론 문장
실제 결과가 지지할 때만 다음과 같은 형태로 보고한다.
Episode별 의미 매핑을 바꾸는 검증 가능한 합성 시각 과제와 전문 분야의 구조화된 표현을 공동 학습한 결과, 지정된 외부 평가에서 원본 모델 대비 정확도와 관계·상태 지표의 개선이 관측되었다. 그 효과는 평가한 표현·원출처·분야 범위에서 확인되었으며, 모든 영역에 대한 일반화나 특정 내부 모듈의 유일한 원인을 입증한 것은 아니다.

실험 전에는 이 문장을 완료된 성과로 쓰지 않는다.
27. 현재 완료 상태와 첫 실행 대상
항목	이번 작업의 상태
직전 실행 계획서의 실제 내용 확인	수행
NVW 관련 공개 카드·원저자 자료 조사	수행
데이터별 수집·Adapter 설계	작성
NVW·전문 bridge·식별 가능성 설계	작성
FL/FM/FV/FML 및 데이터 ablation 설계	작성
실행용 데이터 revision lock 생성	미수행
실제 데이터 전수 다운로드·권리 승인	미수행
Generator·verifier 구현 및 테스트	미수행
Clean training manifest 생성	미수행
실제 파인튜닝·GPU memory 검증	미수행
MMMU/MMMU-Pro 개선 측정	미수행
새 .md 첨부 저장	도구 오류로 미완료


최종 실행 방향
공개 자료는 TRACE·CLEVR·RAVEN·ReGuLaR의 필요한 부분만 선별하고, 자체 NVW는 작은 검증 가능한 세 world family부터 구현한다.
그다음 같은 전문 의미 상태의 표준 표기와 새로운 표기를 연결하는 bridge를 만들고, 동일한 이미지 기반 10K manifest에서 FL·FM·FV·FML을 비교한다. NVW의 비중을 확대하는 기준은 암호 퍼즐 성적이 아니라 미학습 실제 전문 시각 문제의 정확도·관계 보존·중간 상태 추적이 함께 개선되는지로 정한다.
첫 착수 대상은 source inventory → 세 family의 300-episode prototype → 독립 검증 → 120-root 학습 smoke이며, 이를 통과한 뒤에만 공통 10K 학습을 시작한다.